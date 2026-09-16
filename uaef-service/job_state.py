# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Async job state machine for the UAEF Service.

This module owns the lifecycle of an evaluation Job stored in the
``uaef-service-jobs`` DynamoDB table. It is part of the *separate* UAEF Service
application and therefore MUST NOT import the ``uaef`` package — it talks to
DynamoDB directly via ``boto3``.

State machine (see design.md "Async Job State Machine"):

    PENDING --> PROCESSING --> COMPLETED
                          \--> FAILED

Rules enforced here:
  * Transitions only advance forward along the allowed edges above.
  * ``PROCESSING`` can never be skipped (``PENDING`` cannot jump straight to a
    terminal state).
  * Every write is a *conditional* DynamoDB update that only succeeds when the
    stored ``status`` equals the expected ``frm`` value. This makes transitions
    idempotent under duplicate/async worker invocations.
  * Terminal states (``COMPLETED`` / ``FAILED``) are immutable: a conditional
    write whose ``frm`` is a terminal state, or whose stored status is already
    terminal, is rejected and leaves the record unchanged.
  * Error messages are bounded to 1024 characters before being persisted.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger("uaef_service")


def _floats_to_decimal(obj: Any) -> Any:
    """Recursively convert ``float`` values to ``Decimal`` for DynamoDB.

    DynamoDB's boto3 resource interface rejects Python floats ("Float types are
    not supported. Use Decimal types instead."). Evaluation result summaries
    carry float scores, so any attribute written to the jobs table is converted
    here. Uses ``str(value)`` to avoid binary float-rounding artifacts.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, list):
        return [_floats_to_decimal(v) for v in obj]
    if isinstance(obj, tuple):
        return [_floats_to_decimal(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _floats_to_decimal(v) for k, v in obj.items()}
    return obj

# --------------------------------------------------------------------------- #
# Status constants
# --------------------------------------------------------------------------- #

PENDING = "PENDING"
PROCESSING = "PROCESSING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"

#: All valid job statuses.
STATUSES: frozenset[str] = frozenset({PENDING, PROCESSING, COMPLETED, FAILED})

#: Terminal statuses are immutable once written.
TERMINAL_STATUSES: frozenset[str] = frozenset({COMPLETED, FAILED})

#: The only allowed forward transitions. ``PROCESSING`` is never skipped.
ALLOWED_TRANSITIONS: frozenset[Tuple[str, str]] = frozenset(
    {
        (PENDING, PROCESSING),
        (PROCESSING, COMPLETED),
        (PROCESSING, FAILED),
    }
)

#: Maximum persisted length of an error message (Requirement 5.5).
MAX_ERROR_MESSAGE_LENGTH = 1024

#: Environment variable naming the DynamoDB jobs table. Security review L-04:
#: no default value — a hardcoded table name risks colliding with an
#: unrelated table if the env var is ever unset in a real deployment. The
#: CDK stacks always set this explicitly.
JOBS_TABLE_ENV = "JOBS_TABLE_NAME"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class InvalidTransitionError(ValueError):
    """Raised when a (frm, to) pair is not an allowed forward transition."""


# --------------------------------------------------------------------------- #
# Transition validation helpers
# --------------------------------------------------------------------------- #


def is_valid_transition(frm: str, to: str) -> bool:
    """Return ``True`` iff ``frm -> to`` is an allowed forward transition.

    This is a pure predicate over the state machine; it performs no I/O.
    """
    return (frm, to) in ALLOWED_TRANSITIONS


def is_terminal(status: str) -> bool:
    """Return ``True`` iff ``status`` is a terminal (immutable) state."""
    return status in TERMINAL_STATUSES


def allowed_next_states(frm: str) -> frozenset[str]:
    """Return the set of states reachable from ``frm`` in a single transition."""
    return frozenset(to for (f, to) in ALLOWED_TRANSITIONS if f == frm)


def validate_transition(frm: str, to: str) -> None:
    """Validate a transition, raising :class:`InvalidTransitionError` if invalid.

    A transition is invalid when either status is unknown, when the source is a
    terminal state (terminal states are immutable), or when the edge is not one
    of the allowed forward edges (which also rejects skipping ``PROCESSING``).
    """
    if frm not in STATUSES:
        raise InvalidTransitionError(f"Unknown source status: {frm!r}")
    if to not in STATUSES:
        raise InvalidTransitionError(f"Unknown target status: {to!r}")
    if frm in TERMINAL_STATUSES:
        raise InvalidTransitionError(
            f"Cannot transition out of terminal state {frm!r}; terminal states are immutable"
        )
    if not is_valid_transition(frm, to):
        raise InvalidTransitionError(
            f"Illegal transition {frm!r} -> {to!r}; "
            f"allowed next states from {frm!r} are {sorted(allowed_next_states(frm))}"
        )


def bound_error_message(message: Any) -> str:
    """Coerce ``message`` to a string bounded at :data:`MAX_ERROR_MESSAGE_LENGTH`."""
    text = "" if message is None else str(message)
    if len(text) > MAX_ERROR_MESSAGE_LENGTH:
        return text[:MAX_ERROR_MESSAGE_LENGTH]
    return text


def audit_log(event: str, job_id: str, **fields: Any) -> None:
    """Emit a structured audit-trail log line for a job lifecycle event.

    Security review L-01 (inadequate AI-specific logging and monitoring): the
    service had API access logs but no audit trail of job lifecycle events
    (created / started / completed / failed) correlated by job ID, and no
    anomaly signal for e.g. unusual score patterns. This is a minimal,
    additive audit trail — it does NOT change any evaluation behavior.

    Deliberately logs only metadata (job ID, operation, status, aggregate
    scores/counts, error codes) — never raw prompts, agent responses, ground
    truth, or judge reasoning text. Callers must not pass those as ``fields``;
    this function does no content-based redaction of its own (unlike
    ``sanitize_error``, which sanitizes exception text), so it relies on
    call-site discipline to only pass already-aggregate data (e.g. the
    ``resultSummary`` shape built by ``handlers/worker.py::_build_outcome``,
    which carries scores/counts/trace IDs only).

    Emitted via the standard ``logging`` module at INFO level, captured by
    CloudWatch Logs like every other log line in these Lambdas — this is not
    a new storage location or retention policy, just a consistent,
    greppable/queryable line shape (CloudWatch Logs Insights can filter on
    ``event=...`` and ``jobId=...``) for operators.
    """
    parts = [f"event={event}", f"jobId={job_id}"]
    for key, value in fields.items():
        parts.append(f"{key}={value!r}")
    logger.info("AUDIT " + " ".join(parts))


def sanitize_error(exc: BaseException, *, context: str = "") -> Tuple[str, str]:
    """Turn an internal exception into a caller-safe message + correlation ID.

    Security review M-03 (verbose error disclosure): several handlers returned
    ``str(exc)`` directly to API callers, which for boto3/S3/DynamoDB errors can
    include bucket names, S3 keys, ARNs, and other account-internal detail —
    reconnaissance that lowers the effort of exploiting other findings (e.g.
    H-03, H-04).

    This logs the full exception (with a correlation ID) via the standard
    ``logging`` module — CloudWatch Logs captures it for operators — and
    returns only a generic, correlation-ID-bearing message safe to include in
    an API response body or a persisted job/UI error.

    Args:
        exc: The caught exception.
        context: Short human-readable description of what failed (e.g.
            "Failed to read job record"), included in the safe message and the
            server-side log line so both can be correlated by a human without
            needing the correlation ID.

    Returns:
        ``(safe_message, correlation_id)``. ``safe_message`` is safe to return
        to a caller; the full exception detail is only in server-side logs,
        keyed by ``correlation_id``.
    """
    correlation_id = str(uuid4())
    logger.error(
        "%s (correlation_id=%s): %s: %s",
        context or "Unhandled error",
        correlation_id,
        type(exc).__name__,
        exc,
        exc_info=exc,
    )
    prefix = f"{context}. " if context else ""
    safe_message = f"{prefix}Correlation ID: {correlation_id}"
    return safe_message, correlation_id


# --------------------------------------------------------------------------- #
# DynamoDB wiring
# --------------------------------------------------------------------------- #


def _table_name() -> str:
    table = os.environ.get(JOBS_TABLE_ENV)
    if not table:
        raise RuntimeError(
            f"{JOBS_TABLE_ENV} is not set. This Lambda is misconfigured — "
            "the CDK deployment should always set this environment variable."
        )
    return table


_table = None


def _get_table():
    """Return a cached DynamoDB Table resource for the jobs table."""
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb").Table(_table_name())
    return _table


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_created_by(job_id: str) -> Optional[str]:
    """Return the ``createdBy`` value recorded on a job, or ``None`` if not found.

    Security review H-02: the Worker and Reducer Lambdas run the actual
    evaluate/batch_evaluate/invoke_evaluate operation and must stamp
    ``created_by`` on any experiment they persist, but they only receive a
    ``jobId`` (not the caller identity directly) in their event payload. This
    looks up the identity the API Lambda already recorded at job creation
    (``create_job``'s ``created_by``) so the Worker/Reducer never has to be
    separately trusted with — or have the event payload carry — the caller's
    identity redundantly.
    """
    try:
        response = _get_table().get_item(Key={"jobId": job_id})
    except (ClientError, BotoCoreError):
        return None
    item = response.get("Item")
    if not isinstance(item, dict):
        return None
    created_by = item.get("createdBy")
    return created_by if isinstance(created_by, str) and created_by else None


# --------------------------------------------------------------------------- #
# Core conditional transition
# --------------------------------------------------------------------------- #


def _transition(job_id: str, frm: str, to: str, **attrs: Any) -> bool:
    """Conditionally transition a job from ``frm`` to ``to``.

    The update is applied with a DynamoDB ``ConditionExpression`` requiring the
    stored ``status`` to equal ``frm``. This guarantees:

      * Idempotency: a duplicate/async worker invocation that finds the job
        already past ``frm`` performs no write and returns ``False``.
      * Terminal immutability: because ``frm`` for any write into a terminal
        state must be ``PROCESSING``, a record already in a terminal state can
        never be overwritten — the condition fails and ``False`` is returned.

    Args:
        job_id: Partition key (``jobId``) of the job record.
        frm: Expected current status. Must be a non-terminal status.
        to: Target status. ``(frm, to)`` must be an allowed forward transition.
        **attrs: Extra attributes to set on the record (e.g. ``experimentId``,
            ``resultRef``, ``resultSummary``, ``error``). ``error.message`` style
            payloads should already be bounded by the caller; this function also
            bounds a top-level ``error`` dict's ``message`` defensively.

    Returns:
        ``True`` if the conditional write succeeded; ``False`` if the stored
        status did not equal ``frm`` (including when the job is terminal or the
        job does not exist).

    Raises:
        InvalidTransitionError: if ``(frm, to)`` is not an allowed transition.
        ClientError: for DynamoDB errors other than the conditional check
            failure.
    """
    validate_transition(frm, to)

    attrs = _sanitize_attrs(attrs)

    # Build the SET expression: always update status + updatedAt, plus any extras.
    set_parts = ["#status = :to", "#updatedAt = :updatedAt"]
    expr_names: Dict[str, str] = {"#status": "status", "#updatedAt": "updatedAt"}
    expr_values: Dict[str, Any] = {":to": to, ":frm": frm, ":updatedAt": _now_iso()}

    for index, (key, value) in enumerate(attrs.items()):
        name_placeholder = f"#a{index}"
        value_placeholder = f":v{index}"
        set_parts.append(f"{name_placeholder} = {value_placeholder}")
        expr_names[name_placeholder] = key
        expr_values[value_placeholder] = value

    update_expression = "SET " + ", ".join(set_parts)

    try:
        _get_table().update_item(
            Key={"jobId": job_id},
            UpdateExpression=update_expression,
            # Only write when the stored status is exactly `frm` AND the item
            # exists. attribute_exists(jobId) prevents creating a brand-new row.
            ConditionExpression="attribute_exists(jobId) AND #status = :frm",
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            # Stored status != frm (terminal, already advanced, or missing row).
            return False
        raise


def _sanitize_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Defensively bound any error message contained in extra attributes."""
    if not attrs:
        return {}
    sanitized = dict(attrs)
    error = sanitized.get("error")
    if isinstance(error, dict) and "message" in error:
        error = dict(error)
        error["message"] = bound_error_message(error.get("message"))
        sanitized["error"] = error
    # DynamoDB rejects floats; convert any float (e.g. result-summary scores)
    # to Decimal across the whole attribute set before writing.
    return _floats_to_decimal(sanitized)


# --------------------------------------------------------------------------- #
# High-level lifecycle operations
# --------------------------------------------------------------------------- #


def create_job(job_id: str, operation: str, created_by: str, **attrs: Any) -> Dict[str, Any]:
    """Create a new ``PENDING`` job record.

    Uses a conditional put so an existing job is never clobbered. Raises
    :class:`ClientError` (``ConditionalCheckFailedException``) if the job
    already exists.
    """
    now = _now_iso()
    item: Dict[str, Any] = {
        "jobId": job_id,
        "status": PENDING,
        "operation": operation,
        "createdBy": created_by,
        "createdAt": now,
        "updatedAt": now,
    }
    item.update(_sanitize_attrs(attrs))
    _get_table().put_item(
        Item=item,
        ConditionExpression="attribute_not_exists(jobId)",
    )
    audit_log("job_created", job_id, operation=operation, createdBy=created_by)
    return item


def start_processing(job_id: str, **attrs: Any) -> bool:
    """Transition ``PENDING -> PROCESSING`` (idempotent first worker action).

    Returns ``False`` when the job is not in ``PENDING`` (already picked up,
    terminal, or missing), in which case the worker must not evaluate.
    """
    ok = _transition(job_id, PENDING, PROCESSING, **attrs)
    if ok:
        audit_log("job_processing_started", job_id)
    return ok


def complete_job(
    job_id: str,
    *,
    experiment_id: str | None = None,
    result_ref: str | None = None,
    result_summary: Dict[str, Any] | None = None,
    **attrs: Any,
) -> bool:
    """Transition ``PROCESSING -> COMPLETED`` with result references.

    Returns ``False`` if the job is not in ``PROCESSING`` (e.g. already terminal).
    """
    extra: Dict[str, Any] = dict(attrs)
    if experiment_id is not None:
        extra["experimentId"] = experiment_id
    if result_ref is not None:
        extra["resultRef"] = result_ref
    if result_summary is not None:
        extra["resultSummary"] = result_summary
    ok = _transition(job_id, PROCESSING, COMPLETED, **extra)
    if ok:
        # Audit only the aggregate shape (count / overallScore / passed —
        # see _build_outcome / _summarize_one in handlers/worker.py), never
        # raw prompts/responses. resultSummary for a single evaluate() call
        # is already aggregate-only; for batch it's {"count", "results": [...]}
        # where each entry is likewise aggregate-only, so this is safe as-is.
        audit_log(
            "job_completed",
            job_id,
            experimentId=experiment_id,
            resultSummary=result_summary,
        )
    return ok


def fail_job(
    job_id: str,
    *,
    code: str,
    message: str,
    **attrs: Any,
) -> bool:
    """Transition ``PROCESSING -> FAILED`` with a bounded error payload.

    The error message is truncated to :data:`MAX_ERROR_MESSAGE_LENGTH` chars.
    Returns ``False`` if the job is not in ``PROCESSING`` (e.g. already
    ``COMPLETED`` — a terminal record is never overwritten).
    """
    error = {"code": str(code), "message": bound_error_message(message)}
    ok = _transition(job_id, PROCESSING, FAILED, error=error, **attrs)
    if ok:
        # Log the error code only, not the message: callers already sanitize
        # messages via sanitize_error() before reaching here (security review
        # M-03), and that function already logs full detail server-side keyed
        # by its own correlation ID — no need to duplicate it here.
        audit_log("job_failed", job_id, errorCode=code)
    return ok
