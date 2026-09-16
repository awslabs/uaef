# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Job-creation handler for the UAEF Service API Lambda.

This module is part of the *separate* ``uaef-service`` deployable application.
Per **Requirement 6.2** it MUST NOT import the ``uaef`` package or the
``server`` extra — it only validates the request, records a ``PENDING`` job, and
asynchronously hands the work to the Worker Lambda (the sole component that
imports ``uaef``). No synchronous evaluation ever happens on the request path
(**Requirement 4.3**).

Responsibilities of :func:`create_evaluate_job` (this task, 9.2):

  * Validate the request body against the appropriate pydantic schema. A
    malformed JSON body or a schema-invalid body yields a ``400`` and creates
    **no** Job (**Requirement 4.6**).
  * Enforce the curated-metric / no-user-code limitation via the sibling
    ``validation`` module (authored in task 9.6). It is imported **lazily and
    guardedly** so this handler stands alone before that module exists; when it
    is unavailable the curated check is skipped (schema-level enforcement still
    applies, since custom-metric and schema-mapping fields are absent from the
    request schemas).
  * Write a ``PENDING`` Job record, recording the caller's Cognito ``sub`` claim
    as ``createdBy`` (**Requirement 9.3**).
  * Asynchronously invoke the Worker Lambda with ``InvocationType='Event'`` and
    return a ``jobId`` quickly — well within the 3-second / 29-second budgets
    (**Requirements 4.1, 4.2**).
  * If the asynchronous invocation fails to enqueue, advance the Job to the
    terminal ``FAILED`` state and return an error response, performing no
    evaluation (**Requirement 4.5**). Because the job state machine never skips
    ``PROCESSING`` (see ``job_state``), the failure path advances
    ``PENDING -> PROCESSING -> FAILED`` so ``FAILED`` is reached without
    violating monotonicity.

The sibling handlers ``get_job_status`` / ``list_experiments`` /
``get_experiment`` are intentionally **not** defined here — they are authored in
later tasks (9.3). The router (``handlers.api``) loads handlers lazily and
returns ``501`` until they exist.

Requirements: 4.1, 4.2, 4.3, 4.5, 4.6, 9.3
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import uuid
from typing import Any, Dict, Optional

from decimal import Decimal

from pydantic import ValidationError


def _decimals_to_native(obj: Any) -> Any:
    """Recursively convert DynamoDB ``Decimal`` values to ``int``/``float``.

    Job records are written with float scores coerced to ``Decimal`` (DynamoDB
    requirement). On read we convert back so the API returns real JSON numbers
    (e.g. ``0.95``) rather than strings, which clients/charts expect.
    """
    if isinstance(obj, Decimal):
        # Preserve integers as ints; everything else as float.
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, list):
        return [_decimals_to_native(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _decimals_to_native(v) for k, v in obj.items()}
    return obj

# Sibling modules in the same ``uaef-service`` app. These import boto3/pydantic
# only — never ``uaef`` — so importing them keeps the API Lambda free of the
# library/server extras (Requirement 6.2).
try:  # package-relative when deployed as the ``handlers`` package
    from .. import job_state, schemas  # type: ignore[import-not-found]
except (ImportError, ValueError):  # flat layout / top-level modules
    import job_state  # type: ignore[no-redef]
    import schemas  # type: ignore[no-redef]

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Environment variable naming the Worker Lambda function to async-invoke.
WORKER_FUNCTION_ENV = "WORKER_FUNCTION_NAME"

#: Environment variable naming the Step Functions state machine ARN for the
#: parallel-by-dimension evaluation orchestrator.
STATE_MACHINE_ARN_ENV = "EVAL_STATE_MACHINE_ARN"

#: Environment variable naming the S3 bucket holding full result payloads. The
#: Worker stores the full result JSON here (``results/<jobId>.json``) and records
#: its key as the Job's ``resultRef``; on a COMPLETED status read we presign a
#: short-lived GET for that key. Security review L-04: no default value — see
#: ``handlers/payloads.py`` for rationale. The CDK stacks always set this env
#: var explicitly, so all handlers agree on the bucket.
PAYLOAD_BUCKET_ENV = "PAYLOAD_BUCKET_NAME"

#: Lifetime of the presigned result GET URL (15 minutes — the same ceiling used
#: for upload URLs in ``handlers/payloads.py``).
RESULT_PRESIGN_EXPIRY_SECONDS = 900

#: Valid operation discriminators accepted by this handler.
_OPERATIONS = {
    "evaluate": schemas.EvaluateRequest,
    "batch_evaluate": schemas.BatchEvaluateRequest,
}

_JSON_HEADERS = {"Content-Type": "application/json"}

_lambda_client = None
_s3_client = None
_sfn_client = None


def _get_lambda_client():
    """Return a cached boto3 Lambda client (created lazily for testability)."""
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")
    return _lambda_client


def _get_s3_client():
    """Return a cached boto3 S3 client (created lazily for testability)."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def _get_sfn_client():
    """Return a cached boto3 Step Functions client (created lazily for testability)."""
    global _sfn_client
    if _sfn_client is None:
        _sfn_client = boto3.client("stepfunctions")
    return _sfn_client


def _payload_bucket() -> str:
    bucket = os.environ.get(PAYLOAD_BUCKET_ENV)
    if not bucket:
        raise RuntimeError(
            f"{PAYLOAD_BUCKET_ENV} is not set. This Lambda is misconfigured — "
            "the CDK deployment should always set this environment variable."
        )
    return bucket


def _worker_function_name() -> Optional[str]:
    name = os.environ.get(WORKER_FUNCTION_ENV)
    return name if name else None


def _state_machine_arn() -> Optional[str]:
    arn = os.environ.get(STATE_MACHINE_ARN_ENV)
    return arn if arn else None


# --------------------------------------------------------------------------- #
# Proxy response helpers (self-contained so jobs.py needs no router internals)
# --------------------------------------------------------------------------- #


def _response(status_code: int, body: Any) -> Dict[str, Any]:
    """Build an API Gateway proxy response with a JSON-encoded body."""
    encoded = body if isinstance(body, str) else json.dumps(body, default=str)
    return {
        "statusCode": status_code,
        "headers": dict(_JSON_HEADERS),
        "body": encoded,
    }


def _error(status_code: int, message: str, **extra: Any) -> Dict[str, Any]:
    """Build a standard JSON error proxy response."""
    payload: Dict[str, Any] = {"error": message}
    payload.update(extra)
    return _response(status_code, payload)


# --------------------------------------------------------------------------- #
# Event parsing helpers
# --------------------------------------------------------------------------- #


def _extract_caller_sub(event: Dict[str, Any]) -> Optional[str]:
    """Return the caller's verified Cognito ``sub`` claim from the proxy event.

    API Gateway places verified Cognito claims under
    ``requestContext.authorizer.claims`` (Requirement 9.1/9.2). The router has
    already rejected requests without a ``sub``; this re-extraction keeps the
    handler usable in isolation and is defensive only.
    """
    request_context = event.get("requestContext") or {}
    authorizer = request_context.get("authorizer") or {}
    claims = authorizer.get("claims") or {}
    sub = claims.get("sub")
    return sub if isinstance(sub, str) and sub else None


def _parse_body(event: Dict[str, Any]) -> Any:
    """Decode and JSON-parse the proxy event body.

    Handles the base64-encoded body API Gateway delivers when
    ``isBase64Encoded`` is set. Raises :class:`ValueError` on a missing body or
    invalid JSON so the caller can map it to a ``400``.
    """
    raw = event.get("body")
    if raw is None:
        raise ValueError("Request body is required.")

    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(raw).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Body is not valid base64: {exc}") from exc

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Body is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Curated-metric / limitation validation (task 9.6) — imported guardedly
# --------------------------------------------------------------------------- #


def _curated_metric_set() -> Any:
    """Return the trusted curated metric catalog the validator checks against.

    Sourced from the library (single source of truth) via the ``catalog``
    handler's ``get_metric_catalog()`` — which delegates to the Worker Lambda's
    ``uaef.get_full_metric_catalog()``. We never hardcode a metric list here
    (Requirement 12.7). The returned value is the grouped catalog dict, which
    :func:`validation.validate_evaluation_request` flattens internally.

    Best-effort and guarded: if the catalog handler is unavailable or returns an
    error proxy response, an empty set is returned. Callers treat an empty
    curated set as "fail closed" — a request that *names* metrics is then
    rejected rather than silently accepted, preserving the security boundary.

    This indirection is also the seam tests monkeypatch to inject a known
    curated set without standing up a Worker.
    """
    try:  # package-relative then flat layout
        try:
            from . import catalog  # type: ignore[import-not-found]
        except (ImportError, ValueError):
            import catalog  # type: ignore[no-redef]
    except ImportError:
        return set()

    getter = getattr(catalog, "get_metric_catalog", None)
    if not callable(getter):
        return set()
    try:
        result = getter()
    except Exception:  # noqa: BLE001 — never let catalog issues crash validation
        return set()
    # The catalog handler returns either the catalog dict or an error proxy
    # response ({"statusCode", ...}); the latter means "unavailable".
    if isinstance(result, dict) and "statusCode" in result:
        return set()
    return result


def _run_curated_validation(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Enforce the library-only limitation if the ``validation`` module exists.

    Task 9.6 authors ``uaef-service/validation.py`` exposing
    ``validate_evaluation_request(body, curated_metrics)`` which raises
    :class:`validation.ValidationRejection` (carrying ``status_code``,
    ``message``, ``code``) when a request names a non-curated/custom metric
    (Req 11.1/11.3) or specifies a ``GenericJSONAdapter`` schema mapping
    (Req 11.2). The whole *raw* body is inspected so unmodeled fields such as
    ``schema_mapping`` — which the pydantic request schema drops — are still
    caught.

    Called *before* any Job is created or the Worker is invoked, so a rejection
    returns a ``400`` proxy response with **no** Job and **no** Worker invoke
    (Requirements 11.1, 11.2, 11.3).

    This hook is tolerant: if the module or a recognized entry point is not
    available it returns ``None`` (skip), so the handler stands alone before
    task 9.6 lands. The curated metric set is sourced (only when the request
    actually names metrics) from the library catalog via
    :func:`_curated_metric_set` — never hardcoded.
    """
    try:  # package-relative then flat layout
        try:
            from .. import validation  # type: ignore[import-not-found]
        except (ImportError, ValueError):
            import validation  # type: ignore[no-redef]
    except ImportError:
        return None  # task 9.6 not authored yet — schema-level checks still apply

    validator = getattr(validation, "validate_evaluation_request", None)
    if not callable(validator):
        # Backward-compatible fallback to a single-arg metric validator.
        for name in ("validate_metrics", "validate_curated_metrics", "validate_curated"):
            candidate = getattr(validation, name, None)
            if callable(candidate):
                try:
                    candidate(body.get("metrics"))
                except Exception as exc:  # noqa: BLE001
                    return _error(400, str(exc) or "Requested metric is not available in service mode.")
                return None
        return None  # module present but no recognized entry point — skip

    rejection_cls = getattr(validation, "ValidationRejection", Exception)

    # Only pay the cost of sourcing the curated catalog when the request names
    # metrics; the schema-mapping check (Req 11.2) needs no curated set.
    curated = _curated_metric_set() if body.get("metrics") else set()

    try:
        validator(body, curated)
    except rejection_cls as exc:  # type: ignore[misc] — limitation rejection
        status = getattr(exc, "status_code", 400)
        message = getattr(exc, "message", None) or str(exc) or "Request rejected."
        code = getattr(exc, "code", None)
        if code:
            return _error(status, message, code=code)
        return _error(status, message)
    return None


# --------------------------------------------------------------------------- #
# Worker invocation
# --------------------------------------------------------------------------- #


def _invoke_worker_async(job_id: str, operation: str, request: Dict[str, Any]) -> None:
    """Start a Step Functions execution for parallel-by-dimension evaluation.

    The state machine (provisioned by OrchestratorStack) fans out one Worker
    Lambda invocation per metric dimension, then reduces the per-dimension
    results into a single Job outcome. This replaces the former direct
    async-invoke of the Worker, removing the 15-minute single-Lambda timeout
    constraint for multi-dimension evaluations.

    Falls back to direct Worker async-invoke if the state machine ARN is not
    configured (e.g. during development or if the OrchestratorStack is absent).
    """
    sm_arn = _state_machine_arn()
    if sm_arn:
        # Route through Step Functions for parallel-by-dimension evaluation.
        _start_state_machine_execution(job_id, operation, request, sm_arn)
    else:
        # Fallback: direct async-invoke (legacy path).
        _invoke_worker_direct(job_id, operation, request)


def _start_state_machine_execution(
    job_id: str, operation: str, request: Dict[str, Any], sm_arn: str
) -> None:
    """Start a Step Functions execution for the evaluation job.

    The execution input matches what the Splitter Lambda expects:
        {"jobId": "...", "operation": "...", "request": {...}}
    """
    execution_input = json.dumps({
        "jobId": job_id,
        "operation": operation,
        "request": request,
    }, default=str)

    response = _get_sfn_client().start_execution(
        stateMachineArn=sm_arn,
        name=f"eval-{job_id}",
        input=execution_input,
    )
    # start_execution returns immediately (async). If it throws, the caller
    # maps to the FAILED path (Requirement 4.5).
    if not response.get("executionArn"):
        raise RuntimeError("Step Functions start_execution returned no executionArn.")


def _invoke_worker_direct(job_id: str, operation: str, request: Dict[str, Any]) -> None:
    """Directly async-invoke the Worker Lambda (legacy fallback path).

    Used when the Step Functions state machine is not configured.
    """
    function_name = _worker_function_name()
    if not function_name:
        raise RuntimeError(
            f"Worker function name not configured (set ${WORKER_FUNCTION_ENV})."
        )

    payload = {
        "jobId": job_id,
        "operation": operation,
        "request": request,
    }
    response = _get_lambda_client().invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(payload, default=str).encode("utf-8"),
    )
    # For asynchronous ('Event') invocations Lambda returns 202 on a successful
    # enqueue. Treat anything else as a failure to enqueue.
    status = response.get("StatusCode")
    if status not in (200, 202):
        raise RuntimeError(f"Worker enqueue returned unexpected status {status}.")
    if response.get("FunctionError"):
        raise RuntimeError(f"Worker enqueue reported error: {response['FunctionError']}.")


def _mark_failed(job_id: str, code: str, message: str) -> None:
    """Drive a freshly created ``PENDING`` job to terminal ``FAILED``.

    The job state machine never skips ``PROCESSING`` (Requirement 5.1), so the
    only state-machine-compliant way to reach ``FAILED`` from ``PENDING`` is to
    advance through ``PROCESSING``. Both writes are conditional and idempotent;
    if a (hypothetical) racing worker has already moved the job on, these become
    no-ops and the job's own lifecycle wins.
    """
    job_state.start_processing(job_id)
    job_state.fail_job(job_id, code=code, message=message)


# --------------------------------------------------------------------------- #
# Public handler
# --------------------------------------------------------------------------- #


def create_evaluate_job(event: Dict[str, Any], operation: str) -> Dict[str, Any]:
    """Create an async evaluation Job and hand it to the Worker Lambda.

    Steps (Requirements 4.1, 4.2, 4.3, 4.5, 4.6, 9.3):
      1. Validate the body against the schema for ``operation``. Malformed or
         schema-invalid bodies yield ``400`` with **no** Job created (4.6).
      2. Enforce the curated-metric limitation (guarded; task 9.6).
      3. Write a ``PENDING`` Job recording the caller ``sub`` as ``createdBy`` (9.3).
      4. Async-invoke the Worker (``InvocationType='Event'``) and return the
         ``jobId`` immediately — no synchronous evaluation (4.1/4.2/4.3).
      5. On enqueue failure, set the Job ``FAILED`` and return an error (4.5).

    Returns either a plain ``JobCreatedResponse`` dict (which the router wraps in
    ``200``) or a full proxy response (``400`` / ``502``) the router passes
    through.
    """
    if operation not in _OPERATIONS:
        # Defensive: the router only ever passes the two known operations.
        return _error(400, f"Unsupported operation: {operation!r}.")

    # --- 1. Parse + schema-validate the body (400 on failure, no Job) -------
    try:
        body = _parse_body(event)
    except ValueError as exc:
        return _error(400, str(exc))

    schema_cls = _OPERATIONS[operation]
    try:
        request_model = schema_cls.model_validate(body)
    except ValidationError as exc:
        return _error(400, "Request body failed validation.", details=json.loads(exc.json()))

    request_dict = request_model.model_dump()

    # --- 2. Curated-metric / limitation enforcement (guarded; task 9.6) -----
    # Validate the *raw* parsed body (not the pydantic-dumped model) so unmodeled
    # fields like ``schema_mapping`` — which the request schema drops — are still
    # caught (Requirement 11.2). Runs before any Job write / Worker invoke.
    limitation_response = _run_curated_validation(body if isinstance(body, dict) else request_dict)
    if limitation_response is not None:
        return limitation_response  # 400, no Job, no Worker invoke

    # --- 3. Record caller identity + create the PENDING Job (9.3) -----------
    caller_sub = _extract_caller_sub(event)
    if not caller_sub:
        # The Cognito authorizer/router should have rejected this already.
        return _error(401, "Unauthorized: missing caller identity.")

    job_id = str(uuid.uuid4())
    try:
        job_state.create_job(job_id, operation, created_by=caller_sub)
    except Exception as exc:  # noqa: BLE001 — table write failure -> 500, no invoke
        # Security review M-03: don't return DynamoDB/boto3 exception detail.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to create job record"
        )
        return _error(500, safe_message, correlationId=correlation_id)

    # --- 4./5. Async-invoke the Worker; on enqueue failure mark FAILED ------
    try:
        _invoke_worker_async(job_id, operation, request_dict)
    except Exception as exc:  # noqa: BLE001 — failed to enqueue (Requirement 4.5)
        # Security review M-03: the persisted job error (surfaced verbatim by
        # GET /jobs/{jobId}) and the response body must not carry exception
        # detail either.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to enqueue evaluation worker"
        )
        _mark_failed(job_id, code="ENQUEUE_FAILED", message=safe_message)
        return _error(
            502,
            "Failed to enqueue evaluation worker; job marked FAILED.",
            jobId=job_id,
            status=job_state.FAILED,
            correlationId=correlation_id,
        )

    # --- Success: return the jobId immediately (no synchronous evaluation) --
    return schemas.JobCreatedResponse(jobId=job_id).model_dump()


# --------------------------------------------------------------------------- #
# GET /jobs/{jobId} — status read with per-caller authorization (task 9.3)
# --------------------------------------------------------------------------- #


def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the stored Job item for ``job_id``, or ``None`` if it does not exist.

    Reuses the ``job_state`` table accessor so the jobs table name/env resolution
    lives in exactly one place. A ``get_item`` with no matching key returns a
    response without an ``Item`` member, which we surface as ``None`` (Req 4.7).
    """
    response = job_state._get_table().get_item(Key={"jobId": job_id})
    item = response.get("Item")
    return item if isinstance(item, dict) else None


def _presign_result_get(result_ref: Optional[str]) -> Optional[str]:
    """Return a short-lived presigned GET URL for ``result_ref``.

    ``result_ref`` is the S3 object key the Worker recorded for the full result
    JSON. We presign a ``get_object`` URL (15-minute ceiling) so the client can
    download the full result directly from S3. If presigning fails for any
    reason we fall back to returning the raw key rather than failing the whole
    status read — the status itself is still useful to the caller.
    """
    if not result_ref:
        return result_ref
    try:
        return _get_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": _payload_bucket(), "Key": result_ref},
            ExpiresIn=RESULT_PRESIGN_EXPIRY_SECONDS,
        )
    except (ClientError, BotoCoreError, ValueError):
        # Best-effort: keep the raw key so the response still references the
        # result; never let a presign hiccup turn a valid status into an error.
        return result_ref


def get_job_status(job_id: str, caller_sub: str) -> Dict[str, Any]:
    """Return the status of a Job, authorizing the caller by ``createdBy``.

    Behaviour (Requirements 4.4, 4.7, 9.4):
      * If no Job exists for ``job_id`` -> ``404`` (Req 4.7).
      * If the Job's ``createdBy`` does not equal ``caller_sub`` -> ``403`` with
        **no** status, result reference, or summary disclosed (Req 9.4). The
        same 403 is returned whether or not the job exists for that other owner,
        so existence is not leaked to non-owners.
      * Otherwise return a ``JobStatusResponse``-shaped dict carrying the current
        ``status``, ``operation``, ``createdAt`` and ``updatedAt`` (Req 4.4).
        On ``COMPLETED`` it includes ``experimentId``, ``resultSummary`` and a
        presigned-GET ``resultRef``; on ``FAILED`` it includes the ``error``
        ``{code, message}`` payload.

    Returns either a ``JobStatusResponse`` dict (which the router wraps in a
    ``200``) or a full proxy response (``403`` / ``404`` / ``500``) the router
    passes through.
    """
    try:
        item = _read_job(job_id)
    except (ClientError, BotoCoreError) as exc:
        # Security review M-03: don't return DynamoDB exception detail.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to read job record"
        )
        return _error(500, safe_message, correlationId=correlation_id)

    # 404 when the Job does not exist (Requirement 4.7).
    if item is None:
        return _error(404, f"Job not found: {job_id}.")

    # 403 when the caller is not the owner. Per Requirement 9.4 the response must
    # NOT include the Job's status, result reference, or summary.
    if item.get("createdBy") != caller_sub:
        return _error(403, "Forbidden: you do not have access to this job.")

    status = item.get("status")

    response = schemas.JobStatusResponse(
        jobId=job_id,
        status=status,
        operation=item.get("operation"),
        createdAt=item.get("createdAt") or "",
        updatedAt=item.get("updatedAt") or "",
    )

    # COMPLETED: surface the experiment id, the small inline summary, and a
    # presigned GET URL for the full result (Requirement 4.4).
    if status == job_state.COMPLETED:
        response.experimentId = item.get("experimentId")
        response.resultSummary = _decimals_to_native(item.get("resultSummary"))
        response.resultRef = _presign_result_get(item.get("resultRef"))
    # FAILED: surface the error code/message (Requirement 4.4).
    elif status == job_state.FAILED:
        error = item.get("error")
        response.error = error if isinstance(error, dict) else None
    # PROCESSING: surface progress info (dimension completion tracking).
    elif status == job_state.PROCESSING:
        progress = item.get("progress")
        if isinstance(progress, dict):
            response.progress = _decimals_to_native(progress)

    return response.model_dump(exclude_none=True)


# --------------------------------------------------------------------------- #
# GET /experiments/{experimentId} — full experiment detail (delegates to Worker)
# --------------------------------------------------------------------------- #


def get_experiment(experiment_id: str, caller_sub: Optional[str] = None) -> Dict[str, Any]:
    """Return the full detail of a single experiment by ID, authorized by owner.

    Security review H-02: previously returned any experiment_id's detail with
    no ownership check — any authenticated caller could read any other
    caller's experiment data by ID. Now requires ``caller_sub`` to own the
    experiment (checked by the Worker via the experiment's ``created_by``);
    on a mismatch or missing experiment, returns 403 with no experiment
    detail disclosed — the same non-leaking pattern used by
    :func:`get_job_status` for Jobs (Requirement 9.4's approach, applied here
    to experiments).

    Delegates to the Worker Lambda synchronously (the Worker is the only
    component that imports ``uaef`` and can access the storage layer).
    Returns the experiment metadata, average scores, and per-row evaluations.
    """
    function_name = _worker_function_name()
    if not function_name:
        return _error(503, "Experiment detail unavailable: worker not configured.")
    if not caller_sub:
        return _error(401, "Unauthorized: missing caller identity.")

    try:
        response = _get_lambda_client().invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(
                {
                    "action": "get_experiment",
                    "experiment_id": experiment_id,
                    "created_by": caller_sub,
                }
            ).encode("utf-8"),
        )
        payload = response.get("Payload")
        raw = payload.read() if hasattr(payload, "read") else payload
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        result = json.loads(raw) if raw else {}

        if response.get("FunctionError"):
            # Security review M-03: a Lambda FunctionError payload is the
            # Worker's raw traceback — never return it to the caller.
            safe_message, correlation_id = job_state.sanitize_error(
                RuntimeError(result), context="Failed to retrieve experiment from worker"
            )
            return _error(502, safe_message, correlationId=correlation_id)
        if isinstance(result, dict) and result.get("code") == "FORBIDDEN":
            # Security review H-02: no experiment status/summary disclosed,
            # whether the experiment doesn't exist or just isn't the
            # caller's — same non-leaking 403 as get_job_status.
            return _error(403, "Forbidden: you do not have access to this experiment.")
        if isinstance(result, dict) and result.get("error"):
            if "not found" in result["error"].lower():
                # A clean "not found" message from the Worker is safe to pass
                # through as-is — it carries no internal detail.
                return _error(404, result["error"])
            safe_message, correlation_id = job_state.sanitize_error(
                RuntimeError(result["error"]), context="Failed to retrieve experiment detail"
            )
            return _error(502, safe_message, correlationId=correlation_id)
        return result
    except Exception as exc:  # noqa: BLE001
        # Security review M-03: don't return Lambda-invoke exception detail.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to retrieve experiment detail"
        )
        return _error(502, safe_message, correlationId=correlation_id)
