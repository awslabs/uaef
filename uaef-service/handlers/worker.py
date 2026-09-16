# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Worker Lambda for the UAEF Service — the only component that imports ``uaef``.

Unlike the API Lambda (``handlers/api.py``), the Worker Lambda **does** consume
the published library: it ``import uaef`` from the pip-installed ``uaef[server]``
package and runs the actual evaluation. Per the single-source-of-truth boundary
(Requirement 8.5) the worker adds **no** scoring, aggregation, parsing, or
invocation logic of its own — it delegates entirely to the library:

  * ground-truth loading/parsing  -> ``uaef.data``
  * agent invocation              -> ``uaef.adapters`` (AgentCoreAdapter / invoke_*)
  * trace transformation          -> ``uaef.adapters.registry.get_adapter``
  * evaluation                    -> ``uaef.api.evaluate`` / ``batch_evaluate``
  * aggregates (average_scores …) -> read from the library persistence layer
                                     (NEVER recomputed here — Req 12.5)

It only marshals requests and shapes the stored Job/UI payloads for transport.

Operations (async ``InvocationType='Event'`` from the API Lambda):
    evaluate | batch_evaluate | invoke_evaluate
Synchronous actions (``InvocationType='RequestResponse'``):
    {"action": "get_metric_catalog"}   -> library catalog
    {"action": "validate_data", ...}   -> library ground-truth validation

Requirements: 5.2, 5.4, 5.5, 6.1, 8.1, 8.2, 8.3, 8.5, 12.4, 12.5, 2.3, 2.4
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from uuid import uuid4

# matplotlib (used by uaef.reporting.generate_report) needs a writable config
# dir and a headless backend inside Lambda. Set before any matplotlib import.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")

# The Worker Lambda imports the published library from the registry-installed
# ``uaef[server]`` package. (The API Lambda must NOT import this.)
import uaef
from uaef.api import evaluate, batch_evaluate
from uaef.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# job_state import (supports both package and flat module layouts)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - layout shim
    import job_state
    from job_state import (
        start_processing,
        complete_job,
        fail_job,
        bound_error_message,
        sanitize_error as _sanitize_error,
    )
    from schemas import MAX_WORKERS_CEILING, MAX_BATCH_ROWS
except ImportError:  # pragma: no cover - layout shim
    from .. import job_state  # type: ignore[no-redef]
    from ..job_state import (  # type: ignore[no-redef]
        start_processing,
        complete_job,
        sanitize_error as _sanitize_error,
        fail_job,
        bound_error_message,
    )
    from ..schemas import MAX_WORKERS_CEILING, MAX_BATCH_ROWS  # type: ignore[no-redef]


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

#: Security review L-04: no default value — see handlers/payloads.py for
#: rationale. The CDK stacks always set this env var explicitly.
PAYLOAD_BUCKET_ENV = "PAYLOAD_BUCKET_NAME"

#: Comma-separated list of additional S3 bucket names (bare names, no
#: ``s3://`` / ARN) that ``invoke_evaluate``'s ``s3_data_path`` may reference,
#: on top of the service payload bucket. Deployers who want callers to be able
#: to point at their own pre-existing data buckets set this explicitly; it is
#: empty by default so an unconfigured deployment only ever reads its own
#: payload bucket (security review H-03 — the previous behavior accepted any
#: ``s3://bucket/key`` in the account, which combined with a wildcard
#: ``s3:GetObject`` IAM grant let any caller read any object the Worker role
#: could reach).
ALLOWED_DATA_BUCKETS_ENV = "UAEF_ALLOWED_DATA_BUCKETS"


def _payload_bucket() -> str:
    bucket = os.environ.get(PAYLOAD_BUCKET_ENV)
    if not bucket:
        raise RuntimeError(
            f"{PAYLOAD_BUCKET_ENV} is not set. This Lambda is misconfigured — "
            "the CDK deployment should always set this environment variable."
        )
    return bucket


def _allowed_data_buckets() -> set:
    """Bucket names ``_read_data_bytes`` may read from, beyond the payload bucket."""
    raw = os.environ.get(ALLOWED_DATA_BUCKETS_ENV, "")
    return {b.strip() for b in raw.split(",") if b.strip()}


#: Security review M-02: opt-in redaction of the persisted/UI-displayed copy
#: of queries, agent responses, and ground-truth answers. Off by default —
#: existing deployments see no change unless a deployer explicitly sets this.
#: Applied only in _build_ui_result, strictly after batch_evaluate has
#: already scored the unredacted text, so enabling it cannot change any score
#: (see the comment at that call site and redact_for_storage's docstring).
PII_REDACTION_ENV = "UAEF_REDACT_PII_ON_PERSIST"


def _pii_redaction_enabled() -> bool:
    return os.environ.get(PII_REDACTION_ENV, "").strip().lower() in {"true", "1", "yes"}


def _redact_pii(text: str) -> str:
    try:
        from uaef.security.pii import redact_for_storage

        return redact_for_storage(text)
    except Exception:  # noqa: BLE001 — redaction must never break persistence
        return text


EXPECTED_VERSION_ENV = "UAEF_EXPECTED_VERSION"
VERSION_MISMATCH_CODE = "VersionMismatch"


def _expected_version() -> Optional[str]:
    value = os.environ.get(EXPECTED_VERSION_ENV)
    return value or None


def _version_mismatch_message() -> Optional[str]:
    expected = _expected_version()
    if expected is None:
        return None
    loaded = getattr(uaef, "__version__", None)
    if loaded == expected:
        return None
    return (
        f"UAEF library version mismatch: the service is pinned to {expected!r} "
        f"but the Worker loaded {loaded!r}. Refusing to run evaluation against "
        f"an unpinned library version."
    )


_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3")
    return _s3_client


# --------------------------------------------------------------------------- #
# Request loading
# --------------------------------------------------------------------------- #


def _load_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Return the request dict, inline (``request``) or fetched from S3 (``requestRef``)."""
    inline = event.get("request")
    if isinstance(inline, dict):
        return inline
    request_ref = event.get("requestRef")
    if request_ref:
        obj = _get_s3().get_object(Bucket=_payload_bucket(), Key=request_ref)
        return json.loads(obj["Body"].read())
    raise ValueError("Worker event contains neither an inline 'request' nor a 'requestRef'.")


# --------------------------------------------------------------------------- #
# Evaluation paths (thin pass-through to the library)
# --------------------------------------------------------------------------- #


def _run_evaluate(req: Dict[str, Any], created_by: Optional[str] = None):
    return evaluate(
        trace=req.get("trace"),
        ground_truth=req.get("ground_truth"),
        adapter=req.get("adapter"),
        metrics=req.get("metrics"),
        context=req.get("context"),
        persist=bool(req.get("persist", False)),
        experiment_name=req.get("experiment_name"),
        experiment_objective=req.get("experiment_objective"),
        created_by=created_by,
    )


def _run_batch_evaluate(req: Dict[str, Any], created_by: Optional[str] = None) -> List[Any]:
    # Security review M-04: max_workers is schema-capped at the API Lambda
    # (schemas.BatchEvaluateRequest), but the Worker re-validates it here too —
    # requests can also reach this function via the Step Functions per-
    # dimension fan-out (orchestrator.py), which passes the request straight
    # through without re-applying the API schema.
    max_workers = min(int(req.get("max_workers", 4)), MAX_WORKERS_CEILING)
    return batch_evaluate(
        traces=req.get("traces"),
        ground_truths=req.get("ground_truths"),
        adapter=req.get("adapter"),
        metrics=req.get("metrics"),
        max_workers=max_workers,
        persist=bool(req.get("persist", False)),
        experiment_name=req.get("experiment_name"),
        experiment_objective=req.get("experiment_objective"),
        created_by=created_by,
    )


# --------------------------------------------------------------------------- #
# Data references (resolve a UI-provided data ref to bytes + filename)
# --------------------------------------------------------------------------- #


def _split_s3(uri: str):
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def _read_data_bytes(data_ref: str) -> bytes:
    """Read a ground-truth file from S3.

    ``data_ref`` is either a full ``s3://bucket/key`` or a bare key uploaded to
    the service payload bucket via the presigned /payloads flow. Reading bytes
    here; parsing is delegated to ``uaef.data``.

    Security (review H-03): a full ``s3://`` URI is only honored when its
    bucket is the service payload bucket or is explicitly allow-listed via
    ``UAEF_ALLOWED_DATA_BUCKETS``. Without this check, any authenticated
    caller could point ``s3_data_path`` at any bucket the Worker's IAM role
    can read — a confused-deputy read of arbitrary account-internal S3
    objects. Bare keys (the presigned-upload flow) are unaffected; they always
    resolve to the payload bucket.
    """
    if data_ref.startswith("s3://"):
        bucket, key = _split_s3(data_ref)
        payload_bucket = _payload_bucket()
        if bucket != payload_bucket and bucket not in _allowed_data_buckets():
            raise ValueError(
                f"Bucket {bucket!r} is not the service payload bucket "
                f"({payload_bucket!r}) and is not in {ALLOWED_DATA_BUCKETS_ENV}. "
                "Upload the file via the presigned /payloads flow, or ask a "
                "deployer to add this bucket to the allow-list."
            )
    else:
        bucket, key = _payload_bucket(), data_ref
    if not bucket or not key:
        raise ValueError(f"Malformed data reference: {data_ref!r}")
    return _get_s3().get_object(Bucket=bucket, Key=key)["Body"].read()


def _filename_from_ref(data_ref: str) -> str:
    return (data_ref.rsplit("/", 1)[-1] or "data.csv")


# --------------------------------------------------------------------------- #
# Agent invocation (delegates to library adapters / invocation helpers)
# --------------------------------------------------------------------------- #


def _group_into_sessions(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group ground-truth rows into conversations, ordered within each.

    Rows are grouped by ``session_id`` (first-seen order preserved, so the report
    follows the file) and sorted by ``turn_id`` within a session. A row missing
    ``session_id`` becomes a session of its own, which is what makes single-turn
    datasets flow through this path unchanged: every row is a one-turn
    conversation, no history is ever accumulated, and behaviour matches the old
    per-row loop exactly.

    ``turn_id`` is sorted numerically when it parses as a number and otherwise
    falls back to file order, so a stray blank or non-numeric turn doesn't
    reorder a conversation into nonsense.
    """
    # Rows are carried as (file_position, row) so ordering never needs to write a
    # bookkeeping key into the row itself — the rows handed back are the caller's
    # own dicts, unmodified, since they go on to build_ground_truth().
    grouped: "OrderedDict[Any, List[tuple]]" = OrderedDict()
    for position, row in enumerate(rows):
        sid = row.get("session_id")
        # Blank/NaN session ids are per-row sessions, keyed uniquely by position.
        if sid is None or (isinstance(sid, float) and sid != sid) or str(sid).strip() == "":
            key: Any = ("__row__", position)
        else:
            key = str(sid)
        grouped.setdefault(key, []).append((position, row))

    def _turn_sort_key(entry: tuple):
        position, row = entry
        raw = row.get("turn_id")
        try:
            return (0, float(raw), position)
        except (TypeError, ValueError):
            return (1, 0.0, position)

    return [
        [row for _, row in sorted(entries, key=_turn_sort_key)]
        for entries in grouped.values()
    ]


def _invoke_agent(
    framework: str,
    conn: Dict[str, Any],
    query: str,
    *,
    session_id: Optional[str] = None,
    turn_id: Optional[Any] = None,
    payload_session_id: Optional[str] = None,
) -> Any:
    """Invoke a reachable agent once via the library, returning its raw output.

    ``session_id`` / ``turn_id`` carry conversation continuity and are only
    forwarded on the HTTP path — the AgentCore, Bedrock-agent and Langfuse
    invokers have no equivalent parameter, so multi-turn against those frameworks
    stays single-turn until they gain one.

    History is never sent: the endpoint is required to keep the conversation
    itself and key on ``session_id``.

    ``session_id`` is what the agent is told; ``payload_session_id`` is what the
    returned payload is stamped with (see the caller for why they differ).
    ``conn["field_map"]`` renames request fields for endpoints using their own names.
    """
    if framework == "agentcore":
        from uaef.adapters import AgentCoreAdapter

        return AgentCoreAdapter.invoke(
            agent_runtime_arn=conn["agent_runtime_arn"],
            user_input=query,
            region=conn.get("region", "us-east-1"),
            bearer_token=conn.get("bearer_token") or None,
            auto_detect_auth=True,
            # Do NOT poll CloudWatch otel logs after each invocation. That path
            # waits up to ~3 minutes PER row for logs to appear (and fails fast
            # only if the log group exists). The deployed service invokes the
            # agent once per ground-truth row inside a 15-minute Lambda, so a
            # 5-row batch would blow the timeout and the job would be killed
            # mid-run (orphaned in PROCESSING). The agent's response text is
            # sufficient for response-quality evaluation here.
            fetch_logs=False,
        )
    if framework == "bedrock":
        from uaef.adapters import invoke_bedrock_agent

        return invoke_bedrock_agent(
            conn["agent_id"], conn["alias_id"], query, conn.get("region", "us-east-1")
        )
    if framework == "langfuse":
        from uaef.adapters import invoke_langfuse_trace

        return invoke_langfuse_trace(
            query, "",
            conn.get("langfuse_public_key", ""),
            conn.get("langfuse_secret_key", ""),
            conn.get("langfuse_host", "https://cloud.langfuse.com"),
        )
    from uaef.adapters import invoke_http_agent_for_framework

    return invoke_http_agent_for_framework(
        conn["endpoint"],
        query,
        framework,
        session_id=session_id,
        turn_id=turn_id,
        field_map=conn.get("field_map") or None,
        payload_session_id=payload_session_id,
    )


#: Message ``type`` values that denote a user turn, in the spellings UAEF accepts
#: (LangChain class names and its short forms).
_USER_MESSAGE_TYPES = {"humanmessage", "human", "user"}


def _has_user_message(events: List[Dict[str, Any]]) -> bool:
    """Whether these stream events already contain a user turn."""
    for event in events or []:
        if not isinstance(event, dict):
            continue
        for node_data in event.values():
            if not isinstance(node_data, dict):
                continue
            messages = node_data.get("messages")
            if not messages:
                continue
            if not isinstance(messages, list):
                messages = [messages]
            for message in messages:
                kind = (
                    message.get("type") if isinstance(message, dict)
                    else type(message).__name__
                )
                if str(kind or "").lower() in _USER_MESSAGE_TYPES:
                    return True
    return False


def _turn_payload(
    raw: Any,
    *,
    query: str,
    session_id: str,
    turn_id: int,
    latency: float,
) -> Dict[str, Any]:
    """Build one per-turn payload for the adapter's multi-turn session path.

    Prepends the user's own message to the turn's events. This is not cosmetic:
    LangGraph's ``updates`` stream mode emits only *node outputs*, so an agent's
    response carries its assistant messages and never the question that prompted
    them. Left alone, the assembled conversation is assistant-only, and every
    full-trace multi-turn metric — which pairs user->assistant exchanges — judges
    a transcript with zero pairs and reports nothing useful.

    Mirrors the notebook's proven shape (``notebooks/00_langgraph.ipynb``), using
    a synthetic ``__human__`` node. Serialized dicts are used rather than
    LangChain objects because the payload has crossed HTTP; the adapter coerces
    either.
    """
    events = list((raw or {}).get("stream_events") or [])

    # Inject only when the agent didn't already include the user's message. Some
    # endpoints echo it (any agent not using LangGraph's "updates" mode may), and
    # prepending unconditionally would put the question in the transcript twice —
    # every conversation-level judge would then read each question double.
    if not _has_user_message(events):
        events = [
            {"__human__": {"messages": [{"type": "HumanMessage", "content": query}]}}
        ] + events

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "turn_id": turn_id,
        "stream_events": events,
        "latency": latency,
    }
    # Carry through node naming so tool-call extraction keeps working; the
    # adapter defaults to "agent"/"tools" otherwise, and this agent uses
    # "assistant"/"tools".
    for key in ("agent_node_name", "tool_node_name"):
        if isinstance(raw, dict) and raw.get(key):
            payload[key] = raw[key]
    return payload


def _agent_response_text(trace: Any) -> str:
    """Read the assistant's response text from a canonical trace (display only).

    Accepts a multi-turn session dict as well as a trace, reading the
    conversation's ``full_trace`` — otherwise a session would display an empty
    response, since a dict has no ``messages``.
    """
    from uaef.models.message import MessageRole

    if isinstance(trace, dict):
        trace = trace.get("full_trace")

    resp = ""
    for msg in getattr(trace, "messages", []) or []:
        if getattr(msg, "role", None) == MessageRole.ASSISTANT:
            resp = msg.content or ""
    return resp


# --------------------------------------------------------------------------- #
# Invoke-and-evaluate (mirrors the demo backend; all heavy lifting in uaef)
# --------------------------------------------------------------------------- #


def _run_invoke_evaluate(req: Dict[str, Any], created_by: Optional[str] = None):
    """Load ground truth, invoke the agent per row, evaluate, shape a UI result.

    Uses the library for every step — ground-truth loading/parsing
    (``uaef.data``), invocation + transform (``uaef.adapters``), evaluation
    (``uaef.api.batch_evaluate``) — and reads aggregate scores from the library
    persistence layer rather than recomputing them (Req 12.5). Returns
    ``(results, ui_result)``.
    """
    from uaef.data import (
        load_ground_truth_file,
        build_ground_truth,
        extract_context_from_trace,
    )
    from uaef.adapters.registry import get_adapter

    framework = req.get("framework") or "generic"
    connection = req.get("connection") or {}
    metric_list = req.get("metrics") or None
    data_ref = req.get("s3_data_path")
    if not data_ref:
        raise ValueError("invoke_evaluate requires 's3_data_path' (a data reference).")

    # Resolve a display S3 URI for the dataset (full s3:// for the UI link).
    dataset_s3_path = data_ref if data_ref.startswith("s3://") else f"s3://{_payload_bucket()}/{data_ref}"
    dataset_name = _filename_from_ref(data_ref)

    # Library-sourced ground-truth loading + parsing. Use the original filename
    # (when provided) for CSV/XLSX format detection, since payload keys have no
    # extension.
    content = _read_data_bytes(data_ref)
    fname = req.get("filename") or _filename_from_ref(data_ref)
    df = load_ground_truth_file(content=content, filename=fname)
    if df.empty:
        raise ValueError("Ground-truth file is empty.")

    # Security review M-04: row count here comes from the uploaded file, not
    # the request body, so schemas.BatchEvaluateRequest's row cap doesn't
    # apply. Each row drives one live agent invocation plus multiple
    # Bedrock-judge calls, so an unbounded file is a wallet-drain DoS vector.
    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(
            f"Ground-truth file has {len(df)} rows, exceeding the "
            f"{MAX_BATCH_ROWS}-row limit for invoke-evaluate. Split the file "
            "into smaller batches."
        )

    adapter = get_adapter(framework)

    traces: List[Any] = []
    ground_truths: List[Any] = []
    queries: List[str] = []
    errors: List[Dict[str, Any]] = []

    # Drive each conversation in order, sending only the session id: the endpoint is
    # required to keep conversation state itself, so a dependent turn ("Book the
    # cheapest hotel.") is answerable because the agent remembers, not because we
    # replay. Turns within a session MUST still run sequentially — turn N's meaning
    # depends on the agent having processed turn N-1 — while separate sessions remain
    # independent. Datasets without session_id are one-turn conversations.
    from uaef.data import build_multi_turn_ground_truth

    sessions = _group_into_sessions(df.to_dict(orient="records"))

    # Two separate questions, previously conflated:
    #
    #   has_conversations — does the data contain multi-turn conversations? This
    #       drives sequential invocation against one session id, which is what makes
    #       a dependent turn ("Book the cheapest hotel.") answerable. It applies to
    #       every framework, because the sequencing happens on our side of the wire.
    #
    #   use_session_eval — can this adapter assemble a conversation? Only adapters
    #       implementing the list form of transform_to_canonical can return
    #       {"session_id", "per_turn_traces", "full_trace"}, which conversation-level
    #       metrics need. Handing a list to an adapter that expects a dict fails in
    #       whatever way that adapter happens to fail.
    #
    # So a Strands or Bedrock agent with session-shaped data still gets a coherent
    # conversation driven against it, and is then scored per turn rather than per
    # conversation, instead of erroring out.
    has_conversations = any(len(s) > 1 for s in sessions)
    use_session_eval = has_conversations and adapter.supports_session_transform()
    if has_conversations and not use_session_eval:
        logger.info(
            "Data contains multi-turn conversations but the '%s' adapter cannot "
            "assemble sessions; driving turns sequentially and evaluating per turn. "
            "Conversation-level metrics will not be available.",
            framework,
        )

    # Per-turn payloads for the adapter's session path, kept alongside the
    # per-turn traces so either evaluation shape can be built from one drive of
    # the conversations.
    turn_payloads: List[Dict[str, Any]] = []
    session_rows_by_id: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    if has_conversations:
        logger.info("Multi-turn run: %d conversations", len(sessions))

    row_index = 0
    for session_index, session_rows in enumerate(sessions):
        session_id = session_rows[0].get("session_id")
        session_id = str(session_id) if session_id is not None else f"session_{session_index + 1}"

        # Two identifiers, deliberately different.
        #
        # agent_session_id is what the agent is told, and is fresh for every run.
        # Reusing the dataset's literal id against a stateful endpoint would make a
        # second run of the same file continue the first run's conversation: scores
        # drift for reasons invisible in the data, and re-running stops being
        # reproducible. For our own agent reset_db() provides that isolation; a
        # customer's state can't be reset, so it has to come from a new id.
        #
        # session_id (the dataset's) stays the grouping and reporting key, so the
        # adapter groups turns correctly and the UI label reads "session_001".
        agent_session_id = str(uuid4())

        session_rows_by_id[session_id] = session_rows

        for turn_position, row in enumerate(session_rows):
            row_index += 1
            query, gt = build_ground_truth(row)
            queries.append(query)
            started = time.time()
            try:
                raw = _invoke_agent(
                    framework,
                    connection,
                    query,
                    session_id=agent_session_id,
                    turn_id=row.get("turn_id"),
                    payload_session_id=session_id,
                )
                trace = adapter.transform_to_canonical(raw)
                # Ground the hallucination check on what the agent actually
                # retrieved. Datasets rarely carry a `context` column, and
                # without context that metric can only report itself
                # unavailable. Tool output is the closest thing to a source of
                # truth the run produces, so it stands in when the file has no
                # context of its own — never overriding a context column.
                if not gt.context_documents:
                    gt.context_documents = extract_context_from_trace(trace)
                traces.append(trace)
                ground_truths.append(gt)

                if use_session_eval:
                    turn_payloads.append(
                        _turn_payload(
                            raw,
                            query=query,
                            session_id=session_id,
                            turn_id=turn_position + 1,
                            latency=time.time() - started,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 — record per-row invoke failures
                # Security review M-03: per-row errors are persisted verbatim and
                # rendered in the UI. A failure invoking boto3 (AgentCore/Bedrock)
                # can surface IAM/role detail beyond the caller-supplied ARN, so
                # sanitize the same as the other service error paths.
                safe_message, correlation_id = _sanitize_error(
                    exc, context="Agent invocation failed"
                )
                errors.append({
                    "index": row_index,
                    "query": query,
                    "error": safe_message,
                    "correlationId": correlation_id,
                })
                # Nothing to carry forward: the agent owns the conversation, so a
                # failed turn is simply absent from its state. Later turns in this
                # session may therefore reference something that was never said.

    # Multi-turn datasets are evaluated per conversation, not per row: the
    # library's session-dict path (batch_evaluate -> evaluate_multi_turn) runs
    # full-trace metrics once over the whole conversation AND per-turn metrics on
    # each turn, averaging them with each turn's detail preserved in
    # metadata["per_turn"]. So conversation-level scores and per-turn tool
    # correctness both come out of a single pass — see
    # docs/docs/Advanced/multi-turn-metrics.md.
    #
    # Single-turn datasets keep the per-row path untouched. Routing them through
    # evaluate_multi_turn would technically work (a one-turn session) but would
    # silently change which metrics run and how they aggregate for every existing
    # user, so the switch is gated on the data actually having conversations.
    sessions_meta: List[Dict[str, Any]] = []
    if use_session_eval and turn_payloads:
        sessions_by_id = {
            s["session_id"]: s
            for s in adapter.transform_to_canonical(turn_payloads)
        }
        session_traces: List[Any] = []
        session_gts: List[Any] = []
        for session_id, rows_in_session in session_rows_by_id.items():
            session_dict = sessions_by_id.get(session_id)
            if session_dict is None:
                # Every turn in this conversation failed to invoke.
                continue
            session_traces.append(session_dict)
            session_gt = build_multi_turn_ground_truth(rows_in_session)
            # Same fallback as the per-turn path, but pooled across the whole
            # conversation: a claim in turn 5 may rest on what a tool returned
            # in turn 2, so per-turn context alone would look unsupported.
            if not session_gt.context_documents:
                pooled: List[str] = []
                for turn_trace in (session_dict.get("per_turn_traces") or []):
                    pooled.extend(extract_context_from_trace(turn_trace))
                session_gt.context_documents = pooled
            session_gts.append(session_gt)
            sessions_meta.append({
                "session_id": session_id,
                "queries": [str(r.get("query") or "") for r in rows_in_session],
            })

        if session_traces:
            traces, ground_truths = session_traces, session_gts

    if not traces:
        raise RuntimeError(
            f"All {len(queries)} agent invocations failed. First error: "
            f"{errors[0]['error'] if errors else 'unknown'}"
        )

    # Persist so the library computes + stores aggregates (read back below).
    results = list(
        batch_evaluate(
            traces=traces,
            ground_truths=ground_truths,
            metrics=metric_list,
            max_workers=4,
            persist=bool(req.get("persist", True)),
            experiment_name=req.get("experiment_name") or "Ad-hoc evaluation",
            experiment_objective=req.get("experiment_objective"),
            created_by=created_by,
        )
    )

    # Annotate the persisted experiment with service metadata (framework, agent,
    # dataset) so the Experiments/Overview lists can show the agent and link to
    # the dataset. Mirrors the demo backend; best-effort.
    exp_id = next((str(getattr(r, "experiment_id")) for r in results if getattr(r, "experiment_id", None)), None)
    if exp_id:
        _annotate_experiment(
            exp_id,
            framework=framework,
            agent_name=req.get("agent_name") or framework,
            dataset_name=dataset_name,
            dataset_s3_path=dataset_s3_path,
        )

    ui_result = _build_ui_result(
        req, framework, queries, traces, ground_truths, results, errors,
        sessions_meta=sessions_meta,
    )
    return results, ui_result


def _annotate_experiment(exp_id: str, **metadata: Any) -> None:
    """Set service metadata on a persisted experiment record (best-effort)."""
    try:
        from uaef.storage.dynamodb_s3 import get_storage

        get_storage()._table.update_item(
            Key={"experiment_id": exp_id},
            UpdateExpression="SET metadata = :m",
            ExpressionAttributeValues={":m": {k: v for k, v in metadata.items() if v}},
        )
    except Exception:  # noqa: BLE001 — annotation is non-critical
        pass


def _build_ui_result(
    req, framework, queries, traces, ground_truths, results, errors, sessions_meta=None
):
    """Shape results for the UI. Per-row values are READ from library results;
    aggregate scores are READ from the library persistence layer (never recomputed).

    ``sessions_meta`` marks a multi-turn run, where one result covers a whole
    conversation rather than a single query. Row cardinality then follows
    conversations, not spreadsheet rows, so it carries the label and turn queries
    per session — indexing ``queries`` positionally would otherwise pair session 2
    with turn 2's question.
    """
    metric_list = req.get("metrics") or None
    is_multi_turn = bool(sessions_meta)

    # Which dimension each scored metric actually belongs to, taken from the
    # library's DimensionResult rather than guessed. The UI used to infer the
    # dimension from substrings in the metric name, which filed 6 of the 12
    # Multi-Turn metrics under "Reasoning" and put `agent_tone` under
    # "Multi-Agent". Grouping (not re-averaging) keeps Req 12.5 intact.
    dimension_metrics: Dict[str, List[str]] = {}
    # Metrics that were selected but could not produce a score.
    unavailable_all: Dict[str, Dict[str, str]] = {}

    # Security review M-02: opt-in (UAEF_REDACT_PII_ON_PERSIST=true). Applied
    # strictly here, on the already-scored UI-display copy — the judge already
    # evaluated the unredacted traces/ground_truths (batch_evaluate runs before
    # _build_ui_result is called), so this cannot change any score. See
    # uaef.security.pii.redact_for_storage's docstring for the same constraint
    # stated at the primitive level.
    #
    # Every display string below goes through _display(), including the per-turn
    # breakdown: those turn queries/responses/expected answers are persisted and
    # rendered exactly like the conversation-level ones, so redacting only the
    # top-level fields would leave PII in the payload.
    _redact = _pii_redaction_enabled()

    def _display(text: str) -> str:
        return _redact_pii(text) if _redact else text

    rows: List[Dict[str, Any]] = []
    for idx, r in enumerate(results):
        dim_scores: Dict[str, float] = {}
        metric_scores: Dict[str, float] = {}
        unavailable: Dict[str, Dict[str, str]] = {}
        per_turn: Dict[str, Any] = {}
        for dim in getattr(r, "dimension_results", []) or []:
            dim_scores[dim.dimension_name] = float(dim.aggregate_score)
            for ms in getattr(dim, "metric_scores", []) or []:
                if metric_list and ms.metric_name not in metric_list:
                    continue
                if ms.score is not None:
                    metric_scores[ms.metric_name] = float(ms.score)
                    bucket = dimension_metrics.setdefault(dim.dimension_name, [])
                    if ms.metric_name not in bucket:
                        bucket.append(ms.metric_name)
                else:
                    # A metric that *could not run* (judge error, missing
                    # prerequisite) must not be silently dropped. Doing so made
                    # a Bedrock access denial look like "the report only has 2
                    # metrics", indistinguishable from a metric-selection bug.
                    ms_meta = ms.metadata or {}
                    detail = {
                        "warning": str(ms_meta.get("warning") or "unavailable"),
                        "reason": str(ms.reasoning or ms_meta.get("error_details") or ""),
                        "dimension": str(dim.dimension_name),
                    }
                    unavailable[ms.metric_name] = detail
                    unavailable_all.setdefault(ms.metric_name, detail)
                # Per-turn metrics are averaged into one score; keep each turn's
                # breakdown so a conversation row can be expanded to see which
                # turn failed.
                turn_detail = (ms.metadata or {}).get("per_turn")
                if turn_detail:
                    per_turn[ms.metric_name] = turn_detail

        gt = ground_truths[idx] if idx < len(ground_truths) else None
        meta = sessions_meta[idx] if is_multi_turn and idx < len(sessions_meta) else None

        if meta:
            turn_queries = meta.get("queries") or []
            # The label is a session id plus a turn count, not user text, so it
            # carries no PII of its own and is left unredacted.
            label = "%s (%d turns)" % (meta.get("session_id", "session"), len(turn_queries))
        else:
            turn_queries = []
            label = _display(queries[idx] if idx < len(queries) else "")

        # Per-turn breakdown. A conversation row otherwise shows only its final
        # response, which hides where things actually went wrong: a 6-turn
        # session scoring 0 on tool selection says nothing about *which* turn.
        turns: List[Dict[str, Any]] = []
        if meta:
            session_dict = traces[idx] if idx < len(traces) else {}
            per_turn_traces = (session_dict or {}).get("per_turn_traces") or []
            expected_per_turn = (
                (getattr(gt, "expected_arguments", None) or {}).get("per_turn_expected") or []
            )
            # Invert metric -> [{turn, score, ...}] into turn -> {metric: score},
            # so each turn carries its own scores.
            scores_by_turn: Dict[int, Dict[str, Any]] = {}
            for metric_name, entries in per_turn.items():
                for entry in entries or []:
                    turn_no = entry.get("turn")
                    if turn_no is None:
                        continue
                    bucket = scores_by_turn.setdefault(int(turn_no), {})
                    if entry.get("score") is not None:
                        bucket[metric_name] = float(entry["score"])

            for turn_no in range(1, max(len(turn_queries), len(per_turn_traces)) + 1):
                turn_trace = per_turn_traces[turn_no - 1] if turn_no <= len(per_turn_traces) else None
                turns.append({
                    "turn": turn_no,
                    "query": _display(
                        turn_queries[turn_no - 1] if turn_no <= len(turn_queries) else ""
                    ),
                    "response": _display(
                        _agent_response_text(turn_trace)[:2000] if turn_trace else ""
                    ),
                    "expected": _display(str(
                        expected_per_turn[turn_no - 1]
                        if turn_no <= len(expected_per_turn) else ""
                    )[:2000]),
                    "tool_calls": [
                        getattr(tc, "name", "") for tc in (getattr(turn_trace, "tool_calls", []) or [])
                    ],
                    "metric_scores": scores_by_turn.get(turn_no, {}),
                })

        row: Dict[str, Any] = {
            "index": idx + 1,
            "query": label,
            "agent_response": _display(
                _agent_response_text(traces[idx])[:500] if idx < len(traces) else ""
            ),
            "expected_answer": _display(str(getattr(gt, "expected_output", "") or "")[:500]),
            "overall_score": float(getattr(r, "overall_score", 0.0) or 0.0),
            "passed": bool(getattr(r, "passed", False)),
            "dimension_scores": dim_scores,
            "metric_scores": metric_scores,
        }
        if meta:
            row["session_id"] = meta.get("session_id")
            row["turn_count"] = len(turn_queries)
            row["turns"] = turns
        if per_turn:
            row["per_turn"] = per_turn
        if unavailable:
            row["unavailable_metrics"] = unavailable
        rows.append(row)

    # Aggregates: read from the persistence layer (Req 12.5 — never recomputed).
    exp_id = next((str(getattr(r, "experiment_id")) for r in results if getattr(r, "experiment_id", None)), None)
    average_scores: Dict[str, float] = {}
    overall_avg = 0.0
    if exp_id:
        try:
            from uaef.storage.dynamodb_s3 import get_storage

            stored = get_storage().get_experiment(exp_id) or {}
            average_scores = {k: float(v) for k, v in (stored.get("average_scores") or {}).items()}
            overall_avg = float(stored.get("overall_average_score") or 0.0)
        except Exception:  # noqa: BLE001 — aggregates are best-effort for display
            pass

    return {
        "experiment_id": exp_id,
        "experiment_name": req.get("experiment_name") or "Ad-hoc evaluation",
        "experiment_objective": req.get("experiment_objective"),
        "framework": framework,
        "agent_name": req.get("agent_name") or framework,
        "evaluation_count": len(results),
        "total_queries": len(queries),
        # On a multi-turn run one evaluation covers many turns, so successes are
        # counted in turns (queries that ran) rather than in results, which would
        # report "2 of 9 succeeded" for two fully successful conversations.
        "successful_queries": (len(queries) - len(errors)) if is_multi_turn else len(results),
        "failed_queries": len(errors),
        "multi_turn": is_multi_turn,
        "session_count": len(sessions_meta or []),
        "overall_average_score": overall_avg,
        "average_scores": average_scores,
        # Real dimension -> metric grouping, so the UI can build the radar from
        # library truth instead of keyword-matching metric names.
        "dimension_metrics": {k: sorted(v) for k, v in dimension_metrics.items()},
        "unavailable_metrics": unavailable_all,
        "rows": rows,
        "errors": errors,
        "persist": True,
    }


# --------------------------------------------------------------------------- #
# Sync action: ground-truth validation (delegated to uaef.data)
# --------------------------------------------------------------------------- #


def _list_experiments(created_by: Optional[str] = None) -> Dict[str, Any]:
    """List the caller's own persisted experiments via the library storage layer.

    Security review H-02: ``created_by`` (when provided by the API Lambda)
    scopes the listing to that owner via the storage layer's ``created_by``
    GSI query — never a full unscoped scan. Experiments created before this
    fix (no ``created_by`` recorded) are never returned here.

    Sourced entirely from ``uaef.storage`` (the same store the library writes to
    on persist) — the service reshapes records for transport but computes
    nothing. Mirrors the demo backend's /experiments.
    """
    try:
        from uaef.storage.dynamodb_s3 import get_storage

        experiments = get_storage().list_experiments(created_by=created_by)
    except Exception as exc:  # noqa: BLE001 — surface as empty + warning
        # Security review M-03: don't return storage-layer exception detail.
        safe_message, correlation_id = _sanitize_error(
            exc, context="Failed to list experiments"
        )
        return {"experiments": [], "error": safe_message, "correlationId": correlation_id}

    out: List[Dict[str, Any]] = []
    for exp in experiments:
        meta = exp.get("metadata", {}) if isinstance(exp.get("metadata"), dict) else {}
        out.append({
            "experiment_id": exp.get("experiment_id", ""),
            "experiment_name": exp.get("experiment_name", ""),
            "experiment_objective": exp.get("experiment_objective", ""),
            "framework": meta.get("framework", "—"),
            "agent_name": meta.get("agent_name", ""),
            "dataset_name": meta.get("dataset_name", "—"),
            "dataset_s3_path": meta.get("dataset_s3_path", ""),
            "evaluation_count": int(exp.get("evaluation_count", 0) or 0),
            "overall_average_score": float(exp.get("overall_average_score", 0) or 0),
            "result_path": exp.get("result_path", ""),
            "created_at": exp.get("created_at", ""),
        })
    return {"experiments": out}


def _get_experiment(event: Dict[str, Any]) -> Dict[str, Any]:
    """Return the full detail of a single experiment by ID, authorized by owner.

    Security review H-02: when ``created_by`` is present in the event (the API
    Lambda always sets it from the caller's Cognito sub), the experiment's own
    ``created_by`` must match or this returns ``{"code": "FORBIDDEN"}`` with no
    experiment detail — the API Lambda maps that to a 403 disclosing nothing
    further, whether the experiment doesn't exist or just isn't the caller's.

    Fetches the experiment metadata and full evaluation results from the library
    storage layer (``uaef.storage``). Mirrors the UI's experiment detail view:
    metadata, per-row scores, average scores, and traces.
    """
    experiment_id = event.get("experiment_id")
    if not experiment_id:
        return {"error": "experiment_id is required"}
    caller_created_by = event.get("created_by")

    try:
        from uaef.storage.dynamodb_s3 import get_storage

        store = get_storage()
        exp = store.get_experiment(experiment_id)
        if not exp:
            return {"error": f"Experiment not found: {experiment_id}"}
        if caller_created_by and exp.get("created_by") != caller_created_by:
            return {"code": "FORBIDDEN"}

        # Get full results (per-row evaluations)
        full = store.get_full_results(experiment_id)
        evaluations = full.get("evaluations", []) if isinstance(full, dict) else []

        meta = exp.get("metadata", {}) if isinstance(exp.get("metadata"), dict) else {}
        average_scores = {k: float(v) for k, v in (exp.get("average_scores") or {}).items()}

        return {
            "experiment_id": exp.get("experiment_id", experiment_id),
            "experiment_name": exp.get("experiment_name", ""),
            "experiment_objective": exp.get("experiment_objective", ""),
            "framework": meta.get("framework", ""),
            "agent_name": meta.get("agent_name", ""),
            "dataset_name": meta.get("dataset_name", ""),
            "dataset_s3_path": meta.get("dataset_s3_path", ""),
            "evaluation_count": int(exp.get("evaluation_count", 0) or 0),
            "overall_average_score": float(exp.get("overall_average_score", 0) or 0),
            "average_scores": average_scores,
            "created_at": exp.get("created_at", ""),
            "evaluations": evaluations,
        }
    except Exception as exc:  # noqa: BLE001
        # Security review M-03: don't return storage-layer exception detail.
        safe_message, correlation_id = _sanitize_error(
            exc, context="Failed to retrieve experiment detail"
        )
        return {"error": safe_message, "correlationId": correlation_id}


def _inline_report_markdown(report_dir: str) -> str:
    """Read the generated report.md and inline its PNG images as base64 data URIs.

    ``generate_report`` writes Markdown with relative ``images/*.png`` links. We
    inline them so the report is a single self-contained Markdown document the
    browser can render with no extra asset hosting.
    """
    import base64
    import re

    md_path = os.path.join(report_dir, "report.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    def _repl(m):
        alt, rel = m.group(1), m.group(2)
        try:
            with open(os.path.join(report_dir, rel), "rb") as img:
                b64 = base64.b64encode(img.read()).decode("ascii")
            return f"![{alt}](data:image/png;base64,{b64})"
        except Exception:  # noqa: BLE001 — keep the original link if missing
            return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\((images/[^)]+)\)", _repl, md)


def _generate_report_action(event: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the library report for an experiment and return a viewable URL.

    Security review H-02: when ``created_by`` is present in the event, the
    experiment's own ``created_by`` must match or this returns
    ``{"code": "FORBIDDEN"}`` with no report generated and no experiment
    detail disclosed.

    Delegates entirely to ``uaef.reporting.generate_report`` (the same call the
    reporting-pipeline notebook uses): loads the persisted ``EvaluationResult``s
    from the library store, generates the Markdown report + charts, inlines the
    images, uploads the self-contained Markdown to S3, and returns a presigned
    URL the UI fetches and renders.
    """
    experiment_id = event.get("experiment_id")
    report_type = event.get("report_type") or "full"
    if not experiment_id:
        return {"error": "experiment_id is required"}
    caller_created_by = event.get("created_by")

    try:
        from uaef.storage.dynamodb_s3 import get_storage
        from uaef.reporting import generate_report
        from uaef.models.evaluation_result import EvaluationResult

        store = get_storage()
        exp = store.get_experiment(experiment_id) or {}
        if not exp:
            return {"error": f"Experiment not found: {experiment_id}"}
        if caller_created_by and exp.get("created_by") != caller_created_by:
            return {"code": "FORBIDDEN"}
        full = store.get_full_results(experiment_id)
        raw = full.get("evaluations", []) if isinstance(full, dict) else []
        results = []
        for d in raw:
            try:
                results.append(EvaluationResult.model_validate(d))
            except Exception:  # noqa: BLE001 — skip any malformed record
                # The Reducer stores UI-shaped rows; try to construct a
                # minimal EvaluationResult from them for report generation.
                try:
                    if isinstance(d, dict) and "metric_scores" in d and d["metric_scores"]:
                        from uaef.models.dimension_result import DimensionResult
                        from uaef.models.metric_score import MetricScore
                        from uuid import uuid4 as _uuid4
                        dim_results = []
                        for dim_name, dim_score in (d.get("dimension_scores") or {}).items():
                            dim_metrics = []
                            for m_name, m_score in (d.get("metric_scores") or {}).items():
                                dim_metrics.append(MetricScore(metric_name=m_name, score=m_score))
                            if dim_metrics:
                                dim_results.append(DimensionResult(
                                    dimension_name=dim_name,
                                    aggregate_score=float(dim_score) if dim_score else 0.0,
                                    metric_scores=dim_metrics,
                                    weight=1.0,
                                ))
                        if dim_results:
                            results.append(EvaluationResult(
                                trace_id=_uuid4(),
                                overall_score=float(d.get("overall_score") or 0.0),
                                passed=bool(d.get("passed", False)),
                                dimension_results=dim_results,
                            ))
                except Exception:  # noqa: BLE001
                    continue
        if not results:
            return {"error": f"No evaluation results found for experiment {experiment_id}."}

        title = exp.get("experiment_name") or f"Experiment {str(experiment_id)[:8]}"
        report_dir = generate_report(
            results,
            title=title,
            report_type=report_type if report_type in ("full", "executive") else "full",
            output_dir=os.path.join(tempfile.gettempdir(), "uaef-reports"),
        )

        md = _inline_report_markdown(report_dir)
        key = f"reports/{experiment_id}.md"
        _get_s3().put_object(
            Bucket=_payload_bucket(),
            Key=key,
            Body=md.encode("utf-8"),
            ContentType="text/markdown",
        )
        url = _get_s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": _payload_bucket(), "Key": key},
            ExpiresIn=900,
        )
        return {"reportUrl": url, "title": title}
    except Exception as exc:  # noqa: BLE001
        # Security review M-03: don't return exception detail (this handler
        # touches S3 keys/bucket names and the storage layer).
        safe_message, correlation_id = _sanitize_error(
            exc, context="Failed to generate report"
        )
        return {"error": safe_message, "correlationId": correlation_id}


def _compare_experiments_action(event: Dict[str, Any]) -> Dict[str, Any]:
    """Compare multiple experiments by their per-metric average scores.

    Security review H-02: when ``created_by`` is present in the event, every
    named experiment's own ``created_by`` must match or this returns
    ``{"code": "FORBIDDEN"}`` for the whole request with no experiment detail
    disclosed — a caller can't tell "doesn't exist" from "exists but isn't
    yours" by which experiments made it into a partial response, because
    there is no partial response: any mismatch rejects the entire comparison.

    Loads each experiment's stored average_scores from the library storage layer
    and builds a comparison table the UI renders (overall scores, per-metric
    breakdown, spread, and insights).

    Returns a shape the ComparePanel expects:
        {
            "experiments": [{"id", "name", "overall_score", "agent_name", "framework"}],
            "table": [{"metric", "exp_0", "exp_1", ..., "spread", "min", "max"}],
            "insights": ["string", ...]
        }
    """
    experiment_ids = event.get("experiment_ids") or []
    if len(experiment_ids) < 2:
        return {"error": "At least 2 experiment_ids are required."}
    caller_created_by = event.get("created_by")

    try:
        from uaef.storage.dynamodb_s3 import get_storage

        store = get_storage()
    except Exception as exc:  # noqa: BLE001
        # Security review M-03: don't return storage-layer exception detail.
        safe_message, correlation_id = _sanitize_error(
            exc, context="Storage unavailable"
        )
        return {"error": safe_message, "correlationId": correlation_id}

    # Load experiment records.
    experiments_meta: List[Dict[str, Any]] = []
    all_metric_names: set = set()

    for exp_id in experiment_ids:
        exp = store.get_experiment(exp_id)
        if not exp:
            return {"error": f"Experiment not found: {exp_id}"}
        if caller_created_by and exp.get("created_by") != caller_created_by:
            return {"code": "FORBIDDEN"}
        avg_scores = exp.get("average_scores") or {}
        # Convert Decimal values to float.
        avg_scores = {k: float(v) for k, v in avg_scores.items()}
        meta = exp.get("metadata", {}) if isinstance(exp.get("metadata"), dict) else {}
        experiments_meta.append({
            "id": exp_id,
            "name": exp.get("experiment_name") or exp_id[:8],
            "overall_score": float(exp.get("overall_average_score") or 0),
            "agent_name": meta.get("agent_name") or "",
            "framework": meta.get("framework") or "",
            "average_scores": avg_scores,
        })
        all_metric_names.update(avg_scores.keys())

    # Build the comparison table: one row per metric, one column per experiment.
    table: List[Dict[str, Any]] = []
    for metric in sorted(all_metric_names):
        row: Dict[str, Any] = {"metric": metric}
        values: List[float] = []
        for i, exp_meta in enumerate(experiments_meta):
            val = exp_meta["average_scores"].get(metric)
            row[f"exp_{i}"] = val
            if val is not None:
                values.append(val)
        row["min"] = min(values) if values else 0
        row["max"] = max(values) if values else 0
        row["spread"] = (max(values) - min(values)) if len(values) > 1 else 0
        table.append(row)

    # Generate insights.
    insights = _generate_comparison_insights(experiments_meta, table)

    # Build the response (strip average_scores from experiment objects).
    experiments_out = [
        {k: v for k, v in exp.items() if k != "average_scores"}
        for exp in experiments_meta
    ]

    return {
        "experiments": experiments_out,
        "table": table,
        "insights": insights,
    }


def _generate_comparison_insights(
    experiments: List[Dict[str, Any]], table: List[Dict[str, Any]]
) -> List[str]:
    """Generate human-readable insights from the comparison."""
    insights: List[str] = []

    if not experiments:
        return insights

    # Best overall performer.
    best = max(experiments, key=lambda e: e["overall_score"])
    worst = min(experiments, key=lambda e: e["overall_score"])
    if best["overall_score"] > worst["overall_score"]:
        insights.append(
            f'"{best["name"]}" has the highest overall score '
            f'({best["overall_score"]*100:.1f}%), '
            f'{(best["overall_score"] - worst["overall_score"])*100:.1f}pp ahead of '
            f'"{worst["name"]}" ({worst["overall_score"]*100:.1f}%).'
        )

    # Most variable metrics (highest spread).
    variable = sorted(table, key=lambda r: r.get("spread", 0), reverse=True)
    top_variable = [r for r in variable[:3] if r.get("spread", 0) > 0.05]
    if top_variable:
        names = ", ".join(r["metric"] for r in top_variable)
        insights.append(
            f"Metrics with the most variation across experiments: {names}."
        )

    # Metrics where all experiments agree (low spread).
    consistent = [r for r in table if r.get("spread", 0) < 0.02 and r.get("max", 0) > 0]
    if consistent:
        if len(consistent) <= 3:
            names = ", ".join(r["metric"] for r in consistent)
            insights.append(f"Consistent across all experiments: {names}.")
        else:
            insights.append(
                f"{len(consistent)} metrics show minimal variation (< 2pp spread) "
                f"across experiments."
            )

    # Per-experiment strengths.
    for i, exp in enumerate(experiments):
        best_metrics = []
        for row in table:
            val = row.get(f"exp_{i}")
            if val is not None and val == row.get("max") and row.get("spread", 0) > 0.05:
                best_metrics.append(row["metric"])
        if best_metrics and len(best_metrics) <= 5:
            insights.append(
                f'"{exp["name"]}" leads in: {", ".join(best_metrics[:5])}.'
            )

    return insights


def _validate_data(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a ground-truth file using the library and return a preview."""
    from uaef.data import load_ground_truth_file, validate_ground_truth

    data_ref = event.get("data_ref")
    if not data_ref:
        return {"valid": False, "error": "No data reference provided."}
    try:
        content = _read_data_bytes(data_ref)
        fname = event.get("filename") or _filename_from_ref(data_ref)
        df = load_ground_truth_file(content=content, filename=fname)
        result = validate_ground_truth(df, filename=fname).model_dump()

        # Report the conversation shape the run will actually use. Computed with
        # the same _group_into_sessions the evaluation calls, so what the user is
        # told at validation time cannot disagree with what happens later — the
        # grouping is inferred from session_id, and nothing else in the UI reveals
        # whether a file is multi-turn.
        sessions = _group_into_sessions(df.to_dict(orient="records"))
        result["session_count"] = len(sessions)
        result["multi_turn"] = any(len(s) > 1 for s in sessions)
        return result
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Result shaping for the Job record (read-only summary)
# --------------------------------------------------------------------------- #


def _experiment_id_of(result: Any) -> Optional[str]:
    exp = getattr(result, "experiment_id", None)
    return str(exp) if exp is not None else None


def _summarize_one(result: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if hasattr(result, "overall_score"):
        summary["overallScore"] = getattr(result, "overall_score")
    if hasattr(result, "passed"):
        summary["passed"] = getattr(result, "passed")
    trace_id = getattr(result, "trace_id", None)
    if trace_id is not None:
        summary["traceId"] = str(trace_id)
    return summary


def _build_outcome(operation: str, result: Any) -> Dict[str, Any]:
    if operation in ("batch_evaluate", "invoke_evaluate"):
        results: List[Any] = list(result) if result is not None else []
        experiment_id = next(
            (eid for eid in (_experiment_id_of(r) for r in results) if eid is not None),
            None,
        )
        summary = {"count": len(results), "results": [_summarize_one(r) for r in results]}
        return {"experimentId": experiment_id, "resultSummary": summary}
    return {"experimentId": _experiment_id_of(result), "resultSummary": _summarize_one(result)}


# --------------------------------------------------------------------------- #
# Result storage (full result JSON -> S3)
# --------------------------------------------------------------------------- #


def _store_result(job_id: str, result: Any) -> Optional[str]:
    try:  # pragma: no cover - thin delegation
        try:
            from result_store import _store_result as _impl  # type: ignore
        except ImportError:
            from .result_store import _store_result as _impl  # type: ignore
        return _impl(job_id, result)
    except ImportError:
        pass
    key = f"results/{job_id}.json"
    _get_s3().put_object(
        Bucket=_payload_bucket(),
        Key=key,
        Body=_serialize_result(result).encode("utf-8"),
        ContentType="application/json",
    )
    return key


def _store_ui_result(job_id: str, ui_result: Dict[str, Any]) -> str:
    """Persist the UI-shaped result JSON to S3 (always, for the UI's resultRef)."""
    key = f"results/{job_id}.json"
    _get_s3().put_object(
        Bucket=_payload_bucket(),
        Key=key,
        Body=json.dumps(ui_result, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return key


def _serialize_result(result: Any) -> str:
    def _dump(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        return obj

    if isinstance(result, list):
        return json.dumps([_dump(r) for r in result], default=str)
    return json.dumps(_dump(result), default=str)


# --------------------------------------------------------------------------- #
# Partial-dimension evaluation (invoked by Step Functions Map state)
# --------------------------------------------------------------------------- #


def _run_partial_dimension(event: Dict[str, Any], operation: str) -> Any:
    """Run evaluation for a single dimension's metrics (no job lifecycle).

    When the Step Functions orchestrator fans out by dimension, each Worker
    invocation receives ``isPartialDimension=True`` and a request with only that
    dimension's metrics. The Worker evaluates and returns the raw result — the
    Reducer handles job lifecycle and persistence.
    """
    request = event.get("request") or {}
    request_ref = event.get("requestRef")
    job_id = event.get("jobId", "")
    dimension_name = event.get("dimensionName", "unknown")

    # If request comes via S3 ref, fetch it.
    if not request and request_ref:
        obj = _get_s3().get_object(Bucket=_payload_bucket(), Key=request_ref)
        request = json.loads(obj["Body"].read())

    # Version-mismatch guard (still enforced per-dimension for safety).
    mismatch = _version_mismatch_message()
    if mismatch is not None:
        raise RuntimeError(mismatch)

    # Update job progress: mark this dimension as "running".
    _update_dimension_progress(job_id, dimension_name, "running")

    try:
        if operation == "invoke_evaluate":
            # invoke_evaluate in partial mode: agent invocation + evaluation with
            # only this dimension's metrics. Returns the UI-shaped dict.
            _result, ui_result = _run_invoke_evaluate(request)
            result = ui_result
        elif operation == "batch_evaluate":
            result_list = _run_batch_evaluate(request)
            # Serialize for the Map state output.
            result = [_serialize_eval_result(r) for r in result_list]
        else:
            eval_result = _run_evaluate(request)
            result = _serialize_eval_result(eval_result)

        # Update job progress: mark this dimension as "done".
        _update_dimension_progress(job_id, dimension_name, "done")
        return result
    except Exception:
        # Update job progress: mark this dimension as "failed".
        _update_dimension_progress(job_id, dimension_name, "failed")
        raise


def _update_dimension_progress(job_id: str, dimension_name: str, status: str) -> None:
    """Update the job record's progress field with dimension status.

    Best-effort: failures here don't affect the evaluation outcome.
    The progress field is a map: {"dimensions": {"DimName": "running|done|failed"}}
    """
    if not job_id:
        return
    try:
        from job_state import _get_table
    except ImportError:
        try:
            from ..job_state import _get_table
        except ImportError:
            return
    try:
        _get_table().update_item(
            Key={"jobId": job_id},
            UpdateExpression="SET progress.dimensions.#dim = :status",
            ExpressionAttributeNames={"#dim": dimension_name},
            ExpressionAttributeValues={":status": status},
        )
    except Exception:  # noqa: BLE001 — progress is best-effort
        # If progress map doesn't exist yet, create it.
        try:
            _get_table().update_item(
                Key={"jobId": job_id},
                UpdateExpression="SET progress = :prog",
                ExpressionAttributeValues={
                    ":prog": {"dimensions": {dimension_name: status}}
                },
                ConditionExpression="attribute_not_exists(progress)",
            )
        except Exception:  # noqa: BLE001
            # Another worker already created it; try the nested update again.
            try:
                _get_table().update_item(
                    Key={"jobId": job_id},
                    UpdateExpression="SET progress.dimensions.#dim = :status",
                    ExpressionAttributeNames={"#dim": dimension_name},
                    ExpressionAttributeValues={":status": status},
                )
            except Exception:  # noqa: BLE001
                pass


def _serialize_eval_result(result: Any) -> Any:
    """Serialize an EvaluationResult to a JSON-safe dict."""
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    return str(result)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def handler(event: Dict[str, Any], context: Any = None) -> Any:
    """Async Worker entry point (+ synchronous catalog/validate actions)."""
    event = event or {}

    # Synchronous, job-less actions (RequestResponse invoke from the API Lambda).
    action = event.get("action")
    if action == "get_metric_catalog":
        try:
            from uaef.api import get_full_metric_catalog  # type: ignore
        except ImportError:
            from uaef.metrics import get_full_metric_catalog  # type: ignore
        return get_full_metric_catalog()
    if action == "validate_data":
        return _validate_data(event)
    if action == "list_experiments":
        return _list_experiments(created_by=event.get("created_by"))
    if action == "get_experiment":
        return _get_experiment(event)
    if action == "generate_report":
        return _generate_report_action(event)
    if action == "compare_experiments":
        return _compare_experiments_action(event)

    job_id = event.get("jobId")
    operation = event.get("operation") or "evaluate"
    is_partial = event.get("isPartialDimension", False)
    if not job_id:
        return

    # When invoked as a partial-dimension chunk by the Step Functions Map state,
    # skip the job lifecycle transitions (the Splitter already moved the job to
    # PROCESSING, and the Reducer will complete/fail it after merging all chunks).
    if is_partial:
        return _run_partial_dimension(event, operation)

    # PENDING -> PROCESSING (idempotent guard).
    if not start_processing(job_id):
        return

    # Version-mismatch guard (Req 2.3, 2.4).
    mismatch = _version_mismatch_message()
    if mismatch is not None:
        fail_job(job_id, code=VERSION_MISMATCH_CODE, message=bound_error_message(mismatch))
        return

    try:
        request = _load_request(event)
        persist = bool(request.get("persist", False))
        # Security review H-02: the caller's identity was recorded on the Job
        # at creation (API Lambda's create_job); look it up here so any
        # experiment this Worker persists is stamped with an owner. The
        # Worker/Reducer never receive caller identity directly in their
        # event payload — this is the one place that looks it up, from the
        # jobId every invocation already carries.
        created_by = job_state.get_created_by(job_id)

        if operation == "invoke_evaluate":
            # Server invoked the agent(s); always store the rich UI-shaped result
            # so the UI can render it via resultRef (independent of persist).
            result, ui_result = _run_invoke_evaluate(request, created_by=created_by)
            result_ref = _store_ui_result(job_id, ui_result)
        elif operation == "batch_evaluate":
            result = _run_batch_evaluate(request, created_by=created_by)
            result_ref = _store_result(job_id, result) if persist else None
        else:
            result = _run_evaluate(request, created_by=created_by)
            result_ref = _store_result(job_id, result) if persist else None

        outcome = _build_outcome(operation, result)
        complete_job(
            job_id,
            experiment_id=outcome.get("experimentId"),
            result_ref=result_ref,
            result_summary=outcome.get("resultSummary"),
        )
    except Exception as exc:  # noqa: BLE001 — any failure -> bounded FAILED
        # Security review M-03: this message is surfaced verbatim by
        # GET /jobs/{jobId} to the job's owner. This is the main evaluation
        # path (evaluate/batch_evaluate/invoke_evaluate) and can fail with
        # boto3/S3/DynamoDB/library exceptions carrying bucket names, keys, or
        # ARNs — sanitize before persisting. ``type(exc).__name__`` in the
        # ``code`` field is safe (exception class name only, no detail).
        safe_message, _correlation_id = _sanitize_error(
            exc, context="Evaluation failed"
        )
        fail_job(job_id, code=type(exc).__name__, message=safe_message)
