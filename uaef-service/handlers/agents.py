# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Agent connection + invoke-and-evaluate handlers for the UAEF Service API.

These routes let the **deployed UI** drive an agent that is reachable from AWS
(an AgentCore runtime, a Bedrock agent, or a public HTTPS endpoint), collect a
trace per question, and evaluate the traces — the "server invokes the agent"
model. Local/laptop agents are not reachable from the cloud and use the notebook
(UAEFClient) path instead.

Like the rest of the API Lambda (Requirement 6.2) this module MUST NOT import
``uaef``. The actual agent invocation + evaluation happens in the Worker Lambda
(operation ``invoke_evaluate``); here we only:

  * ``GET  /agent-types``        — return the static connection-field config.
  * ``GET  /agentcore/runtimes`` — list deployed AgentCore runtimes (boto3).
  * ``POST /invoke-evaluate``    — validate, create a PENDING job, async-invoke
                                   the Worker, return the jobId.

Job creation reuses the helpers in ``handlers.jobs`` so worker-invoke / failure
handling / response shaping stay in one place.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from pydantic import ValidationError

# Sibling modules (boto3/pydantic only — never ``uaef``).
try:  # package-relative when deployed as the ``handlers`` package
    from . import jobs  # type: ignore[import-not-found]
except (ImportError, ValueError):  # flat layout
    import jobs  # type: ignore[no-redef]

try:
    from .. import job_state, schemas  # type: ignore[import-not-found]
except (ImportError, ValueError):
    import job_state  # type: ignore[no-redef]
    import schemas  # type: ignore[no-redef]

import boto3


#: Static agent-type config the UI renders connection fields from. Only agent
#: types that are reachable from AWS are offered (no "localhost" option — local
#: agents use the notebook path). ``adapter`` is the uaef adapter the Worker uses
#: to transform that agent's raw output into a canonical trace.
# Agent types. IDs equal the library adapter names (uaef.adapters.registry), so
# the Worker can call get_adapter(framework) directly. Only agent types that are
# reachable from AWS are offered (no "localhost" — local agents use the
# notebook). HTTP-based frameworks require a public/reachable endpoint URL.
AGENT_TYPES = [
    {
        "id": "agentcore",
        "label": "AWS AgentCore",
        "fields": [
            {"name": "agent_runtime_arn", "label": "Agent Runtime ARN",
             "placeholder": "arn:aws:bedrock-agentcore:...:runtime/..."},
            {"name": "region", "label": "AWS Region", "default": "us-east-1"},
            {"name": "bearer_token", "label": "Bearer Token (OAuth/JWT)",
             "placeholder": "Leave empty for SigV4", "optional": True},
        ],
    },
    {
        "id": "bedrock",
        "label": "AWS Bedrock Agent",
        "fields": [
            {"name": "agent_id", "label": "Agent ID", "placeholder": "e.g. ABCDEF1234"},
            {"name": "alias_id", "label": "Alias ID", "placeholder": "e.g. TSTALIASID"},
            {"name": "region", "label": "AWS Region", "default": "us-east-1"},
        ],
    },
    {
        "id": "strands",
        "label": "Strands (HTTP endpoint)",
        "fields": [
            {"name": "endpoint", "label": "Agent Endpoint URL", "placeholder": "https://my-agent.example.com/invoke"},
        ],
    },
    {
        "id": "langgraph",
        "label": "LangGraph (HTTP endpoint)",
        "fields": [
            {"name": "endpoint", "label": "Agent Endpoint URL", "placeholder": "https://my-agent.example.com/invoke"},
        ],
    },
    {
        "id": "langchain",
        "label": "LangChain (HTTP endpoint)",
        "fields": [
            {"name": "endpoint", "label": "Agent Endpoint URL", "placeholder": "https://my-agent.example.com/invoke"},
        ],
    },
    {
        "id": "generic",
        "label": "Generic (HTTP endpoint)",
        "fields": [
            {"name": "endpoint", "label": "Agent Endpoint URL", "placeholder": "https://my-agent.example.com/invoke"},
        ],
    },
]

#: AgentCore control-plane region default.
_DEFAULT_REGION = "us-east-1"

_agentcore_control = None


def _get_agentcore_control(region: str):
    """Return a cached AgentCore control-plane client for ``region``."""
    global _agentcore_control
    if _agentcore_control is None:
        _agentcore_control = boto3.client("bedrock-agentcore-control", region_name=region)
    return _agentcore_control


# --------------------------------------------------------------------------- #
# GET /agent-types
# --------------------------------------------------------------------------- #


def get_agent_types() -> Any:
    """Return the static list of supported, AWS-reachable agent types."""
    return AGENT_TYPES


# --------------------------------------------------------------------------- #
# GET /agentcore/runtimes
# --------------------------------------------------------------------------- #


def list_agentcore_runtimes(query: Dict[str, Any]) -> Any:
    """List deployed AgentCore runtimes in the requested region.

    Returns ``{"runtimes": [{name, arn, status}, ...]}``. Best-effort: any AWS
    error returns an empty list (the UI lets the user paste an ARN manually), so
    a discovery hiccup never blocks the flow.
    """
    region = (query or {}).get("region") or _DEFAULT_REGION
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        runtimes = []
        paginator_input: Dict[str, Any] = {}
        while True:
            resp = client.list_agent_runtimes(**paginator_input)
            for rt in resp.get("agentRuntimes", []):
                runtimes.append(
                    {
                        "name": rt.get("agentRuntimeName") or rt.get("agentRuntimeId"),
                        "arn": rt.get("agentRuntimeArn"),
                        "status": rt.get("status", "UNKNOWN"),
                    }
                )
            token = resp.get("nextToken")
            if not token:
                break
            paginator_input = {"nextToken": token}
        return {"runtimes": runtimes}
    except Exception as exc:  # noqa: BLE001 — discovery is best-effort
        # Security review M-03: don't return boto3/IAM exception detail.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Agent runtime discovery failed"
        )
        return {"runtimes": [], "warning": safe_message, "correlationId": correlation_id}


# --------------------------------------------------------------------------- #
# POST /invoke-evaluate
# --------------------------------------------------------------------------- #


def create_invoke_evaluate_job(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an invoke-and-evaluate request, create a job, hand it to the Worker.

    Mirrors ``jobs.create_evaluate_job`` but for the ``invoke_evaluate``
    operation: the Worker invokes the agent per question and batch-evaluates the
    resulting traces. Reuses ``jobs`` helpers for parsing, worker invoke, and
    failure handling so behaviour stays consistent across routes.
    """
    try:
        body = jobs._parse_body(event)
    except ValueError:
        return jobs._error(400, "Request body is not valid JSON or base64.")

    try:
        model = schemas.InvokeEvaluateRequest.model_validate(body)
    except ValidationError as exc:
        return jobs._error(400, "Request body failed validation.", details=_safe_details(exc))

    request_dict = model.model_dump()

    if not request_dict.get("questions") and not request_dict.get("s3_data_path"):
        return jobs._error(400, "Provide either inline questions or an s3_data_path.")

    # Curated-metric enforcement (same guard as the evaluate route).
    limitation = jobs._run_curated_validation(body if isinstance(body, dict) else request_dict)
    if limitation is not None:
        return limitation

    caller_sub = jobs._extract_caller_sub(event)
    if not caller_sub:
        return jobs._error(401, "Unauthorized: missing caller identity.")

    job_id = str(uuid.uuid4())
    try:
        job_state.create_job(job_id, "invoke_evaluate", created_by=caller_sub)
    except Exception as exc:  # noqa: BLE001
        # Security review M-03: don't return DynamoDB/boto3 exception detail.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to create job record"
        )
        return jobs._error(500, safe_message, correlationId=correlation_id)

    try:
        jobs._invoke_worker_async(job_id, "invoke_evaluate", request_dict)
    except Exception as exc:  # noqa: BLE001
        # Security review M-03: the persisted job error (surfaced verbatim by
        # GET /jobs/{jobId}) and the response body must not carry exception
        # detail either.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to enqueue evaluation worker"
        )
        jobs._mark_failed(job_id, code="ENQUEUE_FAILED", message=safe_message)
        return jobs._error(
            502,
            "Failed to enqueue evaluation worker; job marked FAILED.",
            jobId=job_id,
            status=job_state.FAILED,
            correlationId=correlation_id,
        )

    return schemas.JobCreatedResponse(jobId=job_id).model_dump()


def _safe_details(exc: ValidationError) -> Any:
    """Return JSON-serializable validation error details."""
    import json

    try:
        return json.loads(exc.json())
    except Exception:  # noqa: BLE001
        return str(exc)


# --------------------------------------------------------------------------- #
# POST /validate-data  (delegates parsing/validation to the Worker -> uaef.data)
# --------------------------------------------------------------------------- #

_WORKER_FUNCTION_ENV = "WORKER_FUNCTION_NAME"
_lambda_client = None


def _get_lambda():
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")
    return _lambda_client


def validate_data(event: Dict[str, Any]) -> Any:
    """Validate a ground-truth file by reference, via the Worker (library).

    The API Lambda must not import ``uaef``; it invokes the Worker synchronously
    with ``{"action": "validate_data", "data_ref": ...}``. ``data_ref`` is either
    an ``s3://`` URI (a user's bucket) or a payload-bucket key uploaded via the
    presigned /payloads flow. Returns the library ``ValidationResult``.
    """
    import json

    try:
        body = jobs._parse_body(event)
    except ValueError:
        return jobs._error(400, "Request body is not valid JSON or base64.")

    data_ref = (body or {}).get("data_ref") or (body or {}).get("s3_data_path")
    if not data_ref:
        return jobs._error(400, "data_ref is required.")
    filename = (body or {}).get("filename")

    fn = os.environ.get(_WORKER_FUNCTION_ENV)
    if not fn:
        return jobs._error(503, "Validation unavailable: worker not configured.")

    try:
        resp = _get_lambda().invoke(
            FunctionName=fn,
            InvocationType="RequestResponse",
            Payload=json.dumps(
                {"action": "validate_data", "data_ref": data_ref, "filename": filename}
            ).encode("utf-8"),
        )
        payload = resp.get("Payload")
        raw = payload.read() if hasattr(payload, "read") else payload
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        result = json.loads(raw) if raw else {}
        if resp.get("FunctionError"):
            # Security review M-03: a Lambda FunctionError payload is the
            # Worker's raw traceback — never return it to the caller.
            safe_message, correlation_id = job_state.sanitize_error(
                RuntimeError(result), context="Validation failed in the worker"
            )
            return jobs._error(502, safe_message, correlationId=correlation_id)
        return result
    except Exception as exc:  # noqa: BLE001
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to validate data"
        )
        return jobs._error(502, safe_message, correlationId=correlation_id)


# --------------------------------------------------------------------------- #
# GET /experiments  (library-backed list, via the Worker)
# --------------------------------------------------------------------------- #


def list_experiments(query: Dict[str, Any] | None = None, caller_sub: str | None = None) -> Any:
    """List the caller's own persisted experiments, via the Worker (uaef.storage).

    Security review H-02: previously listed every experiment in the
    deployment regardless of caller, letting any authenticated user read
    every other user's experiment data. Now scopes the listing to
    ``caller_sub`` (via the Worker's ``created_by`` GSI query — see
    ``uaef.storage.dynamodb_s3.DynamoS3Storage.list_experiments``), so a
    caller only ever sees experiments they created themselves. Experiments
    created before this fix (with no ``created_by`` recorded) are never
    returned by this owner-scoped listing — see the CDK GSI comment in
    ``infra/storage_stack.py`` for the fail-closed rationale.

    The API Lambda must not import ``uaef``; it invokes the Worker synchronously
    with ``{"action": "list_experiments", "created_by": caller_sub}`` and
    relays the result. Returns ``{"experiments": [...]}``.
    """
    import json

    fn = os.environ.get(_WORKER_FUNCTION_ENV)
    if not fn:
        return {"experiments": [], "error": "worker not configured"}
    if not caller_sub:
        # Defensive: the router should have rejected an unauthenticated
        # request already. Fail closed rather than list everything.
        return {"experiments": [], "error": "Unauthorized: missing caller identity."}
    try:
        resp = _get_lambda().invoke(
            FunctionName=fn,
            InvocationType="RequestResponse",
            Payload=json.dumps(
                {"action": "list_experiments", "created_by": caller_sub}
            ).encode("utf-8"),
        )
        payload = resp.get("Payload")
        raw = payload.read() if hasattr(payload, "read") else payload
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        result = json.loads(raw) if raw else {"experiments": []}
        if resp.get("FunctionError"):
            # Security review M-03: don't return the Worker's raw traceback.
            safe_message, correlation_id = job_state.sanitize_error(
                RuntimeError(result), context="Worker error listing experiments"
            )
            return {"experiments": [], "error": safe_message, "correlationId": correlation_id}
        return result
    except Exception as exc:  # noqa: BLE001
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to list experiments"
        )
        return {"experiments": [], "error": safe_message, "correlationId": correlation_id}


# --------------------------------------------------------------------------- #
# POST /compare-experiments (compare multiple experiments side by side)
# --------------------------------------------------------------------------- #


def compare_experiments(event: Dict[str, Any], caller_sub: str | None = None) -> Any:
    """Compare multiple experiments by their per-metric average scores.

    Security review H-02: previously compared any experiment IDs the caller
    named, with no ownership check — letting any authenticated user pull
    another user's experiment data into a comparison. Now requires
    ``caller_sub`` to own every named experiment (checked by the Worker via
    each experiment's ``created_by``); if any experiment is missing or not
    owned by the caller, the whole request is rejected rather than silently
    comparing only the ones that pass, so a caller can't distinguish
    "doesn't exist" from "exists but isn't yours" by observing which subset
    of IDs made it into the response.

    Delegates to the Worker synchronously (it reads experiment data from the
    library storage layer). Body: ``{experiment_ids: [id1, id2, ...]}``.
    Returns a comparison result shaped for the UI's ComparePanel.
    """
    import json

    try:
        body = jobs._parse_body(event)
    except ValueError:
        return jobs._error(400, "Request body is not valid JSON or base64.")

    experiment_ids = (body or {}).get("experiment_ids") or []
    if not isinstance(experiment_ids, list) or len(experiment_ids) < 2:
        return jobs._error(400, "Provide at least 2 experiment_ids to compare.")

    if not caller_sub:
        return jobs._error(401, "Unauthorized: missing caller identity.")

    fn = os.environ.get(_WORKER_FUNCTION_ENV)
    if not fn:
        return jobs._error(503, "Comparison unavailable: worker not configured.")

    try:
        resp = _get_lambda().invoke(
            FunctionName=fn,
            InvocationType="RequestResponse",
            Payload=json.dumps(
                {
                    "action": "compare_experiments",
                    "experiment_ids": experiment_ids,
                    "created_by": caller_sub,
                }
            ).encode("utf-8"),
        )
        payload = resp.get("Payload")
        raw = payload.read() if hasattr(payload, "read") else payload
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        result = json.loads(raw) if raw else {}
        if resp.get("FunctionError"):
            safe_message, correlation_id = job_state.sanitize_error(
                RuntimeError(result), context="Comparison failed in the worker"
            )
            return jobs._error(502, safe_message, correlationId=correlation_id)
        if isinstance(result, dict) and result.get("code") == "FORBIDDEN":
            # Security review H-02: the Worker found an experiment that
            # either doesn't exist or isn't owned by caller_sub. Return 403
            # with no experiment detail, matching get_job_status's pattern.
            return jobs._error(403, "Forbidden: you do not have access to one or more of the requested experiments.")
        if isinstance(result, dict) and result.get("error"):
            safe_message, correlation_id = job_state.sanitize_error(
                RuntimeError(result["error"]), context="Failed to compare experiments"
            )
            return jobs._error(502, safe_message, correlationId=correlation_id)
        return result
    except Exception as exc:  # noqa: BLE001
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to compare experiments"
        )
        return jobs._error(502, safe_message, correlationId=correlation_id)


# --------------------------------------------------------------------------- #
# POST /reports  (generate a library report for an experiment, via the Worker)
# --------------------------------------------------------------------------- #


def generate_report(event: Dict[str, Any], caller_sub: str | None = None) -> Any:
    """Generate the library report for an experiment, returning a viewable URL.

    Security review H-02: previously generated a report for any
    experiment_id the caller named, with no ownership check. Now requires
    ``caller_sub`` to own the experiment (checked by the Worker via
    ``created_by``); a missing or non-owned experiment is rejected with 403,
    matching get_job_status's non-leaking pattern.

    Delegates to the Worker (``uaef.reporting.generate_report``); the API Lambda
    never imports ``uaef``. Body: ``{experiment_id, report_type?}``. Returns
    ``{reportUrl, title}``.
    """
    import json

    try:
        body = jobs._parse_body(event)
    except ValueError:
        return jobs._error(400, "Request body is not valid JSON or base64.")

    experiment_id = (body or {}).get("experiment_id")
    if not experiment_id:
        return jobs._error(400, "experiment_id is required.")
    report_type = (body or {}).get("report_type") or "full"

    if not caller_sub:
        return jobs._error(401, "Unauthorized: missing caller identity.")

    fn = os.environ.get(_WORKER_FUNCTION_ENV)
    if not fn:
        return jobs._error(503, "Report generation unavailable: worker not configured.")

    try:
        resp = _get_lambda().invoke(
            FunctionName=fn,
            InvocationType="RequestResponse",
            Payload=json.dumps(
                {
                    "action": "generate_report",
                    "experiment_id": experiment_id,
                    "report_type": report_type,
                    "created_by": caller_sub,
                }
            ).encode("utf-8"),
        )
        payload = resp.get("Payload")
        raw = payload.read() if hasattr(payload, "read") else payload
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        result = json.loads(raw) if raw else {}
        if isinstance(result, dict) and result.get("code") == "FORBIDDEN":
            # Security review H-02: no experiment/report detail disclosed.
            return jobs._error(403, "Forbidden: you do not have access to this experiment.")
        if resp.get("FunctionError"):
            safe_message, correlation_id = job_state.sanitize_error(
                RuntimeError(result), context="Report generation failed in the worker"
            )
            return jobs._error(502, safe_message, correlationId=correlation_id)
        if isinstance(result, dict) and result.get("error"):
            safe_message, correlation_id = job_state.sanitize_error(
                RuntimeError(result["error"]), context="Failed to generate report"
            )
            return jobs._error(502, safe_message, correlationId=correlation_id)
        return result
    except Exception as exc:  # noqa: BLE001
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to generate report"
        )
        return jobs._error(502, safe_message, correlationId=correlation_id)
