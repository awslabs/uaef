# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI routes wrapping the existing UAEF Service Lambda handler logic.

Each route maps to the same endpoint the API Gateway fronted, but instead of
Lambda proxy events we use standard HTTP request/response. The handler modules
(jobs, catalog, payloads, agents) are imported and called with adapted arguments.

The Worker Lambda logic runs in-process (no async Lambda invoke). For long
evaluations, it's dispatched to a background thread so the HTTP response returns
immediately with a jobId (same async pattern as the Lambda version).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Optional
from unittest.mock import patch, MagicMock

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Patch: In-process worker invocation (replaces Lambda invoke)
# --------------------------------------------------------------------------- #

def _inline_worker_invoke(**kwargs) -> Dict[str, Any]:
    """Intercept boto3 Lambda invoke calls and run the worker in-process."""
    import handlers.worker as worker_handler

    payload = kwargs.get("Payload", b"{}")
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    event = json.loads(payload)

    invocation_type = kwargs.get("InvocationType", "RequestResponse")

    if invocation_type == "Event":
        # Async invocation — run in background thread
        job_id = event.get("jobId", "unknown")
        thread = threading.Thread(
            target=_run_worker_background,
            args=(event,),
            name=f"worker-{job_id}",
            daemon=True,
        )
        thread.start()
        return {"StatusCode": 202, "Payload": b""}
    else:
        # Synchronous invocation — run inline and return result
        try:
            result = worker_handler.handler(event, None)
            result_bytes = json.dumps(result, default=str).encode("utf-8")
            import io
            return {
                "StatusCode": 200,
                "Payload": io.BytesIO(result_bytes),
            }
        except Exception as exc:
            import io
            error_payload = json.dumps({"error": str(exc)}).encode("utf-8")
            return {
                "StatusCode": 200,
                "FunctionError": "Unhandled",
                "Payload": io.BytesIO(error_payload),
            }


def _run_worker_background(event: Dict[str, Any]) -> None:
    """Run worker in background thread with error handling."""
    try:
        import handlers.worker as worker_handler
        worker_handler.handler(event, None)
    except Exception:
        logger.exception("Background worker failed for event: %s", event.get("jobId"))
        try:
            from job_state import fail_job, bound_error_message
            job_id = event.get("jobId")
            if job_id:
                fail_job(job_id, code="WORKER_CRASH", message=bound_error_message("Worker process crashed unexpectedly"))
        except Exception:
            logger.exception("Failed to mark job as FAILED")


class _InlineLambdaClient:
    """Fake boto3 Lambda client that routes invocations to in-process worker."""

    def invoke(self, **kwargs):
        return _inline_worker_invoke(**kwargs)


_original_boto3_client = None


def _patch_boto3_lambda():
    """Patch boto3.client to intercept Lambda client creation."""
    import boto3
    global _original_boto3_client
    _original_boto3_client = boto3.client

    def _patched_client(service_name, *args, **kwargs):
        if service_name == "lambda":
            return _InlineLambdaClient()
        return _original_boto3_client(service_name, *args, **kwargs)

    boto3.client = _patched_client


# Apply the monkey-patch at import time (worker runs in-process) ONLY when
# running in inline mode. When EVAL_STATE_MACHINE_ARN is set (lambda mode),
# evaluations route through Step Functions → real Worker Lambda, and synchronous
# Worker calls (catalog, experiments, reports) go to the real Worker Lambda.
_WORKER_MODE = os.environ.get("WORKER_MODE", "inline")
if _WORKER_MODE == "inline":
    os.environ.setdefault("WORKER_FUNCTION_NAME", "__INLINE_WORKER__")
    _patch_boto3_lambda()


# --------------------------------------------------------------------------- #
# Helper: build a fake Lambda proxy event from a FastAPI request
# --------------------------------------------------------------------------- #

def _build_proxy_event(request: Request, body: Optional[str] = None) -> Dict[str, Any]:
    """Build a minimal API Gateway proxy event from a FastAPI request."""
    path = request.url.path
    method = request.method

    # Extract path parameters from the URL
    path_params = request.path_params or {}

    # Build query string params
    query_params = dict(request.query_params) if request.query_params else None

    # Build the claims context (from JWT middleware)
    caller_sub = getattr(request.state, "caller_sub", "anonymous")

    event = {
        "httpMethod": method,
        "path": path,
        "resource": path,
        "pathParameters": path_params if path_params else None,
        "queryStringParameters": query_params,
        "headers": dict(request.headers),
        "body": body,
        "requestContext": {
            "authorizer": {
                "claims": {
                    "sub": caller_sub,
                }
            }
        },
    }
    return event


def _proxy_response_to_fastapi(result: Any) -> Response:
    """Convert an API Gateway proxy response (or plain dict) to FastAPI Response."""
    if isinstance(result, dict) and "statusCode" in result:
        status_code = result["statusCode"]
        headers = result.get("headers", {})
        body = result.get("body", "")
        if isinstance(body, dict):
            body = json.dumps(body, default=str)
        return Response(content=body, status_code=status_code, headers=headers, media_type="application/json")
    else:
        return JSONResponse(content=result)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@router.get("/metrics")
async def get_metrics(request: Request):
    """Return the metric catalog."""
    from handlers.catalog import get_metric_catalog
    result = get_metric_catalog()
    return _proxy_response_to_fastapi(result)


@router.post("/evaluate")
async def create_evaluate_job(request: Request):
    """Create an evaluation job."""
    from handlers.api import handler as api_handler
    body = await request.body()
    event = _build_proxy_event(request, body.decode("utf-8") if body else None)
    event["resource"] = "/evaluate"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.post("/batch-evaluate")
async def create_batch_evaluate_job(request: Request):
    """Create a batch evaluation job."""
    from handlers.api import handler as api_handler
    body = await request.body()
    event = _build_proxy_event(request, body.decode("utf-8") if body else None)
    event["resource"] = "/batch-evaluate"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.get("/jobs/{job_id}")
async def get_job_status(request: Request, job_id: str):
    """Get job status by ID."""
    from handlers.api import handler as api_handler
    event = _build_proxy_event(request)
    event["resource"] = "/jobs/{jobId}"
    event["pathParameters"] = {"jobId": job_id}
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.get("/experiments")
async def list_experiments(request: Request):
    """List experiments."""
    from handlers.api import handler as api_handler
    event = _build_proxy_event(request)
    event["resource"] = "/experiments"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.get("/experiments/{experiment_id}")
async def get_experiment(request: Request, experiment_id: str):
    """Get experiment by ID."""
    from handlers.api import handler as api_handler
    event = _build_proxy_event(request)
    event["resource"] = "/experiments/{experimentId}"
    event["pathParameters"] = {"experimentId": experiment_id}
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.post("/payloads")
async def presign_payload_upload(request: Request):
    """Generate a presigned URL for payload upload."""
    from handlers.api import handler as api_handler
    body = await request.body()
    event = _build_proxy_event(request, body.decode("utf-8") if body else None)
    event["resource"] = "/payloads"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.get("/agent-types")
async def get_agent_types(request: Request):
    """Return supported agent connection types."""
    from handlers.api import handler as api_handler
    event = _build_proxy_event(request)
    event["resource"] = "/agent-types"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.get("/agentcore/runtimes")
async def list_agentcore_runtimes(request: Request):
    """List available AgentCore runtimes."""
    from handlers.api import handler as api_handler
    event = _build_proxy_event(request)
    event["resource"] = "/agentcore/runtimes"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.post("/invoke-evaluate")
async def invoke_evaluate(request: Request):
    """Invoke an agent and evaluate the result."""
    from handlers.api import handler as api_handler
    body = await request.body()
    event = _build_proxy_event(request, body.decode("utf-8") if body else None)
    event["resource"] = "/invoke-evaluate"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.post("/validate-data")
async def validate_data(request: Request):
    """Validate ground-truth data format."""
    from handlers.api import handler as api_handler
    body = await request.body()
    event = _build_proxy_event(request, body.decode("utf-8") if body else None)
    event["resource"] = "/validate-data"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.post("/reports")
async def generate_report(request: Request):
    """Generate an evaluation report."""
    from handlers.api import handler as api_handler
    body = await request.body()
    event = _build_proxy_event(request, body.decode("utf-8") if body else None)
    event["resource"] = "/reports"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)


@router.post("/compare-experiments")
async def compare_experiments(request: Request):
    """Compare multiple experiments side by side."""
    from handlers.api import handler as api_handler
    body = await request.body()
    event = _build_proxy_event(request, body.decode("utf-8") if body else None)
    event["resource"] = "/compare-experiments"
    result = api_handler(event, None)
    return _proxy_response_to_fastapi(result)
