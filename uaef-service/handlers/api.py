# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""API Lambda router for the UAEF Service.

This module is the API Gateway proxy entry point. It is part of the *separate*
``uaef-service`` deployable application and, per **Requirement 6.2**, MUST NOT
import the ``uaef`` package or the ``server`` extra. It only routes: it parses
the API Gateway proxy event, authorizes the caller, and dispatches to the
per-route handler functions. All actual evaluation work is delegated to the
Worker Lambda (which is the only component that imports ``uaef``).

Route map (see design.md "API Route Definitions"):

    | Method | Resource                      | Dispatches to                |
    |--------|-------------------------------|------------------------------|
    | POST   | /evaluate                     | create_evaluate_job(..., "evaluate")
    | POST   | /batch-evaluate               | create_evaluate_job(..., "batch_evaluate")
    | GET    | /jobs/{jobId}                 | get_job_status(job_id, sub)  |
    | GET    | /experiments                  | list_experiments(query)      |
    | GET    | /experiments/{experimentId}   | get_experiment(experiment_id)|
    | POST   | /payloads                     | presign_payload_upload(sub)  |
    | GET    | /metrics                      | get_metric_catalog()         |

The per-route handlers live in sibling modules (``jobs``, ``payloads``,
``catalog``) authored in tasks 9.2 / 9.4 / 9.5. They are imported **lazily**
inside the dispatch so this router stands alone before those modules exist; a
missing handler yields a ``501 Not Implemented`` response rather than an import
error at module load.

Handler return-value contract:
    A dispatched handler may return either
      * a full API Gateway proxy response (a dict containing ``statusCode``), in
        which case it is passed through after ensuring the body is JSON-encoded;
        this lets handlers set their own 400/403/404/413 codes, or
      * a plain JSON-serializable object, which the router wraps in a ``200 OK``
        proxy response.

Requirements: 6.2
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Tuple

# NOTE: This module intentionally imports neither ``uaef`` nor any sibling
# handler module at import time. Sibling handlers are imported lazily within the
# dispatch functions below so the router has no hard dependency on code authored
# in later tasks and never bundles the library.
#
# ``job_state`` is the one exception: it carries no ``uaef``/``server``
# dependency itself (Requirement 6.2 only forbids importing ``uaef``), and
# ``sanitize_error`` (security review M-03) is needed at router level to avoid
# leaking exception detail to callers.
try:  # package-relative when deployed as the ``handlers`` package
    from .. import job_state  # type: ignore[import-not-found]
except (ImportError, ValueError):  # flat layout / top-level module
    import job_state  # type: ignore[no-redef]

_JSON_HEADERS = {
    "Content-Type": "application/json",
    # The deployed browser UI reads responses cross-origin; mirror the API
    # Gateway CORS preflight so actual responses are also readable.
    "Access-Control-Allow-Origin": "*",
}


# --------------------------------------------------------------------------- #
# Proxy response helpers
# --------------------------------------------------------------------------- #


def _response(status_code: int, body: Any) -> Dict[str, Any]:
    """Build an API Gateway proxy response with a JSON-encoded body."""
    if isinstance(body, str):
        encoded = body
    else:
        encoded = json.dumps(body, default=str)
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


def _normalize_result(result: Any) -> Dict[str, Any]:
    """Normalize a dispatched handler's return value into a proxy response.

    If the handler already returned a proxy-style response (a dict with a
    ``statusCode``), pass it through after ensuring the body is JSON-encoded.
    Otherwise wrap the value in a ``200 OK`` response.
    """
    if isinstance(result, dict) and "statusCode" in result:
        body = result.get("body")
        if body is not None and not isinstance(body, str):
            result = dict(result)
            result["body"] = json.dumps(body, default=str)
        result.setdefault("headers", dict(_JSON_HEADERS))
        # Ensure CORS is present even when the handler set its own headers, so
        # the browser UI can read responses from every route.
        result["headers"].setdefault("Access-Control-Allow-Origin", "*")
        return result
    return _response(200, result)


# --------------------------------------------------------------------------- #
# Event parsing helpers
# --------------------------------------------------------------------------- #


def _extract_caller_sub(event: Dict[str, Any]) -> Optional[str]:
    """Return the caller's Cognito ``sub`` claim from the proxy event.

    API Gateway places verified Cognito claims under
    ``requestContext.authorizer.claims``. The Cognito authorizer guarantees a
    valid JWT before the API Lambda is invoked (Requirement 9.1/9.2), so a
    missing ``sub`` here is defensive only.
    """
    request_context = event.get("requestContext") or {}
    authorizer = request_context.get("authorizer") or {}
    claims = authorizer.get("claims") or {}
    sub = claims.get("sub")
    return sub if isinstance(sub, str) and sub else None


def _path_parameters(event: Dict[str, Any]) -> Dict[str, Any]:
    return event.get("pathParameters") or {}


def _query_parameters(event: Dict[str, Any]) -> Dict[str, Any]:
    return event.get("queryStringParameters") or {}


def _route_key(event: Dict[str, Any]) -> Tuple[str, str]:
    """Return the ``(httpMethod, resource)`` routing key.

    ``resource`` is the API Gateway resource *template* (e.g. ``/jobs/{jobId}``),
    which is stable regardless of the concrete path, making it ideal for routing.
    """
    method = (event.get("httpMethod") or "").upper()
    resource = event.get("resource") or event.get("path") or ""
    return method, resource


# --------------------------------------------------------------------------- #
# Lazy sibling-handler loading
# --------------------------------------------------------------------------- #


class _HandlerUnavailable(Exception):
    """Raised when a sibling handler module/function is not yet available."""


def _load_handler(module_name: str, func_name: str) -> Callable[..., Any]:
    """Import ``func_name`` from a sibling handler module lazily.

    Importing within dispatch (rather than at module load) keeps this router
    standalone: handler modules authored in later tasks need not exist for the
    router to import and route. A missing module/function surfaces as a
    :class:`_HandlerUnavailable` which the dispatcher maps to ``501``.
    """
    # Support both package-relative and flat module layouts so the router works
    # whether deployed as a package (``handlers.jobs``) or as top-level modules.
    candidates = (f"{__package__}.{module_name}" if __package__ else module_name, module_name)
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            module = __import__(candidate, fromlist=[func_name])
        except ImportError as exc:  # module not authored yet / not on path
            last_error = exc
            continue
        func = getattr(module, func_name, None)
        if func is None:
            raise _HandlerUnavailable(
                f"{candidate}.{func_name} is not implemented yet"
            )
        return func
    raise _HandlerUnavailable(
        f"handler module '{module_name}' is unavailable: {last_error}"
    )


# --------------------------------------------------------------------------- #
# Per-route dispatch
# --------------------------------------------------------------------------- #


def _dispatch(method: str, resource: str, event: Dict[str, Any], caller_sub: str) -> Any:
    """Dispatch a recognized ``(method, resource)`` to its handler.

    Returns the handler's raw return value (normalized by the caller). Raises
    :class:`KeyError` semantics via the route tables in :func:`handler` — this
    function is only called for routes already known to be valid.
    """
    if resource == "/evaluate":
        create_evaluate_job = _load_handler("jobs", "create_evaluate_job")
        return create_evaluate_job(event, "evaluate")

    if resource == "/batch-evaluate":
        create_evaluate_job = _load_handler("jobs", "create_evaluate_job")
        return create_evaluate_job(event, "batch_evaluate")

    if resource == "/jobs/{jobId}":
        job_id = _path_parameters(event).get("jobId")
        if not job_id:
            return _error(400, "Missing path parameter 'jobId'.")
        get_job_status = _load_handler("jobs", "get_job_status")
        return get_job_status(job_id, caller_sub)

    if resource == "/experiments":
        # Security review H-02: scope the listing to the caller's own
        # experiments — see list_experiments's docstring in handlers/agents.py.
        list_experiments = _load_handler("agents", "list_experiments")
        return list_experiments(_query_parameters(event), caller_sub)

    if resource == "/experiments/{experimentId}":
        experiment_id = _path_parameters(event).get("experimentId")
        if not experiment_id:
            return _error(400, "Missing path parameter 'experimentId'.")
        get_experiment = _load_handler("jobs", "get_experiment")
        return get_experiment(experiment_id, caller_sub)

    if resource == "/payloads":
        presign_payload_upload = _load_handler("payloads", "presign_payload_upload")
        return presign_payload_upload(caller_sub)

    if resource == "/metrics":
        get_metric_catalog = _load_handler("catalog", "get_metric_catalog")
        return get_metric_catalog()

    if resource == "/agent-types":
        get_agent_types = _load_handler("agents", "get_agent_types")
        return get_agent_types()

    if resource == "/agentcore/runtimes":
        list_agentcore_runtimes = _load_handler("agents", "list_agentcore_runtimes")
        return list_agentcore_runtimes(_query_parameters(event))

    if resource == "/invoke-evaluate":
        create_invoke_evaluate_job = _load_handler("agents", "create_invoke_evaluate_job")
        return create_invoke_evaluate_job(event)

    if resource == "/validate-data":
        validate_data = _load_handler("agents", "validate_data")
        return validate_data(event)

    if resource == "/reports":
        generate_report = _load_handler("agents", "generate_report")
        return generate_report(event, caller_sub)

    if resource == "/compare-experiments":
        compare_experiments = _load_handler("agents", "compare_experiments")
        return compare_experiments(event, caller_sub)

    # Should be unreachable: only known resources reach _dispatch.
    raise _HandlerUnavailable(f"no dispatch for resource {resource!r}")


# --------------------------------------------------------------------------- #
# Routing tables
# --------------------------------------------------------------------------- #

#: All recognized ``(method, resource)`` pairs.
_ROUTES: frozenset[Tuple[str, str]] = frozenset(
    {
        ("POST", "/evaluate"),
        ("POST", "/batch-evaluate"),
        ("GET", "/jobs/{jobId}"),
        ("GET", "/experiments"),
        ("GET", "/experiments/{experimentId}"),
        ("POST", "/payloads"),
        ("GET", "/metrics"),
        ("GET", "/agent-types"),
        ("GET", "/agentcore/runtimes"),
        ("POST", "/invoke-evaluate"),
        ("POST", "/validate-data"),
        ("POST", "/reports"),
        ("POST", "/compare-experiments"),
    }
)

#: Allowed methods per known resource — used to distinguish 404 from 405.
_RESOURCE_METHODS: Dict[str, frozenset[str]] = {}
for _m, _r in _ROUTES:
    _RESOURCE_METHODS.setdefault(_r, set()).add(_m)  # type: ignore[arg-type]
_RESOURCE_METHODS = {r: frozenset(ms) for r, ms in _RESOURCE_METHODS.items()}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """API Gateway proxy entry point. Routes by ``(httpMethod, resource)``.

    Returns an API Gateway proxy response (``statusCode`` + JSON ``body``).
    Unknown resources yield ``404``; known resources hit with an unsupported
    method yield ``405``. Handlers not yet implemented yield ``501``. Any
    unexpected error is contained as ``500`` so a single bad route never crashes
    the Lambda.
    """
    event = event or {}
    method, resource = _route_key(event)

    if not method or not resource:
        return _error(400, "Malformed request: missing httpMethod or resource.")

    # Unknown resource path -> 404. Known resource, wrong method -> 405.
    allowed_methods = _RESOURCE_METHODS.get(resource)
    if allowed_methods is None:
        return _error(404, f"No such resource: {resource}")
    if method not in allowed_methods:
        response = _error(
            405,
            f"Method {method} not allowed for {resource}.",
        )
        response["headers"]["Allow"] = ", ".join(sorted(allowed_methods))
        return response

    caller_sub = _extract_caller_sub(event)
    if caller_sub is None:
        # The Cognito authorizer should have rejected this already; defensive 401.
        return _error(401, "Unauthorized: missing caller identity.")

    try:
        result = _dispatch(method, resource, event, caller_sub)
    except _HandlerUnavailable as exc:
        return _error(501, f"Not implemented: {exc}")
    except Exception as exc:  # noqa: BLE001 — contain handler errors as 500
        # Security review M-03: never return exception detail (boto3/S3/
        # DynamoDB errors can carry bucket names, keys, ARNs) to the caller.
        # Full detail goes to CloudWatch Logs, keyed by a correlation ID.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Internal server error"
        )
        return _error(500, safe_message, correlationId=correlation_id)

    return _normalize_result(result)
