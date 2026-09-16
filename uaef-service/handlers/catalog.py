# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Metric-catalog route handler for the UAEF Service API Lambda.

Implements ``GET /metrics`` — the *curated server-side metric catalog*. The
catalog lists every metric name the service will accept, grouped by dimension
and integration (built-ins from the metric registry + RAGAS / DeepEval /
Stickler from the integrations registry).

Performance strategy — static catalog.json:
    The catalog is generated at **build time** by ``scripts/generate_catalog.py``
    and baked into the API Lambda's deployment package as ``catalog.json``. This
    means ``GET /metrics`` returns instantly from a pre-computed file with zero
    cold-start latency — no Worker Lambda invoke needed.

    If ``catalog.json`` is missing (e.g. local dev without a build step), the
    handler falls back to invoking the Worker Lambda synchronously as before.

Requirements: 11.3, 12.7
    - The catalog MUST be sourced from the library's ``get_full_metric_catalog()``
      (it is — at build time via the generate script).
    - The API Lambda MUST NOT import or bundle ``uaef`` (it doesn't — it reads
      a pre-generated JSON file).
    - Contains NO hardcoded metric-name list (the JSON is generated from the
      library at each deploy).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import boto3

try:  # package-relative when deployed as the ``handlers`` package
    from .. import job_state  # type: ignore[import-not-found]
except (ImportError, ValueError):  # flat layout / top-level module
    import job_state  # type: ignore[no-redef]

# --------------------------------------------------------------------------- #
# Static catalog (generated at build time by scripts/generate_catalog.py)
# --------------------------------------------------------------------------- #

_CATALOG_JSON_PATH = Path(__file__).resolve().parent / "catalog.json"

_JSON_HEADERS = {"Content-Type": "application/json"}

# --------------------------------------------------------------------------- #
# Configuration (fallback: Worker Lambda invoke)
# --------------------------------------------------------------------------- #

WORKER_FUNCTION_ENV = "WORKER_FUNCTION_NAME"
CATALOG_ACTION = "get_metric_catalog"

_lambda_client = None
_catalog_cache: Optional[Dict[str, Any]] = None


def _get_lambda_client():
    """Return a cached boto3 Lambda client (created on first use)."""
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")
    return _lambda_client


def _worker_function_name() -> Optional[str]:
    """Return the configured Worker Lambda function name, if any."""
    name = os.environ.get(WORKER_FUNCTION_ENV)
    return name if name else None


# --------------------------------------------------------------------------- #
# Response helpers
# --------------------------------------------------------------------------- #


def _error(status_code: int, message: str, **extra: Any) -> Dict[str, Any]:
    """Build an API Gateway proxy error response with a JSON body."""
    body: Dict[str, Any] = {"error": message}
    body.update(extra)
    return {
        "statusCode": status_code,
        "headers": dict(_JSON_HEADERS),
        "body": json.dumps(body, default=str),
    }


# --------------------------------------------------------------------------- #
# Static file loader
# --------------------------------------------------------------------------- #


def _load_static_catalog() -> Optional[Dict[str, Any]]:
    """Load the pre-generated catalog.json if it exists.

    Returns None if the file is missing (triggers fallback to Worker invoke).
    """
    if not _CATALOG_JSON_PATH.exists():
        return None
    try:
        return json.loads(_CATALOG_JSON_PATH.read_text())
    except (ValueError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Fallback: Worker invocation (used only if catalog.json is absent)
# --------------------------------------------------------------------------- #


def _invoke_worker_for_catalog(function_name: str) -> Dict[str, Any]:
    """Synchronously invoke the Worker Lambda and return the parsed catalog."""
    client = _get_lambda_client()
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps({"action": CATALOG_ACTION}).encode("utf-8"),
    )

    if response.get("FunctionError"):
        detail = _read_payload_text(response)
        raise RuntimeError(f"Worker returned an error: {detail}")

    payload_text = _read_payload_text(response)
    if not payload_text:
        raise RuntimeError("Worker returned an empty catalog payload.")

    try:
        catalog = json.loads(payload_text)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Worker returned a non-JSON catalog payload: {exc}") from exc

    catalog = _unwrap_catalog(catalog)

    if not isinstance(catalog, dict):
        raise RuntimeError(
            "Worker returned an unexpected catalog shape "
            f"(expected an object, got {type(catalog).__name__})."
        )
    return catalog


def _read_payload_text(response: Dict[str, Any]) -> str:
    """Read the Payload streaming body from an invoke response as text."""
    payload = response.get("Payload")
    if payload is None:
        return ""
    raw = payload.read() if hasattr(payload, "read") else payload
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _unwrap_catalog(payload: Any) -> Any:
    """Unwrap the catalog if the Worker wrapped it in an envelope."""
    if isinstance(payload, dict):
        if "catalog" in payload and isinstance(payload["catalog"], dict):
            return payload["catalog"]
        if "statusCode" in payload and "body" in payload:
            body = payload["body"]
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except (ValueError, TypeError):
                    return body
            if isinstance(body, dict) and "catalog" in body and isinstance(body["catalog"], dict):
                return body["catalog"]
            return body
    return payload


# --------------------------------------------------------------------------- #
# Route entry point
# --------------------------------------------------------------------------- #


def get_metric_catalog() -> Dict[str, Any]:
    """Return the curated server-side metric catalog.

    Strategy:
        1. Load from pre-generated ``catalog.json`` (instant, no cold start).
        2. Fallback: invoke the Worker Lambda (only if catalog.json is missing).

    Contains NO hardcoded metric-name list (Requirement 12.7) and performs NO
    ``uaef`` import in the API Lambda (Requirement 6.2).
    """
    global _catalog_cache

    if _catalog_cache is not None:
        return _catalog_cache

    # Strategy 1: static file (fast path).
    catalog = _load_static_catalog()
    if catalog is not None:
        _catalog_cache = catalog
        return catalog

    # Strategy 2: fallback to Worker invoke (slow path — cold start).
    function_name = _worker_function_name()
    if function_name is None:
        return _error(
            503,
            "Metric catalog is temporarily unavailable: no catalog.json and no worker configured.",
        )

    try:
        catalog = _invoke_worker_for_catalog(function_name)
    except Exception as exc:  # noqa: BLE001
        # Security review M-03: don't return Lambda-invoke exception detail.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Failed to retrieve the metric catalog from the worker"
        )
        return _error(502, safe_message, correlationId=correlation_id)

    _catalog_cache = catalog
    return catalog
