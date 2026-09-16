# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curated-metric validation and library-only limitation enforcement.

This module is part of the standalone ``uaef-service`` deployable app. Per
**Requirement 6.2** it MUST NOT import the ``uaef`` package or any ``server``
extra — it depends only on the Python standard library. All evaluation work is
delegated to the Worker Lambda; this module is the synchronous request-path
gatekeeper that the API Lambda's :func:`handlers.jobs.create_evaluate_job`
(task 9.2) calls *before* a job is created or the Worker is invoked.

It enforces the documented library-only limitation (design.md "Known
Limitation"):

* **Requirement 11.1 / 11.3** — A service request may name only metrics in the
  ``Curated_Metric_Set`` (the trusted server-side catalog: built-ins plus
  installed integration metrics). Any other name — including a custom metric
  registered via ``register_metric`` — is rejected with a ``400`` whose message
  explains the limitation and points the user at library mode. No job is
  created and the Worker is never invoked (the API Lambda calls this function
  first and aborts on rejection).

* **Requirement 11.2** — A request that specifies a ``GenericJSONAdapter``
  schema mapping (arbitrary user-supplied schema-mapping Python in-process) is
  rejected with the same limitation explanation. No job, no Worker invoke.

Custom metrics and ``GenericJSONAdapter`` schema mappings remain fully
supported in **library mode** (Requirements 11.4 / 11.5); this boundary applies
to the hosted service only.

The trusted curated set is *sourced from the catalog*, not hardcoded: callers
pass the curated metric names (or the grouped catalog produced by the library's
``get_full_metric_catalog()`` via the ``/metrics`` handler) into
:func:`validate_evaluation_request`. This keeps the single source of truth in
the library and avoids a divergent hardcoded list in the service.

Requirements: 11.1, 11.2, 11.3
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Set

# A reference to library mode included in every limitation message so users
# always learn the supported escape hatch.
_LIBRARY_MODE_HINT = (
    "Custom metrics (register_metric) and GenericJSONAdapter schema mappings "
    "run user-supplied Python in-process and are a library-mode feature only. "
    "Install and use the uaef package directly (library mode) to use them; the "
    "hosted service accepts only the curated server-side metric set."
)

# Adapter identifiers that select the GenericJSON adapter. The library registers
# it under both ``generic`` (adapter registry key) and ``generic_json`` (the
# adapter's ``name``); we also tolerate a few obvious spellings so a request
# cannot slip a schema mapping past the boundary on a naming technicality.
_GENERIC_JSON_ADAPTER_ALIASES = frozenset(
    {"generic", "generic_json", "genericjson", "genericjsonadapter"}
)

# Body keys that indicate a user-supplied GenericJSON schema mapping.
_SCHEMA_MAPPING_KEYS = ("schema_mapping", "schemaMapping", "schema_map")


class ValidationRejection(Exception):
    """Raised when a service request violates the library-only limitation.

    Carries an HTTP ``status_code`` and a user-facing ``message`` so the API
    Lambda can turn it directly into an API Gateway proxy error response without
    creating a job or invoking the Worker.

    Attributes:
        status_code: HTTP status to return (``400`` for limitation rejections).
        message: Human-readable explanation of the limitation.
        code: Stable machine-readable error code for clients/tests.
    """

    def __init__(self, status_code: int, message: str, code: str = "VALIDATION_REJECTED") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code

    def to_error_body(self) -> dict:
        """Return a JSON-serializable error body for the API response."""
        return {"error": self.message, "code": self.code}


def _normalize_curated_metrics(curated_metrics: Any) -> Set[str]:
    """Flatten the curated metric source into a set of accepted metric names.

    Accepts either:
      * a flat iterable of metric-name strings, or
      * a grouped catalog (Mapping of dimension/integration -> names), matching
        the shape returned by the library's ``get_full_metric_catalog()``. Each
        group value may itself be a list of strings or a list of dicts carrying
        a ``name``/``metric``/``id`` key.

    Names are compared case-insensitively and surrounding whitespace is ignored.
    """
    names: Set[str] = set()

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized:
                names.add(normalized)
            return
        if isinstance(value, Mapping):
            for key in ("name", "metric", "id"):
                if key in value:
                    _add(value[key])
                    return
            # A nested mapping group (e.g. dimension -> [names]).
            for nested in value.values():
                _add(nested)
            return
        if isinstance(value, Iterable):
            for item in value:
                _add(item)
            return
        # Any other scalar type: coerce to its string form defensively.
        _add(str(value))

    _add(curated_metrics)
    return names


def _extract_requested_metric_names(body: Mapping[str, Any]) -> list[Any]:
    """Return the raw ``metrics`` entries from a request body.

    Returns the list verbatim (entries may be non-strings, e.g. an inline custom
    metric definition object), so the caller can reject those too.
    """
    metrics = body.get("metrics")
    if metrics is None:
        return []
    if isinstance(metrics, str):
        # A single metric name provided as a bare string.
        return [metrics]
    if isinstance(metrics, Iterable):
        return list(metrics)
    return [metrics]


def _has_schema_mapping(body: Mapping[str, Any]) -> bool:
    """Return True if the body carries a GenericJSON schema mapping."""
    for key in _SCHEMA_MAPPING_KEYS:
        value = body.get(key)
        if value:  # non-empty dict/list/str
            return True
    return False


def _is_generic_json_adapter(body: Mapping[str, Any]) -> bool:
    """Return True if the body selects the GenericJSON adapter."""
    adapter = body.get("adapter")
    if not isinstance(adapter, str):
        return False
    return adapter.strip().lower() in _GENERIC_JSON_ADAPTER_ALIASES


def validate_evaluation_request(
    body: Mapping[str, Any],
    curated_metrics: Any,
) -> None:
    """Validate a service evaluation request against the library-only limitation.

    Called by the API Lambda on the synchronous request path *before* any job is
    written or the Worker is invoked. Returns cleanly (``None``) when the request
    is acceptable; raises :class:`ValidationRejection` otherwise. Because the
    caller aborts on the raised exception, a rejected request creates no job and
    never reaches the Worker (Requirements 11.1, 11.2).

    Args:
        body: The parsed JSON request body (``EvaluateRequest`` /
            ``BatchEvaluateRequest`` shape, but inspected as a raw mapping so
            unmodeled fields like ``schema_mapping`` are still caught).
        curated_metrics: The trusted ``Curated_Metric_Set`` — either a flat
            iterable of accepted metric names or the grouped catalog from
            ``get_full_metric_catalog()``. Sourced from the catalog, not
            hardcoded.

    Raises:
        ValidationRejection: With ``status_code=400`` when the request specifies
            a GenericJSON schema mapping (Req 11.2) or any metric name absent
            from the curated set, including custom metrics (Req 11.1 / 11.3).
    """
    if not isinstance(body, Mapping):
        raise ValidationRejection(
            400,
            "Malformed request body: expected a JSON object.",
            code="MALFORMED_BODY",
        )

    # --- Requirement 11.2: reject GenericJSONAdapter schema mappings ---------
    if _has_schema_mapping(body) or _is_generic_json_adapter(body):
        raise ValidationRejection(
            400,
            "GenericJSONAdapter schema mappings are not supported in the hosted "
            "service. " + _LIBRARY_MODE_HINT,
            code="SCHEMA_MAPPING_NOT_SUPPORTED",
        )

    # --- Requirements 11.1 / 11.3: accept only curated metric names ----------
    curated = _normalize_curated_metrics(curated_metrics)
    requested = _extract_requested_metric_names(body)

    rejected: list[str] = []
    for entry in requested:
        if not isinstance(entry, str):
            # A non-string metric entry is an inline/custom metric definition —
            # arbitrary user-supplied logic that is library-mode only.
            rejected.append(repr(entry))
            continue
        if entry.strip().lower() not in curated:
            rejected.append(entry)

    if rejected:
        offending = ", ".join(rejected)
        raise ValidationRejection(
            400,
            f"Unsupported metric(s) requested: {offending}. The hosted service "
            f"accepts only the curated server-side metric set; custom metrics "
            f"registered via register_metric are not available. "
            + _LIBRARY_MODE_HINT,
            code="CUSTOM_METRIC_NOT_SUPPORTED",
        )

    # Acceptable request: returns None.
    return None


__all__ = ["ValidationRejection", "validate_evaluation_request"]
