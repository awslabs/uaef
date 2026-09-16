# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Presigned payload upload handler for the UAEF Service API Lambda.

This module backs ``POST /payloads`` and the request-path payload checks used by
job creation. It is part of the *separate* ``uaef-service`` deployable
application and, per **Requirement 6.2**, MUST NOT import the ``uaef`` package or
the ``server`` extra. It talks to S3 directly via ``boto3``.

API Gateway caps request bodies at 10MB. For large/batch payloads a client first
calls ``POST /payloads`` to obtain a short-lived presigned S3 ``PUT`` URL plus a
unique object key, uploads the payload to S3, then references it by key
(``traceRef`` / ``tracesRef``) in a subsequent evaluation request.

Responsibilities (see design.md "Payload handling"):

  * :func:`presign_payload_upload` — issue a presigned S3 ``PUT`` URL and a
    unique key that expires no later than 15 minutes after issuance and is
    scoped to a single key/upload (Requirements 7.1, 7.4). On failure it returns
    an error response and **no key** (Requirement 7.5).
  * :func:`ensure_inline_payload_within_limit` — detect a payload that exceeds
    the 10MB body limit sent *inline*, returning a ``413`` response with
    guidance to use the presigned path and creating no job (Requirement 7.3).
  * :func:`validate_payload_reference` — confirm a referenced S3 key exists via
    ``head_object``; reject a missing/expired reference without creating a job
    (Requirement 7.6).

Return-value contract: handler/helper functions return either a full API Gateway
proxy response (a dict containing ``statusCode``) for error/non-200 outcomes, or
a plain JSON-serializable object (the ``PresignResponse`` shape) that the router
wraps in a ``200 OK``. The reference/size helpers return ``None`` when the check
passes so callers can treat a truthy return as a short-circuit error response.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

try:  # package-relative when deployed as the ``handlers`` package
    from .. import job_state  # type: ignore[import-not-found]
except (ImportError, ValueError):  # flat layout / top-level module
    import job_state  # type: ignore[no-redef]

# --------------------------------------------------------------------------- #
# Configuration / constants
# --------------------------------------------------------------------------- #

#: Environment variable naming the S3 bucket used for presigned payload uploads.
#: Security review L-04: no default value. A hardcoded global-namespace bucket
#: name invites name-squatting/mis-targeting if the env var is ever unset in a
#: real deployment; the CDK stacks (StorageStack/ApiStack/WorkerStack/
#: OrchestratorStack) always set this explicitly, so an unset env var here
#: means a misconfigured deployment, not a normal state to silently paper over.
PAYLOAD_BUCKET_ENV = "PAYLOAD_BUCKET_NAME"

#: Presigned URL lifetime. Requirement 7.4: URLs expire no later than 15 minutes
#: after issuance. 900 seconds == 15 minutes (the hard ceiling, never exceeded).
PRESIGN_EXPIRY_SECONDS = 900

#: API Gateway request-body hard limit (Requirement 7.3). Payloads larger than
#: this must travel through the presigned S3 path rather than inline.
API_GATEWAY_BODY_LIMIT_BYTES = 10 * 1024 * 1024  # 10MB

#: Prefix under which per-caller uploads are scoped within the bucket.
_UPLOAD_PREFIX = "uploads"

#: Characters allowed in the caller-scoped key segment; everything else is
#: collapsed so a malformed ``sub`` can never escape the upload prefix.
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_-]+")

_JSON_HEADERS = {"Content-Type": "application/json"}


# --------------------------------------------------------------------------- #
# Proxy response helpers (kept local so this handler stands alone)
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
# S3 wiring
# --------------------------------------------------------------------------- #


def _payload_bucket() -> str:
    bucket = os.environ.get(PAYLOAD_BUCKET_ENV)
    if not bucket:
        raise RuntimeError(
            f"{PAYLOAD_BUCKET_ENV} is not set. This Lambda is misconfigured — "
            "the CDK deployment should always set this environment variable."
        )
    return bucket


_s3 = None


def _get_s3_client():
    """Return a cached boto3 S3 client that presigns with Signature V4.

    Two things matter for a browser presigned PUT to succeed:

    * **SigV4** (vs the legacy SigV2 default for the us-east-1 global endpoint)
      does NOT bake Content-Type into the presigned signature, so a browser PUT
      — which sets its own Content-Type from the File — won't fail with
      ``SignatureDoesNotMatch``.
    * **Regional virtual-hosted endpoint.** In us-east-1 a default client uses
      the *global* endpoint ``s3.amazonaws.com``; a virtual-hosted PUT there can
      answer with a 307 redirect to the regional endpoint. The browser's CORS
      preflight does NOT follow redirects, so the upload fails with an opaque
      ``TypeError: Failed to fetch`` (no status/body). Pinning the regional
      endpoint (``...s3.<region>.amazonaws.com``) removes the redirect so the
      preflight + PUT land directly on the bucket.
    """
    global _s3
    if _s3 is None:
        region = os.environ.get("AWS_REGION", "us-east-1")
        _s3 = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(
                signature_version="s3v4",
                s3={
                    "addressing_style": "virtual",
                    "us_east_1_regional_endpoint": "regional",
                },
            ),
        )
    return _s3


def _safe_segment(value: Optional[str]) -> str:
    """Sanitize ``value`` into a single safe S3 key segment.

    Collapses any run of disallowed characters to ``-`` and trims leading/
    trailing separators so a caller ``sub`` can never inject ``/`` or ``..`` and
    escape the per-caller upload prefix. Falls back to ``anonymous`` when empty.
    """
    text = (value or "").strip()
    cleaned = _SAFE_SEGMENT.sub("-", text).strip("-")
    return cleaned or "anonymous"


def _generate_key(caller_sub: Optional[str]) -> str:
    """Generate a unique S3 object key scoped under the caller's prefix.

    Shape: ``uploads/<caller-sub>/<uuid4>``. The ``uuid4`` guarantees a single,
    unguessable key per upload so each presigned URL is scoped to exactly one
    object (Requirement 7.4).
    """
    return f"{_UPLOAD_PREFIX}/{_safe_segment(caller_sub)}/{uuid.uuid4()}"


# --------------------------------------------------------------------------- #
# POST /payloads — issue a presigned PUT URL
# --------------------------------------------------------------------------- #


def presign_payload_upload(caller_sub: str) -> Dict[str, Any]:
    """Issue a presigned S3 ``PUT`` URL and a unique key for one upload.

    Generates a unique object key scoped under the caller's prefix and a
    presigned ``put_object`` URL that expires in :data:`PRESIGN_EXPIRY_SECONDS`
    (15 minutes — the ceiling required by 7.4) and is scoped to that single key.

    Returns a ``PresignResponse``-shaped dict
    ``{"uploadUrl", "key", "expiresIn"}`` on success (the router wraps it in a
    ``200``). On any failure to generate the URL, returns an error proxy
    response and **no key** (Requirement 7.5).

    Args:
        caller_sub: The caller's Cognito ``sub`` claim; used to scope the key.

    Returns:
        A ``PresignResponse``-shaped dict on success, or an error proxy response
        (``statusCode`` 502) carrying no ``key`` on failure.
    """
    bucket = _payload_bucket()
    key = _generate_key(caller_sub)

    try:
        upload_url = _get_s3_client().generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=PRESIGN_EXPIRY_SECONDS,
        )
    except (ClientError, BotoCoreError, ValueError) as exc:
        # Requirement 7.5: report the upload URL could not be issued and return
        # NO object key. Security review M-03: don't return S3/boto3 exception
        # detail (which can include the bucket name / key).
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Could not issue a presigned upload URL"
        )
        return _error(502, safe_message, correlationId=correlation_id)

    # PresignResponse shape (schemas.PresignResponse): uploadUrl, key, expiresIn.
    return {
        "uploadUrl": upload_url,
        "key": key,
        "expiresIn": PRESIGN_EXPIRY_SECONDS,
    }


# --------------------------------------------------------------------------- #
# Inline oversized-payload guard (Requirement 7.3)
# --------------------------------------------------------------------------- #


def _inline_body_size_bytes(raw_body: Any) -> int:
    """Best-effort byte length of an inline request body."""
    if raw_body is None:
        return 0
    if isinstance(raw_body, (bytes, bytearray)):
        return len(raw_body)
    if isinstance(raw_body, str):
        return len(raw_body.encode("utf-8"))
    # Non-string structured body: measure its JSON encoding.
    return len(json.dumps(raw_body, default=str).encode("utf-8"))


def is_inline_payload_oversized(raw_body: Any) -> bool:
    """Return ``True`` iff an inline body exceeds the 10MB API Gateway limit."""
    return _inline_body_size_bytes(raw_body) > API_GATEWAY_BODY_LIMIT_BYTES


def oversized_inline_payload_response() -> Dict[str, Any]:
    """Build the ``413`` response guiding the caller to the presigned path."""
    return _error(
        413,
        (
            "Request payload exceeds the 10MB API Gateway body limit. Upload the "
            "payload via POST /payloads to obtain a presigned S3 URL, then "
            "reference it by key (traceRef/tracesRef) in your request."
        ),
        limitBytes=API_GATEWAY_BODY_LIMIT_BYTES,
        uploadPath="/payloads",
    )


def ensure_inline_payload_within_limit(raw_body: Any) -> Optional[Dict[str, Any]]:
    """Return a ``413`` response if ``raw_body`` is oversized, else ``None``.

    Requirement 7.3: an oversized inline payload yields a ``413`` with guidance
    to use the presigned path and **no job is created** — the caller short-
    circuits on a truthy return before any job write.
    """
    if is_inline_payload_oversized(raw_body):
        return oversized_inline_payload_response()
    return None


# --------------------------------------------------------------------------- #
# Referenced-payload existence check (Requirement 7.6)
# --------------------------------------------------------------------------- #


def payload_reference_exists(key: str) -> bool:
    """Return ``True`` iff an object exists at ``key`` in the payload bucket.

    Uses ``head_object``; a missing object (never uploaded, or whose presigned
    URL expired before upload) returns ``False``. Unexpected S3 errors propagate
    so they are not silently treated as "missing".
    """
    if not key:
        return False
    try:
        _get_s3_client().head_object(Bucket=_payload_bucket(), Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        # 404 / NoSuchKey / NotFound (and 403 on a non-existent key under some
        # bucket policies) all mean the referenced payload is unavailable.
        if code in {"404", "NoSuchKey", "NotFound", "403", "Forbidden"}:
            return False
        raise


def validate_payload_reference(key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return an error response if a referenced key is unavailable, else ``None``.

    Requirement 7.6: an evaluation request referencing an S3 key for which no
    object was uploaded (or whose presigned URL expired) is rejected with an
    error indicating the referenced payload is unavailable, and **no job is
    created** — the caller short-circuits on a truthy return.

    Returns ``None`` when ``key`` is falsy (no reference supplied — nothing to
    validate) or when the object exists.
    """
    if not key:
        return None
    try:
        exists = payload_reference_exists(key)
    except (ClientError, BotoCoreError) as exc:
        # Security review M-03: don't return S3/boto3 exception detail. `key`
        # is safe to echo back — it's the caller's own reference, already
        # known to them from the presign response.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Could not verify the referenced payload"
        )
        return _error(502, safe_message, key=key, correlationId=correlation_id)
    if not exists:
        return _error(
            400,
            (
                "Referenced payload is unavailable: no object was uploaded for "
                "this key, or its presigned upload URL has expired. Request a "
                "new presigned URL via POST /payloads and re-upload."
            ),
            key=key,
        )
    return None
