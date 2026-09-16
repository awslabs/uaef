# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Full-result offload to S3 for the UAEF Service Worker Lambda (task 10.3).

This module owns the **retrieval offload** of a completed evaluation's full
result: it serializes the entire ``EvaluationResult`` (or the list produced by
``batch_evaluate``) to JSON and writes it as a single object to the payload
S3 bucket, returning the object key (``resultRef``) the Worker records on the
``COMPLETED`` Job. The API Lambda later hands clients a presigned GET for that
key (see ``handlers/jobs.get_job_status``).

Persistence boundary (Requirements 8.2, 8.3, 8.4):

  * Actual *experiment* persistence — the Experiment_Store DynamoDB rows **and**
    the canonical S3 result JSON — is performed by UAEF_Library itself when the
    service request passes ``persist=True`` through to ``uaef.api.evaluate`` /
    ``batch_evaluate``. This module does **not** duplicate or re-implement that
    persistence; doing so would fork logic the library already owns
    (Requirement 8.5 / 12.1) and risk schema divergence.
  * The Worker only calls ``_store_result`` when ``persist=True`` (it offloads
    the full result for retrieval). When ``persist`` is omitted or ``False`` the
    Worker never calls this module, so nothing is written to the Payload_Store
    (Requirement 8.3).
  * No partial S3 object is left behind on a serialization failure
    (Requirement 8.4): the full JSON payload is serialized *in memory first* and
    only a single, complete ``put_object`` is issued. If serialization raises,
    the function raises before any S3 call — there is no partial write to clean
    up — and the Worker turns that into a bounded ``FAILED`` Job.

This module is part of the standalone ``uaef-service`` deployable app. It may
use ``boto3`` but it deliberately does **not** ``import uaef`` — it only relies
on the duck-typed pydantic ``model_dump`` method when present.

Requirements: 8.2, 8.3, 8.4
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

#: S3 bucket holding request/result payloads offloaded from API Gateway's body
#: limit. Shared with the Worker/API Lambdas via the ``PAYLOAD_BUCKET_NAME`` env
#: var so all components agree on the location of the stored full result.
#: Security review L-04: no default value — see ``handlers/payloads.py`` for
#: rationale.
PAYLOAD_BUCKET_ENV = "PAYLOAD_BUCKET_NAME"

#: Key prefix under which full results are stored: ``results/<jobId>.json``.
RESULT_KEY_PREFIX = "results"


def _payload_bucket() -> str:
    bucket = os.environ.get(PAYLOAD_BUCKET_ENV)
    if not bucket:
        raise RuntimeError(
            f"{PAYLOAD_BUCKET_ENV} is not set. This Lambda is misconfigured — "
            "the CDK deployment should always set this environment variable."
        )
    return bucket


def _result_key(job_id: str) -> str:
    """Return the deterministic S3 key for a job's stored full result."""
    return f"{RESULT_KEY_PREFIX}/{job_id}.json"


_s3_client = None


def _get_s3():
    """Return a cached boto3 S3 client (lazily created)."""
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3")
    return _s3_client


# --------------------------------------------------------------------------- #
# Serialization (pydantic-aware, no uaef import)
# --------------------------------------------------------------------------- #


def _dump_one(obj: Any) -> Any:
    """Convert a single result object to a JSON-ready value.

    Prefers pydantic v2 ``model_dump(mode="json")`` (which the UAEF
    ``EvaluationResult`` model provides) so enums, datetimes, and nested models
    serialize exactly as library-mode persistence would. Falls back to the
    object itself, letting ``json.dumps(default=str)`` handle the remainder.
    """
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return obj


def _serialize_result(result: Any) -> str:
    """Serialize an ``EvaluationResult`` (or list of them) to a JSON string.

    The entire payload is produced here, before any S3 call, so a failure to
    serialize cannot leave a partial object in the Payload_Store (Req 8.4).
    """
    if isinstance(result, list):
        payload: Any = [_dump_one(r) for r in result]
    else:
        payload = _dump_one(result)
    return json.dumps(payload, default=str)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _store_result(job_id: str, result: Any) -> str:
    """Write the full result JSON to S3 and return its ``resultRef`` key.

    Steps (order matters for Requirement 8.4):
      1. Serialize the complete result to JSON *in memory*. If this raises, no
         S3 write has occurred and there is nothing partial to clean up.
      2. Issue a single ``put_object`` with the fully-formed body.
      3. Return the object key for the Worker to record as ``resultRef``.

    This is the retrieval offload only; the canonical experiment persistence is
    performed by UAEF_Library via ``persist=True`` and is not duplicated here.
    """
    # 1. Serialize fully first — guarantees no partial S3 object on failure.
    body = _serialize_result(result).encode("utf-8")

    # 2. Single, complete write.
    key = _result_key(job_id)
    _get_s3().put_object(
        Bucket=_payload_bucket(),
        Key=key,
        Body=body,
        ContentType="application/json",
    )

    # 3. Return the reference key.
    return key
