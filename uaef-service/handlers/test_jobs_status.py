# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Unit tests for ``handlers.jobs.get_job_status`` (task 9.3).

Covers the per-caller authorization and status-shaping behaviour required by
Requirements 4.4, 4.7 and 9.4:

  * 404 when the Job does not exist (Req 4.7).
  * 403 when ``createdBy`` does not match the caller, disclosing no status,
    result reference, or summary (Req 9.4).
  * COMPLETED status includes ``experimentId``, ``resultSummary`` and a
    presigned-GET ``resultRef`` (Req 4.4).
  * FAILED status includes the ``error`` ``{code, message}`` payload (Req 4.4).
  * PENDING/PROCESSING status omits result/error fields.

These tests use in-memory fakes for the DynamoDB table and the S3 client so they
exercise the real handler logic without AWS. They are written to run either
under pytest or directly via ``python3 handlers/test_jobs_status.py``.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import job_state  # noqa: E402  (top-level module in the app root)
from handlers import jobs  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeTable:
    """Minimal stand-in for a boto3 DynamoDB Table supporting ``get_item``."""

    def __init__(self, items):
        # items: dict keyed by jobId
        self._items = items

    def get_item(self, Key):  # noqa: N803 — match boto3 kwarg name
        item = self._items.get(Key["jobId"])
        return {"Item": item} if item is not None else {}


class _FakeS3:
    """Minimal stand-in returning a deterministic presigned URL."""

    def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
        return (
            f"https://s3.example/{Params['Bucket']}/{Params['Key']}"
            f"?op={op}&exp={ExpiresIn}"
        )


def _install(monkeypatch_items):
    """Point the handler at fake AWS resources holding ``monkeypatch_items``."""
    table = _FakeTable(monkeypatch_items)
    jobs.job_state._get_table = lambda: table  # type: ignore[attr-defined]
    s3 = _FakeS3()
    jobs._get_s3_client = lambda: s3  # type: ignore[attr-defined]


def _body(response):
    """Decode a proxy-response body into a dict."""
    return json.loads(response["body"])


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_missing_job_returns_404():
    _install({})
    resp = jobs.get_job_status("nope", "caller-1")
    assert resp["statusCode"] == 404, resp
    assert "not found" in _body(resp)["error"].lower()


def test_wrong_owner_returns_403_without_disclosure():
    _install(
        {
            "job-1": {
                "jobId": "job-1",
                "status": job_state.COMPLETED,
                "operation": "evaluate",
                "createdBy": "owner-sub",
                "createdAt": "t0",
                "updatedAt": "t1",
                "experimentId": "exp-9",
                "resultSummary": {"overallScore": 0.9},
                "resultRef": "results/job-1.json",
            }
        }
    )
    resp = jobs.get_job_status("job-1", "someone-else")
    assert resp["statusCode"] == 403, resp
    payload = _body(resp)
    # Req 9.4: must not disclose status, result reference, or summary.
    for leaked in ("status", "resultRef", "resultSummary", "experimentId"):
        assert leaked not in payload, f"403 leaked {leaked}: {payload}"


def test_completed_includes_summary_ref_and_presigned_url():
    _install(
        {
            "job-2": {
                "jobId": "job-2",
                "status": job_state.COMPLETED,
                "operation": "evaluate",
                "createdBy": "owner",
                "createdAt": "t0",
                "updatedAt": "t1",
                "experimentId": "exp-42",
                "resultSummary": {"overallScore": 0.75, "passed": True},
                "resultRef": "results/job-2.json",
            }
        }
    )
    resp = jobs.get_job_status("job-2", "owner")
    # Plain dict (router wraps in 200), not a proxy error response.
    assert "statusCode" not in resp, resp
    assert resp["status"] == job_state.COMPLETED
    assert resp["operation"] == "evaluate"
    assert resp["experimentId"] == "exp-42"
    assert resp["resultSummary"] == {"overallScore": 0.75, "passed": True}
    # resultRef is now a presigned GET URL for the stored key.
    assert resp["resultRef"].startswith("https://s3.example/")
    assert "results/job-2.json" in resp["resultRef"]
    assert "op=get_object" in resp["resultRef"]
    # No error field on a successful job.
    assert "error" not in resp


def test_failed_includes_error_code_and_message():
    _install(
        {
            "job-3": {
                "jobId": "job-3",
                "status": job_state.FAILED,
                "operation": "batch_evaluate",
                "createdBy": "owner",
                "createdAt": "t0",
                "updatedAt": "t1",
                "error": {"code": "ValueError", "message": "bad trace"},
            }
        }
    )
    resp = jobs.get_job_status("job-3", "owner")
    assert "statusCode" not in resp, resp
    assert resp["status"] == job_state.FAILED
    assert resp["error"] == {"code": "ValueError", "message": "bad trace"}
    # No result fields on a failed job.
    assert "resultRef" not in resp
    assert "resultSummary" not in resp


def test_pending_omits_result_and_error_fields():
    _install(
        {
            "job-4": {
                "jobId": "job-4",
                "status": job_state.PENDING,
                "operation": "evaluate",
                "createdBy": "owner",
                "createdAt": "t0",
                "updatedAt": "t0",
            }
        }
    )
    resp = jobs.get_job_status("job-4", "owner")
    assert "statusCode" not in resp, resp
    assert resp["status"] == job_state.PENDING
    for absent in ("resultRef", "resultSummary", "experimentId", "error"):
        assert absent not in resp, f"PENDING should omit {absent}: {resp}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
