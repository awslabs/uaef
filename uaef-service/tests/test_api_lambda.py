# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Unit tests for the UAEF Service **API Lambda** (task 9.7).

Code under test (the API Lambda and its dispatch path — none of which imports
the ``uaef`` package, per Requirement 6.2):

  * ``handlers/api.py``      — the router (dispatch by ``(httpMethod, resource)``)
  * ``handlers/jobs.py``     — job creation + status read
  * ``handlers/payloads.py`` — presigned upload + inline-size / reference guards
  * ``validation.py``        — curated-metric / library-only limitation gatekeeper
  * ``job_state.py``         — DynamoDB-backed job lifecycle (used by the above)

What is covered (and the requirement each maps to):

  * Router dispatch by ``(httpMethod, resource)``; unknown route -> 404,
    wrong method -> 405, missing auth -> 401, valid route reaches its handler.
  * ``create_evaluate_job`` returns 400 (no Job) on malformed / schema-invalid
    bodies                                                            (Req 4.6).
  * Async-invoke path uses ``InvocationType='Event'`` and returns a ``jobId``
    without performing synchronous evaluation                    (Req 4.3, 9.3).
  * The created ``PENDING`` Job records the caller's Cognito ``sub`` as
    ``createdBy``                                                     (Req 9.3).
  * On enqueue failure the Job is driven to ``FAILED`` and an error is returned
                                                                       (Req 4.5).
  * ``get_job_status`` -> 404 for a missing Job (Req 4.7); -> 403 for a non-owner
    with no status/result/summary disclosed                          (Req 9.4).
  * Oversized inline payload -> 413 with presigned-path guidance, no Job
                                                                       (Req 7.3).
  * Limitation rejections: a custom (non-curated) metric -> 400 and a
    ``GenericJSONAdapter`` schema mapping -> 400, each creating **no** Job and
    invoking **no** Worker                                     (Req 11.1, 11.2).
  * The API Lambda's dispatch path never imports ``uaef``           (Req 6.2).

Test environment notes:
  * DynamoDB (the jobs table) is provided by ``moto``.
  * The async Worker invoke is exercised via a stubbed boto3 Lambda client
    (``jobs._get_lambda_client``) that records the ``InvocationType`` and can
    simulate enqueue success/failure.
  * ``uaef`` is intentionally NOT installed in this environment, so the
    "never imports uaef" assertions are meaningful: importing/invoking the API
    Lambda must not pull ``uaef`` into ``sys.modules``.

Validates: Requirements 4.3, 4.6, 4.7, 6.2, 7.3, 9.4, 11.1, 11.2
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys

# --------------------------------------------------------------------------- #
# Path + AWS env setup (before importing the app modules / boto3)
# --------------------------------------------------------------------------- #

# Add the uaef-service app root to sys.path so ``import job_state``,
# ``import schemas``, ``import validation`` and ``from handlers import ...``
# resolve regardless of the pytest invocation directory.
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

# moto needs a region + (dummy) credentials.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

import boto3  # noqa: E402
import pytest  # noqa: E402
from moto import mock_aws  # noqa: E402

import job_state  # noqa: E402
import validation as validation_mod  # noqa: E402
from handlers import api as api_mod  # noqa: E402
from handlers import jobs as jobs_mod  # noqa: E402
from handlers import payloads as payloads_mod  # noqa: E402

JOBS_TABLE = "uaef-service-jobs"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeLambdaClient:
    """Stub boto3 Lambda client recording invocations.

    ``invoke`` records its kwargs so tests can assert ``InvocationType='Event'``.
    Configurable to simulate a successful enqueue (202), a non-2xx status, a
    ``FunctionError`` envelope, or a raised client error.
    """

    def __init__(self, status_code=202, function_error=None, raise_exc=None):
        self.invocations = []
        self._status_code = status_code
        self._function_error = function_error
        self._raise_exc = raise_exc

    def invoke(self, **kwargs):  # noqa: N803 — match boto3 kwarg names
        self.invocations.append(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        resp = {"StatusCode": self._status_code}
        if self._function_error is not None:
            resp["FunctionError"] = self._function_error
        return resp


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def aws():
    """Provide a mocked AWS environment with the jobs table created."""
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        ddb.create_table(
            TableName=JOBS_TABLE,
            KeySchema=[{"AttributeName": "jobId", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "jobId", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        # Reset cached clients so they bind to the mocked services.
        job_state._table = None
        jobs_mod._lambda_client = None
        jobs_mod._s3_client = None
        os.environ["JOBS_TABLE_NAME"] = JOBS_TABLE
        os.environ["WORKER_FUNCTION_NAME"] = "uaef-service-worker"
        yield ddb
        job_state._table = None
        jobs_mod._lambda_client = None
        jobs_mod._s3_client = None


@pytest.fixture
def fake_lambda(monkeypatch):
    """Install a recording fake Lambda client; return it for assertions."""
    client = FakeLambdaClient()
    monkeypatch.setattr(jobs_mod, "_get_lambda_client", lambda: client)
    return client


def _jobs_table():
    return boto3.resource("dynamodb", region_name="us-east-1").Table(JOBS_TABLE)


def _scan_count():
    return _jobs_table().scan().get("Count", 0)


def _proxy_event(method, resource, *, sub="user-123", body=None, path_params=None):
    """Build an API Gateway proxy event."""
    event = {
        "httpMethod": method,
        "resource": resource,
        "pathParameters": path_params or {},
    }
    if sub is not None:
        event["requestContext"] = {"authorizer": {"claims": {"sub": sub}}}
    if body is not None:
        event["body"] = body if isinstance(body, str) else json.dumps(body)
    return event


def _status(resp):
    return resp.get("statusCode")


def _body(resp):
    raw = resp.get("body")
    return json.loads(raw) if isinstance(raw, str) else raw


# =========================================================================== #
# Router dispatch (Req 6.2 routing surface)
# =========================================================================== #


def test_router_unknown_resource_returns_404():
    resp = api_mod.handler(_proxy_event("GET", "/does-not-exist"))
    assert _status(resp) == 404
    assert "no such resource" in _body(resp)["error"].lower()


def test_router_wrong_method_returns_405_with_allow_header():
    resp = api_mod.handler(_proxy_event("DELETE", "/evaluate"))
    assert _status(resp) == 405
    # The Allow header advertises the supported method(s).
    assert "POST" in resp["headers"].get("Allow", "")


def test_router_missing_auth_returns_401():
    # Valid route + method, but no Cognito claims -> defensive 401.
    resp = api_mod.handler(_proxy_event("POST", "/evaluate", sub=None, body={"trace": {}}))
    assert _status(resp) == 401
    assert "unauthorized" in _body(resp)["error"].lower()


def test_router_malformed_event_returns_400():
    resp = api_mod.handler({})
    assert _status(resp) == 400


def test_router_dispatches_valid_evaluate_to_job_creation(aws, fake_lambda):
    # POST /evaluate with auth + a schema-valid body should reach
    # create_evaluate_job, which returns a jobId (router wraps plain dict in 200).
    resp = api_mod.handler(
        _proxy_event("POST", "/evaluate", sub="router-user", body={"trace": {"x": 1}})
    )
    assert _status(resp) == 200, resp
    assert "jobId" in _body(resp)
    # Async invoke happened exactly once via Event.
    assert len(fake_lambda.invocations) == 1
    assert fake_lambda.invocations[0]["InvocationType"] == "Event"


def test_router_dispatches_jobs_get_to_status_handler(aws):
    # Seed a job owned by the caller; GET /jobs/{jobId} should return it.
    job_state.create_job("job-router", "evaluate", created_by="router-user")
    resp = api_mod.handler(
        _proxy_event(
            "GET",
            "/jobs/{jobId}",
            sub="router-user",
            path_params={"jobId": "job-router"},
        )
    )
    assert _status(resp) == 200, resp
    assert _body(resp)["status"] == job_state.PENDING


# =========================================================================== #
# create_evaluate_job — 400 on bad body, no Job (Req 4.6)
# =========================================================================== #


def test_create_job_malformed_json_returns_400_no_job(aws, fake_lambda):
    event = _proxy_event("POST", "/evaluate", body="this is not json{{")
    resp = jobs_mod.create_evaluate_job(event, "evaluate")
    assert _status(resp) == 400
    assert _scan_count() == 0
    assert fake_lambda.invocations == []


def test_create_job_schema_invalid_returns_400_no_job(aws, fake_lambda):
    # ``trace`` must be an object; a string fails schema validation.
    event = _proxy_event("POST", "/evaluate", body={"trace": "not-a-dict"})
    resp = jobs_mod.create_evaluate_job(event, "evaluate")
    assert _status(resp) == 400
    assert _scan_count() == 0
    assert fake_lambda.invocations == []


def test_create_job_missing_body_returns_400_no_job(aws, fake_lambda):
    event = _proxy_event("POST", "/evaluate")  # no body
    resp = jobs_mod.create_evaluate_job(event, "evaluate")
    assert _status(resp) == 400
    assert _scan_count() == 0


# =========================================================================== #
# Async-invoke path + createdBy recording (Req 4.3, 9.3)
# =========================================================================== #


def test_create_job_records_createdby_and_pending(aws, fake_lambda):
    event = _proxy_event("POST", "/evaluate", sub="cognito-sub-42", body={"trace": {"a": 1}})
    resp = jobs_mod.create_evaluate_job(event, "evaluate")

    # Plain JobCreatedResponse dict (router wraps in 200); not a proxy error.
    assert "statusCode" not in resp, resp
    job_id = resp["jobId"]
    assert resp["status"] == job_state.PENDING

    item = _jobs_table().get_item(Key={"jobId": job_id})["Item"]
    assert item["createdBy"] == "cognito-sub-42"  # Req 9.3
    assert item["status"] == job_state.PENDING


def test_create_job_async_invoke_uses_event_type(aws, fake_lambda):
    event = _proxy_event("POST", "/batch-evaluate", body={"traces": [{"a": 1}]})
    resp = jobs_mod.create_evaluate_job(event, "batch_evaluate")
    assert "statusCode" not in resp, resp

    assert len(fake_lambda.invocations) == 1
    call = fake_lambda.invocations[0]
    # Req 4.3: asynchronous (Event) invoke — no synchronous evaluation.
    assert call["InvocationType"] == "Event"
    assert call["FunctionName"] == "uaef-service-worker"
    payload = json.loads(call["Payload"].decode("utf-8"))
    assert payload["jobId"] == resp["jobId"]
    assert payload["operation"] == "batch_evaluate"


# =========================================================================== #
# Enqueue failure drives the Job to FAILED (Req 4.5)
# =========================================================================== #


def test_create_job_enqueue_failure_marks_failed(aws, monkeypatch):
    failing = FakeLambdaClient(raise_exc=RuntimeError("throttled"))
    monkeypatch.setattr(jobs_mod, "_get_lambda_client", lambda: failing)

    event = _proxy_event("POST", "/evaluate", body={"trace": {"a": 1}})
    resp = jobs_mod.create_evaluate_job(event, "evaluate")

    assert _status(resp) == 502, resp
    job_id = _body(resp)["jobId"]
    item = _jobs_table().get_item(Key={"jobId": job_id})["Item"]
    assert item["status"] == job_state.FAILED  # Req 4.5
    assert item["error"]["code"] == "ENQUEUE_FAILED"


def test_create_job_enqueue_bad_status_marks_failed(aws, monkeypatch):
    bad = FakeLambdaClient(status_code=500)
    monkeypatch.setattr(jobs_mod, "_get_lambda_client", lambda: bad)

    event = _proxy_event("POST", "/evaluate", body={"trace": {"a": 1}})
    resp = jobs_mod.create_evaluate_job(event, "evaluate")
    assert _status(resp) == 502, resp
    item = _jobs_table().get_item(Key={"jobId": _body(resp)["jobId"]})["Item"]
    assert item["status"] == job_state.FAILED


# =========================================================================== #
# get_job_status — 404 missing, 403 non-owner (Req 4.7, 9.4)
# =========================================================================== #


def test_get_job_status_missing_returns_404(aws):
    resp = jobs_mod.get_job_status("no-such-job", "anyone")
    assert _status(resp) == 404


def test_get_job_status_wrong_owner_returns_403_no_disclosure(aws):
    job_state.create_job("owned-job", "evaluate", created_by="real-owner")
    # Advance + complete so there *is* status/result to (not) leak.
    job_state.start_processing("owned-job")
    job_state.complete_job(
        "owned-job",
        experiment_id="exp-1",
        result_ref="results/owned-job.json",
        result_summary={"overallScore": "0.9", "passed": True},
    )

    resp = jobs_mod.get_job_status("owned-job", "intruder")
    assert _status(resp) == 403, resp
    payload = _body(resp)
    # Req 9.4: must not disclose status, result reference, or summary.
    for leaked in ("status", "resultRef", "resultSummary", "experimentId"):
        assert leaked not in payload, f"403 leaked {leaked}: {payload}"


def test_get_job_status_owner_can_read(aws):
    job_state.create_job("mine", "evaluate", created_by="me")
    resp = jobs_mod.get_job_status("mine", "me")
    assert "statusCode" not in resp, resp
    assert resp["status"] == job_state.PENDING


# =========================================================================== #
# Oversized inline payload -> 413 (Req 7.3)
# =========================================================================== #


def test_oversized_inline_payload_returns_413_with_guidance():
    # An inline body just over the 10MB API Gateway limit.
    oversized = "x" * (payloads_mod.API_GATEWAY_BODY_LIMIT_BYTES + 1)
    assert payloads_mod.is_inline_payload_oversized(oversized) is True

    resp = payloads_mod.ensure_inline_payload_within_limit(oversized)
    assert resp is not None
    assert _status(resp) == 413
    body = _body(resp)
    # Guidance must point at the presigned upload path (Req 7.3).
    assert body["uploadPath"] == "/payloads"
    assert "presigned" in body["error"].lower() or "/payloads" in body["error"]


def test_within_limit_inline_payload_passes():
    small = json.dumps({"trace": {"a": 1}})
    assert payloads_mod.is_inline_payload_oversized(small) is False
    assert payloads_mod.ensure_inline_payload_within_limit(small) is None


# =========================================================================== #
# Limitation rejections (Req 11.1, 11.2)
# =========================================================================== #

#: A representative curated catalog the validator checks against. Sourced via
#: the catalog seam in real deployments; injected here so tests need no Worker.
_CURATED = {
    "faithfulness": ["faithfulness", "answer_relevancy"],
    "builtin": ["exact_match", "rouge_l"],
}


def test_validation_rejects_custom_metric_direct():
    # Direct unit test of the gatekeeper (Req 11.1).
    with pytest.raises(validation_mod.ValidationRejection) as exc:
        validation_mod.validate_evaluation_request(
            {"metrics": ["my_custom_metric"]}, _CURATED
        )
    assert exc.value.status_code == 400
    assert exc.value.code == "CUSTOM_METRIC_NOT_SUPPORTED"


def test_validation_rejects_generic_json_schema_mapping_direct():
    # Direct unit test of the gatekeeper (Req 11.2).
    with pytest.raises(validation_mod.ValidationRejection) as exc:
        validation_mod.validate_evaluation_request(
            {"schema_mapping": {"input": "$.q"}}, _CURATED
        )
    assert exc.value.status_code == 400
    assert exc.value.code == "SCHEMA_MAPPING_NOT_SUPPORTED"


def test_validation_accepts_curated_metric_direct():
    # A curated metric passes cleanly (no exception).
    assert (
        validation_mod.validate_evaluation_request({"metrics": ["faithfulness"]}, _CURATED)
        is None
    )


def test_create_job_rejects_custom_metric_no_job_no_worker(aws, fake_lambda, monkeypatch):
    # End-to-end through the API Lambda: custom metric -> 400, no Job, no invoke.
    monkeypatch.setattr(jobs_mod, "_curated_metric_set", lambda: _CURATED)
    event = _proxy_event("POST", "/evaluate", body={"trace": {"a": 1}, "metrics": ["totally_custom"]})
    resp = jobs_mod.create_evaluate_job(event, "evaluate")

    assert _status(resp) == 400, resp  # Req 11.1
    assert _scan_count() == 0  # no Job created
    assert fake_lambda.invocations == []  # no Worker invoke


def test_create_job_rejects_schema_mapping_no_job_no_worker(aws, fake_lambda):
    # GenericJSON schema mapping -> 400, no Job, no invoke (Req 11.2).
    # No metrics requested, so the curated source is never consulted.
    event = _proxy_event(
        "POST", "/evaluate", body={"trace": {"a": 1}, "schema_mapping": {"input": "$.q"}}
    )
    resp = jobs_mod.create_evaluate_job(event, "evaluate")

    assert _status(resp) == 400, resp
    assert _scan_count() == 0
    assert fake_lambda.invocations == []


def test_create_job_rejects_generic_adapter_no_job_no_worker(aws, fake_lambda):
    # Selecting the GenericJSON adapter by name is also rejected (Req 11.2).
    event = _proxy_event("POST", "/evaluate", body={"trace": {"a": 1}, "adapter": "generic"})
    resp = jobs_mod.create_evaluate_job(event, "evaluate")

    assert _status(resp) == 400, resp
    assert _scan_count() == 0
    assert fake_lambda.invocations == []


def test_create_job_allows_curated_metric(aws, fake_lambda, monkeypatch):
    # A curated metric is accepted: Job created + Worker invoked.
    monkeypatch.setattr(jobs_mod, "_curated_metric_set", lambda: _CURATED)
    event = _proxy_event("POST", "/evaluate", body={"trace": {"a": 1}, "metrics": ["faithfulness"]})
    resp = jobs_mod.create_evaluate_job(event, "evaluate")

    assert "statusCode" not in resp, resp
    assert _scan_count() == 1
    assert len(fake_lambda.invocations) == 1


# =========================================================================== #
# The API Lambda never imports uaef (Req 6.2)
# =========================================================================== #


def test_api_lambda_modules_do_not_import_uaef_at_runtime(aws, fake_lambda, monkeypatch):
    """Exercising the full dispatch path must never pull ``uaef`` into sys.modules."""
    # Defensive: ensure a clean slate.
    sys.modules.pop("uaef", None)

    monkeypatch.setattr(jobs_mod, "_curated_metric_set", lambda: _CURATED)

    # Hit every dispatchable route + the validation gatekeeper.
    api_mod.handler(_proxy_event("GET", "/does-not-exist"))
    api_mod.handler(_proxy_event("DELETE", "/evaluate"))
    api_mod.handler(_proxy_event("POST", "/evaluate", sub=None, body={"trace": {}}))
    api_mod.handler(_proxy_event("POST", "/evaluate", body={"trace": {"a": 1}}))
    api_mod.handler(_proxy_event("POST", "/batch-evaluate", body={"traces": [{"a": 1}]}))
    job_state.create_job("j", "evaluate", created_by="user-123")
    api_mod.handler(_proxy_event("GET", "/jobs/{jobId}", path_params={"jobId": "j"}))
    jobs_mod.create_evaluate_job(
        _proxy_event("POST", "/evaluate", body={"trace": {}, "metrics": ["nope"]}), "evaluate"
    )
    validation_mod.validate_evaluation_request({"metrics": ["faithfulness"]}, _CURATED)
    payloads_mod.ensure_inline_payload_within_limit("small")

    assert "uaef" not in sys.modules, (
        "The API Lambda dispatch path imported `uaef`, violating Requirement 6.2."
    )


def test_api_lambda_source_has_no_uaef_import():
    """Static check: no API-Lambda module imports ``uaef`` at module level."""
    import_pattern = re.compile(r"^\s*(?:import\s+uaef|from\s+uaef\b)", re.MULTILINE)
    for module in (api_mod, jobs_mod, payloads_mod, validation_mod, job_state):
        source = open(module.__file__, "r", encoding="utf-8").read()
        assert not import_pattern.search(source), (
            f"{module.__name__} contains a top-level `uaef` import (Req 6.2)."
        )
