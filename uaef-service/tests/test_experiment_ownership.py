# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Unit tests for owner-scoped experiment access (security review H-02).

Prior to this fix, ``list_experiments``, ``compare_experiments``, and
``generate_report`` invoked the Worker with no caller identity or owner
filter at all — any authenticated user could read, compare, or generate a
report for any other user's experiment data. This covers the fix end to end:

  * Worker-layer actions (``handlers/worker.py``):
      - ``_get_experiment`` returns ``{"code": "FORBIDDEN"}`` (no experiment
        detail) when the experiment's ``created_by`` does not match the
        caller-supplied ``created_by``; succeeds when it matches; is
        permissive when no ``created_by`` is supplied at all (defensive only
        — the API Lambda always supplies it for an authenticated caller).
      - ``_compare_experiments_action`` rejects the **whole** request (not a
        partial result) if any named experiment isn't owned by the caller.
      - ``_generate_report_action`` likewise rejects on a mismatch.
      - ``_list_experiments`` scopes to ``created_by`` via the storage layer
        (see ``tests/test_storage_owner_scoping.py`` in the repo root for the
        storage-layer GSI behavior itself).

  * API-layer handlers (``handlers/jobs.py::get_experiment``,
    ``handlers/agents.py::list_experiments/compare_experiments/generate_report``):
      - Map a Worker ``FORBIDDEN`` response to a 403 with no experiment
        detail in the body — the same non-leaking pattern
        ``get_job_status`` uses for Jobs (Requirement 9.4's approach, applied
        here to experiments).
      - Fail closed (401, not "list everything") when the caller identity is
        missing.
      - Forward the caller's identity to the Worker as ``created_by``.

The Worker Lambda actions are exercised directly against a mocked
``uaef.storage.dynamodb_s3.get_storage()`` (no real AWS calls, no real
``uaef`` persistence layer). The API-layer handlers are exercised against a
stubbed boto3 Lambda client (mirroring ``test_api_lambda.py``'s pattern).
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("PAYLOAD_BUCKET_NAME", "test-payload-bucket")

from handlers import agents as agents_mod  # noqa: E402
from handlers import jobs as jobs_mod  # noqa: E402
from handlers import worker as worker_mod  # noqa: E402

# Import the submodule explicitly (rather than referencing it via a dotted
# string in monkeypatch.setattr) so this doesn't depend on `uaef.storage`
# already being attached as an attribute of the top-level `uaef` package —
# which it may not be if another test module (e.g. test_api_lambda.py, which
# asserts `uaef` is never imported by the API Lambda) has popped `uaef` from
# ``sys.modules`` and it gets freshly re-imported bare.
import uaef.storage.dynamodb_s3 as dynamodb_s3_mod  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared fakes
# --------------------------------------------------------------------------- #


def _fake_store(experiments_by_id):
    """A fake uaef.storage get_storage() returning canned experiment records."""
    store = MagicMock()
    store.get_experiment.side_effect = lambda eid: experiments_by_id.get(eid)
    store.get_full_results.return_value = {"evaluations": []}
    return store


class FakeLambdaClient:
    """Stub boto3 Lambda client for RequestResponse invokes (API-layer tests)."""

    def __init__(self, payload, function_error=None):
        self.invocations = []
        self._payload = payload
        self._function_error = function_error

    def invoke(self, **kwargs):
        self.invocations.append(kwargs)
        resp = {"Payload": MagicMock()}
        resp["Payload"].read.return_value = json.dumps(self._payload).encode("utf-8")
        if self._function_error is not None:
            resp["FunctionError"] = self._function_error
        return resp


# =========================================================================== #
# Worker layer — handlers/worker.py
# =========================================================================== #


def test_worker_get_experiment_forbidden_on_owner_mismatch(monkeypatch):
    experiments = {
        "exp-1": {"experiment_id": "exp-1", "created_by": "alice-sub", "average_scores": {}},
    }
    monkeypatch.setattr(dynamodb_s3_mod, "get_storage", lambda: _fake_store(experiments))

    result = worker_mod._get_experiment({"experiment_id": "exp-1", "created_by": "mallory-sub"})
    assert result == {"code": "FORBIDDEN"}


def test_worker_get_experiment_succeeds_for_actual_owner(monkeypatch):
    experiments = {
        "exp-1": {"experiment_id": "exp-1", "created_by": "alice-sub", "average_scores": {}},
    }
    monkeypatch.setattr(dynamodb_s3_mod, "get_storage", lambda: _fake_store(experiments))

    result = worker_mod._get_experiment({"experiment_id": "exp-1", "created_by": "alice-sub"})
    assert result.get("code") != "FORBIDDEN"
    assert result["experiment_id"] == "exp-1"


def test_worker_get_experiment_permissive_when_created_by_absent(monkeypatch):
    """Defensive-only: if the Worker event carries no created_by at all, don't reject."""
    experiments = {
        "exp-1": {"experiment_id": "exp-1", "created_by": "alice-sub", "average_scores": {}},
    }
    monkeypatch.setattr(dynamodb_s3_mod, "get_storage", lambda: _fake_store(experiments))

    result = worker_mod._get_experiment({"experiment_id": "exp-1"})
    assert result.get("code") != "FORBIDDEN"


def test_worker_get_experiment_missing_is_not_found_not_forbidden(monkeypatch):
    monkeypatch.setattr(dynamodb_s3_mod, "get_storage", lambda: _fake_store({}))

    result = worker_mod._get_experiment({"experiment_id": "exp-missing", "created_by": "alice-sub"})
    assert result.get("code") != "FORBIDDEN"
    assert "error" in result


def test_worker_compare_experiments_rejects_whole_request_on_any_mismatch(monkeypatch):
    experiments = {
        "exp-1": {
            "experiment_id": "exp-1", "created_by": "alice-sub", "average_scores": {"m": 0.5},
            "experiment_name": "one", "overall_average_score": 0.5, "metadata": {},
        },
        "exp-2": {
            "experiment_id": "exp-2", "created_by": "bob-sub", "average_scores": {"m": 0.7},
            "experiment_name": "two", "overall_average_score": 0.7, "metadata": {},
        },
    }
    monkeypatch.setattr(dynamodb_s3_mod, "get_storage", lambda: _fake_store(experiments))

    result = worker_mod._compare_experiments_action(
        {"experiment_ids": ["exp-1", "exp-2"], "created_by": "alice-sub"}
    )
    assert result == {"code": "FORBIDDEN"}


def test_worker_compare_experiments_succeeds_when_all_owned_by_caller(monkeypatch):
    experiments = {
        "exp-1": {
            "experiment_id": "exp-1", "created_by": "alice-sub", "average_scores": {"m": 0.5},
            "experiment_name": "one", "overall_average_score": 0.5, "metadata": {},
        },
        "exp-3": {
            "experiment_id": "exp-3", "created_by": "alice-sub", "average_scores": {"m": 0.9},
            "experiment_name": "three", "overall_average_score": 0.9, "metadata": {},
        },
    }
    monkeypatch.setattr(dynamodb_s3_mod, "get_storage", lambda: _fake_store(experiments))

    result = worker_mod._compare_experiments_action(
        {"experiment_ids": ["exp-1", "exp-3"], "created_by": "alice-sub"}
    )
    assert result.get("code") != "FORBIDDEN"
    assert "table" in result


def test_worker_generate_report_forbidden_on_owner_mismatch(monkeypatch):
    experiments = {
        "exp-1": {"experiment_id": "exp-1", "created_by": "alice-sub", "experiment_name": "one"},
    }
    monkeypatch.setattr(dynamodb_s3_mod, "get_storage", lambda: _fake_store(experiments))

    result = worker_mod._generate_report_action(
        {"experiment_id": "exp-1", "created_by": "mallory-sub"}
    )
    assert result == {"code": "FORBIDDEN"}


def test_worker_list_experiments_forwards_created_by_to_storage(monkeypatch):
    store = MagicMock()
    store.list_experiments.return_value = []
    monkeypatch.setattr(dynamodb_s3_mod, "get_storage", lambda: store)

    worker_mod._list_experiments(created_by="alice-sub")
    store.list_experiments.assert_called_once_with(created_by="alice-sub")


# =========================================================================== #
# API layer — handlers/jobs.py::get_experiment
# =========================================================================== #


def test_api_get_experiment_403_on_worker_forbidden(monkeypatch):
    client = FakeLambdaClient({"code": "FORBIDDEN"})
    monkeypatch.setattr(jobs_mod, "_get_lambda_client", lambda: client)
    monkeypatch.setattr(jobs_mod, "_worker_function_name", lambda: "test-worker-fn")

    result = jobs_mod.get_experiment("exp-1", "mallory-sub")
    assert result["statusCode"] == 403
    body = json.loads(result["body"])
    assert "experiment_id" not in body
    assert "average_scores" not in body


def test_api_get_experiment_forwards_caller_sub_as_created_by(monkeypatch):
    client = FakeLambdaClient({"experiment_id": "exp-1"})
    monkeypatch.setattr(jobs_mod, "_get_lambda_client", lambda: client)
    monkeypatch.setattr(jobs_mod, "_worker_function_name", lambda: "test-worker-fn")

    jobs_mod.get_experiment("exp-1", "alice-sub")
    sent = json.loads(client.invocations[0]["Payload"].decode("utf-8"))
    assert sent["created_by"] == "alice-sub"
    assert sent["experiment_id"] == "exp-1"


def test_api_get_experiment_401_without_caller_sub(monkeypatch):
    monkeypatch.setattr(jobs_mod, "_worker_function_name", lambda: "test-worker-fn")
    result = jobs_mod.get_experiment("exp-1", None)
    assert result["statusCode"] == 401


# =========================================================================== #
# API layer — handlers/agents.py::list_experiments
# =========================================================================== #


def test_api_list_experiments_fails_closed_without_caller_sub(monkeypatch):
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "test-worker-fn")
    result = agents_mod.list_experiments({}, None)
    assert result["experiments"] == []
    assert "error" in result


def test_api_list_experiments_forwards_created_by(monkeypatch):
    client = FakeLambdaClient({"experiments": [{"experiment_id": "exp-1"}]})
    monkeypatch.setattr(agents_mod, "_get_lambda", lambda: client)
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "test-worker-fn")

    result = agents_mod.list_experiments({}, "alice-sub")
    assert result == {"experiments": [{"experiment_id": "exp-1"}]}
    sent = json.loads(client.invocations[0]["Payload"].decode("utf-8"))
    assert sent == {"action": "list_experiments", "created_by": "alice-sub"}


# =========================================================================== #
# API layer — handlers/agents.py::compare_experiments / generate_report
# =========================================================================== #


def _proxy_event(body):
    return {"body": json.dumps(body)}


def test_api_compare_experiments_403_on_worker_forbidden(monkeypatch):
    client = FakeLambdaClient({"code": "FORBIDDEN"})
    monkeypatch.setattr(agents_mod, "_get_lambda", lambda: client)
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "test-worker-fn")

    result = agents_mod.compare_experiments(
        _proxy_event({"experiment_ids": ["exp-1", "exp-2"]}), "mallory-sub"
    )
    assert result["statusCode"] == 403


def test_api_compare_experiments_401_without_caller_sub(monkeypatch):
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "test-worker-fn")
    result = agents_mod.compare_experiments(
        _proxy_event({"experiment_ids": ["exp-1", "exp-2"]}), None
    )
    assert result["statusCode"] == 401


def test_api_generate_report_403_on_worker_forbidden(monkeypatch):
    client = FakeLambdaClient({"code": "FORBIDDEN"})
    monkeypatch.setattr(agents_mod, "_get_lambda", lambda: client)
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "test-worker-fn")

    result = agents_mod.generate_report(_proxy_event({"experiment_id": "exp-1"}), "mallory-sub")
    assert result["statusCode"] == 403


def test_api_generate_report_401_without_caller_sub(monkeypatch):
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "test-worker-fn")
    result = agents_mod.generate_report(_proxy_event({"experiment_id": "exp-1"}), None)
    assert result["statusCode"] == 401
