# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the UAEFClient poll loop (Task 5.5).

Validates Requirements:
  10.4 - submit, poll until a terminal state is reached, then return the result.
  10.5 - while non-terminal, poll the status endpoint at a fixed 2-second interval.
  10.6 - if no terminal state within 15 minutes, stop polling and raise a timeout error.
  10.8 - if the job reaches FAILED, raise an error carrying the job's error
         code and message and return no result.

Test isolation approach
-----------------------
The full ``uaef`` package cannot be imported here (heavy optional deps), so the
client module is loaded directly from ``src/uaef/client/__init__.py`` via
``importlib`` WITHOUT importing the parent ``uaef`` package. HTTP is mocked by
injecting a fake ``requests`` module (the module's ``_require_requests`` is
monkeypatched to return it), and timing is made deterministic by monkeypatching
the client module's ``time.sleep`` and ``time.monotonic`` so the 2s interval and
900s timeout are simulated without real waiting.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CLIENT_SRC = _REPO_ROOT / "src" / "uaef" / "client" / "__init__.py"


def _load_client_module():
    spec = importlib.util.spec_from_file_location(
        "uaef_client_standalone_poll", _CLIENT_SRC
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client_mod = _load_client_module()
UAEFClient = client_mod.UAEFClient
JobFailedError = client_mod.JobFailedError


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeRequests:
    """Minimal stand-in for the ``requests`` module.

    GET to /jobs/{id} returns the next queued status record. POST/PUT are
    recorded but unused by the poll-loop tests.
    """

    def __init__(self, status_sequence):
        self._status_sequence = list(status_sequence)
        self.get_calls = []

    def get(self, url, headers=None, timeout=None):
        self.get_calls.append(url)
        # Serve queued records; repeat the last one if polled beyond the queue.
        if len(self._status_sequence) > 1:
            record = self._status_sequence.pop(0)
        else:
            record = self._status_sequence[0]
        return _FakeResponse(record)

    def post(self, url, json=None, headers=None, timeout=None):  # pragma: no cover - unused
        return _FakeResponse({"jobId": "job-xyz"})

    def put(self, url, data=None, headers=None, timeout=None):  # pragma: no cover - unused
        return _FakeResponse({})


class _FakeClock:
    """Deterministic monotonic clock advanced only by ``sleep`` calls."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _build_client(monkeypatch, fake_requests):
    monkeypatch.setattr(client_mod, "_require_requests", lambda: fake_requests)
    return UAEFClient(endpoint="https://api.example.test", token="jwt-token")


def _install_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(client_mod.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(client_mod.time, "sleep", clock.sleep)
    return clock


# ==========================================================================
# 10.4 / 10.5: poll until terminal, fixed 2s interval
# ==========================================================================
def test_poll_returns_completed_record_after_polling(monkeypatch):
    """10.4: poll until COMPLETED and return the terminal status record."""
    fake_requests = _FakeRequests(
        [
            {"status": "PENDING"},
            {"status": "PROCESSING"},
            {"status": "COMPLETED", "resultSummary": {"overall_score": 0.9}},
        ]
    )
    _install_clock(monkeypatch)
    client = _build_client(monkeypatch, fake_requests)

    record = client._poll("job-1")

    assert record["status"] == "COMPLETED"
    assert record["resultSummary"] == {"overall_score": 0.9}
    # Polled three times (PENDING, PROCESSING, COMPLETED).
    assert len(fake_requests.get_calls) == 3
    assert fake_requests.get_calls[0].endswith("/jobs/job-1")


def test_poll_uses_fixed_2_second_interval(monkeypatch):
    """10.5: every wait between non-terminal polls is exactly 2 seconds."""
    fake_requests = _FakeRequests(
        [
            {"status": "PENDING"},
            {"status": "PROCESSING"},
            {"status": "PROCESSING"},
            {"status": "COMPLETED"},
        ]
    )
    clock = _install_clock(monkeypatch)
    client = _build_client(monkeypatch, fake_requests)
    assert client.poll_interval == 2.0  # default mandated by 10.5

    client._poll("job-2")

    # Three non-terminal polls -> three 2s sleeps before the terminal poll.
    assert clock.sleeps == [2.0, 2.0, 2.0]


# ==========================================================================
# 10.6: 15-minute (900s) timeout raises TimeoutError
# ==========================================================================
def test_poll_raises_timeout_after_900_seconds(monkeypatch):
    """10.6: a job that never finishes raises TimeoutError at the 900s bound."""
    # Always non-terminal -> the loop must give up on the timeout.
    fake_requests = _FakeRequests([{"status": "PROCESSING"}])
    clock = _install_clock(monkeypatch)
    client = _build_client(monkeypatch, fake_requests)
    assert client.timeout == 900.0  # default mandated by 10.6

    with pytest.raises(TimeoutError) as excinfo:
        client._poll("job-timeout")

    assert "job-timeout" in str(excinfo.value)
    # The deadline is monotonic-start + 900s; simulated time must have reached it.
    assert clock.now >= 900.0
    # Each sleep was the fixed 2s interval and never overshot the deadline grossly.
    assert all(s == 2.0 for s in clock.sleeps)


# ==========================================================================
# 10.8: FAILED raises JobFailedError carrying error code + message
# ==========================================================================
def test_poll_raises_job_failed_error_with_code_and_message(monkeypatch):
    """10.8: FAILED status raises an error carrying the job's code + message."""
    fake_requests = _FakeRequests(
        [
            {"status": "PROCESSING"},
            {
                "status": "FAILED",
                "error": {"code": "EVALUATION_ERROR", "message": "metric blew up"},
            },
        ]
    )
    _install_clock(monkeypatch)
    client = _build_client(monkeypatch, fake_requests)

    with pytest.raises(JobFailedError) as excinfo:
        client._poll("job-failed")

    err = excinfo.value
    assert err.code == "EVALUATION_ERROR"
    assert err.message == "metric blew up"
    assert err.job_id == "job-failed"
    # The message surfaces both the code and message for the caller.
    assert "EVALUATION_ERROR" in str(err)
    assert "metric blew up" in str(err)


def test_poll_failed_without_error_block_still_raises(monkeypatch):
    """10.8: FAILED with no error block still raises (code/message default)."""
    fake_requests = _FakeRequests([{"status": "FAILED"}])
    _install_clock(monkeypatch)
    client = _build_client(monkeypatch, fake_requests)

    with pytest.raises(JobFailedError) as excinfo:
        client._poll("job-bare-fail")

    assert excinfo.value.code is None
    assert excinfo.value.message is None
    assert excinfo.value.job_id == "job-bare-fail"


def test_failed_job_returns_no_result(monkeypatch):
    """10.8: a FAILED job surfaces via evaluate() as an error, not a result."""
    fake_requests = _FakeRequests(
        [{"status": "FAILED", "error": {"code": "BAD", "message": "nope"}}]
    )
    _install_clock(monkeypatch)
    client = _build_client(monkeypatch, fake_requests)
    # Avoid real submit; only the poll path is under test here.
    client._submit = lambda path, body: "job-e2e"

    with pytest.raises(JobFailedError):
        client.evaluate(trace={"x": 1})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
