# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for job-state transition guards.

**Validates: Requirements 5.3, 5.5, 5.6**

Covers:
  * rejecting a non-PENDING pickup (5.3),
  * terminal immutability on retry / duplicate invocation (5.6), and
  * the 1024-character error-message bound (5.5).

Pure-logic helpers are exercised directly; the conditional-write behaviour is
exercised against a moto-mocked DynamoDB table.
"""

from __future__ import annotations

import uuid

import pytest

import job_state
from job_state import (
    COMPLETED,
    FAILED,
    MAX_ERROR_MESSAGE_LENGTH,
    PENDING,
    PROCESSING,
    InvalidTransitionError,
    bound_error_message,
    is_valid_transition,
    validate_transition,
)


# --------------------------------------------------------------------------- #
# Pure-logic guards (no I/O)
# --------------------------------------------------------------------------- #


def test_allowed_transitions_only():
    assert is_valid_transition(PENDING, PROCESSING)
    assert is_valid_transition(PROCESSING, COMPLETED)
    assert is_valid_transition(PROCESSING, FAILED)


@pytest.mark.parametrize(
    "frm,to",
    [
        (PENDING, COMPLETED),   # skips PROCESSING
        (PENDING, FAILED),      # skips PROCESSING
        (PROCESSING, PENDING),  # backward
        (COMPLETED, PROCESSING),  # out of terminal
        (FAILED, PROCESSING),     # out of terminal
        (COMPLETED, FAILED),      # terminal -> terminal
        (PENDING, PENDING),       # self-loop, not an advance
    ],
)
def test_illegal_transitions_rejected(frm, to):
    assert not is_valid_transition(frm, to)
    with pytest.raises(InvalidTransitionError):
        validate_transition(frm, to)


def test_validate_transition_rejects_unknown_status():
    with pytest.raises(InvalidTransitionError):
        validate_transition("BOGUS", PROCESSING)
    with pytest.raises(InvalidTransitionError):
        validate_transition(PENDING, "BOGUS")


def test_validate_transition_rejects_terminal_source():
    # Terminal states are immutable (5.6) even at the validation layer.
    with pytest.raises(InvalidTransitionError):
        validate_transition(COMPLETED, FAILED)
    with pytest.raises(InvalidTransitionError):
        validate_transition(FAILED, COMPLETED)


# --------------------------------------------------------------------------- #
# Error-message bound (Requirement 5.5)
# --------------------------------------------------------------------------- #


def test_bound_error_message_truncates_long_message():
    long_message = "x" * (MAX_ERROR_MESSAGE_LENGTH + 500)
    bounded = bound_error_message(long_message)
    assert len(bounded) == MAX_ERROR_MESSAGE_LENGTH


def test_bound_error_message_passes_short_message():
    assert bound_error_message("short") == "short"


def test_bound_error_message_handles_none():
    assert bound_error_message(None) == ""


def test_bound_error_message_exactly_at_limit():
    exact = "y" * MAX_ERROR_MESSAGE_LENGTH
    assert bound_error_message(exact) == exact
    assert len(bound_error_message(exact)) == MAX_ERROR_MESSAGE_LENGTH


# --------------------------------------------------------------------------- #
# Conditional-write guards against DynamoDB (moto)
# --------------------------------------------------------------------------- #


def _new_pending_job(table) -> str:
    job_id = f"job-{uuid.uuid4()}"
    job_state.create_job(job_id, operation="evaluate", created_by="user-1")
    return job_id


def test_start_processing_succeeds_from_pending(jobs_table):
    job_id = _new_pending_job(jobs_table)
    assert job_state.start_processing(job_id) is True
    assert jobs_table.get_item(Key={"jobId": job_id})["Item"]["status"] == PROCESSING


def test_reject_non_pending_pickup(jobs_table):
    """5.3: a worker invoked for a non-PENDING job leaves it unchanged."""
    job_id = _new_pending_job(jobs_table)
    assert job_state.start_processing(job_id) is True  # now PROCESSING

    # A duplicate/second pickup must be rejected (stored status != PENDING).
    assert job_state.start_processing(job_id) is False
    assert jobs_table.get_item(Key={"jobId": job_id})["Item"]["status"] == PROCESSING


def test_start_processing_rejected_when_missing(jobs_table):
    """A pickup for a non-existent job performs no write and returns False."""
    assert job_state.start_processing("does-not-exist") is False
    assert "Item" not in jobs_table.get_item(Key={"jobId": "does-not-exist"})


def test_complete_then_retry_is_immutable(jobs_table):
    """5.6: retry/duplicate invocation on a COMPLETED job is rejected."""
    job_id = _new_pending_job(jobs_table)
    job_state.start_processing(job_id)
    assert job_state.complete_job(job_id, experiment_id="exp-1", result_ref="r/1") is True

    item = jobs_table.get_item(Key={"jobId": job_id})["Item"]
    assert item["status"] == COMPLETED
    assert item["experimentId"] == "exp-1"

    # Duplicate complete + a fail attempt must both be rejected, unchanged.
    assert job_state.complete_job(job_id, experiment_id="exp-2") is False
    assert job_state.fail_job(job_id, code="ERR", message="late failure") is False

    after = jobs_table.get_item(Key={"jobId": job_id})["Item"]
    assert after["status"] == COMPLETED
    assert after["experimentId"] == "exp-1"  # not overwritten
    assert "error" not in after


def test_fail_then_retry_is_immutable(jobs_table):
    """5.6: a FAILED job is never overwritten by a later complete/fail."""
    job_id = _new_pending_job(jobs_table)
    job_state.start_processing(job_id)
    assert job_state.fail_job(job_id, code="ERR", message="boom") is True

    assert jobs_table.get_item(Key={"jobId": job_id})["Item"]["status"] == FAILED

    assert job_state.complete_job(job_id, experiment_id="exp-1") is False
    assert job_state.fail_job(job_id, code="ERR2", message="again") is False

    item = jobs_table.get_item(Key={"jobId": job_id})["Item"]
    assert item["status"] == FAILED
    assert item["error"]["code"] == "ERR"  # original error preserved


def test_complete_rejected_from_pending(jobs_table):
    """Cannot skip PROCESSING: completing a PENDING job is rejected."""
    job_id = _new_pending_job(jobs_table)
    assert job_state.complete_job(job_id, experiment_id="exp-1") is False
    assert jobs_table.get_item(Key={"jobId": job_id})["Item"]["status"] == PENDING


def test_fail_job_persists_bounded_error_message(jobs_table):
    """5.5: a long error message is truncated to 1024 chars when persisted."""
    job_id = _new_pending_job(jobs_table)
    job_state.start_processing(job_id)

    long_message = "z" * (MAX_ERROR_MESSAGE_LENGTH + 1000)
    assert job_state.fail_job(job_id, code="ERR", message=long_message) is True

    error = jobs_table.get_item(Key={"jobId": job_id})["Item"]["error"]
    assert error["code"] == "ERR"
    assert len(error["message"]) == MAX_ERROR_MESSAGE_LENGTH
