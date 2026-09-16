# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Property 3: Job lifecycle monotonicity.

**Validates: Requirements 5.1, 5.2, 5.6**

Generates random sequences of lifecycle operations against a fresh job (backed
by a moto-mocked DynamoDB table) and asserts that the stored status:

  * only ever advances forward along PENDING -> PROCESSING -> COMPLETED|FAILED,
  * never skips PROCESSING (a terminal state is only ever reached after the job
    was observed in PROCESSING), and
  * once terminal (COMPLETED/FAILED) is never overwritten.

Each generated example uses a unique ``jobId`` so examples do not interfere
with one another even though the moto table is shared across the test.
"""

from __future__ import annotations

import uuid

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import job_state
from job_state import (
    COMPLETED,
    FAILED,
    PENDING,
    PROCESSING,
    TERMINAL_STATUSES,
)

# Rank used to assert the status never moves backward. COMPLETED and FAILED are
# both terminal and share the top rank.
_RANK = {PENDING: 0, PROCESSING: 1, COMPLETED: 2, FAILED: 2}

# The set of lifecycle operations a worker (or a buggy/duplicate invocation)
# might attempt, in any order, any number of times.
_OPERATIONS = ("start", "complete", "fail")


def _apply(job_id: str, op: str) -> None:
    """Apply a single lifecycle operation via the public conditional helpers."""
    if op == "start":
        job_state.start_processing(job_id)
    elif op == "complete":
        job_state.complete_job(job_id, experiment_id="exp-1", result_ref="r/1")
    elif op == "fail":
        job_state.fail_job(job_id, code="ERR", message="boom")
    else:  # pragma: no cover - guarded by the strategy
        raise AssertionError(f"unknown op {op!r}")


def _stored_status(table, job_id: str) -> str:
    item = table.get_item(Key={"jobId": job_id})["Item"]
    return item["status"]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ops=st.lists(st.sampled_from(_OPERATIONS), min_size=0, max_size=12))
def test_job_lifecycle_monotonicity(jobs_table, ops):
    """Status advances forward only and terminal records are immutable."""
    job_id = f"job-{uuid.uuid4()}"
    job_state.create_job(job_id, operation="evaluate", created_by="user-1")

    prev_status = _stored_status(jobs_table, job_id)
    assert prev_status == PENDING

    seen_processing = prev_status == PROCESSING
    terminal_value: str | None = None

    for op in ops:
        _apply(job_id, op)
        current = _stored_status(jobs_table, job_id)

        # (5.1) Never moves backward in the lifecycle.
        assert _RANK[current] >= _RANK[prev_status], (
            f"status regressed {prev_status} -> {current} after op {op!r}"
        )

        if current == PROCESSING:
            seen_processing = True

        # (5.1/5.2) PROCESSING is never skipped: reaching a terminal state
        # implies the job passed through PROCESSING at some point.
        if current in TERMINAL_STATUSES:
            assert seen_processing, (
                f"reached terminal {current} without ever observing PROCESSING"
            )

        # (5.6) Terminal states are immutable once written.
        if terminal_value is not None:
            assert current == terminal_value, (
                f"terminal record overwritten {terminal_value} -> {current}"
            )
        elif current in TERMINAL_STATUSES:
            terminal_value = current

        prev_status = current


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    extra_ops=st.lists(st.sampled_from(_OPERATIONS), min_size=1, max_size=8),
    terminal_first=st.sampled_from(["complete", "fail"]),
)
def test_terminal_state_never_overwritten(jobs_table, extra_ops, terminal_first):
    """Once a job is terminal, no subsequent operation changes it."""
    job_id = f"job-{uuid.uuid4()}"
    job_state.create_job(job_id, operation="evaluate", created_by="user-1")

    # Drive the job to a terminal state deterministically.
    assert job_state.start_processing(job_id) is True
    _apply(job_id, terminal_first)
    locked = _stored_status(jobs_table, job_id)
    assert locked in TERMINAL_STATUSES

    # Any further operations must be rejected and leave the record unchanged.
    for op in extra_ops:
        _apply(job_id, op)
        assert _stored_status(jobs_table, job_id) == locked
