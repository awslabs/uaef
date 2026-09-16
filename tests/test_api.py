# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the high-level UAEF API (evaluate, batch_evaluate)."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from uaef.api import evaluate, batch_evaluate
from uaef.models import AgentTrace, Message, GroundTruth, ToolCall
from uaef.models.message import MessageRole


def _make_trace(content="The capital of France is Paris."):
    return AgentTrace(
        trace_id=uuid4(),
        messages=[
            Message(role=MessageRole.USER, content="What is the capital of France?", timestamp=datetime.now(timezone.utc)),
            Message(role=MessageRole.ASSISTANT, content=content, timestamp=datetime.now(timezone.utc)),
        ],
        tool_calls=[],
        session_id="test-session",
    )


def _make_gt():
    return GroundTruth(expected_output="Paris", expected_tool_calls=[])


class TestEvaluate:
    def test_basic_evaluation_returns_result(self):
        result = evaluate(trace=_make_trace(), ground_truth=_make_gt())
        assert result is not None
        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.dimension_results is not None

    def test_evaluation_without_ground_truth(self):
        result = evaluate(trace=_make_trace())
        assert result is not None
        assert 0.0 <= result.overall_score <= 1.0

    def test_evaluation_with_metrics_list(self):
        result = evaluate(
            trace=_make_trace(),
            ground_truth=_make_gt(),
            metrics=["accuracy", "answer_relevance"],
        )
        assert result is not None
        assert result.overall_score >= 0.0

    def test_evaluation_rejects_list_input(self):
        with pytest.raises(ValueError, match="does not accept a list"):
            evaluate(trace=[_make_trace()])

    def test_evaluation_with_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate(trace=_make_trace(), metrics=["nonexistent_metric_xyz"])


class TestBatchEvaluate:
    def test_batch_returns_correct_count(self):
        traces = [_make_trace() for _ in range(3)]
        gts = [_make_gt() for _ in range(3)]
        results = batch_evaluate(traces=traces, ground_truths=gts)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.overall_score <= 1.0

    def test_batch_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            batch_evaluate(traces=[_make_trace()], ground_truths=[_make_gt(), _make_gt()])

    def test_batch_without_ground_truths(self):
        traces = [_make_trace() for _ in range(2)]
        results = batch_evaluate(traces=traces)
        assert len(results) == 2
