# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for UAEF data models."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from uaef.models import AgentTrace, GroundTruth, Message, ToolCall
from uaef.models.message import MessageRole
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.dimension_result import DimensionResult
from uaef.models.metric_score import MetricScore


class TestAgentTrace:
    def test_create_minimal(self):
        trace = AgentTrace(trace_id=uuid4(), messages=[], tool_calls=[])
        assert trace.trace_id is not None
        assert trace.messages == []

    def test_create_with_messages(self):
        trace = AgentTrace(
            trace_id=uuid4(),
            messages=[
                Message(role=MessageRole.USER, content="hello", timestamp=datetime.now(timezone.utc)),
                Message(role=MessageRole.ASSISTANT, content="hi", timestamp=datetime.now(timezone.utc)),
            ],
            tool_calls=[],
        )
        assert len(trace.messages) == 2

    def test_latency_must_be_non_negative(self):
        with pytest.raises(Exception):
            AgentTrace(trace_id=uuid4(), messages=[], tool_calls=[], latency=-1.0)

    def test_latency_zero_is_valid(self):
        trace = AgentTrace(trace_id=uuid4(), messages=[], tool_calls=[], latency=0.0)
        assert trace.latency == 0.0


class TestMessage:
    def test_user_message(self):
        msg = Message(role=MessageRole.USER, content="test", timestamp=datetime.now(timezone.utc))
        assert msg.role == MessageRole.USER
        assert msg.content == "test"

    def test_assistant_message(self):
        msg = Message(role=MessageRole.ASSISTANT, content="response", timestamp=datetime.now(timezone.utc))
        assert msg.role == MessageRole.ASSISTANT


class TestToolCall:
    def test_create(self):
        tc = ToolCall(name="search", arguments={"q": "test"}, timestamp=datetime.now(timezone.utc))
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}


class TestGroundTruth:
    def test_create_minimal(self):
        gt = GroundTruth(expected_output="answer", expected_tool_calls=[])
        assert gt.expected_output == "answer"

    def test_create_with_tool_calls(self):
        gt = GroundTruth(
            expected_output="result",
            expected_tool_calls=[
                ToolCall(name="search", arguments={"q": "test"}, timestamp=datetime.now(timezone.utc))
            ],
        )
        assert len(gt.expected_tool_calls) == 1


class TestMetricScore:
    def test_create(self):
        ms = MetricScore(metric_name="accuracy", score=0.95)
        assert ms.metric_name == "accuracy"
        assert ms.score == 0.95

    def test_score_none_is_valid(self):
        ms = MetricScore(metric_name="test", score=None)
        assert ms.score is None

    def test_score_out_of_range_raises(self):
        with pytest.raises(Exception):
            MetricScore(metric_name="test", score=1.5)

    def test_score_negative_raises(self):
        with pytest.raises(Exception):
            MetricScore(metric_name="test", score=-0.1)


class TestDimensionResult:
    def test_create(self):
        dr = DimensionResult(
            dimension_name="tool_calling",
            metric_scores=[MetricScore(metric_name="accuracy", score=0.9)],
            aggregate_score=0.9,
            weight=0.5,
        )
        assert dr.dimension_name == "tool_calling"
        assert dr.aggregate_score == 0.9

    def test_empty_metric_scores_raises(self):
        with pytest.raises(Exception):
            DimensionResult(
                dimension_name="test",
                metric_scores=[],
                aggregate_score=0.5,
                weight=0.5,
            )


class TestEvaluationResult:
    def test_create(self):
        result = EvaluationResult(
            trace_id=uuid4(),
            dimension_results=[
                DimensionResult(
                    dimension_name="quality",
                    metric_scores=[MetricScore(metric_name="acc", score=0.8)],
                    aggregate_score=0.8,
                    weight=1.0,
                )
            ],
            overall_score=0.8,
            passed=True,
        )
        assert result.overall_score == 0.8
        assert result.passed is True

    def test_score_out_of_range_raises(self):
        with pytest.raises(Exception):
            EvaluationResult(
                trace_id=uuid4(),
                dimension_results=[
                    DimensionResult(
                        dimension_name="q",
                        metric_scores=[MetricScore(metric_name="a", score=0.5)],
                        aggregate_score=0.5,
                        weight=1.0,
                    )
                ],
                overall_score=1.5,
                passed=True,
            )
