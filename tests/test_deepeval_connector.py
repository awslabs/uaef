# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeepEval connector with mocked DeepEval library."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from uaef.integrations.deepeval_connector import DeepEvalConnector
from uaef.models.agent_trace import AgentTrace, Message, ToolCall
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.ground_truth import GroundTruth

NOW = datetime.now(timezone.utc)


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content, timestamp=NOW)


def _tc(name: str, arguments: dict = None) -> ToolCall:
    return ToolCall(name=name, arguments=arguments or {}, timestamp=NOW)


def _make_eval_input(trace, ground_truth=None, context=None):
    return EvaluationInput(trace=trace, ground_truth=ground_truth, context=context or [])


def _mock_metric(score=0.85, reason="test reason"):
    metric = MagicMock()
    metric.score = score
    metric.reason = reason
    return metric


# --- Fixtures ---

@pytest.fixture
def connector():
    c = DeepEvalConnector()
    c._deepeval_available = True
    return c


@pytest.fixture
def basic_trace():
    return AgentTrace(
        messages=[_msg("user", "What is Python?"), _msg("assistant", "Python is a programming language.")],
    )


@pytest.fixture
def trace_with_tools():
    return AgentTrace(
        messages=[_msg("user", "Search for flights"), _msg("assistant", "Found flights.")],
        tool_calls=[_tc("search_flights", {"origin": "SEA"})],
    )


@pytest.fixture
def ground_truth_with_output():
    return GroundTruth(expected_output="Python is a high-level programming language.")


@pytest.fixture
def ground_truth_with_tools():
    return GroundTruth(
        expected_tool_calls=[_tc("search_flights", {"origin": "SEA"})],
    )


@pytest.fixture
def context_docs():
    return [
        "Python is a high-level, interpreted programming language.",
        "Python supports multiple programming paradigms.",
    ]


def _simple_trace(q="q", a="a"):
    return AgentTrace(messages=[_msg("user", q), _msg("assistant", a)])


# --- Tests: transform_to_deepeval_format ---

class TestTransformToDeepEvalFormat:
    def test_basic_transform(self, connector, basic_trace):
        result = connector.transform_to_deepeval_format(_make_eval_input(basic_trace))
        assert result["input"] == "What is Python?"
        assert result["actual_output"] == "Python is a programming language."

    def test_with_ground_truth(self, connector, basic_trace, ground_truth_with_output):
        result = connector.transform_to_deepeval_format(
            _make_eval_input(basic_trace, ground_truth=ground_truth_with_output)
        )
        assert result["expected_output"] == "Python is a high-level programming language."

    def test_with_context(self, connector, basic_trace, context_docs):
        result = connector.transform_to_deepeval_format(
            _make_eval_input(basic_trace, context=context_docs)
        )
        assert len(result["retrieval_context"]) == 2

    def test_with_tools(self, connector, trace_with_tools, ground_truth_with_tools):
        result = connector.transform_to_deepeval_format(
            _make_eval_input(trace_with_tools, ground_truth=ground_truth_with_tools)
        )
        assert len(result["tools_called"]) == 1
        assert len(result["expected_tools"]) == 1

    def test_missing_user_message(self, connector):
        trace = AgentTrace(messages=[_msg("assistant", "hi")])
        result = connector.transform_to_deepeval_format(_make_eval_input(trace))
        # Falls back to first message as question
        assert result["input"] == "hi"

    def test_missing_assistant_message(self, connector):
        trace = AgentTrace(messages=[_msg("user", "hi")])
        with pytest.raises(ValueError, match="No assistant response"):
            connector.transform_to_deepeval_format(_make_eval_input(trace))


# --- Tests: list_available_metrics ---

class TestListAvailableMetrics:
    def test_lists_all_metrics(self, connector):
        metrics = connector.list_available_metrics()
        expected = [
            "contextual_precision", "contextual_recall", "contextual_relevancy",
            "hallucination", "faithfulness", "answer_relevancy",
            "tool_correctness", "geval",
        ]
        assert metrics == expected


# --- Tests: individual metrics (mocked) ---

class TestContextualPrecision:
    def test_success(self, connector):
        mock_metric = _mock_metric(0.9, "precise context")
        mock_test_case_cls = MagicMock()
        mock_metrics_mod = MagicMock()
        mock_metrics_mod.ContextualPrecisionMetric.return_value = mock_metric
        mock_test_case_mod = MagicMock()
        mock_test_case_mod.LLMTestCase = mock_test_case_cls

        with patch.dict("sys.modules", {
            "deepeval": MagicMock(),
            "deepeval.metrics": mock_metrics_mod,
            "deepeval.test_case": mock_test_case_mod,
        }):
            ev = _make_eval_input(_simple_trace(), ground_truth=GroundTruth(expected_output="e"), context=["ctx"])
            score = connector.calculate_contextual_precision(ev)
            assert score.metric_name == "deepeval_contextual_precision"
            assert score.score == 0.9
            assert score.metadata["library"] == "deepeval"

    def test_missing_ground_truth_raises(self, connector, basic_trace):
        with pytest.raises((ValueError, Exception)):
            connector.calculate_contextual_precision(_make_eval_input(basic_trace, context=["ctx"]))


class TestContextualRecall:
    def test_missing_ground_truth_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Ground truth"):
            connector.calculate_contextual_recall(_make_eval_input(basic_trace, context=["ctx"]))


class TestContextualRelevancy:
    def test_success(self, connector):
        mock_metric = _mock_metric(0.88, "relevant context")
        mock_metrics_mod = MagicMock()
        mock_metrics_mod.ContextualRelevancyMetric.return_value = mock_metric
        mock_test_case_mod = MagicMock()

        with patch.dict("sys.modules", {
            "deepeval": MagicMock(),
            "deepeval.metrics": mock_metrics_mod,
            "deepeval.test_case": mock_test_case_mod,
        }):
            ev = _make_eval_input(_simple_trace(), context=["ctx"])
            score = connector.calculate_contextual_relevancy(ev)
            assert score.metric_name == "deepeval_contextual_relevancy"
            assert score.score == 0.88
            assert score.metadata["metric"] == "contextual_relevancy"

    def test_missing_context_returns_none(self, connector, basic_trace):
        score = connector.calculate_contextual_relevancy(_make_eval_input(basic_trace))
        assert score.score is None
        assert "not_applicable" in score.metadata.get("warning", "")


class TestFaithfulness:
    def test_success(self, connector):
        mock_metric = _mock_metric(0.95, "faithful")
        mock_metrics_mod = MagicMock()
        mock_metrics_mod.FaithfulnessMetric.return_value = mock_metric
        mock_test_case_mod = MagicMock()

        with patch.dict("sys.modules", {
            "deepeval": MagicMock(),
            "deepeval.metrics": mock_metrics_mod,
            "deepeval.test_case": mock_test_case_mod,
        }):
            ev = _make_eval_input(_simple_trace(), context=["ctx"])
            score = connector.calculate_faithfulness(ev)
            assert score.metric_name == "deepeval_faithfulness"
            assert score.score == 0.95
            assert score.metadata["metric"] == "faithfulness"

    def test_missing_context_returns_none(self, connector, basic_trace):
        score = connector.calculate_faithfulness(_make_eval_input(basic_trace))
        assert score.score is None
        assert "not_applicable" in score.metadata.get("warning", "")


class TestAnswerRelevancy:
    def test_success(self, connector):
        mock_metric = _mock_metric(0.92, "relevant answer")
        mock_metrics_mod = MagicMock()
        mock_metrics_mod.AnswerRelevancyMetric.return_value = mock_metric
        mock_test_case_mod = MagicMock()

        with patch.dict("sys.modules", {
            "deepeval": MagicMock(),
            "deepeval.metrics": mock_metrics_mod,
            "deepeval.test_case": mock_test_case_mod,
        }):
            ev = _make_eval_input(_simple_trace())
            score = connector.calculate_answer_relevancy(ev)
            assert score.metric_name == "deepeval_answer_relevancy"
            assert score.score == 0.92
            assert score.metadata["metric"] == "answer_relevancy"


class TestHallucination:
    def test_missing_context_returns_none(self, connector, basic_trace):
        score = connector.calculate_hallucination(_make_eval_input(basic_trace))
        assert score.score is None
        assert "not_applicable" in score.metadata.get("warning", "")


class TestToolCorrectness:
    def test_missing_ground_truth_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Ground truth"):
            connector.calculate_tool_correctness(_make_eval_input(basic_trace))


# --- Tests: calculate_metrics dispatcher ---

class TestCalculateMetrics:
    def test_unknown_metric_skipped(self, connector, basic_trace):
        scores = connector.calculate_metrics(_make_eval_input(basic_trace), metric_names=["nonexistent"])
        assert scores == []

    def test_geval_without_criteria_skipped(self, connector, basic_trace):
        scores = connector.calculate_metrics(_make_eval_input(basic_trace), metric_names=["geval"])
        assert scores == []

    def test_failing_metric_continues(self, connector, basic_trace):
        """Metrics without context should return score=None, not crash."""
        scores = connector.calculate_metrics(
            _make_eval_input(basic_trace),
            metric_names=["contextual_precision", "hallucination", "faithfulness"],
        )
        # All return score=None (no context provided) but don't crash
        assert len(scores) == 3
        assert all(s.score is None for s in scores)


# --- Tests: unavailable library ---

class TestUnavailableLibrary:
    def test_is_available_false(self):
        c = DeepEvalConnector()
        c._deepeval_available = False
        assert c.is_available() is False

    def test_calculate_metrics_raises_when_unavailable(self):
        c = DeepEvalConnector()
        c._deepeval_available = False
        # Requirement 3.5: requesting an integration metric without the
        # 'integrations' extra raises an ImportError naming the extra and the
        # exact pip install command, and returns no metric result.
        with pytest.raises(ImportError, match=r"integrations.*pip install 'uaef\[integrations\]'"):
            c.calculate_metrics(_make_eval_input(_simple_trace()), metric_names=["hallucination"])

    def test_transform_raises_when_unavailable(self):
        c = DeepEvalConnector()
        c._deepeval_available = False
        with pytest.raises(ImportError, match=r"integrations.*pip install 'uaef\[integrations\]'"):
            c.transform_to_deepeval_format(_make_eval_input(_simple_trace()))
