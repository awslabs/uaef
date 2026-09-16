# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RAGAS connector with mocked RAGAS library."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from uaef.integrations.ragas_connector import RAGASConnector
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


def _simple_trace(q="q", a="a"):
    return AgentTrace(messages=[_msg("user", q), _msg("assistant", a)])


# --- Fixtures ---

@pytest.fixture
def connector():
    c = RAGASConnector()
    c._ragas_available = True
    return c


@pytest.fixture
def basic_trace():
    return AgentTrace(
        messages=[_msg("user", "What is Python?"), _msg("assistant", "Python is a programming language.")],
    )


@pytest.fixture
def context_docs():
    return ["Python is a high-level programming language.", "Python supports multiple paradigms."]


@pytest.fixture
def ground_truth_with_output():
    return GroundTruth(expected_output="Python is a high-level programming language.")


@pytest.fixture
def ground_truth_with_tools():
    return GroundTruth(expected_tool_calls=[_tc("search_flights", {"origin": "SEA"})])


# --- Tests: transform_to_ragas_format ---

class TestTransformToRagasFormat:
    def test_basic_transform(self, connector, basic_trace):
        result = connector.transform_to_ragas_format(_make_eval_input(basic_trace))
        assert result["question"] == "What is Python?"
        assert result["answer"] == "Python is a programming language."

    def test_with_ground_truth(self, connector, basic_trace, ground_truth_with_output):
        result = connector.transform_to_ragas_format(
            _make_eval_input(basic_trace, ground_truth=ground_truth_with_output)
        )
        assert result["ground_truth"] == "Python is a high-level programming language."

    def test_with_context(self, connector, basic_trace, context_docs):
        result = connector.transform_to_ragas_format(
            _make_eval_input(basic_trace, context=context_docs)
        )
        assert len(result["contexts"]) == 2

    def test_missing_user_message(self, connector):
        trace = AgentTrace(messages=[_msg("assistant", "hi")])
        result = connector.transform_to_ragas_format(_make_eval_input(trace))
        # Falls back to first message as question
        assert result["question"] == "hi"

    def test_missing_assistant_message(self, connector):
        trace = AgentTrace(messages=[_msg("user", "hi")])
        with pytest.raises(ValueError, match="No assistant response"):
            connector.transform_to_ragas_format(_make_eval_input(trace))


# --- Tests: list_available_metrics ---

class TestListAvailableMetrics:
    def test_lists_all_metrics(self, connector):
        metrics = connector.list_available_metrics()
        expected = [
            "faithfulness", "context_precision", "context_recall",
            "answer_precision", "answer_recall", "answer_correctness",
            "tool_call_accuracy",
        ]
        assert metrics == expected


# --- Tests: new ground-truth metrics ---

class TestFaithfulness:
    def test_missing_context_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Context is required"):
            connector.calculate_faithfulness(_make_eval_input(basic_trace))

    def test_success(self, connector):
        mock_result = {"faithfulness": 0.91}
        mock_dataset_cls = MagicMock()
        mock_evaluate = MagicMock(return_value=mock_result)
        mock_ragas_mod = MagicMock()
        mock_ragas_mod.evaluate = mock_evaluate
        mock_metrics_mod = MagicMock()
        mock_datasets_mod = MagicMock()
        mock_datasets_mod.Dataset.from_dict.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "ragas": mock_ragas_mod,
            "ragas.metrics": mock_metrics_mod,
            "datasets": mock_datasets_mod,
        }):
            ev = _make_eval_input(_simple_trace(), context=["ctx"])
            score = connector.calculate_faithfulness(ev)
            assert score.metric_name == "ragas_faithfulness"
            assert score.score == 0.91
            assert score.metadata["metric"] == "faithfulness"


class TestContextPrecision:
    def test_missing_context_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Context is required"):
            connector.calculate_context_precision(_make_eval_input(basic_trace))

    def test_missing_ground_truth_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Ground truth"):
            connector.calculate_context_precision(_make_eval_input(basic_trace, context=["ctx"]))

    def test_success(self, connector):
        mock_result = {"context_precision": 0.87}
        mock_ragas_mod = MagicMock()
        mock_ragas_mod.evaluate = MagicMock(return_value=mock_result)
        mock_metrics_mod = MagicMock()
        mock_datasets_mod = MagicMock()
        mock_datasets_mod.Dataset.from_dict.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "ragas": mock_ragas_mod,
            "ragas.metrics": mock_metrics_mod,
            "datasets": mock_datasets_mod,
        }):
            ev = _make_eval_input(
                _simple_trace(),
                ground_truth=GroundTruth(expected_output="expected"),
                context=["ctx"],
            )
            score = connector.calculate_context_precision(ev)
            assert score.metric_name == "ragas_context_precision"
            assert score.score == 0.87
            assert score.metadata["metric"] == "context_precision"


class TestContextRecall:
    def test_missing_context_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Context is required"):
            connector.calculate_context_recall(_make_eval_input(basic_trace))

    def test_missing_ground_truth_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Ground truth"):
            connector.calculate_context_recall(_make_eval_input(basic_trace, context=["ctx"]))

    def test_success(self, connector):
        mock_result = {"context_recall": 0.75}
        mock_ragas_mod = MagicMock()
        mock_ragas_mod.evaluate = MagicMock(return_value=mock_result)
        mock_metrics_mod = MagicMock()
        mock_datasets_mod = MagicMock()
        mock_datasets_mod.Dataset.from_dict.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "ragas": mock_ragas_mod,
            "ragas.metrics": mock_metrics_mod,
            "datasets": mock_datasets_mod,
        }):
            ev = _make_eval_input(
                _simple_trace(),
                ground_truth=GroundTruth(expected_output="expected"),
                context=["ctx"],
            )
            score = connector.calculate_context_recall(ev)
            assert score.metric_name == "ragas_context_recall"
            assert score.score == 0.75
            assert score.metadata["metric"] == "context_recall"


# --- Tests: existing metrics validation ---

class TestAnswerRecall:
    def test_missing_ground_truth_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Ground truth"):
            connector.calculate_answer_recall(_make_eval_input(basic_trace))


class TestAnswerCorrectness:
    def test_missing_ground_truth_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Ground truth"):
            connector.calculate_answer_correctness(_make_eval_input(basic_trace))


class TestToolCallAccuracy:
    def test_missing_ground_truth_raises(self, connector, basic_trace):
        with pytest.raises(ValueError, match="Ground truth"):
            connector.calculate_tool_call_accuracy(_make_eval_input(basic_trace))


# --- Tests: calculate_metrics dispatcher ---

class TestCalculateMetrics:
    def test_unknown_metric_skipped(self, connector, basic_trace):
        scores = connector.calculate_metrics(_make_eval_input(basic_trace), metric_names=["nonexistent"])
        assert scores == []

    def test_failing_metric_continues(self, connector, basic_trace):
        scores = connector.calculate_metrics(
            _make_eval_input(basic_trace),
            metric_names=["faithfulness", "context_recall", "answer_correctness"],
        )
        assert scores == []


# --- Tests: unavailable library ---

class TestUnavailableLibrary:
    def test_is_available_false(self):
        c = RAGASConnector()
        c._ragas_available = False
        assert c.is_available() is False

    def test_calculate_metrics_raises_when_unavailable(self):
        c = RAGASConnector()
        c._ragas_available = False
        # Requirement 3.5: requesting an integration metric without the
        # 'integrations' extra raises an ImportError naming the extra and the
        # exact pip install command, and returns no metric result.
        with pytest.raises(ImportError, match=r"integrations.*pip install 'uaef\[integrations\]'"):
            c.calculate_metrics(_make_eval_input(_simple_trace()), metric_names=["faithfulness"])

    def test_transform_raises_when_unavailable(self):
        c = RAGASConnector()
        c._ragas_available = False
        with pytest.raises(ImportError, match=r"integrations.*pip install 'uaef\[integrations\]'"):
            c.transform_to_ragas_format(_make_eval_input(_simple_trace()))
