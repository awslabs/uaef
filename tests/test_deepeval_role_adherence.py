# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DeepEval role-adherence wrapper and the tightened
existing `role_adherence` metric.

Bedrock and DeepEval's `RoleAdherenceMetric.measure` are fully mocked.
No live API calls; no live judge calls.

The DeepEval library is required for the helper / connector tests because
they import `Turn` and `ConversationalTestCase` directly. The whole module
is skipped if DeepEval isn't available — matching the existing pattern in
`tests/test_deepeval_connector.py`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

deepeval = pytest.importorskip("deepeval")

from deepeval.test_case import Turn  # noqa: E402

from uaef.integrations.deepeval_connector import (  # noqa: E402
    _messages_to_deepeval_turns,
)
from uaef.models.agent_trace import AgentTrace  # noqa: E402
from uaef.models.message import Message, MessageRole  # noqa: E402


def _msg(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content, timestamp=datetime.now(timezone.utc))


def _trace(messages: List[Message]) -> AgentTrace:
    return AgentTrace(trace_id=uuid4(), messages=messages, tool_calls=[])


# -- Helper: messages → DeepEval Turn list ---------------------------------


class TestMessagesToDeepevalTurns:
    def test_empty_messages_returns_empty(self):
        assert _messages_to_deepeval_turns([]) == []

    def test_user_assistant_pairs_in_order(self):
        msgs = [
            _msg(MessageRole.USER, "u1"),
            _msg(MessageRole.ASSISTANT, "a1"),
            _msg(MessageRole.USER, "u2"),
            _msg(MessageRole.ASSISTANT, "a2"),
        ]
        turns = _messages_to_deepeval_turns(msgs)
        assert all(isinstance(t, Turn) for t in turns)
        assert [(t.role, t.content) for t in turns] == [
            ("user", "u1"), ("assistant", "a1"),
            ("user", "u2"), ("assistant", "a2"),
        ]

    def test_system_and_tool_messages_are_skipped(self):
        msgs = [
            _msg(MessageRole.SYSTEM, "sys"),
            _msg(MessageRole.USER, "u1"),
            _msg(MessageRole.TOOL, "tool"),
            _msg(MessageRole.ASSISTANT, "a1"),
        ]
        turns = _messages_to_deepeval_turns(msgs)
        assert [(t.role, t.content) for t in turns] == [
            ("user", "u1"), ("assistant", "a1"),
        ]

    def test_user_only_passes_through(self):
        msgs = [_msg(MessageRole.USER, "hi"), _msg(MessageRole.USER, "still hi")]
        turns = _messages_to_deepeval_turns(msgs)
        assert [(t.role, t.content) for t in turns] == [
            ("user", "hi"), ("user", "still hi"),
        ]


# -- Connector method ------------------------------------------------------

from uaef.integrations.deepeval_connector import DeepEvalConnector  # noqa: E402
from uaef.models.evaluation_input import EvaluationInput  # noqa: E402
from uaef.models.ground_truth import GroundTruth  # noqa: E402
from uaef.models.metric_score import MetricScore  # noqa: E402


def _three_turn_trace() -> AgentTrace:
    return _trace([
        _msg(MessageRole.USER, "u1"),
        _msg(MessageRole.ASSISTANT, "a1"),
        _msg(MessageRole.USER, "u2"),
        _msg(MessageRole.ASSISTANT, "a2"),
        _msg(MessageRole.USER, "u3"),
        _msg(MessageRole.ASSISTANT, "a3"),
    ])


def _gt_with_role(role: str) -> GroundTruth:
    return GroundTruth(
        expected_output="ignored",
        expected_arguments={"chatbot_role": role},
    )


def _eval_input(trace: AgentTrace, ground_truth: GroundTruth | None) -> EvaluationInput:
    return EvaluationInput(trace=trace, ground_truth=ground_truth)


def _patched_role_metric_returning(score: float, reason: str, verdicts: list) -> MagicMock:
    """Build a MagicMock that mimics a constructed RoleAdherenceMetric.

    The connector calls `RoleAdherenceMetric(model=...)` and then
    `metric.measure(test_case)`. We patch the class with a MagicMock; calling
    it returns the per-instance mock that holds `score`, `reason`, and
    `out_of_character_verdicts.verdicts`.
    """
    instance = MagicMock()
    instance.score = score
    instance.reason = reason
    instance.out_of_character_verdicts = MagicMock(verdicts=verdicts)
    return instance


def _verdict(index: int, ai_message: str, reason: str) -> MagicMock:
    v = MagicMock()
    v.index = index
    v.ai_message = ai_message
    v.reason = reason
    return v


class TestCalculateRoleAdherence:
    def test_happy_path_returns_metric_score(self):
        connector = DeepEvalConnector()
        if not connector.is_available():
            pytest.skip("DeepEval unavailable")
        eval_in = _eval_input(_three_turn_trace(), _gt_with_role("You are an ACME support agent."))

        instance = _patched_role_metric_returning(
            score=2 / 3,
            reason="One turn drifted from the ACME persona.",
            verdicts=[_verdict(1, "a2 (turn #2)", "Switched to a different persona.")],
        )
        with patch("deepeval.metrics.RoleAdherenceMetric", return_value=instance) as cls_mock:
            result = connector.calculate_role_adherence(eval_in)

        assert isinstance(result, MetricScore)
        assert result.metric_name == "deepeval_role_adherence"
        assert result.score == pytest.approx(2 / 3)
        assert "ACME persona" in (result.reasoning or "")
        ooc = result.metadata["out_of_character_turns"]
        assert ooc == [
            {
                "turn_index": 1,
                "ai_message": "a2 (turn #2)",
                "reason": "Switched to a different persona.",
            }
        ]
        assert result.metadata["library"] == "deepeval"
        assert result.metadata["metric"] == "role_adherence"
        assert result.metadata["assistant_turn_count"] == 3

        # measure was actually invoked once on a ConversationalTestCase
        instance.measure.assert_called_once()
        # and the metric class was constructed with model from the connector
        cls_mock.assert_called_once()

    def test_missing_chatbot_role_returns_not_applicable(self):
        connector = DeepEvalConnector()
        if not connector.is_available():
            pytest.skip("DeepEval unavailable")
        eval_in = _eval_input(
            _three_turn_trace(),
            GroundTruth(expected_output="not the role"),  # no chatbot_role key
        )
        result = connector.calculate_role_adherence(eval_in)
        assert result.score is None
        assert result.metadata.get("warning") == "not_applicable"
        assert "chatbot_role" in (result.reasoning or "")

    def test_no_ground_truth_returns_not_applicable(self):
        connector = DeepEvalConnector()
        if not connector.is_available():
            pytest.skip("DeepEval unavailable")
        eval_in = _eval_input(_three_turn_trace(), None)
        result = connector.calculate_role_adherence(eval_in)
        assert result.score is None
        assert result.metadata.get("warning") == "not_applicable"
        assert "chatbot_role" in (result.reasoning or "")

    def test_empty_filtered_turn_list_returns_not_applicable(self):
        connector = DeepEvalConnector()
        if not connector.is_available():
            pytest.skip("DeepEval unavailable")
        eval_in = _eval_input(
            _trace([_msg(MessageRole.SYSTEM, "sys"), _msg(MessageRole.TOOL, "t")]),
            _gt_with_role("ACME"),
        )
        result = connector.calculate_role_adherence(eval_in)
        assert result.score is None
        assert result.metadata.get("warning") == "not_applicable"
        assert "no user/assistant messages" in (result.reasoning or "").lower()

    def test_measure_exception_propagates(self):
        connector = DeepEvalConnector()
        if not connector.is_available():
            pytest.skip("DeepEval unavailable")
        eval_in = _eval_input(_three_turn_trace(), _gt_with_role("ACME"))
        instance = MagicMock()
        instance.measure.side_effect = RuntimeError("boom")
        with patch("deepeval.metrics.RoleAdherenceMetric", return_value=instance):
            with pytest.raises(RuntimeError, match="boom"):
                connector.calculate_role_adherence(eval_in)


# -- Wrapper class ---------------------------------------------------------

from uaef.metrics.deepeval_metrics import DeepEvalRoleAdherenceMetric  # noqa: E402


class TestDeepEvalRoleAdherenceMetric:
    def test_metric_identity(self):
        m = DeepEvalRoleAdherenceMetric()
        assert m.get_name() == "deepeval_role_adherence"
        assert m.get_dimension() == "DeepEval"
        assert m.requires_ground_truth() is True
        assert m.requires_llm_judge() is True
        assert m.get_description()  # non-empty

    def test_unavailable_raises_import_error(self):
        m = DeepEvalRoleAdherenceMetric()
        eval_in = _eval_input(_three_turn_trace(), _gt_with_role("ACME"))

        # Patch the connector factory so is_available() returns False.
        fake_connector = MagicMock()
        fake_connector.is_available.return_value = False
        with patch(
            "uaef.metrics.deepeval_metrics._get_connector",
            return_value=fake_connector,
        ):
            # Requirement 3.5: requesting an integration metric without the
            # 'integrations' extra raises an ImportError naming the extra and
            # the exact pip install command, and returns no metric result.
            with pytest.raises(
                ImportError, match=r"integrations.*pip install 'uaef\[integrations\]'"
            ):
                m.calculate(eval_in)

        # No connector work is attempted once the guard trips.
        fake_connector.calculate_role_adherence.assert_not_called()

    def test_delegates_to_connector(self):
        m = DeepEvalRoleAdherenceMetric()
        eval_in = _eval_input(_three_turn_trace(), _gt_with_role("ACME"))

        fake_connector = MagicMock()
        fake_connector.is_available.return_value = True
        fake_score = MetricScore(
            metric_name="deepeval_role_adherence",
            score=0.75,
            reasoning="ok",
            metadata={"library": "deepeval"},
        )
        fake_connector.calculate_role_adherence.return_value = fake_score
        with patch(
            "uaef.metrics.deepeval_metrics._get_connector",
            return_value=fake_connector,
        ):
            result = m.calculate(eval_in)

        assert result is fake_score
        fake_connector.calculate_role_adherence.assert_called_once_with(eval_in)


# -- Registry --------------------------------------------------------------


class TestRegistry:
    def test_deepeval_role_adherence_registered_by_default(self):
        from uaef.metrics.registry import (
            get_metric,
            list_metrics,
            list_metrics_by_dimension,
            reset_registry,
        )

        reset_registry()
        names = list_metrics()
        assert "deepeval_role_adherence" in names

        by_dim = list_metrics_by_dimension()
        assert "deepeval_role_adherence" in by_dim["DeepEval"]

        m = get_metric("deepeval_role_adherence")
        assert m.get_name() == "deepeval_role_adherence"


# -- Evaluator routing -----------------------------------------------------


class TestEvaluatorRouting:
    def test_listed_in_full_trace_only(self):
        # The DeepEval role-adherence metric reads the entire turn list and
        # produces one MetricScore per evaluation. It must NOT be re-run
        # inside the per-turn loop.
        import inspect
        from uaef.evaluation import base_evaluator

        src = inspect.getsource(base_evaluator)
        assert '"deepeval_role_adherence"' in src


# -- Tightened existing role_adherence -------------------------------------

from uaef.metrics.multi_turn import RoleAdherenceMetric  # noqa: E402


class TestRoleAdherenceTightened:
    def test_missing_chatbot_role_returns_not_applicable_with_clear_reason(self):
        # No expected_arguments at all
        m = RoleAdherenceMetric()
        eval_in = _eval_input(_three_turn_trace(), GroundTruth(expected_output="not the role"))
        result = m.calculate(eval_in)
        assert result.score is None
        assert result.metadata.get("warning") == "not_applicable"
        assert "chatbot_role" in (result.reasoning or "")

    def test_role_in_expected_output_no_longer_picked_up(self):
        # The legacy fallback used to read GroundTruth.expected_output as the
        # role. After tightening, that path is gone.
        m = RoleAdherenceMetric()
        gt = GroundTruth(expected_output="You are an ACME support agent.")  # role-shaped, but in the wrong field
        eval_in = _eval_input(_three_turn_trace(), gt)
        result = m.calculate(eval_in)
        assert result.score is None
        assert result.metadata.get("warning") == "not_applicable"
        assert "chatbot_role" in (result.reasoning or "")

    def test_role_in_expected_arguments_proceeds_to_judge(self):
        m = RoleAdherenceMetric()
        gt = _gt_with_role("You are an ACME support agent.")
        eval_in = _eval_input(_three_turn_trace(), gt)
        with patch(
            "uaef.metrics.multi_turn._invoke_bedrock_judge",
            return_value={"score": 0.8, "reasoning": "Stayed in role most turns."},
        ) as mock_judge:
            result = m.calculate(eval_in)
        mock_judge.assert_called_once()
        assert result.score == pytest.approx(0.8)
        # The role text (trusted ground-truth data) was forwarded into the
        # user message. Security review H-01 split _invoke_bedrock_judge's
        # single `prompt` argument into (system_prompt, user_message,
        # metric_name) — the rubric now lives in system_prompt, and the
        # chatbot_role + conversation live in user_message.
        user_message_arg = mock_judge.call_args[0][1]
        assert "You are an ACME support agent." in user_message_arg
