# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-turn sentiment / agent_tone / naturalness metrics.

Covers:
- The pair-extraction helper (`_extract_user_assistant_pairs`).
- The aggregation / failure logic in the `_AMACEPerTurnMetric` base.
- Parallel dispatch via `asyncio.gather` in `calculate_async`.

Bedrock is fully mocked. No live API calls.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import List
from unittest.mock import patch
from uuid import uuid4

import pytest

from uaef.metrics.multi_turn import (
    PerTurnAgentToneMetric,
    PerTurnNaturalnessMetric,
    PerTurnSentimentMetric,
    _AMACEPerTurnMetric,
    _extract_user_assistant_pairs,
    _normalize_1_to_5,
)
from uaef.models.agent_trace import AgentTrace
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.message import Message, MessageRole
from uaef.models.metric_score import MetricScore


def _msg(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content, timestamp=datetime.now(timezone.utc))


def _trace(messages: List[Message]) -> AgentTrace:
    return AgentTrace(trace_id=uuid4(), messages=messages, tool_calls=[])


# -- Pair extraction --------------------------------------------------------


class TestExtractUserAssistantPairs:
    def test_empty_messages_returns_empty(self):
        assert _extract_user_assistant_pairs([]) == []

    def test_user_only_no_assistant_returns_empty(self):
        msgs = [_msg(MessageRole.USER, "hi")]
        assert _extract_user_assistant_pairs(msgs) == []

    def test_three_pairs_in_order(self):
        msgs = [
            _msg(MessageRole.USER, "u1"),
            _msg(MessageRole.ASSISTANT, "a1"),
            _msg(MessageRole.USER, "u2"),
            _msg(MessageRole.ASSISTANT, "a2"),
            _msg(MessageRole.USER, "u3"),
            _msg(MessageRole.ASSISTANT, "a3"),
        ]
        pairs = _extract_user_assistant_pairs(msgs)
        assert [(u.content, a.content) for u, a in pairs] == [
            ("u1", "a1"), ("u2", "a2"), ("u3", "a3"),
        ]

    def test_system_and_tool_messages_are_skipped(self):
        msgs = [
            _msg(MessageRole.SYSTEM, "sys"),
            _msg(MessageRole.USER, "u1"),
            _msg(MessageRole.TOOL, "tool"),
            _msg(MessageRole.ASSISTANT, "a1"),
            _msg(MessageRole.USER, "u2"),
            _msg(MessageRole.ASSISTANT, "a2"),
        ]
        pairs = _extract_user_assistant_pairs(msgs)
        assert [(u.content, a.content) for u, a in pairs] == [("u1", "a1"), ("u2", "a2")]

    def test_consecutive_assistant_messages_only_first_closes_pair(self):
        msgs = [
            _msg(MessageRole.USER, "u1"),
            _msg(MessageRole.ASSISTANT, "a1_first"),
            _msg(MessageRole.ASSISTANT, "a1_followup"),
            _msg(MessageRole.USER, "u2"),
            _msg(MessageRole.ASSISTANT, "a2"),
        ]
        pairs = _extract_user_assistant_pairs(msgs)
        assert [(u.content, a.content) for u, a in pairs] == [("u1", "a1_first"), ("u2", "a2")]

    def test_trailing_user_with_no_assistant_dropped(self):
        msgs = [
            _msg(MessageRole.USER, "u1"),
            _msg(MessageRole.ASSISTANT, "a1"),
            _msg(MessageRole.USER, "u2_trailing"),
        ]
        pairs = _extract_user_assistant_pairs(msgs)
        assert [(u.content, a.content) for u, a in pairs] == [("u1", "a1")]


# -- _AMACEPerTurnMetric base -----------------------------------------------


class _StubPerTurn(_AMACEPerTurnMetric):
    """Stub subclass for testing the base class in isolation.

    Each `_build_pair_prompt` returns a (system_prompt, user_message) pair
    with a unique sentinel string in the user_message so the test can assert
    one judge call per pair, matching the security review H-01 split of
    `_invoke_bedrock_judge` into separate system/user arguments.
    `_parse_pair_response` returns the raw 1-5 score normalized.
    """

    def get_name(self) -> str:
        return "stub_per_turn"

    def get_description(self) -> str:
        return "stub"

    def _build_pair_prompt(self, user_text: str, assistant_text: str) -> tuple:
        return "SYSTEM", f"PROMPT[{user_text}|{assistant_text}]"

    def _parse_pair_response(self, response_json: dict) -> tuple:
        raw = response_json.get("Score")
        reason = response_json.get("Reason", "")
        return _normalize_1_to_5(raw), reason


def _three_pair_trace() -> AgentTrace:
    return _trace([
        _msg(MessageRole.USER, "u1"),
        _msg(MessageRole.ASSISTANT, "a1"),
        _msg(MessageRole.USER, "u2"),
        _msg(MessageRole.ASSISTANT, "a2"),
        _msg(MessageRole.USER, "u3"),
        _msg(MessageRole.ASSISTANT, "a3"),
    ])


def _eval_input(trace: AgentTrace) -> EvaluationInput:
    return EvaluationInput(trace=trace, ground_truth=None)


class TestAMACEPerTurnMetricBase:
    def test_zero_pairs_returns_not_applicable(self):
        metric = _StubPerTurn()
        eval_in = _eval_input(_trace([]))
        result = metric.calculate(eval_in)
        assert isinstance(result, MetricScore)
        assert result.score is None
        assert result.metadata.get("warning") == "not_applicable"
        assert result.metadata.get("pair_count") == 0

    def test_all_pairs_succeed_aggregates_mean(self):
        metric = _StubPerTurn()
        eval_in = _eval_input(_three_pair_trace())
        # Key responses by prompt content so parallel dispatch order does
        # not affect which response goes to which turn.
        by_prompt = {
            "PROMPT[u1|a1]": {"Score": 5, "Reason": "great"},  # turn 0 → 1.0
            "PROMPT[u2|a2]": {"Score": 3, "Reason": "ok"},     # turn 1 → 0.5
            "PROMPT[u3|a3]": {"Score": 1, "Reason": "bad"},    # turn 2 → 0.0
        }

        def by_prompt_side_effect(system_prompt, user_message, metric_name):
            return by_prompt[user_message]

        with patch(
            "uaef.metrics.multi_turn._invoke_bedrock_judge",
            side_effect=by_prompt_side_effect,
        ) as mock_judge:
            result = metric.calculate(eval_in)
        assert mock_judge.call_count == 3
        assert result.score == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert result.metadata["pair_count"] == 3
        assert result.metadata["scored_count"] == 3
        assert result.metadata["failed_count"] == 0
        assert result.metadata["aggregation"] == "mean"
        per_turn = result.metadata["per_turn_scores"]
        assert [t["turn_index"] for t in per_turn] == [0, 1, 2]
        assert [t["score"] for t in per_turn] == [1.0, 0.5, 0.0]

    def test_partial_failure_excludes_bad_turn_from_mean(self):
        metric = _StubPerTurn()
        eval_in = _eval_input(_three_pair_trace())

        def side_effect(system_prompt, user_message, metric_name):
            if "u2" in user_message:
                raise RuntimeError("Bedrock API call failed: kaboom")
            return {"Score": 5, "Reason": "ok"}

        with patch(
            "uaef.metrics.multi_turn._invoke_bedrock_judge",
            side_effect=side_effect,
        ):
            result = metric.calculate(eval_in)
        assert result.metadata["pair_count"] == 3
        assert result.metadata["scored_count"] == 2
        assert result.metadata["failed_count"] == 1
        assert result.score == pytest.approx(1.0)
        per_turn = result.metadata["per_turn_scores"]
        assert per_turn[1]["score"] is None
        assert "judge error" in per_turn[1]["reasoning"]

    def test_all_pairs_fail_returns_none_score(self):
        metric = _StubPerTurn()
        eval_in = _eval_input(_three_pair_trace())
        with patch(
            "uaef.metrics.multi_turn._invoke_bedrock_judge",
            side_effect=RuntimeError("dead"),
        ):
            result = metric.calculate(eval_in)
        assert result.score is None
        assert result.metadata["scored_count"] == 0
        assert result.metadata["failed_count"] == 3
        assert "all judge calls failed" in (result.reasoning or "").lower()

    def test_headline_reasoning_references_lowest_turn(self):
        metric = _StubPerTurn()
        eval_in = _eval_input(_three_pair_trace())
        by_prompt = {
            "PROMPT[u1|a1]": {"Score": 5, "Reason": "great"},
            "PROMPT[u2|a2]": {"Score": 1, "Reason": "very bad turn"},
            "PROMPT[u3|a3]": {"Score": 4, "Reason": "fine"},
        }

        def by_prompt_side_effect(system_prompt, user_message, metric_name):
            return by_prompt[user_message]

        with patch(
            "uaef.metrics.multi_turn._invoke_bedrock_judge",
            side_effect=by_prompt_side_effect,
        ):
            result = metric.calculate(eval_in)
        assert result.reasoning is not None
        assert "turn 1" in result.reasoning.lower()
        assert "very bad turn" in result.reasoning


class TestAMACEPerTurnMetricAsyncParallelism:
    def test_calculate_async_parallelizes_judge_calls(self):
        metric = _StubPerTurn()
        eval_in = _eval_input(_trace([
            _msg(MessageRole.USER, f"u{i}") if i % 2 == 0
            else _msg(MessageRole.ASSISTANT, f"a{i}")
            for i in range(10)
        ]))

        def slow_judge(system_prompt, user_message, metric_name):
            time.sleep(0.1)
            return {"Score": 4, "Reason": "ok"}

        with patch(
            "uaef.metrics.multi_turn._invoke_bedrock_judge",
            side_effect=slow_judge,
        ):
            t0 = time.perf_counter()
            result = asyncio.run(metric.calculate_async(eval_in))
            elapsed = time.perf_counter() - t0

        assert result.metadata["scored_count"] == 5
        # 5 sequential calls would take ~0.5s. Parallel should be well under
        # half that. Allow generous slack for CI noise.
        assert elapsed < 0.35, f"Expected parallel dispatch, took {elapsed:.2f}s"


# -- Concrete metrics -------------------------------------------------------


class TestPerTurnConcreteMetrics:
    @pytest.mark.parametrize(
        "metric_cls,expected_name",
        [
            (PerTurnSentimentMetric, "per_turn_sentiment"),
            (PerTurnAgentToneMetric, "per_turn_agent_tone"),
            (PerTurnNaturalnessMetric, "per_turn_naturalness"),
        ],
    )
    def test_metric_identity(self, metric_cls, expected_name):
        m = metric_cls()
        assert m.get_name() == expected_name
        assert m.get_dimension() == "Multi-Turn"
        assert m.requires_ground_truth() is False
        assert m.requires_llm_judge() is True
        assert m.get_description()

    @pytest.mark.parametrize(
        "metric_cls,discriminator",
        [
            (PerTurnSentimentMetric, "sentiment analyst"),
            (PerTurnAgentToneMetric, "tone and professionalism"),
            (PerTurnNaturalnessMetric, "naturalness"),
        ],
    )
    def test_pair_prompt_contains_only_single_pair(self, metric_cls, discriminator):
        m = metric_cls()
        # Security review H-01: _build_pair_prompt now returns
        # (system_prompt, user_message) rather than one concatenated
        # string — the rubric/discriminator text lives in system_prompt,
        # and the pair's own text lives in user_message (with the
        # assistant's reply wrapped in a <candidate_assistant_reply> tag).
        system_prompt, user_message = m._build_pair_prompt("hello bot", "hi there")
        full_prompt = system_prompt + "\n" + user_message
        # Per-pair substitution
        assert "hello bot" in full_prompt
        assert "hi there" in full_prompt
        # Per-class identity — guards against subclass prompt swaps
        assert discriminator.lower() in full_prompt.lower()
        # Recency-weighting strings stripped from full-trace prompts
        assert "weigh later interactions more heavily" not in full_prompt.lower()
        assert "consider the entire conversation flow" not in full_prompt.lower()

    @pytest.mark.parametrize(
        "metric_cls,expected_name",
        [
            (PerTurnSentimentMetric, "per_turn_sentiment"),
            (PerTurnAgentToneMetric, "per_turn_agent_tone"),
            (PerTurnNaturalnessMetric, "per_turn_naturalness"),
        ],
    )
    def test_end_to_end_with_mocked_judge(self, metric_cls, expected_name):
        m = metric_cls()
        eval_in = _eval_input(_three_pair_trace())
        responses = [
            {"Score": 4, "Reason": "fine"},
            {"Score": 4, "Reason": "fine"},
            {"Score": 4, "Reason": "fine"},
        ]
        with patch(
            "uaef.metrics.multi_turn._invoke_bedrock_judge",
            side_effect=responses,
        ):
            result = m.calculate(eval_in)
        assert result.metric_name == expected_name
        assert result.metadata["pair_count"] == 3
        assert result.score == pytest.approx(0.75)


# -- Registry --------------------------------------------------------------


class TestRegistry:
    def test_per_turn_metrics_registered_by_default(self):
        from uaef.metrics.registry import (
            get_metric,
            list_metrics,
            list_metrics_by_dimension,
            reset_registry,
        )

        reset_registry()
        names = list_metrics()
        for n in ("per_turn_sentiment", "per_turn_agent_tone", "per_turn_naturalness"):
            assert n in names

        # Each instantiates and reports the Multi-Turn dimension
        by_dim = list_metrics_by_dimension()
        assert "per_turn_sentiment" in by_dim["Multi-Turn"]
        assert "per_turn_agent_tone" in by_dim["Multi-Turn"]
        assert "per_turn_naturalness" in by_dim["Multi-Turn"]

        # Factory returns the right concrete class
        m = get_metric("per_turn_sentiment")
        assert m.get_name() == "per_turn_sentiment"


# -- Evaluator routing -----------------------------------------------------


class TestEvaluatorRouting:
    def test_per_turn_metrics_listed_in_full_trace_only(self):
        # The per-turn metrics extract pairs from the full message list and
        # produce one MetricScore per evaluation. They must NOT be re-run
        # inside the per-turn evaluation loop, which would feed them
        # single-turn slices and break pair extraction.
        import inspect
        from uaef.evaluation import base_evaluator

        src = inspect.getsource(base_evaluator)
        assert '"per_turn_sentiment"' in src
        assert '"per_turn_agent_tone"' in src
        assert '"per_turn_naturalness"' in src
