# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dimension aggregation, overall score, thresholds, and metric failures."""

import asyncio
from uuid import uuid4

import pytest

import uaef.metrics  # noqa: F401  (registers built-in metrics)
from uaef.evaluation import base_evaluator
from uaef.evaluation.multi_agent import MultiAgentEvaluator
from uaef.evaluation.multi_turn import MultiTurnEvaluator
from uaef.evaluation.offline import OfflineEvaluator
from uaef.evaluation.online import OnlineEvaluator
from uaef.evaluation.single_agent import SingleAgentEvaluator
from uaef.metrics.base import BaseMetric
from uaef.metrics.registry import get_registry
from uaef.models import AgentTrace
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore


def _metric_for(dimension: str) -> str:
    return get_registry().list_metrics_by_dimension()[dimension][0]


class _RaisingMetric(BaseMetric):
    def calculate(self, evaluation_input):
        raise RuntimeError("judge timed out")

    async def calculate_async(self, evaluation_input):
        raise RuntimeError("judge timed out")

    def get_name(self):
        return "raising_metric"

    def requires_ground_truth(self):
        return False

    def requires_llm_judge(self):
        return False


@pytest.fixture
def evaluator():
    return SingleAgentEvaluator()


@pytest.fixture
def evaluation_input():
    return EvaluationInput(
        trace=AgentTrace(trace_id=uuid4(), messages=[], tool_calls=[])
    )


class TestDefaultDimensionKeys:
    @pytest.mark.parametrize(
        "evaluator_cls",
        [SingleAgentEvaluator, MultiAgentEvaluator, OfflineEvaluator,
         MultiTurnEvaluator, OnlineEvaluator],
    )
    def test_default_keys_match_registered_dimensions(self, evaluator_cls):
        known = set(get_registry().list_metrics_by_dimension().keys())
        ev = evaluator_cls()
        weights = getattr(ev, "default_dimension_weights", None) or {}
        assert set(weights) <= known
        assert set(ev.default_thresholds) <= known

    def test_default_weights_are_applied(self, evaluator):
        scores = [
            MetricScore(metric_name=_metric_for("Tool Calling"), score=1.0),
            MetricScore(metric_name=_metric_for("Performance"), score=0.0),
        ]
        dims = evaluator.aggregate_dimension_scores(
            scores, evaluator.default_dimension_weights
        )
        by_name = {d.dimension_name: d for d in dims}
        assert by_name["Tool Calling"].weight == 0.25
        assert by_name["Performance"].weight == 0.10
        # (1.0 * 0.25 + 0.0 * 0.10) / 0.35
        assert evaluator.calculate_overall_score(dims) == pytest.approx(0.25 / 0.35)

    def test_default_thresholds_are_enforced(self, evaluator):
        scores = [MetricScore(metric_name=_metric_for("Tool Calling"), score=0.5)]
        dims = evaluator.aggregate_dimension_scores(
            scores, evaluator.default_dimension_weights
        )
        passed, failures = evaluator.check_thresholds(
            dims, evaluator.default_thresholds
        )
        assert not passed
        assert failures and failures[0].startswith("Tool Calling")

    def test_unknown_key_warns_once(self, evaluator, monkeypatch):
        warnings = []
        monkeypatch.setattr(base_evaluator.logger, "warning", warnings.append)
        scores = [MetricScore(metric_name=_metric_for("Tool Calling"), score=1.0)]
        evaluator.aggregate_dimension_scores(scores, {"tool_calling": 1.0})
        evaluator.aggregate_dimension_scores(scores, {"tool_calling": 1.0})
        assert len([w for w in warnings if "'tool_calling'" in w]) == 1


class TestNotApplicableDimension:
    def test_all_none_dimension_is_not_applicable(self, evaluator):
        scores = [
            MetricScore(metric_name=_metric_for("Tool Calling"), score=0.9),
            MetricScore(metric_name=_metric_for("Multi-Agent"), score=None),
        ]
        dims = evaluator.aggregate_dimension_scores(
            scores, evaluator.default_dimension_weights
        )
        by_name = {d.dimension_name: d for d in dims}
        assert by_name["Multi-Agent"].aggregate_score is None

        assert evaluator.calculate_overall_score(dims) == pytest.approx(0.9)
        passed, failures = evaluator.check_thresholds(
            dims, {"Tool Calling": 0.7, "Multi-Agent": 0.7}
        )
        assert passed
        assert failures == []

    def test_only_inapplicable_dimensions(self, evaluator):
        scores = [MetricScore(metric_name=_metric_for("Multi-Agent"), score=None)]
        dims = evaluator.aggregate_dimension_scores(scores)
        assert evaluator.calculate_overall_score(dims) == 0.0


class TestMetricFailures:
    def test_sync_failure_returns_placeholder(self, evaluator, evaluation_input):
        result = evaluator._safe_calculate_metric(_RaisingMetric(), evaluation_input)
        assert result is not None
        assert result.metric_name == "raising_metric"
        assert result.score is None
        assert "Evaluation failed" in result.reasoning
        assert result.metadata["error_type"] == "RuntimeError"

    def test_async_failure_returns_placeholder(self, evaluator, evaluation_input):
        result = asyncio.run(
            evaluator._safe_calculate_metric_async(_RaisingMetric(), evaluation_input)
        )
        assert result is not None
        assert result.score is None
        assert "judge timed out" in result.metadata["error"]

    def test_failed_metric_is_kept_in_parallel_results(self, evaluator, evaluation_input):
        scores = evaluator.calculate_metrics_parallel([_RaisingMetric()], evaluation_input)
        assert [s.metric_name for s in scores] == ["raising_metric"]
