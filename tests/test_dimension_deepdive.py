# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dimension deep-dive must match the display names evaluators emit."""

from uuid import uuid4

import pytest

from uaef.models.dimension_result import DimensionResult
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.metric_score import MetricScore
from uaef.reporting.dimension_deepdive import DimensionDashboardGenerator


def _result(dimension_name, scores):
    return EvaluationResult(
        trace_id=uuid4(),
        dimension_results=[
            DimensionResult(
                dimension_name=dimension_name,
                metric_scores=[MetricScore(metric_name=n, score=s) for n, s in scores],
                aggregate_score=next((s for _, s in scores if s is not None), None),
                weight=1.0,
            )
        ],
        overall_score=0.8,
        passed=True,
    )


@pytest.mark.parametrize(
    "emitted, requested",
    [
        ("Tool Calling", "tool_calling"),
        ("Multi-Turn", "multi_turn"),
        ("tool_calling", "tool_calling"),
    ],
)
def test_matches_display_and_snake_case_names(emitted, requested):
    prefix = "tool_" if requested == "tool_calling" else "context_"
    results = [_result(emitted, [(f"{prefix}a", 0.9), (f"{prefix}b", 0.6)])]
    data = DimensionDashboardGenerator()._extract_dimension_data(results, requested)
    assert sorted(data["metrics"]) == [f"{prefix}a", f"{prefix}b"]
    assert data["dimension_scores"] == [0.9]


def test_none_metric_scores_are_skipped():
    results = [_result("Tool Calling", [("tool_a", None), ("tool_b", 0.7)])]
    data = DimensionDashboardGenerator()._extract_dimension_data(results, "tool_calling")
    assert data["metrics"] == ["tool_b"]


def test_full_dashboard_from_real_names():
    results = [
        _result("Tool Calling", [("tool_a", 0.9), ("tool_b", 0.5)]),
        _result("Tool Calling", [("tool_a", 0.8), ("tool_b", None)]),
    ]
    dashboard = DimensionDashboardGenerator().generate_dashboard(
        results=results, dimension="tool_calling", title="Tool Calling deep-dive"
    )
    assert dashboard.charts
