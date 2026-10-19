# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analysis and reporting must tolerate metric scores of None.

A metric that is not applicable, or that raised (kept as a placeholder score
since the #6 follow-ups), has score=None. Every consumer that compares or
averages scores has to skip those instead of raising TypeError.
"""

from uuid import uuid4

import pytest

from uaef.analysis.clustering import FailureClusterer
from uaef.analysis.improvement_tracker import ImprovementTracker
from uaef.analysis.recommendations import RecommendationEngine
from uaef.analysis.root_cause import RootCauseAnalyzer
from uaef.analysis.trends import TrendAnalyzer
from uaef.evaluation.multi_turn import MultiTurnEvaluator
from uaef.experiments import AgentDesign, ExperimentManager
from uaef.models.dimension_result import DimensionResult
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.metric_score import MetricScore
from uaef.reporting.safety import SafetyReportGenerator


def _result(passed=False, score=0.4):
    return EvaluationResult(
        trace_id=uuid4(),
        dimension_results=[
            DimensionResult(
                dimension_name="Tool Calling",
                metric_scores=[
                    MetricScore(metric_name="tool_selection_accuracy", score=score),
                    MetricScore(metric_name="parameter_quality", score=None),
                ],
                aggregate_score=score,
                weight=1.0,
            ),
            DimensionResult(
                dimension_name="Responsible AI",
                metric_scores=[
                    MetricScore(metric_name="safety_score", score=None),
                    MetricScore(metric_name="toxicity_score", score=0.9),
                ],
                aggregate_score=0.9,
                weight=1.0,
            ),
        ],
        overall_score=score,
        passed=passed,
        failures=[] if passed else ["Tool Calling below threshold"],
    )


@pytest.fixture
def results():
    return [_result(), _result(), _result(passed=True, score=0.95)]


def test_experiment_manager_aggregates(results):
    manager = ExperimentManager()
    exp = manager.create_experiment(name="e", description="d")
    run = manager.create_run(
        experiment_id=exp.experiment_id, run_name="r",
        agent_design=AgentDesign(framework="f", model="m"), config={},
    )
    for r in results:
        manager.record_evaluation(run.run_id, r)
    aggregates = manager.get_run(run.run_id).aggregate_metrics
    assert "parameter_quality" not in aggregates
    assert aggregates["toxicity_score"] == pytest.approx(0.9)


def test_root_cause(results):
    analyzer = RootCauseAnalyzer()
    analyzer.analyze_failure(results[0])
    analyzer.identify_failure_patterns(results)


def test_clustering_feature_vectors(results):
    clusterer = FailureClusterer()
    clusterer._create_feature_vectors(results)
    clusterer._summarize_cluster(0, results)


def test_improvement_tracker(results):
    averages = ImprovementTracker()._calculate_average_metrics(results)
    assert "safety_score" not in averages


def test_recommendations(results):
    engine = RecommendationEngine()
    engine._generate_optimization_recommendations(results, [r for r in results if r.passed])


def test_trends(results):
    data = TrendAnalyzer()._extract_metric_data(results)
    assert all(v is not None for d in data.values() for v in d["values"])


def test_safety_report(results):
    SafetyReportGenerator()._extract_safety_metrics(results)


def test_multi_turn_unmet_criteria():
    scores = [
        MetricScore(metric_name="conversation_completeness", score=None),
        MetricScore(metric_name="context_retention", score=None),
        MetricScore(metric_name="coherence", score=None),
    ]
    MultiTurnEvaluator()._identify_unmet_criteria(metric_scores=scores, goal=None, passed=False)


def test_cluster_failures_end_to_end():
    failed = [_result(score=0.3 + i * 0.01) for i in range(4)]
    clusters = FailureClusterer(min_cluster_size=2).cluster_failures(failed)
    assert isinstance(clusters, list)
