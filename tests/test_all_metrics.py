# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified test suite for all UAEF metrics.
Uses parameterized tests with test data from tests/data/*.json
"""

import pytest
from tests.test_utils import load_test_data, create_evaluation_input, validate_metric_result

# Import all metrics
from uaef.metrics.tool_calling import (
    ToolSelectionAccuracyMetric,
    ToolSequenceCorrectnessMetric,
    ParameterQualityMetric,
    MCPComplianceMetric
)
from uaef.metrics.performance import (
    LatencyScoreMetric,
    TokenEfficiencyMetric,
    CostEfficiencyMetric,
    ThroughputMetric
)
from uaef.metrics.response_quality import (
    AnswerRelevanceMetric,
    CompletenessMetric,
    HallucinationScoreMetric,
    AccuracyMetric
)
from uaef.metrics.responsible_ai import (
    SafetyScoreMetric,
    BiasScoreMetric,
    PromptInjectionDetectionMetric,
    ToxicityScoreMetric
)
from uaef.metrics.multi_turn import (
    ContextRetentionMetric,
    CoherenceMetric,
    ConversationCompletenessMetric,
    TurnEfficiencyMetric
)
from uaef.metrics.multi_agent import (
    AgentUtilizationMetric,
    DelegationQualityMetric,
    WorkflowCompletionMetric,
    CoordinationEfficiencyMetric
)
from uaef.metrics.reasoning import (
    ChainOfThoughtCoherenceMetric,
    LogicalConsistencyMetric,
    ReasoningStepCorrectnessMetric,
    FallacyDetectionMetric
)


# Mapping of metric names to metric classes
METRIC_CLASSES = {
    # Tool Calling
    "tool_selection_accuracy": ToolSelectionAccuracyMetric,
    "tool_sequence_correctness": ToolSequenceCorrectnessMetric,
    "parameter_quality": ParameterQualityMetric,
    "mcp_compliance": MCPComplianceMetric,

    # Performance
    "latency_score": LatencyScoreMetric,
    "token_efficiency": TokenEfficiencyMetric,
    "cost_efficiency": CostEfficiencyMetric,
    "throughput": ThroughputMetric,

    # Response Quality (LLM-based - deterministic edge cases tested separately below)
    # Full LLM-based tests run via tests/run_llm_metrics.py

    # Responsible AI (LLM-based - separate tests)
    # "safety_score": SafetyScoreMetric,
    # "bias_score": BiasScoreMetric,
    "prompt_injection_detection": PromptInjectionDetectionMetric,
    # "toxicity_score": ToxicityScoreMetric,

    # Multi-Turn (deterministic metric)
    "turn_efficiency": TurnEfficiencyMetric,

    # Multi-Agent (deterministic metrics)
    "agent_utilization": AgentUtilizationMetric,
    "coordination_efficiency": CoordinationEfficiencyMetric,

    # Reasoning (LLM-based - separate tests)
    # "chain_of_thought_coherence": ChainOfThoughtCoherenceMetric,
    # "logical_consistency": LogicalConsistencyMetric,
    # "reasoning_step_correctness": ReasoningStepCorrectnessMetric,
    # "fallacy_detection": FallacyDetectionMetric,
}


# Generate test parameters: (metric_name, metric_class, test_case)
def generate_test_params():
    """Generate test parameters for all metrics with test data."""
    params = []
    for metric_name, metric_class in METRIC_CLASSES.items():
        try:
            test_data = load_test_data(metric_name)
            for test_case in test_data["test_cases"]:
                params.append((metric_name, metric_class, test_case))
        except FileNotFoundError:
            # Skip metrics without test data
            pass
    return params


@pytest.mark.parametrize(
    "metric_name,metric_class,test_case",
    generate_test_params(),
    ids=lambda param: f"{param[0]}[{param[2]['name']}]" if isinstance(param, tuple) else str(param)
)
def test_metric(metric_name, metric_class, test_case):
    """
    Unified test for all metrics.
    Parameterized by metric name, metric class, and test case.
    Supports optional 'metric_params' in test case for constructor args.
    """
    # Create metric instance with optional params from test data
    metric_params = test_case.get("metric_params", {})
    metric = metric_class(**metric_params)

    # Create evaluation input from test case
    eval_input = create_evaluation_input(test_case)

    # Calculate metric
    result = metric.calculate(eval_input)

    # Validate result
    validate_metric_result(result, test_case)

    # Additional validation
    assert result.metric_name == metric.get_name(), \
        f"Metric name mismatch: {result.metric_name} != {metric.get_name()}"


# --- LLM-based metrics: deterministic edge cases (no LLM calls) ---
# These metrics normally require LLM judge, but test cases with an explicit
# expected_score hit early-return paths before any API call is made.

LLM_METRIC_CLASSES = {
    # Response Quality
    "answer_relevance": AnswerRelevanceMetric,
    "completeness": CompletenessMetric,
    "hallucination_score": HallucinationScoreMetric,
    "accuracy": AccuracyMetric,
    # Responsible AI
    "safety_score": SafetyScoreMetric,
    "bias_score": BiasScoreMetric,
    "toxicity_score": ToxicityScoreMetric,
    # Reasoning
    "chain_of_thought_coherence": ChainOfThoughtCoherenceMetric,
    "logical_consistency": LogicalConsistencyMetric,
    "reasoning_step_correctness": ReasoningStepCorrectnessMetric,
    "fallacy_detection": FallacyDetectionMetric,
    # Multi-Agent (LLM-based, deterministic edge cases only)
    "delegation_quality": DelegationQualityMetric,
    "workflow_completion": WorkflowCompletionMetric,
    # Multi-Turn (LLM-based, deterministic edge cases only)
    "context_retention": ContextRetentionMetric,
    "coherence": CoherenceMetric,
}


def generate_llm_deterministic_params():
    """Generate params for LLM-metric test cases that have an expected_score."""
    params = []
    for metric_name, metric_class in LLM_METRIC_CLASSES.items():
        try:
            test_data = load_test_data(metric_name)
            for test_case in test_data["test_cases"]:
                if "expected_score" in test_case:
                    params.append((metric_name, metric_class, test_case))
        except FileNotFoundError:
            pass
    return params


@pytest.mark.parametrize(
    "metric_name,metric_class,test_case",
    generate_llm_deterministic_params(),
    ids=lambda param: f"{param[0]}[{param[2]['name']}]" if isinstance(param, tuple) else str(param)
)
def test_llm_metric_deterministic_edge_case(metric_name, metric_class, test_case):
    """
    Test deterministic edge cases for LLM-based response quality metrics.
    These cases return before calling the LLM judge (missing/empty data paths).
    """
    metric_params = test_case.get("metric_params", {})
    metric = metric_class(**metric_params)

    eval_input = create_evaluation_input(test_case)
    result = metric.calculate(eval_input)

    validate_metric_result(result, test_case)

    assert result.metric_name == metric.get_name(), \
        f"Metric name mismatch: {result.metric_name} != {metric.get_name()}"

    # Verify warning metadata is present for None-score edge cases
    if test_case["expected_score"] is None:
        assert "warning" in result.metadata, \
            f"Expected warning in metadata for None-score case '{test_case['name']}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])