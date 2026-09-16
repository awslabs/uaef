#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to run LLM-based metrics with actual API calls.
Results saved to tests/llm_metrics_results.json for manual review.

This is NOT a pytest test - it makes real Bedrock API calls.
Use this to validate LLM-based metrics with real data.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from uaef.metrics.response_quality import (
    AnswerRelevanceMetric,
    HallucinationScoreMetric,
    AccuracyMetric,
    CompletenessMetric
)
from uaef.metrics.responsible_ai import (
    SafetyScoreMetric,
    BiasScoreMetric,
    ToxicityScoreMetric
)
from uaef.metrics.multi_turn import (
    ContextRetentionMetric,
    CoherenceMetric
)
from uaef.metrics.reasoning import (
    ChainOfThoughtCoherenceMetric,
    LogicalConsistencyMetric,
    ReasoningStepCorrectnessMetric,
    FallacyDetectionMetric
)
from uaef.metrics.multi_agent import (
    DelegationQualityMetric,
    WorkflowCompletionMetric
)
from uaef.models import AgentTrace, GroundTruth, EvaluationInput, Message, ToolCall
from tests.test_utils import create_evaluation_input as build_evaluation_input


def load_test_data(metric_name):
    """Load test data for specific metric."""
    test_data_path = Path(__file__).parent / "data" / f"{metric_name}.json"
    if not test_data_path.exists():
        return None
    with open(test_data_path, 'r') as f:
        return json.load(f)


def create_trace_from_dict(trace_dict):
    """Convert dict to AgentTrace."""
    messages = [
        Message(
            role=msg["role"],
            content=msg["content"],
            timestamp=datetime.fromisoformat(msg["timestamp"].replace('Z', '+00:00'))
        )
        for msg in trace_dict["messages"]
    ]

    tool_calls = [
        ToolCall(
            name=tc["name"],
            arguments=tc["arguments"],
            timestamp=datetime.fromisoformat(tc["timestamp"].replace('Z', '+00:00'))
        )
        for tc in trace_dict.get("tool_calls", [])
    ]

    return AgentTrace(
        trace_id=uuid4(),
        messages=messages,
        tool_calls=tool_calls
    )


def create_ground_truth_from_dict(gt_dict):
    """Convert dict to GroundTruth."""
    if not gt_dict:
        return None

    return GroundTruth(
        expected_output=gt_dict.get("expected_output"),
        expected_tool_calls=[]
    )


def run_metric_tests(metric_name, metric_class):
    """
    Run a metric on all test cases and collect results.

    Args:
        metric_name: Name of the metric (used to load JSON file)
        metric_class: Metric class to instantiate

    Returns:
        Dict with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {metric_name}")
    print(f"{'='*80}")

    # Load test data
    test_data = load_test_data(metric_name)
    if not test_data:
        print(f"  ⚠️  No test data file found: {metric_name}.json")
        return {"error": "No test data file", "test_count": 0}

    test_cases = test_data["test_cases"]
    metric = metric_class()
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['name']}: {test_case['description']}")

        try:
            # Build inputs using shared utility
            eval_input = build_evaluation_input(test_case)

            # Calculate metric
            print(f"  Calling {metric_name}...")
            result = metric.calculate(eval_input)

            print(f"  ✓ Score: {result.score}")
            if result.reasoning:
                reasoning_preview = result.reasoning[:150] + "..." if len(result.reasoning) > 150 else result.reasoning
                print(f"  ✓ Reasoning: {reasoning_preview}")

            # Save result
            trace_data = test_case.get("trace") or test_case.get("multi_agent_trace")
            results.append({
                "test_case_name": test_case["name"],
                "description": test_case["description"],
                "scenario_type": test_case["scenario_type"],
                "trace": trace_data,
                "score": result.score,
                "reasoning": result.reasoning,
                "metadata": result.metadata,
                "success": True
            })

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            trace_data = test_case.get("trace") or test_case.get("multi_agent_trace")
            results.append({
                "test_case_name": test_case["name"],
                "description": test_case["description"],
                "scenario_type": test_case["scenario_type"],
                "trace": trace_data,
                "error": str(e),
                "success": False
            })

    return {
        "test_count": len(test_cases),
        "success_count": sum(1 for r in results if r["success"]),
        "failure_count": sum(1 for r in results if not r["success"]),
        "results": results
    }


def main():
    """Run all LLM-based metrics and save results."""
    print("="*80)
    print("LLM-Based Metrics Validation Script")
    print("="*80)
    print("\n⚠️  This script makes ACTUAL Bedrock API calls and costs money!")
    print("Ensure AWS credentials are configured.")
    print("\nResults will be saved to: tests/output/")
    print("="*80)

    input("\nPress Enter to continue or Ctrl+C to cancel...")

    # All available LLM metrics (metric_name must match JSON filename)
    all_llm_metrics = {
        "answer_relevance": AnswerRelevanceMetric,
        "completeness": CompletenessMetric,
        "hallucination_score": HallucinationScoreMetric,
        "accuracy": AccuracyMetric,
        "safety_score": SafetyScoreMetric,
        "bias_score": BiasScoreMetric,
        "toxicity_score": ToxicityScoreMetric,
        "context_retention": ContextRetentionMetric,
        "coherence": CoherenceMetric,
        "chain_of_thought_coherence": ChainOfThoughtCoherenceMetric,
        "logical_consistency": LogicalConsistencyMetric,
        "reasoning_step_correctness": ReasoningStepCorrectnessMetric,
        "fallacy_detection": FallacyDetectionMetric,
        "delegation_quality": DelegationQualityMetric,
        "workflow_completion": WorkflowCompletionMetric,
    }

    # Metric groups for --group filter
    metric_groups = {
        "quality": ["answer_relevance", "completeness", "hallucination_score", "accuracy"],
        "responsible_ai": ["safety_score", "bias_score", "toxicity_score"],
        "multi_turn": ["context_retention", "coherence"],
        "reasoning": ["chain_of_thought_coherence", "logical_consistency", "reasoning_step_correctness", "fallacy_detection"],
        "multi_agent": ["delegation_quality", "workflow_completion"],
    }

    # Filter metrics based on CLI args
    filter_args = sys.argv[1:]
    if filter_args:
        selected = {}
        for arg in filter_args:
            arg_clean = arg.lstrip("-")
            if arg_clean in metric_groups:
                for name in metric_groups[arg_clean]:
                    selected[name] = all_llm_metrics[name]
            elif arg_clean in all_llm_metrics:
                selected[arg_clean] = all_llm_metrics[arg_clean]
            else:
                print(f"Unknown metric or group: {arg_clean}")
                print(f"  Metrics: {', '.join(all_llm_metrics.keys())}")
                print(f"  Groups:  {', '.join(metric_groups.keys())}")
                sys.exit(1)
        llm_metrics = selected
    else:
        llm_metrics = all_llm_metrics

    # Build output filename with filter and timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if filter_args:
        filter_label = "_".join(arg.lstrip("-") for arg in filter_args)
        output_filename = f"llm_metrics_results_{filter_label}_{timestamp}.json"
    else:
        output_filename = f"llm_metrics_results_{timestamp}.json"

    # Collect all results
    all_results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "filter": filter_args if filter_args else "all",
        "metrics": {}
    }

    for metric_name, metric_class in llm_metrics.items():
        try:
            metric_results = run_metric_tests(metric_name, metric_class)
            all_results["metrics"][metric_name] = metric_results
        except Exception as e:
            print(f"\n✗ Failed to run {metric_name}: {str(e)}")
            all_results["metrics"][metric_name] = {
                "error": str(e),
                "test_count": 0
            }

    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*80}")

    # Print summary
    print("\nSummary:")
    total_tests = 0
    total_success = 0
    for metric_name, data in all_results["metrics"].items():
        if "error" in data and data.get("test_count") == 0:
            print(f"  {metric_name}: ERROR - {data['error']}")
        else:
            success = data.get("success_count", 0)
            total = data.get("test_count", 0)
            failures = data.get("failure_count", 0)
            total_tests += total
            total_success += success
            status = "✓" if failures == 0 else "⚠️"
            print(f"  {status} {metric_name}: {success}/{total} successful")

    print(f"\n{'='*80}")
    print(f"Total: {total_success}/{total_tests} tests successful")
    print(f"{'='*80}")
    print(f"\n📄 Review detailed results in: {output_path}")


if __name__ == "__main__":
    main()
