# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for metric testing.
Shared functions for loading test data and creating test objects.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import uuid4

from uaef.models import AgentTrace, GroundTruth, MultiAgentTrace, CoordinationEvent, EvaluationInput, Message, ToolCall
from uaef.models.multi_agent_trace import WorkflowStatus


def load_test_data(metric_name: str) -> Dict[str, Any]:
    """
    Load test cases from JSON file for a given metric.

    Args:
        metric_name: Name of the metric (e.g., 'tool_selection_accuracy')

    Returns:
        Dictionary with test_cases list
    """
    test_data_path = Path(__file__).parent / "data" / f"{metric_name}.json"
    with open(test_data_path, 'r') as f:
        return json.load(f)


def create_trace_from_dict(trace_dict: Dict[str, Any]) -> AgentTrace:
    """
    Convert dict to AgentTrace object.

    Args:
        trace_dict: Dictionary containing trace data

    Returns:
        AgentTrace object
    """
    messages = [
        Message(
            role=msg["role"],
            content=msg["content"],
            timestamp=datetime.fromisoformat(msg["timestamp"].replace('Z', '+00:00'))
        )
        for msg in trace_dict.get("messages", [])
    ]

    tool_calls = []
    for tc in trace_dict.get("tool_calls", []):
        timestamp = None
        if tc.get("timestamp"):
            timestamp = datetime.fromisoformat(tc["timestamp"].replace('Z', '+00:00'))

        tool_calls.append(
            ToolCall(
                name=tc["name"],
                arguments=tc.get("arguments", {}),
                timestamp=timestamp,
                error=tc.get("error"),
                result=tc.get("result"),
                execution_time=tc.get("execution_time")
            )
        )

    return AgentTrace(
        trace_id=uuid4(),
        messages=messages,
        tool_calls=tool_calls,
        input_tokens=trace_dict.get("input_tokens"),
        output_tokens=trace_dict.get("output_tokens"),
        latency=trace_dict.get("latency"),
        cost=trace_dict.get("cost"),
        metadata=trace_dict.get("metadata", {})
    )


def create_ground_truth_from_dict(gt_dict: Optional[Dict[str, Any]]) -> Optional[GroundTruth]:
    """
    Convert dict to GroundTruth object.

    Args:
        gt_dict: Dictionary containing ground truth data (can be None)

    Returns:
        GroundTruth object or None
    """
    if not gt_dict:
        return None

    expected_tool_calls = []
    for tc in gt_dict.get("expected_tool_calls", []):
        expected_tool_calls.append(
            ToolCall(
                name=tc["name"],
                arguments=tc["arguments"],
                timestamp=datetime.now(timezone.utc)
            )
        )

    return GroundTruth(
        expected_output=gt_dict.get("expected_output"),
        expected_tool_calls=expected_tool_calls
    )


def create_multi_agent_trace_from_dict(mat_dict: Dict[str, Any]) -> MultiAgentTrace:
    """
    Convert dict to MultiAgentTrace object.

    Args:
        mat_dict: Dictionary containing multi-agent trace data

    Returns:
        MultiAgentTrace object
    """
    # Build agent traces
    agent_traces = {}
    for agent_id, trace_data in mat_dict.get("agent_traces", {}).items():
        agent_traces[agent_id] = create_trace_from_dict(trace_data)

    # Build coordination events
    coordination_events = []
    for event in mat_dict.get("coordination_events", []):
        coordination_events.append(
            CoordinationEvent(
                from_agent=event["from_agent"],
                to_agent=event["to_agent"],
                event_type=event["event_type"],
                message=event["message"],
                timestamp=datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
            )
        )

    # Map workflow status string to enum
    status_str = mat_dict.get("workflow_status", "completed").lower()
    status_map = {
        "completed": WorkflowStatus.COMPLETED,
        "failed": WorkflowStatus.FAILED,
        "partial": WorkflowStatus.PARTIAL,
    }
    workflow_status = status_map.get(status_str, WorkflowStatus.COMPLETED)

    return MultiAgentTrace(
        trace_id=uuid4(),
        agent_traces=agent_traces,
        coordination_events=coordination_events,
        workflow_status=workflow_status,
        metadata=mat_dict.get("metadata", {})
    )


def create_evaluation_input(test_case: Dict[str, Any]) -> EvaluationInput:
    """
    Create EvaluationInput from test case dict.

    Supports both regular traces (test_case["trace"]) and multi-agent traces
    (test_case["multi_agent_trace"]).

    Args:
        test_case: Test case dictionary

    Returns:
        EvaluationInput object
    """
    # Determine trace type
    if "multi_agent_trace" in test_case and test_case["multi_agent_trace"] is not None:
        trace = create_multi_agent_trace_from_dict(test_case["multi_agent_trace"])
    elif "trace" in test_case:
        trace = create_trace_from_dict(test_case["trace"])
    else:
        # Fallback: create empty AgentTrace for edge cases
        trace = AgentTrace(trace_id=uuid4(), messages=[], tool_calls=[])

    ground_truth = create_ground_truth_from_dict(test_case.get("ground_truth"))
    context = test_case.get("context", [])

    return EvaluationInput(
        trace=trace,
        ground_truth=ground_truth,
        context=context
    )


def validate_metric_result(result, test_case: Dict[str, Any]) -> None:
    """
    Validate metric result against expected values.

    Args:
        result: MetricScore object
        test_case: Test case dictionary with expected values

    Raises:
        AssertionError: If validation fails
    """
    # Check expected exact score
    if "expected_score" in test_case:
        expected = test_case["expected_score"]
        if expected is None:
            # Expecting None (metric not applicable)
            assert result.score is None, \
                f"Expected score None (not applicable), got {result.score}"
        else:
            # Expecting numeric score
            assert result.score is not None, \
                f"Expected score {expected}, got None"
            assert abs(result.score - expected) < 0.02, \
                f"Expected score {expected}, got {result.score}"

    # Check expected score range
    elif "expected_score_range" in test_case:
        min_score, max_score = test_case["expected_score_range"]
        assert result.score is not None, \
            f"Expected score in range [{min_score}, {max_score}], got None"
        assert min_score <= result.score <= max_score, \
            f"Score {result.score} not in expected range [{min_score}, {max_score}]"

    # Validate result structure
    assert result.metric_name is not None, "Metric name should be set"
    assert result.reasoning is not None, "Reasoning should be provided"
    assert result.metadata is not None, "Metadata should be present"
