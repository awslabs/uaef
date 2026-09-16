# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool calling metrics for UAEF.

This module implements metrics for evaluating agent tool calling accuracy,
including tool selection, sequence correctness, and parameter quality.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from uaef.metrics.base import BaseMetric
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore
from uaef.models.tool_call import ToolCall


def _compare_params(expected: Dict[str, Any], actual: Dict[str, Any]) -> float:
    """
    Compare two parameter dictionaries using key-value pair matching.

    This function respects parameter order/binding and types (detects swapped arguments
    and type mismatches).

    Args:
        expected: Expected parameters as key-value pairs
        actual: Actual parameters as key-value pairs

    Returns:
        Match score between 0.0 and 1.0
    """
    if not expected and not actual:
        return 1.0

    if not expected or not actual:
        return 0.0

    # Get all unique keys from both dictionaries
    all_keys = set(expected.keys()) | set(actual.keys())

    if len(all_keys) == 0:
        return 1.0

    # Count matching key-value pairs with type-aware comparison
    matched_keys = 0
    for key in expected.keys():
        if key in actual:
            expected_val = expected[key]
            actual_val = actual[key]

            # Direct comparison (respects types)
            # For nested structures, this will compare recursively
            if expected_val == actual_val:
                matched_keys += 1

    # Score = matched keys / total unique keys
    return matched_keys / len(all_keys)


def _calculate_tool_name_jaccard(
    expected_calls: List[ToolCall],
    actual_calls: List[ToolCall]
) -> Dict[str, Any]:
    """
    Calculate tool selection accuracy using Jaccard similarity on tool names only.

    Treats tool calls as a multiset (bag) where duplicates matter.
    Uses greedy matching to find maximum overlap, then computes:
    Jaccard = matched / (total_expected + total_actual - matched)

    Args:
        expected_calls: List of expected tool calls
        actual_calls: List of actual tool calls

    Returns:
        Dictionary with score and detailed metrics:
        - score: Jaccard similarity score between 0.0 and 1.0
        - matched_count: Number of matched tool names
        - expected_count: Total expected calls
        - actual_count: Total actual calls
        - missing_count: Expected calls not matched
        - extra_count: Actual calls not matched
    """
    expected_count = len(expected_calls)
    actual_count = len(actual_calls)

    # Handle empty cases
    if expected_count == 0 and actual_count == 0:
        return {
            "score": 1.0,
            "matched_count": 0,
            "expected_count": 0,
            "actual_count": 0,
            "missing_count": 0,
            "extra_count": 0
        }

    if expected_count == 0 or actual_count == 0:
        return {
            "score": 0.0,
            "matched_count": 0,
            "expected_count": expected_count,
            "actual_count": actual_count,
            "missing_count": expected_count,
            "extra_count": actual_count
        }

    # Extract tool names
    expected_names = [tc.name for tc in expected_calls]
    actual_names = [tc.name for tc in actual_calls]

    # Use Counter for multiset matching
    expected_counter = Counter(expected_names)
    actual_counter = Counter(actual_names)

    # Calculate intersection (matched calls)
    matched_count = 0
    for tool_name in expected_counter:
        matched_count += min(expected_counter[tool_name], actual_counter.get(tool_name, 0))

    # Jaccard similarity for multisets
    # matched / (total_expected + total_actual - matched)
    union_size = expected_count + actual_count - matched_count
    score = matched_count / union_size if union_size > 0 else 0.0

    missing_count = expected_count - matched_count
    extra_count = actual_count - matched_count

    return {
        "score": score,
        "matched_count": matched_count,
        "expected_count": expected_count,
        "actual_count": actual_count,
        "missing_count": missing_count,
        "extra_count": extra_count
    }


class ToolSelectionAccuracyMetric(BaseMetric):
    """
    Metric for evaluating tool selection accuracy using Jaccard similarity.

    Evaluates tool names only (ignores parameters). Uses multiset/bag semantics
    where duplicate tool calls are meaningful. Computes Jaccard similarity:
    matched / (expected + actual - matched)
    """

    def get_name(self) -> str:
        """Get metric name."""
        return "tool_selection_accuracy"

    def requires_ground_truth(self) -> bool:
        """This metric requires ground truth."""
        return True

    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False

    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates how accurately the agent selects tools based on tool names (Jaccard similarity)"

    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Tool Calling"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate tool selection accuracy using Jaccard similarity."""
        # Validate ground truth is provided
        if not evaluation_input.ground_truth:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no ground truth provided",
                metadata={"warning": "missing_data"}
            )

        expected_tool_calls = evaluation_input.ground_truth.expected_tool_calls

        if not expected_tool_calls:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no expected tool calls provided in ground truth",
                metadata={"warning": "missing_data"}
            )

        # Extract tool calls from trace
        trace = evaluation_input.trace
        actual_tool_calls = trace.tool_calls

        # Calculate Jaccard similarity on tool names
        result = _calculate_tool_name_jaccard(expected_tool_calls, actual_tool_calls)
        score = result["score"]

        # Generate reasoning
        if score == 1.0:
            reasoning = "Perfect tool selection"
        elif score == 0.0:
            if result["expected_count"] == 0:
                reasoning = f"Called {result['actual_count']} unexpected tools"
            elif result["actual_count"] == 0:
                reasoning = f"Expected {result['expected_count']} tools but called none"
            else:
                reasoning = "No matching tools"
        else:
            parts = []
            if result["matched_count"] > 0:
                parts.append(f"{result['matched_count']} matched")
            if result["missing_count"] > 0:
                parts.append(f"{result['missing_count']} missing")
            if result["extra_count"] > 0:
                parts.append(f"{result['extra_count']} extra")
            reasoning = f"Jaccard {score:.2f}: " + ", ".join(parts)

        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "expected_count": result["expected_count"],
                "actual_count": result["actual_count"],
                "matched_count": result["matched_count"],
                "missing_count": result["missing_count"],
                "extra_count": result["extra_count"],
                "jaccard_score": score
            }
        )

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)


class ToolSequenceCorrectnessMetric(BaseMetric):
    """
    Metric for evaluating tool call sequence correctness.

    Validates that tools are called in the correct order compared to ground truth.
    Uses sequence similarity scoring.
    """

    def get_name(self) -> str:
        """Get metric name."""
        return "tool_sequence_correctness"

    def requires_ground_truth(self) -> bool:
        """This metric requires ground truth."""
        return True

    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False

    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates whether tools are called in the correct order"

    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Tool Calling"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate tool sequence correctness."""
        # Validate ground truth is provided
        if evaluation_input.ground_truth is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no ground truth provided",
                metadata={"warning": "missing_data"}
            )

        # Extract tool sequences
        trace = evaluation_input.trace
        called_sequence = [tc.name for tc in trace.tool_calls]

        gt = evaluation_input.ground_truth
        expected_sequence = [tc.name for tc in gt.expected_tool_calls]

        if not expected_sequence:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no expected tool calls provided in ground truth",
                metadata={"warning": "missing_data"}
            )

        # Handle empty cases
        if len(expected_sequence) == 0 and len(called_sequence) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=1.0,
                reasoning="No tool sequence expected and no tools called",
                metadata={"expected_length": 0, "called_length": 0}
            )

        if len(expected_sequence) == 0 or len(called_sequence) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=0.0,
                reasoning="Empty sequence mismatch",
                metadata={"expected_length": len(expected_sequence), "called_length": len(called_sequence)}
            )

        # Calculate sequence similarity using longest common subsequence
        def lcs_length(seq1: List[str], seq2: List[str]) -> int:
            """Calculate longest common subsequence length."""
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i - 1] == seq2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[m][n]

        lcs_len = lcs_length(expected_sequence, called_sequence)
        max_len = max(len(expected_sequence), len(called_sequence))

        # Score based on LCS ratio
        score = lcs_len / max_len if max_len > 0 else 0.0

        # Check for exact match
        exact_match = called_sequence == expected_sequence

        reasoning = (
            "Exact sequence match" if exact_match
            else f"Sequence similarity: {score:.1%} (LCS: {lcs_len}/{max_len})"
        )

        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "expected_sequence": expected_sequence,
                "called_sequence": called_sequence,
                "lcs_length": lcs_len,
                "exact_match": exact_match
            }
        )

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)


class ParameterQualityMetric(BaseMetric):
    """
    Metric for evaluating tool parameter/argument quality.

    Uses per-call key-value pair matching to evaluate parameter correctness.
    Detects swapped arguments and incorrect parameter values.
    """

    def get_name(self) -> str:
        """Get metric name."""
        return "parameter_quality"

    def requires_ground_truth(self) -> bool:
        """This metric requires ground truth."""
        return True

    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False

    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates the quality and correctness of tool arguments"

    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Tool Calling"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate parameter quality."""
        # Validate ground truth is provided
        if not evaluation_input.ground_truth:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no ground truth provided",
                metadata={"warning": "missing_data"}
            )

        expected_tool_calls = evaluation_input.ground_truth.expected_tool_calls

        if not expected_tool_calls:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no expected tool calls provided in ground truth",
                metadata={"warning": "missing_data"}
            )

        # Extract tool calls
        trace = evaluation_input.trace
        actual_tool_calls = trace.tool_calls

        # Handle empty cases
        if len(expected_tool_calls) == 0 and len(actual_tool_calls) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=1.0,
                reasoning="No parameters to evaluate",
                metadata={"expected_count": 0, "actual_count": 0}
            )

        if len(expected_tool_calls) == 0 or len(actual_tool_calls) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=0.0,
                reasoning="Cannot evaluate parameters with empty tool lists",
                metadata={"expected_count": len(expected_tool_calls), "actual_count": len(actual_tool_calls)}
            )

        # Calculate parameter quality using partial matching
        # For each expected call, find best matching actual call and get partial param score
        matched_actual_indices = set()
        param_scores = []

        for expected in expected_tool_calls:
            best_param_score = 0.0
            best_idx = -1

            for idx, actual in enumerate(actual_tool_calls):
                if idx in matched_actual_indices:
                    continue

                # Only evaluate parameters if tool names match
                if expected.name == actual.name:
                    param_score = _compare_params(expected.arguments, actual.arguments)

                    if param_score > best_param_score:
                        best_param_score = param_score
                        best_idx = idx

            # Add the best param score (even if 0.0)
            param_scores.append(best_param_score)

            if best_idx >= 0:
                matched_actual_indices.add(best_idx)

        # Calculate average parameter quality
        base_score = sum(param_scores) / len(param_scores) if param_scores else 0.0

        # Apply penalty for extra calls
        extra_calls = len(actual_tool_calls) - len(matched_actual_indices)
        if extra_calls > 0:
            # Penalize proportionally
            penalty_factor = len(expected_tool_calls) / len(actual_tool_calls)
            score = base_score * penalty_factor
        else:
            score = base_score

        # Generate reasoning
        if score == 1.0:
            reasoning = "Perfect parameter quality"
        else:
            reasoning = f"Parameter quality: {score:.1%}"
            if extra_calls > 0:
                reasoning += f" ({extra_calls} extra calls)"

        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "expected_count": len(expected_tool_calls),
                "actual_count": len(actual_tool_calls),
                "matched_count": len(matched_actual_indices),
                "extra_calls": extra_calls,
                "parameter_score": score
            }
        )

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)


class MCPComplianceMetric(BaseMetric):
    """
    Metric for evaluating Model Context Protocol (MCP) compliance.

    Checks if tool calls follow MCP standards and conventions.
    """

    def get_name(self) -> str:
        """Get metric name."""
        return "mcp_compliance"

    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False

    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False

    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates compliance with Model Context Protocol standards"

    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Tool Calling"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate MCP compliance."""
        trace = evaluation_input.trace
        tool_calls = trace.tool_calls

        if len(tool_calls) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no tool calls in trace",
                metadata={"warning": "not_applicable"}
            )

        # Check if MCP is actually being used by looking for MCP-specific indicators
        has_mcp_indicators = False
        for tc in tool_calls:
            # Check for MCP-specific patterns in tool names or metadata
            if tc.name and ("mcp" in tc.name.lower() or tc.name.startswith("mcp_")):
                has_mcp_indicators = True
                break
            # Check for MCP URIs in arguments
            if isinstance(tc.arguments, dict):
                for val in tc.arguments.values():
                    if isinstance(val, str) and ("mcp://" in val or "mcp:" in val):
                        has_mcp_indicators = True
                        break
            if has_mcp_indicators:
                break

        # Check trace metadata for MCP indicators
        if not has_mcp_indicators and trace.metadata:
            metadata = trace.metadata
            if isinstance(metadata, dict):
                metadata_str = str(metadata).lower()
                if "mcp" in metadata_str or "model_context_protocol" in metadata_str:
                    has_mcp_indicators = True

        if not has_mcp_indicators:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Not applicable: no MCP (Model Context Protocol) usage detected in trace",
                metadata={"warning": "not_applicable", "reason": "no_mcp_indicators"}
            )

        # Check MCP compliance criteria
        violations = []
        compliant_count = 0

        for tc in tool_calls:
            tool_violations = []

            # Check 1: Tool name should not be empty
            if not tc.name or not tc.name.strip():
                tool_violations.append("empty_name")

            # Check 2: Arguments should be a dictionary
            if not isinstance(tc.arguments, dict):
                tool_violations.append("invalid_arguments_type")

            # Check 3: Timestamp should be present
            if not tc.timestamp:
                tool_violations.append("missing_timestamp")

            # Check 4: Tool name should follow naming conventions (lowercase, underscores)
            if tc.name and not tc.name.islower():
                tool_violations.append("invalid_naming_convention")

            if not tool_violations:
                compliant_count += 1
            else:
                violations.extend(tool_violations)

        # Calculate compliance score
        score = compliant_count / len(tool_calls) if len(tool_calls) > 0 else 1.0

        # Generate reasoning
        if score == 1.0:
            reasoning = "All tool calls are MCP compliant"
        else:
            violation_summary = Counter(violations)
            top_violations = violation_summary.most_common(3)
            reasoning = f"MCP compliance: {score:.1%}. Violations: " + ", ".join(
                f"{v}({c})" for v, c in top_violations
            )

        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "total_tools": len(tool_calls),
                "compliant_tools": compliant_count,
                "violations": dict(Counter(violations))
            }
        )

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)
