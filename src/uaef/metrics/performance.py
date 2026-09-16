# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance metrics for UAEF.

This module implements metrics for evaluating agent performance including
latency, token efficiency, cost efficiency, and throughput.

All metrics in this module are deterministic and do not require LLM judge.
"""

from typing import Optional

from uaef.metrics.base import BaseMetric
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore


class LatencyScoreMetric(BaseMetric):
    """
    Metric for evaluating agent latency using threshold-based scoring.
    
    Scores based on how quickly the agent responds. Lower latency = higher score.
    This is a deterministic metric that doesn't require ground truth or LLM judge.
    """
    
    def __init__(self, threshold_ms: float = 2000.0):
        """
        Initialize latency metric.
        
        Args:
            threshold_ms: Latency threshold in milliseconds. Responses faster than
                         this get score 1.0, slower responses get proportionally lower scores.
        """
        self.threshold_ms = threshold_ms
    
    def get_name(self) -> str:
        """Get metric name."""
        return "latency_score"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates agent response latency with threshold-based scoring"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Performance"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate latency score."""
        trace = evaluation_input.trace

        # AgentTrace uses `latency`; MultiAgentTrace uses `total_latency`; both in seconds
        raw_latency = getattr(trace, "latency", None)
        if raw_latency is None:
            raw_latency = getattr(trace, "total_latency", None)
        
        latency_ms = raw_latency * 1000 if raw_latency is not None else None

        if latency_ms is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no latency information in trace",
                metadata={"warning": "missing_data"}
            )
        
        # Validate latency is non-negative
        if latency_ms < 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Invalid negative latency: {latency_ms}ms",
                metadata={"warning": "invalid_data", "latency_ms": latency_ms}
            )
        
        # Calculate score based on threshold
        # Score = 1.0 if latency <= threshold
        # Score decreases linearly as latency increases beyond threshold
        if latency_ms <= self.threshold_ms:
            score = 1.0
            reasoning = f"Latency {latency_ms:.0f}ms is within threshold {self.threshold_ms:.0f}ms"
        else:
            # Linear decay: score = threshold / latency
            score = self.threshold_ms / latency_ms
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            reasoning = f"Latency {latency_ms:.0f}ms exceeds threshold {self.threshold_ms:.0f}ms"
        
        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "latency_ms": latency_ms,
                "threshold_ms": self.threshold_ms,
                "within_threshold": latency_ms <= self.threshold_ms
            }
        )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)


class TokenEfficiencyMetric(BaseMetric):
    """
    Metric for evaluating token efficiency (quality per token).

    Calculates efficiency based on quality score and token usage.
    REQUIRES quality_score to be provided via trace.metadata.

    Formula:
    - If tokens <= expected: efficiency = quality (flat, no penalty)
    - If tokens > expected: efficiency = quality / (ratio ** penalty_exponent)
      where ratio = actual_tokens / expected_tokens

    This is a deterministic metric.
    """

    def __init__(self, expected_tokens: int = 1000, penalty_mode: str = "light"):
        """
        Initialize token efficiency metric.

        Args:
            expected_tokens: Expected/ideal token count. No penalty if under this.
            penalty_mode: "light" (sqrt penalty, 0.5 exponent) or "heavy" (linear, 1.0 exponent)
        """
        self.expected_tokens = expected_tokens
        self.penalty_mode = penalty_mode
        self.penalty_exponent = 0.5 if penalty_mode == "light" else 1.0

    def get_name(self) -> str:
        """Get metric name."""
        return "token_efficiency"

    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False

    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False

    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates token efficiency as quality relative to token usage"

    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Performance"

    def get_dependencies(self) -> dict:
        return {
            "requires_metadata": ["quality_score"],
        }

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate token efficiency."""
        trace = evaluation_input.trace

        # Get token counts from trace
        input_tokens = trace.input_tokens or 0
        output_tokens = trace.output_tokens or 0
        total_tokens = input_tokens + output_tokens

        if total_tokens == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no token information in trace",
                metadata={"warning": "missing_data"}
            )

        # REQUIRE quality score from metadata (0.0 is valid, only None is invalid)
        quality_score = trace.metadata.get("quality_score")
        if quality_score is None or not isinstance(quality_score, (int, float)):
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: quality_score required (calculate quality metrics first and store in trace.metadata['quality_score'])",
                metadata={"warning": "missing_data", "required_field": "quality_score"}
            )

        # Calculate efficiency with threshold and penalty
        if total_tokens <= self.expected_tokens:
            # Within expected: no penalty
            efficiency = quality_score
            reasoning = f"Quality {quality_score:.2f} with {total_tokens} tokens (within expected {self.expected_tokens})"
        else:
            # Exceeds expected: apply penalty
            ratio = total_tokens / self.expected_tokens
            efficiency = quality_score / (ratio ** self.penalty_exponent)
            efficiency = max(0.0, min(1.0, efficiency))
            reasoning = f"Quality {quality_score:.2f} with {total_tokens} tokens (exceeds expected {self.expected_tokens}, {self.penalty_mode} penalty applied)"

        return MetricScore(
            metric_name=self.get_name(),
            score=efficiency,
            reasoning=reasoning,
            metadata={
                "total_tokens": total_tokens,
                "expected_tokens": self.expected_tokens,
                "quality_score": quality_score,
                "penalty_mode": self.penalty_mode,
                "token_ratio": total_tokens / self.expected_tokens
            }
        )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)


class CostEfficiencyMetric(BaseMetric):
    """
    Metric for evaluating cost efficiency (quality per dollar).

    Calculates efficiency based on quality score and cost.
    REQUIRES quality_score to be provided via trace.metadata.

    Formula:
    - If cost <= expected: efficiency = quality (flat, no penalty)
    - If cost > expected: efficiency = quality / (ratio ** penalty_exponent)
      where ratio = actual_cost / expected_cost

    This is a deterministic metric.
    """

    def __init__(self, expected_cost_usd: float = 0.01, penalty_mode: str = "light"):
        """
        Initialize cost efficiency metric.

        Args:
            expected_cost_usd: Expected/ideal cost in USD. No penalty if under this.
            penalty_mode: "light" (sqrt penalty, 0.5 exponent) or "heavy" (linear, 1.0 exponent)
        """
        self.expected_cost_usd = expected_cost_usd
        self.penalty_mode = penalty_mode
        self.penalty_exponent = 0.5 if penalty_mode == "light" else 1.0

    def get_name(self) -> str:
        """Get metric name."""
        return "cost_efficiency"

    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False

    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False

    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates cost efficiency as quality relative to cost"

    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Performance"

    def get_dependencies(self) -> dict:
        """Get metric dependencies."""
        return {
            "requires_metadata": ["quality_score"],
            "requires_metrics": ["answer_relevance", "completeness", "accuracy", "hallucination_score"]
        }

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate cost efficiency."""
        trace = evaluation_input.trace

        # Get cost from trace metadata
        cost_usd = trace.metadata.get("cost_usd")

        if cost_usd is None or cost_usd == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no cost information in trace",
                metadata={"warning": "missing_data"}
            )

        # Validate cost is non-negative
        if cost_usd < 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=0.0,
                reasoning=f"Invalid negative cost: ${cost_usd}",
                metadata={"error": "invalid_cost", "cost_usd": cost_usd}
            )

        # REQUIRE quality score from metadata (0.0 is valid, only None is invalid)
        quality_score = trace.metadata.get("quality_score")
        if quality_score is None or not isinstance(quality_score, (int, float)):
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: quality_score required (calculate quality metrics first and store in trace.metadata['quality_score'])",
                metadata={"warning": "missing_data", "required_field": "quality_score"}
            )

        # Calculate efficiency with threshold and penalty
        if cost_usd <= self.expected_cost_usd:
            # Within expected: no penalty
            efficiency = quality_score
            reasoning = f"Quality {quality_score:.2f} at cost ${cost_usd:.4f} (within expected ${self.expected_cost_usd:.4f})"
        else:
            # Exceeds expected: apply penalty
            ratio = cost_usd / self.expected_cost_usd
            efficiency = quality_score / (ratio ** self.penalty_exponent)
            efficiency = max(0.0, min(1.0, efficiency))
            reasoning = f"Quality {quality_score:.2f} at cost ${cost_usd:.4f} (exceeds expected ${self.expected_cost_usd:.4f}, {self.penalty_mode} penalty applied)"

        return MetricScore(
            metric_name=self.get_name(),
            score=efficiency,
            reasoning=reasoning,
            metadata={
                "cost_usd": cost_usd,
                "expected_cost_usd": self.expected_cost_usd,
                "quality_score": quality_score,
                "penalty_mode": self.penalty_mode,
                "cost_ratio": cost_usd / self.expected_cost_usd
            }
        )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)


class ThroughputMetric(BaseMetric):
    """
    Metric for evaluating agent throughput (requests per second).
    
    Measures how many requests the agent can handle per unit time.
    This is typically calculated at the batch level rather than per-request.
    This is a deterministic metric.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "throughput"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric does not require LLM judge."""
        return False
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates agent throughput in requests per second"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Performance"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate throughput."""
        trace = evaluation_input.trace
        
        # Get throughput from trace metadata (typically set at batch level)
        throughput_rps = trace.metadata.get("throughput_rps")
        
        if throughput_rps is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no throughput information in trace",
                metadata={"warning": "missing_data"}
            )
        
        # Validate throughput is non-negative
        if throughput_rps < 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=0.0,
                reasoning=f"Invalid negative throughput: {throughput_rps} rps",
                metadata={"error": "invalid_throughput", "throughput_rps": throughput_rps}
            )
        
        # Calculate score based on throughput
        # Assume target throughput is 10 requests per second
        target_rps = 10.0
        
        if throughput_rps >= target_rps:
            score = 1.0
            reasoning = f"Throughput {throughput_rps:.2f} rps meets or exceeds target {target_rps:.2f} rps"
        else:
            # Linear scaling: score = throughput / target
            score = throughput_rps / target_rps
            score = max(0.0, min(1.0, score))
            reasoning = f"Throughput {throughput_rps:.2f} rps is below target {target_rps:.2f} rps"
        
        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "throughput_rps": throughput_rps,
                "target_rps": target_rps,
                "meets_target": throughput_rps >= target_rps
            }
        )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Async version (just calls synchronous version for deterministic metric)."""
        return self.calculate(evaluation_input)
