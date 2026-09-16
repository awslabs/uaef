# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-agent evaluator for UAEF.

This module provides evaluation functionality for multi-agent traces,
coordinating metric calculation for multi-agent coordination scenarios.
"""

from typing import Dict, List, Optional

from uaef.evaluation.base_evaluator import BaseEvaluator
from uaef.logging import get_logger
from uaef.models.dimension_result import DimensionResult
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.ground_truth import GroundTruth
from uaef.models.metric_score import MetricScore
from uaef.models.multi_agent_trace import MultiAgentTrace

logger = get_logger(__name__)


class MultiAgentEvaluator(BaseEvaluator):
    """
    Evaluator for multi-agent traces.

    Coordinates the complete evaluation flow for multi-agent coordination:
    1. Create EvaluationInput from multi-agent trace and ground truth
    2. Calculate metrics for each dimension (including multi-agent specific)
    3. Aggregate dimension scores with weighted averages
    4. Calculate overall score
    5. Check thresholds and identify failures
    """

    def __init__(
        self,
        metric_registry: Optional[any] = None,
        max_workers: int = 4,
        default_dimension_weights: Optional[Dict[str, float]] = None,
        default_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multi-agent evaluator.

        Args:
            metric_registry: MetricRegistry instance (uses global if None)
            max_workers: Maximum number of parallel workers for metric calculation
            default_dimension_weights: Default weights for dimensions
            default_thresholds: Default threshold values for dimensions
        """
        super().__init__(metric_registry=metric_registry, max_workers=max_workers)

        # Set default dimension weights if not provided
        self.default_dimension_weights = default_dimension_weights or {
            "tool_calling": 0.20,
            "response_quality": 0.20,
            "responsible_ai": 0.15,
            "performance": 0.10,
            "multi_turn": 0.10,
            "multi_agent": 0.20,  # Higher weight for multi-agent scenarios
            "reasoning": 0.05
        }

        # Set default thresholds if not provided
        self.default_thresholds = default_thresholds or {
            "tool_calling": 0.7,
            "response_quality": 0.7,
            "responsible_ai": 0.8,
            "performance": 0.6,
            "multi_turn": 0.6,
            "multi_agent": 0.7,
            "reasoning": 0.6
        }

        logger.info(
            f"Initialized MultiAgentEvaluator with "
            f"{len(self.default_dimension_weights)} dimensions"
        )

    def evaluate(
        self,
        trace: MultiAgentTrace,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a multi-agent trace.

        Args:
            trace: Multi-agent trace to evaluate
            ground_truth: Expected correct outputs (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names.
                       If None, uses all available metrics.
            dimension_weights: Weights for each dimension (uses defaults if None)
            thresholds: Threshold values for pass/fail (uses defaults if None)
            context: List of context documents (optional)
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with scores and metadata

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If evaluation fails
        """
        logger.info(f"Evaluating multi-agent trace {trace.trace_id}")

        # Validate inputs
        if not isinstance(trace, MultiAgentTrace):
            raise ValueError("trace must be a MultiAgentTrace instance")

        # Use default weights and thresholds if not provided
        dimension_weights = dimension_weights or self.default_dimension_weights
        thresholds = thresholds or self.default_thresholds

        # Create evaluation input
        evaluation_input = EvaluationInput(
            trace=trace,
            ground_truth=ground_truth,
            context=context or (ground_truth.context_documents if ground_truth else [])
        )

        # Determine which metrics to calculate
        metrics_to_calculate = self._get_metrics_to_calculate(metric_set)

        logger.debug(f"Calculating {len(metrics_to_calculate)} metrics")

        # Calculate metrics with automatic dependency resolution
        metric_scores = self.calculate_metrics_with_dependencies(
            metrics=metrics_to_calculate,
            evaluation_input=evaluation_input
        )

        logger.debug(f"Calculated {len(metric_scores)} metric scores")

        # Aggregate scores by dimension
        dimension_results = self.aggregate_dimension_scores(
            metric_scores=metric_scores,
            dimension_weights=dimension_weights
        )

        logger.debug(f"Aggregated scores for {len(dimension_results)} dimensions")

        # Calculate overall score
        overall_score = self.calculate_overall_score(dimension_results)

        logger.debug(f"Overall score: {overall_score:.3f}")

        # Check thresholds
        passed, failures = self.check_thresholds(
            dimension_results=dimension_results,
            thresholds=thresholds
        )

        # Collect warnings
        warnings = []
        if ground_truth is None:
            gt_metrics = self.metric_registry.get_metrics_requiring_ground_truth()
            if any(m.metric_name in gt_metrics for m in metric_scores):
                warnings.append(
                    "Ground truth not provided, some metrics may be skipped"
                )

        # Create evaluation result
        result = EvaluationResult(
            trace_id=trace.trace_id,
            dimension_results=dimension_results,
            overall_score=overall_score,
            passed=passed,
            failures=failures,
            warnings=warnings,
            metadata={
                "num_metrics": len(metric_scores),
                "num_dimensions": len(dimension_results),
                "num_agents": len(trace.agent_traces),
                "workflow_status": trace.workflow_status.value,
                "dimension_weights": dimension_weights,
                "thresholds": thresholds
            }
        )

        logger.info(
            f"Evaluation complete for trace {trace.trace_id}: "
            f"score={overall_score:.3f}, passed={passed}"
        )

        return result

    def evaluate_multi_turn(
        self,
        per_turn_traces: List[MultiAgentTrace],
        full_trace: MultiAgentTrace,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate a multi-agent multi-turn conversation from pre-split per-turn traces.

        Both per_turn_traces and full_trace must be MultiAgentTrace instances.
        Single-agent (AgentTrace) inputs are handled by
        SingleAgentEvaluator.evaluate_multi_turn().

        Per-turn metrics (tool calling, response quality, multi-agent, ...)
        run on each MultiAgentTrace turn and are averaged. Full-trace metrics
        (context_retention, coherence, ...) run on the full MultiAgentTrace,
        whose `messages` property aggregates sub-agent messages in order.

        Args:
            per_turn_traces: List of per-turn MultiAgentTraces (one per user turn)
            full_trace: MultiAgentTrace covering the entire conversation
            ground_truth: Expected correct outputs (optional). May contain
                         per_turn_expected and per_turn_tools arrays in expected_arguments.
            metric_set: Dictionary mapping dimension names to lists of metric names
            dimension_weights: Weights for each dimension (uses defaults if None)
            thresholds: Threshold values for pass/fail (uses defaults if None)
            context: List of context documents (optional)
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with per-turn breakdown in metadata
        """
        if not per_turn_traces:
            raise ValueError("per_turn_traces must contain at least one trace")
        if not isinstance(full_trace, MultiAgentTrace):
            raise ValueError(
                "full_trace must be a MultiAgentTrace instance "
                "(use SingleAgentEvaluator.evaluate_multi_turn for AgentTrace)"
            )
        for i, t in enumerate(per_turn_traces):
            if not isinstance(t, MultiAgentTrace):
                raise ValueError(
                    f"per_turn_traces[{i}] must be a MultiAgentTrace instance "
                    "(use SingleAgentEvaluator.evaluate_multi_turn for AgentTrace)"
                )

        # Reuse the shared multi-turn loop on BaseEvaluator. Because
        # self._get_metrics_to_calculate / self.default_dimension_weights /
        # self.default_thresholds are all overridden on this class, the loop
        # picks up the multi-agent metric registry and weights automatically.
        return self._run_multi_turn(
            per_turn_traces=per_turn_traces,
            full_trace=full_trace,
            ground_truth=ground_truth,
            metric_set=metric_set,
            dimension_weights=dimension_weights,
            thresholds=thresholds,
            context=context,
            **kwargs,
        )

    def _get_metrics_to_calculate(
        self,
        metric_set: Optional[Dict[str, List[str]]] = None
    ) -> List[any]:
        """
        Get list of metric instances to calculate.

        Args:
            metric_set: Dictionary mapping dimension names to lists of metric names.
                       If None, uses all available metrics.

        Returns:
            List of metric instances
        """
        metrics = []

        if metric_set is None:
            # Use all available metrics
            all_metric_names = self.metric_registry.list_metrics()
            for metric_name in all_metric_names:
                try:
                    metric = self.metric_registry.get_metric(metric_name)
                    metrics.append(metric)
                except Exception as e:
                    logger.warning(
                        f"Failed to create metric {metric_name}: {str(e)}"
                    )
        else:
            # Use specified metrics
            for dimension_name, metric_names in metric_set.items():
                for metric_name in metric_names:
                    try:
                        metric = self.metric_registry.get_metric(metric_name)
                        metrics.append(metric)
                    except Exception as e:
                        logger.warning(
                            f"Failed to create metric {metric_name}: {str(e)}"
                        )

        return metrics

    def batch_evaluate(
        self,
        traces: List[MultiAgentTrace],
        ground_truths: Optional[List[Optional[GroundTruth]]] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple multi-agent traces in batch.

        Automatically calculates batch-level metrics for metrics that support it
        (e.g., agent_utilization_batch).

        Args:
            traces: List of multi-agent traces to evaluate
            ground_truths: List of ground truths (optional, can be None for each)
            metric_set: Dictionary mapping dimension names to lists of metric names
            **kwargs: Additional evaluation parameters

        Returns:
            List of EvaluationResult objects, with batch metrics in results[0].metadata

        Raises:
            ValueError: If inputs are invalid
        """
        # Call parent batch_evaluate
        results = super().batch_evaluate(traces, ground_truths, metric_set, **kwargs)

        # Calculate batch-level metrics
        self._calculate_batch_metrics(traces, ground_truths, results, metric_set)

        return results

    def _calculate_batch_metrics(
        self,
        traces: List[MultiAgentTrace],
        ground_truths: List[Optional[GroundTruth]],
        results: List[EvaluationResult],
        metric_set: Optional[Dict[str, List[str]]]
    ) -> None:
        """
        Calculate batch-level metrics for multi-agent scenarios.

        Automatically detects which metrics have calculate_batch() static method
        and calls them to generate aggregate statistics across the batch.

        Args:
            traces: List of multi-agent traces
            ground_truths: List of ground truths
            results: List of evaluation results (will be updated with batch metrics)
            metric_set: Metric set configuration
        """
        from uaef.models.evaluation_input import EvaluationInput

        # Get metrics to calculate
        metrics_to_calculate = self._get_metrics_to_calculate(metric_set)

        # Find metrics that support batch calculation
        batch_metrics = []
        for metric in metrics_to_calculate:
            if hasattr(metric.__class__, 'calculate_batch') and callable(getattr(metric.__class__, 'calculate_batch')):
                batch_metrics.append(metric)

        if not batch_metrics:
            logger.debug("No metrics support batch calculation")
            return

        logger.debug(f"Found {len(batch_metrics)} metrics supporting batch calculation")

        # Build evaluation inputs
        if ground_truths is None:
            ground_truths = [None] * len(traces)

        evaluation_inputs = []
        for trace, gt in zip(traces, ground_truths):
            eval_input = EvaluationInput(trace=trace, ground_truth=gt, context=[])
            evaluation_inputs.append(eval_input)

        # Calculate batch metrics
        batch_scores = []
        for metric in batch_metrics:
            try:
                # Call the static calculate_batch method
                batch_score = metric.__class__.calculate_batch(evaluation_inputs)
                if batch_score.score is not None:
                    batch_scores.append(batch_score)
                    logger.info(f"Calculated batch metric: {batch_score.metric_name} = {batch_score.score:.3f}")
            except Exception as e:
                logger.error(f"Failed to calculate batch metric {metric.get_name()}: {str(e)}")

        # Store batch metrics in first result's metadata
        if results and batch_scores:
            if "batch_metrics" not in results[0].metadata:
                results[0].metadata["batch_metrics"] = {}

            for batch_score in batch_scores:
                results[0].metadata["batch_metrics"][batch_score.metric_name] = {
                    "score": batch_score.score,
                    "reasoning": batch_score.reasoning,
                    "metadata": batch_score.metadata
                }

            logger.info(f"Stored {len(batch_scores)} batch-level metrics in results")
