# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-agent evaluator for UAEF.

This module provides evaluation functionality for single-agent traces,
coordinating the adapter → metrics → LLM judge flow.
"""

from typing import Dict, List, Optional

from uaef.evaluation.base_evaluator import BaseEvaluator
from uaef.logging import get_logger
from uaef.models.agent_trace import AgentTrace
from uaef.models.dimension_result import DimensionResult
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.ground_truth import GroundTruth
from uaef.models.metric_score import MetricScore
from uaef.models.multi_agent_trace import MultiAgentTrace

logger = get_logger(__name__)


class SingleAgentEvaluator(BaseEvaluator):
    """
    Evaluator for single-agent traces.
    
    Coordinates the complete evaluation flow:
    1. Create EvaluationInput from trace and ground truth
    2. Calculate metrics for each dimension
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
        Initialize single-agent evaluator.
        
        Args:
            metric_registry: MetricRegistry instance (uses global if None)
            max_workers: Maximum number of parallel workers for metric calculation
            default_dimension_weights: Default weights for dimensions
            default_thresholds: Default threshold values for dimensions
        """
        super().__init__(metric_registry=metric_registry, max_workers=max_workers)
        
        # Set default dimension weights if not provided
        self.default_dimension_weights = default_dimension_weights or {
            "tool_calling": 0.25,
            "response_quality": 0.25,
            "responsible_ai": 0.20,
            "performance": 0.10,
            "multi_turn": 0.10,
            "reasoning": 0.10
        }
        
        # Set default thresholds if not provided
        self.default_thresholds = default_thresholds or {
            "tool_calling": 0.7,
            "response_quality": 0.7,
            "responsible_ai": 0.8,
            "performance": 0.6,
            "multi_turn": 0.6,
            "reasoning": 0.6
        }
        
        logger.info(
            f"Initialized SingleAgentEvaluator with "
            f"{len(self.default_dimension_weights)} dimensions"
        )
    
    def evaluate(
        self,
        trace: AgentTrace,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single agent trace.
        
        Args:
            trace: Agent trace to evaluate
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
        logger.info(f"Evaluating trace {trace.trace_id}")
        
        # Validate inputs
        if not isinstance(trace, AgentTrace):
            raise ValueError("trace must be an AgentTrace instance")
        
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
                "dimension_weights": dimension_weights,
                "thresholds": thresholds
            }
        )
        
        logger.info(
            f"Evaluation complete for trace {trace.trace_id}: "
            f"score={overall_score:.3f}, passed={passed}"
        )
        
        return result
    
    def evaluate_trace(
        self,
        trace: AgentTrace,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Alias for evaluate() method for backward compatibility.
        
        Args:
            trace: Agent trace to evaluate
            ground_truth: Expected correct outputs (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with scores and metadata
        """
        return self.evaluate(
            trace=trace,
            ground_truth=ground_truth,
            metric_set=metric_set,
            **kwargs
        )

    def evaluate_multi_turn(
        self,
        per_turn_traces: List[AgentTrace],
        full_trace: AgentTrace,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single-agent multi-turn conversation from pre-split per-turn traces.

        Both per_turn_traces and full_trace must be AgentTrace instances.
        Multi-agent traces are handled by MultiAgentEvaluator.evaluate_multi_turn().

        Flow:
        1. Iterate per_turn_traces, run per-turn-eligible metrics on each
           (using `ground_truth.expected_arguments["per_turn_expected"][i]` and
           `["per_turn_tools"][i]` for that turn's ground truth)
        2. Average per-turn scores across turns
        3. Run full-trace-only metrics (multi-turn + throughput) on full_trace
        4. Combine and return EvaluationResult with per-turn breakdown in metadata

        Args:
            per_turn_traces: List of per-turn AgentTraces (one per user turn)
            full_trace: AgentTrace covering the entire conversation
            ground_truth: Expected correct outputs (optional). May contain
                         per_turn_expected and per_turn_tools arrays in expected_arguments.
            metric_set: Dictionary mapping dimension names to lists of metric names
            dimension_weights: Weights for each dimension (uses defaults if None)
            thresholds: Threshold values for pass/fail (uses defaults if None)
            context: List of context documents (optional)
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with scores and metadata (including per_turn breakdown)
        """
        if not per_turn_traces:
            raise ValueError("per_turn_traces must contain at least one trace")
        if not isinstance(full_trace, AgentTrace) or isinstance(full_trace, MultiAgentTrace):
            raise ValueError(
                "full_trace must be an AgentTrace instance "
                "(use MultiAgentEvaluator.evaluate_multi_turn for MultiAgentTrace)"
            )
        for i, t in enumerate(per_turn_traces):
            if not isinstance(t, AgentTrace) or isinstance(t, MultiAgentTrace):
                raise ValueError(
                    f"per_turn_traces[{i}] must be an AgentTrace instance "
                    "(use MultiAgentEvaluator.evaluate_multi_turn for MultiAgentTrace)"
                )

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
    
    def evaluate_with_custom_metrics(
        self,
        trace: AgentTrace,
        metrics: List[str],
        ground_truth: Optional[GroundTruth] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a trace with a custom list of metrics.
        
        Args:
            trace: Agent trace to evaluate
            metrics: List of metric names to calculate
            ground_truth: Expected correct outputs (optional)
            dimension_weights: Weights for each dimension (uses defaults if None)
            thresholds: Threshold values for pass/fail (uses defaults if None)
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with scores and metadata
        """
        # Group metrics by dimension
        metric_set = {}
        
        for metric_name in metrics:
            try:
                metric_info = self.metric_registry.get_metric_info(metric_name)
                dimension = metric_info.get("dimension", "unknown")
                
                if dimension not in metric_set:
                    metric_set[dimension] = []
                
                metric_set[dimension].append(metric_name)
                
            except KeyError:
                logger.warning(f"Metric {metric_name} not found in registry")
        
        return self.evaluate(
            trace=trace,
            ground_truth=ground_truth,
            metric_set=metric_set,
            dimension_weights=dimension_weights,
            thresholds=thresholds,
            **kwargs
        )
