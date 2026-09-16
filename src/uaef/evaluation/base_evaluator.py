# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base evaluator interface for UAEF.

This module provides the abstract base class for all evaluators,
defining the core evaluation interface and common functionality.
"""

import asyncio
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Increase recursion limit for RAGAS/DeepEval + nest_asyncio + Pydantic v2 in Jupyter.
# These libraries create deeply nested async/serialization call stacks that exceed
# Python's default limit of 1000.
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

from uaef.logging import get_logger
from uaef.metrics.base import BaseMetric
from uaef.metrics.registry import get_registry
from uaef.models.dimension_result import DimensionResult
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.ground_truth import GroundTruth
from uaef.models.metric_score import MetricScore

logger = get_logger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Provides common functionality for:
    - Single evaluation
    - Batch evaluation
    - Parallel metric calculation
    - Async support for LLM-based metrics
    """
    
    def __init__(
        self,
        metric_registry: Optional[any] = None,
        max_workers: int = 4
    ):
        """
        Initialize base evaluator.
        
        Args:
            metric_registry: MetricRegistry instance (uses global if None)
            max_workers: Maximum number of parallel workers for metric calculation
        """
        self.metric_registry = metric_registry or get_registry()
        self.max_workers = max_workers
        
        logger.info(
            f"Initialized {self.__class__.__name__} with max_workers={max_workers}"
        )
    
    @abstractmethod
    def evaluate(
        self,
        trace: any,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single trace.
        
        Args:
            trace: Agent trace to evaluate
            ground_truth: Expected correct outputs (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with scores and metadata
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If evaluation fails
        """
        pass
    
    def batch_evaluate(
        self,
        traces: List[any],
        ground_truths: Optional[List[Optional[GroundTruth]]] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple traces in batch.
        
        Args:
            traces: List of agent traces to evaluate
            ground_truths: List of ground truths (optional, can be None for each)
            metric_set: Dictionary mapping dimension names to lists of metric names
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of EvaluationResult objects
            
        Raises:
            ValueError: If inputs are invalid
        """
        logger.info(f"Starting batch evaluation for {len(traces)} traces")
        
        # Validate inputs
        if not traces:
            raise ValueError("traces cannot be empty")
        
        if ground_truths is None:
            ground_truths = [None] * len(traces)
        
        if len(traces) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: {len(traces)} traces but {len(ground_truths)} ground truths"
            )
        
        # Evaluate each trace
        results = []
        for i, (trace, gt) in enumerate(zip(traces, ground_truths)):
            try:
                logger.debug(f"Evaluating trace {i+1}/{len(traces)}")
                result = self.evaluate(
                    trace=trace,
                    ground_truth=gt,
                    metric_set=metric_set,
                    **kwargs
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate trace {i+1}: {str(e)}")
                # Create a failed evaluation result
                results.append(self._create_failed_result(trace, str(e)))
        
        success_count = sum(1 for r in results if r.passed)
        logger.info(
            f"Batch evaluation complete: {success_count}/{len(results)} passed"
        )

        return results
    
    def calculate_metrics_with_dependencies(
        self,
        metrics: List[BaseMetric],
        evaluation_input: EvaluationInput
    ) -> List[MetricScore]:
        """
        Calculate metrics respecting dependencies.

        Simple two-phase approach:
        - Phase 1: Calculate metrics with no dependencies
        - Phase 2: Calculate metrics with dependencies (which can now read metadata)

        Args:
            metrics: List of metric instances to calculate
            evaluation_input: Input data for evaluation

        Returns:
            List of MetricScore objects for all metrics
        """
        logger.debug(f"Calculating {len(metrics)} metrics with dependency resolution")

        # Separate metrics: independent vs dependent
        independent_metrics = []
        dependent_metrics = []

        for metric in metrics:
            deps = metric.get_dependencies()
            has_deps = bool(deps.get("requires_metrics") or deps.get("requires_metadata"))

            if has_deps:
                dependent_metrics.append(metric)
            else:
                independent_metrics.append(metric)

        all_scores = []

        # If no dependent metrics, calculate everything in parallel
        if not dependent_metrics:
            logger.debug("No dependent metrics, calculating all metrics in parallel")
            return self.calculate_metrics_parallel(metrics, evaluation_input)

        # Check which dependencies are already satisfied
        required_metadata = set()
        for metric in dependent_metrics:
            deps = metric.get_dependencies()
            required_metadata.update(deps.get("requires_metadata", []))

        # Filter out independent metrics if their metadata is already present
        missing_metadata = [key for key in required_metadata if key not in evaluation_input.trace.metadata]

        if not missing_metadata:
            # All dependencies already satisfied, calculate everything in parallel
            logger.debug("All dependencies already satisfied in trace.metadata, calculating all metrics in parallel")
            return self.calculate_metrics_parallel(metrics, evaluation_input)

        logger.debug(f"Missing metadata: {missing_metadata}, will calculate provider metrics first")

        # Phase 1: Calculate independent metrics to provide missing metadata
        logger.debug(f"Phase 1: Calculating {len(independent_metrics)} independent metrics")
        phase1_scores = self.calculate_metrics_parallel(independent_metrics, evaluation_input)
        all_scores.extend(phase1_scores)

        # Store metadata from phase 1 for phase 2
        self._store_metadata_from_scores(phase1_scores, evaluation_input)

        # Phase 2: Calculate dependent metrics
        logger.debug(f"Phase 2: Calculating {len(dependent_metrics)} dependent metrics")
        phase2_scores = self.calculate_metrics_parallel(dependent_metrics, evaluation_input)
        all_scores.extend(phase2_scores)

        logger.debug(f"Calculated {len(all_scores)} total metric scores")

        return all_scores

    def _store_metadata_from_scores(
        self,
        scores: List[MetricScore],
        evaluation_input: EvaluationInput
    ) -> None:
        """
        Store metadata in trace from calculated metric scores.

        Automatically extracts and stores:
        - quality_score: Average of quality metrics
        - completeness_score: From conversation_completeness

        Args:
            scores: List of calculated metric scores
            evaluation_input: Evaluation input containing trace
        """
        # Collect quality scores
        quality_metric_names = ["answer_relevance", "completeness", "accuracy", "hallucination_score"]
        quality_scores = []

        for score in scores:
            if score.score is None:
                continue

            # Quality metrics
            if score.metric_name in quality_metric_names:
                if score.metric_name == "hallucination_score":
                    quality_scores.append(1.0 - score.score)  # Invert
                else:
                    quality_scores.append(score.score)

            # Completeness score
            elif score.metric_name == "conversation_completeness":
                evaluation_input.trace.metadata["completeness_score"] = score.score
                logger.debug(f"Stored completeness_score={score.score:.3f}")

        # Store average quality score
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            evaluation_input.trace.metadata["quality_score"] = avg_quality
            logger.debug(f"Stored quality_score={avg_quality:.3f}")

    def calculate_metrics_parallel(
        self,
        metrics: List[BaseMetric],
        evaluation_input: EvaluationInput
    ) -> List[MetricScore]:
        """
        Calculate multiple metrics in parallel.

        Separates deterministic and LLM-based metrics, running deterministic
        metrics in parallel threads and LLM-based metrics asynchronously.

        Args:
            metrics: List of metric instances to calculate
            evaluation_input: Input data for evaluation

        Returns:
            List of MetricScore objects
        """
        logger.debug(f"Calculating {len(metrics)} metrics in parallel")

        # Separate deterministic and LLM-based metrics
        deterministic_metrics = []
        llm_metrics = []

        for metric in metrics:
            if metric.requires_llm_judge():
                llm_metrics.append(metric)
            else:
                deterministic_metrics.append(metric)

        logger.debug(
            f"Split metrics: {len(deterministic_metrics)} deterministic, "
            f"{len(llm_metrics)} LLM-based"
        )

        # Calculate deterministic metrics in parallel
        deterministic_scores = self._calculate_deterministic_parallel(
            deterministic_metrics,
            evaluation_input
        )

        # Calculate LLM-based metrics asynchronously
        llm_scores = self._calculate_llm_async(
            llm_metrics,
            evaluation_input
        )

        # Combine results
        all_scores = deterministic_scores + llm_scores

        logger.debug(f"Calculated {len(all_scores)} metric scores")

        return all_scores
    
    def _calculate_deterministic_parallel(
        self,
        metrics: List[BaseMetric],
        evaluation_input: EvaluationInput
    ) -> List[MetricScore]:
        """
        Calculate deterministic metrics in parallel using ThreadPoolExecutor.
        
        Args:
            metrics: List of deterministic metric instances
            evaluation_input: Input data for evaluation
            
        Returns:
            List of MetricScore objects
        """
        if not metrics:
            return []
        
        scores = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all metric calculations
            future_to_metric = {
                executor.submit(
                    self._safe_calculate_metric,
                    metric,
                    evaluation_input
                ): metric
                for metric in metrics
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_metric):
                metric = future_to_metric[future]
                try:
                    score = future.result()
                    if score is not None:
                        scores.append(score)
                except Exception as e:
                    logger.error(
                        f"Failed to calculate metric {metric.get_name()}: {str(e)}"
                    )
        
        return scores
    
    def _calculate_llm_async(
        self,
        metrics: List[BaseMetric],
        evaluation_input: EvaluationInput
    ) -> List[MetricScore]:
        """
        Calculate LLM-based metrics asynchronously.
        
        Args:
            metrics: List of LLM-based metric instances
            evaluation_input: Input data for evaluation
            
        Returns:
            List of MetricScore objects
        """
        if not metrics:
            return []
        
        # Run async metrics
        # Handle both Jupyter (with running loop) and regular Python environments
        try:
            # Try to get the running loop (Jupyter/IPython)
            loop = asyncio.get_running_loop()
            # If we're here, there's a running loop - use nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                scores = loop.run_until_complete(
                    self._calculate_llm_async_impl(metrics, evaluation_input)
                )
            except ImportError:
                # nest_asyncio not available, create task in existing loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._calculate_llm_async_impl(metrics, evaluation_input)
                    )
                    scores = future.result()
        except RuntimeError:
            # No running loop, create a new one (regular Python)
            scores = asyncio.run(
                self._calculate_llm_async_impl(metrics, evaluation_input)
            )
        
        return scores
    
    async def _calculate_llm_async_impl(
        self,
        metrics: List[BaseMetric],
        evaluation_input: EvaluationInput
    ) -> List[MetricScore]:
        """
        Async implementation of LLM metric calculation.

        Groups metrics by their ``batch_group()`` key. Metrics that share a
        batch group are evaluated together via ``batch_calculate()`` (which can
        make concurrent external calls). Metrics without a batch group are run
        sequentially to avoid event-loop conflicts with libraries that manage
        their own async internals (e.g., DeepEval).

        Args:
            metrics: List of LLM-based metric instances
            evaluation_input: Input data for evaluation

        Returns:
            List of MetricScore objects
        """
        # Separate batchable metrics from unbatchable ones
        batches: Dict[str, List[BaseMetric]] = {}
        sequential_metrics: List[BaseMetric] = []

        for metric in metrics:
            group = metric.batch_group()
            if group is not None:
                batches.setdefault(group, []).append(metric)
            else:
                sequential_metrics.append(metric)

        scores: List[MetricScore] = []

        # Evaluate each batch group via batch_calculate
        for group_key, group_metrics in batches.items():
            try:
                # Use the first metric's class to call the classmethod
                batch_cls = type(group_metrics[0])
                loop = asyncio.get_event_loop()
                batch_scores = await loop.run_in_executor(
                    None,
                    batch_cls.batch_calculate,
                    group_metrics,
                    evaluation_input,
                )
                scores.extend(batch_scores)
            except Exception as e:
                logger.error(f"Batch evaluation failed for group '{group_key}': {e}")
                # Fall back to sequential for this group
                for metric in group_metrics:
                    result = await self._safe_calculate_metric_async(metric, evaluation_input)
                    if result is not None:
                        scores.append(result)

        # Run remaining LLM metrics sequentially
        for metric in sequential_metrics:
            result = await self._safe_calculate_metric_async(metric, evaluation_input)
            if result is not None:
                scores.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Failed to calculate metric {metric.get_name()}: {result}")

        return scores
    
    def _safe_calculate_metric(
        self,
        metric: BaseMetric,
        evaluation_input: EvaluationInput
    ) -> Optional[MetricScore]:
        """
        Safely calculate a metric with error handling.
        
        Args:
            metric: Metric instance to calculate
            evaluation_input: Input data for evaluation
            
        Returns:
            MetricScore or None if calculation fails
        """
        try:
            # Check if ground truth is required but missing
            if metric.requires_ground_truth() and not evaluation_input.ground_truth:
                logger.warning(
                    f"Skipping metric {metric.get_name()}: requires ground truth but none provided"
                )
                return None
            
            # Calculate metric
            score = metric.calculate(evaluation_input)
            return score
            
        except Exception as e:
            logger.error(f"Error calculating metric {metric.get_name()}: {str(e)}")
            return None
    
    async def _safe_calculate_metric_async(
        self,
        metric: BaseMetric,
        evaluation_input: EvaluationInput
    ) -> Optional[MetricScore]:
        """
        Safely calculate a metric asynchronously with error handling.
        
        Runs the metric's synchronous calculate() in a thread executor to
        avoid event loop conflicts with RAGAS/DeepEval which manage their
        own async loops internally.
        
        Args:
            metric: Metric instance to calculate
            evaluation_input: Input data for evaluation
            
        Returns:
            MetricScore or None if calculation fails
        """
        try:
            # Check if ground truth is required but missing
            if metric.requires_ground_truth() and not evaluation_input.ground_truth:
                logger.warning(
                    f"Skipping metric {metric.get_name()}: requires ground truth but none provided"
                )
                return None
            
            # Run synchronous calculate() in a thread to avoid nested event loop issues
            loop = asyncio.get_event_loop()
            score = await loop.run_in_executor(None, metric.calculate, evaluation_input)
            return score
            
        except Exception as e:
            logger.error(f"Error calculating metric {metric.get_name()}: {str(e)}")
            return None

    def _run_multi_turn(
        self,
        per_turn_traces: List[any],
        full_trace: any,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Type-agnostic multi-turn evaluation loop.

        Subclasses (SingleAgentEvaluator, MultiAgentEvaluator) expose their own
        public ``evaluate_multi_turn`` that validates the trace type and then
        delegates here. The body works for both AgentTrace and MultiAgentTrace
        because:
        - per-turn metrics use ``self._get_metrics_to_calculate`` which the
          subclass picks
        - multi-agent metrics guard on ``isinstance(trace, MultiAgentTrace)``
          and skip on plain AgentTrace
        - multi-turn metrics read ``trace.messages`` which exists on both
          (a computed property on MultiAgentTrace)

        Args:
            per_turn_traces: Per-turn traces (one per user turn)
            full_trace: Trace covering the entire conversation
            ground_truth: Expected outputs (optional). May contain
                          per_turn_expected/per_turn_tools in expected_arguments.
            metric_set: Optional dimension → metric-name mapping
            dimension_weights: Per-dimension weights (defaults to subclass defaults)
            thresholds: Per-dimension thresholds (defaults to subclass defaults)
            context: Optional list of context documents
            **kwargs: Forwarded to underlying evaluation calls

        Returns:
            EvaluationResult with per-turn breakdown in metadata
        """
        # Avoid circular import: imported lazily so base_evaluator stays
        # decoupled from the trace types.
        from uaef.models.multi_agent_trace import MultiAgentTrace

        is_multi_agent = isinstance(full_trace, MultiAgentTrace)

        logger.info(
            f"Evaluating multi-turn conversation: trace={full_trace.trace_id}, "
            f"turns={len(per_turn_traces)}"
        )

        dimension_weights = dimension_weights or self.default_dimension_weights
        thresholds = thresholds or self.default_thresholds
        ctx = context or (ground_truth.context_documents if ground_truth else [])

        metrics_to_calculate = self._get_metrics_to_calculate(metric_set)

        # Metrics that only make sense on the full conversation trace.
        # Today every Multi-Turn-dimension metric is full-trace; per-turn
        # relevance scoring is provided by `answer_relevance` (Response Quality
        # dimension), which the per-turn loop already runs and averages.
        FULL_TRACE_ONLY = {
            "context_retention",
            "coherence",
            "conversation_completeness",
            "turn_efficiency",
            "role_adherence",
            "holistic_llm_judge",
            "user_satisfaction",
            "sentiment",
            "agent_tone",
            "naturalness",
            "per_turn_sentiment",
            "per_turn_agent_tone",
            "per_turn_naturalness",
            "deepeval_role_adherence",
            "containment",
            "resolution",
            "instruction_compliance",
            "optimum_turns",
        }

        per_turn_metrics = [
            m for m in metrics_to_calculate
            if m.get_name() not in FULL_TRACE_ONLY
        ]
        full_trace_metrics = [
            m for m in metrics_to_calculate
            if m.get_name() in FULL_TRACE_ONLY
        ]

        # ------------------------------------------------------------------
        # Per-turn evaluation
        # ------------------------------------------------------------------
        per_turn_expected = (
            ground_truth.expected_arguments.get("per_turn_expected", [])
            if ground_truth else []
        )
        per_turn_tools = (
            ground_truth.expected_arguments.get("per_turn_tools", [])
            if ground_truth else []
        )

        per_turn_results: List[Dict] = []
        score_buckets: Dict[str, List[float]] = {}

        for turn_idx, turn_trace in enumerate(per_turn_traces):
            logger.info(f"  Per-turn evaluation: turn {turn_idx + 1}/{len(per_turn_traces)}")

            turn_gt = None
            if ground_truth:
                turn_expected = (
                    per_turn_expected[turn_idx]
                    if turn_idx < len(per_turn_expected)
                    else ground_truth.expected_output
                )
                turn_tool_calls = (
                    per_turn_tools[turn_idx]
                    if turn_idx < len(per_turn_tools)
                    else ground_truth.expected_tool_calls
                )
                turn_gt = GroundTruth(
                    expected_output=turn_expected or None,
                    expected_tool_calls=turn_tool_calls,
                    context_documents=ground_truth.context_documents,
                )

            eval_input = EvaluationInput(
                trace=turn_trace,
                ground_truth=turn_gt,
                context=ctx,
            )
            scores = self.calculate_metrics_with_dependencies(
                metrics=per_turn_metrics,
                evaluation_input=eval_input,
            )
            turn_scores = {s.metric_name: s.score for s in scores}
            turn_reasoning = {s.metric_name: s.reasoning for s in scores}
            per_turn_results.append({
                "turn": turn_idx + 1,
                "metric_scores": turn_scores,
                "reasoning": turn_reasoning,
            })

            for metric_name, score in turn_scores.items():
                if score is not None:
                    score_buckets.setdefault(metric_name, []).append(score)

        per_turn_averages = {
            metric_name: sum(vals) / len(vals)
            for metric_name, vals in score_buckets.items()
        }

        # Build per-metric reasoning bundles so the aggregated MetricScore
        # carries each turn's reasoning instead of a generic "Average across
        # N turns" string. The structured form is in metadata["per_turn"]
        # for programmatic access; the formatted form lands in `reasoning`
        # so notebook-style printing surfaces it without extra plumbing.
        reasoning_buckets: Dict[str, List[Dict[str, any]]] = {}
        for turn_entry in per_turn_results:
            turn_no = turn_entry["turn"]
            for mname, sc in turn_entry["metric_scores"].items():
                reasoning_buckets.setdefault(mname, []).append({
                    "turn": turn_no,
                    "score": sc,
                    "reasoning": turn_entry["reasoning"].get(mname, ""),
                })

        def _format_per_turn_reasoning(name: str) -> str:
            entries = reasoning_buckets.get(name, [])
            if not entries:
                return f"Average across {len(per_turn_traces)} turns"
            lines = [f"Average across {len(per_turn_traces)} turns:"]
            for e in entries:
                sc_str = f"{e['score']:.2f}" if e["score"] is not None else "N/A"
                reason = (e["reasoning"] or "").strip()
                if reason:
                    lines.append(f"  Turn {e['turn']} ({sc_str}): {reason}")
                else:
                    lines.append(f"  Turn {e['turn']} ({sc_str})")
            return "\n".join(lines)

        # Build metric scores from per-turn averages
        metric_scores: List[MetricScore] = []
        for m in per_turn_metrics:
            name = m.get_name()
            if name in per_turn_averages:
                metric_scores.append(MetricScore(
                    metric_name=name,
                    score=per_turn_averages[name],
                    reasoning=_format_per_turn_reasoning(name),
                    metadata={"per_turn": reasoning_buckets.get(name, [])},
                ))

        # ------------------------------------------------------------------
        # Full-trace evaluation
        # ------------------------------------------------------------------
        if full_trace_metrics:
            # Make quality_score available on the full trace so any quality-dependent
            # full-trace metric can resolve it.
            quality_metric_names = ["answer_relevance", "completeness", "accuracy"]
            quality_vals = [per_turn_averages[n] for n in quality_metric_names if n in per_turn_averages]
            if "hallucination_score" in per_turn_averages:
                quality_vals.append(1.0 - per_turn_averages["hallucination_score"])
            if quality_vals:
                full_trace.metadata["quality_score"] = sum(quality_vals) / len(quality_vals)

            full_eval_input = EvaluationInput(
                trace=full_trace,
                ground_truth=ground_truth,
                context=ctx,
            )
            full_trace_scores = self.calculate_metrics_with_dependencies(
                metrics=full_trace_metrics,
                evaluation_input=full_eval_input,
            )
            metric_scores.extend(full_trace_scores)

        # ------------------------------------------------------------------
        # Aggregate, threshold, return
        # ------------------------------------------------------------------
        dimension_results = self.aggregate_dimension_scores(
            metric_scores=metric_scores,
            dimension_weights=dimension_weights,
        )
        overall_score = self.calculate_overall_score(dimension_results)
        passed, failures = self.check_thresholds(
            dimension_results=dimension_results,
            thresholds=thresholds,
        )

        return EvaluationResult(
            trace_id=full_trace.trace_id,
            dimension_results=dimension_results,
            overall_score=overall_score,
            passed=passed,
            failures=failures,
            warnings=[],
            metadata={
                "num_metrics": len(metric_scores),
                "num_dimensions": len(dimension_results),
                "num_turns": len(per_turn_traces),
                "is_multi_agent": is_multi_agent,
                "dimension_weights": dimension_weights,
                "thresholds": thresholds,
                "per_turn": per_turn_results,
                "per_turn_averages": per_turn_averages,
            },
        )

    def aggregate_dimension_scores(
        self,
        metric_scores: List[MetricScore],
        dimension_weights: Optional[Dict[str, float]] = None
    ) -> List[DimensionResult]:
        """
        Aggregate metric scores by dimension.
        
        Args:
            metric_scores: List of metric scores to aggregate
            dimension_weights: Optional weights for each dimension
            
        Returns:
            List of DimensionResult objects
        """
        # Group scores by dimension
        dimension_scores: Dict[str, List[MetricScore]] = {}
        
        for score in metric_scores:
            # Get metric info to determine dimension
            try:
                metric_info = self.metric_registry.get_metric_info(score.metric_name)
                dimension = metric_info.get("dimension", "unknown")
                
                if dimension not in dimension_scores:
                    dimension_scores[dimension] = []
                
                dimension_scores[dimension].append(score)
                
            except KeyError:
                logger.warning(
                    f"Metric {score.metric_name} not found in registry, "
                    f"assigning to 'unknown' dimension"
                )
                if "unknown" not in dimension_scores:
                    dimension_scores["unknown"] = []
                dimension_scores["unknown"].append(score)
        
        # Calculate aggregate score for each dimension
        dimension_results = []

        for dimension_name, scores in dimension_scores.items():
            # Filter out None scores (non-applicable metrics)
            valid_scores = [s for s in scores if s.score is not None]

            # Calculate weighted average from valid scores only
            if valid_scores:
                aggregate_score = sum(s.score for s in valid_scores) / len(valid_scores)
            else:
                aggregate_score = 0.0

            # Get dimension weight
            weight = 1.0
            if dimension_weights and dimension_name in dimension_weights:
                weight = dimension_weights[dimension_name]

            dimension_result = DimensionResult(
                dimension_name=dimension_name,
                metric_scores=scores,  # Include all scores (even None) for transparency
                aggregate_score=aggregate_score,
                weight=weight
            )

            dimension_results.append(dimension_result)
        
        return dimension_results
    
    def calculate_overall_score(
        self,
        dimension_results: List[DimensionResult]
    ) -> float:
        """
        Calculate overall score from dimension results.
        
        Args:
            dimension_results: List of dimension results
            
        Returns:
            Overall weighted score (0-1)
        """
        if not dimension_results:
            return 0.0
        
        # Calculate weighted average
        total_weight = sum(d.weight for d in dimension_results)
        
        if total_weight == 0:
            # Equal weights if all weights are 0
            return sum(d.aggregate_score for d in dimension_results) / len(dimension_results)
        
        weighted_sum = sum(d.aggregate_score * d.weight for d in dimension_results)
        overall_score = weighted_sum / total_weight
        
        return overall_score
    
    def check_thresholds(
        self,
        dimension_results: List[DimensionResult],
        thresholds: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if dimension scores meet thresholds.
        
        Args:
            dimension_results: List of dimension results
            thresholds: Dictionary mapping dimension names to threshold values
            
        Returns:
            Tuple of (passed, failures) where passed is True if all thresholds met,
            and failures is a list of failure messages
        """
        if not thresholds:
            return True, []
        
        failures = []
        
        for dimension_result in dimension_results:
            dimension_name = dimension_result.dimension_name
            
            if dimension_name in thresholds:
                threshold = thresholds[dimension_name]
                
                if dimension_result.aggregate_score < threshold:
                    failures.append(
                        f"{dimension_name}: score {dimension_result.aggregate_score:.3f} "
                        f"below threshold {threshold:.3f}"
                    )
        
        passed = len(failures) == 0
        
        return passed, failures
    
    def _create_failed_result(
        self,
        trace: any,
        error_message: str
    ) -> EvaluationResult:
        """
        Create a failed evaluation result.
        
        Args:
            trace: The trace that failed evaluation
            error_message: Error message describing the failure
            
        Returns:
            EvaluationResult with failed status
        """
        # Extract trace_id if possible
        trace_id = getattr(trace, "trace_id", None)
        
        return EvaluationResult(
            trace_id=trace_id,
            dimension_results=[],
            overall_score=0.0,
            passed=False,
            failures=[f"Evaluation failed: {error_message}"],
            warnings=[],
            metadata={"error": error_message}
        )
