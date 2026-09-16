# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online evaluation mode for UAEF.

This module provides real-time evaluation functionality with:
- Fast deterministic metrics (< 500ms)
- Async LLM-based metrics (non-blocking)
- Partial results for immediate decisions
- Background storage of complete results
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

from uaef.evaluation.base_evaluator import BaseEvaluator
from uaef.logging import get_logger
from uaef.models.agent_trace import AgentTrace
from uaef.models.dimension_result import DimensionResult
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.ground_truth import GroundTruth
from uaef.models.metric_score import MetricScore

logger = get_logger(__name__)


class PartialEvaluationResult:
    """
    Partial evaluation result for online mode.
    
    Contains immediately available deterministic metrics and placeholders
    for async LLM-based metrics that are still being calculated.
    """
    
    def __init__(
        self,
        trace_id: any,
        deterministic_scores: List[MetricScore],
        llm_scores_pending: int,
        overall_score: float,
        passed: bool,
        failures: List[str],
        warnings: List[str],
        metadata: Dict[str, any]
    ):
        """Initialize partial evaluation result."""
        self.trace_id = trace_id
        self.deterministic_scores = deterministic_scores
        self.llm_scores_pending = llm_scores_pending
        self.overall_score = overall_score
        self.passed = passed
        self.failures = failures
        self.warnings = warnings
        self.metadata = metadata
        self.is_complete = llm_scores_pending == 0
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary representation."""
        return {
            "trace_id": str(self.trace_id),
            "deterministic_scores": [
                {
                    "metric_name": s.metric_name,
                    "score": s.score,
                    "reasoning": s.reasoning
                }
                for s in self.deterministic_scores
            ],
            "llm_scores_pending": self.llm_scores_pending,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "is_complete": self.is_complete,
            "metadata": self.metadata
        }


class OnlineEvaluator(BaseEvaluator):
    """
    Evaluator for online/real-time evaluation.
    
    Optimized for low-latency evaluation with:
    - Synchronous calculation of fast deterministic metrics
    - Asynchronous execution of LLM-based metrics
    - Immediate return of partial results
    - Background completion and storage
    - Threshold-based pass/fail decisions
    """
    
    def __init__(
        self,
        metric_registry: Optional[any] = None,
        max_workers: int = 4,
        fast_metrics_only: bool = False,
        latency_threshold_ms: float = 500.0,
        default_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize online evaluator.
        
        Args:
            metric_registry: MetricRegistry instance (uses global if None)
            max_workers: Maximum number of parallel workers
            fast_metrics_only: If True, only calculate deterministic metrics
            latency_threshold_ms: Maximum latency for synchronous metrics (ms)
            default_thresholds: Default threshold values for dimensions
        """
        super().__init__(metric_registry=metric_registry, max_workers=max_workers)
        
        self.fast_metrics_only = fast_metrics_only
        self.latency_threshold_ms = latency_threshold_ms
        
        # Set default thresholds for online evaluation
        self.default_thresholds = default_thresholds or {
            "tool_calling": 0.7,
            "response_quality": 0.6,
            "responsible_ai": 0.8,
            "performance": 0.5
        }
        
        # Storage for async results (in production, use database)
        self._async_results: Dict[str, EvaluationResult] = {}
        
        logger.info(
            f"Initialized OnlineEvaluator with fast_metrics_only={fast_metrics_only}, "
            f"latency_threshold={latency_threshold_ms}ms"
        )
    
    def evaluate(
        self,
        trace: AgentTrace,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a trace in online mode (blocking until complete).
        
        This method blocks until all metrics are calculated. For non-blocking
        evaluation, use evaluate_response() instead.
        
        Args:
            trace: Agent trace to evaluate
            ground_truth: Expected correct outputs (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names
            thresholds: Threshold values for pass/fail (uses defaults if None)
            context: List of context documents (optional)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Complete EvaluationResult
        """
        logger.info(f"Online evaluation (blocking) for trace {trace.trace_id}")
        
        # Use evaluate_response and wait for completion
        partial_result, complete_future = self.evaluate_response(
            trace=trace,
            ground_truth=ground_truth,
            metric_set=metric_set,
            thresholds=thresholds,
            context=context,
            **kwargs
        )
        
        # Wait for completion
        if complete_future:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            complete_result = loop.run_until_complete(complete_future)
            return complete_result
        else:
            # No async metrics, convert partial to complete
            return self._partial_to_complete(partial_result, [])
    
    def evaluate_response(
        self,
        trace: AgentTrace,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        store_async: bool = True,
        **kwargs
    ) -> Tuple[PartialEvaluationResult, Optional[asyncio.Task]]:
        """
        Evaluate a response in online mode (non-blocking).
        
        Returns partial results immediately with deterministic metrics,
        and starts async calculation of LLM-based metrics in background.
        
        Args:
            trace: Agent trace to evaluate
            ground_truth: Expected correct outputs (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names.
                       If None, uses subset of metrics suitable for online evaluation.
            thresholds: Threshold values for pass/fail (uses defaults if None)
            context: List of context documents (optional)
            store_async: If True, store complete results in background
            **kwargs: Additional evaluation parameters
            
        Returns:
            Tuple of (PartialEvaluationResult, Optional[asyncio.Task])
            The task completes when all LLM metrics are calculated.
        """
        start_time = time.time()
        
        logger.info(f"Online evaluation (non-blocking) for trace {trace.trace_id}")
        
        # Validate inputs
        if not isinstance(trace, AgentTrace):
            raise ValueError("trace must be an AgentTrace instance")
        
        # Use default thresholds if not provided
        thresholds = thresholds or self.default_thresholds
        
        # Create evaluation input
        evaluation_input = EvaluationInput(
            trace=trace,
            ground_truth=ground_truth,
            context=context or []
        )
        
        # Determine which metrics to calculate
        metrics_to_calculate = self._get_online_metrics(metric_set)
        
        # Separate deterministic and LLM-based metrics
        deterministic_metrics = []
        llm_metrics = []
        
        for metric in metrics_to_calculate:
            if metric.requires_llm_judge() and not self.fast_metrics_only:
                llm_metrics.append(metric)
            else:
                deterministic_metrics.append(metric)
        
        logger.debug(
            f"Split metrics: {len(deterministic_metrics)} deterministic, "
            f"{len(llm_metrics)} LLM-based"
        )
        
        # Calculate deterministic metrics synchronously
        deterministic_scores = self._calculate_deterministic_parallel(
            deterministic_metrics,
            evaluation_input
        )
        
        deterministic_time = (time.time() - start_time) * 1000
        logger.debug(f"Deterministic metrics calculated in {deterministic_time:.1f}ms")
        
        # Check if we exceeded latency threshold
        warnings = []
        if deterministic_time > self.latency_threshold_ms:
            warnings.append(
                f"Deterministic metrics took {deterministic_time:.1f}ms, "
                f"exceeding threshold of {self.latency_threshold_ms}ms"
            )
        
        # Calculate partial scores and pass/fail
        partial_dimension_results = self.aggregate_dimension_scores(
            metric_scores=deterministic_scores,
            dimension_weights=None  # Use equal weights for partial results
        )
        
        partial_overall_score = self.calculate_overall_score(partial_dimension_results)
        
        passed, failures = self.check_thresholds(
            dimension_results=partial_dimension_results,
            thresholds=thresholds
        )
        
        # Create partial result
        partial_result = PartialEvaluationResult(
            trace_id=trace.trace_id,
            deterministic_scores=deterministic_scores,
            llm_scores_pending=len(llm_metrics),
            overall_score=partial_overall_score,
            passed=passed,
            failures=failures,
            warnings=warnings,
            metadata={
                "num_deterministic_metrics": len(deterministic_scores),
                "num_llm_metrics_pending": len(llm_metrics),
                "deterministic_latency_ms": deterministic_time,
                "fast_metrics_only": self.fast_metrics_only
            }
        )
        
        logger.info(
            f"Partial result ready in {deterministic_time:.1f}ms: "
            f"score={partial_overall_score:.3f}, passed={passed}"
        )
        
        # Start async calculation of LLM metrics if needed
        complete_task = None
        if llm_metrics and not self.fast_metrics_only:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            complete_task = loop.create_task(
                self._complete_evaluation_async(
                    partial_result=partial_result,
                    llm_metrics=llm_metrics,
                    evaluation_input=evaluation_input,
                    thresholds=thresholds,
                    store_async=store_async
                )
            )
            
            logger.debug(f"Started async calculation of {len(llm_metrics)} LLM metrics")
        
        return partial_result, complete_task
    
    async def _complete_evaluation_async(
        self,
        partial_result: PartialEvaluationResult,
        llm_metrics: List[any],
        evaluation_input: EvaluationInput,
        thresholds: Dict[str, float],
        store_async: bool
    ) -> EvaluationResult:
        """
        Complete evaluation asynchronously by calculating LLM metrics.
        
        Args:
            partial_result: Partial result with deterministic metrics
            llm_metrics: List of LLM-based metrics to calculate
            evaluation_input: Input data for evaluation
            thresholds: Threshold values for pass/fail
            store_async: If True, store complete results
            
        Returns:
            Complete EvaluationResult
        """
        start_time = time.time()
        
        logger.debug(f"Starting async LLM metric calculation for {partial_result.trace_id}")
        
        # Calculate LLM metrics asynchronously
        llm_scores = await self._calculate_llm_async_impl(
            llm_metrics,
            evaluation_input
        )
        
        llm_time = (time.time() - start_time) * 1000
        logger.debug(f"LLM metrics calculated in {llm_time:.1f}ms")
        
        # Combine with deterministic scores
        all_scores = partial_result.deterministic_scores + llm_scores
        
        # Create complete result
        complete_result = self._partial_to_complete(
            partial_result,
            llm_scores,
            thresholds
        )
        
        # Store result if requested
        if store_async:
            self._store_result_async(complete_result)
        
        logger.info(
            f"Complete evaluation ready for {partial_result.trace_id}: "
            f"score={complete_result.overall_score:.3f}, "
            f"total_time={llm_time:.1f}ms"
        )
        
        return complete_result
    
    def _get_online_metrics(
        self,
        metric_set: Optional[Dict[str, List[str]]] = None
    ) -> List[any]:
        """
        Get list of metrics suitable for online evaluation.
        
        If metric_set is None, returns a subset of metrics that:
        - Don't require ground truth
        - Are fast to calculate
        - Are relevant for online decisions
        
        Args:
            metric_set: Dictionary mapping dimension names to lists of metric names.
                       If None, uses default online metric set.
            
        Returns:
            List of metric instances
        """
        if metric_set is not None:
            # Use specified metrics
            metrics = []
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
        
        # Use default online metrics (no ground truth required)
        online_metric_names = [
            # Performance metrics (fast, deterministic)
            "latency_score",
            "token_efficiency",
            "throughput",
            
            # Responsible AI metrics (LLM-based but important for online)
            "safety_score",
            "toxicity_score",
            "prompt_injection_detection",
            
            # Response quality metrics (LLM-based, no GT required)
            "answer_relevance",
        ]
        
        metrics = []
        for metric_name in online_metric_names:
            try:
                metric = self.metric_registry.get_metric(metric_name)
                # Skip if requires ground truth
                if not metric.requires_ground_truth():
                    metrics.append(metric)
            except KeyError:
                logger.debug(f"Metric {metric_name} not available in registry")
            except Exception as e:
                logger.warning(f"Failed to create metric {metric_name}: {str(e)}")
        
        return metrics
    
    def _partial_to_complete(
        self,
        partial_result: PartialEvaluationResult,
        llm_scores: List[MetricScore],
        thresholds: Optional[Dict[str, float]] = None
    ) -> EvaluationResult:
        """
        Convert partial result to complete EvaluationResult.
        
        Args:
            partial_result: Partial result with deterministic metrics
            llm_scores: LLM-based metric scores
            thresholds: Threshold values for pass/fail
            
        Returns:
            Complete EvaluationResult
        """
        # Combine all scores
        all_scores = partial_result.deterministic_scores + llm_scores
        
        # Aggregate by dimension
        dimension_results = self.aggregate_dimension_scores(
            metric_scores=all_scores,
            dimension_weights=None  # Use equal weights
        )
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(dimension_results)
        
        # Check thresholds
        thresholds = thresholds or self.default_thresholds
        passed, failures = self.check_thresholds(
            dimension_results=dimension_results,
            thresholds=thresholds
        )
        
        # Create complete result
        metadata = partial_result.metadata.copy()
        metadata.update({
            "num_total_metrics": len(all_scores),
            "num_llm_metrics": len(llm_scores)
        })
        
        return EvaluationResult(
            trace_id=partial_result.trace_id,
            dimension_results=dimension_results,
            overall_score=overall_score,
            passed=passed,
            failures=failures,
            warnings=partial_result.warnings,
            metadata=metadata
        )
    
    def _store_result_async(self, result: EvaluationResult) -> None:
        """
        Store complete result in background.
        
        In production, this would write to a database. For now, stores in memory.
        
        Args:
            result: Complete evaluation result to store
        """
        self._async_results[str(result.trace_id)] = result
        logger.debug(f"Stored complete result for {result.trace_id}")
    
    def get_stored_result(self, trace_id: str) -> Optional[EvaluationResult]:
        """
        Retrieve a stored result by trace ID.
        
        Args:
            trace_id: Trace ID to look up
            
        Returns:
            EvaluationResult if found, None otherwise
        """
        return self._async_results.get(trace_id)

    def aggregate_session_metrics(
        self,
        turn_results: List[EvaluationResult],
        session_id: Optional[str] = None
    ) -> EvaluationResult:
        """
        Aggregate metrics across all turns in a conversation session.
        
        This method combines per-turn evaluation results into session-level metrics,
        providing an overall assessment of the entire conversation including:
        - Average scores across all turns
        - Context retention across the conversation
        - Overall coherence and goal achievement
        - Aggregated performance metrics
        
        Args:
            turn_results: List of EvaluationResult objects, one per turn
            session_id: Optional session identifier
            
        Returns:
            EvaluationResult with aggregated session-level metrics
            
        Example:
            >>> evaluator = OnlineEvaluator()
            >>> turn_results = []
            >>> 
            >>> # Evaluate each turn
            >>> for turn_trace in conversation_turns:
            >>>     result = evaluator.evaluate(turn_trace)
            >>>     turn_results.append(result)
            >>> 
            >>> # Get session-level aggregation
            >>> session_result = evaluator.aggregate_session_metrics(turn_results)
            >>> print(f"Overall conversation score: {session_result.overall_score:.3f}")
        """
        if not turn_results:
            raise ValueError("turn_results cannot be empty")
        
        logger.info(f"Aggregating metrics for session with {len(turn_results)} turns")
        
        # Collect all dimension results across turns
        all_dimension_results = {}
        
        for turn_result in turn_results:
            for dim_name, dim_result in turn_result.dimension_results.items():
                if dim_name not in all_dimension_results:
                    all_dimension_results[dim_name] = []
                all_dimension_results[dim_name].append(dim_result)
        
        # Calculate average scores per dimension
        aggregated_dimensions = {}
        
        for dim_name, dim_results in all_dimension_results.items():
            avg_score = sum(dr.score for dr in dim_results) / len(dim_results)
            
            # Combine all metric scores from this dimension
            all_metric_scores = []
            for dr in dim_results:
                all_metric_scores.extend(dr.metric_scores)
            
            aggregated_dimensions[dim_name] = DimensionResult(
                dimension_name=dim_name,
                score=avg_score,
                metric_scores=all_metric_scores,
                weight=dim_results[0].weight if dim_results else 1.0
            )
        
        # Calculate overall session score
        overall_score = sum(
            dr.score * dr.weight 
            for dr in aggregated_dimensions.values()
        ) / sum(dr.weight for dr in aggregated_dimensions.values())
        
        # Aggregate failures and warnings
        all_failures = []
        all_warnings = []
        
        for turn_result in turn_results:
            all_failures.extend(turn_result.failures)
            all_warnings.extend(turn_result.warnings)
        
        # Determine if session passed (all turns must pass)
        passed = all(tr.passed for tr in turn_results)
        
        # Create session result
        session_result = EvaluationResult(
            trace_id=session_id or f"session_{turn_results[0].trace_id}",
            dimension_results=aggregated_dimensions,
            overall_score=overall_score,
            passed=passed,
            failures=list(set(all_failures)),  # Remove duplicates
            warnings=list(set(all_warnings)),  # Remove duplicates
            metadata={
                "num_turns": len(turn_results),
                "session_id": session_id,
                "turn_scores": [tr.overall_score for tr in turn_results],
                "aggregation_type": "session_level"
            }
        )
        
        logger.info(
            f"Session aggregation complete: {len(turn_results)} turns, "
            f"overall_score={overall_score:.3f}, passed={passed}"
        )
        
        return session_result
