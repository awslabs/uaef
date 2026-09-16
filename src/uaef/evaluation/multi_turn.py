# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn conversation evaluator for UAEF.

This module provides evaluation functionality for multi-turn conversations,
assessing context retention, coherence, and goal achievement across turns.
"""

from typing import Dict, List, Optional

from uaef.evaluation.base_evaluator import BaseEvaluator
from uaef.logging import get_logger
from uaef.models.agent_trace import AgentTrace
from uaef.models.dimension_result import DimensionResult
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.ground_truth import GroundTruth
from uaef.models.message import Message
from uaef.models.metric_score import MetricScore

logger = get_logger(__name__)


class MultiTurnEvaluator(BaseEvaluator):
    """
    Evaluator for multi-turn conversations.
    
    Specializes in evaluating:
    - Context retention across turns
    - Coherence and logical flow
    - Conversation completeness and goal achievement
    - Turn efficiency
    
    Requirements: 10.1-10.6
    """
    
    def __init__(
        self,
        metric_registry: Optional[any] = None,
        max_workers: int = 4,
        default_dimension_weights: Optional[Dict[str, float]] = None,
        default_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multi-turn evaluator.
        
        Args:
            metric_registry: MetricRegistry instance (uses global if None)
            max_workers: Maximum number of parallel workers for metric calculation
            default_dimension_weights: Default weights for dimensions
            default_thresholds: Default threshold values for dimensions
        """
        super().__init__(metric_registry=metric_registry, max_workers=max_workers)
        
        # Set default dimension weights focused on multi-turn metrics
        self.default_dimension_weights = default_dimension_weights or {
            "multi_turn": 1.0  # Focus on multi-turn dimension
        }
        
        # Set default thresholds
        self.default_thresholds = default_thresholds or {
            "multi_turn": 0.7
        }
        
        logger.info("Initialized MultiTurnEvaluator")
    
    def evaluate_conversation(
        self,
        messages: List[Message],
        goal: Optional[str] = None,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a multi-turn conversation.
        
        Requirements:
        - 10.1: Assess context retention across turns
        - 10.2: Evaluate coherence by checking logical flow
        - 10.3: Evaluate conversation completeness
        - 10.4: Verify information from earlier turns is used appropriately
        - 10.5: Support goal-based evaluation
        - 10.6: Identify which success criteria were not met
        
        Args:
            messages: List of conversation messages
            goal: User goal for the conversation (optional)
            ground_truth: Expected correct outputs (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names.
                       If None, uses all multi-turn metrics.
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
        logger.info(f"Evaluating conversation with {len(messages)} messages")
        
        # Validate inputs
        if not messages:
            raise ValueError("messages list cannot be empty")
        
        if not all(isinstance(m, Message) for m in messages):
            raise ValueError("All items in messages must be Message instances")
        
        # Use default weights and thresholds if not provided
        dimension_weights = dimension_weights or self.default_dimension_weights
        thresholds = thresholds or self.default_thresholds
        
        # Create a trace from messages for evaluation
        import uuid
        trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            messages=messages,
            tool_calls=[],  # Tool calls are embedded in messages
            metadata={
                "conversation_goal": goal,
                "num_turns": len(messages)
            }
        )
        
        # Create evaluation input
        evaluation_input = EvaluationInput(
            trace=trace,
            ground_truth=ground_truth,
            context=context or []
        )
        
        # Determine which metrics to calculate
        # Default to multi-turn metrics if not specified
        if metric_set is None:
            metric_set = {"multi_turn": self._get_default_multi_turn_metrics()}
        
        metrics_to_calculate = self._get_metrics_to_calculate(metric_set)
        
        logger.debug(f"Calculating {len(metrics_to_calculate)} metrics")
        
        # Calculate all metrics in parallel
        metric_scores = self.calculate_metrics_parallel(
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
        if goal is None:
            warnings.append(
                "Conversation goal not provided, completeness evaluation may be limited"
            )
        
        if ground_truth is None:
            warnings.append(
                "Ground truth not provided, some metrics may be skipped"
            )
        
        # Identify unmet success criteria (Requirement 10.6)
        unmet_criteria = self._identify_unmet_criteria(
            metric_scores=metric_scores,
            goal=goal,
            passed=passed
        )
        
        if unmet_criteria:
            failures.extend(unmet_criteria)
        
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
                "num_turns": len(messages),
                "conversation_goal": goal,
                "dimension_weights": dimension_weights,
                "thresholds": thresholds
            }
        )
        
        logger.info(
            f"Conversation evaluation complete: "
            f"score={overall_score:.3f}, passed={passed}, turns={len(messages)}"
        )
        
        return result
    
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
        Evaluate a multi-turn conversation from an AgentTrace.
        
        This is an alternative interface that accepts an AgentTrace directly.
        
        Args:
            trace: Agent trace containing conversation messages
            ground_truth: Expected correct outputs (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names
            dimension_weights: Weights for each dimension (uses defaults if None)
            thresholds: Threshold values for pass/fail (uses defaults if None)
            context: List of context documents (optional)
            **kwargs: Additional evaluation parameters (can include 'goal')
            
        Returns:
            EvaluationResult with scores and metadata
        """
        # Extract goal from kwargs or trace metadata
        goal = kwargs.get("goal")
        if goal is None and trace.metadata:
            goal = trace.metadata.get("conversation_goal")
        
        return self.evaluate_conversation(
            messages=trace.messages,
            goal=goal,
            ground_truth=ground_truth,
            metric_set=metric_set,
            dimension_weights=dimension_weights,
            thresholds=thresholds,
            context=context,
            **kwargs
        )
    
    def evaluate_context_retention(
        self,
        messages: List[Message],
        ground_truth: Optional[GroundTruth] = None
    ) -> MetricScore:
        """
        Evaluate context retention across conversation turns.
        
        Requirement 10.1: Assess context retention across turns
        Requirement 10.4: Verify information from earlier turns is used appropriately
        
        Args:
            messages: List of conversation messages
            ground_truth: Expected correct outputs (optional)
            
        Returns:
            MetricScore for context retention
        """
        logger.debug("Evaluating context retention")
        
        # Create trace and evaluation input
        import uuid
        trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            messages=messages,
            tool_calls=[],
            metadata={"num_turns": len(messages)}
        )
        
        evaluation_input = EvaluationInput(
            trace=trace,
            ground_truth=ground_truth,
            context=[]
        )
        
        # Get context retention metric
        try:
            metric = self.metric_registry.get_metric("context_retention")
            score = metric.calculate(evaluation_input)
            logger.debug(f"Context retention score: {score.score:.3f}")
            return score
        except Exception as e:
            logger.error(f"Failed to calculate context retention: {str(e)}")
            return MetricScore(
                metric_name="context_retention",
                score=0.0,
                reasoning=f"Error calculating context retention: {str(e)}"
            )
    
    def evaluate_coherence(
        self,
        messages: List[Message]
    ) -> MetricScore:
        """
        Evaluate coherence and logical flow of conversation.
        
        Requirement 10.2: Evaluate coherence by checking logical flow and consistency
        
        Args:
            messages: List of conversation messages
            
        Returns:
            MetricScore for coherence
        """
        logger.debug("Evaluating coherence")
        
        # Create trace and evaluation input
        import uuid
        trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            messages=messages,
            tool_calls=[],
            metadata={"num_turns": len(messages)}
        )
        
        evaluation_input = EvaluationInput(
            trace=trace,
            ground_truth=None,
            context=[]
        )
        
        # Get coherence metric
        try:
            metric = self.metric_registry.get_metric("coherence")
            score = metric.calculate(evaluation_input)
            logger.debug(f"Coherence score: {score.score:.3f}")
            return score
        except Exception as e:
            logger.error(f"Failed to calculate coherence: {str(e)}")
            return MetricScore(
                metric_name="coherence",
                score=0.0,
                reasoning=f"Error calculating coherence: {str(e)}"
            )
    
    def _get_default_multi_turn_metrics(self) -> List[str]:
        """
        Get default list of multi-turn metric names.
        
        Returns:
            List of metric names for multi-turn evaluation
        """
        return [
            "context_retention",
            "coherence",
            "conversation_completeness",
            "turn_efficiency"
        ]
    
    def _get_metrics_to_calculate(
        self,
        metric_set: Optional[Dict[str, List[str]]] = None
    ) -> List[any]:
        """
        Get list of metric instances to calculate.
        
        Args:
            metric_set: Dictionary mapping dimension names to lists of metric names.
                       If None, uses default multi-turn metrics.
            
        Returns:
            List of metric instances
        """
        metrics = []
        
        if metric_set is None:
            # Use default multi-turn metrics
            metric_names = self._get_default_multi_turn_metrics()
            for metric_name in metric_names:
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
    
    def _identify_unmet_criteria(
        self,
        metric_scores: List[MetricScore],
        goal: Optional[str],
        passed: bool
    ) -> List[str]:
        """
        Identify which success criteria were not met.
        
        Requirement 10.6: Identify which success criteria were not met
        
        Args:
            metric_scores: List of calculated metric scores
            goal: Conversation goal (optional)
            passed: Whether overall evaluation passed
            
        Returns:
            List of unmet criteria descriptions
        """
        unmet_criteria = []
        
        # Check if conversation was incomplete
        completeness_scores = [
            s for s in metric_scores 
            if s.metric_name == "conversation_completeness"
        ]
        
        if completeness_scores:
            completeness_score = completeness_scores[0]
            if completeness_score.score < 0.7:
                unmet_criteria.append(
                    f"Conversation completeness below threshold: "
                    f"{completeness_score.score:.3f} < 0.7"
                )
                if goal:
                    unmet_criteria.append(
                        f"Goal not achieved: {goal}"
                    )
        
        # Check context retention
        context_scores = [
            s for s in metric_scores 
            if s.metric_name == "context_retention"
        ]
        
        if context_scores:
            context_score = context_scores[0]
            if context_score.score < 0.6:
                unmet_criteria.append(
                    f"Context retention insufficient: "
                    f"{context_score.score:.3f} < 0.6"
                )
        
        # Check coherence
        coherence_scores = [
            s for s in metric_scores 
            if s.metric_name == "coherence"
        ]
        
        if coherence_scores:
            coherence_score = coherence_scores[0]
            if coherence_score.score < 0.6:
                unmet_criteria.append(
                    f"Coherence insufficient: "
                    f"{coherence_score.score:.3f} < 0.6"
                )
        
        return unmet_criteria
