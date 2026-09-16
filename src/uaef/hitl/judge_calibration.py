# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM judge calibration workflow for HITL validation.

This module implements the judge calibration workflow, one of three HITL workflows
in UAEF. It validates and calibrates LLM-based evaluation scores by comparing them
with human judgments.

The workflow:
1. Queues low-confidence LLM evaluations for human review
2. Collects human scores for the same evaluations
3. Calculates LLM-human agreement (correlation)
4. Uses feedback to adjust prompt templates
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from scipy import stats

from uaef.hitl.models import AgreementMetrics, ItemType, Priority, ReviewItem
from uaef.hitl.queue import ReviewQueue
from uaef.llm_judge.calibration import CalibrationEngine, CalibrationSample
from uaef.logging import get_logger
from uaef.models.metric_score import MetricScore

logger = get_logger(__name__)


class JudgeCalibrationError(Exception):
    """Base exception for judge calibration errors."""
    pass


class InsufficientDataError(JudgeCalibrationError):
    """Raised when insufficient data is available for calibration."""
    pass


class JudgeCalibrationWorkflow:
    """
    Workflow for calibrating LLM judge prompts using human feedback.
    
    This workflow validates and calibrates LLM-based evaluation scores by:
    1. Identifying low-confidence LLM evaluations that need human review
    2. Queuing them for human validation
    3. Collecting human scores for the same evaluations
    4. Calculating agreement metrics (correlation, MAE, etc.)
    5. Analyzing disagreement patterns
    6. Suggesting prompt improvements based on feedback
    
    The workflow integrates with:
    - ReviewQueue: Manages items needing calibration
    - CalibrationEngine: Performs statistical analysis and tracks performance
    
    Attributes:
        review_queue: ReviewQueue instance for managing review items
        calibration_engine: CalibrationEngine instance for analysis
        confidence_threshold: Threshold below which evaluations are queued (default: 0.7)
        disagreement_threshold: Threshold for identifying disagreements (default: 0.2)
    
    Examples:
        >>> # Initialize workflow
        >>> workflow = JudgeCalibrationWorkflow()
        
        >>> # Queue low-confidence evaluations
        >>> evaluation_results = [
        ...     {
        ...         "evaluation_id": "eval_001",
        ...         "question": "What is AI?",
        ...         "response": "AI is artificial intelligence...",
        ...         "metric_scores": [
        ...             MetricScore(name="answer_relevance", score=0.85, confidence=0.65),
        ...             MetricScore(name="answer_correctness", score=0.90, confidence=0.75)
        ...         ]
        ...     }
        ... ]
        >>> queued_items = workflow.queue_low_confidence_evaluations(
        ...     evaluation_results,
        ...     confidence_threshold=0.7
        ... )
        >>> len(queued_items)
        1
        
        >>> # After human review, calculate agreement
        >>> agreement = workflow.calculate_agreement(
        ...     metric_name="answer_relevance",
        ...     prompt_version="v1.0"
        ... )
        >>> agreement.agreement_score
        0.85
        
        >>> # Generate calibration report
        >>> report = workflow.generate_calibration_report(
        ...     metric_name="answer_relevance",
        ...     prompt_version="v1.0"
        ... )
        
        >>> # Get prompt improvement suggestions
        >>> suggestions = workflow.suggest_prompt_improvements(
        ...     metric_name="answer_relevance",
        ...     prompt_version="v1.0"
        ... )
    """
    
    def __init__(
        self,
        review_queue: Optional[ReviewQueue] = None,
        calibration_engine: Optional[CalibrationEngine] = None,
        confidence_threshold: float = 0.7,
        disagreement_threshold: float = 0.2
    ):
        """
        Initialize the judge calibration workflow.
        
        Args:
            review_queue: ReviewQueue instance (creates new if None)
            calibration_engine: CalibrationEngine instance (creates new if None)
            confidence_threshold: Threshold below which evaluations are queued
            disagreement_threshold: Threshold for identifying disagreements
        
        Raises:
            ValueError: If thresholds are not between 0 and 1
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, got {confidence_threshold}"
            )
        if not 0.0 <= disagreement_threshold <= 1.0:
            raise ValueError(
                f"disagreement_threshold must be between 0 and 1, got {disagreement_threshold}"
            )
        
        self.review_queue = review_queue or ReviewQueue()
        self.calibration_engine = calibration_engine or CalibrationEngine()
        self.confidence_threshold = confidence_threshold
        self.disagreement_threshold = disagreement_threshold
        
        logger.info(
            f"Initialized JudgeCalibrationWorkflow with "
            f"confidence_threshold={confidence_threshold}, "
            f"disagreement_threshold={disagreement_threshold}"
        )
    
    def queue_low_confidence_evaluations(
        self,
        evaluation_results: List[Dict[str, Any]],
        confidence_threshold: Optional[float] = None,
        auto_priority: bool = True
    ) -> List[ReviewItem]:
        """
        Queue low-confidence LLM evaluations for human review.
        
        Identifies evaluations with confidence scores below the threshold and
        adds them to the review queue for human validation.
        
        Args:
            evaluation_results: List of evaluation result dictionaries containing:
                - evaluation_id: Unique identifier for the evaluation
                - question: The question/input
                - response: The agent's response
                - metric_scores: List of MetricScore objects with confidence
                - context: Optional context information
                - ground_truth: Optional ground truth for comparison
            confidence_threshold: Override default confidence threshold
            auto_priority: If True, automatically assign priority based on confidence
        
        Returns:
            List of ReviewItem objects that were added to the queue
        
        Raises:
            ValueError: If evaluation_results format is invalid
        
        Examples:
            >>> workflow = JudgeCalibrationWorkflow()
            >>> results = [
            ...     {
            ...         "evaluation_id": "eval_001",
            ...         "question": "What is AI?",
            ...         "response": "AI is...",
            ...         "metric_scores": [
            ...             MetricScore(name="relevance", score=0.85, confidence=0.65)
            ...         ]
            ...     }
            ... ]
            >>> items = workflow.queue_low_confidence_evaluations(results)
            >>> len(items)
            1
            >>> items[0].confidence_score
            0.65
        """
        threshold = confidence_threshold or self.confidence_threshold
        queued_items = []
        
        for result in evaluation_results:
            # Validate result format
            if "evaluation_id" not in result:
                raise ValueError("evaluation_result must contain 'evaluation_id'")
            if "metric_scores" not in result:
                raise ValueError("evaluation_result must contain 'metric_scores'")
            
            evaluation_id = result["evaluation_id"]
            metric_scores = result["metric_scores"]
            
            # Check each metric score for low confidence
            for metric_score in metric_scores:
                # Get confidence (default to 1.0 if not present)
                confidence = getattr(metric_score, "confidence", 1.0)
                
                # Queue if confidence is below threshold
                if confidence < threshold:
                    # Create review item
                    review_item = ReviewItem(
                        item_type=ItemType.LLM_EVALUATION,
                        content={
                            "question": result.get("question", ""),
                            "agent_response": result.get("response", ""),
                            "llm_scores": {
                                metric_score.name: metric_score.score
                            },
                            "llm_reasoning": getattr(metric_score, "reasoning", ""),
                            "context": result.get("context", ""),
                            "ground_truth": result.get("ground_truth", None)
                        },
                        metadata={
                            "evaluation_id": evaluation_id,
                            "metric_name": metric_score.name,
                            "prompt_version": getattr(metric_score, "prompt_version", "unknown")
                        },
                        priority=Priority.MEDIUM,  # Will be overridden if auto_priority=True
                        confidence_score=confidence
                    )
                    
                    # Add to queue
                    added_item = self.review_queue.add_to_queue(
                        review_item,
                        auto_priority=auto_priority
                    )
                    queued_items.append(added_item)
                    
                    logger.debug(
                        f"Queued evaluation {evaluation_id} for metric "
                        f"{metric_score.name} with confidence {confidence:.3f}"
                    )
        
        logger.info(
            f"Queued {len(queued_items)} low-confidence evaluations "
            f"(threshold={threshold:.2f})"
        )
        
        return queued_items
    
    def calculate_agreement(
        self,
        metric_name: str,
        prompt_version: str,
        min_samples: int = 2
    ) -> AgreementMetrics:
        """
        Calculate agreement between LLM and human scores.
        
        Computes correlation metrics (Pearson, Spearman) and error metrics (MAE, RMSE)
        to measure how well LLM scores align with human judgments.
        
        Args:
            metric_name: Name of the metric to analyze
            prompt_version: Version of the prompt used
            min_samples: Minimum number of samples required for calculation
        
        Returns:
            AgreementMetrics object with correlation and error metrics
        
        Raises:
            InsufficientDataError: If fewer than min_samples are available
        
        Examples:
            >>> workflow = JudgeCalibrationWorkflow()
            >>> # After collecting human reviews...
            >>> agreement = workflow.calculate_agreement(
            ...     metric_name="answer_relevance",
            ...     prompt_version="v1.0"
            ... )
            >>> agreement.agreement_score  # Pearson correlation
            0.85
            >>> agreement.disagreement_details["mean_delta"]
            0.12
        """
        # Get completed review items for this metric
        completed_items = self.review_queue.get_all_items(
            item_type=ItemType.LLM_EVALUATION
        )
        
        # Filter by metric name and prompt version
        relevant_items = [
            item for item in completed_items
            if (
                item.metadata.get("metric_name") == metric_name and
                item.metadata.get("prompt_version") == prompt_version and
                item.human_scores  # Has human scores
            )
        ]
        
        if len(relevant_items) < min_samples:
            raise InsufficientDataError(
                f"Insufficient samples for agreement calculation: "
                f"need at least {min_samples}, got {len(relevant_items)}"
            )
        
        # Extract LLM and human scores
        llm_scores = []
        human_scores_list = []
        
        for item in relevant_items:
            # Get LLM score from content
            llm_score = item.content.get("llm_scores", {}).get(metric_name)
            if llm_score is None:
                continue
            
            # Get human score
            human_score = item.human_scores.get(metric_name)
            if human_score is None:
                continue
            
            llm_scores.append(llm_score)
            human_scores_list.append(human_score)
            
            # Add to calibration engine for tracking
            self.calibration_engine.add_sample(
                sample_id=str(item.item_id),
                metric_name=metric_name,
                prompt_version=prompt_version,
                llm_score=llm_score,
                human_score=human_score,
                question=item.content.get("question", ""),
                response=item.content.get("agent_response", ""),
                llm_reasoning=item.content.get("llm_reasoning", ""),
                human_reasoning=item.feedback
            )
        
        if len(llm_scores) < min_samples:
            raise InsufficientDataError(
                f"Insufficient valid score pairs: "
                f"need at least {min_samples}, got {len(llm_scores)}"
            )
        
        # Convert to numpy arrays
        llm_scores_arr = np.array(llm_scores)
        human_scores_arr = np.array(human_scores_list)
        
        # Calculate Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(llm_scores_arr, human_scores_arr)
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(llm_scores_arr, human_scores_arr)
        
        # Calculate error metrics
        score_diffs = np.abs(llm_scores_arr - human_scores_arr)
        mae = np.mean(score_diffs)
        rmse = np.sqrt(np.mean((llm_scores_arr - human_scores_arr) ** 2))
        
        # Calculate agreement rate (within disagreement threshold)
        agreements = score_diffs <= self.disagreement_threshold
        agreement_rate = np.mean(agreements)
        
        # Identify disagreement samples
        disagreement_indices = np.where(score_diffs > self.disagreement_threshold)[0]
        
        # Create AgreementMetrics
        agreement_metrics = AgreementMetrics(
            metric_name=metric_name,
            llm_score=None,  # Not applicable for aggregate metrics
            human_scores=human_scores_list,
            agreement_score=float(pearson_corr),
            disagreement_details={
                "sample_count": len(llm_scores),
                "pearson_correlation": float(pearson_corr),
                "pearson_p_value": float(pearson_p),
                "spearman_correlation": float(spearman_corr),
                "spearman_p_value": float(spearman_p),
                "mean_absolute_error": float(mae),
                "root_mean_squared_error": float(rmse),
                "agreement_rate": float(agreement_rate),
                "disagreement_count": len(disagreement_indices),
                "mean_delta": float(np.mean(score_diffs)),
                "max_delta": float(np.max(score_diffs)),
                "std_delta": float(np.std(score_diffs)),
                "prompt_version": prompt_version
            }
        )
        
        logger.info(
            f"Calculated agreement for {metric_name} v{prompt_version}: "
            f"pearson={pearson_corr:.3f}, mae={mae:.3f}, "
            f"agreement_rate={agreement_rate:.1%}"
        )
        
        return agreement_metrics
    
    def generate_calibration_report(
        self,
        metric_name: str,
        prompt_version: str
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive calibration report.
        
        The report includes:
        - Agreement metrics (correlation, MAE, RMSE)
        - Disagreement patterns and analysis
        - Sample statistics
        - Prompt improvement suggestions
        
        Args:
            metric_name: Name of the metric to analyze
            prompt_version: Version of the prompt used
        
        Returns:
            Dictionary containing the calibration report
        
        Raises:
            InsufficientDataError: If insufficient data for analysis
        
        Examples:
            >>> workflow = JudgeCalibrationWorkflow()
            >>> report = workflow.generate_calibration_report(
            ...     metric_name="answer_relevance",
            ...     prompt_version="v1.0"
            ... )
            >>> report["agreement_metrics"]["pearson_correlation"]
            0.85
            >>> report["is_well_calibrated"]
            True
            >>> len(report["improvement_suggestions"])
            3
        """
        try:
            # Calculate agreement metrics
            agreement = self.calculate_agreement(metric_name, prompt_version)
            
            # Get disagreement analysis from calibration engine
            disagreement_analysis = self.calibration_engine.analyze_disagreements(
                metric_name=metric_name,
                prompt_version=prompt_version,
                disagreement_threshold=self.disagreement_threshold
            )
            
            # Get improvement suggestions
            suggestions = self.calibration_engine.suggest_improvements(
                metric_name=metric_name,
                prompt_version=prompt_version,
                disagreement_threshold=self.disagreement_threshold
            )
            
            # Get calibration metrics from engine
            calibration_metrics = self.calibration_engine.calculate_metrics(
                metric_name=metric_name,
                prompt_version=prompt_version,
                disagreement_threshold=self.disagreement_threshold
            )
            
            # Build report
            report = {
                "metric_name": metric_name,
                "prompt_version": prompt_version,
                "timestamp": datetime.utcnow().isoformat(),
                "agreement_metrics": {
                    "pearson_correlation": agreement.disagreement_details["pearson_correlation"],
                    "pearson_p_value": agreement.disagreement_details["pearson_p_value"],
                    "spearman_correlation": agreement.disagreement_details["spearman_correlation"],
                    "spearman_p_value": agreement.disagreement_details["spearman_p_value"],
                    "mean_absolute_error": agreement.disagreement_details["mean_absolute_error"],
                    "root_mean_squared_error": agreement.disagreement_details["root_mean_squared_error"],
                    "agreement_rate": agreement.disagreement_details["agreement_rate"]
                },
                "sample_statistics": {
                    "total_samples": agreement.disagreement_details["sample_count"],
                    "disagreement_count": agreement.disagreement_details["disagreement_count"],
                    "mean_delta": agreement.disagreement_details["mean_delta"],
                    "max_delta": agreement.disagreement_details["max_delta"],
                    "std_delta": agreement.disagreement_details["std_delta"]
                },
                "disagreement_analysis": disagreement_analysis,
                "improvement_suggestions": suggestions,
                "is_well_calibrated": calibration_metrics.is_well_calibrated(),
                "calibration_thresholds": {
                    "confidence_threshold": self.confidence_threshold,
                    "disagreement_threshold": self.disagreement_threshold
                }
            }
            
            logger.info(
                f"Generated calibration report for {metric_name} v{prompt_version}: "
                f"well_calibrated={report['is_well_calibrated']}, "
                f"samples={report['sample_statistics']['total_samples']}"
            )
            
            return report
            
        except InsufficientDataError as e:
            logger.warning(f"Cannot generate report: {e}")
            raise
    
    def suggest_prompt_improvements(
        self,
        metric_name: str,
        prompt_version: str
    ) -> List[str]:
        """
        Suggest prompt improvements based on disagreement patterns.
        
        Analyzes disagreements between LLM and human scores to identify
        systematic issues and recommend specific prompt modifications.
        
        Args:
            metric_name: Name of the metric to analyze
            prompt_version: Version of the prompt used
        
        Returns:
            List of actionable improvement suggestions
        
        Examples:
            >>> workflow = JudgeCalibrationWorkflow()
            >>> suggestions = workflow.suggest_prompt_improvements(
            ...     metric_name="answer_relevance",
            ...     prompt_version="v1.0"
            ... )
            >>> suggestions[0]
            'LLM tends to overscore (avg difference: 0.15). Add stricter criteria...'
        """
        suggestions = self.calibration_engine.suggest_improvements(
            metric_name=metric_name,
            prompt_version=prompt_version,
            disagreement_threshold=self.disagreement_threshold
        )
        
        logger.info(
            f"Generated {len(suggestions)} improvement suggestions for "
            f"{metric_name} v{prompt_version}"
        )
        
        return suggestions
    
    def get_calibration_status(
        self,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current calibration status across metrics.
        
        Provides an overview of:
        - Number of items in review queue
        - Number of completed reviews
        - Calibration metrics for each metric/version
        - Overall calibration health
        
        Args:
            metric_name: Optional filter by metric name
        
        Returns:
            Dictionary with calibration status information
        
        Examples:
            >>> workflow = JudgeCalibrationWorkflow()
            >>> status = workflow.get_calibration_status()
            >>> status["queue_stats"]["pending"]
            5
            >>> status["metrics"]["answer_relevance"]["is_well_calibrated"]
            True
        """
        # Get queue statistics
        queue_stats = self.review_queue.get_queue_stats()
        
        # Filter for LLM evaluation items only
        llm_eval_items = self.review_queue.get_all_items(
            item_type=ItemType.LLM_EVALUATION
        )
        
        # Group by metric name
        metrics_status = {}
        metric_names = set(
            item.metadata.get("metric_name")
            for item in llm_eval_items
            if item.metadata.get("metric_name")
        )
        
        # Filter by metric_name if provided
        if metric_name:
            metric_names = {metric_name} if metric_name in metric_names else set()
        
        for m_name in metric_names:
            # Get items for this metric
            metric_items = [
                item for item in llm_eval_items
                if item.metadata.get("metric_name") == m_name
            ]
            
            # Get unique prompt versions
            prompt_versions = set(
                item.metadata.get("prompt_version")
                for item in metric_items
                if item.metadata.get("prompt_version")
            )
            
            versions_status = {}
            for version in prompt_versions:
                version_items = [
                    item for item in metric_items
                    if item.metadata.get("prompt_version") == version
                ]
                
                completed_items = [
                    item for item in version_items
                    if item.human_scores
                ]
                
                # Try to get calibration metrics
                try:
                    calibration_metrics = self.calibration_engine.calculate_metrics(
                        metric_name=m_name,
                        prompt_version=version,
                        disagreement_threshold=self.disagreement_threshold
                    )
                    
                    versions_status[version] = {
                        "total_items": len(version_items),
                        "completed_items": len(completed_items),
                        "pending_items": len(version_items) - len(completed_items),
                        "pearson_correlation": calibration_metrics.pearson_correlation,
                        "mean_absolute_error": calibration_metrics.mean_absolute_error,
                        "agreement_rate": calibration_metrics.agreement_rate,
                        "is_well_calibrated": calibration_metrics.is_well_calibrated()
                    }
                except (InsufficientDataError, ValueError):
                    versions_status[version] = {
                        "total_items": len(version_items),
                        "completed_items": len(completed_items),
                        "pending_items": len(version_items) - len(completed_items),
                        "status": "insufficient_data"
                    }
            
            metrics_status[m_name] = versions_status
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "queue_stats": {
                "total_llm_evaluations": len(llm_eval_items),
                "pending": queue_stats["by_type"]["llm_evaluation"],
                "completed": sum(
                    1 for item in llm_eval_items
                    if item.human_scores
                )
            },
            "metrics": metrics_status,
            "thresholds": {
                "confidence_threshold": self.confidence_threshold,
                "disagreement_threshold": self.disagreement_threshold
            }
        }
