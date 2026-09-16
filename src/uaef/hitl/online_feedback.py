# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online evaluation feedback workflow for HITL validation.

This module implements the online evaluation feedback workflow, one of three HITL
workflows in UAEF. It collects human feedback on agent responses in real-time during
production use.

The workflow:
1. Collects agent responses from online evaluation mode
2. Queues responses for human feedback (thumbs up/down or detailed scores)
3. Aggregates feedback across multiple reviewers
4. Tracks feedback trends over time to identify systematic issues
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from uaef.hitl.models import ItemType, Priority, ReviewItem
from uaef.hitl.queue import ReviewQueue
from uaef.logging import get_logger

logger = get_logger(__name__)


class OnlineFeedbackError(Exception):
    """Base exception for online feedback errors."""
    pass


class InsufficientFeedbackError(OnlineFeedbackError):
    """Raised when insufficient feedback is available for analysis."""
    pass


class FeedbackTrend:
    """
    Represents a trend in feedback over time.
    
    Attributes:
        metric_name: Name of the metric being tracked
        time_period: Time period for the trend (e.g., "last_7_days")
        average_score: Average score over the period
        score_change: Change in score compared to previous period
        feedback_count: Number of feedback items in the period
        trend_direction: Direction of trend ("improving", "declining", "stable")
        issues_detected: List of detected issues or patterns
    """
    
    def __init__(
        self,
        metric_name: str,
        time_period: str,
        average_score: float,
        score_change: float,
        feedback_count: int,
        trend_direction: str,
        issues_detected: List[str]
    ):
        """Initialize feedback trend."""
        self.metric_name = metric_name
        self.time_period = time_period
        self.average_score = average_score
        self.score_change = score_change
        self.feedback_count = feedback_count
        self.trend_direction = trend_direction
        self.issues_detected = issues_detected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "time_period": self.time_period,
            "average_score": self.average_score,
            "score_change": self.score_change,
            "feedback_count": self.feedback_count,
            "trend_direction": self.trend_direction,
            "issues_detected": self.issues_detected
        }


class OnlineFeedbackWorkflow:
    """
    Workflow for collecting human feedback on agent responses in real-time.
    
    This workflow collects and analyzes human feedback on agent responses during
    online evaluation mode. It supports:
    1. Simple feedback (thumbs up/down)
    2. Detailed feedback (metric scores 0-1)
    3. Aggregation across multiple reviewers
    4. Trend tracking over time
    
    The workflow integrates with:
    - ReviewQueue: Manages items needing feedback
    - OnlineEvaluator: Provides agent responses for feedback
    
    Attributes:
        review_queue: ReviewQueue instance for managing review items
        simple_feedback_mapping: Mapping from thumbs up/down to scores
        consensus_threshold: Threshold for reviewer agreement (default: 0.7)
    
    Examples:
        >>> # Initialize workflow
        >>> workflow = OnlineFeedbackWorkflow()
        
        >>> # Collect feedback on an agent response
        >>> response_data = {
        ...     "trace_id": "trace_001",
        ...     "user_query": "Book a flight to NYC",
        ...     "agent_response": "I found 3 flights...",
        ...     "tools_called": ["search_flights", "get_prices"],
        ...     "evaluation_scores": {"answer_relevance": 0.85}
        ... }
        >>> item = workflow.collect_feedback(response_data)
        
        >>> # Submit simple feedback (thumbs up/down)
        >>> workflow.submit_feedback(
        ...     item_id=item.item_id,
        ...     feedback_type="simple",
        ...     simple_rating="thumbs_up",
        ...     reviewer_id="user_123"
        ... )
        
        >>> # Submit detailed feedback (metric scores)
        >>> workflow.submit_feedback(
        ...     item_id=item.item_id,
        ...     feedback_type="detailed",
        ...     detailed_scores={
        ...         "answer_relevance": 0.90,
        ...         "answer_correctness": 0.85,
        ...         "helpfulness": 0.95
        ...     },
        ...     reviewer_id="reviewer_456",
        ...     feedback_text="Good response but could include pricing details"
        ... )
        
        >>> # Aggregate feedback across reviewers
        >>> aggregated = workflow.aggregate_feedback(item.item_id)
        >>> aggregated["consensus_scores"]
        {'answer_relevance': 0.875, 'answer_correctness': 0.85, 'helpfulness': 0.95}
        
        >>> # Get feedback trends
        >>> trends = workflow.get_feedback_trends(
        ...     time_period_days=7,
        ...     metric_name="answer_relevance"
        ... )
        >>> trends[0].trend_direction
        'improving'
    """
    
    def __init__(
        self,
        review_queue: Optional[ReviewQueue] = None,
        simple_feedback_mapping: Optional[Dict[str, float]] = None,
        consensus_threshold: float = 0.7
    ):
        """
        Initialize the online feedback workflow.
        
        Args:
            review_queue: ReviewQueue instance (creates new if None)
            simple_feedback_mapping: Mapping from simple ratings to scores
                Default: {"thumbs_up": 1.0, "thumbs_down": 0.0}
            consensus_threshold: Threshold for reviewer agreement (0-1)
        
        Raises:
            ValueError: If consensus_threshold is not between 0 and 1
        """
        if not 0.0 <= consensus_threshold <= 1.0:
            raise ValueError(
                f"consensus_threshold must be between 0 and 1, got {consensus_threshold}"
            )
        
        self.review_queue = review_queue or ReviewQueue()
        self.simple_feedback_mapping = simple_feedback_mapping or {
            "thumbs_up": 1.0,
            "thumbs_down": 0.0
        }
        self.consensus_threshold = consensus_threshold
        
        # Storage for multiple reviewer feedback (item_id -> list of feedback)
        self._feedback_history: Dict[UUID, List[Dict[str, Any]]] = {}
        
        logger.info(
            f"Initialized OnlineFeedbackWorkflow with "
            f"consensus_threshold={consensus_threshold}"
        )
    
    def collect_feedback(
        self,
        response_data: Dict[str, Any],
        auto_priority: bool = True
    ) -> ReviewItem:
        """
        Collect an agent response for human feedback.
        
        Adds an agent response to the review queue for human feedback collection.
        This is typically called after online evaluation completes.
        
        Args:
            response_data: Dictionary containing:
                - trace_id: Unique identifier for the trace
                - user_query: The user's input/question
                - agent_response: The agent's response
                - tools_called: List of tools the agent called (optional)
                - evaluation_scores: Dict of automated evaluation scores (optional)
                - context: Additional context information (optional)
                - confidence_score: Confidence in the response (optional, default: 0.5)
            auto_priority: If True, automatically assign priority based on confidence
        
        Returns:
            ReviewItem that was added to the queue
        
        Raises:
            ValueError: If response_data format is invalid
        
        Examples:
            >>> workflow = OnlineFeedbackWorkflow()
            >>> response = {
            ...     "trace_id": "trace_001",
            ...     "user_query": "What is AI?",
            ...     "agent_response": "AI is artificial intelligence...",
            ...     "evaluation_scores": {"answer_relevance": 0.85},
            ...     "confidence_score": 0.75
            ... }
            >>> item = workflow.collect_feedback(response)
            >>> item.item_type
            <ItemType.AGENT_RESPONSE: 'agent_response'>
        """
        # Validate required fields
        if "trace_id" not in response_data:
            raise ValueError("response_data must contain 'trace_id'")
        if "user_query" not in response_data:
            raise ValueError("response_data must contain 'user_query'")
        if "agent_response" not in response_data:
            raise ValueError("response_data must contain 'agent_response'")
        
        trace_id = response_data["trace_id"]
        confidence_score = response_data.get("confidence_score", 0.5)
        
        # Create review item
        review_item = ReviewItem(
            item_type=ItemType.AGENT_RESPONSE,
            content={
                "user_query": response_data["user_query"],
                "agent_response": response_data["agent_response"],
                "tools_called": response_data.get("tools_called", []),
                "evaluation_scores": response_data.get("evaluation_scores", {}),
                "context": response_data.get("context", "")
            },
            metadata={
                "trace_id": str(trace_id),
                "timestamp": datetime.utcnow().isoformat()
            },
            priority=Priority.MEDIUM,  # Will be overridden if auto_priority=True
            confidence_score=confidence_score
        )
        
        # Add to queue
        added_item = self.review_queue.add_to_queue(
            review_item,
            auto_priority=auto_priority
        )
        
        # Initialize feedback history for this item
        self._feedback_history[added_item.item_id] = []
        
        logger.debug(
            f"Collected agent response for feedback: trace_id={trace_id}, "
            f"confidence={confidence_score:.3f}"
        )
        
        return added_item
    
    def submit_feedback(
        self,
        item_id: UUID,
        feedback_type: str,
        reviewer_id: str,
        simple_rating: Optional[str] = None,
        detailed_scores: Optional[Dict[str, float]] = None,
        feedback_text: Optional[str] = None
    ) -> ReviewItem:
        """
        Submit human feedback for an agent response.
        
        Supports two types of feedback:
        1. Simple: thumbs up/down rating
        2. Detailed: metric scores (0-1) for multiple dimensions
        
        Args:
            item_id: UUID of the item to provide feedback on
            feedback_type: Type of feedback ("simple" or "detailed")
            reviewer_id: ID of the human reviewer
            simple_rating: Simple rating ("thumbs_up" or "thumbs_down")
                Required if feedback_type="simple"
            detailed_scores: Dictionary of metric_name -> score (0-1)
                Required if feedback_type="detailed"
            feedback_text: Optional textual feedback
        
        Returns:
            Updated ReviewItem
        
        Raises:
            ValueError: If feedback format is invalid
            OnlineFeedbackError: If item not found or invalid operation
        
        Examples:
            >>> workflow = OnlineFeedbackWorkflow()
            >>> # Simple feedback
            >>> item = workflow.submit_feedback(
            ...     item_id=some_uuid,
            ...     feedback_type="simple",
            ...     simple_rating="thumbs_up",
            ...     reviewer_id="user_123"
            ... )
            
            >>> # Detailed feedback
            >>> item = workflow.submit_feedback(
            ...     item_id=some_uuid,
            ...     feedback_type="detailed",
            ...     detailed_scores={
            ...         "answer_relevance": 0.90,
            ...         "helpfulness": 0.85
            ...     },
            ...     reviewer_id="reviewer_456",
            ...     feedback_text="Good response"
            ... )
        """
        # Validate feedback_type
        if feedback_type not in ["simple", "detailed"]:
            raise ValueError(
                f"feedback_type must be 'simple' or 'detailed', got '{feedback_type}'"
            )
        
        # Validate simple feedback
        if feedback_type == "simple":
            if simple_rating is None:
                raise ValueError("simple_rating is required for simple feedback")
            if simple_rating not in self.simple_feedback_mapping:
                raise ValueError(
                    f"simple_rating must be one of {list(self.simple_feedback_mapping.keys())}, "
                    f"got '{simple_rating}'"
                )
        
        # Validate detailed feedback
        if feedback_type == "detailed":
            if detailed_scores is None or not detailed_scores:
                raise ValueError("detailed_scores is required for detailed feedback")
            for metric_name, score in detailed_scores.items():
                if not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"Score for '{metric_name}' must be between 0 and 1, got {score}"
                    )
        
        # Get the item
        try:
            item = self.review_queue.get_item(item_id)
        except Exception as e:
            raise OnlineFeedbackError(f"Failed to get item {item_id}: {e}")
        
        # Convert simple rating to scores
        if feedback_type == "simple":
            human_scores = {
                "overall_rating": self.simple_feedback_mapping[simple_rating]
            }
        else:
            human_scores = detailed_scores
        
        # Store feedback in history (for aggregation across multiple reviewers)
        feedback_entry = {
            "reviewer_id": reviewer_id,
            "timestamp": datetime.utcnow().isoformat(),
            "feedback_type": feedback_type,
            "scores": human_scores,
            "feedback_text": feedback_text
        }
        
        if item_id not in self._feedback_history:
            self._feedback_history[item_id] = []
        self._feedback_history[item_id].append(feedback_entry)
        
        # Update the review item with the latest feedback
        # Note: For multiple reviewers, we'll aggregate in aggregate_feedback()
        try:
            updated_item = self.review_queue.submit_review(
                item_id=item_id,
                human_scores=human_scores,
                reviewer_id=reviewer_id,
                feedback=feedback_text
            )
        except Exception as e:
            raise OnlineFeedbackError(f"Failed to submit review: {e}")
        
        logger.info(
            f"Submitted {feedback_type} feedback for item {item_id} "
            f"from reviewer {reviewer_id}"
        )
        
        return updated_item
    
    def aggregate_feedback(
        self,
        item_id: UUID,
        aggregation_method: str = "mean"
    ) -> Dict[str, Any]:
        """
        Aggregate feedback across multiple reviewers.
        
        Combines feedback from multiple reviewers to produce consensus scores
        and identify areas of agreement/disagreement.
        
        Args:
            item_id: UUID of the item to aggregate feedback for
            aggregation_method: Method for aggregation ("mean", "median", "weighted")
        
        Returns:
            Dictionary containing:
                - consensus_scores: Aggregated scores for each metric
                - reviewer_count: Number of reviewers
                - score_variance: Variance in scores across reviewers
                - agreement_level: Level of agreement ("high", "medium", "low")
                - individual_feedback: List of individual feedback entries
        
        Raises:
            OnlineFeedbackError: If item not found
            InsufficientFeedbackError: If no feedback available
        
        Examples:
            >>> workflow = OnlineFeedbackWorkflow()
            >>> # After multiple reviewers submit feedback...
            >>> aggregated = workflow.aggregate_feedback(item_id)
            >>> aggregated["consensus_scores"]
            {'answer_relevance': 0.875, 'helpfulness': 0.90}
            >>> aggregated["agreement_level"]
            'high'
        """
        # Validate aggregation method
        if aggregation_method not in ["mean", "median", "weighted"]:
            raise ValueError(
                f"aggregation_method must be 'mean', 'median', or 'weighted', "
                f"got '{aggregation_method}'"
            )
        
        # Get feedback history
        if item_id not in self._feedback_history:
            raise OnlineFeedbackError(f"No feedback history found for item {item_id}")
        
        feedback_entries = self._feedback_history[item_id]
        
        if not feedback_entries:
            raise InsufficientFeedbackError(
                f"No feedback available for item {item_id}"
            )
        
        # Collect all scores by metric
        scores_by_metric: Dict[str, List[float]] = {}
        
        for entry in feedback_entries:
            for metric_name, score in entry["scores"].items():
                if metric_name not in scores_by_metric:
                    scores_by_metric[metric_name] = []
                scores_by_metric[metric_name].append(score)
        
        # Aggregate scores
        consensus_scores = {}
        score_variance = {}
        
        for metric_name, scores in scores_by_metric.items():
            scores_arr = np.array(scores)
            
            if aggregation_method == "mean":
                consensus_scores[metric_name] = float(np.mean(scores_arr))
            elif aggregation_method == "median":
                consensus_scores[metric_name] = float(np.median(scores_arr))
            elif aggregation_method == "weighted":
                # For weighted, use recency weighting (more recent = higher weight)
                weights = np.linspace(0.5, 1.0, len(scores))
                consensus_scores[metric_name] = float(
                    np.average(scores_arr, weights=weights)
                )
            
            # Calculate variance
            score_variance[metric_name] = float(np.var(scores_arr))
        
        # Calculate overall agreement level
        avg_variance = np.mean(list(score_variance.values())) if score_variance else 0.0
        
        if avg_variance < 0.01:
            agreement_level = "high"
        elif avg_variance < 0.05:
            agreement_level = "medium"
        else:
            agreement_level = "low"
        
        # Build result
        result = {
            "item_id": str(item_id),
            "consensus_scores": consensus_scores,
            "reviewer_count": len(feedback_entries),
            "score_variance": score_variance,
            "average_variance": float(avg_variance),
            "agreement_level": agreement_level,
            "aggregation_method": aggregation_method,
            "individual_feedback": feedback_entries,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Aggregated feedback for item {item_id}: "
            f"{len(feedback_entries)} reviewers, "
            f"agreement={agreement_level}"
        )
        
        return result
    
    def get_feedback_trends(
        self,
        time_period_days: int = 7,
        metric_name: Optional[str] = None,
        min_feedback_count: int = 3
    ) -> List[FeedbackTrend]:
        """
        Analyze feedback trends over time.
        
        Tracks how feedback scores change over time to identify:
        - Improving or declining quality
        - Systematic issues
        - Patterns in user satisfaction
        
        Args:
            time_period_days: Number of days to analyze
            metric_name: Optional filter by specific metric
            min_feedback_count: Minimum feedback items required for trend
        
        Returns:
            List of FeedbackTrend objects
        
        Raises:
            InsufficientFeedbackError: If insufficient feedback for analysis
        
        Examples:
            >>> workflow = OnlineFeedbackWorkflow()
            >>> trends = workflow.get_feedback_trends(
            ...     time_period_days=7,
            ...     metric_name="answer_relevance"
            ... )
            >>> trends[0].trend_direction
            'improving'
            >>> trends[0].average_score
            0.85
            >>> trends[0].score_change
            0.05
        """
        # Get all agent response items
        all_items = self.review_queue.get_all_items(
            item_type=ItemType.AGENT_RESPONSE
        )
        
        # Filter by time period
        cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
        recent_items = [
            item for item in all_items
            if item.created_at >= cutoff_date and item.human_scores
        ]
        
        if len(recent_items) < min_feedback_count:
            raise InsufficientFeedbackError(
                f"Insufficient feedback for trend analysis: "
                f"need at least {min_feedback_count}, got {len(recent_items)}"
            )
        
        # Collect scores by metric
        scores_by_metric: Dict[str, List[Tuple[datetime, float]]] = {}
        
        for item in recent_items:
            for m_name, score in item.human_scores.items():
                # Filter by metric_name if specified
                if metric_name and m_name != metric_name:
                    continue
                
                if m_name not in scores_by_metric:
                    scores_by_metric[m_name] = []
                scores_by_metric[m_name].append((item.created_at, score))
        
        # Analyze trends for each metric
        trends = []
        
        for m_name, score_data in scores_by_metric.items():
            if len(score_data) < min_feedback_count:
                continue
            
            # Sort by timestamp
            score_data.sort(key=lambda x: x[0])
            
            # Extract scores
            scores = [score for _, score in score_data]
            scores_arr = np.array(scores)
            
            # Calculate average score
            avg_score = float(np.mean(scores_arr))
            
            # Calculate trend (compare first half vs second half)
            mid_point = len(scores) // 2
            first_half_avg = float(np.mean(scores_arr[:mid_point]))
            second_half_avg = float(np.mean(scores_arr[mid_point:]))
            score_change = second_half_avg - first_half_avg
            
            # Determine trend direction
            if score_change > 0.05:
                trend_direction = "improving"
            elif score_change < -0.05:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
            
            # Detect issues
            issues_detected = []
            
            # Check for low scores
            low_score_count = np.sum(scores_arr < 0.5)
            if low_score_count > len(scores) * 0.3:
                issues_detected.append(
                    f"High rate of low scores: {low_score_count}/{len(scores)} "
                    f"({low_score_count/len(scores):.1%})"
                )
            
            # Check for declining trend
            if trend_direction == "declining":
                issues_detected.append(
                    f"Declining trend detected: score dropped by {abs(score_change):.3f}"
                )
            
            # Check for high variance
            score_variance = float(np.var(scores_arr))
            if score_variance > 0.1:
                issues_detected.append(
                    f"High variance in scores: {score_variance:.3f} "
                    "(inconsistent quality)"
                )
            
            # Create trend object
            trend = FeedbackTrend(
                metric_name=m_name,
                time_period=f"last_{time_period_days}_days",
                average_score=avg_score,
                score_change=score_change,
                feedback_count=len(scores),
                trend_direction=trend_direction,
                issues_detected=issues_detected
            )
            
            trends.append(trend)
            
            logger.debug(
                f"Trend for {m_name}: avg={avg_score:.3f}, "
                f"change={score_change:+.3f}, direction={trend_direction}"
            )
        
        if not trends:
            raise InsufficientFeedbackError(
                f"No trends available for the specified criteria"
            )
        
        logger.info(
            f"Analyzed feedback trends over {time_period_days} days: "
            f"{len(trends)} metrics analyzed"
        )
        
        return trends
    
    def get_feedback_summary(
        self,
        time_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of feedback collection status.
        
        Provides an overview of:
        - Total feedback collected
        - Feedback by type (simple vs detailed)
        - Average scores by metric
        - Recent trends
        
        Args:
            time_period_days: Optional time period for filtering (default: all time)
        
        Returns:
            Dictionary with feedback summary
        
        Examples:
            >>> workflow = OnlineFeedbackWorkflow()
            >>> summary = workflow.get_feedback_summary(time_period_days=7)
            >>> summary["total_feedback_items"]
            42
            >>> summary["average_scores"]["answer_relevance"]
            0.85
        """
        # Get all agent response items
        all_items = self.review_queue.get_all_items(
            item_type=ItemType.AGENT_RESPONSE
        )
        
        # Filter by time period if specified
        if time_period_days:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            items = [item for item in all_items if item.created_at >= cutoff_date]
        else:
            items = all_items
        
        # Count feedback by type
        simple_feedback_count = 0
        detailed_feedback_count = 0
        
        for item_id, feedback_entries in self._feedback_history.items():
            for entry in feedback_entries:
                if entry["feedback_type"] == "simple":
                    simple_feedback_count += 1
                else:
                    detailed_feedback_count += 1
        
        # Calculate average scores by metric
        scores_by_metric: Dict[str, List[float]] = {}
        
        for item in items:
            if item.human_scores:
                for metric_name, score in item.human_scores.items():
                    if metric_name not in scores_by_metric:
                        scores_by_metric[metric_name] = []
                    scores_by_metric[metric_name].append(score)
        
        average_scores = {
            metric_name: float(np.mean(scores))
            for metric_name, scores in scores_by_metric.items()
        }
        
        # Get queue stats
        queue_stats = self.review_queue.get_queue_stats()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "time_period": f"last_{time_period_days}_days" if time_period_days else "all_time",
            "total_feedback_items": len(items),
            "pending_feedback": queue_stats["by_type"]["agent_response"],
            "completed_feedback": len([item for item in items if item.human_scores]),
            "feedback_by_type": {
                "simple": simple_feedback_count,
                "detailed": detailed_feedback_count
            },
            "average_scores": average_scores,
            "metrics_tracked": list(scores_by_metric.keys()),
            "total_reviewers": len(set(
                item.reviewer_id for item in items if item.reviewer_id
            ))
        }
