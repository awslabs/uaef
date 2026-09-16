# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation scoring workflow for HITL validation.

This module implements the conversation scoring workflow, one of three HITL workflows
in UAEF. It collects human feedback on simulated conversations to validate and improve
conversation simulation quality.

The workflow:
1. Queues simulated conversations for human scoring
2. Collects quality ratings across multiple dimensions
3. Aggregates scores across multiple reviewers
4. Analyzes simulation quality trends over time
5. Provides improvement suggestions for the conversation simulator
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from uaef.hitl.models import ItemType, Priority, ReviewItem
from uaef.hitl.queue import ReviewQueue
from uaef.logging import get_logger

logger = get_logger(__name__)


class ConversationScoringError(Exception):
    """Base exception for conversation scoring errors."""
    pass


class InsufficientDataError(ConversationScoringError):
    """Raised when insufficient data is available for analysis."""
    pass


class QualityTrend:
    """
    Represents a trend in conversation quality over time.
    
    Attributes:
        dimension: Quality dimension being tracked
        time_period: Time period for the trend
        average_score: Average score over the period
        score_change: Change compared to previous period
        conversation_count: Number of conversations in the period
        trend_direction: Direction of trend
        issues_detected: List of detected issues
    """

    def __init__(
        self,
        dimension: str,
        time_period: str,
        average_score: float,
        score_change: float,
        conversation_count: int,
        trend_direction: str,
        issues_detected: List[str]
    ):
        """Initialize quality trend."""
        self.dimension = dimension
        self.time_period = time_period
        self.average_score = average_score
        self.score_change = score_change
        self.conversation_count = conversation_count
        self.trend_direction = trend_direction
        self.issues_detected = issues_detected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimension": self.dimension,
            "time_period": self.time_period,
            "average_score": self.average_score,
            "score_change": self.score_change,
            "conversation_count": self.conversation_count,
            "trend_direction": self.trend_direction,
            "issues_detected": self.issues_detected
        }



class ConversationScoringWorkflow:
    """
    Workflow for scoring simulated conversations with human feedback.
    
    This workflow validates and improves conversation simulation quality by:
    1. Queuing simulated conversations for human scoring
    2. Collecting quality ratings across dimensions (realism, coherence, etc.)
    3. Aggregating scores across multiple reviewers
    4. Analyzing quality trends over time
    5. Providing improvement suggestions for the simulator
    
    The workflow integrates with:
    - ReviewQueue: Manages conversations needing scoring
    - ConversationSimulator: Provides simulated conversations (Phase 7)
    
    Scoring dimensions:
    - realism: How realistic the conversation feels
    - coherence: Logical flow and consistency
    - goal_achievement: Whether the conversation achieved its goal
    - naturalness: How natural the language and interactions are

    
    Attributes:
        review_queue: ReviewQueue instance for managing review items
        scoring_dimensions: List of quality dimensions to score
        consensus_threshold: Threshold for reviewer agreement (default: 0.7)
    
    Examples:
        >>> # Initialize workflow
        >>> workflow = ConversationScoringWorkflow()
        
        >>> # Queue a simulated conversation for scoring
        >>> conversation_data = {
        ...     "conversation_id": "conv_001",
        ...     "scenario": "Customer support - refund request",
        ...     "conversation": [
        ...         {"role": "user", "content": "I want a refund"},
        ...         {"role": "agent", "content": "I can help with that..."}
        ...     ],
        ...     "goal_achieved": True,
        ...     "turn_count": 5
        ... }
        >>> item = workflow.queue_conversation(conversation_data)
        
        >>> # Submit scores from a reviewer
        >>> workflow.submit_scores(
        ...     item_id=item.item_id,
        ...     scores={
        ...         "realism": 0.85,
        ...         "coherence": 0.90,
        ...         "goal_achievement": 0.95,
        ...         "naturalness": 0.80
        ...     },
        ...     reviewer_id="reviewer_123",
        ...     feedback="Good conversation but agent could be more empathetic"
        ... )
        
        >>> # Analyze simulation quality trends
        >>> trends = workflow.analyze_simulation_quality(time_period_days=7)
        >>> trends[0].trend_direction
        'improving'
        
        >>> # Get improvement suggestions
        >>> suggestions = workflow.get_improvement_suggestions()
        >>> suggestions["realism"]
        ['Add more varied user responses', 'Include edge cases']
    """

    
    def __init__(
        self,
        review_queue: Optional[ReviewQueue] = None,
        scoring_dimensions: Optional[List[str]] = None,
        consensus_threshold: float = 0.7
    ):
        """
        Initialize the conversation scoring workflow.
        
        Args:
            review_queue: ReviewQueue instance (creates new if None)
            scoring_dimensions: List of quality dimensions to score
                Default: ["realism", "coherence", "goal_achievement", "naturalness"]
            consensus_threshold: Threshold for reviewer agreement (0-1)
        
        Raises:
            ValueError: If consensus_threshold is not between 0 and 1
        """
        if not 0.0 <= consensus_threshold <= 1.0:
            raise ValueError(
                f"consensus_threshold must be between 0 and 1, got {consensus_threshold}"
            )
        
        self.review_queue = review_queue or ReviewQueue()
        self.scoring_dimensions = scoring_dimensions or [
            "realism",
            "coherence",
            "goal_achievement",
            "naturalness"
        ]
        self.consensus_threshold = consensus_threshold
        
        # Storage for multiple reviewer scores (item_id -> list of score entries)
        self._score_history: Dict[UUID, List[Dict[str, Any]]] = {}
        
        logger.info(
            f"Initialized ConversationScoringWorkflow with "
            f"dimensions={self.scoring_dimensions}, "
            f"consensus_threshold={consensus_threshold}"
        )

    
    def queue_conversation(
        self,
        conversation_data: Dict[str, Any],
        auto_priority: bool = True
    ) -> ReviewItem:
        """
        Queue a simulated conversation for human scoring.
        
        Adds a simulated conversation to the review queue for quality assessment.
        This is typically called after the conversation simulator generates a conversation.
        
        Args:
            conversation_data: Dictionary containing:
                - conversation_id: Unique identifier for the conversation
                - scenario: Description of the scenario
                - conversation: List of message dictionaries with role and content
                - goal_achieved: Whether the conversation achieved its goal (optional)
                - turn_count: Number of turns in the conversation (optional)
                - user_model: User behavior model used (optional)
                - metadata: Additional metadata (optional)
                - confidence_score: Confidence in simulation quality (optional, default: 0.5)
            auto_priority: If True, automatically assign priority based on confidence
        
        Returns:
            ReviewItem that was added to the queue
        
        Raises:
            ValueError: If conversation_data format is invalid
        
        Examples:
            >>> workflow = ConversationScoringWorkflow()
            >>> conversation = {
            ...     "conversation_id": "conv_001",
            ...     "scenario": "Customer support - refund request",
            ...     "conversation": [
            ...         {"role": "user", "content": "I want a refund"},
            ...         {"role": "agent", "content": "I can help with that..."}
            ...     ],
            ...     "goal_achieved": True,
            ...     "turn_count": 5,
            ...     "confidence_score": 0.75
            ... }
            >>> item = workflow.queue_conversation(conversation)
            >>> item.item_type
            <ItemType.SIMULATED_CONVERSATION: 'simulated_conversation'>
        """
        # Validate required fields
        if "conversation_id" not in conversation_data:
            raise ValueError("conversation_data must contain 'conversation_id'")
        if "scenario" not in conversation_data:
            raise ValueError("conversation_data must contain 'scenario'")
        if "conversation" not in conversation_data:
            raise ValueError("conversation_data must contain 'conversation'")
        
        conversation_id = conversation_data["conversation_id"]
        confidence_score = conversation_data.get("confidence_score", 0.5)
        
        # Create review item
        review_item = ReviewItem(
            item_type=ItemType.SIMULATED_CONVERSATION,
            content={
                "scenario": conversation_data["scenario"],
                "conversation": conversation_data["conversation"],
                "goal_achieved": conversation_data.get("goal_achieved", None),
                "turn_count": conversation_data.get("turn_count", len(conversation_data["conversation"])),
                "user_model": conversation_data.get("user_model", ""),
                "metadata": conversation_data.get("metadata", {})
            },
            metadata={
                "conversation_id": str(conversation_id),
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
        
        # Initialize score history for this item
        self._score_history[added_item.item_id] = []
        
        logger.debug(
            f"Queued conversation for scoring: conversation_id={conversation_id}, "
            f"turns={review_item.content['turn_count']}, "
            f"confidence={confidence_score:.3f}"
        )
        
        return added_item

    
    def submit_scores(
        self,
        item_id: UUID,
        scores: Dict[str, float],
        reviewer_id: str,
        feedback: Optional[str] = None
    ) -> ReviewItem:
        """
        Submit quality scores for a simulated conversation.
        
        Collects human ratings across multiple quality dimensions to assess
        the realism and quality of simulated conversations.
        
        Args:
            item_id: UUID of the conversation item to score
            scores: Dictionary of dimension -> score (0-1)
                Should include scores for dimensions in self.scoring_dimensions
            reviewer_id: ID of the human reviewer
            feedback: Optional textual feedback about the conversation
        
        Returns:
            Updated ReviewItem
        
        Raises:
            ValueError: If scores format is invalid
            ConversationScoringError: If item not found or invalid operation
        
        Examples:
            >>> workflow = ConversationScoringWorkflow()
            >>> item = workflow.submit_scores(
            ...     item_id=some_uuid,
            ...     scores={
            ...         "realism": 0.85,
            ...         "coherence": 0.90,
            ...         "goal_achievement": 0.95,
            ...         "naturalness": 0.80
            ...     },
            ...     reviewer_id="reviewer_123",
            ...     feedback="Good conversation but agent could be more empathetic"
            ... )
            >>> item.human_scores
            {'realism': 0.85, 'coherence': 0.9, ...}
        """
        # Validate scores
        if not scores:
            raise ValueError("scores cannot be empty")
        
        for dimension, score in scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Score for '{dimension}' must be between 0 and 1, got {score}"
                )
        
        # Validate reviewer_id
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("reviewer_id cannot be empty")
        
        # Get the item
        try:
            item = self.review_queue.get_item(item_id)
        except Exception as e:
            raise ConversationScoringError(f"Failed to get item {item_id}: {e}")
        
        # Store scores in history (for aggregation across multiple reviewers)
        score_entry = {
            "reviewer_id": reviewer_id,
            "timestamp": datetime.utcnow().isoformat(),
            "scores": scores,
            "feedback": feedback
        }
        
        if item_id not in self._score_history:
            self._score_history[item_id] = []
        self._score_history[item_id].append(score_entry)
        
        # Update the review item with the latest scores
        try:
            updated_item = self.review_queue.submit_review(
                item_id=item_id,
                human_scores=scores,
                reviewer_id=reviewer_id,
                feedback=feedback
            )
        except Exception as e:
            raise ConversationScoringError(f"Failed to submit scores: {e}")
        
        logger.info(
            f"Submitted scores for conversation {item_id} from reviewer {reviewer_id}: "
            f"avg_score={np.mean(list(scores.values())):.3f}"
        )
        
        return updated_item

    
    def analyze_simulation_quality(
        self,
        time_period_days: int = 7,
        dimension: Optional[str] = None,
        min_conversation_count: int = 3
    ) -> List[QualityTrend]:
        """
        Analyze simulation quality trends over time.
        
        Tracks how conversation quality changes over time to identify:
        - Improving or declining simulation quality
        - Systematic issues in conversation generation
        - Patterns in quality across dimensions
        
        Args:
            time_period_days: Number of days to analyze
            dimension: Optional filter by specific dimension
            min_conversation_count: Minimum conversations required for trend
        
        Returns:
            List of QualityTrend objects, one per dimension
        
        Raises:
            InsufficientDataError: If insufficient data for analysis
        
        Examples:
            >>> workflow = ConversationScoringWorkflow()
            >>> trends = workflow.analyze_simulation_quality(
            ...     time_period_days=7,
            ...     dimension="realism"
            ... )
            >>> trends[0].trend_direction
            'improving'
            >>> trends[0].average_score
            0.85
            >>> trends[0].score_change
            0.05
        """
        # Get all simulated conversation items
        all_items = self.review_queue.get_all_items(
            item_type=ItemType.SIMULATED_CONVERSATION
        )
        
        # Filter by time period
        cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
        recent_items = [
            item for item in all_items
            if item.created_at >= cutoff_date and item.human_scores
        ]
        
        if len(recent_items) < min_conversation_count:
            raise InsufficientDataError(
                f"Insufficient conversations for quality analysis: "
                f"need at least {min_conversation_count}, got {len(recent_items)}"
            )
        
        # Collect scores by dimension
        scores_by_dimension: Dict[str, List[Tuple[datetime, float]]] = {}
        
        for item in recent_items:
            for dim, score in item.human_scores.items():
                # Filter by dimension if specified
                if dimension and dim != dimension:
                    continue
                
                if dim not in scores_by_dimension:
                    scores_by_dimension[dim] = []
                scores_by_dimension[dim].append((item.created_at, score))
        
        # Analyze trends for each dimension
        trends = []
        
        for dim, score_data in scores_by_dimension.items():
            if len(score_data) < min_conversation_count:
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
            low_score_count = np.sum(scores_arr < 0.6)
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
            
            # Dimension-specific issues
            if dim == "realism" and avg_score < 0.7:
                issues_detected.append(
                    "Low realism scores suggest conversations feel artificial"
                )
            elif dim == "coherence" and avg_score < 0.7:
                issues_detected.append(
                    "Low coherence scores suggest logical flow issues"
                )
            elif dim == "goal_achievement" and avg_score < 0.7:
                issues_detected.append(
                    "Low goal achievement suggests conversations don't reach objectives"
                )
            elif dim == "naturalness" and avg_score < 0.7:
                issues_detected.append(
                    "Low naturalness scores suggest unnatural language patterns"
                )
            
            # Create trend object
            trend = QualityTrend(
                dimension=dim,
                time_period=f"last_{time_period_days}_days",
                average_score=avg_score,
                score_change=score_change,
                conversation_count=len(scores),
                trend_direction=trend_direction,
                issues_detected=issues_detected
            )
            
            trends.append(trend)
            
            logger.debug(
                f"Quality trend for {dim}: avg={avg_score:.3f}, "
                f"change={score_change:+.3f}, direction={trend_direction}"
            )
        
        if not trends:
            raise InsufficientDataError(
                f"No quality trends available for the specified criteria"
            )
        
        logger.info(
            f"Analyzed simulation quality over {time_period_days} days: "
            f"{len(trends)} dimensions analyzed"
        )
        
        return trends

    
    def get_improvement_suggestions(
        self,
        time_period_days: int = 30,
        min_conversation_count: int = 5
    ) -> Dict[str, List[str]]:
        """
        Generate improvement suggestions for the conversation simulator.
        
        Analyzes scored conversations to identify patterns and provide
        actionable recommendations for improving simulation quality.
        
        Args:
            time_period_days: Number of days to analyze
            min_conversation_count: Minimum conversations required
        
        Returns:
            Dictionary mapping dimension -> list of suggestions
        
        Raises:
            InsufficientDataError: If insufficient data for analysis
        
        Examples:
            >>> workflow = ConversationScoringWorkflow()
            >>> suggestions = workflow.get_improvement_suggestions()
            >>> suggestions["realism"]
            ['Add more varied user responses', 'Include edge cases']
            >>> suggestions["coherence"]
            ['Improve context tracking', 'Validate logical flow']
        """
        # Get all simulated conversation items
        all_items = self.review_queue.get_all_items(
            item_type=ItemType.SIMULATED_CONVERSATION
        )
        
        # Filter by time period
        cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
        recent_items = [
            item for item in all_items
            if item.created_at >= cutoff_date and item.human_scores
        ]
        
        if len(recent_items) < min_conversation_count:
            raise InsufficientDataError(
                f"Insufficient conversations for improvement suggestions: "
                f"need at least {min_conversation_count}, got {len(recent_items)}"
            )
        
        # Collect scores and feedback by dimension
        dimension_data: Dict[str, Dict[str, Any]] = {}
        
        for dim in self.scoring_dimensions:
            dimension_data[dim] = {
                "scores": [],
                "low_score_feedback": [],
                "high_score_feedback": []
            }
        
        for item in recent_items:
            for dim in self.scoring_dimensions:
                if dim in item.human_scores:
                    score = item.human_scores[dim]
                    dimension_data[dim]["scores"].append(score)
                    
                    # Collect feedback for low and high scores
                    if item.feedback:
                        if score < 0.6:
                            dimension_data[dim]["low_score_feedback"].append(item.feedback)
                        elif score > 0.8:
                            dimension_data[dim]["high_score_feedback"].append(item.feedback)
        
        # Generate suggestions for each dimension
        suggestions: Dict[str, List[str]] = {}
        
        for dim, data in dimension_data.items():
            if not data["scores"]:
                continue
            
            dim_suggestions = []
            scores_arr = np.array(data["scores"])
            avg_score = float(np.mean(scores_arr))
            score_variance = float(np.var(scores_arr))
            
            # General suggestions based on average score
            if avg_score < 0.6:
                dim_suggestions.append(
                    f"Critical: {dim} scores are low (avg={avg_score:.2f}). "
                    "Major improvements needed."
                )
            elif avg_score < 0.7:
                dim_suggestions.append(
                    f"{dim} scores need improvement (avg={avg_score:.2f}). "
                    "Focus on quality enhancements."
                )
            
            # Suggestions based on variance
            if score_variance > 0.1:
                dim_suggestions.append(
                    f"High variance in {dim} scores (var={score_variance:.3f}). "
                    "Improve consistency across conversations."
                )
            
            # Dimension-specific suggestions
            if dim == "realism":
                if avg_score < 0.7:
                    dim_suggestions.extend([
                        "Add more varied and realistic user responses",
                        "Include edge cases and unexpected user behaviors",
                        "Model real user frustration and patience patterns",
                        "Use more natural language variations"
                    ])
                if score_variance > 0.1:
                    dim_suggestions.append(
                        "Standardize realism across different scenarios"
                    )
            
            elif dim == "coherence":
                if avg_score < 0.7:
                    dim_suggestions.extend([
                        "Improve context tracking across conversation turns",
                        "Validate logical flow between messages",
                        "Ensure agent responses address user questions directly",
                        "Maintain consistency in conversation state"
                    ])
                if score_variance > 0.1:
                    dim_suggestions.append(
                        "Apply coherence checks consistently across all conversations"
                    )
            
            elif dim == "goal_achievement":
                if avg_score < 0.7:
                    dim_suggestions.extend([
                        "Improve goal tracking and completion logic",
                        "Add clearer success criteria for scenarios",
                        "Ensure conversations progress toward goals",
                        "Validate goal achievement before termination"
                    ])
                # Check actual goal achievement rate
                goal_achieved_count = sum(
                    1 for item in recent_items
                    if item.content.get("goal_achieved", False)
                )
                goal_rate = goal_achieved_count / len(recent_items)
                if goal_rate < 0.5:
                    dim_suggestions.append(
                        f"Low goal achievement rate ({goal_rate:.1%}). "
                        "Review scenario difficulty and agent capabilities."
                    )
            
            elif dim == "naturalness":
                if avg_score < 0.7:
                    dim_suggestions.extend([
                        "Use more natural language patterns",
                        "Reduce overly formal or robotic phrasing",
                        "Add conversational fillers and acknowledgments",
                        "Vary sentence structure and length"
                    ])
                if score_variance > 0.1:
                    dim_suggestions.append(
                        "Maintain consistent naturalness across user models"
                    )
            
            # Add feedback-based suggestions
            if data["low_score_feedback"]:
                dim_suggestions.append(
                    f"Review {len(data['low_score_feedback'])} low-score feedback items "
                    "for specific improvement areas"
                )
            
            if data["high_score_feedback"]:
                dim_suggestions.append(
                    f"Analyze {len(data['high_score_feedback'])} high-score examples "
                    "to identify successful patterns"
                )
            
            suggestions[dim] = dim_suggestions
        
        logger.info(
            f"Generated improvement suggestions for {len(suggestions)} dimensions "
            f"based on {len(recent_items)} conversations"
        )
        
        return suggestions

    
    def aggregate_scores(
        self,
        item_id: UUID,
        aggregation_method: str = "mean"
    ) -> Dict[str, Any]:
        """
        Aggregate scores across multiple reviewers for a conversation.
        
        Combines scores from multiple reviewers to produce consensus ratings
        and identify areas of agreement/disagreement.
        
        Args:
            item_id: UUID of the conversation to aggregate scores for
            aggregation_method: Method for aggregation ("mean", "median", "weighted")
        
        Returns:
            Dictionary containing:
                - consensus_scores: Aggregated scores for each dimension
                - reviewer_count: Number of reviewers
                - score_variance: Variance in scores across reviewers
                - agreement_level: Level of agreement ("high", "medium", "low")
                - individual_scores: List of individual score entries
        
        Raises:
            ConversationScoringError: If item not found
            InsufficientDataError: If no scores available
        
        Examples:
            >>> workflow = ConversationScoringWorkflow()
            >>> # After multiple reviewers submit scores...
            >>> aggregated = workflow.aggregate_scores(item_id)
            >>> aggregated["consensus_scores"]
            {'realism': 0.85, 'coherence': 0.88, 'goal_achievement': 0.90}
            >>> aggregated["agreement_level"]
            'high'
        """
        # Validate aggregation method
        if aggregation_method not in ["mean", "median", "weighted"]:
            raise ValueError(
                f"aggregation_method must be 'mean', 'median', or 'weighted', "
                f"got '{aggregation_method}'"
            )
        
        # Get score history
        if item_id not in self._score_history:
            raise ConversationScoringError(
                f"No score history found for conversation {item_id}"
            )
        
        score_entries = self._score_history[item_id]
        
        if not score_entries:
            raise InsufficientDataError(
                f"No scores available for conversation {item_id}"
            )
        
        # Collect all scores by dimension
        scores_by_dimension: Dict[str, List[float]] = {}
        
        for entry in score_entries:
            for dimension, score in entry["scores"].items():
                if dimension not in scores_by_dimension:
                    scores_by_dimension[dimension] = []
                scores_by_dimension[dimension].append(score)
        
        # Aggregate scores
        consensus_scores = {}
        score_variance = {}
        
        for dimension, scores in scores_by_dimension.items():
            scores_arr = np.array(scores)
            
            if aggregation_method == "mean":
                consensus_scores[dimension] = float(np.mean(scores_arr))
            elif aggregation_method == "median":
                consensus_scores[dimension] = float(np.median(scores_arr))
            elif aggregation_method == "weighted":
                # For weighted, use recency weighting (more recent = higher weight)
                weights = np.linspace(0.5, 1.0, len(scores))
                consensus_scores[dimension] = float(
                    np.average(scores_arr, weights=weights)
                )
            
            # Calculate variance
            score_variance[dimension] = float(np.var(scores_arr))
        
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
            "reviewer_count": len(score_entries),
            "score_variance": score_variance,
            "average_variance": float(avg_variance),
            "agreement_level": agreement_level,
            "aggregation_method": aggregation_method,
            "individual_scores": score_entries,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Aggregated scores for conversation {item_id}: "
            f"{len(score_entries)} reviewers, "
            f"agreement={agreement_level}"
        )
        
        return result
    
    def get_scoring_summary(
        self,
        time_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of conversation scoring status.
        
        Provides an overview of:
        - Total conversations scored
        - Average scores by dimension
        - Scoring progress
        - Quality trends
        
        Args:
            time_period_days: Optional time period for filtering (default: all time)
        
        Returns:
            Dictionary with scoring summary
        
        Examples:
            >>> workflow = ConversationScoringWorkflow()
            >>> summary = workflow.get_scoring_summary(time_period_days=7)
            >>> summary["total_conversations"]
            42
            >>> summary["average_scores"]["realism"]
            0.85
            >>> summary["quality_status"]
            'good'
        """
        # Get all simulated conversation items
        all_items = self.review_queue.get_all_items(
            item_type=ItemType.SIMULATED_CONVERSATION
        )
        
        # Filter by time period if specified
        if time_period_days:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            items = [item for item in all_items if item.created_at >= cutoff_date]
        else:
            items = all_items
        
        # Calculate average scores by dimension
        scores_by_dimension: Dict[str, List[float]] = {}
        
        for item in items:
            if item.human_scores:
                for dimension, score in item.human_scores.items():
                    if dimension not in scores_by_dimension:
                        scores_by_dimension[dimension] = []
                    scores_by_dimension[dimension].append(score)
        
        average_scores = {
            dimension: float(np.mean(scores))
            for dimension, scores in scores_by_dimension.items()
        }
        
        # Calculate overall quality status
        if average_scores:
            overall_avg = np.mean(list(average_scores.values()))
            if overall_avg >= 0.8:
                quality_status = "excellent"
            elif overall_avg >= 0.7:
                quality_status = "good"
            elif overall_avg >= 0.6:
                quality_status = "fair"
            else:
                quality_status = "needs_improvement"
        else:
            quality_status = "unknown"
        
        # Get queue stats
        queue_stats = self.review_queue.get_queue_stats()
        
        # Calculate goal achievement rate
        goal_achieved_count = sum(
            1 for item in items
            if item.content.get("goal_achieved", False)
        )
        goal_achievement_rate = (
            goal_achieved_count / len(items) if items else 0.0
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "time_period": f"last_{time_period_days}_days" if time_period_days else "all_time",
            "total_conversations": len(items),
            "pending_scoring": queue_stats["by_type"]["simulated_conversation"],
            "completed_scoring": len([item for item in items if item.human_scores]),
            "average_scores": average_scores,
            "quality_status": quality_status,
            "goal_achievement_rate": float(goal_achievement_rate),
            "dimensions_tracked": list(scores_by_dimension.keys()),
            "total_reviewers": len(set(
                item.reviewer_id for item in items if item.reviewer_id
            ))
        }
