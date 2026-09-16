# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review queue manager for HITL workflows."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from .models import ItemType, Priority, ReviewItem, ReviewStatus


class ReviewQueueError(Exception):
    """Base exception for review queue errors."""
    pass


class ItemNotFoundError(ReviewQueueError):
    """Raised when a review item is not found in the queue."""
    pass


class InvalidOperationError(ReviewQueueError):
    """Raised when an invalid operation is attempted."""
    pass


class ReviewQueue:
    """
    Manager for the review queue supporting three HITL workflows:
    1. LLM judge calibration workflow
    2. Online evaluation feedback workflow
    3. Simulated conversation scoring workflow
    
    The ReviewQueue manages the lifecycle of ReviewItem objects from creation
    through completion, with priority-based ordering and confidence-based filtering.
    
    This implementation uses an in-memory queue that can be extended to use
    a database backend for persistence and scalability.
    
    Attributes:
        _queue: Internal storage for review items (item_id -> ReviewItem)
    
    Examples:
        >>> # Create a review queue
        >>> queue = ReviewQueue()
        
        >>> # Add items with automatic priority assignment
        >>> item = ReviewItem(
        ...     item_type=ItemType.LLM_EVALUATION,
        ...     content={"question": "What is AI?", "response": "AI is..."},
        ...     confidence_score=0.65
        ... )
        >>> queue.add_to_queue(item)
        
        >>> # Get next item for review (highest priority, lowest confidence)
        >>> next_item = queue.get_next_review_item()
        
        >>> # Submit review with human scores
        >>> queue.submit_review(
        ...     item_id=next_item.item_id,
        ...     human_scores={"answer_relevance": 0.85, "answer_correctness": 0.90},
        ...     feedback="Good response but could be more detailed",
        ...     reviewer_id="reviewer_123"
        ... )
        
        >>> # Filter by confidence level
        >>> low_confidence_items = queue.get_next_review_item(
        ...     confidence_threshold=0.7
        ... )
        
        >>> # Filter by item type
        >>> agent_responses = queue.get_next_review_item(
        ...     item_type=ItemType.AGENT_RESPONSE
        ... )
    """
    
    def __init__(self):
        """Initialize an empty review queue."""
        self._queue: Dict[UUID, ReviewItem] = {}
    
    def add_to_queue(
        self,
        item: ReviewItem,
        auto_priority: bool = True
    ) -> ReviewItem:
        """
        Add a new item to the review queue.
        
        If auto_priority is True, automatically assigns priority based on
        confidence score:
        - confidence < 0.5: CRITICAL
        - 0.5 <= confidence < 0.7: HIGH
        - 0.7 <= confidence < 0.85: MEDIUM
        - confidence >= 0.85: LOW
        
        Args:
            item: ReviewItem to add to the queue
            auto_priority: If True, automatically assign priority based on confidence
        
        Returns:
            The added ReviewItem (with potentially updated priority)
        
        Raises:
            InvalidOperationError: If item already exists in queue
        
        Examples:
            >>> queue = ReviewQueue()
            >>> item = ReviewItem(
            ...     item_type=ItemType.LLM_EVALUATION,
            ...     content={"question": "What is AI?"},
            ...     confidence_score=0.45,
            ...     priority=Priority.LOW  # Will be overridden
            ... )
            >>> added_item = queue.add_to_queue(item, auto_priority=True)
            >>> added_item.priority
            <Priority.CRITICAL: 'critical'>
            
            >>> # Manual priority assignment
            >>> item2 = ReviewItem(
            ...     item_type=ItemType.AGENT_RESPONSE,
            ...     content={"response": "..."},
            ...     confidence_score=0.95,
            ...     priority=Priority.HIGH
            ... )
            >>> added_item2 = queue.add_to_queue(item2, auto_priority=False)
            >>> added_item2.priority
            <Priority.HIGH: 'high'>
        """
        if item.item_id in self._queue:
            raise InvalidOperationError(
                f"Item with ID {item.item_id} already exists in queue"
            )
        
        # Auto-assign priority based on confidence score
        if auto_priority:
            if item.confidence_score < 0.5:
                item.priority = Priority.CRITICAL
            elif item.confidence_score < 0.7:
                item.priority = Priority.HIGH
            elif item.confidence_score < 0.85:
                item.priority = Priority.MEDIUM
            else:
                item.priority = Priority.LOW
        
        # Add to queue
        self._queue[item.item_id] = item
        return item
    
    def get_next_review_item(
        self,
        item_type: Optional[ItemType] = None,
        confidence_threshold: Optional[float] = None
    ) -> Optional[ReviewItem]:
        """
        Retrieve the highest priority pending item from the queue.
        
        Items are ordered by:
        1. Priority (CRITICAL > HIGH > MEDIUM > LOW)
        2. Confidence score (lower confidence first)
        3. Creation time (older items first)
        
        Args:
            item_type: Optional filter by item type
            confidence_threshold: Optional filter for items with confidence below threshold
        
        Returns:
            The next ReviewItem to review, or None if no items match criteria
        
        Raises:
            ValueError: If confidence_threshold is not between 0 and 1
        
        Examples:
            >>> queue = ReviewQueue()
            >>> # Add multiple items
            >>> item1 = ReviewItem(
            ...     item_type=ItemType.LLM_EVALUATION,
            ...     content={"q": "Q1"},
            ...     confidence_score=0.45,
            ...     priority=Priority.CRITICAL
            ... )
            >>> item2 = ReviewItem(
            ...     item_type=ItemType.AGENT_RESPONSE,
            ...     content={"q": "Q2"},
            ...     confidence_score=0.65,
            ...     priority=Priority.HIGH
            ... )
            >>> queue.add_to_queue(item1, auto_priority=False)
            >>> queue.add_to_queue(item2, auto_priority=False)
            
            >>> # Get highest priority item
            >>> next_item = queue.get_next_review_item()
            >>> next_item.priority
            <Priority.CRITICAL: 'critical'>
            
            >>> # Filter by item type
            >>> agent_item = queue.get_next_review_item(
            ...     item_type=ItemType.AGENT_RESPONSE
            ... )
            >>> agent_item.item_type
            <ItemType.AGENT_RESPONSE: 'agent_response'>
            
            >>> # Filter by confidence threshold
            >>> low_conf_item = queue.get_next_review_item(
            ...     confidence_threshold=0.7
            ... )
            >>> low_conf_item.confidence_score < 0.7
            True
        """
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError(
                    f"confidence_threshold must be between 0 and 1, got {confidence_threshold}"
                )
        
        # Filter pending items
        pending_items = [
            item for item in self._queue.values()
            if item.status == ReviewStatus.PENDING
        ]
        
        # Apply item_type filter
        if item_type is not None:
            pending_items = [
                item for item in pending_items
                if item.item_type == item_type
            ]
        
        # Apply confidence_threshold filter
        if confidence_threshold is not None:
            pending_items = [
                item for item in pending_items
                if item.confidence_score < confidence_threshold
            ]
        
        # Return None if no items match criteria
        if not pending_items:
            return None
        
        # Define priority order for sorting
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }
        
        # Sort by priority (highest first), then confidence (lowest first), then created_at (oldest first)
        sorted_items = sorted(
            pending_items,
            key=lambda item: (
                priority_order[item.priority],
                item.confidence_score,
                item.created_at
            )
        )
        
        return sorted_items[0] if sorted_items else None
    
    def submit_review(
        self,
        item_id: UUID,
        human_scores: Dict[str, float],
        reviewer_id: str,
        feedback: Optional[str] = None
    ) -> ReviewItem:
        """
        Submit a review for an item, updating it with human scores and feedback.
        
        This method:
        1. Validates the item exists and is in a reviewable state
        2. Updates the item with human scores, feedback, and reviewer_id
        3. Marks the item as completed with reviewed_at timestamp
        
        Args:
            item_id: UUID of the item to review
            human_scores: Dictionary of metric_name -> score (0-1)
            reviewer_id: ID of the human reviewer
            feedback: Optional textual feedback from the reviewer
        
        Returns:
            The updated ReviewItem
        
        Raises:
            ItemNotFoundError: If item_id is not found in queue
            InvalidOperationError: If item is already completed
            ValueError: If human_scores are invalid (not between 0-1)
        
        Examples:
            >>> queue = ReviewQueue()
            >>> item = ReviewItem(
            ...     item_type=ItemType.LLM_EVALUATION,
            ...     content={"question": "What is AI?"},
            ...     confidence_score=0.65,
            ...     priority=Priority.MEDIUM
            ... )
            >>> queue.add_to_queue(item, auto_priority=False)
            
            >>> # Submit review
            >>> reviewed_item = queue.submit_review(
            ...     item_id=item.item_id,
            ...     human_scores={
            ...         "answer_relevance": 0.85,
            ...         "answer_correctness": 0.90
            ...     },
            ...     reviewer_id="reviewer_123",
            ...     feedback="Good response but could be more detailed"
            ... )
            >>> reviewed_item.status
            <ReviewStatus.COMPLETED: 'completed'>
            >>> reviewed_item.human_scores
            {'answer_relevance': 0.85, 'answer_correctness': 0.9}
            
            >>> # Attempting to review again raises error
            >>> queue.submit_review(
            ...     item_id=item.item_id,
            ...     human_scores={"score": 0.5},
            ...     reviewer_id="reviewer_456"
            ... )
            Traceback (most recent call last):
                ...
            InvalidOperationError: Item ... is already completed
        """
        # Check if item exists
        if item_id not in self._queue:
            raise ItemNotFoundError(
                f"Item with ID {item_id} not found in queue"
            )
        
        item = self._queue[item_id]
        
        # Check if item is already completed
        if item.status == ReviewStatus.COMPLETED:
            raise InvalidOperationError(
                f"Item {item_id} is already completed"
            )
        
        # Validate human_scores
        for metric_name, score in human_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Human score for '{metric_name}' must be between 0 and 1, got {score}"
                )
        
        # Validate reviewer_id
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("reviewer_id cannot be empty")
        
        # Update item
        item.human_scores = human_scores
        item.reviewer_id = reviewer_id.strip()
        item.feedback = feedback
        item.status = ReviewStatus.COMPLETED
        item.reviewed_at = datetime.utcnow()
        
        return item
    
    def get_item(self, item_id: UUID) -> ReviewItem:
        """
        Retrieve a specific item from the queue by ID.
        
        Args:
            item_id: UUID of the item to retrieve
        
        Returns:
            The ReviewItem
        
        Raises:
            ItemNotFoundError: If item_id is not found in queue
        
        Examples:
            >>> queue = ReviewQueue()
            >>> item = ReviewItem(
            ...     item_type=ItemType.LLM_EVALUATION,
            ...     content={"question": "What is AI?"},
            ...     confidence_score=0.65,
            ...     priority=Priority.MEDIUM
            ... )
            >>> queue.add_to_queue(item, auto_priority=False)
            >>> retrieved = queue.get_item(item.item_id)
            >>> retrieved.item_id == item.item_id
            True
        """
        if item_id not in self._queue:
            raise ItemNotFoundError(
                f"Item with ID {item_id} not found in queue"
            )
        return self._queue[item_id]
    
    def get_all_items(
        self,
        status: Optional[ReviewStatus] = None,
        item_type: Optional[ItemType] = None
    ) -> List[ReviewItem]:
        """
        Retrieve all items from the queue, optionally filtered by status and type.
        
        Args:
            status: Optional filter by review status
            item_type: Optional filter by item type
        
        Returns:
            List of ReviewItems matching the criteria
        
        Examples:
            >>> queue = ReviewQueue()
            >>> item1 = ReviewItem(
            ...     item_type=ItemType.LLM_EVALUATION,
            ...     content={"q": "Q1"},
            ...     confidence_score=0.65,
            ...     priority=Priority.MEDIUM
            ... )
            >>> item2 = ReviewItem(
            ...     item_type=ItemType.AGENT_RESPONSE,
            ...     content={"q": "Q2"},
            ...     confidence_score=0.75,
            ...     priority=Priority.LOW
            ... )
            >>> queue.add_to_queue(item1, auto_priority=False)
            >>> queue.add_to_queue(item2, auto_priority=False)
            
            >>> # Get all items
            >>> all_items = queue.get_all_items()
            >>> len(all_items)
            2
            
            >>> # Filter by status
            >>> pending = queue.get_all_items(status=ReviewStatus.PENDING)
            >>> len(pending)
            2
            
            >>> # Filter by item type
            >>> llm_items = queue.get_all_items(item_type=ItemType.LLM_EVALUATION)
            >>> len(llm_items)
            1
        """
        items = list(self._queue.values())
        
        # Apply status filter
        if status is not None:
            items = [item for item in items if item.status == status]
        
        # Apply item_type filter
        if item_type is not None:
            items = [item for item in items if item.item_type == item_type]
        
        return items
    
    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get statistics about the current queue state.
        
        Returns:
            Dictionary with queue statistics:
            - total: Total number of items in queue
            - pending: Number of pending items
            - in_review: Number of items currently in review
            - completed: Number of completed items
            - by_priority: Count of items by priority level
            - by_type: Count of items by item type
        
        Examples:
            >>> queue = ReviewQueue()
            >>> item1 = ReviewItem(
            ...     item_type=ItemType.LLM_EVALUATION,
            ...     content={"q": "Q1"},
            ...     confidence_score=0.45,
            ...     priority=Priority.CRITICAL
            ... )
            >>> item2 = ReviewItem(
            ...     item_type=ItemType.AGENT_RESPONSE,
            ...     content={"q": "Q2"},
            ...     confidence_score=0.85,
            ...     priority=Priority.LOW
            ... )
            >>> queue.add_to_queue(item1, auto_priority=False)
            >>> queue.add_to_queue(item2, auto_priority=False)
            
            >>> stats = queue.get_queue_stats()
            >>> stats['total']
            2
            >>> stats['pending']
            2
            >>> stats['by_priority']['critical']
            1
        """
        items = list(self._queue.values())
        
        # Count by status
        pending = sum(1 for item in items if item.status == ReviewStatus.PENDING)
        in_review = sum(1 for item in items if item.status == ReviewStatus.IN_REVIEW)
        completed = sum(1 for item in items if item.status == ReviewStatus.COMPLETED)
        
        # Count by priority
        by_priority = {
            "critical": sum(1 for item in items if item.priority == Priority.CRITICAL),
            "high": sum(1 for item in items if item.priority == Priority.HIGH),
            "medium": sum(1 for item in items if item.priority == Priority.MEDIUM),
            "low": sum(1 for item in items if item.priority == Priority.LOW)
        }
        
        # Count by type
        by_type = {
            "llm_evaluation": sum(1 for item in items if item.item_type == ItemType.LLM_EVALUATION),
            "agent_response": sum(1 for item in items if item.item_type == ItemType.AGENT_RESPONSE),
            "simulated_conversation": sum(1 for item in items if item.item_type == ItemType.SIMULATED_CONVERSATION)
        }
        
        return {
            "total": len(items),
            "pending": pending,
            "in_review": in_review,
            "completed": completed,
            "by_priority": by_priority,
            "by_type": by_type
        }
