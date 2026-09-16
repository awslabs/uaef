# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HITL (Human-in-the-Loop) review queue data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class ItemType(str, Enum):
    """Type of item in the review queue."""
    
    LLM_EVALUATION = "llm_evaluation"
    AGENT_RESPONSE = "agent_response"
    SIMULATED_CONVERSATION = "simulated_conversation"


class Priority(str, Enum):
    """Priority level for review items."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewStatus(str, Enum):
    """Status of a review item."""
    
    PENDING = "pending"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"


class ReviewItem(BaseModel):
    """
    Item in the review queue for human validation.
    
    This model supports three distinct HITL workflows:
    1. LLM judge calibration workflow - validating LLM evaluation scores
    2. Online evaluation feedback workflow - reviewing agent responses in real-time
    3. Simulated conversation scoring workflow - scoring generated conversations
    
    Attributes:
        item_id: Unique identifier for this review item
        item_type: Type of item (llm_evaluation, agent_response, simulated_conversation)
        content: The content to be reviewed (question, response, conversation, etc.)
        metadata: Additional context for the review (trace_id, evaluation_id, etc.)
        priority: Priority level (low, medium, high, critical)
        confidence_score: Confidence score from automated evaluation (0-1)
        status: Current status (pending, in_review, completed)
        created_at: When the item was added to the queue
        reviewed_at: When the review was completed (None if not yet reviewed)
        reviewer_id: ID of the human reviewer (None if not yet assigned)
        human_scores: Scores provided by human reviewers (metric_name -> score)
        feedback: Textual feedback from human reviewers
    
    Examples:
        >>> # LLM judge calibration item
        >>> item = ReviewItem(
        ...     item_type=ItemType.LLM_EVALUATION,
        ...     content={
        ...         "question": "What is the capital of France?",
        ...         "agent_response": "The capital of France is Paris.",
        ...         "llm_scores": {"answer_relevance": 0.95}
        ...     },
        ...     metadata={"evaluation_id": "550e8400-e29b-41d4-a716-446655440000"},
        ...     priority=Priority.MEDIUM,
        ...     confidence_score=0.75
        ... )
        
        >>> # Online evaluation feedback item
        >>> item = ReviewItem(
        ...     item_type=ItemType.AGENT_RESPONSE,
        ...     content={
        ...         "user_query": "Book a flight to NYC",
        ...         "agent_response": "I've found 3 flights...",
        ...         "tools_called": ["search_flights", "get_prices"]
        ...     },
        ...     priority=Priority.HIGH,
        ...     confidence_score=0.65
        ... )
        
        >>> # Simulated conversation scoring item
        >>> item = ReviewItem(
        ...     item_type=ItemType.SIMULATED_CONVERSATION,
        ...     content={
        ...         "scenario": "Customer support - refund request",
        ...         "conversation": [
        ...             {"role": "user", "content": "I want a refund"},
        ...             {"role": "agent", "content": "I can help with that..."}
        ...         ],
        ...         "goal_achieved": True
        ...     },
        ...     priority=Priority.LOW,
        ...     confidence_score=0.85
        ... )
    """
    
    item_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this review item"
    )
    item_type: ItemType = Field(
        ...,
        description="Type of item (llm_evaluation, agent_response, simulated_conversation)"
    )
    content: Dict[str, Any] = Field(
        ...,
        description="The content to be reviewed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the review"
    )
    priority: Priority = Field(
        ...,
        description="Priority level (low, medium, high, critical)"
    )
    confidence_score: float = Field(
        ...,
        description="Confidence score from automated evaluation",
        ge=0.0,
        le=1.0
    )
    status: ReviewStatus = Field(
        default=ReviewStatus.PENDING,
        description="Current status (pending, in_review, completed)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the item was added to the queue"
    )
    reviewed_at: Optional[datetime] = Field(
        None,
        description="When the review was completed"
    )
    reviewer_id: Optional[str] = Field(
        None,
        description="ID of the human reviewer"
    )
    human_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores provided by human reviewers (metric_name -> score)"
    )
    feedback: Optional[str] = Field(
        None,
        description="Textual feedback from human reviewers"
    )
    
    @field_validator("item_id")
    @classmethod
    def validate_item_id(cls, v: UUID) -> UUID:
        """Validate that item_id is not None."""
        if v is None:
            raise ValueError("item_id cannot be None")
        return v
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that content is not empty."""
        if not v:
            raise ValueError("content cannot be empty")
        return v
    
    @field_validator("confidence_score")
    @classmethod
    def validate_confidence_score(cls, v: float) -> float:
        """Validate that confidence_score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence_score must be between 0 and 1, got {v}")
        return v
    
    @field_validator("human_scores")
    @classmethod
    def validate_human_scores(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all human scores are between 0 and 1."""
        for metric_name, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Human score for '{metric_name}' must be between 0 and 1, got {score}"
                )
        return v
    
    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Validate that reviewed_at is set only when status is completed."""
        # Note: info.data contains the other field values
        status = info.data.get("status")
        if status == ReviewStatus.COMPLETED and v is None:
            raise ValueError("reviewed_at must be set when status is completed")
        return v
    
    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate that reviewer_id is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("reviewer_id cannot be empty string")
        return v.strip() if v else None
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "item_id": "550e8400-e29b-41d4-a716-446655440000",
                "item_type": "llm_evaluation",
                "content": {
                    "question": "What is the capital of France?",
                    "agent_response": "The capital of France is Paris.",
                    "llm_scores": {"answer_relevance": 0.95, "answer_correctness": 0.98}
                },
                "metadata": {
                    "evaluation_id": "660e8400-e29b-41d4-a716-446655440001",
                    "trace_id": "770e8400-e29b-41d4-a716-446655440002"
                },
                "priority": "medium",
                "confidence_score": 0.75,
                "status": "completed",
                "created_at": "2024-01-15T10:00:00Z",
                "reviewed_at": "2024-01-15T10:30:00Z",
                "reviewer_id": "reviewer_123",
                "human_scores": {
                    "answer_relevance": 0.90,
                    "answer_correctness": 1.0
                },
                "feedback": "Response is accurate but could be more detailed"
            }
        }


class AgreementMetrics(BaseModel):
    """
    Metrics for measuring agreement between LLM judges and human reviewers.
    
    This model tracks agreement statistics to calibrate LLM-based evaluation
    and identify areas where automated evaluation needs improvement.
    
    Attributes:
        metric_name: Name of the metric being evaluated
        llm_score: Score from the LLM judge (None if not applicable)
        human_scores: List of scores from human reviewers
        agreement_score: Correlation coefficient between LLM and human scores (0-1)
        disagreement_details: Details about disagreements (score deltas, patterns, etc.)
    
    Examples:
        >>> # High agreement case
        >>> metrics = AgreementMetrics(
        ...     metric_name="answer_relevance",
        ...     llm_score=0.85,
        ...     human_scores=[0.80, 0.85, 0.90],
        ...     agreement_score=0.95,
        ...     disagreement_details={
        ...         "mean_delta": 0.02,
        ...         "max_delta": 0.05,
        ...         "inter_rater_agreement": 0.92
        ...     }
        ... )
        
        >>> # Low agreement case requiring calibration
        >>> metrics = AgreementMetrics(
        ...     metric_name="safety_score",
        ...     llm_score=0.90,
        ...     human_scores=[0.50, 0.55, 0.60],
        ...     agreement_score=0.45,
        ...     disagreement_details={
        ...         "mean_delta": 0.35,
        ...         "max_delta": 0.40,
        ...         "pattern": "LLM consistently overestimates safety",
        ...         "inter_rater_agreement": 0.88
        ...     }
        ... )
    """
    
    metric_name: str = Field(
        ...,
        description="Name of the metric being evaluated",
        min_length=1
    )
    llm_score: Optional[float] = Field(
        None,
        description="Score from the LLM judge",
        ge=0.0,
        le=1.0
    )
    human_scores: List[float] = Field(
        ...,
        description="List of scores from human reviewers"
    )
    agreement_score: float = Field(
        ...,
        description="Correlation coefficient between LLM and human scores",
        ge=0.0,
        le=1.0
    )
    disagreement_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details about disagreements (score deltas, patterns, etc.)"
    )
    
    @field_validator("metric_name")
    @classmethod
    def validate_metric_name(cls, v: str) -> str:
        """Validate that metric_name is not empty."""
        if not v or not v.strip():
            raise ValueError("metric_name cannot be empty")
        return v.strip()
    
    @field_validator("llm_score")
    @classmethod
    def validate_llm_score(cls, v: Optional[float]) -> Optional[float]:
        """Validate that llm_score is between 0 and 1 if provided."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"llm_score must be between 0 and 1, got {v}")
        return v
    
    @field_validator("human_scores")
    @classmethod
    def validate_human_scores(cls, v: List[float]) -> List[float]:
        """Validate that all human scores are between 0 and 1."""
        if not v:
            raise ValueError("human_scores cannot be empty")
        for i, score in enumerate(v):
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Human score at index {i} must be between 0 and 1, got {score}"
                )
        return v
    
    @field_validator("agreement_score")
    @classmethod
    def validate_agreement_score(cls, v: float) -> float:
        """Validate that agreement_score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"agreement_score must be between 0 and 1, got {v}")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_name": "answer_relevance",
                "llm_score": 0.85,
                "human_scores": [0.80, 0.85, 0.90],
                "agreement_score": 0.95,
                "disagreement_details": {
                    "mean_delta": 0.02,
                    "max_delta": 0.05,
                    "std_delta": 0.03,
                    "inter_rater_agreement": 0.92,
                    "sample_size": 3,
                    "correlation_type": "pearson"
                }
            }
        }
