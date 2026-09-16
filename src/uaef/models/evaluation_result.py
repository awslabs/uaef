# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""EvaluationResult data model."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from uaef.models.dimension_result import DimensionResult


class EvaluationResult(BaseModel):
    """
    Complete evaluation result for an agent trace.
    
    Attributes:
        evaluation_id: Unique identifier for this evaluation
        trace_id: ID of the trace that was evaluated
        dimension_results: List of results for each dimension
        overall_score: Overall weighted score across all dimensions (0-1)
        passed: Whether the evaluation passed all thresholds
        failures: List of failure messages for threshold violations
        warnings: List of warning messages
        metadata: Additional metadata about the evaluation
        timestamp: When the evaluation was completed
    """
    
    evaluation_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this evaluation"
    )
    trace_id: UUID = Field(..., description="ID of the trace that was evaluated")
    experiment_id: Optional[UUID] = Field(
        None,
        description="ID of the experiment this evaluation belongs to (set when persist=True)"
    )
    dimension_results: List[DimensionResult] = Field(
        ...,
        description="List of results for each dimension"
    )
    overall_score: float = Field(
        ...,
        description="Overall weighted score across all dimensions",
        ge=0.0,
        le=1.0
    )
    passed: bool = Field(..., description="Whether the evaluation passed all thresholds")
    failures: List[str] = Field(
        default_factory=list,
        description="List of failure messages for threshold violations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the evaluation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the evaluation was completed"
    )
    
    @field_validator("evaluation_id", "trace_id")
    @classmethod
    def validate_uuid(cls, v: UUID) -> UUID:
        """Validate that UUIDs are not None."""
        if v is None:
            raise ValueError("UUID cannot be None")
        return v
    
    @field_validator("overall_score")
    @classmethod
    def validate_score_bounds(cls, v: float) -> float:
        """Validate that overall score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Overall score must be between 0 and 1, got {v}")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "evaluation_id": "550e8400-e29b-41d4-a716-446655440000",
                "trace_id": "660e8400-e29b-41d4-a716-446655440001",
                "dimension_results": [
                    {
                        "dimension_name": "tool_calling",
                        "metric_scores": [],
                        "aggregate_score": 0.85,
                        "weight": 0.4
                    }
                ],
                "overall_score": 0.82,
                "passed": True,
                "failures": [],
                "warnings": ["Ground truth not provided for some metrics"],
                "metadata": {"evaluation_mode": "offline"},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
