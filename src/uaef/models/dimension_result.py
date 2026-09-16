# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DimensionResult data model."""

from typing import List

from pydantic import BaseModel, Field, field_validator

from uaef.models.metric_score import MetricScore


class DimensionResult(BaseModel):
    """
    Aggregated result for a single evaluation dimension.
    
    Attributes:
        dimension_name: Name of the dimension (e.g., "tool_calling", "response_quality")
        metric_scores: List of individual metric scores in this dimension
        aggregate_score: Weighted average score for the dimension (0-1)
        weight: Weight of this dimension in overall score calculation
    """
    
    dimension_name: str = Field(
        ...,
        description="Name of the dimension",
        min_length=1
    )
    metric_scores: List[MetricScore] = Field(
        ...,
        description="List of individual metric scores in this dimension",
        min_length=1
    )
    aggregate_score: float = Field(
        ...,
        description="Weighted average score for the dimension",
        ge=0.0,
        le=1.0
    )
    weight: float = Field(
        ...,
        description="Weight of this dimension in overall score calculation",
        ge=0.0,
        le=1.0
    )
    
    @field_validator("dimension_name")
    @classmethod
    def validate_dimension_name(cls, v: str) -> str:
        """Validate that dimension name is not empty."""
        if not v or not v.strip():
            raise ValueError("Dimension name cannot be empty")
        return v.strip()
    
    @field_validator("aggregate_score", "weight")
    @classmethod
    def validate_score_bounds(cls, v: float) -> float:
        """Validate that scores and weights are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be between 0 and 1, got {v}")
        return v
    
    @field_validator("metric_scores")
    @classmethod
    def validate_metric_scores(cls, v: List[MetricScore]) -> List[MetricScore]:
        """Validate that metric_scores is not empty."""
        if not v:
            raise ValueError("metric_scores cannot be empty")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "dimension_name": "tool_calling",
                "metric_scores": [
                    {
                        "metric_name": "tool_accuracy",
                        "score": 0.85,
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    {
                        "metric_name": "parameter_quality",
                        "score": 0.90,
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "aggregate_score": 0.875,
                "weight": 0.4
            }
        }
