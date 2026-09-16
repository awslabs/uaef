# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MetricScore data model."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class MetricScore(BaseModel):
    """
    Score for a single metric.

    Attributes:
        metric_name: Name of the metric
        score: Numeric score between 0 and 1, or None if metric is not applicable
        reasoning: Explanation of the score (optional, typically from LLM judge)
        metadata: Additional metadata about the score calculation
        timestamp: When the score was calculated
    """

    metric_name: str = Field(..., description="Name of the metric", min_length=1)
    score: Optional[float] = Field(None, description="Numeric score between 0 and 1, or None if not applicable")
    reasoning: Optional[str] = Field(
        None,
        description="Explanation of the score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the score calculation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the score was calculated"
    )
    
    @field_validator("metric_name")
    @classmethod
    def validate_metric_name(cls, v: str) -> str:
        """Validate that metric name is not empty."""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        return v.strip()
    
    @field_validator("score")
    @classmethod
    def validate_score_bounds(cls, v: Optional[float]) -> Optional[float]:
        """Validate that score is between 0 and 1, or None."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_name": "tool_accuracy",
                "score": 0.85,
                "reasoning": "Agent called the correct tools with mostly accurate arguments",
                "metadata": {"incorrect_count": 1, "missed_count": 0},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
