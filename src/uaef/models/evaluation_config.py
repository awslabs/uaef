# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""EvaluationConfig and MetricSet data models."""

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class EvaluationMode(str, Enum):
    """Mode of evaluation execution."""
    
    ONLINE = "online"
    OFFLINE = "offline"


class MetricSet(BaseModel):
    """
    Configuration for which metrics to calculate.
    
    Attributes:
        dimensions: List of dimension names to evaluate
        metrics: List of specific metric names to calculate
        weights: Dictionary mapping dimension names to weights (0-1)
        thresholds: Dictionary mapping dimension names to minimum passing scores (0-1)
    """
    
    dimensions: List[str] = Field(
        ...,
        description="List of dimension names to evaluate",
        min_length=1
    )
    metrics: List[str] = Field(
        ...,
        description="List of specific metric names to calculate",
        min_length=1
    )
    weights: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping dimension names to weights"
    )
    thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary mapping dimension names to minimum passing scores"
    )
    
    @field_validator("dimensions", "metrics")
    @classmethod
    def validate_non_empty_list(cls, v: List[str]) -> List[str]:
        """Validate that lists are not empty."""
        if not v:
            raise ValueError("List cannot be empty")
        return v
    
    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all weights are between 0 and 1."""
        for dimension, weight in v.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Weight for {dimension} must be between 0 and 1, got {weight}")
        return v
    
    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all thresholds are between 0 and 1."""
        for dimension, threshold in v.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"Threshold for {dimension} must be between 0 and 1, got {threshold}"
                )
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "dimensions": ["tool_calling", "response_quality", "performance"],
                "metrics": ["tool_accuracy", "answer_relevance", "latency_score"],
                "weights": {
                    "tool_calling": 0.4,
                    "response_quality": 0.4,
                    "performance": 0.2
                },
                "thresholds": {
                    "tool_calling": 0.7,
                    "response_quality": 0.8,
                    "performance": 0.6
                }
            }
        }


class EvaluationConfig(BaseModel):
    """
    Configuration for evaluation execution.
    
    Attributes:
        metric_set: Set of metrics to calculate
        mode: Evaluation mode (online or offline)
        llm_judge_config: Configuration for LLM judge (optional)
        parallel_execution: Whether to execute metrics in parallel
        timeout_seconds: Timeout for evaluation in seconds (optional)
        metadata: Additional configuration metadata
    """
    
    metric_set: MetricSet = Field(..., description="Set of metrics to calculate")
    mode: EvaluationMode = Field(
        default=EvaluationMode.OFFLINE,
        description="Evaluation mode"
    )
    llm_judge_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for LLM judge"
    )
    parallel_execution: bool = Field(
        default=True,
        description="Whether to execute metrics in parallel"
    )
    timeout_seconds: int = Field(
        default=300,
        description="Timeout for evaluation in seconds",
        gt=0
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration metadata"
    )
    
    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: EvaluationMode) -> EvaluationMode:
        """Validate that mode is a valid enum value."""
        if not isinstance(v, EvaluationMode):
            raise ValueError(f"Invalid mode: {v}")
        return v
    
    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate that timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_set": {
                    "dimensions": ["tool_calling", "response_quality"],
                    "metrics": ["tool_accuracy", "answer_relevance"],
                    "weights": {"tool_calling": 0.5, "response_quality": 0.5}
                },
                "mode": "offline",
                "llm_judge_config": {
                    "model_id": "us.anthropic.claude-sonnet-4-6",
                    "region": "us-east-1"
                },
                "parallel_execution": True,
                "timeout_seconds": 300,
                "metadata": {}
            }
        }
