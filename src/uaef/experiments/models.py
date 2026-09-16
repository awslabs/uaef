# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment data models for versioning and comparison."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class AgentDesign(BaseModel):
    """
    Agent design configuration for reproducibility.
    
    Attributes:
        framework: Agent framework (e.g., "langgraph", "bedrock", "langchain")
        model: LLM model identifier (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
        tools: List of tool names available to the agent
        system_prompt: System prompt used for the agent
        parameters: Additional agent parameters (temperature, max_tokens, etc.)
    """
    
    framework: str = Field(..., description="Agent framework")
    model: str = Field(..., description="LLM model identifier")
    tools: List[str] = Field(
        default_factory=list,
        description="List of tool names available to the agent"
    )
    system_prompt: Optional[str] = Field(
        None,
        description="System prompt used for the agent"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent parameters"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "framework": "langgraph",
                "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                "tools": ["search", "calculator", "database_query"],
                "system_prompt": "You are a helpful assistant...",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.9
                }
            }
        }


class Experiment(BaseModel):
    """
    Experiment for tracking agent versions and evaluations.
    
    Attributes:
        experiment_id: Unique identifier for the experiment
        name: Human-readable experiment name
        description: Detailed description of the experiment
        created_at: When the experiment was created
        metadata: Additional metadata (tags, owner, project, etc.)
        baseline_run_id: ID of the run designated as baseline for comparison
    """
    
    experiment_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the experiment"
    )
    name: str = Field(..., description="Human-readable experiment name")
    description: Optional[str] = Field(
        None,
        description="Detailed description of the experiment"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the experiment was created"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tags, owner, project, etc.)"
    )
    baseline_run_id: Optional[UUID] = Field(
        None,
        description="ID of the run designated as baseline for comparison"
    )
    
    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, v: UUID) -> UUID:
        """Validate that experiment_id is not None."""
        if v is None:
            raise ValueError("experiment_id cannot be None")
        return v
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError("Experiment name cannot be empty")
        return v.strip()
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Customer Support Agent v2",
                "description": "Testing improved tool selection logic",
                "created_at": "2024-01-15T10:00:00Z",
                "metadata": {
                    "owner": "ml-team",
                    "project": "customer-support",
                    "tags": ["production", "tool-calling"]
                },
                "baseline_run_id": "660e8400-e29b-41d4-a716-446655440001"
            }
        }


class ExperimentRun(BaseModel):
    """
    A specific run within an experiment with config snapshot.
    
    Attributes:
        run_id: Unique identifier for this run
        experiment_id: ID of the parent experiment
        run_name: Human-readable name for this run
        agent_design: Complete agent design configuration
        config_snapshot: Full configuration snapshot for reproducibility
        git_commit: Git commit hash for code versioning
        timestamp: When the run was created
        evaluation_count: Number of evaluations in this run
        aggregate_metrics: Aggregate metrics across all evaluations
        metadata: Additional run metadata
    """
    
    run_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this run"
    )
    experiment_id: UUID = Field(
        ...,
        description="ID of the parent experiment"
    )
    run_name: str = Field(..., description="Human-readable name for this run")
    agent_design: AgentDesign = Field(
        ...,
        description="Complete agent design configuration"
    )
    config_snapshot: Dict[str, Any] = Field(
        ...,
        description="Full configuration snapshot for reproducibility"
    )
    git_commit: Optional[str] = Field(
        None,
        description="Git commit hash for code versioning"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the run was created"
    )
    evaluation_count: int = Field(
        default=0,
        description="Number of evaluations in this run",
        ge=0
    )
    aggregate_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Aggregate metrics across all evaluations"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional run metadata"
    )
    
    @field_validator("run_id", "experiment_id")
    @classmethod
    def validate_uuid(cls, v: UUID) -> UUID:
        """Validate that UUIDs are not None."""
        if v is None:
            raise ValueError("UUID cannot be None")
        return v
    
    @field_validator("run_name")
    @classmethod
    def validate_run_name(cls, v: str) -> str:
        """Validate that run_name is not empty."""
        if not v or not v.strip():
            raise ValueError("Run name cannot be empty")
        return v.strip()
    
    @field_validator("git_commit")
    @classmethod
    def validate_git_commit(cls, v: Optional[str]) -> Optional[str]:
        """Validate git commit hash format (40 hex characters)."""
        if v is not None:
            v = v.strip()
            if len(v) != 40 or not all(c in "0123456789abcdef" for c in v.lower()):
                raise ValueError(
                    f"Invalid git commit hash: {v}. Must be 40 hexadecimal characters"
                )
        return v
    
    @field_validator("evaluation_count")
    @classmethod
    def validate_evaluation_count(cls, v: int) -> int:
        """Validate that evaluation_count is non-negative."""
        if v < 0:
            raise ValueError(f"evaluation_count must be non-negative, got {v}")
        return v
    
    @field_validator("aggregate_metrics")
    @classmethod
    def validate_aggregate_metrics(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all metric scores are between 0 and 1."""
        for metric_name, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Metric '{metric_name}' score must be between 0 and 1, got {score}"
                )
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "run_id": "770e8400-e29b-41d4-a716-446655440002",
                "experiment_id": "550e8400-e29b-41d4-a716-446655440000",
                "run_name": "baseline-v1",
                "agent_design": {
                    "framework": "langgraph",
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "tools": ["search", "calculator"],
                    "system_prompt": "You are a helpful assistant...",
                    "parameters": {"temperature": 0.7}
                },
                "config_snapshot": {
                    "metric_set": ["tool_accuracy", "answer_relevance"],
                    "thresholds": {"tool_calling": 0.8},
                    "evaluation_mode": "offline"
                },
                "git_commit": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",  # pragma: allowlist secret
                "timestamp": "2024-01-15T10:30:00Z",
                "evaluation_count": 100,
                "aggregate_metrics": {
                    "tool_accuracy": 0.85,
                    "answer_relevance": 0.92
                },
                "metadata": {
                    "environment": "staging",
                    "test_suite": "customer_support_v1"
                }
            }
        }
