# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentTrace data model."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from uaef.models.message import Message
from uaef.models.tool_call import ToolCall


class AgentTrace(BaseModel):
    """
    Canonical representation of an agent execution trace.
    
    This is the standardized format that all framework-specific traces
    are transformed into for evaluation.
    
    Attributes:
        trace_id: Unique identifier for this trace
        session_id: Identifier for the conversation session (optional)
        messages: List of messages in the conversation
        tool_calls: List of all tool calls made during execution
        input_tokens: Number of input tokens consumed (optional)
        output_tokens: Number of output tokens generated (optional)
        latency: Total execution time in seconds (optional)
        cost: Estimated cost in USD (optional)
        metadata: Additional metadata about the trace
        framework: Name of the agent framework used (optional)
        timestamp: When the trace was created
    """
    
    trace_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this trace")
    session_id: Optional[str] = Field(None, description="Identifier for the conversation session")
    messages: List[Message] = Field(
        default_factory=list,
        description="List of messages in the conversation"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of all tool calls made during execution"
    )
    input_tokens: Optional[int] = Field(
        None,
        description="Number of input tokens consumed",
        ge=0
    )
    output_tokens: Optional[int] = Field(
        None,
        description="Number of output tokens generated",
        ge=0
    )
    latency: Optional[float] = Field(
        None,
        description="Total execution time in seconds",
        ge=0.0
    )
    cost: Optional[float] = Field(
        None,
        description="Estimated cost in USD",
        ge=0.0
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the trace"
    )
    framework: Optional[str] = Field(None, description="Name of the agent framework used")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the trace was created"
    )
    
    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, v: UUID) -> UUID:
        """Validate that trace_id is a valid UUID."""
        if v is None:
            raise ValueError("trace_id cannot be None")
        return v
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Validate timestamp format."""
        if v is None:
            raise ValueError("Timestamp cannot be None")
        return v
    
    @field_validator("input_tokens", "output_tokens")
    @classmethod
    def validate_token_counts(cls, v: Optional[int]) -> Optional[int]:
        """Validate that token counts are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Token counts must be non-negative")
        return v
    
    @field_validator("latency", "cost")
    @classmethod
    def validate_positive_values(cls, v: Optional[float]) -> Optional[float]:
        """Validate that latency and cost are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Latency and cost must be non-negative")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_id": "session_123",
                "messages": [
                    {
                        "role": "user",
                        "content": "Find customer orders",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "tool_calls": [],
                "input_tokens": 150,
                "output_tokens": 75,
                "latency": 1.234,
                "cost": 0.0025,
                "metadata": {"model": "gpt-4"},
                "framework": "langgraph",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
