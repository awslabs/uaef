# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ToolCall data model."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class ToolCall(BaseModel):
    """
    Represents a single tool invocation by an agent.
    
    Attributes:
        name: Name of the tool that was called
        arguments: Dictionary of arguments passed to the tool
        result: Result returned by the tool (optional)
        timestamp: When the tool was called
        execution_time: Time taken to execute the tool in seconds (optional)
        error: Error message if tool execution failed (optional)
    """
    
    name: str = Field(..., description="Name of the tool that was called", min_length=1)
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of arguments passed to the tool"
    )
    result: Optional[Any] = Field(default=None, description="Result returned by the tool")
    timestamp: datetime = Field(..., description="When the tool was called")
    execution_time: Optional[float] = Field(
        default=None,
        description="Time taken to execute the tool in seconds",
        ge=0.0
    )
    error: Optional[str] = Field(default=None, description="Error message if tool execution failed")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that tool name is not empty."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Validate timestamp format."""
        if v is None:
            raise ValueError("Timestamp cannot be None")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "name": "search_database",
                "arguments": {"query": "customer orders", "limit": 10},
                "result": {"count": 5, "items": []},
                "timestamp": "2024-01-15T10:30:00Z",
                "execution_time": 0.234,
                "error": None
            }
        }
