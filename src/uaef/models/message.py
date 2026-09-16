# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message data model."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from uaef.models.tool_call import ToolCall


class MessageRole(str, Enum):
    """Role of the message sender."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """
    Represents a single message in an agent conversation.
    
    Attributes:
        role: Role of the message sender (user, assistant, system, tool)
        content: Text content of the message
        tool_calls: List of tool calls made in this message (optional)
        metadata: Additional metadata about the message (optional)
        timestamp: When the message was created
    """
    
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Text content of the message")
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of tool calls made in this message"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the message"
    )
    timestamp: datetime = Field(..., description="When the message was created")
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v: MessageRole) -> MessageRole:
        """Validate that role is a valid enum value."""
        if not isinstance(v, MessageRole):
            raise ValueError(f"Invalid role: {v}. Must be one of {list(MessageRole)}")
        return v
    
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
                "role": "assistant",
                "content": "I'll search the database for customer orders.",
                "tool_calls": [
                    {
                        "name": "search_database",
                        "arguments": {"query": "customer orders"},
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "metadata": {"model": "gpt-4", "temperature": 0.7},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
