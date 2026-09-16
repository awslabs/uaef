# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MultiAgentTrace data model."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, field_validator

from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message
from uaef.models.tool_call import ToolCall


class WorkflowStatus(str, Enum):
    """Status of a multi-agent workflow."""
    
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class CoordinationEvent(BaseModel):
    """
    Represents a coordination event between agents.
    
    Attributes:
        event_id: Unique identifier for this event
        from_agent: ID of the agent initiating the coordination
        to_agent: ID of the agent receiving the coordination
        event_type: Type of coordination (delegation, handoff, collaboration)
        message: Description of the coordination event
        timestamp: When the event occurred
        metadata: Additional metadata about the event
    """
    
    event_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this event")
    from_agent: str = Field(..., description="ID of the agent initiating the coordination")
    to_agent: str = Field(..., description="ID of the agent receiving the coordination")
    event_type: str = Field(..., description="Type of coordination event")
    message: str = Field(..., description="Description of the coordination event")
    timestamp: datetime = Field(..., description="When the event occurred")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the event"
    )
    
    @field_validator("from_agent", "to_agent", "event_type", "message")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate that string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class MultiAgentTrace(BaseModel):
    """
    Canonical representation of a multi-agent execution trace.
    
    Attributes:
        trace_id: Unique identifier for this multi-agent trace
        session_id: Identifier for the conversation session (optional)
        agent_traces: Dictionary mapping agent_id to AgentTrace
        coordination_events: List of coordination events between agents
        workflow_status: Status of the overall workflow
        total_latency: Total execution time across all agents in seconds (optional)
        total_cost: Total estimated cost across all agents in USD (optional)
        metadata: Additional metadata about the multi-agent trace
        timestamp: When the trace was created
    """
    
    trace_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this trace")
    session_id: Optional[str] = Field(None, description="Identifier for the conversation session")
    agent_traces: Dict[str, AgentTrace] = Field(
        default_factory=dict,
        description="Dictionary mapping agent_id to AgentTrace"
    )
    coordination_events: List[CoordinationEvent] = Field(
        default_factory=list,
        description="List of coordination events between agents"
    )
    workflow_status: WorkflowStatus = Field(
        ...,
        description="Status of the overall workflow"
    )
    total_latency: Optional[float] = Field(
        None,
        description="Total execution time across all agents in seconds",
        ge=0.0
    )
    total_cost: Optional[float] = Field(
        None,
        description="Total estimated cost across all agents in USD",
        ge=0.0
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the multi-agent trace"
    )
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
    
    @field_validator("workflow_status")
    @classmethod
    def validate_workflow_status(cls, v: WorkflowStatus) -> WorkflowStatus:
        """Validate that workflow_status is a valid enum value."""
        if not isinstance(v, WorkflowStatus):
            raise ValueError(f"Invalid workflow_status: {v}")
        return v
    
    @field_validator("total_latency", "total_cost")
    @classmethod
    def validate_positive_values(cls, v: Optional[float]) -> Optional[float]:
        """Validate that latency and cost are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Latency and cost must be non-negative")
        return v

    @computed_field  # type: ignore[misc]
    @property
    def messages(self) -> List[Message]:
        """Aggregate messages from all sub-agent traces in order."""
        all_messages: List[Message] = []
        for trace in self.agent_traces.values():
            all_messages.extend(trace.messages)
        return all_messages

    @computed_field  # type: ignore[misc]
    @property
    def tool_calls(self) -> List[ToolCall]:
        """Aggregate tool calls from all sub-agent traces in order."""
        all_tool_calls: List[ToolCall] = []
        for trace in self.agent_traces.values():
            all_tool_calls.extend(trace.tool_calls)
        return all_tool_calls

    @computed_field  # type: ignore[misc]
    @property
    def input_tokens(self) -> Optional[int]:
        """Sum input tokens from all sub-agent traces."""
        totals = [t.input_tokens for t in self.agent_traces.values() if t.input_tokens is not None]
        return sum(totals) if totals else None

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens(self) -> Optional[int]:
        """Sum output tokens from all sub-agent traces."""
        totals = [t.output_tokens for t in self.agent_traces.values() if t.output_tokens is not None]
        return sum(totals) if totals else None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_id": "session_123",
                "agent_traces": {
                    "coordinator": {},
                    "specialist": {}
                },
                "coordination_events": [
                    {
                        "from_agent": "coordinator",
                        "to_agent": "specialist",
                        "event_type": "delegation",
                        "message": "Delegating task to specialist",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "workflow_status": "completed",
                "total_latency": 2.5,
                "total_cost": 0.005,
                "metadata": {},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
