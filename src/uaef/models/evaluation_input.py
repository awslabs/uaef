# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""EvaluationInput data model."""

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from uaef.models.agent_trace import AgentTrace
from uaef.models.ground_truth import GroundTruth
from uaef.models.multi_agent_trace import MultiAgentTrace


class EvaluationInput(BaseModel):
    """
    Input data for evaluation.
    
    Attributes:
        trace: Agent trace to evaluate (single or multi-agent)
        ground_truth: Expected correct outputs (optional)
        context: List of context documents (optional)
        evaluation_config: Configuration for evaluation (will be added separately)
    """
    
    trace: Union[AgentTrace, MultiAgentTrace] = Field(
        ...,
        description="Agent trace to evaluate (single or multi-agent)"
    )
    ground_truth: Optional[GroundTruth] = Field(
        None,
        description="Expected correct outputs"
    )
    context: List[str] = Field(
        default_factory=list,
        description="List of context documents"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "trace": {
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                    "messages": [],
                    "tool_calls": []
                },
                "ground_truth": {
                    "expected_output": "Expected response",
                    "expected_tool_calls": []
                },
                "context": ["Context document 1", "Context document 2"]
            }
        }
