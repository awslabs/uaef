# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GroundTruth data model."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from uaef.models.tool_call import ToolCall


class GroundTruth(BaseModel):
    """
    Expected correct outputs for validating agent responses.
    
    Attributes:
        expected_output: Expected text response from the agent (optional)
        expected_tool_calls: List of expected tool calls (optional)
        expected_arguments: Dictionary of expected arguments for tools (optional)
        context_documents: List of context documents that should be used (optional)
    """
    
    expected_output: Optional[str] = Field(
        None,
        description="Expected text response from the agent"
    )
    expected_tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of expected tool calls"
    )
    expected_arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of expected arguments for tools"
    )
    context_documents: List[str] = Field(
        default_factory=list,
        description="List of context documents that should be used"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "expected_output": "I found 5 customer orders matching your query.",
                "expected_tool_calls": [
                    {
                        "name": "search_database",
                        "arguments": {"query": "customer orders", "limit": 10},
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "expected_arguments": {
                    "search_database": {"query": "customer orders"}
                },
                "context_documents": [
                    "Customer database contains order history...",
                    "Search API documentation..."
                ]
            }
        }
