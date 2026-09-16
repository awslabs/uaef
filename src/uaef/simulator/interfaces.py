# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Conversation Simulator Interface Definitions.

This module defines the input and output specifications for the conversation simulator.
These interfaces provide the API contract for future simulator implementations.

PLACEHOLDER: Full implementation deferred to future work.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from uaef.models.message import Message
from uaef.models.tool_call import ToolCall


# ============================================================================
# Input Specifications
# ============================================================================


class ComplexityLevel(str, Enum):
    """Complexity level for scenario generation."""
    
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ScenarioConfig(BaseModel):
    """
    Configuration for scenario generation.
    
    Defines the domain, user goal, success criteria, and complexity level
    for generating a test scenario.
    
    Attributes:
        domain: The domain or topic area (e.g., "customer_support", "data_analysis")
        user_goal: The goal the user is trying to achieve
        success_criteria: List of criteria that define successful completion
        complexity_level: How complex the scenario should be
        initial_context: Optional initial context or background information
        constraints: Optional constraints or limitations for the scenario
    """
    
    domain: str = Field(..., description="Domain or topic area for the scenario")
    user_goal: str = Field(..., description="Goal the user is trying to achieve")
    success_criteria: List[str] = Field(
        ...,
        description="Criteria that define successful completion"
    )
    complexity_level: ComplexityLevel = Field(
        default=ComplexityLevel.MODERATE,
        description="Complexity level for the scenario"
    )
    initial_context: Optional[str] = Field(
        None,
        description="Initial context or background information"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints or limitations for the scenario"
    )


class CommunicationStyle(str, Enum):
    """Communication style for user behavior."""
    
    DIRECT = "direct"
    VERBOSE = "verbose"
    TERSE = "terse"
    TECHNICAL = "technical"
    NON_TECHNICAL = "non_technical"


class KnowledgeLevel(str, Enum):
    """Knowledge level of the simulated user."""
    
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class UserBehaviorModel(BaseModel):
    """
    Model for simulating user behavior in conversations.
    
    Defines personality traits, patience level, knowledge level, and
    communication style for generating realistic user messages.
    
    Attributes:
        personality: Personality traits (e.g., "frustrated", "patient", "curious")
        patience_level: How patient the user is (1-10, higher = more patient)
        knowledge_level: User's knowledge level in the domain
        communication_style: How the user communicates
        error_tolerance: How tolerant the user is of errors (0-1)
        clarification_frequency: How often user asks for clarification (0-1)
    """
    
    personality: str = Field(
        default="neutral",
        description="Personality traits of the user"
    )
    patience_level: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Patience level (1-10, higher = more patient)"
    )
    knowledge_level: KnowledgeLevel = Field(
        default=KnowledgeLevel.INTERMEDIATE,
        description="User's knowledge level in the domain"
    )
    communication_style: CommunicationStyle = Field(
        default=CommunicationStyle.DIRECT,
        description="How the user communicates"
    )
    error_tolerance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Tolerance for errors (0-1)"
    )
    clarification_frequency: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Frequency of clarification requests (0-1)"
    )


class AgentConfig(BaseModel):
    """
    Configuration for the agent being tested.
    
    Defines the framework, model, tools, and system prompt for the agent
    that will participate in the simulated conversation.
    
    Attributes:
        framework: Agent framework (e.g., "langgraph", "bedrock", "langchain")
        model: LLM model identifier (e.g., "claude-3-sonnet", "gpt-4")
        tools: List of available tool names
        system_prompt: System prompt for the agent
        parameters: Additional agent parameters (temperature, max_tokens, etc.)
    """
    
    framework: str = Field(..., description="Agent framework")
    model: str = Field(..., description="LLM model identifier")
    tools: List[str] = Field(
        default_factory=list,
        description="List of available tool names"
    )
    system_prompt: str = Field(..., description="System prompt for the agent")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent parameters"
    )


class GenerationParameters(BaseModel):
    """
    Parameters for conversation generation.
    
    Controls the generation process including max turns, adversarial mode,
    and ambiguity level.
    
    Attributes:
        max_turns: Maximum number of conversation turns
        adversarial_mode: Whether to generate adversarial user behavior
        ambiguity_level: Level of ambiguity in user messages (0-1)
        include_edge_cases: Whether to include edge cases
        seed: Random seed for reproducibility
    """
    
    max_turns: int = Field(
        default=10,
        ge=1,
        description="Maximum number of conversation turns"
    )
    adversarial_mode: bool = Field(
        default=False,
        description="Whether to generate adversarial user behavior"
    )
    ambiguity_level: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Level of ambiguity in user messages (0-1)"
    )
    include_edge_cases: bool = Field(
        default=False,
        description="Whether to include edge cases"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )


# ============================================================================
# Output Specifications
# ============================================================================


class GeneratedScenario(BaseModel):
    """
    A generated test scenario.
    
    Contains the scenario definition including ID, name, description,
    initial context, user goal, success criteria, and expected turns.
    
    Attributes:
        scenario_id: Unique identifier for the scenario
        name: Human-readable name for the scenario
        description: Detailed description of the scenario
        initial_context: Initial context for the conversation
        user_goal: The goal the user is trying to achieve
        success_criteria: Criteria that define successful completion
        expected_turns: Expected number of conversation turns
        metadata: Additional metadata about the scenario
    """
    
    scenario_id: str = Field(..., description="Unique identifier for the scenario")
    name: str = Field(..., description="Human-readable name for the scenario")
    description: str = Field(..., description="Detailed description of the scenario")
    initial_context: str = Field(..., description="Initial context for the conversation")
    user_goal: str = Field(..., description="Goal the user is trying to achieve")
    success_criteria: List[str] = Field(
        ...,
        description="Criteria that define successful completion"
    )
    expected_turns: int = Field(
        ...,
        ge=1,
        description="Expected number of conversation turns"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the scenario"
    )


class GeneratedConversation(BaseModel):
    """
    A generated conversation trace.
    
    Contains the complete conversation as a list of Message objects with
    role, content, tool calls, and timestamps.
    
    Attributes:
        conversation_id: Unique identifier for the conversation
        scenario_id: ID of the scenario this conversation is based on
        messages: List of messages in the conversation
        turn_count: Number of turns in the conversation
        completed: Whether the conversation completed successfully
        completion_reason: Reason for conversation completion
        metadata: Additional metadata about the conversation
    """
    
    conversation_id: str = Field(
        ...,
        description="Unique identifier for the conversation"
    )
    scenario_id: str = Field(
        ...,
        description="ID of the scenario this conversation is based on"
    )
    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation"
    )
    turn_count: int = Field(..., ge=0, description="Number of turns in the conversation")
    completed: bool = Field(..., description="Whether the conversation completed successfully")
    completion_reason: str = Field(
        ...,
        description="Reason for conversation completion"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the conversation"
    )


class TurnGroundTruth(BaseModel):
    """
    Ground truth for a single conversation turn.
    
    Contains expected output, tool calls, and arguments for validation.
    
    Attributes:
        turn_number: Turn number (0-indexed)
        expected_output: Expected agent output for this turn
        expected_tool_calls: Expected tool calls for this turn
        expected_arguments: Expected arguments for each tool call
        validation_criteria: Additional validation criteria for this turn
    """
    
    turn_number: int = Field(..., ge=0, description="Turn number (0-indexed)")
    expected_output: Optional[str] = Field(
        None,
        description="Expected agent output for this turn"
    )
    expected_tool_calls: List[str] = Field(
        default_factory=list,
        description="Expected tool calls for this turn"
    )
    expected_arguments: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Expected arguments for each tool call"
    )
    validation_criteria: List[str] = Field(
        default_factory=list,
        description="Additional validation criteria for this turn"
    )


class GeneratedGroundTruth(BaseModel):
    """
    Ground truth for the entire conversation.
    
    Contains expected outputs, tool calls, and arguments for each turn
    to enable automated evaluation.
    
    Attributes:
        conversation_id: ID of the conversation this ground truth is for
        turn_ground_truths: Ground truth for each turn
        overall_success_criteria: Overall success criteria for the conversation
        metadata: Additional metadata about the ground truth
    """
    
    conversation_id: str = Field(
        ...,
        description="ID of the conversation this ground truth is for"
    )
    turn_ground_truths: List[TurnGroundTruth] = Field(
        ...,
        description="Ground truth for each turn"
    )
    overall_success_criteria: List[str] = Field(
        default_factory=list,
        description="Overall success criteria for the conversation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the ground truth"
    )


class TestPackage(BaseModel):
    """
    Complete test package for evaluation.
    
    Bundles scenario, conversation, ground truth, and metadata into a
    single package ready for automated evaluation.
    
    Attributes:
        package_id: Unique identifier for the test package
        scenario: The generated scenario
        conversation: The generated conversation
        ground_truth: The generated ground truth
        generation_timestamp: When the package was generated
        generation_parameters: Parameters used for generation
        metadata: Additional metadata about the test package
    """
    
    package_id: str = Field(..., description="Unique identifier for the test package")
    scenario: GeneratedScenario = Field(..., description="The generated scenario")
    conversation: GeneratedConversation = Field(..., description="The generated conversation")
    ground_truth: GeneratedGroundTruth = Field(..., description="The generated ground truth")
    generation_timestamp: datetime = Field(
        ...,
        description="When the package was generated"
    )
    generation_parameters: GenerationParameters = Field(
        ...,
        description="Parameters used for generation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the test package"
    )
    
    @field_validator("generation_timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Validate timestamp format."""
        if v is None:
            raise ValueError("Generation timestamp cannot be None")
        return v
