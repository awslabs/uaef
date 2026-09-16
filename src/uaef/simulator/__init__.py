# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Conversation Simulator Package - PLACEHOLDER Implementation.

This package provides interfaces and placeholder implementations for conversation
simulation and test package generation. Full implementation will be added in future work.

PLACEHOLDER: This is a stub implementation. The simulator provides:
- Interface definitions for input/output specifications
- Placeholder classes for conversation generation
- Placeholder classes for scenario orchestration
- Placeholder classes for test package generation

All methods raise NotImplementedError with descriptive messages indicating
the expected behavior for future implementation.

Example usage (future):
    from uaef.simulator import (
        ConversationGenerator,
        ScenarioOrchestrator,
        TestPackageGenerator,
        ScenarioConfig,
        UserBehaviorModel,
        AgentConfig,
        GenerationParameters,
    )
    
    # Configure scenario
    scenario = ScenarioConfig(
        domain="customer_support",
        user_goal="Get refund for defective product",
        success_criteria=["Refund approved", "Customer satisfied"],
        complexity_level="moderate"
    )
    
    # Configure user behavior
    user_model = UserBehaviorModel(
        personality="frustrated",
        patience_level=3,
        knowledge_level="novice",
        communication_style="direct"
    )
    
    # Configure agent
    agent_config = AgentConfig(
        framework="langgraph",
        model="claude-3-sonnet",
        tools=["search_orders", "process_refund"],
        system_prompt="You are a helpful customer support agent."
    )
    
    # Generate test package
    generator = TestPackageGenerator()
    package = generator.generate_test_package(
        scenario_config=scenario,
        user_model=user_model,
        agent_config=agent_config,
        parameters=GenerationParameters(max_turns=10)
    )
"""

# Interface definitions
from uaef.simulator.interfaces import (
    AgentConfig,
    CommunicationStyle,
    ComplexityLevel,
    GeneratedConversation,
    GeneratedGroundTruth,
    GeneratedScenario,
    GenerationParameters,
    KnowledgeLevel,
    ScenarioConfig,
    TestPackage,
    TurnGroundTruth,
    UserBehaviorModel,
)

# Placeholder implementations
from uaef.simulator.generator import ConversationGenerator
from uaef.simulator.orchestrator import ScenarioOrchestrator
from uaef.simulator.test_package import TestPackageGenerator

__all__ = [
    # Interface definitions - Input specifications
    "ScenarioConfig",
    "UserBehaviorModel",
    "AgentConfig",
    "GenerationParameters",
    "ComplexityLevel",
    "CommunicationStyle",
    "KnowledgeLevel",
    # Interface definitions - Output specifications
    "GeneratedScenario",
    "GeneratedConversation",
    "GeneratedGroundTruth",
    "TurnGroundTruth",
    "TestPackage",
    # Placeholder implementations
    "ConversationGenerator",
    "ScenarioOrchestrator",
    "TestPackageGenerator",
]
