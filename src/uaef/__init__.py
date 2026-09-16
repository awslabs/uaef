# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Universal Agent Evaluation Framework (UAEF)

A comprehensive, framework-agnostic evaluation system for AI agents.
"""

__version__ = "0.2.0"

from uaef.models import (
    AgentTrace,
    Message,
    ToolCall,
    EvaluationResult,
    MetricScore,
)

__all__ = [
    "__version__",
    "AgentTrace",
    "Message",
    "ToolCall",
    "EvaluationResult",
    "MetricScore",
]
