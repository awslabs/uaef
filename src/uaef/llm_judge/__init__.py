# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM Judge module for UAEF.

This module provides LLM-based evaluation capabilities using AWS Bedrock,
including prompt templates, calibration, and batch processing.
"""

from uaef.llm_judge.bedrock_client import (
    BedrockClient,
    BedrockClientError,
    BedrockRateLimitError,
    BedrockServiceError,
    BedrockTimeoutError
)
from uaef.llm_judge.calibration import (
    CalibrationEngine,
    CalibrationMetrics,
    CalibrationSample
)
from uaef.llm_judge.judge import (
    LLMJudge,
    LLMJudgeError,
    get_llm_judge,
    reset_llm_judge,
    set_llm_judge
)
from uaef.llm_judge.prompt_templates import (
    PromptTemplateError,
    PromptTemplateLoader,
    PromptTemplateNotFoundError,
    PromptTemplateValidationError,
    get_template_loader,
    reset_template_loader,
    set_template_loader
)

__all__ = [
    # Bedrock Client
    "BedrockClient",
    "BedrockClientError",
    "BedrockRateLimitError",
    "BedrockServiceError",
    "BedrockTimeoutError",
    # LLM Judge
    "LLMJudge",
    "LLMJudgeError",
    "get_llm_judge",
    "set_llm_judge",
    "reset_llm_judge",
    # Prompt Templates
    "PromptTemplateLoader",
    "PromptTemplateError",
    "PromptTemplateNotFoundError",
    "PromptTemplateValidationError",
    "get_template_loader",
    "set_template_loader",
    "reset_template_loader",
    # Calibration
    "CalibrationEngine",
    "CalibrationSample",
    "CalibrationMetrics",
]
