# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tool Calling Metrics
tool_metrics = ["tool_selection_accuracy", "tool_sequence_correctness", "parameter_quality", "mcp_compliance"]

# Response Quality Metrics
quality_metrics = ["answer_relevance", "completeness", "hallucination_score", "accuracy"]

# Responsible AI Metrics
responsible_metrics = ["safety_score", "bias_score", "prompt_injection_detection", "toxicity_score"]

# Performance Metrics
performance_metrics = ["latency_score", "token_efficiency", "cost_efficiency", "throughput"]

# Multi-Turn Metrics
multi_turn_metrics = ["context_retention", "coherence", "conversation_completeness", "turn_efficiency"]

# Multi-Agent Metrics
multi_ag_metrics = ["agent_utilization", "delegation_quality", "workflow_completion", "coordination_efficiency"]

# Reasoning Metrics
reasoning_metrics = ["chain_of_thought_coherence", "logical_consistency", "reasoning_step_correctness", "fallacy_detection"]

# # DeepEval Integration Metrics
deepeval_metrics = [
    "deepeval_contextual_precision",
    "deepeval_contextual_recall",
    "deepeval_contextual_relevancy",
    "deepeval_hallucination",
    "deepeval_faithfulness",
    "deepeval_answer_relevancy",
    "deepeval_tool_correctness",
]

# RAGAS Integration Metrics
ragas_metrics = [
    "ragas_faithfulness",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_answer_precision",
    "ragas_answer_recall",
    "ragas_answer_correctness",
    "ragas_tool_call_accuracy",
]

# # All metrics combined
all_metrics = (
    tool_metrics + quality_metrics + responsible_metrics + performance_metrics
    + multi_turn_metrics + multi_ag_metrics + reasoning_metrics + deepeval_metrics + ragas_metrics
)

# All metrics combined without deepeval
# all_metrics = (
#     tool_metrics + quality_metrics + responsible_metrics + performance_metrics
#     + multi_turn_metrics + multi_ag_metrics + reasoning_metrics + ragas_metrics
# )

# Single agent metrics 
single_ag_metrics = (
    tool_metrics + quality_metrics + responsible_metrics + performance_metrics
    + multi_turn_metrics + reasoning_metrics
)