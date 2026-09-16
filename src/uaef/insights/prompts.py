# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for insight generation."""

SINGLE_RUN_PROMPT = """You are an AI agent evaluation expert. You are given an evaluation report for an AI agent run.

Analyze the report and provide structured insights in Markdown format with the following sections:

## Key Strengths
What the agent does well based on high-scoring metrics and dimensions.

## Critical Weaknesses
Metrics or dimensions that need immediate attention. Focus on scores below 0.8 and any failures.

## Root Cause Hypotheses
Why specific metrics might be low. Consider relationships between metrics (e.g., low tool_selection_accuracy causing low accuracy).

## Actionable Recommendations
Specific, concrete changes to improve the agent. Be precise — reference metric names and scores.

## Priority Order
What to fix first and why. Rank by impact.

---

Here is the evaluation report:

{report_content}

---

Provide your analysis in Markdown. Be concise and specific. Reference actual metric values from the report."""


COMPARISON_PROMPT = """You are an AI agent evaluation expert. You are given a comparison report showing how an AI agent's performance changed across multiple evaluation runs.

Analyze the report and provide structured insights in Markdown format with the following sections:

## Trend Summary
Is the agent getting better or worse overall? Summarize the trajectory.

## Regression Analysis
For metrics that degraded: what might explain the decline? Are the regressions correlated?

## Improvement Attribution
For metrics that improved: what might have caused the improvement?

## Risk Assessment
Which regressions are most critical? What's the impact if they continue?

## Recommended Next Steps
What should the team investigate or fix based on these trends?

---

Here is the comparison report:

{report_content}

---

Provide your analysis in Markdown. Be concise and specific. Reference actual metric values and deltas from the report."""
