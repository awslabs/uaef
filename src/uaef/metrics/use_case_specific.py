# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use-case-specific evaluation metrics.

This module hosts metrics whose semantics only apply to specific agent use
cases (currently: customer-service / contact-center agents). They are NOT
included in the default registry — UAEF is a general-purpose agent
evaluation framework and metrics like "did the agent escalate to a human?"
or "was the customer's issue resolved?" are meaningless for, e.g., a coding
assistant or an internal data-pipeline agent.

To opt in, call ``register_use_case_specific_metrics()`` once at startup:

    from uaef.metrics.registry import register_use_case_specific_metrics

    register_use_case_specific_metrics()
    # "containment" and "resolution" are now selectable by name:
    result = evaluate(trace, metrics=["containment", "resolution"])

The classes themselves remain importable directly for advanced users who
want to register them manually via ``register_metric(MetricCls)``.
"""

from typing import Optional

from uaef.metrics.multi_turn import _AMACEFullTraceMetric
from uaef.models.evaluation_input import EvaluationInput


# --- Containment ----------------------------------------------------------


class ContainmentMetric(_AMACEFullTraceMetric):
    """
    Boolean: was the conversation handled by the chatbot without escalating
    or redirecting to a human agent? Returns 1.0 (contained) or 0.0 (not).
    """

    def get_name(self) -> str:
        return "containment"

    def get_description(self) -> Optional[str]:
        return "Whether the conversation was handled by the agent without escalation"

    def _build_prompt(self, conversation: str, evaluation_input: EvaluationInput) -> Optional[str]:
        prompt = """
You are an expert in conversation analysis. Your task is to determine if the conversation is effectively contained,
meaning the agent did NOT escalate, transfer, or redirect to a human agent.

Conversation:
<conversation>{conversation}</conversation>

Output should be in JSON format with keys "Score" and "Reason". "Score" value must be a number 1 or 0.
Conversation is NOT contained (Score = 0) if the agent escalated or transferred/redirected the user
to additional assistance, or directed the user to an official customer-support channel.
Otherwise the conversation is contained (Score = 1).

Provide reasoning under "Reason" as a string.
Do not restate the question. Only return JSON output.
"""
        return prompt.format(conversation=conversation)

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        raw = response_json.get("Score", response_json.get("score"))
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        try:
            v = int(float(raw))
        except (TypeError, ValueError):
            return None, reason, {"raw": raw}
        if v not in (0, 1):
            return None, reason, {"raw": raw}
        return float(v), reason, {"raw": v}


# --- Resolution -----------------------------------------------------------


class ResolutionMetric(_AMACEFullTraceMetric):
    """
    Boolean: was the customer's issue resolved by the end of the conversation?
    Returns 1.0 (resolved) or 0.0 (unresolved).
    """

    def get_name(self) -> str:
        return "resolution"

    def get_description(self) -> Optional[str]:
        return "Whether the customer's issue was resolved by the end of the conversation"

    def _build_prompt(self, conversation: str, evaluation_input: EvaluationInput) -> Optional[str]:
        prompt = """
You are an expert in conversation analysis. Analyze the conversation between a customer and a chatbot agent
and determine if the customer's issue has been resolved.

Guidelines:
1. Review the entire conversation carefully.
2. Rate resolution as follows:
   - If the issue is not resolved, set "Resolution" to 0.
   - If the issue is resolved, set "Resolution" to 1.
3. Notes:
   - If the chatbot provides instructions but lacks the ability to fully solve the issue and transfers
     the customer to human support, treat the issue as RESOLVED (Resolution = 1).
   - If the customer directly asks the chatbot to transfer to human support, treat as UNRESOLVED (Resolution = 0).
4. Provide a brief explanation in the "Reason" field.
5. Output JSON with keys "Resolution" (integer 0 or 1) and "Reason" (string).

<conversation>
{conversation}
</conversation>

Do not restate the question. Only return JSON output.
"""
        return prompt.format(conversation=conversation)

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        raw = response_json.get("Resolution", response_json.get("resolution", response_json.get("Score")))
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        # Accept booleans, "true"/"false" strings, and 0/1.
        if isinstance(raw, bool):
            v = 1 if raw else 0
        elif isinstance(raw, str):
            sv = raw.strip().lower()
            if sv in ("true", "yes", "1"):
                v = 1
            elif sv in ("false", "no", "0"):
                v = 0
            else:
                return None, reason, {"raw": raw}
        else:
            try:
                v = int(float(raw))
            except (TypeError, ValueError):
                return None, reason, {"raw": raw}
            if v not in (0, 1):
                return None, reason, {"raw": raw}
        return float(v), reason, {"raw": v}


__all__ = ["ContainmentMetric", "ResolutionMetric"]
