# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn conversation metrics for UAEF.

This module implements metrics for evaluating multi-turn conversations including
context retention, coherence, conversation completeness, turn efficiency, plus
additional customer-service-flavored LLM-judge metrics ported from the AMACE
Conversation Evaluator (role adherence, holistic LLM judge, user satisfaction,
sentiment, agent tone, naturalness, instruction compliance, optimum turns).

Per-turn variants of sentiment / agent_tone / naturalness — `per_turn_sentiment`,
`per_turn_agent_tone`, `per_turn_naturalness` — score each user→assistant
exchange independently and aggregate by mean. They sit alongside the full-trace
versions; users opt in by metric name.
"""

import asyncio
import json
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from uaef.config import get_config
from uaef.logging import get_logger
from uaef.metrics.base import BaseMetric
from uaef.metrics.utils import (
    build_claude_judge_body,
    build_judge_system_prompt,
    extract_json_from_llm_response,
    validate_judge_response,
    wrap_untrusted,
)
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.message import Message, MessageRole
from uaef.models.metric_score import MetricScore

logger = get_logger(__name__)


_RESP_FMT_SCORE_REASONING = """{
                       "score":float,
                       "reasoning": str
                   }
               """


def _format_conversation(messages: list) -> str:
    """Build a `ROLE: content` transcript from a list of trace messages."""
    return "\n\n".join(f"{msg.role.upper()}: {msg.content}" for msg in messages)


def _extract_user_assistant_pairs(messages: list) -> list[tuple[Message, Message]]:
    """Return ordered (user_msg, assistant_msg) pairs from a message list.

    Walks the list once. A USER message opens a pending pair; the next
    ASSISTANT message closes it. SYSTEM and TOOL messages are skipped.
    Consecutive assistant messages without an intervening user message:
    only the first closes the pending pair (subsequent assistant messages
    are typically tool-result follow-ups and are ignored for scoring).
    A trailing user message with no assistant reply emits no pair.
    """
    pairs: list[tuple[Message, Message]] = []
    pending_user = None
    for msg in messages:
        if msg.role == MessageRole.USER:
            pending_user = msg
        elif msg.role == MessageRole.ASSISTANT:
            if pending_user is not None:
                pairs.append((pending_user, msg))
                pending_user = None
        # SYSTEM / TOOL: skip
    return pairs


def _invoke_bedrock_judge(system_prompt: str, user_message: str, metric_name: str) -> dict:
    """
    Invoke the configured Bedrock judge with a system/user prompt split and
    return parsed JSON.

    Centralizes the boto3 client init, request formatting, invocation, and
    JSON extraction shared by every multi-turn LLM-judge metric in this module.

    Security review H-01: takes a system_prompt (rubric, built via
    build_judge_system_prompt) and a user_message (the untrusted conversation
    /pair content, wrapped via wrap_untrusted) as separate arguments, rather
    than one pre-concatenated prompt string, so callers cannot accidentally
    fall back to mixing trusted instructions and untrusted content in the
    same turn. See src/uaef/metrics/utils.py module docstring for the full
    rationale; this mirrors the split already applied in response_quality.py,
    reasoning.py, responsible_ai.py, and multi_agent.py.

    Args:
        system_prompt: The rubric/instructions (already passed through
            build_judge_system_prompt). Goes in Bedrock's `system` field.
        user_message: The untrusted content, already wrapped (via
            wrap_untrusted) plus any trusted plain-text context — goes in
            the single `user` turn.
        metric_name: Used for log context only.

    Returns:
        Parsed JSON dict returned by the judge.

    Raises:
        ConnectionError: Bedrock client could not be initialized.
        RuntimeError:    Bedrock invoke_model call failed.
        ValueError:      Judge response was not parseable as JSON.
    """
    try:
        config = get_config()
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=config.aws.region,
            aws_access_key_id=config.aws.access_key_id,
            aws_secret_access_key=config.aws.secret_access_key,
            aws_session_token=config.aws.session_token,
        )
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Bedrock client: {str(e)}")

    body = build_claude_judge_body(
        system_prompt=system_prompt,
        user_message=user_message,
        max_tokens=config.llm_judge.max_tokens,
        temperature=config.llm_judge.temperature,
    )

    try:
        response_obj = bedrock_client.invoke_model(
            modelId=config.llm_judge.model_id,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
    except Exception as e:
        raise RuntimeError(f"Bedrock API call failed: {str(e)}")

    response_body = json.loads(response_obj.get("body").read())
    raw_text = response_body.get("content")[0]["text"]
    try:
        return extract_json_from_llm_response(raw_text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"{metric_name}: LLM judge returned non-JSON response: {repr(raw_text)}")
        raise ValueError(str(e))


def _judge_failure_score(
    metric_name: str, exc: Exception, error_kind: str
) -> MetricScore:
    """Standard failure-path MetricScore for judge invocation errors."""
    return MetricScore(
        metric_name=metric_name,
        score=None,
        reasoning=f"Cannot evaluate: {error_kind}: {str(exc)}",
        metadata={"warning": "api_error", "error_details": str(exc)},
    )


# ---------------------------------------------------------------------------
# Existing UAEF multi-turn metrics
# ---------------------------------------------------------------------------


class ContextRetentionMetric(BaseMetric):
    """
    Metric for evaluating context retention across conversation turns.

    Evaluates whether the agent maintains and uses information from earlier turns
    in later responses. Uses LLM judge to assess context carryover.
    """

    def get_name(self) -> str:
        return "context_retention"

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Evaluates whether the agent maintains context across conversation turns"

    def get_dimension(self) -> Optional[str]:
        return "Multi-Turn"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        trace = evaluation_input.trace

        if len(trace.messages) < 4:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: fewer than 2 conversation turns",
                metadata={"warning": "not_applicable", "message_count": len(trace.messages)},
            )

        conversation = _format_conversation(trace.messages)
        system_prompt, user_message = self._build_prompt(conversation)

        try:
            response_json = _invoke_bedrock_judge(system_prompt, user_message, self.get_name())
            # Security review M-05: validate shape/types before use. This
            # metric's own prompt documents null as a valid "not applicable"
            # score (see _build_prompt above), so allow_null_score=True.
            response_json = validate_judge_response(response_json, allow_null_score=True)
            response_score = response_json["score"]
            response_reasoning = response_json.get("reasoning", "")

            if response_score is None:
                return MetricScore(
                    metric_name=self.get_name(),
                    score=None,
                    reasoning=f"Cannot evaluate: {response_reasoning}",
                    metadata={
                        "warning": "not_applicable",
                        "message_count": len(trace.messages),
                        "turn_count": len(trace.messages) // 2,
                    },
                )

            return MetricScore(
                metric_name=self.get_name(),
                score=response_score,
                reasoning=response_reasoning,
                metadata={
                    "message_count": len(trace.messages),
                    "turn_count": len(trace.messages) // 2,
                },
            )
        except ConnectionError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock client initialization failed")
        except RuntimeError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock API call failed")
        except ValueError as e:
            return _judge_failure_score(self.get_name(), e, "Invalid score from LLM judge")
        except ClientError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock API error")
        except Exception as e:
            return _judge_failure_score(self.get_name(), e, "LLM judge error")

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)

    @staticmethod
    def _build_prompt(conversation: str) -> tuple[str, str]:
        # Security review H-01: rubric in system field; conversation is the
        # untrusted transcript (includes the agent's own turns) and is
        # wrapped.
        system_rubric = """
            You are a strict AI evaluator that assesses context retention in multi-turn conversations.
            Evaluate whether the agent maintains and appropriately uses information from earlier turns
            in its later responses. You will be given the full conversation history.

            IMPORTANT: First determine if context retention is applicable:
            - If the conversation consists of INDEPENDENT questions with no shared context to retain, set score to null
            - Only evaluate retention if there is context that SHOULD be retained across turns

            EVALUATION CRITERIA (apply strictly):

            1. **Remembers stated facts**: If the user provided personal details, preferences, or constraints
               in an earlier turn, the agent MUST use them — not ask again.
               - Asking for information the user already provided is a CRITICAL failure (score 0.0).

            2. **Consistency**: The agent must not contradict its own earlier statements.
               - Giving a different answer to the same question is a CRITICAL failure (score 0.0).

            3. **Builds on context**: Later responses should reference and build upon earlier exchanges,
               not treat each turn as a fresh conversation.

            4. **No unnecessary repetition**: The agent should not re-ask questions that were already answered.

            SCORING GUIDE (be strict):
            - 1.0: Perfect — agent uses all prior context accurately
            - 0.7-0.9: Good — agent uses most context, minor omissions
            - 0.3-0.6: Partial — agent remembers some things but misses important details
            - 0.0: Failure — agent forgets key facts, asks for already-provided info, or contradicts itself

            After providing your explanation in the "reasoning" field, score in the "score" field:
            - Use null if context retention is not applicable (independent questions, no shared context)
            - Use 0 to 1 scale if applicable, following the scoring guide above

            Strictly follow the below json format:{resp_fmt}."""
        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=_RESP_FMT_SCORE_REASONING))
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message


class CoherenceMetric(BaseMetric):
    """
    Metric for evaluating conversation coherence and logical flow.

    Evaluates whether the conversation flows logically and maintains coherence
    across turns. Uses LLM judge to assess logical consistency.
    """

    def get_name(self) -> str:
        return "coherence"

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Evaluates logical flow and coherence across conversation turns"

    def get_dimension(self) -> Optional[str]:
        return "Multi-Turn"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        trace = evaluation_input.trace

        if len(trace.messages) < 4:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: fewer than 2 conversation turns",
                metadata={"warning": "not_applicable", "message_count": len(trace.messages)},
            )

        conversation = _format_conversation(trace.messages)
        system_prompt, user_message = self._build_prompt(conversation)

        try:
            response_json = _invoke_bedrock_judge(system_prompt, user_message, self.get_name())
            # Security review M-05: validate shape/types before use, not
            # just the score's numeric range.
            response_json = validate_judge_response(response_json)
            response_score = response_json["score"]
            response_reasoning = response_json.get("reasoning", "")

            return MetricScore(
                metric_name=self.get_name(),
                score=response_score,
                reasoning=response_reasoning,
                metadata={
                    "message_count": len(trace.messages),
                    "turn_count": len(trace.messages) // 2,
                },
            )
        except ConnectionError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock client initialization failed")
        except RuntimeError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock API call failed")
        except ValueError as e:
            return _judge_failure_score(self.get_name(), e, "Invalid score from LLM judge")
        except ClientError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock API error")
        except Exception as e:
            return _judge_failure_score(self.get_name(), e, "LLM judge error")

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)

    @staticmethod
    def _build_prompt(conversation: str) -> tuple[str, str]:
        # Security review H-01: rubric in system field; conversation is the
        # untrusted transcript and is wrapped.
        system_rubric = """
            You are a strict AI evaluator that assesses coherence in multi-turn conversations.
            Evaluate whether the conversation flows logically and maintains coherence across turns.
            You will be given the full conversation history.

            EVALUATION CRITERIA (apply strictly):

            1. **Relevance**: Each assistant response MUST address the user's actual question or request.
               - Responding with completely unrelated content is a CRITICAL failure (score 0.0-0.1).

            2. **Consistency**: The agent must not contradict its own earlier statements.
               - Saying "it's raining" then "it's sunny" about the same moment is a CRITICAL failure (score 0.0-0.1).

            3. **Logical flow**: The conversation should progress naturally from one turn to the next.
               - Abrupt, unexplained topic jumps by the assistant are a significant flaw.

            4. **Topic tracking**: When the user changes topic, the agent should follow appropriately.
               - User-initiated topic changes are fine; agent-initiated random jumps are not.

            SCORING GUIDE (be strict):
            - 0.9-1.0: Excellent — every response is relevant, consistent, and flows naturally
            - 0.7-0.8: Good — mostly coherent with minor awkwardness
            - 0.4-0.6: Mediocre — some irrelevant or inconsistent responses
            - 0.1-0.3: Poor — multiple irrelevant responses or contradictions
            - 0.0: Incoherent — responses bear no relation to user queries, or directly contradict each other

            After providing your explanation in the "reasoning" field, score on a 0 to 1 scale in the "score" field.
            Strictly follow the below json format:{resp_fmt}."""
        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=_RESP_FMT_SCORE_REASONING))
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message


class ConversationCompletenessMetric(BaseMetric):
    """
    Metric for evaluating conversation completeness (goal achievement).

    Evaluates whether the user's goal was achieved by the end of the conversation.
    Optionally uses ground-truth success criteria; if absent, the judge extracts
    the user's intentions from the conversation itself.
    """

    def get_name(self) -> str:
        return "conversation_completeness"

    def requires_ground_truth(self) -> bool:
        return True

    def requires_llm_judge(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Evaluates whether the user's goal was achieved in the conversation"

    def get_dimension(self) -> Optional[str]:
        return "Multi-Turn"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        success_criteria = (
            evaluation_input.ground_truth.expected_output
            if evaluation_input.ground_truth
            else None
        )
        trace = evaluation_input.trace
        conversation = _format_conversation(trace.messages)
        system_prompt, user_message = self._build_prompt(conversation, success_criteria)

        try:
            response_json = _invoke_bedrock_judge(system_prompt, user_message, self.get_name())
            # Security review M-05: validate shape/types before use, not
            # just the score's numeric range.
            response_json = validate_judge_response(response_json)
            response_score = response_json["score"]
            response_reasoning = response_json.get("reasoning", "")

            return MetricScore(
                metric_name=self.get_name(),
                score=response_score,
                reasoning=response_reasoning,
                metadata={
                    "message_count": len(trace.messages),
                    "turn_count": len(trace.messages) // 2,
                    "success_criteria": (success_criteria or "")[:100],
                },
            )
        except ConnectionError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock client initialization failed")
        except RuntimeError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock API call failed")
        except ValueError as e:
            return _judge_failure_score(self.get_name(), e, "Invalid score from LLM judge")
        except ClientError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock API error")
        except Exception as e:
            return _judge_failure_score(self.get_name(), e, "LLM judge error")

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)

    @staticmethod
    def _build_prompt(conversation: str, success_criteria: Optional[str]) -> tuple[str, str]:
        # Security review H-01: rubric in system field; success_criteria is
        # trusted ground-truth data and stays as plain text; conversation is
        # the untrusted transcript and is wrapped.
        if success_criteria:
            system_rubric = """
                You are an AI evaluator that assesses conversation completeness and goal achievement.
                Please evaluate whether the user's goal was achieved by the end of the conversation.
                You will be given the full conversation history and the success criteria.

                In your evaluation, check whether:
                - All success criteria were met
                - The user's questions were answered
                - The user's problem was solved
                - The conversation reached a satisfactory conclusion
                - No critical information is missing

                After providing your explanation in the "reasoning" tab, you must score the conversation on a scale of 0 to 1 in the "score" tab,
                where 1 means fully complete (all success criteria met, goal achieved)
                and 0 means incomplete (success criteria not met, goal not achieved).
                Strictly follow the below json format:{resp_fmt}."""
            system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=_RESP_FMT_SCORE_REASONING))
            user_message = (
                f"[Success Criteria]\n{success_criteria}\n\n"
                + wrap_untrusted(conversation, "candidate_conversation")
            )
            return system_prompt, user_message

        system_rubric = """
            You are an AI evaluator that assesses conversation completeness and goal achievement.
            Please evaluate whether the user's goal was achieved by the end of the conversation.
            You will be given the full conversation history.

            IMPORTANT: Since no explicit success criteria is provided, first extract the user's intentions and goals from the conversation,
            then evaluate whether those goals were achieved.

            In your evaluation:
            1. Identify what the user wanted to accomplish (their goals/intentions)
            2. Check whether those goals were met by the end of the conversation
            3. Assess if the user's questions were answered
            4. Determine if the conversation reached a satisfactory conclusion

            After providing your explanation in the "reasoning" tab, you must score the conversation on a scale of 0 to 1 in the "score" tab,
            where 1 means fully complete (user's extracted goals achieved)
            and 0 means incomplete (user's goals not achieved).
            Strictly follow the below json format:{resp_fmt}."""
        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=_RESP_FMT_SCORE_REASONING))
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message


class TurnEfficiencyMetric(BaseMetric):
    """
    Metric for evaluating turn efficiency (how quickly goal is achieved).

    Deterministic — incorporates a completeness signal from trace metadata
    when available so unfinished conversations are penalized.
    """

    def __init__(self, expected_turns: int = 5):
        """Initialize. Fewer-than-expected turns = higher score."""
        self.expected_turns = expected_turns

    def get_name(self) -> str:
        return "turn_efficiency"

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return False

    def get_description(self) -> Optional[str]:
        return "Evaluates efficiency based on number of turns to achieve goal"

    def get_dimension(self) -> Optional[str]:
        return "Multi-Turn"

    def get_dependencies(self) -> dict:
        return {
            "requires_metadata": ["completeness_score"],
            "requires_metrics": ["conversation_completeness"],
        }

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        trace = evaluation_input.trace
        turn_count = len(trace.messages) // 2

        if turn_count == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no conversation turns",
                metadata={"warning": "not_applicable", "turn_count": 0},
            )

        completeness = trace.metadata.get("completeness_score", 1.0)

        if turn_count <= self.expected_turns:
            turn_efficiency = 1.0
        else:
            turn_efficiency = max(0.0, min(1.0, self.expected_turns / turn_count))

        score = completeness * turn_efficiency

        if completeness < 1.0:
            reasoning = (
                f"Completed in {turn_count} turns with {completeness:.1%} goal achievement "
                f"(efficiency: {score:.1%})"
            )
        elif turn_count <= self.expected_turns:
            reasoning = f"Completed in {turn_count} turns (within expected {self.expected_turns})"
        else:
            reasoning = (
                f"Completed in {turn_count} turns "
                f"(exceeds expected {self.expected_turns}, efficiency: {score:.1%})"
            )

        return MetricScore(
            metric_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            metadata={
                "turn_count": turn_count,
                "expected_turns": self.expected_turns,
                "completeness": completeness,
                "turn_efficiency": turn_efficiency,
                "within_expected": turn_count <= self.expected_turns,
            },
        )

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


# ---------------------------------------------------------------------------
# AMACE-derived multi-turn LLM-judge metrics
# ---------------------------------------------------------------------------


class _AMACEFullTraceMetric(BaseMetric):
    """
    Shared scaffolding for full-trace AMACE-derived metrics.

    Subclasses define ``_PROMPT_TEMPLATE`` (via ``_build_prompt``) and
    ``_parse_response`` to translate the judge JSON into a (score, reasoning,
    metadata) triple. The score returned by ``_parse_response`` must already be
    in [0, 1]; subclasses normalize 1–5 raw scores before returning.
    """

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return True

    def get_dimension(self) -> Optional[str]:
        return "Multi-Turn"

    def _min_messages(self) -> int:
        """Subclasses can require more than the default 2-message floor."""
        return 2

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        """Return (system_prompt, user_message), or None to short-circuit as
        not_applicable.

        Security review H-01: subclasses build their rubric via
        build_judge_system_prompt (-> system_prompt) and wrap the untrusted
        conversation via wrap_untrusted (-> part of user_message), rather
        than returning one pre-concatenated string.
        """
        raise NotImplementedError

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        """Translate judge JSON into (score, reasoning, extra_metadata)."""
        raise NotImplementedError

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        trace = evaluation_input.trace

        if len(trace.messages) < self._min_messages():
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: insufficient conversation length",
                metadata={"warning": "not_applicable", "message_count": len(trace.messages)},
            )

        conversation = _format_conversation(trace.messages)
        built = self._build_prompt(conversation, evaluation_input)
        if built is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: required input missing",
                metadata={"warning": "not_applicable"},
            )
        system_prompt, user_message = built

        try:
            response_json = _invoke_bedrock_judge(system_prompt, user_message, self.get_name())
            score, reasoning, extra_meta = self._parse_response(response_json, evaluation_input)

            if score is not None and not (0.0 <= score <= 1.0):
                raise ValueError(f"Invalid score after normalization: {score}")

            metadata = {
                "message_count": len(trace.messages),
                "turn_count": len(trace.messages) // 2,
            }
            metadata.update(extra_meta)

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata=metadata,
            )
        except ConnectionError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock client initialization failed")
        except RuntimeError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock API call failed")
        except ValueError as e:
            return _judge_failure_score(self.get_name(), e, "Invalid score from LLM judge")
        except KeyError as e:
            return _judge_failure_score(self.get_name(), e, "Missing key in LLM judge response")
        except ClientError as e:
            return _judge_failure_score(self.get_name(), e, "Bedrock API error")
        except Exception as e:
            return _judge_failure_score(self.get_name(), e, "LLM judge error")

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        return self.calculate(evaluation_input)


def _normalize_1_to_5(raw: Any) -> Optional[float]:
    """Normalize a raw 1–5 integer/float to [0, 1]. Returns None if not 1–5."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if not (1.0 <= v <= 5.0):
        return None
    return (v - 1.0) / 4.0


class _AMACEPerTurnMetric(BaseMetric):
    """Shared scaffolding for per-turn AMACE-derived metrics.

    Each user→assistant pair is judged independently. Subclasses provide a
    pair-scoped prompt and parser; the base handles pair extraction,
    parallel dispatch via `asyncio.gather`, mean aggregation across turns,
    and metadata assembly.

    Methodology contrast with `_AMACEFullTraceMetric`:
      - Full-trace: one judge call sees the entire conversation and is
        instructed to weigh later turns more heavily.
      - Per-turn:   N independent judge calls, each seeing exactly one
        user→assistant pair. No cross-turn context, no recency weighting.

    The `MetricScore.score` returned by `calculate` is the **mean** of the
    per-turn scores (failed turns excluded). Per-turn scores live in
    `metadata["per_turn_scores"]` so consumers can pinpoint bad turns.

    Error-handling divergence from `_AMACEFullTraceMetric`: the full-trace
    base collapses every judge failure into a single top-level
    ``MetricScore`` with ``score=None`` and ``metadata["warning"]="api_error"``
    via ``_judge_failure_score``. The per-turn base instead records each
    turn's failure independently in
    ``metadata["per_turn_scores"][idx]["reasoning"]`` as a
    ``"judge error: ..."`` string and continues aggregating the surviving
    turns. The metric only returns a top-level ``score=None`` when *every*
    turn failed.
    """

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return True

    def get_dimension(self) -> Optional[str]:
        return "Multi-Turn"

    # --- subclass hooks ----------------------------------------------------

    def _build_pair_prompt(self, user_text: str, assistant_text: str) -> tuple[str, str]:
        """Return (system_prompt, user_message) for a single user→assistant pair.

        Security review H-01: subclasses build their rubric via
        build_judge_system_prompt (-> system_prompt) and wrap the untrusted
        assistant_text via wrap_untrusted (-> part of user_message). The
        user_text (the human's own turn, not the agent's) is trusted input
        and stays as plain text.
        """
        raise NotImplementedError

    def _parse_pair_response(self, response_json: dict) -> tuple[Optional[float], str]:
        """Translate judge JSON into (score in [0,1] or None, reasoning).

        Default implementation: read `Score`/`score` and `Reason`/`reasoning`
        from the judge response and run `_normalize_1_to_5` on the raw score.
        Subclasses may override to handle non-1–5 scoring schemes.

        Security review M-05: validates shape/types via validate_judge_response
        before normalizing. Raises ValueError on a malformed response — callers
        (calculate_async / _calculate_sync_fallback) catch this per-turn so one
        bad turn degrades to score=None instead of aborting the whole metric.
        """
        response_json = validate_judge_response(
            response_json,
            score_key=["Score", "score"],
            reasoning_key=["Reason", "reasoning"],
            score_range=(1.0, 5.0),
        )
        raw = response_json.get("Score", response_json.get("score"))
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        return _normalize_1_to_5(raw), reason

    # --- core calculation --------------------------------------------------

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        try:
            return asyncio.run(self.calculate_async(evaluation_input))
        except RuntimeError as e:
            # Already inside an event loop (notebook, async test runner).
            # Fall back to a sequential synchronous loop.
            if "asyncio.run() cannot be called" not in str(e):
                raise
            return self._calculate_sync_fallback(evaluation_input)

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        trace = evaluation_input.trace
        pairs = _extract_user_assistant_pairs(trace.messages)
        if not pairs:
            return self._not_applicable(0)

        prompts = [self._build_pair_prompt(u.content, a.content) for u, a in pairs]
        results = await asyncio.gather(
            *[
                asyncio.to_thread(_invoke_bedrock_judge, sys_p, user_p, self.get_name())
                for sys_p, user_p in prompts
            ],
            return_exceptions=True,
        )

        per_turn = []
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                per_turn.append({
                    "turn_index": idx,
                    "score": None,
                    "reasoning": f"judge error: {type(res).__name__}: {res}",
                })
                continue
            try:
                # Security review M-05: _parse_pair_response's overrides call
                # validate_judge_response, which can raise ValueError on a
                # malformed judge response. Catch it here (like the
                # gather(return_exceptions=True) above already does for
                # invoke failures) so one bad turn degrades to score=None
                # instead of aborting the whole metric.
                score, reasoning = self._parse_pair_response(res)
            except ValueError as e:
                per_turn.append({
                    "turn_index": idx,
                    "score": None,
                    "reasoning": f"invalid judge response: {e}",
                })
                continue
            if score is not None and not (0.0 <= score <= 1.0):
                per_turn.append({
                    "turn_index": idx,
                    "score": None,
                    "reasoning": f"invalid score from judge: {score}",
                })
                continue
            per_turn.append({
                "turn_index": idx,
                "score": score,
                "reasoning": reasoning,
            })

        return self._build_metric_score(pairs, per_turn)

    # --- helpers -----------------------------------------------------------

    def _calculate_sync_fallback(self, evaluation_input: EvaluationInput) -> MetricScore:
        trace = evaluation_input.trace
        pairs = _extract_user_assistant_pairs(trace.messages)
        if not pairs:
            return self._not_applicable(0)

        per_turn = []
        for idx, (u, a) in enumerate(pairs):
            system_prompt, user_message = self._build_pair_prompt(u.content, a.content)
            try:
                response_json = _invoke_bedrock_judge(system_prompt, user_message, self.get_name())
                score, reasoning = self._parse_pair_response(response_json)
                if score is not None and not (0.0 <= score <= 1.0):
                    per_turn.append({
                        "turn_index": idx,
                        "score": None,
                        "reasoning": f"invalid score from judge: {score}",
                    })
                    continue
                per_turn.append({"turn_index": idx, "score": score, "reasoning": reasoning})
            except Exception as e:  # noqa: BLE001 - mirrors gather(return_exceptions=True)
                per_turn.append({
                    "turn_index": idx,
                    "score": None,
                    "reasoning": f"judge error: {type(e).__name__}: {e}",
                })

        return self._build_metric_score(pairs, per_turn)

    def _not_applicable(self, pair_count: int) -> MetricScore:
        return MetricScore(
            metric_name=self.get_name(),
            score=None,
            reasoning="Cannot evaluate: no user→assistant pairs",
            metadata={"warning": "not_applicable", "pair_count": pair_count},
        )

    def _build_metric_score(self, pairs: list[tuple[Message, Message]], per_turn: list[dict]) -> MetricScore:
        usable = [t["score"] for t in per_turn if t["score"] is not None]
        scored = len(usable)
        failed = len(per_turn) - scored

        metadata = {
            "pair_count": len(pairs),
            "scored_count": scored,
            "failed_count": failed,
            "per_turn_scores": per_turn,
            "aggregation": "mean",
        }

        if not usable:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: all judge calls failed",
                metadata=metadata,
            )

        mean_score = sum(usable) / len(usable)

        scored_turns = [t for t in per_turn if t["score"] is not None]
        lowest = min(scored_turns, key=lambda t: t["score"])
        headline = (
            f"Mean of {scored} per-turn score(s). "
            f"Lowest at turn {lowest['turn_index']} ({lowest['score']:.2f}): "
            f"{lowest['reasoning']}"
        )

        return MetricScore(
            metric_name=self.get_name(),
            score=mean_score,
            reasoning=headline,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Per-turn concrete metrics (opt-in alternatives to the full-trace versions)
# ---------------------------------------------------------------------------


class PerTurnSentimentMetric(_AMACEPerTurnMetric):
    """User sentiment per user→assistant exchange (1–5 normalized to 0–1).

    See `SentimentMetric` for the full-trace counterpart. Per-turn scores
    are independent: each judge call sees exactly one exchange, with no
    cross-turn context.
    """

    def get_name(self) -> str:
        return "per_turn_sentiment"

    def get_description(self) -> Optional[str]:
        return (
            "User sentiment scored independently per user→assistant turn "
            "(1–5 normalized to 0–1, mean across turns). "
            "Per-turn scores in metadata['per_turn_scores']."
        )

    def _build_pair_prompt(self, user_text: str, assistant_text: str) -> tuple[str, str]:
        # Security review H-01: rubric in system field; assistant_text is
        # the untrusted agent output and is wrapped. user_text is the
        # human's own turn (trusted) and stays as plain text.
        system_rubric = """
Act as a principal sentiment analyst with PhDs in Linguistics and Psychology.
Determine the user's sentiment in the single exchange below, given what the
assistant just said.

Your output should be in JSON format only and must be deserializable. Use the following structure:
{
    "Score": number, // estimated sentiment score from 1-5 reflecting very unhappy to very happy
    "Reason": string // reason for the above sentiment rating
}

Additional Instructions:
1. Score only the user's affect in this single exchange, not the broader conversation.
2. Provide detailed reasoning.
3. If uncertain about specific details, indicate lower confidence in your analysis.
4. Do not restate the question. Only return JSON output.
"""
        system_prompt = build_judge_system_prompt(system_rubric)
        user_message = (
            f"User: {user_text}\n"
            + wrap_untrusted(assistant_text, "candidate_assistant_reply")
        )
        return system_prompt, user_message


class PerTurnAgentToneMetric(_AMACEPerTurnMetric):
    """Agent tone per user→assistant exchange (1–5 normalized to 0–1).

    See `AgentToneMetric` for the full-trace counterpart. Per-turn scores
    are independent: each judge call sees exactly one exchange, with no
    cross-turn context.
    """

    def get_name(self) -> str:
        return "per_turn_agent_tone"

    def get_description(self) -> Optional[str]:
        return (
            "Agent tone professionalism scored independently per user→assistant "
            "turn (1–5 normalized to 0–1, mean across turns). "
            "Per-turn scores in metadata['per_turn_scores']."
        )

    def _build_pair_prompt(self, user_text: str, assistant_text: str) -> tuple[str, str]:
        # Security review H-01: rubric in system field; assistant_text is
        # the untrusted agent output and is wrapped. user_text is the
        # human's own turn (trusted) and stays as plain text.
        system_rubric = """
Act as a Principal Linguist specializing in evaluating chatbot tone and professionalism.
Rate the agent's tone in the single reply below.

Expected Tone: Professional and Friendly

Examples of what the tone should reflect:
- A proactive friend
- A well-informed advisor
- An accomplished peer

Examples of what the tone shouldn't reflect:
- a lecturing teacher
- A smug elitist
- An overprotective parent

Your output should be in JSON format only and must be deserializable:
{
    "Score": number, // 1 to 5 (1 = super unprofessional, 5 = super professional)
    "Reason": string // brief reason for the rating
}

Additional Instructions:
1. Score only the assistant's tone in this single reply.
2. Provide detailed reasoning.
3. If uncertain about specific details, indicate lower confidence.
4. Do not restate the question. Only return JSON output.
"""
        system_prompt = build_judge_system_prompt(system_rubric)
        user_message = (
            f"User: {user_text}\n"
            + wrap_untrusted(assistant_text, "candidate_assistant_reply")
        )
        return system_prompt, user_message


class PerTurnNaturalnessMetric(_AMACEPerTurnMetric):
    """Agent naturalness per user→assistant exchange (1–5 normalized to 0–1).

    See `NaturalnessMetric` for the full-trace counterpart. Per-turn scores
    are independent: each judge call sees exactly one exchange, with no
    cross-turn context.
    """

    def get_name(self) -> str:
        return "per_turn_naturalness"

    def get_description(self) -> Optional[str]:
        return (
            "Naturalness of the agent's response scored independently per "
            "user→assistant turn (1–5 normalized to 0–1, mean across turns). "
            "Per-turn scores in metadata['per_turn_scores']."
        )

    def _build_pair_prompt(self, user_text: str, assistant_text: str) -> tuple[str, str]:
        # Security review H-01: rubric in system field; assistant_text is
        # the untrusted agent output and is wrapped. user_text is the
        # human's own turn (trusted) and stays as plain text.
        system_rubric = """
Act as a Principal Linguist evaluating the naturalness of a chatbot's reply.
Determine whether the assistant's single reply below sounds formulaically
robotic or naturally human-like.

Your output should be in JSON format only and must be deserializable:
{
    "Score": number, // 1 to 5 (1 = robotic / unnatural, 5 = natural / human-like)
    "Reason": string // reasoning for the rating
}

Additional Instructions:
1. Score only the assistant's single reply, not the conversational arc.
2. Provide detailed reasoning.
3. If uncertain about specific details, indicate lower confidence.
4. Do not restate the question. Only return JSON output.
"""
        system_prompt = build_judge_system_prompt(system_rubric)
        user_message = (
            f"User: {user_text}\n"
            + wrap_untrusted(assistant_text, "candidate_assistant_reply")
        )
        return system_prompt, user_message


# --- 1. Role Adherence -----------------------------------------------------


class RoleAdherenceMetric(_AMACEFullTraceMetric):
    """
    Metric for evaluating chatbot role adherence across the conversation.

    Computes the ratio of turns whose assistant response stayed in-role versus
    drifting away from the persona / system message. Requires the chatbot's
    role/persona via ``GroundTruth.expected_arguments['chatbot_role']``.
    """

    def get_name(self) -> str:
        return "role_adherence"

    def requires_ground_truth(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Evaluates whether the chatbot adheres to its given role across the conversation"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        gt = evaluation_input.ground_truth
        role = (
            gt.expected_arguments.get("chatbot_role")
            if gt is not None and gt.expected_arguments
            else None
        )
        if not role:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: expected_arguments['chatbot_role'] is required",
                metadata={"warning": "not_applicable"},
            )
        return super().calculate(evaluation_input)

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        gt = evaluation_input.ground_truth
        role = (
            gt.expected_arguments.get("chatbot_role")
            if gt is not None and gt.expected_arguments
            else None
        )
        if not role:
            return None

        # Security review H-01: rubric in system field; role is trusted
        # deployer-provided data and stays as plain text; conversation is
        # the untrusted transcript and is wrapped.
        system_rubric = """
            You are evaluating whether a chatbot adhered to its assigned role throughout a multi-turn conversation.
            You will be given the chatbot's role description and the full conversation history.

            For each ASSISTANT turn, judge whether that turn stays in-role.
            Then compute (turns_in_role / total_assistant_turns) and return that ratio as the score in [0, 1].

            EVALUATION GUIDANCE:
            - Stays in-role: tone, scope, and constraints match the role description.
            - Out of role: the assistant breaks persona, exceeds its scope, contradicts its constraints, or speaks as a different persona.

            After providing your explanation in the "reasoning" field, return the per-turn ratio in the "score" field on a 0 to 1 scale.
            Strictly follow the below json format:{resp_fmt}."""
        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=_RESP_FMT_SCORE_REASONING))
        user_message = (
            f"[Chatbot Role]\n{role}\n\n"
            + wrap_untrusted(conversation, "candidate_conversation")
        )
        return system_prompt, user_message

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        # Security review M-05: validate shape/types before use.
        response_json = validate_judge_response(response_json)
        score = response_json["score"]
        reasoning = response_json.get("reasoning", "")
        return float(score), reasoning, {}


# --- 2. Holistic LLM Judge -------------------------------------------------


class HolisticLLMJudgeMetric(_AMACEFullTraceMetric):
    """
    Composite multi-factor LLM judge ported from AMACE's LLMJudgeScore.

    Evaluates the conversation across completeness, relevancy, knowledge
    retention, sentiment, tone, satisfaction, and containment, weighting
    containment more heavily. Returns a single 0-1 quality score.
    """

    def get_name(self) -> str:
        return "holistic_llm_judge"

    def get_description(self) -> Optional[str]:
        return "Composite multi-factor LLM judge over the full conversation"

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        # Security review H-01: rubric in system field; conversation is the
        # untrusted transcript and is wrapped.
        system_rubric = """
            You are an expert conversation evaluator. Evaluate the conversation from all angles.
            Consider its completeness, relevancy, knowledge retention, sentiment, tone,
            user satisfaction, and containment. Give an overall quality score for the
            agent's behavior across the entire conversation, weighting CONTAINMENT more heavily.
            If the conversation had to be escalated to a human agent, do not give it a high score.

            After providing your explanation in the "reasoning" field, return a 0 to 1 score in the "score" field.
            Strictly follow the below json format:{resp_fmt}."""
        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=_RESP_FMT_SCORE_REASONING))
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        # Security review M-05: validate shape/types before use.
        response_json = validate_judge_response(response_json)
        score = float(response_json["score"])
        reasoning = response_json.get("reasoning", "")
        return score, reasoning, {}


# --- 3. User Satisfaction --------------------------------------------------


class UserSatisfactionMetric(_AMACEFullTraceMetric):
    """
    Metric estimating end-of-conversation user satisfaction on a 1-5 scale,
    normalized to 0-1 with the raw value preserved in metadata.
    """

    def get_name(self) -> str:
        return "user_satisfaction"

    def get_description(self) -> Optional[str]:
        return "Evaluates end-of-conversation user satisfaction (1-5, normalized to 0-1)"

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        # Security review H-01: rubric in system field; conversation is the
        # untrusted transcript and is wrapped.
        system_rubric = """
            You are evaluating overall user satisfaction at the end of a conversation between a user and a chatbot.
            Determine whether the user's query was resolved properly and how satisfied the user appears to be by the end.

            Score on a 1-5 scale:
            1 = very dissatisfied (query not resolved, user frustrated)
            2 = dissatisfied
            3 = neutral
            4 = satisfied
            5 = very satisfied (query resolved cleanly, user happy)

            Output JSON only, with this structure:
            {
                "Score": number,   // integer between 1 and 5
                "Reason": string   // explanation
            }"""
        system_prompt = build_judge_system_prompt(system_rubric)
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        # Security review M-05: validate shape/types before use.
        response_json = validate_judge_response(
            response_json,
            score_key=["Score", "score"],
            reasoning_key=["Reason", "reasoning"],
            score_range=(1.0, 5.0),
        )
        raw = response_json.get("Score", response_json.get("score"))
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        score = _normalize_1_to_5(raw)
        return score, reason, {"raw_score": raw}


# --- 4. Sentiment ----------------------------------------------------------


class SentimentMetric(_AMACEFullTraceMetric):
    """
    Estimates the user's sentiment across the conversation on a 1-5 scale,
    weighting later interactions more heavily. Normalizes to 0-1.
    """

    def get_name(self) -> str:
        return "sentiment"

    def get_description(self) -> Optional[str]:
        return "User sentiment across the conversation (1-5, normalized to 0-1)"

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        # Security review H-01: rubric in system field; conversation is the
        # untrusted transcript and is wrapped.
        system_rubric = """
Act as a principal sentiment analyst with PhDs in Linguistics and Psychology. You will analyze user interaction transcripts involving an AI assistant. Review the provided transcript and determine the user's sentiment throughout the interaction.

Your output should be in JSON format only and must be deserializable. Use the following structure:
{
    "Score": number, // estimated sentiment score from 1-5 reflecting very unhappy to very happy
    "Reason": string // reason for the above sentiment rating
}

Additional Instructions:
1. Consider the entire conversation flow, weighing later interactions more heavily.
2. Account for repetitions, and misunderstandings in your scoring.
3. Provide detailed reasoning for your analysis and scoring.
4. If uncertain about specific details, indicate lower confidence in your analysis.
5. Do not restate the question. Only return JSON output.
"""
        system_prompt = build_judge_system_prompt(system_rubric)
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        # Security review M-05: validate shape/types before use.
        response_json = validate_judge_response(
            response_json,
            score_key=["Score", "score"],
            reasoning_key=["Reason", "reasoning"],
            score_range=(1.0, 5.0),
        )
        raw = response_json.get("Score", response_json.get("score"))
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        score = _normalize_1_to_5(raw)
        return score, reason, {"raw_score": raw}


# --- 5. Agent Tone ---------------------------------------------------------


class AgentToneMetric(_AMACEFullTraceMetric):
    """
    Evaluates the agent's tone on a 1-5 professionalism scale (super
    unprofessional → super professional). Normalized to 0-1.
    """

    def get_name(self) -> str:
        return "agent_tone"

    def get_description(self) -> Optional[str]:
        return "Agent tone professionalism across the conversation (1-5, normalized to 0-1)"

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        # Security review H-01: rubric in system field; conversation is the
        # untrusted transcript and is wrapped.
        system_rubric = """
Act as a Principal Linguist specializing in evaluating chatbot tone and professionalism.
Review the provided transcript of a chatbot interaction and rate the agent's tone.

Expected Tone: Professional and Friendly

Examples of what the tone should reflect:
- A proactive friend
- A well-informed advisor
- An accomplished peer

Examples of what the tone shouldn't reflect:
- a lecturing teacher
- A smug elitist
- An overprotective parent

Your output should be in JSON format only and must be deserializable:
{
    "Score": number, // 1 to 5 (1 = super unprofessional, 5 = super professional)
    "Reason": string // brief reason for the rating
}

Additional Instructions:
1. Consider the entire conversation flow, weighing later interactions more heavily.
2. Account for repetitions and misunderstandings in your scoring.
3. Provide detailed reasoning.
4. If uncertain about specific details, indicate lower confidence.
5. Do not restate the question. Only return JSON output.
"""
        system_prompt = build_judge_system_prompt(system_rubric)
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        # Security review M-05: validate shape/types before use.
        response_json = validate_judge_response(
            response_json,
            score_key=["Score", "score"],
            reasoning_key=["Reason", "reasoning"],
            score_range=(1.0, 5.0),
        )
        raw = response_json.get("Score", response_json.get("score"))
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        score = _normalize_1_to_5(raw)
        return score, reason, {"raw_score": raw}


# --- 6. Naturalness --------------------------------------------------------


class NaturalnessMetric(_AMACEFullTraceMetric):
    """
    Evaluates how natural and human-like the agent's responses sound across
    the conversation, on a 1-5 scale (1 = robotic / unnatural,
    5 = natural / human-like). Normalized to 0-1.
    """

    def get_name(self) -> str:
        return "naturalness"

    def get_description(self) -> Optional[str]:
        return "Naturalness of agent responses (1-5, normalized to 0-1)"

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        # Security review H-01: rubric in system field; conversation is the
        # untrusted transcript and is wrapped.
        system_rubric = """
Act as a Principal Linguist evaluating the naturalness of a chatbot's responses.
Determine whether the bot's responses sound formulaically robotic or naturally human-like across the conversation.

Your output should be in JSON format only and must be deserializable:
{
    "Score": number, // 1 to 5 (1 = robotic / unnatural, 5 = natural / human-like)
    "Reason": string // reasoning for the rating
}

Additional Instructions:
1. Consider the entire conversation flow, weighing later interactions more heavily.
2. Account for repetitions and misunderstandings.
3. Provide detailed reasoning.
4. If uncertain about specific details, indicate lower confidence.
5. Do not restate the question. Only return JSON output.
"""
        system_prompt = build_judge_system_prompt(system_rubric)
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        # Security review M-05: validate shape/types before use.
        response_json = validate_judge_response(
            response_json,
            score_key=["Score", "score"],
            reasoning_key=["Reason", "reasoning"],
            score_range=(1.0, 5.0),
        )
        raw = response_json.get("Score", response_json.get("score"))
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        score = _normalize_1_to_5(raw)
        return score, reason, {"raw_score": raw}


# --- 7. Instruction Compliance ---------------------------------------------


class InstructionComplianceMetric(_AMACEFullTraceMetric):
    """
    Evaluates whether the agent complied with the original system prompt /
    instructions across the conversation. Returns a 0-1 score.

    Reads the instruction text from
    ``GroundTruth.expected_arguments['system_instructions']`` (preferred) or
    falls back to ``GroundTruth.expected_output``.
    """

    def get_name(self) -> str:
        return "instruction_compliance"

    def requires_ground_truth(self) -> bool:
        return True

    def get_description(self) -> Optional[str]:
        return "Whether the agent complies with the original system prompt instructions"

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        gt = evaluation_input.ground_truth
        instructions = None
        if gt is not None:
            if gt.expected_arguments:
                instructions = gt.expected_arguments.get("system_instructions")
            if not instructions:
                instructions = gt.expected_output
        if not instructions:
            return None

        # Security review H-01: rubric in system field; instructions is
        # trusted deployer-provided data (the original system prompt) and
        # stays as plain text; conversation is the untrusted transcript
        # (the agent's own responses being checked for compliance) and is
        # wrapped.
        system_rubric = """
Review the following chat conversation and evaluate whether the agent's responses comply
with the original prompt instructions on how to handle user queries/issues.

Original Prompt:
<original_prompt>{instruction}</original_prompt>

Respond with JSON output containing keys "Score" and "Reason". "Score" is a floating-point
number ranging between 0 and 1.
- "Score" of 0 means agent responses are not compliant with the original instructions.
- "Score" of 1 means agent responses fully comply with the original instructions.
Provide reasoning under "Reason" as a string.

Additional instructions:
1. Read the full chat conversation transcript.
2. Break down each instruction from the original prompt.
3. Compare whether the agent's responses adhere to those instructions overall.
4. Indicate any deviations or partial compliance.
5. Return JSON only output and do not restate the question.
"""
        system_prompt = build_judge_system_prompt(system_rubric.format(instruction=instructions))
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        # Security review M-05: validate shape (required key present, no
        # unexpected fields) before use. Range is intentionally NOT enforced
        # here — this metric accepts either a 0-1 score or a 1-5 raw rating
        # (disambiguated below), a wider contract than validate_judge_response's
        # single score_range param models, so allow the full union here and
        # keep relying on this method's own float-parse + dual-range handling,
        # which already degrades gracefully (returns None, not an exception)
        # on a non-numeric or out-of-both-ranges value.
        response_json = validate_judge_response(
            response_json,
            score_key=["Score", "score"],
            reasoning_key=["Reason", "reasoning"],
            score_range=(0.0, 5.0),
            allow_null_score=True,
        )
        raw = response_json.get("Score", response_json.get("score"))
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, reason, {"raw_score": raw}
        # Accept either a 0-1 score or a 1-5 raw rating, normalize as needed.
        if 0.0 <= v <= 1.0:
            return v, reason, {"raw_score": raw}
        normalized = _normalize_1_to_5(v)
        if normalized is None:
            return None, reason, {"raw_score": raw}
        return normalized, reason, {"raw_score": raw}


# --- 8. Optimum Turns ------------------------------------------------------


class OptimumTurnsMetric(_AMACEFullTraceMetric):
    """
    LLM-estimated optimum-turn-count efficiency.

    Asks the judge to estimate the optimum number of turns this conversation
    should have taken, then computes a 0-1 efficiency score:

        score = min(1.0, optimum / actual)

    The raw integer optimum count is preserved in metadata['optimum_turns'].
    """

    def get_name(self) -> str:
        return "optimum_turns"

    def get_description(self) -> Optional[str]:
        return "LLM-estimated optimum-turn efficiency (optimum / actual, capped at 1.0)"

    def _build_prompt(
        self, conversation: str, evaluation_input: EvaluationInput
    ) -> Optional[tuple[str, str]]:
        # Security review H-01: rubric in system field; conversation is the
        # untrusted transcript and is wrapped.
        system_rubric = """
You are an expert in customer service optimization and conversation efficiency analysis.
Estimate the optimum number of turns required for the agent to resolve this customer's query.
A turn is defined as a pair of interactions: a customer message and the corresponding agent response.

Output JSON only with this structure:
{
    "Optimum_Turns": number, // integer >= 1
    "Reason": string         // why this is the optimum
}

Do not restate the question. Only return JSON output.
"""
        system_prompt = build_judge_system_prompt(system_rubric)
        user_message = wrap_untrusted(conversation, "candidate_conversation")
        return system_prompt, user_message

    def _parse_response(
        self, response_json: dict, evaluation_input: EvaluationInput
    ) -> tuple[Optional[float], str, dict]:
        # Security review M-05: this metric's judge contract doesn't fit
        # validate_judge_response's {score, reasoning} shape — the field is
        # "Optimum_Turns"/"optimum_turns" (an integer turn count, not a
        # bounded score; the actual 0-1 score is computed afterward from
        # optimum/actual). Applying the shared validator here would either
        # misrepresent this field as "score" or need a range that doesn't
        # correspond to anything meaningful for a turn count. The existing
        # try/except + "< 1" check below already provides equivalent
        # type/shape safety for this metric's actual contract.
        optimum_raw = response_json.get(
            "Optimum_Turns", response_json.get("optimum_turns", response_json.get("Score"))
        )
        reason = response_json.get("Reason", response_json.get("reasoning", ""))
        try:
            optimum = int(float(optimum_raw))
        except (TypeError, ValueError):
            return None, reason, {"optimum_turns": optimum_raw}
        if optimum < 1:
            return None, reason, {"optimum_turns": optimum}

        actual_turns = max(
            1, len(evaluation_input.trace.messages) // 2
        )
        score = min(1.0, optimum / actual_turns)
        return score, reason, {
            "optimum_turns": optimum,
            "actual_turns": actual_turns,
        }
