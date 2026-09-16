# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning quality metrics for UAEF.

This module implements metrics for evaluating agent reasoning quality including
chain-of-thought coherence, logical consistency, reasoning step correctness,
and fallacy detection.
"""

import json
from typing import List, Optional, Tuple

import boto3

from uaef.config import get_config
from uaef.metrics.base import BaseMetric
from uaef.metrics.utils import (
    build_claude_judge_body,
    build_judge_system_prompt,
    extract_json_from_llm_response,
    validate_judge_response,
    wrap_untrusted,
)
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore
from uaef.models.multi_agent_trace import MultiAgentTrace


def _extract_reasoning_text(evaluation_input: EvaluationInput) -> Tuple[str, int]:
    """
    Extract reasoning text from a trace, labeling by agent for multi-agent traces.

    For single-agent traces, returns all assistant messages joined.
    For multi-agent traces, labels each message with its agent name so the
    LLM judge can understand inter-agent reasoning flow.

    Returns:
        Tuple of (reasoning_text, step_count). reasoning_text is empty if
        no assistant messages found.
    """
    trace = evaluation_input.trace

    if isinstance(trace, MultiAgentTrace):
        steps: List[str] = []
        for agent_id, agent_trace in trace.agent_traces.items():
            for msg in agent_trace.messages:
                if msg.role == "assistant" and msg.content:
                    steps.append(f"[{agent_id}] {msg.content}")
        return "\n\n".join(steps), len(steps)

    reasoning_steps = []
    for msg in trace.messages:
        if msg.role == "assistant" and msg.content:
            reasoning_steps.append(msg.content)
    return "\n\n".join(reasoning_steps), len(reasoning_steps)


class ChainOfThoughtCoherenceMetric(BaseMetric):
    """
    Metric for evaluating chain-of-thought reasoning coherence.
    
    Evaluates whether the agent's reasoning flows logically from one step
    to the next. Uses LLM judge to assess reasoning flow.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "chain_of_thought_coherence"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric requires LLM judge."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates coherence and logical flow of chain-of-thought reasoning"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Reasoning"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate chain-of-thought coherence score."""
        full_reasoning, step_count = _extract_reasoning_text(evaluation_input)

        if not full_reasoning:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent reasoning found in trace",
                metadata={"warning": "missing_data"}
            )

        # Use LLM judge to evaluate coherence
        try:
            config = get_config()
            score, reasoning = self._llm_judge_coherence(
                reasoning_text=full_reasoning,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={"reasoning_step_count": step_count}
            )
        except ConnectionError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Bedrock client initialization failed: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except RuntimeError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Bedrock API call failed: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except ValueError as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: Invalid score from LLM judge: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
        except Exception as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: LLM judge error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate chain-of-thought coherence asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_coherence(
        self,
        reasoning_text: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to evaluate chain-of-thought coherence.
        
        Args:
            reasoning_text: The agent's reasoning text
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning)
        """
        # Initialize AWS Bedrock client
        try:
            config = get_config()
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=config.aws.region,
                aws_access_key_id=config.aws.access_key_id,
                aws_secret_access_key=config.aws.secret_access_key,
                aws_session_token=config.aws.session_token
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Bedrock client: {str(e)}")
        
        # Prepare prompt
        resp_fmt = """{
                       "score":float,
                       "reasoning": str
                   }
               """
        
        # Security review H-01: rubric in system field; reasoning_text is
        # entirely the untrusted agent's own output and is wrapped.
        system_rubric = """
            You are an AI evaluator that assesses chain-of-thought reasoning coherence.
            Please evaluate whether the reasoning flows logically from one step to the next.
            You will be given the agent's reasoning text. If multiple agents are involved
            (indicated by [agent_name] labels), evaluate the coherence of the overall
            reasoning flow across agents — whether information passes correctly between
            them and whether the combined reasoning forms a coherent chain.

            In your evaluation, check for:
            - Logical progression from one step to the next
            - Clear connections between reasoning steps
            - Consistent line of thought throughout
            - Absence of logical jumps or gaps
            - Well-structured reasoning flow
            - Each step building upon previous steps
            - For multi-agent: whether each agent's output provides coherent input for the next

            After providing your explanation in the "reasoning" tab, you must score the reasoning on a scale of 0 to 1 in the "score" tab,
            where 1 means highly coherent reasoning (clear logical flow, well-connected steps)
            and 0 means incoherent reasoning (disjointed, illogical jumps, unclear connections).
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))
        user_message = wrap_untrusted(reasoning_text, "candidate_reasoning")

        # Claude 3+ rejects requests with both temperature and top_p;
        # temperature alone is sufficient for deterministic judge scoring.
        body = build_claude_judge_body(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Make API call
        try:
            response_obj = bedrock_client.invoke_model(
                modelId=judge_id,
                body=body,
                accept='application/json',
                contentType='application/json'
            )
        except Exception as e:
            raise RuntimeError(f"Bedrock API call failed: {str(e)}")
        
        # Parse response
        response_body = json.loads(response_obj.get('body').read())
        raw_text = response_body.get('content')[0]['text']
        response_json = extract_json_from_llm_response(raw_text)
        # Security review M-05: validate shape/types before use, not just
        # the score's numeric range.
        response_json = validate_judge_response(response_json)
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning


class LogicalConsistencyMetric(BaseMetric):
    """
    Metric for detecting logical contradictions in agent reasoning.
    
    Evaluates whether the agent's reasoning contains contradictions or
    inconsistencies. Uses LLM judge to detect logical issues.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "logical_consistency"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric requires LLM judge."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Detects logical contradictions and inconsistencies in reasoning"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Reasoning"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate logical consistency score."""
        full_reasoning, step_count = _extract_reasoning_text(evaluation_input)

        if not full_reasoning:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent reasoning found in trace",
                metadata={"warning": "missing_data"}
            )

        # Use LLM judge to evaluate consistency
        try:
            config = get_config()
            score, reasoning = self._llm_judge_consistency(
                reasoning_text=full_reasoning,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={"reasoning_step_count": step_count}
            )
        except Exception as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: LLM judge error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate logical consistency asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_consistency(
        self,
        reasoning_text: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to evaluate logical consistency.
        
        Args:
            reasoning_text: The agent's reasoning text
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning)
        """
        # Initialize AWS Bedrock client
        try:
            config = get_config()
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=config.aws.region,
                aws_access_key_id=config.aws.access_key_id,
                aws_secret_access_key=config.aws.secret_access_key,
                aws_session_token=config.aws.session_token
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Bedrock client: {str(e)}")
        
        # Prepare prompt
        resp_fmt = """{
                       "score":float,
                       "reasoning": str
                   }
               """
        
        # Security review H-01: rubric in system field; reasoning_text is
        # entirely the untrusted agent's own output and is wrapped.
        system_rubric = """
            You are an AI evaluator that assesses logical consistency in reasoning.
            Please evaluate whether the reasoning contains contradictions or inconsistencies.
            You will be given the agent's reasoning text. If multiple agents are involved
            (indicated by [agent_name] labels), also check for contradictions between
            agents — e.g., one agent claiming something that another agent contradicts.

            In your evaluation, check for:
            - Contradictory statements or claims
            - Inconsistent assumptions or premises
            - Self-contradicting conclusions
            - Conflicting facts or data
            - Logical inconsistencies between steps
            - Statements that contradict earlier statements
            - For multi-agent: contradictions between different agents' outputs

            After providing your explanation in the "reasoning" tab, you must score the reasoning on a scale of 0 to 1 in the "score" tab,
            where 1 means fully consistent (no contradictions, all statements align)
            and 0 means highly inconsistent (multiple contradictions, conflicting statements).
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))
        user_message = wrap_untrusted(reasoning_text, "candidate_reasoning")

        # Claude 3+ rejects requests with both temperature and top_p;
        # temperature alone is sufficient for deterministic judge scoring.
        body = build_claude_judge_body(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Make API call
        try:
            response_obj = bedrock_client.invoke_model(
                modelId=judge_id,
                body=body,
                accept='application/json',
                contentType='application/json'
            )
        except Exception as e:
            raise RuntimeError(f"Bedrock API call failed: {str(e)}")
        
        # Parse response
        response_body = json.loads(response_obj.get('body').read())
        raw_text = response_body.get('content')[0]['text']
        response_json = extract_json_from_llm_response(raw_text)
        # Security review M-05: validate shape/types before use, not just
        # the score's numeric range.
        response_json = validate_judge_response(response_json)
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning


class ReasoningStepCorrectnessMetric(BaseMetric):
    """
    Metric for validating correctness of individual reasoning steps.
    
    Evaluates whether each reasoning step is logically sound and correct.
    Uses LLM judge to validate reasoning steps.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "reasoning_step_correctness"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric requires LLM judge."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Validates correctness of individual reasoning steps"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Reasoning"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate reasoning step correctness score."""
        full_reasoning, step_count = _extract_reasoning_text(evaluation_input)

        if not full_reasoning:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent reasoning found in trace",
                metadata={"warning": "missing_data"}
            )

        # Get context if available
        context = "\n\n".join(evaluation_input.context) if evaluation_input.context else ""

        # Use LLM judge to evaluate step correctness
        try:
            config = get_config()
            score, reasoning = self._llm_judge_step_correctness(
                reasoning_text=full_reasoning,
                context=context,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={"reasoning_step_count": step_count}
            )
        except Exception as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: LLM judge error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate reasoning step correctness asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_step_correctness(
        self,
        reasoning_text: str,
        context: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to evaluate reasoning step correctness.
        
        Args:
            reasoning_text: The agent's reasoning text
            context: Context information (if available)
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning)
        """
        # Initialize AWS Bedrock client
        try:
            config = get_config()
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=config.aws.region,
                aws_access_key_id=config.aws.access_key_id,
                aws_secret_access_key=config.aws.secret_access_key,
                aws_session_token=config.aws.session_token
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Bedrock client: {str(e)}")
        
        # Prepare prompt
        resp_fmt = """{
                       "score":float,
                       "reasoning": str
                   }
               """
        
        # Security review H-01: rubric in system field; context (if any) is
        # trusted supplementary info and stays in the user turn as plain
        # text; reasoning_text is the untrusted agent output and is wrapped.
        system_rubric = """
            You are an AI evaluator that assesses the correctness of reasoning steps.
            Please evaluate whether each reasoning step is logically sound and correct.
            You will be given the agent's reasoning text{context_note}. If multiple agents
            are involved (indicated by [agent_name] labels), evaluate whether each agent's
            reasoning steps are correct and whether information passed between agents is
            used accurately.

            In your evaluation, check for:
            - Logical soundness of each step
            - Correct application of logic and inference
            - Valid conclusions from premises
            - Accurate use of information
            - Proper reasoning methodology
            - Absence of logical errors or fallacies
            - For multi-agent: correct use of other agents' outputs as inputs
            
            After providing your explanation in the "reasoning" tab, you must score the reasoning on a scale of 0 to 1 in the "score" tab,
            where 1 means all steps are correct (logically sound, valid inferences)
            and 0 means steps contain errors (logical mistakes, invalid inferences).
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(
            system_rubric.format(
                resp_fmt=resp_fmt,
                context_note=" and context information" if context else "",
            )
        )

        context_section = f"[Context]\n{context}\n\n" if context else ""
        user_message = context_section + wrap_untrusted(reasoning_text, "candidate_reasoning")

        # Claude 3+ rejects requests with both temperature and top_p;
        # temperature alone is sufficient for deterministic judge scoring.
        body = build_claude_judge_body(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Make API call
        try:
            response_obj = bedrock_client.invoke_model(
                modelId=judge_id,
                body=body,
                accept='application/json',
                contentType='application/json'
            )
        except Exception as e:
            raise RuntimeError(f"Bedrock API call failed: {str(e)}")
        
        # Parse response
        response_body = json.loads(response_obj.get('body').read())
        raw_text = response_body.get('content')[0]['text']
        response_json = extract_json_from_llm_response(raw_text)
        # Security review M-05: validate shape/types before use, not just
        # the score's numeric range.
        response_json = validate_judge_response(response_json)
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning


class FallacyDetectionMetric(BaseMetric):
    """
    Metric for detecting logical fallacies in agent reasoning.
    
    Identifies common logical fallacies such as ad hominem, straw man,
    false dichotomy, etc. Uses LLM judge to detect fallacies.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "fallacy_detection"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric requires LLM judge."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Detects logical fallacies in reasoning"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Reasoning"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate fallacy detection score."""
        full_reasoning, step_count = _extract_reasoning_text(evaluation_input)

        if not full_reasoning:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent reasoning found in trace",
                metadata={"warning": "missing_data"}
            )

        # Use LLM judge to detect fallacies
        try:
            config = get_config()
            score, reasoning = self._llm_judge_fallacies(
                reasoning_text=full_reasoning,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={"reasoning_step_count": step_count}
            )
        except Exception as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: LLM judge error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )
    
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate fallacy detection asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_fallacies(
        self,
        reasoning_text: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to detect logical fallacies.
        
        Args:
            reasoning_text: The agent's reasoning text
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning) where 1.0 = no fallacies, 0.0 = many fallacies
        """
        # Initialize AWS Bedrock client
        try:
            config = get_config()
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=config.aws.region,
                aws_access_key_id=config.aws.access_key_id,
                aws_secret_access_key=config.aws.secret_access_key,
                aws_session_token=config.aws.session_token
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Bedrock client: {str(e)}")
        
        # Prepare prompt
        resp_fmt = """{
                       "score":float,
                       "reasoning": str
                   }
               """
        
        # Security review H-01: rubric in system field; reasoning_text is
        # entirely the untrusted agent's own output and is wrapped.
        system_rubric = """
            You are an AI evaluator that detects logical fallacies in reasoning.
            Please evaluate whether the reasoning contains logical fallacies.
            You will be given the agent's reasoning text. If multiple agents are involved
            (indicated by [agent_name] labels), check for fallacies both within individual
            agents' reasoning and in the logical connections between agents.

            In your evaluation, check for common logical fallacies including:
            - Ad hominem (attacking the person instead of the argument)
            - Straw man (misrepresenting an argument to make it easier to attack)
            - False dichotomy (presenting only two options when more exist)
            - Slippery slope (assuming one thing will lead to extreme consequences)
            - Circular reasoning (using the conclusion as a premise)
            - Appeal to authority (claiming something is true because an authority says so)
            - Hasty generalization (drawing conclusions from insufficient evidence)
            - Post hoc ergo propter hoc (assuming causation from correlation)
            - Red herring (introducing irrelevant information to distract)
            - Begging the question (assuming what you're trying to prove)
            
            After providing your explanation in the "reasoning" tab, you must score the reasoning on a scale of 0 to 1 in the "score" tab,
            where 1 means no fallacies detected (sound logical reasoning)
            and 0 means multiple fallacies detected (flawed reasoning).
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))
        user_message = wrap_untrusted(reasoning_text, "candidate_reasoning")

        # Claude 3+ rejects requests with both temperature and top_p;
        # temperature alone is sufficient for deterministic judge scoring.
        body = build_claude_judge_body(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Make API call
        try:
            response_obj = bedrock_client.invoke_model(
                modelId=judge_id,
                body=body,
                accept='application/json',
                contentType='application/json'
            )
        except Exception as e:
            raise RuntimeError(f"Bedrock API call failed: {str(e)}")
        
        # Parse response
        response_body = json.loads(response_obj.get('body').read())
        raw_text = response_body.get('content')[0]['text']
        response_json = extract_json_from_llm_response(raw_text)
        # Security review M-05: validate shape/types before use, not just
        # the score's numeric range.
        response_json = validate_judge_response(response_json)
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning
