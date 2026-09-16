# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response quality metrics for UAEF.

This module implements metrics for evaluating agent response quality,
including relevance, completeness, hallucination detection, and accuracy.

Reuses LLM judge logic from agenticevaluationframework-main/frontend/evaluation.py.
"""

import json
from typing import Optional

import boto3

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

logger = get_logger(__name__)
from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore


class AnswerRelevanceMetric(BaseMetric):
    """
    Metric for evaluating answer relevance using LLM judge.
    
    Evaluates if the response is relevant to the question without requiring ground truth.
    Reuses llm_as_judge_score_answer_relevancy from evaluation.py (lines 462-551).
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "answer_relevance"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric requires LLM judge."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates if the response is relevant to the user's question using LLM judge"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Response Quality"

    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """
        Calculate answer relevance synchronously.
        
        Note: This is a blocking call. Use calculate_async for non-blocking evaluation.
        """
        # Extract question and response from trace
        trace = evaluation_input.trace
        
        # Get the user's question (first user message)
        question = None
        for msg in trace.messages:
            if msg.role == "user":
                question = msg.content
                break
        
        if question is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no user question found in trace",
                metadata={"warning": "missing_data"}
            )
        if isinstance(question, str) and not question.strip():
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: user question is empty",
                metadata={"warning": "invalid_data"}
            )
        
        # Get the agent's response (last assistant message)
        response = None
        for msg in reversed(trace.messages):
            if msg.role == "assistant":
                response = msg.content
                break
        
        if response is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent response found in trace",
                metadata={"warning": "missing_data"}
            )
        if isinstance(response, str) and not response.strip():
            return MetricScore(
                metric_name=self.get_name(),
                score=0.0,
                reasoning="Agent response is empty",
                metadata={"warning": "invalid_data"}
            )
        
        # Call LLM judge
        try:
            config = get_config()
            score, reasoning = self._llm_judge_relevance(
                question=question,
                response=response,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={"question": question[:100], "response": response[:100]}
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
                reasoning=f"Cannot evaluate: {str(e)}",
                metadata={"warning": "missing_data", "error_details": str(e)}
            )
        except Exception as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: LLM judge error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate answer relevance asynchronously."""
        # For now, just call synchronous version
        # TODO: Implement true async with aioboto3
        return self.calculate(evaluation_input)
    
    def _llm_judge_relevance(
        self,
        question: str,
        response: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to evaluate answer relevance.
        
        Reused from evaluation.py llm_as_judge_score_answer_relevancy (lines 462-551).
        
        Args:
            question: The user's question
            response: The agent's response
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning)
            
        Raises:
            ValueError: If inputs are invalid
            ConnectionError: If Bedrock client initialization fails
            RuntimeError: If Bedrock API call fails
        """
        # Input validation
        if not all(isinstance(x, str) for x in [question, response]):
            raise ValueError("Question and response must be strings")
        if not all(len(x.strip()) > 0 for x in [question, response]):
            raise ValueError("Question and response cannot be empty")
        
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
        
        # Prepare prompt (security review H-01: rubric goes in the system
        # field; the question is trusted deployer/caller input and stays in
        # the user turn alongside the untrusted response, which is wrapped
        # in an explicit boundary — see build_judge_system_prompt/
        # wrap_untrusted docstrings in uaef.metrics.utils).
        resp_fmt = """{
                       "score":float,
                       "reasoning": str
                   }
               """

        system_rubric = """
            You are an AI evaluator that helps in evaluating final response from LLM agent. 
            Please act as an impartial judge and evaluate ONLY the relevance of the response
            provided by an AI agent to the user question displayed below. You will be given the question and 
            the agent's answer. In your evaluation, focus exclusively on whether the agent's answer is relevant 
            to the user's question. Do NOT check for answer completeness or accuracy — only assess relevance.
            A relevant answer directly addresses the topic of the question, even if it is incomplete or partially correct.
            An irrelevant answer discusses unrelated topics or fails to address the question at all.
            After providing your explanation in the "reasoning" tab, 
            you must score the response on a scale of 0 to 1 in the "score" tab, 
            where 1 means fully relevant and 0 means completely irrelevant.
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))

        user_message = (
            f"[Question]\n{question}\n\n"
            + wrap_untrusted(response, "candidate_response")
        )

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
        try:
            response_json = validate_judge_response(response_json)
        except ValueError as e:
            logger.error(f"Malformed judge response: {e}. Raw response: {raw_text!r}")
            raise
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning



class CompletenessMetric(BaseMetric):
    """
    Metric for evaluating response completeness against ground truth.
    
    Compares the response against expected output to identify missing information.
    Requires ground truth for comparison.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "completeness"
    
    def requires_ground_truth(self) -> bool:
        """This metric requires ground truth."""
        return True
    
    def requires_llm_judge(self) -> bool:
        """This metric uses LLM judge for evaluation."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates if the response contains all expected information from ground truth"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Response Quality"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate completeness score."""
        # Validate expected output is provided
        expected_output = (evaluation_input.ground_truth.expected_output
                           if evaluation_input.ground_truth else None)
        if expected_output is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no expected output provided",
                metadata={"warning": "missing_data"}
            )
        if isinstance(expected_output, str) and not expected_output.strip():
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: expected output is empty",
                metadata={"warning": "invalid_data"}
            )
        
        # Extract response from trace
        trace = evaluation_input.trace
        response = None
        for msg in reversed(trace.messages):
            if msg.role == "assistant":
                response = msg.content
                break
        
        if response is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent response found in trace",
                metadata={"warning": "missing_data"}
            )
        if isinstance(response, str) and not response.strip():
            return MetricScore(
                metric_name=self.get_name(),
                score=0.0,
                reasoning="Agent response is empty",
                metadata={"warning": "invalid_data"}
            )
        
        # Use LLM judge to evaluate completeness
        try:
            config = get_config()
            score, reasoning = self._llm_judge_completeness(
                expected_output=expected_output,
                response=response,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={"expected_output": expected_output[:100], "response": response[:100]}
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
        """Calculate completeness asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_completeness(
        self,
        expected_output: str,
        response: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to evaluate completeness.
        
        Args:
            expected_output: Expected output from ground truth
            response: The agent's response
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

        # Security review H-01: rubric in system field; expected_output is
        # trusted ground truth and stays in the user turn, response is
        # untrusted agent output and is wrapped.
        system_rubric = """
            You are an AI evaluator that helps in evaluating final response from LLM agent. 
            Please act as an impartial judge and evaluate the completeness of the response
            provided by an AI agent. You will be given the expected output and the agent's answer. 
            In your evaluation, check if the agent's answer contains all the key information from the expected output.
            Identify any missing information or incomplete coverage.
            IMPORTANT: Only give credit for information that is both present AND correct. 
            If the agent provides incorrect information for a key point, do NOT give any credit for that point — 
            treat it the same as missing information.
            After providing your explanation in the "reasoning" tab, 
            you must score the response on a scale of 0 to 1 in the "score" tab, where 1 means completely covers all expected information
            and 0 means missing most or all expected information.
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))

        user_message = (
            f"[Expected Output]\n{expected_output}\n\n"
            + wrap_untrusted(response, "candidate_response")
        )

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
        try:
            response_json = validate_judge_response(response_json)
        except ValueError as e:
            logger.error(f"Malformed judge response: {e}. Raw response: {raw_text!r}")
            raise
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning



class HallucinationScoreMetric(BaseMetric):
    """
    Metric for detecting hallucinations in agent responses.
    
    Validates that claims in the response are supported by the provided context.
    Does not require ground truth, uses context documents for validation.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "hallucination_score"
    
    def requires_ground_truth(self) -> bool:
        """This metric does not require ground truth."""
        return False
    
    def requires_llm_judge(self) -> bool:
        """This metric uses LLM judge for evaluation."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Detects hallucinations by validating claims against provided context"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Response Quality"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate hallucination score."""
        # Extract response from trace
        trace = evaluation_input.trace
        response = None
        for msg in reversed(trace.messages):
            if msg.role == "assistant":
                response = msg.content
                break
        
        if response is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent response found in trace",
                metadata={"warning": "missing_data"}
            )
        if isinstance(response, str) and not response.strip():
            return MetricScore(
                metric_name=self.get_name(),
                score=0.0,
                reasoning="Agent response is empty",
                metadata={"warning": "invalid_data"}
            )
        
        # Get context documents
        context = evaluation_input.context
        if not context or len(context) == 0:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no context provided to detect hallucinations",
                metadata={"warning": "missing_data"}
            )
        
        # Use LLM judge to detect hallucinations
        try:
            config = get_config()
            score, reasoning = self._llm_judge_hallucination(
                context="\n\n".join(context),
                response=response,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={"context_count": len(context), "response": response[:100]}
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
        """Calculate hallucination score asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_hallucination(
        self,
        context: str,
        response: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to detect hallucinations.
        
        Args:
            context: Context documents concatenated
            response: The agent's response
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning) where score is 1.0 for no hallucinations, 0.0 for severe hallucinations
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
        
        # Security review H-01: rubric in system field; response is the
        # untrusted agent output and is wrapped.
        system_rubric = """
            You are an AI evaluator that helps in detecting hallucinations in LLM agent responses. 
            Please act as an impartial judge and evaluate whether the agent's response contains claims
            that are not supported by the provided context. You will be given the context documents and the agent's answer.
            In your evaluation, check if all claims and facts in the agent's answer are supported by the context.
            Identify any unsupported claims, fabricated information, or contradictions with the context.
            After providing your explanation in the "reasoning" tab, you must score the response on a scale of 0 to 1 in the "score" tab,
            where 1 means no hallucinations (all claims supported by context) and 0 means severe hallucinations (most claims unsupported).
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))

        user_message = (
            f"[Context]\n{context}\n\n"
            + wrap_untrusted(response, "candidate_response")
        )

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
        try:
            response_json = validate_judge_response(response_json)
        except ValueError as e:
            logger.error(f"Malformed judge response: {e}. Raw response: {raw_text!r}")
            raise
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning



class AccuracyMetric(BaseMetric):
    """
    Metric for evaluating response accuracy against reference answer.
    
    Compares the response against ground truth reference answer for correctness.
    Reuses llm_as_judge_score from evaluation.py (lines 360-459).
    Requires ground truth for comparison.
    """
    
    def get_name(self) -> str:
        """Get metric name."""
        return "accuracy"
    
    def requires_ground_truth(self) -> bool:
        """This metric requires ground truth."""
        return True
    
    def requires_llm_judge(self) -> bool:
        """This metric uses LLM judge for evaluation."""
        return True
    
    def get_description(self) -> Optional[str]:
        """Get metric description."""
        return "Evaluates response accuracy by comparing against reference answer using LLM judge"
    
    def get_dimension(self) -> Optional[str]:
        """Get metric dimension."""
        return "Response Quality"
    
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate accuracy score."""
        # Validate reference answer is provided
        reference = (evaluation_input.ground_truth.expected_output
                     if evaluation_input.ground_truth else None)
        if reference is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no reference answer provided",
                metadata={"warning": "missing_data"}
            )
        if isinstance(reference, str) and not reference.strip():
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: reference answer is empty",
                metadata={"warning": "invalid_data"}
            )
        
        # Extract question and response from trace
        trace = evaluation_input.trace
        
        # Get the user's question (first user message)
        question = None
        for msg in trace.messages:
            if msg.role == "user":
                question = msg.content
                break
        
        if question is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no user question found in trace",
                metadata={"warning": "missing_data"}
            )
        if isinstance(question, str) and not question.strip():
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: user question is empty",
                metadata={"warning": "invalid_data"}
            )
        
        # Get the agent's response (last assistant message)
        response = None
        for msg in reversed(trace.messages):
            if msg.role == "assistant":
                response = msg.content
                break
        
        if response is None:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning="Cannot evaluate: no agent response found in trace",
                metadata={"warning": "missing_data"}
            )
        if isinstance(response, str) and not response.strip():
            return MetricScore(
                metric_name=self.get_name(),
                score=0.0,
                reasoning="Agent response is empty",
                metadata={"warning": "invalid_data"}
            )
        
        # Use LLM judge to evaluate accuracy
        try:
            config = get_config()
            score, reasoning = self._llm_judge_accuracy(
                question=question,
                reference=reference,
                response=response,
                judge_id=config.llm_judge.model_id,
                max_tokens=config.llm_judge.max_tokens,
                temperature=config.llm_judge.temperature
            )

            return MetricScore(
                metric_name=self.get_name(),
                score=score,
                reasoning=reasoning,
                metadata={
                    "question": question[:100],
                    "reference": reference[:100],
                    "response": response[:100]
                }
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
                reasoning=f"Cannot evaluate: {str(e)}",
                metadata={"warning": "missing_data", "error_details": str(e)}
            )
        except Exception as e:
            return MetricScore(
                metric_name=self.get_name(),
                score=None,
                reasoning=f"Cannot evaluate: LLM judge error: {str(e)}",
                metadata={"warning": "api_error", "error_details": str(e)}
            )

    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """Calculate accuracy asynchronously."""
        return self.calculate(evaluation_input)
    
    def _llm_judge_accuracy(
        self,
        question: str,
        reference: str,
        response: str,
        judge_id: str,
        max_tokens: int,
        temperature: float
    ) -> tuple[float, str]:
        """
        Call LLM judge to evaluate accuracy.
        
        Reused from evaluation.py llm_as_judge_score (lines 360-459).
        
        Args:
            question: The user's question
            reference: Reference/ground truth answer
            response: The agent's response
            judge_id: Bedrock model ID
            max_tokens: Maximum tokens for response
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (score, reasoning)
            
        Raises:
            ValueError: If inputs are invalid
            ConnectionError: If Bedrock client initialization fails
            RuntimeError: If Bedrock API call fails
        """
        # Input validation
        if not all(isinstance(x, str) for x in [question, reference, response]):
            raise ValueError("Question, reference and response must be strings")
        if not all(len(x.strip()) > 0 for x in [question, reference, response]):
            raise ValueError("Question, reference and response cannot be empty")
        
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
        
        # Security review H-01: rubric in system field; question/reference
        # are trusted and stay in the user turn, response is the untrusted
        # agent output and is wrapped.
        system_rubric = """
            You are an AI evaluator that helps in evaluating final response from LLM agent. 
            Please act as an impartial judge and evaluate the correctness and format of the response
            provided by an AI agent to the user question displayed below. You will be given a reference answer 
            and the agent's answer. Begin your evaluation by comparing the agents's answer with the reference answer. 
            Identify any mistakes or missing information. After providing your explanation in the "reasoning" tab, 
            you must score the response on a scale of 0 to 1 in the "score" tab. 
            Strictly follow the below json format:{resp_fmt}."""

        system_prompt = build_judge_system_prompt(system_rubric.format(resp_fmt=resp_fmt))

        user_message = (
            f"[Question]\n{question}\n\n[Reference Answer]\n{reference}\n\n"
            + wrap_untrusted(response, "candidate_response")
        )

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
        try:
            response_json = validate_judge_response(response_json)
        except ValueError as e:
            logger.error(f"Malformed judge response: {e}. Raw response: {raw_text!r}")
            raise
        response_score = response_json["score"]
        response_reasoning = response_json.get("reasoning", "")
        
        return response_score, response_reasoning
