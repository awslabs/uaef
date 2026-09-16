# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bedrock client wrapper for UAEF LLM Judge.

This module provides a centralized Bedrock client with retry logic,
error handling, and batch processing capabilities.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from uaef.config import get_config
from uaef.logging import get_logger

logger = get_logger(__name__)


class BedrockClientError(Exception):
    """Base exception for Bedrock client errors."""
    pass


class BedrockRateLimitError(BedrockClientError):
    """Exception raised when rate limit is exceeded."""
    pass


class BedrockTimeoutError(BedrockClientError):
    """Exception raised when request times out."""
    pass


class BedrockServiceError(BedrockClientError):
    """Exception raised for Bedrock service errors."""
    pass


class BedrockClient:
    """
    Wrapper for AWS Bedrock client with retry logic and error handling.
    
    Provides:
    - Automatic retry with exponential backoff
    - Error handling for rate limits, timeouts, service errors
    - Support for multiple Bedrock models
    - Batch invocation for offline evaluation
    """
    
    def __init__(
        self,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        retry_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region: AWS region (defaults to config)
            access_key_id: AWS access key ID (defaults to config)
            secret_access_key: AWS secret access key (defaults to config)
            session_token: AWS session token (defaults to config)
            retry_attempts: Number of retry attempts (defaults to config)
            retry_delay: Initial retry delay in seconds (defaults to config)
        """
        config = get_config()
        
        # Use provided values or fall back to config
        self.region = region or config.aws.region
        self.access_key_id = access_key_id or config.aws.access_key_id
        self.secret_access_key = secret_access_key or config.aws.secret_access_key
        self.session_token = session_token or config.aws.session_token
        self.retry_attempts = retry_attempts or config.llm_judge.retry_attempts
        self.retry_delay = retry_delay or config.llm_judge.retry_delay
        
        # Initialize boto3 client
        self._client = self._create_client()
        
        logger.info(
            f"Initialized BedrockClient with region={self.region}, "
            f"retry_attempts={self.retry_attempts}"
        )
    
    def _create_client(self):
        """Create boto3 Bedrock runtime client."""
        try:
            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": self.region
            }
            
            # Add credentials if provided
            if self.access_key_id:
                client_kwargs["aws_access_key_id"] = self.access_key_id
            if self.secret_access_key:
                client_kwargs["aws_secret_access_key"] = self.secret_access_key
            if self.session_token:
                client_kwargs["aws_session_token"] = self.session_token
            
            return boto3.client(**client_kwargs)
        except Exception as e:
            raise BedrockClientError(f"Failed to create Bedrock client: {str(e)}")
    
    def invoke_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.1,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
        system: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Invoke a Bedrock model with retry logic.
        
        Args:
            model_id: Bedrock model ID (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of stop sequences
            system: Optional system-level instruction, sent via the model's
                dedicated system field (Anthropic Claude's top-level `system`)
                where supported. Security review H-01: callers judging
                untrusted, evaluated-agent-controlled text should pass the
                evaluation rubric here and keep only the boundary-wrapped
                untrusted text in `prompt`, so the trust boundary is enforced
                by the API shape itself rather than by convention alone. For
                model families without a distinct system field, `system` is
                prepended to `prompt` as a best-effort fallback.
            
        Returns:
            Tuple of (response_text, metadata) where metadata includes usage stats
            
        Raises:
            BedrockRateLimitError: If rate limit exceeded after retries
            BedrockTimeoutError: If request times out after retries
            BedrockServiceError: If service error occurs after retries
            BedrockClientError: For other errors
        """
        # Prepare request body based on model family
        body = self._prepare_request_body(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences or ["Human"],
            system=system
        )
        
        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(
                    f"Invoking Bedrock model {model_id} (attempt {attempt + 1}/{self.retry_attempts})"
                )
                
                response = self._client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body),
                    accept='application/json',
                    contentType='application/json'
                )
                
                # Parse response
                response_body = json.loads(response.get('body').read())
                response_text, metadata = self._parse_response(model_id, response_body)
                
                logger.debug(
                    f"Successfully invoked Bedrock model {model_id} "
                    f"(tokens: {metadata.get('usage', {}).get('total_tokens', 'unknown')})"
                )
                
                return response_text, metadata
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                # Classify error type
                if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                    last_exception = BedrockRateLimitError(
                        f"Rate limit exceeded: {error_message}"
                    )
                elif error_code in ['TimeoutError', 'RequestTimeout']:
                    last_exception = BedrockTimeoutError(
                        f"Request timeout: {error_message}"
                    )
                elif error_code in ['ServiceUnavailable', 'InternalServerError']:
                    last_exception = BedrockServiceError(
                        f"Service error: {error_message}"
                    )
                else:
                    last_exception = BedrockClientError(
                        f"Bedrock API error ({error_code}): {error_message}"
                    )
                
                # Log and retry if not last attempt
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Bedrock API call failed (attempt {attempt + 1}/{self.retry_attempts}): "
                        f"{error_message}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Bedrock API call failed after {self.retry_attempts} attempts: "
                        f"{error_message}"
                    )
            
            except Exception as e:
                last_exception = BedrockClientError(f"Unexpected error: {str(e)}")
                logger.error(f"Unexpected error invoking Bedrock: {str(e)}")
                break  # Don't retry on unexpected errors
        
        # All retries exhausted
        raise last_exception
    
    def batch_invoke(
        self,
        model_id: str,
        prompts: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.1,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None
    ) -> List[Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Exception]]]:
        """
        Invoke Bedrock model for multiple prompts (batch processing).
        
        This method processes prompts sequentially with retry logic for each.
        For true parallel batch processing, use AWS Bedrock Batch Inference API.
        
        Args:
            model_id: Bedrock model ID
            prompts: List of prompts to process
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of stop sequences
            
        Returns:
            List of tuples (response_text, metadata, error) for each prompt.
            If successful, error is None. If failed, response_text and metadata are None.
        """
        logger.info(f"Starting batch invocation for {len(prompts)} prompts")
        
        results = []
        for i, prompt in enumerate(prompts):
            try:
                response_text, metadata = self.invoke_model(
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences
                )
                results.append((response_text, metadata, None))
                logger.debug(f"Batch item {i+1}/{len(prompts)} succeeded")
            except Exception as e:
                logger.warning(f"Batch item {i+1}/{len(prompts)} failed: {str(e)}")
                results.append((None, None, e))
        
        success_count = sum(1 for r in results if r[2] is None)
        logger.info(
            f"Batch invocation complete: {success_count}/{len(prompts)} succeeded"
        )
        
        return results
    
    def _prepare_request_body(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_sequences: List[str],
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare request body based on model family.
        
        Different Bedrock models have different request formats.

        Security review H-01: ``system``, when provided, is sent via the
        model's own system-level field (Anthropic Claude's top-level
        `system`) so an evaluation rubric stays structurally separate from
        the (boundary-wrapped) untrusted text in `prompt`. Model families
        with no distinct system field fall back to prepending `system` to
        `prompt` — weaker than a true field separation, but still better
        than the caller having to concatenate it in themselves.
        """
        # Anthropic Claude models — Claude 3+ rejects requests with both temperature and top_p;
        # temperature=0.0 alone is sufficient for deterministic judge scoring.
        if "anthropic.claude" in model_id:
            body: Dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }]
                }],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop_sequences": stop_sequences
            }
            if system:
                body["system"] = system
            return body
        
        # Amazon Titan models
        elif "amazon.titan" in model_id:
            return {
                "inputText": f"{system}\n\n{prompt}" if system else prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                    "stopSequences": stop_sequences
                }
            }
        
        # AI21 Jurassic models
        elif "ai21.j2" in model_id:
            return {
                "prompt": f"{system}\n\n{prompt}" if system else prompt,
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
                "stopSequences": stop_sequences
            }
        
        # Cohere Command models
        elif "cohere.command" in model_id:
            return {
                "prompt": f"{system}\n\n{prompt}" if system else prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "p": top_p,
                "k": top_k,
                "stop_sequences": stop_sequences
            }
        
        # Meta Llama models
        elif "meta.llama" in model_id:
            return {
                "prompt": f"{system}\n\n{prompt}" if system else prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        
        # Default format (Anthropic-style)
        else:
            logger.warning(
                f"Unknown model family for {model_id}, using Anthropic format"
            )
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }]
                }],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop_sequences": stop_sequences
            }
            if system:
                body["system"] = system
            return body
    
    def _parse_response(
        self,
        model_id: str,
        response_body: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Parse response based on model family.
        
        Returns:
            Tuple of (response_text, metadata)
        """
        metadata = {}
        
        # Anthropic Claude models
        if "anthropic.claude" in model_id:
            response_text = response_body.get('content', [{}])[0].get('text', '')
            metadata['usage'] = response_body.get('usage', {})
            metadata['stop_reason'] = response_body.get('stop_reason')
        
        # Amazon Titan models
        elif "amazon.titan" in model_id:
            results = response_body.get('results', [{}])
            response_text = results[0].get('outputText', '') if results else ''
            metadata['usage'] = {
                'input_tokens': response_body.get('inputTextTokenCount', 0),
                'output_tokens': results[0].get('tokenCount', 0) if results else 0
            }
        
        # AI21 Jurassic models
        elif "ai21.j2" in model_id:
            completions = response_body.get('completions', [{}])
            response_text = completions[0].get('data', {}).get('text', '') if completions else ''
            metadata['usage'] = {
                'total_tokens': response_body.get('prompt', {}).get('tokens', [])
            }
        
        # Cohere Command models
        elif "cohere.command" in model_id:
            generations = response_body.get('generations', [{}])
            response_text = generations[0].get('text', '') if generations else ''
            metadata['usage'] = {}
        
        # Meta Llama models
        elif "meta.llama" in model_id:
            response_text = response_body.get('generation', '')
            metadata['usage'] = {
                'prompt_token_count': response_body.get('prompt_token_count', 0),
                'generation_token_count': response_body.get('generation_token_count', 0)
            }
        
        # Default (Anthropic-style)
        else:
            response_text = response_body.get('content', [{}])[0].get('text', '')
            metadata['usage'] = response_body.get('usage', {})
        
        return response_text, metadata
