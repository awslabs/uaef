# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base metric interface for UAEF."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import boto3

from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore


def get_bedrock_client():
    """
    Create a Bedrock runtime client using the default credential chain.

    Uses config for region but lets boto3 resolve credentials from
    environment variables, IAM roles, or credential files — never passes
    None explicitly which would break the default chain.
    """
    from uaef.config import get_config
    config = get_config()
    client_kwargs = {
        "service_name": "bedrock-runtime",
        "region_name": config.aws.region,
    }
    if config.aws.access_key_id and config.aws.secret_access_key:
        client_kwargs["aws_access_key_id"] = config.aws.access_key_id
        client_kwargs["aws_secret_access_key"] = config.aws.secret_access_key
        if config.aws.session_token:
            client_kwargs["aws_session_token"] = config.aws.session_token
    return boto3.client(**client_kwargs)


class BaseMetric(ABC):
    """
    Abstract base class for all metrics in UAEF.
    
    All metrics must inherit from this class and implement the calculate method.
    Metrics can be deterministic (rule-based), ground truth-based (require expected outputs),
    or LLM-based (use LLM judge for subjective evaluation).
    """
    
    @abstractmethod
    def calculate(self, evaluation_input: EvaluationInput) -> MetricScore:
        """
        Calculate the metric score for the given evaluation input.
        
        Args:
            evaluation_input: Input data containing trace, ground truth, and context
            
        Returns:
            MetricScore object with score, reasoning, and metadata
            
        Raises:
            ValueError: If required data is missing (e.g., ground truth when required)
        """
        pass
    
    @abstractmethod
    async def calculate_async(self, evaluation_input: EvaluationInput) -> MetricScore:
        """
        Asynchronously calculate the metric score (for LLM-based metrics).
        
        Args:
            evaluation_input: Input data containing trace, ground truth, and context
            
        Returns:
            MetricScore object with score, reasoning, and metadata
            
        Raises:
            ValueError: If required data is missing
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this metric.
        
        Returns:
            String name of the metric (e.g., "tool_selection_accuracy")
        """
        pass
    
    @abstractmethod
    def requires_ground_truth(self) -> bool:
        """
        Check if this metric requires ground truth data.
        
        Returns:
            True if ground truth is required, False otherwise
        """
        pass
    
    @abstractmethod
    def requires_llm_judge(self) -> bool:
        """
        Check if this metric requires an LLM judge for evaluation.
        
        Returns:
            True if LLM judge is required, False otherwise
        """
        pass
    
    def get_description(self) -> Optional[str]:
        """
        Get a human-readable description of what this metric measures.
        
        Returns:
            Description string or None
        """
        return None
    
    def get_dimension(self) -> Optional[str]:
        """
        Get the dimension this metric belongs to.

        Returns:
            Dimension name (e.g., "Tool Calling", "Response Quality") or None
        """
        return None

    def get_dependencies(self) -> Dict[str, List[str]]:
        """
        Get dependencies for this metric.

        Returns:
            Dictionary with dependency information:
            - "requires_metrics": List of metric names that must run first
            - "provides_metadata": List of metadata keys this metric stores in trace
            - "requires_metadata": List of metadata keys this metric needs from trace

        Example:
            {"requires_metadata": ["quality_score"]}
            {"requires_metrics": ["conversation_completeness"], "requires_metadata": ["completeness_score"]}
        """
        return {}

    def batch_group(self) -> Optional[str]:
        """Return a batch group key if this metric supports batched evaluation.

        Metrics that return the same non-None key will be grouped together and
        evaluated via ``batch_calculate`` in a single call, which can be much
        faster for metrics that make external LLM calls.

        Returns:
            A string key identifying the batch group, or None (default) to
            indicate this metric does not support batching.
        """
        return None

    @classmethod
    def batch_calculate(
        cls,
        metrics: List["BaseMetric"],
        evaluation_input: "EvaluationInput",
    ) -> List["MetricScore"]:
        """Evaluate multiple metrics of this batch group in a single operation.

        Override this in subclasses that support batching. The base evaluator
        calls this instead of individual ``calculate()`` when metrics share the
        same ``batch_group()`` key.

        Args:
            metrics: List of metric instances in this batch group.
            evaluation_input: Shared evaluation input for all metrics.

        Returns:
            List of MetricScore objects (one per successfully evaluated metric).
        """
        raise NotImplementedError(
            "Subclasses that return a batch_group() must implement batch_calculate()"
        )
