# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric registry and factory for UAEF.

This module provides a centralized registry for all metrics and a factory
pattern for creating metric instances by name.
"""

from typing import Dict, List, Optional, Type

from uaef.metrics.base import BaseMetric
from uaef.metrics.multi_agent import (
    AgentUtilizationMetric,
    CoordinationEfficiencyMetric,
    DelegationQualityMetric,
    WorkflowCompletionMetric,
)
from uaef.metrics.multi_turn import (
    AgentToneMetric,
    CoherenceMetric,
    ContextRetentionMetric,
    ConversationCompletenessMetric,
    HolisticLLMJudgeMetric,
    InstructionComplianceMetric,
    NaturalnessMetric,
    OptimumTurnsMetric,
    PerTurnAgentToneMetric,
    PerTurnNaturalnessMetric,
    PerTurnSentimentMetric,
    RoleAdherenceMetric,
    SentimentMetric,
    TurnEfficiencyMetric,
    UserSatisfactionMetric,
)
from uaef.metrics.performance import (
    CostEfficiencyMetric,
    LatencyScoreMetric,
    ThroughputMetric,
    TokenEfficiencyMetric,
)
from uaef.metrics.reasoning import (
    ChainOfThoughtCoherenceMetric,
    FallacyDetectionMetric,
    LogicalConsistencyMetric,
    ReasoningStepCorrectnessMetric,
)
from uaef.metrics.responsible_ai import (
    BiasScoreMetric,
    PromptInjectionDetectionMetric,
    SafetyScoreMetric,
    ToxicityScoreMetric,
)
from uaef.metrics.response_quality import (
    AccuracyMetric,
    AnswerRelevanceMetric,
    CompletenessMetric,
    HallucinationScoreMetric,
)
from uaef.metrics.tool_calling import (
    MCPComplianceMetric,
    ParameterQualityMetric,
    ToolSelectionAccuracyMetric,
    ToolSequenceCorrectnessMetric,
)


from uaef.metrics.deepeval_metrics import (
    DeepEvalAnswerRelevancyMetric,
    DeepEvalContextualPrecisionMetric,
    DeepEvalContextualRecallMetric,
    DeepEvalContextualRelevancyMetric,
    DeepEvalFaithfulnessMetric,
    DeepEvalHallucinationMetric,
    DeepEvalRoleAdherenceMetric,
    DeepEvalToolCorrectnessMetric,
)
from uaef.metrics.ragas_metrics import (
    RagasAnswerCorrectnessMetric,
    RagasAnswerPrecisionMetric,
    RagasAnswerRecallMetric,
    RagasContextPrecisionMetric,
    RagasContextRecallMetric,
    RagasFaithfulnessMetric,
    RagasToolCallAccuracyMetric,
)


class MetricRegistry:
    """
    Registry for all available metrics in UAEF.
    
    Provides centralized registration and factory methods for creating
    metric instances. Supports both built-in and custom metrics.
    """
    
    def __init__(self):
        """Initialize the metric registry."""
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._register_builtin_metrics()
    
    def _register_builtin_metrics(self) -> None:
        """Register all built-in UAEF metrics."""
        # Tool Calling metrics
        self.register(ToolSelectionAccuracyMetric)
        self.register(ToolSequenceCorrectnessMetric)
        self.register(ParameterQualityMetric)
        self.register(MCPComplianceMetric)
        
        # Response Quality metrics
        self.register(AnswerRelevanceMetric)
        self.register(CompletenessMetric)
        self.register(HallucinationScoreMetric)
        self.register(AccuracyMetric)
        
        # Responsible AI metrics
        self.register(SafetyScoreMetric)
        self.register(BiasScoreMetric)
        self.register(PromptInjectionDetectionMetric)
        self.register(ToxicityScoreMetric)
        
        # Performance metrics
        self.register(LatencyScoreMetric)
        self.register(TokenEfficiencyMetric)
        self.register(CostEfficiencyMetric)
        self.register(ThroughputMetric)
        
        # Multi-Turn metrics
        self.register(ContextRetentionMetric)
        self.register(CoherenceMetric)
        self.register(ConversationCompletenessMetric)
        self.register(TurnEfficiencyMetric)
        self.register(RoleAdherenceMetric)
        self.register(HolisticLLMJudgeMetric)
        self.register(UserSatisfactionMetric)
        self.register(SentimentMetric)
        self.register(AgentToneMetric)
        self.register(NaturalnessMetric)
        self.register(PerTurnSentimentMetric)
        self.register(PerTurnAgentToneMetric)
        self.register(PerTurnNaturalnessMetric)
        self.register(InstructionComplianceMetric)
        self.register(OptimumTurnsMetric)
        # Note: ContainmentMetric and ResolutionMetric are NOT registered here.
        # They are use-case-specific (contact-center) metrics — opt in via
        # register_use_case_specific_metrics() at module bottom.
        
        # Multi-Agent metrics
        self.register(AgentUtilizationMetric)
        self.register(DelegationQualityMetric)
        self.register(WorkflowCompletionMetric)
        self.register(CoordinationEfficiencyMetric)
        
        # Reasoning metrics
        self.register(ChainOfThoughtCoherenceMetric)
        self.register(LogicalConsistencyMetric)
        self.register(ReasoningStepCorrectnessMetric)
        self.register(FallacyDetectionMetric)

        # DeepEval integration metrics
        self.register(DeepEvalContextualPrecisionMetric)
        self.register(DeepEvalContextualRecallMetric)
        self.register(DeepEvalContextualRelevancyMetric)
        self.register(DeepEvalHallucinationMetric)
        self.register(DeepEvalFaithfulnessMetric)
        self.register(DeepEvalAnswerRelevancyMetric)
        self.register(DeepEvalToolCorrectnessMetric)
        self.register(DeepEvalRoleAdherenceMetric)

        # RAGAS integration metrics
        self.register(RagasFaithfulnessMetric)
        self.register(RagasContextPrecisionMetric)
        self.register(RagasContextRecallMetric)
        self.register(RagasAnswerPrecisionMetric)
        self.register(RagasAnswerRecallMetric)
        self.register(RagasAnswerCorrectnessMetric)
        self.register(RagasToolCallAccuracyMetric)
    
    def register(self, metric_class: Type[BaseMetric]) -> None:
        """
        Register a metric class in the registry.
        
        Args:
            metric_class: The metric class to register (must inherit from BaseMetric)
            
        Raises:
            ValueError: If metric_class is not a subclass of BaseMetric
            ValueError: If a metric with the same name is already registered
        """
        # Validate that metric_class is a subclass of BaseMetric
        if not issubclass(metric_class, BaseMetric):
            raise ValueError(f"{metric_class.__name__} must inherit from BaseMetric")
        
        # Create a temporary instance to get the metric name
        temp_instance = metric_class()
        metric_name = temp_instance.get_name()
        
        # Check if metric is already registered
        if metric_name in self._metrics:
            raise ValueError(f"Metric '{metric_name}' is already registered")
        
        # Register the metric
        self._metrics[metric_name] = metric_class
    
    def unregister(self, metric_name: str) -> None:
        """
        Unregister a metric from the registry.
        
        Args:
            metric_name: Name of the metric to unregister
            
        Raises:
            KeyError: If metric is not found in registry
        """
        if metric_name not in self._metrics:
            raise KeyError(f"Metric '{metric_name}' is not registered")
        
        del self._metrics[metric_name]
    
    def get_metric(self, metric_name: str, **kwargs) -> BaseMetric:
        """
        Factory method to create a metric instance by name.
        
        Args:
            metric_name: Name of the metric to create
            **kwargs: Additional arguments to pass to the metric constructor
            
        Returns:
            Instance of the requested metric
            
        Raises:
            KeyError: If metric is not found in registry
        """
        if metric_name not in self._metrics:
            raise KeyError(
                f"Metric '{metric_name}' is not registered. "
                f"Available metrics: {', '.join(self.list_metrics())}"
            )
        
        metric_class = self._metrics[metric_name]
        return metric_class(**kwargs)
    
    def list_metrics(self) -> List[str]:
        """
        Get a list of all registered metric names.
        
        Returns:
            List of metric names
        """
        return sorted(self._metrics.keys())
    
    def list_metrics_by_dimension(self) -> Dict[str, List[str]]:
        """
        Get metrics grouped by dimension.
        
        Returns:
            Dictionary mapping dimension names to lists of metric names
        """
        dimensions: Dict[str, List[str]] = {}
        
        for metric_name, metric_class in self._metrics.items():
            temp_instance = metric_class()
            dimension = temp_instance.get_dimension()
            
            if dimension:
                if dimension not in dimensions:
                    dimensions[dimension] = []
                dimensions[dimension].append(metric_name)
        
        # Sort metrics within each dimension
        for dimension in dimensions:
            dimensions[dimension].sort()
        
        return dimensions
    
    def get_metrics_requiring_ground_truth(self) -> List[str]:
        """
        Get list of metrics that require ground truth.
        
        Returns:
            List of metric names that require ground truth
        """
        metrics_with_gt = []
        
        for metric_name, metric_class in self._metrics.items():
            temp_instance = metric_class()
            if temp_instance.requires_ground_truth():
                metrics_with_gt.append(metric_name)
        
        return sorted(metrics_with_gt)
    
    def get_metrics_requiring_llm_judge(self) -> List[str]:
        """
        Get list of metrics that require LLM judge.
        
        Returns:
            List of metric names that require LLM judge
        """
        metrics_with_llm = []
        
        for metric_name, metric_class in self._metrics.items():
            temp_instance = metric_class()
            if temp_instance.requires_llm_judge():
                metrics_with_llm.append(metric_name)
        
        return sorted(metrics_with_llm)
    
    def validate_metric_implementation(self, metric_class: Type[BaseMetric]) -> bool:
        """
        Validate that a metric class properly implements the BaseMetric interface.
        
        Args:
            metric_class: The metric class to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if it's a subclass of BaseMetric
            if not issubclass(metric_class, BaseMetric):
                return False
            
            # Try to instantiate it
            instance = metric_class()
            
            # Check that all required methods are implemented
            required_methods = [
                'calculate',
                'calculate_async',
                'get_name',
                'requires_ground_truth',
                'requires_llm_judge'
            ]
            
            for method_name in required_methods:
                if not hasattr(instance, method_name):
                    return False
                
                method = getattr(instance, method_name)
                if not callable(method):
                    return False
            
            # Check that get_name returns a non-empty string
            name = instance.get_name()
            if not isinstance(name, str) or not name.strip():
                return False
            
            return True
        
        except Exception:
            return False
    
    def get_metric_info(self, metric_name: str) -> Dict[str, any]:
        """
        Get detailed information about a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary with metric information
            
        Raises:
            KeyError: If metric is not found in registry
        """
        if metric_name not in self._metrics:
            raise KeyError(f"Metric '{metric_name}' is not registered")
        
        metric_class = self._metrics[metric_name]
        temp_instance = metric_class()
        
        return {
            "name": temp_instance.get_name(),
            "description": temp_instance.get_description(),
            "dimension": temp_instance.get_dimension(),
            "requires_ground_truth": temp_instance.requires_ground_truth(),
            "requires_llm_judge": temp_instance.requires_llm_judge(),
            "class_name": metric_class.__name__
        }
    
    def get_all_metrics_info(self) -> List[Dict[str, any]]:
        """
        Get detailed information about all registered metrics.
        
        Returns:
            List of dictionaries with metric information
        """
        return [
            self.get_metric_info(metric_name)
            for metric_name in self.list_metrics()
        ]


# Global registry instance
_registry: Optional[MetricRegistry] = None


def get_registry() -> MetricRegistry:
    """
    Get the global metric registry instance.
    
    Creates the registry on first call with all built-in metrics registered.
    
    Returns:
        MetricRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = MetricRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry to None."""
    global _registry
    _registry = None


def register_metric(metric_class: Type[BaseMetric]) -> None:
    """
    Register a custom metric in the global registry.
    
    Args:
        metric_class: The metric class to register
    """
    registry = get_registry()
    registry.register(metric_class)


def get_metric(metric_name: str, **kwargs) -> BaseMetric:
    """
    Factory function to create a metric instance by name.
    
    Args:
        metric_name: Name of the metric to create
        **kwargs: Additional arguments to pass to the metric constructor
        
    Returns:
        Instance of the requested metric
    """
    registry = get_registry()
    return registry.get_metric(metric_name, **kwargs)


def list_metrics() -> List[str]:
    """
    Get a list of all registered metric names.
    
    Returns:
        List of metric names
    """
    registry = get_registry()
    return registry.list_metrics()


def list_metrics_by_dimension() -> Dict[str, List[str]]:
    """
    Get metrics grouped by dimension.

    Returns:
        Dictionary mapping dimension names to lists of metric names
    """
    registry = get_registry()
    return registry.list_metrics_by_dimension()


def register_use_case_specific_metrics() -> None:
    """
    Opt-in registration for use-case-specific metrics.

    These metrics evaluate behavior that's only meaningful for specific
    use cases (e.g., customer-service / contact-center agents) and are
    deliberately excluded from the default registry. Call this once at
    startup if you want them to participate in evaluations.

    Currently registers:
        - ContainmentMetric ("containment")
        - ResolutionMetric ("resolution")

    The call is idempotent — invoking it multiple times has no effect.

    Example:
        from uaef.metrics.registry import register_use_case_specific_metrics

        register_use_case_specific_metrics()
        result = evaluate(trace, metrics=["containment", "resolution"])
    """
    from uaef.metrics.use_case_specific import ContainmentMetric, ResolutionMetric

    registry = get_registry()
    for cls in (ContainmentMetric, ResolutionMetric):
        try:
            registry.register(cls)
        except ValueError:
            # Already registered — calling twice is a no-op.
            pass
