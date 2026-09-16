# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics module for UAEF."""

from typing import Dict, List

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
from uaef.metrics.use_case_specific import (
    ContainmentMetric,
    ResolutionMetric,
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
from uaef.metrics.registry import (
    MetricRegistry,
    get_metric,
    get_registry,
    list_metrics,
    list_metrics_by_dimension,
    register_metric,
    register_use_case_specific_metrics,
    reset_registry,
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


def get_full_metric_catalog() -> Dict[str, List[str]]:
    """Return every available metric name grouped by dimension/integration.

    This is the single source of truth for the metric catalog presented by
    the demo backend and the UAEF service. It merges two registries:

    - The built-in metric registry, grouped by dimension (e.g. "Response
      Quality", "Tool Calling"), via :func:`list_metrics_by_dimension`.
    - The external integrations registry, grouped by integration name (e.g.
      "ragas", "deepeval"), via the integrations registry's
      ``list_all_metrics()``.

    Integration groups are included only when their optional extra is
    installed. If an integration extra is not installed, that integration's
    group is omitted from the returned catalog without raising
    (Requirement 12.8). Callers must not hardcode metric-name lists
    (Requirement 12.6).

    Returns:
        Dict mapping each dimension or integration group name to a sorted
        list of the metric names available in that group.
    """
    catalog: Dict[str, List[str]] = {}

    # Built-in metrics grouped by dimension.
    for dimension, metric_names in list_metrics_by_dimension().items():
        catalog[dimension] = sorted(metric_names)

    # Integration metrics grouped by integration name. The integrations
    # registry only reports connectors whose backing library is installed, so
    # groups for uninstalled extras are naturally omitted. We additionally
    # guard the import so a missing integrations subpackage does not raise.
    #
    # De-duplication: the built-in metric registry is the authoritative source.

    try:
        from uaef.integrations.registry import get_registry as _get_integrations_registry

        _existing_upper = {k.upper() for k in catalog}
        integrations_registry = _get_integrations_registry()
        for integration_name, metric_names in integrations_registry.list_all_metrics().items():
            # Skip if a built-in dimension already covers this integration.
            if integration_name.upper() in _existing_upper:
                continue
            catalog[integration_name] = sorted(metric_names)
    except ImportError:
        # Integrations subpackage or its extras are unavailable; omit
        # integration groups rather than failing the catalog.
        pass

    return catalog


__all__ = [
    # Base
    "BaseMetric",
    # Tool Calling
    "ToolSelectionAccuracyMetric",
    "ToolSequenceCorrectnessMetric",
    "ParameterQualityMetric",
    "MCPComplianceMetric",
    # Response Quality
    "AnswerRelevanceMetric",
    "CompletenessMetric",
    "HallucinationScoreMetric",
    "AccuracyMetric",
    # Responsible AI
    "SafetyScoreMetric",
    "BiasScoreMetric",
    "PromptInjectionDetectionMetric",
    "ToxicityScoreMetric",
    # Performance
    "LatencyScoreMetric",
    "TokenEfficiencyMetric",
    "CostEfficiencyMetric",
    "ThroughputMetric",
    # Multi-Turn
    "ContextRetentionMetric",
    "CoherenceMetric",
    "ConversationCompletenessMetric",
    "TurnEfficiencyMetric",
    "RoleAdherenceMetric",
    "HolisticLLMJudgeMetric",
    "UserSatisfactionMetric",
    "SentimentMetric",
    "AgentToneMetric",
    "NaturalnessMetric",
    "InstructionComplianceMetric",
    "OptimumTurnsMetric",
    "PerTurnSentimentMetric",
    "PerTurnAgentToneMetric",
    "PerTurnNaturalnessMetric",
    # Use-Case-Specific (opt-in via register_use_case_specific_metrics)
    "ContainmentMetric",
    "ResolutionMetric",
    # Multi-Agent
    "AgentUtilizationMetric",
    "DelegationQualityMetric",
    "WorkflowCompletionMetric",
    "CoordinationEfficiencyMetric",
    # Reasoning
    "ChainOfThoughtCoherenceMetric",
    "LogicalConsistencyMetric",
    "ReasoningStepCorrectnessMetric",
    "FallacyDetectionMetric",
    # Registry
    "MetricRegistry",
    "get_registry",
    "reset_registry",
    "register_metric",
    "register_use_case_specific_metrics",
    "get_metric",
    "list_metrics",
    "list_metrics_by_dimension",
    # Catalog
    "get_full_metric_catalog",
]
