# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root cause analyzer for evaluation failures."""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from uaef.models.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class FailureCategory(BaseModel):
    """
    Category of failure with associated metrics.
    
    Attributes:
        category: Type of failure (tool_selection, reasoning, context, response)
        failed_metrics: List of metric names that failed in this category
        severity: Severity level (low, medium, high, critical)
        description: Human-readable description of the failure
    """
    
    category: str = Field(..., description="Type of failure")
    failed_metrics: List[str] = Field(
        default_factory=list,
        description="List of metric names that failed in this category"
    )
    severity: str = Field(..., description="Severity level")
    description: str = Field(..., description="Human-readable description of the failure")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "category": "tool_selection",
                "failed_metrics": ["tool_accuracy", "tool_sequence_accuracy"],
                "severity": "high",
                "description": "Agent selected incorrect tools or used them in wrong order"
            }
        }


class RootCauseReport(BaseModel):
    """
    Root cause analysis report for a failed evaluation.
    
    Attributes:
        evaluation_id: ID of the evaluated trace
        trace_id: ID of the agent trace
        failure_categories: List of identified failure categories
        primary_root_cause: Most likely root cause
        recommendations: List of actionable recommendations
        details: Additional details about the analysis
    """
    
    evaluation_id: str = Field(..., description="ID of the evaluated trace")
    trace_id: str = Field(..., description="ID of the agent trace")
    failure_categories: List[FailureCategory] = Field(
        default_factory=list,
        description="List of identified failure categories"
    )
    primary_root_cause: str = Field(..., description="Most likely root cause")
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of actionable recommendations"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the analysis"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "evaluation_id": "550e8400-e29b-41d4-a716-446655440000",
                "trace_id": "660e8400-e29b-41d4-a716-446655440001",
                "failure_categories": [],
                "primary_root_cause": "tool_selection",
                "recommendations": [
                    "Review tool selection logic in agent prompt",
                    "Add examples of correct tool usage"
                ],
                "details": {"overall_score": 0.45, "failed_dimensions": ["tool_calling"]}
            }
        }


class FailurePattern(BaseModel):
    """
    Pattern identified across multiple evaluation failures.
    
    Attributes:
        pattern_type: Type of pattern (e.g., "consistent_tool_error", "context_loss")
        frequency: Number of evaluations exhibiting this pattern
        affected_metrics: List of metrics commonly affected
        description: Description of the pattern
        severity: Average severity across occurrences
        recommendations: Recommendations to address this pattern
    """
    
    pattern_type: str = Field(..., description="Type of pattern")
    frequency: int = Field(..., description="Number of evaluations exhibiting this pattern")
    affected_metrics: List[str] = Field(
        default_factory=list,
        description="List of metrics commonly affected"
    )
    description: str = Field(..., description="Description of the pattern")
    severity: str = Field(..., description="Average severity across occurrences")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations to address this pattern"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "pattern_type": "consistent_tool_error",
                "frequency": 15,
                "affected_metrics": ["tool_accuracy", "tool_sequence_accuracy"],
                "description": "Agent consistently selects wrong tools for data retrieval tasks",
                "severity": "high",
                "recommendations": [
                    "Add more tool selection examples to system prompt",
                    "Consider simplifying tool descriptions"
                ]
            }
        }


class RootCauseAnalyzer:
    """
    Analyzer for identifying root causes of evaluation failures.
    
    Provides methods for:
    - Analyzing individual evaluation failures
    - Identifying patterns across multiple failures
    - Categorizing failures by type
    - Generating actionable recommendations
    """
    
    # Metric to category mapping
    METRIC_CATEGORIES = {
        # Tool calling metrics
        "tool_accuracy": "tool_selection",
        "tool_call_accuracy": "tool_selection",
        "tool_sequence_accuracy": "tool_selection",
        "parameter_quality": "tool_selection",
        "incorrect_tool_rate": "tool_selection",
        "missed_tool_rate": "tool_selection",
        
        # Response quality metrics
        "answer_relevance": "response",
        "answer_correctness": "response",
        "completeness": "response",
        "hallucination_score": "response",
        "faithfulness": "response",
        
        # Context metrics
        "context_retention": "context",
        "context_relevance": "context",
        "context_precision": "context",
        "context_recall": "context",
        
        # Reasoning metrics
        "reasoning_coherence": "reasoning",
        "logical_consistency": "reasoning",
        "reasoning_completeness": "reasoning",
        "fallacy_detection": "reasoning",
        "chain_of_thought_quality": "reasoning",
    }
    
    def __init__(self, severity_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the root cause analyzer.
        
        Args:
            severity_thresholds: Custom thresholds for severity classification
                                Format: {"critical": 0.3, "high": 0.5, "medium": 0.7}
        """
        self.severity_thresholds = severity_thresholds or {
            "critical": 0.3,
            "high": 0.5,
            "medium": 0.7,
            "low": 1.0
        }
        logger.info("RootCauseAnalyzer initialized")
    
    def analyze_failure(
        self,
        evaluation_result: EvaluationResult
    ) -> RootCauseReport:
        """
        Analyze a single failed evaluation to identify root causes.
        
        Args:
            evaluation_result: EvaluationResult with failures
            
        Returns:
            RootCauseReport with identified root causes and recommendations
            
        Raises:
            ValueError: If evaluation_result passed (no failures to analyze)
            
        Example:
            >>> analyzer = RootCauseAnalyzer()
            >>> report = analyzer.analyze_failure(failed_evaluation)
            >>> print(f"Primary root cause: {report.primary_root_cause}")
            >>> print(f"Recommendations: {report.recommendations}")
        """
        if evaluation_result.passed:
            raise ValueError(
                f"Cannot analyze failure for passed evaluation {evaluation_result.evaluation_id}"
            )
        
        logger.info(
            f"Analyzing failure for evaluation {evaluation_result.evaluation_id}, "
            f"trace {evaluation_result.trace_id}"
        )
        
        # Collect failed metrics by dimension
        failed_metrics_by_dimension: Dict[str, List[str]] = defaultdict(list)
        failed_metrics_by_category: Dict[str, List[str]] = defaultdict(list)
        
        for dimension_result in evaluation_result.dimension_results:
            for metric_score in dimension_result.metric_scores:
                # Consider a metric failed if score is below 0.7 (typical threshold)
                if metric_score.score < 0.7:
                    failed_metrics_by_dimension[dimension_result.dimension_name].append(
                        metric_score.metric_name
                    )
                    
                    # Map to category
                    category = self._get_metric_category(metric_score.metric_name)
                    failed_metrics_by_category[category].append(metric_score.metric_name)
        
        # Create failure categories
        failure_categories = []
        for category, metrics in failed_metrics_by_category.items():
            if not metrics:
                continue
            
            # Calculate severity based on number of failed metrics and their scores
            severity = self._calculate_severity(evaluation_result, metrics)
            
            # Generate description
            description = self._generate_category_description(category, metrics)
            
            failure_categories.append(FailureCategory(
                category=category,
                failed_metrics=list(set(metrics)),  # Remove duplicates
                severity=severity,
                description=description
            ))
        
        # Determine primary root cause (category with most failures or highest severity)
        primary_root_cause = self._determine_primary_root_cause(failure_categories)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            failure_categories,
            primary_root_cause,
            evaluation_result
        )
        
        # Collect details
        details = {
            "overall_score": evaluation_result.overall_score,
            "failed_dimensions": list(failed_metrics_by_dimension.keys()),
            "total_failed_metrics": sum(len(m) for m in failed_metrics_by_category.values()),
            "failures": evaluation_result.failures,
            "warnings": evaluation_result.warnings
        }
        
        report = RootCauseReport(
            evaluation_id=str(evaluation_result.evaluation_id),
            trace_id=str(evaluation_result.trace_id),
            failure_categories=failure_categories,
            primary_root_cause=primary_root_cause,
            recommendations=recommendations,
            details=details
        )
        
        logger.info(
            f"Root cause analysis complete for {evaluation_result.evaluation_id}: "
            f"primary cause = {primary_root_cause}, "
            f"{len(failure_categories)} categories identified"
        )
        
        return report
    
    def identify_failure_patterns(
        self,
        evaluation_results: List[EvaluationResult]
    ) -> List[FailurePattern]:
        """
        Identify patterns across multiple evaluation failures.
        
        Args:
            evaluation_results: List of EvaluationResult instances (passed and failed)
            
        Returns:
            List of FailurePattern instances, sorted by frequency (descending)
            
        Example:
            >>> analyzer = RootCauseAnalyzer()
            >>> patterns = analyzer.identify_failure_patterns(evaluation_results)
            >>> for pattern in patterns:
            ...     print(f"{pattern.pattern_type}: {pattern.frequency} occurrences")
        """
        if not evaluation_results:
            logger.warning("No evaluation results provided for pattern identification")
            return []
        
        # Filter to failed evaluations
        failed_evaluations = [r for r in evaluation_results if not r.passed]
        
        if not failed_evaluations:
            logger.info("No failed evaluations found in provided results")
            return []
        
        logger.info(
            f"Identifying failure patterns across {len(failed_evaluations)} failed evaluations "
            f"(out of {len(evaluation_results)} total)"
        )
        
        # Track metrics that fail together
        category_failures: Dict[str, int] = Counter()
        metric_failures: Dict[str, int] = Counter()
        category_to_metrics: Dict[str, List[str]] = defaultdict(list)
        
        for result in failed_evaluations:
            failed_metrics_in_eval = set()
            
            for dimension_result in result.dimension_results:
                for metric_score in dimension_result.metric_scores:
                    if metric_score.score < 0.7:
                        metric_name = metric_score.metric_name
                        failed_metrics_in_eval.add(metric_name)
                        metric_failures[metric_name] += 1
                        
                        category = self._get_metric_category(metric_name)
                        category_to_metrics[category].append(metric_name)
            
            # Count category failures
            categories_in_eval = set(
                self._get_metric_category(m) for m in failed_metrics_in_eval
            )
            for category in categories_in_eval:
                category_failures[category] += 1
        
        # Generate patterns from categories
        patterns = []
        for category, frequency in category_failures.most_common():
            # Get most common metrics in this category
            metrics_in_category = category_to_metrics[category]
            metric_counter = Counter(metrics_in_category)
            affected_metrics = [m for m, _ in metric_counter.most_common(5)]
            
            # Determine severity based on frequency
            severity = self._calculate_pattern_severity(frequency, len(failed_evaluations))
            
            # Generate pattern description
            description = self._generate_pattern_description(
                category,
                frequency,
                len(failed_evaluations),
                affected_metrics
            )
            
            # Generate recommendations
            recommendations = self._generate_pattern_recommendations(
                category,
                affected_metrics,
                frequency
            )
            
            pattern = FailurePattern(
                pattern_type=f"{category}_failures",
                frequency=frequency,
                affected_metrics=affected_metrics,
                description=description,
                severity=severity,
                recommendations=recommendations
            )
            patterns.append(pattern)
        
        logger.info(f"Identified {len(patterns)} failure patterns")
        
        return patterns
    
    def _get_metric_category(self, metric_name: str) -> str:
        """
        Get the category for a metric name.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Category name (tool_selection, reasoning, context, response, or unknown)
        """
        return self.METRIC_CATEGORIES.get(metric_name, "unknown")
    
    def _calculate_severity(
        self,
        evaluation_result: EvaluationResult,
        failed_metrics: List[str]
    ) -> str:
        """
        Calculate severity level based on scores and number of failures.
        
        Args:
            evaluation_result: The evaluation result
            failed_metrics: List of failed metric names
            
        Returns:
            Severity level: "critical", "high", "medium", or "low"
        """
        # Get average score of failed metrics
        scores = []
        for dimension_result in evaluation_result.dimension_results:
            for metric_score in dimension_result.metric_scores:
                if metric_score.metric_name in failed_metrics:
                    scores.append(metric_score.score)
        
        if not scores:
            return "low"
        
        avg_score = sum(scores) / len(scores)
        
        # Determine severity based on thresholds
        for severity, threshold in sorted(
            self.severity_thresholds.items(),
            key=lambda x: x[1]
        ):
            if avg_score <= threshold:
                return severity
        
        return "low"
    
    def _generate_category_description(
        self,
        category: str,
        failed_metrics: List[str]
    ) -> str:
        """
        Generate a human-readable description for a failure category.
        
        Args:
            category: Category name
            failed_metrics: List of failed metrics in this category
            
        Returns:
            Human-readable description
        """
        # Get unique metrics and limit to first 3
        unique_metrics = list(set(failed_metrics))[:3]
        metrics_str = ', '.join(unique_metrics)
        
        descriptions = {
            "tool_selection": (
                f"Agent had issues with tool selection and usage. "
                f"Failed metrics: {metrics_str}"
            ),
            "reasoning": (
                f"Agent's reasoning process had logical issues or inconsistencies. "
                f"Failed metrics: {metrics_str}"
            ),
            "context": (
                f"Agent failed to properly retain or use context information. "
                f"Failed metrics: {metrics_str}"
            ),
            "response": (
                f"Agent's response quality was insufficient. "
                f"Failed metrics: {metrics_str}"
            ),
            "unknown": (
                f"Unclassified failure. "
                f"Failed metrics: {metrics_str}"
            )
        }
        
        return descriptions.get(category, descriptions["unknown"])
    
    def _determine_primary_root_cause(
        self,
        failure_categories: List[FailureCategory]
    ) -> str:
        """
        Determine the primary root cause from failure categories.
        
        Args:
            failure_categories: List of failure categories
            
        Returns:
            Primary root cause category name
        """
        if not failure_categories:
            return "unknown"
        
        # Severity weights
        severity_weights = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        # Calculate weighted score for each category
        category_scores = []
        for fc in failure_categories:
            weight = severity_weights.get(fc.severity, 1)
            score = len(fc.failed_metrics) * weight
            category_scores.append((fc.category, score))
        
        # Return category with highest score
        primary = max(category_scores, key=lambda x: x[1])
        return primary[0]
    
    def _generate_recommendations(
        self,
        failure_categories: List[FailureCategory],
        primary_root_cause: str,
        evaluation_result: EvaluationResult
    ) -> List[str]:
        """
        Generate actionable recommendations based on failure analysis.
        
        Args:
            failure_categories: List of failure categories
            primary_root_cause: Primary root cause
            evaluation_result: The evaluation result
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Category-specific recommendations
        category_recommendations = {
            "tool_selection": [
                "Review and improve tool selection logic in agent prompt",
                "Add more examples of correct tool usage to system prompt",
                "Verify tool descriptions are clear and unambiguous",
                "Consider simplifying the tool set or combining similar tools"
            ],
            "reasoning": [
                "Enhance chain-of-thought prompting in system instructions",
                "Add explicit reasoning validation steps",
                "Include examples of correct reasoning patterns",
                "Consider using a reasoning-optimized model"
            ],
            "context": [
                "Improve context retention mechanisms",
                "Add explicit context summarization steps",
                "Reduce context window usage to prevent information loss",
                "Implement better context retrieval strategies"
            ],
            "response": [
                "Improve response generation prompts",
                "Add quality checks before returning responses",
                "Include more examples of high-quality responses",
                "Consider post-processing to improve response quality"
            ]
        }
        
        # Add recommendations for primary root cause
        if primary_root_cause in category_recommendations:
            recommendations.extend(category_recommendations[primary_root_cause][:2])
        
        # Add recommendations for other significant categories
        for fc in failure_categories:
            if fc.category != primary_root_cause and fc.severity in ["critical", "high"]:
                if fc.category in category_recommendations:
                    recommendations.append(category_recommendations[fc.category][0])
        
        # Add general recommendation based on overall score
        if evaluation_result.overall_score < 0.3:
            recommendations.insert(
                0,
                "Overall score is critically low - consider comprehensive agent redesign"
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_pattern_severity(
        self,
        frequency: int,
        total_failures: int
    ) -> str:
        """
        Calculate severity for a failure pattern based on frequency.
        
        Args:
            frequency: Number of times pattern occurred
            total_failures: Total number of failed evaluations
            
        Returns:
            Severity level: "critical", "high", "medium", or "low"
        """
        percentage = (frequency / total_failures) * 100 if total_failures > 0 else 0
        
        if percentage >= 75:
            return "critical"
        elif percentage >= 50:
            return "high"
        elif percentage >= 25:
            return "medium"
        else:
            return "low"
    
    def _generate_pattern_description(
        self,
        category: str,
        frequency: int,
        total_failures: int,
        affected_metrics: List[str]
    ) -> str:
        """
        Generate description for a failure pattern.
        
        Args:
            category: Category name
            frequency: Number of occurrences
            total_failures: Total number of failures
            affected_metrics: List of commonly affected metrics
            
        Returns:
            Human-readable description
        """
        percentage = (frequency / total_failures) * 100 if total_failures > 0 else 0
        
        category_descriptions = {
            "tool_selection": "tool selection and usage",
            "reasoning": "reasoning and logical consistency",
            "context": "context retention and usage",
            "response": "response quality and accuracy",
            "unknown": "unclassified issues"
        }
        
        category_desc = category_descriptions.get(category, category)
        
        return (
            f"Agent consistently fails on {category_desc} "
            f"({frequency}/{total_failures} failures, {percentage:.1f}%). "
            f"Most affected metrics: {', '.join(affected_metrics[:3])}"
        )
    
    def _generate_pattern_recommendations(
        self,
        category: str,
        affected_metrics: List[str],
        frequency: int
    ) -> List[str]:
        """
        Generate recommendations for addressing a failure pattern.
        
        Args:
            category: Category name
            affected_metrics: List of affected metrics
            frequency: Number of occurrences
            
        Returns:
            List of recommendations
        """
        base_recommendations = {
            "tool_selection": [
                "Systematically review and improve tool selection prompts",
                "Add comprehensive tool usage examples to training data",
                "Consider implementing tool selection validation logic",
                "Simplify tool descriptions and reduce tool set complexity"
            ],
            "reasoning": [
                "Implement structured reasoning frameworks (e.g., ReAct, Chain-of-Thought)",
                "Add reasoning validation checkpoints in agent workflow",
                "Use reasoning-optimized models or fine-tuning",
                "Include diverse reasoning examples in system prompt"
            ],
            "context": [
                "Implement robust context management strategies",
                "Add context summarization between conversation turns",
                "Use retrieval-augmented generation (RAG) for long contexts",
                "Optimize context window usage and prioritization"
            ],
            "response": [
                "Enhance response generation with quality templates",
                "Implement response validation before returning to user",
                "Add post-processing steps for response refinement",
                "Use higher-quality models for response generation"
            ]
        }
        
        recommendations = base_recommendations.get(category, [
            "Investigate root cause through detailed trace analysis",
            "Consider agent architecture improvements",
            "Add monitoring and alerting for this failure type"
        ])
        
        # Add urgency note for high-frequency patterns
        if frequency > 10:
            recommendations.insert(
                0,
                f"HIGH PRIORITY: This pattern occurs frequently ({frequency} times) - "
                "immediate action recommended"
            )
        
        return recommendations[:4]  # Limit to top 4 recommendations
