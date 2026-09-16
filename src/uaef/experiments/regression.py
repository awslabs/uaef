# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression detector for identifying performance degradations."""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from uaef.experiments.models import ExperimentRun

logger = logging.getLogger(__name__)


class RegressionSeverity(str, Enum):
    """Severity levels for regressions."""
    
    MINOR = "minor"  # 5-10% degradation
    MODERATE = "moderate"  # 10-20% degradation
    SEVERE = "severe"  # >20% degradation


class Regression(BaseModel):
    """
    A detected regression in a metric.
    
    Attributes:
        metric_name: Name of the metric with regression
        baseline_score: Score from baseline run
        current_score: Score from current run
        delta: Absolute difference (current - baseline)
        percent_change: Percentage change
        severity: Severity level of the regression
        dimension: Dimension the metric belongs to (if known)
    """
    
    metric_name: str = Field(..., description="Name of the metric with regression")
    baseline_score: float = Field(..., description="Score from baseline run")
    current_score: float = Field(..., description="Score from current run")
    delta: float = Field(..., description="Absolute difference (current - baseline)")
    percent_change: float = Field(..., description="Percentage change")
    severity: RegressionSeverity = Field(..., description="Severity level of the regression")
    dimension: Optional[str] = Field(
        None,
        description="Dimension the metric belongs to (if known)"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_name": "tool_accuracy",
                "baseline_score": 0.85,
                "current_score": 0.65,
                "delta": -0.20,
                "percent_change": -23.53,
                "severity": "severe",
                "dimension": "tool_calling"
            }
        }


class RegressionReport(BaseModel):
    """
    Complete regression detection report.
    
    Attributes:
        baseline_run_id: ID of the baseline run
        current_run_id: ID of the current run
        regressions: List of detected regressions
        regressions_by_severity: Regressions grouped by severity
        regressions_by_dimension: Regressions grouped by dimension
        summary: Summary statistics
    """
    
    baseline_run_id: UUID = Field(..., description="ID of the baseline run")
    current_run_id: UUID = Field(..., description="ID of the current run")
    regressions: List[Regression] = Field(..., description="List of detected regressions")
    regressions_by_severity: Dict[str, List[str]] = Field(
        ...,
        description="Regressions grouped by severity"
    )
    regressions_by_dimension: Dict[str, List[str]] = Field(
        ...,
        description="Regressions grouped by dimension"
    )
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "baseline_run_id": "550e8400-e29b-41d4-a716-446655440000",
                "current_run_id": "660e8400-e29b-41d4-a716-446655440001",
                "regressions": [],
                "regressions_by_severity": {
                    "severe": ["tool_accuracy"],
                    "moderate": ["answer_relevance"],
                    "minor": []
                },
                "regressions_by_dimension": {
                    "tool_calling": ["tool_accuracy"],
                    "response_quality": ["answer_relevance"]
                },
                "summary": {
                    "total_regressions": 2,
                    "severe_count": 1,
                    "moderate_count": 1,
                    "minor_count": 0
                }
            }
        }


class RegressionDetector:
    """
    Detector for identifying performance regressions.
    
    Provides methods for:
    - Detecting regressions between runs
    - Calculating severity based on magnitude
    - Generating regression reports
    """
    
    # Default thresholds for severity levels
    DEFAULT_MINOR_THRESHOLD = 5.0  # 5-10% degradation
    DEFAULT_MODERATE_THRESHOLD = 10.0  # 10-20% degradation
    DEFAULT_SEVERE_THRESHOLD = 20.0  # >20% degradation
    
    # Mapping of metric names to dimensions (can be extended)
    METRIC_DIMENSION_MAP = {
        "tool_accuracy": "tool_calling",
        "tool_selection_accuracy": "tool_calling",
        "tool_sequence_correctness": "tool_calling",
        "parameter_quality": "tool_calling",
        "answer_relevance": "response_quality",
        "completeness": "response_quality",
        "hallucination_score": "response_quality",
        "accuracy": "response_quality",
        "safety_score": "responsible_ai",
        "bias_score": "responsible_ai",
        "toxicity_score": "responsible_ai",
        "latency_score": "performance",
        "token_efficiency": "performance",
        "cost_efficiency": "performance",
        "context_retention": "multi_turn",
        "coherence": "multi_turn",
        "conversation_completeness": "multi_turn",
        "agent_utilization": "multi_agent",
        "delegation_quality": "multi_agent",
        "workflow_completion": "multi_agent",
        "chain_of_thought_coherence": "reasoning",
        "logical_consistency": "reasoning",
    }
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        minor_threshold: float = DEFAULT_MINOR_THRESHOLD,
        moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
        severe_threshold: float = DEFAULT_SEVERE_THRESHOLD
    ):
        """
        Initialize the regression detector.
        
        Args:
            thresholds: Custom thresholds per metric (metric_name -> threshold %)
            minor_threshold: Global threshold for minor regressions (default: 5%)
            moderate_threshold: Global threshold for moderate regressions (default: 10%)
            severe_threshold: Global threshold for severe regressions (default: 20%)
        """
        self.thresholds = thresholds or {}
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold
        
        logger.info(
            f"RegressionDetector initialized with thresholds: "
            f"minor={minor_threshold}%, moderate={moderate_threshold}%, "
            f"severe={severe_threshold}%"
        )
    
    def detect_regressions(
        self,
        baseline_run: ExperimentRun,
        current_run: ExperimentRun,
        thresholds: Optional[Dict[str, float]] = None
    ) -> RegressionReport:
        """
        Detect regressions between baseline and current runs.
        
        Args:
            baseline_run: Baseline run for comparison
            current_run: Current run to check for regressions
            thresholds: Custom thresholds per metric (overrides defaults)
            
        Returns:
            RegressionReport with detected regressions
            
        Example:
            >>> detector = RegressionDetector()
            >>> report = detector.detect_regressions(baseline_run, current_run)
            >>> print(f"Found {len(report.regressions)} regressions")
            >>> for regression in report.regressions:
            ...     print(f"  {regression.metric_name}: {regression.severity}")
        """
        # Use custom thresholds if provided
        metric_thresholds = thresholds or self.thresholds
        
        # Get aggregate metrics from both runs
        baseline_metrics = baseline_run.aggregate_metrics
        current_metrics = current_run.aggregate_metrics
        
        # Find common metrics
        common_metrics = set(baseline_metrics.keys()) & set(current_metrics.keys())
        
        if not common_metrics:
            logger.warning(
                f"No common metrics found between runs {baseline_run.run_id} "
                f"and {current_run.run_id}"
            )
        
        # Detect regressions
        regressions: List[Regression] = []
        
        for metric_name in sorted(common_metrics):
            baseline_score = baseline_metrics[metric_name]
            current_score = current_metrics[metric_name]
            
            # Get threshold for this metric
            threshold = metric_thresholds.get(metric_name, self.minor_threshold)
            
            # Check if this is a regression
            regression = self._check_regression(
                metric_name=metric_name,
                baseline_score=baseline_score,
                current_score=current_score,
                threshold=threshold
            )
            
            if regression:
                regressions.append(regression)
        
        # Group regressions by severity
        regressions_by_severity = {
            "severe": [],
            "moderate": [],
            "minor": []
        }
        
        for regression in regressions:
            regressions_by_severity[regression.severity.value].append(
                regression.metric_name
            )
        
        # Group regressions by dimension
        regressions_by_dimension: Dict[str, List[str]] = {}
        
        for regression in regressions:
            dimension = regression.dimension or "unknown"
            if dimension not in regressions_by_dimension:
                regressions_by_dimension[dimension] = []
            regressions_by_dimension[dimension].append(regression.metric_name)
        
        # Generate summary
        summary = {
            "total_regressions": len(regressions),
            "severe_count": len(regressions_by_severity["severe"]),
            "moderate_count": len(regressions_by_severity["moderate"]),
            "minor_count": len(regressions_by_severity["minor"]),
            "dimensions_affected": len(regressions_by_dimension)
        }
        
        report = RegressionReport(
            baseline_run_id=baseline_run.run_id,
            current_run_id=current_run.run_id,
            regressions=regressions,
            regressions_by_severity=regressions_by_severity,
            regressions_by_dimension=regressions_by_dimension,
            summary=summary
        )
        
        logger.info(
            f"Detected {len(regressions)} regressions between runs "
            f"{baseline_run.run_id} and {current_run.run_id}: "
            f"{summary['severe_count']} severe, {summary['moderate_count']} moderate, "
            f"{summary['minor_count']} minor"
        )
        
        return report
    
    def _check_regression(
        self,
        metric_name: str,
        baseline_score: float,
        current_score: float,
        threshold: float
    ) -> Optional[Regression]:
        """
        Check if a metric has regressed.
        
        Args:
            metric_name: Name of the metric
            baseline_score: Score from baseline run
            current_score: Score from current run
            threshold: Threshold for regression detection
            
        Returns:
            Regression instance if regression detected, None otherwise
        """
        # Calculate delta and percent change
        delta = current_score - baseline_score
        
        # Calculate percent change (handle division by zero)
        if baseline_score == 0:
            if current_score == 0:
                percent_change = 0.0
            else:
                percent_change = 100.0 if current_score > 0 else -100.0
        else:
            percent_change = (delta / baseline_score) * 100.0
        
        # Check if this is a regression (negative change beyond threshold)
        if percent_change >= -threshold:
            return None  # Not a regression
        
        # Determine severity based on magnitude
        abs_percent_change = abs(percent_change)
        
        if abs_percent_change >= self.severe_threshold:
            severity = RegressionSeverity.SEVERE
        elif abs_percent_change >= self.moderate_threshold:
            severity = RegressionSeverity.MODERATE
        else:
            severity = RegressionSeverity.MINOR
        
        # Get dimension for this metric
        dimension = self.METRIC_DIMENSION_MAP.get(metric_name)
        
        return Regression(
            metric_name=metric_name,
            baseline_score=baseline_score,
            current_score=current_score,
            delta=delta,
            percent_change=percent_change,
            severity=severity,
            dimension=dimension
        )
    
    def get_regression_summary(
        self,
        report: RegressionReport
    ) -> str:
        """
        Generate a human-readable summary of a regression report.
        
        Args:
            report: RegressionReport to summarize
            
        Returns:
            Human-readable summary string
            
        Example:
            >>> summary = detector.get_regression_summary(report)
            >>> print(summary)
        """
        lines = [
            f"Regression Detection Report",
            f"===========================",
            f"Baseline Run: {report.baseline_run_id}",
            f"Current Run: {report.current_run_id}",
            f"",
            f"Summary:",
            f"  Total Regressions: {report.summary['total_regressions']}",
            f"  Severe: {report.summary['severe_count']}",
            f"  Moderate: {report.summary['moderate_count']}",
            f"  Minor: {report.summary['minor_count']}",
            f"  Dimensions Affected: {report.summary['dimensions_affected']}",
            f""
        ]
        
        if not report.regressions:
            lines.append("No regressions detected.")
            return "\n".join(lines)
        
        # Group by severity
        for severity in ["severe", "moderate", "minor"]:
            metric_names = report.regressions_by_severity.get(severity, [])
            if metric_names:
                lines.append(f"{severity.capitalize()} Regressions:")
                for metric_name in metric_names:
                    regression = next(
                        r for r in report.regressions
                        if r.metric_name == metric_name
                    )
                    lines.append(
                        f"  - {metric_name}: {regression.baseline_score:.3f} → "
                        f"{regression.current_score:.3f} ({regression.percent_change:+.1f}%)"
                    )
                lines.append("")
        
        # Group by dimension
        if report.regressions_by_dimension:
            lines.append("Regressions by Dimension:")
            for dimension, metric_names in sorted(report.regressions_by_dimension.items()):
                lines.append(f"  {dimension}: {', '.join(metric_names)}")
            lines.append("")
        
        return "\n".join(lines)
