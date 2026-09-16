# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comparison engine for comparing experiment runs."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from uaef.experiments.models import ExperimentRun

logger = logging.getLogger(__name__)


class MetricComparison(BaseModel):
    """
    Comparison result for a single metric.
    
    Attributes:
        metric_name: Name of the metric
        baseline_score: Score from baseline run
        current_score: Score from current run
        delta: Absolute difference (current - baseline)
        percent_change: Percentage change ((current - baseline) / baseline * 100)
        is_regression: Whether this is a regression (negative change > threshold)
        is_improvement: Whether this is an improvement (positive change > threshold)
    """
    
    metric_name: str = Field(..., description="Name of the metric")
    baseline_score: float = Field(..., description="Score from baseline run")
    current_score: float = Field(..., description="Score from current run")
    delta: float = Field(..., description="Absolute difference (current - baseline)")
    percent_change: float = Field(
        ...,
        description="Percentage change ((current - baseline) / baseline * 100)"
    )
    is_regression: bool = Field(
        ...,
        description="Whether this is a regression (negative change > threshold)"
    )
    is_improvement: bool = Field(
        ...,
        description="Whether this is an improvement (positive change > threshold)"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_name": "tool_accuracy",
                "baseline_score": 0.85,
                "current_score": 0.78,
                "delta": -0.07,
                "percent_change": -8.24,
                "is_regression": True,
                "is_improvement": False
            }
        }


class ComparisonReport(BaseModel):
    """
    Complete comparison report between two runs.
    
    Attributes:
        baseline_run_id: ID of the baseline run
        current_run_id: ID of the current run
        metric_comparisons: List of metric-by-metric comparisons
        regressions: List of metrics with regressions
        improvements: List of metrics with improvements
        unchanged: List of metrics with no significant change
        summary: Summary statistics
    """
    
    baseline_run_id: UUID = Field(..., description="ID of the baseline run")
    current_run_id: UUID = Field(..., description="ID of the current run")
    metric_comparisons: List[MetricComparison] = Field(
        ...,
        description="List of metric-by-metric comparisons"
    )
    regressions: List[str] = Field(
        ...,
        description="List of metrics with regressions"
    )
    improvements: List[str] = Field(
        ...,
        description="List of metrics with improvements"
    )
    unchanged: List[str] = Field(
        ...,
        description="List of metrics with no significant change"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "baseline_run_id": "550e8400-e29b-41d4-a716-446655440000",
                "current_run_id": "660e8400-e29b-41d4-a716-446655440001",
                "metric_comparisons": [],
                "regressions": ["tool_accuracy"],
                "improvements": ["answer_relevance"],
                "unchanged": ["latency_score"],
                "summary": {
                    "total_metrics": 3,
                    "regression_count": 1,
                    "improvement_count": 1,
                    "unchanged_count": 1
                }
            }
        }


class ComparisonEngine:
    """
    Engine for comparing experiment runs.
    
    Provides methods for:
    - Comparing two runs metric-by-metric
    - Identifying regressions and improvements
    - Generating comparison reports
    """
    
    def __init__(
        self,
        regression_threshold: float = 5.0,
        improvement_threshold: float = 5.0
    ):
        """
        Initialize the comparison engine.
        
        Args:
            regression_threshold: Percentage threshold for regression detection (default: 5.0%)
            improvement_threshold: Percentage threshold for improvement detection (default: 5.0%)
        """
        self.regression_threshold = regression_threshold
        self.improvement_threshold = improvement_threshold
        logger.info(
            f"ComparisonEngine initialized with regression_threshold={regression_threshold}%, "
            f"improvement_threshold={improvement_threshold}%"
        )
    
    def compare_runs(
        self,
        baseline_run: ExperimentRun,
        current_run: ExperimentRun,
        regression_threshold: Optional[float] = None,
        improvement_threshold: Optional[float] = None
    ) -> ComparisonReport:
        """
        Compare two experiment runs metric-by-metric.
        
        Args:
            baseline_run: Baseline run for comparison
            current_run: Current run to compare against baseline
            regression_threshold: Custom regression threshold (overrides default)
            improvement_threshold: Custom improvement threshold (overrides default)
            
        Returns:
            ComparisonReport with detailed comparison results
            
        Example:
            >>> engine = ComparisonEngine()
            >>> report = engine.compare_runs(baseline_run, current_run)
            >>> print(f"Regressions: {report.regressions}")
            >>> print(f"Improvements: {report.improvements}")
        """
        # Use custom thresholds if provided
        reg_threshold = regression_threshold or self.regression_threshold
        imp_threshold = improvement_threshold or self.improvement_threshold
        
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
        
        # Compare each metric
        metric_comparisons: List[MetricComparison] = []
        regressions: List[str] = []
        improvements: List[str] = []
        unchanged: List[str] = []
        
        for metric_name in sorted(common_metrics):
            baseline_score = baseline_metrics[metric_name]
            current_score = current_metrics[metric_name]
            
            comparison = self._compare_metric(
                metric_name=metric_name,
                baseline_score=baseline_score,
                current_score=current_score,
                regression_threshold=reg_threshold,
                improvement_threshold=imp_threshold
            )
            
            metric_comparisons.append(comparison)
            
            # Categorize the change
            if comparison.is_regression:
                regressions.append(metric_name)
            elif comparison.is_improvement:
                improvements.append(metric_name)
            else:
                unchanged.append(metric_name)
        
        # Generate summary
        summary = {
            "total_metrics": len(common_metrics),
            "regression_count": len(regressions),
            "improvement_count": len(improvements),
            "unchanged_count": len(unchanged),
            "regression_threshold": reg_threshold,
            "improvement_threshold": imp_threshold
        }
        
        report = ComparisonReport(
            baseline_run_id=baseline_run.run_id,
            current_run_id=current_run.run_id,
            metric_comparisons=metric_comparisons,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            summary=summary
        )
        
        logger.info(
            f"Compared runs {baseline_run.run_id} and {current_run.run_id}: "
            f"{len(regressions)} regressions, {len(improvements)} improvements, "
            f"{len(unchanged)} unchanged"
        )
        
        return report
    
    def _compare_metric(
        self,
        metric_name: str,
        baseline_score: float,
        current_score: float,
        regression_threshold: float,
        improvement_threshold: float
    ) -> MetricComparison:
        """
        Compare a single metric between baseline and current.
        
        Args:
            metric_name: Name of the metric
            baseline_score: Score from baseline run
            current_score: Score from current run
            regression_threshold: Threshold for regression detection
            improvement_threshold: Threshold for improvement detection
            
        Returns:
            MetricComparison instance
        """
        # Calculate delta
        delta = current_score - baseline_score
        
        # Calculate percent change (handle division by zero)
        if baseline_score == 0:
            if current_score == 0:
                percent_change = 0.0
            else:
                # If baseline is 0 and current is not, treat as 100% change
                percent_change = 100.0 if current_score > 0 else -100.0
        else:
            percent_change = (delta / baseline_score) * 100.0
        
        # Determine if regression or improvement
        is_regression = percent_change < -regression_threshold
        is_improvement = percent_change > improvement_threshold
        
        return MetricComparison(
            metric_name=metric_name,
            baseline_score=baseline_score,
            current_score=current_score,
            delta=delta,
            percent_change=percent_change,
            is_regression=is_regression,
            is_improvement=is_improvement
        )
    
    def compare_multiple_runs(
        self,
        baseline_run: ExperimentRun,
        current_runs: List[ExperimentRun],
        regression_threshold: Optional[float] = None,
        improvement_threshold: Optional[float] = None
    ) -> List[ComparisonReport]:
        """
        Compare multiple runs against a baseline.
        
        Args:
            baseline_run: Baseline run for comparison
            current_runs: List of runs to compare against baseline
            regression_threshold: Custom regression threshold (overrides default)
            improvement_threshold: Custom improvement threshold (overrides default)
            
        Returns:
            List of ComparisonReport instances, one for each current run
            
        Example:
            >>> engine = ComparisonEngine()
            >>> reports = engine.compare_multiple_runs(baseline_run, [run1, run2, run3])
            >>> for report in reports:
            ...     print(f"Run {report.current_run_id}: {len(report.regressions)} regressions")
        """
        reports = []
        
        for current_run in current_runs:
            report = self.compare_runs(
                baseline_run=baseline_run,
                current_run=current_run,
                regression_threshold=regression_threshold,
                improvement_threshold=improvement_threshold
            )
            reports.append(report)
        
        logger.info(
            f"Compared baseline {baseline_run.run_id} against {len(current_runs)} runs"
        )
        
        return reports
    
    def get_comparison_summary(
        self,
        report: ComparisonReport
    ) -> str:
        """
        Generate a human-readable summary of a comparison report.
        
        Args:
            report: ComparisonReport to summarize
            
        Returns:
            Human-readable summary string
            
        Example:
            >>> summary = engine.get_comparison_summary(report)
            >>> print(summary)
        """
        lines = [
            f"Comparison Report",
            f"=================",
            f"Baseline Run: {report.baseline_run_id}",
            f"Current Run: {report.current_run_id}",
            f"",
            f"Summary:",
            f"  Total Metrics: {report.summary['total_metrics']}",
            f"  Regressions: {report.summary['regression_count']}",
            f"  Improvements: {report.summary['improvement_count']}",
            f"  Unchanged: {report.summary['unchanged_count']}",
            f""
        ]
        
        if report.regressions:
            lines.append("Regressions:")
            for metric_name in report.regressions:
                comparison = next(
                    c for c in report.metric_comparisons
                    if c.metric_name == metric_name
                )
                lines.append(
                    f"  - {metric_name}: {comparison.baseline_score:.3f} → "
                    f"{comparison.current_score:.3f} ({comparison.percent_change:+.1f}%)"
                )
            lines.append("")
        
        if report.improvements:
            lines.append("Improvements:")
            for metric_name in report.improvements:
                comparison = next(
                    c for c in report.metric_comparisons
                    if c.metric_name == metric_name
                )
                lines.append(
                    f"  - {metric_name}: {comparison.baseline_score:.3f} → "
                    f"{comparison.current_score:.3f} ({comparison.percent_change:+.1f}%)"
                )
            lines.append("")
        
        return "\n".join(lines)
