# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executive dashboard generator for high-level performance summaries."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from uaef.analysis.recommendations import RecommendationEngine
from uaef.analysis.trends import TrendAnalyzer
from uaef.models.evaluation_result import EvaluationResult
from uaef.reporting.models import Chart, ChartType, Dashboard

logger = logging.getLogger(__name__)


class ExecutiveDashboardGenerator:
    """
    Generator for executive dashboards with high-level performance summaries.
    
    Creates dashboards suitable for stakeholders showing:
    - Overall performance scores across all dimensions
    - Key metrics with current values and trends
    - Critical issues and failures
    - Visual indicators for improvement/decline
    - Executive summary with actionable insights
    
    Integrates with TrendAnalyzer and RecommendationEngine for comprehensive insights.
    
    Examples:
        >>> # Generate dashboard from experiment run results
        >>> generator = ExecutiveDashboardGenerator()
        >>> results = [result1, result2, result3]  # EvaluationResult objects
        >>> dashboard = generator.generate_dashboard(
        ...     results=results,
        ...     title="Customer Support Agent - Q1 2024",
        ...     run_id="550e8400-e29b-41d4-a716-446655440000"
        ... )
        >>> print(f"Dashboard: {dashboard.title}")
        >>> print(f"Charts: {len(dashboard.charts)}")
        
        >>> # Generate with trend analysis
        >>> trend_analyzer = TrendAnalyzer()
        >>> generator = ExecutiveDashboardGenerator(trend_analyzer=trend_analyzer)
        >>> dashboard = generator.generate_dashboard(
        ...     results=results,
        ...     title="Agent Performance Dashboard",
        ...     include_trends=True
        ... )
        
        >>> # Generate with recommendations
        >>> rec_engine = RecommendationEngine()
        >>> generator = ExecutiveDashboardGenerator(recommendation_engine=rec_engine)
        >>> dashboard = generator.generate_dashboard(
        ...     results=results,
        ...     title="Executive Dashboard",
        ...     include_recommendations=True
        ... )
    """
    
    def __init__(
        self,
        trend_analyzer: Optional[TrendAnalyzer] = None,
        recommendation_engine: Optional[RecommendationEngine] = None
    ):
        """
        Initialize the executive dashboard generator.
        
        Args:
            trend_analyzer: Optional TrendAnalyzer for trend indicators
            recommendation_engine: Optional RecommendationEngine for insights
        """
        self.trend_analyzer = trend_analyzer or TrendAnalyzer()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
    
    def generate_dashboard(
        self,
        results: List[EvaluationResult],
        title: str,
        run_id: Optional[UUID] = None,
        experiment_id: Optional[UUID] = None,
        include_trends: bool = True,
        include_recommendations: bool = True,
        description: Optional[str] = None
    ) -> Dashboard:
        """
        Generate an executive dashboard from evaluation results.
        
        Args:
            results: List of evaluation results to analyze
            title: Dashboard title
            run_id: Optional experiment run ID
            experiment_id: Optional experiment ID
            include_trends: Whether to include trend indicators
            include_recommendations: Whether to include recommendations
            description: Optional dashboard description
        
        Returns:
            Dashboard with executive summary charts
        
        Raises:
            ValueError: If results list is empty
            ValueError: If title is empty
        
        Examples:
            >>> generator = ExecutiveDashboardGenerator()
            >>> results = [result1, result2, result3]
            >>> dashboard = generator.generate_dashboard(
            ...     results=results,
            ...     title="Q1 Performance Dashboard"
            ... )
        """
        if not results:
            raise ValueError("Cannot generate dashboard from empty results list")
        
        if not title or not title.strip():
            raise ValueError("Dashboard title cannot be empty")
        
        logger.info(f"Generating executive dashboard for {len(results)} evaluation results")
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        # Generate charts
        charts = []
        
        # Chart 1: Overall performance score
        charts.append(self._create_overall_score_chart(aggregate_metrics))
        
        # Chart 2: Dimension breakdown
        charts.append(self._create_dimension_breakdown_chart(aggregate_metrics))
        
        # Chart 3: Key metrics summary
        charts.append(self._create_key_metrics_chart(aggregate_metrics))
        
        # Chart 4: Pass/Fail distribution
        charts.append(self._create_pass_fail_chart(results))
        
        # Chart 5: Critical issues (if any)
        critical_issues = self._identify_critical_issues(results, aggregate_metrics)
        if critical_issues:
            charts.append(self._create_critical_issues_chart(critical_issues))
        
        # Chart 6: Trend indicators (if enabled and available)
        if include_trends and len(results) > 1:
            try:
                trend_chart = self._create_trend_indicators_chart(results, aggregate_metrics)
                if trend_chart:
                    charts.append(trend_chart)
            except Exception as e:
                logger.warning(f"Failed to generate trend chart: {e}")
        
        # Prepare metadata
        metadata = {
            "dashboard_type": "executive",
            "evaluation_count": len(results),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        if run_id:
            metadata["run_id"] = str(run_id)
        if experiment_id:
            metadata["experiment_id"] = str(experiment_id)
        
        # Add aggregate metrics to metadata
        metadata["aggregate_metrics"] = aggregate_metrics
        
        # Add critical issues summary
        if critical_issues:
            metadata["critical_issues_count"] = len(critical_issues)
            metadata["critical_issues"] = critical_issues
        
        # Add recommendations if enabled
        if include_recommendations:
            try:
                rec_report = self.recommendation_engine.generate_recommendations(results)
                top_recommendations = [
                    {
                        "title": rec.title,
                        "priority": rec.priority,
                        "category": rec.category
                    }
                    for rec in rec_report.recommendations[:3]  # Top 3
                ]
                metadata["top_recommendations"] = top_recommendations
            except Exception as e:
                logger.warning(f"Failed to generate recommendations: {e}")
        
        # Generate description if not provided
        if not description:
            description = self._generate_executive_summary(
                results, aggregate_metrics, critical_issues
            )
        
        # Create dashboard
        dashboard = Dashboard(
            title=title,
            description=description,
            charts=charts,
            metadata=metadata
        )
        
        logger.info(f"Generated executive dashboard with {len(charts)} charts")
        return dashboard
    
    def _calculate_aggregate_metrics(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all evaluation results.
        
        Args:
            results: List of evaluation results
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        # Calculate overall score
        overall_scores = [r.overall_score for r in results]
        avg_overall_score = sum(overall_scores) / len(overall_scores)
        
        # Calculate dimension scores
        dimension_scores = defaultdict(list)
        for result in results:
            for dim_result in result.dimension_results:
                dimension_scores[dim_result.dimension_name].append(
                    dim_result.aggregate_score
                )
        
        avg_dimension_scores = {
            dim: sum(scores) / len(scores)
            for dim, scores in dimension_scores.items()
        }
        
        # Calculate metric scores
        metric_scores = defaultdict(list)
        none_metrics = set()
        for result in results:
            for dim_result in result.dimension_results:
                for metric_score in dim_result.metric_scores:
                    if metric_score.score is not None:
                        metric_scores[metric_score.metric_name].append(metric_score.score)
                    else:
                        none_metrics.add(metric_score.metric_name)

        if none_metrics:
            logger.warning(
                f"Skipped {len(none_metrics)} metric(s) with None scores: "
                f"{sorted(none_metrics)}"
            )

        avg_metric_scores = {
            metric: sum(scores) / len(scores)
            for metric, scores in metric_scores.items()
            if scores
        }
        
        # Calculate pass/fail statistics
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        pass_rate = passed_count / len(results) if results else 0.0
        
        return {
            "overall_score": avg_overall_score,
            "dimension_scores": avg_dimension_scores,
            "metric_scores": avg_metric_scores,
            "total_evaluations": len(results),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "pass_rate": pass_rate
        }
    
    def _create_overall_score_chart(self, aggregate_metrics: Dict[str, Any]) -> Chart:
        """Create chart showing overall performance score."""
        overall_score = aggregate_metrics.get("overall_score", 0.0)
        
        # Determine color based on score
        if overall_score >= 0.8:
            color = "#4CAF50"  # Green
        elif overall_score >= 0.6:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        
        return Chart(
            title="Overall Performance Score",
            chart_type=ChartType.BAR,
            data={
                "labels": ["Overall Score"],
                "values": [round(overall_score, 3)]
            },
            labels=["Overall Score"],
            colors=[color],
            options={
                "show_values": True,
                "max_value": 1.0,
                "horizontal": False
            },
            description=f"Average overall score: {overall_score:.1%}"
        )
    
    def _create_dimension_breakdown_chart(
        self, aggregate_metrics: Dict[str, Any]
    ) -> Chart:
        """Create radar chart showing dimension breakdown."""
        dimension_scores = aggregate_metrics.get("dimension_scores", {})
        
        if not dimension_scores:
            # Return empty chart if no dimensions
            return Chart(
                title="Dimension Breakdown",
                chart_type=ChartType.RADAR,
                data={"dimensions": [], "scores": []},
                labels=[],
                colors=["#2196F3"],
                description="No dimension data available"
            )
        
        dimensions = list(dimension_scores.keys())
        scores = [round(dimension_scores[dim], 3) for dim in dimensions]
        
        return Chart(
            title="Dimension Breakdown",
            chart_type=ChartType.RADAR,
            data={
                "dimensions": dimensions,
                "scores": scores
            },
            labels=dimensions,
            colors=["#2196F3"],
            options={
                "max_value": 1.0,
                "show_legend": True
            },
            description=f"Performance across {len(dimensions)} dimensions"
        )
    
    def _create_key_metrics_chart(self, aggregate_metrics: Dict[str, Any]) -> Chart:
        """Create bar chart showing top key metrics."""
        metric_scores = aggregate_metrics.get("metric_scores", {})
        
        if not metric_scores:
            return Chart(
                title="Key Metrics",
                chart_type=ChartType.BAR,
                data={"metrics": [], "scores": []},
                labels=[],
                colors=[],
                description="No metric data available"
            )
        
        # Sort metrics by score and take top 10
        sorted_metrics = sorted(
            metric_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        metrics = [m[0] for m in sorted_metrics]
        scores = [round(m[1], 3) for m in sorted_metrics]
        
        # Color code based on score
        colors = []
        for score in scores:
            if score >= 0.8:
                colors.append("#4CAF50")  # Green
            elif score >= 0.6:
                colors.append("#FF9800")  # Orange
            else:
                colors.append("#F44336")  # Red
        
        return Chart(
            title="Key Metrics",
            chart_type=ChartType.BAR,
            data={
                "metrics": metrics,
                "scores": scores
            },
            labels=metrics,
            colors=colors,
            options={
                "show_values": True,
                "horizontal": True,
                "max_value": 1.0
            },
            description=f"Top {len(metrics)} metrics by performance"
        )
    
    def _create_pass_fail_chart(self, results: List[EvaluationResult]) -> Chart:
        """Create pie chart showing pass/fail distribution."""
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        return Chart(
            title="Pass/Fail Distribution",
            chart_type=ChartType.PIE,
            data={
                "labels": ["Passed", "Failed"],
                "values": [passed_count, failed_count]
            },
            labels=["Passed", "Failed"],
            colors=["#4CAF50", "#F44336"],
            options={
                "show_percentages": True,
                "show_legend": True
            },
            description=f"Pass rate: {passed_count}/{len(results)} ({passed_count/len(results):.1%})"
        )
    
    def _identify_critical_issues(
        self,
        results: List[EvaluationResult],
        aggregate_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify critical issues from evaluation results.
        
        Args:
            results: List of evaluation results
            aggregate_metrics: Aggregate metrics
        
        Returns:
            List of critical issues
        """
        critical_issues = []
        
        # Check for low overall score
        overall_score = aggregate_metrics.get("overall_score", 1.0)
        if overall_score < 0.6:
            critical_issues.append({
                "type": "low_overall_score",
                "severity": "critical",
                "description": f"Overall score ({overall_score:.1%}) is below acceptable threshold (60%)",
                "metric": "overall_score",
                "value": overall_score
            })
        
        # Check for low dimension scores
        dimension_scores = aggregate_metrics.get("dimension_scores", {})
        for dim, score in dimension_scores.items():
            if score < 0.6:
                critical_issues.append({
                    "type": "low_dimension_score",
                    "severity": "high" if score < 0.5 else "medium",
                    "description": f"{dim} dimension score ({score:.1%}) is below threshold",
                    "metric": dim,
                    "value": score
                })
        
        # Check for high failure rate
        pass_rate = aggregate_metrics.get("pass_rate", 1.0)
        if pass_rate < 0.7:
            critical_issues.append({
                "type": "high_failure_rate",
                "severity": "critical" if pass_rate < 0.5 else "high",
                "description": f"Pass rate ({pass_rate:.1%}) is below acceptable threshold (70%)",
                "metric": "pass_rate",
                "value": pass_rate
            })
        
        # Check for specific metric failures
        metric_scores = aggregate_metrics.get("metric_scores", {})
        critical_metrics = ["safety_score", "tool_accuracy", "answer_relevance"]
        for metric in critical_metrics:
            if metric in metric_scores and metric_scores[metric] < 0.7:
                critical_issues.append({
                    "type": "critical_metric_failure",
                    "severity": "high",
                    "description": f"Critical metric {metric} ({metric_scores[metric]:.1%}) is below threshold",
                    "metric": metric,
                    "value": metric_scores[metric]
                })
        
        return critical_issues
    
    def _create_critical_issues_chart(
        self, critical_issues: List[Dict[str, Any]]
    ) -> Chart:
        """Create chart highlighting critical issues."""
        # Group by severity
        severity_counts = defaultdict(int)
        for issue in critical_issues:
            severity_counts[issue["severity"]] += 1
        
        severities = ["critical", "high", "medium", "low"]
        counts = [severity_counts.get(sev, 0) for sev in severities]
        
        # Filter out zero counts
        filtered_data = [(sev, count) for sev, count in zip(severities, counts) if count > 0]
        if filtered_data:
            severities, counts = zip(*filtered_data)
        else:
            severities, counts = [], []
        
        colors = {
            "critical": "#D32F2F",
            "high": "#F44336",
            "medium": "#FF9800",
            "low": "#FFC107"
        }
        
        chart_colors = [colors[sev] for sev in severities]
        
        return Chart(
            title="Critical Issues",
            chart_type=ChartType.BAR,
            data={
                "severities": list(severities),
                "counts": list(counts)
            },
            labels=list(severities),
            colors=chart_colors,
            options={
                "show_values": True,
                "horizontal": False
            },
            description=f"{len(critical_issues)} critical issues detected"
        )
    
    def _create_trend_indicators_chart(
        self,
        results: List[EvaluationResult],
        aggregate_metrics: Dict[str, Any]
    ) -> Optional[Chart]:
        """
        Create chart showing trend indicators for key metrics.
        
        Args:
            results: List of evaluation results
            aggregate_metrics: Aggregate metrics
        
        Returns:
            Chart with trend indicators or None if trends cannot be calculated
        """
        try:
            # Analyze trends
            trend_report = self.trend_analyzer.analyze_metric_trends(results)
            
            # Get trends for key metrics
            key_metrics = ["tool_accuracy", "answer_relevance", "safety_score"]
            trends_data = []
            
            for metric in key_metrics:
                if metric in trend_report.metric_trends:
                    trend = trend_report.metric_trends[metric]
                    trends_data.append({
                        "metric": metric,
                        "direction": trend.trend_direction,
                        "change": trend.change_percentage
                    })
            
            if not trends_data:
                return None
            
            # Create chart data
            metrics = [t["metric"] for t in trends_data]
            changes = [t["change"] for t in trends_data]
            
            # Color code by direction
            colors = []
            for t in trends_data:
                if t["direction"] == "improving":
                    colors.append("#4CAF50")  # Green
                elif t["direction"] == "declining":
                    colors.append("#F44336")  # Red
                else:
                    colors.append("#9E9E9E")  # Gray
            
            return Chart(
                title="Metric Trends",
                chart_type=ChartType.BAR,
                data={
                    "metrics": metrics,
                    "changes": [round(c, 2) for c in changes]
                },
                labels=metrics,
                colors=colors,
                options={
                    "show_values": True,
                    "horizontal": True,
                    "show_zero_line": True
                },
                description="Percentage change in key metrics"
            )
        
        except Exception as e:
            logger.warning(f"Failed to create trend indicators chart: {e}")
            return None
    
    def _generate_executive_summary(
        self,
        results: List[EvaluationResult],
        aggregate_metrics: Dict[str, Any],
        critical_issues: List[Dict[str, Any]]
    ) -> str:
        """
        Generate executive summary text.
        
        Args:
            results: List of evaluation results
            aggregate_metrics: Aggregate metrics
            critical_issues: List of critical issues
        
        Returns:
            Executive summary string
        """
        overall_score = aggregate_metrics.get("overall_score", 0.0)
        pass_rate = aggregate_metrics.get("pass_rate", 0.0)
        total_evals = len(results)
        
        summary_parts = []
        
        # Overall performance
        summary_parts.append(
            f"Analyzed {total_evals} evaluations with overall score of {overall_score:.1%} "
            f"and pass rate of {pass_rate:.1%}."
        )
        
        # Critical issues
        if critical_issues:
            critical_count = sum(1 for i in critical_issues if i["severity"] == "critical")
            high_count = sum(1 for i in critical_issues if i["severity"] == "high")
            
            if critical_count > 0:
                summary_parts.append(
                    f"{critical_count} critical issue(s) detected requiring immediate attention."
                )
            if high_count > 0:
                summary_parts.append(
                    f"{high_count} high-priority issue(s) identified."
                )
        else:
            summary_parts.append("No critical issues detected.")
        
        # Dimension performance
        dimension_scores = aggregate_metrics.get("dimension_scores", {})
        if dimension_scores:
            best_dim = max(dimension_scores.items(), key=lambda x: x[1])
            worst_dim = min(dimension_scores.items(), key=lambda x: x[1])
            
            summary_parts.append(
                f"Best performing dimension: {best_dim[0]} ({best_dim[1]:.1%}). "
                f"Lowest performing: {worst_dim[0]} ({worst_dim[1]:.1%})."
            )
        
        return " ".join(summary_parts)
