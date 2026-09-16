# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression report generator for detecting and analyzing metric regressions."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from uaef.analysis.recommendations import RecommendationEngine
from uaef.analysis.root_cause import RootCauseAnalyzer
from uaef.experiments.comparison import ComparisonEngine, ComparisonReport, MetricComparison
from uaef.experiments.models import ExperimentRun
from uaef.reporting.models import Chart, ChartType, Dashboard

logger = logging.getLogger(__name__)


class RegressionReportGenerator:
    """
    Generator for detailed regression reports when metrics decline.
    
    Creates comprehensive regression dashboards showing:
    - All regressed metrics with before/after comparisons
    - Context about when regression occurred (run metadata, timestamps)
    - Root cause analysis for regressions
    - Severity classifications (critical, high, medium, low)
    - Impact on overall system performance
    - Actionable recommendations for fixing regressions
    - Historical regression tracking (if multiple runs provided)
    
    Integrates with ComparisonEngine for regression detection,
    RootCauseAnalyzer for root cause analysis, and RecommendationEngine
    for regression-specific recommendations.
    
    Severity Classification:
    - Critical: Regression > 20% or key metrics below 0.5
    - High: Regression 10-20% or important metrics below 0.7
    - Medium: Regression 5-10% or metrics below 0.85
    - Low: Regression < 5% or minor metrics affected
    
    Examples:
        >>> # Generate regression report from two runs
        >>> generator = RegressionReportGenerator()
        >>> dashboard = generator.generate_regression_dashboard(
        ...     baseline_run=baseline_run,
        ...     current_run=current_run,
        ...     title="Regression Report - v1 to v2"
        ... )
        >>> print(f"Regressions: {len(dashboard.metadata.get('regressions', []))}")
        
        >>> # Generate with custom thresholds
        >>> generator = RegressionReportGenerator(
        ...     regression_threshold=10.0,
        ...     critical_threshold=0.5
        ... )
        >>> dashboard = generator.generate_regression_dashboard(
        ...     baseline_run=baseline_run,
        ...     current_run=current_run,
        ...     title="Critical Regression Analysis"
        ... )
        
        >>> # Generate historical regression tracking
        >>> dashboard = generator.generate_historical_regression_dashboard(
        ...     baseline_run=baseline_run,
        ...     comparison_runs=[run1, run2, run3],
        ...     title="Regression Tracking - Last 4 Runs"
        ... )
    """
    
    def __init__(
        self,
        comparison_engine: Optional[ComparisonEngine] = None,
        root_cause_analyzer: Optional[RootCauseAnalyzer] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        regression_threshold: float = 5.0,
        critical_threshold: float = 0.5,
        high_threshold: float = 0.7,
        medium_threshold: float = 0.85
    ):
        """
        Initialize the regression report generator.
        
        Args:
            comparison_engine: Optional ComparisonEngine for regression detection
            root_cause_analyzer: Optional RootCauseAnalyzer for root cause analysis
            recommendation_engine: Optional RecommendationEngine for recommendations
            regression_threshold: Percentage threshold for regression detection (default: 5.0%)
            critical_threshold: Score threshold for critical severity (default: 0.5)
            high_threshold: Score threshold for high severity (default: 0.7)
            medium_threshold: Score threshold for medium severity (default: 0.85)
        """
        self.comparison_engine = comparison_engine or ComparisonEngine(
            regression_threshold=regression_threshold
        )
        self.root_cause_analyzer = root_cause_analyzer or RootCauseAnalyzer()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.regression_threshold = regression_threshold
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
    
    def generate_regression_dashboard(
        self,
        baseline_run: ExperimentRun,
        current_run: ExperimentRun,
        title: str,
        description: Optional[str] = None,
        include_root_cause: bool = True,
        include_recommendations: bool = True
    ) -> Dashboard:
        """
        Generate a regression dashboard comparing two experiment runs.
        
        Args:
            baseline_run: Baseline experiment run
            current_run: Current experiment run to check for regressions
            title: Dashboard title
            description: Optional dashboard description
            include_root_cause: Whether to include root cause analysis
            include_recommendations: Whether to include recommendations
        
        Returns:
            Dashboard with regression analysis charts
        
        Raises:
            ValueError: If title is empty
            ValueError: If baseline and current runs are the same
            ValueError: If no regressions detected
        
        Examples:
            >>> generator = RegressionReportGenerator()
            >>> dashboard = generator.generate_regression_dashboard(
            ...     baseline_run=baseline_run,
            ...     current_run=current_run,
            ...     title="Regression Report - v1 to v2"
            ... )
        """
        if not title or not title.strip():
            raise ValueError("Dashboard title cannot be empty")
        
        if baseline_run.run_id == current_run.run_id:
            raise ValueError("Cannot compare a run with itself")
        
        logger.info(
            f"Generating regression dashboard between runs {baseline_run.run_id} "
            f"and {current_run.run_id}"
        )
        
        # Generate comparison report
        comparison_report = self.comparison_engine.compare_runs(
            baseline_run=baseline_run,
            current_run=current_run
        )
        
        # Check if there are any regressions
        if not comparison_report.regressions:
            raise ValueError(
                f"No regressions detected between runs {baseline_run.run_id} "
                f"and {current_run.run_id}"
            )
        
        # Classify regressions by severity
        regression_data = self._classify_regressions(comparison_report)
        
        # Generate charts
        charts = []
        
        # Chart 1: Regression overview (bar chart showing all regressed metrics)
        charts.append(self._create_regression_overview_chart(regression_data))
        
        # Chart 2: Before/after comparison (grouped bar chart)
        charts.append(self._create_before_after_chart(regression_data))
        
        # Chart 3: Severity distribution (pie chart)
        charts.append(self._create_severity_distribution_chart(regression_data))
        
        # Chart 4: Impact analysis (heatmap showing affected dimensions)
        charts.append(self._create_impact_heatmap(regression_data))
        
        # Chart 5: Regression magnitude (bar chart showing percentage changes)
        charts.append(self._create_regression_magnitude_chart(regression_data))
        
        # Prepare metadata
        metadata = {
            "dashboard_type": "regression",
            "baseline_run_id": str(baseline_run.run_id),
            "current_run_id": str(current_run.run_id),
            "generated_at": datetime.utcnow().isoformat(),
            "regression_threshold": self.regression_threshold,
            "total_regressions": len(comparison_report.regressions),
            "critical_count": regression_data["severity_counts"]["critical"],
            "high_count": regression_data["severity_counts"]["high"],
            "medium_count": regression_data["severity_counts"]["medium"],
            "low_count": regression_data["severity_counts"]["low"],
            "regressions": regression_data["regressions"]
        }
        
        # Add context about when regression occurred
        metadata["baseline_context"] = {
            "run_name": baseline_run.run_name,
            "timestamp": baseline_run.timestamp.isoformat() if baseline_run.timestamp else None,
            "config_snapshot": baseline_run.config_snapshot
        }
        metadata["current_context"] = {
            "run_name": current_run.run_name,
            "timestamp": current_run.timestamp.isoformat() if current_run.timestamp else None,
            "config_snapshot": current_run.config_snapshot
        }
        
        # Add root cause analysis if enabled
        if include_root_cause:
            try:
                root_causes = self._analyze_root_causes(
                    regression_data, baseline_run, current_run
                )
                metadata["root_causes"] = root_causes
                
                # Add root cause breakdown chart
                if root_causes:
                    charts.append(self._create_root_cause_chart(root_causes))
            except Exception as e:
                logger.warning(f"Failed to generate root cause analysis: {e}")
        
        # Add recommendations if enabled
        if include_recommendations:
            try:
                recommendations = self._generate_regression_recommendations(
                    regression_data, comparison_report
                )
                metadata["recommendations"] = recommendations
            except Exception as e:
                logger.warning(f"Failed to generate recommendations: {e}")
        
        # Generate description if not provided
        if not description:
            description = self._generate_regression_summary(
                regression_data, baseline_run, current_run
            )
        
        # Create dashboard
        dashboard = Dashboard(
            title=title,
            description=description,
            charts=charts,
            metadata=metadata
        )
        
        logger.info(
            f"Generated regression dashboard with {len(charts)} charts and "
            f"{len(comparison_report.regressions)} regressions"
        )
        return dashboard
    
    def generate_historical_regression_dashboard(
        self,
        baseline_run: ExperimentRun,
        comparison_runs: List[ExperimentRun],
        title: str,
        description: Optional[str] = None
    ) -> Dashboard:
        """
        Generate dashboard tracking regressions across multiple runs.
        
        Args:
            baseline_run: Baseline experiment run
            comparison_runs: List of runs to track regressions across
            title: Dashboard title
            description: Optional dashboard description
        
        Returns:
            Dashboard with historical regression tracking
        
        Raises:
            ValueError: If comparison_runs is empty
            ValueError: If title is empty
        
        Examples:
            >>> generator = RegressionReportGenerator()
            >>> dashboard = generator.generate_historical_regression_dashboard(
            ...     baseline_run=baseline_run,
            ...     comparison_runs=[run1, run2, run3],
            ...     title="Regression Tracking - Last 4 Runs"
            ... )
        """
        if not comparison_runs:
            raise ValueError("comparison_runs cannot be empty")
        
        if not title or not title.strip():
            raise ValueError("Dashboard title cannot be empty")
        
        logger.info(
            f"Generating historical regression dashboard for {len(comparison_runs)} runs"
        )
        
        # Generate comparison reports for all runs
        all_reports = []
        for run in comparison_runs:
            try:
                report = self.comparison_engine.compare_runs(
                    baseline_run=baseline_run,
                    current_run=run
                )
                all_reports.append((run, report))
            except Exception as e:
                logger.warning(f"Failed to compare run {run.run_id}: {e}")
        
        if not all_reports:
            raise ValueError("Failed to generate any comparison reports")
        
        # Generate charts
        charts = []
        
        # Chart 1: Regression timeline (line chart showing regression count over time)
        charts.append(self._create_regression_timeline_chart(baseline_run, all_reports))
        
        # Chart 2: Metric regression heatmap (which metrics regressed in which runs)
        charts.append(self._create_metric_regression_heatmap(all_reports))
        
        # Chart 3: Severity trends (stacked area chart)
        charts.append(self._create_severity_trends_chart(all_reports))
        
        # Prepare metadata
        metadata = {
            "dashboard_type": "historical_regression",
            "baseline_run_id": str(baseline_run.run_id),
            "comparison_run_ids": [str(run.run_id) for run, _ in all_reports],
            "generated_at": datetime.utcnow().isoformat(),
            "run_count": len(all_reports),
            "total_regressions": sum(len(report.regressions) for _, report in all_reports)
        }
        
        # Generate description if not provided
        if not description:
            description = self._generate_historical_summary(baseline_run, all_reports)
        
        # Create dashboard
        dashboard = Dashboard(
            title=title,
            description=description,
            charts=charts,
            metadata=metadata
        )
        
        logger.info(f"Generated historical regression dashboard with {len(charts)} charts")
        return dashboard
    
    def _classify_regressions(
        self, comparison_report: ComparisonReport
    ) -> Dict[str, Any]:
        """
        Classify regressions by severity and extract detailed information.
        
        Severity levels:
        - Critical: Regression > 20% or current score < 0.5
        - High: Regression 10-20% or current score < 0.7
        - Medium: Regression 5-10% or current score < 0.85
        - Low: Regression < 5% or minor metrics
        
        Args:
            comparison_report: Comparison report with regressions
        
        Returns:
            Dictionary with classified regression data
        """
        regressions = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for comp in comparison_report.metric_comparisons:
            if not comp.is_regression:
                continue
            
            # Determine severity
            severity = self._determine_severity(comp)
            severity_counts[severity] += 1
            
            regressions.append({
                "metric_name": comp.metric_name,
                "baseline_score": comp.baseline_score,
                "current_score": comp.current_score,
                "delta": comp.delta,
                "percent_change": comp.percent_change,
                "severity": severity
            })
        
        # Sort by severity (critical first) then by magnitude
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        regressions.sort(
            key=lambda x: (severity_order[x["severity"]], abs(x["delta"])),
            reverse=True
        )
        
        return {
            "regressions": regressions,
            "severity_counts": severity_counts
        }
    
    def _determine_severity(self, comparison: MetricComparison) -> str:
        """
        Determine severity level for a regression.
        
        Args:
            comparison: Metric comparison with regression
        
        Returns:
            Severity level (critical, high, medium, low)
        """
        # Check if current score is critically low
        if comparison.current_score < self.critical_threshold:
            return "critical"
        
        # Check percentage change magnitude
        if abs(comparison.percent_change) > 20:
            return "critical"
        elif abs(comparison.percent_change) > 10:
            return "high"
        elif abs(comparison.percent_change) > 5:
            return "medium"
        
        # Check if current score is below thresholds
        if comparison.current_score < self.high_threshold:
            return "high"
        elif comparison.current_score < self.medium_threshold:
            return "medium"
        
        return "low"
    
    def _create_regression_overview_chart(
        self, regression_data: Dict[str, Any]
    ) -> Chart:
        """Create bar chart showing all regressed metrics."""
        regressions = regression_data["regressions"]
        
        if not regressions:
            return Chart(
                title="Regression Overview",
                chart_type=ChartType.BAR,
                data={"metrics": [], "deltas": []},
                labels=[],
                colors=[],
                description="No regressions detected"
            )
        
        # Take top 15 regressions by magnitude
        top_regressions = regressions[:15]
        
        metrics = [r["metric_name"] for r in top_regressions]
        deltas = [round(r["delta"], 3) for r in top_regressions]
        
        # Color code by severity
        severity_colors = {
            "critical": "#D32F2F",
            "high": "#F44336",
            "medium": "#FF9800",
            "low": "#FFC107"
        }
        colors = [severity_colors[r["severity"]] for r in top_regressions]
        
        return Chart(
            title="Regression Overview",
            chart_type=ChartType.BAR,
            data={
                "metrics": metrics,
                "deltas": deltas
            },
            labels=metrics,
            colors=colors,
            options={
                "show_values": True,
                "horizontal": True,
                "show_zero_line": True
            },
            description=f"Top {len(top_regressions)} regressed metrics by magnitude"
        )
    
    def _create_before_after_chart(
        self, regression_data: Dict[str, Any]
    ) -> Chart:
        """Create grouped bar chart showing before/after comparison."""
        regressions = regression_data["regressions"]
        
        if not regressions:
            return Chart(
                title="Before/After Comparison",
                chart_type=ChartType.BAR,
                data={"metrics": [], "baseline": [], "current": []},
                labels=[],
                colors=[],
                description="No regressions to compare"
            )
        
        # Take top 10 regressions
        top_regressions = regressions[:10]
        
        metrics = [r["metric_name"] for r in top_regressions]
        baseline_scores = [round(r["baseline_score"], 3) for r in top_regressions]
        current_scores = [round(r["current_score"], 3) for r in top_regressions]
        
        return Chart(
            title="Before/After Comparison",
            chart_type=ChartType.BAR,
            data={
                "metrics": metrics,
                "baseline": baseline_scores,
                "current": current_scores
            },
            labels=metrics,
            colors=["#2196F3", "#F44336"],
            options={
                "grouped": True,
                "horizontal": True,
                "show_values": True,
                "max_value": 1.0,
                "legend": ["Baseline", "Current"]
            },
            description=f"Before/after scores for top {len(top_regressions)} regressions"
        )
    
    def _create_severity_distribution_chart(
        self, regression_data: Dict[str, Any]
    ) -> Chart:
        """Create pie chart showing severity distribution."""
        severity_counts = regression_data["severity_counts"]
        
        # Filter out zero counts
        severities = []
        counts = []
        for severity in ["critical", "high", "medium", "low"]:
            count = severity_counts[severity]
            if count > 0:
                severities.append(severity)
                counts.append(count)
        
        if not severities:
            return Chart(
                title="Severity Distribution",
                chart_type=ChartType.PIE,
                data={"severities": [], "counts": []},
                labels=[],
                colors=[],
                description="No regressions to classify"
            )
        
        severity_colors = {
            "critical": "#D32F2F",
            "high": "#F44336",
            "medium": "#FF9800",
            "low": "#FFC107"
        }
        colors = [severity_colors[sev] for sev in severities]
        
        return Chart(
            title="Severity Distribution",
            chart_type=ChartType.PIE,
            data={
                "severities": severities,
                "counts": counts
            },
            labels=[f"{sev.capitalize()} ({count})" for sev, count in zip(severities, counts)],
            colors=colors,
            options={
                "show_percentages": True,
                "show_legend": True
            },
            description=f"Distribution of {sum(counts)} regressions by severity"
        )
    
    def _create_impact_heatmap(
        self, regression_data: Dict[str, Any]
    ) -> Chart:
        """Create heatmap showing affected dimensions."""
        regressions = regression_data["regressions"]
        
        if not regressions:
            return Chart(
                title="Impact Analysis by Dimension",
                chart_type=ChartType.HEATMAP,
                data={"matrix": [], "x_labels": [], "y_labels": []},
                labels=[],
                colors=[],
                description="No regressions to analyze"
            )
        
        # Group regressions by dimension
        dimension_regressions = defaultdict(lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0})
        
        for reg in regressions:
            # Determine dimension from metric name
            dimension = self._get_dimension_from_metric(reg["metric_name"])
            dimension_regressions[dimension][reg["severity"]] += 1
        
        # Build heatmap matrix
        dimensions = sorted(dimension_regressions.keys())
        severities = ["critical", "high", "medium", "low"]
        
        matrix = []
        for dimension in dimensions:
            row = [dimension_regressions[dimension][sev] for sev in severities]
            matrix.append(row)
        
        return Chart(
            title="Impact Analysis by Dimension",
            chart_type=ChartType.HEATMAP,
            data={
                "matrix": matrix,
                "x_labels": [s.capitalize() for s in severities],
                "y_labels": dimensions
            },
            labels=dimensions,
            colors=["#FFFFFF", "#F44336"],
            options={
                "show_values": True,
                "color_scale": "Reds"
            },
            description="Regression severity distribution across dimensions"
        )
    
    def _create_regression_magnitude_chart(
        self, regression_data: Dict[str, Any]
    ) -> Chart:
        """Create bar chart showing percentage changes."""
        regressions = regression_data["regressions"]
        
        if not regressions:
            return Chart(
                title="Regression Magnitude (% Change)",
                chart_type=ChartType.BAR,
                data={"metrics": [], "percent_changes": []},
                labels=[],
                colors=[],
                description="No regressions to display"
            )
        
        # Take top 15 regressions by percentage change
        top_regressions = sorted(
            regressions,
            key=lambda x: abs(x["percent_change"]),
            reverse=True
        )[:15]
        
        metrics = [r["metric_name"] for r in top_regressions]
        percent_changes = [round(r["percent_change"], 2) for r in top_regressions]
        
        # Color code by severity
        severity_colors = {
            "critical": "#D32F2F",
            "high": "#F44336",
            "medium": "#FF9800",
            "low": "#FFC107"
        }
        colors = [severity_colors[r["severity"]] for r in top_regressions]
        
        return Chart(
            title="Regression Magnitude (% Change)",
            chart_type=ChartType.BAR,
            data={
                "metrics": metrics,
                "percent_changes": percent_changes
            },
            labels=metrics,
            colors=colors,
            options={
                "show_values": True,
                "horizontal": True,
                "show_zero_line": True,
                "value_suffix": "%"
            },
            description=f"Percentage change for top {len(top_regressions)} regressions"
        )
    
    def _create_root_cause_chart(
        self, root_causes: List[Dict[str, Any]]
    ) -> Chart:
        """Create bar chart showing root cause breakdown."""
        if not root_causes:
            return Chart(
                title="Root Cause Analysis",
                chart_type=ChartType.BAR,
                data={"causes": [], "counts": []},
                labels=[],
                colors=[],
                description="No root causes identified"
            )
        
        # Count occurrences of each root cause category
        cause_counts = defaultdict(int)
        for cause in root_causes:
            cause_counts[cause["category"]] += 1
        
        # Sort by count
        sorted_causes = sorted(
            cause_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        categories = [c[0] for c in sorted_causes]
        counts = [c[1] for c in sorted_causes]
        
        # Color code by category
        category_colors = {
            "tool_selection": "#FF5722",
            "response": "#2196F3",
            "context": "#4CAF50",
            "reasoning": "#9C27B0",
            "performance": "#FF9800",
            "safety": "#F44336"
        }
        colors = [category_colors.get(cat, "#9E9E9E") for cat in categories]
        
        return Chart(
            title="Root Cause Breakdown",
            chart_type=ChartType.BAR,
            data={
                "categories": categories,
                "counts": counts
            },
            labels=categories,
            colors=colors,
            options={
                "show_values": True,
                "horizontal": False
            },
            description=f"Root cause distribution across {len(root_causes)} regressions"
        )
    
    def _create_regression_timeline_chart(
        self,
        baseline_run: ExperimentRun,
        all_reports: List[Tuple[ExperimentRun, ComparisonReport]]
    ) -> Chart:
        """Create line chart showing regression count over time."""
        if not all_reports:
            return Chart(
                title="Regression Timeline",
                chart_type=ChartType.LINE,
                data={"runs": [], "regression_counts": []},
                labels=[],
                colors=[],
                description="No data available"
            )
        
        # Sort by run timestamp
        sorted_reports = sorted(
            all_reports,
            key=lambda x: x[0].timestamp if x[0].timestamp else datetime.min
        )
        
        run_labels = [f"Run {i+1}" for i in range(len(sorted_reports))]
        regression_counts = [len(report.regressions) for _, report in sorted_reports]
        
        return Chart(
            title="Regression Timeline",
            chart_type=ChartType.LINE,
            data={
                "runs": run_labels,
                "regression_counts": regression_counts
            },
            labels=run_labels,
            colors=["#F44336"],
            options={
                "show_points": True,
                "show_legend": False
            },
            description=f"Regression count across {len(sorted_reports)} runs"
        )
    
    def _create_metric_regression_heatmap(
        self,
        all_reports: List[Tuple[ExperimentRun, ComparisonReport]]
    ) -> Chart:
        """Create heatmap showing which metrics regressed in which runs."""
        if not all_reports:
            return Chart(
                title="Metric Regression Heatmap",
                chart_type=ChartType.HEATMAP,
                data={"matrix": [], "x_labels": [], "y_labels": []},
                labels=[],
                colors=[],
                description="No data available"
            )
        
        # Collect all metrics that regressed in any run
        all_regressed_metrics = set()
        for _, report in all_reports:
            all_regressed_metrics.update(report.regressions)
        
        if not all_regressed_metrics:
            return Chart(
                title="Metric Regression Heatmap",
                chart_type=ChartType.HEATMAP,
                data={"matrix": [], "x_labels": [], "y_labels": []},
                labels=[],
                colors=[],
                description="No regressions detected across runs"
            )
        
        # Sort metrics alphabetically
        metrics = sorted(all_regressed_metrics)
        
        # Build matrix (1 if metric regressed in run, 0 otherwise)
        run_labels = [f"Run {i+1}" for i in range(len(all_reports))]
        matrix = []
        
        for metric in metrics:
            row = []
            for _, report in all_reports:
                row.append(1 if metric in report.regressions else 0)
            matrix.append(row)
        
        return Chart(
            title="Metric Regression Heatmap",
            chart_type=ChartType.HEATMAP,
            data={
                "matrix": matrix,
                "x_labels": run_labels,
                "y_labels": metrics
            },
            labels=metrics,
            colors=["#FFFFFF", "#F44336"],
            options={
                "show_values": False,
                "color_scale": "Reds"
            },
            description="Which metrics regressed in which runs"
        )
    
    def _create_severity_trends_chart(
        self,
        all_reports: List[Tuple[ExperimentRun, ComparisonReport]]
    ) -> Chart:
        """Create stacked area chart showing severity trends."""
        if not all_reports:
            return Chart(
                title="Severity Trends",
                chart_type=ChartType.BAR,
                data={"runs": [], "critical": [], "high": [], "medium": [], "low": []},
                labels=[],
                colors=[],
                description="No data available"
            )
        
        # Sort by run timestamp
        sorted_reports = sorted(
            all_reports,
            key=lambda x: x[0].timestamp if x[0].timestamp else datetime.min
        )
        
        run_labels = [f"Run {i+1}" for i in range(len(sorted_reports))]
        
        # Count regressions by severity for each run
        critical_counts = []
        high_counts = []
        medium_counts = []
        low_counts = []
        
        for _, report in sorted_reports:
            # Classify regressions for this run
            regression_data = self._classify_regressions(report)
            severity_counts = regression_data["severity_counts"]
            
            critical_counts.append(severity_counts["critical"])
            high_counts.append(severity_counts["high"])
            medium_counts.append(severity_counts["medium"])
            low_counts.append(severity_counts["low"])
        
        return Chart(
            title="Severity Trends",
            chart_type=ChartType.BAR,
            data={
                "runs": run_labels,
                "critical": critical_counts,
                "high": high_counts,
                "medium": medium_counts,
                "low": low_counts
            },
            labels=run_labels,
            colors=["#D32F2F", "#F44336", "#FF9800", "#FFC107"],
            options={
                "stacked": True,
                "show_legend": True,
                "legend": ["Critical", "High", "Medium", "Low"]
            },
            description="Regression severity distribution over time"
        )
    
    def _analyze_root_causes(
        self,
        regression_data: Dict[str, Any],
        baseline_run: ExperimentRun,
        current_run: ExperimentRun
    ) -> List[Dict[str, Any]]:
        """
        Analyze root causes for regressions.
        
        Args:
            regression_data: Classified regression data
            baseline_run: Baseline run
            current_run: Current run
        
        Returns:
            List of root cause analyses
        """
        root_causes = []
        
        for reg in regression_data["regressions"]:
            # Determine category from metric name
            category = self._get_dimension_from_metric(reg["metric_name"])
            
            # Generate root cause description
            description = self._generate_root_cause_description(reg, category)
            
            root_causes.append({
                "metric_name": reg["metric_name"],
                "category": category,
                "severity": reg["severity"],
                "description": description,
                "delta": reg["delta"],
                "percent_change": reg["percent_change"]
            })
        
        return root_causes
    
    def _generate_regression_recommendations(
        self,
        regression_data: Dict[str, Any],
        comparison_report: ComparisonReport
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations for fixing regressions.
        
        Args:
            regression_data: Classified regression data
            comparison_report: Comparison report
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Group regressions by category
        category_regressions = defaultdict(list)
        for reg in regression_data["regressions"]:
            category = self._get_dimension_from_metric(reg["metric_name"])
            category_regressions[category].append(reg)
        
        # Generate recommendations per category
        for category, regs in category_regressions.items():
            # Count critical and high severity regressions
            critical_count = sum(1 for r in regs if r["severity"] == "critical")
            high_count = sum(1 for r in regs if r["severity"] == "high")
            
            if critical_count > 0 or high_count > 0:
                priority = "critical" if critical_count > 0 else "high"
                
                recommendation = {
                    "category": category,
                    "priority": priority,
                    "title": f"Address {category} regressions",
                    "description": self._generate_category_recommendation(category, regs),
                    "affected_metrics": [r["metric_name"] for r in regs],
                    "severity_breakdown": {
                        "critical": critical_count,
                        "high": high_count,
                        "medium": sum(1 for r in regs if r["severity"] == "medium"),
                        "low": sum(1 for r in regs if r["severity"] == "low")
                    }
                }
                recommendations.append(recommendation)
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])
        
        return recommendations
    
    def _get_dimension_from_metric(self, metric_name: str) -> str:
        """
        Determine dimension from metric name.
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Dimension name
        """
        metric_lower = metric_name.lower()
        
        if any(prefix in metric_lower for prefix in ["tool_", "mcp_"]):
            return "Tool Calling"
        elif any(prefix in metric_lower for prefix in ["answer_", "completeness", "hallucination", "accuracy"]):
            return "Response Quality"
        elif any(prefix in metric_lower for prefix in ["safety_", "bias_", "toxicity", "prompt_injection"]):
            return "Responsible AI"
        elif any(prefix in metric_lower for prefix in ["latency_", "token_", "cost_", "throughput"]):
            return "Performance"
        elif any(prefix in metric_lower for prefix in ["context_", "coherence", "conversation_", "turn_"]):
            return "Multi-Turn"
        elif any(prefix in metric_lower for prefix in ["agent_", "delegation_", "workflow_", "coordination_"]):
            return "Multi-Agent"
        elif any(prefix in metric_lower for prefix in ["chain_of_thought", "logical_", "reasoning_", "fallacy"]):
            return "Reasoning"
        else:
            return "Other"
    
    def _generate_root_cause_description(
        self, regression: Dict[str, Any], category: str
    ) -> str:
        """
        Generate human-readable root cause description.
        
        Args:
            regression: Regression data
            category: Dimension category
        
        Returns:
            Root cause description
        """
        metric_name = regression["metric_name"]
        percent_change = regression["percent_change"]
        
        descriptions = {
            "Tool Calling": f"{metric_name} decreased by {abs(percent_change):.1f}%, indicating issues with tool selection or usage",
            "Response Quality": f"{metric_name} decreased by {abs(percent_change):.1f}%, suggesting response quality degradation",
            "Responsible AI": f"{metric_name} decreased by {abs(percent_change):.1f}%, indicating potential safety or bias concerns",
            "Performance": f"{metric_name} decreased by {abs(percent_change):.1f}%, showing performance degradation",
            "Multi-Turn": f"{metric_name} decreased by {abs(percent_change):.1f}%, indicating context retention or coherence issues",
            "Multi-Agent": f"{metric_name} decreased by {abs(percent_change):.1f}%, suggesting coordination or delegation problems",
            "Reasoning": f"{metric_name} decreased by {abs(percent_change):.1f}%, indicating reasoning quality degradation"
        }
        
        return descriptions.get(
            category,
            f"{metric_name} decreased by {abs(percent_change):.1f}%"
        )
    
    def _generate_category_recommendation(
        self, category: str, regressions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate category-specific recommendation.
        
        Args:
            category: Dimension category
            regressions: List of regressions in this category
        
        Returns:
            Recommendation description
        """
        count = len(regressions)
        avg_change = sum(abs(r["percent_change"]) for r in regressions) / count
        
        recommendations = {
            "Tool Calling": (
                f"{count} tool calling metric(s) regressed by an average of {avg_change:.1f}%. "
                "Review tool selection logic, tool descriptions, and usage examples in the system prompt. "
                "Consider adding more tool usage examples or simplifying tool interfaces."
            ),
            "Response Quality": (
                f"{count} response quality metric(s) regressed by an average of {avg_change:.1f}%. "
                "Review response generation prompts, context handling, and answer completeness. "
                "Consider improving retrieval quality or adding more response examples."
            ),
            "Responsible AI": (
                f"{count} safety/bias metric(s) regressed by an average of {avg_change:.1f}%. "
                "Review safety guardrails, bias detection, and content filtering. "
                "Consider strengthening safety prompts or adding additional safety checks."
            ),
            "Performance": (
                f"{count} performance metric(s) regressed by an average of {avg_change:.1f}%. "
                "Review latency, token usage, and cost efficiency. "
                "Consider optimizing prompts, reducing unnecessary tool calls, or using more efficient models."
            ),
            "Multi-Turn": (
                f"{count} multi-turn metric(s) regressed by an average of {avg_change:.1f}%. "
                "Review context retention, conversation coherence, and goal tracking. "
                "Consider improving context management or conversation state handling."
            ),
            "Multi-Agent": (
                f"{count} multi-agent metric(s) regressed by an average of {avg_change:.1f}%. "
                "Review agent coordination, task delegation, and workflow management. "
                "Consider improving inter-agent communication or task routing logic."
            ),
            "Reasoning": (
                f"{count} reasoning metric(s) regressed by an average of {avg_change:.1f}%. "
                "Review chain-of-thought prompts, logical consistency, and reasoning steps. "
                "Consider adding more reasoning examples or improving step-by-step guidance."
            )
        }
        
        return recommendations.get(
            category,
            f"{count} metric(s) in {category} regressed by an average of {avg_change:.1f}%. Review and address these issues."
        )
    
    def _generate_regression_summary(
        self,
        regression_data: Dict[str, Any],
        baseline_run: ExperimentRun,
        current_run: ExperimentRun
    ) -> str:
        """
        Generate executive summary for regression report.
        
        Args:
            regression_data: Classified regression data
            baseline_run: Baseline run
            current_run: Current run
        
        Returns:
            Summary string
        """
        total_regressions = len(regression_data["regressions"])
        severity_counts = regression_data["severity_counts"]
        
        summary_parts = []
        
        # Overall summary
        summary_parts.append(
            f"Detected {total_regressions} regression(s) comparing {baseline_run.run_name} "
            f"(baseline) to {current_run.run_name} (current)."
        )
        
        # Severity breakdown
        if severity_counts["critical"] > 0:
            summary_parts.append(
                f"{severity_counts['critical']} critical regression(s) requiring immediate attention."
            )
        if severity_counts["high"] > 0:
            summary_parts.append(
                f"{severity_counts['high']} high-priority regression(s) identified."
            )
        if severity_counts["medium"] > 0:
            summary_parts.append(
                f"{severity_counts['medium']} medium-priority regression(s) detected."
            )
        if severity_counts["low"] > 0:
            summary_parts.append(
                f"{severity_counts['low']} low-priority regression(s) noted."
            )
        
        # Top regressions
        if regression_data["regressions"]:
            top_3 = regression_data["regressions"][:3]
            top_metrics = [r["metric_name"] for r in top_3]
            summary_parts.append(
                f"Most affected metrics: {', '.join(top_metrics)}."
            )
        
        return " ".join(summary_parts)
    
    def _generate_historical_summary(
        self,
        baseline_run: ExperimentRun,
        all_reports: List[Tuple[ExperimentRun, ComparisonReport]]
    ) -> str:
        """
        Generate summary for historical regression tracking.
        
        Args:
            baseline_run: Baseline run
            all_reports: List of (run, report) tuples
        
        Returns:
            Summary string
        """
        total_runs = len(all_reports)
        total_regressions = sum(len(report.regressions) for _, report in all_reports)
        
        # Count runs with regressions
        runs_with_regressions = sum(1 for _, report in all_reports if report.regressions)
        
        # Find most commonly regressed metrics
        all_regressed_metrics = []
        for _, report in all_reports:
            all_regressed_metrics.extend(report.regressions)
        
        if all_regressed_metrics:
            from collections import Counter
            metric_counts = Counter(all_regressed_metrics)
            top_3_metrics = [m[0] for m in metric_counts.most_common(3)]
            
            summary = (
                f"Tracked regressions across {total_runs} runs compared to baseline {baseline_run.run_name}. "
                f"Detected {total_regressions} total regressions across {runs_with_regressions} runs. "
                f"Most frequently regressed metrics: {', '.join(top_3_metrics)}."
            )
        else:
            summary = (
                f"Tracked regressions across {total_runs} runs compared to baseline {baseline_run.run_name}. "
                f"No regressions detected in any run."
            )
        
        return summary
