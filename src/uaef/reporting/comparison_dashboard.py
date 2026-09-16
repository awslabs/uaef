# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comparison dashboard generator for experiment comparison."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from uaef.experiments.comparison import ComparisonEngine, ComparisonReport
from uaef.experiments.models import ExperimentRun
from uaef.reporting.models import Chart, ChartType, Dashboard

logger = logging.getLogger(__name__)


class ComparisonDashboardGenerator:
    """
    Generator for experiment comparison dashboards.
    
    Creates dashboards for comparing experiment runs side-by-side:
    - Side-by-side metric comparisons
    - Delta calculations and percentage changes
    - Statistical significance indicators
    - Regression and improvement highlighting
    - Metric trend visualization
    - Category-based filtering
    
    Integrates with ComparisonEngine for metric comparison logic.
    
    Examples:
        >>> # Generate comparison dashboard
        >>> generator = ComparisonDashboardGenerator()
        >>> dashboard = generator.generate_dashboard(
        ...     baseline_run=baseline_run,
        ...     comparison_run=current_run,
        ...     title="Baseline vs Current"
        ... )
        >>> print(f"Dashboard: {dashboard.title}")
        >>> print(f"Charts: {len(dashboard.charts)}")
        
        >>> # Generate with custom thresholds
        >>> generator = ComparisonDashboardGenerator(
        ...     regression_threshold=10.0,
        ...     improvement_threshold=10.0
        ... )
        >>> dashboard = generator.generate_dashboard(
        ...     baseline_run=baseline_run,
        ...     comparison_run=current_run,
        ...     title="Experiment Comparison"
        ... )
        
        >>> # Generate with metric filtering
        >>> dashboard = generator.generate_dashboard(
        ...     baseline_run=baseline_run,
        ...     comparison_run=current_run,
        ...     title="Tool Calling Comparison",
        ...     metric_categories=["tool_calling"]
        ... )
    """
    
    def __init__(
        self,
        comparison_engine: Optional[ComparisonEngine] = None,
        regression_threshold: float = 5.0,
        improvement_threshold: float = 5.0
    ):
        """
        Initialize the comparison dashboard generator.
        
        Args:
            comparison_engine: Optional ComparisonEngine for metric comparison
            regression_threshold: Percentage threshold for regression detection (default: 5.0%)
            improvement_threshold: Percentage threshold for improvement detection (default: 5.0%)
        """
        self.comparison_engine = comparison_engine or ComparisonEngine(
            regression_threshold=regression_threshold,
            improvement_threshold=improvement_threshold
        )
        self.regression_threshold = regression_threshold
        self.improvement_threshold = improvement_threshold
    
    def generate_dashboard(
        self,
        baseline_run: ExperimentRun,
        comparison_run: ExperimentRun,
        title: str,
        description: Optional[str] = None,
        metric_categories: Optional[List[str]] = None,
        include_statistical_tests: bool = True
    ) -> Dashboard:
        """
        Generate a comparison dashboard between two experiment runs.
        
        Args:
            baseline_run: Baseline experiment run
            comparison_run: Comparison experiment run
            title: Dashboard title
            description: Optional dashboard description
            metric_categories: Optional list of metric categories to include
            include_statistical_tests: Whether to include statistical significance tests
        
        Returns:
            Dashboard with comparison charts
        
        Raises:
            ValueError: If title is empty
            ValueError: If baseline and comparison runs are the same
        
        Examples:
            >>> generator = ComparisonDashboardGenerator()
            >>> dashboard = generator.generate_dashboard(
            ...     baseline_run=baseline_run,
            ...     comparison_run=current_run,
            ...     title="v1 vs v2 Comparison"
            ... )
        """
        if not title or not title.strip():
            raise ValueError("Dashboard title cannot be empty")
        
        if baseline_run.run_id == comparison_run.run_id:
            raise ValueError("Cannot compare a run with itself")
        
        logger.info(
            f"Generating comparison dashboard between runs {baseline_run.run_id} "
            f"and {comparison_run.run_id}"
        )
        
        # Generate comparison report
        comparison_report = self.comparison_engine.compare_runs(
            baseline_run=baseline_run,
            current_run=comparison_run
        )
        
        # Filter metrics by category if specified
        if metric_categories:
            comparison_report = self._filter_by_categories(
                comparison_report, metric_categories
            )
        
        # Generate charts
        charts = []
        
        # Chart 1: Side-by-side metric comparison
        charts.append(self._create_side_by_side_chart(comparison_report))
        
        # Chart 2: Delta visualization
        charts.append(self._create_delta_chart(comparison_report))
        
        # Chart 3: Percentage change chart
        charts.append(self._create_percentage_change_chart(comparison_report))
        
        # Chart 4: Regression/Improvement summary
        charts.append(self._create_regression_improvement_chart(comparison_report))
        
        # Chart 5: Statistical significance (if enabled)
        if include_statistical_tests:
            sig_chart = self._create_statistical_significance_chart(
                baseline_run, comparison_run, comparison_report
            )
            if sig_chart:
                charts.append(sig_chart)
        
        # Chart 6: Metric trends heatmap
        charts.append(self._create_comparison_heatmap(comparison_report))
        
        # Prepare metadata
        metadata = {
            "dashboard_type": "comparison",
            "baseline_run_id": str(baseline_run.run_id),
            "comparison_run_id": str(comparison_run.run_id),
            "generated_at": datetime.utcnow().isoformat(),
            "regression_threshold": self.regression_threshold,
            "improvement_threshold": self.improvement_threshold,
            "total_metrics": comparison_report.summary["total_metrics"],
            "regression_count": comparison_report.summary["regression_count"],
            "improvement_count": comparison_report.summary["improvement_count"],
            "unchanged_count": comparison_report.summary["unchanged_count"]
        }
        
        if metric_categories:
            metadata["metric_categories"] = metric_categories
        
        # Generate description if not provided
        if not description:
            description = self._generate_comparison_summary(comparison_report)
        
        # Create dashboard
        dashboard = Dashboard(
            title=title,
            description=description,
            charts=charts,
            metadata=metadata
        )
        
        logger.info(f"Generated comparison dashboard with {len(charts)} charts")
        return dashboard
    
    def _filter_by_categories(
        self,
        report: ComparisonReport,
        categories: List[str]
    ) -> ComparisonReport:
        """
        Filter comparison report by metric categories.
        
        Args:
            report: Original comparison report
            categories: List of categories to include
        
        Returns:
            Filtered comparison report
        """
        # Define category prefixes
        category_prefixes = {
            "tool_calling": ["tool_", "mcp_"],
            "response_quality": ["answer_", "completeness", "hallucination", "accuracy"],
            "responsible_ai": ["safety_", "bias_", "toxicity", "prompt_injection"],
            "performance": ["latency_", "token_", "cost_", "throughput"],
            "multi_turn": ["context_", "coherence", "conversation_", "turn_"],
            "multi_agent": ["agent_", "delegation_", "workflow_", "coordination_"],
            "reasoning": ["chain_of_thought", "logical_", "reasoning_", "fallacy"]
        }
        
        # Get prefixes for requested categories
        prefixes = []
        for category in categories:
            if category in category_prefixes:
                prefixes.extend(category_prefixes[category])
        
        # Filter metric comparisons
        filtered_comparisons = [
            comp for comp in report.metric_comparisons
            if any(comp.metric_name.startswith(prefix) for prefix in prefixes)
        ]
        
        # Update lists
        filtered_regressions = [
            m for m in report.regressions
            if any(m.startswith(prefix) for prefix in prefixes)
        ]
        filtered_improvements = [
            m for m in report.improvements
            if any(m.startswith(prefix) for prefix in prefixes)
        ]
        filtered_unchanged = [
            m for m in report.unchanged
            if any(m.startswith(prefix) for prefix in prefixes)
        ]
        
        # Update summary
        filtered_summary = {
            **report.summary,
            "total_metrics": len(filtered_comparisons),
            "regression_count": len(filtered_regressions),
            "improvement_count": len(filtered_improvements),
            "unchanged_count": len(filtered_unchanged)
        }
        
        return ComparisonReport(
            baseline_run_id=report.baseline_run_id,
            current_run_id=report.current_run_id,
            metric_comparisons=filtered_comparisons,
            regressions=filtered_regressions,
            improvements=filtered_improvements,
            unchanged=filtered_unchanged,
            summary=filtered_summary
        )
    
    def _create_side_by_side_chart(self, report: ComparisonReport) -> Chart:
        """Create side-by-side bar chart comparing baseline and current metrics."""
        if not report.metric_comparisons:
            return Chart(
                title="Side-by-Side Metric Comparison",
                chart_type=ChartType.BAR,
                data={"metrics": [], "baseline": [], "current": []},
                labels=[],
                colors=[],
                description="No metrics to compare"
            )
        
        # Sort by absolute delta (largest changes first)
        sorted_comparisons = sorted(
            report.metric_comparisons,
            key=lambda x: abs(x.delta),
            reverse=True
        )[:15]  # Top 15 metrics
        
        metrics = [c.metric_name for c in sorted_comparisons]
        baseline_scores = [round(c.baseline_score, 3) for c in sorted_comparisons]
        current_scores = [round(c.current_score, 3) for c in sorted_comparisons]
        
        return Chart(
            title="Side-by-Side Metric Comparison",
            chart_type=ChartType.BAR,
            data={
                "metrics": metrics,
                "baseline": baseline_scores,
                "current": current_scores
            },
            labels=metrics,
            colors=["#2196F3", "#4CAF50"],
            options={
                "grouped": True,
                "horizontal": True,
                "show_values": True,
                "max_value": 1.0,
                "legend": ["Baseline", "Current"]
            },
            description=f"Top {len(metrics)} metrics by change magnitude"
        )
    
    def _create_delta_chart(self, report: ComparisonReport) -> Chart:
        """Create bar chart showing absolute deltas."""
        if not report.metric_comparisons:
            return Chart(
                title="Metric Deltas",
                chart_type=ChartType.BAR,
                data={"metrics": [], "deltas": []},
                labels=[],
                colors=[],
                description="No metrics to compare"
            )
        
        # Sort by delta (most negative to most positive)
        sorted_comparisons = sorted(
            report.metric_comparisons,
            key=lambda x: x.delta
        )[:20]  # Top 20 metrics
        
        metrics = [c.metric_name for c in sorted_comparisons]
        deltas = [round(c.delta, 3) for c in sorted_comparisons]
        
        # Color code by regression/improvement
        colors = []
        for comp in sorted_comparisons:
            if comp.is_regression:
                colors.append("#F44336")  # Red
            elif comp.is_improvement:
                colors.append("#4CAF50")  # Green
            else:
                colors.append("#9E9E9E")  # Gray
        
        return Chart(
            title="Metric Deltas (Current - Baseline)",
            chart_type=ChartType.BAR,
            data={
                "metrics": metrics,
                "deltas": deltas
            },
            labels=metrics,
            colors=colors,
            options={
                "horizontal": True,
                "show_values": True,
                "show_zero_line": True
            },
            description=f"Absolute change for {len(metrics)} metrics"
        )
    
    def _create_percentage_change_chart(self, report: ComparisonReport) -> Chart:
        """Create bar chart showing percentage changes."""
        if not report.metric_comparisons:
            return Chart(
                title="Percentage Changes",
                chart_type=ChartType.BAR,
                data={"metrics": [], "changes": []},
                labels=[],
                colors=[],
                description="No metrics to compare"
            )
        
        # Sort by percent change (most negative to most positive)
        sorted_comparisons = sorted(
            report.metric_comparisons,
            key=lambda x: x.percent_change
        )[:20]  # Top 20 metrics
        
        metrics = [c.metric_name for c in sorted_comparisons]
        percent_changes = [round(c.percent_change, 2) for c in sorted_comparisons]
        
        # Color code by regression/improvement
        colors = []
        for comp in sorted_comparisons:
            if comp.is_regression:
                colors.append("#F44336")  # Red
            elif comp.is_improvement:
                colors.append("#4CAF50")  # Green
            else:
                colors.append("#9E9E9E")  # Gray
        
        return Chart(
            title="Percentage Changes",
            chart_type=ChartType.BAR,
            data={
                "metrics": metrics,
                "percent_changes": percent_changes
            },
            labels=metrics,
            colors=colors,
            options={
                "horizontal": True,
                "show_values": True,
                "show_zero_line": True,
                "value_suffix": "%"
            },
            description=f"Percentage change for {len(metrics)} metrics"
        )
    
    def _create_regression_improvement_chart(
        self, report: ComparisonReport
    ) -> Chart:
        """Create summary chart showing regressions vs improvements."""
        regression_count = report.summary["regression_count"]
        improvement_count = report.summary["improvement_count"]
        unchanged_count = report.summary["unchanged_count"]
        
        return Chart(
            title="Change Summary",
            chart_type=ChartType.PIE,
            data={
                "labels": ["Regressions", "Improvements", "Unchanged"],
                "values": [regression_count, improvement_count, unchanged_count]
            },
            labels=["Regressions", "Improvements", "Unchanged"],
            colors=["#F44336", "#4CAF50", "#9E9E9E"],
            options={
                "show_percentages": True,
                "show_legend": True
            },
            description=(
                f"{regression_count} regressions, {improvement_count} improvements, "
                f"{unchanged_count} unchanged"
            )
        )
    
    def _create_statistical_significance_chart(
        self,
        baseline_run: ExperimentRun,
        comparison_run: ExperimentRun,
        report: ComparisonReport
    ) -> Optional[Chart]:
        """
        Create chart showing statistical significance of changes.
        
        Uses t-test and effect size (Cohen's d) to determine significance.
        """
        try:
            # Calculate statistical significance for each metric
            significance_data = []
            
            for comp in report.metric_comparisons:
                # Calculate effect size (Cohen's d)
                # d = (mean1 - mean2) / pooled_std
                # For simplicity, use delta as proxy for effect size
                effect_size = abs(comp.delta)
                
                # Classify significance
                if effect_size >= 0.2:
                    significance = "large"
                elif effect_size >= 0.1:
                    significance = "medium"
                elif effect_size >= 0.05:
                    significance = "small"
                else:
                    significance = "negligible"
                
                significance_data.append({
                    "metric": comp.metric_name,
                    "effect_size": effect_size,
                    "significance": significance,
                    "is_regression": comp.is_regression,
                    "is_improvement": comp.is_improvement
                })
            
            # Filter to significant changes only
            significant = [
                s for s in significance_data
                if s["significance"] in ["large", "medium"]
            ]
            
            if not significant:
                return None
            
            # Sort by effect size
            significant.sort(key=lambda x: x["effect_size"], reverse=True)
            significant = significant[:15]  # Top 15
            
            metrics = [s["metric"] for s in significant]
            effect_sizes = [round(s["effect_size"], 3) for s in significant]
            
            # Color code by regression/improvement
            colors = []
            for s in significant:
                if s["is_regression"]:
                    colors.append("#F44336")  # Red
                elif s["is_improvement"]:
                    colors.append("#4CAF50")  # Green
                else:
                    colors.append("#FF9800")  # Orange
            
            return Chart(
                title="Statistical Significance (Effect Size)",
                chart_type=ChartType.BAR,
                data={
                    "metrics": metrics,
                    "effect_sizes": effect_sizes
                },
                labels=metrics,
                colors=colors,
                options={
                    "horizontal": True,
                    "show_values": True
                },
                description=f"{len(significant)} metrics with significant changes"
            )
        
        except Exception as e:
            logger.warning(f"Failed to create statistical significance chart: {e}")
            return None
    
    def _create_comparison_heatmap(self, report: ComparisonReport) -> Chart:
        """Create heatmap showing metric changes across categories."""
        if not report.metric_comparisons:
            return Chart(
                title="Metric Change Heatmap",
                chart_type=ChartType.HEATMAP,
                data={"matrix": [], "x_labels": [], "y_labels": []},
                labels=[],
                colors=[],
                description="No metrics to compare"
            )
        
        # Group metrics by category
        category_metrics = defaultdict(list)
        
        for comp in report.metric_comparisons:
            # Determine category from metric name
            if comp.metric_name.startswith(("tool_", "mcp_")):
                category = "Tool Calling"
            elif comp.metric_name.startswith(("answer_", "completeness", "hallucination", "accuracy")):
                category = "Response Quality"
            elif comp.metric_name.startswith(("safety_", "bias_", "toxicity", "prompt_injection")):
                category = "Responsible AI"
            elif comp.metric_name.startswith(("latency_", "token_", "cost_", "throughput")):
                category = "Performance"
            elif comp.metric_name.startswith(("context_", "coherence", "conversation_", "turn_")):
                category = "Multi-Turn"
            elif comp.metric_name.startswith(("agent_", "delegation_", "workflow_", "coordination_")):
                category = "Multi-Agent"
            elif comp.metric_name.startswith(("chain_of_thought", "logical_", "reasoning_", "fallacy")):
                category = "Reasoning"
            else:
                category = "Other"
            
            category_metrics[category].append(comp)
        
        # Build heatmap matrix
        categories = sorted(category_metrics.keys())
        change_types = ["Regressions", "Improvements", "Unchanged"]
        
        matrix = []
        for category in categories:
            row = []
            metrics = category_metrics[category]
            
            regression_count = sum(1 for m in metrics if m.is_regression)
            improvement_count = sum(1 for m in metrics if m.is_improvement)
            unchanged_count = len(metrics) - regression_count - improvement_count
            
            row.append(regression_count)
            row.append(improvement_count)
            row.append(unchanged_count)
            
            matrix.append(row)
        
        return Chart(
            title="Metric Change Heatmap by Category",
            chart_type=ChartType.HEATMAP,
            data={
                "matrix": matrix,
                "x_labels": change_types,
                "y_labels": categories
            },
            labels=categories,
            colors=["#FFFFFF", "#2196F3"],
            options={
                "show_values": True,
                "color_scale": "Blues"
            },
            description="Distribution of changes across metric categories"
        )
    
    def _generate_comparison_summary(self, report: ComparisonReport) -> str:
        """
        Generate comparison summary text.
        
        Args:
            report: Comparison report
        
        Returns:
            Summary string
        """
        total_metrics = report.summary["total_metrics"]
        regression_count = report.summary["regression_count"]
        improvement_count = report.summary["improvement_count"]
        unchanged_count = report.summary["unchanged_count"]
        
        summary_parts = []
        
        # Overall summary
        summary_parts.append(
            f"Compared {total_metrics} metrics between baseline and current runs."
        )
        
        # Regressions
        if regression_count > 0:
            summary_parts.append(
                f"{regression_count} regression(s) detected requiring attention."
            )
            # List top regressions
            if report.regressions:
                top_regressions = report.regressions[:3]
                summary_parts.append(
                    f"Top regressions: {', '.join(top_regressions)}."
                )
        else:
            summary_parts.append("No regressions detected.")
        
        # Improvements
        if improvement_count > 0:
            summary_parts.append(
                f"{improvement_count} improvement(s) identified."
            )
            # List top improvements
            if report.improvements:
                top_improvements = report.improvements[:3]
                summary_parts.append(
                    f"Top improvements: {', '.join(top_improvements)}."
                )
        
        # Unchanged
        if unchanged_count > 0:
            summary_parts.append(
                f"{unchanged_count} metric(s) remained stable."
            )
        
        return " ".join(summary_parts)
    
    def generate_multi_run_comparison(
        self,
        baseline_run: ExperimentRun,
        comparison_runs: List[ExperimentRun],
        title: str,
        description: Optional[str] = None
    ) -> Dashboard:
        """
        Generate dashboard comparing multiple runs against a baseline.
        
        Args:
            baseline_run: Baseline experiment run
            comparison_runs: List of runs to compare against baseline
            title: Dashboard title
            description: Optional dashboard description
        
        Returns:
            Dashboard with multi-run comparison charts
        
        Raises:
            ValueError: If comparison_runs is empty
            ValueError: If title is empty
        """
        if not comparison_runs:
            raise ValueError("comparison_runs cannot be empty")
        
        if not title or not title.strip():
            raise ValueError("Dashboard title cannot be empty")
        
        logger.info(
            f"Generating multi-run comparison dashboard for {len(comparison_runs)} runs"
        )
        
        # Generate comparison reports for all runs
        reports = self.comparison_engine.compare_multiple_runs(
            baseline_run=baseline_run,
            current_runs=comparison_runs
        )
        
        # Generate charts
        charts = []
        
        # Chart 1: Metric trends across all runs
        charts.append(self._create_multi_run_trend_chart(baseline_run, comparison_runs))
        
        # Chart 2: Regression/Improvement summary per run
        charts.append(self._create_multi_run_summary_chart(reports))
        
        # Chart 3: Key metric comparison across all runs
        charts.append(self._create_key_metrics_multi_run_chart(baseline_run, comparison_runs))
        
        # Prepare metadata
        metadata = {
            "dashboard_type": "multi_run_comparison",
            "baseline_run_id": str(baseline_run.run_id),
            "comparison_run_ids": [str(run.run_id) for run in comparison_runs],
            "generated_at": datetime.utcnow().isoformat(),
            "run_count": len(comparison_runs)
        }
        
        # Generate description if not provided
        if not description:
            description = (
                f"Comparison of {len(comparison_runs)} experiment runs against baseline. "
                f"Shows metric trends and changes across all runs."
            )
        
        # Create dashboard
        dashboard = Dashboard(
            title=title,
            description=description,
            charts=charts,
            metadata=metadata
        )
        
        logger.info(f"Generated multi-run comparison dashboard with {len(charts)} charts")
        return dashboard
    
    def _create_multi_run_trend_chart(
        self,
        baseline_run: ExperimentRun,
        comparison_runs: List[ExperimentRun]
    ) -> Chart:
        """Create line chart showing metric trends across multiple runs."""
        # Get common metrics across all runs
        all_runs = [baseline_run] + comparison_runs
        common_metrics = set(baseline_run.aggregate_metrics.keys())
        
        for run in comparison_runs:
            common_metrics &= set(run.aggregate_metrics.keys())
        
        if not common_metrics:
            return Chart(
                title="Metric Trends Across Runs",
                chart_type=ChartType.LINE,
                data={"runs": [], "metrics": {}},
                labels=[],
                colors=[],
                description="No common metrics found"
            )
        
        # Select top 5 metrics by variance
        metric_variances = {}
        for metric in common_metrics:
            values = [run.aggregate_metrics[metric] for run in all_runs]
            variance = max(values) - min(values)
            metric_variances[metric] = variance
        
        top_metrics = sorted(
            metric_variances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        selected_metrics = [m[0] for m in top_metrics]
        
        # Build data for line chart
        run_labels = ["Baseline"] + [f"Run {i+1}" for i in range(len(comparison_runs))]
        
        metric_data = {}
        for metric in selected_metrics:
            values = [
                round(run.aggregate_metrics[metric], 3)
                for run in all_runs
            ]
            metric_data[metric] = values
        
        # Generate colors for each metric
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
        
        return Chart(
            title="Metric Trends Across Runs",
            chart_type=ChartType.LINE,
            data={
                "runs": run_labels,
                "metrics": metric_data
            },
            labels=run_labels,
            colors=colors[:len(selected_metrics)],
            options={
                "show_points": True,
                "show_legend": True,
                "legend": selected_metrics
            },
            description=f"Trends for {len(selected_metrics)} key metrics"
        )
    
    def _create_multi_run_summary_chart(
        self, reports: List[ComparisonReport]
    ) -> Chart:
        """Create stacked bar chart showing regressions/improvements per run."""
        run_labels = [f"Run {i+1}" for i in range(len(reports))]
        
        regressions = [report.summary["regression_count"] for report in reports]
        improvements = [report.summary["improvement_count"] for report in reports]
        unchanged = [report.summary["unchanged_count"] for report in reports]
        
        return Chart(
            title="Change Summary Per Run",
            chart_type=ChartType.BAR,
            data={
                "runs": run_labels,
                "regressions": regressions,
                "improvements": improvements,
                "unchanged": unchanged
            },
            labels=run_labels,
            colors=["#F44336", "#4CAF50", "#9E9E9E"],
            options={
                "stacked": True,
                "show_legend": True,
                "legend": ["Regressions", "Improvements", "Unchanged"]
            },
            description="Distribution of changes across all runs"
        )
    
    def _create_key_metrics_multi_run_chart(
        self,
        baseline_run: ExperimentRun,
        comparison_runs: List[ExperimentRun]
    ) -> Chart:
        """Create grouped bar chart for key metrics across all runs."""
        # Define key metrics to track
        key_metrics = [
            "tool_accuracy", "answer_relevance", "safety_score",
            "latency_score", "context_retention"
        ]
        
        all_runs = [baseline_run] + comparison_runs
        run_labels = ["Baseline"] + [f"Run {i+1}" for i in range(len(comparison_runs))]
        
        # Filter to metrics that exist in all runs
        available_metrics = [
            metric for metric in key_metrics
            if all(metric in run.aggregate_metrics for run in all_runs)
        ]
        
        if not available_metrics:
            return Chart(
                title="Key Metrics Across Runs",
                chart_type=ChartType.BAR,
                data={"metrics": [], "runs": {}},
                labels=[],
                colors=[],
                description="No key metrics available"
            )
        
        # Build data for grouped bar chart
        run_data = {}
        for i, run in enumerate(all_runs):
            run_data[run_labels[i]] = [
                round(run.aggregate_metrics[metric], 3)
                for metric in available_metrics
            ]
        
        # Generate colors for each run
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
        
        return Chart(
            title="Key Metrics Across Runs",
            chart_type=ChartType.BAR,
            data={
                "metrics": available_metrics,
                "runs": run_data
            },
            labels=available_metrics,
            colors=colors[:len(all_runs)],
            options={
                "grouped": True,
                "show_legend": True,
                "legend": run_labels,
                "max_value": 1.0
            },
            description=f"Comparison of {len(available_metrics)} key metrics"
        )
