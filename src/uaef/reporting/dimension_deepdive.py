# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dimension deep-dive dashboard generator for detailed dimension analysis."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from uaef.analysis.clustering import FailureClusterer
from uaef.analysis.recommendations import RecommendationEngine
from uaef.analysis.trends import TrendAnalyzer
from uaef.models.evaluation_result import EvaluationResult
from uaef.reporting.models import Chart, ChartType, Dashboard

logger = logging.getLogger(__name__)


# Dimension definitions with metric prefixes
DIMENSION_DEFINITIONS = {
    "tool_calling": {
        "name": "Tool Calling",
        "prefixes": ["tool_", "mcp_"],
        "description": "Tool selection, usage, and MCP compliance"
    },
    "response_quality": {
        "name": "Response Quality",
        "prefixes": ["answer_", "completeness", "hallucination", "accuracy"],
        "description": "Answer relevance, completeness, and accuracy"
    },
    "responsible_ai": {
        "name": "Responsible AI",
        "prefixes": ["safety_", "bias_", "toxicity", "prompt_injection"],
        "description": "Safety, bias, and ethical considerations"
    },
    "performance": {
        "name": "Performance",
        "prefixes": ["latency_", "token_", "cost_", "throughput"],
        "description": "Latency, token efficiency, and cost"
    },
    "multi_turn": {
        "name": "Multi-Turn",
        "prefixes": ["context_", "coherence", "conversation_", "turn_"],
        "description": "Context retention and conversation flow"
    },
    "multi_agent": {
        "name": "Multi-Agent",
        "prefixes": ["agent_", "delegation_", "workflow_", "coordination_"],
        "description": "Agent coordination and workflow"
    },
    "reasoning": {
        "name": "Reasoning",
        "prefixes": ["chain_of_thought", "logical_", "reasoning_", "fallacy"],
        "description": "Logical consistency and reasoning quality"
    }
}


class DimensionDashboardGenerator:
    """
    Generator for dimension deep-dive dashboards.
    
    Creates detailed dashboards for individual evaluation dimensions:
    - All metrics within the dimension with score distributions
    - Metric correlation heatmap within dimension
    - Top failures by metric with examples
    - Metric trends over time (if historical data available)
    - Dimension-specific recommendations
    - Failure pattern analysis for the dimension
    
    Integrates with TrendAnalyzer, FailureClusterer, and RecommendationEngine
    for comprehensive dimension analysis.
    
    Examples:
        >>> # Generate dashboard for tool calling dimension
        >>> generator = DimensionDashboardGenerator()
        >>> dashboard = generator.generate_dashboard(
        ...     results=results,
        ...     dimension="tool_calling",
        ...     title="Tool Calling Deep Dive"
        ... )
        >>> print(f"Dashboard: {dashboard.title}")
        >>> print(f"Charts: {len(dashboard.charts)}")
        
        >>> # Generate with trend analysis
        >>> trend_analyzer = TrendAnalyzer()
        >>> generator = DimensionDashboardGenerator(trend_analyzer=trend_analyzer)
        >>> dashboard = generator.generate_dashboard(
        ...     results=results,
        ...     dimension="response_quality",
        ...     title="Response Quality Analysis",
        ...     include_trends=True
        ... )
        
        >>> # Generate with recommendations
        >>> rec_engine = RecommendationEngine()
        >>> generator = DimensionDashboardGenerator(recommendation_engine=rec_engine)
        >>> dashboard = generator.generate_dashboard(
        ...     results=results,
        ...     dimension="responsible_ai",
        ...     title="Safety Analysis",
        ...     include_recommendations=True
        ... )
    """
    
    def __init__(
        self,
        trend_analyzer: Optional[TrendAnalyzer] = None,
        failure_clusterer: Optional[FailureClusterer] = None,
        recommendation_engine: Optional[RecommendationEngine] = None
    ):
        """
        Initialize the dimension dashboard generator.
        
        Args:
            trend_analyzer: Optional TrendAnalyzer for trend indicators
            failure_clusterer: Optional FailureClusterer for failure analysis
            recommendation_engine: Optional RecommendationEngine for insights
        """
        self.trend_analyzer = trend_analyzer or TrendAnalyzer()
        self.failure_clusterer = failure_clusterer
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
    
    def generate_dashboard(
        self,
        results: List[EvaluationResult],
        dimension: str,
        title: str,
        run_id: Optional[UUID] = None,
        experiment_id: Optional[UUID] = None,
        include_trends: bool = True,
        include_recommendations: bool = True,
        description: Optional[str] = None
    ) -> Dashboard:
        """
        Generate a dimension deep-dive dashboard from evaluation results.
        
        Args:
            results: List of evaluation results to analyze
            dimension: Dimension to analyze (e.g., "tool_calling", "response_quality")
            title: Dashboard title
            run_id: Optional experiment run ID
            experiment_id: Optional experiment ID
            include_trends: Whether to include trend indicators
            include_recommendations: Whether to include recommendations
            description: Optional dashboard description
        
        Returns:
            Dashboard with dimension deep-dive charts
        
        Raises:
            ValueError: If results list is empty
            ValueError: If title is empty
            ValueError: If dimension is invalid
        
        Examples:
            >>> generator = DimensionDashboardGenerator()
            >>> results = [result1, result2, result3]
            >>> dashboard = generator.generate_dashboard(
            ...     results=results,
            ...     dimension="tool_calling",
            ...     title="Tool Calling Deep Dive"
            ... )
        """
        if not results:
            raise ValueError("Cannot generate dashboard from empty results list")
        
        if not title or not title.strip():
            raise ValueError("Dashboard title cannot be empty")
        
        if dimension not in DIMENSION_DEFINITIONS:
            raise ValueError(
                f"Invalid dimension: {dimension}. "
                f"Valid dimensions: {', '.join(DIMENSION_DEFINITIONS.keys())}"
            )
        
        logger.info(
            f"Generating dimension deep-dive dashboard for {dimension} "
            f"with {len(results)} evaluation results"
        )
        
        # Get dimension definition
        dim_def = DIMENSION_DEFINITIONS[dimension]
        
        # Extract dimension metrics
        dimension_data = self._extract_dimension_data(results, dimension)
        
        if not dimension_data["metrics"]:
            raise ValueError(
                f"No metrics found for dimension {dimension} in evaluation results"
            )
        
        # Generate charts
        charts = []
        
        # Chart 1: Metric score distribution
        charts.append(self._create_metric_distribution_chart(dimension_data, dim_def))
        
        # Chart 2: Score distribution histogram/box plot
        charts.append(self._create_score_distribution_chart(dimension_data, dim_def))
        
        # Chart 3: Metric correlation heatmap
        if len(dimension_data["metrics"]) > 1:
            corr_chart = self._create_correlation_heatmap(dimension_data, dim_def)
            if corr_chart:
                charts.append(corr_chart)
        
        # Chart 4: Top failures by metric
        failure_chart = self._create_top_failures_chart(dimension_data, results, dim_def)
        if failure_chart:
            charts.append(failure_chart)
        
        # Chart 5: Metric trends over time (if enabled and sufficient data)
        if include_trends and len(results) > 2:
            try:
                trend_chart = self._create_metric_trends_chart(
                    results, dimension, dim_def
                )
                if trend_chart:
                    charts.append(trend_chart)
            except Exception as e:
                logger.warning(f"Failed to generate trend chart: {e}")
        
        # Chart 6: Failure patterns (if clusterer available)
        if self.failure_clusterer and dimension_data["failed_count"] > 0:
            try:
                pattern_chart = self._create_failure_patterns_chart(
                    results, dimension, dim_def
                )
                if pattern_chart:
                    charts.append(pattern_chart)
            except Exception as e:
                logger.warning(f"Failed to generate failure patterns chart: {e}")
        
        # Prepare metadata
        metadata = {
            "dashboard_type": "dimension_deep_dive",
            "dimension": dimension,
            "dimension_name": dim_def["name"],
            "evaluation_count": len(results),
            "metric_count": len(dimension_data["metrics"]),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        if run_id:
            metadata["run_id"] = str(run_id)
        if experiment_id:
            metadata["experiment_id"] = str(experiment_id)
        
        # Add dimension statistics
        metadata["dimension_statistics"] = {
            "average_score": dimension_data["avg_score"],
            "min_score": dimension_data["min_score"],
            "max_score": dimension_data["max_score"],
            "std_dev": dimension_data["std_dev"],
            "pass_rate": dimension_data["pass_rate"],
            "failed_count": dimension_data["failed_count"]
        }
        
        # Add recommendations if enabled
        if include_recommendations and dimension_data["failed_count"] > 0:
            try:
                rec_report = self.recommendation_engine.generate_recommendations(results)
                # Filter recommendations for this dimension
                dim_recommendations = [
                    {
                        "title": rec.title,
                        "priority": rec.priority,
                        "category": rec.category,
                        "description": rec.description
                    }
                    for rec in rec_report.recommendations
                    if any(
                        metric in rec.affected_metrics
                        for metric in dimension_data["metrics"]
                    )
                ][:3]  # Top 3
                
                if dim_recommendations:
                    metadata["recommendations"] = dim_recommendations
            except Exception as e:
                logger.warning(f"Failed to generate recommendations: {e}")
        
        # Generate description if not provided
        if not description:
            description = self._generate_dimension_summary(
                dimension_data, dim_def, results
            )
        
        # Create dashboard
        dashboard = Dashboard(
            title=title,
            description=description,
            charts=charts,
            metadata=metadata
        )
        
        logger.info(
            f"Generated dimension deep-dive dashboard with {len(charts)} charts "
            f"for {dimension}"
        )
        return dashboard
    
    def _extract_dimension_data(
        self,
        results: List[EvaluationResult],
        dimension: str
    ) -> Dict[str, Any]:
        """
        Extract all data for a specific dimension from evaluation results.
        
        Args:
            results: List of evaluation results
            dimension: Dimension to extract
        
        Returns:
            Dictionary with dimension data including metrics, scores, and statistics
        """
        dim_def = DIMENSION_DEFINITIONS[dimension]
        prefixes = dim_def["prefixes"]
        
        # Collect all metrics and scores for this dimension
        metric_scores = defaultdict(list)
        dimension_scores = []
        failed_count = 0
        
        for result in results:
            # Find dimension result
            dim_result = None
            for dr in result.dimension_results:
                if dr.dimension_name.lower() == dimension.lower():
                    dim_result = dr
                    break
            
            if dim_result:
                dimension_scores.append(dim_result.aggregate_score)
                
                # Check if dimension failed
                if dim_result.aggregate_score < 0.7:  # Common threshold
                    failed_count += 1
                
                # Collect metric scores
                for metric_score in dim_result.metric_scores:
                    # Check if metric belongs to this dimension
                    if any(
                        metric_score.metric_name.startswith(prefix)
                        for prefix in prefixes
                    ):
                        metric_scores[metric_score.metric_name].append({
                            "score": metric_score.score,
                            "evaluation_id": str(result.evaluation_id),
                            "timestamp": result.timestamp,
                            "reasoning": metric_score.reasoning
                        })
        
        # Calculate statistics
        if dimension_scores:
            avg_score = sum(dimension_scores) / len(dimension_scores)
            min_score = min(dimension_scores)
            max_score = max(dimension_scores)
            
            # Calculate standard deviation
            variance = sum((s - avg_score) ** 2 for s in dimension_scores) / len(dimension_scores)
            std_dev = variance ** 0.5
            
            pass_rate = (len(dimension_scores) - failed_count) / len(dimension_scores)
        else:
            avg_score = 0.0
            min_score = 0.0
            max_score = 0.0
            std_dev = 0.0
            pass_rate = 0.0
        
        return {
            "metrics": list(metric_scores.keys()),
            "metric_scores": dict(metric_scores),
            "dimension_scores": dimension_scores,
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "std_dev": std_dev,
            "pass_rate": pass_rate,
            "failed_count": failed_count
        }
    
    def _create_metric_distribution_chart(
        self,
        dimension_data: Dict[str, Any],
        dim_def: Dict[str, str]
    ) -> Chart:
        """Create bar chart showing average scores for all metrics in dimension."""
        metric_scores = dimension_data["metric_scores"]
        
        # Calculate average score for each metric
        metrics = []
        avg_scores = []
        
        for metric, scores in metric_scores.items():
            metrics.append(metric)
            avg_score = sum(s["score"] for s in scores) / len(scores)
            avg_scores.append(round(avg_score, 3))
        
        # Sort by score (lowest first to highlight issues)
        sorted_data = sorted(zip(metrics, avg_scores), key=lambda x: x[1])
        metrics, avg_scores = zip(*sorted_data) if sorted_data else ([], [])
        
        # Color code based on score
        colors = []
        for score in avg_scores:
            if score >= 0.8:
                colors.append("#4CAF50")  # Green
            elif score >= 0.6:
                colors.append("#FF9800")  # Orange
            else:
                colors.append("#F44336")  # Red
        
        return Chart(
            title=f"{dim_def['name']} - Metric Scores",
            chart_type=ChartType.BAR,
            data={
                "metrics": list(metrics),
                "scores": list(avg_scores)
            },
            labels=list(metrics),
            colors=list(colors),
            options={
                "show_values": True,
                "horizontal": True,
                "max_value": 1.0
            },
            description=f"Average scores for {len(metrics)} metrics in {dim_def['name']} dimension"
        )
    
    def _create_score_distribution_chart(
        self,
        dimension_data: Dict[str, Any],
        dim_def: Dict[str, str]
    ) -> Chart:
        """Create box plot or histogram showing score distribution."""
        metric_scores = dimension_data["metric_scores"]
        
        # Collect all scores for each metric
        metrics = []
        score_distributions = []
        
        for metric, scores in metric_scores.items():
            metrics.append(metric)
            score_values = [s["score"] for s in scores]
            
            # Calculate quartiles for box plot
            sorted_scores = sorted(score_values)
            n = len(sorted_scores)
            
            q1_idx = n // 4
            q2_idx = n // 2
            q3_idx = (3 * n) // 4
            
            distribution = {
                "min": min(sorted_scores),
                "q1": sorted_scores[q1_idx] if n > 0 else 0.0,
                "median": sorted_scores[q2_idx] if n > 0 else 0.0,
                "q3": sorted_scores[q3_idx] if n > 0 else 0.0,
                "max": max(sorted_scores)
            }
            score_distributions.append(distribution)
        
        return Chart(
            title=f"{dim_def['name']} - Score Distribution",
            chart_type=ChartType.BOX,
            data={
                "metrics": metrics,
                "distributions": score_distributions
            },
            labels=metrics,
            colors=["#2196F3"],
            options={
                "show_outliers": True,
                "horizontal": False
            },
            description=f"Score distribution across {len(metrics)} metrics showing quartiles and outliers"
        )
    
    def _create_correlation_heatmap(
        self,
        dimension_data: Dict[str, Any],
        dim_def: Dict[str, str]
    ) -> Optional[Chart]:
        """Create heatmap showing metric correlations within dimension."""
        try:
            metric_scores = dimension_data["metric_scores"]
            metrics = list(metric_scores.keys())
            
            if len(metrics) < 2:
                return None
            
            # Build correlation matrix
            # For each pair of metrics, calculate correlation
            correlation_matrix = []
            
            for metric1 in metrics:
                row = []
                scores1 = [s["score"] for s in metric_scores[metric1]]
                
                for metric2 in metrics:
                    scores2 = [s["score"] for s in metric_scores[metric2]]
                    
                    # Ensure same length (should be same evaluations)
                    min_len = min(len(scores1), len(scores2))
                    scores1_aligned = scores1[:min_len]
                    scores2_aligned = scores2[:min_len]
                    
                    # Calculate Pearson correlation
                    if len(scores1_aligned) > 1:
                        mean1 = sum(scores1_aligned) / len(scores1_aligned)
                        mean2 = sum(scores2_aligned) / len(scores2_aligned)
                        
                        numerator = sum(
                            (s1 - mean1) * (s2 - mean2)
                            for s1, s2 in zip(scores1_aligned, scores2_aligned)
                        )
                        
                        denom1 = sum((s1 - mean1) ** 2 for s1 in scores1_aligned) ** 0.5
                        denom2 = sum((s2 - mean2) ** 2 for s2 in scores2_aligned) ** 0.5
                        
                        if denom1 > 0 and denom2 > 0:
                            correlation = numerator / (denom1 * denom2)
                        else:
                            correlation = 0.0
                    else:
                        correlation = 0.0
                    
                    row.append(round(correlation, 3))
                
                correlation_matrix.append(row)
            
            return Chart(
                title=f"{dim_def['name']} - Metric Correlations",
                chart_type=ChartType.HEATMAP,
                data={
                    "matrix": correlation_matrix,
                    "x_labels": metrics,
                    "y_labels": metrics
                },
                labels=metrics,
                colors=["#FFFFFF", "#2196F3"],
                options={
                    "show_values": True,
                    "color_scale": "Blues",
                    "min_value": -1.0,
                    "max_value": 1.0
                },
                description=f"Correlation matrix showing relationships between {len(metrics)} metrics"
            )
        
        except Exception as e:
            logger.warning(f"Failed to create correlation heatmap: {e}")
            return None
    
    def _create_top_failures_chart(
        self,
        dimension_data: Dict[str, Any],
        results: List[EvaluationResult],
        dim_def: Dict[str, str]
    ) -> Optional[Chart]:
        """Create chart showing top failures by metric with examples."""
        metric_scores = dimension_data["metric_scores"]
        
        # Count failures per metric (score < 0.7)
        failure_counts = {}
        failure_examples = {}
        
        for metric, scores in metric_scores.items():
            failures = [s for s in scores if s["score"] < 0.7]
            if failures:
                failure_counts[metric] = len(failures)
                # Get worst example
                worst = min(failures, key=lambda x: x["score"])
                failure_examples[metric] = {
                    "score": worst["score"],
                    "reasoning": worst.get("reasoning", "No reasoning provided")[:100]
                }
        
        if not failure_counts:
            return None
        
        # Sort by failure count
        sorted_failures = sorted(
            failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        metrics = [m[0] for m in sorted_failures]
        counts = [m[1] for m in sorted_failures]
        
        # Color code by severity
        colors = []
        for metric in metrics:
            avg_score = sum(s["score"] for s in metric_scores[metric]) / len(metric_scores[metric])
            if avg_score < 0.5:
                colors.append("#D32F2F")  # Dark red
            elif avg_score < 0.6:
                colors.append("#F44336")  # Red
            else:
                colors.append("#FF9800")  # Orange
        
        return Chart(
            title=f"{dim_def['name']} - Top Failures by Metric",
            chart_type=ChartType.BAR,
            data={
                "metrics": metrics,
                "failure_counts": counts,
                "examples": [failure_examples.get(m, {}) for m in metrics]
            },
            labels=metrics,
            colors=colors,
            options={
                "show_values": True,
                "horizontal": True
            },
            description=f"Top {len(metrics)} metrics by failure count with example failures"
        )
    
    def _create_metric_trends_chart(
        self,
        results: List[EvaluationResult],
        dimension: str,
        dim_def: Dict[str, str]
    ) -> Optional[Chart]:
        """Create line chart showing metric trends over time."""
        try:
            # Analyze trends for dimension metrics
            trend_report = self.trend_analyzer.analyze_metric_trends(results)
            
            # Filter to dimension metrics
            dim_prefixes = dim_def["prefixes"]
            dim_trends = {
                metric: trend
                for metric, trend in trend_report.metric_trends.items()
                if any(metric.startswith(prefix) for prefix in dim_prefixes)
            }
            
            if not dim_trends:
                return None
            
            # Select top 5 metrics by variance (most interesting trends)
            metric_variances = {
                metric: trend.max_value - trend.min_value
                for metric, trend in dim_trends.items()
            }
            
            top_metrics = sorted(
                metric_variances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            selected_metrics = [m[0] for m in top_metrics]
            
            # Build trend data
            # Assume all metrics have same timestamps (from same evaluations)
            if selected_metrics:
                first_trend = dim_trends[selected_metrics[0]]
                timestamps = [t.isoformat() for t in first_trend.timestamps]
                
                metric_data = {}
                for metric in selected_metrics:
                    trend = dim_trends[metric]
                    metric_data[metric] = [round(v, 3) for v in trend.values]
                
                # Generate colors based on trend direction
                colors = []
                for metric in selected_metrics:
                    trend = dim_trends[metric]
                    if trend.trend_direction == "improving":
                        colors.append("#4CAF50")  # Green
                    elif trend.trend_direction == "declining":
                        colors.append("#F44336")  # Red
                    else:
                        colors.append("#2196F3")  # Blue
                
                return Chart(
                    title=f"{dim_def['name']} - Metric Trends Over Time",
                    chart_type=ChartType.LINE,
                    data={
                        "timestamps": timestamps,
                        "metrics": metric_data
                    },
                    labels=timestamps,
                    colors=colors,
                    options={
                        "show_points": True,
                        "show_legend": True,
                        "legend": selected_metrics
                    },
                    description=f"Trends for {len(selected_metrics)} key metrics over time"
                )
            
            return None
        
        except Exception as e:
            logger.warning(f"Failed to create metric trends chart: {e}")
            return None
    
    def _create_failure_patterns_chart(
        self,
        results: List[EvaluationResult],
        dimension: str,
        dim_def: Dict[str, str]
    ) -> Optional[Chart]:
        """Create chart showing failure patterns for dimension."""
        try:
            # Cluster failures
            clusters = self.failure_clusterer.cluster_failures(
                results,
                method="hierarchical"
            )
            
            # Filter to clusters relevant to this dimension
            dim_prefixes = dim_def["prefixes"]
            dim_clusters = [
                cluster for cluster in clusters
                if any(
                    any(metric.startswith(prefix) for prefix in dim_prefixes)
                    for metric in cluster.common_metrics
                )
            ]
            
            if not dim_clusters:
                return None
            
            # Sort by size
            dim_clusters.sort(key=lambda c: c.size, reverse=True)
            dim_clusters = dim_clusters[:5]  # Top 5
            
            # Build chart data
            cluster_labels = [f"Pattern {c.cluster_id + 1}" for c in dim_clusters]
            cluster_sizes = [c.size for c in dim_clusters]
            cluster_summaries = [c.summary[:50] + "..." if len(c.summary) > 50 else c.summary for c in dim_clusters]
            
            # Color by severity
            severity_colors = {
                "critical": "#D32F2F",
                "high": "#F44336",
                "medium": "#FF9800",
                "low": "#FFC107"
            }
            colors = [severity_colors.get(c.severity, "#9E9E9E") for c in dim_clusters]
            
            return Chart(
                title=f"{dim_def['name']} - Failure Patterns",
                chart_type=ChartType.BAR,
                data={
                    "patterns": cluster_labels,
                    "sizes": cluster_sizes,
                    "summaries": cluster_summaries
                },
                labels=cluster_labels,
                colors=colors,
                options={
                    "show_values": True,
                    "horizontal": False
                },
                description=f"{len(dim_clusters)} common failure patterns identified in {dim_def['name']}"
            )
        
        except Exception as e:
            logger.warning(f"Failed to create failure patterns chart: {e}")
            return None
    
    def _generate_dimension_summary(
        self,
        dimension_data: Dict[str, Any],
        dim_def: Dict[str, str],
        results: List[EvaluationResult]
    ) -> str:
        """
        Generate summary text for dimension dashboard.
        
        Args:
            dimension_data: Extracted dimension data
            dim_def: Dimension definition
            results: Evaluation results
        
        Returns:
            Summary string
        """
        avg_score = dimension_data["avg_score"]
        pass_rate = dimension_data["pass_rate"]
        metric_count = len(dimension_data["metrics"])
        failed_count = dimension_data["failed_count"]
        
        summary_parts = []
        
        # Overall performance
        summary_parts.append(
            f"Analyzed {len(results)} evaluations for {dim_def['name']} dimension "
            f"with average score of {avg_score:.1%} and pass rate of {pass_rate:.1%}."
        )
        
        # Failures
        if failed_count > 0:
            summary_parts.append(
                f"{failed_count} evaluation(s) failed in this dimension."
            )
        else:
            summary_parts.append("No failures detected in this dimension.")
        
        # Metric performance
        metric_scores = dimension_data["metric_scores"]
        if metric_scores:
            # Find best and worst metrics
            avg_by_metric = {
                metric: sum(s["score"] for s in scores) / len(scores)
                for metric, scores in metric_scores.items()
            }
            
            best_metric = max(avg_by_metric.items(), key=lambda x: x[1])
            worst_metric = min(avg_by_metric.items(), key=lambda x: x[1])
            
            summary_parts.append(
                f"Best performing metric: {best_metric[0]} ({best_metric[1]:.1%}). "
                f"Lowest performing: {worst_metric[0]} ({worst_metric[1]:.1%})."
            )
        
        return " ".join(summary_parts)

