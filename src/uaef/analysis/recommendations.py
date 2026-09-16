# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recommendation engine for agent improvement insights."""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from uaef.analysis.clustering import FailureCluster, FailureClusterer
from uaef.analysis.root_cause import FailurePattern, RootCauseAnalyzer, RootCauseReport
from uaef.analysis.trends import MetricTrend, TrendAnalyzer, TrendReport
from uaef.models.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class Recommendation(BaseModel):
    """
    A specific, actionable recommendation for agent improvement.
    
    Attributes:
        category: Category of recommendation (prompt, tool, architecture, reasoning)
        priority: Priority level (critical, high, medium, low)
        title: Short title of the recommendation
        description: Detailed description of the issue and recommendation
        rationale: Why this recommendation is being made
        impact: Expected impact if implemented
        examples: Examples from high-scoring test cases (if applicable)
        affected_metrics: Metrics that would improve with this recommendation
        implementation_steps: Specific steps to implement the recommendation
    """
    
    category: Literal["prompt", "tool", "architecture", "reasoning", "context", "safety"] = Field(
        ...,
        description="Category of recommendation"
    )
    priority: Literal["critical", "high", "medium", "low"] = Field(
        ...,
        description="Priority level"
    )
    title: str = Field(..., description="Short title of the recommendation")
    description: str = Field(
        ...,
        description="Detailed description of the issue and recommendation"
    )
    rationale: str = Field(..., description="Why this recommendation is being made")
    impact: str = Field(..., description="Expected impact if implemented")
    examples: List[str] = Field(
        default_factory=list,
        description="Examples from high-scoring test cases"
    )
    affected_metrics: List[str] = Field(
        default_factory=list,
        description="Metrics that would improve with this recommendation"
    )
    implementation_steps: List[str] = Field(
        default_factory=list,
        description="Specific steps to implement the recommendation"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "category": "tool",
                "priority": "high",
                "title": "Add data validation tool",
                "description": "Agent frequently fails when validating data formats",
                "rationale": "15 failures related to data validation across test cases",
                "impact": "Expected 20% improvement in tool_accuracy metric",
                "examples": ["High-scoring case used explicit validation step"],
                "affected_metrics": ["tool_accuracy", "tool_sequence_accuracy"],
                "implementation_steps": [
                    "Create validate_data_format tool",
                    "Add tool to agent's tool set",
                    "Update prompt with validation examples"
                ]
            }
        }


class RecommendationReport(BaseModel):
    """
    Comprehensive recommendation report for agent improvement.
    
    Attributes:
        recommendations: List of recommendations sorted by priority
        summary: High-level summary of key findings
        total_evaluations: Total number of evaluations analyzed
        failed_evaluations: Number of failed evaluations
        success_rate: Overall success rate (0-1)
        top_issues: Top 3 most common issues identified
        improvement_potential: Estimated improvement potential if recommendations implemented
    """
    
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="List of recommendations sorted by priority"
    )
    summary: str = Field(..., description="High-level summary of key findings")
    total_evaluations: int = Field(..., description="Total number of evaluations analyzed")
    failed_evaluations: int = Field(..., description="Number of failed evaluations")
    success_rate: float = Field(..., description="Overall success rate", ge=0.0, le=1.0)
    top_issues: List[str] = Field(
        default_factory=list,
        description="Top 3 most common issues identified"
    )
    improvement_potential: str = Field(
        ...,
        description="Estimated improvement potential if recommendations implemented"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "recommendations": [],
                "summary": "Analyzed 100 evaluations with 35% failure rate",
                "total_evaluations": 100,
                "failed_evaluations": 35,
                "success_rate": 0.65,
                "top_issues": [
                    "Tool selection errors (15 cases)",
                    "Context retention failures (12 cases)",
                    "Response quality issues (8 cases)"
                ],
                "improvement_potential": "Implementing top 3 recommendations could improve success rate to ~80%"
            }
        }


class RecommendationEngine:
    """
    Engine for generating actionable recommendations for agent improvement.
    
    Integrates with RootCauseAnalyzer, FailureClusterer, and TrendAnalyzer to:
    - Identify systemic issues across evaluations
    - Generate category-specific recommendations
    - Prioritize recommendations by impact
    - Provide examples from successful test cases
    - Suggest specific implementation steps
    """
    
    def __init__(
        self,
        root_cause_analyzer: Optional[RootCauseAnalyzer] = None,
        failure_clusterer: Optional[FailureClusterer] = None,
        trend_analyzer: Optional[TrendAnalyzer] = None
    ):
        """
        Initialize the recommendation engine.
        
        Args:
            root_cause_analyzer: RootCauseAnalyzer instance (creates default if None)
            failure_clusterer: FailureClusterer instance (optional, None if not available)
            trend_analyzer: TrendAnalyzer instance (creates default if None)
        """
        self.root_cause_analyzer = root_cause_analyzer or RootCauseAnalyzer()
        self.failure_clusterer = failure_clusterer  # May be None if sklearn not available
        self.trend_analyzer = trend_analyzer or TrendAnalyzer()
        
        logger.info("RecommendationEngine initialized")
    
    def generate_recommendations(
        self,
        evaluation_results: List[EvaluationResult],
        include_examples: bool = True
    ) -> RecommendationReport:
        """
        Generate comprehensive recommendations based on evaluation results.
        
        Analyzes failure patterns, clusters, and trends to provide:
        - Specific, actionable recommendations
        - Prioritization by impact
        - Examples from high-scoring test cases
        - Implementation guidance
        
        Args:
            evaluation_results: List of evaluation results to analyze
            include_examples: Whether to include examples from successful cases
        
        Returns:
            RecommendationReport with prioritized recommendations
        
        Raises:
            ValueError: If no evaluation results provided or analysis fails
        
        Example:
            >>> engine = RecommendationEngine()
            >>> report = engine.generate_recommendations(evaluation_results)
            >>> for rec in report.recommendations[:3]:
            ...     print(f"{rec.priority}: {rec.title}")
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        logger.info(f"Generating recommendations for {len(evaluation_results)} evaluations")
        
        try:
            # Separate passed and failed evaluations
            failed_results = [r for r in evaluation_results if not r.passed]
            passed_results = [r for r in evaluation_results if r.passed]
            
            if not failed_results:
                logger.info("No failures found - generating optimization recommendations")
                return self._generate_optimization_recommendations(
                    evaluation_results,
                    passed_results
                )
            
            # Perform root cause analysis
            failure_patterns = self.root_cause_analyzer.identify_failure_patterns(
                evaluation_results
            )
            
            # Perform failure clustering
            failure_clusters = []
            if self.failure_clusterer:
                try:
                    failure_clusters = self.failure_clusterer.cluster_failures(
                        evaluation_results,
                        method="hierarchical"
                    )
                except Exception as e:
                    logger.warning(f"Failure clustering skipped: {e}")
            else:
                logger.info("Failure clustering skipped (clusterer not available)")
            
            # Perform trend analysis (if enough data points)
            trend_report = None
            if len(evaluation_results) >= 3:
                try:
                    trend_report = self.trend_analyzer.analyze_metric_trends(
                        evaluation_results
                    )
                except ValueError as e:
                    logger.warning(f"Trend analysis skipped: {e}")
            
            # Generate recommendations from different sources
            recommendations = []
            
            # From failure patterns
            recommendations.extend(
                self._generate_pattern_recommendations(failure_patterns)
            )
            
            # From failure clusters
            recommendations.extend(
                self._generate_cluster_recommendations(failure_clusters)
            )
            
            # From trend analysis
            if trend_report:
                recommendations.extend(
                    self._generate_trend_recommendations(trend_report)
                )
            
            # Add examples from successful cases
            if include_examples and passed_results:
                self._add_success_examples(recommendations, passed_results)
            
            # Deduplicate and prioritize
            recommendations = self._deduplicate_recommendations(recommendations)
            recommendations = self._prioritize_recommendations(
                recommendations,
                failed_results,
                evaluation_results
            )
            
            # Generate summary and metrics
            success_rate = len(passed_results) / len(evaluation_results)
            top_issues = self._identify_top_issues(
                failure_patterns,
                failure_clusters
            )
            summary = self._generate_summary(
                len(evaluation_results),
                len(failed_results),
                success_rate,
                recommendations
            )
            improvement_potential = self._estimate_improvement_potential(
                recommendations,
                failed_results
            )
            
            report = RecommendationReport(
                recommendations=recommendations,
                summary=summary,
                total_evaluations=len(evaluation_results),
                failed_evaluations=len(failed_results),
                success_rate=success_rate,
                top_issues=top_issues,
                improvement_potential=improvement_potential
            )
            
            logger.info(
                f"Generated {len(recommendations)} recommendations "
                f"(success rate: {success_rate:.1%})"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            raise ValueError(f"Failed to generate recommendations: {e}") from e
    
    def _generate_pattern_recommendations(
        self,
        failure_patterns: List[FailurePattern]
    ) -> List[Recommendation]:
        """Generate recommendations from failure patterns."""
        recommendations = []
        
        for pattern in failure_patterns:
            # Determine category from pattern type
            category = self._categorize_pattern(pattern.pattern_type)
            
            # Map severity to priority
            priority_map = {
                "critical": "critical",
                "high": "high",
                "medium": "medium",
                "low": "low"
            }
            priority = priority_map.get(pattern.severity, "medium")
            
            # Generate specific recommendation based on category
            rec = self._create_category_recommendation(
                category=category,
                priority=priority,
                pattern=pattern
            )
            
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_cluster_recommendations(
        self,
        failure_clusters: List[FailureCluster]
    ) -> List[Recommendation]:
        """Generate recommendations from failure clusters."""
        recommendations = []
        
        for cluster in failure_clusters:
            # Skip small clusters
            if cluster.size < 3:
                continue
            
            # Determine category from common metrics
            category = self._categorize_metrics(cluster.common_metrics)
            
            # Map severity to priority
            priority_map = {
                "critical": "critical",
                "high": "high",
                "medium": "medium",
                "low": "low"
            }
            priority = priority_map.get(cluster.severity, "medium")
            
            # Create recommendation
            title = f"Address {category} issues affecting {cluster.size} test cases"
            
            description = (
                f"A cluster of {cluster.size} similar failures was identified. "
                f"{cluster.summary}. "
                f"Common failures: {', '.join(cluster.common_failures[:2])}"
            )
            
            rationale = (
                f"This pattern affects {cluster.size} test cases with {cluster.severity} "
                f"severity. Addressing this cluster could significantly improve overall "
                f"performance."
            )
            
            impact = self._estimate_cluster_impact(cluster)
            
            implementation_steps = self._generate_implementation_steps(
                category,
                cluster.common_metrics
            )
            
            rec = Recommendation(
                category=category,
                priority=priority,
                title=title,
                description=description,
                rationale=rationale,
                impact=impact,
                examples=[],
                affected_metrics=cluster.common_metrics,
                implementation_steps=implementation_steps
            )
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_trend_recommendations(
        self,
        trend_report: TrendReport
    ) -> List[Recommendation]:
        """Generate recommendations from trend analysis."""
        recommendations = []
        
        # Focus on declining trends
        for metric_name, trend in trend_report.metric_trends.items():
            if trend.trend_direction == "declining":
                category = self._categorize_metric_name(metric_name)
                
                # Determine priority based on decline severity
                if trend.change_percentage < -20:
                    priority = "critical"
                elif trend.change_percentage < -10:
                    priority = "high"
                else:
                    priority = "medium"
                
                title = f"Address declining {metric_name} metric"
                
                description = (
                    f"The {metric_name} metric has been declining over time. "
                    f"Current trend shows {abs(trend.change_percentage):.1f}% decrease "
                    f"from initial value of {trend.values[0]:.3f} to {trend.values[-1]:.3f}."
                )
                
                rationale = (
                    f"Declining trend detected with slope {trend.slope:.4f}. "
                    f"This indicates a systematic issue that needs attention."
                )
                
                impact = (
                    f"Reversing this trend could improve {metric_name} by "
                    f"~{abs(trend.change_percentage):.1f}%"
                )
                
                implementation_steps = self._generate_implementation_steps(
                    category,
                    [metric_name]
                )
                
                rec = Recommendation(
                    category=category,
                    priority=priority,
                    title=title,
                    description=description,
                    rationale=rationale,
                    impact=impact,
                    examples=[],
                    affected_metrics=[metric_name],
                    implementation_steps=implementation_steps
                )
                
                recommendations.append(rec)
        
        # Add recommendations for critical anomalies
        for anomaly in trend_report.anomalies:
            if anomaly.severity in ["critical", "high"]:
                category = self._categorize_metric_name(anomaly.metric_name)
                
                title = f"Investigate {anomaly.metric_name} anomaly"
                
                description = (
                    f"An anomaly was detected in {anomaly.metric_name}: {anomaly.description}"
                )
                
                rationale = (
                    f"This anomaly has {anomaly.severity} severity and may indicate "
                    f"a significant issue that needs immediate attention."
                )
                
                impact = "Resolving this anomaly could prevent further degradation"
                
                rec = Recommendation(
                    category=category,
                    priority=anomaly.severity,
                    title=title,
                    description=description,
                    rationale=rationale,
                    impact=impact,
                    examples=[],
                    affected_metrics=[anomaly.metric_name],
                    implementation_steps=[
                        "Investigate root cause of anomaly",
                        "Review changes made around anomaly timestamp",
                        "Implement fixes and monitor metric recovery"
                    ]
                )
                
                recommendations.append(rec)
        
        return recommendations
    
    def _add_success_examples(
        self,
        recommendations: List[Recommendation],
        passed_results: List[EvaluationResult]
    ) -> None:
        """Add examples from successful test cases to recommendations."""
        if not passed_results:
            return
        
        # Get high-scoring results (top 20%)
        sorted_results = sorted(
            passed_results,
            key=lambda r: r.overall_score,
            reverse=True
        )
        top_results = sorted_results[:max(1, len(sorted_results) // 5)]
        
        # Extract patterns from successful cases
        for rec in recommendations:
            examples = []
            
            # Find successful cases with high scores in affected metrics
            for result in top_results:
                for dim_result in result.dimension_results:
                    for metric_score in dim_result.metric_scores:
                        if metric_score.metric_name in rec.affected_metrics:
                            if metric_score.score >= 0.9:
                                example = (
                                    f"High-scoring case (score: {metric_score.score:.2f}) "
                                    f"for {metric_score.metric_name}"
                                )
                                if metric_score.reasoning:
                                    example += f": {metric_score.reasoning[:100]}"
                                examples.append(example)
                                break
            
            # Add unique examples (limit to 3)
            rec.examples = list(set(examples))[:3]
    
    def _deduplicate_recommendations(
        self,
        recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Remove duplicate recommendations based on title similarity."""
        seen_titles = set()
        unique_recs = []
        
        for rec in recommendations:
            # Normalize title for comparison
            normalized_title = rec.title.lower().strip()
            
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_recs.append(rec)
        
        return unique_recs
    
    def _prioritize_recommendations(
        self,
        recommendations: List[Recommendation],
        failed_results: List[EvaluationResult],
        all_results: List[EvaluationResult]
    ) -> List[Recommendation]:
        """Prioritize recommendations by impact and frequency."""
        # Calculate impact scores
        for rec in recommendations:
            # Count how many failures this recommendation could address
            affected_count = 0
            for result in failed_results:
                for dim_result in result.dimension_results:
                    for metric_score in dim_result.metric_scores:
                        if metric_score.metric_name in rec.affected_metrics:
                            if metric_score.score < 0.7:
                                affected_count += 1
                                break
            
            # Calculate impact percentage
            impact_percentage = (affected_count / len(all_results)) * 100
            
            # Update impact description
            rec.impact = (
                f"{rec.impact}. "
                f"Could address {affected_count} failures ({impact_percentage:.1f}% of evaluations)"
            )
        
        # Sort by priority and impact
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(
            key=lambda r: (
                priority_order.get(r.priority, 3),
                -len(r.affected_metrics)
            )
        )
        
        return recommendations
    
    def _categorize_pattern(self, pattern_type: str) -> str:
        """Categorize a failure pattern."""
        pattern_lower = pattern_type.lower()
        
        if "tool" in pattern_lower:
            return "tool"
        elif "reasoning" in pattern_lower or "logic" in pattern_lower:
            return "reasoning"
        elif "context" in pattern_lower:
            return "context"
        elif "response" in pattern_lower:
            return "prompt"
        elif "safety" in pattern_lower or "bias" in pattern_lower:
            return "safety"
        else:
            return "architecture"
    
    def _categorize_metrics(self, metrics: List[str]) -> str:
        """Categorize based on metric names."""
        if not metrics:
            return "architecture"
        
        # Count metrics by category
        categories = Counter()
        for metric in metrics:
            category = self._categorize_metric_name(metric)
            categories[category] += 1
        
        # Return most common category
        return categories.most_common(1)[0][0]
    
    def _categorize_metric_name(self, metric_name: str) -> str:
        """Categorize a single metric name."""
        metric_lower = metric_name.lower()
        
        if any(x in metric_lower for x in ["tool", "parameter"]):
            return "tool"
        elif any(x in metric_lower for x in ["reasoning", "logic", "coherence", "fallacy"]):
            return "reasoning"
        elif any(x in metric_lower for x in ["context", "retention"]):
            return "context"
        elif any(x in metric_lower for x in ["relevance", "completeness", "hallucination", "answer"]):
            return "prompt"
        elif any(x in metric_lower for x in ["safety", "bias", "toxicity"]):
            return "safety"
        else:
            return "architecture"
    
    def _create_category_recommendation(
        self,
        category: str,
        priority: str,
        pattern: FailurePattern
    ) -> Optional[Recommendation]:
        """Create a recommendation for a specific category."""
        # Category-specific templates
        templates = {
            "tool": {
                "title": "Improve tool selection and usage",
                "description": (
                    f"Agent shows consistent issues with tool selection. "
                    f"{pattern.description}. "
                    f"This affects {pattern.frequency} test cases."
                ),
                "steps": [
                    "Review and clarify tool descriptions in system prompt",
                    "Add 3-5 examples of correct tool usage for each tool",
                    "Consider simplifying tool set by combining similar tools",
                    "Add tool selection validation logic"
                ]
            },
            "reasoning": {
                "title": "Enhance reasoning capabilities",
                "description": (
                    f"Agent demonstrates logical inconsistencies and reasoning errors. "
                    f"{pattern.description}. "
                    f"This affects {pattern.frequency} test cases."
                ),
                "steps": [
                    "Implement structured reasoning framework (e.g., Chain-of-Thought)",
                    "Add reasoning validation checkpoints in agent workflow",
                    "Include diverse reasoning examples in system prompt",
                    "Consider using reasoning-optimized model"
                ]
            },
            "context": {
                "title": "Improve context management",
                "description": (
                    f"Agent fails to properly retain and use context information. "
                    f"{pattern.description}. "
                    f"This affects {pattern.frequency} test cases."
                ),
                "steps": [
                    "Implement explicit context summarization between turns",
                    "Add context retrieval mechanisms for long conversations",
                    "Optimize context window usage and prioritization",
                    "Add context validation checks"
                ]
            },
            "prompt": {
                "title": "Enhance response generation",
                "description": (
                    f"Agent produces low-quality or irrelevant responses. "
                    f"{pattern.description}. "
                    f"This affects {pattern.frequency} test cases."
                ),
                "steps": [
                    "Refine system prompt with quality guidelines",
                    "Add response quality templates and examples",
                    "Implement response validation before returning",
                    "Add post-processing for response refinement"
                ]
            },
            "safety": {
                "title": "Address safety and bias concerns",
                "description": (
                    f"Agent shows safety or bias issues. "
                    f"{pattern.description}. "
                    f"This affects {pattern.frequency} test cases."
                ),
                "steps": [
                    "Add explicit safety guidelines to system prompt",
                    "Implement content filtering and validation",
                    "Add bias detection and mitigation strategies",
                    "Review and update safety training examples"
                ]
            },
            "architecture": {
                "title": "Consider architecture improvements",
                "description": (
                    f"Multiple failure types suggest architectural issues. "
                    f"{pattern.description}. "
                    f"This affects {pattern.frequency} test cases."
                ),
                "steps": [
                    "Review overall agent architecture and workflow",
                    "Consider adding validation layers between components",
                    "Evaluate alternative agent frameworks or patterns",
                    "Implement comprehensive error handling"
                ]
            }
        }
        
        template = templates.get(category)
        if not template:
            return None
        
        rationale = (
            f"This pattern occurs in {pattern.frequency} test cases with {pattern.severity} "
            f"severity. Addressing it is a high-impact improvement opportunity."
        )
        
        impact = (
            f"Expected to improve {', '.join(pattern.affected_metrics[:3])} metrics. "
            f"Could resolve {pattern.frequency} failures."
        )
        
        return Recommendation(
            category=category,
            priority=priority,
            title=template["title"],
            description=template["description"],
            rationale=rationale,
            impact=impact,
            examples=[],
            affected_metrics=pattern.affected_metrics,
            implementation_steps=template["steps"]
        )
    
    def _generate_implementation_steps(
        self,
        category: str,
        metrics: List[str]
    ) -> List[str]:
        """Generate implementation steps for a category."""
        base_steps = {
            "tool": [
                "Audit current tool set and descriptions",
                "Add clear examples of tool usage to prompt",
                "Implement tool selection validation",
                "Test with diverse tool usage scenarios"
            ],
            "reasoning": [
                "Add structured reasoning framework to prompt",
                "Include reasoning validation checkpoints",
                "Add diverse reasoning examples",
                "Monitor reasoning quality metrics"
            ],
            "context": [
                "Implement context summarization strategy",
                "Add context retrieval mechanisms",
                "Optimize context window management",
                "Test with long conversation scenarios"
            ],
            "prompt": [
                "Review and refine system prompt",
                "Add quality guidelines and examples",
                "Implement response validation",
                "Test with diverse query types"
            ],
            "safety": [
                "Add safety guidelines to system prompt",
                "Implement content filtering",
                "Add bias detection mechanisms",
                "Review with safety experts"
            ],
            "architecture": [
                "Review overall agent design",
                "Identify architectural bottlenecks",
                "Implement validation layers",
                "Consider alternative patterns"
            ]
        }
        
        return base_steps.get(category, [
            "Analyze root cause in detail",
            "Design and implement solution",
            "Test thoroughly",
            "Monitor metrics post-deployment"
        ])
    
    def _estimate_cluster_impact(self, cluster: FailureCluster) -> str:
        """Estimate the impact of addressing a failure cluster."""
        percentage = (cluster.size / 100) * 100  # Assuming ~100 test cases
        
        return (
            f"Addressing this cluster could improve success rate by ~{min(percentage, 20):.1f}%. "
            f"Affects {cluster.size} test cases with {cluster.severity} severity."
        )
    
    def _identify_top_issues(
        self,
        failure_patterns: List[FailurePattern],
        failure_clusters: List[FailureCluster]
    ) -> List[str]:
        """Identify top 3 most common issues."""
        issues = []
        
        # Add top patterns
        for pattern in sorted(failure_patterns, key=lambda p: p.frequency, reverse=True)[:2]:
            issues.append(
                f"{pattern.pattern_type.replace('_', ' ').title()} ({pattern.frequency} cases)"
            )
        
        # Add largest cluster if not already covered
        if failure_clusters:
            largest_cluster = max(failure_clusters, key=lambda c: c.size)
            cluster_desc = f"{largest_cluster.summary.split(':')[0]} ({largest_cluster.size} cases)"
            if cluster_desc not in issues:
                issues.append(cluster_desc)
        
        return issues[:3]
    
    def _generate_summary(
        self,
        total: int,
        failed: int,
        success_rate: float,
        recommendations: List[Recommendation]
    ) -> str:
        """Generate high-level summary."""
        critical_count = sum(1 for r in recommendations if r.priority == "critical")
        high_count = sum(1 for r in recommendations if r.priority == "high")
        
        summary_parts = [
            f"Analyzed {total} evaluations with {failed} failures ({success_rate:.1%} success rate)"
        ]
        
        if critical_count > 0:
            summary_parts.append(f"{critical_count} critical recommendations")
        if high_count > 0:
            summary_parts.append(f"{high_count} high-priority recommendations")
        
        if recommendations:
            summary_parts.append(
                f"Top recommendation: {recommendations[0].title}"
            )
        
        return ". ".join(summary_parts) + "."
    
    def _estimate_improvement_potential(
        self,
        recommendations: List[Recommendation],
        failed_results: List[EvaluationResult]
    ) -> str:
        """Estimate improvement potential if recommendations are implemented."""
        if not recommendations or not failed_results:
            return "No improvement potential estimated"
        
        # Count high-priority recommendations
        high_priority = sum(
            1 for r in recommendations
            if r.priority in ["critical", "high"]
        )
        
        # Estimate based on number of high-priority recommendations
        if high_priority >= 3:
            improvement = "15-25%"
        elif high_priority >= 2:
            improvement = "10-15%"
        else:
            improvement = "5-10%"
        
        return (
            f"Implementing top {min(3, len(recommendations))} recommendations "
            f"could improve success rate by {improvement}"
        )
    
    def _generate_optimization_recommendations(
        self,
        all_results: List[EvaluationResult],
        passed_results: List[EvaluationResult]
    ) -> RecommendationReport:
        """Generate optimization recommendations when there are no failures."""
        recommendations = []
        
        # Find metrics with room for improvement (< 0.95)
        metric_scores = defaultdict(list)
        for result in passed_results:
            for dim_result in result.dimension_results:
                for metric_score in dim_result.metric_scores:
                    metric_scores[metric_score.metric_name].append(metric_score.score)
        
        # Identify metrics with lowest average scores
        avg_scores = {
            metric: sum(scores) / len(scores)
            for metric, scores in metric_scores.items()
            if scores
        }
        
        # Sort by score (lowest first)
        sorted_metrics = sorted(avg_scores.items(), key=lambda x: x[1])
        
        # Generate optimization recommendations for lowest-scoring metrics
        for metric_name, avg_score in sorted_metrics[:3]:
            if avg_score < 0.95:
                category = self._categorize_metric_name(metric_name)
                
                rec = Recommendation(
                    category=category,
                    priority="low",
                    title=f"Optimize {metric_name} performance",
                    description=(
                        f"While {metric_name} is passing (avg: {avg_score:.3f}), "
                        f"there is room for optimization to reach excellence (>0.95)."
                    ),
                    rationale=(
                        f"Current average score of {avg_score:.3f} indicates good but "
                        f"not excellent performance. Optimization could push this metric "
                        f"to excellence."
                    ),
                    impact=f"Could improve {metric_name} by {(0.95 - avg_score):.2f} points",
                    examples=[],
                    affected_metrics=[metric_name],
                    implementation_steps=self._generate_implementation_steps(
                        category,
                        [metric_name]
                    )
                )
                recommendations.append(rec)
        
        summary = (
            f"Analyzed {len(all_results)} evaluations with 100% success rate. "
            f"Generated {len(recommendations)} optimization recommendations for excellence."
        )
        
        return RecommendationReport(
            recommendations=recommendations,
            summary=summary,
            total_evaluations=len(all_results),
            failed_evaluations=0,
            success_rate=1.0,
            top_issues=["No failures detected - focus on optimization"],
            improvement_potential=(
                "Agent is performing well. Optimization recommendations focus on "
                "achieving excellence (>0.95) across all metrics."
            )
        )
