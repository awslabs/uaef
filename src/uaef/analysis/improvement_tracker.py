# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Improvement tracker for monitoring recommendation implementation and impact."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from uaef.analysis.recommendations import Recommendation
from uaef.models.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class RecommendationStatus(BaseModel):
    """
    Status tracking for a recommendation implementation.
    
    Attributes:
        recommendation_id: Unique identifier for this recommendation tracking
        recommendation: The recommendation being tracked
        status: Current implementation status
        baseline_metrics: Metric values before implementation
        post_implementation_metrics: Metric values after implementation (if completed)
        created_at: When tracking started
        started_at: When implementation started (if in_progress or completed)
        completed_at: When implementation completed (if completed)
        notes: Implementation notes or observations
    """
    
    recommendation_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this recommendation tracking"
    )
    recommendation: Recommendation = Field(
        ...,
        description="The recommendation being tracked"
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending",
        description="Current implementation status"
    )
    baseline_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metric values before implementation"
    )
    post_implementation_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metric values after implementation"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When tracking started"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="When implementation started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When implementation completed"
    )
    notes: str = Field(
        default="",
        description="Implementation notes or observations"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "recommendation_id": "550e8400-e29b-41d4-a716-446655440000",
                "recommendation": {
                    "category": "tool",
                    "priority": "high",
                    "title": "Add data validation tool",
                    "description": "Agent frequently fails when validating data formats",
                    "rationale": "15 failures related to data validation",
                    "impact": "Expected 20% improvement in tool_accuracy",
                    "affected_metrics": ["tool_accuracy"]
                },
                "status": "completed",
                "baseline_metrics": {"tool_accuracy": 0.65},
                "post_implementation_metrics": {"tool_accuracy": 0.82},
                "created_at": "2024-01-01T10:00:00Z",
                "started_at": "2024-01-02T10:00:00Z",
                "completed_at": "2024-01-05T10:00:00Z",
                "notes": "Added validate_data_format tool and updated prompt"
            }
        }


class ImprovementDelta(BaseModel):
    """
    Calculated improvement delta for a metric.
    
    Attributes:
        metric_name: Name of the metric
        baseline_value: Value before implementation
        post_implementation_value: Value after implementation
        absolute_change: Absolute change (post - baseline)
        percentage_change: Percentage change ((post - baseline) / baseline * 100)
        improved: Whether the metric improved (positive change)
    """
    
    metric_name: str = Field(..., description="Name of the metric")
    baseline_value: float = Field(..., description="Value before implementation")
    post_implementation_value: float = Field(..., description="Value after implementation")
    absolute_change: float = Field(..., description="Absolute change")
    percentage_change: float = Field(..., description="Percentage change")
    improved: bool = Field(..., description="Whether the metric improved")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_name": "tool_accuracy",
                "baseline_value": 0.65,
                "post_implementation_value": 0.82,
                "absolute_change": 0.17,
                "percentage_change": 26.15,
                "improved": True
            }
        }


class ImprovementReport(BaseModel):
    """
    Report showing before/after comparison for a recommendation.
    
    Attributes:
        recommendation_id: ID of the tracked recommendation
        recommendation_title: Title of the recommendation
        recommendation_category: Category of the recommendation
        status: Current implementation status
        metric_deltas: List of improvement deltas for affected metrics
        overall_improvement: Overall improvement summary
        implementation_duration_days: Days from start to completion (if completed)
        created_at: When tracking started
        completed_at: When implementation completed (if completed)
        notes: Implementation notes
    """
    
    recommendation_id: UUID = Field(
        ...,
        description="ID of the tracked recommendation"
    )
    recommendation_title: str = Field(..., description="Title of the recommendation")
    recommendation_category: str = Field(..., description="Category of the recommendation")
    status: Literal["pending", "in_progress", "completed"] = Field(
        ...,
        description="Current implementation status"
    )
    metric_deltas: List[ImprovementDelta] = Field(
        default_factory=list,
        description="List of improvement deltas for affected metrics"
    )
    overall_improvement: str = Field(
        ...,
        description="Overall improvement summary"
    )
    implementation_duration_days: Optional[float] = Field(
        default=None,
        description="Days from start to completion"
    )
    created_at: datetime = Field(..., description="When tracking started")
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When implementation completed"
    )
    notes: str = Field(default="", description="Implementation notes")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "recommendation_id": "550e8400-e29b-41d4-a716-446655440000",
                "recommendation_title": "Add data validation tool",
                "recommendation_category": "tool",
                "status": "completed",
                "metric_deltas": [
                    {
                        "metric_name": "tool_accuracy",
                        "baseline_value": 0.65,
                        "post_implementation_value": 0.82,
                        "absolute_change": 0.17,
                        "percentage_change": 26.15,
                        "improved": True
                    }
                ],
                "overall_improvement": "Improved 1/1 metrics (100% success rate)",
                "implementation_duration_days": 3.0,
                "created_at": "2024-01-01T10:00:00Z",
                "completed_at": "2024-01-05T10:00:00Z",
                "notes": "Added validate_data_format tool"
            }
        }


class ImprovementTracker:
    """
    Tracker for monitoring recommendation implementation and measuring impact.
    
    Tracks:
    - Recommendation implementation status (pending, in_progress, completed)
    - Baseline metrics before implementation
    - Post-implementation metrics
    - Improvement deltas and percentage changes
    - Multiple recommendations simultaneously
    
    Provides:
    - Before/after comparison reports
    - Impact measurement for completed recommendations
    - Status tracking across multiple recommendations
    """
    
    def __init__(self):
        """Initialize the improvement tracker."""
        self._tracked_recommendations: Dict[UUID, RecommendationStatus] = {}
        logger.info("ImprovementTracker initialized")
    
    def start_tracking(
        self,
        recommendation: Recommendation,
        baseline_evaluation_results: List[EvaluationResult]
    ) -> UUID:
        """
        Start tracking a recommendation implementation.
        
        Records baseline metrics from evaluation results before implementation begins.
        
        Args:
            recommendation: The recommendation to track
            baseline_evaluation_results: Evaluation results before implementation
        
        Returns:
            UUID: Recommendation tracking ID
        
        Raises:
            ValueError: If no baseline evaluation results provided
        
        Example:
            >>> tracker = ImprovementTracker()
            >>> rec_id = tracker.start_tracking(recommendation, baseline_results)
            >>> print(f"Tracking recommendation: {rec_id}")
        """
        if not baseline_evaluation_results:
            raise ValueError("No baseline evaluation results provided")
        
        # Calculate baseline metrics
        baseline_metrics = self._calculate_average_metrics(baseline_evaluation_results)
        
        # Create tracking status
        status = RecommendationStatus(
            recommendation=recommendation,
            status="pending",
            baseline_metrics=baseline_metrics
        )
        
        self._tracked_recommendations[status.recommendation_id] = status
        
        logger.info(
            f"Started tracking recommendation '{recommendation.title}' "
            f"(ID: {status.recommendation_id})"
        )
        
        return status.recommendation_id
    
    def mark_in_progress(
        self,
        recommendation_id: UUID,
        notes: str = ""
    ) -> None:
        """
        Mark a recommendation as in progress.
        
        Args:
            recommendation_id: ID of the tracked recommendation
            notes: Optional notes about the implementation
        
        Raises:
            ValueError: If recommendation ID not found or already completed
        
        Example:
            >>> tracker.mark_in_progress(rec_id, "Started implementing tool")
        """
        status = self._get_status(recommendation_id)
        
        if status.status == "completed":
            raise ValueError(
                f"Recommendation {recommendation_id} is already completed"
            )
        
        status.status = "in_progress"
        status.started_at = datetime.utcnow()
        if notes:
            status.notes = notes
        
        logger.info(f"Marked recommendation {recommendation_id} as in_progress")
    
    def mark_completed(
        self,
        recommendation_id: UUID,
        post_implementation_evaluation_results: List[EvaluationResult],
        notes: str = ""
    ) -> ImprovementReport:
        """
        Mark a recommendation as completed and record post-implementation metrics.
        
        Calculates improvement deltas and generates a before/after report.
        
        Args:
            recommendation_id: ID of the tracked recommendation
            post_implementation_evaluation_results: Evaluation results after implementation
            notes: Optional notes about the implementation
        
        Returns:
            ImprovementReport: Report showing before/after comparison
        
        Raises:
            ValueError: If recommendation ID not found or no post-implementation results
        
        Example:
            >>> report = tracker.mark_completed(rec_id, post_results, "Tool added successfully")
            >>> print(f"Overall improvement: {report.overall_improvement}")
        """
        status = self._get_status(recommendation_id)
        
        if not post_implementation_evaluation_results:
            raise ValueError("No post-implementation evaluation results provided")
        
        # Calculate post-implementation metrics
        post_metrics = self._calculate_average_metrics(
            post_implementation_evaluation_results
        )
        
        # Update status
        status.status = "completed"
        status.completed_at = datetime.utcnow()
        status.post_implementation_metrics = post_metrics
        if notes:
            status.notes = notes
        
        # If not already marked in_progress, set started_at to completed_at
        if status.started_at is None:
            status.started_at = status.completed_at
        
        logger.info(f"Marked recommendation {recommendation_id} as completed")
        
        # Generate improvement report
        return self.generate_improvement_report(recommendation_id)
    
    def generate_improvement_report(
        self,
        recommendation_id: UUID
    ) -> ImprovementReport:
        """
        Generate an improvement report for a tracked recommendation.
        
        Shows before/after comparison with improvement deltas.
        
        Args:
            recommendation_id: ID of the tracked recommendation
        
        Returns:
            ImprovementReport: Report with before/after comparison
        
        Raises:
            ValueError: If recommendation ID not found or not completed
        
        Example:
            >>> report = tracker.generate_improvement_report(rec_id)
            >>> for delta in report.metric_deltas:
            ...     print(f"{delta.metric_name}: {delta.percentage_change:.1f}%")
        """
        status = self._get_status(recommendation_id)
        
        if status.status != "completed":
            raise ValueError(
                f"Recommendation {recommendation_id} is not completed yet "
                f"(status: {status.status})"
            )
        
        # Calculate improvement deltas
        metric_deltas = self._calculate_improvement_deltas(status)
        
        # Calculate implementation duration
        duration_days = None
        if status.started_at and status.completed_at:
            duration = status.completed_at - status.started_at
            duration_days = duration.total_seconds() / 86400  # Convert to days
        
        # Generate overall improvement summary
        improved_count = sum(1 for delta in metric_deltas if delta.improved)
        total_count = len(metric_deltas)
        
        if total_count == 0:
            overall_improvement = "No metrics tracked"
        else:
            success_rate = (improved_count / total_count) * 100
            overall_improvement = (
                f"Improved {improved_count}/{total_count} metrics "
                f"({success_rate:.1f}% success rate)"
            )
        
        report = ImprovementReport(
            recommendation_id=status.recommendation_id,
            recommendation_title=status.recommendation.title,
            recommendation_category=status.recommendation.category,
            status=status.status,
            metric_deltas=metric_deltas,
            overall_improvement=overall_improvement,
            implementation_duration_days=duration_days,
            created_at=status.created_at,
            completed_at=status.completed_at,
            notes=status.notes
        )
        
        logger.info(
            f"Generated improvement report for {recommendation_id}: "
            f"{overall_improvement}"
        )
        
        return report
    
    def get_all_tracked_recommendations(self) -> List[RecommendationStatus]:
        """
        Get all tracked recommendations.
        
        Returns:
            List of all RecommendationStatus objects
        
        Example:
            >>> all_recs = tracker.get_all_tracked_recommendations()
            >>> for rec in all_recs:
            ...     print(f"{rec.recommendation.title}: {rec.status}")
        """
        return list(self._tracked_recommendations.values())
    
    def get_recommendations_by_status(
        self,
        status: Literal["pending", "in_progress", "completed"]
    ) -> List[RecommendationStatus]:
        """
        Get tracked recommendations filtered by status.
        
        Args:
            status: Status to filter by
        
        Returns:
            List of RecommendationStatus objects with the specified status
        
        Example:
            >>> pending = tracker.get_recommendations_by_status("pending")
            >>> print(f"Pending recommendations: {len(pending)}")
        """
        return [
            rec for rec in self._tracked_recommendations.values()
            if rec.status == status
        ]
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of all tracked recommendations.
        
        Returns:
            Dictionary with summary statistics and reports
        
        Example:
            >>> summary = tracker.generate_summary_report()
            >>> print(f"Total tracked: {summary['total_tracked']}")
            >>> print(f"Completed: {summary['completed_count']}")
        """
        all_recs = self.get_all_tracked_recommendations()
        
        pending = self.get_recommendations_by_status("pending")
        in_progress = self.get_recommendations_by_status("in_progress")
        completed = self.get_recommendations_by_status("completed")
        
        # Generate reports for completed recommendations
        completed_reports = []
        for rec in completed:
            try:
                report = self.generate_improvement_report(rec.recommendation_id)
                completed_reports.append(report)
            except Exception as e:
                logger.warning(
                    f"Failed to generate report for {rec.recommendation_id}: {e}"
                )
        
        # Calculate overall statistics
        total_improved_metrics = sum(
            sum(1 for delta in report.metric_deltas if delta.improved)
            for report in completed_reports
        )
        total_tracked_metrics = sum(
            len(report.metric_deltas)
            for report in completed_reports
        )
        
        overall_success_rate = 0.0
        if total_tracked_metrics > 0:
            overall_success_rate = (total_improved_metrics / total_tracked_metrics) * 100
        
        summary = {
            "total_tracked": len(all_recs),
            "pending_count": len(pending),
            "in_progress_count": len(in_progress),
            "completed_count": len(completed),
            "completed_reports": completed_reports,
            "total_improved_metrics": total_improved_metrics,
            "total_tracked_metrics": total_tracked_metrics,
            "overall_success_rate": overall_success_rate,
            "pending_recommendations": [
                {
                    "id": str(rec.recommendation_id),
                    "title": rec.recommendation.title,
                    "category": rec.recommendation.category,
                    "priority": rec.recommendation.priority
                }
                for rec in pending
            ],
            "in_progress_recommendations": [
                {
                    "id": str(rec.recommendation_id),
                    "title": rec.recommendation.title,
                    "category": rec.recommendation.category,
                    "priority": rec.recommendation.priority,
                    "started_at": rec.started_at.isoformat() if rec.started_at else None
                }
                for rec in in_progress
            ]
        }
        
        logger.info(
            f"Generated summary report: {len(completed)} completed, "
            f"{len(in_progress)} in progress, {len(pending)} pending"
        )
        
        return summary
    
    def _get_status(self, recommendation_id: UUID) -> RecommendationStatus:
        """
        Get recommendation status by ID.
        
        Args:
            recommendation_id: ID of the tracked recommendation
        
        Returns:
            RecommendationStatus object
        
        Raises:
            ValueError: If recommendation ID not found
        """
        if recommendation_id not in self._tracked_recommendations:
            raise ValueError(
                f"Recommendation {recommendation_id} not found in tracker"
            )
        
        return self._tracked_recommendations[recommendation_id]
    
    def _calculate_average_metrics(
        self,
        evaluation_results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """
        Calculate average metric values across evaluation results.
        
        Args:
            evaluation_results: List of evaluation results
        
        Returns:
            Dictionary mapping metric names to average values
        """
        metric_values: Dict[str, List[float]] = {}
        
        # Collect all metric values
        for result in evaluation_results:
            for dimension_result in result.dimension_results:
                for metric_score in dimension_result.metric_scores:
                    metric_name = metric_score.metric_name
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(metric_score.score)
        
        # Calculate averages
        average_metrics = {
            metric_name: sum(values) / len(values)
            for metric_name, values in metric_values.items()
        }
        
        return average_metrics
    
    def _calculate_improvement_deltas(
        self,
        status: RecommendationStatus
    ) -> List[ImprovementDelta]:
        """
        Calculate improvement deltas for all tracked metrics.
        
        Args:
            status: RecommendationStatus with baseline and post-implementation metrics
        
        Returns:
            List of ImprovementDelta objects
        """
        deltas = []
        
        # Get all metrics that appear in either baseline or post-implementation
        all_metrics = set(status.baseline_metrics.keys()) | set(
            status.post_implementation_metrics.keys()
        )
        
        for metric_name in all_metrics:
            baseline = status.baseline_metrics.get(metric_name, 0.0)
            post = status.post_implementation_metrics.get(metric_name, 0.0)
            
            absolute_change = post - baseline
            
            # Calculate percentage change (handle division by zero)
            if baseline == 0.0:
                if post == 0.0:
                    percentage_change = 0.0
                else:
                    percentage_change = 100.0  # Arbitrary large value
            else:
                percentage_change = (absolute_change / baseline) * 100
            
            improved = absolute_change > 0
            
            delta = ImprovementDelta(
                metric_name=metric_name,
                baseline_value=baseline,
                post_implementation_value=post,
                absolute_change=absolute_change,
                percentage_change=percentage_change,
                improved=improved
            )
            
            deltas.append(delta)
        
        # Sort by absolute change (descending)
        deltas.sort(key=lambda d: abs(d.absolute_change), reverse=True)
        
        return deltas
