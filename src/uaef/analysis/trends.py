# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trend analyzer for tracking metric changes over time."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from uaef.models.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class MetricTrend(BaseModel):
    """
    Trend information for a specific metric.
    
    Attributes:
        metric_name: Name of the metric
        values: List of metric values over time
        timestamps: List of timestamps corresponding to values
        trend_direction: Overall trend direction (improving, declining, stable)
        slope: Linear regression slope (positive = improving)
        mean: Mean value across all data points
        std_dev: Standard deviation
        min_value: Minimum value
        max_value: Maximum value
        change_percentage: Percentage change from first to last value
    """
    
    metric_name: str = Field(..., description="Name of the metric")
    values: List[float] = Field(
        default_factory=list,
        description="List of metric values over time"
    )
    timestamps: List[datetime] = Field(
        default_factory=list,
        description="List of timestamps corresponding to values"
    )
    trend_direction: Literal["improving", "declining", "stable"] = Field(
        ...,
        description="Overall trend direction"
    )
    slope: float = Field(..., description="Linear regression slope")
    mean: float = Field(..., description="Mean value across all data points")
    std_dev: float = Field(..., description="Standard deviation")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    change_percentage: float = Field(
        ...,
        description="Percentage change from first to last value"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_name": "tool_accuracy",
                "values": [0.75, 0.78, 0.82, 0.85],
                "timestamps": ["2024-01-01T10:00:00Z", "2024-01-02T10:00:00Z"],
                "trend_direction": "improving",
                "slope": 0.033,
                "mean": 0.80,
                "std_dev": 0.042,
                "min_value": 0.75,
                "max_value": 0.85,
                "change_percentage": 13.33
            }
        }


class Anomaly(BaseModel):
    """
    Detected anomaly in metric values.
    
    Attributes:
        metric_name: Name of the metric with anomaly
        value: Anomalous value
        timestamp: When the anomaly occurred
        expected_range: Expected range (min, max) based on historical data
        z_score: Z-score of the anomalous value
        severity: Severity level (low, medium, high, critical)
        description: Human-readable description of the anomaly
    """
    
    metric_name: str = Field(..., description="Name of the metric with anomaly")
    value: float = Field(..., description="Anomalous value")
    timestamp: datetime = Field(..., description="When the anomaly occurred")
    expected_range: Tuple[float, float] = Field(
        ...,
        description="Expected range (min, max) based on historical data"
    )
    z_score: float = Field(..., description="Z-score of the anomalous value")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description="Severity level"
    )
    description: str = Field(
        ...,
        description="Human-readable description of the anomaly"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_name": "tool_accuracy",
                "value": 0.45,
                "timestamp": "2024-01-15T10:00:00Z",
                "expected_range": (0.75, 0.90),
                "z_score": -3.5,
                "severity": "critical",
                "description": "tool_accuracy dropped to 0.45, significantly below expected range [0.75, 0.90]"
            }
        }


class TrendReport(BaseModel):
    """
    Comprehensive trend analysis report.
    
    Attributes:
        metric_trends: Dictionary mapping metric names to their trends
        anomalies: List of detected anomalies
        summary: High-level summary of trends
        time_range: Time range covered by the analysis (start, end)
        total_data_points: Total number of data points analyzed
        metrics_analyzed: Number of unique metrics analyzed
    """
    
    metric_trends: Dict[str, MetricTrend] = Field(
        default_factory=dict,
        description="Dictionary mapping metric names to their trends"
    )
    anomalies: List[Anomaly] = Field(
        default_factory=list,
        description="List of detected anomalies"
    )
    summary: str = Field(..., description="High-level summary of trends")
    time_range: Tuple[datetime, datetime] = Field(
        ...,
        description="Time range covered by the analysis"
    )
    total_data_points: int = Field(
        ...,
        description="Total number of data points analyzed"
    )
    metrics_analyzed: int = Field(
        ...,
        description="Number of unique metrics analyzed"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_trends": {},
                "anomalies": [],
                "summary": "Overall improving trend across 5 metrics with 2 anomalies detected",
                "time_range": ("2024-01-01T00:00:00Z", "2024-01-15T00:00:00Z"),
                "total_data_points": 100,
                "metrics_analyzed": 5
            }
        }


class TrendAnalyzer:
    """
    Analyzer for tracking metric trends over time.
    
    Provides methods for:
    - Analyzing metric trends across experiment runs
    - Detecting anomalies using statistical methods
    - Tracking metric changes over time
    - Generating trend reports with visualizations
    """
    
    def __init__(
        self,
        min_data_points: int = 3,
        anomaly_threshold: float = 2.5,
        stable_threshold: float = 0.01
    ):
        """
        Initialize the trend analyzer.
        
        Args:
            min_data_points: Minimum number of data points required for trend analysis
            anomaly_threshold: Z-score threshold for anomaly detection (default: 2.5)
            stable_threshold: Threshold for considering trend stable (default: 0.01)
        
        Raises:
            ValueError: If parameters are invalid
        """
        if min_data_points < 2:
            raise ValueError("min_data_points must be at least 2")
        if anomaly_threshold <= 0:
            raise ValueError("anomaly_threshold must be positive")
        if stable_threshold <= 0:
            raise ValueError("stable_threshold must be positive")
        
        self.min_data_points = min_data_points
        self.anomaly_threshold = anomaly_threshold
        self.stable_threshold = stable_threshold
        
        logger.info(
            f"TrendAnalyzer initialized with min_data_points={min_data_points}, "
            f"anomaly_threshold={anomaly_threshold}, stable_threshold={stable_threshold}"
        )
    
    def analyze_metric_trends(
        self,
        evaluation_results: List[EvaluationResult],
        metric_name: Optional[str] = None
    ) -> TrendReport:
        """
        Analyze metric trends over time across evaluation results.
        
        Args:
            evaluation_results: List of evaluation results ordered by time
            metric_name: Specific metric to analyze (None = analyze all metrics)
        
        Returns:
            TrendReport with trend analysis for requested metrics
        
        Raises:
            ValueError: If insufficient data or invalid inputs
        
        Example:
            >>> analyzer = TrendAnalyzer()
            >>> report = analyzer.analyze_metric_trends(results, "tool_accuracy")
            >>> print(f"Trend: {report.metric_trends['tool_accuracy'].trend_direction}")
            >>> print(f"Change: {report.metric_trends['tool_accuracy'].change_percentage:.1f}%")
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        if len(evaluation_results) < self.min_data_points:
            raise ValueError(
                f"Insufficient data points: {len(evaluation_results)} provided, "
                f"{self.min_data_points} required"
            )
        
        logger.info(
            f"Analyzing trends for {len(evaluation_results)} evaluation results"
            + (f" (metric: {metric_name})" if metric_name else " (all metrics)")
        )
        
        # Sort by timestamp
        sorted_results = sorted(evaluation_results, key=lambda r: r.timestamp)
        
        # Extract metric values over time
        metric_data = self._extract_metric_data(sorted_results, metric_name)
        
        if not metric_data:
            raise ValueError(
                f"No data found for metric: {metric_name}" if metric_name
                else "No metric data found in evaluation results"
            )
        
        # Analyze trends for each metric
        metric_trends = {}
        for metric, data in metric_data.items():
            if len(data["values"]) >= self.min_data_points:
                trend = self._analyze_single_metric_trend(
                    metric,
                    data["values"],
                    data["timestamps"]
                )
                metric_trends[metric] = trend
            else:
                logger.warning(
                    f"Skipping metric {metric}: only {len(data['values'])} data points"
                )
        
        # Detect anomalies
        anomalies = self._detect_all_anomalies(metric_data)
        
        # Generate summary
        summary = self._generate_trend_summary(metric_trends, anomalies)
        
        # Determine time range
        all_timestamps = [r.timestamp for r in sorted_results]
        time_range = (min(all_timestamps), max(all_timestamps))
        
        report = TrendReport(
            metric_trends=metric_trends,
            anomalies=anomalies,
            summary=summary,
            time_range=time_range,
            total_data_points=len(sorted_results),
            metrics_analyzed=len(metric_trends)
        )
        
        logger.info(
            f"Trend analysis complete: {len(metric_trends)} metrics analyzed, "
            f"{len(anomalies)} anomalies detected"
        )
        
        return report
    
    def detect_anomalies(
        self,
        evaluation_results: List[EvaluationResult],
        metric_name: Optional[str] = None,
        method: Literal["zscore", "iqr"] = "zscore"
    ) -> List[Anomaly]:
        """
        Detect statistical anomalies in metric values.
        
        Args:
            evaluation_results: List of evaluation results
            metric_name: Specific metric to check (None = check all metrics)
            method: Anomaly detection method ("zscore" or "iqr")
        
        Returns:
            List of detected anomalies
        
        Raises:
            ValueError: If insufficient data or invalid inputs
        
        Example:
            >>> analyzer = TrendAnalyzer()
            >>> anomalies = analyzer.detect_anomalies(results, method="zscore")
            >>> for anomaly in anomalies:
            ...     print(f"{anomaly.metric_name}: {anomaly.description}")
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        if len(evaluation_results) < self.min_data_points:
            raise ValueError(
                f"Insufficient data points: {len(evaluation_results)} provided, "
                f"{self.min_data_points} required"
            )
        
        logger.info(
            f"Detecting anomalies in {len(evaluation_results)} evaluation results "
            f"using {method} method"
            + (f" (metric: {metric_name})" if metric_name else " (all metrics)")
        )
        
        # Extract metric data
        metric_data = self._extract_metric_data(evaluation_results, metric_name)
        
        if not metric_data:
            raise ValueError(
                f"No data found for metric: {metric_name}" if metric_name
                else "No metric data found in evaluation results"
            )
        
        # Detect anomalies based on method
        if method == "zscore":
            anomalies = self._detect_anomalies_zscore(metric_data)
        elif method == "iqr":
            anomalies = self._detect_anomalies_iqr(metric_data)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        logger.info(f"Detected {len(anomalies)} anomalies using {method} method")
        
        return anomalies
    
    def _extract_metric_data(
        self,
        evaluation_results: List[EvaluationResult],
        metric_name: Optional[str] = None
    ) -> Dict[str, Dict[str, List]]:
        """
        Extract metric values and timestamps from evaluation results.
        
        Args:
            evaluation_results: List of evaluation results
            metric_name: Specific metric to extract (None = extract all)
        
        Returns:
            Dictionary mapping metric names to {"values": [...], "timestamps": [...]}
        """
        metric_data: Dict[str, Dict[str, List]] = defaultdict(
            lambda: {"values": [], "timestamps": []}
        )
        
        for result in evaluation_results:
            for dim_result in result.dimension_results:
                for metric_score in dim_result.metric_scores:
                    # Filter by metric name if specified
                    if metric_name and metric_score.metric_name != metric_name:
                        continue
                    
                    metric_data[metric_score.metric_name]["values"].append(metric_score.score)
                    metric_data[metric_score.metric_name]["timestamps"].append(result.timestamp)
        
        return dict(metric_data)
    
    def _analyze_single_metric_trend(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime]
    ) -> MetricTrend:
        """
        Analyze trend for a single metric.
        
        Args:
            metric_name: Name of the metric
            values: List of metric values
            timestamps: List of timestamps
        
        Returns:
            MetricTrend with analysis results
        """
        values_array = np.array(values)
        
        # Calculate statistics
        mean = float(np.mean(values_array))
        std_dev = float(np.std(values_array))
        min_value = float(np.min(values_array))
        max_value = float(np.max(values_array))
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        slope = float(np.polyfit(x, values_array, 1)[0])
        
        # Determine trend direction
        if abs(slope) < self.stable_threshold:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"
        
        # Calculate percentage change
        if values[0] != 0:
            change_percentage = ((values[-1] - values[0]) / values[0]) * 100
        else:
            change_percentage = 0.0
        
        return MetricTrend(
            metric_name=metric_name,
            values=values,
            timestamps=timestamps,
            trend_direction=trend_direction,
            slope=slope,
            mean=mean,
            std_dev=std_dev,
            min_value=min_value,
            max_value=max_value,
            change_percentage=change_percentage
        )
    
    def _detect_all_anomalies(
        self,
        metric_data: Dict[str, Dict[str, List]]
    ) -> List[Anomaly]:
        """
        Detect anomalies across all metrics using z-score method.
        
        Args:
            metric_data: Dictionary of metric data
        
        Returns:
            List of detected anomalies
        """
        return self._detect_anomalies_zscore(metric_data)
    
    def _detect_anomalies_zscore(
        self,
        metric_data: Dict[str, Dict[str, List]]
    ) -> List[Anomaly]:
        """
        Detect anomalies using z-score method.
        
        Args:
            metric_data: Dictionary of metric data
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for metric_name, data in metric_data.items():
            values = np.array(data["values"])
            timestamps = data["timestamps"]
            
            if len(values) < self.min_data_points:
                continue
            
            # Calculate mean and standard deviation
            mean = np.mean(values)
            std_dev = np.std(values)
            
            # Skip if std_dev is too small (no variation)
            if std_dev < 1e-6:
                continue
            
            # Calculate z-scores
            z_scores = (values - mean) / std_dev
            
            # Detect anomalies
            for i, (value, timestamp, z_score) in enumerate(
                zip(values, timestamps, z_scores)
            ):
                if abs(z_score) > self.anomaly_threshold:
                    # Calculate expected range (mean ± 2 std_dev)
                    expected_min = max(0.0, mean - 2 * std_dev)
                    expected_max = min(1.0, mean + 2 * std_dev)
                    
                    # Determine severity based on z-score magnitude
                    severity = self._calculate_anomaly_severity(abs(z_score))
                    
                    # Generate description
                    description = self._generate_anomaly_description(
                        metric_name,
                        value,
                        (expected_min, expected_max),
                        z_score
                    )
                    
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        value=float(value),
                        timestamp=timestamp,
                        expected_range=(float(expected_min), float(expected_max)),
                        z_score=float(z_score),
                        severity=severity,
                        description=description
                    )
                    anomalies.append(anomaly)
        
        # Sort by severity and z-score magnitude
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(
            key=lambda a: (severity_order[a.severity], -abs(a.z_score))
        )
        
        return anomalies
    
    def _detect_anomalies_iqr(
        self,
        metric_data: Dict[str, Dict[str, List]]
    ) -> List[Anomaly]:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            metric_data: Dictionary of metric data
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for metric_name, data in metric_data.items():
            values = np.array(data["values"])
            timestamps = data["timestamps"]
            
            if len(values) < self.min_data_points:
                continue
            
            # Calculate quartiles and IQR
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            # Skip if IQR is too small (no variation)
            if iqr < 1e-6:
                continue
            
            # Calculate bounds (1.5 * IQR is standard for outliers)
            lower_bound = max(0.0, q1 - 1.5 * iqr)
            upper_bound = min(1.0, q3 + 1.5 * iqr)
            
            # Detect anomalies
            for value, timestamp in zip(values, timestamps):
                if value < lower_bound or value > upper_bound:
                    # Calculate pseudo z-score for severity
                    mean = np.mean(values)
                    std_dev = np.std(values)
                    z_score = (value - mean) / std_dev if std_dev > 1e-6 else 0.0
                    
                    # Determine severity
                    severity = self._calculate_anomaly_severity(abs(z_score))
                    
                    # Generate description
                    description = self._generate_anomaly_description(
                        metric_name,
                        value,
                        (lower_bound, upper_bound),
                        z_score
                    )
                    
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        value=float(value),
                        timestamp=timestamp,
                        expected_range=(float(lower_bound), float(upper_bound)),
                        z_score=float(z_score),
                        severity=severity,
                        description=description
                    )
                    anomalies.append(anomaly)
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(
            key=lambda a: (severity_order[a.severity], -abs(a.z_score))
        )
        
        return anomalies
    
    def _calculate_anomaly_severity(self, abs_z_score: float) -> str:
        """
        Calculate anomaly severity based on z-score magnitude.
        
        Args:
            abs_z_score: Absolute value of z-score
        
        Returns:
            Severity level: "low", "medium", "high", or "critical"
        """
        if abs_z_score >= 4.0:
            return "critical"
        elif abs_z_score >= 3.5:
            return "high"
        elif abs_z_score >= 3.0:
            return "medium"
        else:
            return "low"
    
    def _generate_anomaly_description(
        self,
        metric_name: str,
        value: float,
        expected_range: Tuple[float, float],
        z_score: float
    ) -> str:
        """
        Generate human-readable description for an anomaly.
        
        Args:
            metric_name: Name of the metric
            value: Anomalous value
            expected_range: Expected range (min, max)
            z_score: Z-score of the value
        
        Returns:
            Human-readable description
        """
        direction = "above" if z_score > 0 else "below"
        
        return (
            f"{metric_name} value {value:.3f} is significantly {direction} "
            f"expected range [{expected_range[0]:.3f}, {expected_range[1]:.3f}] "
            f"(z-score: {z_score:.2f})"
        )
    
    def _generate_trend_summary(
        self,
        metric_trends: Dict[str, MetricTrend],
        anomalies: List[Anomaly]
    ) -> str:
        """
        Generate high-level summary of trends.
        
        Args:
            metric_trends: Dictionary of metric trends
            anomalies: List of detected anomalies
        
        Returns:
            Human-readable summary
        """
        if not metric_trends:
            return "No trends to analyze"
        
        # Count trend directions
        improving = sum(1 for t in metric_trends.values() if t.trend_direction == "improving")
        declining = sum(1 for t in metric_trends.values() if t.trend_direction == "declining")
        stable = sum(1 for t in metric_trends.values() if t.trend_direction == "stable")
        
        # Count anomaly severities
        critical_anomalies = sum(1 for a in anomalies if a.severity == "critical")
        high_anomalies = sum(1 for a in anomalies if a.severity == "high")
        
        # Build summary
        parts = []
        
        if improving > declining:
            parts.append(f"Overall improving trend across {improving} metrics")
        elif declining > improving:
            parts.append(f"Overall declining trend across {declining} metrics")
        else:
            parts.append(f"Mixed trends: {improving} improving, {declining} declining, {stable} stable")
        
        if critical_anomalies > 0:
            parts.append(f"{critical_anomalies} critical anomalies detected")
        elif high_anomalies > 0:
            parts.append(f"{high_anomalies} high-severity anomalies detected")
        elif len(anomalies) > 0:
            parts.append(f"{len(anomalies)} anomalies detected")
        else:
            parts.append("no anomalies detected")
        
        return "; ".join(parts)
