# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt calibration engine for UAEF LLM Judge.

This module provides tools for calibrating LLM judge prompts by comparing
LLM scores with human judgments and suggesting improvements.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from uaef.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationSample:
    """A single calibration sample with LLM and human scores."""
    
    sample_id: str
    metric_name: str
    prompt_version: str
    llm_score: float
    human_score: float
    question: str
    response: str
    llm_reasoning: str
    human_reasoning: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate scores are in valid range."""
        if not (0 <= self.llm_score <= 1):
            raise ValueError(f"LLM score must be between 0 and 1, got {self.llm_score}")
        if not (0 <= self.human_score <= 1):
            raise ValueError(f"Human score must be between 0 and 1, got {self.human_score}")
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def score_difference(self) -> float:
        """Calculate absolute difference between LLM and human scores."""
        return abs(self.llm_score - self.human_score)
    
    @property
    def is_disagreement(self, threshold: float = 0.2) -> bool:
        """Check if LLM and human scores disagree significantly."""
        return self.score_difference > threshold


@dataclass
class CalibrationMetrics:
    """Metrics for prompt calibration."""
    
    metric_name: str
    prompt_version: str
    sample_count: int
    pearson_correlation: float
    pearson_p_value: float
    spearman_correlation: float
    spearman_p_value: float
    mean_absolute_error: float
    root_mean_squared_error: float
    agreement_rate: float  # Percentage within threshold
    disagreement_samples: List[str]  # Sample IDs with high disagreement
    timestamp: datetime
    
    def is_well_calibrated(
        self,
        min_correlation: float = 0.7,
        max_mae: float = 0.15
    ) -> bool:
        """
        Check if prompt is well calibrated.
        
        Args:
            min_correlation: Minimum acceptable correlation
            max_mae: Maximum acceptable mean absolute error
            
        Returns:
            True if well calibrated
        """
        return (
            self.pearson_correlation >= min_correlation and
            self.mean_absolute_error <= max_mae
        )


class CalibrationEngine:
    """
    Engine for calibrating LLM judge prompts.
    
    Provides:
    - Comparison of LLM scores with human judgments
    - Correlation metrics (Pearson, Spearman)
    - Identification of disagreement patterns
    - Prompt improvement suggestions
    - Performance tracking over time
    """
    
    def __init__(self):
        """Initialize calibration engine."""
        self.samples: List[CalibrationSample] = []
        self.calibration_history: List[CalibrationMetrics] = []
        
        logger.info("Initialized CalibrationEngine")
    
    def add_sample(
        self,
        sample_id: str,
        metric_name: str,
        prompt_version: str,
        llm_score: float,
        human_score: float,
        question: str,
        response: str,
        llm_reasoning: str,
        human_reasoning: Optional[str] = None
    ) -> CalibrationSample:
        """
        Add a calibration sample.
        
        Args:
            sample_id: Unique sample identifier
            metric_name: Name of the metric
            prompt_version: Version of the prompt used
            llm_score: Score from LLM judge
            human_score: Score from human reviewer
            question: The question/input
            response: The agent's response
            llm_reasoning: LLM's reasoning
            human_reasoning: Human's reasoning (optional)
            
        Returns:
            CalibrationSample instance
        """
        sample = CalibrationSample(
            sample_id=sample_id,
            metric_name=metric_name,
            prompt_version=prompt_version,
            llm_score=llm_score,
            human_score=human_score,
            question=question,
            response=response,
            llm_reasoning=llm_reasoning,
            human_reasoning=human_reasoning
        )
        
        self.samples.append(sample)
        
        logger.debug(
            f"Added calibration sample: {sample_id}, "
            f"llm_score={llm_score}, human_score={human_score}, "
            f"difference={sample.score_difference:.3f}"
        )
        
        return sample
    
    def calculate_metrics(
        self,
        metric_name: str,
        prompt_version: str,
        disagreement_threshold: float = 0.2
    ) -> CalibrationMetrics:
        """
        Calculate calibration metrics for a specific metric and prompt version.
        
        Args:
            metric_name: Name of the metric
            prompt_version: Version of the prompt
            disagreement_threshold: Threshold for identifying disagreements
            
        Returns:
            CalibrationMetrics instance
            
        Raises:
            ValueError: If insufficient samples
        """
        # Filter samples for this metric and version
        filtered_samples = [
            s for s in self.samples
            if s.metric_name == metric_name and s.prompt_version == prompt_version
        ]
        
        if len(filtered_samples) < 2:
            raise ValueError(
                f"Insufficient samples for calibration: need at least 2, got {len(filtered_samples)}"
            )
        
        # Extract scores
        llm_scores = np.array([s.llm_score for s in filtered_samples])
        human_scores = np.array([s.human_score for s in filtered_samples])
        
        # Calculate Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(llm_scores, human_scores)
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(llm_scores, human_scores)
        
        # Calculate error metrics
        mae = np.mean(np.abs(llm_scores - human_scores))
        rmse = np.sqrt(np.mean((llm_scores - human_scores) ** 2))
        
        # Calculate agreement rate
        agreements = np.abs(llm_scores - human_scores) <= disagreement_threshold
        agreement_rate = np.mean(agreements)
        
        # Identify disagreement samples
        disagreement_samples = [
            s.sample_id for s in filtered_samples
            if s.is_disagreement(disagreement_threshold)
        ]
        
        metrics = CalibrationMetrics(
            metric_name=metric_name,
            prompt_version=prompt_version,
            sample_count=len(filtered_samples),
            pearson_correlation=float(pearson_corr),
            pearson_p_value=float(pearson_p),
            spearman_correlation=float(spearman_corr),
            spearman_p_value=float(spearman_p),
            mean_absolute_error=float(mae),
            root_mean_squared_error=float(rmse),
            agreement_rate=float(agreement_rate),
            disagreement_samples=disagreement_samples,
            timestamp=datetime.now()
        )
        
        # Add to history
        self.calibration_history.append(metrics)
        
        logger.info(
            f"Calculated calibration metrics for {metric_name} v{prompt_version}: "
            f"pearson={pearson_corr:.3f}, spearman={spearman_corr:.3f}, "
            f"mae={mae:.3f}, agreement_rate={agreement_rate:.1%}"
        )
        
        return metrics
    
    def analyze_disagreements(
        self,
        metric_name: str,
        prompt_version: str,
        disagreement_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Analyze disagreement patterns between LLM and human scores.
        
        Args:
            metric_name: Name of the metric
            prompt_version: Version of the prompt
            disagreement_threshold: Threshold for identifying disagreements
            
        Returns:
            Dictionary with disagreement analysis
        """
        # Filter samples
        filtered_samples = [
            s for s in self.samples
            if s.metric_name == metric_name and s.prompt_version == prompt_version
        ]
        
        if not filtered_samples:
            return {
                "total_samples": 0,
                "disagreement_count": 0,
                "disagreement_rate": 0.0,
                "patterns": []
            }
        
        # Identify disagreements
        disagreements = [
            s for s in filtered_samples
            if s.is_disagreement(disagreement_threshold)
        ]
        
        # Analyze patterns
        patterns = []
        
        # Pattern 1: LLM consistently higher than human
        llm_higher = [s for s in disagreements if s.llm_score > s.human_score]
        if len(llm_higher) > len(disagreements) * 0.6:
            patterns.append({
                "pattern": "llm_overscores",
                "description": "LLM tends to give higher scores than humans",
                "count": len(llm_higher),
                "percentage": len(llm_higher) / len(disagreements),
                "avg_difference": np.mean([s.llm_score - s.human_score for s in llm_higher])
            })
        
        # Pattern 2: LLM consistently lower than human
        llm_lower = [s for s in disagreements if s.llm_score < s.human_score]
        if len(llm_lower) > len(disagreements) * 0.6:
            patterns.append({
                "pattern": "llm_underscores",
                "description": "LLM tends to give lower scores than humans",
                "count": len(llm_lower),
                "percentage": len(llm_lower) / len(disagreements),
                "avg_difference": np.mean([s.human_score - s.llm_score for s in llm_lower])
            })
        
        # Pattern 3: High variance in disagreements
        score_diffs = [s.score_difference for s in disagreements]
        if np.std(score_diffs) > 0.15:
            patterns.append({
                "pattern": "high_variance",
                "description": "Inconsistent disagreements with high variance",
                "std_dev": float(np.std(score_diffs)),
                "max_difference": float(np.max(score_diffs))
            })
        
        return {
            "total_samples": len(filtered_samples),
            "disagreement_count": len(disagreements),
            "disagreement_rate": len(disagreements) / len(filtered_samples),
            "patterns": patterns,
            "disagreement_samples": [
                {
                    "sample_id": s.sample_id,
                    "llm_score": s.llm_score,
                    "human_score": s.human_score,
                    "difference": s.score_difference,
                    "question": s.question[:100],
                    "response": s.response[:100]
                }
                for s in disagreements[:10]  # Limit to first 10
            ]
        }
    
    def suggest_improvements(
        self,
        metric_name: str,
        prompt_version: str,
        disagreement_threshold: float = 0.2
    ) -> List[str]:
        """
        Suggest prompt improvements based on disagreement analysis.
        
        Args:
            metric_name: Name of the metric
            prompt_version: Version of the prompt
            disagreement_threshold: Threshold for identifying disagreements
            
        Returns:
            List of improvement suggestions
        """
        # Get disagreement analysis
        analysis = self.analyze_disagreements(
            metric_name=metric_name,
            prompt_version=prompt_version,
            disagreement_threshold=disagreement_threshold
        )
        
        suggestions = []
        
        # Check disagreement rate
        if analysis["disagreement_rate"] > 0.3:
            suggestions.append(
                f"High disagreement rate ({analysis['disagreement_rate']:.1%}). "
                "Consider revising prompt for clearer evaluation criteria."
            )
        
        # Check patterns
        for pattern in analysis["patterns"]:
            if pattern["pattern"] == "llm_overscores":
                suggestions.append(
                    f"LLM tends to overscore (avg difference: {pattern['avg_difference']:.2f}). "
                    "Add stricter criteria or examples of lower-quality responses."
                )
            elif pattern["pattern"] == "llm_underscores":
                suggestions.append(
                    f"LLM tends to underscore (avg difference: {pattern['avg_difference']:.2f}). "
                    "Clarify what constitutes good performance or add positive examples."
                )
            elif pattern["pattern"] == "high_variance":
                suggestions.append(
                    f"Inconsistent scoring (std dev: {pattern['std_dev']:.2f}). "
                    "Add more specific scoring guidelines or rubric."
                )
        
        # Check correlation
        try:
            metrics = self.calculate_metrics(metric_name, prompt_version)
            if metrics.pearson_correlation < 0.5:
                suggestions.append(
                    f"Low correlation with human judgments ({metrics.pearson_correlation:.2f}). "
                    "Prompt may not be capturing the right evaluation criteria."
                )
        except ValueError:
            pass
        
        if not suggestions:
            suggestions.append("Prompt appears well-calibrated. No major improvements needed.")
        
        logger.info(
            f"Generated {len(suggestions)} improvement suggestions for "
            f"{metric_name} v{prompt_version}"
        )
        
        return suggestions
    
    def track_performance(
        self,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Track prompt performance over time across versions.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary with performance tracking data
        """
        # Filter history for this metric
        metric_history = [
            m for m in self.calibration_history
            if m.metric_name == metric_name
        ]
        
        if not metric_history:
            return {
                "metric_name": metric_name,
                "versions": [],
                "trend": "no_data"
            }
        
        # Sort by timestamp
        metric_history.sort(key=lambda m: m.timestamp)
        
        # Extract version data
        versions = []
        for metrics in metric_history:
            versions.append({
                "version": metrics.prompt_version,
                "timestamp": metrics.timestamp.isoformat(),
                "sample_count": metrics.sample_count,
                "pearson_correlation": metrics.pearson_correlation,
                "spearman_correlation": metrics.spearman_correlation,
                "mae": metrics.mean_absolute_error,
                "rmse": metrics.root_mean_squared_error,
                "agreement_rate": metrics.agreement_rate,
                "is_well_calibrated": metrics.is_well_calibrated()
            })
        
        # Determine trend
        if len(versions) >= 2:
            recent_corr = versions[-1]["pearson_correlation"]
            previous_corr = versions[-2]["pearson_correlation"]
            
            if recent_corr > previous_corr + 0.05:
                trend = "improving"
            elif recent_corr < previous_corr - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "metric_name": metric_name,
            "versions": versions,
            "trend": trend,
            "latest_version": versions[-1] if versions else None
        }
    
    def get_samples(
        self,
        metric_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        min_disagreement: Optional[float] = None
    ) -> List[CalibrationSample]:
        """
        Get calibration samples with optional filtering.
        
        Args:
            metric_name: Filter by metric name (optional)
            prompt_version: Filter by prompt version (optional)
            min_disagreement: Minimum disagreement threshold (optional)
            
        Returns:
            List of CalibrationSample instances
        """
        samples = self.samples
        
        if metric_name:
            samples = [s for s in samples if s.metric_name == metric_name]
        
        if prompt_version:
            samples = [s for s in samples if s.prompt_version == prompt_version]
        
        if min_disagreement is not None:
            samples = [s for s in samples if s.score_difference >= min_disagreement]
        
        return samples
    
    def export_samples(
        self,
        metric_name: Optional[str] = None,
        prompt_version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Export calibration samples as dictionaries.
        
        Args:
            metric_name: Filter by metric name (optional)
            prompt_version: Filter by prompt version (optional)
            
        Returns:
            List of sample dictionaries
        """
        samples = self.get_samples(metric_name, prompt_version)
        
        return [
            {
                "sample_id": s.sample_id,
                "metric_name": s.metric_name,
                "prompt_version": s.prompt_version,
                "llm_score": s.llm_score,
                "human_score": s.human_score,
                "score_difference": s.score_difference,
                "question": s.question,
                "response": s.response,
                "llm_reasoning": s.llm_reasoning,
                "human_reasoning": s.human_reasoning,
                "timestamp": s.timestamp.isoformat()
            }
            for s in samples
        ]
