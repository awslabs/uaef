# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agreement calculation utilities for HITL workflows.

This module provides statistical measures of agreement between reviewers
and between LLM judges and human reviewers. It supports:
- Inter-rater agreement (Krippendorff's alpha, Cohen's kappa)
- LLM-human correlation (Pearson, Spearman)
- Disputed item identification
- Comprehensive agreement reporting
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from scipy import stats

from uaef.hitl.models import AgreementMetrics, ReviewItem
from uaef.logging import get_logger

logger = get_logger(__name__)


class AgreementCalculationError(Exception):
    """Base exception for agreement calculation errors."""
    pass


class InsufficientDataError(AgreementCalculationError):
    """Raised when there is insufficient data for agreement calculation."""
    pass


class AgreementCalculator:
    """
    Calculate agreement metrics for HITL workflows.
    
    This class provides statistical measures of agreement between reviewers
    and between LLM judges and human reviewers. It supports all three HITL
    workflows:
    1. LLM judge calibration - measuring LLM-human agreement
    2. Online evaluation feedback - tracking reviewer consensus
    3. Simulated conversation scoring - validating conversation quality scores
    
    The calculator implements:
    - Inter-rater agreement: Krippendorff's alpha, Cohen's kappa
    - LLM-human correlation: Pearson, Spearman
    - Disputed item identification: Items with low agreement
    - Agreement reporting: Comprehensive statistical summaries
    
    Attributes:
        disagreement_threshold: Score difference threshold for disputes (default: 0.2)
        min_samples: Minimum samples required for calculations (default: 2)
    
    Examples:
        >>> # Calculate inter-rater agreement
        >>> calculator = AgreementCalculator()
        >>> ratings = np.array([[0.8, 0.85, 0.9], [0.5, 0.6, 0.55]])
        >>> alpha = calculator.calculate_krippendorff_alpha(ratings)
        >>> print(f"Krippendorff's alpha: {alpha:.3f}")
        
        >>> # Calculate LLM-human correlation
        >>> llm_scores = [0.8, 0.7, 0.9, 0.6]
        >>> human_scores = [0.85, 0.75, 0.88, 0.65]
        >>> correlation = calculator.calculate_llm_human_correlation(
        ...     llm_scores, human_scores
        ... )
        >>> print(f"Pearson: {correlation['pearson']:.3f}")
        
        >>> # Identify disputed items
        >>> review_items = [...]  # List of ReviewItem objects
        >>> disputed = calculator.identify_disputed_items(
        ...     review_items, metric_name="answer_relevance"
        ... )
        >>> print(f"Found {len(disputed)} disputed items")
        
        >>> # Generate comprehensive report
        >>> report = calculator.generate_agreement_report(
        ...     review_items, metric_name="answer_relevance"
        ... )
        >>> print(f"Inter-rater agreement: {report['inter_rater_agreement']:.3f}")
    """
    
    def __init__(
        self,
        disagreement_threshold: float = 0.2,
        min_samples: int = 2
    ):
        """
        Initialize the agreement calculator.
        
        Args:
            disagreement_threshold: Score difference threshold for disputes.
                Scores differing by more than this are considered disagreements.
                Must be between 0 and 1. Default: 0.2
            min_samples: Minimum number of samples required for calculations.
                Must be at least 2. Default: 2
        
        Raises:
            ValueError: If thresholds are invalid
        
        Examples:
            >>> # Default thresholds
            >>> calculator = AgreementCalculator()
            
            >>> # Custom thresholds
            >>> calculator = AgreementCalculator(
            ...     disagreement_threshold=0.15,
            ...     min_samples=3
            ... )
        """
        if not 0.0 <= disagreement_threshold <= 1.0:
            raise ValueError(
                f"disagreement_threshold must be between 0 and 1, "
                f"got {disagreement_threshold}"
            )
        
        if min_samples < 2:
            raise ValueError(
                f"min_samples must be at least 2, got {min_samples}"
            )
        
        self.disagreement_threshold = disagreement_threshold
        self.min_samples = min_samples
        
        logger.info(
            f"Initialized AgreementCalculator with "
            f"disagreement_threshold={disagreement_threshold}, "
            f"min_samples={min_samples}"
        )

    
    def calculate_krippendorff_alpha(
        self,
        ratings: np.ndarray,
        metric: str = "interval"
    ) -> float:
        """
        Calculate Krippendorff's alpha for inter-rater agreement.
        
        Krippendorff's alpha is a reliability coefficient that measures agreement
        among multiple raters. It handles missing data, works with any number of
        raters, and supports different measurement levels (nominal, ordinal, interval).
        
        Alpha ranges from 0 to 1:
        - 1.0: Perfect agreement
        - 0.8+: Good agreement (generally acceptable)
        - 0.67-0.8: Tentative agreement
        - < 0.67: Low agreement (not reliable)
        
        Args:
            ratings: 2D array of shape (n_items, n_raters) where each row is an item
                and each column is a rater. Use np.nan for missing ratings.
            metric: Distance metric to use. Options:
                - "interval": For continuous scores (default)
                - "nominal": For categorical data
                - "ordinal": For ranked data
        
        Returns:
            Krippendorff's alpha coefficient (0-1)
        
        Raises:
            InsufficientDataError: If fewer than min_samples items
            ValueError: If ratings array is invalid
        
        Examples:
            >>> # Perfect agreement
            >>> ratings = np.array([[0.8, 0.8, 0.8], [0.5, 0.5, 0.5]])
            >>> alpha = calculator.calculate_krippendorff_alpha(ratings)
            >>> print(f"Alpha: {alpha:.3f}")  # 1.0
            
            >>> # With missing data
            >>> ratings = np.array([[0.8, 0.85, np.nan], [0.5, np.nan, 0.55]])
            >>> alpha = calculator.calculate_krippendorff_alpha(ratings)
            
            >>> # Binary data (nominal)
            >>> ratings = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 1]])
            >>> alpha = calculator.calculate_krippendorff_alpha(
            ...     ratings, metric="nominal"
            ... )
        """
        if ratings.ndim != 2:
            raise ValueError(
                f"ratings must be 2D array, got shape {ratings.shape}"
            )
        
        n_items, n_raters = ratings.shape
        
        if n_items < self.min_samples:
            raise InsufficientDataError(
                f"Insufficient items for Krippendorff's alpha: "
                f"need at least {self.min_samples}, got {n_items}"
            )
        
        if n_raters < 2:
            raise ValueError(
                f"Need at least 2 raters, got {n_raters}"
            )
        
        # Calculate observed disagreement
        observed_disagreement = self._calculate_disagreement(ratings, metric)
        
        # Calculate expected disagreement
        expected_disagreement = self._calculate_expected_disagreement(ratings, metric)
        
        # Avoid division by zero
        if expected_disagreement == 0:
            # If expected disagreement is 0, all values are identical
            return 1.0
        
        # Krippendorff's alpha = 1 - (observed / expected)
        alpha = 1.0 - (observed_disagreement / expected_disagreement)
        
        logger.debug(
            f"Krippendorff's alpha: {alpha:.3f} "
            f"(observed={observed_disagreement:.3f}, "
            f"expected={expected_disagreement:.3f})"
        )
        
        return float(alpha)
    
    def _calculate_disagreement(
        self,
        ratings: np.ndarray,
        metric: str
    ) -> float:
        """Calculate observed disagreement for Krippendorff's alpha."""
        n_items, n_raters = ratings.shape
        total_disagreement = 0.0
        total_comparisons = 0
        
        for i in range(n_items):
            # Get non-missing ratings for this item
            item_ratings = ratings[i, ~np.isnan(ratings[i, :])]
            n_valid = len(item_ratings)
            
            if n_valid < 2:
                continue
            
            # Calculate pairwise disagreements
            for j in range(n_valid):
                for k in range(j + 1, n_valid):
                    disagreement = self._distance(
                        item_ratings[j], item_ratings[k], metric
                    )
                    total_disagreement += disagreement
                    total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
        
        return total_disagreement / total_comparisons
    
    def _calculate_expected_disagreement(
        self,
        ratings: np.ndarray,
        metric: str
    ) -> float:
        """Calculate expected disagreement for Krippendorff's alpha."""
        # Flatten all non-missing ratings
        all_ratings = ratings[~np.isnan(ratings)]
        n_total = len(all_ratings)
        
        if n_total < 2:
            return 0.0
        
        # Calculate expected disagreement from all possible pairs
        total_disagreement = 0.0
        total_comparisons = 0
        
        for i in range(n_total):
            for j in range(i + 1, n_total):
                disagreement = self._distance(
                    all_ratings[i], all_ratings[j], metric
                )
                total_disagreement += disagreement
                total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
        
        return total_disagreement / total_comparisons
    
    def _distance(self, value1: float, value2: float, metric: str) -> float:
        """Calculate distance between two values based on metric type."""
        if metric == "nominal":
            # Binary: 0 if same, 1 if different
            return 0.0 if value1 == value2 else 1.0
        elif metric == "ordinal":
            # Rank-based distance
            return abs(value1 - value2)
        elif metric == "interval":
            # Squared difference for interval data
            return (value1 - value2) ** 2
        else:
            raise ValueError(f"Unknown metric: {metric}")

    
    def calculate_cohen_kappa(
        self,
        rater1_scores: List[float],
        rater2_scores: List[float],
        weights: Optional[str] = None
    ) -> float:
        """
        Calculate Cohen's kappa for agreement between two raters.
        
        Cohen's kappa measures agreement between two raters, correcting for
        chance agreement. It's particularly useful for binary or categorical
        ratings but can also be used with continuous scores.
        
        Kappa ranges from -1 to 1:
        - 1.0: Perfect agreement
        - 0.8-1.0: Almost perfect agreement
        - 0.6-0.8: Substantial agreement
        - 0.4-0.6: Moderate agreement
        - 0.2-0.4: Fair agreement
        - 0.0-0.2: Slight agreement
        - < 0: Less than chance agreement
        
        Args:
            rater1_scores: Scores from first rater
            rater2_scores: Scores from second rater (must be same length)
            weights: Weighting scheme for disagreements:
                - None: Unweighted (default)
                - "linear": Linear weights (for ordinal data)
                - "quadratic": Quadratic weights (for interval data)
        
        Returns:
            Cohen's kappa coefficient (-1 to 1)
        
        Raises:
            InsufficientDataError: If fewer than min_samples
            ValueError: If score lists have different lengths
        
        Examples:
            >>> # Perfect agreement
            >>> rater1 = [0.8, 0.5, 0.9, 0.6]
            >>> rater2 = [0.8, 0.5, 0.9, 0.6]
            >>> kappa = calculator.calculate_cohen_kappa(rater1, rater2)
            >>> print(f"Kappa: {kappa:.3f}")  # 1.0
            
            >>> # Moderate agreement
            >>> rater1 = [0.8, 0.5, 0.9, 0.6]
            >>> rater2 = [0.85, 0.55, 0.88, 0.65]
            >>> kappa = calculator.calculate_cohen_kappa(rater1, rater2)
            
            >>> # With quadratic weights (for continuous scores)
            >>> kappa = calculator.calculate_cohen_kappa(
            ...     rater1, rater2, weights="quadratic"
            ... )
        """
        if len(rater1_scores) != len(rater2_scores):
            raise ValueError(
                f"Rater score lists must have same length: "
                f"got {len(rater1_scores)} and {len(rater2_scores)}"
            )
        
        if len(rater1_scores) < self.min_samples:
            raise InsufficientDataError(
                f"Insufficient samples for Cohen's kappa: "
                f"need at least {self.min_samples}, got {len(rater1_scores)}"
            )
        
        # Convert to numpy arrays
        rater1 = np.array(rater1_scores)
        rater2 = np.array(rater2_scores)
        
        # For continuous scores, we need to discretize or use weighted kappa
        # We'll bin the scores into categories for standard kappa
        if weights is None:
            # Discretize continuous scores into bins
            n_bins = min(10, len(rater1_scores))
            rater1_binned = np.digitize(rater1, np.linspace(0, 1, n_bins))
            rater2_binned = np.digitize(rater2, np.linspace(0, 1, n_bins))
            
            # Calculate observed agreement
            observed_agreement = np.mean(rater1_binned == rater2_binned)
            
            # Calculate expected agreement
            unique_bins = np.unique(np.concatenate([rater1_binned, rater2_binned]))
            expected_agreement = 0.0
            
            for bin_val in unique_bins:
                p1 = np.mean(rater1_binned == bin_val)
                p2 = np.mean(rater2_binned == bin_val)
                expected_agreement += p1 * p2
            
        else:
            # Weighted kappa for continuous scores
            observed_agreement = self._calculate_weighted_agreement(
                rater1, rater2, weights
            )
            expected_agreement = self._calculate_expected_weighted_agreement(
                rater1, rater2, weights
            )
        
        # Avoid division by zero
        if expected_agreement >= 1.0:
            return 1.0
        
        # Cohen's kappa = (observed - expected) / (1 - expected)
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        
        logger.debug(
            f"Cohen's kappa: {kappa:.3f} "
            f"(observed={observed_agreement:.3f}, "
            f"expected={expected_agreement:.3f})"
        )
        
        return float(kappa)
    
    def _calculate_weighted_agreement(
        self,
        rater1: np.ndarray,
        rater2: np.ndarray,
        weights: str
    ) -> float:
        """Calculate weighted observed agreement."""
        n = len(rater1)
        total_weight = 0.0
        
        for i in range(n):
            weight = self._get_weight(rater1[i], rater2[i], weights)
            total_weight += weight
        
        return total_weight / n
    
    def _calculate_expected_weighted_agreement(
        self,
        rater1: np.ndarray,
        rater2: np.ndarray,
        weights: str
    ) -> float:
        """Calculate expected weighted agreement."""
        n = len(rater1)
        expected = 0.0
        
        # For continuous data, approximate with histogram
        bins = np.linspace(0, 1, 11)
        hist1, _ = np.histogram(rater1, bins=bins, density=True)
        hist2, _ = np.histogram(rater2, bins=bins, density=True)
        
        # Normalize
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Calculate expected agreement
        for i in range(len(hist1)):
            for j in range(len(hist2)):
                weight = self._get_weight(
                    bins[i], bins[j], weights
                )
                expected += hist1[i] * hist2[j] * weight
        
        return expected
    
    def _get_weight(self, value1: float, value2: float, weights: str) -> float:
        """Get weight for a pair of values based on weighting scheme."""
        diff = abs(value1 - value2)
        
        if weights == "linear":
            # Linear weight: 1 - |diff|
            return 1.0 - diff
        elif weights == "quadratic":
            # Quadratic weight: 1 - diff^2
            return 1.0 - (diff ** 2)
        else:
            # Binary: 1 if same, 0 if different
            return 1.0 if diff < 0.01 else 0.0

    
    def calculate_inter_rater_agreement(
        self,
        review_items: List[ReviewItem],
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Calculate inter-rater agreement between multiple human reviewers.
        
        This method analyzes agreement between multiple human reviewers who
        scored the same items. It calculates both Krippendorff's alpha (for
        multiple raters) and Cohen's kappa (for pairwise comparisons).
        
        Args:
            review_items: List of review items with human scores
            metric_name: Name of the metric to analyze
        
        Returns:
            Dictionary containing:
                - krippendorff_alpha: Agreement across all raters (0-1)
                - pairwise_kappa: Dict of Cohen's kappa for each rater pair
                - n_items: Number of items analyzed
                - n_raters: Number of unique raters
                - mean_score: Mean score across all raters
                - std_score: Standard deviation across all raters
        
        Raises:
            InsufficientDataError: If insufficient data for calculation
            ValueError: If no valid scores found
        
        Examples:
            >>> # Multiple reviewers scored the same items
            >>> review_items = [
            ...     ReviewItem(
            ...         item_id=uuid4(),
            ...         item_type=ItemType.LLM_EVALUATION,
            ...         content={"question": "Q1"},
            ...         priority=Priority.MEDIUM,
            ...         confidence_score=0.7,
            ...         reviewer_id="reviewer_1",
            ...         human_scores={"answer_relevance": 0.8}
            ...     ),
            ...     ReviewItem(
            ...         item_id=uuid4(),
            ...         item_type=ItemType.LLM_EVALUATION,
            ...         content={"question": "Q1"},
            ...         priority=Priority.MEDIUM,
            ...         confidence_score=0.7,
            ...         reviewer_id="reviewer_2",
            ...         human_scores={"answer_relevance": 0.85}
            ...     )
            ... ]
            >>> agreement = calculator.calculate_inter_rater_agreement(
            ...     review_items, "answer_relevance"
            ... )
            >>> print(f"Alpha: {agreement['krippendorff_alpha']:.3f}")
        """
        # Group items by content (same item reviewed by multiple raters)
        item_groups = self._group_items_by_content(review_items, metric_name)
        
        if not item_groups:
            raise ValueError(
                f"No valid scores found for metric '{metric_name}'"
            )
        
        # Build ratings matrix: rows = items, columns = raters
        all_reviewers = set()
        for scores_dict in item_groups.values():
            all_reviewers.update(scores_dict.keys())
        
        reviewer_list = sorted(all_reviewers)
        n_raters = len(reviewer_list)
        n_items = len(item_groups)
        
        if n_items < self.min_samples:
            raise InsufficientDataError(
                f"Insufficient items for inter-rater agreement: "
                f"need at least {self.min_samples}, got {n_items}"
            )
        
        if n_raters < 2:
            raise InsufficientDataError(
                f"Need at least 2 raters for inter-rater agreement, got {n_raters}"
            )
        
        # Create ratings matrix
        ratings = np.full((n_items, n_raters), np.nan)
        
        for item_idx, (item_id, scores_dict) in enumerate(item_groups.items()):
            for reviewer_idx, reviewer_id in enumerate(reviewer_list):
                if reviewer_id in scores_dict:
                    ratings[item_idx, reviewer_idx] = scores_dict[reviewer_id]
        
        # Calculate Krippendorff's alpha
        alpha = self.calculate_krippendorff_alpha(ratings, metric="interval")
        
        # Calculate pairwise Cohen's kappa
        pairwise_kappa = {}
        
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                # Get scores for this pair (excluding missing values)
                mask = ~(np.isnan(ratings[:, i]) | np.isnan(ratings[:, j]))
                
                if mask.sum() >= self.min_samples:
                    rater1_scores = ratings[mask, i].tolist()
                    rater2_scores = ratings[mask, j].tolist()
                    
                    try:
                        kappa = self.calculate_cohen_kappa(
                            rater1_scores, rater2_scores, weights="quadratic"
                        )
                        pair_key = f"{reviewer_list[i]}_vs_{reviewer_list[j]}"
                        pairwise_kappa[pair_key] = kappa
                    except (InsufficientDataError, ValueError) as e:
                        logger.warning(
                            f"Could not calculate kappa for pair "
                            f"{reviewer_list[i]}, {reviewer_list[j]}: {e}"
                        )
        
        # Calculate summary statistics
        all_scores = ratings[~np.isnan(ratings)]
        
        result = {
            "krippendorff_alpha": float(alpha),
            "pairwise_kappa": pairwise_kappa,
            "n_items": n_items,
            "n_raters": n_raters,
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores)),
            "min_score": float(np.min(all_scores)),
            "max_score": float(np.max(all_scores))
        }
        
        logger.info(
            f"Inter-rater agreement for {metric_name}: "
            f"alpha={alpha:.3f}, n_items={n_items}, n_raters={n_raters}"
        )
        
        return result
    
    def _group_items_by_content(
        self,
        review_items: List[ReviewItem],
        metric_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Group review items by content to identify same items reviewed by multiple raters.
        
        Returns:
            Dict mapping item identifier to dict of {reviewer_id: score}
        """
        item_groups = {}
        
        for item in review_items:
            # Skip items without human scores for this metric
            if metric_name not in item.human_scores:
                continue
            
            # Use item_id or content hash as identifier
            # For items reviewed by multiple people, they should have same content
            item_key = str(item.item_id)
            
            # Check if this is a duplicate review (same content, different reviewer)
            # Use content hash for grouping
            content_str = str(sorted(item.content.items()))
            
            # Find existing group with same content
            found_group = None
            for existing_key, scores_dict in item_groups.items():
                # Check if content matches (simple heuristic)
                if content_str in existing_key or existing_key in content_str:
                    found_group = existing_key
                    break
            
            if found_group:
                item_key = found_group
            else:
                item_key = content_str
            
            if item_key not in item_groups:
                item_groups[item_key] = {}
            
            # Add this reviewer's score
            reviewer_id = item.reviewer_id or "unknown"
            item_groups[item_key][reviewer_id] = item.human_scores[metric_name]
        
        # Filter to only items with multiple reviewers
        multi_reviewer_items = {
            k: v for k, v in item_groups.items() if len(v) >= 2
        }
        
        return multi_reviewer_items

    
    def calculate_llm_human_correlation(
        self,
        llm_scores: List[float],
        human_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate correlation between LLM judge and human reviewer scores.
        
        This method computes multiple correlation metrics to measure how well
        LLM scores align with human judgments. It includes both parametric
        (Pearson) and non-parametric (Spearman) correlations, plus error metrics.
        
        Args:
            llm_scores: Scores from LLM judge
            human_scores: Scores from human reviewers (must be same length)
        
        Returns:
            Dictionary containing:
                - pearson_correlation: Pearson correlation coefficient (-1 to 1)
                - pearson_p_value: Statistical significance of Pearson correlation
                - spearman_correlation: Spearman rank correlation (-1 to 1)
                - spearman_p_value: Statistical significance of Spearman correlation
                - mean_absolute_error: Mean absolute difference between scores
                - root_mean_squared_error: RMSE between scores
                - agreement_rate: Percentage within disagreement threshold
                - mean_delta: Mean signed difference (LLM - human)
                - max_delta: Maximum absolute difference
                - std_delta: Standard deviation of differences
                - n_samples: Number of score pairs
        
        Raises:
            InsufficientDataError: If fewer than min_samples
            ValueError: If score lists have different lengths
        
        Examples:
            >>> # High correlation
            >>> llm_scores = [0.8, 0.7, 0.9, 0.6, 0.85]
            >>> human_scores = [0.85, 0.75, 0.88, 0.65, 0.82]
            >>> correlation = calculator.calculate_llm_human_correlation(
            ...     llm_scores, human_scores
            ... )
            >>> print(f"Pearson: {correlation['pearson_correlation']:.3f}")
            >>> print(f"MAE: {correlation['mean_absolute_error']:.3f}")
            
            >>> # Check if correlation is significant
            >>> if correlation['pearson_p_value'] < 0.05:
            ...     print("Correlation is statistically significant")
        """
        if len(llm_scores) != len(human_scores):
            raise ValueError(
                f"Score lists must have same length: "
                f"got {len(llm_scores)} and {len(human_scores)}"
            )
        
        if len(llm_scores) < self.min_samples:
            raise InsufficientDataError(
                f"Insufficient samples for correlation: "
                f"need at least {self.min_samples}, got {len(llm_scores)}"
            )
        
        # Convert to numpy arrays
        llm_arr = np.array(llm_scores)
        human_arr = np.array(human_scores)
        
        # Calculate Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(llm_arr, human_arr)
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(llm_arr, human_arr)
        
        # Calculate error metrics
        score_diffs = np.abs(llm_arr - human_arr)
        mae = np.mean(score_diffs)
        rmse = np.sqrt(np.mean((llm_arr - human_arr) ** 2))
        
        # Calculate agreement rate (within threshold)
        agreements = score_diffs <= self.disagreement_threshold
        agreement_rate = np.mean(agreements)
        
        # Calculate signed differences
        signed_diffs = llm_arr - human_arr
        mean_delta = np.mean(signed_diffs)
        max_delta = np.max(score_diffs)
        std_delta = np.std(signed_diffs)
        
        result = {
            "pearson_correlation": float(pearson_corr),
            "pearson_p_value": float(pearson_p),
            "spearman_correlation": float(spearman_corr),
            "spearman_p_value": float(spearman_p),
            "mean_absolute_error": float(mae),
            "root_mean_squared_error": float(rmse),
            "agreement_rate": float(agreement_rate),
            "mean_delta": float(mean_delta),
            "max_delta": float(max_delta),
            "std_delta": float(std_delta),
            "n_samples": len(llm_scores)
        }
        
        logger.info(
            f"LLM-human correlation: "
            f"pearson={pearson_corr:.3f} (p={pearson_p:.4f}), "
            f"spearman={spearman_corr:.3f} (p={spearman_p:.4f}), "
            f"mae={mae:.3f}"
        )
        
        return result

    
    def identify_disputed_items(
        self,
        review_items: List[ReviewItem],
        metric_name: str,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify items with low agreement (disputed items).
        
        This method finds review items where reviewers significantly disagree,
        either between multiple human reviewers or between LLM and human scores.
        Disputed items may need additional review or dispute resolution.
        
        Args:
            review_items: List of review items to analyze
            metric_name: Name of the metric to check for disputes
            threshold: Disagreement threshold (uses instance default if None)
        
        Returns:
            List of disputed items, each containing:
                - item_id: UUID of the disputed item
                - item_type: Type of review item
                - content: Item content
                - scores: Dict of all scores for this item
                - disagreement_level: Maximum score difference
                - mean_score: Mean of all scores
                - std_score: Standard deviation of scores
                - n_reviewers: Number of reviewers
        
        Examples:
            >>> # Find disputed items
            >>> disputed = calculator.identify_disputed_items(
            ...     review_items, "answer_relevance"
            ... )
            >>> 
            >>> for item in disputed:
            ...     print(f"Item {item['item_id']}: "
            ...           f"disagreement={item['disagreement_level']:.3f}")
            ...     print(f"  Scores: {item['scores']}")
            
            >>> # Use custom threshold
            >>> disputed = calculator.identify_disputed_items(
            ...     review_items, "answer_relevance", threshold=0.15
            ... )
        """
        if threshold is None:
            threshold = self.disagreement_threshold
        
        disputed_items = []
        
        # Group items by content to find items with multiple reviews
        item_groups = self._group_items_by_content(review_items, metric_name)
        
        for item_key, scores_dict in item_groups.items():
            scores = list(scores_dict.values())
            
            if len(scores) < 2:
                continue
            
            # Calculate disagreement level (max difference)
            min_score = min(scores)
            max_score = max(scores)
            disagreement = max_score - min_score
            
            # Check if this is a disputed item
            if disagreement > threshold:
                # Find the original review item for metadata
                matching_item = None
                for item in review_items:
                    if metric_name in item.human_scores:
                        content_str = str(sorted(item.content.items()))
                        if content_str in item_key or item_key in content_str:
                            matching_item = item
                            break
                
                if matching_item:
                    disputed_items.append({
                        "item_id": str(matching_item.item_id),
                        "item_type": matching_item.item_type.value,
                        "content": matching_item.content,
                        "scores": scores_dict,
                        "disagreement_level": float(disagreement),
                        "mean_score": float(np.mean(scores)),
                        "std_score": float(np.std(scores)),
                        "n_reviewers": len(scores),
                        "threshold": threshold
                    })
        
        # Also check for LLM-human disagreements
        for item in review_items:
            if metric_name not in item.human_scores:
                continue
            
            # Check if item has LLM score
            llm_score = item.content.get("llm_scores", {}).get(metric_name)
            if llm_score is None:
                continue
            
            human_score = item.human_scores[metric_name]
            disagreement = abs(llm_score - human_score)
            
            if disagreement > threshold:
                # Check if not already in disputed list
                item_id_str = str(item.item_id)
                if not any(d["item_id"] == item_id_str for d in disputed_items):
                    disputed_items.append({
                        "item_id": item_id_str,
                        "item_type": item.item_type.value,
                        "content": item.content,
                        "scores": {
                            "llm": llm_score,
                            item.reviewer_id or "human": human_score
                        },
                        "disagreement_level": float(disagreement),
                        "mean_score": float((llm_score + human_score) / 2),
                        "std_score": float(np.std([llm_score, human_score])),
                        "n_reviewers": 2,
                        "threshold": threshold,
                        "llm_human_dispute": True
                    })
        
        # Sort by disagreement level (highest first)
        disputed_items.sort(key=lambda x: x["disagreement_level"], reverse=True)
        
        logger.info(
            f"Identified {len(disputed_items)} disputed items for {metric_name} "
            f"(threshold={threshold})"
        )
        
        return disputed_items

    
    def generate_agreement_report(
        self,
        review_items: List[ReviewItem],
        metric_name: str,
        include_llm_correlation: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive agreement report.
        
        This method produces a complete statistical summary of agreement
        across all reviewers and between LLM and human scores. It combines
        inter-rater agreement, LLM-human correlation, and disputed item
        identification into a single comprehensive report.
        
        Args:
            review_items: List of review items to analyze
            metric_name: Name of the metric to analyze
            include_llm_correlation: Whether to include LLM-human correlation
                (requires items with LLM scores)
        
        Returns:
            Dictionary containing:
                - metric_name: Name of the analyzed metric
                - inter_rater_agreement: Inter-rater agreement metrics
                    - krippendorff_alpha: Agreement across all raters
                    - pairwise_kappa: Cohen's kappa for each pair
                    - n_items: Number of items
                    - n_raters: Number of raters
                    - mean_score: Mean score
                    - std_score: Standard deviation
                - llm_human_correlation: LLM-human correlation metrics (if enabled)
                    - pearson_correlation: Pearson coefficient
                    - spearman_correlation: Spearman coefficient
                    - mean_absolute_error: MAE
                    - agreement_rate: Percentage within threshold
                - disputed_items: List of items with low agreement
                - summary: High-level summary statistics
                    - total_items: Total items analyzed
                    - disputed_count: Number of disputed items
                    - dispute_rate: Percentage of disputed items
                    - overall_agreement_quality: "excellent", "good", "fair", "poor"
        
        Raises:
            InsufficientDataError: If insufficient data for analysis
            ValueError: If no valid scores found
        
        Examples:
            >>> # Generate full report
            >>> report = calculator.generate_agreement_report(
            ...     review_items, "answer_relevance"
            ... )
            >>> 
            >>> print(f"Inter-rater alpha: "
            ...       f"{report['inter_rater_agreement']['krippendorff_alpha']:.3f}")
            >>> print(f"LLM-human correlation: "
            ...       f"{report['llm_human_correlation']['pearson_correlation']:.3f}")
            >>> print(f"Disputed items: {report['summary']['disputed_count']}")
            >>> print(f"Overall quality: {report['summary']['overall_agreement_quality']}")
            
            >>> # Report without LLM correlation
            >>> report = calculator.generate_agreement_report(
            ...     review_items, "answer_relevance", include_llm_correlation=False
            ... )
        """
        report = {
            "metric_name": metric_name,
            "timestamp": np.datetime64('now').astype(str)
        }
        
        # Calculate inter-rater agreement
        try:
            inter_rater = self.calculate_inter_rater_agreement(
                review_items, metric_name
            )
            report["inter_rater_agreement"] = inter_rater
        except (InsufficientDataError, ValueError) as e:
            logger.warning(f"Could not calculate inter-rater agreement: {e}")
            report["inter_rater_agreement"] = None
        
        # Calculate LLM-human correlation if requested
        if include_llm_correlation:
            try:
                # Extract LLM and human scores
                llm_scores = []
                human_scores = []
                
                for item in review_items:
                    if metric_name not in item.human_scores:
                        continue
                    
                    llm_score = item.content.get("llm_scores", {}).get(metric_name)
                    if llm_score is None:
                        continue
                    
                    llm_scores.append(llm_score)
                    human_scores.append(item.human_scores[metric_name])
                
                if len(llm_scores) >= self.min_samples:
                    llm_correlation = self.calculate_llm_human_correlation(
                        llm_scores, human_scores
                    )
                    report["llm_human_correlation"] = llm_correlation
                else:
                    logger.warning(
                        f"Insufficient LLM-human pairs for correlation: "
                        f"got {len(llm_scores)}"
                    )
                    report["llm_human_correlation"] = None
            except (InsufficientDataError, ValueError) as e:
                logger.warning(f"Could not calculate LLM-human correlation: {e}")
                report["llm_human_correlation"] = None
        else:
            report["llm_human_correlation"] = None
        
        # Identify disputed items
        try:
            disputed = self.identify_disputed_items(review_items, metric_name)
            report["disputed_items"] = disputed
        except Exception as e:
            logger.warning(f"Could not identify disputed items: {e}")
            report["disputed_items"] = []
        
        # Generate summary
        total_items = len([
            item for item in review_items
            if metric_name in item.human_scores
        ])
        disputed_count = len(report["disputed_items"])
        dispute_rate = disputed_count / total_items if total_items > 0 else 0.0
        
        # Determine overall agreement quality
        quality = self._assess_agreement_quality(report)
        
        report["summary"] = {
            "total_items": total_items,
            "disputed_count": disputed_count,
            "dispute_rate": float(dispute_rate),
            "overall_agreement_quality": quality,
            "disagreement_threshold": self.disagreement_threshold
        }
        
        logger.info(
            f"Generated agreement report for {metric_name}: "
            f"quality={quality}, disputed={disputed_count}/{total_items}"
        )
        
        return report
    
    def _assess_agreement_quality(self, report: Dict[str, Any]) -> str:
        """
        Assess overall agreement quality based on multiple metrics.
        
        Returns: "excellent", "good", "fair", or "poor"
        """
        # Check inter-rater agreement
        inter_rater = report.get("inter_rater_agreement")
        if inter_rater:
            alpha = inter_rater.get("krippendorff_alpha", 0.0)
        else:
            alpha = None
        
        # Check LLM-human correlation
        llm_human = report.get("llm_human_correlation")
        if llm_human:
            pearson = llm_human.get("pearson_correlation", 0.0)
        else:
            pearson = None
        
        # Check dispute rate
        summary = report.get("summary", {})
        dispute_rate = summary.get("dispute_rate", 1.0)
        
        # Assess quality based on available metrics
        scores = []
        
        if alpha is not None:
            if alpha >= 0.8:
                scores.append(4)  # Excellent
            elif alpha >= 0.67:
                scores.append(3)  # Good
            elif alpha >= 0.5:
                scores.append(2)  # Fair
            else:
                scores.append(1)  # Poor
        
        if pearson is not None:
            if pearson >= 0.8:
                scores.append(4)
            elif pearson >= 0.6:
                scores.append(3)
            elif pearson >= 0.4:
                scores.append(2)
            else:
                scores.append(1)
        
        # Dispute rate (inverse scoring)
        if dispute_rate <= 0.1:
            scores.append(4)
        elif dispute_rate <= 0.2:
            scores.append(3)
        elif dispute_rate <= 0.3:
            scores.append(2)
        else:
            scores.append(1)
        
        if not scores:
            return "unknown"
        
        # Average score
        avg_score = np.mean(scores)
        
        if avg_score >= 3.5:
            return "excellent"
        elif avg_score >= 2.5:
            return "good"
        elif avg_score >= 1.5:
            return "fair"
        else:
            return "poor"
