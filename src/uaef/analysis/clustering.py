# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure clustering for identifying common failure modes."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

try:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    AgglomerativeClustering = None
    KMeans = None
    cosine_similarity = None

from uaef.models.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class FailureCluster(BaseModel):
    """
    A cluster of similar failures.
    
    Attributes:
        cluster_id: Unique identifier for this cluster
        size: Number of failures in this cluster
        evaluation_ids: List of evaluation IDs in this cluster
        common_failures: List of failure messages common to this cluster
        common_metrics: Metrics that commonly failed in this cluster
        severity: Average severity of failures in this cluster
        summary: Human-readable summary of the cluster
        representative_failure: Most representative failure in the cluster
    """
    
    cluster_id: int = Field(..., description="Unique identifier for this cluster")
    size: int = Field(..., description="Number of failures in this cluster")
    evaluation_ids: List[str] = Field(
        default_factory=list,
        description="List of evaluation IDs in this cluster"
    )
    common_failures: List[str] = Field(
        default_factory=list,
        description="List of failure messages common to this cluster"
    )
    common_metrics: List[str] = Field(
        default_factory=list,
        description="Metrics that commonly failed in this cluster"
    )
    severity: str = Field(..., description="Average severity of failures in this cluster")
    summary: str = Field(..., description="Human-readable summary of the cluster")
    representative_failure: Optional[str] = Field(
        None,
        description="Most representative failure in the cluster"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "cluster_id": 0,
                "size": 15,
                "evaluation_ids": ["eval-1", "eval-2", "eval-3"],
                "common_failures": [
                    "tool_calling dimension score 0.45 below threshold 0.70"
                ],
                "common_metrics": ["tool_accuracy", "tool_sequence_accuracy"],
                "severity": "high",
                "summary": "Tool selection errors: Agent consistently selects wrong tools",
                "representative_failure": "tool_calling dimension score 0.45 below threshold 0.70"
            }
        }


class FailureClusterer:
    """
    Clusterer for identifying common failure modes across evaluations.
    
    Provides methods for:
    - Clustering failures based on similarity
    - Supporting multiple clustering algorithms (k-means, hierarchical)
    - Generating cluster summaries
    - Identifying common failure patterns
    """
    
    def __init__(
        self,
        min_cluster_size: int = 2,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize the failure clusterer.
        
        Args:
            min_cluster_size: Minimum number of failures to form a cluster
            similarity_threshold: Threshold for hierarchical clustering (0-1)
        
        Raises:
            ValueError: If parameters are invalid
            ImportError: If scikit-learn is not installed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for failure clustering. "
                "Install it with: pip install scikit-learn>=1.3.0"
            )
        
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        logger.info(
            f"Initialized FailureClusterer with min_cluster_size={min_cluster_size}, "
            f"similarity_threshold={similarity_threshold}"
        )
    
    def cluster_failures(
        self,
        evaluation_results: List[EvaluationResult],
        method: Literal["kmeans", "hierarchical"] = "hierarchical",
        n_clusters: Optional[int] = None
    ) -> List[FailureCluster]:
        """
        Cluster failures based on similarity.
        
        Args:
            evaluation_results: List of evaluation results to cluster
            method: Clustering method ("kmeans" or "hierarchical")
            n_clusters: Number of clusters for k-means (auto-detected if None)
        
        Returns:
            List of failure clusters
        
        Raises:
            ValueError: If inputs are invalid or clustering fails
        """
        try:
            # Filter to only failed evaluations
            failed_results = [r for r in evaluation_results if not r.passed]
            
            if not failed_results:
                logger.info("No failed evaluations to cluster")
                return []
            
            if len(failed_results) < self.min_cluster_size:
                logger.info(
                    f"Only {len(failed_results)} failures, less than min_cluster_size "
                    f"{self.min_cluster_size}. Creating single cluster."
                )
                return [self._create_single_cluster(failed_results)]
            
            logger.info(
                f"Clustering {len(failed_results)} failures using {method} method"
            )
            
            # Create feature vectors for clustering
            feature_vectors, feature_names = self._create_feature_vectors(failed_results)
            
            # Perform clustering
            if method == "kmeans":
                labels = self._cluster_kmeans(feature_vectors, n_clusters)
            elif method == "hierarchical":
                labels = self._cluster_hierarchical(feature_vectors)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Generate cluster summaries
            clusters = self._generate_clusters(
                failed_results,
                labels,
                feature_names
            )
            
            logger.info(f"Created {len(clusters)} failure clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering failures: {e}", exc_info=True)
            raise ValueError(f"Failed to cluster failures: {e}") from e
    
    def _create_feature_vectors(
        self,
        results: List[EvaluationResult]
    ) -> tuple[np.ndarray, List[str]]:
        """
        Create feature vectors for clustering.
        
        Features include:
        - Which dimensions failed
        - Which specific metrics failed
        - Severity of failures (based on scores)
        
        Args:
            results: List of evaluation results
        
        Returns:
            Tuple of (feature_vectors, feature_names)
        """
        # Collect all unique dimension names and metric names
        all_dimensions = set()
        all_metrics = set()
        
        for result in results:
            for dim_result in result.dimension_results:
                all_dimensions.add(dim_result.dimension_name)
                for metric_score in dim_result.metric_scores:
                    all_metrics.add(metric_score.name)
        
        dimension_list = sorted(all_dimensions)
        metric_list = sorted(all_metrics)
        
        # Create feature names
        feature_names = (
            [f"dim_{dim}" for dim in dimension_list] +
            [f"metric_{metric}" for metric in metric_list] +
            ["overall_score", "num_failures"]
        )
        
        # Create feature vectors
        vectors = []
        for result in results:
            vector = []
            
            # Dimension features (1 if dimension failed, 0 otherwise)
            dim_scores = {
                dr.dimension_name: dr.aggregate_score
                for dr in result.dimension_results
            }
            for dim in dimension_list:
                # Consider failed if score < 0.7 (common threshold)
                score = dim_scores.get(dim, 1.0)
                vector.append(1.0 if score < 0.7 else 0.0)
            
            # Metric features (1 if metric failed, 0 otherwise)
            metric_scores = {}
            for dim_result in result.dimension_results:
                for metric_score in dim_result.metric_scores:
                    metric_scores[metric_score.name] = metric_score.score
            
            for metric in metric_list:
                score = metric_scores.get(metric, 1.0)
                vector.append(1.0 if score < 0.7 else 0.0)
            
            # Overall score (inverted so lower scores = higher feature value)
            vector.append(1.0 - result.overall_score)
            
            # Number of failures (normalized)
            vector.append(min(len(result.failures) / 10.0, 1.0))
            
            vectors.append(vector)
        
        return np.array(vectors), feature_names
    
    def _cluster_kmeans(
        self,
        feature_vectors: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Perform k-means clustering.
        
        Args:
            feature_vectors: Feature vectors for clustering
            n_clusters: Number of clusters (auto-detected if None)
        
        Returns:
            Cluster labels for each sample
        """
        n_samples = len(feature_vectors)
        
        # Auto-detect number of clusters if not specified
        if n_clusters is None:
            # Use elbow method heuristic: sqrt(n/2)
            n_clusters = max(2, min(int(np.sqrt(n_samples / 2)), n_samples // 2))
            logger.info(f"Auto-detected n_clusters={n_clusters} for {n_samples} samples")
        
        # Ensure n_clusters is valid
        n_clusters = max(1, min(n_clusters, n_samples))
        
        if n_clusters == 1:
            return np.zeros(n_samples, dtype=int)
        
        # Perform k-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(feature_vectors)
        
        return labels
    
    def _cluster_hierarchical(
        self,
        feature_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Perform hierarchical clustering.
        
        Args:
            feature_vectors: Feature vectors for clustering
        
        Returns:
            Cluster labels for each sample
        """
        n_samples = len(feature_vectors)
        
        if n_samples == 1:
            return np.zeros(1, dtype=int)
        
        # Use distance threshold based on similarity threshold
        # distance = 1 - similarity, so distance_threshold = 1 - similarity_threshold
        distance_threshold = 1.0 - self.similarity_threshold
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="average"
        )
        labels = clustering.fit_predict(feature_vectors)
        
        return labels
    
    def _generate_clusters(
        self,
        results: List[EvaluationResult],
        labels: np.ndarray,
        feature_names: List[str]
    ) -> List[FailureCluster]:
        """
        Generate cluster summaries from clustering results.
        
        Args:
            results: List of evaluation results
            labels: Cluster labels for each result
            feature_names: Names of features used for clustering
        
        Returns:
            List of failure clusters
        """
        clusters = []
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            # Get results in this cluster
            cluster_mask = labels == cluster_id
            cluster_results = [r for i, r in enumerate(results) if cluster_mask[i]]
            
            # Skip if cluster is too small
            if len(cluster_results) < self.min_cluster_size:
                logger.debug(
                    f"Skipping cluster {cluster_id} with only "
                    f"{len(cluster_results)} members"
                )
                continue
            
            # Generate cluster summary
            cluster = self._summarize_cluster(cluster_id, cluster_results)
            clusters.append(cluster)
        
        return clusters
    
    def _summarize_cluster(
        self,
        cluster_id: int,
        results: List[EvaluationResult]
    ) -> FailureCluster:
        """
        Generate a summary for a cluster of failures.
        
        Args:
            cluster_id: ID of the cluster
            results: Evaluation results in this cluster
        
        Returns:
            FailureCluster summary
        """
        # Collect common failures
        failure_counts = defaultdict(int)
        for result in results:
            for failure in result.failures:
                failure_counts[failure] += 1
        
        # Get failures that appear in at least 50% of cluster members
        threshold = len(results) * 0.5
        common_failures = [
            failure for failure, count in failure_counts.items()
            if count >= threshold
        ]
        
        # Collect common metrics
        metric_counts = defaultdict(int)
        for result in results:
            for dim_result in result.dimension_results:
                for metric_score in dim_result.metric_scores:
                    if metric_score.score < 0.7:  # Failed metric
                        metric_counts[metric_score.name] += 1
        
        # Get metrics that failed in at least 50% of cluster members
        common_metrics = [
            metric for metric, count in metric_counts.items()
            if count >= threshold
        ]
        
        # Calculate average severity
        avg_score = np.mean([r.overall_score for r in results])
        if avg_score < 0.4:
            severity = "critical"
        elif avg_score < 0.6:
            severity = "high"
        elif avg_score < 0.8:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate summary
        summary = self._generate_cluster_summary(
            common_failures,
            common_metrics,
            severity
        )
        
        # Get representative failure (most common)
        representative_failure = None
        if common_failures:
            representative_failure = max(
                failure_counts.items(),
                key=lambda x: x[1]
            )[0]
        
        return FailureCluster(
            cluster_id=cluster_id,
            size=len(results),
            evaluation_ids=[str(r.evaluation_id) for r in results],
            common_failures=common_failures,
            common_metrics=common_metrics,
            severity=severity,
            summary=summary,
            representative_failure=representative_failure
        )
    
    def _generate_cluster_summary(
        self,
        common_failures: List[str],
        common_metrics: List[str],
        severity: str
    ) -> str:
        """
        Generate a human-readable summary for a cluster.
        
        Args:
            common_failures: Common failure messages
            common_metrics: Common failed metrics
            severity: Severity level
        
        Returns:
            Human-readable summary
        """
        if not common_metrics and not common_failures:
            return f"Cluster of {severity} severity failures with varied issues"
        
        # Categorize metrics
        tool_metrics = [m for m in common_metrics if "tool" in m.lower()]
        response_metrics = [m for m in common_metrics if any(
            x in m.lower() for x in ["relevance", "completeness", "hallucination"]
        )]
        safety_metrics = [m for m in common_metrics if any(
            x in m.lower() for x in ["safety", "bias", "toxicity"]
        )]
        performance_metrics = [m for m in common_metrics if any(
            x in m.lower() for x in ["latency", "token", "cost"]
        )]
        
        # Generate summary based on dominant category
        if tool_metrics:
            return f"Tool calling issues: Failures in {', '.join(tool_metrics[:3])}"
        elif response_metrics:
            return f"Response quality issues: Failures in {', '.join(response_metrics[:3])}"
        elif safety_metrics:
            return f"Safety concerns: Failures in {', '.join(safety_metrics[:3])}"
        elif performance_metrics:
            return f"Performance issues: Failures in {', '.join(performance_metrics[:3])}"
        else:
            return f"Mixed failures: {', '.join(common_metrics[:3])}"
    
    def _create_single_cluster(
        self,
        results: List[EvaluationResult]
    ) -> FailureCluster:
        """
        Create a single cluster containing all results.
        
        Args:
            results: All evaluation results
        
        Returns:
            Single failure cluster
        """
        return self._summarize_cluster(0, results)
