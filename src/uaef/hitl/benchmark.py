# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark versioning for HITL workflows.

This module provides versioned benchmarks from human-validated review items,
enabling regression testing and tracking evaluation quality over time.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from uaef.hitl.agreement import AgreementCalculator
from uaef.hitl.models import ItemType, ReviewItem
from uaef.logging import get_logger

logger = get_logger(__name__)


class BenchmarkVersioningError(Exception):
    """Base exception for benchmark versioning errors."""
    pass


class BenchmarkNotFoundError(BenchmarkVersioningError):
    """Raised when a benchmark version is not found."""
    pass


class InvalidBenchmarkError(BenchmarkVersioningError):
    """Raised when a benchmark is invalid or corrupted."""
    pass


class BenchmarkMetadata(BaseModel):
    """
    Metadata for a benchmark version.
    
    Attributes:
        version: Unique version identifier (e.g., "v1.0.0", "2024-01-15")
        creation_date: When the benchmark was created
        sample_size: Number of validated items in the benchmark
        agreement_metrics: Agreement statistics for the benchmark
        description: Human-readable description of the benchmark
        metric_names: List of metrics included in the benchmark
        item_types: List of item types included in the benchmark
        tags: Optional tags for categorization
    
    Examples:
        >>> metadata = BenchmarkMetadata(
        ...     version="v1.0.0",
        ...     creation_date=datetime.utcnow(),
        ...     sample_size=100,
        ...     agreement_metrics={
        ...         "answer_relevance": {"krippendorff_alpha": 0.85}
        ...     },
        ...     description="Initial benchmark with high-quality samples",
        ...     metric_names=["answer_relevance", "answer_correctness"],
        ...     item_types=["llm_evaluation"]
        ... )
    """
    
    version: str = Field(
        ...,
        description="Unique version identifier",
        min_length=1
    )
    creation_date: datetime = Field(
        ...,
        description="When the benchmark was created"
    )
    sample_size: int = Field(
        ...,
        description="Number of validated items in the benchmark",
        ge=1
    )
    agreement_metrics: Dict[str, Any] = Field(
        ...,
        description="Agreement statistics for the benchmark"
    )
    description: str = Field(
        ...,
        description="Human-readable description of the benchmark",
        min_length=1
    )
    metric_names: List[str] = Field(
        ...,
        description="List of metrics included in the benchmark"
    )
    item_types: List[str] = Field(
        ...,
        description="List of item types included in the benchmark"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional tags for categorization"
    )
    
    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate that version is not empty."""
        if not v or not v.strip():
            raise ValueError("version cannot be empty")
        return v.strip()
    
    @field_validator("sample_size")
    @classmethod
    def validate_sample_size(cls, v: int) -> int:
        """Validate that sample_size is positive."""
        if v < 1:
            raise ValueError(f"sample_size must be at least 1, got {v}")
        return v
    
    @field_validator("metric_names")
    @classmethod
    def validate_metric_names(cls, v: List[str]) -> List[str]:
        """Validate that metric_names is not empty."""
        if not v:
            raise ValueError("metric_names cannot be empty")
        return v
    
    @field_validator("item_types")
    @classmethod
    def validate_item_types(cls, v: List[str]) -> List[str]:
        """Validate that item_types is not empty."""
        if not v:
            raise ValueError("item_types cannot be empty")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "version": "v1.0.0",
                "creation_date": "2024-01-15T10:00:00Z",
                "sample_size": 100,
                "agreement_metrics": {
                    "answer_relevance": {
                        "krippendorff_alpha": 0.85,
                        "mean_score": 0.78
                    }
                },
                "description": "Initial benchmark with high-quality samples",
                "metric_names": ["answer_relevance", "answer_correctness"],
                "item_types": ["llm_evaluation"],
                "tags": ["production", "validated"]
            }
        }


class Benchmark(BaseModel):
    """
    A versioned benchmark containing human-validated review items.
    
    Attributes:
        benchmark_id: Unique identifier for this benchmark
        metadata: Benchmark metadata (version, date, metrics, etc.)
        items: List of validated review items
    
    Examples:
        >>> benchmark = Benchmark(
        ...     benchmark_id=uuid4(),
        ...     metadata=BenchmarkMetadata(...),
        ...     items=[review_item1, review_item2, ...]
        ... )
    """
    
    benchmark_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this benchmark"
    )
    metadata: BenchmarkMetadata = Field(
        ...,
        description="Benchmark metadata"
    )
    items: List[ReviewItem] = Field(
        ...,
        description="List of validated review items"
    )
    
    @field_validator("items")
    @classmethod
    def validate_items(cls, v: List[ReviewItem]) -> List[ReviewItem]:
        """Validate that items list is not empty."""
        if not v:
            raise ValueError("items cannot be empty")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "benchmark_id": "550e8400-e29b-41d4-a716-446655440000",
                "metadata": {
                    "version": "v1.0.0",
                    "creation_date": "2024-01-15T10:00:00Z",
                    "sample_size": 100,
                    "agreement_metrics": {},
                    "description": "Initial benchmark",
                    "metric_names": ["answer_relevance"],
                    "item_types": ["llm_evaluation"]
                },
                "items": []
            }
        }


class BenchmarkVersioning:
    """
    Manager for versioned benchmarks from human-validated review items.
    
    This class creates and manages versioned benchmarks that can be used for:
    - Regression testing: Ensure evaluation quality doesn't degrade over time
    - LLM judge calibration: Track how well LLM scores align with human judgments
    - Quality tracking: Monitor agreement metrics across benchmark versions
    - Benchmark comparison: Compare evaluation quality between versions
    
    Benchmarks are stored in-memory by default but can be exported/imported
    to/from files for persistence and sharing.
    
    Attributes:
        _benchmarks: Internal storage for benchmarks (version -> Benchmark)
        _agreement_calculator: Calculator for agreement metrics
    
    Examples:
        >>> # Create benchmark versioning manager
        >>> versioning = BenchmarkVersioning()
        
        >>> # Create a new benchmark from validated review items
        >>> benchmark = versioning.create_benchmark(
        ...     version="v1.0.0",
        ...     review_items=validated_items,
        ...     description="Initial production benchmark",
        ...     tags=["production", "validated"]
        ... )
        
        >>> # Retrieve a specific benchmark
        >>> benchmark = versioning.get_benchmark("v1.0.0")
        
        >>> # List all available benchmarks
        >>> benchmarks = versioning.list_benchmarks()
        
        >>> # Compare two benchmarks
        >>> comparison = versioning.compare_benchmarks("v1.0.0", "v2.0.0")
        
        >>> # Export benchmark to file
        >>> versioning.export_benchmark("v1.0.0", "benchmark_v1.json")
        
        >>> # Import benchmark from file
        >>> versioning.import_benchmark("benchmark_v1.json")
    """
    
    def __init__(
        self,
        disagreement_threshold: float = 0.2,
        min_samples: int = 2
    ):
        """
        Initialize the benchmark versioning manager.
        
        Args:
            disagreement_threshold: Threshold for identifying disputed items
            min_samples: Minimum samples required for agreement calculations
        
        Examples:
            >>> # Default settings
            >>> versioning = BenchmarkVersioning()
            
            >>> # Custom settings
            >>> versioning = BenchmarkVersioning(
            ...     disagreement_threshold=0.15,
            ...     min_samples=3
            ... )
        """
        self._benchmarks: Dict[str, Benchmark] = {}
        self._agreement_calculator = AgreementCalculator(
            disagreement_threshold=disagreement_threshold,
            min_samples=min_samples
        )
        
        logger.info(
            f"Initialized BenchmarkVersioning with "
            f"disagreement_threshold={disagreement_threshold}, "
            f"min_samples={min_samples}"
        )
    
    def create_benchmark(
        self,
        version: str,
        review_items: List[ReviewItem],
        description: str,
        tags: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
        item_types: Optional[List[str]] = None
    ) -> Benchmark:
        """
        Create a new benchmark version from validated review items.
        
        This method:
        1. Validates that all review items have human scores
        2. Calculates agreement metrics for each metric
        3. Creates benchmark metadata
        4. Stores the benchmark for future use
        
        Args:
            version: Unique version identifier (e.g., "v1.0.0", "2024-01-15")
            review_items: List of validated review items with human scores
            description: Human-readable description of the benchmark
            tags: Optional tags for categorization
            metric_names: Optional list of metrics to include (defaults to all)
            item_types: Optional list of item types to include (defaults to all)
        
        Returns:
            The created Benchmark object
        
        Raises:
            ValueError: If version already exists or review_items is empty
            InvalidBenchmarkError: If review items are invalid
        
        Examples:
            >>> # Create benchmark with all items
            >>> benchmark = versioning.create_benchmark(
            ...     version="v1.0.0",
            ...     review_items=validated_items,
            ...     description="Initial production benchmark"
            ... )
            
            >>> # Create benchmark with specific metrics
            >>> benchmark = versioning.create_benchmark(
            ...     version="v1.1.0",
            ...     review_items=validated_items,
            ...     description="Benchmark for answer quality metrics",
            ...     metric_names=["answer_relevance", "answer_correctness"],
            ...     tags=["production", "answer_quality"]
            ... )
            
            >>> # Create benchmark with specific item types
            >>> benchmark = versioning.create_benchmark(
            ...     version="v1.2.0",
            ...     review_items=validated_items,
            ...     description="LLM evaluation benchmark",
            ...     item_types=["llm_evaluation"]
            ... )
        """
        # Validate version doesn't already exist
        if version in self._benchmarks:
            raise ValueError(f"Benchmark version '{version}' already exists")
        
        # Validate review_items
        if not review_items:
            raise ValueError("review_items cannot be empty")
        
        # Filter items by type if specified
        if item_types:
            review_items = [
                item for item in review_items
                if item.item_type.value in item_types
            ]
            
            if not review_items:
                raise ValueError(
                    f"No items found matching item_types: {item_types}"
                )
        
        # Validate all items have human scores
        items_without_scores = [
            item for item in review_items
            if not item.human_scores
        ]
        
        if items_without_scores:
            raise InvalidBenchmarkError(
                f"Found {len(items_without_scores)} items without human scores. "
                f"All items must be validated before creating a benchmark."
            )
        
        # Extract all metric names if not specified
        if metric_names is None:
            metric_names_set = set()
            for item in review_items:
                metric_names_set.update(item.human_scores.keys())
            metric_names = sorted(metric_names_set)
        
        # Extract all item types if not specified
        if item_types is None:
            item_types_set = set()
            for item in review_items:
                item_types_set.add(item.item_type.value)
            item_types = sorted(item_types_set)
        
        # Calculate agreement metrics for each metric
        agreement_metrics = {}
        
        for metric_name in metric_names:
            # Filter items that have this metric
            items_with_metric = [
                item for item in review_items
                if metric_name in item.human_scores
            ]
            
            if not items_with_metric:
                logger.warning(
                    f"No items found with metric '{metric_name}', skipping"
                )
                continue
            
            try:
                # Generate agreement report for this metric
                report = self._agreement_calculator.generate_agreement_report(
                    items_with_metric,
                    metric_name,
                    include_llm_correlation=True
                )
                
                # Extract key metrics for metadata
                agreement_metrics[metric_name] = {
                    "inter_rater_agreement": report.get("inter_rater_agreement"),
                    "llm_human_correlation": report.get("llm_human_correlation"),
                    "disputed_count": len(report.get("disputed_items", [])),
                    "overall_quality": report.get("summary", {}).get(
                        "overall_agreement_quality"
                    )
                }
                
            except Exception as e:
                logger.warning(
                    f"Could not calculate agreement for metric '{metric_name}': {e}"
                )
                agreement_metrics[metric_name] = {
                    "error": str(e)
                }
        
        # Create benchmark metadata
        metadata = BenchmarkMetadata(
            version=version,
            creation_date=datetime.utcnow(),
            sample_size=len(review_items),
            agreement_metrics=agreement_metrics,
            description=description,
            metric_names=metric_names,
            item_types=item_types,
            tags=tags or []
        )
        
        # Create benchmark
        benchmark = Benchmark(
            benchmark_id=uuid4(),
            metadata=metadata,
            items=review_items
        )
        
        # Store benchmark
        self._benchmarks[version] = benchmark
        
        logger.info(
            f"Created benchmark version '{version}' with "
            f"{len(review_items)} items, {len(metric_names)} metrics"
        )
        
        return benchmark
    
    def get_benchmark(self, version: str) -> Benchmark:
        """
        Retrieve a specific benchmark version.
        
        Args:
            version: Version identifier of the benchmark to retrieve
        
        Returns:
            The Benchmark object
        
        Raises:
            BenchmarkNotFoundError: If version is not found
        
        Examples:
            >>> benchmark = versioning.get_benchmark("v1.0.0")
            >>> print(f"Sample size: {benchmark.metadata.sample_size}")
            >>> print(f"Metrics: {benchmark.metadata.metric_names}")
        """
        if version not in self._benchmarks:
            raise BenchmarkNotFoundError(
                f"Benchmark version '{version}' not found. "
                f"Available versions: {list(self._benchmarks.keys())}"
            )
        
        return self._benchmarks[version]
    
    def list_benchmarks(
        self,
        metric_name: Optional[str] = None,
        item_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List all available benchmarks with optional filtering.
        
        Args:
            metric_name: Optional filter by metric name
            item_type: Optional filter by item type
            start_date: Optional filter by creation date (inclusive)
            end_date: Optional filter by creation date (inclusive)
            tags: Optional filter by tags (must have all specified tags)
        
        Returns:
            List of benchmark summaries, each containing:
                - version: Version identifier
                - creation_date: When the benchmark was created
                - sample_size: Number of items
                - metric_names: List of metrics
                - item_types: List of item types
                - description: Benchmark description
                - tags: List of tags
        
        Examples:
            >>> # List all benchmarks
            >>> benchmarks = versioning.list_benchmarks()
            >>> for b in benchmarks:
            ...     print(f"{b['version']}: {b['sample_size']} items")
            
            >>> # Filter by metric
            >>> benchmarks = versioning.list_benchmarks(
            ...     metric_name="answer_relevance"
            ... )
            
            >>> # Filter by date range
            >>> benchmarks = versioning.list_benchmarks(
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 12, 31)
            ... )
            
            >>> # Filter by tags
            >>> benchmarks = versioning.list_benchmarks(
            ...     tags=["production", "validated"]
            ... )
        """
        results = []
        
        for version, benchmark in self._benchmarks.items():
            metadata = benchmark.metadata
            
            # Apply filters
            if metric_name and metric_name not in metadata.metric_names:
                continue
            
            if item_type and item_type not in metadata.item_types:
                continue
            
            if start_date and metadata.creation_date < start_date:
                continue
            
            if end_date and metadata.creation_date > end_date:
                continue
            
            if tags and not all(tag in metadata.tags for tag in tags):
                continue
            
            # Add to results
            results.append({
                "version": metadata.version,
                "creation_date": metadata.creation_date.isoformat(),
                "sample_size": metadata.sample_size,
                "metric_names": metadata.metric_names,
                "item_types": metadata.item_types,
                "description": metadata.description,
                "tags": metadata.tags,
                "benchmark_id": str(benchmark.benchmark_id)
            })
        
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x["creation_date"], reverse=True)
        
        logger.info(f"Listed {len(results)} benchmarks")
        
        return results

    
    def compare_benchmarks(
        self,
        version1: str,
        version2: str,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two benchmark versions to track evaluation quality changes.
        
        This method compares agreement metrics, sample sizes, and quality
        indicators between two benchmark versions to identify improvements
        or regressions in evaluation quality.
        
        Args:
            version1: First benchmark version (baseline)
            version2: Second benchmark version (comparison)
            metric_names: Optional list of metrics to compare (defaults to all)
        
        Returns:
            Dictionary containing:
                - version1: First version identifier
                - version2: Second version identifier
                - date1: Creation date of first benchmark
                - date2: Creation date of second benchmark
                - sample_size_change: Change in sample size
                - metric_comparisons: Per-metric comparison results
                    - metric_name: Name of the metric
                    - agreement_change: Change in agreement metrics
                    - quality_change: Change in quality assessment
                - summary: High-level comparison summary
                    - overall_trend: "improved", "degraded", "stable"
                    - significant_changes: List of metrics with significant changes
        
        Raises:
            BenchmarkNotFoundError: If either version is not found
            ValueError: If versions are the same
        
        Examples:
            >>> # Compare two versions
            >>> comparison = versioning.compare_benchmarks("v1.0.0", "v2.0.0")
            >>> print(f"Overall trend: {comparison['summary']['overall_trend']}")
            >>> 
            >>> # Check specific metrics
            >>> for metric_comp in comparison['metric_comparisons']:
            ...     print(f"{metric_comp['metric_name']}: "
            ...           f"{metric_comp['quality_change']}")
            
            >>> # Compare specific metrics only
            >>> comparison = versioning.compare_benchmarks(
            ...     "v1.0.0", "v2.0.0",
            ...     metric_names=["answer_relevance", "answer_correctness"]
            ... )
        """
        # Validate versions
        if version1 == version2:
            raise ValueError("Cannot compare a benchmark version with itself")
        
        benchmark1 = self.get_benchmark(version1)
        benchmark2 = self.get_benchmark(version2)
        
        # Determine which metrics to compare
        if metric_names is None:
            # Use intersection of metrics from both benchmarks
            metrics1 = set(benchmark1.metadata.metric_names)
            metrics2 = set(benchmark2.metadata.metric_names)
            metric_names = sorted(metrics1.intersection(metrics2))
            
            if not metric_names:
                logger.warning(
                    f"No common metrics between {version1} and {version2}"
                )
        
        # Compare each metric
        metric_comparisons = []
        
        for metric_name in metric_names:
            # Get agreement metrics for both versions
            metrics1 = benchmark1.metadata.agreement_metrics.get(metric_name, {})
            metrics2 = benchmark2.metadata.agreement_metrics.get(metric_name, {})
            
            if not metrics1 or not metrics2:
                logger.warning(
                    f"Metric '{metric_name}' not found in one or both benchmarks"
                )
                continue
            
            # Compare inter-rater agreement
            inter_rater1 = metrics1.get("inter_rater_agreement")
            inter_rater2 = metrics2.get("inter_rater_agreement")
            
            agreement_change = {}
            
            if inter_rater1 and inter_rater2:
                alpha1 = inter_rater1.get("krippendorff_alpha", 0.0)
                alpha2 = inter_rater2.get("krippendorff_alpha", 0.0)
                agreement_change["krippendorff_alpha_delta"] = alpha2 - alpha1
                agreement_change["krippendorff_alpha_percent"] = (
                    ((alpha2 - alpha1) / alpha1 * 100) if alpha1 > 0 else 0.0
                )
            
            # Compare LLM-human correlation
            llm_human1 = metrics1.get("llm_human_correlation")
            llm_human2 = metrics2.get("llm_human_correlation")
            
            if llm_human1 and llm_human2:
                pearson1 = llm_human1.get("pearson_correlation", 0.0)
                pearson2 = llm_human2.get("pearson_correlation", 0.0)
                agreement_change["pearson_correlation_delta"] = pearson2 - pearson1
                agreement_change["pearson_correlation_percent"] = (
                    ((pearson2 - pearson1) / abs(pearson1) * 100)
                    if pearson1 != 0 else 0.0
                )
            
            # Compare quality assessments
            quality1 = metrics1.get("overall_quality", "unknown")
            quality2 = metrics2.get("overall_quality", "unknown")
            
            quality_change = self._assess_quality_change(quality1, quality2)
            
            # Compare disputed counts
            disputed1 = metrics1.get("disputed_count", 0)
            disputed2 = metrics2.get("disputed_count", 0)
            
            metric_comparisons.append({
                "metric_name": metric_name,
                "agreement_change": agreement_change,
                "quality_change": quality_change,
                "quality1": quality1,
                "quality2": quality2,
                "disputed_count_change": disputed2 - disputed1
            })
        
        # Calculate sample size change
        sample_size_change = (
            benchmark2.metadata.sample_size - benchmark1.metadata.sample_size
        )
        sample_size_percent = (
            (sample_size_change / benchmark1.metadata.sample_size * 100)
            if benchmark1.metadata.sample_size > 0 else 0.0
        )
        
        # Assess overall trend
        overall_trend, significant_changes = self._assess_overall_trend(
            metric_comparisons
        )
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "date1": benchmark1.metadata.creation_date.isoformat(),
            "date2": benchmark2.metadata.creation_date.isoformat(),
            "sample_size_change": {
                "absolute": sample_size_change,
                "percent": sample_size_percent,
                "size1": benchmark1.metadata.sample_size,
                "size2": benchmark2.metadata.sample_size
            },
            "metric_comparisons": metric_comparisons,
            "summary": {
                "overall_trend": overall_trend,
                "significant_changes": significant_changes,
                "metrics_compared": len(metric_comparisons)
            }
        }
        
        logger.info(
            f"Compared benchmarks {version1} vs {version2}: "
            f"trend={overall_trend}, {len(significant_changes)} significant changes"
        )
        
        return comparison
    
    def _assess_quality_change(self, quality1: str, quality2: str) -> str:
        """
        Assess the change in quality between two benchmarks.
        
        Returns: "improved", "degraded", or "stable"
        """
        quality_order = {
            "poor": 1,
            "fair": 2,
            "good": 3,
            "excellent": 4,
            "unknown": 0
        }
        
        score1 = quality_order.get(quality1, 0)
        score2 = quality_order.get(quality2, 0)
        
        if score2 > score1:
            return "improved"
        elif score2 < score1:
            return "degraded"
        else:
            return "stable"
    
    def _assess_overall_trend(
        self,
        metric_comparisons: List[Dict[str, Any]]
    ) -> tuple[str, List[str]]:
        """
        Assess overall trend across all metric comparisons.
        
        Returns:
            Tuple of (overall_trend, significant_changes)
        """
        if not metric_comparisons:
            return "stable", []
        
        # Count improvements and degradations
        improvements = 0
        degradations = 0
        significant_changes = []
        
        for comp in metric_comparisons:
            quality_change = comp["quality_change"]
            metric_name = comp["metric_name"]
            
            if quality_change == "improved":
                improvements += 1
                significant_changes.append(f"{metric_name}: improved")
            elif quality_change == "degraded":
                degradations += 1
                significant_changes.append(f"{metric_name}: degraded")
            
            # Check for significant agreement changes
            agreement_change = comp.get("agreement_change", {})
            alpha_percent = agreement_change.get("krippendorff_alpha_percent", 0.0)
            
            if abs(alpha_percent) > 10:  # More than 10% change
                if alpha_percent > 0:
                    significant_changes.append(
                        f"{metric_name}: agreement +{alpha_percent:.1f}%"
                    )
                else:
                    significant_changes.append(
                        f"{metric_name}: agreement {alpha_percent:.1f}%"
                    )
        
        # Determine overall trend
        if improvements > degradations:
            overall_trend = "improved"
        elif degradations > improvements:
            overall_trend = "degraded"
        else:
            overall_trend = "stable"
        
        return overall_trend, significant_changes
    
    def export_benchmark(
        self,
        version: str,
        filepath: str
    ) -> None:
        """
        Export a benchmark to a JSON file for sharing or persistence.
        
        The exported file contains the complete benchmark including metadata
        and all review items, allowing it to be imported into another system
        or used for regression testing.
        
        Args:
            version: Version identifier of the benchmark to export
            filepath: Path to the output JSON file
        
        Raises:
            BenchmarkNotFoundError: If version is not found
            IOError: If file cannot be written
        
        Examples:
            >>> # Export to file
            >>> versioning.export_benchmark("v1.0.0", "benchmark_v1.json")
            
            >>> # Export to specific directory
            >>> versioning.export_benchmark(
            ...     "v1.0.0",
            ...     "/path/to/benchmarks/benchmark_v1.json"
            ... )
        """
        benchmark = self.get_benchmark(version)
        
        # Convert to dict for JSON serialization
        benchmark_dict = {
            "benchmark_id": str(benchmark.benchmark_id),
            "metadata": benchmark.metadata.model_dump(mode="json"),
            "items": [item.model_dump(mode="json") for item in benchmark.items]
        }
        
        # Write to file
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w") as f:
                json.dump(benchmark_dict, f, indent=2, default=str)
            
            logger.info(
                f"Exported benchmark '{version}' to {filepath} "
                f"({len(benchmark.items)} items)"
            )
            
        except Exception as e:
            raise IOError(f"Failed to export benchmark to {filepath}: {e}")
    
    def import_benchmark(
        self,
        filepath: str,
        overwrite: bool = False
    ) -> Benchmark:
        """
        Import a benchmark from a JSON file.
        
        This method loads a previously exported benchmark from a file,
        validates it, and adds it to the available benchmarks.
        
        Args:
            filepath: Path to the JSON file containing the benchmark
            overwrite: If True, overwrite existing benchmark with same version
        
        Returns:
            The imported Benchmark object
        
        Raises:
            FileNotFoundError: If file does not exist
            InvalidBenchmarkError: If file is invalid or corrupted
            ValueError: If version already exists and overwrite is False
        
        Examples:
            >>> # Import from file
            >>> benchmark = versioning.import_benchmark("benchmark_v1.json")
            >>> print(f"Imported {benchmark.metadata.version}")
            
            >>> # Import and overwrite existing
            >>> benchmark = versioning.import_benchmark(
            ...     "benchmark_v1.json",
            ...     overwrite=True
            ... )
        """
        # Read file
        try:
            path = Path(filepath)
            
            if not path.exists():
                raise FileNotFoundError(f"Benchmark file not found: {filepath}")
            
            with open(path, "r") as f:
                benchmark_dict = json.load(f)
            
        except json.JSONDecodeError as e:
            raise InvalidBenchmarkError(f"Invalid JSON in {filepath}: {e}")
        except Exception as e:
            raise IOError(f"Failed to read benchmark from {filepath}: {e}")
        
        # Validate and reconstruct benchmark
        try:
            # Reconstruct metadata
            metadata = BenchmarkMetadata(**benchmark_dict["metadata"])
            
            # Reconstruct review items
            items = [
                ReviewItem(**item_dict)
                for item_dict in benchmark_dict["items"]
            ]
            
            # Reconstruct benchmark
            benchmark = Benchmark(
                benchmark_id=UUID(benchmark_dict["benchmark_id"]),
                metadata=metadata,
                items=items
            )
            
        except Exception as e:
            raise InvalidBenchmarkError(
                f"Failed to reconstruct benchmark from {filepath}: {e}"
            )
        
        # Check if version already exists
        version = metadata.version
        
        if version in self._benchmarks and not overwrite:
            raise ValueError(
                f"Benchmark version '{version}' already exists. "
                f"Use overwrite=True to replace it."
            )
        
        # Store benchmark
        self._benchmarks[version] = benchmark
        
        logger.info(
            f"Imported benchmark '{version}' from {filepath} "
            f"({len(items)} items)"
        )
        
        return benchmark
