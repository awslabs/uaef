# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline batch evaluation mode for UAEF.

This module provides batch evaluation functionality with:
- Parallel processing of multiple traces
- Batch LLM API calls for efficiency
- Aggregate statistics across batch
- Configurable parallelism
- Graceful error handling
- Export to various formats
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from uaef.evaluation.base_evaluator import BaseEvaluator
from uaef.logging import get_logger
from uaef.models.agent_trace import AgentTrace
from uaef.models.evaluation_result import EvaluationResult
from uaef.models.ground_truth import GroundTruth

logger = get_logger(__name__)


class BatchStatistics:
    """
    Aggregate statistics for a batch evaluation.
    """
    
    def __init__(self, results: List[EvaluationResult]):
        """
        Initialize batch statistics from evaluation results.
        
        Args:
            results: List of evaluation results
        """
        self.total_count = len(results)
        self.passed_count = sum(1 for r in results if r.passed)
        self.failed_count = self.total_count - self.passed_count
        self.pass_rate = self.passed_count / self.total_count if self.total_count > 0 else 0.0
        
        # Calculate score statistics
        scores = [r.overall_score for r in results]
        self.mean_score = sum(scores) / len(scores) if scores else 0.0
        self.min_score = min(scores) if scores else 0.0
        self.max_score = max(scores) if scores else 0.0
        
        # Calculate dimension statistics
        self.dimension_stats = self._calculate_dimension_stats(results)
        
        # Calculate metric statistics
        self.metric_stats = self._calculate_metric_stats(results)
    
    def _calculate_dimension_stats(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics per dimension."""
        dimension_scores: Dict[str, List[float]] = {}
        
        for result in results:
            for dim_result in result.dimension_results:
                dim_name = dim_result.dimension_name
                if dim_name not in dimension_scores:
                    dimension_scores[dim_name] = []
                dimension_scores[dim_name].append(dim_result.aggregate_score)
        
        stats = {}
        for dim_name, scores in dimension_scores.items():
            stats[dim_name] = {
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "count": len(scores)
            }
        
        return stats
    
    def _calculate_metric_stats(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics per metric."""
        metric_scores: Dict[str, List[float]] = {}
        
        for result in results:
            for dim_result in result.dimension_results:
                for metric_score in dim_result.metric_scores:
                    metric_name = metric_score.metric_name
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(metric_score.score)
        
        stats = {}
        for metric_name, scores in metric_scores.items():
            stats[metric_name] = {
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "count": len(scores)
            }
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_count": self.total_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "dimension_stats": self.dimension_stats,
            "metric_stats": self.metric_stats
        }


class OfflineEvaluator(BaseEvaluator):
    """
    Evaluator for offline/batch evaluation.
    
    Optimized for throughput and efficiency with:
    - Parallel processing of multiple traces
    - Batch LLM API calls
    - Aggregate statistics calculation
    - Configurable parallelism
    - Graceful error handling
    - Export to various formats
    """
    
    def __init__(
        self,
        metric_registry: Optional[any] = None,
        max_workers: int = 8,
        batch_size: int = 10,
        default_dimension_weights: Optional[Dict[str, float]] = None,
        default_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize offline evaluator.
        
        Args:
            metric_registry: MetricRegistry instance (uses global if None)
            max_workers: Maximum number of parallel workers for trace processing
            batch_size: Number of traces to process in each batch
            default_dimension_weights: Default weights for dimensions
            default_thresholds: Default threshold values for dimensions
        """
        super().__init__(metric_registry=metric_registry, max_workers=max_workers)
        
        self.batch_size = batch_size
        
        # Set default dimension weights
        self.default_dimension_weights = default_dimension_weights or {
            "tool_calling": 0.25,
            "response_quality": 0.25,
            "responsible_ai": 0.20,
            "performance": 0.10,
            "multi_turn": 0.10,
            "multi_agent": 0.05,
            "reasoning": 0.05
        }
        
        # Set default thresholds
        self.default_thresholds = default_thresholds or {
            "tool_calling": 0.7,
            "response_quality": 0.7,
            "responsible_ai": 0.8,
            "performance": 0.6,
            "multi_turn": 0.6,
            "multi_agent": 0.6,
            "reasoning": 0.6
        }
        
        logger.info(
            f"Initialized OfflineEvaluator with max_workers={max_workers}, "
            f"batch_size={batch_size}"
        )
    
    def evaluate(
        self,
        trace: AgentTrace,
        ground_truth: Optional[GroundTruth] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single trace (delegates to batch_evaluate).
        
        Args:
            trace: Agent trace to evaluate
            ground_truth: Expected correct outputs (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names
            dimension_weights: Weights for each dimension (uses defaults if None)
            thresholds: Threshold values for pass/fail (uses defaults if None)
            context: List of context documents (optional)
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with scores and metadata
        """
        results = self.batch_evaluate(
            traces=[trace],
            ground_truths=[ground_truth] if ground_truth else None,
            metric_set=metric_set,
            dimension_weights=dimension_weights,
            thresholds=thresholds,
            contexts=[context] if context else None,
            **kwargs
        )
        
        return results[0]
    
    def evaluate_dataset(
        self,
        dataset: List[Tuple[AgentTrace, Optional[GroundTruth]]],
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        contexts: Optional[List[Optional[List[str]]]] = None,
        **kwargs
    ) -> Tuple[List[EvaluationResult], BatchStatistics]:
        """
        Evaluate a complete dataset of traces.
        
        Args:
            dataset: List of (trace, ground_truth) tuples
            metric_set: Dictionary mapping dimension names to lists of metric names
            dimension_weights: Weights for each dimension (uses defaults if None)
            thresholds: Threshold values for pass/fail (uses defaults if None)
            contexts: List of context documents for each trace (optional)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Tuple of (results, statistics)
        """
        logger.info(f"Evaluating dataset with {len(dataset)} traces")
        
        start_time = time.time()
        
        # Separate traces and ground truths
        traces = [item[0] for item in dataset]
        ground_truths = [item[1] for item in dataset]
        
        # Evaluate batch
        results = self.batch_evaluate(
            traces=traces,
            ground_truths=ground_truths,
            metric_set=metric_set,
            dimension_weights=dimension_weights,
            thresholds=thresholds,
            contexts=contexts,
            **kwargs
        )
        
        # Calculate statistics
        statistics = BatchStatistics(results)
        
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"Dataset evaluation complete: {len(results)} traces in {elapsed_time:.1f}s "
            f"({len(results)/elapsed_time:.1f} traces/sec), "
            f"pass_rate={statistics.pass_rate:.2%}"
        )
        
        return results, statistics
    
    def batch_evaluate(
        self,
        traces: List[AgentTrace],
        ground_truths: Optional[List[Optional[GroundTruth]]] = None,
        metric_set: Optional[Dict[str, List[str]]] = None,
        dimension_weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        contexts: Optional[List[Optional[List[str]]]] = None,
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple traces in parallel with batch optimization.
        
        Args:
            traces: List of agent traces to evaluate
            ground_truths: List of ground truths (optional)
            metric_set: Dictionary mapping dimension names to lists of metric names
            dimension_weights: Weights for each dimension (uses defaults if None)
            thresholds: Threshold values for pass/fail (uses defaults if None)
            contexts: List of context documents for each trace (optional)
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"Starting batch evaluation for {len(traces)} traces")
        
        start_time = time.time()
        
        # Validate inputs
        if not traces:
            raise ValueError("traces cannot be empty")
        
        if ground_truths is None:
            ground_truths = [None] * len(traces)
        
        if contexts is None:
            contexts = [None] * len(traces)
        
        if len(traces) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: {len(traces)} traces but {len(ground_truths)} ground truths"
            )
        
        if len(traces) != len(contexts):
            raise ValueError(
                f"Length mismatch: {len(traces)} traces but {len(contexts)} contexts"
            )
        
        # Use default weights and thresholds if not provided
        dimension_weights = dimension_weights or self.default_dimension_weights
        thresholds = thresholds or self.default_thresholds
        
        # Process traces in parallel
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all trace evaluations
            future_to_index = {}
            
            for i, (trace, gt, ctx) in enumerate(zip(traces, ground_truths, contexts)):
                future = executor.submit(
                    self._evaluate_single_trace,
                    trace=trace,
                    ground_truth=gt,
                    metric_set=metric_set,
                    dimension_weights=dimension_weights,
                    thresholds=thresholds,
                    context=ctx,
                    **kwargs
                )
                future_to_index[future] = i
            
            # Collect results in order
            results_dict = {}
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results_dict[index] = result
                    
                    if (index + 1) % 10 == 0:
                        logger.debug(f"Completed {index + 1}/{len(traces)} evaluations")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate trace {index}: {str(e)}")
                    # Create failed result
                    results_dict[index] = self._create_failed_result(
                        traces[index],
                        str(e)
                    )
            
            # Sort results by index
            results = [results_dict[i] for i in range(len(traces))]
        
        elapsed_time = time.time() - start_time
        success_count = sum(1 for r in results if r.passed)
        
        logger.info(
            f"Batch evaluation complete: {success_count}/{len(results)} passed "
            f"in {elapsed_time:.1f}s ({len(results)/elapsed_time:.1f} traces/sec)"
        )
        
        return results
    
    def _evaluate_single_trace(
        self,
        trace: AgentTrace,
        ground_truth: Optional[GroundTruth],
        metric_set: Optional[Dict[str, List[str]]],
        dimension_weights: Dict[str, float],
        thresholds: Dict[str, float],
        context: Optional[List[str]],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single trace (used by parallel workers).

        This is a wrapper around the evaluation logic that can be called
        by ThreadPoolExecutor workers.
        """
        from uaef.evaluation.multi_agent import MultiAgentEvaluator
        from uaef.evaluation.single_agent import SingleAgentEvaluator
        from uaef.models.multi_agent_trace import MultiAgentTrace

        if isinstance(trace, MultiAgentTrace):
            evaluator = MultiAgentEvaluator(
                metric_registry=self.metric_registry,
                max_workers=4,
                default_dimension_weights=dimension_weights,
                default_thresholds=thresholds
            )
        else:
            evaluator = SingleAgentEvaluator(
                metric_registry=self.metric_registry,
                max_workers=4,
                default_dimension_weights=dimension_weights,
                default_thresholds=thresholds
            )

        return evaluator.evaluate(
            trace=trace,
            ground_truth=ground_truth,
            metric_set=metric_set,
            dimension_weights=dimension_weights,
            thresholds=thresholds,
            context=context,
            **kwargs
        )
    
    def export_results(
        self,
        results: List[EvaluationResult],
        statistics: Optional[BatchStatistics],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export evaluation results to file.
        
        Args:
            results: List of evaluation results
            statistics: Batch statistics (optional)
            output_path: Path to output file
            format: Export format ("json", "csv", "html")
            
        Raises:
            ValueError: If format is not supported
        """
        logger.info(f"Exporting {len(results)} results to {output_path} ({format})")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            self._export_json(results, statistics, output_file)
        elif format == "csv":
            self._export_csv(results, statistics, output_file)
        elif format == "html":
            self._export_html(results, statistics, output_file)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {output_path}")
    
    def _export_json(
        self,
        results: List[EvaluationResult],
        statistics: Optional[BatchStatistics],
        output_file: Path
    ) -> None:
        """Export results to JSON format."""
        data = {
            "results": [
                {
                    "trace_id": str(r.trace_id),
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                    "failures": r.failures,
                    "warnings": r.warnings,
                    "dimensions": [
                        {
                            "name": d.dimension_name,
                            "score": d.aggregate_score,
                            "weight": d.weight,
                            "metrics": [
                                {
                                    "name": m.metric_name,
                                    "score": m.score,
                                    "reasoning": m.reasoning
                                }
                                for m in d.metric_scores
                            ]
                        }
                        for d in r.dimension_results
                    ],
                    "metadata": r.metadata,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        }
        
        if statistics:
            data["statistics"] = statistics.to_dict()
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(
        self,
        results: List[EvaluationResult],
        statistics: Optional[BatchStatistics],
        output_file: Path
    ) -> None:
        """Export results to CSV format."""
        import csv
        
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            header = ["trace_id", "overall_score", "passed"]
            
            # Add dimension columns
            if results:
                for dim in results[0].dimension_results:
                    header.append(f"{dim.dimension_name}_score")
            
            writer.writerow(header)
            
            # Write data rows
            for result in results:
                row = [
                    str(result.trace_id),
                    f"{result.overall_score:.3f}",
                    str(result.passed)
                ]
                
                for dim in result.dimension_results:
                    row.append(f"{dim.aggregate_score:.3f}")
                
                writer.writerow(row)
            
            # Write statistics if available
            if statistics:
                writer.writerow([])
                writer.writerow(["Statistics"])
                writer.writerow(["Total Count", statistics.total_count])
                writer.writerow(["Passed Count", statistics.passed_count])
                writer.writerow(["Failed Count", statistics.failed_count])
                writer.writerow(["Pass Rate", f"{statistics.pass_rate:.2%}"])
                writer.writerow(["Mean Score", f"{statistics.mean_score:.3f}"])
                writer.writerow(["Min Score", f"{statistics.min_score:.3f}"])
                writer.writerow(["Max Score", f"{statistics.max_score:.3f}"])
    
    def _export_html(
        self,
        results: List[EvaluationResult],
        statistics: Optional[BatchStatistics],
        output_file: Path
    ) -> None:
        """Export results to HTML format."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>UAEF Evaluation Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .stats { background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .passed { color: green; font-weight: bold; }
        .failed { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>UAEF Evaluation Results</h1>
"""
        
        # Add statistics
        if statistics:
            html += f"""
    <div class="stats">
        <h2>Statistics</h2>
        <p><strong>Total Count:</strong> {statistics.total_count}</p>
        <p><strong>Passed:</strong> {statistics.passed_count} ({statistics.pass_rate:.1%})</p>
        <p><strong>Failed:</strong> {statistics.failed_count}</p>
        <p><strong>Mean Score:</strong> {statistics.mean_score:.3f}</p>
        <p><strong>Score Range:</strong> {statistics.min_score:.3f} - {statistics.max_score:.3f}</p>
    </div>
"""
        
        # Add results table
        html += """
    <h2>Results</h2>
    <table>
        <tr>
            <th>Trace ID</th>
            <th>Overall Score</th>
            <th>Status</th>
            <th>Dimensions</th>
        </tr>
"""
        
        for result in results:
            status_class = "passed" if result.passed else "failed"
            status_text = "PASSED" if result.passed else "FAILED"
            
            dimensions_html = "<br>".join([
                f"{d.dimension_name}: {d.aggregate_score:.3f}"
                for d in result.dimension_results
            ])
            
            html += f"""
        <tr>
            <td>{result.trace_id}</td>
            <td>{result.overall_score:.3f}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{dimensions_html}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        with open(output_file, "w") as f:
            f.write(html)
