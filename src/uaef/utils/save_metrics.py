# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility for saving and loading evaluation metric results.

Uses the same serialization format as the S3 persistence layer
(EvaluationResult.model_dump(mode="json")), ensuring that results
saved locally can be loaded identically to those retrieved from S3.

The JSON format is:
    {
        "experiment_name": "...",
        "evaluations": [ {evaluation_result_dict}, ... ]
    }

This matches the structure written by DynamoS3Storage._write_s3_results().
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from uaef.utils.constants import BATCH_EVAL_OUTPUT_DIR


def save_metric_results(
    results,
    test_cases: Optional[List[Dict]] = None,
    *,
    prefix: str = "batch_results",
    output_dir: Optional[str] = None,
    query_key: str = "query",
    expected_key: str = "expected_output",
    experiment_name: Optional[str] = None,
) -> str:
    """
    Save evaluation metric results to timestamped JSON and CSV files.

    The JSON uses the same format as DynamoS3Storage (with an "evaluations"
    wrapper), ensuring local saves and S3 downloads are interchangeable.
    The CSV is a flat, human-readable companion artifact.

    Args:
        results: List of EvaluationResult objects from evaluate() or batch_evaluate()
        test_cases: Optional list of test case dicts for the CSV (not required for JSON)
        prefix: Filename prefix (default: "batch_results")
        output_dir: Output directory (default: BATCH_EVAL_OUTPUT_DIR)
        query_key: Key in test_cases for the query text (CSV only)
        expected_key: Key in test_cases for the expected output text (CSV only)
        experiment_name: Optional name stored in the JSON metadata

    Returns:
        Path to the saved JSON file (use this path with load_metric_results)

    Examples:
        >>> from uaef.utils import save_metric_results
        >>> filepath = save_metric_results(results, test_cases, prefix="strands_weather")
        >>> # Or without test_cases (JSON-only, no CSV):
        >>> filepath = save_metric_results(results, prefix="my_run")
    """
    output_dir = output_dir or BATCH_EVAL_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{timestamp}"

    # --- Save JSON (source of truth, same format as S3) ---
    json_path = os.path.join(output_dir, f"{base_name}.json")
    from uaef.utils.serialization import serialize_results
    json_data = {
        "experiment_name": experiment_name or prefix,
        "saved_at": datetime.now().isoformat(),
        "evaluation_count": len(results),
        "evaluations": serialize_results(results),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)

    print(f"\u2713 Results saved to {json_path}")

    # --- Save CSV (human-readable companion) ---
    if test_cases is not None:
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        rows = []
        for i, r in enumerate(results):
            row = {
                "query_id": i + 1,
                "query": str(test_cases[i].get(query_key, "")) if i < len(test_cases) else "",
                "expected_output": str(test_cases[i].get(expected_key, "")) if i < len(test_cases) else "",
                "overall_score": r.overall_score,
                "passed": r.passed,
            }
            for dim in r.dimension_results:
                row[f"dim_{dim.dimension_name}"] = dim.aggregate_score
                for metric in dim.metric_scores:
                    row[metric.metric_name] = metric.score
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"\u2713 CSV saved to {csv_path}")

    return json_path


def load_metric_results(path: str) -> list:
    """
    Load EvaluationResult objects from a saved JSON file.

    Handles both formats:
    - Local format: {"evaluations": [...], ...}  (same as S3)
    - Legacy format: [...] (flat list of evaluation dicts)

    Args:
        path: Path to the JSON file produced by save_metric_results()
              or downloaded from S3 via get_full_results()

    Returns:
        List of EvaluationResult objects

    Examples:
        >>> from uaef.utils import load_metric_results
        >>> results = load_metric_results("output/evaluation-results/my_run.json")
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats
    if isinstance(data, dict) and "evaluations" in data:
        eval_list = data["evaluations"]
    elif isinstance(data, list):
        eval_list = data
    else:
        raise ValueError(
            f"Unrecognized format in {path}. "
            "Expected a dict with 'evaluations' key or a list of evaluation dicts."
        )

    from uaef.utils.serialization import deserialize_results
    return deserialize_results(eval_list)
