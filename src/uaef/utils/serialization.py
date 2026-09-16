# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared serialization for EvaluationResult objects.

This is the single source of truth for converting EvaluationResult objects
to/from JSON-compatible dicts. Used by both:
- Local file persistence (save_metric_results / load_metric_results)
- S3 persistence (DynamoS3Storage.save_experiment)
"""

from typing import Any, Dict, List

from uaef.models.evaluation_result import EvaluationResult


def serialize_results(results: List[EvaluationResult]) -> List[Dict[str, Any]]:
    """
    Serialize a list of EvaluationResult objects to JSON-compatible dicts.

    This is the canonical serialization used everywhere results are persisted
    (local JSON files, S3 storage, API responses).

    Args:
        results: List of EvaluationResult objects

    Returns:
        List of JSON-serializable dicts
    """
    return [r.model_dump(mode="json") for r in results]


def deserialize_results(data: List[Dict[str, Any]]) -> List[EvaluationResult]:
    """
    Deserialize a list of dicts back into EvaluationResult objects.

    Args:
        data: List of dicts (as produced by serialize_results)

    Returns:
        List of EvaluationResult objects
    """
    return [EvaluationResult(**item) for item in data]
