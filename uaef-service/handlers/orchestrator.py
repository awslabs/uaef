# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Orchestrator handlers: Splitter and Reducer for the Step Functions state machine.

The Splitter partitions an evaluation request's metrics into per-dimension work
items that the Map state fans out to the Worker Lambda. The Reducer merges the
per-dimension Worker results into a single combined outcome and completes the
Job.

Neither handler imports ``uaef`` — the Splitter reads the static
``catalog.json`` to resolve metric→dimension mappings, and the Reducer merges
result payloads that the Worker already shaped.

Requirements: 6.4
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3

# --------------------------------------------------------------------------- #
# Shared: job_state import (supports both package and flat module layouts)
# --------------------------------------------------------------------------- #
try:
    from .. import job_state  # type: ignore[import-not-found]
except (ImportError, ValueError):
    import job_state  # type: ignore[no-redef]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Security review L-04: no default value — see handlers/payloads.py for
#: rationale. The CDK stacks always set this env var explicitly.
PAYLOAD_BUCKET_ENV = "PAYLOAD_BUCKET_NAME"
WORKER_FUNCTION_ENV = "WORKER_FUNCTION_NAME"

_CATALOG_JSON_PATH = Path(__file__).resolve().parent / "catalog.json"

_s3_client = None
_lambda_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def _get_lambda():
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")
    return _lambda_client


def _payload_bucket() -> str:
    bucket = os.environ.get(PAYLOAD_BUCKET_ENV)
    if not bucket:
        raise RuntimeError(
            f"{PAYLOAD_BUCKET_ENV} is not set. This Lambda is misconfigured — "
            "the CDK deployment should always set this environment variable."
        )
    return bucket


def _init_progress(job_id: str, dimension_names: List[str]) -> None:
    """Initialize the progress field on the job record with all dimensions as pending."""
    if not job_id:
        return
    try:
        progress = {
            "total": len(dimension_names),
            "completed": 0,
            "dimensions": {name: "pending" for name in dimension_names},
        }
        job_state._get_table().update_item(
            Key={"jobId": job_id},
            UpdateExpression="SET progress = :prog",
            ExpressionAttributeValues={":prog": progress},
        )
    except Exception:  # noqa: BLE001 — progress is best-effort
        pass


# --------------------------------------------------------------------------- #
# Catalog loading (metric → dimension mapping)
# --------------------------------------------------------------------------- #


def _load_catalog() -> Dict[str, Any]:
    """Load the metric catalog (static JSON or Worker fallback).

    Returns the catalog dict: {"dimensions": {"DimName": {"metrics": [...]}}}
    """
    if _CATALOG_JSON_PATH.exists():
        with open(_CATALOG_JSON_PATH) as f:
            return json.load(f)

    # Fallback: invoke the Worker synchronously for the catalog.
    fn = os.environ.get(WORKER_FUNCTION_ENV)
    if not fn:
        return {}
    resp = _get_lambda().invoke(
        FunctionName=fn,
        InvocationType="RequestResponse",
        Payload=json.dumps({"action": "get_metric_catalog"}).encode("utf-8"),
    )
    payload = resp.get("Payload")
    raw = payload.read() if hasattr(payload, "read") else payload
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw) if raw else {}


def _build_metric_to_dimension_map(catalog: Dict[str, Any]) -> Dict[str, str]:
    """Build a metric_name → dimension_name lookup from the catalog.

    The catalog can be in several formats:
    1. Direct: {"DimensionName": ["metric1", "metric2"], ...}  (from get_full_metric_catalog)
    2. Nested: {"dimensions": {"DimensionName": {"metrics": [...]}}}
    3. List: {"dimensions": [{"dimension": "X", "metrics": [...]}]}
    """
    mapping: Dict[str, str] = {}

    # Try nested format first: {"dimensions": ...} or {"groups": ...}
    dimensions = catalog.get("dimensions") or catalog.get("groups")

    # If no "dimensions"/"groups" key, the catalog itself may be the direct format:
    # {"DimensionName": ["metric1", ...], "AnotherDimension": ["metric2", ...]}
    if dimensions is None:
        dimensions = catalog

    if isinstance(dimensions, dict):
        for dim_name, dim_data in dimensions.items():
            metrics = []
            if isinstance(dim_data, dict):
                metrics = dim_data.get("metrics") or dim_data.get("metric_names") or []
                # Some catalog formats nest metrics as list of dicts with "name" key.
                if metrics and isinstance(metrics[0], dict):
                    metrics = [m.get("name") or m.get("metric_name") for m in metrics]
            elif isinstance(dim_data, list):
                metrics = dim_data
            for m in metrics:
                if m:
                    mapping[m] = dim_name
    elif isinstance(dimensions, list):
        # List-of-dicts format: [{"dimension": "X", "metrics": [...]}]
        for item in dimensions:
            if isinstance(item, dict):
                dim_name = item.get("dimension") or item.get("dimension_name") or "unknown"
                metrics = item.get("metrics") or item.get("metric_names") or []
                if metrics and isinstance(metrics[0], dict):
                    metrics = [m.get("name") or m.get("metric_name") for m in metrics]
                for m in metrics:
                    if m:
                        mapping[m] = dim_name

    return mapping


def _split_metrics_by_dimension(
    metrics: Optional[List[str]], catalog: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Partition a flat metrics list into {dimension_name: [metric_names]}.

    If metrics is None (meaning "all metrics"), return one item per dimension
    with all its metrics.
    """
    metric_to_dim = _build_metric_to_dimension_map(catalog)

    if metrics is None:
        # All metrics: return every dimension with all its metrics.
        result: Dict[str, List[str]] = {}
        for metric_name, dim_name in metric_to_dim.items():
            result.setdefault(dim_name, []).append(metric_name)
        return result

    # Partition the requested metrics by their dimension.
    result = {}
    for m in metrics:
        dim = metric_to_dim.get(m, "unknown")
        result.setdefault(dim, []).append(m)
    return result


# --------------------------------------------------------------------------- #
# Splitter handler
# --------------------------------------------------------------------------- #


def splitter_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Split an evaluation request into per-dimension work items.

    Input (from Step Functions start_execution):
        {
            "jobId": "...",
            "operation": "evaluate|batch_evaluate|invoke_evaluate",
            "request": { ... full request body ... },
            "requestRef": "s3-key"  (optional, for large payloads)
        }

    Output (consumed by the Map state):
        {
            "jobId": "...",
            "operation": "...",
            "dimensionItems": [
                {
                    "dimensionName": "Response Quality",
                    "request": { ... request with metrics=[only this dim's metrics] },
                    "requestRef": null
                },
                ...
            ]
        }
    """
    job_id = event.get("jobId", "")
    operation = event.get("operation", "evaluate")
    request = event.get("request") or {}
    request_ref = event.get("requestRef")

    # If request is provided via S3 reference, fetch it.
    if not request and request_ref:
        obj = _get_s3().get_object(Bucket=_payload_bucket(), Key=request_ref)
        request = json.loads(obj["Body"].read())

    # Mark job as PROCESSING (idempotent). If another process already moved it
    # past PENDING, the Map items will still run but the Worker's own
    # start_processing calls will be no-ops (they check isPartialDimension).
    job_state.start_processing(job_id)

    # Resolve metrics from the request.
    metrics: Optional[List[str]] = request.get("metrics")

    # Load catalog and split by dimension.
    catalog = _load_catalog()
    dimension_groups = _split_metrics_by_dimension(metrics, catalog)

    # If no dimensions resolved (empty catalog or no metrics), fall back to a
    # single item with all metrics — the Worker will handle it as-is.
    if not dimension_groups:
        dimension_groups = {"all": metrics or []}

    # Initialize the progress field on the job record so the UI can show
    # which dimensions are pending/running/done.
    _init_progress(job_id, list(dimension_groups.keys()))

    # Build per-dimension work items. Each gets the full request but with only
    # that dimension's metrics. persist=False on per-dimension calls (the reducer
    # handles the final persist).
    dimension_items: List[Dict[str, Any]] = []
    for dim_name, dim_metrics in dimension_groups.items():
        dim_request = dict(request)
        dim_request["metrics"] = dim_metrics
        # Per-dimension Workers must NOT persist individually — the reducer does
        # the combined persist at the end.
        dim_request["persist"] = False
        dimension_items.append({
            "dimensionName": dim_name,
            "request": dim_request,
            "requestRef": None,
        })

    return {
        "jobId": job_id,
        "operation": operation,
        "originalRequest": request,
        "dimensionItems": dimension_items,
    }


# --------------------------------------------------------------------------- #
# Reducer handler
# --------------------------------------------------------------------------- #


def reducer_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Merge per-dimension Worker results and complete the Job.

    This handler is invoked in two modes:
    1. Normal: after the Map state, with mapResults containing per-dimension
       Worker outputs.
    2. Failure: with action="fail_job" when the Map or a prior step failed.

    Input (normal mode — output of the Map + Splitter passthrough):
        {
            "jobId": "...",
            "operation": "...",
            "originalRequest": { ... },
            "dimensionItems": [...],
            "mapResults": [ <per-dimension Worker outputs> ]
        }

    Input (failure mode):
        {
            "action": "fail_job",
            "jobId": "...",
            "error": "..."
        }
    """
    # Failure mode: mark job as FAILED and exit.
    if event.get("action") == "fail_job":
        return _handle_fail_job(event)

    # Catch-path mode: the Map or Reducer step failed and the catch routed here.
    if event.get("mapError") or event.get("reducerError"):
        return _handle_fail_job(event)

    job_id = event.get("jobId", "")
    operation = event.get("operation", "evaluate")
    original_request = event.get("originalRequest") or {}
    map_results = event.get("mapResults") or []

    try:
        if operation == "invoke_evaluate":
            merged = _merge_invoke_evaluate_results(map_results, original_request)
        elif operation == "batch_evaluate":
            merged = _merge_batch_results(map_results, original_request)
        else:
            merged = _merge_single_results(map_results, original_request)

        # Persist the combined experiment to DynamoDB so it appears in the
        # Experiments tab and Compare panel. Security review H-02: stamp the
        # job's owner (recorded by the API Lambda at job creation) on the
        # experiment so list/read access can be scoped to it.
        created_by = job_state.get_created_by(job_id)
        experiment_id = _persist_combined_experiment(
            merged, original_request, operation, created_by=created_by
        )
        merged["experiment_id"] = experiment_id

        # Store the merged result to S3.
        result_ref = _store_merged_result(job_id, merged)

        # Build result summary for the job record.
        result_summary = _build_result_summary(merged, operation)

        # Complete the job.
        job_state.complete_job(
            job_id,
            experiment_id=experiment_id,
            result_ref=result_ref,
            result_summary=result_summary,
        )

        return {"status": "COMPLETED", "jobId": job_id}

    except Exception as exc:  # noqa: BLE001
        # Security review M-03: the persisted job error is surfaced verbatim
        # by GET /jobs/{jobId} to the job's owner — don't put raw exception
        # detail (which can carry S3/DynamoDB internals from the merge step)
        # into it. The Step Functions return value here feeds back into the
        # state machine, not directly to an API caller, but is sanitized too
        # for consistency and because it's logged/visible in execution history.
        safe_message, correlation_id = job_state.sanitize_error(
            exc, context="Reducer failed to merge results"
        )
        job_state.fail_job(
            job_id,
            code="REDUCER_ERROR",
            message=safe_message,
        )
        return {
            "status": "FAILED",
            "jobId": job_id,
            "error": safe_message,
            "correlationId": correlation_id,
        }


def _handle_fail_job(event: Dict[str, Any]) -> Dict[str, Any]:
    """Mark a job as FAILED from the error-catch path."""
    job_id = event.get("jobId", "")
    # The error can come from different paths:
    # - Direct: event["error"] (from the FailJob payload)
    # - Map catch: event["mapError"] (Step Functions error object)
    # - Reducer catch: event["reducerError"] (Step Functions error object)
    error_info = (
        event.get("error")
        or event.get("mapError")
        or event.get("reducerError")
        or "Unknown orchestration error"
    )
    if isinstance(error_info, dict):
        # Security review M-03: for a Lambda task failure, Step Functions'
        # error object puts the raw exception message — and often a full
        # stack trace — into "Cause". This is the biggest leak on the
        # orchestration path if returned/persisted unsanitized: it can carry
        # boto3/S3/DynamoDB/library internals from the Worker or Reducer.
        error_msg = error_info.get("Cause") or error_info.get("Error") or json.dumps(error_info)
    else:
        error_msg = str(error_info)
    safe_message, correlation_id = job_state.sanitize_error(
        RuntimeError(error_msg), context="Orchestration failed"
    )

    # Ensure job is in PROCESSING first (idempotent).
    job_state.start_processing(job_id)
    job_state.fail_job(
        job_id,
        code="ORCHESTRATION_ERROR",
        message=safe_message,
    )
    return {
        "status": "FAILED",
        "jobId": job_id,
        "error": safe_message,
        "correlationId": correlation_id,
    }


# --------------------------------------------------------------------------- #
# Experiment persistence (Reducer writes the combined experiment to DynamoDB)
# --------------------------------------------------------------------------- #

UAEF_EXPERIMENT_TABLE_ENV = "UAEF_DYNAMODB_TABLE"
UAEF_RESULTS_BUCKET_ENV = "UAEF_S3_BUCKET"

_dynamo_resource = None


def _get_dynamo_resource():
    global _dynamo_resource
    if _dynamo_resource is None:
        _dynamo_resource = boto3.resource("dynamodb")
    return _dynamo_resource


def _persist_combined_experiment(
    merged: Dict[str, Any],
    original_request: Dict[str, Any],
    operation: str,
    created_by: Optional[str] = None,
) -> str:
    """Persist the combined experiment to the UAEF experiment table.

    Creates a single experiment record with the merged average_scores so the
    Experiments tab and Compare panel can find it. Also writes the full results
    to the UAEF results bucket so report generation can find them.
    Returns the experiment_id.
    """
    import uuid
    from datetime import datetime, timezone
    from decimal import Decimal

    experiment_id = merged.get("experiment_id") or str(uuid.uuid4())
    experiment_name = original_request.get("experiment_name") or "Ad-hoc evaluation"
    experiment_objective = original_request.get("experiment_objective") or ""
    framework = original_request.get("framework") or ""
    agent_name = original_request.get("agent_name") or framework

    # Compute average_scores from the merged result.
    average_scores = merged.get("average_scores") or {}
    overall_avg = merged.get("overall_average_score") or merged.get("overall_score") or 0.0

    # Count evaluations.
    if operation == "invoke_evaluate":
        eval_count = len(merged.get("rows") or [])
    elif operation == "batch_evaluate":
        eval_count = len(merged.get("results") or [])
    else:
        eval_count = 1

    # Convert floats to Decimal for DynamoDB.
    def _to_decimal(obj):
        if isinstance(obj, float):
            return Decimal(str(round(obj, 4)))
        if isinstance(obj, dict):
            return {k: _to_decimal(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_decimal(v) for v in obj]
        return obj

    now = datetime.now(timezone.utc).isoformat()

    table_name = os.environ.get(UAEF_EXPERIMENT_TABLE_ENV)
    if not table_name:
        return experiment_id  # Can't persist without table name, return ID anyway.

    # Write the full results to the UAEF results bucket so report generation works.
    results_bucket = os.environ.get(UAEF_RESULTS_BUCKET_ENV)
    s3_prefix = os.environ.get("UAEF_S3_PREFIX", "evaluations/")
    if results_bucket:
        try:
            # The library reads from "{s3_prefix}{experiment_id}.json" where
            # s3_prefix defaults to "evaluations/".
            result_key = f"{s3_prefix}{experiment_id}.json"
            # Build the results payload in the format the library expects.
            results_data = {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "experiment_objective": experiment_objective,
                "evaluations": merged.get("rows") or merged.get("results") or [merged],
            }
            _get_s3().put_object(
                Bucket=results_bucket,
                Key=result_key,
                Body=json.dumps(results_data, default=str).encode("utf-8"),
                ContentType="application/json",
            )
        except Exception:  # noqa: BLE001 — best-effort
            pass

    try:
        table = _get_dynamo_resource().Table(table_name)
        item = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "experiment_objective": experiment_objective,
            "created_at": now,
            "updated_at": now,
            "evaluation_count": eval_count,
            "average_scores": _to_decimal(average_scores),
            "overall_average_score": _to_decimal(overall_avg),
            "result_path": f"s3://{results_bucket}/{s3_prefix}{experiment_id}.json" if results_bucket else "",
            "metadata": {
                "framework": framework,
                "agent_name": agent_name,
                "dataset_name": original_request.get("filename") or "",
            },
        }
        # Security review H-02: stamp the owner so list/read access can be
        # scoped to it. Preserve an existing experiment's created_by on
        # update rather than overwriting it (mirrors
        # DynamoS3Storage.save_experiment's same rule).
        existing_created_by = None
        try:
            existing_item = table.get_item(Key={"experiment_id": experiment_id}).get("Item")
            if isinstance(existing_item, dict):
                existing_created_by = existing_item.get("created_by")
        except Exception:  # noqa: BLE001 — best-effort
            pass
        if existing_created_by:
            item["created_by"] = existing_created_by
        elif created_by:
            item["created_by"] = created_by
        table.put_item(Item=item)
    except Exception:  # noqa: BLE001 — persistence is best-effort; job still completes
        pass

    return experiment_id


# --------------------------------------------------------------------------- #
# Result merging
# --------------------------------------------------------------------------- #


def _merge_single_results(
    map_results: List[Any], original_request: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge per-dimension results for a single evaluate operation.

    Each Worker invocation returns an EvaluationResult dict (or the raw result
    from uaef.api.evaluate). We combine dimension_results from all into one.
    """
    merged_dimension_results: List[Dict[str, Any]] = []
    overall_scores: List[float] = []
    experiment_id = None

    for result in map_results:
        if result is None:
            continue
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                continue

        # The Worker returns the raw EvaluationResult model_dump or a dict.
        dim_results = result.get("dimension_results") or []
        merged_dimension_results.extend(dim_results)

        score = result.get("overall_score")
        if score is not None:
            overall_scores.append(float(score))

        if not experiment_id:
            experiment_id = result.get("experiment_id")

    # Recompute overall score as the average of dimension aggregate scores.
    combined_overall = _compute_overall_score(merged_dimension_results)

    return {
        "experiment_id": experiment_id,
        "overall_score": combined_overall,
        "passed": combined_overall >= 0.7,  # default threshold
        "dimension_results": merged_dimension_results,
    }


def _merge_batch_results(
    map_results: List[Any], original_request: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge per-dimension results for a batch_evaluate operation.

    Each Worker returns a list of EvaluationResults (one per trace). We need to
    merge dimension_results across dimensions for each trace index.
    """
    if not map_results:
        return {"results": [], "experiment_id": None}

    # Parse all results into lists.
    parsed_lists: List[List[Dict[str, Any]]] = []
    for result in map_results:
        if result is None:
            continue
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(result, list):
            parsed_lists.append(result)
        elif isinstance(result, dict) and "results" in result:
            parsed_lists.append(result.get("results", []))

    if not parsed_lists:
        return {"results": [], "experiment_id": None}

    # Determine the number of traces from the longest list.
    num_traces = max(len(lst) for lst in parsed_lists) if parsed_lists else 0
    merged_results: List[Dict[str, Any]] = []
    experiment_id = None

    for i in range(num_traces):
        merged_dims: List[Dict[str, Any]] = []
        trace_experiment_id = None
        for lst in parsed_lists:
            if i < len(lst):
                item = lst[i]
                if isinstance(item, dict):
                    dims = item.get("dimension_results") or []
                    merged_dims.extend(dims)
                    if not trace_experiment_id:
                        trace_experiment_id = item.get("experiment_id")

        combined_score = _compute_overall_score(merged_dims)
        merged_results.append({
            "experiment_id": trace_experiment_id,
            "overall_score": combined_score,
            "passed": combined_score >= 0.7,
            "dimension_results": merged_dims,
        })
        if not experiment_id and trace_experiment_id:
            experiment_id = trace_experiment_id

    return {"results": merged_results, "experiment_id": experiment_id}


def _merge_invoke_evaluate_results(
    map_results: List[Any], original_request: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge per-dimension results for an invoke_evaluate operation.

    The invoke_evaluate Worker returns a UI-shaped result dict. When split by
    dimension, each Worker only evaluates its dimensions' metrics on the same
    traces. We merge the per-row metric_scores and dimension_scores, and
    recompute averages.
    """
    if not map_results:
        return {"rows": [], "experiment_id": None}

    # Parse results. invoke_evaluate Workers return the UI-shaped dict directly.
    parsed: List[Dict[str, Any]] = []
    for result in map_results:
        if result is None:
            continue
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(result, dict):
            parsed.append(result)

    if not parsed:
        return {"rows": [], "experiment_id": None}

    # Use the first result as a base (it has the row structure: query,
    # agent_response, expected_answer, etc.) and merge in dimension/metric
    # scores from the others.
    base = parsed[0]
    base_rows = base.get("rows") or []
    experiment_id = base.get("experiment_id")

    for other in parsed[1:]:
        other_rows = other.get("rows") or []
        if not experiment_id:
            experiment_id = other.get("experiment_id")
        for i, other_row in enumerate(other_rows):
            if i < len(base_rows):
                # Merge dimension_scores.
                other_dims = other_row.get("dimension_scores") or {}
                base_rows[i].setdefault("dimension_scores", {}).update(other_dims)
                # Merge metric_scores.
                other_metrics = other_row.get("metric_scores") or {}
                base_rows[i].setdefault("metric_scores", {}).update(other_metrics)
                # Merge the unavailable-metric detail.
                other_unavail = other_row.get("unavailable_metrics") or {}
                if other_unavail:
                    base_rows[i].setdefault("unavailable_metrics", {}).update(other_unavail)

    # Recompute per-row overall scores.
    all_scores: List[float] = []
    for row in base_rows:
        dim_scores = row.get("dimension_scores") or {}
        if dim_scores:
            avg = sum(dim_scores.values()) / len(dim_scores)
            row["overall_score"] = avg
            all_scores.append(avg)

    # Merge the dimension -> metric grouping and the unavailable-metric detail
    # from every partial. These are top-level fields, so taking parsed[0] alone
    # would report a single dimension: selecting Tool Calling and Multi-Turn
    # rendered only one radar axis, whichever dimension the Map happened to
    # return first.
    merged_dimension_metrics: Dict[str, List[str]] = {}
    merged_unavailable: Dict[str, Any] = {}
    for part in parsed:
        for dim_name, metric_names in (part.get("dimension_metrics") or {}).items():
            bucket = merged_dimension_metrics.setdefault(dim_name, [])
            for name in metric_names or []:
                if name not in bucket:
                    bucket.append(name)
        for metric_name, detail in (part.get("unavailable_metrics") or {}).items():
            merged_unavailable.setdefault(metric_name, detail)

    # Recompute average_scores: per-metric averages across all rows. The UI
    # builds its radar from dimension_metrics above, not from these names.
    all_metric_scores: Dict[str, List[float]] = {}
    for row in base_rows:
        for metric_name, score in (row.get("metric_scores") or {}).items():
            if score is not None:
                all_metric_scores.setdefault(metric_name, []).append(float(score))
    average_scores = {
        metric: sum(scores) / len(scores)
        for metric, scores in all_metric_scores.items()
        if scores
    }

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    base["experiment_id"] = experiment_id
    base["overall_average_score"] = overall_avg
    base["average_scores"] = average_scores
    base["rows"] = base_rows
    base["dimension_metrics"] = {
        k: sorted(v) for k, v in merged_dimension_metrics.items()
    }
    base["unavailable_metrics"] = merged_unavailable

    return base


def _compute_overall_score(dimension_results: List[Dict[str, Any]]) -> float:
    """Compute an overall score as the average of dimension aggregate scores."""
    if not dimension_results:
        return 0.0

    scores = []
    for dim in dimension_results:
        score = dim.get("aggregate_score")
        if score is not None:
            scores.append(float(score))
    return sum(scores) / len(scores) if scores else 0.0


# --------------------------------------------------------------------------- #
# Result storage
# --------------------------------------------------------------------------- #


def _store_merged_result(job_id: str, merged: Dict[str, Any]) -> str:
    """Store the merged result JSON to S3 and return the key."""
    key = f"results/{job_id}.json"
    _get_s3().put_object(
        Bucket=_payload_bucket(),
        Key=key,
        Body=json.dumps(merged, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return key


def _build_result_summary(merged: Dict[str, Any], operation: str) -> Dict[str, Any]:
    """Build a small inline result summary for the job record."""
    if operation in ("batch_evaluate", "invoke_evaluate"):
        results = merged.get("results") or merged.get("rows") or []
        return {
            "count": len(results),
            "overallAverageScore": merged.get("overall_average_score") or merged.get("overall_score") or 0.0,
        }
    return {
        "overallScore": merged.get("overall_score") or 0.0,
        "passed": merged.get("passed", False),
    }
