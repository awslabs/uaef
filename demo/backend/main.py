# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
UAEF Demo Backend — FastAPI server for the evaluation UI.

Flow:
1. User uploads ground truth (xlsx/csv with Question + Answer)
2. User selects agent type and provides connection info
3. UAEF invokes the agent for each question, gets raw traces
4. Adapter converts raw traces to canonical AgentTrace
5. Evaluation engine scores each trace
6. Results returned to UI
"""

import os, sys, json, time, logging
from pathlib import Path
from uuid import uuid4, UUID
from typing import Optional
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3

from uaef.api import evaluate, batch_evaluate
from uaef.experiments.comparison import ComparisonEngine
from uaef.experiments.models import ExperimentRun, AgentDesign
from uaef.models import AgentTrace, Message, GroundTruth, ToolCall
from uaef.models.message import MessageRole
from uaef.adapters import AgentCoreAdapter
from uaef.adapters import (
    invoke_http_agent,
    invoke_bedrock_agent,
    invoke_langfuse_trace,
    invoke_http_agent_for_framework,
)
from uaef.adapters.registry import get_adapter, list_adapters
from uaef.metrics import get_full_metric_catalog
from uaef.data import (
    parse_ground_truth_row,
    load_ground_truth_file,
    validate_ground_truth,
)

logger = logging.getLogger(__name__)

# The library's agent-invocation SSRF guard blocks private/loopback/metadata
# endpoints by default. The demo runs locally and its whole point is evaluating
# a localhost agent, so opt in to private endpoints here (setdefault, so an
# operator can still force strict mode). The deployed Worker never does this.
os.environ.setdefault("UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS", "true")

# API-key authentication for the demo API. Fails closed: if UAEF_DEMO_API_KEY
# is unset, every request is rejected with a 503 rather than silently allowed.
# (Security review M-06: auth used to default to *disabled* when the key was
# unset, so anyone who copied the demo as a starting point — the explicit
# intent of a shared reusable asset — could easily end up deploying an
# unauthenticated service fronting live agent invocation and Bedrock spend.)
#
# For local development where you deliberately want no auth, set
# UAEF_DEMO_ALLOW_NO_AUTH=true explicitly — this makes skipping auth a
# conscious opt-in rather than the silent default. See demo/README.md.
_DEMO_API_KEY = os.environ.get("UAEF_DEMO_API_KEY", "")
_ALLOW_NO_AUTH = os.environ.get("UAEF_DEMO_ALLOW_NO_AUTH", "").strip().lower() in {"true", "1", "yes"}

if not _DEMO_API_KEY and not _ALLOW_NO_AUTH:
    logger.warning(
        "UAEF_DEMO_API_KEY is not set. All requests will be rejected with 503. "
        "Set UAEF_DEMO_API_KEY to require an X-API-Key header, or set "
        "UAEF_DEMO_ALLOW_NO_AUTH=true to explicitly run without authentication "
        "(local development only — never on a shared/exposed host)."
    )


def require_api_key(x_api_key: str = Header(default="")):
    """Require X-API-Key to match UAEF_DEMO_API_KEY, or an explicit no-auth opt-in.

    Fails closed: with neither UAEF_DEMO_API_KEY nor UAEF_DEMO_ALLOW_NO_AUTH
    set, every request is rejected rather than silently allowed.
    """
    if _DEMO_API_KEY:
        if x_api_key != _DEMO_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return
    if _ALLOW_NO_AUTH:
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "This server is not configured for authentication. Set "
            "UAEF_DEMO_API_KEY, or set UAEF_DEMO_ALLOW_NO_AUTH=true to "
            "explicitly run without auth (local development only)."
        ),
    )


app = FastAPI(title="UAEF Demo API", dependencies=[Depends(require_api_key)])
# CORS origins are configurable via CORS_ALLOWED_ORIGINS (comma-separated);
# defaults to "*" so behavior is unchanged unless explicitly set.
_cors_allowed_origins = [o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins, allow_methods=["*"], allow_headers=["*"],
)



# ── Agent invocation helpers ──────────────────────────────────────────────

def _invoke_agentcore(
    agent_runtime_arn: str, query: str,
    region: str = "us-east-1", bearer_token: str = "",
) -> dict:
    """Invoke an AgentCore agent via the adapter (handles auth auto-detection)."""
    return AgentCoreAdapter.invoke(
        agent_runtime_arn=agent_runtime_arn,
        user_input=query,
        region=region,
        bearer_token=bearer_token or None,
        auto_detect_auth=True,
    )


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/api/aws-identity")
def get_aws_identity():
    """Return the current AWS account and identity info."""
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        return {
            "authenticated": True,
            "account": identity.get("Account", ""),
            "arn": identity.get("Arn", ""),
            "user_id": identity.get("UserId", ""),
        }
    except Exception as e:
        logger.warning("Failed to resolve AWS identity: %s", e)
        return {
            "authenticated": False,
            "error": "Unable to determine AWS identity.",
        }


# NOTE: The former POST /api/aws-configure endpoint (which accepted AWS
# credentials in the request body and set them into os.environ for the whole
# process) has been removed. The demo now relies on the backend's ambient AWS
# credentials via boto3's default provider chain (environment variables, the
# shared config from `aws configure`, SSO, or an IAM role). Set your credentials
# in the shell before starting uvicorn; GET /api/aws-identity reports the
# resolved identity. This avoids accepting credentials over the API.


@app.get("/api/agentcore/env")
def get_agentcore_env():
    """Load AgentCore config from .env.agentcore if it exists."""
    from dotenv import load_dotenv
    import os

    env_path = Path(__file__).resolve().parent.parent.parent / "notebooks" / ".env.agentcore"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent.parent / ".env.agentcore"

    result = {}
    if env_path.exists():
        load_dotenv(env_path, override=True)
        token = os.getenv("BEARER_TOKEN", "")
        # Treat "None" string as empty
        if token and token.strip().lower() == "none":
            token = ""
        result["agent_runtime_arn"] = os.getenv("AGENT_RUNTIME_ARN", "")
        result["agent_runtime_id"] = os.getenv("AGENT_RUNTIME_ID", "")
        result["bearer_token"] = token
        result["found"] = True
        result["path"] = str(env_path)
        logger.debug(  # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure -- False positive: logs only whether the values are SET (booleans via bool(...)), never the token/ARN value itself.
            ".env.agentcore loaded (agent_runtime_arn set=%s, bearer_token set=%s)",
            bool(result["agent_runtime_arn"]), bool(token),
        )
    else:
        result["found"] = False
        logger.debug(".env.agentcore not found at %s", env_path)

    return result


@app.get("/api/metrics")
def get_metrics():
    """Return all metrics grouped by dimension/integration.

    Sourced entirely from the library's canonical catalog
    (``get_full_metric_catalog``), which merges built-in metrics with the
    installed integration metrics. No metric names are hardcoded here
    (Requirement 12.7); integration groups (e.g. RAGAS, DeepEval) appear only
    when their optional extra is installed.
    """
    # Built-in dimensions first, then integration groups, for stable display.
    catalog = get_full_metric_catalog()
    return {key: catalog[key] for key in sorted(catalog.keys())}


@app.get("/api/agent-types")
def get_agent_types():
    """Return supported agent types with their connection requirements."""
    return [
        {
            "id": "agentcore",
            "label": "AWS AgentCore",
            "fields": [
                {"name": "agent_runtime_arn", "label": "Agent Runtime ARN", "placeholder": "arn:aws:bedrock-agentcore:..."},
                {"name": "region", "label": "AWS Region", "default": "us-east-1"},
                {"name": "bearer_token", "label": "Bearer Token (optional, for OAuth/JWT auth)", "placeholder": "Leave empty for SigV4", "optional": True},
            ],
        },
        {
            "id": "strands",
            "label": "Strands",
            "fields": [
                {"name": "endpoint", "label": "Agent Endpoint URL", "placeholder": "http://localhost:8080/invoke"},
            ],
        },
        {
            "id": "langfuse",
            "label": "Langfuse (fetch existing traces)",
            "fields": [
                {"name": "langfuse_public_key", "label": "Langfuse Public Key", "placeholder": "pk-lf-..."},
                {"name": "langfuse_secret_key", "label": "Langfuse Secret Key", "placeholder": "sk-lf-..."},
                {"name": "langfuse_host", "label": "Langfuse Host", "default": "https://cloud.langfuse.com"},
            ],
        },
        {
            "id": "langgraph",
            "label": "LangGraph",
            "fields": [
                {"name": "endpoint", "label": "Agent Endpoint URL", "placeholder": "http://localhost:8080/invoke"},
            ],
        },
        {
            "id": "bedrock",
            "label": "AWS Bedrock Agent",
            "fields": [
                {"name": "agent_id", "label": "Agent ID", "placeholder": "e.g. ABCDEF1234"},
                {"name": "alias_id", "label": "Alias ID", "placeholder": "e.g. TSTALIASID"},
                {"name": "region", "label": "AWS Region", "default": "us-east-1"},
            ],
        },
        {
            "id": "langchain",
            "label": "LangChain",
            "fields": [
                {"name": "endpoint", "label": "Agent Endpoint URL", "placeholder": "http://localhost:8080/invoke"},
            ],
        },
        {
            "id": "generic",
            "label": "Generic",
            "fields": [
                {"name": "endpoint", "label": "Agent Endpoint URL", "placeholder": "http://localhost:8080/invoke"},
            ],
        },
    ]


@app.get("/api/experiments")
def get_experiments():
    """List all persisted experiments from DynamoDB."""
    try:
        from uaef.storage.dynamodb_s3 import get_storage
        store = get_storage()
        experiments = store.list_experiments()
        return {
            "experiments": [
                {
                    "experiment_id": exp.get("experiment_id", ""),
                    "experiment_name": exp.get("experiment_name", ""),
                    "experiment_objective": exp.get("experiment_objective", ""),
                    "framework": exp.get("metadata", {}).get("framework", "—") if isinstance(exp.get("metadata"), dict) else "—",
                    "agent_name": exp.get("metadata", {}).get("agent_name", "") if isinstance(exp.get("metadata"), dict) else "",
                    "dataset_name": exp.get("metadata", {}).get("dataset_name", "—") if isinstance(exp.get("metadata"), dict) else "—",
                    "dataset_s3_path": exp.get("metadata", {}).get("dataset_s3_path", "") if isinstance(exp.get("metadata"), dict) else "",
                    "evaluation_count": int(exp.get("evaluation_count", 0)),
                    "overall_average_score": float(exp.get("overall_average_score", 0)),
                    "result_path": exp.get("result_path", ""),
                    "created_at": exp.get("created_at", ""),
                    "updated_at": exp.get("updated_at", ""),
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        logger.warning("Failed to list experiments: %s", e)
        return {"experiments": [], "error": "Failed to list experiments."}


@app.get("/api/experiments/{experiment_id}")
def get_experiment_results(experiment_id: str):
    """Get full results for a specific experiment from S3."""
    try:
        from uaef.storage.dynamodb_s3 import get_storage
        store = get_storage()

        # Get experiment metadata from DynamoDB
        exp = store.get_experiment(experiment_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")

        # Get full results from S3
        full_results = store.get_full_results(experiment_id)
        evaluations = full_results.get("evaluations", [])

        # Build per-row presentation data. Aggregates are read from the
        # stored experiment record (average_scores / overall_average_score)
        # computed by the persistence layer — the presentation layer never
        # recomputes them here.
        metadata = exp.get("metadata", {}) if isinstance(exp.get("metadata"), dict) else {}
        rows = []

        for idx, ev in enumerate(evaluations):
            overall = float(ev.get("overall_score", 0))

            # Extract metric scores from dimension_results structure
            dimension_scores = {}
            metric_scores = {}
            for dim in ev.get("dimension_results", []):
                dim_name = dim.get("dimension_name", "")
                dimension_scores[dim_name] = dim.get("aggregate_score", 0)
                for ms in dim.get("metric_scores", []):
                    m_name = ms.get("metric_name", "")
                    m_score = ms.get("score")
                    if m_name and m_score is not None:
                        metric_scores[m_name] = m_score

            # Query/response may be stored in evaluation metadata
            ev_metadata = ev.get("metadata", {}) or {}

            row_data = {
                "index": idx + 1,
                "query": ev_metadata.get("query", ""),
                "agent_response": ev_metadata.get("agent_response", "")[:500],
                "expected_answer": ev_metadata.get("expected_output", "")[:500],
                "overall_score": overall,
                "passed": ev.get("passed", False),
                "dimension_scores": dimension_scores,
                "metric_scores": metric_scores,
            }
            rows.append(row_data)

        # Read the stored aggregates. If they are absent, return an error
        # indication rather than recomputing in the presentation layer (Req 12.5).
        stored_average_scores = exp.get("average_scores")
        stored_overall_avg = exp.get("overall_average_score")
        if stored_average_scores is None or stored_overall_avg is None:
            raise HTTPException(
                422,
                f"Experiment {experiment_id} has no stored aggregates "
                "(average_scores / overall_average_score) to report.",
            )
        average_scores = {k: float(v) for k, v in stored_average_scores.items()}
        overall_avg = float(stored_overall_avg)

        return {
            "experiment_id": experiment_id,
            "experiment_name": exp.get("experiment_name", ""),
            "experiment_objective": exp.get("experiment_objective", ""),
            "framework": metadata.get("framework", ""),
            "agent_name": metadata.get("agent_name", ""),
            "dataset_name": metadata.get("dataset_name", ""),
            "evaluation_count": len(evaluations),
            "total_queries": len(evaluations),
            "successful_queries": len(evaluations),
            "failed_queries": 0,
            "overall_average_score": overall_avg,
            "average_scores": average_scores,
            "rows": rows,
            "errors": [],
            "persist": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load experiment results: {e}")


@app.post("/api/validate-data")
async def validate_data(
    file: UploadFile = File(None),
    s3_path: str = Form(""),
    source: str = Form("upload"),
):
    """Validate uploaded ground truth file or S3 path and return a preview."""
    try:
        if source == "s3" and s3_path:
            import re
            match = re.match(r"s3://([^/]+)/(.+)", s3_path)
            if not match:
                return {"valid": False, "error": "Invalid S3 path. Expected format: s3://bucket/key"}
            bucket, key = match.group(1), match.group(2)
            s3 = boto3.client("s3")
            resp = s3.get_object(Bucket=bucket, Key=key)
            content = resp["Body"].read()
            fname = key.split("/")[-1]
        elif file:
            content = await file.read()
            fname = file.filename or ""
        else:
            return {"valid": False, "error": "No file or S3 path provided"}

        df = load_ground_truth_file(content=content, filename=fname)
    except Exception as e:
        logger.warning("Failed to read ground truth data (%s): %s", fname, e)
        return {"valid": False, "error": "Failed to read data. Check the file format and try again."}

    result = validate_ground_truth(df, filename=fname)
    return result.model_dump()


@app.get("/api/agentcore/runtimes")
def list_agentcore_runtimes(region: str = "us-east-1"):
    """List deployed AgentCore runtimes in the account."""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=region)
        runtimes = []
        paginator_token = None

        while True:
            kwargs = {"maxResults": 50}
            if paginator_token:
                kwargs["nextToken"] = paginator_token

            resp = client.list_agent_runtimes(**kwargs)
            for rt in resp.get("agentRuntimes", []):
                runtimes.append({
                    "arn": rt.get("agentRuntimeArn", ""),
                    "id": rt.get("agentRuntimeId", ""),
                    "name": rt.get("agentRuntimeName", ""),
                    "description": rt.get("description", ""),
                    "status": rt.get("status", ""),
                })

            paginator_token = resp.get("nextToken")
            if not paginator_token:
                break

        return {"runtimes": runtimes, "region": region}

    except Exception as e:
        logger.warning("Failed to list AgentCore runtimes: %s", e)
        return {"runtimes": [], "region": region, "error": "Failed to list runtimes."}


class CompareRequest(BaseModel):
    experiment_ids: list[str]


@app.post("/api/compare-experiments")
def compare_experiments(req: CompareRequest):
    """Compare multiple experiments using the library ComparisonEngine.

    All comparison logic is delegated to the library
    (uaef.experiments.comparison.ComparisonEngine — the same engine behind
    uaef.api.compare_runs). The presentation layer only loads stored experiment
    aggregates, builds ExperimentRun objects, and shapes the library's
    ComparisonReport output for transport. If the library comparison fails, an
    error indication is returned — there is no presentation-layer fallback
    comparison (Requirement 12.9).
    """
    if len(req.experiment_ids) < 2:
        raise HTTPException(400, "Select at least 2 experiments to compare")

    from uaef.storage.dynamodb_s3 import get_storage
    store = get_storage()

    # Load stored experiments. Read the stored aggregates only — the
    # presentation layer never recomputes average_scores / overall scores.
    loaded = []
    for eid in req.experiment_ids:
        exp = store.get_experiment(eid)
        if not exp:
            raise HTTPException(404, f"Experiment {eid} not found")
        meta = exp.get("metadata", {}) if isinstance(exp.get("metadata"), dict) else {}
        average_scores = exp.get("average_scores")
        if not average_scores:
            # Stored aggregates are required to compare; do not recompute here.
            raise HTTPException(
                422, f"Experiment {eid} has no stored average_scores to compare"
            )
        loaded.append({
            "id": eid,
            "name": exp.get("experiment_name", "") or eid,
            "framework": meta.get("framework", "—"),
            "agent_name": meta.get("agent_name", ""),
            "overall_score": float(exp.get("overall_average_score", 0)),
            "scores": {k: float(v) for k, v in average_scores.items()},
        })

    def _to_run(item: dict) -> ExperimentRun:
        """Build a library ExperimentRun from stored experiment aggregates."""
        return ExperimentRun(
            experiment_id=UUID(item["id"]),
            run_name=item["name"],
            agent_design=AgentDesign(
                framework=item["framework"] or "unknown",
                model=item["agent_name"] or "unknown",
            ),
            config_snapshot={},
            aggregate_metrics=item["scores"],
        )

    # Delegate the comparison entirely to the library ComparisonEngine. The
    # first selected experiment is the baseline; the rest are compared against
    # it. Any failure here surfaces as an error indication — never a
    # presentation-layer fallback comparison (Requirement 12.9).
    try:
        engine = ComparisonEngine()
        baseline_run = _to_run(loaded[0])
        current_runs = [_to_run(item) for item in loaded[1:]]
        reports = engine.compare_multiple_runs(baseline_run, current_runs)
    except Exception as e:
        raise HTTPException(502, f"Comparison failed in the evaluation library: {e}")

    # Shape the library's ComparisonReport output for transport. Per-metric
    # scores and the regression/improvement classifications all come from the
    # library reports; the layer below only formats them for the UI.
    experiments_out = [{
        "id": item["id"],
        "name": item["name"],
        "framework": item["framework"],
        "agent_name": item["agent_name"],
        "overall_score": item["overall_score"],
    } for item in loaded]

    # exp_0 is the baseline; exp_{j+1} corresponds to reports[j].
    metric_scores: dict = {}  # metric -> {experiment_index: score}
    for j, report in enumerate(reports):
        for mc in report.metric_comparisons:
            entry = metric_scores.setdefault(mc.metric_name, {})
            entry[0] = mc.baseline_score
            entry[j + 1] = mc.current_score

    table = []
    for metric in sorted(metric_scores.keys()):
        entry = metric_scores[metric]
        row = {"metric": metric}
        values = []
        for i in range(len(loaded)):
            val = entry.get(i)
            row[f"exp_{i}"] = round(val, 4) if val is not None else None
            if val is not None:
                values.append(val)
        # Descriptive stats over the library-provided scores (display aids).
        row["min"] = round(min(values), 4) if values else None
        row["max"] = round(max(values), 4) if values else None
        row["spread"] = round(max(values) - min(values), 4) if len(values) > 1 else 0
        table.append(row)

    # Insights are derived from the library's regression/improvement
    # classifications, not hand-rolled in the presentation layer.
    insights = []
    baseline_name = loaded[0]["name"]
    for j, report in enumerate(reports):
        current_name = loaded[j + 1]["name"]
        if report.improvements:
            insights.append(
                f"{current_name} improved vs {baseline_name}: "
                + ", ".join(report.improvements)
            )
        if report.regressions:
            insights.append(
                f"{current_name} regressed vs {baseline_name}: "
                + ", ".join(report.regressions)
            )
        if not report.improvements and not report.regressions:
            insights.append(
                f"{current_name} shows no significant change vs {baseline_name} "
                f"({report.summary.get('unchanged_count', 0)} metrics within threshold)."
            )

    return {
        "experiments": experiments_out,
        "table": table,
        "insights": insights,
    }


@app.post("/api/evaluate")
async def run_evaluation(
    file: UploadFile = File(None),
    data_source: str = Form("upload"),
    s3_data_path: str = Form(""),
    framework: str = Form("generic"),
    metrics: str = Form(""),
    # HTTP agent fields
    endpoint: str = Form(""),
    # Bedrock fields
    agent_id: str = Form(""),
    alias_id: str = Form(""),
    region: str = Form("us-east-1"),
    # AgentCore fields
    agent_runtime_arn: str = Form(""),
    bearer_token: str = Form(""),
    # Langfuse fields
    langfuse_public_key: str = Form(""),
    langfuse_secret_key: str = Form(""),
    langfuse_host: str = Form("https://cloud.langfuse.com"),
    # Persistence
    persist: str = Form("false"),
    experiment_name: str = Form(""),
    experiment_objective: str = Form(""),
    agent_name: str = Form(""),
):
    """
    Full evaluation flow:
    1. Parse uploaded ground truth file
    2. For each row, invoke the agent and get raw trace
    3. Use adapter to convert to canonical AgentTrace
    4. Run UAEF evaluation
    5. Return aggregated results
    """
    # Validate agent connection
    if framework == "bedrock":
        if not agent_id or not alias_id:
            raise HTTPException(400, "Bedrock agents require agent_id and alias_id")
    elif framework == "agentcore":
        if not agent_runtime_arn:
            raise HTTPException(400, "AgentCore requires agent_runtime_arn")
        logger.debug("AgentCore invoke: region=%s (agent_runtime_arn set=%s, bearer_token set=%s)", region, bool(agent_runtime_arn), bool(bearer_token))  # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure -- False positive: logs only booleans (whether the ARN/token are set), never their values.
    elif framework == "langfuse":
        if not langfuse_public_key or not langfuse_secret_key:
            raise HTTPException(400, "Langfuse requires public_key and secret_key")
    else:
        if not endpoint:
            raise HTTPException(400, f"{framework} agents require an endpoint URL")

    # Parse ground truth data (from upload or S3)
    import re as _re

    dataset_name = ""
    dataset_s3_path = ""

    if data_source == "s3" and s3_data_path:
        # Download from S3
        match = _re.match(r"s3://([^/]+)/(.+)", s3_data_path)
        if not match:
            raise HTTPException(400, "Invalid S3 path. Expected: s3://bucket/key")
        bucket, key = match.group(1), match.group(2)
        s3_client = boto3.client("s3")
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=key)
            content = resp["Body"].read()
        except Exception as e:
            raise HTTPException(400, f"Failed to read from S3: {e}")
        fname = key.split("/")[-1]
        dataset_name = fname
        dataset_s3_path = s3_data_path
    elif file:
        content = await file.read()
        fname = file.filename or "upload.csv"
        dataset_name = fname

        # Save uploaded file to S3 (same bucket as results)
        try:
            from uaef.config import get_config
            cfg = get_config()
            s3_client = boto3.client("s3")
            dataset_key = f"datasets/{fname}"
            s3_client.put_object(
                Bucket=cfg.storage.s3_bucket,
                Key=dataset_key,
                Body=content,
                ContentType="application/octet-stream",
            )
            dataset_s3_path = f"s3://{cfg.storage.s3_bucket}/{dataset_key}"
        except Exception as e:
            logger.warning("Failed to save dataset to S3: %s", e)
            dataset_s3_path = ""
    else:
        raise HTTPException(400, "No file or S3 path provided")

    try:
        df = load_ground_truth_file(content=content, filename=fname)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {e}")

    if df.empty:
        raise HTTPException(400, "Uploaded file is empty")

    gt_rows = df.to_dict(orient="records")

    # Resolve metrics
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None

    # Get adapter from registry
    try:
        adapter = get_adapter(framework)
    except ValueError:
        raise HTTPException(400, f"Unknown agent type: {framework}. Available: {list_adapters()}")

    # Process each ground truth row
    traces = []
    ground_truths = []
    errors = []
    progress = []

    for i, row in enumerate(gt_rows):
        query, expected, context, expected_tools = parse_ground_truth_row(row)

        try:
            # Invoke agent based on framework
            if framework == "bedrock":
                raw_output = invoke_bedrock_agent(agent_id, alias_id, query, region)
            elif framework == "agentcore":
                raw_output = _invoke_agentcore(agent_runtime_arn, query, region, bearer_token)
            elif framework == "langfuse":
                raw_output = invoke_langfuse_trace(
                    query, expected, langfuse_public_key, langfuse_secret_key, langfuse_host,
                )
            else:
                raw_output = invoke_http_agent_for_framework(endpoint, query, framework)

            # Transform through adapter
            trace = adapter.transform_to_canonical(raw_output)
            traces.append(trace)

            # Build ground truth
            gt = GroundTruth(
                expected_output=expected,
                expected_tool_calls=expected_tools,
                context_documents=[context] if context else [],
            )
            ground_truths.append(gt)

            # Extract agent response for display
            agent_response = ""
            for msg in trace.messages:
                if msg.role == MessageRole.ASSISTANT:
                    agent_response = msg.content or ""

            progress.append({
                "index": i + 1,
                "query": query,
                "agent_response": agent_response[:500],
                "status": "ok",
            })

        except Exception as e:
            logger.warning("Query %d failed during evaluation: %s", i + 1, e)
            client_error = "Agent invocation or evaluation failed for this query."
            errors.append({
                "index": i + 1,
                "query": query,
                "error": client_error,
            })
            progress.append({
                "index": i + 1,
                "query": query,
                "status": "error",
                "error": client_error,
            })

    if not traces:
        raise HTTPException(500, f"All {len(gt_rows)} queries failed. See server logs for details.")

    # Run batch evaluation
    eval_kwargs = {}
    if metric_list:
        eval_kwargs["metrics"] = metric_list

    # Enable debug logging for evaluation
    import logging
    logging.getLogger("uaef").setLevel(logging.DEBUG)

    # Log trace info before evaluation
    for i, t in enumerate(traces):
        msgs = [(m.role.value, m.content[:80] if m.content else "") for m in t.messages]
        logger.debug("Trace %d: messages=%s, tool_calls=%d, latency=%s, tokens=%s/%s", i, msgs, len(t.tool_calls), t.latency, t.input_tokens, t.output_tokens)  # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure -- False positive: logs integer token COUNTS (usage metrics), not credentials.

    should_persist = persist.lower() == "true"
    if should_persist:
        eval_kwargs["persist"] = True
        if experiment_name:
            eval_kwargs["experiment_name"] = experiment_name
        if experiment_objective:
            eval_kwargs["experiment_objective"] = experiment_objective

    results = batch_evaluate(
        traces=traces,
        ground_truths=ground_truths,
        max_workers=4,
        **eval_kwargs,
    )

    # Debug: log what came back from evaluation
    for i, r in enumerate(results):
        logger.debug("Result %d: overall=%s, dims=%d", i, r.overall_score, len(r.dimension_results))
        for dim in r.dimension_results:
            scores_str = ", ".join(f"{ms.metric_name}={ms.score}" for ms in dim.metric_scores)
            logger.debug("  %s: agg=%s, metrics=[%s]", dim.dimension_name, dim.aggregate_score, scores_str)

    # Build per-row presentation data. Aggregates are read from the stored
    # experiment record below; the presentation layer never recomputes them.
    rows_out = []
    for idx, r in enumerate(results):
        # Extract agent response
        agent_resp = ""
        for msg in traces[idx].messages:
            if msg.role == MessageRole.ASSISTANT:
                agent_resp = msg.content or ""

        row_data = {
            "index": idx + 1,
            "query": progress[idx]["query"] if idx < len(progress) else "",
            "agent_response": agent_resp[:500],
            "expected_answer": str(ground_truths[idx].expected_output)[:500],
            "overall_score": r.overall_score,
            "passed": r.passed,
            "dimension_scores": {},
            "metric_scores": {},
        }
        for dim in r.dimension_results:
            row_data["dimension_scores"][dim.dimension_name] = dim.aggregate_score
            for ms in dim.metric_scores:
                # Only include metrics the user selected
                if metric_list and ms.metric_name not in metric_list:
                    continue
                row_data["metric_scores"][ms.metric_name] = ms.score
        rows_out.append(row_data)

    exp_id = None
    if results and results[0].experiment_id:
        exp_id = str(results[0].experiment_id)

        # Update DynamoDB row with framework and dataset info
        # Also re-save S3 results with query/response metadata for later viewing
        if should_persist:
            try:
                from uaef.storage.dynamodb_s3 import get_storage
                store = get_storage()
                store._table.update_item(
                    Key={"experiment_id": exp_id},
                    UpdateExpression="SET metadata = :m",
                    ExpressionAttributeValues={":m": {
                        "framework": framework,
                        "agent_name": agent_name,
                        "dataset_name": dataset_name,
                        "dataset_s3_path": dataset_s3_path,
                    }},
                )

                # Enrich stored results with query/response for later retrieval
                enriched_results = []
                for idx, r in enumerate(results):
                    r_dict = r.model_dump(mode="json")
                    agent_resp = ""
                    for msg in traces[idx].messages:
                        if msg.role == MessageRole.ASSISTANT:
                            agent_resp = msg.content or ""
                    r_dict.setdefault("metadata", {})
                    r_dict["metadata"]["query"] = progress[idx]["query"] if idx < len(progress) else ""
                    r_dict["metadata"]["agent_response"] = agent_resp[:500]
                    r_dict["metadata"]["expected_output"] = str(ground_truths[idx].expected_output)[:500]
                    enriched_results.append(r_dict)

                store._write_s3_results(exp_id, {
                    "experiment_id": exp_id,
                    "experiment_name": experiment_name or "evaluation",
                    "experiment_objective": experiment_objective or "",
                    "evaluations": enriched_results,
                })
            except Exception as e:
                logger.warning("Failed to enrich persisted results: %s", e)

    # Report aggregates read from the stored experiment record. The
    # persistence layer computes and stores average_scores /
    # overall_average_score; the presentation layer never recomputes them.
    # If the aggregates are absent (e.g. the run was not persisted), return
    # an error indication rather than recomputing here (Req 12.4, 12.5).
    if not (should_persist and exp_id):
        raise HTTPException(
            422,
            "Stored experiment aggregates are unavailable because the "
            "evaluation was not persisted. Re-run with persistence enabled; "
            "the presentation layer does not recompute aggregates.",
        )

    from uaef.storage.dynamodb_s3 import get_storage
    stored_exp = get_storage().get_experiment(exp_id)
    stored_average_scores = stored_exp.get("average_scores") if stored_exp else None
    stored_overall_avg = stored_exp.get("overall_average_score") if stored_exp else None
    if stored_average_scores is None or stored_overall_avg is None:
        raise HTTPException(
            422,
            f"Experiment {exp_id} has no stored aggregates "
            "(average_scores / overall_average_score) to report.",
        )
    average_scores = {k: float(v) for k, v in stored_average_scores.items()}
    overall_avg = float(stored_overall_avg)

    return {
        "experiment_id": exp_id,
        "experiment_name": experiment_name or "evaluation",
        "experiment_objective": experiment_objective or "",
        "framework": framework,
        "dataset_name": dataset_name,
        "dataset_s3_path": dataset_s3_path,
        "evaluation_count": len(results),
        "total_queries": len(gt_rows),
        "successful_queries": len(traces),
        "failed_queries": len(errors),
        "overall_average_score": overall_avg,
        "average_scores": average_scores,
        "rows": rows_out,
        "errors": errors,
        "persist": should_persist,
    }
