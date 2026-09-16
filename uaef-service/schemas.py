# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response schemas for the uaef-service application.

This module is part of the standalone ``uaef-service`` deployable app. It MUST
NOT import UAEF library source — the service consumes the published ``uaef``
package only at the Worker Lambda layer. These schemas depend solely on
``pydantic``.

Custom-metric and GenericJSONAdapter schema-mapping fields are intentionally
excluded: the service accepts only curated server-side metric names. Custom
metrics and schema mappings remain available in library mode.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Ceiling on ``max_workers`` (security review M-04): the requested worker
#: pool fans out to concurrent Bedrock-judge / agent-invoke calls, so an
#: unbounded value lets a single request drive unbounded concurrent paid
#: inference. 10 comfortably covers the library default of 4 while capping
#: per-request concurrency.
MAX_WORKERS_CEILING = 10

#: Ceiling on the number of rows a single ``/batch-evaluate`` request may
#: submit inline (security review M-04). Each row can trigger multiple
#: Bedrock-judge calls; an unbounded row count lets a single authenticated
#: request fan out to an unbounded number of paid invocations (wallet-drain
#: DoS). Larger datasets should be split across multiple requests.
MAX_BATCH_ROWS = 500


# ---- POST /evaluate (request) ----
class EvaluateRequest(BaseModel):
    trace: dict | None = None             # inline canonical/raw trace
    traceRef: str | None = None           # OR S3 key for large inline trace
    ground_truth: dict | None = None
    adapter: str | None = None
    metrics: list[str] | None = None      # curated server-side names only
    context: list[str] | None = None
    persist: bool = False
    experiment_name: str | None = None
    experiment_objective: str | None = None
    # custom metrics / GenericJSON schema mappings are NOT accepted server-side


# ---- POST /batch-evaluate (request) ----
class BatchEvaluateRequest(BaseModel):
    traces: list[dict] | None = Field(default=None, max_length=MAX_BATCH_ROWS)
    tracesRef: str | None = None          # S3 key (preferred for large batches)
    ground_truths: list[dict] | None = Field(default=None, max_length=MAX_BATCH_ROWS)
    adapter: str | None = None
    metrics: list[str] | None = None
    max_workers: int = Field(default=4, ge=1, le=MAX_WORKERS_CEILING)
    persist: bool = False
    experiment_name: str | None = None
    experiment_objective: str | None = None


# ---- POST /invoke-evaluate (request) ----
class InvokeEvaluateRequest(BaseModel):
    """Invoke a network-reachable agent per question, then evaluate the traces.

    The deployed Worker reaches out to the agent (AgentCore runtime, Bedrock
    agent, or HTTP endpoint — anything reachable from AWS), collects a trace per
    question, and batch-evaluates them. Local/laptop agents are NOT reachable
    from the cloud and must use the notebook (UAEFClient) path instead.
    """
    framework: str                         # agentcore|bedrock_agent|http|langgraph|...
    connection: dict = {}                  # agent connection fields (arn, endpoint, region…)
    s3_data_path: str | None = None        # s3:// URI OR a payload-bucket key; Worker parses via uaef.data
    filename: str | None = None            # original filename, for CSV/XLSX format detection
    metrics: list[str] | None = None       # curated server-side names only
    persist: bool = False
    experiment_name: str | None = None
    experiment_objective: str | None = None
    agent_name: str | None = None


# ---- POST /evaluate | /batch-evaluate (response) ----
class JobCreatedResponse(BaseModel):
    jobId: str
    status: str = "PENDING"
    message: str = "Job started. Poll GET /jobs/{jobId} for results."


# ---- GET /jobs/{jobId} (response) ----
class JobStatusResponse(BaseModel):
    jobId: str
    status: str                            # PENDING|PROCESSING|COMPLETED|FAILED
    operation: str                         # evaluate|batch_evaluate
    experimentId: str | None = None
    resultSummary: dict | None = None      # small inline summary
    resultRef: str | None = None           # S3 key / presigned GET for full result
    error: dict | None = None              # {code, message} when FAILED
    progress: dict | None = None           # {total, completed, dimensions: {name: status}}
    createdAt: str
    updatedAt: str


# ---- POST /payloads (response) ----
class PresignResponse(BaseModel):
    uploadUrl: str                         # presigned S3 PUT
    key: str                               # reference to pass back as traceRef/tracesRef
    expiresIn: int = 900
