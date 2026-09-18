# Universal Agent Evaluation Framework (UAEF)

A framework-agnostic evaluation system for AI agents. Plug it into any agentic platform to measure agent quality across 36 built-in metrics, persist results to AWS, and track performance over time.

UAEF works with LangGraph, LangChain, AWS Bedrock, Langfuse, Strands, AgentCore, or any custom framework via adapters. One `evaluate()` call gives you scores across tool calling, response quality, safety, performance, reasoning, and more.

| I want to… | Go to |
|---|---|
| Use UAEF as a Python library | [Installation](#installation) · [Quick Start](#quick-start) |
| Deploy the service to AWS | [Deployment](#deployment) → **[uaef-service/README.md](uaef-service/README.md)** |
| Run the demo app locally | [demo/README.md](demo/README.md) |
| See every metric | [docs/docs/Guides/metrics-catalog.md](docs/docs/Guides/metrics-catalog.md) |
| Browse all documentation | **[docs/docs/index.md](docs/docs/index.md)** — or build the site: `cd docs && make docs` |

## Architecture

```
                        ┌──────────────────┐
                        │  User / Platform  │
                        └──┬────────┬───┬──┘
                           │        │   │
         agent traces +    │        │   │  create experiment
         ground truth      │        │   │  (persist=True or direct)
                           │        │   │
                           ▼        │   ▼
┌──────────────────────────────────────────────────────────────┐
│                        UAEF API                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                 Framework Adapters                      │  │
│  │  LangGraph │ Bedrock │ LangChain │ Langfuse │ Strands  │  │
│  │  AgentCore │ GenericJSON (custom schema mapping)        │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │ canonical AgentTrace               │
│                          ▼                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │               Evaluation Engine                        │  │
│  │  SingleAgent │ MultiTurn │ MultiAgent │ Offline(batch) │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                    │
│                          ▼                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  Metrics (36 built-in)                  │  │
│  │                                                        │  │
│  │  Tool Calling      Response Quality   Responsible AI   │  │
│  │  Performance       Multi-Turn         Multi-Agent      │  │
│  │  Reasoning         + Custom metrics   + RAGAS/DeepEval │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                    │
│                          ▼                                    │
│                   EvaluationResult ─────────────────────┐     │
│                    │           │                        │     │
│          ┌─────────┘           └──────────┐            │     │
│          ▼                                ▼            │     │
│   returned to caller            persist=True?          │     │
│          │                            │                │     │
│          │                            ▼                │     │
│          │   ┌────────────────────────────────────┐    │     │
│          │   │     AWS Persistence Layer          │    │     │
│          │   │                                    │    │     │
│          │   │  ┌────────────┐  ┌──────────────┐ │    │     │
│          │   │  │  DynamoDB   │  │     S3       │ │    │     │
│          │   │  │            │  │              │ │    │     │
│          │   │  │ experiment │  │ Full results │ │    │     │
│          │   │  │ avg_scores │  │ JSON per     │ │    │     │
│          │   │  │ eval_count │  │ experiment   │ │    │     │
│          │   │  │ result_path│─▶│              │ │    │     │
│          │   │  └────────────┘  └──────────────┘ │    │     │
│          │   └──────────────────────┬─────────────┘    │     │
│          │                          │                  │     │
│          │   ┌──────────────────────┘                  │     │
│          │   │                                         │     │
│          ▼   ▼                                         │     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Analysis & Reporting                      │  │
│  │  Root cause │ Clustering │ Trends │ Recommendations    │  │
│  │  Dashboards │ Regression detection │ Safety reports    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

See [docs/docs/Getting-Started/core-concepts.md](docs/docs/Getting-Started/core-concepts.md) for the canonical trace format, evaluation dimensions, and evaluation modes.

## Installation

### From the repository (recommended)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. `pyproject.toml` + `uv.lock` is the single source of truth for all dependencies.

```bash
git clone https://github.com/awslabs/uaef.git
cd uaef
uv sync            # install all dependencies from the lockfile
```

### From a distributed package

If you received UAEF as a `.whl` or `.tar.gz`:

```bash
uv pip install ./packages/uaef-0.2.0-py3-none-any.whl
```

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- AWS credentials — for LLM-based metrics via Bedrock, and for persistence via DynamoDB/S3
- `boto3`, included as a core dependency

`uv sync` installs **everything needed to run every metric and use every capability** — all metric backends (RAGAS, DeepEval), every adapter, analysis and reporting, the Bedrock LLM judge, and the `UAEFClient` remote SDK. Nothing else to install. The only optional add-on is a Jupyter kernel for notebook use:

```bash
uv sync --extra notebooks   # or --extra all, which is core + the notebook kernel
```

The capability extras (`integrations`, `langgraph`, `analysis`, `judge`, `client`, `server`) are retained as no-op aliases so older install commands keep resolving; they add nothing.

## Quick Start

### 1. Evaluate a single trace

```python
from uaef.api import evaluate
from uaef.models import AgentTrace, Message, MessageRole, GroundTruth
from datetime import datetime

trace = AgentTrace(
    trace_id="trace_001",
    messages=[
        Message(role=MessageRole.USER, content="What is the capital of France?", timestamp=datetime.utcnow()),
        Message(role=MessageRole.ASSISTANT, content="The capital of France is Paris.", timestamp=datetime.utcnow()),
    ],
    tool_calls=[],
    session_id="session_123",
)

ground_truth = GroundTruth(expected_output="Paris", expected_tool_calls=[])

result = evaluate(trace=trace, ground_truth=ground_truth)

print(f"Score: {result.overall_score:.2f}")
print(f"Passed: {result.passed}")
```

### 2. Evaluate and persist to AWS

```python
result = evaluate(
    trace=trace,
    ground_truth=ground_truth,
    persist=True,
    experiment_name="Customer Support Agent v2",
    experiment_objective="Measure quality after prompt changes",
)

print(f"Experiment ID: {result.experiment_id}")
```

Results go to DynamoDB (one row per experiment) and S3 (full JSON). See [docs/docs/Guides/persistence-and-configuration.md](docs/docs/Guides/persistence-and-configuration.md) for the schema, experiment resolution rules, and configuration.

**Next:** [batch evaluation, CI/CD gating, production monitoring, and platform middleware →](docs/docs/Guides/use-cases.md)

## Metrics

36 built-in metrics across 7 dimensions, plus 2 opt-in use-case-specific metrics and 15 more via the RAGAS and DeepEval integrations.

| Dimension | Metrics |
|-----------|---------|
| Tool Calling | `tool_selection_accuracy`, `tool_sequence_correctness`, `parameter_quality`, `mcp_compliance` |
| Response Quality | `answer_relevance`, `completeness`, `hallucination_score`, `accuracy` |
| Responsible AI | `safety_score`, `bias_score`, `prompt_injection_detection`, `toxicity_score` |
| Performance | `latency_score`, `token_efficiency`, `cost_efficiency`, `throughput` |
| Multi-Turn | `context_retention`, `coherence`, `conversation_completeness`, `turn_efficiency`, `role_adherence`, `holistic_llm_judge`, `user_satisfaction`, `sentiment`, `agent_tone`, `naturalness`, `instruction_compliance`, `optimum_turns` |
| Multi-Agent | `agent_utilization`, `delegation_quality`, `workflow_completion`, `coordination_efficiency` |
| Reasoning | `chain_of_thought_coherence`, `logical_consistency`, `reasoning_step_correctness`, `fallacy_detection` |

Run a subset by name:

```python
result = evaluate(trace=trace, metrics=["answer_relevance", "safety_score", "latency_score"])
```

**Full catalog** — RAGAS and DeepEval metric tables, the opt-in contact-center metrics, Stickler, and how to register your own: [docs/docs/Guides/metrics-catalog.md](docs/docs/Guides/metrics-catalog.md)

## Supported Adapters

| Adapter | Framework |
|---------|-----------|
| `LangGraphAdapter` | LangGraph |
| `LangChainAdapter` | LangChain |
| `BedrockAgentAdapter` | AWS Bedrock Agents |
| `LangfuseAdapter` | Langfuse |
| `StrandsAdapter` | Strands |
| `AgentCoreAdapter` | AgentCore |
| `GenericJSONAdapter` | Any framework, via custom schema mapping |

All import from `uaef.adapters`. See [docs/docs/Guides/adapters.md](docs/docs/Guides/adapters.md) for usage and for mapping a custom schema.

## Deployment

UAEF has three deployable surfaces:

| Target | What you get | Guide |
|--------|--------------|-------|
| Local demo | React + FastAPI on `localhost`, for exploration | [demo/README.md](demo/README.md) |
| AWS `with_ui` | Fully serverless: Cognito, API Gateway, Worker Lambda, Step Functions, DynamoDB + S3, and a CloudFront-hosted React UI | **[uaef-service/README.md](uaef-service/README.md)** |
| AWS `with_eks` | Same backend, but the HTTP API is served by a FastAPI container on EKS instead of API Gateway. Bring your own UI. | **[uaef-service/README.md](uaef-service/README.md)** |

Both AWS modes are driven by one config file and one script:

```bash
cd uaef-service
cp config.yaml.example config.yaml     # set deploy_suffix, deploy_region, deploy_mode, auth.mode
                                       # and cors_allowed_origins (required — see below)
./scripts/cloud_deploy.sh
```

`cors_allowed_origins` is **required and has no default**: it is the allow-list of
origins permitted to call the API and to read/write the payload S3 bucket via
presigned URLs. The deploy script stops before making any change if it is unset,
so a fresh copy of `config.yaml.example` needs this one field filled in before the
first deploy.

For `with_ui`, **leave `cors_allowed_origins` blank** — `cloud_deploy.sh` derives the
UI's CloudFront origin itself, so one run of the script is all you need. On a first
deploy the origin does not exist yet, so CodeBuild deploys once to create the
distribution and then immediately redeploys with the real origin; that is two CDK
passes inside one build, not two script calls. Set a value only to allow *extra*
origins (e.g. `http://localhost:5173`), which are merged with the derived one.
`with_eks` deploys no CloudFront distribution, so it still requires an explicit origin.

> **Upgrading an existing deployment:** `cors_allowed_origins` was introduced by a
> security review and is absent from `config.yaml` files created before it. Add it
> (and `acknowledge_insecure_cors: false`) to your existing config before
> redeploying, using your current `UiUrl` as the origin. `with_eks` needs it too,
> for the payload bucket, even though it deploys no API Gateway.

See [uaef-service/README.md](uaef-service/README.md) for prerequisites (including the one-time `cdk bootstrap`), the full `config.yaml` reference, architecture diagrams, redeploy behavior, and teardown.

> The local demo under `demo/` is **not** what the AWS modes deploy — the deployed UI is built from `uaef-service/ui/app` by `uaef-service/buildspec.yml`.

### Tenant isolation

UAEF is a reference architecture designed to run **embedded inside a platform that provides user/tenant isolation**. Direct deployment in a multi-tenant environment without a host platform enforcing tenant isolation is not supported.

Cross-tenant separation is deployment separation: the host platform gives each tenant its own UAEF deployment (its own AWS account, or at minimum its own stack with its own DynamoDB tables, S3 buckets, and Cognito user pool), and authenticates callers through that deployment's Cognito user pool. Within a single deployment, UAEF additionally scopes every job and experiment read to the caller's own Cognito `sub`, so one authenticated user cannot read another's evaluation data — but that is defense in depth inside a tenant, not a tenant boundary.

Read **[SECURITY.md](SECURITY.md)** for the full model: what the host platform must provide, what UAEF enforces itself, the accepted residual risk of standalone multi-user deployment, and why evaluated-agent output is treated as untrusted input in every LLM-judge prompt.

## Additional Capabilities

| Module | What it does |
|--------|-------------|
| `uaef.analysis` | Root cause analysis, failure clustering, trend analysis, improvement tracking, recommendations |
| `uaef.reporting` | Executive dashboards, comparison dashboards, safety reports, HTML/PDF export |
| `uaef.security` | PII detection/redaction, data encryption for traces |
| `uaef.llm_judge` | Bedrock-based LLM judge with prompt templates and calibration |

## Project Structure

```
src/uaef/          # the library
uaef-service/      # AWS deployment — CDK stacks, Lambda handlers, deployed UI
deploy_eks/        # EKS container (FastAPI) + Helm chart
demo/              # local demo app (React + FastAPI)
docs/              # MkDocs site — see docs/README.md
notebooks/         # worked examples and sample agents
scripts/           # repo tooling
tests/
```

The `src/uaef/` breakdown is in [docs/docs/Contributing/development.md](docs/docs/Contributing/development.md#library-layout).

## Documentation

Full index: **[docs/docs/index.md](docs/docs/index.md)**. The pages below are Markdown and
render in the repo, but they are also the source for a MkDocs site — run
`cd docs && make docs` to browse it locally with search and navigation. See
[docs/README.md](docs/README.md) for how the site is built and published.

| Topic | Doc |
|-------|-----|
| First evaluation, core concepts, troubleshooting | [docs/docs/Getting-Started/quickstart.md](docs/docs/Getting-Started/quickstart.md) |
| Every metric, including integrations | [docs/docs/Guides/metrics-catalog.md](docs/docs/Guides/metrics-catalog.md) |
| Batch, CI/CD, production monitoring patterns | [docs/docs/Guides/use-cases.md](docs/docs/Guides/use-cases.md) |
| AWS persistence schema and configuration | [docs/docs/Guides/persistence-and-configuration.md](docs/docs/Guides/persistence-and-configuration.md) |
| Framework adapters | [docs/docs/Guides/adapters.md](docs/docs/Guides/adapters.md) |
| Experiment tracking and regressions | [docs/docs/Guides/experiments.md](docs/docs/Guides/experiments.md) |
| Human-in-the-loop review | [docs/docs/Guides/hitl.md](docs/docs/Guides/hitl.md) |
| Writing custom metrics | [docs/docs/Guides/custom-metrics.md](docs/docs/Guides/custom-metrics.md) |
| AWS deployment | [uaef-service/README.md](uaef-service/README.md) |
| Security model, tenant isolation, judge trust boundary | [SECURITY.md](SECURITY.md) |
