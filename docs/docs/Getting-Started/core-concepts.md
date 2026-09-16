# Core Concepts

The vocabulary UAEF uses, and how the pieces fit together.

## Layered architecture

1. **Adapter layer** — transforms framework-specific traces into the canonical format
2. **Core data models** — standardized representation of traces and evaluations
3. **Metrics layer** — 36 built-in metrics across 7 dimensions
4. **Evaluation engine** — orchestrates metric calculation and aggregation
5. **Experiment management** — versions configurations and tracks results
6. **Analysis & insights** — root cause analysis and recommendations
7. **Reporting layer** — dashboards and exportable reports

## Canonical trace format

Every adapter produces the same shape, so metrics never need to know the source framework.

`AgentTrace`
:   The complete execution record.

    - `trace_id` — unique identifier
    - `messages` — the conversation
    - `tool_calls` — tool invocations
    - `metadata` — latency, tokens, cost

`Message`
:   A single message.

    - `role` — `user`, `assistant`, `system`, or `tool`
    - `content` — message text
    - `tool_calls` — associated calls, for assistant messages

`ToolCall`
:   A tool invocation.

    - `name` — tool identifier
    - `arguments` — input parameters
    - `result` — tool output
    - `timestamp` — execution time

Full field reference: [`uaef.models`](../API-Reference/models.md).

## Evaluation dimensions

| Dimension | What it measures |
|-----------|------------------|
| Tool Calling | Tool selection, sequencing, parameter quality, MCP compliance |
| Response Quality | Relevance, completeness, hallucination, accuracy |
| Responsible AI | Safety, bias, prompt injection, toxicity |
| Performance | Latency, token efficiency, cost efficiency, throughput |
| Multi-Turn | Context retention, coherence, conversation completeness, turn efficiency |
| Multi-Agent | Agent utilization, delegation quality, workflow completion, coordination |
| Reasoning | Chain-of-thought coherence, logical consistency, step correctness, fallacies |

The per-metric list is in the [Metrics Catalog](../Guides/metrics-catalog.md).

## Metric types

**Deterministic**
:   Rule-based, no ground truth needed. Latency, token count, cost. Fast (under 100 ms) and
    reproducible.

**Ground-truth-based**
:   Compares output against expected results. Tool accuracy, completeness. Requires test
    cases with expected outputs.

**LLM-based**
:   Uses an LLM judge for subjective evaluation. Relevance, coherence, safety. Requires AWS
    Bedrock access, and returns reasoning alongside the score.

## Evaluation modes

**Online**
:   Real-time, during agent execution. Deterministic metrics run synchronously (under
    500 ms); LLM metrics run asynchronously so partial results are available immediately.
    Used for production quality gates.

**Offline**
:   Batch evaluation of historical traces. Parallel trace processing, batched LLM calls,
    aggregate statistics. Used for comprehensive analysis.

See [Multi-Turn Evaluation Workflow](../Advanced/multi-turn-evaluation-workflow.md) for the end-to-end flow.

## Ground truth

Expected outputs for validation:

- `expected_output` — the correct response text
- `expected_tool_calls` — tools that should be called
- `expected_arguments` — expected tool parameters
- `context_documents` — reference information

Required for accuracy and completeness metrics, optional for relevance and coherence.
Building datasets: [Ground Truth Datasets](../Guides/ground-truth-datasets.md).

## Experiments

**Experiment**
:   A versioned agent configuration — framework, model, prompt, tools — plus its metric set
    and thresholds. Holds multiple runs.

**Run**
:   One execution of an experiment: configuration snapshot, evaluation results, aggregate
    metrics, comparison against baseline.

**Baseline**
:   The reference run for regression detection. Updated as the agent improves.

Workflow: [Experiments](../Guides/experiments.md).

## LLM judge

UAEF uses AWS Bedrock as the judge for subjective metrics: validated prompt templates per
metric, versioned for reproducibility, structured JSON output with reasoning, and retry
with exponential backoff.

Calibration compares judge scores against human judgments, computes agreement, and feeds
prompt adjustments — see [Human-in-the-Loop](../Guides/hitl.md).

!!! warning "Evaluated output is untrusted input"
    Agent output under evaluation is treated as untrusted data in every judge prompt. See
    `SECURITY.md`.

## Analysis and reporting

Analysis covers root cause categorization, failure clustering to surface systematic
issues, trend analysis over time, and prioritized recommendations.

Reports come as an executive dashboard, a side-by-side comparison dashboard, per-dimension
deep dives, a safety report for Responsible AI concerns, and a regression report.
