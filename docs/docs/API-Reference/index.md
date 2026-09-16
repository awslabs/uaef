# API Reference

Generated from docstrings in `src/uaef/`. If something here looks wrong, fix the docstring.

## Core

| Module | What it holds |
|--------|---------------|
| [`uaef.models`](models.md) | `AgentTrace`, `Message`, `ToolCall`, `GroundTruth`, `MetricScore`, `EvaluationResult` |
| [`uaef.adapters`](adapters.md) | Framework adapters and the adapter registry |
| [`uaef.metrics`](metrics.md) | All built-in metrics, `BaseMetric`, the metric registry |
| [`uaef.evaluation`](evaluation.md) | `evaluate()` and the evaluator classes |

## Advanced

| Module | What it holds |
|--------|---------------|
| [`uaef.llm_judge`](llm_judge.md) | Bedrock judge, prompt templates, calibration |
| [`uaef.experiments`](experiments.md) | Experiment management, comparison, regression detection |
| [`uaef.analysis`](analysis.md) | Root cause, clustering, trends, recommendations |
| [`uaef.reporting`](reporting.md) | Dashboards, deep dives, safety reports, export |

## Specialized

| Module | What it holds |
|--------|---------------|
| [`uaef.hitl`](hitl.md) | Human-in-the-loop review queues, agreement, calibration |
| [`uaef.security`](security.md) | PII detection and redaction, trace encryption |
| [`uaef.storage`](storage.md) | DynamoDB + S3 persistence |
| [`uaef.client`](client.md) | `UAEFClient` remote SDK |
| [`uaef.integrations`](integrations.md) | RAGAS and DeepEval connectors |
| [`uaef.simulator`](simulator.md) | Conversation simulation — **placeholder, not implemented** |

## Common entry point

Most callers need only `evaluate()`:

```python
from uaef.api import evaluate

result = evaluate(
    trace=trace,
    ground_truth=ground_truth,
    metrics=["answer_relevance", "safety_score"],
    persist=True,
    experiment_name="Customer Support Agent v2",
)
```

Worked examples live in the [Guides](../Guides/use-cases.md).
