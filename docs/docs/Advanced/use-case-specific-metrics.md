# Use-Case-Specific Metrics (opt-in)

UAEF is a general-purpose agent evaluation framework. A few metrics only
make sense for specific use cases — the prompts they ask an LLM judge to
answer aren't meaningful for every agent. Rather than pollute the default
metric set with judgments that are noise for most users, those metrics live
in [`src/uaef/metrics/use_case_specific.py`](https://github.com/awslabs/uaef/blob/main/src/uaef/metrics/use_case_specific.py) and are **opt-in**.

Today this is the **contact-center / customer-service** family:

| Metric | Phase | What it measures | Score | Ground truth | LLM judge |
|--------|-------|-------------------|-------|--------------|-----------|
| `containment` | full-trace | Was the conversation handled by the agent without escalating, transferring, or redirecting to a human or external support channel? | 0.0 (escalated) or 1.0 (contained); raw 0/1 in `metadata["raw"]` | no | yes (Bedrock) |
| `resolution` | full-trace | Was the customer's issue resolved by end of conversation? An agent transferring to human support after providing instructions is treated as resolved; the customer asking for a human transfer is unresolved. | 0.0 or 1.0; raw 0/1 in `metadata["raw"]` | no | yes (Bedrock) |

Both inherit `_AMACEFullTraceMetric` and reuse the shared Bedrock judge
helper from `multi_turn.py`. The classes are 100% self-contained beyond
that — no other module in `uaef.metrics` references them.

## Opting in

One call, anywhere before you invoke `evaluate()`:

```python
from uaef.api import evaluate
from uaef.metrics.registry import register_use_case_specific_metrics

register_use_case_specific_metrics()

result = evaluate(
    trace=trace,
    ground_truth=gt,
    metrics=["containment", "resolution"],
)
```

The call is **idempotent** — re-running it (e.g., a notebook cell
re-executed) is a no-op rather than raising `ValueError("Metric already
registered")`. Once registered, both metrics behave like every other
Multi-Turn metric: they appear in `list_metrics()`, group under the
`"Multi-Turn"` dimension, and route to the full-trace phase via the existing
`FULL_TRACE_ONLY` set in `BaseEvaluator._run_multi_turn`.

## Direct class import (advanced)

If you'd rather register them yourself (e.g., to wrap them, subclass them,
or selectively register only one):

```python
from uaef.metrics import register_metric
from uaef.metrics.use_case_specific import ContainmentMetric

register_metric(ContainmentMetric)   # standard custom-metric path
```

## Without opting in

If you call `evaluate(trace, metrics=["containment"])` without first
registering, you get a clear error from
`uaef.api.__init__:_metrics_to_metric_set`:

```
ValueError: Unknown metric 'containment'. Available: [<list of registered metrics>]
```

This is by design — the goal of opting in is to keep general-purpose
evaluations from accidentally running irrelevant judges and racking up
Bedrock cost.

## When to use

These metrics fit when:

- Your agent is a customer-service / IVR / contact-center bot.
- The conversation has a notion of "escalation to human" (whether the agent
  has a transfer tool, or simply says "please contact our support team").
- You care about end-of-conversation issue resolution as a binary.

They do *not* fit when:

- The agent is a coding assistant, internal tooling agent, RAG bot,
  question-answering bot, etc. — there's no "human escalation" semantics
  and "resolution" is overloaded.
- You're evaluating multi-agent orchestration where "did the agent escalate"
  is replaced by inter-agent delegation (use `delegation_quality` instead).

## Code locations

| Component | File |
|-----------|------|
| Metric implementations | [`src/uaef/metrics/use_case_specific.py`](https://github.com/awslabs/uaef/blob/main/src/uaef/metrics/use_case_specific.py) |
| Opt-in registration helper | `src/uaef/metrics/registry.py:register_use_case_specific_metrics` |
| Public re-exports | [`src/uaef/metrics/__init__.py`](https://github.com/awslabs/uaef/blob/main/src/uaef/metrics/__init__.py) (`ContainmentMetric`, `ResolutionMetric`, `register_use_case_specific_metrics`) |
| Phase routing | `src/uaef/evaluation/base_evaluator.py:_run_multi_turn` (`FULL_TRACE_ONLY` set still contains both names) |
| Shared judge helpers (reused via import) | [`src/uaef/metrics/multi_turn.py`](https://github.com/awslabs/uaef/blob/main/src/uaef/metrics/multi_turn.py) (`_AMACEFullTraceMetric`, `_invoke_bedrock_judge`, etc.) |
