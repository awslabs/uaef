# Use Cases

Worked patterns for the places UAEF usually gets wired in. Each is
self-contained — pick the one that matches your situation.

## Batch evaluation

Evaluate many traces at once, in parallel.

```python
from uaef.api import batch_evaluate

results = batch_evaluate(
    traces=[trace1, trace2, trace3],
    ground_truths=[gt1, gt2, gt3],
    max_workers=4,
    persist=True,
    experiment_name="Batch Regression Test",
)

avg = sum(r.overall_score for r in results) / len(results)
print(f"Average score: {avg:.2f}")
```

## Development: spot-check during iteration

Score a single trace and drill into the dimensions that failed.

```python
result = evaluate(
    trace=trace,
    ground_truth=gt,
    thresholds={"Tool Calling": 0.8, "Response Quality": 0.8},
)
if not result.passed:
    for dim in result.dimension_results:
        print(f"{dim.dimension_name}: {dim.aggregate_score:.2f}")
```

Set the `thresholds` you want to gate on, keyed by the exact dimension name — see
[Multi-Turn Evaluation Workflow](../Advanced/multi-turn-evaluation-workflow.md#default-dimension-weights).
The CI example below takes the other approach and asserts on `overall_score` directly.

## CI/CD: automated regression testing

Fail the build when quality drops below a threshold.

```python
results = batch_evaluate(
    traces=test_traces,
    ground_truths=test_gts,
    persist=True,
    experiment_name=f"ci-build-{build_number}",
)

avg = sum(r.overall_score for r in results) / len(results)
assert avg >= 0.8, f"Quality regression: {avg:.2f}"
```

## Production: real-time monitoring

Score live traffic and alert on low scores.

```python
def handle_request(user_input):
    response = agent.run(user_input)
    trace = build_trace(response)

    result = evaluate(
        trace=trace,
        persist=True,
        experiment_name="production-monitoring",
    )

    if result.overall_score < 0.6:
        alert_team(f"Low score: {result.overall_score:.2f}")

    return response
```

Persisting every request to the same experiment keeps a running average in
DynamoDB — see [Persistence and Configuration](persistence-and-configuration.md).

## Platform integration: plug into any agentic system

Wrap the adapter and `evaluate()` in middleware so every agent call on your
platform is scored.

```python
from uaef.api import evaluate
from uaef.adapters import get_adapter

class EvaluationMiddleware:
    def __init__(self, framework="langgraph", experiment_name="my-platform"):
        self.adapter = get_adapter(framework)
        self.experiment_name = experiment_name

    def evaluate_trace(self, raw_trace, ground_truth=None):
        trace = self.adapter.transform_to_canonical(raw_trace)
        return evaluate(
            trace=trace,
            ground_truth=ground_truth,
            persist=True,
            experiment_name=self.experiment_name,
        )
```

## Using a framework adapter directly

```python
from uaef.adapters import LangGraphAdapter

adapter = LangGraphAdapter()
trace = adapter.transform_to_canonical(langgraph_output)

result = evaluate(trace=trace, persist=True, experiment_name="LangGraph Agent")
```

Or resolve one by name:

```python
from uaef.adapters import get_adapter

adapter = get_adapter("langgraph")  # bedrock, langchain, langfuse, strands, agentcore
trace = adapter.transform_to_canonical(raw_output)
```

For a custom framework, `GenericJSONAdapter` maps your schema onto the canonical
trace — see [Adapters](adapters.md).

## Calling RAGAS and DeepEval directly

See [Metrics Catalog](metrics-catalog.md#calling-the-connectors-directly).

## Related

- [Multi-Turn Evaluation Workflow](../Advanced/multi-turn-evaluation-workflow.md) — the end-to-end evaluation flow
- [Experiments](experiments.md) — tracking runs and comparing them
- [Human-in-the-Loop](hitl.md) — adding human review
- `notebooks/` — further examples, including analysis and reporting
