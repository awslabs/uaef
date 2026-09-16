# Getting Started with UAEF

## Overview

The Universal Agent Evaluation Framework (UAEF) is a comprehensive, framework-agnostic evaluation system that enables you to assess AI agent performance across any framework using consistent metrics, experiment tracking, and automated quality assurance.

This guide will help you get started with UAEF in minutes.

## Before you start

You need UAEF installed and AWS credentials with Bedrock access for the LLM-judge metrics.
See **[Installation](installation.md)** — it is the single source of truth for requirements
and install commands.

## Quick Start: Your First Evaluation

Let's evaluate a simple agent interaction in 5 minutes.

### Step 1: Import UAEF

```python
from datetime import datetime
from uuid import uuid4

from uaef.api import evaluate
from uaef.models import AgentTrace, Message, MessageRole, ToolCall, GroundTruth
```

### Step 2: Create an Agent Trace

An agent trace represents a complete interaction with your agent:

```python
# Create a trace of your agent's interaction
trace = AgentTrace(
    trace_id=uuid4(),
    framework="langgraph",  # or "bedrock", "langchain", etc.
    messages=[
        Message(
            role=MessageRole.USER,
            content="What's the weather in Paris?",
            timestamp=datetime.now()
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="I'll check the weather for you.",
            timestamp=datetime.now()
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="The weather in Paris is sunny with 22°C.",
            timestamp=datetime.now()
        )
    ],
    tool_calls=[
        ToolCall(
            name="get_weather",
            arguments={"city": "Paris", "units": "celsius"},
            result={"temperature": 22, "condition": "sunny"},
            timestamp=datetime.now()
        )
    ],
    input_tokens=150,
    output_tokens=75,
    latency=1.5
)
```

### Step 3: Create Ground Truth (Optional)

Ground truth helps evaluate accuracy:

```python
ground_truth = GroundTruth(
    expected_output="The weather in Paris is sunny with 22°C.",
    expected_tool_calls=[
        ToolCall(
            name="get_weather",
            arguments={"city": "Paris", "units": "celsius"},
            timestamp=datetime.now()
        )
    ]
)
```

### Step 4: Run Evaluation

`evaluate()` takes the trace directly and picks the right evaluator for you. The three
metrics below are all deterministic, so this first run needs no AWS credentials.

```python
result = evaluate(
    trace=trace,
    ground_truth=ground_truth,
    metrics=["tool_selection_accuracy", "mcp_compliance", "latency_score"],
    context=["Paris is the capital of France."],
)

print(f"Overall Score: {result.overall_score:.2f}")
print(f"Passed: {result.passed}")

for dimension in result.dimension_results:
    print(f"\n{dimension.dimension_name}: {dimension.aggregate_score:.2f}")
    for metric in dimension.metric_scores:
        # score is None when a metric had insufficient input to judge
        value = "n/a" if metric.score is None else f"{metric.score:.2f}"
        print(f"  - {metric.metric_name}: {value}")
```

### Expected Output

```
Overall Score: 1.00
Passed: True

Performance: 1.00
  - latency_score: 1.00

Tool Calling: 1.00
  - mcp_compliance: n/a
  - tool_selection_accuracy: 1.00
```

Dimension order is not guaranteed — don't rely on it.

`tool_selection_accuracy` is 1.0 because the agent called exactly the tool the ground truth
expected. `latency_score` is 1.0 because 1.5 s is under the 2000 ms default threshold.
`mcp_compliance` returns `None` rather than a score, with
`metric.metadata["warning"] == "not_applicable"` — every metric returns `None` and a warning
instead of guessing when the input doesn't apply or is insufficient, and `None` scores are
excluded from the dimension average rather than counted as zero. See
[Metrics Catalog](../Guides/metrics-catalog.md#edge-cases-all-metrics).

### Step 5: Add LLM-judged metrics

Once AWS credentials with Bedrock access are configured, add judge-based metrics to the same
call — nothing else changes:

```python
result = evaluate(
    trace=trace,
    ground_truth=ground_truth,
    metrics=["tool_selection_accuracy", "answer_relevance", "safety_score", "latency_score"],
    context=["Paris is the capital of France."],
)
```

Each LLM-judged metric is one Bedrock call per evaluation, so curate the list in tight
loops. The full list is in the [Metrics Catalog](../Guides/metrics-catalog.md).

## Tracking experiments

Version agent configurations, compare runs, and detect regressions:

```python
from uaef.experiments import ExperimentManager

manager = ExperimentManager()
experiment = manager.create_experiment(
    name="Customer Support Agent v2",
    description="Testing improved tool selection"
)
```

Full workflow — runs, baselines, regression detection: [Experiments](../Guides/experiments.md).

## Configuration

### AWS Credentials

For LLM-based metrics, configure AWS credentials:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

Or use AWS credential files (`~/.aws/credentials`).

### UAEF Configuration

Create a configuration file `uaef_config.yaml`:

```yaml
# LLM Judge Configuration
llm_judge:
  model: "anthropic.claude-3-sonnet-20240229-v1:0"
  region: "us-east-1"
  max_retries: 3
  timeout: 30

# Metric Configuration
metrics:
  default_set:
    - tool_selection_accuracy
    - answer_relevance
    - safety_score
    - latency_score
  
  thresholds:
    tool_calling: 0.8
    response_quality: 0.75
    responsible_ai: 0.9
    performance: 0.7

# Database Configuration
database:
  type: "sqlite"  # or "postgresql"
  path: "uaef.db"
  # For PostgreSQL:
  # host: "localhost"
  # port: 5432
  # database: "uaef"
  # user: "uaef_user"
  # password: "your_password"  # pragma: allowlist secret
```

Load configuration:

```python
from uaef.config import load_config

config = load_config("uaef_config.yaml")
```

## Framework Integration

UAEF supports multiple agent frameworks out of the box.

### LangGraph

```python
from uaef.adapters import get_adapter

adapter = get_adapter("langgraph")
trace = adapter.transform_to_canonical(langgraph_output)
```

### Bedrock Agents

```python
adapter = get_adapter("bedrock")
trace = adapter.transform_to_canonical(bedrock_response)
```

### LangChain

```python
adapter = get_adapter("langchain")
trace = adapter.transform_to_canonical(langchain_result)
```

### Custom Framework

```python
from uaef.adapters.generic import GenericJSONAdapter

schema_mapping = {
    "messages": {
        "path": "$.conversation.messages[*]",
        "role_field": "sender",
        "content_field": "text"
    },
    "tool_calls": {
        "path": "$.execution.tools[*]",
        "name_field": "tool_name",
        "arguments_field": "params"
    }
}

adapter = GenericJSONAdapter(schema_mapping)
trace = adapter.transform_to_canonical(custom_data)
```

## Common Patterns

### Pattern 1: Evaluate Multiple Traces

```python
from uaef.api import batch_evaluate

results = batch_evaluate(
    traces=[trace1, trace2, trace3],
    ground_truths=[gt1, gt2, gt3],      # optional; omit for metrics that don't need it
    metrics=["tool_selection_accuracy", "latency_score"],
)

avg_score = sum(r.overall_score for r in results) / len(results)
pass_rate = sum(1 for r in results if r.passed) / len(results)

print(f"Average Score: {avg_score:.2f}")
print(f"Pass Rate: {pass_rate:.1%}")
```

`batch_evaluate` runs traces in parallel via `OfflineEvaluator`. Control the pool with
`max_workers=`.

### Pattern 2: Custom Metric Selection

Metric selection happens per call, not on the evaluator. Pass a flat list of names:

```python
result = evaluate(
    trace=trace,
    ground_truth=ground_truth,
    metrics=[
        "tool_selection_accuracy",
        "parameter_quality",
        "answer_relevance",
        "hallucination_score",
    ],
)
```

Or group by dimension with `metric_set=` when you want dimension-level control:

```python
result = evaluate(
    trace=trace,
    ground_truth=ground_truth,
    metric_set={
        "tool_calling": ["tool_selection_accuracy", "parameter_quality"],
        "response_quality": ["answer_relevance"],
    },
)
```

### Pattern 3: Threshold-Based Pass/Fail

Thresholds are per-dimension and passed to the call. `result.passed` is `False` when any
evaluated dimension falls below its threshold:

```python
result = evaluate(
    trace=trace,
    ground_truth=ground_truth,
    metrics=["tool_selection_accuracy", "answer_relevance"],
    thresholds={
        "Tool Calling": 0.85,
        "Response Quality": 0.80,
    },
)

if not result.passed:
    print("Failures:")
    for failure in result.failures:
        print(f"  - {failure}")
```

!!! note "Use the exact dimension name as the key"
    Threshold and weight keys are matched against `DimensionResult.dimension_name`, which is
    Title Case with spaces. The seven valid keys are `"Tool Calling"`, `"Response Quality"`,
    `"Responsible AI"`, `"Performance"`, `"Multi-Turn"`, `"Multi-Agent"`, and `"Reasoning"`.

`dimension_weights=` takes the same keys.
[Multi-Turn Evaluation Workflow](../Advanced/multi-turn-evaluation-workflow.md#default-dimension-weights) lists the built-in
defaults.

## Troubleshooting

### Issue: AWS Credentials Not Found

**Error**: `NoCredentialsError: Unable to locate credentials`

**Solution**: Configure AWS credentials:
```bash
aws configure
# or
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Issue: A metric scored `None` instead of a number

**Symptom**: `metric.score is None`, and `metric.metadata["warning"]` is `missing_data`.

UAEF does not raise when a metric lacks its inputs — it returns `None` with a warning, and
excludes it from the dimension average. A metric requiring ground truth will do this whenever
`ground_truth` is absent or missing the field it needs.

**Solution**: supply the ground truth, or select metrics that don't need it:

```python
result = evaluate(
    trace=trace,
    metrics=[
        "answer_relevance",  # LLM judge, no ground truth needed
        "latency_score",     # deterministic, no ground truth needed
        "safety_score",      # LLM judge, no ground truth needed
    ],
)
```

The ground-truth requirement for every metric is in the
[Metrics Catalog](../Guides/metrics-catalog.md#all-built-in-metrics), and the full warning
list is under [Edge cases](../Guides/metrics-catalog.md#edge-cases-all-metrics).

### Issue: Slow Evaluation

**Problem**: Evaluation takes too long

**Solution**: Use online mode for fast deterministic metrics:
```python
from uaef.evaluation import OnlineEvaluator

evaluator = OnlineEvaluator(
    fast_metrics=["latency_score", "token_efficiency"],
    async_metrics=["answer_relevance", "safety_score"]
)

# Get immediate results for fast metrics
result = evaluator.evaluate(evaluation_input)
```

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'ragas'`

**Solution**: RAGAS and DeepEval are required core dependencies, not extras — this means the
environment is incomplete rather than missing an add-on. Re-sync from the lockfile:

```bash
uv sync
```

If you are running outside the project environment, make sure you invoke it through `uv run`
(or activate `.venv`) so the synced interpreter is the one executing your code.

## Best Practices

### 1. Start Simple

Begin with a few key metrics:
- Tool selection accuracy
- Answer relevance
- Latency

Add more metrics as needed.

### 2. Use Ground Truth Strategically

Ground truth is valuable but time-consuming to create. Focus on:
- Critical user journeys
- Edge cases
- Regression test cases

### 3. Set Realistic Thresholds

Start with lower thresholds and increase as your agent improves:
- Initial: 0.70
- Production-ready: 0.80
- High-quality: 0.90

### 4. Track Experiments

Always use experiment management to track changes:
```python
manager = ExperimentManager()
experiment = manager.create_experiment(name="My Agent v1")
run = manager.create_run(experiment_id=experiment.experiment_id)
```

### 5. Monitor Trends

Use offline batch evaluation to analyze trends:
```python
from uaef.evaluation import OfflineEvaluator

evaluator = OfflineEvaluator(parallelism=4)
results = evaluator.batch_evaluate(evaluation_inputs)
```

## Next Steps

Now that you understand the basics, explore:

1. **[Adapter Integration Guide](../Guides/adapters.md)**: Integrate your agent framework
2. **[Custom Metrics Guide](../Guides/custom-metrics.md)**: Create custom evaluation metrics
3. **[Experiment Workflow Guide](../Guides/experiments.md)**: Track and compare agent versions
4. **[HITL Workflow Guide](../Guides/hitl.md)**: Incorporate human feedback

## Additional Resources

- **API Documentation**: Complete API reference
- **Examples**: Sample code for common scenarios
- **Repository**: source code and issue tracker
- **Community**: Discussion forum and support

## Getting Help

- **Documentation**: Check the full documentation
- **Examples**: Browse example code in `examples/` directory
- **Issues**: report bugs in the issue tracker
- **Community**: Ask questions in discussions

---

**Ready to evaluate your agents?** Continue to the [Adapter Integration Guide](../Guides/adapters.md) to connect your agent framework.
