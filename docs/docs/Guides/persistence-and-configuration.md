# Persistence and Configuration

How UAEF stores evaluation results in AWS, and how to configure it.

## Persistence Model

When you call `evaluate(persist=True, ...)`, UAEF writes to two AWS services:

| Layer | What | Schema |
|-------|------|--------|
| DynamoDB | One row per experiment | `experiment_id` (PK), `experiment_name`, `experiment_objective`, `evaluation_count`, `average_scores`, `overall_average_score`, `result_path`, `created_at`, `updated_at` |
| S3 | Full results JSON per experiment | All `EvaluationResult` objects with dimension scores, metric scores, metadata |

Each experiment is a self-contained evaluation. To run a new evaluation with
different settings, create a new experiment rather than reusing an old one.
Average scores in DynamoDB are recomputed on every write, so the row always
reflects the current state.

```python
result = evaluate(
    trace=trace,
    ground_truth=ground_truth,
    persist=True,
    experiment_name="Customer Support Agent v2",
    experiment_objective="Measure quality after prompt changes",
)

print(f"Experiment ID: {result.experiment_id}")
# Pass this ID to later calls to append results to the same experiment
```

### Experiment resolution

When you call `evaluate(persist=True, ...)`, UAEF resolves which experiment to
write to in this order:

1. `experiment_id` provided → use it directly (it must already exist)
2. `experiment_name` provided → find by name, or create a new experiment
3. Neither → use or create an experiment named `"default"`

## Configuration

### Via environment variables

```bash
# AWS credentials use standard boto3 resolution
export AWS_REGION=us-east-1

# Storage
export UAEF_DYNAMODB_TABLE=uaef-experiments    # default
export UAEF_S3_BUCKET=uaef-results             # default
export UAEF_S3_PREFIX=evaluations/             # default

# LLM Judge (for subjective metrics)
export UAEF_LLM_MODEL_ID=us.anthropic.claude-sonnet-4-6
```

### Via a YAML config file

```yaml
aws:
  region: us-east-1

storage:
  dynamodb_table_name: uaef-experiments
  s3_bucket: uaef-results
  s3_prefix: evaluations/

llm_judge:
  model_id: us.anthropic.claude-sonnet-4-6
  temperature: 0.0
  max_tokens: 2048
```

Load it explicitly:

```python
from uaef.config import UAEFConfig

config = UAEFConfig.from_file("uaef_config.yaml")
```

## AWS prerequisites

- **Credentials** resolved by boto3 (env vars, shared config, instance role, …).
- **Bedrock model access** enabled for the judge model in your region. The
  default `us.anthropic.claude-sonnet-4-6` is a `us.` cross-region inference
  profile, so it works in `us-east-1`, `us-east-2`, and `us-west-2`.
- **DynamoDB table and S3 bucket** — created for you if you deploy the service
  (see `uaef-service/README.md`); create
  them yourself if you are using the library standalone with `persist=True`.

## Related

- [Experiment workflow](experiments.md) — comparing runs, detecting regressions
- Deploying the service — provisions these resources for you
