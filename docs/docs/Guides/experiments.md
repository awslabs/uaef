# Experiment Workflow Guide

## Overview

UAEF's experiment management system helps you track agent versions, compare performance, detect regressions, and make data-driven improvements. This guide covers the complete experiment workflow from creation to analysis.

## Why Use Experiments?

Experiments provide:

- **Version Control**: Track agent configurations and changes
- **Performance Tracking**: Monitor metrics across iterations
- **Regression Detection**: Automatically identify performance degradation
- **Comparison**: Side-by-side analysis of different approaches
- **Reproducibility**: Capture complete configuration for exact reproduction
- **History**: Maintain audit trail of all changes

## Core Concepts

### Experiment

An experiment represents a specific agent project or feature:

```python
from uaef.experiments import ExperimentManager

manager = ExperimentManager()
experiment = manager.create_experiment(
    name="Customer Support Agent v2",
    description="Testing improved tool selection logic",
    metadata={
        "owner": "ml-team",
        "project": "customer-support",
        "tags": ["production", "tool-calling"]
    }
)
```

### Experiment Run

A run is a specific version or iteration of your agent:

```python
from uaef.experiments import AgentDesign

# Define agent configuration
agent_design = AgentDesign(
    framework="langgraph",
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    tools=["search", "calculator", "database_query"],
    system_prompt="You are a helpful customer support agent.",
    parameters={"temperature": 0.7, "max_tokens": 2048}
)

# Create run
run = manager.create_run(
    experiment_id=experiment.experiment_id,
    run_name="baseline-v1",
    agent_design=agent_design,
    config={
        "metric_set": ["tool_accuracy", "answer_relevance"],
        "thresholds": {"tool_calling": 0.8},
        "evaluation_mode": "offline"
    },
    git_commit="a1b2c3d4"  # Optional: track code version
)
```

### Baseline

A baseline is the reference point for comparisons:

```python
# Set a run as baseline
manager.set_baseline(experiment.experiment_id, run.run_id)
```

## Complete Workflow

### Step 1: Create Experiment

```python
from uaef.experiments import ExperimentManager

# Initialize manager
manager = ExperimentManager()

# Create experiment
experiment = manager.create_experiment(
    name="Weather Agent Optimization",
    description="Improving weather query accuracy and response time",
    metadata={
        "owner": "data-science-team",
        "priority": "high",
        "start_date": "2024-01-15"
    }
)

print(f"Created experiment: {experiment.experiment_id}")
```

### Step 2: Create Baseline Run

```python
from uaef.experiments import AgentDesign

# Define baseline agent
baseline_design = AgentDesign(
    framework="langgraph",
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    tools=["get_weather", "get_forecast"],
    system_prompt="You are a weather information assistant.",
    parameters={"temperature": 0.5, "max_tokens": 1024}
)

# Create baseline run
baseline_run = manager.create_run(
    experiment_id=experiment.experiment_id,
    run_name="baseline-v1.0",
    agent_design=baseline_design,
    config={
        "metric_set": [
            "tool_selection_accuracy",
            "answer_relevance",
            "latency_score"
        ],
        "thresholds": {
            "tool_calling": 0.80,
            "response_quality": 0.75,
            "performance": 0.70
        }
    },
    git_commit="abc123"
)

print(f"Created baseline run: {baseline_run.run_id}")
```

### Step 3: Run Evaluations

```python
from uaef.evaluation import SingleAgentEvaluator
from uaef.models import EvaluationInput
from uuid import uuid4

# Create evaluator
evaluator = SingleAgentEvaluator(
    metrics=[
        "tool_selection_accuracy",
        "answer_relevance",
        "latency_score"
    ]
)

# Evaluate test cases
test_cases = load_test_cases()  # Your test data

for test_case in test_cases:
    # Get agent trace
    trace = run_agent(test_case)
    
    # Evaluate
    evaluation_input = EvaluationInput(
        trace=trace,
        ground_truth=test_case.ground_truth
    )
    result = evaluator.evaluate(evaluation_input)
    
    # Record in experiment
    manager.record_evaluation(baseline_run.run_id, result)

print(f"Recorded {baseline_run.evaluation_count} evaluations")
print(f"Aggregate metrics: {baseline_run.aggregate_metrics}")
```

### Step 4: Set Baseline

```python
# Set this run as the baseline for comparison
manager.set_baseline(experiment.experiment_id, baseline_run.run_id)
print("Baseline set successfully")
```

### Step 5: Create Improved Version

```python
# Make improvements to agent
improved_design = AgentDesign(
    framework="langgraph",
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    tools=["get_weather", "get_forecast", "get_historical_weather"],  # Added tool
    system_prompt="You are a weather information assistant with access to historical data.",
    parameters={"temperature": 0.3, "max_tokens": 1024}  # Lower temperature
)

# Create new run
improved_run = manager.create_run(
    experiment_id=experiment.experiment_id,
    run_name="improved-v1.1",
    agent_design=improved_design,
    config={
        "metric_set": [
            "tool_selection_accuracy",
            "answer_relevance",
            "latency_score"
        ],
        "thresholds": {
            "tool_calling": 0.80,
            "response_quality": 0.75,
            "performance": 0.70
        }
    },
    git_commit="def456"
)

# Run same evaluations
for test_case in test_cases:
    trace = run_agent(test_case)
    evaluation_input = EvaluationInput(
        trace=trace,
        ground_truth=test_case.ground_truth
    )
    result = evaluator.evaluate(evaluation_input)
    manager.record_evaluation(improved_run.run_id, result)
```

### Step 6: Compare Runs

```python
from uaef.experiments import ComparisonEngine

# Create comparison engine
comparison_engine = ComparisonEngine(
    regression_threshold=5.0,  # 5% decrease is a regression
    improvement_threshold=5.0   # 5% increase is an improvement
)

# Compare runs
comparison_report = comparison_engine.compare_runs(
    baseline_run,
    improved_run
)

# Print summary
print("\n=== Comparison Summary ===")
print(f"Total metrics: {comparison_report.summary['total_metrics']}")
print(f"Regressions: {len(comparison_report.regressions)}")
print(f"Improvements: {len(comparison_report.improvements)}")
print(f"Unchanged: {len(comparison_report.unchanged)}")

# Show details
if comparison_report.regressions:
    print("\nRegressions detected:")
    for metric_name in comparison_report.regressions:
        comparison = next(
            c for c in comparison_report.metric_comparisons
            if c.metric_name == metric_name
        )
        print(f"  - {metric_name}: {comparison.baseline_score:.3f} → "
              f"{comparison.current_score:.3f} ({comparison.percent_change:+.1f}%)")

if comparison_report.improvements:
    print("\nImprovements detected:")
    for metric_name in comparison_report.improvements:
        comparison = next(
            c for c in comparison_report.metric_comparisons
            if c.metric_name == metric_name
        )
        print(f"  - {metric_name}: {comparison.baseline_score:.3f} → "
              f"{comparison.current_score:.3f} ({comparison.percent_change:+.1f}%)")
```

### Step 7: Detect Regressions with Severity

```python
from uaef.experiments import RegressionDetector

# Create regression detector
regression_detector = RegressionDetector(
    minor_threshold=5.0,      # 5-10% decrease
    moderate_threshold=10.0,  # 10-20% decrease
    severe_threshold=20.0     # >20% decrease
)

# Detect regressions
regression_report = regression_detector.detect_regressions(
    baseline_run,
    improved_run
)

# Print regression details
print("\n=== Regression Analysis ===")
print(f"Total regressions: {regression_report.summary['total_regressions']}")
print(f"Severe: {regression_report.summary['severe_count']}")
print(f"Moderate: {regression_report.summary['moderate_count']}")
print(f"Minor: {regression_report.summary['minor_count']}")

if regression_report.regressions:
    print("\nRegressions by severity:")
    for regression in regression_report.regressions:
        print(f"  [{regression.severity.value.upper()}] {regression.metric_name}")
        print(f"    {regression.baseline_score:.3f} → {regression.current_score:.3f} "
              f"({regression.percent_change:+.1f}%)")
        print(f"    Impact: {regression.impact_description}")
```

## Advanced Patterns

### Pattern 1: A/B Testing

```python
# Create two variants
variant_a = manager.create_run(
    experiment_id=experiment.experiment_id,
    run_name="variant-a-high-temp",
    agent_design=AgentDesign(
        framework="langgraph",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        tools=["search"],
        system_prompt="You are helpful.",
        parameters={"temperature": 0.9}  # High temperature
    )
)

variant_b = manager.create_run(
    experiment_id=experiment.experiment_id,
    run_name="variant-b-low-temp",
    agent_design=AgentDesign(
        framework="langgraph",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        tools=["search"],
        system_prompt="You are helpful.",
        parameters={"temperature": 0.1}  # Low temperature
    )
)

# Evaluate both variants
for test_case in test_cases:
    # Variant A
    trace_a = run_agent_with_config(test_case, variant_a.agent_design)
    result_a = evaluator.evaluate(EvaluationInput(trace=trace_a))
    manager.record_evaluation(variant_a.run_id, result_a)
    
    # Variant B
    trace_b = run_agent_with_config(test_case, variant_b.agent_design)
    result_b = evaluator.evaluate(EvaluationInput(trace=trace_b))
    manager.record_evaluation(variant_b.run_id, result_b)

# Compare variants
comparison = comparison_engine.compare_runs(variant_a, variant_b)
print(f"Winner: {'Variant A' if variant_a.aggregate_metrics['overall_score'] > variant_b.aggregate_metrics['overall_score'] else 'Variant B'}")
```

### Pattern 2: Multi-Metric Optimization

```python
# Track multiple objectives
runs = []

# Try different configurations
configs = [
    {"temperature": 0.3, "max_tokens": 1024},
    {"temperature": 0.5, "max_tokens": 1024},
    {"temperature": 0.7, "max_tokens": 1024},
    {"temperature": 0.5, "max_tokens": 2048},
]

for i, config in enumerate(configs):
    run = manager.create_run(
        experiment_id=experiment.experiment_id,
        run_name=f"config-{i}",
        agent_design=AgentDesign(
            framework="langgraph",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            tools=["search"],
            system_prompt="You are helpful.",
            parameters=config
        )
    )
    
    # Evaluate
    for test_case in test_cases:
        trace = run_agent_with_config(test_case, run.agent_design)
        result = evaluator.evaluate(EvaluationInput(trace=trace))
        manager.record_evaluation(run.run_id, result)
    
    runs.append(run)

# Find best configuration
best_run = max(runs, key=lambda r: r.aggregate_metrics['overall_score'])
print(f"Best configuration: {best_run.run_name}")
print(f"Score: {best_run.aggregate_metrics['overall_score']:.3f}")
print(f"Parameters: {best_run.agent_design.parameters}")
```

### Pattern 3: Iterative Improvement

```python
# Iterative improvement loop
current_run = baseline_run
iteration = 1

while iteration <= 5:
    print(f"\n=== Iteration {iteration} ===")
    
    # Analyze current performance
    from uaef.analysis import RootCauseAnalyzer
    analyzer = RootCauseAnalyzer()
    
    evaluations = manager.get_evaluations(current_run.run_id)
    analysis = analyzer.analyze(evaluations)
    
    # Get recommendations
    from uaef.analysis import RecommendationEngine
    recommender = RecommendationEngine()
    recommendations = recommender.generate_recommendations(analysis)
    
    print(f"Top recommendation: {recommendations[0].title}")
    print(f"Expected impact: {recommendations[0].expected_impact}")
    
    # Apply recommendation (manual or automated)
    improved_design = apply_recommendation(
        current_run.agent_design,
        recommendations[0]
    )
    
    # Create new run
    new_run = manager.create_run(
        experiment_id=experiment.experiment_id,
        run_name=f"iteration-{iteration}",
        agent_design=improved_design
    )
    
    # Evaluate
    for test_case in test_cases:
        trace = run_agent_with_config(test_case, new_run.agent_design)
        result = evaluator.evaluate(EvaluationInput(trace=trace))
        manager.record_evaluation(new_run.run_id, result)
    
    # Check if improved
    comparison = comparison_engine.compare_runs(current_run, new_run)
    
    if len(comparison.improvements) > len(comparison.regressions):
        print("✓ Improvement detected, continuing...")
        current_run = new_run
    else:
        print("✗ No improvement, stopping...")
        break
    
    iteration += 1
```

### Pattern 4: Regression Testing

```python
# Create regression test suite
regression_suite = {
    "critical_cases": load_critical_test_cases(),
    "edge_cases": load_edge_cases(),
    "common_scenarios": load_common_scenarios()
}

def run_regression_tests(run_id):
    """Run complete regression test suite."""
    results = {}
    
    for suite_name, test_cases in regression_suite.items():
        suite_results = []
        
        for test_case in test_cases:
            trace = run_agent(test_case)
            result = evaluator.evaluate(EvaluationInput(
                trace=trace,
                ground_truth=test_case.ground_truth
            ))
            manager.record_evaluation(run_id, result)
            suite_results.append(result)
        
        # Calculate suite metrics
        pass_rate = sum(1 for r in suite_results if r.passed) / len(suite_results)
        avg_score = sum(r.overall_score for r in suite_results) / len(suite_results)
        
        results[suite_name] = {
            "pass_rate": pass_rate,
            "avg_score": avg_score,
            "total_cases": len(suite_results)
        }
    
    return results

# Run regression tests on new version
regression_results = run_regression_tests(improved_run.run_id)

print("\n=== Regression Test Results ===")
for suite_name, metrics in regression_results.items():
    print(f"\n{suite_name}:")
    print(f"  Pass rate: {metrics['pass_rate']:.1%}")
    print(f"  Avg score: {metrics['avg_score']:.3f}")
    print(f"  Total cases: {metrics['total_cases']}")
```

## Database Storage

### SQLite (Default)

```python
from uaef.storage import DatabaseStorage

# Initialize with SQLite
db = DatabaseStorage(db_type="sqlite", db_path="uaef.db")
db.create_tables()

# Store experiment
db.create_experiment(experiment)

# Store runs
db.create_run(baseline_run)
db.create_run(improved_run)

# Store evaluations
evaluations = manager.get_evaluations(baseline_run.run_id)
for evaluation in evaluations:
    db.create_evaluation(evaluation, run_id=baseline_run.run_id)
```

### PostgreSQL (Production)

```python
# Initialize with PostgreSQL
db = DatabaseStorage(
    db_type="postgresql",
    host="localhost",
    port=5432,
    database="uaef",
    user="uaef_user",
    password="your_password"  # pragma: allowlist secret
)
db.create_tables()

# Same API as SQLite
db.create_experiment(experiment)
db.create_run(baseline_run)
```

### Querying Data

```python
# Get experiment by ID
experiment = db.get_experiment(experiment_id)

# Get all runs for experiment
runs = db.get_runs_by_experiment(experiment_id)

# Get evaluations for run
evaluations = db.get_evaluations_by_run(run_id)

# Query by date range
from datetime import datetime, timedelta
start_date = datetime.now() - timedelta(days=7)
recent_runs = db.get_runs_by_date_range(start_date, datetime.now())

# Query by metric threshold
high_performing_runs = db.get_runs_by_metric_threshold(
    metric_name="overall_score",
    threshold=0.85
)
```

## Visualization and Reporting

### Generate Comparison Dashboard

```python
from uaef.reporting import ComparisonDashboard

dashboard = ComparisonDashboard()

# Generate dashboard
html_report = dashboard.generate(
    baseline_run=baseline_run,
    current_run=improved_run,
    comparison_report=comparison_report
)

# Save to file
with open("comparison_dashboard.html", "w") as f:
    f.write(html_report)

print("Dashboard saved to comparison_dashboard.html")
```

### Generate Regression Report

```python
from uaef.reporting import RegressionReport

report = RegressionReport()

# Generate report
pdf_report = report.generate(
    regression_report=regression_report,
    baseline_run=baseline_run,
    current_run=improved_run
)

# Save to file
with open("regression_report.pdf", "wb") as f:
    f.write(pdf_report)

print("Report saved to regression_report.pdf")
```

### Track Trends Over Time

```python
from uaef.analysis import TrendAnalyzer

analyzer = TrendAnalyzer()

# Get all runs for experiment
runs = db.get_runs_by_experiment(experiment_id)

# Analyze trends
trends = analyzer.analyze_trends(
    runs=runs,
    metrics=["tool_accuracy", "answer_relevance", "latency_score"]
)

# Print trends
for metric_name, trend_data in trends.items():
    print(f"\n{metric_name}:")
    print(f"  Trend: {trend_data['direction']}")  # improving, declining, stable
    print(f"  Change: {trend_data['total_change']:+.1%}")
    print(f"  Volatility: {trend_data['volatility']:.3f}")
```

## Best Practices

### 1. Consistent Test Sets

```python
# Use same test cases for all runs
test_cases = load_test_cases("test_suite_v1.json")

# Store test set version in metadata
run = manager.create_run(
    experiment_id=experiment.experiment_id,
    run_name="run-1",
    agent_design=design,
    config={
        "test_set_version": "v1.0",
        "test_set_size": len(test_cases)
    }
)
```

### 2. Meaningful Run Names

```python
# Good: Descriptive names
run_name = "baseline-claude3-temp0.7-v1.0"
run_name = "improved-added-database-tool-v1.1"
run_name = "experiment-lower-temperature-v1.2"

# Bad: Generic names
run_name = "run1"
run_name = "test"
run_name = "new"
```

### 3. Track Code Versions

```python
import subprocess

# Get current git commit
git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]
).decode().strip()

run = manager.create_run(
    experiment_id=experiment.experiment_id,
    run_name="run-1",
    agent_design=design,
    git_commit=git_commit
)
```

### 4. Document Changes

```python
run = manager.create_run(
    experiment_id=experiment.experiment_id,
    run_name="improved-v1.1",
    agent_design=design,
    config={
        "changes": [
            "Added database_query tool",
            "Reduced temperature from 0.7 to 0.5",
            "Updated system prompt for clarity"
        ],
        "hypothesis": "Lower temperature will improve accuracy",
        "expected_impact": "5-10% improvement in tool_accuracy"
    }
)
```

### 5. Set Appropriate Thresholds

```python
# Start conservative
initial_thresholds = {
    "tool_calling": 0.70,
    "response_quality": 0.65,
    "performance": 0.60
}

# Increase as agent improves
production_thresholds = {
    "tool_calling": 0.85,
    "response_quality": 0.80,
    "performance": 0.75
}
```

## Troubleshooting

### Issue: Runs Not Comparable

**Problem**: Comparison fails because runs use different test sets

**Solution**: Ensure consistent test sets:
```python
# Store test set hash
import hashlib
test_set_hash = hashlib.md5(
    str(test_cases).encode()
).hexdigest()

run.config["test_set_hash"] = test_set_hash

# Verify before comparison
if baseline_run.config["test_set_hash"] != improved_run.config["test_set_hash"]:
    print("Warning: Runs use different test sets")
```

### Issue: Missing Aggregate Metrics

**Problem**: `aggregate_metrics` is empty

**Solution**: Ensure evaluations are recorded:
```python
# Check evaluation count
print(f"Evaluations: {run.evaluation_count}")

if run.evaluation_count == 0:
    print("No evaluations recorded for this run")
    # Record evaluations
    for test_case in test_cases:
        result = evaluator.evaluate(...)
        manager.record_evaluation(run.run_id, result)
```

### Issue: Regression Detection Too Sensitive

**Problem**: Too many false positive regressions

**Solution**: Adjust thresholds:
```python
# Increase threshold for less sensitivity
regression_detector = RegressionDetector(
    minor_threshold=10.0,     # Was 5.0
    moderate_threshold=20.0,  # Was 10.0
    severe_threshold=30.0     # Was 20.0
)
```

## Next Steps

Now that you understand experiment workflows:

1. **[HITL Workflow Guide](hitl.md)**: Add human validation to experiments
2. **[Getting Started Guide](../Getting-Started/quickstart.md)**: Review basics
3. **[Custom Metrics Guide](custom-metrics.md)**: Create experiment-specific metrics

## Additional Resources

- **API Reference**: Complete experiment API documentation
- **Examples**: Sample workflows in `examples/phase6_experiment_management.py`
- **Database Schema**: Storage layer documentation

---

**Ready to add human feedback?** Continue to the [HITL Workflow Guide](hitl.md).
