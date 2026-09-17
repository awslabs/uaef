# Development Setup

Working on UAEF itself. For using UAEF as a library, see the
root README.

## Setup

```bash
git clone <repository-url>
cd agenticevaluationframework
uv sync                    # install all dependencies from the lockfile
```

Add the developer tooling (pytest, hypothesis, black, mypy, ruff) with the `dev`
extra:

```bash
uv sync --extra dev
```

## Dependency management

This project uses [uv](https://docs.astral.sh/uv/). [`pyproject.toml`](https://github.com/awslabs/uaef/blob/main/pyproject.toml) + [`uv.lock`](https://github.com/awslabs/uaef/blob/main/uv.lock)
are the single source of truth. **Do not use `pip install` to add
dependencies.**

- **Add a dependency** — edit [`pyproject.toml`](https://github.com/awslabs/uaef/blob/main/pyproject.toml), then `uv lock && uv sync`
- **Install from the lockfile** — `uv sync`
- The `requirements.txt` files under [`demo/`](https://github.com/awslabs/uaef/tree/main/demo), [`deploy_eks/`](https://github.com/awslabs/uaef/tree/main/deploy_eks), and [`uaef-service/`](https://github.com/awslabs/uaef/tree/main/uaef-service)
  belong to those standalone apps. They are not used for the core `uaef`
  package.

### Security constraints

[`pyproject.toml`](https://github.com/awslabs/uaef/blob/main/pyproject.toml) has a `# Security constraints` block where dependency floors
are raised to clear specific advisories, each with an inline GHSA comment. When a
vulnerability scan flags a new advisory, the convention is to **raise the floor
there** and re-run `uv lock`, rather than suppressing the finding.

## Notebook output filter (one-time setup)

Notebooks are stripped of outputs on `git add` via a git filter. After cloning,
run this once:

```bash
git config filter.strip-notebook-output.clean "python3 scripts/clean-notebook.py"
git config filter.strip-notebook-output.smudge cat
```

This keeps committed notebooks free of cell outputs and execution counts,
regardless of your local working state.

## Running tests

```bash
uv run pytest                      # core suite
uv run --extra dev pytest          # includes the hypothesis property tests
```

The deployment service has its own suite:

```bash
cd uaef-service && python3 -m pytest tests/ -q
```

## Code style

```bash
uv run black src/ tests/
uv run ruff check src/
```

Line length is 100 (see `[tool.black]` and `[tool.ruff]` in [`pyproject.toml`](https://github.com/awslabs/uaef/blob/main/pyproject.toml)).

## Library layout

```
src/uaef/
├── api/            # High-level Python API (evaluate, batch_evaluate)
├── adapters/       # Framework adapters (LangGraph, Bedrock, LangChain, ...)
├── data/           # Ground truth parsing, loading, and validation
├── metrics/        # 36 built-in metrics across 7 dimensions
├── evaluation/     # Evaluation engines (single-agent, multi-turn, multi-agent, offline)
├── experiments/    # Experiment management and comparison
├── storage/        # DynamoDB + S3 persistence layer
├── models/         # Pydantic data models (AgentTrace, EvaluationResult, ...)
├── analysis/       # Root cause, clustering, trends, recommendations
├── reporting/      # Dashboards and report generation
├── integrations/   # RAGAS, DeepEval connectors
├── llm_judge/      # Bedrock LLM judge with prompt templates
├── hitl/           # Human-in-the-loop workflows
├── security/       # PII detection, encryption
├── simulator/      # Conversation simulation (placeholder)
└── config.py       # Configuration management
```

## Related

- [Contributing](contributing.md) — pull request process, adding metrics and adapters
- [Metrics Catalog](../Guides/metrics-catalog.md) — every metric, with formulas and examples
- [Custom Metrics](../Guides/custom-metrics.md) — metrics you keep in your own codebase
- [Adapters](../Guides/adapters.md) — writing a new adapter
