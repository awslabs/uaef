# Installation

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- AWS credentials — for LLM-based metrics via Bedrock, and for persistence via DynamoDB/S3

## From the repository (recommended)

[`pyproject.toml`](https://github.com/awslabs/uaef/blob/main/pyproject.toml) + [`uv.lock`](https://github.com/awslabs/uaef/blob/main/uv.lock) are the single source of truth for dependencies.

```bash
git clone <repo-url>
cd uaef
uv sync            # install everything from the lockfile
```

`uv sync` installs **everything needed to run every metric and use every capability** — all
metric backends (RAGAS, DeepEval), every adapter, analysis and reporting, the Bedrock LLM
judge, and the `UAEFClient` remote SDK. Nothing else to install.

The only optional add-on is a Jupyter kernel for notebook use:

```bash
uv sync --extra notebooks   # or --extra all, which is core + the notebook kernel
```

!!! note "The capability extras are no-ops"
    `integrations`, `langgraph`, `analysis`, `judge`, `client`, and `server` are retained as
    aliases so older install commands keep resolving. Every capability already ships in the
    required core, so these extras add nothing.

## From a distributed package

If you received UAEF as a `.whl` or `.tar.gz`:

```bash
uv pip install ./packages/uaef-0.2.0-py3-none-any.whl
```

## For development

```bash
uv sync --extra dev        # adds pytest, black, ruff, mypy, hypothesis
uv run pytest
```

See [Development](../Contributing/development.md) for the dependency rules, the notebook
output filter, and code style.

## Next

[Run your first evaluation →](quickstart.md)
