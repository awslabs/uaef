# Contributing

## Setup

```bash
git clone <repo-url>
cd uaef
uv sync --extra dev
uv run pytest
```

Full environment details, the `uv` dependency rules, and the notebook output filter are in
[Development](development.md).

## Code style

```bash
uv run black src/
uv run ruff check src/
uv run mypy src/
```

Line length is 100. Targets are Python 3.10–3.12.

## Tests

Write tests for all new features — unit tests for components, integration tests for
workflows, property-based tests (Hypothesis) where the input space is wide.

```bash
uv run pytest
uv run pytest --cov=uaef --cov-report=html
```

## Documentation

The site is MkDocs, built from `docs/`. Add or update the page alongside the code change:

```bash
cd docs
make install     # first time only
make docs        # serve at http://127.0.0.1:8000 with live reload
make build       # strict build — fails on broken links
```

Public API changes need docstrings; the [API Reference](../API-Reference/index.md) is
generated from them by mkdocstrings, so no separate file needs editing when you add a
function to an existing module.

## Pull requests

1. Create a feature branch
2. Make the change and add tests
3. Update documentation
4. Open the pull request

Checklist:

- [ ] Tests pass
- [ ] Formatted (`black`)
- [ ] Lint clean (`ruff`)
- [ ] Type checks pass (`mypy`)
- [ ] Docs updated, and `make build` succeeds
- [ ] `CHANGELOG.md` updated

## Adding a metric

Subclass `BaseMetric`, implement the required methods, register it, and add tests:

```python
from uaef.metrics.base import BaseMetric
from uaef.models import MetricScore, EvaluationInput


class MyMetric(BaseMetric):
    def get_name(self) -> str:
        return "my_metric"

    def requires_ground_truth(self) -> bool:
        return False

    def requires_llm_judge(self) -> bool:
        return False

    def calculate(self, eval_input: EvaluationInput) -> MetricScore:
        ...
```

Add it to the [Metrics Catalog](../Guides/metrics-catalog.md) with a formula and worked
good / partial / bad examples, matching the existing entries. For metrics you keep in your
own codebase rather than contributing upstream, see
[Custom Metrics](../Guides/custom-metrics.md).

## Adding a framework adapter

Subclass `BaseAdapter` and implement `transform_to_canonical`:

```python
from uaef.adapters.base import BaseAdapter
from uaef.models import AgentTrace


class MyFrameworkAdapter(BaseAdapter):
    def transform_to_canonical(self, trace: dict) -> AgentTrace:
        ...
```

Add tests with a real sample trace from the framework, and register the adapter. See
[Adapters](../Guides/adapters.md) — if the framework emits JSON you may not need an adapter
at all, since `GenericJSONAdapter` maps arbitrary schemas by configuration.

## License

UAEF is released under the Apache License 2.0. See
`LICENSE` and
`NOTICE`.
