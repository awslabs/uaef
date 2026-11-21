# Harness Layout — Phase 3 generation spec

This is the file-by-file contract for everything Phase 3 emits under `eval/`. Follow it exactly —
the point of the layout is that only one file (`adapters/agent_adapter.py`) knows anything about the
specific agent being evaluated. Everything else must keep working unchanged if that agent is
rewritten in a different framework.

```
eval/
  eval-plan.md              Phase 1 output, approved before this phase runs
  test-cases.jsonl          Phase 2 output, per references/case-format.md
  red_team/cases.jsonl
  chaos/scenarios.yaml
  evaluators/
    deterministic.py
    judges.py
    trajectory.py
  adapters/
    agent_adapter.py
  run_evals.py
  report.py
  results/
    <timestamp>/
      runs.jsonl
      scores.jsonl
      scorecard.json
      eval-report.md
  README.md
```

## `adapters/agent_adapter.py` — the only agent-specific file

Exposes exactly one public entry point:

```python
def invoke(case: Case) -> AgentRun:
    ...
```

`AgentRun` is a plain dataclass/TypedDict with these fields, all required (use `None`/`[]`/`{}` for
anything the agent genuinely can't produce, don't omit the field):

| Field | Type | Content |
|---|---|---|
| `output` | `str` | The agent's final response text. |
| `trajectory` | `list[str]` | Ordered list of tool names actually called. Derivable from `tool_calls`; kept separate because order-matching evaluators only need the names. |
| `tool_calls` | `list[dict]` | Ordered, structured record of each call: `{"name", "arguments", "result", "timestamp", "execution_time", "error"}`. **Required for tool-parameter accuracy / data-precision evaluation** — you cannot check whether arguments were grounded in context if you only captured tool names. Populate from framework hooks, not from the model's own account of what it did. |
| `spans` | `list[dict]` | Raw or lightly normalised trace spans (OTel or framework-native), for TRACE-level evaluators that need more than tool names — e.g. arguments per call, timing. |
| `state` | `dict` | Any observed side effect relevant to `expected_state` — read back from wherever the agent writes, not inferred from its claims. |
| `usage` | `dict` | Token counts and cost inputs, e.g. `{"input_tokens": ..., "output_tokens": ..., "cost_usd": ...}`. |
| `latency` | `float` | Wall-clock seconds for the invocation. |
| `error` | `str \| None` | Set if the invocation raised or the agent returned an error state; evaluators must treat a non-`None` error as an automatic fail rather than trying to score `output`. |

Rules:
- No other file imports the target agent's package, framework SDK, or credentials. If an evaluator
  needs framework-specific data (e.g. Strands span names), it reads that data out of `AgentRun.spans`
  as plain dicts — it does not import Strands.
- `agent_adapter.py` also exposes `ADAPTER_VERSION: str` — a short string (git SHA, semver, or hash of
  the adapter file) used as part of the cache key so a change to the adapter invalidates cached runs.
- Construction/config (endpoints, model ids, session setup) happens inside this file, lazily, not at
  import time — importing `agent_adapter` must never make a network call or require credentials, so
  that `--validate` and `--dry-run` work without them.
- Provide a second, trivial `StubAdapter` (or a `--dry-run` branch inside the same file) that returns
  a canned `AgentRun` without touching the real agent, so the rest of the pipeline is exercisable with
  no credentials.

## `run_evals.py` — CLI contract

One CLI, subcommands or flags — either is fine, but the flags below must all exist and behave as
specified:

| Flag | Behaviour |
|---|---|
| `--validate` | Schema-validates `test-cases.jsonl` (and red-team/chaos files if present) against `case-format.md`. **No model calls, no adapter import side effects.** Exit code non-zero on any malformed line, printing the line number and the field that failed. This is the CI gate — it must run with zero credentials and zero network access. |
| `--dry-run` | Runs the full pipeline — load cases, invoke, evaluate, report — using the stub adapter instead of the real one. Must also work with no credentials, proving the plumbing before spending money. |
| `--subset <ids>` | Comma-separated case ids or a glob against `id`; restricts the run to those cases. |
| `--tags <tags>` | Comma-separated tags; restricts the run to cases whose `tags` intersect. Combine with `--subset` as AND, not OR. |
| `--n <int>` | Repeats per case (default 1). Use ≥3 for any real run against a stochastic agent — record every repeat's score, don't just average silently; `report.py` needs the spread. |
| `--level <levels>` | Comma-separated subset of `output,tool,trace,session` — restricts which evaluator level(s) run. |
| `--no-cache` | Bypasses the result cache and re-invokes the agent even if a cached run exists for `(case.id, adapter_version, run_index)`. |
| `--out <dir>` | Overrides the default `results/<timestamp>/` output location. |

`--validate` and `--dry-run` are the two things a reviewer or CI runs without ever touching
credentials — both must be provably free of side effects; if either imports something that requires
AWS/Bedrock access, that's a bug in the generated harness, not an acceptable limitation.

## `evaluators/*.py` — common interface

All three evaluator modules implement the same shape:

```python
def evaluate(run: AgentRun, case: Case) -> list[EvaluationOutput]:
    ...
```

One case can produce evaluations from more than one module (e.g. a deterministic check on `output`
plus a trajectory check on `trajectory`) — `run_evals.py` calls whichever modules are applicable
based on which fields the case populates, and concatenates the results.

`EvaluationOutput` fields:

| Field | Type | Content |
|---|---|---|
| `score` | `float` (0–1) | Numeric score. Binary checks use 0.0/1.0. |
| `passed` | `bool` | Whether this evaluation met its own pass bar — not the same as `score == 1.0` if the evaluator has a threshold below perfect. |
| `reason` | `str` | Free text — **never emit a score without one**. For a judge this is the rationale; for a deterministic check it's what was compared against what. |
| `label` | `str` | The dimension this evaluation belongs to, e.g. `functional_correctness`, `tool_selection`, `data_precision` — matches the plan's evaluator-selection table so results roll up cleanly in the report. |
| `level` | `str` | One of `output` \| `tool` \| `trace` \| `session`, matching `--level` filtering. |

- **`deterministic.py`** — exact/contains/regex/schema matching against `expected_output`, and
  `state-equals` diffing against `expected_state`. Add domain-specific checks here too (e.g. a numeric
  tolerance check, a currency-format check) rather than reaching for a judge.
- **`judges.py`** — one class per subjective dimension, each wrapping a rubric prompt built from
  `case.expected_assertion`. Follow `references/judge-design.md` for CoT-before-verdict and
  calibration; this file only defines the interface and where to plug UAEF/strands-evals in.
- **`trajectory.py`** — tool-selection correctness against `expected_trajectory` (state explicitly per
  case whether order matters), and any data-precision checks that need `AgentRun.spans` (e.g. did the
  arguments passed to a tool call actually appear in the retrieved context, rather than being
  fabricated).

## Result caching

Cache key: `(case.id, adapter_version, run_index)`. Store the raw `AgentRun` (not the evaluation) so
that evaluators can be edited and rerun without re-invoking the agent — this is what makes iterating
on judge rubrics affordable. `--no-cache` bypasses reads and writes; otherwise a cache hit skips
`adapter.invoke()` entirely. Cache location is an implementation detail of `run_evals.py` (e.g.
`results/.cache/`), but it must survive between invocations of the CLI, not just within one process.

## `results/<timestamp>/`

- `runs.jsonl` — one line per `(case.id, run_index)`, the raw `AgentRun` plus latency/usage. This is
  the cache's persisted form and the input to every evaluator run.
- `scores.jsonl` — one line per `EvaluationOutput`, joined back to its case id and run index.
- `scorecard.json` — machine-readable rollup: per-category, per-label aggregate scores, pass rates,
  cost, p50/p95 latency, and failure counts. Structured so two `scorecard.json` files from different
  timestamps can be diffed programmatically (stable keys, no timestamps embedded inside the per-case
  records) — this is what lets `report.py` show a regression comparison.
- The human report (`eval-report.md`, per `assets/eval-report-template.md`) is generated from
  `scorecard.json`, not recomputed independently from `scores.jsonl` — one source of truth.

## `report.py`

Reads `scorecard.json` (and, if present, the previous run's `scorecard.json` for comparison) and
produces:
- per-category breakdown (not just an aggregate — aggregates hide regressions in one category)
- cost and p50/p95 latency, next to the quality numbers, not in a separate section nobody reads
- a failure taxonomy: cluster failing cases by `label` + a short cause tag, with counts — the
  histogram is what makes the report actionable rather than descriptive

## Dependency rule

**UAEF is a hard dependency.** Import `uaef.api` and `uaef.models` at the top of `judges.py` /
`trajectory.py` without a guard, and route scoring through `evaluate()` / `batch_evaluate()`. If UAEF
cannot be imported, that is a broken environment and the run should fail loudly — `uv sync` is the fix,
not a silent fallback that reports different numbers under the same metric names. Write a local
evaluator only where UAEF has no metric for the check, and name that gap in the eval plan.

**`strands-evals` is genuinely optional.** Import it inside a `try/except ImportError`. If present,
prefer it for OTel-trace-native trajectory scoring on Strands agents; if absent, fall back to the
self-contained trajectory evaluator in the same file. Never let its absence raise past import time or
abort a run — catch it, print one line naming what was skipped and why (e.g. `"strands-evals not
installed — using local trajectory check instead"`), and continue.

## Generated code standards

- Type hints on every public function and dataclass; no bare `dict`/`Any` where a `TypedDict` or
  dataclass is cheap to write.
- No network calls, no credential resolution, no environment-dependent behaviour at **import** time —
  only inside functions that are actually invoked (this is what makes `--validate`/`--dry-run` free).
- Any randomness (case sampling, judge temperature, chaos fault selection) takes an explicit seed,
  and the seed used is recorded in `runs.jsonl`/`scorecard.json` so a run is reproducible.
- Secrets (API keys, endpoint URLs with credentials) come from environment variables only — never
  hardcoded, never written into `results/`.
- Every `EvaluationOutput` carries a non-empty `reason`. A generated evaluator that can only produce a
  bare number is incomplete.
