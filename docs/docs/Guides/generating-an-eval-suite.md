# Generating an Eval Suite

UAEF scores a trace against ground truth. It does not tell you *which* dimensions matter for your
agent, or where the test cases come from — and on a real engagement, building that dataset is the
hard part.

`agent-eval-suite` is an **agent skill** that fills that gap. Point it at an existing agent's
repository and it derives the evaluation suite from that agent's own tools, prompts, data sources
and guardrails, then emits a runnable harness that scores through UAEF.

```
agent's source code  ──►  agent-eval-suite  ──►  cases + harness  ──►  uaef.api.evaluate
   tools, prompts,          derives what              eval/                36 metrics,
   guardrails               to measure                                     persistence, trends
```

!!! note "This is a skill, not a Python module"
    It lives in `skills/agent-eval-suite/`. Copy or symlink that directory into your agentic coding tool's
    skills directory (e.g. `~/.claude/skills/` for Claude Code, `.kiro/skills/` for Kiro). There is nothing
    to `import`. It reads code and writes files; UAEF does the scoring.

## When to use it

- Before a POC handover, or before an agent goes to production
- When an agent's quality is disputed and there is no suite to settle it
- When you have UAEF installed and a ground-truth dataset is the thing you're missing
- To compare several candidate architectures on one fixed suite

## Usage

From a session whose working directory is the agent's repository (or with the repo path to hand):

```
Build an evaluation suite for this agent.
```

The skill triggers on description matching, so an explicit invocation is not required — but
`/agent-eval-suite` works if your tool supports slash invocation.

## The four phases

| Phase | Produces | Gate |
|---|---|---|
| 1 — Analyse + plan | `eval/eval-plan.md` — architecture as found, nondeterminism map, ground-truth source per dimension, evaluator selection, numeric thresholds, risks, open questions | **Approval required before any code is generated** |
| 2 — Generate cases | `eval/test-cases.jsonl`, `eval/red_team/cases.jsonl`, `eval/chaos/scenarios.yaml` | — |
| 3 — Generate harness | `eval/evaluators/`, `eval/adapters/agent_adapter.py`, `eval/run_evals.py`, `eval/report.py` | **Confirmation required before a real run** |
| 4 — Run + report (opt-in) | `eval/eval-report.md`, `eval/results/<ts>/scorecard.json` | — |

The two gates are the point. An eval plan is cheap to change; a generated suite that measures the
wrong thing is worse than no suite, because it produces confident wrong numbers. And Phase 4 costs
money and may touch real systems, so it never runs without being asked for.

## How it uses UAEF

The generated harness treats UAEF as the scoring engine and does not reimplement it:

| Concern | Owned by |
|---|---|
| Metric registry, LLM judge, judge calibration, PII redaction | **UAEF** — `uaef.metrics`, `uaef.llm_judge`, `uaef.hitl.judge_calibration`, `uaef.security.pii` |
| Persistence, experiment tracking, regression detection, dashboards | **UAEF** — `evaluate(persist=True)`, `uaef.experiments`, `uaef.reporting` |
| Which dimensions apply, and why | the skill's Phase 1 |
| The case corpus and its stratification | the skill's Phase 2 |
| Invoking the agent, capturing the trajectory | the skill's `agent_adapter.py` |
| Checks UAEF has no metric for | local evaluators, with the gap named in the plan |

Phase 4 uses `batch_evaluate(..., persist=True, experiment_name=...)` so that each run becomes an
experiment row the next run can be diffed against.

### Four field constraints worth knowing

The bridge from the harness's own run record to `AgentTrace` trips on the same four things every
time. They are documented in full in the skill's `references/framework-adapters.md`:

- `AgentTrace.trace_id` is a `UUID`, not a `str`. A readable id like `"refund-012"` raises a
  `ValidationError`. Derive a stable UUID from the case id (`uuid5`) rather than calling `uuid4()`,
  or the same case gets a fresh id on every run and nothing diffs.
- `Message.timestamp` and `ToolCall.timestamp` are **required** and have no defaults.
- `GroundTruth.expected_output` is a single `Optional[str]`. The case format allows a list meaning
  "any of these is acceptable"; that any-of check has to stay local.
- `GroundTruth.expected_tool_calls` is `List[ToolCall]`, not a list of names.

## What it covers beyond the metric set

- **Evaluation levels** — output, tool, trace and session, matched to the question being asked
- **Deterministic before judged** — anything code can check is checked by code; a judge is used only
  for what code genuinely cannot answer, and the plan says why
- **Ground-truth ladder** — system of record → programmatic verification → golden data → calibrated
  judge → nothing yet, chosen per dimension rather than once for the whole suite
- **Negative cases** — where the correct behaviour is to refuse, escalate, or return nothing.
  Agents are rarely tested on "the right answer is zero results", and it is a common real bug
- **Adversarial and red-team cases**, and **chaos / degraded-mode scenarios**
- **Prompt-only guardrails** — a constraint enforced only in the system prompt with no code backstop
  is named explicitly and routed to a judge plus a trajectory check, because a deterministic tool
  test will pass while the real risk goes unmeasured
- **Diagnosis** — each failure class is routed to a fix type: system prompt, tool description, code,
  or data

## Relationship to `uaef.simulator`

[`uaef.simulator`](../API-Reference/simulator.md) declares `ConversationGenerator`,
`ScenarioOrchestrator` and `TestPackageGenerator`, and every method currently raises
`NotImplementedError`. The skill covers the same ground at the workflow level — deriving scenarios
and extracting ground truth — but as generated files rather than a Python API.

These are complementary, not duplicates: the skill produces a corpus a human reviews and commits,
while `simulator` is intended as a programmatic interface. If `simulator` is implemented, the skill's
case taxonomy (`category`, `difficulty`, `provenance`, `expected_trajectory`, `forbidden_tools`) is
the natural input to it.

## Limitations

- **Phases 3 and 4 have not been exercised end to end against a live agent.** Phases 1 and 2 were
  dogfooded on a real customer fraud-investigation agent; harness generation and a real scored run
  are the parts to treat with most suspicion on first use.
- The case format's `expected_output` list and `forbidden_tools` have no UAEF equivalent, so those
  checks run locally rather than through the metric registry.
- It does not fix the agent, and it does not invent ground truth. Where none exists, the plan says so
  and recommends building it as the first step rather than presenting a judge's opinion as fact.

## Reference

The skill's own documentation is the source of truth for the workflow:

| File | Contents |
|---|---|
| `skills/agent-eval-suite/SKILL.md` | The four-phase workflow and the non-negotiables |
| `references/eval-design-principles.md` | The quality bar and the anti-patterns |
| `references/evaluator-catalogue.md` | The evaluator menu, by level |
| `references/judge-design.md` | Rubrics, CoT-before-verdict, calibration, bias controls |
| `references/red-teaming.md` | Risk taxonomy, attack strategies, breach scoring |
| `references/chaos-and-resilience.md` | Fault injection and the resilience dimensions |
| `references/case-format.md` | The `test-cases.jsonl` schema |
| `references/harness-layout.md` | File-by-file spec of the generated harness |
| `references/framework-adapters.md` | The verified UAEF bridge, plus Strands / AgentCore / LangGraph |
| `references/diagnosis.md` | Failure taxonomy and fix-type routing |
