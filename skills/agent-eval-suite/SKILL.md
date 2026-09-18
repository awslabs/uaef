---
name: agent-eval-suite
description: Generate a complete, agent-specific evaluation suite for an existing AI agent — eval plan, test cases, deterministic checks, LLM judges, trajectory and tool-level evals, adversarial and red-team cases, chaos/degraded-mode tests, and a runnable harness with a report. Use this whenever someone asks to evaluate an agent, build evals, create an eval suite or test harness for an agent, measure agent quality or accuracy, red-team or stress-test an agent, set up regression tests for an agentic system, or asks "how do I know this agent is working / has it got better?". Also use before a POC handover, before putting an agent into production, or when an agent's quality is disputed and there is no suite to settle it. Reads the agent's own code to derive what to measure rather than applying a generic checklist.
dependencies:
  - Read access to the target agent's repository (the skill derives the suite from the agent's tools, prompts, and data sources)
  - Python 3.10+ and UAEF installed (`uv sync`) — the generated harness scores through `uaef.api.evaluate` and persists through UAEF
  - Optional — `strands-evals` when the target agent is a Strands agent and OTel traces are available
  - Optional — AWS credentials with Bedrock access, only for the opt-in real run (Phase 4)
---

## What this skill is for

Agent evaluation gets rebuilt from scratch on every engagement, at a different standard each time. This
skill standardises it: point it at an agent, and it produces the whole suite — **specific to that
agent**, derived from its actual tools, prompts and failure surface, not a generic template.

**The bar:** an engineer who has never evaluated an agent should end up with the same suite a
specialist would build, and be able to run it with one command.

## Non-negotiables

1. **Derive, never assume.** Read the agent's code — tools, system prompt, data sources, guardrails —
   before proposing a single metric. A suite that doesn't match the agent's real failure surface is
   worse than no suite, because it produces confident wrong numbers.
2. **Human gate after Phase 1.** Present the eval plan and get approval before generating code. The
   plan is cheap to change; a generated suite measuring the wrong thing is not.
3. **Deterministic before judged.** Every dimension that a machine can check must be checked by code.
   Reach for an LLM judge only for what code genuinely cannot answer, and say why in the plan.
4. **Never run against a live agent without explicit confirmation.** Generation and schema validation
   are free. Execution costs money, may hit production systems, and needs credentials. Ask.
5. **State what you could not derive.** If you can't find ground truth, or the agent has no test data,
   say so in the plan as an open question. Do not invent expected outputs and present them as truth.
6. **No PII or customer secrets in generated artefacts.** Test cases drawn from real data must be
   scrubbed. Flag anything you're unsure about rather than committing it.

## Workflow

Four phases. **Stop at the end of Phase 1 and Phase 3.**

```
Phase 1  ANALYSE + PLAN     read the agent → nondeterminism map → eval-plan.md    ← APPROVAL GATE
Phase 2  GENERATE CASES     test-cases.jsonl + red-team + chaos cases
Phase 3  GENERATE HARNESS   evaluators, judges, runner, dry-run validation        ← CONFIRM BEFORE RUN
Phase 4  RUN + REPORT       execute (opt-in) → results → eval-report.md
```

---

### Phase 1 — Analyse the agent and write the plan

**Step 1.1 — Locate and read the agent.** Ask for the repo path if not given. Then read, in this order:
- the agent definition (framework, model, system prompt, tool registry, executor)
- **every tool**: name, parameters, what it reads or mutates, and how it fails
- data sources and any system of record (a database, an API, a spreadsheet the business trusts)
- existing tests, eval scripts, or golden data — **reuse these, don't replace them** (procedure below)
- guardrails, permissions, and anything the agent must never do

Record what you found in the plan. **Dispatch subagents to read tool groups in parallel when the agent
has more than ~15 tools or any single file over ~1,500 lines** — below that, read it directly. Never
pull whole tool files into the orchestrator's context; ask for summaries plus the specific failure
surface per tool.

**Step 1.1a — If the agent already has eval assets, decide their fate explicitly.** "Reuse" is a
procedure, not a sentiment. Classify what you found and record the decision in the plan:

| What exists | Do this |
|---|---|
| A case set covering dimensions you'd have derived anyway | **Import by reference.** Keep it as the bulk corpus the harness runs. Re-express a representative subset (20–50) in this skill's case format for the plan's stratification table, and cite the original file as `provenance`. Do not retype hundreds of cases. |
| A judge or rubric | **Reuse it**, and check whether anyone calibrated it against human labels. If not, that's a finding for the plan, not a blocker. |
| A runner or harness | **Wrap, don't rewrite.** Your `agent_adapter.py` can call their runner. Only replace it if it can't express the levels the plan needs. |
| Ad-hoc scripts with no ground truth | Treat as inspiration for cases; build the suite properly. |

If the existing corpus is **larger** than this skill's 20–50 target, don't down-sample the corpus —
down-sample only for the plan's worked example, and say in the plan that the full existing set is what
Phase 3 runs in bulk. More real cases is never the problem.

**Step 1.1b — If the repo contains more than one candidate architecture** (e.g. a single-agent, an
orchestrator, and a dynamic-sub-agent variant all under comparison), write **one plan covering all
variants** under a shared case set — comparing architectures on a fixed suite is exactly what the
suite is for. Mark which eval dimensions apply to which variants; handoff accuracy, for instance,
exists only for the variants that actually hand off.

**Step 1.2 — Build the nondeterminism map.** This decides *what* to evaluate. Walk the ladder and
record which rungs this agent is on:

| Rung | Source of nondeterminism | Eval dimensions it forces |
|---|---|---|
| Single-turn | inputs (developer + user) | **instruction following** (incl. does the system prompt win over a conflicting user prompt) |
| | outputs from the model | **functional correctness** |
| Workflow | no new sources — same two, × N steps | the same two, measured per step in isolation |
| Single agent | the model chooses tools | **tool selection** (right tool, right moment) · **data precision** (arguments grounded in context, not fabricated) |
| Multi-agent | handoffs between agents | **agent handoff accuracy** — including the return path |

**Step 1.3 — Find the ground truth, per dimension.** Most agents need **more than one source at once** —
a pure-function tool can be pinned programmatically while narrative quality needs a judge. So produce a
**table of dimension → source → why**, not a single answer. Work the list in priority order and take the
highest source available *for each dimension*:
1. **A system of record** the business already trusts (the strongest — nobody argues with the metric)
2. **Programmatic verification** — tests, compiler, type checker, schema, executable checks
3. **Golden data** — expected outputs from a domain expert, or manufactured by reverting real fixes
4. **A calibrated LLM judge** — only for what none of the above can answer
5. **Nothing yet** — say so explicitly and make building it the first recommendation

**Watch for the prompt-only guardrail.** Where a constraint is enforced *only* in the system prompt with
no code-level backstop, the tool surface and the guardrail surface have come apart: you can prove
programmatically that the tool cannot be abused, and still not know whether the model will *claim* it did
something it didn't. Name every such guardrail in the plan and route it to a judge plus a
trajectory check — a deterministic tool test will pass while the real risk goes unmeasured. Conversely,
if a constraint *can* be moved into code, recommend that as a finding: a guardrail in code needs no eval.

**Step 1.4 — Pick evaluation levels.** Match the level to the question, per
`references/evaluator-catalogue.md`. Getting this wrong is the most common mistake:

| Level | Unit | Answers |
|---|---|---|
| OUTPUT | one response | is the final answer right? |
| TOOL | each individual tool call | right tool? arguments grounded? |
| TRACE | the whole path | sensible route? answer faithful to what it read? |
| SESSION | a multi-turn conversation | did the user's goal actually get achieved? |

**Step 1.5 — Write `eval/eval-plan.md`** using `assets/eval-plan-template.md`. It must contain: the
agent's architecture as you found it; the nondeterminism map; the chosen ground truth; the evaluator
selection with a one-line justification each; **numeric success thresholds on a sized set**; the risk
and failure-mode analysis; cost and latency budgets; and open questions.

Apply the quality bar from `references/eval-design-principles.md` — cases must be thematically
consistent, **challenging but achievable by a domain expert**, directionally clear, and carefully
crafted. If the agent already passes everything you propose, the suite is too easy to be useful.

**→ STOP. Present the plan. Get approval before Phase 2.**

---

### Phase 2 — Generate the cases

Write `eval/test-cases.jsonl`, one JSON object per line, per `references/case-format.md`. **Stratify
deliberately** and record the counts in the plan:

- **normal path** — the agent's bread and butter
- **edge cases** — from `references/edge-cases.md`: minimal-context requests, typos, multiple intents
  in one message, very long inputs, ambiguous tool results, non-English or non-prose input formats
- **failure scenarios** — missing data, unavailable tools, contradictory sources
- **adversarial** — from `references/red-teaming.md`
- **negative cases** — where the correct behaviour is to refuse, escalate, or return nothing. Agents
  are rarely tested on "the right answer is zero results", and it's a common real bug.

Aim for **20–50 real cases over 500 synthetic ones** — a floor for a suite built from nothing, not a
cap. If existing assets already give you more real cases than that (see Step 1.1a), keep them all as
the corpus the harness runs and express only a representative subset here. Harvest from the customer's tickets, logs and
transcripts wherever possible; note the provenance of each case. Where you must synthesise, generate
across scenarios and difficulty bands and label them as synthetic.

Write red-team cases to `eval/red_team/cases.jsonl` and chaos scenarios to `eval/chaos/scenarios.yaml`.

---

### Phase 3 — Generate the harness

Emit the structure in `references/harness-layout.md`:

```
eval/
  eval-plan.md              Phase 1, approved
  test-cases.jsonl          Phase 2
  red_team/cases.jsonl
  chaos/scenarios.yaml
  evaluators/
    deterministic.py        exact/contains/schema/tool-called/state-equals + domain checks
    judges.py               rubric-based judges, one class per subjective dimension
    trajectory.py           tool-selection + data-precision + path scoring
  adapters/agent_adapter.py the ONLY file that knows how to invoke this agent
  run_evals.py              CLI: --dry-run, --validate, --subset, --n, --level
  report.py                 scorecard → eval-report.md
  results/                  timestamped run folders
  README.md                 how to run it, what each number means
```

Rules for the generated code:
- **`agent_adapter.py` is the only agent-specific integration point.** Everything else must work
  unchanged if the agent is rewritten.
- **`--validate` and `--dry-run` must work with no credentials and no model calls.** Schema validation
  exits non-zero on a malformed case — that's the CI gate.
- **Cache task outputs per case** so evaluators can be iterated without re-invoking the agent.
- **Emit a machine-readable scorecard** (`results/<ts>/scorecard.json`) plus the markdown report, so
  two runs can be diffed.
- **Every score carries a `reason`.** A number without a reason can't be acted on.
- **Report cost and p50/p95 latency next to quality**, always. Quality bought with unbudgeted money
  isn't a win.
- **Score through UAEF, don't reimplement it.** Convert each `AgentRun` to a `uaef.models.AgentTrace`
  and call `uaef.api.evaluate` / `batch_evaluate` — UAEF already owns the metric registry, the Bedrock
  judge, judge calibration against human labels (`uaef.hitl.judge_calibration`), PII redaction
  (`uaef.security.pii`), persistence and trend reporting. Write a local evaluator only for a check
  UAEF has no metric for, and say so in the plan. `references/framework-adapters.md` has the verified
  bridge. Degrade gracefully for genuinely optional extras such as `strands-evals` — print one line
  naming what was skipped and continue — but do not degrade away from UAEF itself.

Then **run `--validate` and `--dry-run` yourself** and fix anything that fails.

**→ STOP. Report what was generated and what a real run will cost and touch. Ask before Phase 4.**

---

### Phase 4 — Run and report (opt-in)

Run with `n≥3` per case where budget allows, because the system is stochastic. Then produce
`eval/eval-report.md` per `assets/eval-report-template.md`:
- headline metrics **per category**, not just aggregate — aggregates hide regressions
- cost and latency
- **the failure taxonomy with counts** — the histogram is the roadmap
- root cause per failure class, routed to a **fix type**: system prompt / tool description / code / data
- the top three recommended fixes, in ROI order
- what the suite does *not* cover

Close by telling the user how to wire `--validate` and the suite into CI, and that **every production
failure should become a permanent case**.

---

## Reference files

Load these on demand — do not read them all up front.

| File | When |
|---|---|
| `references/eval-design-principles.md` | Phase 1 — the quality bar, the five-step workflow, anti-patterns |
| `references/evaluator-catalogue.md` | Phase 1 — the full evaluator menu with levels and when each applies |
| `references/judge-design.md` | Phase 1/3 — rubrics, CoT-before-verdict, calibration, bias controls |
| `references/red-teaming.md` | Phase 2 — risk taxonomy, attack strategies, breach scoring |
| `references/edge-cases.md` | Phase 2 — the edge-case checklist by bucket |
| `references/chaos-and-resilience.md` | Phase 2/3 — fault injection and the three resilience dimensions |
| `references/case-format.md` | Phase 2 — the JSONL schema |
| `references/harness-layout.md` | Phase 3 — file-by-file spec of the generated harness |
| `references/diagnosis.md` | Phase 4 — failure taxonomy, causal chains, fix-type routing |
| `references/framework-adapters.md` | Phase 3 — Strands / AgentCore / LangGraph / plain-Python integration |

## Anti-patterns

| Don't | Do |
|---|---|
| Apply a generic checklist | Derive the suite from the agent's actual tools and failure surface |
| Judge everything with an LLM | Deterministic first; judge only what code can't check |
| Score only the final answer | Add tool-level and trace-level, or you can measure but not debug |
| Report one aggregate number | Break down by category, with cost and p95 beside it |
| Invent expected outputs | Find real ground truth, or declare its absence as the finding |
| Test only the happy path | Edge, adversarial, negative and degraded-mode cases too |
| Run against production on first invocation | Validate and dry-run free; ask before spending |
| Trust a judge because it sounds confident | Calibrate against human labels and report agreement |
