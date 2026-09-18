# Agent Eval Suite

> **This README is for humans.** It explains why the skill exists, when to use it, and what's inside. The agent-facing workflow lives in [`SKILL.md`](SKILL.md).

Generates a complete, agent-specific evaluation suite — plan, test cases, deterministic checks, LLM judges, trajectory and tool-level evals, adversarial and red-team cases, chaos tests, and a runnable harness with a report — derived from an agent's own tools, prompts, and failure surface, not from a generic checklist.

## Why It Exists

Agent evaluation gets rebuilt from scratch on every engagement, at a different standard each time. That inconsistency costs the team:

- Nobody can say with confidence whether an agent got better or worse between two versions
- POCs get handed over to customers with no regression suite, so the next change is a guess
- "How do you know it works?" gets answered with a demo, not a number
- Every engineer reinvents case design, judge rubrics, and trajectory scoring from first principles

This skill standardises the process without standardising the content — point it at an agent and it reads that agent's actual tools, prompts, and data sources to derive what to measure, then generates the suite that measures it.

## When to Use

- Before handing a POC over to a customer
- Before putting an agent into production
- When an agent's quality is disputed and there's no suite to settle it
- When you inherit an agent someone else built and need to know its actual failure surface
- When a customer asks "how do I know this works?" or "has it got better?"

## Example triggers

- "Build an eval suite for this agent"
- "Evaluate this agent's tool selection and see where it fails"
- "Set up regression tests before we hand this POC over"
- "Red-team this agent and stress-test it"
- "How do I know this agent actually works?"
- "Has this agent gotten better since last month?"
- "We disagree on whether this agent is good enough — settle it"

## What you get

```
eval/
  eval-plan.md              approved plan: architecture, ground truth, evaluator selection, thresholds
  test-cases.jsonl          normal / edge / failure / adversarial / negative cases
  red_team/cases.jsonl
  chaos/scenarios.yaml
  evaluators/               deterministic.py, judges.py, trajectory.py
  adapters/agent_adapter.py the one file that knows how to invoke this specific agent
  run_evals.py              CLI: --validate, --dry-run, --subset, --tags, --n, --level
  report.py
  results/<timestamp>/      runs.jsonl, scores.jsonl, scorecard.json, eval-report.md
  README.md
```

## The four phases

1. **Analyse + plan** — read the agent, map its nondeterminism, find (or flag the absence of) ground truth, choose evaluators and levels, set numeric thresholds. **Stop here — this plan is the approval gate.**
2. **Generate cases** — `test-cases.jsonl` stratified across normal/edge/failure/adversarial/negative, plus red-team and chaos files. Real cases from tickets and logs preferred over synthetic.
3. **Generate the harness** — the file tree above, validated by running `--validate` and `--dry-run` before handing it back. **Stop here — confirm before Phase 4 spends money or touches a live system.**
4. **Run + report** (opt-in) — execute the suite, produce `eval-report.md` with per-category breakdown, cost, latency, failure taxonomy, and fix recommendations.

## Requirements

- Read access to the target agent's repository
- Python 3.10+ and UAEF installed (`uv sync` from the repo root) — the generated harness scores through
  `uaef.api.evaluate` and persists through UAEF, so this is a hard dependency, not an optional one
- Optional: `strands-evals` for OTel-trace-native trajectory scoring on Strands agents
- AWS/Bedrock credentials are needed **only** for the opt-in real run in Phase 4 — generation, `--validate`, and `--dry-run` all work with none

## What this deliberately does not do

- It does not fix the agent. It tells you what's broken and routes each failure to a fix type (system prompt / tool description / code / data) — the fix itself is separate work.
- It does not invent ground truth. If none exists, the plan says so and recommends building it as the first step, rather than presenting a judge's opinion as fact.
- It does not run against a live agent — let alone production — without explicit confirmation at the Phase 3→4 gate.
