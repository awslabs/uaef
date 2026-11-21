# Evaluation Plan — [agent name]

[This is the Phase 1 deliverable. Fill every section from what you actually found reading the
agent's code — do not fill a section from a generic template. If you could not derive something, say
so under Open Questions rather than guessing. This document is the approval gate: present it and get
sign-off before Phase 2 generates a single test case.]

## Agent under evaluation

[Architecture as found, not as assumed. Framework (Strands / AgentCore / LangGraph / custom), model(s)
in use, system prompt summary, and a table of every tool with what it reads or mutates and how it
fails. Cite file paths and line numbers for each claim, e.g. `agent/tools/refund.py:42`. If the agent
is large, note which parts were read directly vs. summarised by a sub-agent.]

## Nondeterminism map

[Walk the ladder from `SKILL.md` and record which rungs this agent is actually on — do not assume all
four apply. For each rung present, name which eval dimensions it forces.]

| Rung | Present? | Scope | Eval dimensions it forces |
|---|---|---|---|
| Single-turn (inputs) | | all variants | instruction following |
| Single-turn (model output) | | all variants | functional correctness |
| Workflow (multi-step) | | | same two, per step |
| Single agent (tool choice) | | | tool selection, data precision |
| Multi-agent (handoff) | | [e.g. "Patterns B and C only"] | handoff accuracy |

Use the **Scope** column when the repo holds more than one candidate architecture under comparison —
a rung can be present for one variant and absent for another. One plan, one case set, scoped rows.

## Ground truth

[Most agents use MORE THAN ONE source — fill the table per dimension, taking the highest available
source for each. If a dimension has no ground truth, say so plainly and make building it
recommendation #1 — do not invent expected outputs.]

| Dimension | Ground-truth source | Why this one | Notes |
|---|---|---|---|
| [e.g. deterministic calculation] | programmatic — direct tool-call diff | pure function of fixed inputs | pinnable exactly |
| [e.g. narrative quality] | calibrated judge | no code can answer it | **judge not yet calibrated — open question** |
| [e.g. tool selection] | golden data | domain-expert authored | reused from [file] |
| [dimension] | [source] | [why] | [notes] |

Availability check, for the record:
1. System of record: [name it, or "none found" — and say whether that's a gap or structural, e.g. the
   agent deliberately never mutates anything]
2. Programmatic verification: [tests/schema/compiler available? where?]
3. Golden data: [domain-expert-authored, or reverted-fix-derived — source?]
4. Calibrated LLM judge: [for which dimensions, and whether calibration against human labels exists]
5. Gap: [what has no ground truth at all — an open question, not a thing to paper over]

**Prompt-only guardrails** (constraints with no code-level backstop): [list them, or "none". For each,
note that a deterministic tool test cannot cover it and say which judge plus trajectory check does.]

## Evaluation dimensions and levels

[One row per dimension, matched to OUTPUT / TOOL / TRACE / SESSION per the level table in SKILL.md.
Getting the level wrong is the most common mistake — a dimension scored at the wrong level either
can't be debugged (too coarse) or can't answer the real question (too fine).]

| Dimension | Level | Why this level |
|---|---|---|
| | | |

## Evaluator selection

[Every dimension above gets an evaluator. Deterministic before judged — justify any judge with what
code specifically cannot check.]

| Dimension | Evaluator | Deterministic or judge | Level | Why |
|---|---|---|---|---|
| | | | | |

## Case set design

[Counts per category and provenance — this is what gets checked against `test-cases.jsonl` in Phase
2. State the stratification rationale, e.g. why 30 normal / 10 edge / 5 failure / 5 adversarial / 5
negative for this agent specifically.]

| Category | Count | Provenance breakdown |
|---|---|---|
| normal | | |
| edge | | |
| failure | | |
| adversarial | | |
| negative | | |

## Success thresholds

[Numeric, on a sized held-out set — not "the agent should do well." E.g. "≥90% pass rate on
deterministic cases in `normal`, ≥0.8 mean judge score on `expected_assertion` cases in `edge`, zero
tool-selection errors on `adversarial`." State the set size these thresholds are calibrated against.]

## Risks and expected failure modes

[What you expect to break, based on reading the tools and prompt — not generic agent failure modes.
Tie each to a dimension above.]

## Cost and latency budget

[Per case and per full run: model calls × cost per call, judge calls × cost per call, expected wall
clock for `-n 3` across the full case set. State what "too expensive to run every time" means for
this suite and how that shapes CI cadence.]

## What this suite will NOT cover

[Be explicit about scope cuts — dimensions identified but not evaluated this round, and why (no
ground truth, out of budget, out of scope for this engagement).]

## Open questions for the customer/engineer

[Anything blocking Phase 2 that only the customer or the agent's owner can answer — missing ground
truth, ambiguous success criteria, access needed for real data.]
