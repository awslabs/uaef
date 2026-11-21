# Diagnosis: From Score to Fix

Evaluators give you a score. Detectors give you a diagnosis. A score tells you whether to ship — pass
rate, breach threshold, resilience cliff. A diagnosis tells you what to change on Monday. Run
diagnosis on top of the scorecard, not instead of it, and run it only on the cases that failed —
diagnosing a pass is a wasted model call and adds nothing to the backlog.

## Failure taxonomy

Tag every failing case (or failing span, if you have trace-level detail) against this taxonomy. Use
it as a fixed tagging scheme so tags can be counted across a run, not as free-text description.

| Parent category | Leaves |
|---|---|
| Execution error | auth · rate-limiting · timeout · tool schema |
| Task / instruction | non-compliance · problem misidentification |
| Incorrect actions | wrong tool selection · poor information retrieval |
| Context handling | — |
| Hallucination | of capabilities · of usage · of parameters · fabricated tool outputs |
| Repetitive behaviour / loops | — |
| Orchestration | reasoning-mismatch · goal deviation · premature termination |
| Nonsensical output | — |
| Configuration mismatch | — |

A single failing span can carry more than one tag — a timeout (execution error) that triggers the
agent to fabricate a plausible result (hallucination) is two tags on one root incident, not one.

## Root-cause analysis dimensions

For each tagged failure, record:

- **Causality** — `primary` / `secondary` / `tertiary` / `unclear`. Build a **causal chain**, not a
  flat list of things that went wrong. If a tool timeout caused a retry loop that caused the agent to
  give up early, the timeout is primary, the loop is secondary, the early termination is tertiary.
  Treating all three as independent findings produces three fixes for one bug.
- **Propagation impact** — `task termination` / `quality degradation` / `incorrect path` /
  `state corruption` / `none`. What did this failure actually do downstream, if anything?
- **Detection timing** — `immediately` / `several steps later` / `only at task end` /
  **`silent-undetected`**. This last category is the hardest and most important thing to measure: a
  failure nobody — not the agent, not the user, not the eval — noticed at the time. A suite that only
  measures the failures the agent itself flagged is measuring the easy half of the problem.
- **Completion status** — did the task ultimately complete, and at what quality, despite the failure.

## Fix-type routing

Every root cause routes to exactly one place you intervene. Naming the location turns error analysis
from a reading exercise into a routed backlog rather than a pile of observations.

| Root cause pattern | Route to | Example fix |
|---|---|---|
| Model ignores or misweights an instruction | System prompt | Add an explicit priority rule; move the instruction earlier |
| Model picks the wrong tool, or the right tool at the wrong moment | Tool description | Clarify when the tool should and should not be used |
| Tool crashes or returns malformed data on an edge case | Tool code | Return a structured failure with a note instead of raising |
| Missing, stale, or contradictory reference data | Data | Fix the source data or add a freshness check |
| A capability the model has is never exercised because the harness doesn't expose it | Harness | Wire the missing hook, retry, or timeout handling |
| The failure requires a different decomposition entirely (e.g. should be multi-agent) | Architecture | Redesign the flow, driven by the eval evidence, not instinct |

## Method

Read roughly 20 failing traces by hand before trusting any automated detector's output. Tag each one
against the taxonomy above. **Count the tags** — the histogram is the roadmap. Fix the tallest bar
first, one change per iteration, then re-run the suite. Do not batch multiple unrelated fixes into
one iteration; you lose the ability to attribute the resulting score change to a specific change.

## Localising a regression by elimination

When a score has dropped between two runs and you don't yet know why, prove which layer moved before
changing anything, cheapest check first:

1. **Diff the tool code** between the two runs. If tool I/O contracts are provably identical — same
   inputs produce byte-identical outputs — the regression is not in the tools.
2. **Diff the data** the tools are reading. Stale or changed reference data can look exactly like a
   reasoning regression.
3. **Diff the prompt.** If tools and data are ruled out and the prompt changed, the regression is
   almost certainly there.
4. **Diff the model version.** Only reach here once the above are ruled out — a model-version change
   is the most expensive hypothesis to act on and the easiest to blame prematurely.

The worked pattern: if tool I/O contracts are provably identical across two runs, the change is in
the prompt or schema, not the tools — stop investigating the tools and move on.

## Diagnostics worth running

- **The oracle-context test.** Re-run the failing cases with the ground-truth context handed to the
  agent directly, bypassing retrieval. If performance jumps, the problem is retrieval. If performance
  is still bad with oracle context, the problem is reasoning or tool use, not retrieval — don't spend
  another cycle tuning the retriever.
- **Separating a harness cap from a reasoning cap.** Before concluding the model is the bottleneck,
  confirm the harness isn't capping the score for unrelated reasons (see `chaos-and-resilience.md` on
  tools that crash rather than degrade). A harness cap looks identical to a reasoning ceiling in the
  aggregate number and is far cheaper to fix.

## Cost discipline

Run detectors only on failing cases. A passing case does not need a root-cause analysis, and running
RCA on every case regardless of outcome multiplies your model spend for no additional information —
wire the trigger to fire on failure only.
