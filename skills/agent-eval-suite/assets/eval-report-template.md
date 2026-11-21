# Evaluation Report — [agent name] — [run timestamp]

[This is the Phase 4 deliverable, generated from `results/<timestamp>/scorecard.json`. Every number
here must trace back to that file — do not hand-write a score that isn't in the scorecard.]

## Headline

[One sentence verdict: did the agent meet the thresholds set in the eval plan, and is it ready for
what the customer asked (POC handover / production / settling a quality dispute)? Say this plainly —
"meets thresholds, ready to hand over" or "fails on X, not ready" — don't hedge it into ambiguity.]

## Scorecard per category

[Never report only an aggregate — aggregates hide regressions in one category.]

| Category | Cases | Pass rate | Mean score | Cost (this run) | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| normal | | | | | | |
| edge | | | | | | |
| failure | | | | | | |
| adversarial | | | | | | |
| negative | | | | | | |
| **Overall** | | | | | | |

## Comparison against the previous run

[If `results/<previous-timestamp>/scorecard.json` exists, diff it against this run per category —
improved / regressed / unchanged, with the delta. If there is no previous run, say so explicitly:
"first run — no baseline to compare against."]

## Failure taxonomy

[Cluster failing cases by dimension/label and a short cause tag. The histogram is the roadmap — list
counts, not just examples.]

| Failure class | Count | Example case id(s) |
|---|---|---|
| | | |

## Root causes and fix routing

[One row per failure class above, routed to a fix type so the next step is obvious.]

| Failure class | Count | Root cause | Fix type | Recommendation |
|---|---|---|---|---|
| | | | system prompt / tool description / code / data | |

## Top three fixes, in ROI order

1. [Highest-impact, lowest-effort fix first — name the failure count it resolves.]
2. [...]
3. [...]

## Adversarial and degraded-mode results

[Red-team case outcomes (breach rate, refusal correctness) and chaos-scenario outcomes (did the agent
degrade gracefully when a tool failed, timed out, or returned corrupted data) — reported separately
from the main scorecard because they measure resilience, not baseline quality.]

## Judge calibration status

[If any dimension used an LLM judge: agreement rate against human labels if calibration was run, or
an explicit note that it was not calibrated this round and scores should be read with that caveat.]

## Coverage gaps

[What the eval plan explicitly said this suite would not cover — restate it here so the reader isn't
surprised by silence on a dimension.]

## How to reproduce this run

```bash
python eval/run_evals.py --n 3 --out results/<timestamp>
```

[State the seed used, the adapter version, and the case-set file/commit this run was executed
against, so the exact run can be repeated.]
