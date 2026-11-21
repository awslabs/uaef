# Judge Design

## Only judge what code cannot check

Before writing a judge prompt, state in the plan why this dimension needs one. If a deterministic
check — exact match, schema validity, a compiler, a diff against a system of record — can answer the
question, use that instead and skip the judge entirely. A judge is slower, costs money, and adds a
second source of non-determinism on top of the agent's own. Reserve it for genuinely subjective or
open-ended dimensions: faithfulness, helpfulness, coherence, whether a trajectory deviation was
reasonable.

## Prefer discrimination over generation

Judges are more reliable discriminating between options than generating a free-form score from
nothing. In order of preference:

1. **Pairwise comparison** — "is response A better than response B" — the most reliable format.
2. **Pass/fail against a criterion** — a binary verdict against a stated bar.
3. **Scoring with discrete anchors that carry descriptions** — e.g. 0.0 / 0.3 / 0.7 / 1.0, each with
   a one-line description of what that level looks like. Never ask for a bare 1–10 score with no
   anchor text; graders drift and disagree with each other on what a "7" means.

## Chain-of-thought before the verdict

Structure every judge prompt so reasoning comes first and the verdict comes last:

- Instruct the judge to write out its reasoning before giving an answer.
- Explicitly tell it **not** to state the verdict at the outset — anchoring on an early answer and
  then rationalising it is the dominant failure mode of judge prompts that ask for the verdict first.
- End with a single, parseable choice on the final line, in a fixed format the parser can match
  against a small closed set of strings.
- Parse from the end of the output, not the beginning — read the last line as the verdict and treat
  everything before it as reasoning to log, not to parse.

## Handle unparsable output explicitly

If the judge's output does not parse against the expected choice set, do not drop the sample and do
not silently default it to a passing score. Assign it to an explicit `__invalid__` bucket and score
it at the **minimum of the scale**. This keeps a broken judge call visible in the aggregate — a
silently dropped sample or a defaulted mid-range score will hide judge failures inside a metric that
looks healthy.

## Include an "immaterial difference" category

Give the judge an explicit category for "the answers differ, but the difference does not matter" —
separate from both "correct" and "incorrect." Without it, cosmetic variation (different phrasing,
different but equivalent formatting, a synonym) gets folded into whichever adjacent bucket the judge
happens to lean toward, adding noise that looks like a real disagreement.

## Every score carries a reason

No judge output should be a bare number or a bare verdict. Require a `reason` field on every score,
even for binary pass/fail evaluators. A number with no reason cannot be debugged, and a reviewer
spot-checking results has nothing to check the number against.

## Bias controls

| Bias | What it does | Control |
|---|---|---|
| Position bias | The judge favours whichever response appears first (or second) in a pairwise prompt | Randomise the order presented to the judge, and if feasible run both orders and check for flips |
| Verbosity / length bias | The judge favours the longer response regardless of quality | Control for length — either equalise it, or instruct the judge explicitly to ignore length as a signal of quality |
| Self-preference bias | A judge favours output from its own model family | Judge with a different model family than the system under test |

Treat all three as table stakes on every judge you ship, not as optional hardening.

## Calibration is a gate, not a nice-to-have

Never trust a judge because its reasoning reads convincingly. Calibrate it before it scores anything
that a decision depends on:

1. Hand-label roughly 50–100 examples with a human verdict.
2. Run the judge on the same examples and measure agreement against the human labels — report the
   number (plain accuracy, or Cohen's kappa if the scale has more than two levels).
3. If agreement is weak, fix the rubric — tighten the wording, add missing anchor descriptions, add
   an immaterial-difference category — and recalibrate. Do not tune the judge to pass calibration by
   relabelling the humans' answers.
4. Only scale the judge up once it is **faster, cheaper, and consistently agrees with the human
   annotations** — all three, not just the first two. A judge that is fast and cheap but disagrees
   with humans is not ready; it is just wrong more efficiently.

Re-run calibration whenever the rubric changes materially — calibration is a property of a specific
rubric version, not of the judge in general.

## The judge output contract

Model every judge's output on a strict schema, not free prose. At minimum:

- **Per-finding fields**: a `title`, a `body` explaining the finding, a `confidence` score, and a
  `severity` on a fixed scale (e.g. P0–P3).
- **A mandatory location**: a file/line reference for code, or a step/span id for a trajectory —
  never a finding with nowhere to look.
- **A forced overall verdict**: the judge must commit to a single top-level pass/fail or score, even
  if it also lists multiple findings underneath.
- **An explicit exclusion criterion.** State plainly what the judge must *not* flag — e.g. "do not
  flag pre-existing issues that predate this change, only ones this change introduced." Exclusions
  are what stop a judge from generating noise; a judge with no exclusion criterion will flag
  everything it can find, whether or not it is in scope.

## Rubric wording steers generation, not just scoring

Loaded or leading phrasing in a rubric does not just bias how the judge scores — if the same prompt
template is reused to *generate* candidate outputs (e.g. in a battle/ranking setup), loaded language
causes outputs to converge toward whatever the wording implies is "good." Write rubric criteria in
neutral, descriptive language, and review rubric drafts for phrasing that presupposes the answer.

## Worked example: judge prompt skeleton

```
ROLE
You are evaluating whether an agent's response satisfies the stated criteria for this task.
You are not the agent's author; judge only what is in front of you.

CRITERIA
1. The response correctly answers the user's question, grounded in the retrieved context.
2. The response does not fabricate information not present in the context.
3. The response follows the requested output format.

EXCLUSIONS
- Do not penalise phrasing, tone, or formatting differences that do not change the substance
  of the answer — score these as "immaterial difference," not as a failure.
- Do not penalise the response for information the context itself does not contain.

REASONING INSTRUCTION
Think through each criterion in turn, citing the specific part of the context or response
that supports your reasoning. Do not state your final verdict until you have finished this
reasoning. Do not state the verdict at the outset and work backward to justify it.

VERDICT FORMAT
After your reasoning, output exactly one line in this form, and nothing after it:
VERDICT: <PASS|FAIL|IMMATERIAL_DIFFERENCE|__invalid__>

Include a one-sentence `reason` before the verdict line summarising why.
```
