# Eval Design Principles

## What an eval is for

An eval suite is a regression detector for a system whose behaviour you do not fully control, not a
leaderboard number. Its job is to detect the impact of change — a new model version, a prompt edit, a
tool rewrite — before the customer does. Treat the suite as dual-purpose: it is both the test suite
that gates a release and the training signal that tells you what to fix next. If a suite cannot tell
you whether last week's change helped, it has failed at its one job, however polished its dashboard
looks.

## The four criteria for a good eval case set

Every case set you build must satisfy all four. A set that fails one is not ready to ship.

| Criterion | What it means | How to check it |
|---|---|---|
| Thematically consistent | Each case tests one coherent thing. Do not mix unrelated capabilities into a single set and average the result. | Read the set back — can you name the one capability it probes? |
| Challenging | A domain expert could do the task, but the agent does not currently ace it. If the agent scores near-perfect on everything, the suite is too easy to be useful. | Run the current agent against a draft set before finalising it; if it aces every case, add harder ones. |
| Directionally clear | There is real signal on what "right" looks like — high-quality reference answers, or an exhaustive rubric. Vague or contested ground truth produces vague or contested scores. | Ask whether two competent reviewers would agree on the right answer without discussion. |
| Carefully crafted | The prompts are engineered for good performance, the best-fitting eval template was chosen, and someone has spot-checked the results. | Actually read a sample of completions before trusting any aggregate score. |

## The five-step workflow

1. **Define the eval objective.** State the success criterion in one sentence before writing a
   single case. If you cannot state it, you are not ready to collect data.
2. **Collect the dataset.** Prefer, in this order: production traffic, historical data (tickets,
   transcripts, logs), domain-expert-authored cases, then synthetic cases generated to fill gaps in
   scenario or difficulty coverage. Synthetic cases should be labelled as such and used to extend
   coverage, not to replace real cases.
3. **Define metrics — with numeric thresholds on a sized set.** "Better" is not a metric. State the
   set size and the threshold together: *"on a held-out set of 40 cases, achieve tool-selection
   accuracy ≥ 90% and a groundedness score ≥ 0.8."* A threshold with no denominator is unfalsifiable.
4. **Run and compare.** Execute the suite, compare against the previous run or against a baseline
   configuration, and read the failures before reading the aggregate.
5. **Continuously evaluate.** Run the suite on every material change, watch production for new
   failure modes the suite missed, and fold every new failure back into the case set. The suite
   should be larger and sharper next quarter than it is today.

## Tips

- **Eval-driven development from day one.** Write the first scoped tests before or alongside the
  first version of the agent, not after it ships. Retrofitting an eval suite onto a shipped agent
  finds problems later and more expensively than building the suite first.
- **Task-specific, not generic.** Metrics must reflect this agent's real-world distribution of
  inputs, not a general capability benchmark. A generic metric that does not track this agent's
  actual failure surface is decoration.
- **Log everything.** Instrument the agent so every run is logged, then mine those logs for eval
  cases — the best cases are the ones production actually produced, not the ones you imagined.
- **Automate scoring wherever possible.** A suite that requires a human to read every result will
  not be run often enough to matter.
- **It is continuous, not a one-off deliverable.** Evaluation is a practice, not a milestone on a
  project plan.
- **Maintain agreement with human labels.** Automated scores are only trustworthy to the extent they
  track what a human reviewer would say. Measure that agreement, do not assume it.

## The four anti-patterns

Name these explicitly when reviewing a proposed suite — each is a specific, recognisable failure,
not a vague warning.

1. **Overly generic academic metrics.** Relying on perplexity, BLEU, ROUGE, or similar scores that
   were designed for a different problem and do not track whether this agent does its job.
2. **Biased design.** An eval set that does not faithfully reproduce production traffic patterns —
   built from imagination rather than from what customers actually send.
3. **Vibe-based evals.** "It seems to be working" as the entire evaluation strategy, or waiting
   until after shipping to build any evaluation at all. This is the most common failure mode and the
   most damaging, because it is invisible until something breaks in front of a customer.
4. **Ignoring human feedback.** Running an automated score without ever checking whether it agrees
   with what a human reviewer would conclude.

## LLMs discriminate better than they generate

Design evaluation tasks around this asymmetry. A model asked to produce open-ended free text is a
weaker measurement instrument than the same model asked to choose between two options, classify an
input, or score against an explicit criterion. Wherever the task allows it, prefer pairwise
comparison, classification, or scoring against a specific rubric over open-ended generation — both
for the agent under test and for any LLM judge scoring it.

## Statistics discipline

Both the agent and any LLM judge are non-deterministic — treat every score as a sample, not a fact.

- **Run multiple rounds.** A single run of a stochastic system tells you almost nothing about
  whether a change helped.
- **Paired comparison on identical cases.** Compare the same case across two configurations, not
  two different samples across two configurations — otherwise sampling noise dominates the signal
  you are trying to measure.
- **Do not believe a delta inside the noise band.** If the change between two runs is smaller than
  the run-to-run variance you would see from re-running the *same* configuration, it is not a
  finding.
- **Use a bootstrap estimate or a sign test, not eyeballing.** Resample and report a spread, or run
  a paired sign test across cases, before declaring one configuration better than another.

## Version the eval, not just the model

Give every eval a version, and bump it whenever its meaning changes — a rubric edit, a new
threshold, a changed case set. An eval that silently changes what it measures produces "evaluation
drift": every historical number quietly becomes incomparable to the current one, and nobody notices
until a regression is missed or a false alarm is raised. If you cannot say which version of the eval
produced a given number, treat that number as unusable.

## Cheap-iteration mechanics

Evaluation work is mostly iterating on scoring logic, not re-running the agent. Design for that:

- **Cache task outputs per case.** Run the agent once, store its output keyed by case, and let every
  subsequent iteration on evaluators or judges read from the cache. Re-invoking a non-deterministic,
  costly agent every time you tweak a rubric wastes money and reintroduces noise into a comparison
  that should be held constant.
- **Gate on schema validation before spending a model call.** A fast, free check that a case or a
  judge's expected output conforms to its schema should exit non-zero immediately on malformed
  input. Run this in CI before any evaluator that costs money or time.
