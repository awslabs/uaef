# Evaluator Catalogue

## The four evaluation levels

Match the level to the question before picking an evaluator. This is the single most common mistake
in agent evaluation — scoring only the final answer and then being unable to debug why it is wrong.

| Level | Unit | Answers |
|---|---|---|
| OUTPUT | one response | Is the final answer correct, concise, well-formed? |
| TOOL | each individual tool call | Was *this* call the right tool, with grounded arguments? |
| TRACE | the full path, including tool calls | Was the route sensible? Is the answer faithful to what it actually read? |
| SESSION | the whole multi-turn conversation | Did the user's goal actually get achieved? |

A TOOL-level evaluator does not return one verdict per case — it returns one verdict per call. A
case with three tool calls produces three TOOL-level outputs. Aggregate carefully: a single bad call
buried in an otherwise-clean trajectory should not disappear into an averaged score.

## Deterministic evaluators

Run these first, on every case, before spending a single model call on a judge. They are free, fast,
and cannot be talked around.

| Evaluator | Checks | Notes |
|---|---|---|
| Exact match | Output equals a reference string exactly | Narrow answer space only |
| Contains | Reference substring appears in the output | Looser than exact match |
| Starts-with | Output begins with an expected prefix | Useful for structured response headers |
| Schema / JSON validity | Output parses and conforms to an expected schema | Invalid JSON should never count as a match |
| Tool-called | A specific tool was invoked (optionally with expected arguments) | TOOL level |
| State-equals | A named piece of environment state has the expected value after the run | Verifies the tool's **side effect** actually happened — the row really changed, not just that the tool returned success |
| Domain-specific programmatic checks | Tests pass, code compiles, types check, lints clean, diff-vs-system-of-record | The strongest ground truth available for coding agents and agents that mutate a system of record — prefer this over any judge whenever it exists |

## LLM-judge evaluators

Reach for a judge only once the deterministic layer is exhausted — only for what code genuinely
cannot check.

| Evaluator | What it measures | Level | Scale |
|---|---|---|---|
| Custom rubric / output | Any bespoke criterion specific to this agent | Output | 0–1 or discrete anchors |
| Trajectory | Whether the tool-call sequence followed an appropriate path | Session | 0–1 |
| Interactions | Quality of inter-agent handoffs and exchanges in a multi-agent workflow | Per interaction | 0–1 |
| Correctness | Factual correctness, optionally against a reference | Trace | 0–1 |
| Faithfulness / groundedness | Whether the answer is supported by what was actually retrieved or read | Trace | 0–1 |
| Helpfulness | User-perceived usefulness of the response | Trace | 0–1 |
| Coherence | Internal contradictions, logical consistency | Trace | 0–1 |
| Conciseness | Brevity and efficiency of the response | Trace | 0–1 |
| Response relevance | On-topic vs. tangential | Trace | 0–1 |
| Instruction following | Format, length, and style compliance | Output | Binary |
| Harmfulness | Dangerous or offensive content present | Output | Binary |
| Refusal / over-refusal | The agent refused when it should have helped | Output | Binary |
| Stereotyping / bias | Bias against a group present in the output | Output | Binary |
| Tool-selection accuracy | Was the right tool chosen, at the right moment | Per tool call | Binary |
| Tool-parameter accuracy ("data precision") | Are the arguments grounded in the conversation, or fabricated | Per tool call | Binary |
| Goal success rate | Did the user's goal actually get achieved | Session | 0–1 |
| Failure communication | Clarity, actionability, and transparency of failure messages | Session | 5-level |
| Partial completion | Fraction of the goal achieved despite a failure | Session | 0–1 |
| Recovery strategy | Retry discipline, exploration breadth, variation in approach after a failure | Session | 5-level |
| Multimodal variants | Same axes applied to image/document→text tasks (VQA, OCR, captioning) | Varies | Usually Likert-5 |

Three pairings and distinctions to hold onto:

- **Harmfulness and refusal are a pair, not two independent metrics.** Refusal is the failure mode
  you *create* the moment you push hard on harmfulness — an agent tuned to refuse more will start
  refusing legitimate requests. Measure both together, or a harmfulness fix will look like a pure win
  when it has actually traded one failure for another.
- **Tool-selection and tool-parameter accuracy are orthogonal failure modes.** Tool-selection asks
  "was this the right tool" — a planning error. Tool-parameter accuracy asks "are the arguments
  hallucinated or actually grounded in context" — a hallucination error. They need different fixes
  (routing logic vs. extraction grounding), so score and report them separately.
- **State-equals checks side effects, text-based checks check descriptions.** For any agent that
  mutates the world, prefer asserting on the world.

## How trajectory evaluation actually works

Trajectory judging is the part most suites get wrong by treating it as pure judge vibes. Do this
instead:

1. **Feed the judge a trace or session object, not raw text.** This means either OTel spans keyed by
   session/conversation ID, or the ordered message list for the run. Session-ID tagging is mandatory
   — without it, spans from different test cases can mix together in the trace store.
2. **Anchor the judge with deterministic comparators it can call**, rather than asking it to eyeball
   the whole sequence:
   - **exact match** — the tool-call sequence matches the expected sequence exactly
   - **in-order match** — the expected steps are present, in order, extra steps allowed
   - **any-order match** — all expected steps are present, in any order
3. **Give the judge a rubric and, where possible, an expected trajectory description.** The
   deterministic comparators handle the matching; the judge's job shrinks to interpreting whether a
   deviation from the expected path was reasonable, not doing the matching itself from scratch.
4. **Compress long traces before judging.** Summarise to a tool-call summary (tools used, arguments,
   outcomes) rather than passing the full raw trace. An uncompressed long trace will overflow the
   judge's context on exactly the runs you most need to grade — typically the long, messy,
   interesting failures.

## Selection guidance

Deterministic checks run first, always, as a cheap gate — a judge should never be the first thing
that touches a case.

| What you want to know | Evaluator to pick |
|---|---|
| Did the tool actually change the system? | State-equals (deterministic) |
| Did the code the agent wrote actually work? | Compile / test / lint / type-check (deterministic) |
| Is the final answer factually right? | Correctness judge, or diff against a system of record if one exists |
| Is the answer grounded in what was retrieved? | Faithfulness / groundedness judge |
| Did it pick the right tool? | Tool-selection accuracy, per call |
| Did it invent arguments? | Tool-parameter accuracy, per call |
| Did it follow a sensible path to the answer? | Trajectory judge, anchored by exact/in-order/any-order comparators |
| Did the user's actual goal get met, across the whole conversation? | Goal success rate, session level |
| Does it refuse things it shouldn't, or comply with things it shouldn't? | Harmfulness + refusal, as a pair |
| How does it behave when a tool fails? | Failure communication + partial completion + recovery strategy, together |
| Is it biased against a group? | Stereotyping / bias judge |
| Did multiple agents hand off sensibly, including back again? | Interactions judge at the handoff level |

The rule underneath the table: pick the cheapest layer that can answer the question, and never ask a
TOOL-level evaluator to answer a SESSION-level question or vice versa.
