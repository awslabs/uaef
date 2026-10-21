# Edge-Case Checklist

A happy-path suite tells you the agent works when everything is normal. It tells you nothing about
the cases that generate the support tickets. Walk every bucket below for the target agent and write
at least one case per bucket that applies. Skip a bucket only if it genuinely cannot occur for this
agent, and say so in the plan rather than silently dropping it.

## Input variability

- **Non-English / multilingual input.** A user writing in a language other than the one the system
  prompt assumes.
- **Non-prose formats.** Input arriving as XML, JSON, Markdown, or CSV rather than a sentence — e.g.
  a support ticket pasted as a JSON blob.
- **Other modalities.** An image attached where the agent expects text (a screenshot of an error, a
  photo of a damaged item).
- **Very long input.** A transcript, log dump, or document far past the agent's usual input size.
- **Empty or near-empty input.** A blank message, a single emoji, whitespace only.

## Contextual complexity

- **Multiple intents in one request.** "Cancel my order and also update my address" — does the agent
  handle both, or silently drop one?
- **Typos and misspellings.** Realistic user typing, not clean test-writer prose.
- **Minimal-context requests.** A user who just says "returns" with no other detail — does the agent
  ask a clarifying question or guess?
- **Long-running conversations.** Does behaviour degrade or drift after 20+ turns?
- **Ambiguous tool-result property names.** A tool returns a field like `on: 123` — is `on` the order
  number, a boolean, a timestamp? Write a case where the agent must not guess.
- **Multiple tool calls leading to wrong arguments.** A case chained across several tool calls where
  an earlier result's shape misleads the argument extraction for a later call.
- **Circular handoffs in multi-agent systems.** Agent A hands to Agent B, which hands back to Agent A
  for the same reason, with no progress — test that the loop terminates or escalates.

## Personalisation and conflict

- **Jailbreak attempts** — covered in depth in `red-teaming.md`; include at least a light pass here
  so the eval-cases file has coverage even without a full red-team pass.
- **Formatting demands** that conflict with the agent's normal output shape (a user demanding raw
  JSON when the agent is meant to answer conversationally).
- **User prompt conflicts with system prompt.** The user directly asks the agent to ignore, override,
  or contradict an instruction in its system prompt. Define evals for **what you want to support and
  what you want to block** — a case testing that a benign override is honoured, and a case testing
  that a policy-violating override is refused, are both needed; they are not the same test.

## Data and state edge cases

- **Missing reference data.** A lookup the agent depends on returns nothing because the record does
  not exist yet.
- **Stale data.** A tool returns a cached or outdated value that contradicts what the user is telling
  the agent live.
- **Empty result sets.** A search or query tool legitimately returns zero rows — does the agent say
  so, or hallucinate a result?
- **Duplicate records.** Two records match a lookup that the agent's logic assumes is unique.
- **Success with an empty payload.** A tool call returns `200 OK` / `success: true` with no data —
  the failure mode that looks like success and is the easiest to miss in review.

## Negative and zero-result cases

The correct answer is sometimes **"none", "refuse", or "escalate"** — and this is the bucket agents
are least often tested on, because it is easy to write cases where something should happen and hard
to remember to write cases where nothing should. Write explicit cases where:

- No matching record exists and the correct response is "I couldn't find that", not a fabricated
  answer.
- The request is outside the agent's remit and the correct response is a refusal or a handoff.
- A search legitimately returns nothing, and the agent must say so rather than paraphrasing the empty
  result into something that sounds like an answer.

This is a common real bug in production agents: the model is trained to be helpful, and "there is
nothing to report" reads as unhelpful, so it invents something plausible instead.

## Boundary arithmetic

- **Off-by-one on thresholds.** A discount that applies "for orders over $100" tested at exactly
  $100 and at $99.99.
- **Date boundaries.** A policy that changes "after 30 days" tested at day 29, 30, and 31; timezone
  edges around midnight.
- **Unit conversions.** Mixed units in the same request (kg vs lb, miles vs km) that the agent must
  normalise before comparing.
- **"Earliest of several thresholds" logic.** A case with three candidate deadlines where the correct
  answer is the minimum, not the first one mentioned or the maximum.

Each bucket above should map to at least one concrete case in `test-cases.jsonl`, tagged with the
bucket name in `metadata` so the eventual failure-taxonomy pass (see `diagnosis.md`) can be sliced by
bucket, not just by pass/fail.
