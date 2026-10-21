# Chaos and Resilience Testing

Traditional evaluation tests an agent under ideal conditions: tools respond correctly, the model
returns cleanly, data is present. In production, none of that is guaranteed — tools time out, APIs
return malformed responses, upstream data goes stale. Happy-path evals are table stakes; they answer
"does it work". Degraded-mode behaviour answers a different question, and it is usually the one that
decides whether the customer keeps the agent switched on: does it fail honestly, make partial
progress, and recover sensibly, or does it fall over silently and hand the user a confident wrong
answer.

## Fault injection mechanics

Inject faults through the agent framework's tool and model hooks rather than by breaking a real
backend — this keeps failures reproducible and keeps you off systems you don't control.

- **Pre-tool effects** — fire before the tool call executes. Cancel the call outright, return an
  error, or simulate the tool being unavailable.
- **Post-tool effects** — fire after the tool call returns. Corrupt the response, truncate it, or
  swap in a plausible-but-wrong payload.
- **Model-level effects** — cancel the model call, or corrupt the model's output before the agent
  acts on it.
- **The nastiest case: a tool that returns plausible-but-wrong data.** An error is easy to detect and
  usually handled. A tool that returns a well-formed, plausible, *incorrect* value — a stale price, a
  transposed order number — gives the agent nothing to react to. Include at least one case per
  critical tool that injects this rather than an outright failure.

## The three resilience dimensions

Score each of these separately. They are genuinely independent — an agent can fail communication
while completing the task, or communicate honestly while accomplishing nothing.

| Dimension | Question |
|---|---|
| Failure communication | Is the failure message clear, actionable, and honest about what actually broke — or vague, misleading, or silent? |
| Partial completion | How much of the user's goal was still achieved despite the failure? |
| Recovery strategy | Does the agent retry sensibly, vary its approach across attempts, and explore alternatives — or repeat the same failing action? |

## Method

1. **Always run a no-effect baseline first.** You need the clean-run score to know how much of any
   drop is caused by the injected fault versus baseline agent behaviour.
2. **Escalate severity gradually.** Start with a single low-severity fault before combining several
   or maxing out corruption rate.
3. **Run a degradation sweep.** Vary the failure or corruption rate from 0% to 100% to find the
   **resilience cliff** — the point where behaviour drops off sharply rather than degrading smoothly.
   The cliff's location is more informative than any single-point score.
4. **Use the Cartesian product of effects where budget allows.** Combining fault types
   systematically (tool timeout × model corruption × stale data) finds interaction failures a single
   fault at a time will miss.

## Tool simulation

Replace real tools with schema-constrained fakes for chaos runs. This makes failures reproducible on
demand and keeps you off real backends during fault injection. Share state across simulated tools so
call history is consistent — if a simulated tool has already returned a result once, a later call in
the same conversation should see that history rather than a fresh, contradictory fake.

## User simulation for multi-turn cases

For multi-turn resilience testing, drive the conversation with a simulated actor that has a defined
persona and goal, rather than a fixed transcript. Use a **structured stop condition** — a field the
actor emits in its own output — to end the conversation, not string-matching for something like
"goodbye" in the agent's reply. String matching against a stochastic model's phrasing is itself a
source of nondeterminism you don't want in your harness.

## Harness failures are not reasoning failures

A related but distinct class of failure caps your score for reasons that have nothing to do with the
model's reasoning: a tool that raises a hard exception the moment reference data is missing, rather
than returning a structured failure. This looks identical to a reasoning failure in the aggregate
score, but the fix is completely different — the fix is in the tool, not the prompt.

**Before blaming the agent for a low resilience score, check whether the cap is a harness problem.**
Diff the tool's behaviour on the failing input directly, outside the agent loop. If a tool crashes
rather than degrades, fix it by having it return a structured failure object — `success: false` plus
a note telling the agent what to do instead — rather than letting an exception propagate. A tool's
error message is effectively part of your prompt; a raw stack trace is the worst version of that
message you can ship.
