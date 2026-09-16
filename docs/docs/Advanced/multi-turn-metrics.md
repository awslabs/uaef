# Multi-Turn Conversation Metrics

UAEF ships a `Multi-Turn` metric dimension for evaluating multi-turn agent
conversations. The dimension contains 15 metrics — 4 originals plus 11
ported from Amazon's AMACE Conversation Evaluator — and is integrated into
the standard evaluation flow described in
[Multi-Turn Evaluation Workflow](multi-turn-evaluation-workflow.md).

This document is the catalog: what each metric measures, how it scores,
whether it needs ground truth, and where it fits in the per-turn vs.
full-trace dispatch.

## Where these metrics run

The evaluator routes metrics in two phases (see
`src/uaef/evaluation/base_evaluator.py:_run_multi_turn`):

- **Per-turn phase** — judges each (user, assistant) exchange in isolation.
  The loop averages the per-turn scores and preserves each turn's reasoning
  in `metric_score.metadata["per_turn"]`.
- **Full-trace phase** — judges once over the full conversation transcript.

A metric is per-turn iff its `get_name()` is **not** in the
`FULL_TRACE_ONLY` set in `base_evaluator.py`. Today every Multi-Turn metric
is full-trace. For per-turn relevance scoring across a multi-turn dialog,
include `answer_relevance` (Response Quality dimension) in your metric list
— the per-turn dispatch loop runs it on each turn and averages.

## The metrics

### Originals

| Metric | Phase | Description | Score | Ground truth | LLM judge |
|--------|-------|-------------|-------|--------------|-----------|
| `context_retention` | full-trace | Whether the agent maintains and uses information from earlier turns. Returns `null` when the conversation has no shared context to retain. | 0–1 | no | yes (Bedrock) |
| `coherence` | full-trace | Logical flow, relevance, consistency, and topic-tracking across the conversation. | 0–1 | no | yes (Bedrock) |
| `conversation_completeness` | full-trace | Whether the user's goals were achieved by end of conversation. Uses `GroundTruth.expected_output` as success criteria when present; otherwise the judge extracts intentions from the conversation. | 0–1 | optional | yes (Bedrock) |
| `turn_efficiency` | full-trace | Deterministic. Score is `completeness × min(1, expected_turns / actual_turns)`. Reads `completeness_score` from `trace.metadata` if present. | 0–1 | no | no |

### AMACE-derived (ported in 2026)

All of the following inherit the existing UAEF Bedrock-judge pattern and
return a `MetricScore` with `metric_name`, `score` in [0, 1], `reasoning`,
and a `metadata` dict. For the 1–5 scale metrics, the raw integer is
preserved in `metadata["raw_score"]`. For boolean metrics, the original
0/1 is stored in `metadata["raw"]`.

| Metric | Phase | Description | Score | Ground truth | LLM judge |
|--------|-------|-------------|-------|--------------|-----------|
| `role_adherence` | full-trace | Ratio of assistant turns that stay in the assigned role/persona. | 0–1 | required (`expected_arguments["chatbot_role"]`) | yes (Bedrock) |
| `holistic_llm_judge` | full-trace | Composite multi-factor judge weighing completeness, relevancy, knowledge retention, sentiment, tone, satisfaction, and (heavily) containment. | 0–1 | no | yes (Bedrock) |
| `user_satisfaction` | full-trace | End-of-conversation user satisfaction. Raw 1–5 normalized to 0–1. | 0–1 (raw 1–5 in `metadata["raw_score"]`) | no | yes (Bedrock) |
| `sentiment` | full-trace | User's sentiment across the conversation. The judge weighs later interactions more heavily. Raw 1–5 normalized to 0–1. | 0–1 (raw 1–5 in `metadata["raw_score"]`) | no | yes (Bedrock) |
| `agent_tone` | full-trace | Agent's tone professionalism. Raw 1–5 normalized to 0–1. | 0–1 (raw 1–5 in `metadata["raw_score"]`) | no | yes (Bedrock) |
| `naturalness` | full-trace | Naturalness of the agent's responses across the conversation. Raw 1–5 (1 = robotic / unnatural, 5 = natural / human-like) normalized to 0–1. | 0–1 (raw 1–5 in `metadata["raw_score"]`) | no | yes (Bedrock) |
| `per_turn_sentiment` | per-turn (mean) | User sentiment scored independently per user→assistant exchange. Mean across turns. Per-turn scores in `metadata["per_turn_scores"]`. | 0–1 (per-turn array in metadata) | no | yes (Bedrock) |
| `per_turn_agent_tone` | per-turn (mean) | Agent tone professionalism scored independently per user→assistant exchange. Mean across turns. | 0–1 (per-turn array in metadata) | no | yes (Bedrock) |
| `per_turn_naturalness` | per-turn (mean) | Naturalness of the agent's response scored independently per user→assistant exchange. Mean across turns. | 0–1 (per-turn array in metadata) | no | yes (Bedrock) |
| `instruction_compliance` | full-trace | Whether the agent followed the original system-prompt instructions across the conversation. | 0–1 | required (`expected_arguments["system_instructions"]`, falls back to `expected_output`) | yes (Bedrock) |
| `optimum_turns` | full-trace | LLM-estimated optimum turn count divided by actual turns, capped at 1.0. Raw integer optimum stored in `metadata["optimum_turns"]`. | 0–1 (raw counts in `metadata`) | no | yes (Bedrock) |

> **Note:** `containment` and `resolution` were originally part of this set but
> were carved out as **opt-in use-case-specific metrics** (contact-center focus).
> They are not in the default registry. See
> [Use-Case-Specific Metrics](use-case-specific-metrics.md) for activation
> and usage.

For a list of which specific turns drifted out of role, opt into
`deepeval_role_adherence` instead — it wraps DeepEval's
`RoleAdherenceMetric`, which returns per-turn out-of-character verdicts in
`metadata["out_of_character_turns"]`.

## Ground-truth shape

Two AMACE metrics need extra context that you supply via
`GroundTruth.expected_arguments`:

```python
from uaef.models.ground_truth import GroundTruth

gt = GroundTruth(
    expected_output="Resolve the customer's billing question.",  # used by conversation_completeness
    expected_arguments={
        "chatbot_role": (
            "You are a customer support specialist for ACME Corp's "
            "billing team. You can answer billing questions and create "
            "tickets, but you cannot make refunds."
        ),
        "system_instructions": (
            "Always greet the customer by name. Never quote prices "
            "without confirming the latest pricing. Escalate to a human "
            "agent if the customer asks for a refund."
        ),
    },
)
```

When these arguments are missing, `role_adherence`
(no fallback — supply this key explicitly) and
`instruction_compliance` short-circuit to `score=None` with
`metadata["warning"] = "not_applicable"` rather than calling the judge.

## Picking a metric set

Each new metric belongs to the `Multi-Turn` dimension already configured
in `SingleAgentEvaluator` / `MultiAgentEvaluator` (default weight 0.10,
threshold 0.6). You don't need to register or weight new dimensions to use
them. To run a curated subset, pass a `metric_set` to `evaluate()` /
`batch_evaluate()`:

```python
from uaef.api import evaluate

result = evaluate(
    trace=trace,
    ground_truth=gt,
    metric_set={
        "Multi-Turn": [
            "context_retention",
            "coherence",
            "role_adherence",
        ],
        # Add per-turn relevance via the Response Quality dimension:
        "Response Quality": ["answer_relevance"],
    },
)
```

If you omit `metric_set` entirely, the evaluator runs every metric in the
registry that matches the configured dimensions — including all Multi-Turn
metrics. Each LLM-judge metric is one Bedrock call per evaluation, so
curate aggressively in tight loops.

## How AMACE prompts were adapted

- **Verbatim where possible** — the sentiment, agent-tone, uncanny-valley,
  containment, resolution, and optimum-turns prompts are direct ports of
  AMACE's `Libraries/Evaluation/src/metrics/*.py` prompts, lightly edited
  for the UAEF transcript format (`USER:` / `ASSISTANT:` rather than
  AMACE's `[Customer]` / `[Bot]`).
- **Newly authored** — `role_adherence` maps to a DeepEval library metric
  in AMACE; UAEF reimplements it as a native Bedrock judge following the
  same scoring formula stated in the AMACE README ("turns adhered / total
  turns").
- **Composite** — `holistic_llm_judge` ports AMACE's `LLMJudgeScore` rubric
  (completeness, relevancy, retention, sentiment, tone, satisfaction,
  containment, with extra weight on containment) into a single 0–1 judge.

`eval()` calls in the source repo's `resolution.py` / `optimum_turns.py`
were not preserved — the ports use UAEF's
`uaef.metrics.utils.extract_json_from_llm_response` for safe parsing.

## Cost and behavior notes

- **Each LLM-judge metric is one Bedrock call per evaluation.** Multi-Turn
  metrics are full-trace, so each runs once per conversation regardless of
  turn count. Per-turn metrics (e.g., `answer_relevance` in the Response
  Quality dimension) run once per turn and are averaged.
- **Sentiment / tone / naturalness ship in two flavors.** The full-trace
  versions (`sentiment`, `agent_tone`, `naturalness`) evaluate the
  conversational arc holistically — one judge call, with the prompt
  weighing later turns more heavily. The per-turn variants
  (`per_turn_sentiment`, `per_turn_agent_tone`, `per_turn_naturalness`)
  score each user→assistant exchange independently and aggregate by mean,
  exposing per-turn scores in `metadata["per_turn_scores"]` so consumers
  can pinpoint the specific turn that went bad. Per-turn variants make
  N judge calls per conversation but parallelize them via
  `asyncio.gather`, so wall-clock stays roughly flat as conversations
  grow.
- **`turn_efficiency` reads `trace.metadata["completeness_score"]`** that
  `conversation_completeness` writes during the same evaluation. The
  evaluator's dependency-aware scheduler runs `conversation_completeness`
  first; if you skip it, `turn_efficiency` falls back to assuming
  completeness 1.0.
- **All metrics handle Bedrock failures uniformly** — they return
  `MetricScore(score=None, ..., metadata={"warning": "api_error", ...})`
  rather than raising, so a single judge outage doesn't fail an entire
  evaluation.

## Code locations

| Component | File |
|-----------|------|
| Metric implementations (all 15) | `src/uaef/metrics/multi_turn.py` |
| Shared Bedrock helper | `src/uaef/metrics/multi_turn.py:_invoke_bedrock_judge` |
| AMACE-derived base | `src/uaef/metrics/multi_turn.py:_AMACEFullTraceMetric` |
| 1–5 normalizer | `src/uaef/metrics/multi_turn.py:_normalize_1_to_5` |
| Registry entries | `src/uaef/metrics/registry.py:_register_builtin_metrics` |
| Phase routing | `src/uaef/evaluation/base_evaluator.py:_run_multi_turn` (`FULL_TRACE_ONLY`) |
| Public exports | `src/uaef/metrics/__init__.py` |
