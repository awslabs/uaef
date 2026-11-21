# Framework Adapters — writing `agent_adapter.py` per stack

`agent_adapter.py` is the only file that knows the target agent's framework. Pick the section below
matching the agent under test, adapt the skeleton, and confirm any marked placeholder against the
installed package version before shipping — do not guess an API signature that isn't shown here.

## Strands

Trajectory comes from the message list (tool-use/tool-result pairs); traces come from OTel spans if
the deployment exports them, otherwise from `EventLoopMetrics`/`ToolMetrics` on the agent instance.

Span names to look for: `invoke_agent` (top-level), `execute_event_loop_cycle` (per loop cycle),
`chat` (per model call), `execute_tool` (per tool call). Useful attributes: `gen_ai.usage.*_tokens`,
`gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.server.time_to_first_token`. If OTel export isn't
wired up in the target deployment, fall back to `agent.event_loop_metrics.get_summary()` for
usage/latency and reconstruct `trajectory` from the returned message list's `toolUse` blocks.

`BeforeToolCallEvent` / `AfterToolCallEvent` hooks are the injection point both for recording tool
calls precisely (arguments, results, timing) and — separately, for chaos/red-team testing — for
forcing a tool to fail, delay, or return a corrupted result without touching the agent's own code.

```python
import time
from datetime import datetime, timezone

from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent
from your_agent_package import build_agent  # wherever the target agent is constructed

ADAPTER_VERSION = "strands-v1"

def invoke(case) -> AgentRun:
    agent = build_agent()  # construct lazily, inside the function — not at import time
    trajectory: list[str] = []
    tool_calls: list[dict] = []

    def _before_tool(event: BeforeToolCallEvent) -> None:
        trajectory.append(event.tool_use["name"])
        tool_calls.append({
            "name": event.tool_use["name"],
            "arguments": event.tool_use.get("input", {}),
            "result": None,
            "timestamp": datetime.now(timezone.utc),
            "execution_time": None,
            "error": None,
        })

    def _after_tool(event: AfterToolCallEvent) -> None:
        if tool_calls:
            tool_calls[-1]["result"] = getattr(event, "result", None)

    # Registration is by EVENT TYPE, not by string name:
    #   add_callback(event_type, callback, *, order=HookOrder.DEFAULT)
    # Passing event_type=None infers the type from the callback's type hint.
    agent.hooks.add_callback(BeforeToolCallEvent, _before_tool)
    agent.hooks.add_callback(AfterToolCallEvent, _after_tool)

    started = time.perf_counter()
    result = agent(case.input)
    elapsed = time.perf_counter() - started
    metrics = agent.event_loop_metrics.get_summary()

    return AgentRun(
        output=str(result),
        trajectory=trajectory,
        tool_calls=tool_calls,
        spans=[],  # populate from OTel exporter if configured
        state={},  # read back from whatever system of record the agent's tools mutate
        usage={"total_tokens": metrics.get("total_tokens")},
        latency=elapsed,  # measure it yourself; don't rely on a metrics field being present
        error=None,
    )
```

Verified against the installed `strands-py`: `HookRegistry.add_callback(event_type, callback, *,
order=HookOrder.DEFAULT)` — registration is by event **type**, and `add_hook(provider)` is the bulk
alternative for a `HookProvider`. Check `event.tool_use` / `event.result` attribute names against your
installed version, as the event payload shape moves faster than the registration API.

## Bedrock AgentCore

Invoking a deployed agent endpoint means calling the AgentCore runtime's invoke API with a session
id, not importing the agent's Python code directly — the adapter is closer to an HTTP client.

- Maintain one `session_id` per `case.session_id` (or a fresh one per case if unset) so multi-turn
  cases replay against continuous agent state.
- Traces surface through AgentCore's own observability integration (CloudWatch / OTel export) rather
  than being returned inline with the invoke response in every deployment — confirm what the specific
  target deployment exposes (inline trace vs. separate trace fetch by session/request id) before
  assuming `spans` can be populated synchronously.
- Placeholder — the exact invoke call signature (boto3 client name, method, and request/response
  shape) depends on the AgentCore runtime API version in use; verify it against the installed SDK
  rather than assuming a signature here.

```python
import boto3  # confirm the exact client name/method against the installed AgentCore SDK

ADAPTER_VERSION = "agentcore-v1"

def invoke(case) -> AgentRun:
    client = boto3.client("bedrock-agentcore")  # placeholder — verify service/client name
    session_id = case.session_id or f"eval-{case.id}"
    response = client.invoke_agent_runtime(  # placeholder — verify method name and params
        sessionId=session_id,
        payload=case.input,
    )
    return AgentRun(
        output=response.get("output", ""),
        trajectory=[],  # populate from the runtime's trace/tool-call log if the response includes it
        tool_calls=[],  # same source as trajectory; empty means tool-level evaluators must be skipped, not silently scored
        spans=[],
        state={},
        usage={},
        latency=0.0,
        error=None,
    )
```

## LangGraph / LangChain

Trajectory and trace both come from the graph's state object and its callback handler — attach a
callback before invoking and read the accumulated tool-call events, rather than parsing the final
state alone (the final state won't show intermediate tool calls that were later overwritten).

```python
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

ADAPTER_VERSION = "langgraph-v1"

class _ToolCallRecorder(BaseCallbackHandler):
    def __init__(self):
        self.calls: list[str] = []
        self.tool_calls: list[dict] = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown")
        self.calls.append(name)
        self.tool_calls.append({
            "name": name,
            "arguments": input_str,
            "result": None,
            "timestamp": datetime.now(timezone.utc),
            "execution_time": None,
            "error": None,
        })

    def on_tool_end(self, output, **kwargs):
        if self.tool_calls:
            self.tool_calls[-1]["result"] = output

def invoke(case) -> AgentRun:
    graph = build_graph()  # construct lazily
    recorder = _ToolCallRecorder()
    result = graph.invoke({"input": case.input}, config={"callbacks": [recorder]})

    return AgentRun(
        output=result.get("output", str(result)),
        trajectory=recorder.calls,
        tool_calls=recorder.tool_calls,
        spans=[],
        state=result.get("state", {}),
        usage={},
        latency=0.0,
        error=None,
    )
```

## Plain Python / custom agent

No framework to lean on — the minimum viable adapter wraps the entry point and records tool calls
itself, either with a decorator around each tool function or a logging shim around whatever dispatch
mechanism the agent uses.

```python
import time
from datetime import datetime, timezone

ADAPTER_VERSION = "custom-v1"

_tool_calls: list[dict] = []

def record_tool(name):
    """Wrap each of the agent's tool functions to capture name, arguments, result and timing."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            entry = {
                "name": name,
                "arguments": kwargs or {"args": args},
                "result": None,
                "timestamp": datetime.now(timezone.utc),
                "execution_time": None,
                "error": None,
            }
            _tool_calls.append(entry)
            started = time.perf_counter()
            try:
                entry["result"] = fn(*args, **kwargs)
                return entry["result"]
            except Exception as exc:
                entry["error"] = str(exc)
                raise
            finally:
                entry["execution_time"] = time.perf_counter() - started
        return wrapper
    return decorator

def invoke(case) -> AgentRun:
    _tool_calls.clear()
    start = time.monotonic()
    try:
        output = run_agent(case.input)  # the agent's actual entry point, with its tools wrapped by record_tool
        error = None
    except Exception as exc:
        output, error = "", str(exc)

    return AgentRun(
        output=output,
        trajectory=[tc["name"] for tc in _tool_calls],
        tool_calls=list(_tool_calls),
        spans=[],
        state={},
        usage={},
        latency=time.monotonic() - start,
        error=error,
    )
```

## UAEF (`agenticevaluationframework`)

UAEF is the scoring engine, not an optional extra. It wants a canonical `AgentTrace`, not `AgentRun`
directly — build one from the `AgentRun` fields and hand it to `evaluate()` to reuse UAEF's 36-metric
registry, its Bedrock judge, and its DynamoDB/S3 persistence. The evaluators in `evaluators/` exist for
checks UAEF has no metric for, not to duplicate the ones it has.

API surface: `uaef.api.evaluate(trace, ground_truth=None, persist=False, metrics=None, ...)` and
`uaef.api.batch_evaluate(traces=..., ground_truths=..., max_workers=..., persist=..., experiment_name=...)`;
canonical models `uaef.models.AgentTrace`, `Message`, `MessageRole`, `ToolCall`, `GroundTruth`; adapters
via `uaef.adapters.get_adapter("strands" | "langgraph" | "langchain" | "bedrock" | "langfuse" | "agentcore")`
exposing `.transform_to_canonical(raw_output)`, and `uaef.adapters.GenericJSONAdapter(schema_mapping=...)`
for anything else.

**Four field constraints that will bite you.** All verified against `src/uaef/models/` — these are the
things a plausible-looking bridge gets wrong and only finds out at runtime:

| Field | Constraint | Consequence if ignored |
|---|---|---|
| `AgentTrace.trace_id` | typed `UUID`, not `str` | A readable id like `"refund-012"` raises `ValidationError: Input should be a valid UUID`. Derive a stable UUID from the case id instead — don't use `uuid4()`, or the same case gets a new id every run and nothing diffs. |
| `Message.timestamp` | **required**, no default | Omitting it fails validation, even though every other `Message` field looks optional. |
| `ToolCall.timestamp` | **required**, no default | Same. Fall back to "now" only when the trajectory genuinely carries no timing. |
| `GroundTruth.expected_output` | `Optional[str]` — a **string**, not a list | The case format allows `expected_output` to be a list meaning "any of these is acceptable". UAEF has no equivalent, so pick one representative string for UAEF and keep the any-of check local. |

Also note `GroundTruth.expected_tool_calls` is `List[ToolCall]`, not a list of names — an
`expected_trajectory` of bare tool names has to be wrapped before UAEF will accept it.

```python
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from uaef.api import evaluate
from uaef.models import AgentTrace, GroundTruth, Message, MessageRole, ToolCall

ADAPTER_VERSION = "uaef-bridge-v1"


def _trace_id(case_id: str, run_index: int):
    """Stable UUID per (case, run) so two runs of the same case are comparable."""
    return uuid5(NAMESPACE_URL, f"{ADAPTER_VERSION}/{case_id}/{run_index}")


def to_uaef_trace(run: AgentRun, case, run_index: int = 0) -> AgentTrace:
    now = datetime.now(timezone.utc)
    return AgentTrace(
        trace_id=_trace_id(case.id, run_index),
        framework=run.framework,          # optional, but it shows up in reports
        messages=[
            Message(role=MessageRole.USER, content=case.input, timestamp=run.started_at or now),
            Message(role=MessageRole.ASSISTANT, content=run.output, timestamp=run.ended_at or now),
        ],
        tool_calls=[
            ToolCall(
                name=tc["name"],
                arguments=tc.get("arguments", {}),
                result=tc.get("result"),
                timestamp=tc.get("timestamp") or now,   # required — no default
                execution_time=tc.get("execution_time"),
                error=tc.get("error"),
            )
            for tc in run.tool_calls
        ],
        latency=run.latency,
        cost=run.cost,
        session_id=case.session_id or case.id,
        metadata={"case_category": case.category, "provenance": case.provenance},
    )


def _expected_output(case) -> str | None:
    """UAEF takes a single string; the case format allows a list of acceptable answers."""
    if not case.expected_output:
        return None
    if isinstance(case.expected_output, list):
        return case.expected_output[0]      # any-of stays a local deterministic check
    return case.expected_output


def score_with_uaef(run: AgentRun, case, run_index: int = 0):
    trace = to_uaef_trace(run, case, run_index)
    expected = _expected_output(case)
    now = datetime.now(timezone.utc)
    ground_truth = None
    if expected or case.expected_trajectory:
        ground_truth = GroundTruth(
            expected_output=expected,
            # a list of names is not enough — UAEF wants ToolCall objects
            expected_tool_calls=[
                ToolCall(name=name, timestamp=now)
                for name in (case.expected_trajectory or [])
            ],
        )
    return evaluate(trace=trace, ground_truth=ground_truth, persist=False)
```

Call `score_with_uaef` from `evaluators/judges.py` or `evaluators/trajectory.py`, and map UAEF's
returned `dimension_results`/`metric_score` objects onto this harness's `EvaluationOutput` shape
(`score`, `passed`, `reason`, `label`, `level`) rather than passing UAEF's result object through
unchanged — the report and the scorecard diff both depend on that one shape.

For a real Phase 4 run, prefer `batch_evaluate(..., persist=True, experiment_name=...)` over
per-case `evaluate()`: it parallelises, and it writes one experiment row that
`uaef.experiments`/`uaef.reporting.regression` can then diff against the next run. Persisting is how
the suite becomes a trend rather than a one-off number.

## Summary table

| Stack | Trajectory source | Span/trace source | Notes |
|---|---|---|---|
| Strands | Message list (`toolUse` blocks) or `BeforeToolCallEvent`/`AfterToolCallEvent` hooks | OTel spans (`invoke_agent`, `execute_event_loop_cycle`, `chat`, `execute_tool`) if exported; else `EventLoopMetrics`/`ToolMetrics` | Hooks double as the chaos-injection point |
| Bedrock AgentCore | Runtime invoke response or session trace fetch | CloudWatch/OTel via the runtime's own observability integration | Invoke signature varies by runtime API version — verify against the installed SDK (marked placeholder above) |
| LangGraph / LangChain | Callback handler (`on_tool_start`/`on_tool_end`) | Same callback handler, or LangSmith if configured | Read final graph state for `state`, not just trajectory |
| Plain Python / custom | Manual decorator or logging shim around tool dispatch | None by default — add your own timing/logging | Cheapest to write, least observability for free |
| UAEF (scoring engine) | Built from the harness's own `AgentRun`/trajectory, converted to `AgentTrace` | Same, via `AgentTrace.messages`/`tool_calls` | Always on. Watch `trace_id` (UUID), the required `timestamp` fields, and `expected_output` being a single string |
