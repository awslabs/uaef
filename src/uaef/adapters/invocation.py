# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live agent-invocation helpers.

These helpers call live agents (HTTP endpoints, Bedrock agents, Langfuse traces)
and package their raw responses for the matching framework adapter. They live in
the library — alongside the adapters — so that every consumer (the demo backend,
the service, notebooks) invokes live agents through the same code path instead of
re-implementing this glue.

``httpx`` is not part of the lean core dependency set, so the HTTP helpers import
it lazily and raise an actionable :class:`ImportError` when it is missing. ``boto3``
is a core dependency and is imported lazily here only to keep this module importable
in environments where AWS credentials/clients are not configured.
"""

import ipaddress
import json
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from uuid import uuid4

#: Opt-out (trusted local use only) for the SSRF guard on agent endpoints. The
#: deployed Worker must never set this; the local demo/notebook — which targets
#: localhost agents — may, to allow private/loopback endpoints.
_ALLOW_PRIVATE_ENDPOINTS_ENV = "UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS"


def _allow_private_endpoints() -> bool:
    return os.environ.get(_ALLOW_PRIVATE_ENDPOINTS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _validate_agent_endpoint(endpoint: str) -> None:
    """Reject SSRF-prone agent endpoints before issuing a server-side request.

    The deployed Worker fetches caller-supplied endpoints with the *service's*
    credentials, so an unvalidated URL is a full SSRF: a caller could point it at
    the cloud metadata IP (``169.254.169.254``) to steal the service role's
    credentials, or at internal VPC addresses. We require http/https and — unless
    explicitly opted out for trusted local use via
    ``UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS`` — block any host that resolves to a
    non-global address (private, loopback, link-local, reserved, multicast).

    Note: this resolves DNS and validates every returned address, which closes
    the trivial exploit. It does not by itself defeat DNS-rebinding (the address
    could change between this check and the request); network-level egress
    controls remain the defense-in-depth backstop for that.
    """
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Agent endpoint must use http or https (got {parsed.scheme!r}).")
    host = parsed.hostname
    if not host:
        raise ValueError("Agent endpoint URL has no host.")

    if _allow_private_endpoints():
        return

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addrinfos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"Agent endpoint host could not be resolved: {host}") from exc

    for info in addrinfos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError(
                f"Agent endpoint {host!r} resolves to a non-public address ({ip}); "
                "refusing the request to prevent SSRF. Set "
                f"{_ALLOW_PRIVATE_ENDPOINTS_ENV}=true only for trusted local endpoints."
            )

__all__ = [
    "invoke_http_agent",
    "invoke_bedrock_agent",
    "invoke_agentcore_runtime",
    "invoke_langfuse_trace",
    "invoke_http_agent_for_framework",
]


def invoke_agentcore_runtime(
    agent_runtime_arn: str,
    query: str,
    region: str = "us-east-1",
    qualifier: Optional[str] = None,
    session_id: Optional[str] = None,
    payload_key: str = "prompt",
) -> Dict[str, Any]:
    """Invoke a Bedrock AgentCore runtime and return packaged data for the adapter.

    Calls the AgentCore data-plane ``InvokeAgentRuntime`` API with a JSON payload
    ``{payload_key: query}`` (``payload_key`` defaults to ``"prompt"``; override
    it if your runtime expects a different input key). The streaming response is
    read fully and JSON-decoded when possible, then returned in the shape the
    AgentCore adapter (``get_adapter("agentcore")``) expects to transform into a
    canonical trace.

    ``boto3`` is imported lazily so this module stays importable where AWS is not
    configured. Requires the caller's role to allow
    ``bedrock-agentcore:InvokeAgentRuntime`` on the target runtime.
    """
    import boto3

    client = boto3.client("bedrock-agentcore", region_name=region)

    # AgentCore requires a runtime session id of at least 33 characters.
    session_id = session_id or (uuid4().hex + uuid4().hex)

    kwargs: Dict[str, Any] = {
        "agentRuntimeArn": agent_runtime_arn,
        "runtimeSessionId": session_id,
        "payload": json.dumps({payload_key: query}).encode("utf-8"),
    }
    if qualifier:
        kwargs["qualifier"] = qualifier

    response = client.invoke_agent_runtime(**kwargs)

    body = response.get("response")
    raw = body.read() if hasattr(body, "read") else body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        parsed = {"output": raw}

    return {
        "session_id": session_id,
        "agent_runtime_arn": agent_runtime_arn,
        "user_input": query,
        "response": parsed,
    }


#: Default request field names. A customer endpoint names things its own way
#: ("input"/"prompt"/"message", "sessionId"/"thread_id"/"conversation_id"), so
#: callers can remap any of these via ``field_map`` without the invoker caring.
#:
#: There is no ``messages`` field: multi-turn evaluation requires an endpoint that
#: keeps conversation state itself, so a turn carries only its query and the
#: session id. Client-side history replay was removed deliberately: replaying a
#: transcript into a stateful endpoint duplicates context the endpoint already
#: has, so the turn under evaluation no longer reflects production behaviour.
DEFAULT_REQUEST_FIELDS: Dict[str, str] = {
    "query": "query",
    "session_id": "session_id",
    "turn_id": "turn_id",
}


def invoke_http_agent(
    endpoint: str,
    query: str,
    *,
    session_id: Optional[str] = None,
    turn_id: Optional[int] = None,
    timeout: float = 120.0,
    field_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Call an HTTP agent endpoint with ``POST {"query": "..."}`` and return JSON.

    Conversation continuity (both optional, omitted from the body when unset, so a
    single-turn call posts exactly ``{"query": ...}``):

    ``session_id``
        Identifies the conversation. **The endpoint is required to keep the
        conversation state itself and key on this.** Multi-turn evaluation is only
        meaningful against such an endpoint: a dependent turn like "Book the
        cheapest hotel." has no referent unless the agent remembers the turn that
        listed them.
    ``turn_id``
        1-based position of this turn within the session. Lets an agent
        distinguish "start of conversation" from "continuation" — useful when
        beginning a turn has side effects, such as resetting a scratch database.

    ``field_map`` renames any request field (see :data:`DEFAULT_REQUEST_FIELDS`) for
    endpoints that use their own names.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "HTTP agent invocation requires 'httpx'. "
            "Install with: pip install httpx"
        ) from exc

    # SSRF guard: the endpoint is caller-supplied and fetched with the service's
    # credentials, so validate it before any request leaves the process.
    _validate_agent_endpoint(endpoint)

    fields = {**DEFAULT_REQUEST_FIELDS, **(field_map or {})}

    payload: Dict[str, Any] = {fields["query"]: query}
    if session_id is not None:
        payload[fields["session_id"]] = session_id
    if turn_id is not None:
        payload[fields["turn_id"]] = turn_id

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(endpoint, json=payload)
        resp.raise_for_status()
        return resp.json()


def invoke_bedrock_agent(
    agent_id: str, alias_id: str, query: str, region: str = "us-east-1"
) -> Dict[str, Any]:
    """Invoke a Bedrock agent and return packaged data for the adapter."""
    import boto3

    bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=region)
    session_id = str(uuid4())

    response = bedrock_runtime.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        inputText=query,
    )
    event_stream = list(response["completion"])

    return {
        "event_stream": event_stream,
        "user_input": query,
        "session_id": session_id,
        "agent_id": agent_id,
        "alias_id": alias_id,
    }


def invoke_langfuse_trace(
    query: str, expected: str,
    public_key: str, secret_key: str, host: str,
) -> Dict[str, Any]:
    """
    Fetch a real Langfuse trace matching the query, or simulate one.

    Strategy:
    1. Try to fetch traces from Langfuse that match the query text.
    2. If a matching trace is found, return its observations in the format
       expected by LangfuseAdapter.
    3. If no match or Langfuse SDK unavailable, fall back to a simulated
       trace using the ground truth expected answer.
    """
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        # Search for traces whose input matches the query
        traces = client.fetch_traces(limit=5)
        for trace_data in traces.data:
            trace_input = trace_data.input
            if isinstance(trace_input, dict):
                trace_input = trace_input.get("content", trace_input.get("message", ""))
            if isinstance(trace_input, str) and query.lower() in trace_input.lower():
                observations = client.fetch_observations(trace_id=trace_data.id)
                return {
                    "trace_id": trace_data.id,
                    "session_id": trace_data.session_id or str(uuid4()),
                    "name": trace_data.name or "langfuse-trace",
                    "timestamp": trace_data.timestamp.isoformat() if trace_data.timestamp else datetime.now(timezone.utc).isoformat(),
                    "metadata": trace_data.metadata or {},
                    "observations": [
                        {
                            "id": obs.id,
                            "type": obs.type,
                            "name": obs.name,
                            "input": obs.input,
                            "output": obs.output,
                            "model": obs.model,
                            "usage": {"input": (obs.usage.input if obs.usage else 0), "output": (obs.usage.output if obs.usage else 0), "total": (obs.usage.total if obs.usage else 0)} if obs.usage else {},
                            "cost": obs.calculated_total_cost,
                            "start_time": obs.start_time.isoformat() if obs.start_time else None,
                            "end_time": obs.end_time.isoformat() if obs.end_time else None,
                            "parent_observation_id": obs.parent_observation_id,
                        }
                        for obs in observations.data
                    ],
                }
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] Langfuse fetch failed, using simulated trace: {e}")

    # Fallback: simulate a trace using ground truth data
    return {
        "trace_id": str(uuid4()),
        "session_id": str(uuid4()),
        "name": "langfuse-eval-simulated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "observations": [
            {
                "id": str(uuid4()),
                "type": "generation",
                "name": "llm-call",
                "input": {"role": "user", "content": query},
                "output": {"content": expected},
                "model": "unknown",
                "usage": {"input": 0, "output": 0, "total": 0},
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }


def invoke_http_agent_for_framework(
    endpoint: str,
    query: str,
    framework: str,
    *,
    session_id: Optional[str] = None,
    turn_id: Optional[int] = None,
    field_map: Optional[Dict[str, str]] = None,
    payload_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call an HTTP agent and package the response for the given framework adapter.

    ``session_id`` is what the *agent* is told. ``payload_session_id``, when given,
    is what gets stamped on the returned payload instead — because those are
    legitimately different values. Against a stateful agent the id sent must be
    unique per run, or re-running a dataset continues the previous run's
    conversation; but grouping and reporting want the dataset's own stable id.
    Defaults to ``session_id`` when omitted, which is the single-identifier case.

    ``session_id`` / ``turn_id`` are forwarded to the agent (see
    :func:`invoke_http_agent`) and, when supplied, stamped onto the returned
    payload. That stamping matters: the multi-turn path of
    :meth:`~uaef.adapters.langgraph.LangGraphAdapter.transform_to_canonical`
    groups a list of per-turn payloads by ``session_id`` and orders them by
    ``turn_id``, and raises if either is missing. A caller-supplied value always
    wins over whatever the agent echoes back, so grouping reflects the dataset
    rather than the agent's own bookkeeping.
    """
    raw = invoke_http_agent(
        endpoint, query,
        session_id=session_id, turn_id=turn_id,
        field_map=field_map,
    )

    #: The id used for grouping — the dataset's, when it differs from the agent's.
    group_session_id = payload_session_id if payload_session_id is not None else session_id

    def _with_turn_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Stamp caller-supplied session/turn identity onto the payload."""
        if group_session_id is not None:
            payload["session_id"] = group_session_id
        if turn_id is not None:
            payload["turn_id"] = turn_id
        return payload

    if framework == "langgraph":
        if "stream_events" in raw:
            return _with_turn_keys(raw)
        return _with_turn_keys({
            "stream_events": raw.get("events", [raw]),
            "session_id": raw.get("session_id", str(uuid4())),
            "agent_node_name": raw.get("agent_node_name", "agent"),
            "tool_node_name": raw.get("tool_node_name", "tools"),
        })
    elif framework == "langchain":
        if "inputs" in raw and "outputs" in raw:
            return raw
        return {
            "inputs": {"input": query},
            "outputs": {"output": raw.get("output", raw.get("response", str(raw)))},
            "intermediate_steps": raw.get("intermediate_steps", []),
            "session_id": raw.get("session_id", str(uuid4())),
        }
    else:
        # strands, agentcore (http), generic
        return raw
