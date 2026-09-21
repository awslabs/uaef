# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-turn conversation support in the HTTP agent invokers.

Covers the wire format (what actually reaches the agent), the session/turn
stamping the LangGraph multi-turn path depends on, and history serialization.
The POST is intercepted at the ``httpx.Client`` boundary so these run offline.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from uaef.adapters.invocation import (
    invoke_http_agent,
    invoke_http_agent_for_framework,
)


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeClient:
    """Captures the POSTs an invoker makes, returning a canned payload."""

    calls: List[Dict[str, Any]] = []
    response: Dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        type(self).init_kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, endpoint, json=None):
        type(self).calls.append({"endpoint": endpoint, "json": json})
        return _FakeResponse(type(self).response)


@pytest.fixture
def capture(monkeypatch):
    """Patch httpx.Client so no network is used, and expose the captured calls."""
    import httpx

    _FakeClient.calls = []
    _FakeClient.response = {"stream_events": [], "response": "ok"}
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    # These tests use fake hosts (e.g. "agent") and never touch the network, so
    # opt out of the SSRF endpoint guard (which would otherwise try to resolve them).
    monkeypatch.setenv("UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS", "true")
    return _FakeClient


class TestSingleTurnUnchanged:
    def test_body_is_query_only(self, capture):
        invoke_http_agent("https://agent/invoke", "What time is my flight?")
        assert capture.calls[0]["json"] == {"query": "What time is my flight?"}

    def test_none_valued_continuity_fields_are_omitted(self, capture):
        # Explicit Nones must not appear as null keys — an agent branching on
        # `"turn_id" in body` would otherwise see a continuation on turn one.
        invoke_http_agent(
            "https://agent/invoke", "q", session_id=None, turn_id=None
        )
        assert capture.calls[0]["json"] == {"query": "q"}


class TestContinuityOnTheWire:
    def test_all_fields_sent(self, capture):
        invoke_http_agent(
            "https://agent/invoke",
            "Book the cheapest hotel.",
            session_id="session_001",
            turn_id=4,
        )
        body = capture.calls[0]["json"]
        assert body == {
            "query": "Book the cheapest hotel.",
            "session_id": "session_001",
            "turn_id": 4,
        }

    def test_turn_id_zero_is_sent_not_dropped(self, capture):
        # 0 is falsy; it must survive as an explicit value.
        invoke_http_agent("https://agent/invoke", "q", turn_id=0)
        assert capture.calls[0]["json"]["turn_id"] == 0

    def test_timeout_is_forwarded_to_the_client(self, capture):
        invoke_http_agent("https://agent/invoke", "q", timeout=45.0)
        assert capture.init_kwargs.get("timeout") == 45.0


class TestFrameworkWrapperStamping:
    """The adapter's multi-turn grouping requires session_id and turn_id."""

    def test_stamps_onto_stream_events_payload(self, capture):
        capture.response = {
            "stream_events": [{"assistant": {"messages": []}}],
            "session_id": "agents-own-id",       # must lose to the caller's
            "agent_node_name": "assistant",
        }
        raw = invoke_http_agent_for_framework(
            "https://agent/invoke", "q", "langgraph",
            session_id="session_001", turn_id=2,
        )
        assert raw["session_id"] == "session_001"
        assert raw["turn_id"] == 2
        # Untouched agent-supplied fields survive.
        assert raw["agent_node_name"] == "assistant"

    def test_stamps_onto_wrapped_payload(self, capture):
        # No stream_events -> the wrapping branch.
        capture.response = {"events": [{"agent": {"messages": []}}]}
        raw = invoke_http_agent_for_framework(
            "https://agent/invoke", "q", "langgraph",
            session_id="session_002", turn_id=3,
        )
        assert raw["session_id"] == "session_002"
        assert raw["turn_id"] == 3
        assert raw["stream_events"] == [{"agent": {"messages": []}}]

    def test_forwards_continuity_to_the_agent(self, capture):
        invoke_http_agent_for_framework(
            "https://agent/invoke", "q", "langgraph",
            session_id="s1", turn_id=2,
        )
        body = capture.calls[0]["json"]
        assert body["session_id"] == "s1"
        assert body["turn_id"] == 2
        assert "messages" not in body

    def test_single_turn_payload_has_no_turn_id(self, capture):
        capture.response = {"stream_events": []}
        raw = invoke_http_agent_for_framework("https://agent/invoke", "q", "langgraph")
        assert "turn_id" not in raw


class TestFieldMapping:
    """A customer endpoint names request fields its own way."""

    def test_renames_every_field(self, capture):
        invoke_http_agent(
            "https://agent/invoke", "hello",
            session_id="s1", turn_id=2,
            field_map={
                "query": "input",
                "session_id": "sessionId",
                "turn_id": "turnNumber",
            },
        )
        body = capture.calls[0]["json"]
        assert set(body) == {"input", "sessionId", "turnNumber"}
        assert body["input"] == "hello"
        assert body["sessionId"] == "s1"

    def test_partial_map_leaves_others_default(self):
        from uaef.adapters.invocation import DEFAULT_REQUEST_FIELDS

        assert DEFAULT_REQUEST_FIELDS["query"] == "query"

    def test_partial_map_on_the_wire(self, capture):
        invoke_http_agent(
            "https://agent/invoke", "hello", session_id="s1",
            field_map={"session_id": "sessionId"},
        )
        body = capture.calls[0]["json"]
        assert "query" in body and "sessionId" in body and "session_id" not in body


class TestHistoryIsNeverSent:
    """A turn carries only its query and identity; the endpoint owns the thread.

    This is the property the whole multi-turn design now rests on. If a `messages`
    field ever reappears in a request, a stateful agent silently receives every
    earlier turn twice and returns subtly worse answers with plausible scores —
    which is why it is asserted rather than left implicit.
    """

    def test_sends_session_without_history(self, capture):
        invoke_http_agent(
            "https://agent/invoke", "Book the cheapest hotel.",
            session_id="run-uuid", turn_id=4,
        )
        body = capture.calls[0]["json"]
        assert body["session_id"] == "run-uuid"
        assert body["turn_id"] == 4
        assert "messages" not in body

    def test_history_cannot_be_passed(self):
        """Removed from the signature, so a caller can't reintroduce a replay."""
        import inspect

        params = inspect.signature(invoke_http_agent).parameters
        assert "messages" not in params

    def test_no_messages_field_to_map(self):
        from uaef.adapters.invocation import DEFAULT_REQUEST_FIELDS

        assert "messages" not in DEFAULT_REQUEST_FIELDS

    def test_serialize_history_is_gone(self):
        import uaef.adapters as adapters

        assert not hasattr(adapters, "serialize_history")


class TestSessionIdSplit:
    """The id sent to the agent and the id used for grouping can differ.

    Against a stateful endpoint the sent id must be unique per run, or a second run
    of the same dataset continues the first run's conversation. Grouping and the UI
    label still want the dataset's stable id.
    """

    def test_payload_session_id_overrides_the_stamp(self, capture):
        capture.response = {"stream_events": []}
        raw = invoke_http_agent_for_framework(
            "https://agent/invoke", "q", "langgraph",
            session_id="d290f1ee-6c54-4b01-90e6-d701748f0851",  # per-run uuid
            turn_id=2,
            payload_session_id="session_001",                    # dataset id
        )
        # Agent was told the uuid...
        assert capture.calls[0]["json"]["session_id"] == "d290f1ee-6c54-4b01-90e6-d701748f0851"
        # ...while grouping/reporting sees the dataset id.
        assert raw["session_id"] == "session_001"
        assert raw["turn_id"] == 2

    def test_defaults_to_the_sent_id_when_not_split(self, capture):
        capture.response = {"stream_events": []}
        raw = invoke_http_agent_for_framework(
            "https://agent/invoke", "q", "langgraph",
            session_id="session_001", turn_id=1,
        )
        assert raw["session_id"] == "session_001"

    def test_split_applies_to_the_wrapped_branch_too(self, capture):
        capture.response = {"events": [{"agent": {"messages": []}}]}
        raw = invoke_http_agent_for_framework(
            "https://agent/invoke", "q", "langgraph",
            session_id="run-uuid", turn_id=3, payload_session_id="session_002",
        )
        assert raw["session_id"] == "session_002"


class TestCoerceMessageStillWorks:
    """serialize_history is gone, but the shape it produced still arrives.

    Agents echo `{"type", "content"}` dicts back inside their traces, so the
    adapter's rehydration path is still load-bearing even though nothing on our
    side constructs that shape any more.
    """

    def test_round_trips_serialized_message_dicts(self):
        from uaef.adapters.langgraph import LangGraphAdapter

        hist = [
            {"type": "HumanMessage", "content": "q1"},
            {"type": "AIMessage", "content": "a1"},
        ]
        coerced = [LangGraphAdapter._coerce_message(m) for m in hist]
        assert [type(c).__name__ for c in coerced] == ["HumanMessage", "AIMessage"]


class TestSsrfEndpointGuard:
    """The endpoint is caller-supplied and fetched server-side, so it must be
    validated against SSRF before any request is issued (unless explicitly
    opted out for trusted local use)."""

    def _fake_resolve(self, monkeypatch, ip: str):
        import uaef.adapters.invocation as inv

        def fake_getaddrinfo(host, port, *a, **k):
            return [(2, 1, 6, "", (ip, port))]

        monkeypatch.setattr(inv.socket, "getaddrinfo", fake_getaddrinfo)

    def test_blocks_cloud_metadata_ip(self, monkeypatch):
        monkeypatch.delenv("UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS", raising=False)
        self._fake_resolve(monkeypatch, "169.254.169.254")
        with pytest.raises(ValueError, match="non-public address"):
            invoke_http_agent("http://metadata.example/latest/meta-data/", "q")

    def test_blocks_loopback(self, monkeypatch):
        monkeypatch.delenv("UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS", raising=False)
        self._fake_resolve(monkeypatch, "127.0.0.1")
        with pytest.raises(ValueError, match="non-public address"):
            invoke_http_agent("http://localhost:8080/invoke", "q")

    def test_blocks_private_rfc1918(self, monkeypatch):
        monkeypatch.delenv("UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS", raising=False)
        self._fake_resolve(monkeypatch, "10.0.0.5")
        with pytest.raises(ValueError, match="non-public address"):
            invoke_http_agent("http://internal.svc/invoke", "q")

    def test_rejects_non_http_scheme(self, monkeypatch):
        monkeypatch.delenv("UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS", raising=False)
        with pytest.raises(ValueError, match="http or https"):
            invoke_http_agent("file:///etc/passwd", "q")

    def test_allows_public_endpoint(self, monkeypatch, capture):
        # capture sets the opt-out; clear it so we exercise the real guard, then
        # resolve to a public IP -> the request should go through.
        monkeypatch.delenv("UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS", raising=False)
        self._fake_resolve(monkeypatch, "8.8.8.8")
        invoke_http_agent("https://agent.example.com/invoke", "q")
        assert capture.calls[0]["endpoint"] == "https://agent.example.com/invoke"

    def test_opt_out_allows_localhost(self, monkeypatch, capture):
        # capture already sets UAEF_ALLOW_PRIVATE_AGENT_ENDPOINTS=true.
        invoke_http_agent("http://localhost:8080/invoke", "q")
        assert capture.calls[0]["endpoint"] == "http://localhost:8080/invoke"
