# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Worker's multi-turn conversation driving.

Two concerns:

  - ``_group_into_sessions`` — how ground-truth rows become conversations.
  - ``_invoke_agent`` — that continuity reaches the HTTP invoker, and is not
    passed to invokers that have no parameter for it.

No AWS and no live agent: the library invoker is monkeypatched, so these assert
the orchestration rather than the transport (which
``tests/test_http_agent_multi_turn.py`` covers at the httpx boundary).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from handlers.worker import _group_into_sessions, _invoke_agent  # noqa: E402


class TestGroupIntoSessions:
    def test_groups_by_session_and_orders_by_turn(self):
        rows = [
            {"session_id": "s1", "turn_id": 1},
            {"session_id": "s1", "turn_id": 3},
            {"session_id": "s2", "turn_id": 1},
            {"session_id": "s1", "turn_id": 2},
        ]
        got = _group_into_sessions(rows)
        assert [[r["turn_id"] for r in s] for s in got] == [[1, 2, 3], [1]]

    def test_preserves_first_seen_session_order(self):
        # Report order should follow the file, not sort keys.
        rows = [
            {"session_id": "zebra", "turn_id": 1},
            {"session_id": "alpha", "turn_id": 1},
        ]
        got = _group_into_sessions(rows)
        assert [s[0]["session_id"] for s in got] == ["zebra", "alpha"]

    def test_no_session_id_gives_one_turn_sessions(self):
        # This is what keeps single-turn datasets behaving exactly as before:
        # every row is its own conversation, so no history is accumulated.
        rows = [{"query": "a"}, {"query": "b"}, {"query": "c"}]
        got = _group_into_sessions(rows)
        assert [len(s) for s in got] == [1, 1, 1]

    @pytest.mark.parametrize("blank", [None, "", "   ", float("nan")])
    def test_blank_session_ids_are_not_grouped_together(self, blank):
        # Two rows with a missing id are two conversations, not one two-turn one.
        rows = [{"session_id": blank, "query": "a"}, {"session_id": blank, "query": "b"}]
        got = _group_into_sessions(rows)
        assert [len(s) for s in got] == [1, 1]

    def test_numeric_string_turn_ids_sort_numerically(self):
        # "10" must not sort before "9" as it would lexically.
        rows = [
            {"session_id": "s1", "turn_id": "9"},
            {"session_id": "s1", "turn_id": "10"},
            {"session_id": "s1", "turn_id": "1"},
        ]
        got = _group_into_sessions(rows)
        assert [r["turn_id"] for r in got[0]] == ["1", "9", "10"]

    def test_unparseable_turn_id_falls_back_to_file_order(self):
        # A blank turn shouldn't scramble the conversation; parseable turns lead.
        rows = [
            {"session_id": "s1", "turn_id": 2},
            {"session_id": "s1", "turn_id": None},
            {"session_id": "s1", "turn_id": 1},
        ]
        got = _group_into_sessions(rows)
        assert [r["turn_id"] for r in got[0]] == [1, 2, None]

    def test_session_ids_are_compared_as_strings(self):
        # 1 and "1" from a spreadsheet are the same conversation.
        rows = [{"session_id": 1, "turn_id": 1}, {"session_id": "1", "turn_id": 2}]
        got = _group_into_sessions(rows)
        assert len(got) == 1 and len(got[0]) == 2

    def test_empty_input(self):
        assert _group_into_sessions([]) == []

    def test_rows_are_not_mutated(self):
        rows = [{"session_id": "s1", "turn_id": 1}]
        _group_into_sessions(rows)
        assert rows[0] == {"session_id": "s1", "turn_id": 1}  # no _position leak


class TestTurnPayloadUserMessage:
    """The assembled conversation needs exactly one user turn per turn.

    LangGraph's "updates" mode emits only node outputs, so the question is missing
    and must be injected — without it a conversation is assistant-only and every
    full-trace metric judges an empty transcript. But some endpoints *do* echo the
    user message, and injecting unconditionally puts the question in twice, so every
    conversation-level judge reads each question double.
    """

    @staticmethod
    def _user_turns(payload):
        found = []
        for event in payload["stream_events"]:
            for node_data in event.values():
                for msg in (node_data.get("messages") or []):
                    kind = msg.get("type") if isinstance(msg, dict) else type(msg).__name__
                    if str(kind or "").lower() in {"humanmessage", "human", "user"}:
                        found.append(kind)
        return found

    def test_injects_when_agent_omits_the_question(self):
        from handlers.worker import _turn_payload

        raw = {"stream_events": [
            {"assistant": {"messages": [{"type": "AIMessage", "content": "a1"}]}},
        ]}
        payload = _turn_payload(raw, query="q1", session_id="s", turn_id=1, latency=0.1)
        assert len(self._user_turns(payload)) == 1
        assert list(payload["stream_events"][0]) == ["__human__"]

    def test_does_not_duplicate_when_agent_echoes_the_question(self):
        from handlers.worker import _turn_payload

        raw = {"stream_events": [
            {"__human__": {"messages": [{"type": "HumanMessage", "content": "q1"}]}},
            {"assistant": {"messages": [{"type": "AIMessage", "content": "a1"}]}},
        ]}
        payload = _turn_payload(raw, query="q1", session_id="s", turn_id=1, latency=0.1)
        assert len(self._user_turns(payload)) == 1

    @pytest.mark.parametrize("kind", ["HumanMessage", "human", "user"])
    def test_recognises_every_user_spelling(self, kind):
        from handlers.worker import _turn_payload

        raw = {"stream_events": [{"n": {"messages": [{"type": kind, "content": "q1"}]}}]}
        payload = _turn_payload(raw, query="q1", session_id="s", turn_id=1, latency=0.1)
        assert len(self._user_turns(payload)) == 1

    def test_handles_a_bare_message_not_in_a_list(self):
        from handlers.worker import _turn_payload

        raw = {"stream_events": [{"n": {"messages": {"type": "HumanMessage", "content": "q1"}}}]}
        payload = _turn_payload(raw, query="q1", session_id="s", turn_id=1, latency=0.1)
        # Detected, so nothing injected — the bare form is what LangGraph yields.
        assert not any("__human__" in e for e in payload["stream_events"])

    def test_empty_response_still_gets_the_question(self):
        from handlers.worker import _turn_payload

        payload = _turn_payload({}, query="q1", session_id="s", turn_id=1, latency=0.1)
        assert len(self._user_turns(payload)) == 1


class TestNoHistoryIsReplayed:
    """The endpoint owns the conversation, so a turn carries only its session id.

    Replaced TestStatefulModeDeclaration: there is no longer a mode to declare.
    Kept as a test rather than deleted outright because "we never send history" is
    the load-bearing property of the current design — if a `messages` field ever
    reappears in a request, a stateful agent silently sees every earlier turn twice.
    """

    def test_mode_helper_is_gone(self):
        import handlers.worker as worker

        assert not hasattr(worker, "_is_stateful")

    def test_invoker_rejects_a_messages_argument(self):
        """The library invoker no longer accepts history at all."""
        import inspect

        from uaef.adapters.invocation import invoke_http_agent

        assert "messages" not in inspect.signature(invoke_http_agent).parameters

    def test_no_messages_field_in_request_defaults(self):
        from uaef.adapters.invocation import DEFAULT_REQUEST_FIELDS

        assert "messages" not in DEFAULT_REQUEST_FIELDS


class TestInvokeAgentReachesTheRealInvoker:
    """Exercises the worker/library seam that the monkeypatched tests cannot.

    Every other test in this file replaces the library invoker, so a signature
    change in ``uaef.adapters`` passes here and fails in Lambda. That is not
    hypothetical: removing ``messages`` from the library while the worker still
    passed ``messages=None`` produced "All 9 agent invocations failed:
    invoke_http_agent_for_framework() got an unexpected keyword argument
    'messages'" in production, with a green local suite.

    So this one calls through the real invoker and stops only at httpx.
    """

    @pytest.fixture
    def capture(self, monkeypatch):
        import httpx

        calls: list = []

        class _FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"stream_events": [], "response": "ok"}

        class _FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, json=None, **kwargs):
                calls.append({"url": url, "json": json})
                return _FakeResponse()

        monkeypatch.setattr(httpx, "Client", _FakeClient)
        return calls

    def test_continuity_reaches_the_wire_and_history_does_not(self, capture):
        raw = _invoke_agent(
            "langgraph",
            {"endpoint": "https://agent/invoke"},
            "Book the cheapest hotel.",
            session_id="run-uuid",
            turn_id=4,
            payload_session_id="session_001",
        )

        body = capture[0]["json"]
        assert body == {
            "query": "Book the cheapest hotel.",
            "session_id": "run-uuid",
            "turn_id": 4,
        }
        # The dataset id is what grouping sees; the agent was told the run uuid.
        assert raw["session_id"] == "session_001"
        assert raw["turn_id"] == 4


class TestSessionEvalGating:
    """Session assembly requires an adapter that accepts a list of turn payloads.

    Only LangGraphAdapter does. Handing a list to any other adapter fails in
    whatever way that adapter happens to fail, which is not a useful error — so the
    session path is gated on the capability, while history replay (which happens
    entirely on our side of the wire) is not.
    """

    def test_only_langgraph_declares_session_support(self):
        from uaef.adapters import (
            AgentCoreAdapter, BedrockAgentAdapter, GenericJSONAdapter,
            LangChainAdapter, LangfuseAdapter, LangGraphAdapter, StrandsAdapter,
        )

        assert LangGraphAdapter().supports_session_transform() is True
        for cls in (LangChainAdapter, BedrockAgentAdapter, GenericJSONAdapter,
                    LangfuseAdapter, AgentCoreAdapter, StrandsAdapter):
            assert cls().supports_session_transform() is False, cls.__name__

    def test_base_default_is_opt_out(self):
        # A future adapter inherits False rather than silently claiming support.
        from uaef.adapters.base import BaseAdapter

        assert BaseAdapter.supports_session_transform(object()) is False

    def test_gate_expression_matches_intent(self):
        """has_conversations drives history replay; use_session_eval drives session
        assembly. The distinction is the point: a Strands agent with session-shaped
        data should still get a coherent conversation, scored per turn."""
        def gate(session_sizes, adapter_supports):
            has_conversations = any(n > 1 for n in session_sizes)
            return has_conversations, has_conversations and adapter_supports

        # multi-turn data, capable adapter -> both
        assert gate([6, 3], True) == (True, True)
        # multi-turn data, incapable adapter -> replay history, per-turn eval
        assert gate([6, 3], False) == (True, False)
        # single-turn data -> neither, regardless of capability
        assert gate([1, 1], True) == (False, False)
        assert gate([1, 1], False) == (False, False)


class TestInvokeAgentContinuity:
    def _capture(self, monkeypatch):
        calls = []

        def fake(endpoint, query, framework, **kwargs):
            calls.append({"endpoint": endpoint, "query": query,
                          "framework": framework, **kwargs})
            return {"stream_events": []}

        import uaef.adapters as adapters

        monkeypatch.setattr(adapters, "invoke_http_agent_for_framework", fake)
        return calls

    def test_http_path_forwards_continuity(self, monkeypatch):
        calls = self._capture(monkeypatch)
        _invoke_agent(
            "langgraph",
            {"endpoint": "https://agent/invoke"},
            "Book the cheapest hotel.",
            session_id="s1",
            turn_id=4,
        )
        assert calls[0]["session_id"] == "s1"
        assert calls[0]["turn_id"] == 4
        # History is never forwarded; the endpoint owns the conversation.
        assert "messages" not in calls[0]

    def test_single_turn_sends_no_continuity(self, monkeypatch):
        calls = self._capture(monkeypatch)
        _invoke_agent("langgraph", {"endpoint": "https://agent/invoke"}, "q")
        assert calls[0]["session_id"] is None
        assert calls[0]["turn_id"] is None
        assert "messages" not in calls[0]

    def test_field_map_and_payload_session_id_are_forwarded(self, monkeypatch):
        calls = self._capture(monkeypatch)
        _invoke_agent(
            "langgraph",
            {"endpoint": "https://agent/invoke", "field_map": {"query": "input"}},
            "q",
            session_id="run-uuid",
            payload_session_id="session_001",
        )
        assert calls[0]["field_map"] == {"query": "input"}
        assert calls[0]["session_id"] == "run-uuid"
        assert calls[0]["payload_session_id"] == "session_001"

    def test_absent_field_map_is_none_not_empty_dict(self, monkeypatch):
        # None means "use defaults"; an empty dict would too, but None keeps the
        # library's own default handling in one place.
        calls = self._capture(monkeypatch)
        _invoke_agent("langgraph", {"endpoint": "https://agent/invoke"}, "q")
        assert calls[0]["field_map"] is None

    def test_bedrock_path_ignores_continuity(self, monkeypatch):
        # These invokers take no continuity args; passing them must not TypeError.
        seen = {}

        def fake_bedrock(agent_id, alias_id, query, region):
            seen.update(agent_id=agent_id, query=query)
            return {"ok": True}

        import uaef.adapters as adapters

        monkeypatch.setattr(adapters, "invoke_bedrock_agent", fake_bedrock)
        _invoke_agent(
            "bedrock",
            {"agent_id": "A1", "alias_id": "L1"},
            "q",
            session_id="s1",
            turn_id=2,
        )
        assert seen == {"agent_id": "A1", "query": "q"}
