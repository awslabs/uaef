# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangGraph adapter: single-agent, serialized messages,
multi-agent, and multi-turn session input."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    messages_to_dict,
)

from uaef.adapters.base import AdapterTransformationError
from uaef.adapters.langgraph import LangGraphAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import MessageRole
from uaef.models.multi_agent_trace import MultiAgentTrace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return LangGraphAdapter()


def _usage(input_tokens, output_tokens):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _tool_events():
    """Agent calls a tool, the tool node answers, the agent replies."""
    return [
        {"agent": {"messages": [
            HumanMessage("Weather in Paris?"),
            AIMessage(
                "",
                tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}],
                usage_metadata=_usage(10, 5),
            ),
        ]}},
        {"tools": {"messages": [ToolMessage("Sunny", tool_call_id="c1")]}},
        {"agent": {"messages": [AIMessage("It is sunny.", usage_metadata=_usage(20, 7))]}},
    ]


def _multi_agent_events(serialized=False):
    events = [
        {"researcher": {"messages": [
            HumanMessage("Write about bees"),
            AIMessage("Bees pollinate.", usage_metadata=_usage(10, 4)),
        ]}},
        {"writer": {"messages": [AIMessage("Draft about bees.", usage_metadata=_usage(30, 8))]}},
    ]
    if serialized:
        for event in events:
            for value in event.values():
                value["messages"] = messages_to_dict(value["messages"])
    return events


# ---------------------------------------------------------------------------
# Single agent
# ---------------------------------------------------------------------------

class TestSingleAgent:
    def test_returns_agent_trace(self, adapter):
        trace = adapter.transform_to_canonical({"stream_events": _tool_events()})
        assert isinstance(trace, AgentTrace)
        assert trace.framework == "langgraph"

    def test_messages_extracted(self, adapter):
        trace = adapter.transform_to_canonical({"stream_events": _tool_events()})
        assert [(m.role, m.content) for m in trace.messages] == [
            (MessageRole.USER, "Weather in Paris?"),
            (MessageRole.ASSISTANT, ""),
            (MessageRole.ASSISTANT, "It is sunny."),
        ]

    def test_tool_call_with_args_and_result(self, adapter):
        trace = adapter.transform_to_canonical({"stream_events": _tool_events()})
        assert len(trace.tool_calls) == 1
        call = trace.tool_calls[0]
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Paris"}
        assert call.result is not None

    def test_tokens_summed_across_agent_turns(self, adapter):
        trace = adapter.transform_to_canonical({"stream_events": _tool_events()})
        assert trace.input_tokens == 30
        assert trace.output_tokens == 12

    def test_no_usage_means_no_tokens(self, adapter):
        events = [{"agent": {"messages": [HumanMessage("hi"), AIMessage("hello")]}}]
        trace = adapter.transform_to_canonical({"stream_events": events})
        assert trace.input_tokens is None
        assert trace.output_tokens is None

    def test_latency_and_model_metadata(self, adapter):
        trace = adapter.transform_to_canonical(
            {"stream_events": _tool_events(), "latency": 1.5, "model": "m-1"}
        )
        assert trace.latency == 1.5
        assert trace.metadata["model"] == "m-1"

    def test_custom_node_names_are_inferred(self, adapter):
        events = [
            {"assistant": {"messages": [
                HumanMessage("q"),
                AIMessage("", tool_calls=[{"name": "lookup", "args": {}, "id": "c1"}]),
            ]}},
            {"executor": {"messages": [ToolMessage("found", tool_call_id="c1")]}},
            {"assistant": {"messages": [AIMessage("done")]}},
        ]
        trace = adapter.transform_to_canonical({"stream_events": events})
        assert isinstance(trace, AgentTrace)
        assert [m.content for m in trace.messages] == ["q", "", "done"]
        assert trace.tool_calls[0].name == "lookup"
        assert trace.tool_calls[0].result is not None

    def test_explicit_agent_node_name_wins(self, adapter):
        trace = adapter.transform_to_canonical(
            {"stream_events": _tool_events(), "agent_node_name": "agent", "tool_node_name": "tools"}
        )
        assert len(trace.tool_calls) == 1

    def test_missing_stream_events_raises(self, adapter):
        with pytest.raises(AdapterTransformationError):
            adapter.transform_to_canonical({"latency": 1.0})


# ---------------------------------------------------------------------------
# Serialized (HTTP-served) messages
# ---------------------------------------------------------------------------

class TestSerializedMessages:
    def _class_name_form(self):
        return [
            {"agent": {"messages": [
                {"type": "HumanMessage", "content": "Weather in Paris?"},
                {
                    "type": "AIMessage",
                    "content": "",
                    "tool_calls": [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}],
                    "usage_metadata": _usage(10, 5),
                },
            ]}},
            {"tools": {"messages": [
                {"type": "ToolMessage", "content": "Sunny", "tool_call_id": "c1"},
            ]}},
            {"agent": {"messages": [
                {"type": "AIMessage", "content": "It is sunny.", "usage_metadata": _usage(20, 7)},
            ]}},
        ]

    def _native_form(self):
        events = _tool_events()
        for event in events:
            for value in event.values():
                value["messages"] = messages_to_dict(value["messages"])
        return events

    @pytest.mark.parametrize("form", ["_class_name_form", "_native_form"])
    def test_messages_and_tool_calls_match_live(self, adapter, form):
        live = adapter.transform_to_canonical({"stream_events": _tool_events()})
        serialized = adapter.transform_to_canonical({"stream_events": getattr(self, form)()})
        assert [(m.role, m.content) for m in serialized.messages] == [
            (m.role, m.content) for m in live.messages
        ]
        assert [(c.name, c.arguments) for c in serialized.tool_calls] == [
            (c.name, c.arguments) for c in live.tool_calls
        ]

    @pytest.mark.parametrize("form", ["_class_name_form", "_native_form"])
    def test_tokens_from_usage_metadata(self, adapter, form):
        trace = adapter.transform_to_canonical({"stream_events": getattr(self, form)()})
        assert trace.input_tokens == 30
        assert trace.output_tokens == 12


# ---------------------------------------------------------------------------
# Multi agent
# ---------------------------------------------------------------------------

class TestMultiAgent:
    def test_returns_multi_agent_trace(self, adapter):
        trace = adapter.transform_to_canonical({"stream_events": _multi_agent_events()})
        assert isinstance(trace, MultiAgentTrace)
        assert list(trace.agent_traces) == ["researcher", "writer"]

    def test_handoff_event(self, adapter):
        trace = adapter.transform_to_canonical({"stream_events": _multi_agent_events()})
        assert [(e.from_agent, e.to_agent, e.event_type) for e in trace.coordination_events] == [
            ("researcher", "writer", "HANDOFF"),
        ]

    @pytest.mark.parametrize("serialized", [False, True])
    def test_per_agent_messages(self, adapter, serialized):
        trace = adapter.transform_to_canonical(
            {"stream_events": _multi_agent_events(serialized=serialized)}
        )
        researcher = trace.agent_traces["researcher"].messages
        writer = trace.agent_traces["writer"].messages
        assert [(m.role, m.content) for m in researcher] == [
            (MessageRole.USER, "Write about bees"),
            (MessageRole.ASSISTANT, "Bees pollinate."),
        ]
        assert [(m.role, m.content) for m in writer] == [
            (MessageRole.ASSISTANT, "Draft about bees."),
        ]

    @pytest.mark.parametrize("serialized", [False, True])
    def test_per_agent_and_total_tokens(self, adapter, serialized):
        trace = adapter.transform_to_canonical(
            {"stream_events": _multi_agent_events(serialized=serialized)}
        )
        assert trace.agent_traces["researcher"].input_tokens == 10
        assert trace.agent_traces["writer"].output_tokens == 8
        assert trace.input_tokens == 40
        assert trace.output_tokens == 12

    def test_explicit_agent_node_names(self, adapter):
        trace = adapter.transform_to_canonical({
            "stream_events": _multi_agent_events(),
            "agent_node_names": ["researcher", "writer"],
        })
        assert isinstance(trace, MultiAgentTrace)


# ---------------------------------------------------------------------------
# Multi-turn sessions (list input)
# ---------------------------------------------------------------------------

def _turn(session_id, turn_id, text, **extra):
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "stream_events": [{"agent": {"messages": [HumanMessage(f"q{turn_id}"), AIMessage(text)]}}],
        **extra,
    }


class TestSessions:
    def test_grouped_in_first_seen_order_and_sorted_by_turn(self, adapter):
        sessions = adapter.transform_to_canonical([
            _turn("b", 2, "b2"),
            _turn("a", 1, "a1"),
            _turn("b", 1, "b1"),
        ])
        assert [s["session_id"] for s in sessions] == ["b", "a"]
        b_turns = sessions[0]["per_turn_traces"]
        assert [t.messages[-1].content for t in b_turns] == ["b1", "b2"]
        assert all(t.session_id == "b" for t in b_turns)

    def test_full_trace_aggregates(self, adapter):
        sessions = adapter.transform_to_canonical([
            _turn("s", 1, "one", latency=1.0, input_tokens=10, output_tokens=2, cost_usd=0.01),
            _turn("s", 2, "two", latency=2.0, input_tokens=20, output_tokens=3, cost_usd=0.02),
        ])
        full = sessions[0]["full_trace"]
        assert full.session_id == "s"
        assert full.latency == pytest.approx(3.0)
        assert full.input_tokens == 30
        assert full.output_tokens == 5
        assert full.metadata["cost_usd"] == pytest.approx(0.03)
        assert full.metadata["num_turns"] == 2
        assert [m.content for m in full.messages] == ["q1", "one", "q2", "two"]

    def test_empty_list_raises(self, adapter):
        with pytest.raises(AdapterTransformationError):
            adapter.transform_to_canonical([])

    @pytest.mark.parametrize("missing", ["session_id", "turn_id", "stream_events"])
    def test_missing_required_turn_field_raises(self, adapter, missing):
        turn = _turn("s", 1, "x")
        del turn[missing]
        with pytest.raises(AdapterTransformationError):
            adapter.transform_to_canonical([turn])

    def test_mixed_topology_in_one_session_raises(self, adapter):
        multi = {"session_id": "s", "turn_id": 2, "stream_events": _multi_agent_events()}
        with pytest.raises(AdapterTransformationError):
            adapter.transform_to_canonical([_turn("s", 1, "single"), multi])
