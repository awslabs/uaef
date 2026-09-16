# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Strands adapter, including multi-agent support."""

import pytest
from datetime import datetime, timezone

from uaef.adapters.base import AdapterTransformationError
from uaef.adapters.strands import StrandsAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.multi_agent_trace import CoordinationEvent, MultiAgentTrace, WorkflowStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return StrandsAdapter()


def _make_agent_data(
    user_text="Hello",
    assistant_text="Hi there",
    input_tokens=100,
    output_tokens=50,
    latency_ms=1200,
    stop_reason="end_turn",
    tool_use=None,
):
    """Build a minimal single-agent raw_data dict."""
    assistant_content = [{"text": assistant_text}]
    if tool_use:
        assistant_content.insert(0, {"toolUse": tool_use})

    return {
        "messages": [
            {"role": "user", "content": [{"text": user_text}]},
            {"role": "assistant", "content": assistant_content},
        ],
        "metrics_summary": {
            "accumulated_usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
            },
            "accumulated_metrics": {"latencyMs": latency_ms},
        },
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Single-agent tests
# ---------------------------------------------------------------------------

class TestSingleAgent:
    def test_returns_agent_trace(self, adapter):
        raw = _make_agent_data()
        result = adapter.transform_to_canonical(raw)
        assert isinstance(result, AgentTrace)

    def test_messages_extracted(self, adapter):
        raw = _make_agent_data(user_text="What is 2+2?", assistant_text="4")
        trace = adapter.transform_to_canonical(raw)
        assert len(trace.messages) == 2
        assert trace.messages[0].content == "What is 2+2?"
        assert trace.messages[1].content == "4"

    def test_token_counts(self, adapter):
        raw = _make_agent_data(input_tokens=200, output_tokens=80)
        trace = adapter.transform_to_canonical(raw)
        assert trace.input_tokens == 200
        assert trace.output_tokens == 80

    def test_latency(self, adapter):
        raw = _make_agent_data(latency_ms=2500)
        trace = adapter.transform_to_canonical(raw)
        assert trace.latency == pytest.approx(2.5)

    def test_tool_calls_extracted(self, adapter):
        tool_use = {
            "name": "get_weather",
            "toolUseId": "tu_1",
            "input": {"location": "Paris"},
        }
        raw = _make_agent_data(tool_use=tool_use)
        trace = adapter.transform_to_canonical(raw)
        assert len(trace.tool_calls) == 1
        assert trace.tool_calls[0].name == "get_weather"
        assert trace.tool_calls[0].arguments == {"location": "Paris"}

    def test_missing_required_keys_raises(self, adapter):
        with pytest.raises(AdapterTransformationError):
            adapter.transform_to_canonical({"messages": []})

    def test_single_agent_in_agents_dict_returns_agent_trace(self, adapter):
        """An 'agents' dict with only one entry should still return AgentTrace."""
        raw = {"agents": {"only_agent": _make_agent_data()}}
        result = adapter.transform_to_canonical(raw)
        assert isinstance(result, AgentTrace)


# ---------------------------------------------------------------------------
# Multi-agent tests
# ---------------------------------------------------------------------------

class TestMultiAgent:
    def _make_multi_raw(self, **overrides):
        base = {
            "agents": {
                "research_agent": _make_agent_data(
                    user_text="Search for: capital of France",
                    assistant_text="Paris is the capital.",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1200,
                ),
                "responder_agent": _make_agent_data(
                    user_text="Synthesize answer",
                    assistant_text="The capital of France is Paris.",
                    input_tokens=80,
                    output_tokens=30,
                    latency_ms=800,
                ),
            },
            "session_id": "sess-1",
        }
        base.update(overrides)
        return base

    def test_returns_multi_agent_trace(self, adapter):
        result = adapter.transform_to_canonical(self._make_multi_raw())
        assert isinstance(result, MultiAgentTrace)

    def test_agent_traces_keyed_correctly(self, adapter):
        result = adapter.transform_to_canonical(self._make_multi_raw())
        assert set(result.agent_traces.keys()) == {"research_agent", "responder_agent"}

    def test_per_agent_messages(self, adapter):
        result = adapter.transform_to_canonical(self._make_multi_raw())
        research = result.agent_traces["research_agent"]
        assert any("capital" in m.content.lower() for m in research.messages)

    def test_aggregated_tokens(self, adapter):
        result = adapter.transform_to_canonical(self._make_multi_raw())
        assert result.input_tokens == 180
        assert result.output_tokens == 80

    def test_aggregated_latency(self, adapter):
        result = adapter.transform_to_canonical(self._make_multi_raw())
        assert result.total_latency == pytest.approx(2.0)

    def test_session_id_propagated(self, adapter):
        result = adapter.transform_to_canonical(self._make_multi_raw())
        assert result.session_id == "sess-1"
        for trace in result.agent_traces.values():
            assert trace.session_id == "sess-1"

    # -- Coordination events -----------------------------------------------

    def test_inferred_coordination_events(self, adapter):
        result = adapter.transform_to_canonical(self._make_multi_raw())
        assert len(result.coordination_events) == 1
        evt = result.coordination_events[0]
        assert evt.from_agent == "research_agent"
        assert evt.to_agent == "responder_agent"
        assert evt.event_type == "HANDOFF"

    def test_explicit_coordination_events(self, adapter):
        explicit = [
            {
                "from_agent": "research_agent",
                "to_agent": "responder_agent",
                "event_type": "custom_handoff",
                "message": "Passing context",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        result = adapter.transform_to_canonical(
            self._make_multi_raw(coordination_events=explicit)
        )
        assert len(result.coordination_events) == 1
        assert result.coordination_events[0].event_type == "custom_handoff"

    def test_three_agents_produce_two_coordination_events(self, adapter):
        raw = self._make_multi_raw()
        raw["agents"]["summarizer"] = _make_agent_data(
            user_text="Summarize", assistant_text="Summary.", latency_ms=500
        )
        result = adapter.transform_to_canonical(raw)
        assert len(result.coordination_events) == 2

    # -- Workflow status ----------------------------------------------------

    def test_completed_status(self, adapter):
        result = adapter.transform_to_canonical(self._make_multi_raw())
        assert result.workflow_status == WorkflowStatus.COMPLETED

    def test_failed_status_on_error(self, adapter):
        raw = self._make_multi_raw()
        raw["agents"]["research_agent"]["stop_reason"] = "error"
        result = adapter.transform_to_canonical(raw)
        assert result.workflow_status == WorkflowStatus.FAILED

    def test_partial_status_on_mixed(self, adapter):
        raw = self._make_multi_raw()
        raw["agents"]["research_agent"]["stop_reason"] = "max_tokens"
        result = adapter.transform_to_canonical(raw)
        assert result.workflow_status == WorkflowStatus.PARTIAL
