# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict

from uaef.adapters.langgraph import LangGraphAdapter


@pytest.mark.parametrize("encoding", ["live", "native", "role"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("multi_agent", [False, True])
def test_parallel_tool_results(encoding, reverse, multi_agent):
    calls = [
        {"name": "lookup", "args": {"key": key}, "id": key} for key in ["a", "b", "c"]
    ]
    request = AIMessage(content="", tool_calls=calls)
    replies = [
        ToolMessage(content="timeout", tool_call_id="a", status="error"),
        ToolMessage(content="found", tool_call_id="b"),
        ToolMessage(content="unrelated", tool_call_id="unknown"),
    ]
    if reverse:
        replies.reverse()
    def encode(messages):
        if encoding == "native":
            return messages_to_dict(messages)
        if encoding == "role":
            return [msg.model_dump() for msg in messages]
        return messages

    events = [
        {"agent": {"messages": encode([request])}},
        {"tools": {"messages": encode(replies)}},
    ]
    trace = LangGraphAdapter().transform_to_canonical({
        "agent_node_names": ["agent", "reviewer"] if multi_agent else ["agent"],
        "stream_events": events,
    })
    if multi_agent:
        trace = trace.agent_traces["agent"]
    failed, succeeded, unanswered = trace.tool_calls
    assert failed.error == "timeout"
    assert succeeded.result == "found"
    assert succeeded.error is None
    assert unanswered.result is None
