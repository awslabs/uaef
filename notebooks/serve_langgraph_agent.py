# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Serve the LangGraph weather agent on http://localhost:8123/invoke

Usage:
    python serve_langgraph_agent.py

Then in the notebook, set:
    AGENT_ENDPOINT = "http://localhost:8123/invoke"
"""

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_aws import ChatBedrock
import boto3
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import asyncio


# --- Build the LangGraph agent (same as notebook Step 1) ---

@tool
def get_weather(location: str) -> dict:
    """Get the current weather for a location."""
    return {"temp": 65, "condition": "cloudy", "location": location}

class State(TypedDict):
    messages: Annotated[list, add_messages]

region_name = "us-east-1"
my_config = Config(
    region_name=region_name,
    signature_version="v4",
    retries={"max_attempts": 3, "mode": "standard"},
)
bedrock_runtime = boto3.client(service_name="bedrock-runtime", config=my_config)
bedrock_llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"max_tokens": 1024, "temperature": 0.0},
)

tools = [get_weather]
llm_with_tools = bedrock_llm.bind_tools(tools)

def agent_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


# --- HTTP server ---

app = FastAPI()

MAX_INPUT_LENGTH = 10_000
REQUEST_TIMEOUT_SECONDS = 60

class InvokeRequest(BaseModel):
    input: str = Field(..., max_length=MAX_INPUT_LENGTH)
    session_id: str = Field(default="default", max_length=256)

def _serialize_message(msg):
    d = {"type": msg.__class__.__name__, "content": msg.content}
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    if hasattr(msg, "tool_call_id"):
        d["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        d["usage_metadata"] = msg.usage_metadata
    return d

@app.post("/invoke")
async def invoke(req: InvokeRequest):
    try:
        events = await asyncio.wait_for(
            asyncio.to_thread(_run_graph, req.input),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    serialized = []
    for evt in events:
        s_evt = {}
        for node_name, node_data in evt.items():
            if isinstance(node_data, dict) and "messages" in node_data:
                s_evt[node_name] = {
                    "messages": [_serialize_message(m) for m in node_data["messages"]]
                }
            else:
                s_evt[node_name] = node_data
        serialized.append(s_evt)
    return {"events": serialized}


def _run_graph(user_input: str):
    events = []
    for event in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="updates",
    ):
        events.append(event)
    return events


if __name__ == "__main__":
    print("✓ LangGraph agent server starting on http://localhost:8123")
    print("  POST /invoke  {\"input\": \"your question here\"}")
    print("  Press Ctrl+C to stop\n")
    uvicorn.run(app, host="127.0.0.1", port=8123)
