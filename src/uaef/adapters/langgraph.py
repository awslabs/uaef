# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangGraph adapter for transforming LangGraph traces to canonical format."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Union
from uuid import uuid4

try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
        messages_from_dict,
    )
    _LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    # langchain-core ships behind the optional 'langgraph' extra. Keep this
    # module importable on a core-only install (so ``import uaef.adapters``
    # succeeds); the LangGraph adapter raises an actionable ImportError only
    # when it is actually instantiated.
    AIMessage = HumanMessage = ToolMessage = None  # type: ignore[assignment]
    BaseMessage = SystemMessage = None  # type: ignore[assignment]
    messages_from_dict = None  # type: ignore[assignment]
    _LANGCHAIN_CORE_AVAILABLE = False

from uaef.adapters.base import AdapterTransformationError, BaseAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message, MessageRole
from uaef.models.multi_agent_trace import CoordinationEvent, MultiAgentTrace, WorkflowStatus
from uaef.models.tool_call import ToolCall


class LangGraphAdapter(BaseAdapter):
    """
    Adapter for transforming LangGraph stream events to canonical AgentTrace format.
    
    This adapter processes LangGraph execution traces that contain:
    - Stream events from graph.stream() calls
    - Messages with tool calls and responses
    - Token usage metadata
    - Agent and tool node outputs
    
    The adapter reuses logic from the get_answers_detail_langgraph() function
    in agenticevaluationframework-main/frontend/evaluation.py.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter, requiring the optional 'langgraph' extra.

        The adapter depends on ``langchain-core`` message types, which ship
        behind the optional ``langgraph`` extra. Instantiating it on a
        core-only install raises an actionable ImportError naming the extra
        and the exact ``pip install`` command.
        """
        if not _LANGCHAIN_CORE_AVAILABLE:
            raise ImportError(
                "The LangGraph adapter requires the 'langgraph' extra. "
                "Install with: pip install 'uaef[langgraph]'"
            )
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "langgraph"

    def supports_session_transform(self) -> bool:
        """This adapter accepts a list of per-turn payloads (see
        :meth:`_transform_sessions`) and returns one session dict per
        conversation, so multi-turn evaluation can use it."""
        return True

    
    def _infer_agent_and_tool_nodes(self, stream_events: List[Any]) -> tuple:
        """
        Classify graph nodes by inspecting message types in stream events.

        A node is an agent node if it emits any AIMessage (type == "ai").
        A node is a tool node if it emits any ToolMessage (type == "tool").
        Preserves first-seen order for agent nodes.

        Returns:
            (agent_node_names, tool_node_names) — both as lists
        """
        agent_nodes: List[str] = []
        tool_nodes: List[str] = []

        for event in stream_events:
            if not isinstance(event, dict):
                continue
            for key, value in event.items():
                if not isinstance(value, dict):
                    continue
                messages = value.get("messages", [])
                if not isinstance(messages, list):
                    messages = [messages]
                for raw_msg in messages:
                    msg = self._coerce_message(raw_msg)
                    if isinstance(msg, AIMessage) and key not in agent_nodes:
                        agent_nodes.append(key)
                    elif isinstance(msg, ToolMessage) and key not in tool_nodes:
                        tool_nodes.append(key)

        return agent_nodes, tool_nodes

    def transform_to_canonical(self, raw_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[AgentTrace, MultiAgentTrace, Dict[str, Any]]:
        """
        Transform LangGraph trace data to canonical format.

        Supports two input shapes:

        **Single-turn (dict)** — returns AgentTrace or MultiAgentTrace as before.
        Expected for single-agent:
        {
            "stream_events": [...],
            "agent_node_name": "agent",   # optional; inferred if omitted
            "tool_node_name": "tools",
        }

        Expected for multi-agent:
        {
            "stream_events": [...],
            "agent_node_names": ["researcher", "writer", "reviewer"],  # optional; inferred if omitted
            "tool_node_name": "tools",
            "orchestrator_node_name": "orchestrator",  # optional
        }

        **Multi-turn / multi-session (list)** — returns a list of session dicts.

        Input is a flat list of per-turn payloads. Turns are grouped by
        `session_id` (preserving first-seen order) and sorted within each
        session by `turn_id`. Each unique session_id produces one session
        dict in the output.

        Per-turn payload schema:
        [
            {
                "session_id": "...",          # required, used to group turns
                "turn_id": 1,                  # required, used to order turns within a session
                "stream_events": [...],        # required, raw LangGraph events for this turn
                "latency": 1.23,               # optional, per-turn latency in seconds
                "input_tokens": 100,           # optional, per-turn input tokens
                "output_tokens": 200,          # optional, per-turn output tokens
                "cost_usd": 0.0015,            # optional, per-turn cost in USD
            },
            ...
        ]

        Returns a list with one entry per unique session_id:
        [
            {
                "session_id": "session_a",
                "per_turn_traces": [AgentTrace, AgentTrace, ...],  # one per turn in this session
                "full_trace": AgentTrace,                           # concatenated across all turns
            },
            {
                "session_id": "session_b",
                "per_turn_traces": [AgentTrace],                    # single-turn session is a list of 1
                "full_trace": AgentTrace,
            },
            ...
        ]

        Mixed-cardinality batches (some sessions with one turn, others with
        many) are handled uniformly — single-turn sessions just produce a
        per_turn_traces list of length 1.

        Multi-agent runs follow the same shape: if a session's events involve
        more than one agent node, both `per_turn_traces` entries and
        `full_trace` come back as `MultiAgentTrace` instead of `AgentTrace`,
        which lets multi-agent metrics score on top of multi-turn metrics.
        Mixing single-agent and multi-agent turns within the same session
        raises an error.

        Args:
            raw_data: LangGraph trace data (dict for single-turn,
                      list for multi-turn / multi-session)

        Returns:
            AgentTrace or MultiAgentTrace for dict input, or list of session
            dicts for list input.
        """
        # Multi-turn / multi-session input: list of per-turn payloads
        if isinstance(raw_data, list):
            return self._transform_sessions(raw_data)

        try:
            self.validate_raw_data(raw_data, ["stream_events"])

            stream_events = raw_data.get("stream_events", [])

            # Resolve agent/tool node names: explicit values > infer from message types
            agent_node_names = raw_data.get("agent_node_names")
            if not isinstance(agent_node_names, list):
                agent_node_names, inferred_tool_nodes = self._infer_agent_and_tool_nodes(stream_events)
                tool_node_name = (
                    raw_data.get("tool_node_name")
                    or (inferred_tool_nodes[0] if inferred_tool_nodes else "tools")
                )
                raw_data = {**raw_data, "agent_node_names": agent_node_names, "tool_node_name": tool_node_name}
            else:
                tool_node_name = raw_data.get("tool_node_name", "tools")

            if len(agent_node_names) > 1:
                return self._transform_multi_agent(raw_data)

            # Single-agent: fall back using the (possibly inferred) sole agent name
            if agent_node_names and "agent_node_name" not in raw_data:
                raw_data = {**raw_data, "agent_node_name": agent_node_names[0]}

            return self._transform_single_agent(raw_data)

        except AdapterTransformationError:
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform LangGraph trace: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )

    def _transform_sessions(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform a flat list of per-turn payloads into a list of session dicts.

        Turns are grouped by session_id (preserving first-seen order), sorted by
        turn_id within each session, and each session is materialized into a
        {"session_id", "per_turn_traces", "full_trace"} dict.

        Args:
            turns: Flat list of per-turn payloads. Each must include "session_id"
                   and "stream_events"; "turn_id", "latency", "input_tokens",
                   "output_tokens", "cost_usd" are optional.

        Returns:
            List of session dicts, one per unique session_id:
            [
                {
                    "session_id": "...",
                    "per_turn_traces": [AgentTrace, ...],
                    "full_trace": AgentTrace,
                },
                ...
            ]

        Raises:
            AdapterTransformationError: If transformation fails
        """
        try:
            if not turns:
                raise AdapterTransformationError(
                    "Multi-turn input list is empty",
                    adapter_name=self.name,
                )

            # Group turns by session_id, preserving first-seen order.
            session_order: List[str] = []
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for turn in turns:
                if "stream_events" not in turn:
                    raise AdapterTransformationError(
                        f"Turn payload missing 'stream_events': turn_id={turn.get('turn_id')}",
                        adapter_name=self.name,
                    )
                sid = turn.get("session_id")
                if sid is None:
                    raise AdapterTransformationError(
                        "Turn payload missing 'session_id'",
                        adapter_name=self.name,
                    )
                if turn.get("turn_id") is None:
                    raise AdapterTransformationError(
                        f"Turn payload missing 'turn_id' (session_id={sid})",
                        adapter_name=self.name,
                    )
                if sid not in grouped:
                    grouped[sid] = []
                    session_order.append(sid)
                grouped[sid].append(turn)

            sessions: List[Dict[str, Any]] = []
            for sid in session_order:
                session_turns = grouped[sid]
                # Sort within-session by turn_id (always present after validation above)
                session_turns = sorted(session_turns, key=lambda t: t["turn_id"])
                sessions.append(self._build_session(sid, session_turns))

            return sessions

        except AdapterTransformationError:
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform multi-turn LangGraph trace: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    def _build_session(self, session_id: str, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a single session dict from its ordered list of turn payloads.

        Each per-turn trace is built by reusing the same per-turn-vs-multi-turn
        dispatch as the single-call path: if the turn's events involve more
        than one agent node, we produce a MultiAgentTrace; otherwise an
        AgentTrace. The full session trace is built the same way from the
        concatenated events. Latency / tokens / cost are aggregated by
        summation across turns.

        Args:
            session_id: Session identifier shared by all turns
            turns: Ordered list of per-turn payloads for this session

        Returns:
            {"session_id", "per_turn_traces", "full_trace"}
        """
        per_turn_traces: List[Union[AgentTrace, MultiAgentTrace]] = []
        all_events: List[Any] = []

        agg_latency = 0.0
        agg_input_tokens = 0
        agg_output_tokens = 0
        agg_cost_usd = 0.0
        has_latency = False
        has_tokens = False
        has_cost = False

        def _build_trace_for_events(stream_events):
            """Build a per-turn or full trace from raw events.

            Picks single-agent vs. multi-agent based on inferred agent nodes,
            mirroring the dispatch in transform_to_canonical().
            """
            agent_nodes, tool_nodes = self._infer_agent_and_tool_nodes(stream_events)
            payload: Dict[str, Any] = {"stream_events": stream_events}
            if tool_nodes:
                payload["tool_node_name"] = tool_nodes[0]
            if len(agent_nodes) > 1:
                payload["agent_node_names"] = agent_nodes
                return self._transform_multi_agent(payload)
            if agent_nodes:
                payload["agent_node_name"] = agent_nodes[0]
            return self._transform_single_agent(payload)

        def _set_trace_field(trace, field, value):
            """Set a field on either an AgentTrace or a MultiAgentTrace.

            MultiAgentTrace stores latency on `total_latency`, and
            input/output tokens are computed properties — so we cache the
            per-turn values inside metadata to keep them visible.
            """
            if isinstance(trace, MultiAgentTrace):
                if field == "latency":
                    trace.total_latency = value
                elif field in ("input_tokens", "output_tokens"):
                    trace.metadata[field] = value
            else:
                setattr(trace, field, value)

        for turn in turns:
            turn_trace = _build_trace_for_events(turn["stream_events"])

            if turn.get("latency") is not None:
                _set_trace_field(turn_trace, "latency", turn["latency"])
                agg_latency += float(turn["latency"])
                has_latency = True
            if turn.get("input_tokens") is not None:
                _set_trace_field(turn_trace, "input_tokens", turn["input_tokens"])
                agg_input_tokens += int(turn["input_tokens"])
                has_tokens = True
            if turn.get("output_tokens") is not None:
                _set_trace_field(turn_trace, "output_tokens", turn["output_tokens"])
                agg_output_tokens += int(turn["output_tokens"])
                has_tokens = True
            if turn.get("cost_usd") is not None:
                turn_trace.metadata["cost_usd"] = turn["cost_usd"]
                agg_cost_usd += float(turn["cost_usd"])
                has_cost = True
            if turn.get("turn_id") is not None:
                turn_trace.metadata["turn_id"] = turn["turn_id"]
            turn_trace.session_id = session_id

            per_turn_traces.append(turn_trace)
            all_events.extend(turn["stream_events"])

        full_trace = _build_trace_for_events(all_events)
        full_trace.session_id = session_id
        if has_latency:
            _set_trace_field(full_trace, "latency", agg_latency)
        if has_tokens:
            _set_trace_field(full_trace, "input_tokens", agg_input_tokens)
            _set_trace_field(full_trace, "output_tokens", agg_output_tokens)
        if has_cost:
            full_trace.metadata["cost_usd"] = agg_cost_usd
        full_trace.metadata["num_turns"] = len(per_turn_traces)

        # Mixing single-agent and multi-agent traces within one session is rejected
        # by evaluate_multi_turn; surface the error here with a clearer message.
        per_turn_types = {type(t) for t in per_turn_traces}
        if len(per_turn_types) > 1 or type(full_trace) not in per_turn_types:
            raise AdapterTransformationError(
                f"Session '{session_id}' mixes single-agent and multi-agent turns; "
                "every turn in a session must use the same graph topology.",
                adapter_name=self.name,
            )

        return {
            "session_id": session_id,
            "per_turn_traces": per_turn_traces,
            "full_trace": full_trace,
        }

    def _transform_single_agent(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """Transform to single AgentTrace (original behavior)."""
        messages = self.extract_messages(raw_data)
        tool_calls = self.extract_tool_calls(raw_data)
        metadata = self.extract_metadata(raw_data)

        return AgentTrace(
            trace_id=raw_data.get("trace_id", uuid4()),
            session_id=raw_data.get("session_id"),
            messages=messages,
            tool_calls=tool_calls,
            input_tokens=metadata.get("input_tokens"),
            output_tokens=metadata.get("output_tokens"),
            latency=metadata.get("latency"),
            cost=metadata.get("cost"),
            metadata=metadata,
            framework="langgraph",
            timestamp=datetime.now(timezone.utc)
        )

    def _transform_multi_agent(self, raw_data: Dict[str, Any]) -> MultiAgentTrace:
        """
        Transform to MultiAgentTrace by splitting stream events per agent node.

        Each agent node name becomes a separate AgentTrace. Coordination events
        are inferred from the sequential ordering of agent activations.
        """
        stream_events = raw_data.get("stream_events", [])
        agent_node_names = raw_data.get("agent_node_names", [])
        tool_node_name = raw_data.get("tool_node_name", "tools")
        orchestrator_name = raw_data.get("orchestrator_node_name")

        # Build per-agent data by scanning events
        agent_messages: Dict[str, List[Message]] = {name: [] for name in agent_node_names}
        agent_tool_calls: Dict[str, List[ToolCall]] = {name: [] for name in agent_node_names}
        coordination_events: List[CoordinationEvent] = []
        activation_order: List[str] = []
        # HumanMessages are prepended to the first agent that activates
        human_messages: List[Message] = []
        first_agent_populated = False

        for event in stream_events:
            if not isinstance(event, dict):
                continue

            for key, value in event.items():
                if not isinstance(value, dict):
                    continue
                value_messages = value.get("messages")
                if not value_messages:
                    continue
                if not isinstance(value_messages, list):
                    value_messages = [value_messages]

                # Collect HumanMessages before any agent fires
                for msg in value_messages:
                    if isinstance(msg, HumanMessage) and not first_agent_populated:
                        human_messages.append(Message(
                            role=MessageRole.USER,
                            content=str(getattr(msg, "content", "") or ""),
                            timestamp=datetime.now(timezone.utc)
                        ))

                if key in agent_node_names:
                    activation_order.append(key)
                    msg = value_messages[-1]

                    if not first_agent_populated and human_messages:
                        agent_messages[key].extend(human_messages)
                        first_agent_populated = True

                    content = getattr(msg, "content", "") or ""
                    tc_data = getattr(msg, "tool_calls", []) or []

                    message_tool_calls = []
                    for tc in tc_data:
                        if isinstance(tc, dict) and tc.get("name"):
                            tool_call = ToolCall(
                                name=tc.get("name", ""),
                                arguments=tc.get("args", {}),
                                timestamp=datetime.now(timezone.utc)
                            )
                            message_tool_calls.append(tool_call)
                            agent_tool_calls[key].append(tool_call)

                    agent_messages[key].append(Message(
                        role=MessageRole.ASSISTANT,
                        content=str(content),
                        tool_calls=message_tool_calls,
                        timestamp=datetime.now(timezone.utc)
                    ))

        # Infer coordination events from sequential agent activations
        for i in range(1, len(activation_order)):
            from_agent = activation_order[i - 1]
            to_agent = activation_order[i]
            if from_agent != to_agent:
                coordination_events.append(CoordinationEvent(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    event_type="HANDOFF",
                    message=f"Control passed from {from_agent} to {to_agent}",
                    timestamp=datetime.now(timezone.utc)
                ))

        # Build per-agent AgentTraces
        agent_traces = {}
        for name in agent_node_names:
            agent_traces[name] = AgentTrace(
                trace_id=uuid4(),
                messages=agent_messages[name],
                tool_calls=agent_tool_calls[name],
                framework="langgraph",
                timestamp=datetime.now(timezone.utc)
            )

        metadata = self.extract_metadata(raw_data)
        return MultiAgentTrace(
            trace_id=raw_data.get("trace_id", uuid4()),
            session_id=raw_data.get("session_id"),
            agent_traces=agent_traces,
            coordination_events=coordination_events,
            workflow_status=WorkflowStatus.COMPLETED,
            total_latency=metadata.get("latency"),
            metadata=metadata,
            timestamp=datetime.now(timezone.utc)
        )
    
    @staticmethod
    def _coerce_message(msg: Any) -> Any:
        """Reconstruct a live LangChain message from a serialized dict.

        In-process (notebook) runs hand the adapter real ``AIMessage`` /
        ``HumanMessage`` objects. When a LangGraph agent is served over HTTP
        (see ``notebooks/serve_langgraph_agent.py``), those objects are
        serialized to JSON and arrive here as **dicts** — so the downstream
        ``isinstance`` checks would match nothing and the trace would come back
        empty. This helper turns the dict back into the corresponding LangChain
        message object so the rest of the extraction logic is unchanged.

        Handles three shapes:
          * live message objects — returned as-is,
          * LangChain-native serialized form ``{"type": "ai", "data": {...}}``
            (``messages_to_dict`` output) — restored via ``messages_from_dict``,
          * the class-name / role form used by the reference HTTP server
            ``{"type": "AIMessage"|..., "content": ..., "tool_calls": [...],
            "tool_call_id": "..."}`` (or ``{"role": "assistant", ...}``).

        Returns the message object, or ``None`` when the dict can't be mapped
        (callers skip ``None``).
        """
        # Already a live LangChain message.
        if BaseMessage is not None and isinstance(msg, BaseMessage):
            return msg
        if not isinstance(msg, dict):
            return msg

        # LangChain-native serialized form: {"type": "...", "data": {...}}.
        if messages_from_dict is not None and isinstance(msg.get("data"), dict):
            try:
                restored = messages_from_dict([msg])
                if restored:
                    return restored[0]
            except Exception:  # noqa: BLE001 — fall through to the role/class form
                pass

        type_hint = str(msg.get("type") or msg.get("role") or "").lower()
        content = msg.get("content", "")
        try:
            if type_hint in ("aimessage", "aimessagechunk", "ai", "assistant"):
                tool_calls = msg.get("tool_calls") or []
                try:
                    return AIMessage(content=content, tool_calls=tool_calls)
                except Exception:  # noqa: BLE001 — malformed tool_calls; keep the text
                    return AIMessage(content=content)
            if type_hint in ("humanmessage", "human", "user"):
                return HumanMessage(content=content)
            if type_hint in ("toolmessage", "tool"):
                return ToolMessage(
                    content=content, tool_call_id=msg.get("tool_call_id", "") or ""
                )
            if type_hint in ("systemmessage", "system"):
                return SystemMessage(content=content)
        except Exception:  # noqa: BLE001 — unmappable dict; skip it
            return None
        return None

    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from LangGraph stream events.
        
        Messages are extracted from agent node outputs in the stream events.
        Each agent node output contains a message with content and optional tool calls.
        Serialized (dict) messages from an HTTP-served agent are reconstructed
        into LangChain message objects via :meth:`_coerce_message` first.
        
        Args:
            raw_data: LangGraph trace data
            
        Returns:
            List of Message objects
            
        Raises:
            AdapterTransformationError: If message extraction fails
        """
        try:
            messages = []
            stream_events = raw_data.get("stream_events", [])
            agent_node_name = raw_data.get("agent_node_name", "agent")

            for event in stream_events:
                if not isinstance(event, dict):
                    continue

                for key, value in event.items():
                    if not isinstance(value, dict):
                        continue
                    value_messages = value.get("messages")
                    if not value_messages:
                        continue
                    if not isinstance(value_messages, list):
                        value_messages = [value_messages]

                    for raw_msg in value_messages:
                        msg = self._coerce_message(raw_msg)
                        if msg is None:
                            continue
                        if isinstance(msg, HumanMessage):
                            messages.append(Message(
                                role=MessageRole.USER,
                                content=str(getattr(msg, "content", "") or ""),
                                timestamp=datetime.now(timezone.utc)
                            ))
                        elif key == agent_node_name and isinstance(msg, AIMessage):
                            tool_calls_data = getattr(msg, "tool_calls", []) or []
                            message_tool_calls = [
                                ToolCall(
                                    name=tc.get("name", ""),
                                    arguments=tc.get("args", {}),
                                    timestamp=datetime.now(timezone.utc)
                                )
                                for tc in tool_calls_data if isinstance(tc, dict)
                            ]
                            messages.append(Message(
                                role=MessageRole.ASSISTANT,
                                content=str(getattr(msg, "content", "") or ""),
                                tool_calls=message_tool_calls,
                                timestamp=datetime.now(timezone.utc)
                            ))

            return messages
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract messages: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract tool calls from LangGraph stream events.
        
        Tool calls are extracted from:
        1. Agent node outputs (tool_calls field)
        2. Tool node outputs (tool results)
        
        Args:
            raw_data: LangGraph trace data
            
        Returns:
            List of ToolCall objects
            
        Raises:
            AdapterTransformationError: If tool call extraction fails
        """
        try:
            tool_calls = []
            stream_events = raw_data.get("stream_events", [])
            agent_node_name = raw_data.get("agent_node_name", "agent")
            tool_node_name = raw_data.get("tool_node_name", "tools")
            
            # Track tool calls and their results
            tool_call_map = {}  # Maps tool call index to ToolCall object
            tool_call_index = 0
            
            for event in stream_events:
                if not isinstance(event, dict):
                    continue
                
                for key, value in event.items():
                    # Extract tool calls from agent node
                    if key == agent_node_name and isinstance(value, dict):
                        value_messages = value.get("messages")
                        if not value_messages:
                            continue
                        
                        # Handle both list and single message
                        if isinstance(value_messages, list):
                            msg = value_messages[-1] if value_messages else None
                        else:
                            msg = value_messages

                        # Reconstruct a live message from a serialized dict
                        # (HTTP-served agents) so getattr(tool_calls) works.
                        msg = self._coerce_message(msg) if msg is not None else None

                        if msg:
                            tool_calls_data = getattr(msg, "tool_calls", []) or []
                            for tc in tool_calls_data:
                                if isinstance(tc, dict) and tc.get("name"):
                                    tool_call = ToolCall(
                                        name=tc.get("name", ""),
                                        arguments=tc.get("args", {}),
                                        timestamp=datetime.now(timezone.utc)
                                    )
                                    tool_call_map[tool_call_index] = tool_call
                                    tool_calls.append(tool_call)
                                    tool_call_index += 1
                    
                    # Extract tool results from tool node
                    elif key == tool_node_name and value:
                        # Tool node output contains results
                        # Try to match with previous tool calls
                        # Note: This is a simplified approach; actual matching may need more logic
                        if isinstance(value, dict):
                            # Store as result for the most recent tool call
                            if tool_calls:
                                tool_calls[-1].result = value
            
            return tool_calls
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract tool calls: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from LangGraph stream events.
        
        Metadata includes:
        - input_tokens: Total input tokens from usage_metadata
        - output_tokens: Total output tokens from usage_metadata
        - latency: Execution time (if provided)
        - model: Model name (if available)
        
        Args:
            raw_data: LangGraph trace data
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterTransformationError: If metadata extraction fails
        """
        try:
            metadata = {}
            stream_events = raw_data.get("stream_events", [])
            agent_node_name = raw_data.get("agent_node_name", "agent")
            
            input_tokens = 0
            output_tokens = 0
            
            # Accumulate token usage from all agent node outputs
            for event in stream_events:
                if not isinstance(event, dict):
                    continue
                
                for key, value in event.items():
                    if key == agent_node_name and isinstance(value, dict):
                        value_messages = value.get("messages")
                        if not value_messages:
                            continue
                        
                        # Handle both list and single message
                        if isinstance(value_messages, list):
                            msg = value_messages[-1] if value_messages else None
                        else:
                            msg = value_messages
                        
                        if msg:
                            # Extract usage metadata
                            usage_metadata = getattr(msg, "usage_metadata", {}) or {}
                            if isinstance(usage_metadata, dict):
                                input_tokens += usage_metadata.get("input_tokens", 0)
                                output_tokens += usage_metadata.get("output_tokens", 0)
            
            # Add token counts to metadata
            if input_tokens > 0:
                metadata["input_tokens"] = input_tokens
            if output_tokens > 0:
                metadata["output_tokens"] = output_tokens
            
            # Add any additional metadata from raw_data
            if "latency" in raw_data:
                metadata["latency"] = raw_data["latency"]
            if "model" in raw_data:
                metadata["model"] = raw_data["model"]
            if "config" in raw_data:
                metadata["config"] = raw_data["config"]
            
            return metadata
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract metadata: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
