# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands Agents adapter for transforming Strands traces to canonical format.

Uses native Strands telemetry (EventLoopMetrics, AgentResult) rather than
requiring manual cycle construction.  Supports both single-agent and
multi-agent traces.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Union
from uuid import uuid4

from uaef.adapters.base import AdapterTransformationError, BaseAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message, MessageRole
from uaef.models.multi_agent_trace import CoordinationEvent, MultiAgentTrace, WorkflowStatus
from uaef.models.tool_call import ToolCall


class StrandsAdapter(BaseAdapter):
    """
    Adapter for transforming Strands Agents SDK traces to canonical AgentTrace format.

    This adapter consumes the native telemetry objects produced by the Strands SDK:
    - ``AgentResult.messages`` — the full conversation history
    - ``AgentResult.metrics`` — an ``EventLoopMetrics`` instance with traces,
      accumulated token usage, tool metrics, and per-cycle breakdowns

    Expected raw_data for single-agent::

        {
            "messages": [...],          # AgentResult.messages (list of message dicts)
            "metrics_summary": {...},   # EventLoopMetrics.get_summary()
            "stop_reason": "end_turn",  # AgentResult.stop_reason (optional)
            "session_id": "...",        # Optional session identifier
            "trace_id": "...",          # Optional trace identifier (UUID string)
            "model": "...",             # Optional model identifier
            "cost": 0.005,             # Optional estimated cost in USD
        }

    Expected raw_data for multi-agent::

        {
            "agents": {
                "research_agent": {
                    "messages": [...],
                    "metrics_summary": {...},
                    "stop_reason": "end_turn",
                },
                "responder_agent": {
                    "messages": [...],
                    "metrics_summary": {...},
                    "stop_reason": "end_turn",
                },
            },
            "coordination_events": [...],  # Optional; inferred from agent ordering
            "session_id": "...",
            "trace_id": "...",
        }
    """

    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "strands"

    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> Union[AgentTrace, MultiAgentTrace]:
        """Transform Strands trace data to canonical format.

        Returns ``MultiAgentTrace`` when ``raw_data`` contains an ``agents``
        dict with more than one entry.  Otherwise returns a single
        ``AgentTrace``.

        Args:
            raw_data: Strands trace data.

        Returns:
            AgentTrace for single-agent, MultiAgentTrace for multi-agent.

        Raises:
            AdapterTransformationError: If transformation fails.
        """
        try:
            agents = raw_data.get("agents")
            if isinstance(agents, dict):
                if not agents:
                    raise AdapterTransformationError(
                        "'agents' dict is empty",
                        adapter_name=self.name,
                    )
                if len(agents) > 1:
                    return self._transform_multi_agent(raw_data)
                only_data = next(iter(agents.values()))
                merged = {**only_data, "session_id": raw_data.get("session_id")}
                if raw_data.get("trace_id") is not None:
                    merged["trace_id"] = raw_data["trace_id"]
                raw_data = merged
            return self._transform_single_agent(raw_data)
        except AdapterTransformationError:
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform Strands trace: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    # ------------------------------------------------------------------
    # Single-agent transformation
    # ------------------------------------------------------------------

    def _transform_single_agent(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """Transform to a single AgentTrace (original behaviour)."""
        self.validate_raw_data(raw_data, ["messages", "metrics_summary"])

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
            framework="strands",
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Multi-agent transformation
    # ------------------------------------------------------------------

    def _transform_multi_agent(self, raw_data: Dict[str, Any]) -> MultiAgentTrace:
        """Transform to MultiAgentTrace by building per-agent traces.

        Each key in ``raw_data["agents"]`` becomes a separate AgentTrace.
        Coordination events are taken from ``raw_data["coordination_events"]``
        if provided, otherwise inferred from agent ordering.
        """
        agents_data: Dict[str, Dict[str, Any]] = raw_data.get("agents", {})
        agent_names = list(agents_data.keys())
        session_id = raw_data.get("session_id")

        # Build per-agent AgentTraces by reusing existing extraction methods
        agent_traces: Dict[str, AgentTrace] = {}
        total_latency = 0.0
        total_cost = 0.0
        for name, agent_raw in agents_data.items():
            trace = self._transform_single_agent({
                **agent_raw,
                "session_id": session_id,
            })
            agent_traces[name] = trace
            if trace.latency:
                total_latency += trace.latency
            if trace.cost:
                total_cost += trace.cost

        # Coordination events: use explicit list or infer from ordering
        coordination_events = self._build_coordination_events(
            raw_data.get("coordination_events"), agent_names
        )

        # Determine workflow status from individual agent stop reasons
        statuses = [a.get("stop_reason") for a in agents_data.values()]
        if all(s in ("end_turn", None) for s in statuses):
            workflow_status = WorkflowStatus.COMPLETED
        elif any(s == "error" for s in statuses):
            workflow_status = WorkflowStatus.FAILED
        else:
            workflow_status = WorkflowStatus.PARTIAL

        return MultiAgentTrace(
            trace_id=raw_data.get("trace_id", uuid4()),
            session_id=session_id,
            agent_traces=agent_traces,
            coordination_events=coordination_events,
            workflow_status=workflow_status,
            total_latency=total_latency or None,
            total_cost=total_cost or None,
            metadata={"framework": "strands"},
            timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def _build_coordination_events(
        explicit_events: Any, agent_names: List[str]
    ) -> List[CoordinationEvent]:
        """Return explicit CoordinationEvents or infer from agent ordering."""
        if isinstance(explicit_events, list) and explicit_events:
            return [
                e if isinstance(e, CoordinationEvent)
                else CoordinationEvent(**e)
                for e in explicit_events
            ]
        events: List[CoordinationEvent] = []
        for i in range(1, len(agent_names)):
            events.append(CoordinationEvent(
                from_agent=agent_names[i - 1],
                to_agent=agent_names[i],
                event_type="HANDOFF",
                message=f"Control passed from {agent_names[i - 1]} to {agent_names[i]}",
                timestamp=datetime.now(timezone.utc),
            ))
        return events

    # ------------------------------------------------------------------
    # Message extraction
    # ------------------------------------------------------------------

    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """Extract messages from the Strands conversation history.

        Walks ``raw_data["messages"]`` (the native list stored on
        ``AgentResult.messages``).  Each dict has ``role`` and ``content``
        (a list of content blocks).

        Args:
            raw_data: Strands trace data.

        Returns:
            List of canonical Message objects.
        """
        try:
            messages: List[Message] = []
            raw_messages = raw_data.get("messages", [])

            for msg_data in raw_messages:
                if not isinstance(msg_data, dict):
                    continue

                role_str = msg_data.get("role", "assistant")
                content_blocks = msg_data.get("content", [])

                # Content can be a plain string (system prompt) or list of blocks
                if isinstance(content_blocks, str):
                    messages.append(
                        Message(
                            role=self._map_role(role_str),
                            content=content_blocks,
                            timestamp=datetime.now(timezone.utc),
                        )
                    )
                    continue

                text_parts: List[str] = []
                message_tool_calls: List[ToolCall] = []

                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue

                    # Text content
                    if "text" in block:
                        text_parts.append(str(block["text"]))

                    # Tool-use content block
                    tool_use = block.get("toolUse")
                    if isinstance(tool_use, dict) and tool_use.get("name"):
                        message_tool_calls.append(
                            ToolCall(
                                name=tool_use["name"],
                                arguments=tool_use.get("input", {}),
                                timestamp=datetime.now(timezone.utc),
                            )
                        )

                    # Tool-result content block (inside user messages)
                    tool_result = block.get("toolResult")
                    if isinstance(tool_result, dict):
                        result_text = self._extract_tool_result_text(tool_result)
                        text_parts.append(result_text)

                role = self._map_role(role_str)
                messages.append(
                    Message(
                        role=role,
                        content="\n".join(text_parts) if text_parts else "",
                        tool_calls=message_tool_calls,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

            return messages

        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract messages: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    # ------------------------------------------------------------------
    # Tool-call extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """Extract tool calls from the Strands conversation and metrics.

        Tool invocations are found in two places:
        1. ``toolUse`` content blocks inside assistant messages
        2. ``tool_usage`` in ``metrics_summary`` for execution stats

        The method walks the messages for call details and enriches them
        with duration/success info from the metrics summary.

        Args:
            raw_data: Strands trace data.

        Returns:
            List of ToolCall objects.
        """
        try:
            tool_calls: List[ToolCall] = []
            raw_messages = raw_data.get("messages", [])
            summary = raw_data.get("metrics_summary", {})

            # Build execution stats lookup from metrics_summary.tool_usage
            tool_stats: Dict[str, Dict[str, Any]] = {}
            for tool_name, info in summary.get("tool_usage", {}).items():
                exec_stats = info.get("execution_stats", {})
                tool_stats[tool_name] = exec_stats

            # Build a map of toolUseId → tool result from user messages
            result_map: Dict[str, Dict[str, Any]] = {}
            for msg_data in raw_messages:
                if not isinstance(msg_data, dict) or msg_data.get("role") != "user":
                    continue
                for block in msg_data.get("content", []):
                    if isinstance(block, dict) and "toolResult" in block:
                        tr = block["toolResult"]
                        if isinstance(tr, dict) and "toolUseId" in tr:
                            result_map[tr["toolUseId"]] = tr

            # Walk assistant messages for toolUse blocks
            for msg_data in raw_messages:
                if not isinstance(msg_data, dict) or msg_data.get("role") != "assistant":
                    continue

                for block in msg_data.get("content", []):
                    if not isinstance(block, dict):
                        continue

                    tool_use = block.get("toolUse")
                    if not isinstance(tool_use, dict) or not tool_use.get("name"):
                        continue

                    tool_name = tool_use["name"]
                    tool_use_id = tool_use.get("toolUseId", "")

                    # Match result
                    result = result_map.get(tool_use_id)
                    result_value = self._extract_tool_result_text(result) if result else None
                    status = result.get("status") if result else None
                    error = None
                    if status and status != "success":
                        error = result_value

                    # Enrich with execution time from metrics
                    stats = tool_stats.get(tool_name, {})
                    call_count = stats.get("call_count", 0)
                    avg_time = stats.get("average_time") if call_count else None

                    tool_calls.append(
                        ToolCall(
                            name=tool_name,
                            arguments=tool_use.get("input", {}),
                            result=result_value,
                            timestamp=datetime.now(timezone.utc),
                            execution_time=avg_time,
                            error=error,
                        )
                    )

            return tool_calls

        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract tool calls: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from the Strands metrics summary.

        Reads ``accumulated_usage`` and ``accumulated_metrics`` from the
        ``metrics_summary`` dict produced by ``EventLoopMetrics.get_summary()``.

        Args:
            raw_data: Strands trace data.

        Returns:
            Dictionary of metadata.
        """
        try:
            metadata: Dict[str, Any] = {}
            summary = raw_data.get("metrics_summary", {})

            # Token usage from accumulated_usage
            usage = summary.get("accumulated_usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            if input_tokens > 0:
                metadata["input_tokens"] = input_tokens
            if output_tokens > 0:
                metadata["output_tokens"] = output_tokens

            # Latency from accumulated_metrics (ms → seconds)
            acc_metrics = summary.get("accumulated_metrics", {})
            latency_ms = acc_metrics.get("latencyMs", 0)
            if latency_ms > 0:
                metadata["latency"] = latency_ms / 1000.0

            # Cycle stats
            total_cycles = summary.get("total_cycles", 0)
            if total_cycles > 0:
                metadata["total_cycles"] = total_cycles
                metadata["total_duration"] = summary.get("total_duration", 0)
                metadata["average_cycle_time"] = summary.get("average_cycle_time", 0)

            # Tool usage summary
            if summary.get("tool_usage"):
                metadata["tool_usage"] = summary["tool_usage"]

            # Per-invocation breakdown
            if summary.get("agent_invocations"):
                metadata["agent_invocations"] = summary["agent_invocations"]

            # Pass-through optional top-level fields
            if "model" in raw_data:
                metadata["model"] = raw_data["model"]
            if "cost" in raw_data:
                metadata["cost"] = raw_data["cost"]
            if "stop_reason" in raw_data:
                metadata["stop_reason"] = raw_data["stop_reason"]

            return metadata

        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract metadata: {str(e)}",
                adapter_name=self.name,
                original_error=e,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_role(role_str: str) -> MessageRole:
        """Map a Strands role string to a canonical MessageRole."""
        mapping = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "tool": MessageRole.TOOL,
        }
        return mapping.get(role_str, MessageRole.ASSISTANT)

    @staticmethod
    def _extract_tool_result_text(tool_result: Dict[str, Any]) -> str:
        """Extract a text representation from a tool-result dict."""
        content = tool_result.get("content", [])
        if isinstance(content, list):
            parts = [
                str(c.get("text", ""))
                for c in content
                if isinstance(c, dict) and "text" in c
            ]
            return "\n".join(parts)
        return str(content)
