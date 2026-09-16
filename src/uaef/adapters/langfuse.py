# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Langfuse adapter for transforming Langfuse traces to canonical format."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from uaef.adapters.base import AdapterTransformationError, BaseAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message, MessageRole
from uaef.models.tool_call import ToolCall


class LangfuseAdapter(BaseAdapter):
    """
    Adapter for transforming Langfuse observation data to canonical AgentTrace format.
    
    This adapter processes Langfuse traces that contain:
    - Hierarchical observation trees (spans, generations, events)
    - Messages from LLM generations
    - Tool calls from function/tool observations
    - Token usage and cost metadata from generations
    - Latency information from span durations
    
    Langfuse uses a hierarchical structure where:
    - Trace: Top-level container for a single execution
    - Observations: Individual steps within a trace (spans, generations, events)
    - Spans: Units of work with duration
    - Generations: LLM calls with token counts and I/O
    - Events: Point-in-time occurrences
    
    The adapter transforms this hierarchical structure into the flat canonical format.
    """
    
    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "langfuse"
    
    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """
        Transform Langfuse trace data to canonical AgentTrace format.
        
        Expected raw_data structure:
        {
            "trace_id": "...",  # Langfuse trace ID
            "session_id": "...",  # Optional session identifier
            "observations": [...],  # List of observation objects (spans, generations, events)
            "input": {...},  # Optional trace-level input
            "output": {...},  # Optional trace-level output
            "metadata": {...},  # Optional trace metadata
            "timestamp": "...",  # ISO 8601 timestamp
        }
        
        Alternative structure (from Observations API):
        {
            "data": [
                {
                    "id": "...",
                    "trace_id": "...",
                    "type": "generation|span|event",
                    "name": "...",
                    "input": {...},
                    "output": {...},
                    "metadata": {...},
                    "usage": {"input": 100, "output": 50, "total": 150},
                    "cost": 0.0025,
                    "start_time": "...",
                    "end_time": "...",
                    "parent_observation_id": "...",
                },
                ...
            ]
        }
        
        Args:
            raw_data: Langfuse trace data
            
        Returns:
            Canonical AgentTrace object
            
        Raises:
            AdapterTransformationError: If transformation fails
        """
        try:
            # Validate that we have some data to work with
            if not raw_data:
                raise AdapterTransformationError(
                    "Empty raw_data provided",
                    adapter_name=self.name
                )
            
            # Extract components
            messages = self.extract_messages(raw_data)
            tool_calls = self.extract_tool_calls(raw_data)
            metadata = self.extract_metadata(raw_data)
            
            # Determine trace_id
            trace_id = raw_data.get("trace_id")
            if trace_id:
                # Keep as string if it's not a UUID format
                try:
                    from uuid import UUID
                    if isinstance(trace_id, str):
                        trace_id = UUID(trace_id)
                except (ValueError, AttributeError):
                    # If not a valid UUID, generate a new one and store original in metadata
                    metadata["langfuse_trace_id"] = trace_id
                    trace_id = uuid4()
            else:
                trace_id = uuid4()
            
            # Create AgentTrace
            trace = AgentTrace(
                trace_id=trace_id,
                session_id=raw_data.get("session_id"),
                messages=messages,
                tool_calls=tool_calls,
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                latency=metadata.get("latency"),
                cost=metadata.get("cost"),
                metadata=metadata,
                framework="langfuse",
                timestamp=self._parse_timestamp(raw_data.get("timestamp"))
            )
            
            return trace
            
        except AdapterTransformationError:
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform Langfuse trace: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from Langfuse observations.
        
        Messages are extracted from:
        1. Generation observations (LLM calls with input/output)
        2. Trace-level input/output
        3. Span observations with message content
        
        Langfuse generations typically contain:
        - input: The prompt or messages sent to the LLM
        - output: The LLM's response
        
        Args:
            raw_data: Langfuse trace data
            
        Returns:
            List of Message objects in chronological order
            
        Raises:
            AdapterTransformationError: If message extraction fails
        """
        try:
            messages = []
            
            # Get observations list (handle both direct and nested formats)
            observations = raw_data.get("observations", [])
            if not observations and "data" in raw_data:
                observations = raw_data["data"]
            
            # Sort observations by timestamp to maintain chronological order
            sorted_observations = self._sort_observations_by_time(observations)
            
            # Extract messages from each observation
            for obs in sorted_observations:
                if not isinstance(obs, dict):
                    continue
                
                obs_type = obs.get("type", "").lower()
                
                # Extract from generation observations (LLM calls)
                if obs_type == "generation":
                    obs_messages = self._extract_messages_from_generation(obs)
                    messages.extend(obs_messages)
                
                # Extract from span observations if they contain message data
                elif obs_type == "span":
                    obs_messages = self._extract_messages_from_span(obs)
                    messages.extend(obs_messages)
            
            # If no messages found in observations, try trace-level input/output
            if not messages:
                if "input" in raw_data:
                    input_msg = self._parse_input_to_message(raw_data["input"])
                    if input_msg:
                        messages.append(input_msg)
                
                if "output" in raw_data:
                    output_msg = self._parse_output_to_message(raw_data["output"])
                    if output_msg:
                        messages.append(output_msg)
            
            return messages
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract messages: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def _extract_messages_from_generation(self, generation: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from a Langfuse generation observation.
        
        Args:
            generation: Generation observation dict
            
        Returns:
            List of Message objects
        """
        messages = []
        timestamp = self._parse_timestamp(generation.get("start_time"))
        
        # Extract input (user message or prompt)
        if "input" in generation:
            input_msg = self._parse_input_to_message(generation["input"], timestamp)
            if input_msg:
                messages.append(input_msg)
        
        # Extract output (assistant response)
        if "output" in generation:
            output_msg = self._parse_output_to_message(generation["output"], timestamp)
            if output_msg:
                messages.append(output_msg)
        
        return messages
    
    def _extract_messages_from_span(self, span: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from a Langfuse span observation.
        
        Spans may contain input/output that represents messages.
        
        Args:
            span: Span observation dict
            
        Returns:
            List of Message objects
        """
        messages = []
        timestamp = self._parse_timestamp(span.get("start_time"))
        
        # Check if span has message-like input/output
        if "input" in span:
            input_msg = self._parse_input_to_message(span["input"], timestamp)
            if input_msg:
                messages.append(input_msg)
        
        if "output" in span:
            output_msg = self._parse_output_to_message(span["output"], timestamp)
            if output_msg:
                messages.append(output_msg)
        
        return messages
    
    def _parse_input_to_message(
        self,
        input_data: Any,
        timestamp: Optional[datetime] = None
    ) -> Optional[Message]:
        """
        Parse input data to a Message object.
        
        Input can be:
        - String: Direct message content
        - Dict with "content" or "message" key
        - List of message objects
        
        Args:
            input_data: Input data from observation
            timestamp: Optional timestamp for the message
            
        Returns:
            Message object or None
        """
        if not input_data:
            return None
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Handle string input
        if isinstance(input_data, str):
            return Message(
                role=MessageRole.USER,
                content=input_data,
                timestamp=timestamp
            )
        
        # Handle dict input
        if isinstance(input_data, dict):
            # Check for direct content
            content = input_data.get("content") or input_data.get("message") or input_data.get("text")
            if content:
                role = self._parse_role(input_data.get("role", "user"))
                return Message(
                    role=role,
                    content=str(content),
                    timestamp=timestamp
                )
            
            # Check for messages array
            if "messages" in input_data and isinstance(input_data["messages"], list):
                # Return first message (others will be handled separately)
                for msg in input_data["messages"]:
                    if isinstance(msg, dict) and msg.get("content"):
                        role = self._parse_role(msg.get("role", "user"))
                        return Message(
                            role=role,
                            content=str(msg["content"]),
                            timestamp=timestamp
                        )
        
        # Handle list input (take first message)
        if isinstance(input_data, list) and input_data:
            first_item = input_data[0]
            if isinstance(first_item, dict) and first_item.get("content"):
                role = self._parse_role(first_item.get("role", "user"))
                return Message(
                    role=role,
                    content=str(first_item["content"]),
                    timestamp=timestamp
                )
        
        return None
    
    def _parse_output_to_message(
        self,
        output_data: Any,
        timestamp: Optional[datetime] = None
    ) -> Optional[Message]:
        """
        Parse output data to a Message object.
        
        Output can be:
        - String: Direct message content
        - Dict with "content", "message", or "text" key
        - Dict with "choices" array (OpenAI format)
        
        Args:
            output_data: Output data from observation
            timestamp: Optional timestamp for the message
            
        Returns:
            Message object or None
        """
        if not output_data:
            return None
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Handle string output
        if isinstance(output_data, str):
            return Message(
                role=MessageRole.ASSISTANT,
                content=output_data,
                timestamp=timestamp
            )
        
        # Handle dict output
        if isinstance(output_data, dict):
            # Check for direct content
            content = output_data.get("content") or output_data.get("message") or output_data.get("text")
            if content:
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=str(content),
                    timestamp=timestamp
                )
            
            # Check for OpenAI-style choices
            if "choices" in output_data and isinstance(output_data["choices"], list):
                if output_data["choices"]:
                    first_choice = output_data["choices"][0]
                    if isinstance(first_choice, dict):
                        message_data = first_choice.get("message", {})
                        if isinstance(message_data, dict):
                            content = message_data.get("content")
                            if content:
                                return Message(
                                    role=MessageRole.ASSISTANT,
                                    content=str(content),
                                    timestamp=timestamp
                                )
        
        return None
    
    def _parse_role(self, role_str: str) -> MessageRole:
        """
        Parse role string to MessageRole enum.
        
        Args:
            role_str: Role string from Langfuse
            
        Returns:
            MessageRole enum value
        """
        role_lower = str(role_str).lower()
        
        if role_lower in ["user", "human"]:
            return MessageRole.USER
        elif role_lower in ["assistant", "ai", "bot"]:
            return MessageRole.ASSISTANT
        elif role_lower in ["system"]:
            return MessageRole.SYSTEM
        elif role_lower in ["tool", "function"]:
            return MessageRole.TOOL
        else:
            # Default to assistant for unknown types
            return MessageRole.ASSISTANT
    
    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract tool calls from Langfuse observations.
        
        Tool calls are extracted from:
        1. Span observations with type "tool" or name containing "tool"
        2. Event observations representing function calls
        3. Generation observations with function_call metadata
        
        Args:
            raw_data: Langfuse trace data
            
        Returns:
            List of ToolCall objects in chronological order
            
        Raises:
            AdapterTransformationError: If tool call extraction fails
        """
        try:
            tool_calls = []
            
            # Get observations list (handle both direct and nested formats)
            observations = raw_data.get("observations", [])
            if not observations and "data" in raw_data:
                observations = raw_data["data"]
            
            # Sort observations by timestamp
            sorted_observations = self._sort_observations_by_time(observations)
            
            # Extract tool calls from each observation
            for obs in sorted_observations:
                if not isinstance(obs, dict):
                    continue
                
                obs_type = obs.get("type", "").lower()
                obs_name = obs.get("name", "").lower()
                
                # Check if this is a tool-related observation
                is_tool_span = obs_type == "span" and ("tool" in obs_name or "function" in obs_name)
                is_tool_event = obs_type == "event" and ("tool" in obs_name or "function" in obs_name)
                
                if is_tool_span or is_tool_event:
                    tool_call = self._parse_observation_to_tool_call(obs)
                    if tool_call:
                        tool_calls.append(tool_call)
                
                # Check for function calls in generation metadata
                elif obs_type == "generation":
                    metadata = obs.get("metadata", {})
                    if isinstance(metadata, dict) and "function_call" in metadata:
                        tool_call = self._parse_function_call_to_tool_call(
                            metadata["function_call"],
                            obs
                        )
                        if tool_call:
                            tool_calls.append(tool_call)
            
            return tool_calls
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract tool calls: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def _parse_observation_to_tool_call(self, observation: Dict[str, Any]) -> Optional[ToolCall]:
        """
        Parse a Langfuse observation to a ToolCall object.
        
        Args:
            observation: Observation dict from Langfuse
            
        Returns:
            ToolCall object or None
        """
        try:
            # Extract tool name from observation name
            tool_name = observation.get("name", "")
            
            # Clean up tool name (remove prefixes like "tool_", "function_")
            for prefix in ["tool_", "function_", "call_"]:
                if tool_name.lower().startswith(prefix):
                    tool_name = tool_name[len(prefix):]
            
            if not tool_name:
                return None
            
            # Extract arguments from input
            arguments = {}
            if "input" in observation:
                input_data = observation["input"]
                if isinstance(input_data, dict):
                    arguments = input_data
                elif isinstance(input_data, str):
                    # Try to parse as JSON
                    try:
                        import json
                        arguments = json.loads(input_data)
                    except (json.JSONDecodeError, ValueError):
                        arguments = {"input": input_data}
            
            # Extract result from output
            result = None
            if "output" in observation:
                result = observation["output"]
            
            # Calculate execution time if available
            execution_time = None
            if "start_time" in observation and "end_time" in observation:
                start = self._parse_timestamp(observation["start_time"])
                end = self._parse_timestamp(observation["end_time"])
                if start and end:
                    execution_time = (end - start).total_seconds()
            
            # Get timestamp
            timestamp = self._parse_timestamp(observation.get("start_time"))
            
            return ToolCall(
                name=tool_name,
                arguments=arguments,
                result=result,
                execution_time=execution_time,
                timestamp=timestamp
            )
            
        except Exception:
            return None
    
    def _parse_function_call_to_tool_call(
        self,
        function_call: Dict[str, Any],
        observation: Dict[str, Any]
    ) -> Optional[ToolCall]:
        """
        Parse a function call from generation metadata to a ToolCall object.
        
        Args:
            function_call: Function call dict from metadata
            observation: Parent observation for context
            
        Returns:
            ToolCall object or None
        """
        try:
            tool_name = function_call.get("name", "")
            if not tool_name:
                return None
            
            # Parse arguments
            arguments = function_call.get("arguments", {})
            if isinstance(arguments, str):
                # Try to parse as JSON
                try:
                    import json
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    arguments = {"arguments": arguments}
            
            # Get timestamp from observation
            timestamp = self._parse_timestamp(observation.get("start_time"))
            
            return ToolCall(
                name=tool_name,
                arguments=arguments,
                timestamp=timestamp
            )
            
        except Exception:
            return None
    
    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from Langfuse observations.
        
        Metadata includes:
        - input_tokens: Total input tokens from generation usage
        - output_tokens: Total output tokens from generation usage
        - latency: Total execution time from span durations
        - cost: Total cost from generation cost data
        - model: Model name from generations
        
        Args:
            raw_data: Langfuse trace data
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterTransformationError: If metadata extraction fails
        """
        try:
            metadata = {}
            
            # Get observations list (handle both direct and nested formats)
            observations = raw_data.get("observations", [])
            if not observations and "data" in raw_data:
                observations = raw_data["data"]
            
            # Accumulate token usage and cost
            input_tokens = 0
            output_tokens = 0
            total_cost = 0.0
            total_latency = 0.0
            model_names = set()
            
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                
                # Extract token usage from generations
                if obs.get("type") == "generation":
                    usage = obs.get("usage", {})
                    if isinstance(usage, dict):
                        input_tokens += usage.get("input", 0) or usage.get("promptTokens", 0)
                        output_tokens += usage.get("output", 0) or usage.get("completionTokens", 0)
                    
                    # Extract cost
                    if "cost" in obs and obs["cost"] is not None:
                        total_cost += float(obs["cost"])
                    
                    # Extract model name
                    if "model" in obs and obs["model"]:
                        model_names.add(obs["model"])
                
                # Calculate latency from spans
                if obs.get("type") == "span":
                    if "start_time" in obs and "end_time" in obs:
                        start = self._parse_timestamp(obs["start_time"])
                        end = self._parse_timestamp(obs["end_time"])
                        if start and end:
                            duration = (end - start).total_seconds()
                            total_latency += duration
            
            # Add aggregated values to metadata
            if input_tokens > 0:
                metadata["input_tokens"] = input_tokens
            if output_tokens > 0:
                metadata["output_tokens"] = output_tokens
            if total_cost > 0:
                metadata["cost"] = total_cost
            if total_latency > 0:
                metadata["latency"] = total_latency
            if model_names:
                metadata["model"] = ", ".join(sorted(model_names))
            
            # Add trace-level metadata
            if "metadata" in raw_data and isinstance(raw_data["metadata"], dict):
                # Merge trace metadata, but don't overwrite calculated values
                for key, value in raw_data["metadata"].items():
                    if key not in metadata:
                        metadata[key] = value
            
            # Add trace name if present
            if "name" in raw_data:
                metadata["trace_name"] = raw_data["name"]
            
            # Add user information if present
            if "user_id" in raw_data:
                metadata["user_id"] = raw_data["user_id"]
            
            # Add tags if present
            if "tags" in raw_data:
                metadata["tags"] = raw_data["tags"]
            
            return metadata
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract metadata: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def _sort_observations_by_time(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort observations by start time in chronological order.
        
        Args:
            observations: List of observation dicts
            
        Returns:
            Sorted list of observations
        """
        def get_sort_key(obs: Dict[str, Any]) -> datetime:
            """Get sort key (start time) for an observation."""
            timestamp = self._parse_timestamp(obs.get("start_time"))
            return timestamp if timestamp else datetime.min
        
        try:
            return sorted(observations, key=get_sort_key)
        except Exception:
            # If sorting fails, return original list
            return observations
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """
        Parse ISO 8601 timestamp string to datetime object.
        
        Args:
            timestamp_str: ISO 8601 timestamp string
            
        Returns:
            datetime object or current time if parsing fails
        """
        if not timestamp_str:
            return datetime.utcnow()
        
        try:
            # Handle ISO 8601 format with timezone
            if isinstance(timestamp_str, str):
                # Remove timezone suffix for parsing
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1]
                
                # Try parsing with microseconds
                try:
                    return datetime.fromisoformat(timestamp_str)
                except ValueError:
                    # Try without microseconds
                    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
            
            return datetime.utcnow()
            
        except Exception:
            return datetime.utcnow()
