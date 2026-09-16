# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bedrock Agent adapter for transforming Bedrock Agent traces to canonical format."""

from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

from uaef.adapters.base import AdapterTransformationError, BaseAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message, MessageRole
from uaef.models.tool_call import ToolCall


class BedrockAgentAdapter(BaseAdapter):
    """
    Adapter for transforming Bedrock Agent invoke_agent responses to canonical AgentTrace format.
    
    This adapter processes Bedrock Agent execution traces that contain:
    - Event stream from invoke_agent response
    - Chunk events with response text
    - Trace events with orchestration information
    - Tool invocation details
    - Token usage metadata
    
    The adapter reuses logic from the get_answers_detail_bedrock() function
    in agenticevaluationframework-main/frontend/evaluation.py.
    """
    
    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "bedrock"
    
    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """
        Transform Bedrock Agent response data to canonical AgentTrace format.
        
        Expected raw_data structure:
        {
            "event_stream": [...],  # List of events from invoke_agent response
            "user_input": "...",  # Original user input
            "session_id": "...",  # Session identifier
            "trace_id": "...",  # Optional trace identifier (UUID string)
            "agent_id": "...",  # Bedrock agent ID
            "alias_id": "...",  # Agent alias ID
        }
        
        Args:
            raw_data: Bedrock Agent response data
            
        Returns:
            Canonical AgentTrace object
            
        Raises:
            AdapterTransformationError: If transformation fails
        """
        try:
            # Validate required fields
            self.validate_raw_data(raw_data, ["event_stream"])
            
            # Extract components
            messages = self.extract_messages(raw_data)
            tool_calls = self.extract_tool_calls(raw_data)
            metadata = self.extract_metadata(raw_data)
            
            # Create AgentTrace
            trace = AgentTrace(
                trace_id=raw_data.get("trace_id", uuid4()),
                session_id=raw_data.get("session_id"),
                messages=messages,
                tool_calls=tool_calls,
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                latency=metadata.get("latency"),
                cost=metadata.get("cost"),
                metadata=metadata,
                framework="bedrock",
                timestamp=datetime.utcnow()
            )
            
            return trace
            
        except AdapterTransformationError:
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform Bedrock Agent trace: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from Bedrock Agent event stream.
        
        Messages are extracted from:
        1. User input (if provided)
        2. Chunk events containing response text
        
        Args:
            raw_data: Bedrock Agent response data
            
        Returns:
            List of Message objects
            
        Raises:
            AdapterTransformationError: If message extraction fails
        """
        try:
            messages = []
            
            # Add user input message if provided
            user_input = raw_data.get("user_input")
            if user_input:
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=user_input,
                        timestamp=datetime.utcnow()
                    )
                )
            
            # Extract response chunks from event stream
            event_stream = raw_data.get("event_stream", [])
            response_chunks = []
            
            for event in event_stream:
                if not isinstance(event, dict):
                    continue
                
                # Check for chunk events containing response text
                if "chunk" in event:
                    chunk_data = event["chunk"]
                    if isinstance(chunk_data, dict) and "bytes" in chunk_data:
                        # Decode bytes to string
                        try:
                            text = chunk_data["bytes"].decode("utf-8")
                            response_chunks.append(text)
                        except (AttributeError, UnicodeDecodeError):
                            # If bytes is already a string or decoding fails
                            response_chunks.append(str(chunk_data["bytes"]))
            
            # Combine response chunks into a single assistant message
            if response_chunks:
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content="".join(response_chunks),
                        timestamp=datetime.utcnow()
                    )
                )
            
            return messages
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract messages: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract tool calls from Bedrock Agent trace events.
        
        Tool calls are extracted from orchestrationTrace events that contain
        invocationInput with actionGroupInvocationInput information.
        
        Args:
            raw_data: Bedrock Agent response data
            
        Returns:
            List of ToolCall objects
            
        Raises:
            AdapterTransformationError: If tool call extraction fails
        """
        try:
            tool_calls = []
            event_stream = raw_data.get("event_stream", [])
            
            for event in event_stream:
                if not isinstance(event, dict):
                    continue
                
                # Check for trace events
                if "trace" in event:
                    trace_data = event["trace"]
                    if not isinstance(trace_data, dict):
                        continue
                    
                    # Navigate to orchestrationTrace
                    trace_info = trace_data.get("trace", {})
                    if not isinstance(trace_info, dict):
                        continue
                    
                    orchestration_trace = trace_info.get("orchestrationTrace", {})
                    if not isinstance(orchestration_trace, dict):
                        continue
                    
                    # Extract tool invocation information
                    if "invocationInput" in orchestration_trace:
                        invocation_input = orchestration_trace["invocationInput"]
                        if not isinstance(invocation_input, dict):
                            continue
                        
                        action_group_input = invocation_input.get("actionGroupInvocationInput")
                        if isinstance(action_group_input, dict):
                            # Extract tool name
                            tool_name = action_group_input.get("function", "")
                            
                            # Extract tool parameters
                            parameters = action_group_input.get("parameters", [])
                            tool_args = {}
                            
                            if isinstance(parameters, list):
                                for param in parameters:
                                    if isinstance(param, dict):
                                        param_name = param.get("name")
                                        param_value = param.get("value")
                                        if param_name:
                                            tool_args[param_name] = param_value
                            
                            # Create ToolCall object
                            if tool_name:
                                tool_call = ToolCall(
                                    name=tool_name,
                                    arguments=tool_args,
                                    timestamp=datetime.utcnow()
                                )
                                tool_calls.append(tool_call)
            
            return tool_calls
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract tool calls: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from Bedrock Agent trace events.
        
        Metadata includes:
        - input_tokens: Total input tokens from modelInvocationOutput
        - output_tokens: Total output tokens from modelInvocationOutput
        - latency: Execution time (if provided)
        - agent_id: Bedrock agent ID
        - alias_id: Agent alias ID
        
        Args:
            raw_data: Bedrock Agent response data
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterTransformationError: If metadata extraction fails
        """
        try:
            metadata = {}
            event_stream = raw_data.get("event_stream", [])
            
            input_tokens = 0
            output_tokens = 0
            
            # Accumulate token usage from trace events
            for event in event_stream:
                if not isinstance(event, dict):
                    continue
                
                # Check for trace events with token usage
                if "trace" in event:
                    trace_data = event["trace"]
                    if not isinstance(trace_data, dict):
                        continue
                    
                    # Navigate to orchestrationTrace
                    trace_info = trace_data.get("trace", {})
                    if not isinstance(trace_info, dict):
                        continue
                    
                    orchestration_trace = trace_info.get("orchestrationTrace", {})
                    if not isinstance(orchestration_trace, dict):
                        continue
                    
                    # Extract token usage from modelInvocationOutput
                    if "modelInvocationOutput" in orchestration_trace:
                        model_output = orchestration_trace["modelInvocationOutput"]
                        if isinstance(model_output, dict):
                            metadata_info = model_output.get("metadata", {})
                            if isinstance(metadata_info, dict):
                                usage = metadata_info.get("usage", {})
                                if isinstance(usage, dict):
                                    input_tokens += usage.get("inputTokens", 0)
                                    output_tokens += usage.get("outputTokens", 0)
            
            # Add token counts to metadata
            if input_tokens > 0:
                metadata["input_tokens"] = input_tokens
            if output_tokens > 0:
                metadata["output_tokens"] = output_tokens
            
            # Add agent identifiers
            if "agent_id" in raw_data:
                metadata["agent_id"] = raw_data["agent_id"]
            if "alias_id" in raw_data:
                metadata["alias_id"] = raw_data["alias_id"]
            
            # Add latency if provided
            if "latency" in raw_data:
                metadata["latency"] = raw_data["latency"]
            
            return metadata
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract metadata: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
