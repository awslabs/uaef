# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain adapter for transforming LangChain traces to canonical format."""

from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

from uaef.adapters.base import AdapterTransformationError, BaseAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message, MessageRole
from uaef.models.tool_call import ToolCall


class LangChainAdapter(BaseAdapter):
    """
    Adapter for transforming LangChain callback data to canonical AgentTrace format.
    
    This adapter processes LangChain execution traces that contain:
    - Callback handler data from LangChain runs
    - Messages from chat models and agents
    - Tool invocations and results
    - Token usage metadata
    - Run information and metadata
    
    LangChain provides callback handlers that capture execution details.
    This adapter transforms that data into the canonical UAEF format.
    """
    
    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "langchain"
    
    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """
        Transform LangChain trace data to canonical AgentTrace format.
        
        Expected raw_data structure:
        {
            "runs": [...],  # List of run data from callback handlers
            "messages": [...],  # Optional: Direct message list
            "session_id": "...",  # Optional session identifier
            "trace_id": "...",  # Optional trace identifier (UUID string)
            "metadata": {...},  # Optional additional metadata
        }
        
        Alternative structure (single run):
        {
            "run_id": "...",
            "inputs": {...},
            "outputs": {...},
            "actions": [...],  # Tool calls/actions
            "llm_output": {...},  # LLM response with token usage
            "messages": [...],  # Chat messages
            "metadata": {...},
        }
        
        Args:
            raw_data: LangChain trace data
            
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
                framework="langchain",
                timestamp=datetime.utcnow()
            )
            
            return trace
            
        except AdapterTransformationError:
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform LangChain trace: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from LangChain callback data.
        
        Messages can come from:
        1. Direct "messages" field (list of message objects)
        2. "runs" field containing run data with inputs/outputs
        3. "inputs" and "outputs" fields in a single run
        
        LangChain messages may have different formats:
        - HumanMessage, AIMessage, SystemMessage, ToolMessage
        - Dict with "role" and "content" keys
        - String content with role inferred from context
        
        Args:
            raw_data: LangChain trace data
            
        Returns:
            List of Message objects
            
        Raises:
            AdapterTransformationError: If message extraction fails
        """
        try:
            messages = []
            
            # Case 1: Direct messages field
            if "messages" in raw_data and raw_data["messages"]:
                messages.extend(self._parse_message_list(raw_data["messages"]))
            
            # Case 2: Runs field with multiple runs
            elif "runs" in raw_data and raw_data["runs"]:
                for run in raw_data["runs"]:
                    if not isinstance(run, dict):
                        continue
                    
                    # Extract messages from run inputs
                    if "inputs" in run:
                        input_messages = self._extract_messages_from_inputs(run["inputs"])
                        messages.extend(input_messages)
                    
                    # Extract messages from run outputs
                    if "outputs" in run:
                        output_messages = self._extract_messages_from_outputs(run["outputs"])
                        messages.extend(output_messages)
            
            # Case 3: Single run with inputs/outputs
            elif "inputs" in raw_data or "outputs" in raw_data:
                if "inputs" in raw_data:
                    input_messages = self._extract_messages_from_inputs(raw_data["inputs"])
                    messages.extend(input_messages)
                
                if "outputs" in raw_data:
                    output_messages = self._extract_messages_from_outputs(raw_data["outputs"])
                    messages.extend(output_messages)
            
            return messages
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract messages: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def _parse_message_list(self, messages_data: List[Any]) -> List[Message]:
        """
        Parse a list of LangChain message objects.
        
        Args:
            messages_data: List of message objects or dicts
            
        Returns:
            List of Message objects
        """
        messages = []
        
        for msg in messages_data:
            if not msg:
                continue
            
            # Handle dict format
            if isinstance(msg, dict):
                role = self._parse_role(msg.get("role", msg.get("type", "assistant")))
                content = msg.get("content", "")
                tool_calls_data = msg.get("tool_calls", [])
                
                # Parse tool calls if present
                message_tool_calls = []
                if tool_calls_data:
                    for tc in tool_calls_data:
                        if isinstance(tc, dict):
                            message_tool_calls.append(
                                ToolCall(
                                    name=tc.get("name", tc.get("function", {}).get("name", "")),
                                    arguments=tc.get("args", tc.get("function", {}).get("arguments", {})),
                                    timestamp=datetime.utcnow()
                                )
                            )
                
                messages.append(
                    Message(
                        role=role,
                        content=str(content),
                        tool_calls=message_tool_calls,
                        timestamp=datetime.utcnow()
                    )
                )
            
            # Handle LangChain message objects (with type attribute)
            elif hasattr(msg, "type"):
                role = self._parse_role(msg.type)
                content = getattr(msg, "content", "")
                
                # Extract tool calls if present
                message_tool_calls = []
                if hasattr(msg, "tool_calls"):
                    tool_calls_data = getattr(msg, "tool_calls", [])
                    for tc in tool_calls_data:
                        if isinstance(tc, dict):
                            message_tool_calls.append(
                                ToolCall(
                                    name=tc.get("name", ""),
                                    arguments=tc.get("args", {}),
                                    timestamp=datetime.utcnow()
                                )
                            )
                
                messages.append(
                    Message(
                        role=role,
                        content=str(content),
                        tool_calls=message_tool_calls,
                        timestamp=datetime.utcnow()
                    )
                )
        
        return messages
    
    def _extract_messages_from_inputs(self, inputs: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from run inputs.
        
        Args:
            inputs: Input dictionary from a run
            
        Returns:
            List of Message objects
        """
        messages = []
        
        # Check for direct message content
        if "input" in inputs:
            input_content = inputs["input"]
            if isinstance(input_content, str):
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=input_content,
                        timestamp=datetime.utcnow()
                    )
                )
            elif isinstance(input_content, list):
                messages.extend(self._parse_message_list(input_content))
        
        # Check for messages field in inputs
        if "messages" in inputs:
            messages.extend(self._parse_message_list(inputs["messages"]))
        
        # Check for question field (common in QA chains)
        if "question" in inputs and isinstance(inputs["question"], str):
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=inputs["question"],
                    timestamp=datetime.utcnow()
                )
            )
        
        return messages
    
    def _extract_messages_from_outputs(self, outputs: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from run outputs.
        
        Args:
            outputs: Output dictionary from a run
            
        Returns:
            List of Message objects
        """
        messages = []
        
        # Check for direct output content
        if "output" in outputs:
            output_content = outputs["output"]
            if isinstance(output_content, str):
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=output_content,
                        timestamp=datetime.utcnow()
                    )
                )
            elif isinstance(output_content, list):
                messages.extend(self._parse_message_list(output_content))
        
        # Check for messages field in outputs
        if "messages" in outputs:
            messages.extend(self._parse_message_list(outputs["messages"]))
        
        # Check for answer field (common in QA chains)
        if "answer" in outputs and isinstance(outputs["answer"], str):
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=outputs["answer"],
                    timestamp=datetime.utcnow()
                )
            )
        
        # Check for text field
        if "text" in outputs and isinstance(outputs["text"], str):
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=outputs["text"],
                    timestamp=datetime.utcnow()
                )
            )
        
        return messages
    
    def _parse_role(self, role_str: str) -> MessageRole:
        """
        Parse role string to MessageRole enum.
        
        Args:
            role_str: Role string from LangChain
            
        Returns:
            MessageRole enum value
        """
        role_lower = str(role_str).lower()
        
        # Map LangChain role types to MessageRole
        if role_lower in ["human", "user", "humanmessage"]:
            return MessageRole.USER
        elif role_lower in ["ai", "assistant", "aimessage"]:
            return MessageRole.ASSISTANT
        elif role_lower in ["system", "systemmessage"]:
            return MessageRole.SYSTEM
        elif role_lower in ["tool", "function", "toolmessage", "functionmessage"]:
            return MessageRole.TOOL
        else:
            # Default to assistant for unknown types
            return MessageRole.ASSISTANT
    
    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract tool calls from LangChain callback data.
        
        Tool calls can come from:
        1. "actions" field in agent execution
        2. "tool_calls" in messages
        3. "intermediate_steps" in agent runs
        
        Args:
            raw_data: LangChain trace data
            
        Returns:
            List of ToolCall objects
            
        Raises:
            AdapterTransformationError: If tool call extraction fails
        """
        try:
            tool_calls = []
            
            # Case 1: Direct actions field
            if "actions" in raw_data and raw_data["actions"]:
                for action in raw_data["actions"]:
                    if not isinstance(action, dict):
                        continue
                    
                    tool_call = self._parse_action_to_tool_call(action)
                    if tool_call:
                        tool_calls.append(tool_call)
            
            # Case 2: Intermediate steps (agent execution)
            if "intermediate_steps" in raw_data and raw_data["intermediate_steps"]:
                for step in raw_data["intermediate_steps"]:
                    if not isinstance(step, (list, tuple)) or len(step) < 2:
                        continue
                    
                    # Step format: (action, observation)
                    action = step[0]
                    observation = step[1] if len(step) > 1 else None
                    
                    tool_call = self._parse_action_to_tool_call(action, observation)
                    if tool_call:
                        tool_calls.append(tool_call)
            
            # Case 3: Runs with actions
            if "runs" in raw_data and raw_data["runs"]:
                for run in raw_data["runs"]:
                    if not isinstance(run, dict):
                        continue
                    
                    # Check for actions in run
                    if "actions" in run:
                        for action in run["actions"]:
                            tool_call = self._parse_action_to_tool_call(action)
                            if tool_call:
                                tool_calls.append(tool_call)
                    
                    # Check for intermediate_steps in run
                    if "intermediate_steps" in run:
                        for step in run["intermediate_steps"]:
                            if isinstance(step, (list, tuple)) and len(step) >= 2:
                                action = step[0]
                                observation = step[1]
                                tool_call = self._parse_action_to_tool_call(action, observation)
                                if tool_call:
                                    tool_calls.append(tool_call)
            
            # Case 4: Extract from messages with tool_calls
            messages = raw_data.get("messages", [])
            if messages:
                for msg in messages:
                    if isinstance(msg, dict) and "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            if isinstance(tc, dict):
                                tool_calls.append(
                                    ToolCall(
                                        name=tc.get("name", tc.get("function", {}).get("name", "")),
                                        arguments=tc.get("args", tc.get("function", {}).get("arguments", {})),
                                        result=tc.get("result"),
                                        timestamp=datetime.utcnow()
                                    )
                                )
            
            return tool_calls
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract tool calls: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def _parse_action_to_tool_call(
        self,
        action: Any,
        observation: Any = None
    ) -> ToolCall:
        """
        Parse a LangChain action to a ToolCall object.
        
        Args:
            action: Action object or dict from LangChain
            observation: Optional observation/result from tool execution
            
        Returns:
            ToolCall object or None if parsing fails
        """
        try:
            # Handle dict format
            if isinstance(action, dict):
                tool_name = action.get("tool", action.get("name", ""))
                tool_input = action.get("tool_input", action.get("args", {}))
                
                # Ensure tool_input is a dict
                if not isinstance(tool_input, dict):
                    tool_input = {"input": tool_input}
                
                return ToolCall(
                    name=tool_name,
                    arguments=tool_input,
                    result=observation,
                    timestamp=datetime.utcnow()
                )
            
            # Handle object format (AgentAction)
            elif hasattr(action, "tool"):
                tool_name = getattr(action, "tool", "")
                tool_input = getattr(action, "tool_input", {})
                
                # Ensure tool_input is a dict
                if not isinstance(tool_input, dict):
                    tool_input = {"input": tool_input}
                
                return ToolCall(
                    name=tool_name,
                    arguments=tool_input,
                    result=observation,
                    timestamp=datetime.utcnow()
                )
            
            return None
            
        except Exception:
            return None
    
    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from LangChain callback data.
        
        Metadata includes:
        - input_tokens: Total input tokens from llm_output
        - output_tokens: Total output tokens from llm_output
        - latency: Execution time (if provided)
        - model: Model name (if available)
        - run_id: LangChain run identifier
        
        Args:
            raw_data: LangChain trace data
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterTransformationError: If metadata extraction fails
        """
        try:
            metadata = {}
            
            input_tokens = 0
            output_tokens = 0
            
            # Extract from llm_output field
            if "llm_output" in raw_data and isinstance(raw_data["llm_output"], dict):
                llm_output = raw_data["llm_output"]
                
                # Check for token_usage
                if "token_usage" in llm_output:
                    token_usage = llm_output["token_usage"]
                    if isinstance(token_usage, dict):
                        input_tokens += token_usage.get("prompt_tokens", 0)
                        output_tokens += token_usage.get("completion_tokens", 0)
                
                # Check for model_name
                if "model_name" in llm_output:
                    metadata["model"] = llm_output["model_name"]
            
            # Extract from runs
            if "runs" in raw_data and raw_data["runs"]:
                for run in raw_data["runs"]:
                    if not isinstance(run, dict):
                        continue
                    
                    # Extract token usage from run
                    if "llm_output" in run and isinstance(run["llm_output"], dict):
                        llm_output = run["llm_output"]
                        if "token_usage" in llm_output:
                            token_usage = llm_output["token_usage"]
                            if isinstance(token_usage, dict):
                                input_tokens += token_usage.get("prompt_tokens", 0)
                                output_tokens += token_usage.get("completion_tokens", 0)
            
            # Add token counts to metadata
            if input_tokens > 0:
                metadata["input_tokens"] = input_tokens
            if output_tokens > 0:
                metadata["output_tokens"] = output_tokens
            
            # Add run_id if present
            if "run_id" in raw_data:
                metadata["run_id"] = str(raw_data["run_id"])
            
            # Add latency if provided
            if "latency" in raw_data:
                metadata["latency"] = raw_data["latency"]
            
            # Add model if provided directly
            if "model" in raw_data:
                metadata["model"] = raw_data["model"]
            
            # Add any additional metadata from raw_data
            if "metadata" in raw_data and isinstance(raw_data["metadata"], dict):
                metadata.update(raw_data["metadata"])
            
            return metadata
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract metadata: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
