# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic JSON adapter for transforming arbitrary JSON traces to canonical format."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import jsonpath_ng
from jsonpath_ng.ext import parse

from uaef.adapters.base import AdapterTransformationError, BaseAdapter
from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message, MessageRole
from uaef.models.tool_call import ToolCall


class GenericJSONAdapter(BaseAdapter):
    """
    Adapter for transforming arbitrary JSON traces to canonical AgentTrace format.
    
    This adapter uses schema mapping configuration to extract fields from
    JSON data with flexible structure. It supports JSONPath expressions for
    nested field extraction.
    
    Schema mapping format:
    {
        "messages": {
            "path": "$.conversation.messages[*]",
            "role_field": "sender",
            "content_field": "text",
            "timestamp_field": "created_at"
        },
        "tool_calls": {
            "path": "$.execution.tools[*]",
            "name_field": "tool_name",
            "arguments_field": "params",
            "result_field": "output",
            "timestamp_field": "executed_at"
        },
        "metadata": {
            "input_tokens": "$.usage.input",
            "output_tokens": "$.usage.output",
            "latency": "$.performance.duration",
            "model": "$.config.model_name"
        },
        "trace_id": "$.trace.id",
        "session_id": "$.session.id"
    }
    """
    
    DEFAULT_SCHEMA_MAPPING: Dict[str, Any] = {
        "messages": {
            "path": "$.messages[*]",
            "role_field": "role",
            "content_field": "content",
            "timestamp_field": "timestamp",
        },
        "tool_calls": {
            "path": "$.tool_calls[*]",
            "name_field": "name",
            "arguments_field": "arguments",
            "result_field": "result",
        },
        "metadata": {
            "input_tokens": "$.usage.input_tokens",
            "output_tokens": "$.usage.output_tokens",
            "latency": "$.latency",
            "model": "$.model",
        },
        "trace_id": "$.trace_id",
        "session_id": "$.session_id",
    }

    def __init__(self, schema_mapping: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with schema mapping configuration.

        Args:
            schema_mapping: Dictionary defining how to extract fields from JSON.
                If None, a default mapping is used that expects common field names
                (messages, tool_calls, usage, etc.).
        """
        self.schema_mapping = schema_mapping or self.DEFAULT_SCHEMA_MAPPING
    
    @property
    def name(self) -> str:
        """Return the adapter name."""
        return "generic_json"
    
    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> AgentTrace:
        """
        Transform generic JSON data to canonical AgentTrace format.
        
        Args:
            raw_data: Generic JSON trace data
            
        Returns:
            Canonical AgentTrace object
            
        Raises:
            AdapterTransformationError: If transformation fails
        """
        try:
            # Extract components using schema mapping
            messages = self.extract_messages(raw_data)
            tool_calls = self.extract_tool_calls(raw_data)
            metadata = self.extract_metadata(raw_data)
            
            # Extract trace_id and session_id
            trace_id = self._extract_field(raw_data, self.schema_mapping.get("trace_id"))
            if trace_id and isinstance(trace_id, str):
                try:
                    from uuid import UUID
                    trace_id = UUID(trace_id)
                except (ValueError, AttributeError):
                    trace_id = uuid4()
            else:
                trace_id = uuid4()
            
            session_id = self._extract_field(raw_data, self.schema_mapping.get("session_id"))
            if session_id:
                session_id = str(session_id)
            
            # Create AgentTrace
            trace = AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                messages=messages,
                tool_calls=tool_calls,
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                latency=metadata.get("latency"),
                cost=metadata.get("cost"),
                metadata=metadata,
                framework=metadata.get("framework", "generic"),
                timestamp=datetime.utcnow()
            )
            
            return trace
            
        except AdapterTransformationError:
            raise
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to transform generic JSON trace: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from generic JSON using schema mapping.
        
        Args:
            raw_data: Generic JSON trace data
            
        Returns:
            List of Message objects
            
        Raises:
            AdapterTransformationError: If message extraction fails
        """
        try:
            messages = []
            messages_config = self.schema_mapping.get("messages", {})
            
            if not messages_config:
                return messages
            
            # Extract message objects using JSONPath
            messages_path = messages_config.get("path")
            if not messages_path:
                return messages
            
            message_objects = self._extract_field(raw_data, messages_path, multiple=True)
            if not message_objects:
                return messages
            
            # Extract fields from each message object
            role_field = messages_config.get("role_field", "role")
            content_field = messages_config.get("content_field", "content")
            timestamp_field = messages_config.get("timestamp_field")
            
            for msg_obj in message_objects:
                if not isinstance(msg_obj, dict):
                    continue
                
                # Extract role
                role_str = msg_obj.get(role_field, "assistant")
                role = self._parse_role(role_str)
                
                # Extract content
                content = msg_obj.get(content_field, "")
                
                # Extract timestamp
                timestamp = datetime.utcnow()
                if timestamp_field and timestamp_field in msg_obj:
                    timestamp = self._parse_timestamp(msg_obj[timestamp_field])
                
                # Create message
                message = Message(
                    role=role,
                    content=str(content),
                    timestamp=timestamp
                )
                messages.append(message)
            
            return messages
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract messages: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract tool calls from generic JSON using schema mapping.
        
        Args:
            raw_data: Generic JSON trace data
            
        Returns:
            List of ToolCall objects
            
        Raises:
            AdapterTransformationError: If tool call extraction fails
        """
        try:
            tool_calls = []
            tool_calls_config = self.schema_mapping.get("tool_calls", {})
            
            if not tool_calls_config:
                return tool_calls
            
            # Extract tool call objects using JSONPath
            tool_calls_path = tool_calls_config.get("path")
            if not tool_calls_path:
                return tool_calls
            
            tool_call_objects = self._extract_field(raw_data, tool_calls_path, multiple=True)
            if not tool_call_objects:
                return tool_calls
            
            # Extract fields from each tool call object
            name_field = tool_calls_config.get("name_field", "name")
            arguments_field = tool_calls_config.get("arguments_field", "arguments")
            result_field = tool_calls_config.get("result_field", "result")
            timestamp_field = tool_calls_config.get("timestamp_field")
            execution_time_field = tool_calls_config.get("execution_time_field")
            error_field = tool_calls_config.get("error_field")
            
            for tc_obj in tool_call_objects:
                if not isinstance(tc_obj, dict):
                    continue
                
                # Extract tool name
                name = tc_obj.get(name_field, "")
                if not name:
                    continue
                
                # Extract arguments
                arguments = tc_obj.get(arguments_field, {})
                if not isinstance(arguments, dict):
                    arguments = {}
                
                # Extract result
                result = tc_obj.get(result_field) if result_field else None
                
                # Extract timestamp
                timestamp = datetime.utcnow()
                if timestamp_field and timestamp_field in tc_obj:
                    timestamp = self._parse_timestamp(tc_obj[timestamp_field])
                
                # Extract execution time
                execution_time = None
                if execution_time_field and execution_time_field in tc_obj:
                    try:
                        execution_time = float(tc_obj[execution_time_field])
                    except (ValueError, TypeError):
                        pass
                
                # Extract error
                error = tc_obj.get(error_field) if error_field else None
                if error:
                    error = str(error)
                
                # Create ToolCall object
                tool_call = ToolCall(
                    name=name,
                    arguments=arguments,
                    result=result,
                    timestamp=timestamp,
                    execution_time=execution_time,
                    error=error
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
        Extract metadata from generic JSON using schema mapping.
        
        Args:
            raw_data: Generic JSON trace data
            
        Returns:
            Dictionary of metadata
            
        Raises:
            AdapterTransformationError: If metadata extraction fails
        """
        try:
            metadata = {}
            metadata_config = self.schema_mapping.get("metadata", {})
            
            if not metadata_config:
                return metadata
            
            # Extract each metadata field
            for key, path in metadata_config.items():
                if not path:
                    continue
                
                value = self._extract_field(raw_data, path)
                if value is not None:
                    # Convert numeric fields
                    if key in ["input_tokens", "output_tokens"]:
                        try:
                            metadata[key] = int(value)
                        except (ValueError, TypeError):
                            pass
                    elif key in ["latency", "cost"]:
                        try:
                            metadata[key] = float(value)
                        except (ValueError, TypeError):
                            pass
                    else:
                        metadata[key] = value
            
            return metadata
            
        except Exception as e:
            raise AdapterTransformationError(
                f"Failed to extract metadata: {str(e)}",
                adapter_name=self.name,
                original_error=e
            )
    
    def _extract_field(
        self,
        data: Dict[str, Any],
        path: Optional[str],
        multiple: bool = False
    ) -> Any:
        """
        Extract a field from JSON data using JSONPath.
        
        Args:
            data: JSON data
            path: JSONPath expression
            multiple: If True, return all matches; if False, return first match
            
        Returns:
            Extracted value(s) or None if not found
        """
        if not path:
            return None
        
        try:
            # Parse JSONPath expression
            jsonpath_expr = parse(path)
            matches = jsonpath_expr.find(data)
            
            if not matches:
                return None
            
            if multiple:
                return [match.value for match in matches]
            else:
                return matches[0].value
                
        except Exception:
            # If JSONPath fails, try simple dict access
            if "." not in path and "$" not in path:
                return data.get(path)
            return None
    
    def _parse_role(self, role_str: Any) -> MessageRole:
        """
        Parse role string to MessageRole enum.
        
        Args:
            role_str: Role string or value
            
        Returns:
            MessageRole enum value
        """
        if isinstance(role_str, MessageRole):
            return role_str
        
        role_str = str(role_str).lower()
        
        if role_str in ["user", "human"]:
            return MessageRole.USER
        elif role_str in ["assistant", "ai", "agent", "bot"]:
            return MessageRole.ASSISTANT
        elif role_str in ["system"]:
            return MessageRole.SYSTEM
        elif role_str in ["tool", "function"]:
            return MessageRole.TOOL
        else:
            return MessageRole.ASSISTANT
    
    def _parse_timestamp(self, timestamp_value: Any) -> datetime:
        """
        Parse timestamp value to datetime object.
        
        Args:
            timestamp_value: Timestamp string or value
            
        Returns:
            datetime object
        """
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        
        if isinstance(timestamp_value, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(timestamp_value)
        
        if isinstance(timestamp_value, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        
        # Default to current time
        return datetime.utcnow()
