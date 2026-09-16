# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base adapter interface for framework-specific trace transformation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from uaef.models.agent_trace import AgentTrace
from uaef.models.message import Message
from uaef.models.multi_agent_trace import MultiAgentTrace
from uaef.models.tool_call import ToolCall


class AdapterTransformationError(Exception):
    """
    Exception raised when adapter transformation fails.
    
    This exception should be raised when:
    - Required fields are missing from the raw data
    - Data format is invalid or unexpected
    - Transformation logic encounters an error
    """
    
    def __init__(self, message: str, adapter_name: str = None, original_error: Exception = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            adapter_name: Name of the adapter that raised the error
            original_error: Original exception that caused the transformation failure
        """
        self.adapter_name = adapter_name
        self.original_error = original_error
        
        full_message = message
        if adapter_name:
            full_message = f"[{adapter_name}] {message}"
        if original_error:
            full_message = f"{full_message} (caused by: {str(original_error)})"
        
        super().__init__(full_message)


class BaseAdapter(ABC):
    """
    Abstract base class for framework-specific trace adapters.
    
    All adapters must implement this interface to transform framework-specific
    traces into the canonical UAEF AgentTrace format. This ensures consistent
    evaluation across different agent frameworks.
    
    The transformation process should:
    1. Extract all messages from the framework-specific trace
    2. Extract all tool calls with their arguments and results
    3. Extract metadata (tokens, latency, cost)
    4. Construct a canonical AgentTrace object
    5. Preserve all information from the original trace (no data loss)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this adapter.
        
        Returns:
            Adapter name (e.g., "langgraph", "bedrock", "langchain")
        """
        pass
    
    @abstractmethod
    def transform_to_canonical(self, raw_data: Dict[str, Any]) -> Union[AgentTrace, MultiAgentTrace]:
        """
        Transform framework-specific trace data to canonical format.
        
        Returns AgentTrace for single-agent traces, or MultiAgentTrace when
        multiple agent nodes are detected.
        
        Args:
            raw_data: Framework-specific trace data as a dictionary
            
        Returns:
            AgentTrace or MultiAgentTrace
            
        Raises:
            AdapterTransformationError: If transformation fails
        """
        pass
    
    def supports_session_transform(self) -> bool:
        """
        Whether ``transform_to_canonical`` accepts a *list* of per-turn payloads
        and returns one session dict per conversation.

        Multi-turn evaluation needs a whole conversation assembled into
        ``{"session_id", "per_turn_traces", "full_trace"}``. Only adapters that
        implement that list form can produce it; the rest take a single payload
        and know nothing about turns.

        Concrete (not abstract) and defaulting to ``False`` so existing adapters
        keep working untouched and simply opt out. Callers must check this before
        handing a list to :meth:`transform_to_canonical` — passing one to an
        adapter that expects a dict fails in whatever way that adapter happens to
        fail, which is not a useful error.

        Returns:
            True if list/session input is supported, False otherwise.
        """
        return False

    @abstractmethod
    def extract_messages(self, raw_data: Dict[str, Any]) -> List[Message]:
        """
        Extract messages from framework-specific trace data.
        
        Messages should be extracted in chronological order and include:
        - User messages (inputs)
        - Assistant messages (agent responses)
        - System messages (if present)
        - Tool messages (if present)
        
        Args:
            raw_data: Framework-specific trace data
            
        Returns:
            List of Message objects in chronological order
            
        Raises:
            AdapterTransformationError: If message extraction fails
        """
        pass
    
    @abstractmethod
    def extract_tool_calls(self, raw_data: Dict[str, Any]) -> List[ToolCall]:
        """
        Extract tool calls from framework-specific trace data.
        
        Tool calls should include:
        - Tool name
        - Arguments passed to the tool
        - Result returned by the tool (if available)
        - Timestamp of the call
        - Execution time (if available)
        - Error information (if the call failed)
        
        Args:
            raw_data: Framework-specific trace data
            
        Returns:
            List of ToolCall objects in chronological order
            
        Raises:
            AdapterTransformationError: If tool call extraction fails
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from framework-specific trace data.
        
        Metadata should include (when available):
        - input_tokens: Number of input tokens consumed
        - output_tokens: Number of output tokens generated
        - latency: Total execution time in seconds
        - cost: Estimated cost in USD
        - model: Model name/identifier
        - framework_version: Version of the framework
        - Any other framework-specific metadata
        
        Args:
            raw_data: Framework-specific trace data
            
        Returns:
            Dictionary of metadata key-value pairs
            
        Raises:
            AdapterTransformationError: If metadata extraction fails
        """
        pass
    
    def validate_raw_data(self, raw_data: Dict[str, Any], required_keys: List[str]) -> None:
        """
        Validate that raw data contains required keys.
        
        This is a helper method that adapters can use to validate input data.
        
        Args:
            raw_data: Framework-specific trace data
            required_keys: List of required keys that must be present
            
        Raises:
            AdapterTransformationError: If any required key is missing
        """
        missing_keys = [key for key in required_keys if key not in raw_data]
        if missing_keys:
            raise AdapterTransformationError(
                f"Missing required keys: {', '.join(missing_keys)}",
                adapter_name=self.name
            )
