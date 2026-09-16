# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter registry and factory for managing framework adapters."""

from typing import Any, Dict, Optional, Type

from uaef.adapters.base import AdapterTransformationError, BaseAdapter


class AdapterRegistry:
    """
    Registry for managing framework-specific adapters.
    
    This registry allows:
    - Registration of built-in adapters
    - Registration of custom adapters
    - Retrieval of adapters by framework name
    - Validation of adapter implementations
    """
    
    def __init__(self):
        """Initialize the adapter registry."""
        self._adapters: Dict[str, Type[BaseAdapter]] = {}
        self._adapter_instances: Dict[str, BaseAdapter] = {}
        self._register_builtin_adapters()
    
    def _register_builtin_adapters(self) -> None:
        """Register built-in adapters."""
        try:
            from uaef.adapters.langgraph import LangGraphAdapter
            self.register("langgraph", LangGraphAdapter)
        except ImportError:
            pass
        
        try:
            from uaef.adapters.langchain import LangChainAdapter
            self.register("langchain", LangChainAdapter)
        except ImportError:
            pass
        
        try:
            from uaef.adapters.bedrock import BedrockAgentAdapter
            self.register("bedrock", BedrockAgentAdapter)
        except ImportError:
            pass
        
        try:
            from uaef.adapters.agentcore import AgentCoreAdapter
            self.register("agentcore", AgentCoreAdapter)
        except ImportError:
            pass
        
        try:
            from uaef.adapters.langfuse import LangfuseAdapter
            self.register("langfuse", LangfuseAdapter)
        except ImportError:
            pass
        
        try:
            from uaef.adapters.strands import StrandsAdapter
            self.register("strands", StrandsAdapter)
        except ImportError:
            pass

        try:
            from uaef.adapters.generic import GenericJSONAdapter
            self.register("generic", GenericJSONAdapter)
        except ImportError:
            pass
    
    def register(
        self,
        name: str,
        adapter_class: Type[BaseAdapter],
        override: bool = False
    ) -> None:
        """
        Register an adapter class.
        
        Args:
            name: Framework name (e.g., "langgraph", "bedrock")
            adapter_class: Adapter class that implements BaseAdapter
            override: If True, allow overriding existing adapters
            
        Raises:
            ValueError: If adapter is already registered and override is False
            TypeError: If adapter_class does not implement BaseAdapter
        """
        # Validate adapter class
        if not issubclass(adapter_class, BaseAdapter):
            raise TypeError(
                f"Adapter class {adapter_class.__name__} must inherit from BaseAdapter"
            )
        
        # Check for existing registration
        if name in self._adapters and not override:
            raise ValueError(
                f"Adapter '{name}' is already registered. "
                f"Use override=True to replace it."
            )
        
        # Register the adapter
        self._adapters[name] = adapter_class
        
        # Clear cached instance if it exists
        if name in self._adapter_instances:
            del self._adapter_instances[name]
    
    def get_adapter(
        self,
        name: str,
        adapter_config: Optional[Dict[str, Any]] = None
    ) -> BaseAdapter:
        """
        Get an adapter instance by framework name.
        
        Args:
            name: Framework name (e.g., "langgraph", "bedrock")
            adapter_config: Optional configuration for adapter initialization
            
        Returns:
            Adapter instance
            
        Raises:
            ValueError: If adapter is not registered
        """
        if name not in self._adapters:
            raise ValueError(
                f"Adapter '{name}' is not registered. "
                f"Available adapters: {', '.join(self.list_adapters())}"
            )
        
        # Check if we need to create a new instance (for adapters with config)
        if adapter_config:
            adapter_class = self._adapters[name]
            return adapter_class(**adapter_config)
        
        # Return cached instance or create new one
        if name not in self._adapter_instances:
            adapter_class = self._adapters[name]
            self._adapter_instances[name] = adapter_class()
        
        return self._adapter_instances[name]
    
    def list_adapters(self) -> list[str]:
        """
        List all registered adapter names.
        
        Returns:
            List of adapter names
        """
        return list(self._adapters.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        Check if an adapter is registered.
        
        Args:
            name: Framework name
            
        Returns:
            True if adapter is registered, False otherwise
        """
        return name in self._adapters
    
    def unregister(self, name: str) -> None:
        """
        Unregister an adapter.
        
        Args:
            name: Framework name
            
        Raises:
            ValueError: If adapter is not registered
        """
        if name not in self._adapters:
            raise ValueError(f"Adapter '{name}' is not registered")
        
        del self._adapters[name]
        
        if name in self._adapter_instances:
            del self._adapter_instances[name]


# Global adapter registry instance
_global_registry = AdapterRegistry()


def register_adapter(
    name: str,
    adapter_class: Type[BaseAdapter],
    override: bool = False
) -> None:
    """
    Register an adapter in the global registry.
    
    Args:
        name: Framework name (e.g., "langgraph", "bedrock")
        adapter_class: Adapter class that implements BaseAdapter
        override: If True, allow overriding existing adapters
        
    Raises:
        ValueError: If adapter is already registered and override is False
        TypeError: If adapter_class does not implement BaseAdapter
    """
    _global_registry.register(name, adapter_class, override)


def get_adapter(
    name: str,
    adapter_config: Optional[Dict[str, Any]] = None
) -> BaseAdapter:
    """
    Get an adapter instance from the global registry.
    
    Args:
        name: Framework name (e.g., "langgraph", "bedrock")
        adapter_config: Optional configuration for adapter initialization
        
    Returns:
        Adapter instance
        
    Raises:
        ValueError: If adapter is not registered
    """
    return _global_registry.get_adapter(name, adapter_config)


def list_adapters() -> list[str]:
    """
    List all registered adapter names in the global registry.
    
    Returns:
        List of adapter names
    """
    return _global_registry.list_adapters()


def is_adapter_registered(name: str) -> bool:
    """
    Check if an adapter is registered in the global registry.
    
    Args:
        name: Framework name
        
    Returns:
        True if adapter is registered, False otherwise
    """
    return _global_registry.is_registered(name)


def get_registry() -> AdapterRegistry:
    """
    Get the global adapter registry instance.
    
    Returns:
        Global AdapterRegistry instance
    """
    return _global_registry
