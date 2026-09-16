# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""External metric connector registry and factory for managing external library integrations."""

from typing import Any, Dict, List, Optional, Type

from uaef.models.evaluation_input import EvaluationInput
from uaef.models.metric_score import MetricScore
from uaef.logging import get_logger

logger = get_logger(__name__)


class ExternalMetricConnector:
    """
    Base interface for external metric connectors.
    
    All external metric connectors should implement this interface to ensure
    consistent behavior and integration with UAEF.
    """
    
    def is_available(self) -> bool:
        """
        Check if external library is available.
        
        Returns:
            True if external library is installed and available, False otherwise
        """
        raise NotImplementedError("Subclasses must implement is_available()")
    
    def calculate_metrics(
        self,
        evaluation_input: EvaluationInput,
        metric_names: List[str],
        **kwargs: Any
    ) -> List[MetricScore]:
        """
        Calculate multiple metrics and return UAEF MetricScore objects.
        
        Args:
            evaluation_input: UAEF evaluation input
            metric_names: List of metric names to calculate
            **kwargs: Additional connector-specific parameters
            
        Returns:
            List of MetricScore objects
            
        Raises:
            RuntimeError: If external library is not available
        """
        raise NotImplementedError("Subclasses must implement calculate_metrics()")
    
    def list_available_metrics(self) -> List[str]:
        """
        List all metrics supported by this connector.
        
        Returns:
            List of metric names
        """
        raise NotImplementedError("Subclasses must implement list_available_metrics()")


class ExternalMetricRegistry:
    """
    Registry for managing external metric library connectors.
    
    This registry allows:
    - Registration of built-in connectors (RAGAS, DeepEval)
    - Registration of custom connectors
    - Retrieval of connectors by library name
    - Discovery of available connectors and their metrics
    - Graceful handling of unavailable libraries
    """
    
    def __init__(self):
        """Initialize the external metric registry."""
        self._connectors: Dict[str, Type[ExternalMetricConnector]] = {}
        self._connector_instances: Dict[str, ExternalMetricConnector] = {}
        self._register_builtin_connectors()
    
    def _register_builtin_connectors(self) -> None:
        """Register built-in external metric connectors."""
        # Register RAGAS connector
        try:
            from uaef.integrations.ragas_connector import RAGASConnector
            self.register("ragas", RAGASConnector)
            logger.info("Registered RAGAS connector")
        except ImportError as e:
            logger.debug(f"RAGAS connector not available: {e}")
        
        # Register DeepEval connector
        try:
            from uaef.integrations.deepeval_connector import DeepEvalConnector
            self.register("deepeval", DeepEvalConnector)
            logger.info("Registered DeepEval connector")
        except ImportError as e:
            logger.debug(f"DeepEval connector not available: {e}")
    
    def register(
        self,
        name: str,
        connector_class: Type[ExternalMetricConnector],
        override: bool = False
    ) -> None:
        """
        Register an external metric connector class.
        
        Args:
            name: Library name (e.g., "ragas", "deepeval")
            connector_class: Connector class that implements ExternalMetricConnector interface
            override: If True, allow overriding existing connectors
            
        Raises:
            ValueError: If connector is already registered and override is False
            TypeError: If connector_class does not implement required methods
        """
        # Validate connector class has required methods
        required_methods = ["is_available", "calculate_metrics", "list_available_metrics"]
        for method in required_methods:
            if not hasattr(connector_class, method):
                raise TypeError(
                    f"Connector class {connector_class.__name__} must implement {method}() method"
                )
        
        # Check for existing registration
        if name in self._connectors and not override:
            raise ValueError(
                f"Connector '{name}' is already registered. "
                f"Use override=True to replace it."
            )
        
        # Register the connector
        self._connectors[name] = connector_class
        logger.info(f"Registered external metric connector: {name}")
        
        # Clear cached instance if it exists
        if name in self._connector_instances:
            del self._connector_instances[name]
    
    def get_connector(
        self,
        name: str,
        connector_config: Optional[Dict[str, Any]] = None
    ) -> ExternalMetricConnector:
        """
        Get a connector instance by library name.
        
        Args:
            name: Library name (e.g., "ragas", "deepeval")
            connector_config: Optional configuration for connector initialization
            
        Returns:
            Connector instance
            
        Raises:
            ValueError: If connector is not registered
        """
        if name not in self._connectors:
            raise ValueError(
                f"Connector '{name}' is not registered. "
                f"Available connectors: {', '.join(self.list_connectors())}"
            )
        
        # Check if we need to create a new instance (for connectors with config)
        if connector_config:
            connector_class = self._connectors[name]
            return connector_class(**connector_config)
        
        # Return cached instance or create new one
        if name not in self._connector_instances:
            connector_class = self._connectors[name]
            self._connector_instances[name] = connector_class()
        
        return self._connector_instances[name]
    
    def list_connectors(self) -> List[str]:
        """
        List all registered connector names.
        
        Returns:
            List of connector names
        """
        return list(self._connectors.keys())
    
    def list_available_connectors(self) -> List[str]:
        """
        List connectors that are available (library is installed).
        
        Returns:
            List of available connector names
        """
        available = []
        for name in self._connectors.keys():
            try:
                connector = self.get_connector(name)
                if connector.is_available():
                    available.append(name)
            except Exception as e:
                logger.debug(f"Error checking availability of {name}: {e}")
        return available
    
    def list_all_metrics(self) -> Dict[str, List[str]]:
        """
        List all metrics from all available connectors.
        
        Returns:
            Dictionary mapping connector name to list of metric names
        """
        all_metrics = {}
        for name in self.list_available_connectors():
            try:
                connector = self.get_connector(name)
                metrics = connector.list_available_metrics()
                all_metrics[name] = metrics
            except Exception as e:
                logger.warning(f"Error listing metrics for {name}: {e}")
        return all_metrics
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a connector is registered.
        
        Args:
            name: Library name
            
        Returns:
            True if connector is registered, False otherwise
        """
        return name in self._connectors
    
    def is_available(self, name: str) -> bool:
        """
        Check if a connector is registered and its library is available.
        
        Args:
            name: Library name
            
        Returns:
            True if connector is registered and library is available, False otherwise
        """
        if not self.is_registered(name):
            return False
        
        try:
            connector = self.get_connector(name)
            return connector.is_available()
        except Exception as e:
            logger.debug(f"Error checking availability of {name}: {e}")
            return False
    
    def unregister(self, name: str) -> None:
        """
        Unregister a connector.
        
        Args:
            name: Library name
            
        Raises:
            ValueError: If connector is not registered
        """
        if name not in self._connectors:
            raise ValueError(f"Connector '{name}' is not registered")
        
        del self._connectors[name]
        logger.info(f"Unregistered external metric connector: {name}")
        
        if name in self._connector_instances:
            del self._connector_instances[name]
    
    def calculate_metrics(
        self,
        connector_name: str,
        evaluation_input: EvaluationInput,
        metric_names: List[str],
        **kwargs: Any
    ) -> List[MetricScore]:
        """
        Calculate metrics using a specific connector.
        
        This is a convenience method that gets the connector and calls its
        calculate_metrics method. Handles errors gracefully per requirement 14.6.
        
        Args:
            connector_name: Name of the connector to use
            evaluation_input: UAEF evaluation input
            metric_names: List of metric names to calculate
            **kwargs: Additional connector-specific parameters
            
        Returns:
            List of MetricScore objects (may be empty if connector unavailable)
        """
        try:
            # Check if connector is registered
            if not self.is_registered(connector_name):
                logger.warning(
                    f"Connector '{connector_name}' is not registered. "
                    f"Available: {', '.join(self.list_connectors())}"
                )
                return []
            
            # Get connector instance
            connector = self.get_connector(connector_name)
            
            # Check if library is available
            if not connector.is_available():
                logger.warning(
                    f"Connector '{connector_name}' library is not available. "
                    f"Skipping metrics: {', '.join(metric_names)}"
                )
                return []
            
            # Calculate metrics
            scores = connector.calculate_metrics(
                evaluation_input,
                metric_names,
                **kwargs
            )
            
            logger.info(
                f"Calculated {len(scores)} metrics using {connector_name} connector"
            )
            return scores
            
        except Exception as e:
            # Handle errors gracefully per requirement 14.6
            logger.error(
                f"Error calculating metrics with {connector_name} connector: {e}",
                exc_info=True
            )
            return []


# Global external metric registry instance
_global_registry = ExternalMetricRegistry()


def register_connector(
    name: str,
    connector_class: Type[ExternalMetricConnector],
    override: bool = False
) -> None:
    """
    Register a connector in the global registry.
    
    Args:
        name: Library name (e.g., "ragas", "deepeval")
        connector_class: Connector class that implements ExternalMetricConnector interface
        override: If True, allow overriding existing connectors
        
    Raises:
        ValueError: If connector is already registered and override is False
        TypeError: If connector_class does not implement required methods
    """
    _global_registry.register(name, connector_class, override)


def get_connector(
    name: str,
    connector_config: Optional[Dict[str, Any]] = None
) -> ExternalMetricConnector:
    """
    Get a connector instance from the global registry.
    
    Args:
        name: Library name (e.g., "ragas", "deepeval")
        connector_config: Optional configuration for connector initialization
        
    Returns:
        Connector instance
        
    Raises:
        ValueError: If connector is not registered
    """
    return _global_registry.get_connector(name, connector_config)


def list_connectors() -> List[str]:
    """
    List all registered connector names in the global registry.
    
    Returns:
        List of connector names
    """
    return _global_registry.list_connectors()


def list_available_connectors() -> List[str]:
    """
    List connectors that are available (library is installed) in the global registry.
    
    Returns:
        List of available connector names
    """
    return _global_registry.list_available_connectors()


def list_all_metrics() -> Dict[str, List[str]]:
    """
    List all metrics from all available connectors in the global registry.
    
    Returns:
        Dictionary mapping connector name to list of metric names
    """
    return _global_registry.list_all_metrics()


def is_connector_registered(name: str) -> bool:
    """
    Check if a connector is registered in the global registry.
    
    Args:
        name: Library name
        
    Returns:
        True if connector is registered, False otherwise
    """
    return _global_registry.is_registered(name)


def is_connector_available(name: str) -> bool:
    """
    Check if a connector is registered and its library is available in the global registry.
    
    Args:
        name: Library name
        
    Returns:
        True if connector is registered and library is available, False otherwise
    """
    return _global_registry.is_available(name)


def calculate_external_metrics(
    connector_name: str,
    evaluation_input: EvaluationInput,
    metric_names: List[str],
    **kwargs: Any
) -> List[MetricScore]:
    """
    Calculate metrics using a specific connector from the global registry.
    
    This is a convenience function that handles errors gracefully per requirement 14.6.
    
    Args:
        connector_name: Name of the connector to use
        evaluation_input: UAEF evaluation input
        metric_names: List of metric names to calculate
        **kwargs: Additional connector-specific parameters
        
    Returns:
        List of MetricScore objects (may be empty if connector unavailable)
    """
    return _global_registry.calculate_metrics(
        connector_name,
        evaluation_input,
        metric_names,
        **kwargs
    )


def get_registry() -> ExternalMetricRegistry:
    """
    Get the global external metric registry instance.
    
    Returns:
        Global ExternalMetricRegistry instance
    """
    return _global_registry
