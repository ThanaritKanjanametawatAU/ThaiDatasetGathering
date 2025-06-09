"""
Factory patterns for component instantiation.
Supports dependency injection and lazy initialization.
"""

import logging
from typing import Dict, Type, Any, Optional, List, Callable
from abc import ABC
import inspect

from .base import AudioProcessor, QualityMetric, DecisionEngine, AudioAnalyzer

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Registry for components that can be created by factories.
    """
    
    def __init__(self):
        """Initialize component registry."""
        self._components: Dict[str, Type] = {}
        self._factories: Dict[str, Callable] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register(self, name: str, component_class: Type,
                factory: Optional[Callable] = None,
                dependencies: Optional[List[str]] = None) -> None:
        """
        Register a component.
        
        Args:
            name: Component name
            component_class: Component class
            factory: Optional factory function
            dependencies: Optional list of dependency names
        """
        self._components[name] = component_class
        if factory:
            self._factories[name] = factory
        if dependencies:
            self._dependencies[name] = dependencies
        
        logger.debug(f"Registered component: {name}")
    
    def get_class(self, name: str) -> Type:
        """
        Get component class.
        
        Args:
            name: Component name
            
        Returns:
            Component class
        """
        if name not in self._components:
            raise KeyError(f"Component '{name}' not registered")
        return self._components[name]
    
    def get_factory(self, name: str) -> Optional[Callable]:
        """
        Get component factory.
        
        Args:
            name: Component name
            
        Returns:
            Factory function or None
        """
        return self._factories.get(name)
    
    def get_dependencies(self, name: str) -> List[str]:
        """
        Get component dependencies.
        
        Args:
            name: Component name
            
        Returns:
            List of dependency names
        """
        return self._dependencies.get(name, [])
    
    def list_components(self) -> List[str]:
        """
        List all registered components.
        
        Returns:
            List of component names
        """
        return list(self._components.keys())


class BaseFactory(ABC):
    """
    Base factory class with common functionality.
    """
    
    def __init__(self):
        """Initialize base factory."""
        self.registry = ComponentRegistry()
        self._instances: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, name: str, component_class: Type,
                factory: Optional[Callable] = None,
                dependencies: Optional[List[str]] = None) -> None:
        """
        Register a component with the factory.
        
        Args:
            name: Component name
            component_class: Component class
            factory: Optional factory function
            dependencies: Optional list of dependency names
        """
        self.registry.register(name, component_class, factory, dependencies)
    
    def create(self, name: str, config: Optional[Dict[str, Any]] = None,
              singleton: bool = False, **kwargs) -> Any:
        """
        Create a component instance.
        
        Args:
            name: Component name
            config: Component configuration
            singleton: Whether to create as singleton
            **kwargs: Additional keyword arguments
            
        Returns:
            Component instance
        """
        # Check for existing singleton
        if singleton and name in self._singletons:
            return self._singletons[name]
        
        # Get component class and factory
        component_class = self.registry.get_class(name)
        factory = self.registry.get_factory(name)
        
        # Create instance
        if factory:
            instance = factory(config, **kwargs)
        else:
            instance = self._create_with_dependencies(
                name, component_class, config, **kwargs
            )
        
        # Store singleton if requested
        if singleton:
            self._singletons[name] = instance
        
        return instance
    
    def _create_with_dependencies(self, name: str, component_class: Type,
                                 config: Optional[Dict[str, Any]] = None,
                                 **kwargs) -> Any:
        """
        Create instance with dependency injection.
        
        Args:
            name: Component name
            component_class: Component class
            config: Component configuration
            **kwargs: Additional arguments
            
        Returns:
            Component instance
        """
        # Get dependencies
        dependencies = self.registry.get_dependencies(name)
        
        if dependencies:
            # Create dependency instances
            dep_instances = {}
            for dep_name in dependencies:
                if dep_name not in self._instances:
                    self._instances[dep_name] = self.create(dep_name)
                dep_instances[dep_name] = self._instances[dep_name]
            
            # Check if constructor accepts dependencies
            sig = inspect.signature(component_class.__init__)
            params = list(sig.parameters.keys())
            
            # Inject dependencies based on parameter names
            for param in params[1:]:  # Skip 'self'
                if param in dep_instances:
                    kwargs[param] = dep_instances[param]
                elif param == 'dependencies':
                    kwargs['dependencies'] = dep_instances
        
        # Create instance
        if config is not None:
            return component_class(config, **kwargs)
        else:
            return component_class(**kwargs)
    
    def get_instance(self, name: str) -> Optional[Any]:
        """
        Get existing instance if available.
        
        Args:
            name: Component name
            
        Returns:
            Instance or None
        """
        return self._singletons.get(name) or self._instances.get(name)
    
    def clear_instances(self) -> None:
        """Clear all cached instances."""
        self._instances.clear()
        self._singletons.clear()


class ProcessorFactory(BaseFactory):
    """
    Factory for creating AudioProcessor instances.
    """
    
    def __init__(self):
        """Initialize processor factory."""
        super().__init__()
        self._register_default_processors()
    
    def _register_default_processors(self) -> None:
        """Register default processor types."""
        # These would be imported from actual implementations
        # For now, we'll use placeholders
        pass
    
    def create_processor(self, name: str, config: Optional[Dict[str, Any]] = None,
                        **kwargs) -> AudioProcessor:
        """
        Create a processor instance.
        
        Args:
            name: Processor name
            config: Processor configuration
            **kwargs: Additional arguments
            
        Returns:
            Processor instance
        """
        processor = self.create(name, config, **kwargs)
        if not isinstance(processor, AudioProcessor):
            raise TypeError(f"Component '{name}' is not an AudioProcessor")
        return processor
    
    def create_chain(self, processor_names: List[str],
                    configs: Optional[List[Dict[str, Any]]] = None) -> List[AudioProcessor]:
        """
        Create a chain of processors.
        
        Args:
            processor_names: List of processor names
            configs: Optional list of configurations
            
        Returns:
            List of processor instances
        """
        configs = configs or [None] * len(processor_names)
        return [
            self.create_processor(name, config)
            for name, config in zip(processor_names, configs)
        ]


class MetricFactory(BaseFactory):
    """
    Factory for creating QualityMetric instances.
    """
    
    def create_metric(self, name: str, config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> QualityMetric:
        """
        Create a metric instance.
        
        Args:
            name: Metric name
            config: Metric configuration
            **kwargs: Additional arguments
            
        Returns:
            Metric instance
        """
        metric = self.create(name, config, **kwargs)
        if not isinstance(metric, QualityMetric):
            raise TypeError(f"Component '{name}' is not a QualityMetric")
        return metric
    
    def create_metric_suite(self, metric_names: List[str]) -> Dict[str, QualityMetric]:
        """
        Create a suite of metrics.
        
        Args:
            metric_names: List of metric names
            
        Returns:
            Dictionary of metric instances
        """
        return {
            name: self.create_metric(name)
            for name in metric_names
        }


class ComponentFactory(BaseFactory):
    """
    General factory for all component types.
    """
    
    def __init__(self):
        """Initialize component factory."""
        super().__init__()
        self.processor_factory = ProcessorFactory()
        self.metric_factory = MetricFactory()
    
    def create_component(self, component_type: str, name: str,
                        config: Optional[Dict[str, Any]] = None,
                        **kwargs) -> Any:
        """
        Create any type of component.
        
        Args:
            component_type: Type of component (processor, metric, etc.)
            name: Component name
            config: Component configuration
            **kwargs: Additional arguments
            
        Returns:
            Component instance
        """
        if component_type == "processor":
            return self.processor_factory.create_processor(name, config, **kwargs)
        elif component_type == "metric":
            return self.metric_factory.create_metric(name, config, **kwargs)
        else:
            return self.create(name, config, **kwargs)
    
    def register_component(self, component_type: str, name: str,
                          component_class: Type, **kwargs) -> None:
        """
        Register a component with appropriate factory.
        
        Args:
            component_type: Type of component
            name: Component name
            component_class: Component class
            **kwargs: Additional registration arguments
        """
        if component_type == "processor":
            self.processor_factory.register(name, component_class, **kwargs)
        elif component_type == "metric":
            self.metric_factory.register(name, component_class, **kwargs)
        else:
            self.register(name, component_class, **kwargs)


class SingletonFactory:
    """
    Factory that ensures only one instance of each component exists.
    """
    
    def __init__(self, base_factory: BaseFactory):
        """
        Initialize singleton factory.
        
        Args:
            base_factory: Base factory to wrap
        """
        self.base_factory = base_factory
        self._instances: Dict[str, Any] = {}
    
    def get_instance(self, name: str, config: Optional[Dict[str, Any]] = None,
                    **kwargs) -> Any:
        """
        Get singleton instance.
        
        Args:
            name: Component name
            config: Component configuration (ignored if instance exists)
            **kwargs: Additional arguments (ignored if instance exists)
            
        Returns:
            Singleton instance
        """
        if name not in self._instances:
            self._instances[name] = self.base_factory.create(
                name, config, singleton=True, **kwargs
            )
        return self._instances[name]
    
    def clear(self) -> None:
        """Clear all singleton instances."""
        self._instances.clear()
        self.base_factory.clear_instances()


# Convenience functions

def create_processor(name: str, config: Optional[Dict[str, Any]] = None) -> AudioProcessor:
    """
    Create a processor using the default factory.
    
    Args:
        name: Processor name
        config: Processor configuration
        
    Returns:
        Processor instance
    """
    factory = ProcessorFactory()
    return factory.create_processor(name, config)


def create_metric(name: str, config: Optional[Dict[str, Any]] = None) -> QualityMetric:
    """
    Create a metric using the default factory.
    
    Args:
        name: Metric name
        config: Metric configuration
        
    Returns:
        Metric instance
    """
    factory = MetricFactory()
    return factory.create_metric(name, config)


def create_pipeline_from_config(config: Dict[str, Any]) -> List[AudioProcessor]:
    """
    Create a processor pipeline from configuration.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        List of processor instances
    """
    factory = ProcessorFactory()
    processors = []
    
    for stage in config.get('stages', []):
        processor = factory.create_processor(
            stage['processor'],
            stage.get('config')
        )
        processors.append(processor)
    
    return processors