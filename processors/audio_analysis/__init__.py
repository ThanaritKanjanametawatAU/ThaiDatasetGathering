"""
Audio Analysis Architecture Module.

This module provides the core framework for autonomous audio processing with:
- Plugin-based processor architecture
- Dynamic pipeline execution
- Quality metrics and decision engines
- Memory-efficient processing
"""

from .base import (
    # Core classes
    AudioProcessor,
    QualityMetric,
    DecisionEngine,
    AudioAnalyzer,
    
    # Data classes
    ProcessResult,
    ProcessingStatus,
    PluginInfo,
    StageConfig,
    PipelineConfig,
    
    # Exceptions
    PipelineError,
    PluginLoadError,
    StageExecutionError,
    ValidationError,
    
    # Utility functions
    validate_audio_shape,
    validate_sample_rate
)

from .plugin_system import (
    PluginRegistry,
    PluginDiscovery,
    PluginLoader,
    PluginManager
)

from .pipeline import (
    Pipeline,
    PipelineStage,
    StageMetrics
)

from .factory import (
    ProcessorFactory,
    MetricFactory,
    ComponentFactory,
    SingletonFactory,
    create_processor,
    create_metric,
    create_pipeline_from_config
)

from .config import (
    Config,
    ConfigLoader,
    ConfigBuilder,
    PluginConfig,
    PerformanceConfig
)

from .memory import (
    MemoryPool,
    LazyAudioLoader,
    ResourceManager,
    get_resource_manager,
    monitor_memory_usage
)

__all__ = [
    # Base classes
    'AudioProcessor',
    'QualityMetric', 
    'DecisionEngine',
    'AudioAnalyzer',
    
    # Data classes
    'ProcessResult',
    'ProcessingStatus',
    'PluginInfo',
    'StageConfig',
    'PipelineConfig',
    
    # Plugin system
    'PluginRegistry',
    'PluginDiscovery',
    'PluginLoader',
    'PluginManager',
    
    # Pipeline
    'Pipeline',
    'PipelineStage',
    'StageMetrics',
    
    # Factory
    'ProcessorFactory',
    'MetricFactory',
    'ComponentFactory',
    'SingletonFactory',
    'create_processor',
    'create_metric',
    'create_pipeline_from_config',
    
    # Configuration
    'Config',
    'ConfigLoader',
    'ConfigBuilder',
    'PluginConfig',
    'PerformanceConfig',
    
    # Memory management
    'MemoryPool',
    'LazyAudioLoader',
    'ResourceManager',
    'get_resource_manager',
    'monitor_memory_usage',
    
    # Exceptions
    'PipelineError',
    'PluginLoadError',
    'StageExecutionError',
    'ValidationError',
    
    # Utilities
    'validate_audio_shape',
    'validate_sample_rate'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Thai Audio Dataset Collection Team'