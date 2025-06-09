"""
Base abstract classes for the audio analysis architecture.
Provides the foundation for all audio processing components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import numpy as np

# Try to import pydantic for enhanced validation
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object


class ProcessingStatus(Enum):
    """Status of processing operation."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class ProcessResult:
    """Result of audio processing operation."""
    audio: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    
    def merge_metadata(self, other_metadata: Dict[str, Any]) -> None:
        """Merge additional metadata into result."""
        self.metadata.update(other_metadata)


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    version: str
    author: str
    description: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    min_python_version: str = "3.8"
    requirements: List[str] = field(default_factory=list)


# Configuration classes with pydantic validation when available
if PYDANTIC_AVAILABLE:
    class StageConfigPydantic(BaseModel):
        """Configuration for a pipeline stage with pydantic validation."""
        name: str
        processor: str
        config: Dict[str, Any] = Field(default_factory=dict)
        dependencies: List[str] = Field(default_factory=list)
        error_handling: Dict[str, Any] = Field(default_factory=dict)
        parallel: bool = Field(default=False)
        required: bool = Field(default=True)
        timeout: Optional[float] = Field(default=None, ge=0)
        
        @validator('name', 'processor')
        def validate_non_empty(cls, v):
            if not v or not v.strip():
                raise ValueError("Cannot be empty")
            return v
        
        @validator('error_handling')
        def validate_error_handling(cls, v):
            if 'strategy' in v and v['strategy'] not in ['fail', 'skip', 'retry', 'fallback']:
                raise ValueError(f"Invalid error handling strategy: {v['strategy']}")
            return v

    class PipelineConfigPydantic(BaseModel):
        """Configuration for entire pipeline with pydantic validation."""
        name: str
        version: str
        stages: List[StageConfigPydantic]
        description: str = Field(default="")
        author: str = Field(default="")
        tags: List[str] = Field(default_factory=list)
        performance: Dict[str, Any] = Field(default_factory=dict)
        
        @validator('name', 'version')
        def validate_required_strings(cls, v):
            if not v or not v.strip():
                raise ValueError("Cannot be empty")
            return v
        
        @validator('stages')
        def validate_stages(cls, v):
            if not v:
                raise ValueError("Pipeline must have at least one stage")
            return v
    
    # Use pydantic models
    StageConfig = StageConfigPydantic
    PipelineConfig = PipelineConfigPydantic
else:
    # Use dataclasses as fallback
    @dataclass
    class StageConfig:
        """Configuration for a pipeline stage."""
        name: str
        processor: str
        config: Dict[str, Any] = field(default_factory=dict)
        dependencies: List[str] = field(default_factory=list)
        error_handling: Dict[str, Any] = field(default_factory=dict)
        parallel: bool = False
        required: bool = True
        timeout: Optional[float] = None


    @dataclass
    class PipelineConfig:
        """Configuration for entire pipeline."""
        name: str
        version: str
        stages: List[StageConfig]
        description: str = ""
        author: str = ""
        tags: List[str] = field(default_factory=list)
        performance: Dict[str, Any] = field(default_factory=dict)


class AudioProcessor(ABC):
    """
    Base class for all audio processing modules.
    Implements the plugin interface for dynamic loading.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor with configuration.
        
        Args:
            config: Configuration dictionary for the processor
        """
        self.config = config or {}
        self._initialized = False
        self._metadata = {}
    
    @abstractmethod
    def process(self, audio: np.ndarray, sr: int, **kwargs) -> ProcessResult:
        """
        Process audio data.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessResult with processed audio and metadata
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get processor capabilities.
        
        Returns:
            Dictionary describing processor capabilities:
            - supported_formats: List of audio formats
            - supported_sample_rates: List of sample rates
            - max_channels: Maximum number of channels
            - processing_type: Type of processing (enhancement, analysis, etc.)
            - gpu_acceleration: Whether GPU acceleration is supported
            - real_time_capable: Whether real-time processing is supported
        """
        pass
    
    @abstractmethod
    def validate_input(self, audio: np.ndarray, sr: int) -> bool:
        """
        Validate input audio data.
        
        Args:
            audio: Audio data to validate
            sr: Sample rate to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_plugin_info(cls) -> PluginInfo:
        """
        Get plugin information for discovery.
        
        Returns:
            PluginInfo object with plugin metadata
        """
        pass
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of required plugins.
        
        Returns:
            List of plugin names this processor depends on
        """
        return []
    
    def initialize(self) -> None:
        """
        Initialize processor resources.
        Called once before first use.
        """
        if not self._initialized:
            self._setup()
            self._initialized = True
    
    def _setup(self) -> None:
        """
        Setup processor resources.
        Override in subclasses for custom initialization.
        """
        pass
    
    def cleanup(self) -> None:
        """
        Cleanup processor resources.
        Called when processor is no longer needed.
        """
        if self._initialized:
            self._teardown()
            self._initialized = False
    
    def _teardown(self) -> None:
        """
        Teardown processor resources.
        Override in subclasses for custom cleanup.
        """
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for validation.
        
        Returns:
            JSON schema for configuration validation
        """
        return {}
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update processor configuration.
        
        Args:
            config: New configuration values
        """
        self.config.update(config)
        if self._initialized:
            self._teardown()
            self._setup()


class QualityMetric(ABC):
    """
    Base class for quality assessment metrics.
    Used to evaluate audio quality at various pipeline stages.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metric with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def calculate(self, audio: np.ndarray, sr: int, 
                 reference: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        Calculate quality metric.
        
        Args:
            audio: Audio to evaluate
            sr: Sample rate
            reference: Reference audio for comparison metrics
            **kwargs: Additional parameters
            
        Returns:
            Metric value (higher is better)
        """
        pass
    
    @abstractmethod
    def get_metric_info(self) -> Dict[str, Any]:
        """
        Get information about the metric.
        
        Returns:
            Dictionary with metric information:
            - name: Metric name
            - range: (min, max) tuple
            - requires_reference: Whether reference audio is needed
            - description: Human-readable description
        """
        pass
    
    def normalize(self, value: float) -> float:
        """
        Normalize metric value to 0-1 range.
        
        Args:
            value: Raw metric value
            
        Returns:
            Normalized value between 0 and 1
        """
        info = self.get_metric_info()
        min_val, max_val = info['range']
        return (value - min_val) / (max_val - min_val)


class DecisionEngine(ABC):
    """
    Base class for autonomous decision making.
    Implements multi-criteria decision making (MCDM) for processing choices.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize decision engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.history = []
    
    @abstractmethod
    def decide(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make processing decision based on features.
        
        Args:
            features: Extracted audio features
            context: Additional context for decision
            
        Returns:
            Decision dictionary with:
            - action: Processing action to take
            - confidence: Confidence score (0-1)
            - reasoning: Explanation for decision
            - parameters: Processing parameters
        """
        pass
    
    @abstractmethod
    def update(self, decision: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """
        Update decision engine based on outcome.
        
        Args:
            decision: Decision that was made
            outcome: Outcome of the decision
        """
        pass
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """
        Get history of decisions made.
        
        Returns:
            List of past decisions and outcomes
        """
        return self.history.copy()
    
    def reset_history(self) -> None:
        """
        Reset decision history.
        """
        self.history.clear()


class AudioAnalyzer(ABC):
    """
    Base class for audio analysis modules.
    Extracts features and characteristics from audio.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def analyze(self, audio: np.ndarray, sr: int, **kwargs) -> Dict[str, Any]:
        """
        Analyze audio and extract features.
        
        Args:
            audio: Audio data to analyze
            sr: Sample rate
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary of extracted features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of features this analyzer extracts.
        
        Returns:
            List of feature names
        """
        pass
    
    def get_feature_description(self, feature_name: str) -> str:
        """
        Get description of a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Human-readable description
        """
        return f"Feature: {feature_name}"


# Exception classes for error handling

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    
    def __init__(self, message: str, stage: Optional[str] = None, 
                 recoverable: bool = False):
        self.message = message
        self.stage = stage
        self.recoverable = recoverable
        super().__init__(self.message)


class PluginLoadError(PipelineError):
    """Exception for plugin loading failures."""
    
    def __init__(self, plugin_name: str, reason: str):
        message = f"Failed to load plugin '{plugin_name}': {reason}"
        super().__init__(message, recoverable=False)
        self.plugin_name = plugin_name
        self.reason = reason


class StageExecutionError(PipelineError):
    """Exception for stage execution failures."""
    
    def __init__(self, stage_name: str, reason: str, recoverable: bool = True):
        message = f"Stage '{stage_name}' failed: {reason}"
        super().__init__(message, stage=stage_name, recoverable=recoverable)
        self.stage_name = stage_name
        self.reason = reason


class ValidationError(PipelineError):
    """Exception for validation failures."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, recoverable=False)
        self.field = field


# Utility functions

def validate_audio_shape(audio: np.ndarray) -> bool:
    """
    Validate audio array shape.
    
    Args:
        audio: Audio array to validate
        
    Returns:
        True if shape is valid
    """
    if audio.ndim == 1:
        return True
    elif audio.ndim == 2:
        # Either (samples, channels) or (channels, samples)
        return audio.shape[0] > 0 and audio.shape[1] > 0
    else:
        return False


def validate_sample_rate(sr: int) -> bool:
    """
    Validate sample rate.
    
    Args:
        sr: Sample rate to validate
        
    Returns:
        True if sample rate is valid
    """
    valid_rates = [8000, 16000, 22050, 44100, 48000, 96000, 192000]
    return sr in valid_rates or (sr > 0 and sr <= 384000)