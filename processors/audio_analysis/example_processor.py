"""
Example audio processor implementation demonstrating the plugin architecture.
"""

import numpy as np
from typing import Dict, Any, List

from .base import AudioProcessor, ProcessResult, PluginInfo, ProcessingStatus


class ExampleAudioLoader(AudioProcessor):
    """
    Example audio loader processor that demonstrates the plugin interface.
    """
    
    def process(self, audio: np.ndarray, sr: int, **kwargs) -> ProcessResult:
        """
        Process audio by validating and normalizing it.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            **kwargs: Additional parameters
            
        Returns:
            ProcessResult with processed audio
        """
        # Simulate processing
        metadata = {
            "original_shape": audio.shape,
            "original_sr": sr,
            "normalized": False
        }
        
        # Normalize if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            metadata["normalized"] = True
        
        # Add some processing metadata
        metadata["audio_length_seconds"] = len(audio) / sr
        metadata["peak_amplitude"] = np.max(np.abs(audio))
        
        return ProcessResult(
            audio=audio,
            metadata=metadata,
            status=ProcessingStatus.SUCCESS,
            processing_time=0.001  # Simulated time
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get processor capabilities."""
        return {
            "supported_formats": ["wav", "mp3", "flac"],
            "supported_sample_rates": [8000, 16000, 22050, 44100, 48000],
            "max_channels": 2,
            "processing_type": "loader",
            "gpu_acceleration": False,
            "real_time_capable": True
        }
    
    def validate_input(self, audio: np.ndarray, sr: int) -> bool:
        """Validate input audio."""
        # Check audio shape
        if audio.ndim not in [1, 2]:
            return False
        
        # Check sample rate
        if sr <= 0 or sr > 192000:
            return False
        
        # Check for NaN or Inf
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return False
        
        return True
    
    @classmethod
    def get_plugin_info(cls) -> PluginInfo:
        """Get plugin information."""
        return PluginInfo(
            name="ExampleAudioLoader",
            version="1.0.0",
            author="Audio Analysis Team",
            description="Example audio loader for demonstration",
            category="loader",
            tags=["example", "loader", "basic"]
        )
    
    def get_dependencies(self) -> List[str]:
        """No dependencies for this basic loader."""
        return []


class ExampleNoiseReducer(AudioProcessor):
    """
    Example noise reduction processor.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        super().__init__(config)
        self.reduction_strength = self.config.get("reduction_strength", 0.5)
    
    def process(self, audio: np.ndarray, sr: int, **kwargs) -> ProcessResult:
        """
        Apply simulated noise reduction.
        
        Args:
            audio: Input audio
            sr: Sample rate
            **kwargs: Additional parameters
            
        Returns:
            ProcessResult with processed audio
        """
        # Simulate noise reduction by slight smoothing
        if len(audio) > 3:
            # Simple moving average as fake noise reduction
            window_size = 3
            smoothed = np.convolve(audio, np.ones(window_size)/window_size, mode='same')
            # Mix with original based on reduction strength
            processed = audio * (1 - self.reduction_strength) + smoothed * self.reduction_strength
        else:
            processed = audio
        
        metadata = {
            "noise_reduction_applied": True,
            "reduction_strength": self.reduction_strength,
            "estimated_noise_level": 0.1  # Fake metric
        }
        
        return ProcessResult(
            audio=processed,
            metadata=metadata,
            status=ProcessingStatus.SUCCESS,
            processing_time=0.005
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get processor capabilities."""
        return {
            "processing_type": "enhancement",
            "noise_types": ["white", "pink", "background"],
            "gpu_acceleration": False,
            "real_time_capable": True,
            "configurable_parameters": ["reduction_strength"]
        }
    
    def validate_input(self, audio: np.ndarray, sr: int) -> bool:
        """Validate input."""
        if audio.ndim != 1:
            return False
        if len(audio) == 0:
            return False
        return True
    
    @classmethod
    def get_plugin_info(cls) -> PluginInfo:
        """Get plugin information."""
        return PluginInfo(
            name="ExampleNoiseReducer",
            version="1.0.0",
            author="Audio Analysis Team",
            description="Example noise reduction processor",
            category="enhancement",
            tags=["example", "noise_reduction", "enhancement"]
        )
    
    def get_dependencies(self) -> List[str]:
        """This processor depends on having audio loaded first."""
        return ["ExampleAudioLoader"]