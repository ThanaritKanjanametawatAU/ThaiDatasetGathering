"""
Speaker Separation Module - DEPRECATED

This module has been replaced by speechbrain_separator.py which provides
superior speaker separation using SpeechBrain's SepFormer model.

For backward compatibility, this module redirects to the new implementation.
"""

import logging
from typing import Dict, Union, List, Optional
import numpy as np

# Import the new SpeechBrain implementation
from .speechbrain_separator import (
    SpeechBrainSeparator,
    SeparationConfig,
    SeparationOutput,
    SeparationInput,
    ProcessingMetrics
)

logger = logging.getLogger(__name__)

# Log deprecation warning once
_deprecation_logged = False

def _log_deprecation():
    global _deprecation_logged
    if not _deprecation_logged:
        logger.warning(
            "speaker_separation.py is deprecated. Using speechbrain_separator.py instead. "
            "Please update your imports to use speechbrain_separator directly."
        )
        _deprecation_logged = True


class SpeakerSeparator(SpeechBrainSeparator):
    """
    Backward compatibility wrapper for SpeechBrainSeparator.
    
    This class maintains the old interface while using the new SpeechBrain
    implementation under the hood.
    """
    
    def __init__(self, config=None):
        """
        Initialize with backward compatibility.
        
        Args:
            config: Can be old-style config dict or new SeparationConfig
        """
        _log_deprecation()
        
        # Convert old config format if needed
        if isinstance(config, dict):
            new_config = SeparationConfig()
            # Map old config keys to new ones
            if 'confidence_threshold' in config:
                new_config.confidence_threshold = config['confidence_threshold']
            if 'suppression_strength' in config:
                # Old suppression_strength maps to confidence_threshold
                new_config.confidence_threshold = max(0.5, config['suppression_strength'])
            if 'use_sepformer' in config:
                logger.info("use_sepformer parameter ignored - SpeechBrain always uses SepFormer")
            config = new_config
            
        super().__init__(config)
        logger.info("SpeakerSeparator initialized with SpeechBrain backend")
    
    def separate_speakers(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Backward compatibility method that returns dict format.
        
        Args:
            audio_array: Input audio
            sample_rate: Sample rate
            
        Returns:
            Dictionary with 'audio', 'detections', 'metrics' for compatibility
        """
        # Call new implementation
        result = super().separate_speakers(audio_array, sample_rate)
        
        # Convert to old format
        return {
            'audio': result.audio,
            'detections': [],  # No longer used
            'overlaps': [],    # No longer used
            'metrics': {
                **result.metrics,
                'confidence': result.confidence,
                'rejected': result.rejected,
                'rejection_reason': result.rejection_reason,
                'processing_time_ms': result.processing_time_ms,
                'num_speakers_detected': result.num_speakers_detected
            }
        }
    
    def adaptive_suppression(self, 
                           audio_array: np.ndarray,
                           sample_rate: int = 16000,
                           target_similarity: float = 0.95) -> np.ndarray:
        """
        Legacy method - now just calls regular separation.
        
        Args:
            audio_array: Input audio
            sample_rate: Sample rate
            target_similarity: Ignored (no longer used)
            
        Returns:
            Separated audio
        """
        logger.info("adaptive_suppression is deprecated, using standard separation")
        result = self.separate_speakers(audio_array, sample_rate)
        return result['audio']


# Legacy config class for compatibility
class LegacySeparationConfig:
    """Legacy configuration class for backward compatibility"""
    def __init__(self, **kwargs):
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        self.suppression_strength = kwargs.get('suppression_strength', 0.6)
        self.use_sepformer = kwargs.get('use_sepformer', True)
        self.preserve_main_speaker = kwargs.get('preserve_main_speaker', True)
        self.artifact_removal = kwargs.get('artifact_removal', True)


# Re-export new classes for compatibility
__all__ = [
    'SpeakerSeparator',
    'SeparationConfig',
    'SeparationOutput',
    'SeparationInput',
    'ProcessingMetrics',
    'LegacySeparationConfig'
]