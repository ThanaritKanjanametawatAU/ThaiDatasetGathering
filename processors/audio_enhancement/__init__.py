"""
Audio Enhancement Module

Provides audio quality enhancement capabilities including noise reduction,
voice enhancement, audio restoration, and speaker separation.
"""

# Core enhancement
from .core import AudioEnhancer

# Speaker separation
from .speaker_separation import SpeakerSeparator, SeparationConfig

# Detection modules
from .detection.overlap_detector import OverlapDetector
from .detection.secondary_speaker import (
    AdaptiveSecondaryDetection,
    DetectionResult,
    SecondarySpeckerDetector
)

__all__ = [
    # Core
    "AudioEnhancer",
    # Speaker separation
    "SpeakerSeparator", 
    "SeparationConfig",
    # Detection
    "OverlapDetector",
    "AdaptiveSecondaryDetection",
    "DetectionResult",
    "SecondarySpeckerDetector"
]
