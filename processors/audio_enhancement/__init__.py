"""
Audio Enhancement Module

Provides audio quality enhancement capabilities including noise reduction,
voice enhancement, audio restoration, and speaker separation.
"""

from .speaker_separation import SpeakerSeparator, SeparationConfig
from .detection.overlap_detector import OverlapDetector
from .detection.secondary_speaker import (
    AdaptiveSecondaryDetection,
    DetectionResult,
    SecondarySpeckerDetector
)

__all__ = [
    "SpeakerSeparator", 
    "SeparationConfig",
    "OverlapDetector",
    "AdaptiveSecondaryDetection",
    "DetectionResult",
    "SecondarySpeckerDetector"
]