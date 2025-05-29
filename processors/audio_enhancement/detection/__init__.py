"""
Audio Enhancement Detection Module

Provides detection capabilities for speaker separation and audio quality enhancement.
"""

from .overlap_detector import OverlapDetector
from .secondary_speaker import (
    SecondarySpeakerDetector,
    AdaptiveSecondaryDetection,
    DetectionResult
)

__all__ = [
    "OverlapDetector",
    "SecondarySpeakerDetector", 
    "AdaptiveSecondaryDetection",
    "DetectionResult"
]