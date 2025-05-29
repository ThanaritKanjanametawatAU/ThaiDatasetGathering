"""
Audio Enhancement Detection Module

Provides detection capabilities for speaker separation and audio quality enhancement.
"""

from .overlap_detector import OverlapDetector
from .secondary_speaker import (
    SecondarySpeckerDetector,
    AdaptiveSecondaryDetection,
    DetectionResult
)

__all__ = [
    "OverlapDetector",
    "SecondarySpeckerDetector", 
    "AdaptiveSecondaryDetection",
    "DetectionResult"
]