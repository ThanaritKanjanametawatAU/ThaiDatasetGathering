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
from .pattern_detector import (
    PatternDetector,
    TemporalPattern,
    SpectralPattern,
    NoiseProfile,
    CodecArtifact,
    PatternType,
    PatternSeverity,
    PatternReport
)

__all__ = [
    "OverlapDetector",
    "SecondarySpeakerDetector", 
    "AdaptiveSecondaryDetection",
    "DetectionResult",
    "PatternDetector",
    "TemporalPattern",
    "SpectralPattern",
    "NoiseProfile",
    "CodecArtifact",
    "PatternType",
    "PatternSeverity",
    "PatternReport"
]