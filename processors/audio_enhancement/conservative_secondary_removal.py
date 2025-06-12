"""
Conservative Secondary Speaker Removal Module

This module implements a conservative approach to secondary speaker detection and removal
that prioritizes preserving the primary speaker over aggressive removal.

Key principles:
1. Require strong evidence before classifying audio as secondary speaker
2. Never remove more than 20% of total audio duration
3. Use multiple validation checks before removal
4. Preserve primary speaker continuity at all costs
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import librosa
from scipy import signal
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class ConservativeDetectionConfig:
    """Configuration for conservative secondary speaker detection."""
    
    # Evidence thresholds (all must be met for removal)
    min_speaker_change_confidence: float = 0.85  # Very high confidence required
    min_spectral_change_threshold: float = 0.7   # Strong spectral difference required
    min_energy_change_ratio: float = 3.0         # Significant energy change required
    min_fundamental_freq_change: float = 50      # Hz difference in F0
    
    # Safety limits
    max_removal_ratio: float = 0.20              # Never remove > 20% of audio
    min_remaining_duration: float = 2.0          # Always keep at least 2 seconds
    min_segment_duration: float = 0.5            # Only analyze segments > 0.5s
    
    # Analysis parameters
    spectral_analysis_window: float = 0.25       # 250ms windows for analysis
    overlap_ratio: float = 0.5                   # 50% overlap between windows
    mfcc_coefficients: int = 13                  # Number of MFCC features
    
    # Validation requirements
    require_all_evidence: bool = True            # All checks must pass
    validate_speaker_continuity: bool = True     # Check speaker profile consistency
    preserve_speech_transitions: bool = True     # Don't cut mid-word


@dataclass
class EvidenceReport:
    """Report of evidence for secondary speaker presence."""
    
    spectral_change: float
    energy_change: float
    fundamental_freq_change: float
    speaker_similarity: float
    speech_continuity: bool
    meets_safety_limits: bool
    
    @property
    def has_strong_evidence(self) -> bool:
        """Check if we have strong evidence for secondary speaker."""
        return (
            self.spectral_change > 0.7 and
            self.energy_change > 3.0 and
            self.fundamental_freq_change > 50 and
            self.speaker_similarity < 0.6 and  # Low similarity = different speaker
            not self.speech_continuity and     # Not continuous speech
            self.meets_safety_limits           # Within safety limits
        )


class ConservativeSecondaryRemoval:
    """
    Conservative approach to secondary speaker removal.
    
    This class implements a highly conservative strategy that only removes audio
    when there is overwhelming evidence of a secondary speaker, while always
    preserving the primary speaker's content.
    """
    
    def __init__(self, config: ConservativeDetectionConfig = None):
        """Initialize conservative secondary speaker removal."""
        self.config = config or ConservativeDetectionConfig()
        
        logger.info("Initialized Conservative Secondary Speaker Removal")
        logger.info(f"Max removal ratio: {self.config.max_removal_ratio}")
        logger.info(f"Min confidence required: {self.config.min_speaker_change_confidence}")
    
    def process_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process audio with conservative secondary speaker removal.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        if len(audio) < self.config.min_remaining_duration * sample_rate:
            logger.info("Audio too short for secondary speaker analysis")
            return audio, {"action": "skipped", "reason": "too_short"}
        
        # Step 1: Analyze audio for potential secondary speaker regions
        candidate_regions = self._identify_candidate_regions(audio, sample_rate)
        
        if not candidate_regions:
            logger.info("No candidate secondary speaker regions found")
            return audio, {"action": "no_change", "reason": "no_candidates"}
        
        # Step 2: Gather evidence for each candidate region
        evidence_reports = []
        for region in candidate_regions:
            evidence = self._gather_evidence(audio, region, sample_rate)
            evidence_reports.append((region, evidence))
        
        # Step 3: Apply conservative removal only for regions with strong evidence
        processed_audio, removal_metadata = self._apply_conservative_removal(
            audio, evidence_reports, sample_rate
        )
        
        return processed_audio, removal_metadata
    
    def _identify_candidate_regions(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, float]]:
        """
        Identify candidate regions that might contain secondary speakers.
        
        This is intentionally conservative - we only look for obvious changes.
        """
        candidates = []
        
        # Use larger analysis windows for stability
        window_size = int(self.config.spectral_analysis_window * sample_rate)
        hop_size = int(window_size * (1 - self.config.overlap_ratio))
        
        # Extract features for each window
        windows = []
        for i in range(0, len(audio) - window_size + 1, hop_size):
            window = audio[i:i + window_size]
            windows.append({
                'start_time': i / sample_rate,
                'end_time': (i + window_size) / sample_rate,
                'start_idx': i,
                'end_idx': i + window_size,
                'audio': window
            })
        
        if len(windows) < 3:  # Need at least 3 windows for comparison
            return candidates
        
        # Calculate features for each window
        for window in windows:
            window['features'] = self._extract_window_features(window['audio'], sample_rate)
        
        # Look for significant changes between windows
        for i in range(1, len(windows) - 1):  # Skip first and last (boundary effects)
            prev_window = windows[i - 1]
            curr_window = windows[i]
            
            # Calculate feature differences
            spectral_change = self._calculate_spectral_distance(
                prev_window['features']['mfcc'],
                curr_window['features']['mfcc']
            )
            
            energy_change = abs(
                curr_window['features']['energy'] - prev_window['features']['energy']
            ) / max(prev_window['features']['energy'], 1e-6)
            
            # Only consider as candidate if change is significant
            if (spectral_change > self.config.min_spectral_change_threshold and 
                energy_change > self.config.min_energy_change_ratio):
                
                # Extend candidate region to end of audio (conservative assumption)
                candidates.append({
                    'start_time': curr_window['start_time'],
                    'end_time': len(audio) / sample_rate,
                    'start_idx': curr_window['start_idx'],
                    'end_idx': len(audio),
                    'spectral_change': spectral_change,
                    'energy_change': energy_change
                })
                
                logger.info(f"Candidate region found at {curr_window['start_time']:.2f}s "
                          f"(spectral_change={spectral_change:.3f}, "
                          f"energy_change={energy_change:.3f})")
                break  # Only take first significant change to be conservative
        
        return candidates
    
    def _extract_window_features(self, window: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract features from an audio window."""
        features = {}
        
        # Energy
        features['energy'] = np.sqrt(np.mean(window ** 2))
        
        # MFCC features
        try:
            mfcc = librosa.feature.mfcc(
                y=window, 
                sr=sample_rate, 
                n_mfcc=self.config.mfcc_coefficients
            )
            features['mfcc'] = np.mean(mfcc, axis=1)  # Average over time
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            features['mfcc'] = np.zeros(self.config.mfcc_coefficients)
        
        # Fundamental frequency
        try:
            f0, _, _ = librosa.pyin(
                window, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate
            )
            valid_f0 = f0[~np.isnan(f0)]
            features['f0'] = np.median(valid_f0) if len(valid_f0) > 0 else 0
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}")
            features['f0'] = 0
        
        # Spectral centroid
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=window, sr=sample_rate)
            features['spectral_centroid'] = np.mean(spectral_centroid)
        except Exception as e:
            logger.warning(f"Spectral centroid extraction failed: {e}")
            features['spectral_centroid'] = 0
        
        return features
    
    def _calculate_spectral_distance(self, mfcc1: np.ndarray, mfcc2: np.ndarray) -> float:
        """Calculate spectral distance between two MFCC vectors."""
        try:
            # Use cosine distance for robustness
            distance = cosine(mfcc1, mfcc2)
            return distance
        except Exception as e:
            logger.warning(f"Spectral distance calculation failed: {e}")
            return 0.0
    
    def _gather_evidence(self, audio: np.ndarray, region: Dict[str, Any], sample_rate: int) -> EvidenceReport:
        """
        Gather comprehensive evidence for secondary speaker presence in a region.
        """
        start_idx = region['start_idx']
        end_idx = region['end_idx']
        
        # Extract primary speaker profile from first 30% of audio (conservative)
        primary_end = min(len(audio), int(0.3 * len(audio)))
        primary_segment = audio[:primary_end]
        
        # Extract candidate secondary segment
        candidate_segment = audio[start_idx:end_idx]
        
        if len(candidate_segment) < 0.1 * sample_rate:  # Less than 100ms
            return EvidenceReport(
                spectral_change=0, energy_change=0, fundamental_freq_change=0,
                speaker_similarity=1.0, speech_continuity=True, meets_safety_limits=False
            )
        
        # Extract features
        primary_features = self._extract_window_features(primary_segment, sample_rate)
        candidate_features = self._extract_window_features(candidate_segment, sample_rate)
        
        # Calculate evidence metrics
        spectral_change = self._calculate_spectral_distance(
            primary_features['mfcc'], candidate_features['mfcc']
        )
        
        energy_change = abs(
            candidate_features['energy'] - primary_features['energy']
        ) / max(primary_features['energy'], 1e-6)
        
        f0_change = abs(candidate_features['f0'] - primary_features['f0'])
        
        # Speaker similarity (1.0 = identical, 0.0 = completely different)
        speaker_similarity = 1.0 - spectral_change
        
        # Check speech continuity (look for natural pauses)
        speech_continuity = self._check_speech_continuity(audio, start_idx, sample_rate)
        
        # Safety limits check
        removal_duration = (end_idx - start_idx) / sample_rate
        total_duration = len(audio) / sample_rate
        removal_ratio = removal_duration / total_duration
        remaining_duration = total_duration - removal_duration
        
        meets_safety_limits = (
            removal_ratio <= self.config.max_removal_ratio and
            remaining_duration >= self.config.min_remaining_duration
        )
        
        evidence = EvidenceReport(
            spectral_change=spectral_change,
            energy_change=energy_change,
            fundamental_freq_change=f0_change,
            speaker_similarity=speaker_similarity,
            speech_continuity=speech_continuity,
            meets_safety_limits=meets_safety_limits
        )
        
        logger.info(f"Evidence for region {start_idx/sample_rate:.2f}-{end_idx/sample_rate:.2f}s:")
        logger.info(f"  Spectral change: {spectral_change:.3f}")
        logger.info(f"  Energy change: {energy_change:.3f}")
        logger.info(f"  F0 change: {f0_change:.1f} Hz")
        logger.info(f"  Speaker similarity: {speaker_similarity:.3f}")
        logger.info(f"  Speech continuity: {speech_continuity}")
        logger.info(f"  Safety limits met: {meets_safety_limits}")
        logger.info(f"  Strong evidence: {evidence.has_strong_evidence}")
        
        return evidence
    
    def _check_speech_continuity(self, audio: np.ndarray, transition_idx: int, sample_rate: int) -> bool:
        """
        Check if there's natural speech continuity at the transition point.
        
        Returns True if speech appears continuous (shouldn't cut here).
        """
        # Look at 200ms before and after transition
        window_size = int(0.2 * sample_rate)
        start = max(0, transition_idx - window_size)
        end = min(len(audio), transition_idx + window_size)
        
        transition_region = audio[start:end]
        
        # Check for natural pause or energy drop
        pre_transition = audio[max(0, transition_idx - window_size):transition_idx]
        post_transition = audio[transition_idx:min(len(audio), transition_idx + window_size)]
        
        if len(pre_transition) == 0 or len(post_transition) == 0:
            return False
        
        pre_energy = np.sqrt(np.mean(pre_transition ** 2))
        post_energy = np.sqrt(np.mean(post_transition ** 2))
        
        # If there's a significant energy drop, it might be a natural pause
        if post_energy < 0.3 * pre_energy:
            return False  # Natural pause, okay to cut
        
        # Otherwise, assume continuous speech
        return True
    
    def _apply_conservative_removal(
        self, 
        audio: np.ndarray, 
        evidence_reports: List[Tuple[Dict[str, Any], EvidenceReport]], 
        sample_rate: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply conservative removal only to regions with overwhelming evidence.
        """
        regions_to_remove = []
        
        for region, evidence in evidence_reports:
            if evidence.has_strong_evidence:
                regions_to_remove.append(region)
                logger.info(f"Approved for removal: {region['start_time']:.2f}-{region['end_time']:.2f}s")
            else:
                logger.info(f"Insufficient evidence for removal: {region['start_time']:.2f}-{region['end_time']:.2f}s")
        
        if not regions_to_remove:
            return audio, {
                "action": "no_change",
                "reason": "insufficient_evidence",
                "candidates_analyzed": len(evidence_reports)
            }
        
        # Apply removal (for now, just the first region to be extra conservative)
        region_to_remove = regions_to_remove[0]
        start_idx = region_to_remove['start_idx']
        
        # Apply gradual fade instead of hard cut to preserve naturalness
        processed_audio = audio.copy()
        fade_duration = int(0.1 * sample_rate)  # 100ms fade
        
        if start_idx + fade_duration < len(processed_audio):
            # Apply fade
            fade_curve = np.linspace(1.0, 0.0, fade_duration)
            processed_audio[start_idx:start_idx + fade_duration] *= fade_curve
            
            # Remove the rest
            processed_audio[start_idx + fade_duration:] = 0
        else:
            # Just fade to end
            remaining = len(processed_audio) - start_idx
            if remaining > 0:
                fade_curve = np.linspace(1.0, 0.0, remaining)
                processed_audio[start_idx:] *= fade_curve
        
        metadata = {
            "action": "conservative_removal",
            "removed_start_time": region_to_remove['start_time'],
            "removed_end_time": region_to_remove['end_time'],
            "removal_duration": region_to_remove['end_time'] - region_to_remove['start_time'],
            "evidence_score": evidence_reports[0][1].spectral_change,
            "total_candidates": len(evidence_reports),
            "regions_removed": len(regions_to_remove)
        }
        
        logger.info(f"Conservative removal applied from {region_to_remove['start_time']:.2f}s to end")
        
        return processed_audio, metadata


def test_conservative_removal():
    """Test function for conservative secondary speaker removal."""
    # Create test audio with obvious speaker change
    sample_rate = 16000
    duration = 5.0
    
    # Primary speaker: 440 Hz tone for first 3 seconds
    t1 = np.linspace(0, 3, int(3 * sample_rate))
    primary = 0.5 * np.sin(2 * np.pi * 440 * t1)
    
    # Secondary speaker: 880 Hz tone for last 2 seconds
    t2 = np.linspace(0, 2, int(2 * sample_rate))
    secondary = 0.3 * np.sin(2 * np.pi * 880 * t2)
    
    # Combine with small gap
    gap = np.zeros(int(0.1 * sample_rate))
    test_audio = np.concatenate([primary, gap, secondary])
    
    # Test conservative removal
    remover = ConservativeSecondaryRemoval()
    processed, metadata = remover.process_audio(test_audio, sample_rate)
    
    print("Test Results:")
    print(f"Original duration: {len(test_audio) / sample_rate:.2f}s")
    print(f"Processed duration: {len(processed) / sample_rate:.2f}s")
    print(f"Metadata: {metadata}")
    
    # Check that primary speaker is preserved
    primary_end = int(3 * sample_rate)
    primary_preserved = np.mean(np.abs(processed[:primary_end] - test_audio[:primary_end])) < 0.01
    print(f"Primary speaker preserved: {primary_preserved}")


if __name__ == "__main__":
    test_conservative_removal()