"""
End-Aware Secondary Speaker Removal Module

Specifically designed to handle secondary speakers appearing at the end of audio files.
Implements enhanced processing for the S5 pattern where secondary speakers interrupt
at the end of recordings.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import librosa
from scipy import signal

from .speechbrain_separator import SpeechBrainSeparator, SeparationConfig, SeparationOutput

logger = logging.getLogger(__name__)


@dataclass
class EndSegmentAnalysis:
    """Analysis results for end segment of audio"""
    has_secondary_speaker: bool
    secondary_start_time: float
    confidence: float
    energy_ratio: float
    spectral_change: float


class EndAwareSecondaryRemoval:
    """
    Enhanced secondary speaker removal with special handling for end-of-audio cases.
    """
    
    def __init__(self, 
                 base_separator: Optional[SpeechBrainSeparator] = None,
                 end_analysis_duration: float = 3.0,  # Analyze last 3 seconds
                 end_processing_overlap: float = 0.5,  # Overlap for smooth transition
                 confidence_threshold: float = 0.7):
        """
        Initialize end-aware secondary removal.
        
        Args:
            base_separator: Base SpeechBrain separator (will create if None)
            end_analysis_duration: Duration in seconds to analyze at end
            end_processing_overlap: Overlap duration for processing chunks
            confidence_threshold: Confidence threshold for separation
        """
        self.end_analysis_duration = end_analysis_duration
        self.end_processing_overlap = end_processing_overlap
        self.confidence_threshold = confidence_threshold
        
        # Initialize base separator if not provided
        if base_separator is None:
            config = SeparationConfig(
                confidence_threshold=confidence_threshold,
                speaker_selection="embedding",  # Use embedding for better consistency
                quality_thresholds={
                    "min_stoi": 0.9,  # Higher quality requirement
                    "min_pesq": 4.0,
                    "max_spectral_distortion": 0.1
                }
            )
            self.base_separator = SpeechBrainSeparator(config)
        else:
            self.base_separator = base_separator
        
        self.sample_rate = 16000
        
    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Process audio with enhanced end-segment handling.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            
        Returns:
            Processed audio with secondary speakers removed
        """
        if len(audio) < sample_rate:  # Less than 1 second
            # Too short for meaningful processing
            return self._process_short_audio(audio, sample_rate)
        
        # Analyze end segment
        end_analysis = self._analyze_end_segment(audio, sample_rate)
        
        if not end_analysis.has_secondary_speaker:
            # No secondary speaker detected at end, use standard processing
            result = self.base_separator.separate_speakers(audio, sample_rate)
            # Handle both dict and SeparationOutput formats
            if hasattr(result, 'rejected'):
                return result.audio if not result.rejected else audio
            else:
                # Old dict format
                return result.get('audio', audio)
        
        # Secondary speaker detected at end - use specialized processing
        logger.info(f"Secondary speaker detected at end (start: {end_analysis.secondary_start_time}s)")
        
        # Process in segments with special handling for the end
        processed = self._process_with_end_awareness(audio, sample_rate, end_analysis)
        
        return processed
    
    def _analyze_end_segment(self, audio: np.ndarray, sample_rate: int) -> EndSegmentAnalysis:
        """
        Analyze the end segment of audio for secondary speaker presence.
        """
        end_samples = int(self.end_analysis_duration * sample_rate)
        if len(audio) < end_samples:
            end_samples = len(audio)
        
        end_segment = audio[-end_samples:]
        
        # Detect energy changes that might indicate new speaker
        energy_profile = self._compute_energy_profile(end_segment, sample_rate)
        has_change, change_point = self._detect_energy_change(energy_profile)
        
        # Detect spectral changes
        spectral_change = self._detect_spectral_change(end_segment, sample_rate)
        
        # Detect pitch changes
        pitch_change = self._detect_pitch_change(end_segment, sample_rate)
        
        # Combine evidence
        has_secondary = (has_change and spectral_change > 0.3) or pitch_change > 0.4
        
        if has_secondary and change_point is not None:
            secondary_start_time = len(audio) / sample_rate - (end_samples - change_point) / sample_rate
        else:
            secondary_start_time = len(audio) / sample_rate
        
        return EndSegmentAnalysis(
            has_secondary_speaker=has_secondary,
            secondary_start_time=secondary_start_time,
            confidence=max(spectral_change, pitch_change),
            energy_ratio=1.0,  # Will be calculated if needed
            spectral_change=spectral_change
        )
    
    def _process_with_end_awareness(self, 
                                   audio: np.ndarray, 
                                   sample_rate: int,
                                   end_analysis: EndSegmentAnalysis) -> np.ndarray:
        """
        Process audio with special handling for end segment.
        """
        # Determine split point
        split_point = int(end_analysis.secondary_start_time * sample_rate)
        
        # Add some buffer before the detected point
        buffer_samples = int(0.5 * sample_rate)  # 0.5 second buffer
        split_point = max(sample_rate, split_point - buffer_samples)
        
        # Process main segment (before secondary speaker)
        main_segment = audio[:split_point]
        main_result = self.base_separator.separate_speakers(main_segment, sample_rate)
        
        # Handle both formats
        if hasattr(main_result, 'rejected'):
            if main_result.rejected:
                # Fallback to simple processing
                processed_main = main_segment
            else:
                processed_main = main_result.audio
        else:
            # Old dict format
            processed_main = main_result.get('audio', main_segment)
        
        # Process end segment with enhanced separation
        end_segment = audio[split_point:]
        processed_end = self._process_end_segment_enhanced(end_segment, sample_rate)
        
        # Combine with smooth transition
        combined = self._combine_segments(processed_main, processed_end, sample_rate)
        
        return combined
    
    def _process_end_segment_enhanced(self, segment: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhanced processing for end segment with secondary speaker.
        """
        # Use more aggressive separation for end segment
        config = SeparationConfig(
            confidence_threshold=0.5,  # Lower threshold to ensure separation
            speaker_selection="embedding",
            quality_thresholds={
                "min_stoi": 0.7,  # Relax quality to ensure removal
                "min_pesq": 3.0,
                "max_spectral_distortion": 0.3
            }
        )
        
        # Create temporary separator with aggressive settings
        aggressive_separator = SpeechBrainSeparator(config)
        result = aggressive_separator.separate_speakers(segment, sample_rate)
        
        # Handle both formats
        if hasattr(result, 'rejected'):
            if result.rejected:
                # If separation fails, use frequency-based suppression
                return self._frequency_based_suppression(segment, sample_rate)
            processed = result.audio
            num_speakers = result.num_speakers_detected
        else:
            # Old dict format
            if result.get('metrics', {}).get('rejected', False):
                return self._frequency_based_suppression(segment, sample_rate)
            processed = result.get('audio', segment)
            num_speakers = result.get('metrics', {}).get('num_speakers_detected', 1)
        
        # Check if we successfully removed secondary speaker
        if num_speakers > 1:
            # Apply additional suppression
            processed = self._apply_secondary_suppression(processed, segment, sample_rate)
        
        return processed
    
    def _frequency_based_suppression(self, segment: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Fallback frequency-based suppression for secondary speaker.
        """
        # Estimate primary speaker frequency range (typically 100-300 Hz fundamental)
        # Secondary speakers often have different pitch ranges
        
        # Apply bandpass filter to preserve primary speaker range
        nyquist = sample_rate / 2
        low_freq = 80 / nyquist
        high_freq = 400 / nyquist
        
        # Design filter
        sos = signal.butter(4, [low_freq, high_freq], btype='band', output='sos')
        filtered = signal.sosfilt(sos, segment)
        
        # Mix with original to preserve some naturalness
        return 0.8 * filtered + 0.2 * segment
    
    def _apply_secondary_suppression(self, 
                                    processed: np.ndarray,
                                    original: np.ndarray,
                                    sample_rate: int) -> np.ndarray:
        """
        Apply additional suppression to ensure secondary speaker is removed.
        """
        # Compute spectrograms
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Original spectrum
        orig_stft = librosa.stft(original, n_fft=n_fft, hop_length=hop_length)
        orig_mag = np.abs(orig_stft)
        orig_phase = np.angle(orig_stft)
        
        # Processed spectrum
        proc_stft = librosa.stft(processed, n_fft=n_fft, hop_length=hop_length)
        proc_mag = np.abs(proc_stft)
        
        # Identify frequency bins with significant changes (likely secondary speaker)
        mag_change = np.abs(proc_mag - orig_mag) / (orig_mag + 1e-10)
        secondary_mask = mag_change > 0.5  # Bins with >50% change
        
        # Suppress these bins completely
        suppressed_mag = proc_mag.copy()
        suppressed_mag[secondary_mask] = 0  # Complete suppression
        
        # Reconstruct
        suppressed_stft = suppressed_mag * np.exp(1j * orig_phase)
        suppressed = librosa.istft(suppressed_stft, hop_length=hop_length)
        
        # Ensure same length
        if len(suppressed) > len(processed):
            suppressed = suppressed[:len(processed)]
        elif len(suppressed) < len(processed):
            suppressed = np.pad(suppressed, (0, len(processed) - len(suppressed)))
        
        return suppressed
    
    def _combine_segments(self, 
                         main_segment: np.ndarray,
                         end_segment: np.ndarray,
                         sample_rate: int) -> np.ndarray:
        """
        Combine main and end segments with smooth transition.
        """
        # Create overlap region
        overlap_samples = int(self.end_processing_overlap * sample_rate)
        overlap_samples = min(overlap_samples, len(main_segment) // 4, len(end_segment) // 4)
        
        if overlap_samples > 0:
            # Create smooth transition
            fade_out = np.linspace(1, 0, overlap_samples)
            fade_in = np.linspace(0, 1, overlap_samples)
            
            # Apply fades
            main_segment[-overlap_samples:] *= fade_out
            end_segment[:overlap_samples] *= fade_in
            
            # Combine with overlap
            combined_length = len(main_segment) + len(end_segment) - overlap_samples
            combined = np.zeros(combined_length)
            
            # Copy non-overlapping parts
            combined[:len(main_segment) - overlap_samples] = main_segment[:-overlap_samples]
            combined[len(main_segment):] = end_segment[overlap_samples:]
            
            # Add overlapping part
            combined[len(main_segment) - overlap_samples:len(main_segment)] = (
                main_segment[-overlap_samples:] + end_segment[:overlap_samples]
            )
        else:
            # Simple concatenation
            combined = np.concatenate([main_segment, end_segment])
        
        return combined
    
    def _process_short_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process very short audio clips.
        """
        # For very short audio, just use base separator
        result = self.base_separator.separate_speakers(audio, sample_rate)
        # Handle both formats
        if hasattr(result, 'rejected'):
            return result.audio if not result.rejected else audio
        else:
            # Old dict format
            return result.get('audio', audio)
    
    def _compute_energy_profile(self, segment: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute energy profile over time.
        """
        # Use 50ms windows with 25ms hop
        window_size = int(0.05 * sample_rate)
        hop_size = int(0.025 * sample_rate)
        
        energy = []
        for i in range(0, len(segment) - window_size, hop_size):
            window = segment[i:i + window_size]
            energy.append(np.sqrt(np.mean(window ** 2)))
        
        return np.array(energy)
    
    def _detect_energy_change(self, energy_profile: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Detect significant energy changes that might indicate new speaker.
        """
        if len(energy_profile) < 10:
            return False, None
        
        # Smooth energy profile
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(energy_profile, sigma=2)
        
        # Compute derivative
        energy_diff = np.diff(smoothed)
        
        # Find significant changes
        threshold = np.std(energy_diff) * 2
        change_points = np.where(np.abs(energy_diff) > threshold)[0]
        
        if len(change_points) > 0:
            # Return the last significant change
            return True, change_points[-1]
        
        return False, None
    
    def _detect_spectral_change(self, segment: np.ndarray, sample_rate: int) -> float:
        """
        Detect spectral changes that might indicate different speaker.
        """
        # Compute spectral features over time
        n_fft = 2048
        hop_length = n_fft // 4
        
        stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(stft)
        
        # Compute spectral centroids
        centroids = librosa.feature.spectral_centroid(S=mag, sr=sample_rate)[0]
        
        if len(centroids) < 10:
            return 0.0
        
        # Look for sudden changes in spectral centroid
        first_half = centroids[:len(centroids)//2]
        second_half = centroids[len(centroids)//2:]
        
        mean_change = abs(np.mean(second_half) - np.mean(first_half)) / np.mean(first_half)
        
        return min(mean_change, 1.0)
    
    def _detect_pitch_change(self, segment: np.ndarray, sample_rate: int) -> float:
        """
        Detect pitch changes that might indicate different speaker.
        """
        try:
            # Extract pitch using librosa
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate
            )
            
            # Remove unvoiced frames
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) < 10:
                return 0.0
            
            # Compare pitch statistics between first and second half
            mid_point = len(f0_voiced) // 2
            first_half = f0_voiced[:mid_point]
            second_half = f0_voiced[mid_point:]
            
            # Remove NaN values
            first_half = first_half[~np.isnan(first_half)]
            second_half = second_half[~np.isnan(second_half)]
            
            if len(first_half) == 0 or len(second_half) == 0:
                return 0.0
            
            # Compute change in mean pitch
            pitch_change = abs(np.mean(second_half) - np.mean(first_half)) / np.mean(first_half)
            
            return min(pitch_change, 1.0)
            
        except Exception as e:
            logger.warning(f"Pitch detection failed: {e}")
            return 0.0