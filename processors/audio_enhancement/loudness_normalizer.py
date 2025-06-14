#!/usr/bin/env python3
"""
Loudness Normalizer - Matches processed audio loudness to original

This module provides loudness normalization after audio preprocessing to ensure
the processed audio has the same loudness as the original while preserving
the quality improvements from noise reduction.
"""

import numpy as np
import warnings
from typing import Union, List, Optional, Literal
import pyloudnorm as pyln

warnings.filterwarnings('ignore')


class LoudnessNormalizer:
    """
    Normalizes audio loudness to match a reference level.
    
    Supports multiple normalization methods:
    - RMS (Root Mean Square) - Simple and effective for most cases
    - Peak - Normalizes based on maximum amplitude
    - LUFS (Loudness Units Full Scale) - Broadcast standard loudness
    """
    
    def __init__(self):
        self.soft_limiting_applied = False
        self._lufs_meter = None
    
    def normalize_loudness(
        self, 
        audio: np.ndarray, 
        reference: np.ndarray, 
        sr: int,
        method: Literal['rms', 'peak', 'lufs'] = 'rms',
        headroom_db: float = -1.0,
        soft_limit: bool = True,
        target_lufs: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Normalize audio loudness to match reference.
        
        Args:
            audio: Processed audio to normalize
            reference: Original audio to match loudness
            sr: Sample rate
            method: Normalization method ('rms', 'peak', 'lufs')
            headroom_db: Headroom in dB to prevent clipping (default -1dB)
            soft_limit: Apply soft limiting if needed to prevent clipping
            target_lufs: Target LUFS for LUFS method (uses reference if None)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Normalized audio with loudness matching reference
        """
        self.soft_limiting_applied = False
        
        # Handle edge cases
        if len(audio) == 0 or len(reference) == 0:
            return audio
        
        # Check if either audio is silent
        if np.max(np.abs(reference)) < 1e-8:
            # If reference is silent, return silent audio
            return np.zeros_like(audio)
        
        if np.max(np.abs(audio)) < 1e-6:
            # If processed audio is essentially silent, keep it silent
            return audio
        
        # Don't trim - use full audio for better loudness estimation
        # Calculate gain based on method
        if method == 'rms':
            gain = self._calculate_rms_gain(audio, reference)
        elif method == 'peak':
            gain = self._calculate_peak_gain(audio, reference)
        elif method == 'lufs':
            gain = self._calculate_lufs_gain(
                audio, reference, sr, target_lufs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply gain - for extreme cases, use a careful approach
        if gain > 2.0:  # Lower threshold for special handling
            # For large gains, apply directly but check for clipping
            normalized = audio * gain
            
            # Apply headroom and soft limiting if needed
            headroom_linear = 10 ** (headroom_db / 20)
            max_val = np.max(np.abs(normalized))
            
            if max_val > headroom_linear:
                if soft_limit:
                    normalized = self._apply_soft_limiting(normalized, headroom_linear)
                    self.soft_limiting_applied = True
                else:
                    normalized = normalized * (headroom_linear / max_val)
        else:
            # Normal gain case
            normalized = audio * gain
            
            # Apply headroom
            headroom_linear = 10 ** (headroom_db / 20)
            max_val = np.max(np.abs(normalized))
            
            if max_val > headroom_linear:
                if soft_limit:
                    normalized = self._apply_soft_limiting(normalized, headroom_linear)
                    self.soft_limiting_applied = True
                else:
                    normalized = normalized * (headroom_linear / max_val)
        
        return normalized
    
    def _calculate_rms_gain(self, audio: np.ndarray, reference: np.ndarray) -> float:
        """Calculate gain to match RMS levels"""
        # Calculate RMS
        audio_rms = np.sqrt(np.mean(audio**2))
        reference_rms = np.sqrt(np.mean(reference**2))
        
        # Handle silence
        if audio_rms < 1e-8 or reference_rms < 1e-8:
            return 1.0
        
        gain = reference_rms / audio_rms
        
        # Don't clip gain - let the soft limiter handle extreme cases
        return gain
    
    def _calculate_peak_gain(self, audio: np.ndarray, reference: np.ndarray) -> float:
        """Calculate gain to match peak levels"""
        audio_peak = np.max(np.abs(audio))
        reference_peak = np.max(np.abs(reference))
        
        # Handle silence
        if audio_peak < 1e-8 or reference_peak < 1e-8:
            return 1.0
        
        gain = reference_peak / audio_peak
        
        # Don't clip gain - let the soft limiter handle extreme cases
        return gain
    
    def _calculate_lufs_gain(
        self, 
        audio: np.ndarray, 
        reference: np.ndarray, 
        sr: int,
        target_lufs: Optional[float] = None
    ) -> float:
        """Calculate gain to match LUFS levels"""
        # Calculate LUFS for both
        audio_lufs = self.calculate_lufs(audio, sr)
        
        if target_lufs is None:
            reference_lufs = self.calculate_lufs(reference, sr)
            target_lufs = reference_lufs
        
        # Handle very quiet signals
        if audio_lufs < -60:  # Very quiet
            # Fall back to RMS-based calculation
            return self._calculate_rms_gain(audio, reference)
        
        # Calculate gain in dB
        gain_db = target_lufs - audio_lufs
        
        # Limit extreme gains
        gain_db = np.clip(gain_db, -20, 40)  # -20dB to +40dB range
        
        # Convert to linear
        return 10 ** (gain_db / 20)
    
    def calculate_lufs(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate integrated LUFS (Loudness Units Full Scale).
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Integrated LUFS value
        """
        # Initialize meter if needed
        if self._lufs_meter is None or self._lufs_meter.rate != sr:
            self._lufs_meter = pyln.Meter(sr)
        
        # Handle silence
        if np.max(np.abs(audio)) < 1e-8:
            return -70.0  # Very quiet
        
        # Calculate integrated loudness
        try:
            loudness = self._lufs_meter.integrated_loudness(audio)
            # Handle -inf for very quiet signals
            if np.isinf(loudness):
                return -70.0
            return float(loudness)
        except Exception:
            # Fallback to RMS-based estimate if LUFS fails
            rms = np.sqrt(np.mean(audio**2))
            if rms < 1e-8:
                return -70.0
            return 20 * np.log10(rms) - 3.0  # Rough approximation
    
    def _apply_soft_limiting(self, audio: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply transparent soft limiting that preserves perceived loudness.
        
        Uses a combination of soft clipping and dynamic range compression.
        """
        output = audio.copy()
        
        # Parameters for transparent limiting
        knee_start = threshold * 0.7  # Start compression at 70% of threshold
        
        # Apply different processing for different ranges
        abs_audio = np.abs(audio)
        
        # Region 1: Below knee - no processing
        below_knee = abs_audio <= knee_start
        
        # Region 2: Soft knee region - gentle compression
        in_knee = (abs_audio > knee_start) & (abs_audio <= threshold)
        if np.any(in_knee):
            # Smooth compression in knee region
            knee_samples = abs_audio[in_knee]
            # Quadratic compression curve
            position_in_knee = (knee_samples - knee_start) / (threshold - knee_start)
            compression_amount = position_in_knee ** 2
            target_levels = knee_start + (threshold - knee_start) * (1 - compression_amount * 0.3)
            gain = target_levels / knee_samples
            output[in_knee] = audio[in_knee] * gain
        
        # Region 3: Above threshold - soft clipping
        above_threshold = abs_audio > threshold
        if np.any(above_threshold):
            over_samples = audio[above_threshold]
            abs_over = abs_audio[above_threshold]
            
            # Use sigmoid-style soft clipping
            excess = (abs_over - threshold) / threshold
            # Soft clipping curve that approaches but never exceeds threshold * 1.1
            clipped_excess = (2/np.pi) * np.arctan(excess * np.pi)
            new_magnitude = threshold + clipped_excess * threshold * 0.1
            
            # Apply with sign preservation
            output[above_threshold] = np.sign(over_samples) * new_magnitude
        
        return output
    
    def _apply_adaptive_compression(self, audio: np.ndarray, reference: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply adaptive compression to match reference loudness without extreme gain.
        
        This is used when simple gain would cause too much amplification.
        """
        # Calculate current and target RMS
        current_rms = np.sqrt(np.mean(audio**2))
        target_rms = np.sqrt(np.mean(reference**2))
        
        if current_rms < 1e-8:
            return audio
        
        # Apply multiband compression approach
        # Split into frequency bands for more natural compression
        nyquist = sr // 2
        
        # Simple 2-band approach
        low_cutoff = 200  # Hz
        
        # Apply highpass and lowpass filters
        from scipy import signal
        
        # Design filters
        sos_hp = signal.butter(4, low_cutoff / nyquist, 'high', output='sos')
        sos_lp = signal.butter(4, low_cutoff / nyquist, 'low', output='sos')
        
        # Split into bands
        low_band = signal.sosfilt(sos_lp, audio)
        high_band = signal.sosfilt(sos_hp, audio)
        
        # Apply different compression to each band
        # Low frequencies: less aggressive
        low_rms = np.sqrt(np.mean(low_band**2)) + 1e-8
        low_gain = min((target_rms * 0.4) / low_rms, 3.0)  # Max 3x gain for lows
        low_band *= low_gain
        
        # High frequencies: more aggressive 
        high_rms = np.sqrt(np.mean(high_band**2)) + 1e-8
        high_gain = min((target_rms * 0.6) / high_rms, 6.0)  # Max 6x gain for highs
        high_band *= high_gain
        
        # Recombine
        compressed = low_band + high_band
        
        # Final adjustment to match target
        compressed_rms = np.sqrt(np.mean(compressed**2))
        if compressed_rms > 1e-8:
            final_gain = target_rms / compressed_rms
            # Limit final gain to prevent artifacts
            final_gain = min(final_gain, 2.0)
            compressed *= final_gain
        
        return compressed
    
    def _apply_frequency_weighted_gain(self, audio: np.ndarray, gain: float, sr: int) -> np.ndarray:
        """
        Apply gain with frequency weighting to preserve natural sound.
        
        Higher frequencies get slightly less gain to avoid harshness.
        """
        if gain <= 1.5:
            # For small gains, just apply directly
            return audio * gain
        
        # For larger gains, use frequency weighting
        from scipy import signal
        
        # Design a simple 3-band system
        low_freq = 250  # Hz
        high_freq = 2000  # Hz
        nyquist = sr // 2
        
        # Create filters
        sos_low = signal.butter(4, low_freq / nyquist, 'low', output='sos')
        sos_mid = signal.butter(4, [low_freq / nyquist, high_freq / nyquist], 'band', output='sos')
        sos_high = signal.butter(4, high_freq / nyquist, 'high', output='sos')
        
        # Split into bands
        low_band = signal.sosfilt(sos_low, audio)
        mid_band = signal.sosfilt(sos_mid, audio)
        high_band = signal.sosfilt(sos_high, audio)
        
        # Apply frequency-dependent gain with scaling based on gain magnitude
        if gain > 10:
            # For very large gains, be more conservative with high frequencies
            low_gain = min(gain, 15.0)
            mid_gain = min(gain * 0.7, 12.0)
            high_gain = min(gain * 0.4, 8.0)
        else:
            # For moderate gains, use gentler frequency shaping
            low_gain = gain
            mid_gain = gain * 0.85
            high_gain = gain * 0.7
        
        low_band *= low_gain
        mid_band *= mid_gain
        high_band *= high_gain
        
        # Recombine
        return low_band + mid_band + high_band
    
    def normalize_batch(
        self, 
        audio_list: List[np.ndarray], 
        reference_list: List[np.ndarray], 
        sr: int,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Normalize a batch of audio files.
        
        Args:
            audio_list: List of processed audio arrays
            reference_list: List of reference audio arrays
            sr: Sample rate (same for all)
            **kwargs: Arguments passed to normalize_loudness
            
        Returns:
            List of normalized audio arrays
        """
        if len(audio_list) != len(reference_list):
            raise ValueError("Audio and reference lists must have same length")
        
        normalized_list = []
        
        for audio, reference in zip(audio_list, reference_list):
            normalized = self.normalize_loudness(audio, reference, sr, **kwargs)
            normalized_list.append(normalized)
        
        return normalized_list