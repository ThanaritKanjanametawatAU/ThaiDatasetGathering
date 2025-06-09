"""
Post-processing module for audio enhancement
Includes artifact removal, spectral smoothing, and level normalization
"""

import numpy as np
import logging
from typing import Optional, Dict, Any
from scipy import signal
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


class ArtifactRemover:
    """Remove clicks, pops, and other artifacts from audio"""
    
    def __init__(self, threshold_factor: float = 3.0):
        """
        Initialize artifact remover.
        
        Args:
            threshold_factor: Factor for determining outliers (default: 3.0 std devs)
        """
        self.threshold_factor = threshold_factor
        
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Remove artifacts from audio.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Audio with artifacts removed
        """
        # Make a copy to avoid modifying input
        cleaned = audio.copy()
        
        # Detect clicks/pops using differential
        diff = np.abs(np.diff(cleaned))
        median_diff = np.median(diff)
        std_diff = np.std(diff)
        
        # Threshold for detecting artifacts
        threshold = median_diff + self.threshold_factor * std_diff
        
        # Find artifact locations
        artifact_indices = np.where(diff > threshold)[0]
        
        if len(artifact_indices) > 0:
            # Interpolate over artifacts
            for idx in artifact_indices:
                if 0 < idx < len(cleaned) - 1:
                    # Simple linear interpolation
                    cleaned[idx] = (cleaned[idx-1] + cleaned[idx+1]) / 2
                    
            # Apply median filter to smooth any remaining artifacts
            window_size = int(0.001 * sample_rate)  # 1ms window
            if window_size % 2 == 0:
                window_size += 1
            cleaned = median_filter(cleaned, size=window_size)
            
        return cleaned


class SpectralSmoother:
    """Apply spectral smoothing to reduce harshness"""
    
    def __init__(self, smoothing_factor: float = 0.8):
        """
        Initialize spectral smoother.
        
        Args:
            smoothing_factor: Amount of smoothing (0-1, higher = more smoothing)
        """
        self.smoothing_factor = np.clip(smoothing_factor, 0.0, 1.0)
        
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply spectral smoothing.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Spectrally smoothed audio
        """
        # Convert float16 to float32 for scipy compatibility
        original_dtype = audio.dtype
        if audio.dtype == np.float16:
            audio = audio.astype(np.float32)
            
        # STFT parameters
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Compute STFT
        f, t, stft = signal.stft(audio, fs=sample_rate, nperseg=n_fft, 
                                noverlap=n_fft-hop_length)
        
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply spectral smoothing
        # Reduce high frequency content based on smoothing factor
        freq_bins = magnitude.shape[0]
        
        # Create frequency-dependent smoothing curve
        # More smoothing at higher frequencies
        smoothing_curve = np.ones(freq_bins)
        cutoff_bin = int(4000 * freq_bins / (sample_rate / 2))  # Start smoothing at 4kHz
        
        if cutoff_bin < freq_bins:
            # Linear ramp from 1.0 to (1 - smoothing_factor)
            ramp_bins = freq_bins - cutoff_bin
            smoothing_curve[cutoff_bin:] = np.linspace(
                1.0, 1.0 - self.smoothing_factor, ramp_bins
            )
            
        # Apply smoothing
        magnitude *= smoothing_curve[:, np.newaxis]
        
        # Reconstruct
        stft_smoothed = magnitude * np.exp(1j * phase)
        _, smoothed = signal.istft(stft_smoothed, fs=sample_rate, 
                                  nperseg=n_fft, noverlap=n_fft-hop_length)
        
        # Ensure same length as input
        if len(smoothed) > len(audio):
            smoothed = smoothed[:len(audio)]
        elif len(smoothed) < len(audio):
            smoothed = np.pad(smoothed, (0, len(audio) - len(smoothed)))
            
        return smoothed.astype(original_dtype)


class LevelNormalizer:
    """Normalize audio levels to target dB"""
    
    def __init__(self, target_db: float = -20.0, headroom_db: float = -1.0):
        """
        Initialize level normalizer.
        
        Args:
            target_db: Target RMS level in dB
            headroom_db: Maximum peak level in dB (for limiting)
        """
        self.target_db = target_db
        self.headroom_db = headroom_db
        
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Normalize audio level.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Level-normalized audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        
        # Avoid log of zero
        if rms < 1e-10:
            return audio
            
        # Current level in dB
        current_db = 20 * np.log10(rms)
        
        # Calculate gain needed
        gain_db = self.target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain_linear
        
        # Check for clipping and apply limiting if needed
        peak = np.max(np.abs(normalized))
        max_peak = 10 ** (self.headroom_db / 20)
        
        if peak > max_peak:
            # Simple limiting
            limit_gain = max_peak / peak
            normalized *= limit_gain
            
        return normalized