"""
Full audio secondary speaker removal.
When secondary speakers are detected, process the entire audio instead of just detected segments.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from scipy.signal import butter, sosfilt, stft, istft
from scipy.ndimage import gaussian_filter1d

from .simple_secondary_removal import SimpleSecondaryRemoval

logger = logging.getLogger(__name__)


class FullAudioSecondaryRemoval:
    """
    Enhanced secondary speaker removal that processes entire audio
    when secondary speakers are detected.
    """
    
    def __init__(self,
                 detection_threshold: float = 0.5,
                 min_silence_duration: float = 0.05,
                 preserve_main_freq_range: Tuple[int, int] = (250, 3500),
                 suppression_strength: float = 0.7,
                 use_adaptive_filtering: bool = True):
        """
        Initialize full audio secondary speaker remover.
        
        Args:
            detection_threshold: Threshold for detecting secondary speakers
            min_silence_duration: Minimum silence between speakers
            preserve_main_freq_range: Frequency range to preserve (Hz)
            suppression_strength: How aggressively to suppress (0-1)
            use_adaptive_filtering: Use adaptive filtering based on audio characteristics
        """
        self.detector = SimpleSecondaryRemoval(
            energy_threshold=detection_threshold,
            min_silence_duration=min_silence_duration,
            suppression_db=-40,
            use_spectral_masking=True
        )
        
        self.preserve_main_freq_range = preserve_main_freq_range
        self.suppression_strength = suppression_strength
        self.use_adaptive_filtering = use_adaptive_filtering
    
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Remove secondary speakers from audio.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        metadata = {
            'secondary_speaker_detected': False,
            'num_segments': 0,
            'processing_applied': False,
            'suppression_method': 'none'
        }
        
        # Ensure float32 for compatibility
        if audio.dtype == np.float16:
            audio = audio.astype(np.float32)
        
        # Detect secondary speakers
        segments = self.detector.detect_speaker_changes(audio, sample_rate)
        
        if not segments:
            logger.debug("No secondary speakers detected")
            return audio, metadata
        
        # Update metadata
        metadata['secondary_speaker_detected'] = True
        metadata['num_segments'] = len(segments)
        metadata['processing_applied'] = True
        
        logger.info(f"Secondary speakers detected in {len(segments)} segments, applying full audio processing")
        
        # Apply full audio processing
        if self.use_adaptive_filtering:
            processed = self._adaptive_full_audio_removal(audio, sample_rate, segments)
            metadata['suppression_method'] = 'adaptive_full'
        else:
            processed = self._basic_full_audio_removal(audio, sample_rate)
            metadata['suppression_method'] = 'basic_full'
        
        # Calculate improvement
        original_power = np.mean(audio**2)
        processed_power = np.mean(processed**2)
        if original_power > 0:
            metadata['power_reduction_db'] = 10 * np.log10(processed_power / original_power)
        
        return processed, metadata
    
    def _basic_full_audio_removal(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Basic full audio removal using bandpass filtering and spectral masking.
        """
        # Bandpass filter to preserve main speech frequencies
        nyquist = sample_rate / 2
        low_freq = self.preserve_main_freq_range[0] / nyquist
        high_freq = self.preserve_main_freq_range[1] / nyquist
        
        # Ensure frequencies are valid
        low_freq = max(0.01, min(low_freq, 0.98))
        high_freq = max(low_freq + 0.01, min(high_freq, 0.99))
        
        # Design filter
        sos = butter(4, [low_freq, high_freq], btype='band', output='sos')
        filtered = sosfilt(sos, audio)
        
        # Apply spectral consistency masking
        cleaned = self._apply_spectral_consistency_mask(filtered, sample_rate)
        
        # Mix with original based on suppression strength
        result = cleaned * self.suppression_strength + audio * (1 - self.suppression_strength)
        
        return result
    
    def _adaptive_full_audio_removal(self, audio: np.ndarray, sample_rate: int, 
                                   segments: list) -> np.ndarray:
        """
        Adaptive removal that adjusts based on detected segments.
        """
        # Analyze main speaker characteristics from non-secondary segments
        main_speaker_mask = np.ones(len(audio), dtype=bool)
        for start, end in segments:
            start_idx = int(start * sample_rate)
            end_idx = int(end * sample_rate)
            main_speaker_mask[start_idx:end_idx] = False
        
        # If we have enough main speaker audio, analyze it
        if np.sum(main_speaker_mask) > sample_rate * 0.5:  # At least 0.5s
            main_audio = audio[main_speaker_mask]
            main_freq_profile = self._analyze_frequency_profile(main_audio, sample_rate)
        else:
            # Fallback to basic method
            return self._basic_full_audio_removal(audio, sample_rate)
        
        # Apply adaptive filtering based on main speaker profile
        cleaned = self._apply_adaptive_filter(audio, sample_rate, main_freq_profile)
        
        return cleaned
    
    def _apply_spectral_consistency_mask(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply spectral consistency masking to remove variable frequency components.
        """
        # STFT parameters
        nperseg = min(512, len(audio) - 1)
        if nperseg < 64:
            return audio  # Too short for STFT
        
        # Perform STFT
        f, t, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg)
        
        # Calculate magnitude statistics
        magnitude = np.abs(Zxx)
        
        # Smooth magnitude over time
        for i in range(magnitude.shape[0]):
            magnitude[i, :] = gaussian_filter1d(magnitude[i, :], sigma=2)
        
        # Calculate consistency (low variance = consistent = likely main speaker)
        mean_mag = np.mean(magnitude, axis=1, keepdims=True) + 1e-10
        std_mag = np.std(magnitude, axis=1, keepdims=True)
        consistency = 1 - (std_mag / mean_mag)
        consistency = np.clip(consistency, 0, 1)
        
        # Create soft mask based on consistency
        mask = consistency ** 2  # Square to make it more selective
        
        # Apply mask
        Zxx_masked = Zxx * mask
        
        # Reconstruct
        _, cleaned = istft(Zxx_masked, fs=sample_rate, nperseg=nperseg)
        
        # Ensure same length
        if len(cleaned) > len(audio):
            cleaned = cleaned[:len(audio)]
        elif len(cleaned) < len(audio):
            cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))
        
        return cleaned
    
    def _analyze_frequency_profile(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Analyze frequency profile of audio (presumably main speaker).
        """
        # Get frequency spectrum
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Smooth spectrum
        magnitude = gaussian_filter1d(magnitude, sigma=10)
        
        # Normalize
        magnitude = magnitude / (np.max(magnitude) + 1e-10)
        
        return magnitude
    
    def _apply_adaptive_filter(self, audio: np.ndarray, sample_rate: int,
                              main_freq_profile: np.ndarray) -> np.ndarray:
        """
        Apply adaptive filtering based on main speaker frequency profile.
        """
        # Get audio spectrum
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        
        # Interpolate main speaker profile to match current audio length
        if len(main_freq_profile) != len(fft):
            x_old = np.linspace(0, 1, len(main_freq_profile))
            x_new = np.linspace(0, 1, len(fft))
            main_freq_profile = np.interp(x_new, x_old, main_freq_profile)
        
        # Create adaptive mask
        # Boost frequencies that match main speaker profile
        mask = main_freq_profile ** 0.5  # Soften the profile
        
        # Apply frequency-dependent suppression
        fft_filtered = fft * mask
        
        # Reconstruct
        cleaned = np.fft.irfft(fft_filtered, n=len(audio))
        
        return cleaned