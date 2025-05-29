"""
Spectral Gating Fallback Engine
CPU-based noise reduction using spectral gating.
"""

import numpy as np
import logging
from typing import Dict, Any
from scipy import signal

logger = logging.getLogger(__name__)


class SpectralGatingEngine:
    """
    Spectral gating implementation for CPU-based noise reduction.
    Uses noisereduce library as primary method with custom fallback.
    """
    
    def __init__(self):
        """Initialize spectral gating engine."""
        # Try to import noisereduce
        try:
            import noisereduce as nr
            self.nr = nr
            self.noisereduce_available = True
            self.available = True  # Add this for compatibility
            logger.info("noisereduce library available")
        except ImportError:
            logger.warning("noisereduce not installed. Using basic spectral gating.")
            self.noisereduce_available = False
            self.available = True  # Basic spectral gating is always available
            
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        spectral_gate_freq: float = 1500,
        preserve_ratio: float = 0.7,
        stationary_only: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Process audio with spectral gating.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            spectral_gate_freq: Gate frequency threshold
            preserve_ratio: Amount of original signal to preserve
            stationary_only: Only remove stationary noise
            **kwargs: Additional parameters
            
        Returns:
            Enhanced audio array
        """
        try:
            # Store original properties
            original_dtype = audio.dtype
            original_max = np.max(np.abs(audio))
            
            # Convert to float for processing
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Normalize
            if original_max > 0:
                audio = audio / original_max
                
            # Apply noise reduction
            if self.noisereduce_available:
                # Use noisereduce library
                enhanced = self._noisereduce_process(
                    audio, sample_rate, stationary_only
                )
            else:
                # Use custom spectral gating
                enhanced = self._custom_spectral_gate(
                    audio, sample_rate, spectral_gate_freq
                )
                
            # Apply preservation ratio
            if preserve_ratio > 0:
                enhanced = preserve_ratio * audio + (1 - preserve_ratio) * enhanced
                
            # Restore scale
            if original_max > 0:
                enhanced = enhanced * original_max
                
            # Restore dtype
            if original_dtype != np.float32:
                enhanced = enhanced.astype(original_dtype)
                
            return enhanced
            
        except Exception as e:
            logger.error(f"Spectral gating failed: {e}")
            return audio
            
    def _noisereduce_process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stationary: bool
    ) -> np.ndarray:
        """
        Process using noisereduce library.
        
        Args:
            audio: Normalized audio
            sample_rate: Sample rate
            stationary: Use stationary noise reduction
            
        Returns:
            Enhanced audio
        """
        try:
            # Estimate noise from first 0.5 seconds
            noise_sample_length = int(0.5 * sample_rate)
            if len(audio) > noise_sample_length:
                noise_sample = audio[:noise_sample_length]
            else:
                # Use quietest 10% as noise estimate
                sorted_audio = np.sort(np.abs(audio))
                noise_level = sorted_audio[int(len(sorted_audio) * 0.1)]
                noise_sample = audio[np.abs(audio) < noise_level]
                
            # Apply noise reduction
            enhanced = self.nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                y_noise=noise_sample,
                stationary=stationary,
                prop_decrease=1.0
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"noisereduce processing failed: {e}")
            # Fallback to custom method
            return self._custom_spectral_gate(audio, sample_rate, 1500)
            
    def _custom_spectral_gate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        gate_freq: float
    ) -> np.ndarray:
        """
        Custom spectral gating implementation.
        
        Args:
            audio: Normalized audio
            sample_rate: Sample rate
            gate_freq: Gate frequency threshold
            
        Returns:
            Enhanced audio
        """
        # Parameters
        frame_len = 2048
        hop_len = frame_len // 4
        
        # Compute STFT
        f, t, stft = signal.stft(
            audio,
            fs=sample_rate,
            window='hann',
            nperseg=frame_len,
            noverlap=frame_len - hop_len
        )
        
        # Magnitude and phase
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor
        # Use bottom 20% of magnitude values as noise estimate
        noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
        
        # Create gate mask
        # Frequencies below gate_freq get more aggressive gating
        freq_mask = np.ones_like(f)
        gate_idx = np.where(f < gate_freq)[0]
        if len(gate_idx) > 0:
            freq_mask[gate_idx] = 0.5  # More aggressive below gate frequency
            
        # Apply spectral gate
        gate_threshold = noise_floor * 2.0  # Gate at 2x noise floor
        gate_mask = magnitude > gate_threshold
        
        # Smooth the mask to avoid artifacts
        from scipy.ndimage import binary_dilation
        gate_mask = binary_dilation(gate_mask, iterations=1)
        
        # Apply frequency-dependent gating
        gated_magnitude = magnitude * gate_mask
        
        # Apply frequency mask
        gated_magnitude = gated_magnitude * freq_mask[:, np.newaxis]
        
        # Reconstruct
        gated_stft = gated_magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        _, enhanced = signal.istft(
            gated_stft,
            fs=sample_rate,
            window='hann',
            nperseg=frame_len,
            noverlap=frame_len - hop_len
        )
        
        # Ensure same length as input
        if len(enhanced) > len(audio):
            enhanced = enhanced[:len(audio)]
        elif len(enhanced) < len(audio):
            enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
            
        return enhanced