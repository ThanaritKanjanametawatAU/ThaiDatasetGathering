"""
Simple but effective secondary speaker removal using energy-based detection.
This module provides a straightforward approach to removing secondary speakers
by detecting and suppressing segments with different energy patterns.
"""

import numpy as np
from typing import Tuple, List
import logging
from scipy import signal
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class SimpleSecondaryRemoval:
    """
    Simple secondary speaker removal using energy-based detection and suppression.
    """
    
    def __init__(self, 
                 energy_threshold: float = 0.7,
                 min_silence_duration: float = 0.1,
                 suppression_db: float = -40,
                 use_spectral_masking: bool = True):
        """
        Initialize simple secondary speaker remover.
        
        Args:
            energy_threshold: Ratio threshold for detecting different speakers (0.0-1.0)
            min_silence_duration: Minimum silence duration between speakers (seconds)
            suppression_db: Suppression level in dB (negative value)
            use_spectral_masking: Whether to apply aggressive spectral masking
        """
        self.energy_threshold = energy_threshold
        self.min_silence_duration = min_silence_duration
        self.suppression_db = suppression_db
        self.use_spectral_masking = use_spectral_masking
        
    def remove_secondary_speakers(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Remove secondary speakers from audio using energy-based detection.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Audio with secondary speakers removed
        """
        # Calculate energy envelope
        frame_size = int(0.02 * sample_rate)  # 20ms frames
        hop_size = int(0.01 * sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            frame_energy = np.sqrt(np.mean(frame**2))
            energy.append(frame_energy)
            
        energy = np.array(energy)
        
        # Smooth energy curve
        smoothed_energy = gaussian_filter1d(energy, sigma=5)
        
        # Find main speaker energy level (use median as it's more robust)
        energy_median = np.median(smoothed_energy[smoothed_energy > 0])
        energy_std = np.std(smoothed_energy[smoothed_energy > 0])
        
        # Detect segments with significantly different energy
        # (potential secondary speakers or overlapping speech)
        secondary_mask = np.zeros(len(energy), dtype=bool)
        
        # Method 1: Detect high energy bursts (overlapping speech)
        high_threshold = energy_median + 2 * energy_std
        secondary_mask |= smoothed_energy > high_threshold
        
        # Method 2: Detect segments after silence (potential speaker change)
        silence_threshold = energy_median * 0.1
        is_silence = smoothed_energy < silence_threshold
        
        # Find silence segments
        silence_starts = []
        silence_ends = []
        in_silence = False
        
        for i in range(len(is_silence)):
            if is_silence[i] and not in_silence:
                silence_starts.append(i)
                in_silence = True
            elif not is_silence[i] and in_silence:
                silence_ends.append(i)
                in_silence = False
                
        if in_silence:
            silence_ends.append(len(is_silence))
            
        # Check segments after silence
        min_silence_frames = int(self.min_silence_duration * (1.0 / (hop_size / sample_rate)))
        
        for start, end in zip(silence_starts, silence_ends):
            silence_duration = end - start
            
            if silence_duration >= min_silence_frames and end < len(smoothed_energy) - 10:
                # Check energy pattern after silence
                post_silence_energy = smoothed_energy[end:min(end+20, len(smoothed_energy))]
                
                if len(post_silence_energy) > 0:
                    # If energy pattern is significantly different, mark as potential secondary speaker
                    energy_ratio = np.mean(post_silence_energy) / (energy_median + 1e-8)
                    
                    if energy_ratio > self.energy_threshold * 1.5 or energy_ratio < self.energy_threshold * 0.5:
                        # Mark segment as secondary speaker
                        segment_end = min(end + 50, len(secondary_mask))  # ~500ms segment
                        secondary_mask[end:segment_end] = True
        
        # Method 3: Detect sudden energy changes
        energy_diff = np.abs(np.diff(smoothed_energy))
        energy_diff_threshold = np.median(energy_diff) + 3 * np.std(energy_diff)
        
        for i in range(1, len(energy_diff)):
            if energy_diff[i] > energy_diff_threshold:
                # Potential speaker change
                # Mark surrounding area
                start_idx = max(0, i - 5)
                end_idx = min(len(secondary_mask), i + 20)
                secondary_mask[start_idx:end_idx] = True
        
        # Expand mask to cover full segments
        secondary_mask = self._expand_mask(secondary_mask, expansion=10)
        
        # Apply suppression
        result = audio.copy()
        suppression_factor = 10 ** (self.suppression_db / 20)  # Convert dB to linear
        
        # Convert frame-based mask to sample-based
        for i in range(len(secondary_mask)):
            if secondary_mask[i]:
                start_sample = i * hop_size
                end_sample = min((i + 1) * hop_size, len(audio))
                
                # Apply graduated suppression
                segment = result[start_sample:end_sample]
                
                # Apply fade to avoid clicks
                fade_len = min(len(segment) // 4, int(0.005 * sample_rate))  # 5ms fade
                if fade_len > 0 and len(segment) > 2 * fade_len:
                    fade_in = np.linspace(1, suppression_factor, fade_len)
                    fade_out = np.linspace(suppression_factor, 1, fade_len)
                    
                    segment[:fade_len] *= fade_in
                    segment[-fade_len:] *= fade_out
                    segment[fade_len:-fade_len] *= suppression_factor
                else:
                    segment *= suppression_factor
                    
                result[start_sample:end_sample] = segment
        
        # Final pass: Apply noise gate to very quiet segments
        gate_threshold = np.max(np.abs(result)) * 0.01
        result[np.abs(result) < gate_threshold] *= 0.1
        
        # Apply aggressive spectral masking if enabled
        if self.use_spectral_masking:
            result = self._apply_aggressive_spectral_masking(result, sample_rate, secondary_mask, hop_size)
        
        return result
    
    def _expand_mask(self, mask: np.ndarray, expansion: int = 5) -> np.ndarray:
        """
        Expand boolean mask to cover surrounding areas.
        
        Args:
            mask: Boolean mask
            expansion: Number of frames to expand
            
        Returns:
            Expanded mask
        """
        expanded = mask.copy()
        
        for i in range(len(mask)):
            if mask[i]:
                start = max(0, i - expansion)
                end = min(len(mask), i + expansion + 1)
                expanded[start:end] = True
                
        return expanded
    
    def _apply_aggressive_spectral_masking(self, audio: np.ndarray, sample_rate: int, 
                                          secondary_mask: np.ndarray, hop_size: int) -> np.ndarray:
        """
        Apply aggressive spectral masking to remove secondary speakers.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            secondary_mask: Boolean mask indicating secondary speaker regions
            hop_size: Hop size used in frame processing
            
        Returns:
            Audio with spectral masking applied
        """
        # STFT parameters
        n_fft = 2048
        hop_length = 256
        
        # Pad audio if necessary
        audio_padded = audio.copy()
        if len(audio_padded) % hop_length != 0:
            pad_length = hop_length - (len(audio_padded) % hop_length)
            audio_padded = np.pad(audio_padded, (0, pad_length), mode='constant')
        
        # Compute STFT using scipy
        from scipy.signal import stft as scipy_stft
        f, t, stft = scipy_stft(audio_padded, fs=sample_rate, nperseg=n_fft, 
                                 noverlap=n_fft-hop_length, boundary=None, padded=False)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Create frequency mask based on typical speech characteristics
        freq_bins = magnitude.shape[0]  # Frequency bins are in first dimension
        freq_mask = np.ones(freq_bins)
        
        # Aggressive masking of non-speech frequencies
        # Human speech fundamental frequency: 85-255 Hz
        # Important formants: up to 4000 Hz
        nyquist = sample_rate / 2
        
        # Calculate frequency bins
        freq_bin_hz = nyquist / freq_bins
        
        # Mask very low frequencies (below 80 Hz)
        low_cutoff = int(80 / freq_bin_hz)
        freq_mask[:low_cutoff] = 0.1
        
        # Mask high frequencies (above 4000 Hz) more aggressively in secondary regions
        high_cutoff = int(4000 / freq_bin_hz)
        freq_mask[high_cutoff:] = 0.3
        
        # Apply temporal-spectral masking based on secondary speaker detection
        for i in range(len(secondary_mask)):
            if secondary_mask[i]:
                # Time range in samples
                start_sample = i * hop_size
                end_sample = min((i + 1) * hop_size, len(audio))
                
                # Convert to STFT frame indices
                start_frame = start_sample // hop_length
                end_frame = end_sample // hop_length
                
                if start_frame < magnitude.shape[1] and end_frame <= magnitude.shape[1]:
                    # Apply aggressive frequency-selective masking
                    # Focus on removing harmonics and formants of secondary speaker
                    for frame in range(start_frame, min(end_frame, magnitude.shape[1])):
                        # Find peaks in spectrum (likely formants)
                        spectrum = magnitude[:, frame]
                        
                        # Dynamic threshold based on spectrum statistics
                        threshold = np.mean(spectrum) + 0.5 * np.std(spectrum)
                        
                        # Mask frequencies above threshold (likely secondary speaker)
                        mask = spectrum > threshold
                        
                        # Apply frequency mask with aggressive suppression
                        magnitude[:, frame] *= (1 - mask * 0.95) * freq_mask
        
        # Reconstruct signal
        stft_masked = magnitude * np.exp(1j * phase)
        
        # Use scipy istft for reconstruction
        from scipy.signal import istft as scipy_istft
        _, audio_masked = scipy_istft(stft_masked, fs=sample_rate, nperseg=n_fft, 
                                      noverlap=n_fft-hop_length, boundary=False)
        
        # Ensure same length as input
        if len(audio_masked) > len(audio):
            audio_masked = audio_masked[:len(audio)]
        elif len(audio_masked) < len(audio):
            audio_masked = np.pad(audio_masked, (0, len(audio) - len(audio_masked)))
        
        return audio_masked.astype(audio.dtype)
    
    def detect_speaker_changes(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float]]:
        """
        Detect potential speaker change points in audio.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            List of (start_time, end_time) tuples for detected segments
        """
        segments = []
        
        # Calculate energy
        frame_size = int(0.02 * sample_rate)
        hop_size = int(0.01 * sample_rate)
        
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            frame_energy = np.sqrt(np.mean(frame**2))
            energy.append(frame_energy)
            
        energy = np.array(energy)
        smoothed_energy = gaussian_filter1d(energy, sigma=5)
        
        # Find significant changes
        energy_diff = np.abs(np.diff(smoothed_energy))
        threshold = np.mean(energy_diff) + 2 * np.std(energy_diff)
        
        change_points = np.where(energy_diff > threshold)[0]
        
        # Group nearby change points into segments
        if len(change_points) > 0:
            current_start = change_points[0]
            
            for i in range(1, len(change_points)):
                if change_points[i] - change_points[i-1] > 20:  # 200ms gap
                    # End current segment
                    start_time = current_start * hop_size / sample_rate
                    end_time = change_points[i-1] * hop_size / sample_rate
                    segments.append((start_time, end_time))
                    current_start = change_points[i]
                    
            # Add last segment
            start_time = current_start * hop_size / sample_rate
            end_time = change_points[-1] * hop_size / sample_rate
            segments.append((start_time, end_time))
            
        return segments