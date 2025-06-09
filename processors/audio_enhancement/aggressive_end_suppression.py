"""
Aggressive End Suppression Module

A more direct approach to completely remove secondary speakers from the end of audio files.
Uses multiple techniques including energy-based gating, spectral subtraction, and 
voice activity detection to ensure complete removal.
"""

import numpy as np
import logging
from typing import Tuple, Optional
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class AggressiveEndSuppression:
    """
    Aggressively suppress any secondary speakers at the end of audio.
    """
    
    def __init__(self,
                 end_duration: float = 3.0,
                 primary_speaker_duration: float = 5.0,
                 suppression_threshold: float = 0.3,
                 gate_threshold: float = 0.1):
        """
        Initialize aggressive end suppression.
        
        Args:
            end_duration: Duration in seconds to analyze/process at end
            primary_speaker_duration: Duration to analyze for primary speaker characteristics
            suppression_threshold: Threshold for detecting changes (lower = more aggressive)
            gate_threshold: Energy gate threshold
        """
        self.end_duration = end_duration
        self.primary_speaker_duration = primary_speaker_duration
        self.suppression_threshold = suppression_threshold
        self.gate_threshold = gate_threshold
        self.sample_rate = 16000
        
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Aggressively process audio to remove secondary speakers from the end.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            Processed audio with end secondary speakers removed
        """
        if len(audio) < sample_rate * 2:  # Less than 2 seconds
            return audio
        
        # Analyze for secondary speaker at end
        has_secondary, start_idx = self._detect_end_secondary(audio, sample_rate)
        
        if not has_secondary:
            logger.info("No secondary speaker detected at end")
            return audio
        
        logger.info(f"Secondary speaker detected at end, starting at {start_idx/sample_rate:.2f}s")
        
        # Apply aggressive suppression
        processed = self._suppress_end_secondary(audio, start_idx, sample_rate)
        
        return processed
    
    def _detect_end_secondary(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, int]:
        """
        Detect if there's a secondary speaker at the end.
        
        Returns:
            (has_secondary, start_index)
        """
        end_samples = int(self.end_duration * sample_rate)
        end_samples = min(end_samples, len(audio) // 2)  # Don't analyze more than half
        
        # Get primary speaker characteristics from earlier in the audio
        primary_samples = int(self.primary_speaker_duration * sample_rate)
        primary_samples = min(primary_samples, len(audio) - end_samples)
        
        if primary_samples < sample_rate:  # Too short
            return False, 0
        
        # Extract features
        primary_segment = audio[:primary_samples]
        end_segment = audio[-end_samples:]
        
        # 1. Energy-based detection
        energy_change = self._detect_energy_change(primary_segment, end_segment, sample_rate)
        
        # 2. Spectral change detection
        spectral_change = self._detect_spectral_change(primary_segment, end_segment, sample_rate)
        
        # 3. Pitch change detection
        pitch_change = self._detect_pitch_change(primary_segment, end_segment, sample_rate)
        
        # 4. Zero-crossing rate change
        zcr_change = self._detect_zcr_change(primary_segment, end_segment)
        
        # Combine evidence
        total_change = (energy_change * 0.3 + spectral_change * 0.3 + 
                       pitch_change * 0.3 + zcr_change * 0.1)
        
        has_secondary = total_change > self.suppression_threshold
        
        if has_secondary:
            # Find exact start point
            start_idx = self._find_change_point(audio, sample_rate)
            return True, start_idx
        
        return False, 0
    
    def _suppress_end_secondary(self, audio: np.ndarray, start_idx: int, sample_rate: int) -> np.ndarray:
        """
        Aggressively suppress secondary speaker from start_idx to end.
        """
        processed = audio.copy()
        
        # Method 1: Energy gating - remove low energy portions
        processed = self._apply_energy_gate(processed, start_idx, sample_rate)
        
        # Method 2: Spectral subtraction based on primary speaker
        processed = self._apply_spectral_subtraction(processed, start_idx, sample_rate)
        
        # Method 3: Aggressive low-pass filtering
        processed = self._apply_aggressive_filtering(processed, start_idx, sample_rate)
        
        # Method 4: Fade out if still problematic
        processed = self._apply_smart_fadeout(processed, start_idx, sample_rate)
        
        return processed
    
    def _apply_energy_gate(self, audio: np.ndarray, start_idx: int, sample_rate: int) -> np.ndarray:
        """
        Apply energy-based gating to remove secondary speaker.
        """
        # Calculate energy profile for the end segment
        end_segment = audio[start_idx:]
        
        # Frame-based energy calculation
        frame_length = int(0.02 * sample_rate)  # 20ms frames
        hop_length = frame_length // 2
        
        # Calculate primary speaker energy threshold
        primary_segment = audio[:start_idx]
        primary_energy = np.sqrt(np.mean(primary_segment ** 2))
        
        # Gate threshold based on primary speaker energy
        gate_threshold = primary_energy * self.gate_threshold
        
        # Apply gating to end segment
        gated_segment = end_segment.copy()
        
        for i in range(0, len(end_segment) - frame_length, hop_length):
            frame = end_segment[i:i + frame_length]
            frame_energy = np.sqrt(np.mean(frame ** 2))
            
            if frame_energy < gate_threshold:
                # Suppress this frame completely
                gated_segment[i:i + frame_length] = 0
        
        # Reconstruct
        audio_gated = audio.copy()
        audio_gated[start_idx:] = gated_segment
        
        return audio_gated
    
    def _apply_spectral_subtraction(self, audio: np.ndarray, start_idx: int, sample_rate: int) -> np.ndarray:
        """
        Apply spectral subtraction to remove secondary speaker frequencies.
        """
        # Get primary speaker spectral profile
        primary_segment = audio[:start_idx]
        end_segment = audio[start_idx:]
        
        # Compute spectrograms
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Primary speaker spectrum (average)
        primary_stft = librosa.stft(primary_segment, n_fft=n_fft, hop_length=hop_length)
        primary_mag_avg = np.mean(np.abs(primary_stft), axis=1, keepdims=True)
        
        # End segment spectrum
        end_stft = librosa.stft(end_segment, n_fft=n_fft, hop_length=hop_length)
        end_mag = np.abs(end_stft)
        end_phase = np.angle(end_stft)
        
        # Find frequencies that are prominent in end but not in primary
        mag_ratio = end_mag / (primary_mag_avg + 1e-10)
        
        # Suppress frequencies that are unusually strong (likely secondary speaker)
        suppression_mask = mag_ratio > 3.0  # 3x stronger than primary
        end_mag[suppression_mask] = 0  # Complete suppression
        
        # Also suppress high frequencies (often secondary speakers)
        freq_bins = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2+1]
        high_freq_mask = freq_bins > 3500  # Above 3.5kHz
        end_mag[high_freq_mask, :] = 0  # Complete suppression
        
        # Reconstruct
        end_stft_suppressed = end_mag * np.exp(1j * end_phase)
        end_suppressed = librosa.istft(end_stft_suppressed, hop_length=hop_length)
        
        # Combine
        audio_suppressed = audio.copy()
        # Ensure same length
        if len(end_suppressed) > len(end_segment):
            end_suppressed = end_suppressed[:len(end_segment)]
        elif len(end_suppressed) < len(end_segment):
            end_suppressed = np.pad(end_suppressed, (0, len(end_segment) - len(end_suppressed)))
        
        audio_suppressed[start_idx:] = end_suppressed
        
        return audio_suppressed
    
    def _apply_aggressive_filtering(self, audio: np.ndarray, start_idx: int, sample_rate: int) -> np.ndarray:
        """
        Apply aggressive frequency filtering to preserve only primary speaker range.
        """
        # Primary speaker frequency range (typically 100-3000 Hz for speech)
        nyquist = sample_rate / 2
        low_freq = 100 / nyquist
        high_freq = 3000 / nyquist
        
        # Design aggressive bandpass filter
        sos = signal.butter(6, [low_freq, high_freq], btype='band', output='sos')
        
        # Apply to end segment
        end_segment = audio[start_idx:]
        filtered_end = signal.sosfilt(sos, end_segment)
        
        # Mix with original to preserve some naturalness
        mixed_end = 0.7 * filtered_end + 0.3 * end_segment
        
        # Reconstruct
        audio_filtered = audio.copy()
        audio_filtered[start_idx:] = mixed_end
        
        return audio_filtered
    
    def _apply_smart_fadeout(self, audio: np.ndarray, start_idx: int, sample_rate: int) -> np.ndarray:
        """
        Apply smart fadeout if secondary speaker is still present.
        """
        # Check if we need fadeout by analyzing the very end
        last_500ms = int(0.5 * sample_rate)
        if len(audio) - start_idx > last_500ms:
            end_energy = np.sqrt(np.mean(audio[-last_500ms:] ** 2))
            primary_energy = np.sqrt(np.mean(audio[:start_idx] ** 2))
            
            # If end is still too loud, apply fadeout
            if end_energy > primary_energy * 0.5:
                fade_duration = min(len(audio) - start_idx, int(2.0 * sample_rate))
                fade_start = len(audio) - fade_duration
                
                # Create smooth fade to complete silence
                fade = np.linspace(1.0, 0.0, fade_duration)
                
                audio_faded = audio.copy()
                audio_faded[fade_start:] *= fade
                
                return audio_faded
        
        return audio
    
    def _detect_energy_change(self, primary: np.ndarray, end: np.ndarray, sample_rate: int) -> float:
        """Detect energy profile changes."""
        # RMS energy
        primary_rms = np.sqrt(np.mean(primary ** 2))
        end_rms = np.sqrt(np.mean(end ** 2))
        
        # Energy variability
        frame_length = int(0.02 * sample_rate)
        primary_energies = [np.sqrt(np.mean(primary[i:i+frame_length] ** 2)) 
                           for i in range(0, len(primary) - frame_length, frame_length)]
        end_energies = [np.sqrt(np.mean(end[i:i+frame_length] ** 2)) 
                       for i in range(0, len(end) - frame_length, frame_length)]
        
        primary_std = np.std(primary_energies)
        end_std = np.std(end_energies)
        
        # Combined energy change metric
        rms_change = abs(end_rms - primary_rms) / (primary_rms + 1e-10)
        std_change = abs(end_std - primary_std) / (primary_std + 1e-10)
        
        return (rms_change + std_change) / 2
    
    def _detect_spectral_change(self, primary: np.ndarray, end: np.ndarray, sample_rate: int) -> float:
        """Detect spectral characteristic changes."""
        # Spectral centroids
        primary_centroid = np.mean(librosa.feature.spectral_centroid(y=primary, sr=sample_rate))
        end_centroid = np.mean(librosa.feature.spectral_centroid(y=end, sr=sample_rate))
        
        # Spectral rolloff
        primary_rolloff = np.mean(librosa.feature.spectral_rolloff(y=primary, sr=sample_rate))
        end_rolloff = np.mean(librosa.feature.spectral_rolloff(y=end, sr=sample_rate))
        
        # Combined spectral change
        centroid_change = abs(end_centroid - primary_centroid) / (primary_centroid + 1e-10)
        rolloff_change = abs(end_rolloff - primary_rolloff) / (primary_rolloff + 1e-10)
        
        return (centroid_change + rolloff_change) / 2
    
    def _detect_pitch_change(self, primary: np.ndarray, end: np.ndarray, sample_rate: int) -> float:
        """Detect pitch changes."""
        try:
            # Extract pitch
            primary_f0, primary_voiced, _ = librosa.pyin(
                primary, fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'), sr=sample_rate
            )
            end_f0, end_voiced, _ = librosa.pyin(
                end, fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'), sr=sample_rate
            )
            
            # Get voiced portions
            primary_f0_voiced = primary_f0[primary_voiced]
            end_f0_voiced = end_f0[end_voiced]
            
            if len(primary_f0_voiced) > 0 and len(end_f0_voiced) > 0:
                primary_mean = np.nanmean(primary_f0_voiced)
                end_mean = np.nanmean(end_f0_voiced)
                
                return abs(end_mean - primary_mean) / (primary_mean + 1e-10)
            
        except:
            pass
        
        return 0.0
    
    def _detect_zcr_change(self, primary: np.ndarray, end: np.ndarray) -> float:
        """Detect zero-crossing rate changes."""
        primary_zcr = np.mean(librosa.feature.zero_crossing_rate(primary))
        end_zcr = np.mean(librosa.feature.zero_crossing_rate(end))
        
        return abs(end_zcr - primary_zcr) / (primary_zcr + 1e-10)
    
    def _find_change_point(self, audio: np.ndarray, sample_rate: int) -> int:
        """Find the exact point where secondary speaker starts."""
        # Analyze last 3 seconds
        analysis_duration = int(3.0 * sample_rate)
        analysis_start = max(0, len(audio) - analysis_duration)
        
        # Energy profile
        frame_length = int(0.05 * sample_rate)  # 50ms frames
        hop_length = int(0.025 * sample_rate)  # 25ms hop
        
        energies = []
        for i in range(analysis_start, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energies.append(np.sqrt(np.mean(frame ** 2)))
        
        if len(energies) < 10:
            return analysis_start
        
        # Smooth and find changes
        energies_smooth = gaussian_filter1d(energies, sigma=3)
        
        # Find significant changes
        energy_diff = np.diff(energies_smooth)
        threshold = np.std(energy_diff) * 1.5
        
        change_points = np.where(np.abs(energy_diff) > threshold)[0]
        
        if len(change_points) > 0:
            # Return the last significant change
            last_change = change_points[-1]
            return analysis_start + last_change * hop_length
        
        # Default to 2 seconds from end
        return max(0, len(audio) - int(2.0 * sample_rate))