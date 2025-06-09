"""
Intelligent End Silencer for Secondary Speaker Removal
Specifically targets secondary speakers that appear at the end of audio
"""
import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class IntelligentEndSilencer:
    """Intelligently detects and silences secondary speakers at the end of audio"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Detection parameters
        self.min_silence_before_end = 0.2  # Minimum silence before end speaker (200ms)
        self.max_end_duration = 2.0  # Maximum duration to consider as "end" (2s)
        self.energy_change_threshold = 2.0  # Energy ratio threshold
        self.spectral_change_threshold = 0.3  # Spectral centroid change threshold
        
        # Silence parameters
        self.fade_duration = 0.02  # 20ms fade out
        self.safety_margin = 0.05  # 50ms safety margin
        
    def detect_end_secondary_speaker(self, audio: np.ndarray, sample_rate: int) -> Optional[int]:
        """
        Detect if there's a secondary speaker at the end of audio.
        
        Returns:
            Start sample index of secondary speaker, or None if not detected
        """
        duration = len(audio) / sample_rate
        
        # Only check if audio is long enough
        if duration < 1.0:
            return None
        
        # Analyze the last portion of audio
        check_duration = min(self.max_end_duration, duration * 0.4)  # Check last 40% or 2s
        check_samples = int(check_duration * sample_rate)
        check_start = len(audio) - check_samples
        
        # 1. Energy-based detection
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = int(0.05 * sample_rate)    # 50ms hop
        
        energy_profile = []
        for i in range(check_start, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            energy = np.sqrt(np.mean(window**2))
            energy_profile.append((i, energy))
        
        # Find significant energy changes
        for i in range(len(energy_profile) - 5):  # Need at least 5 windows after
            curr_energy = energy_profile[i][1]
            
            # Check if there's a quiet period followed by energy
            if curr_energy < 0.01:  # Quiet
                # Check next few windows
                future_energies = [e[1] for _, e in energy_profile[i+1:i+6]]
                max_future = max(future_energies) if future_energies else 0
                
                if max_future > 0.02:  # Significant energy appears
                    # Found potential secondary speaker
                    potential_start = energy_profile[i+1][0]
                    
                    # 2. Spectral analysis to confirm it's a different speaker
                    if self._confirm_different_speaker(audio, potential_start, sample_rate):
                        logger.info(f"End secondary speaker detected at {potential_start/sample_rate:.2f}s")
                        return potential_start
        
        # 3. Abrupt energy increase detection
        for i in range(1, len(energy_profile)):
            prev_energy = energy_profile[i-1][1]
            curr_energy = energy_profile[i][1]
            
            if prev_energy > 0 and curr_energy / (prev_energy + 1e-10) > self.energy_change_threshold:
                potential_start = energy_profile[i][0]
                
                # Verify it's at the end and different speaker
                if (len(audio) - potential_start) / sample_rate < 1.5:  # Within last 1.5s
                    if self._confirm_different_speaker(audio, potential_start, sample_rate):
                        logger.info(f"Abrupt end speaker detected at {potential_start/sample_rate:.2f}s")
                        return potential_start
        
        return None
    
    def _confirm_different_speaker(self, audio: np.ndarray, start_idx: int, sample_rate: int) -> bool:
        """Confirm that audio after start_idx is a different speaker"""
        if start_idx < int(0.5 * sample_rate):  # Need at least 0.5s before
            return False
        
        # Compare spectral characteristics before and after
        before_start = max(0, start_idx - int(0.5 * sample_rate))
        before_audio = audio[before_start:start_idx]
        after_audio = audio[start_idx:]
        
        # Skip if segments too short
        if len(before_audio) < int(0.1 * sample_rate) or len(after_audio) < int(0.1 * sample_rate):
            return False
        
        # Compute spectral centroids
        before_centroid = self._compute_spectral_centroid(before_audio, sample_rate)
        after_centroid = self._compute_spectral_centroid(after_audio, sample_rate)
        
        # Significant change in spectral centroid indicates different speaker
        if before_centroid > 0 and after_centroid > 0:
            centroid_ratio = abs(after_centroid - before_centroid) / before_centroid
            if centroid_ratio > self.spectral_change_threshold:
                return True
        
        # Also check pitch variation
        before_pitch_var = self._estimate_pitch_variation(before_audio, sample_rate)
        after_pitch_var = self._estimate_pitch_variation(after_audio, sample_rate)
        
        # Different pitch characteristics
        if abs(before_pitch_var - after_pitch_var) > 50:  # 50 Hz difference
            return True
        
        return False
    
    def _compute_spectral_centroid(self, audio: np.ndarray, sample_rate: int) -> float:
        """Compute spectral centroid of audio segment"""
        # Use multiple windows and average
        window_size = min(1024, len(audio) // 4)
        if window_size < 256:
            return 0
        
        centroids = []
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i+window_size] * signal.windows.hann(window_size)
            
            # Compute FFT
            fft = np.fft.rfft(window)
            magnitude = np.abs(fft)
            
            # Compute frequencies
            freqs = np.fft.rfftfreq(window_size, 1/sample_rate)
            
            # Compute centroid
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                centroids.append(centroid)
        
        return np.median(centroids) if centroids else 0
    
    def _estimate_pitch_variation(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate pitch variation using autocorrelation"""
        # Simple pitch estimation
        if len(audio) < int(0.05 * sample_rate):  # Too short
            return 0
        
        # Use autocorrelation on small windows
        window_size = int(0.03 * sample_rate)  # 30ms
        pitches = []
        
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i+window_size]
            
            # Autocorrelation
            corr = np.correlate(window, window, mode='full')[len(window)-1:]
            
            # Find first peak after initial decline
            d = np.diff(corr)
            start = np.where(d > 0)[0]
            if len(start) > 1:
                peak_idx = start[0] + np.argmax(corr[start[0]:])
                if peak_idx > 0:
                    pitch = sample_rate / peak_idx
                    if 80 < pitch < 400:  # Reasonable speech pitch range
                        pitches.append(pitch)
        
        if pitches:
            return np.std(pitches)  # Return variation
        return 0
    
    def silence_end_speaker(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
        """
        Detect and silence secondary speaker at end of audio.
        
        Returns:
            Processed audio and metrics
        """
        # Detect end speaker
        end_speaker_start = self.detect_end_secondary_speaker(audio, sample_rate)
        
        metrics = {
            'end_speaker_detected': end_speaker_start is not None,
            'end_speaker_start_time': end_speaker_start / sample_rate if end_speaker_start else None,
            'original_end_energy': self._calculate_end_energy(audio, sample_rate)
        }
        
        if end_speaker_start is None:
            # No end speaker detected
            metrics['processed_end_energy'] = metrics['original_end_energy']
            return audio.copy(), metrics
        
        # Apply silencing
        result = audio.copy()
        
        # Apply safety margin
        silence_start = max(0, end_speaker_start - int(self.safety_margin * sample_rate))
        
        # Create fade out
        fade_samples = int(self.fade_duration * sample_rate)
        if fade_samples > 0 and silence_start > fade_samples:
            fade = np.linspace(1, 0, fade_samples)
            result[silence_start-fade_samples:silence_start] *= fade
        
        # Complete silence after fade
        result[silence_start:] = 0.0
        
        metrics['processed_end_energy'] = self._calculate_end_energy(result, sample_rate)
        metrics['silence_start_time'] = silence_start / sample_rate
        
        logger.info(f"Silenced end speaker from {silence_start/sample_rate:.2f}s to end")
        
        return result, metrics
    
    def _calculate_end_energy(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate energy in last 0.5s of audio"""
        last_samples = int(0.5 * sample_rate)
        if len(audio) > last_samples:
            segment = audio[-last_samples:]
            if np.any(segment != 0):
                return 20 * np.log10(np.sqrt(np.mean(segment**2)) + 1e-10)
        return -100.0
    
    def process_with_verification(self, audio: np.ndarray, sample_rate: int,
                                  force_silence_threshold: float = -45) -> Tuple[np.ndarray, Dict]:
        """
        Process audio and verify end is properly silenced.
        If not below threshold after processing, force complete silence.
        """
        # First attempt
        result, metrics = self.silence_end_speaker(audio, sample_rate)
        
        # Verify
        if metrics['processed_end_energy'] > force_silence_threshold:
            logger.warning(f"End energy still {metrics['processed_end_energy']:.1f}dB, forcing silence")
            
            # Force silence in last 1s if energy too high
            force_start = len(result) - sample_rate
            if force_start > 0:
                # Fade out
                fade_samples = int(0.05 * sample_rate)  # 50ms fade
                if force_start > fade_samples:
                    fade = np.linspace(1, 0, fade_samples)
                    result[force_start-fade_samples:force_start] *= fade
                
                # Complete silence
                result[force_start:] = 0.0
                
                # Recalculate metrics
                metrics['forced_silence'] = True
                metrics['forced_silence_start'] = force_start / sample_rate
                metrics['processed_end_energy'] = self._calculate_end_energy(result, sample_rate)
        else:
            metrics['forced_silence'] = False
        
        return result, metrics