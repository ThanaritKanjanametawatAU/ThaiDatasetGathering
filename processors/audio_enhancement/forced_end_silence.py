"""
Forced End Silence Module

A last-resort approach to ensure secondary speakers at the end are removed.
If all other methods fail, this module will silence or heavily attenuate
the end portion where secondary speakers are detected.
"""

import numpy as np
import logging
from typing import Tuple
import librosa
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class ForcedEndSilence:
    """
    Force silence or heavy attenuation at the end of audio to remove secondary speakers.
    This is a last-resort method when other approaches fail.
    """
    
    def __init__(self,
                 analysis_duration: float = 4.0,  # Increased from 3.0
                 silence_threshold: float = 0.05,  # Lower threshold for more aggressive detection
                 fade_duration: float = 0.5):
        """
        Initialize forced end silence.
        
        Args:
            analysis_duration: Duration to analyze at end (seconds)
            silence_threshold: Threshold for forcing silence
            fade_duration: Duration of fade out (seconds)
        """
        self.analysis_duration = analysis_duration
        self.silence_threshold = silence_threshold
        self.fade_duration = fade_duration
        self.sample_rate = 16000
        
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Process audio to force silence at end if secondary speaker detected.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            Processed audio
        """
        if len(audio) < sample_rate * 2:
            return audio
        
        # Analyze end for secondary speaker
        has_secondary, confidence, start_idx = self._analyze_end(audio, sample_rate)
        
        if not has_secondary:
            logger.info("No forced silence needed - no secondary speaker detected")
            return audio
        
        logger.info(f"Forcing end silence - secondary speaker detected with {confidence:.2f} confidence at {start_idx/sample_rate:.2f}s")
        
        # Apply forced silence/attenuation
        processed = self._force_silence(audio, start_idx, sample_rate)
        
        return processed
    
    def _analyze_end(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, float, int]:
        """
        Analyze end of audio for secondary speaker presence.
        
        Returns:
            (has_secondary, confidence, start_index)
        """
        analysis_samples = int(self.analysis_duration * sample_rate)
        analysis_samples = min(analysis_samples, len(audio) // 2)
        
        # Get segments
        primary_segment = audio[:len(audio) - analysis_samples]
        end_segment = audio[-analysis_samples:]
        
        # Compare characteristics
        primary_char = self._get_characteristics(primary_segment)
        end_char = self._get_characteristics(end_segment)
        
        # Calculate differences
        pitch_diff = abs(end_char['pitch'] - primary_char['pitch']) / (primary_char['pitch'] + 1e-10)
        energy_diff = abs(end_char['energy'] - primary_char['energy']) / (primary_char['energy'] + 1e-10)
        spectral_diff = abs(end_char['spectral_centroid'] - primary_char['spectral_centroid']) / (primary_char['spectral_centroid'] + 1e-10)
        
        # Combined confidence
        confidence = (pitch_diff + energy_diff + spectral_diff) / 3
        
        # Determine if secondary speaker present
        has_secondary = confidence > self.silence_threshold
        
        # Find start point
        if has_secondary:
            start_idx = self._find_transition_point(audio, sample_rate)
        else:
            start_idx = len(audio)
        
        return has_secondary, confidence, start_idx
    
    def _get_characteristics(self, segment: np.ndarray) -> dict:
        """Get audio characteristics."""
        char = {}
        
        # Energy
        char['energy'] = np.sqrt(np.mean(segment ** 2))
        
        # Pitch
        try:
            f0, voiced, _ = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            valid_f0 = f0[voiced]
            char['pitch'] = np.mean(valid_f0) if len(valid_f0) > 0 else 0
        except:
            char['pitch'] = 0
        
        # Spectral centroid
        try:
            char['spectral_centroid'] = np.mean(
                librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate)
            )
        except:
            char['spectral_centroid'] = 0
        
        return char
    
    def _find_transition_point(self, audio: np.ndarray, sample_rate: int) -> int:
        """Find where secondary speaker likely starts."""
        # More aggressive approach: check last 4 seconds
        analysis_start = max(0, len(audio) - int(4 * sample_rate))
        
        # Energy profile
        frame_size = int(0.05 * sample_rate)
        hop_size = int(0.025 * sample_rate)
        
        energies = []
        for i in range(analysis_start, len(audio) - frame_size, hop_size):
            frame_energy = np.sqrt(np.mean(audio[i:i+frame_size] ** 2))
            energies.append(frame_energy)
        
        if len(energies) < 5:
            return analysis_start
        
        # Smooth and find changes
        energies = gaussian_filter1d(energies, sigma=2)
        energy_diff = np.diff(energies)
        
        # Find significant changes with lower threshold for more sensitivity
        threshold = np.std(energy_diff) * 0.5  # Reduced from 1.5 for more sensitivity
        changes = np.where(np.abs(energy_diff) > threshold)[0]
        
        if len(changes) > 0:
            # Use FIRST significant change to be more conservative
            # This ensures we silence from the earliest possible secondary speaker
            first_change_idx = changes[0]
            return analysis_start + first_change_idx * hop_size
        
        # Default: Be very aggressive - 3 seconds from end
        return max(0, len(audio) - int(3 * sample_rate))
    
    def _force_silence(self, audio: np.ndarray, start_idx: int, sample_rate: int) -> np.ndarray:
        """
        Force silence or heavy attenuation from start_idx to end.
        """
        processed = audio.copy()
        
        # Option 1: Complete silence (most aggressive) - ACTIVATED
        # Since 5% attenuation was not sufficient (correlation still 0.534),
        # we need complete silence to ensure no secondary speaker is audible
        processed[start_idx:] = 0
        
        # Option 2: Heavy attenuation with fade - DISABLED
        # This option was not aggressive enough to remove secondary speakers
        # fade_samples = int(self.fade_duration * sample_rate)
        # fade_samples = min(fade_samples, len(audio) - start_idx)
        # 
        # if fade_samples > 0:
        #     # Create fade curve
        #     fade = np.linspace(1.0, 0.05, fade_samples)  # Fade to 5% of original
        #     
        #     # Apply fade
        #     processed[start_idx:start_idx + fade_samples] *= fade
        #     
        #     # Near silence for the rest
        #     if start_idx + fade_samples < len(processed):
        #         processed[start_idx + fade_samples:] *= 0.05
        # else:
        #     # Just attenuate heavily
        #     processed[start_idx:] *= 0.05
        
        logger.info(f"Applied COMPLETE SILENCE from {start_idx/sample_rate:.2f}s to end")
        
        return processed