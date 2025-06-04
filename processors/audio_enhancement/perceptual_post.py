"""Perceptual post-processing for natural sound quality."""

import numpy as np
from scipy import signal
import torch


class PerceptualPostProcessor:
    """Post-processing to improve perceptual quality and naturalness."""
    
    def __init__(self):
        self.comfort_noise_level = -50  # dB
        self.pre_emphasis_alpha = 0.97
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply perceptual post-processing.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Perceptually enhanced audio
        """
        if len(audio) == 0:
            return audio
        
        # Step 1: Pre-emphasis (boost high frequencies)
        audio = self._apply_pre_emphasis(audio)
        
        # Step 2: Dynamic range adjustment
        audio = self._adjust_dynamics(audio)
        
        # Step 3: Comfort noise injection
        audio = self._add_comfort_noise(audio)
        
        # Step 4: Smoothing filter
        audio = self._apply_smoothing_filter(audio, sample_rate)
        
        # Step 5: De-emphasis (compensate for pre-emphasis)
        audio = self._apply_de_emphasis(audio)
        
        # Final normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        return audio
    
    def _apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies."""
        # Simple first-order high-pass filter
        emphasized = np.zeros_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - self.pre_emphasis_alpha * audio[:-1]
        return emphasized
    
    def _apply_de_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply de-emphasis filter to compensate for pre-emphasis."""
        # Inverse of pre-emphasis
        de_emphasized = np.zeros_like(audio)
        de_emphasized[0] = audio[0]
        for i in range(1, len(audio)):
            de_emphasized[i] = audio[i] + self.pre_emphasis_alpha * de_emphasized[i-1]
        return de_emphasized
    
    def _adjust_dynamics(self, audio: np.ndarray) -> np.ndarray:
        """Apply gentle dynamic range compression."""
        # Simple soft-knee compressor
        threshold = 0.5
        ratio = 3.0
        knee_width = 0.1
        
        # Convert to dB scale
        eps = 1e-10
        audio_db = 20 * np.log10(np.abs(audio) + eps)
        threshold_db = 20 * np.log10(threshold)
        
        # Soft knee compression
        compressed_db = np.zeros_like(audio_db)
        
        for i, level in enumerate(audio_db):
            if level < threshold_db - knee_width:
                # Below knee - no compression
                compressed_db[i] = level
            elif level > threshold_db + knee_width:
                # Above knee - full compression
                compressed_db[i] = threshold_db + (level - threshold_db) / ratio
            else:
                # In knee region - smooth transition
                knee_factor = (level - (threshold_db - knee_width)) / (2 * knee_width)
                compressed_db[i] = level + (1 - knee_factor) * 0 + knee_factor * ((threshold_db + knee_width - level) * (1 - 1/ratio))
        
        # Convert back to linear scale
        compressed = np.sign(audio) * (10 ** (compressed_db / 20))
        
        # Smooth transitions
        attack_time = 0.005  # 5ms
        release_time = 0.05  # 50ms
        compressed = self._smooth_envelope(compressed, audio, attack_time, release_time, 16000)
        
        return compressed
    
    def _add_comfort_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add minimal comfort noise to prevent dead air."""
        # Detect silence regions
        frame_length = 1024
        hop_length = 512
        
        # Calculate frame energies
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        if not energies:
            return audio
        
        # Threshold for silence detection
        silence_threshold = np.percentile(energies, 10)
        
        # Generate pink noise
        pink_noise = self._generate_pink_noise(len(audio))
        
        # Scale comfort noise
        noise_amplitude = 10 ** (self.comfort_noise_level / 20)
        comfort_noise = pink_noise * noise_amplitude
        
        # Apply comfort noise only to silence regions
        output = audio.copy()
        for i in range(0, len(audio) - frame_length, hop_length):
            frame_energy = np.sqrt(np.mean(audio[i:i + frame_length] ** 2))
            
            if frame_energy < silence_threshold:
                # Silence region - add comfort noise with smooth transition
                fade_length = hop_length // 2
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)
                
                # Apply fades at boundaries
                if i > 0:
                    output[i:i + fade_length] += comfort_noise[i:i + fade_length] * fade_in
                
                output[i + fade_length:i + frame_length - fade_length] += comfort_noise[i + fade_length:i + frame_length - fade_length]
                
                if i + frame_length < len(audio):
                    output[i + frame_length - fade_length:i + frame_length] += comfort_noise[i + frame_length - fade_length:i + frame_length] * fade_out
        
        return output
    
    def _apply_smoothing_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply smoothing filter to remove remaining artifacts."""
        # Design a gentle low-pass filter
        cutoff_freq = 7000  # Hz
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Butterworth filter for smooth response
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        
        # Apply filter
        smoothed = signal.filtfilt(b, a, audio)
        
        # Blend with original to preserve some high-frequency detail
        blend_factor = 0.9
        output = blend_factor * smoothed + (1 - blend_factor) * audio
        
        return output
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        # Generate white noise
        white = np.random.randn(length)
        
        # Apply 1/f filter
        # Simple approximation using multiple first-order filters
        b1 = [0.049922, -0.095993, 0.050612]
        a1 = [1.0, -2.494956, 2.017265]
        
        b2 = [0.012093, -0.019287, 0.007194]
        a2 = [1.0, -0.994949, 0.994949]
        
        # Apply filters in cascade
        pink = signal.lfilter(b1, a1, white)
        pink = signal.lfilter(b2, a2, pink)
        
        # Normalize
        pink = pink / np.std(pink)
        
        return pink
    
    def _smooth_envelope(self, compressed: np.ndarray, original: np.ndarray, 
                        attack_time: float, release_time: float, sample_rate: int) -> np.ndarray:
        """Smooth dynamic changes with attack and release times."""
        # Calculate envelope of gain changes
        gain = np.abs(compressed) / (np.abs(original) + 1e-10)
        
        # Smooth the gain envelope
        attack_samples = int(attack_time * sample_rate)
        release_samples = int(release_time * sample_rate)
        
        smoothed_gain = np.zeros_like(gain)
        smoothed_gain[0] = gain[0]
        
        for i in range(1, len(gain)):
            if gain[i] > smoothed_gain[i-1]:
                # Attack - fast response
                alpha = 1.0 - np.exp(-1.0 / attack_samples)
                smoothed_gain[i] = alpha * gain[i] + (1 - alpha) * smoothed_gain[i-1]
            else:
                # Release - slow response
                alpha = 1.0 - np.exp(-1.0 / release_samples)
                smoothed_gain[i] = alpha * gain[i] + (1 - alpha) * smoothed_gain[i-1]
        
        # Apply smoothed gain
        return original * smoothed_gain