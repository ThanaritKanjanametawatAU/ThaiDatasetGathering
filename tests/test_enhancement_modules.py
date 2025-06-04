"""Test individual enhancement stages."""

import unittest
import numpy as np
from scipy import signal

from processors.audio_enhancement.adaptive_spectral import AdaptiveSpectralSubtraction
from processors.audio_enhancement.wiener_filter import AdaptiveWienerFilter
from processors.audio_enhancement.harmonic_enhancer import HarmonicEnhancer
from processors.audio_enhancement.perceptual_post import PerceptualPostProcessor


class TestEnhancementModules(unittest.TestCase):
    """Test individual enhancement stages."""
    
    def setUp(self):
        self.sample_rate = 16000
    
    def test_spectral_subtraction_gentle(self):
        """Test 2.1: Spectral subtraction preserves speech quality."""
        # Create test signal with known characteristics
        signal_clean = self._create_speech_like_signal()
        noise = 0.1 * np.random.randn(len(signal_clean))
        noisy = signal_clean + noise
        
        # Apply spectral subtraction
        enhancer = AdaptiveSpectralSubtraction()
        enhanced = enhancer.process(noisy, self.sample_rate)
        
        # Verify noise reduction
        residual_noise = enhanced - signal_clean
        original_noise_power = np.mean(noise ** 2)
        residual_noise_power = np.mean(residual_noise ** 2)
        
        self.assertLess(residual_noise_power, original_noise_power * 0.5,
                       "Should reduce noise by at least 50%")
        
        # Verify speech preservation
        speech_distortion = np.mean((enhanced - signal_clean) ** 2) / np.mean(signal_clean ** 2)
        self.assertLess(speech_distortion, 0.1, "Speech distortion should be <10%")
    
    def test_wiener_filter_adaptation(self):
        """Test 2.2: Wiener filter adapts to changing noise."""
        # Create signal with varying noise levels
        signal_clean = self._create_speech_like_signal()
        
        # First half: low noise, second half: high noise
        noise = np.zeros_like(signal_clean)
        half = len(signal_clean) // 2
        noise[:half] = 0.05 * np.random.randn(half)
        noise[half:] = 0.2 * np.random.randn(len(signal_clean) - half)
        
        noisy = signal_clean + noise
        
        # Apply Wiener filter
        wiener = AdaptiveWienerFilter()
        enhanced = wiener.process(noisy, self.sample_rate)
        
        # Check adaptation: should preserve more in low noise region
        distortion_low = np.mean((enhanced[:half] - signal_clean[:half]) ** 2)
        distortion_high = np.mean((enhanced[half:] - signal_clean[half:]) ** 2)
        
        # Low noise region should have less distortion
        self.assertLess(distortion_low, distortion_high * 0.7)
    
    def test_harmonic_enhancement_pitch_tracking(self):
        """Test 2.3: Harmonic enhancer correctly tracks pitch."""
        # Create signal with varying pitch
        duration = 2.0
        sr = self.sample_rate
        t = np.linspace(0, duration, int(duration * sr))
        
        # Pitch glide from 100 Hz to 200 Hz
        pitch_curve = 100 + 50 * t / duration
        phase = 2 * np.pi * np.cumsum(pitch_curve) / sr
        signal_clean = np.sin(phase)
        
        # Add inter-harmonic noise
        noise = 0.1 * np.random.randn(len(signal_clean))
        noisy = signal_clean + noise
        
        # Apply harmonic enhancement
        enhancer = HarmonicEnhancer()
        enhanced = enhancer.enhance(noisy, sr)
        
        # Verify harmonic structure preserved
        # Check power at harmonic frequencies increased
        fft_enhanced = np.abs(np.fft.rfft(enhanced))
        fft_noisy = np.abs(np.fft.rfft(noisy))
        
        # Sample some harmonic frequencies
        test_freqs = [100, 150, 200]  # Hz
        for freq in test_freqs:
            bin_idx = int(freq * len(fft_enhanced) / (sr / 2))
            if bin_idx < len(fft_enhanced):
                harmonic_gain = fft_enhanced[bin_idx] / (fft_noisy[bin_idx] + 1e-10)
                self.assertGreater(harmonic_gain, 1.2, 
                                 f"Harmonic at {freq}Hz not enhanced")
    
    def test_perceptual_post_processing(self):
        """Test 2.4: Perceptual post-processing improves quality."""
        # Create processed signal with some artifacts
        signal_clean = self._create_speech_like_signal()
        
        # Add some processing artifacts
        # 1. Harsh high frequencies
        signal_harsh = signal_clean.copy()
        high_freq_noise = 0.05 * np.random.randn(len(signal_clean))
        # Apply high-pass filter to noise
        nyquist = self.sample_rate / 2
        high_cutoff = 8000 / nyquist
        b, a = signal.butter(5, high_cutoff, 'high')
        high_freq_noise = signal.filtfilt(b, a, high_freq_noise)
        signal_harsh += high_freq_noise
        
        # 2. Unnaturally silent gaps
        signal_harsh[1000:1100] = 0
        signal_harsh[5000:5100] = 0
        
        # Apply perceptual post-processing
        processor = PerceptualPostProcessor()
        processed = processor.process(signal_harsh, self.sample_rate)
        
        # Check comfort noise added to gaps
        gap_power = np.mean(processed[1000:1100] ** 2)
        self.assertGreater(gap_power, 1e-6, "Should add comfort noise to gaps")
        
        # Check high frequency harshness reduced
        fft_original = np.abs(np.fft.rfft(signal_harsh))
        fft_processed = np.abs(np.fft.rfft(processed))
        
        high_freq_idx = len(fft_original) * 3 // 4  # Upper quarter of spectrum
        hf_reduction = np.mean(fft_processed[high_freq_idx:]) / (np.mean(fft_original[high_freq_idx:]) + 1e-10)
        self.assertLess(hf_reduction, 0.8, "Should reduce harsh high frequencies")
    
    def _create_speech_like_signal(self, duration: float = 3.0) -> np.ndarray:
        """Create a speech-like signal with formants and modulation."""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Base frequency with slight variation
        f0 = 150 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
        
        # Generate harmonics
        signal = np.zeros_like(t)
        for harmonic in range(1, 10):
            signal += np.sin(2 * np.pi * f0 * harmonic * t) / harmonic
        
        # Apply amplitude envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        signal *= envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return signal
    
    def _high_pass(self, cutoff: float, sample_rate: int, length: int) -> np.ndarray:
        """Generate high-pass filter response."""
        # Simple high-pass filter implementation
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        # Create frequency response
        freqs = np.fft.rfftfreq(length, 1/sample_rate)
        response = np.zeros_like(freqs)
        response[freqs > cutoff] = 1.0
        
        # Smooth transition
        transition_width = 1000  # Hz
        transition_mask = (freqs > (cutoff - transition_width)) & (freqs < cutoff)
        response[transition_mask] = (freqs[transition_mask] - (cutoff - transition_width)) / transition_width
        
        return response


if __name__ == "__main__":
    unittest.main()