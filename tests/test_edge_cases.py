"""Test handling of edge cases and difficult scenarios."""

import unittest
import numpy as np
from scipy import signal

from processors.audio_enhancement.core import AudioEnhancer
from utils.snr_measurement import SNRMeasurement


class TestEdgeCases(unittest.TestCase):
    """Test handling of edge cases and difficult scenarios."""
    
    def setUp(self):
        self.sample_rate = 16000
        self.enhancer = AudioEnhancer(enable_35db_enhancement=True)
        self.snr_calc = SNRMeasurement()
    
    def test_clipped_audio_handling(self):
        """Test 5.1: Handle clipped/saturated audio."""
        # Create clipped signal
        signal_clean = self._create_realistic_speech() * 3.0  # Amplify to cause clipping
        clipped = np.clip(signal_clean, -1.0, 1.0)
        
        # Add noise
        noisy_clipped = clipped + 0.1 * np.random.randn(len(clipped))
        
        # Process
        enhanced, metadata = self.enhancer.enhance_to_target_snr(noisy_clipped, self.sample_rate)
        
        # Should handle without crashing
        self.assertIsNotNone(enhanced)
        self.assertIsNotNone(metadata)
        
        # Should not amplify clipped regions further
        max_val = np.max(np.abs(enhanced))
        self.assertLessEqual(max_val, 1.0, "Should not exceed clipping range")
    
    def test_very_low_snr_input(self):
        """Test 5.2: Handle extremely noisy input (SNR < 10dB)."""
        signal_clean = self._create_realistic_speech()
        
        # Create very noisy signal (5dB SNR)
        very_noisy = self._add_noise_at_snr(signal_clean, 5)
        
        # Process
        enhanced, metadata = self.enhancer.enhance_to_target_snr(very_noisy, self.sample_rate)
        
        # Should improve SNR significantly even if not reaching 35dB
        output_snr = metadata["snr_db"]
        input_snr = self.snr_calc.measure_snr(very_noisy, self.sample_rate)
        
        self.assertGreater(output_snr, input_snr + 10, "Should achieve significant improvement")
        
        # Should maintain some quality
        self.assertGreater(metadata["naturalness_score"], 0.7)
    
    def test_silence_and_speech_transitions(self):
        """Test 5.3: Handle audio with frequent silence-speech transitions."""
        # Create audio with alternating speech and silence
        duration = 4.0
        sr = self.sample_rate
        audio = np.zeros(int(duration * sr))
        
        # Add 0.5s speech segments with 0.5s gaps
        for i in range(4):
            start = int(i * sr)
            end = int((i + 0.5) * sr)
            t = np.linspace(0, 0.5, end - start)
            audio[start:end] = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 2)
        
        # Add noise
        noisy = audio + 0.05 * np.random.randn(len(audio))
        
        # Process
        enhanced, metadata = self.enhancer.enhance_to_target_snr(noisy, sr)
        
        # Check that transitions are preserved
        # Simple energy-based transition detection
        frame_size = int(0.01 * sr)  # 10ms frames
        original_energy = []
        enhanced_energy = []
        
        for i in range(0, len(audio) - frame_size, frame_size):
            original_energy.append(np.mean(audio[i:i+frame_size]**2))
            enhanced_energy.append(np.mean(enhanced[i:i+frame_size]**2))
        
        # Convert to binary (speech/silence)
        threshold = 0.01
        original_binary = [1 if e > threshold else 0 for e in original_energy]
        enhanced_binary = [1 if e > threshold else 0 for e in enhanced_energy]
        
        # Count transitions
        original_transitions = sum(abs(original_binary[i] - original_binary[i-1]) 
                                 for i in range(1, len(original_binary)))
        enhanced_transitions = sum(abs(enhanced_binary[i] - enhanced_binary[i-1]) 
                                 for i in range(1, len(enhanced_binary)))
        
        # Should preserve transition structure (within 20%)
        self.assertAlmostEqual(enhanced_transitions, original_transitions, 
                             delta=original_transitions * 0.2)
    
    def test_multi_speaker_audio(self):
        """Test 5.4: Handle audio with multiple speakers."""
        # Simulate two speakers
        speaker1 = self._create_realistic_speech(pitch=120)
        speaker2 = self._create_realistic_speech(pitch=200)
        
        # Mix speakers with overlap
        mixed = speaker1 * 0.7 + speaker2 * 0.3
        
        # Add noise
        noisy = mixed + 0.1 * np.random.randn(len(mixed))
        
        # Process
        enhanced, metadata = self.enhancer.enhance_to_target_snr(noisy, self.sample_rate)
        
        # Should preserve both speakers
        # Check spectral peaks at both fundamental frequencies
        fft_result = np.abs(np.fft.rfft(enhanced))
        freqs = np.fft.rfftfreq(len(enhanced), 1/self.sample_rate)
        
        # Find peaks near expected frequencies
        peak1_idx = np.argmin(np.abs(freqs - 120))
        peak2_idx = np.argmin(np.abs(freqs - 200))
        
        # Both speakers should be preserved
        median_magnitude = np.median(fft_result)
        self.assertGreater(fft_result[peak1_idx], median_magnitude * 3,
                         "First speaker not preserved")
        self.assertGreater(fft_result[peak2_idx], median_magnitude * 2,
                         "Second speaker not preserved")
    
    def test_varying_sample_rates(self):
        """Test 5.5: Handle different sample rates correctly."""
        sample_rates = [8000, 16000, 22050, 44100, 48000]
        
        for sr in sample_rates:
            # Create appropriate signal for sample rate
            duration = 2.0
            t = np.linspace(0, duration, int(duration * sr))
            signal_clean = np.sin(2 * np.pi * 200 * t)
            
            # Add noise
            noisy = signal_clean + 0.1 * np.random.randn(len(signal_clean))
            
            # Process
            enhanced, metadata = self.enhancer.enhance_to_target_snr(noisy, sr)
            
            # Should handle any sample rate
            self.assertIsNotNone(enhanced)
            self.assertEqual(len(enhanced), len(noisy))
            self.assertIn("snr_db", metadata)
            
            # Should improve SNR
            input_snr = self.snr_calc.measure_snr(noisy, sr)
            output_snr = metadata["snr_db"]
            self.assertGreater(output_snr, input_snr, f"No improvement at {sr}Hz")
    
    def test_zero_length_audio(self):
        """Test 5.6: Handle zero-length audio gracefully."""
        empty_audio = np.array([])
        
        # Should not crash
        enhanced, metadata = self.enhancer.enhance_to_target_snr(empty_audio, self.sample_rate)
        
        self.assertEqual(len(enhanced), 0)
        self.assertIsNotNone(metadata)
    
    def test_extreme_amplitude_ranges(self):
        """Test 5.7: Handle extreme amplitude ranges."""
        # Very quiet signal
        quiet_signal = self._create_realistic_speech() * 0.001
        quiet_noisy = quiet_signal + 0.0001 * np.random.randn(len(quiet_signal))
        
        enhanced_quiet, metadata_quiet = self.enhancer.enhance_to_target_snr(
            quiet_noisy, self.sample_rate
        )
        
        # Should handle without numerical issues
        self.assertFalse(np.any(np.isnan(enhanced_quiet)))
        self.assertFalse(np.any(np.isinf(enhanced_quiet)))
        
        # Very loud signal (near clipping)
        loud_signal = self._create_realistic_speech() * 0.99
        loud_noisy = loud_signal + 0.1 * np.random.randn(len(loud_signal))
        
        enhanced_loud, metadata_loud = self.enhancer.enhance_to_target_snr(
            loud_noisy, self.sample_rate
        )
        
        # Should handle without clipping
        self.assertLessEqual(np.max(np.abs(enhanced_loud)), 1.0)
    
    def test_single_tone_input(self):
        """Test 5.8: Handle pure tone input."""
        # Pure 1kHz tone
        duration = 2.0
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        pure_tone = np.sin(2 * np.pi * 1000 * t)
        
        # Add noise
        noisy_tone = pure_tone + 0.05 * np.random.randn(len(pure_tone))
        
        # Process
        enhanced, metadata = self.enhancer.enhance_to_target_snr(noisy_tone, self.sample_rate)
        
        # Should preserve the tone frequency
        fft_result = np.abs(np.fft.rfft(enhanced))
        freqs = np.fft.rfftfreq(len(enhanced), 1/self.sample_rate)
        
        # Find peak near 1kHz
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        
        self.assertAlmostEqual(peak_freq, 1000, delta=50, 
                             msg="Tone frequency not preserved")
    
    def _create_realistic_speech(self, duration: float = 3.0, pitch: float = 150) -> np.ndarray:
        """Create speech-like signal."""
        sr = self.sample_rate
        t = np.linspace(0, duration, int(duration * sr))
        
        # Fundamental with vibrato
        f0 = pitch * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
        
        # Harmonics
        signal_clean = np.zeros_like(t)
        for i in range(1, 10):
            signal_clean += np.sin(2 * np.pi * f0 * i * t) / i
        
        # Envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        signal_clean *= envelope
        
        # Normalize
        signal_clean = signal_clean / np.max(np.abs(signal_clean)) * 0.8
        
        return signal_clean
    
    def _add_noise_at_snr(self, signal_clean: np.ndarray, target_snr_db: float) -> np.ndarray:
        """Add noise to achieve specific SNR."""
        signal_power = np.mean(signal_clean ** 2)
        snr_linear = 10 ** (target_snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power) * np.random.randn(len(signal_clean))
        return signal_clean + noise


if __name__ == "__main__":
    unittest.main()