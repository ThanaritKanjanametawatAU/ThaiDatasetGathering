"""Comprehensive SNR measurement validation tests."""

import unittest
import numpy as np
from utils.snr_measurement import SNRMeasurement


class TestSNRMeasurement(unittest.TestCase):
    """Comprehensive SNR measurement validation."""
    
    def setUp(self):
        self.snr_calc = SNRMeasurement()
        self.sample_rate = 16000
        
    def test_known_snr_accuracy(self):
        """Test 1.1: Validate SNR calculation with synthetic signals of known SNR."""
        test_cases = [
            {"true_snr": 0, "tolerance": 0.5},   # Equal signal and noise
            {"true_snr": 10, "tolerance": 0.5},  # 10 dB SNR
            {"true_snr": 20, "tolerance": 0.5},  # 20 dB SNR
            {"true_snr": 35, "tolerance": 0.5},  # Target SNR
            {"true_snr": 40, "tolerance": 0.5},  # High SNR
        ]
        
        for case in test_cases:
            # Generate clean signal (sine wave)
            duration = 3.0
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            
            # Generate white noise at specific SNR
            signal_power = np.mean(signal ** 2)
            target_snr_linear = 10 ** (case["true_snr"] / 10)
            noise_power = signal_power / target_snr_linear
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
            
            # Mix signal and noise
            mixed = signal + noise
            
            # Measure SNR
            measured_snr = self.snr_calc.measure_snr(mixed, self.sample_rate)
            
            # Assert within tolerance
            self.assertAlmostEqual(
                measured_snr, case["true_snr"], 
                delta=case["tolerance"],
                msg=f"SNR measurement failed for {case['true_snr']} dB"
            )
    
    def test_vad_integration(self):
        """Test 1.2: Ensure VAD correctly identifies speech vs silence."""
        # Create signal with clear speech and silence segments
        duration = 4.0
        samples = int(duration * self.sample_rate)
        
        # First 1s: silence
        # Next 2s: speech
        # Last 1s: silence
        signal = np.zeros(samples)
        speech_start = int(1.0 * self.sample_rate)
        speech_end = int(3.0 * self.sample_rate)
        
        # Add speech in middle
        t_speech = np.linspace(0, 2.0, speech_end - speech_start)
        signal[speech_start:speech_end] = np.sin(2 * np.pi * 200 * t_speech)
        
        # Add noise throughout
        noise = 0.01 * np.random.randn(samples)
        mixed = signal + noise
        
        # Get VAD segments
        vad_segments = self.snr_calc._get_vad_segments(mixed, self.sample_rate)
        
        # Verify VAD detected speech correctly
        self.assertEqual(len(vad_segments), 1, "Should detect one speech segment")
        self.assertAlmostEqual(vad_segments[0][0], 1.0, delta=0.1)
        self.assertAlmostEqual(vad_segments[0][1], 3.0, delta=0.1)
    
    def test_edge_case_all_silence(self):
        """Test 1.3: Handle audio that is all silence."""
        # Create pure silence with minimal noise
        silence = 0.0001 * np.random.randn(self.sample_rate * 2)
        
        # Should return very low SNR or handle gracefully
        snr = self.snr_calc.measure_snr(silence, self.sample_rate)
        self.assertIsNotNone(snr)
        self.assertLess(snr, 0, "All silence should have negative SNR")
    
    def test_edge_case_all_speech(self):
        """Test 1.4: Handle audio with no silence segments."""
        # Create continuous speech
        t = np.linspace(0, 3.0, 3 * self.sample_rate)
        speech = np.sin(2 * np.pi * 200 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        
        # Should estimate noise floor from quietest parts
        snr = self.snr_calc.measure_snr(speech, self.sample_rate)
        self.assertIsNotNone(snr)
        self.assertGreater(snr, 20, "Clean speech should have high SNR")
    
    def test_short_audio_handling(self):
        """Test 1.5: Handle very short audio clips (<1 second)."""
        short_durations = [0.1, 0.3, 0.5, 0.8]
        
        for duration in short_durations:
            samples = int(duration * self.sample_rate)
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
            audio += 0.1 * np.random.randn(samples)
            
            # Should handle without crashing
            snr = self.snr_calc.measure_snr(audio, self.sample_rate)
            self.assertIsNotNone(snr, f"Failed for {duration}s audio")
    
    def test_different_noise_types(self):
        """Test 1.6: Accurate measurement with different noise types."""
        duration = 3.0
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        signal = np.sin(2 * np.pi * 440 * t)
        
        noise_types = {
            "white": np.random.randn(len(t)),
            "pink": self._generate_pink_noise(len(t)),
            "brown": self._generate_brown_noise(len(t)),
            "periodic": np.sin(2 * np.pi * 60 * t),  # 60 Hz hum
        }
        
        for noise_type, noise in noise_types.items():
            # Normalize noise for 20 dB SNR
            noise = noise / np.std(noise) * 0.1
            mixed = signal + noise
            
            snr = self.snr_calc.measure_snr(mixed, self.sample_rate)
            self.assertAlmostEqual(snr, 20, delta=2, 
                                 msg=f"Failed for {noise_type} noise")
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        # Simple pink noise approximation
        white = np.random.randn(length)
        
        # Apply basic 1/f filtering
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        
        # Apply 1/f envelope (skip DC)
        fft[1:] = fft[1:] / np.sqrt(freqs[1:])
        
        pink = np.fft.irfft(fft, length)
        return pink / np.std(pink)
    
    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown noise (1/f^2 spectrum)."""
        # Brown noise is integrated white noise
        white = np.random.randn(length)
        brown = np.cumsum(white)
        brown = brown - np.mean(brown)
        return brown / np.std(brown)


if __name__ == "__main__":
    unittest.main()