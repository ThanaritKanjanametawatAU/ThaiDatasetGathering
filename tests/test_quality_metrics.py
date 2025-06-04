"""Test quality assessment metrics."""

import unittest
import numpy as np
from scipy import signal

from processors.audio_enhancement.quality_monitor import QualityMonitor


class TestQualityMetrics(unittest.TestCase):
    """Test quality assessment metrics."""
    
    def setUp(self):
        self.monitor = QualityMonitor()
        self.sample_rate = 16000
    
    def test_naturalness_detection(self):
        """Test 3.1: Naturalness score detects over-processing."""
        signal_clean = self._create_speech_like_signal()
        
        # Test cases with varying degrees of processing
        test_cases = [
            {
                "name": "mild_enhancement",
                "process": lambda x: x * 0.95 + 0.05 * self._gentle_denoise(x),
                "expected_naturalness": 0.95,
                "tolerance": 0.05
            },
            {
                "name": "moderate_enhancement", 
                "process": lambda x: self._moderate_denoise(x),
                "expected_naturalness": 0.85,
                "tolerance": 0.05
            },
            {
                "name": "over_processed",
                "process": lambda x: self._aggressive_denoise(x),
                "expected_naturalness": 0.65,
                "tolerance": 0.1
            },
            {
                "name": "destroyed",
                "process": lambda x: np.sign(x) * 0.5,  # Extreme clipping
                "expected_naturalness": 0.2,
                "tolerance": 0.1
            }
        ]
        
        for case in test_cases:
            processed = case["process"](signal_clean)
            naturalness = self.monitor.check_naturalness(signal_clean, processed)
            
            self.assertAlmostEqual(
                naturalness, 
                case["expected_naturalness"],
                delta=case["tolerance"],
                msg=f"Failed for {case['name']}"
            )
    
    def test_spectral_distortion_measurement(self):
        """Test 3.2: Spectral distortion metric accuracy."""
        signal_clean = self._create_speech_like_signal()
        
        # Add known spectral distortions
        distortions = [
            {"type": "lowpass", "cutoff": 4000, "expected_distortion": 0.15},
            {"type": "highpass", "cutoff": 300, "expected_distortion": 0.10},
            {"type": "notch", "freq": 1000, "expected_distortion": 0.08},
            {"type": "amplify_band", "band": [2000, 3000], "expected_distortion": 0.12},
        ]
        
        for dist in distortions:
            distorted = self._apply_spectral_distortion(signal_clean, dist)
            measured = self.monitor._measure_spectral_distortion(signal_clean, distorted)
            
            self.assertAlmostEqual(
                measured,
                dist["expected_distortion"],
                delta=0.03,
                msg=f"Failed for {dist['type']}"
            )
    
    def test_phase_coherence_detection(self):
        """Test 3.3: Phase coherence detects phase artifacts."""
        signal_clean = self._create_speech_like_signal()
        
        # Test phase manipulations
        test_cases = [
            {
                "name": "original",
                "process": lambda x: x,
                "expected_coherence": 1.0
            },
            {
                "name": "minor_phase_shift",
                "process": lambda x: self._add_phase_shift(x, 0.1),
                "expected_coherence": 0.95
            },
            {
                "name": "major_phase_jumps",
                "process": lambda x: self._add_phase_jumps(x, 10),
                "expected_coherence": 0.7
            },
            {
                "name": "random_phase",
                "process": lambda x: self._randomize_phase(x),
                "expected_coherence": 0.3
            }
        ]
        
        for case in test_cases:
            processed = case["process"](signal_clean)
            coherence = self.monitor._measure_phase_coherence(signal_clean, processed)
            
            self.assertAlmostEqual(
                coherence,
                case["expected_coherence"],
                delta=0.1,
                msg=f"Failed for {case['name']}"
            )
    
    def _create_speech_like_signal(self, duration: float = 3.0) -> np.ndarray:
        """Create a speech-like signal."""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Base frequency with variation
        f0 = 150 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
        
        # Generate harmonics
        signal_clean = np.zeros_like(t)
        for h in range(1, 10):
            signal_clean += np.sin(2 * np.pi * f0 * h * t) / h
        
        # Apply envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        signal_clean *= envelope
        
        # Normalize
        signal_clean = signal_clean / np.max(np.abs(signal_clean)) * 0.8
        
        return signal_clean
    
    def _gentle_denoise(self, x: np.ndarray) -> np.ndarray:
        """Apply gentle denoising."""
        # Simple low-pass filter
        b, a = signal.butter(2, 0.9)
        return signal.filtfilt(b, a, x)
    
    def _moderate_denoise(self, x: np.ndarray) -> np.ndarray:
        """Apply moderate denoising."""
        # Stronger low-pass filter
        b, a = signal.butter(4, 0.7)
        filtered = signal.filtfilt(b, a, x)
        # Mix with original
        return 0.7 * filtered + 0.3 * x
    
    def _aggressive_denoise(self, x: np.ndarray) -> np.ndarray:
        """Apply aggressive denoising."""
        # Very strong low-pass filter
        b, a = signal.butter(6, 0.5)
        filtered = signal.filtfilt(b, a, x)
        # Heavy processing
        return 0.9 * filtered + 0.1 * x
    
    def _apply_spectral_distortion(self, signal_clean: np.ndarray, dist: dict) -> np.ndarray:
        """Apply specific spectral distortion."""
        nyquist = self.sample_rate / 2
        
        if dist["type"] == "lowpass":
            b, a = signal.butter(4, dist["cutoff"] / nyquist, 'low')
            return signal.filtfilt(b, a, signal_clean)
        
        elif dist["type"] == "highpass":
            b, a = signal.butter(4, dist["cutoff"] / nyquist, 'high')
            return signal.filtfilt(b, a, signal_clean)
        
        elif dist["type"] == "notch":
            # Notch filter
            Q = 30.0  # Quality factor
            freq = dist["freq"] / nyquist
            b, a = signal.iirnotch(freq, Q)
            return signal.filtfilt(b, a, signal_clean)
        
        elif dist["type"] == "amplify_band":
            # Amplify specific frequency band
            low_freq, high_freq = dist["band"]
            sos = signal.butter(4, [low_freq / nyquist, high_freq / nyquist], 'band', output='sos')
            band_signal = signal.sosfiltfilt(sos, signal_clean)
            
            # Mix amplified band with original
            return signal_clean + 0.5 * band_signal
        
        return signal_clean
    
    def _add_phase_shift(self, x: np.ndarray, shift_amount: float) -> np.ndarray:
        """Add constant phase shift."""
        fft = np.fft.rfft(x)
        phases = np.angle(fft)
        magnitudes = np.abs(fft)
        
        # Add phase shift
        new_phases = phases + shift_amount
        
        # Reconstruct
        new_fft = magnitudes * np.exp(1j * new_phases)
        return np.fft.irfft(new_fft, len(x))
    
    def _add_phase_jumps(self, x: np.ndarray, num_jumps: int) -> np.ndarray:
        """Add random phase jumps."""
        segment_length = len(x) // (num_jumps + 1)
        result = x.copy()
        
        for i in range(num_jumps):
            start = (i + 1) * segment_length
            if start < len(x):
                # Add random phase shift to segment
                shift = np.random.uniform(-np.pi, np.pi)
                result[start:] = self._add_phase_shift(result[start:], shift)
        
        return result
    
    def _randomize_phase(self, x: np.ndarray) -> np.ndarray:
        """Randomize phase completely."""
        fft = np.fft.rfft(x)
        magnitudes = np.abs(fft)
        
        # Random phases
        random_phases = np.random.uniform(-np.pi, np.pi, len(fft))
        
        # Reconstruct with random phases
        new_fft = magnitudes * np.exp(1j * random_phases)
        return np.fft.irfft(new_fft, len(x))


if __name__ == "__main__":
    unittest.main()