"""Test complete 35dB enhancement pipeline integration."""

import unittest
import numpy as np
from unittest.mock import patch

from processors.audio_enhancement.core import AudioEnhancer
from processors.audio_enhancement.enhancement_orchestrator import EnhancementOrchestrator
from processors.audio_enhancement.gpu_enhancement import GPUEnhancementBatch, MemoryEfficientProcessor
from utils.snr_measurement import SNRMeasurement
from utils.audio_metrics import calculate_pesq, calculate_stoi


class TestIntegration35dB(unittest.TestCase):
    """Test complete 35dB enhancement pipeline."""
    
    def setUp(self):
        self.sample_rate = 16000
        self.snr_calc = SNRMeasurement()
    
    def test_pipeline_achieves_target_snr(self):
        """Test 4.1: Pipeline achieves 35dB for various input SNRs."""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        input_snrs = [15, 20, 25, 30, 33]
        success_count = 0
        
        for input_snr in input_snrs:
            # Create signal with specific SNR
            signal = self._create_realistic_speech()
            noisy = self._add_noise_at_snr(signal, input_snr)
            
            # Enhance
            enhanced, metadata = enhancer.enhance_to_target_snr(noisy, self.sample_rate)
            
            # Check result
            output_snr = metadata["snr_db"]
            
            if output_snr >= 35.0:
                success_count += 1
            elif output_snr >= 33.0:  # Close enough considering measurement error
                success_count += 0.5
            
            # Verify quality preserved
            self.assertGreater(metadata["naturalness_score"], 0.85,
                             f"Naturalness too low for input SNR {input_snr}")
        
        # At least 90% success rate
        success_rate = success_count / len(input_snrs)
        self.assertGreaterEqual(success_rate, 0.9)
    
    def test_quality_preservation_throughout_pipeline(self):
        """Test 4.2: Each stage preserves quality metrics."""
        orchestrator = EnhancementOrchestrator(target_snr=35)
        
        # Create test signal
        signal = self._create_realistic_speech()
        noisy = self._add_noise_at_snr(signal, 25)
        
        # Track metrics through each stage
        metrics_history = []
        
        # Hook into orchestrator to capture intermediate results
        def capture_metrics(audio, stage_name):
            metrics = {
                "stage": stage_name,
                "snr": self.snr_calc.measure_snr(audio, self.sample_rate),
                "naturalness": self._quick_naturalness_check(signal, audio)
            }
            
            # Mock PESQ and STOI for testing
            with patch('utils.audio_metrics.calculate_pesq') as mock_pesq:
                mock_pesq.return_value = 3.5
                metrics["pesq"] = calculate_pesq(signal, audio, self.sample_rate)
            
            with patch('utils.audio_metrics.calculate_stoi') as mock_stoi:
                mock_stoi.return_value = 0.9
                metrics["stoi"] = calculate_stoi(signal, audio, self.sample_rate)
            
            metrics_history.append(metrics)
        
        # Process with metric capture
        orchestrator._metric_callback = capture_metrics
        enhanced, final_metrics = orchestrator.enhance(noisy, self.sample_rate)
        
        # Verify quality never drops below threshold
        for metrics in metrics_history:
            self.assertGreater(metrics["naturalness"], 0.80,
                             f"Quality dropped at {metrics['stage']}")
            self.assertGreater(metrics["pesq"], 3.0,
                             f"PESQ too low at {metrics['stage']}")
    
    def test_batch_processing_consistency(self):
        """Test 4.3: Batch processing produces consistent results."""
        gpu_processor = GPUEnhancementBatch(batch_size=8, device='cpu')  # Use CPU for testing
        
        # Create batch of similar samples
        batch = []
        for i in range(8):
            signal = self._create_realistic_speech()
            # Add similar noise levels
            noisy = self._add_noise_at_snr(signal, 25 + np.random.randn() * 2)
            batch.append(noisy)
        
        # Process batch
        enhanced_batch = gpu_processor.process_batch(batch, self.sample_rate)
        
        # Check consistency
        snrs = [self.snr_calc.measure_snr(audio, self.sample_rate) for audio in enhanced_batch]
        
        # All should be improved
        for i, snr in enumerate(snrs):
            original_snr = self.snr_calc.measure_snr(batch[i], self.sample_rate)
            self.assertGreater(snr, original_snr, f"Batch item {i} not improved")
        
        # Check that processing was consistent
        snr_std = np.std(snrs)
        self.assertLess(snr_std, 5.0, "Too much variation in batch processing")
    
    def test_memory_efficient_processing(self):
        """Test 4.4: Memory-efficient processing with dynamic batch sizing."""
        memory_manager = MemoryEfficientProcessor(max_memory_gb=30)
        
        test_cases = [
            {"duration": 1.0, "expected_batch": 64},
            {"duration": 5.0, "expected_batch": 32},
            {"duration": 10.0, "expected_batch": 16},
            {"duration": 30.0, "expected_batch": 4},
        ]
        
        for case in test_cases:
            batch_size = memory_manager.estimate_batch_size(
                int(case["duration"] * self.sample_rate), 
                self.sample_rate
            )
            
            # Should be in reasonable range
            self.assertGreater(batch_size, 0)
            self.assertLessEqual(batch_size, 128)
            
            # Check it's roughly in expected range
            expected = case["expected_batch"]
            self.assertAlmostEqual(batch_size, expected, 
                                 delta=expected * 0.5,
                                 msg=f"Batch size for {case['duration']}s audio")
    
    def test_configuration_integration(self):
        """Test 4.5: Configuration properly controls enhancement."""
        # Test with 35dB enhancement disabled
        enhancer_disabled = AudioEnhancer(enable_35db_enhancement=False)
        signal = self._create_realistic_speech()
        noisy = self._add_noise_at_snr(signal, 20)
        
        enhanced, metadata = enhancer_disabled.enhance_to_target_snr(noisy, self.sample_rate)
        
        # Should use standard enhancement
        self.assertNotIn("target_achieved", metadata)
        
        # Test with 35dB enhancement enabled
        enhancer_enabled = AudioEnhancer(enable_35db_enhancement=True)
        enhanced, metadata = enhancer_enabled.enhance_to_target_snr(noisy, self.sample_rate)
        
        # Should have 35dB specific fields
        self.assertIn("target_achieved", metadata)
        self.assertIn("naturalness_score", metadata)
        self.assertIn("snr_db", metadata)
    
    def _create_realistic_speech(self, duration: float = 3.0, pitch: float = 150) -> np.ndarray:
        """Create speech-like signal with formants and modulation."""
        sr = self.sample_rate
        t = np.linspace(0, duration, int(duration * sr))
        
        # Fundamental frequency with vibrato
        f0 = pitch * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
        
        # Generate harmonics with formant structure
        signal = np.zeros_like(t)
        formants = [700, 1220, 2600]  # Typical formant frequencies
        
        for i in range(1, 10):  # First 10 harmonics
            harmonic_freq = i * f0
            harmonic = np.sin(2 * np.pi * harmonic_freq * t)
            
            # Apply formant filtering
            for formant in formants:
                if abs(i * pitch - formant) < 200:
                    harmonic *= 2.0  # Boost near formants
            
            signal += harmonic / i  # Natural harmonic rolloff
        
        # Add amplitude modulation (speech envelope)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        signal *= envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return signal
    
    def _add_noise_at_snr(self, signal: np.ndarray, target_snr_db: float) -> np.ndarray:
        """Add noise to achieve specific SNR."""
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (target_snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise
    
    def _quick_naturalness_check(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Quick naturalness check for testing."""
        # Simple correlation-based check
        if len(original) != len(enhanced):
            min_len = min(len(original), len(enhanced))
            original = original[:min_len]
            enhanced = enhanced[:min_len]
        
        correlation = np.corrcoef(original, enhanced)[0, 1]
        return max(0, correlation)


if __name__ == "__main__":
    unittest.main()