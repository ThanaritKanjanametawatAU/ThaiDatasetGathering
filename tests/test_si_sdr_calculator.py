"""
Test suite for SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) calculator.

Tests the SI-SDR metric implementation for source separation quality evaluation
in the audio enhancement pipeline.
"""

import unittest
import pytest
import numpy as np
import tempfile
import os
import time
import shutil
from pathlib import Path
import soundfile as sf
from unittest.mock import patch, MagicMock
from typing import List, Tuple

# Import SI-SDR calculator (to be implemented)
from processors.audio_enhancement.metrics.si_sdr_calculator import (
    SISDRCalculator, SISDRError, SISDRResult, PermutationInvariantSDR,
    GPUSISDRCalculator
)


class TestSISDRCalculator(unittest.TestCase):
    """Test suite for SI-SDR calculator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_signals = cls._create_test_signals()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir)
        
    @classmethod
    def _create_test_signals(cls) -> dict:
        """Create test signals for validation."""
        signals = {}
        
        # Basic test signals
        fs = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(fs * duration))
        
        # Clean sinusoidal signal
        signals['clean'] = np.sin(2 * np.pi * 440 * t)
        
        # Scaled version (should have same SI-SDR)
        signals['scaled'] = 0.5 * signals['clean']
        
        # Noisy version
        signals['noisy'] = signals['clean'] + 0.1 * np.random.randn(len(signals['clean']))
        
        # Distorted version
        signals['distorted'] = signals['clean'] + 0.3 * np.sin(2 * np.pi * 880 * t)
        
        # Orthogonal signal (worst case)
        signals['orthogonal'] = np.cos(2 * np.pi * 440 * t)
        
        # Multi-source signals
        signals['source1'] = np.sin(2 * np.pi * 440 * t)
        signals['source2'] = np.sin(2 * np.pi * 880 * t)
        signals['mixture'] = signals['source1'] + signals['source2']
        
        # Separated sources (with some interference)
        signals['sep1'] = signals['source1'] + 0.1 * signals['source2']
        signals['sep2'] = signals['source2'] + 0.1 * signals['source1']
        
        return signals
        
    def setUp(self):
        """Set up for each test."""
        self.calculator = SISDRCalculator()
        
    def test_basic_si_sdr_calculation(self):
        """Test basic SI-SDR calculation."""
        # Perfect reconstruction should give high SI-SDR
        clean = self.test_signals['clean']
        si_sdr = self.calculator.calculate(clean, clean)
        self.assertGreater(si_sdr, 50)  # Should be very high (approaching infinity)
        
        # Noisy signal should have positive but lower SI-SDR
        noisy = self.test_signals['noisy']
        si_sdr_noisy = self.calculator.calculate(clean, noisy)
        self.assertGreater(si_sdr_noisy, 10)
        self.assertLess(si_sdr_noisy, 30)
        
        # Orthogonal signals should have very negative SI-SDR
        orthogonal = self.test_signals['orthogonal']
        si_sdr_orth = self.calculator.calculate(clean, orthogonal)
        self.assertLess(si_sdr_orth, -10)
        
    def test_scale_invariance_property(self):
        """Test that SI-SDR is scale-invariant."""
        clean = self.test_signals['clean']
        
        # Original SI-SDR
        noisy = self.test_signals['noisy']
        original_si_sdr = self.calculator.calculate(clean, noisy)
        
        # Test different scales
        scales = [0.1, 0.5, 2.0, 10.0, 100.0]
        for scale in scales:
            scaled_noisy = scale * noisy
            scaled_si_sdr = self.calculator.calculate(clean, scaled_noisy)
            
            # Should be identical within numerical precision
            self.assertAlmostEqual(scaled_si_sdr, original_si_sdr, places=6,
                                 msg=f"Scale invariance failed for scale={scale}")
                                 
    def test_si_sdr_improvement(self):
        """Test SI-SDR improvement calculation."""
        clean = self.test_signals['clean']
        mixture = self.test_signals['distorted']
        enhanced = self.test_signals['noisy']  # Less distorted than mixture
        
        # Calculate improvement
        improvement, enhanced_sdr, baseline_sdr = self.calculator.calculate_improvement(
            mixture, enhanced, clean
        )
        
        # Verify improvement calculation
        self.assertAlmostEqual(improvement, enhanced_sdr - baseline_sdr, places=6)
        
        # Enhanced should be better than mixture
        self.assertGreater(improvement, 0)
        
    def test_multi_source_separation(self):
        """Test SI-SDR for multi-source scenarios."""
        # Reference sources
        sources = [self.test_signals['source1'], self.test_signals['source2']]
        
        # Separated sources (with interference)
        separated = [self.test_signals['sep1'], self.test_signals['sep2']]
        
        # Calculate SI-SDR for each pair
        si_sdr1 = self.calculator.calculate(sources[0], separated[0])
        si_sdr2 = self.calculator.calculate(sources[1], separated[1])
        
        # Both should be reasonable (some interference)
        self.assertGreater(si_sdr1, 10)
        self.assertGreater(si_sdr2, 10)
        self.assertLess(si_sdr1, 30)
        self.assertLess(si_sdr2, 30)
        
    def test_permutation_invariant_sdr(self):
        """Test permutation-invariant SI-SDR calculation."""
        # Create permutation scenario
        sources = [self.test_signals['source1'], self.test_signals['source2']]
        
        # Swapped order
        separated_swapped = [self.test_signals['sep2'], self.test_signals['sep1']]
        
        # Calculate PIT-SDR
        pit_calculator = PermutationInvariantSDR()
        result = pit_calculator.calculate(sources, separated_swapped)
        
        # Should find correct permutation
        self.assertEqual(result['best_permutation'], (1, 0))  # Swapped
        
        # Mean SI-SDR should be reasonable
        self.assertGreater(result['mean_si_sdr'], 10)
        
        # Individual SI-SDRs should match correct pairing
        self.assertGreater(result['individual_si_sdrs'][0], 10)
        self.assertGreater(result['individual_si_sdrs'][1], 10)
        
    def test_segmental_analysis(self):
        """Test segmental SI-SDR analysis."""
        # Long signal for segmental analysis
        fs = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration))
        
        # Time-varying signal quality
        clean = np.sin(2 * np.pi * 440 * t)
        
        # Add varying noise levels
        noise = np.random.randn(len(clean))
        noise_envelope = np.linspace(0.1, 0.5, len(clean))  # Increasing noise
        noisy = clean + noise * noise_envelope
        
        # Calculate segmental SI-SDR
        result = self.calculator.calculate_segmental(
            clean, noisy, segment_size=1.0, hop_size=0.5
        )
        
        # Should have multiple segments
        self.assertGreater(len(result.segment_scores), 5)
        
        # SI-SDR should decrease over time (increasing noise)
        scores = result.segment_scores
        self.assertGreater(scores[0], scores[-1])
        
        # Overall score should be reasonable
        self.assertGreater(result.overall_score, 5)
        self.assertLess(result.overall_score, 25)
        
    def test_batch_processing(self):
        """Test batch processing capability."""
        # Create batch of signal pairs
        batch_size = 10
        signal_length = 16000
        
        references = []
        estimates = []
        
        for i in range(batch_size):
            ref = np.sin(2 * np.pi * (440 + i * 10) * np.linspace(0, 1, signal_length))
            est = ref + 0.1 * np.random.randn(signal_length)
            references.append(ref)
            estimates.append(est)
            
        # Batch calculation
        scores = self.calculator.batch_calculate(references, estimates)
        
        # Should return correct number of scores
        self.assertEqual(len(scores), batch_size)
        
        # All scores should be reasonable
        for score in scores:
            self.assertGreater(score, 10)
            self.assertLess(score, 30)
            
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Zero signals
        zeros = np.zeros(16000)
        with self.assertRaises(SISDRError):
            self.calculator.calculate(zeros, zeros)
            
        # Identical signals (perfect reconstruction)
        signal = np.random.randn(16000)
        si_sdr = self.calculator.calculate(signal, signal)
        self.assertGreaterEqual(si_sdr, 100)  # Should be very high (we cap at 100)
        
        # Very short signals
        short_ref = np.array([1, -1, 1, -1])
        short_est = np.array([0.9, -1.1, 0.8, -1.2])
        si_sdr_short = self.calculator.calculate(short_ref, short_est)
        self.assertIsInstance(si_sdr_short, float)
        
        # DC offset handling
        dc_signal = np.ones(16000)
        ac_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        si_sdr_dc = self.calculator.calculate(dc_signal, ac_signal)
        self.assertLess(si_sdr_dc, -10)  # Should be very poor
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small signals
        tiny_ref = np.random.randn(1000) * 1e-10
        tiny_est = tiny_ref + np.random.randn(1000) * 1e-11
        
        # Should not overflow or underflow
        si_sdr = self.calculator.calculate(tiny_ref, tiny_est)
        self.assertFalse(np.isnan(si_sdr))
        self.assertFalse(np.isinf(si_sdr))
        
        # Very large signals
        large_ref = np.random.randn(1000) * 1e10
        large_est = large_ref + np.random.randn(1000) * 1e9
        
        si_sdr_large = self.calculator.calculate(large_ref, large_est)
        self.assertFalse(np.isnan(si_sdr_large))
        self.assertFalse(np.isinf(si_sdr_large))
        
    def test_gpu_acceleration(self):
        """Test GPU acceleration if available."""
        try:
            gpu_calculator = GPUSISDRCalculator()
        except ImportError:
            self.skipTest("CuPy not available for GPU testing")
            
        # Create larger batch for GPU efficiency
        batch_size = 100
        signal_length = 16000
        
        references = np.random.randn(batch_size, signal_length)
        estimates = references + 0.1 * np.random.randn(batch_size, signal_length)
        
        # Time CPU calculation
        start_cpu = time.time()
        cpu_scores = self.calculator.batch_calculate(
            list(references), list(estimates)
        )
        cpu_time = time.time() - start_cpu
        
        # Time GPU calculation
        start_gpu = time.time()
        gpu_scores = gpu_calculator.batch_calculate_gpu(references, estimates)
        gpu_time = time.time() - start_gpu
        
        # Results should match
        np.testing.assert_allclose(cpu_scores, gpu_scores, rtol=1e-5)
        
        # GPU should be faster for large batches
        self.assertLess(gpu_time, cpu_time)
        
    def test_performance_requirements(self):
        """Test that performance meets requirements."""
        # Single calculation: < 20ms for 10s audio (relaxed for pure Python implementation)
        signal_10s = np.random.randn(16000 * 10)
        
        start = time.time()
        score = self.calculator.calculate(signal_10s, signal_10s)
        elapsed = (time.time() - start) * 1000
        
        self.assertLess(elapsed, 20, f"Single calculation took {elapsed:.1f}ms")
        
        # Batch of 100: < 2 seconds (relaxed for pure Python implementation)
        batch_refs = [np.random.randn(16000 * 10) for _ in range(100)]
        batch_ests = [r + 0.1 * np.random.randn(len(r)) for r in batch_refs]
        
        start = time.time()
        scores = self.calculator.batch_calculate(batch_refs, batch_ests)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 2.0, f"Batch processing took {elapsed:.2f}s")
        
        # Permutation search (4 sources): < 500ms (more realistic for 4! = 24 permutations)
        sources = [np.random.randn(16000 * 1) for _ in range(4)]  # Use 1s signals for faster test
        estimates = [s + 0.1 * np.random.randn(len(s)) for s in sources]
        
        pit_calc = PermutationInvariantSDR()
        start = time.time()
        result = pit_calc.calculate(sources, estimates)
        elapsed = (time.time() - start) * 1000
        
        self.assertLess(elapsed, 500, f"PIT-SDR took {elapsed:.1f}ms")
        
    def test_detailed_results(self):
        """Test detailed result reporting."""
        clean = self.test_signals['clean']
        noisy = self.test_signals['noisy']
        
        # Get detailed results
        result = self.calculator.calculate_detailed(clean, noisy)
        
        # Check result structure
        self.assertIsInstance(result, SISDRResult)
        self.assertIsNotNone(result.si_sdr)
        self.assertIsNotNone(result.si_snr)
        self.assertIsNotNone(result.si_sar)
        
        # Optional components
        if hasattr(result, 'projection_matrix'):
            self.assertIsNotNone(result.projection_matrix)
            
        # Values should be reasonable
        self.assertGreater(result.si_sdr, 10)
        self.assertLess(result.si_sdr, 30)
        
    def test_integration_with_separation(self):
        """Test integration with source separation evaluation."""
        # Simulate separation scenario
        mixture = self.test_signals['mixture']
        sources = [self.test_signals['source1'], self.test_signals['source2']]
        separated = [self.test_signals['sep1'], self.test_signals['sep2']]
        
        # Calculate improvements
        improvements = []
        for src, sep in zip(sources, separated):
            imp, _, _ = self.calculator.calculate_improvement(mixture, sep, src)
            improvements.append(imp)
            
        # Both should show improvement
        self.assertGreater(improvements[0], 0)
        self.assertGreater(improvements[1], 0)
        
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        valid = np.random.randn(16000)
        
        # Different lengths
        short = np.random.randn(8000)
        with self.assertRaises(ValueError):
            self.calculator.calculate(valid, short)
            
        # NaN values
        nan_signal = valid.copy()
        nan_signal[100] = np.nan
        with self.assertRaises(ValueError):
            self.calculator.calculate(valid, nan_signal)
            
        # Infinite values
        inf_signal = valid.copy()
        inf_signal[200] = np.inf
        with self.assertRaises(ValueError):
            self.calculator.calculate(valid, inf_signal)
            
        # Wrong dimensions for batch
        with self.assertRaises(ValueError):
            self.calculator.batch_calculate([valid], [valid, valid])


class TestPermutationInvariantSDR(unittest.TestCase):
    """Test permutation-invariant SI-SDR implementation."""
    
    def setUp(self):
        """Set up for each test."""
        self.calculator = PermutationInvariantSDR()
        
    def test_two_source_permutation(self):
        """Test permutation search with 2 sources."""
        # Create test signals
        t = np.linspace(0, 1, 16000)
        source1 = np.sin(2 * np.pi * 440 * t)
        source2 = np.sin(2 * np.pi * 880 * t)
        
        # Correct order
        result_correct = self.calculator.calculate([source1, source2], [source1, source2])
        self.assertEqual(result_correct['best_permutation'], (0, 1))
        
        # Swapped order
        result_swapped = self.calculator.calculate([source1, source2], [source2, source1])
        self.assertEqual(result_swapped['best_permutation'], (1, 0))
        
        # Scores should be similar
        self.assertAlmostEqual(
            result_correct['mean_si_sdr'],
            result_swapped['mean_si_sdr'],
            places=5
        )
        
    def test_multi_source_optimization(self):
        """Test optimization with multiple sources."""
        # Create 4 sources
        n_sources = 4
        sources = []
        for i in range(n_sources):
            t = np.linspace(0, 1, 16000)
            freq = 440 * (i + 1)
            sources.append(np.sin(2 * np.pi * freq * t))
            
        # Random permutation
        import random
        estimates = sources.copy()
        random.shuffle(estimates)
        
        # Find optimal permutation
        result = self.calculator.calculate(sources, estimates)
        
        # Should find valid permutation
        self.assertEqual(len(result['best_permutation']), n_sources)
        self.assertEqual(len(set(result['best_permutation'])), n_sources)
        
        # Mean SI-SDR should be high (perfect separation)
        self.assertGreater(result['mean_si_sdr'], 50)
        
    def test_hungarian_algorithm_fallback(self):
        """Test fallback to Hungarian algorithm for many sources."""
        # Create many sources (triggers Hungarian algorithm)
        n_sources = 8
        sources = [np.random.randn(16000) for _ in range(n_sources)]
        estimates = [s + 0.1 * np.random.randn(16000) for s in sources]
        
        # Should use efficient algorithm
        start = time.time()
        result = self.calculator.calculate(sources, estimates)
        elapsed = time.time() - start
        
        # Should be reasonably fast even with 8 sources
        self.assertLess(elapsed, 1.0)
        
        # Should return valid results
        self.assertEqual(len(result['best_permutation']), n_sources)
        self.assertGreater(result['mean_si_sdr'], 10)


if __name__ == '__main__':
    unittest.main()