"""
Test suite for PESQ (Perceptual Evaluation of Speech Quality) calculator.

Tests ITU-T P.862 compliance, both narrowband and wideband modes,
batch processing, and integration with the audio enhancement pipeline.
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
from unittest.mock import patch, MagicMock, Mock

# Import PESQ calculator (to be implemented)
from processors.audio_enhancement.metrics.pesq_calculator import (
    PESQCalculator, PESQMode, PESQError, PESQResult,
    GPUPESQCalculator, OptimizedPESQCalculator
)


class TestPESQCalculator(unittest.TestCase):
    """Test suite for PESQ calculator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_files = cls._create_test_audio_files()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.temp_dir)
        
    @classmethod
    def _create_test_audio_files(cls):
        """Create test audio files for validation."""
        files = {}
        
        # Clean reference signal
        reference_8k = cls._generate_speech_like_signal(8000, 2.0)
        reference_16k = cls._generate_speech_like_signal(16000, 2.0)
        
        # Save reference files
        ref_8k_path = os.path.join(cls.temp_dir, 'reference_8k.wav')
        ref_16k_path = os.path.join(cls.temp_dir, 'reference_16k.wav')
        sf.write(ref_8k_path, reference_8k, 8000)
        sf.write(ref_16k_path, reference_16k, 16000)
        
        files['reference_8k'] = ref_8k_path
        files['reference_16k'] = ref_16k_path
        
        # Create degraded versions
        degradation_levels = {
            'slight': 0.05,
            'moderate': 0.15,
            'severe': 0.3
        }
        
        for level_name, noise_level in degradation_levels.items():
            # 8kHz degraded
            degraded_8k = reference_8k + np.random.normal(0, noise_level, len(reference_8k))
            deg_8k_path = os.path.join(cls.temp_dir, f'degraded_8k_{level_name}.wav')
            sf.write(deg_8k_path, degraded_8k, 8000)
            files[f'degraded_8k_{level_name}'] = deg_8k_path
            
            # 16kHz degraded
            degraded_16k = reference_16k + np.random.normal(0, noise_level, len(reference_16k))
            deg_16k_path = os.path.join(cls.temp_dir, f'degraded_16k_{level_name}.wav')
            sf.write(deg_16k_path, degraded_16k, 16000)
            files[f'degraded_16k_{level_name}'] = deg_16k_path
            
        return files
        
    @staticmethod
    def _generate_speech_like_signal(sample_rate, duration):
        """Generate a speech-like test signal."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simulate speech with multiple formants
        signal = np.zeros_like(t)
        
        # F1: ~700 Hz (first formant)
        signal += 0.3 * np.sin(2 * np.pi * 700 * t)
        
        # F2: ~1220 Hz (second formant)  
        signal += 0.2 * np.sin(2 * np.pi * 1220 * t)
        
        # F3: ~2600 Hz (third formant)
        signal += 0.1 * np.sin(2 * np.pi * 2600 * t)
        
        # Add amplitude modulation to simulate speech envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4 Hz modulation
        signal *= envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return signal
        
    def setUp(self):
        """Set up for each test."""
        self.calculator = PESQCalculator()
        
    def test_narrowband_calculation(self):
        """Test PESQ calculation in narrowband mode (8kHz)."""
        # Load test files
        reference = self._load_audio(self.test_files['reference_8k'])
        degraded_slight = self._load_audio(self.test_files['degraded_8k_slight'])
        degraded_moderate = self._load_audio(self.test_files['degraded_8k_moderate'])
        degraded_severe = self._load_audio(self.test_files['degraded_8k_severe'])
        
        # Calculate PESQ scores
        score_slight = self.calculator.calculate(reference, degraded_slight, sample_rate=8000, mode=PESQMode.NARROWBAND)
        score_moderate = self.calculator.calculate(reference, degraded_moderate, sample_rate=8000, mode=PESQMode.NARROWBAND)
        score_severe = self.calculator.calculate(reference, degraded_severe, sample_rate=8000, mode=PESQMode.NARROWBAND)
        
        # Verify score ranges (PESQ scores: -0.5 to 4.5)
        self.assertGreaterEqual(score_slight, -0.5)
        self.assertLessEqual(score_slight, 4.5)
        
        # Verify degradation ordering
        self.assertGreater(score_slight, score_moderate)
        self.assertGreater(score_moderate, score_severe)
        
        # Realistic PESQ scores for synthetic signals with noise
        # PESQ scores are typically much lower than expected for simple test signals
        # Slight degradation: typically 1.0-2.0 for synthetic signals
        self.assertGreater(score_slight, 0.8)
        self.assertLess(score_slight, 3.0)
        
        # Severe degradation should have lower score
        self.assertLess(score_severe, score_slight)
        self.assertGreater(score_severe, -0.5)  # Still within valid range
        
    def test_wideband_calculation(self):
        """Test PESQ calculation in wideband mode (16kHz)."""
        # Load test files
        reference = self._load_audio(self.test_files['reference_16k'])
        degraded_slight = self._load_audio(self.test_files['degraded_16k_slight'])
        degraded_moderate = self._load_audio(self.test_files['degraded_16k_moderate'])
        
        # Calculate PESQ scores
        score_slight = self.calculator.calculate(reference, degraded_slight, mode=PESQMode.WIDEBAND)
        score_moderate = self.calculator.calculate(reference, degraded_moderate, mode=PESQMode.WIDEBAND)
        
        # Verify score ranges
        self.assertGreaterEqual(score_slight, -0.5)
        self.assertLessEqual(score_slight, 4.5)
        
        # Verify degradation ordering
        self.assertGreater(score_slight, score_moderate)
        
    def test_auto_mode_detection(self):
        """Test automatic mode detection based on sample rate."""
        # 8kHz should use narrowband
        reference_8k = self._load_audio(self.test_files['reference_8k'])
        degraded_8k = self._load_audio(self.test_files['degraded_8k_slight'])
        
        calculator_auto = PESQCalculator(mode='auto')
        score_8k = calculator_auto.calculate(reference_8k, degraded_8k, sample_rate=8000)
        
        # Should detect narrowband mode
        self.assertEqual(calculator_auto.detected_mode, PESQMode.NARROWBAND)
        
        # 16kHz should use wideband
        reference_16k = self._load_audio(self.test_files['reference_16k'])
        degraded_16k = self._load_audio(self.test_files['degraded_16k_slight'])
        
        score_16k = calculator_auto.calculate(reference_16k, degraded_16k, sample_rate=16000)
        
        # Should detect wideband mode
        self.assertEqual(calculator_auto.detected_mode, PESQMode.WIDEBAND)
        
    def test_batch_processing(self):
        """Test batch processing capability."""
        # Prepare batch data
        reference = self._load_audio(self.test_files['reference_16k'])
        
        degraded_batch = []
        expected_order = []
        
        for level in ['slight', 'moderate', 'severe']:
            degraded = self._load_audio(self.test_files[f'degraded_16k_{level}'])
            degraded_batch.append(degraded)
            expected_order.append(level)
            
        # Process batch
        scores = self.calculator.batch_calculate([reference] * 3, degraded_batch)
        
        # Verify batch size
        self.assertEqual(len(scores), 3)
        
        # Verify score ordering (slight > moderate > severe)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
        
        # Verify performance - batch should be faster than sequential
        start_sequential = time.time()
        for degraded in degraded_batch:
            _ = self.calculator.calculate(reference, degraded, sample_rate=16000)
        sequential_time = time.time() - start_sequential
        
        start_batch = time.time()
        _ = self.calculator.batch_calculate([reference] * 3, degraded_batch, sample_rate=16000)
        batch_time = time.time() - start_batch
        
        # Batch should be at least 2x faster
        self.assertLess(batch_time * 2, sequential_time)
        
    def test_sample_rate_handling(self):
        """Test handling of different sample rates."""
        # Test unsupported sample rate
        reference = np.random.randn(44100)  # 44.1kHz
        degraded = np.random.randn(44100)
        
        with self.assertRaises(ValueError) as context:
            self.calculator.calculate(reference, degraded, sample_rate=44100)
            
        self.assertIn("sample rate", str(context.exception).lower())
        
        # Test mismatched lengths
        reference = np.random.randn(16000)
        degraded = np.random.randn(8000)
        
        with self.assertRaises(ValueError) as context:
            self.calculator.calculate(reference, degraded)
            
        self.assertIn("length", str(context.exception).lower())
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with silence
        silence = np.zeros(16000)
        reference = self._generate_speech_like_signal(16000, 1.0)
        
        score = self.calculator.calculate(reference, silence)
        self.assertLess(score, 1.0)  # Very poor score for silence
        
        # Test with identical signals
        identical_score = self.calculator.calculate(reference, reference)
        self.assertGreater(identical_score, 4.0)  # Near perfect score
        
        # Test with very short audio
        short_ref = reference[:int(16000 * 0.1)]  # 100ms
        short_deg = short_ref + np.random.normal(0, 0.05, len(short_ref))
        
        with self.assertRaises(ValueError) as context:
            self.calculator.calculate(short_ref, short_deg)
            
        self.assertIn("duration", str(context.exception).lower())
        
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        reference = np.random.randn(16000)
        
        # Test with NaN values
        degraded_nan = reference.copy()
        degraded_nan[1000:1100] = np.nan
        
        with self.assertRaises(ValueError) as context:
            self.calculator.calculate(reference, degraded_nan)
            
        self.assertIn("nan", str(context.exception).lower())
        
        # Test with infinite values
        degraded_inf = reference.copy()
        degraded_inf[500] = np.inf
        
        with self.assertRaises(ValueError) as context:
            self.calculator.calculate(reference, degraded_inf)
            
        self.assertIn("inf", str(context.exception).lower())
        
    def test_detailed_results(self):
        """Test calculation with detailed results."""
        reference = self._load_audio(self.test_files['reference_16k'])
        degraded = self._load_audio(self.test_files['degraded_16k_moderate'])
        
        # Get detailed results
        result = self.calculator.calculate_with_details(reference, degraded)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('mos', result)
        self.assertIn('level_difference', result)
        self.assertIn('delay', result)
        self.assertIn('disturbance_profile', result)
        
        # Verify MOS score
        self.assertGreaterEqual(result['mos'], -0.5)
        self.assertLessEqual(result['mos'], 4.5)
        
        # Verify other metrics
        self.assertIsInstance(result['level_difference'], float)
        self.assertIsInstance(result['delay'], int)
        self.assertIsInstance(result['disturbance_profile'], np.ndarray)
        
    def test_gpu_acceleration(self):
        """Test GPU-accelerated PESQ calculation."""
        try:
            import cupy as cp
            gpu_available = cp.cuda.is_available()
        except ImportError:
            gpu_available = False
            
        if not gpu_available:
            self.skipTest("GPU not available")
            
        # Create GPU calculator
        gpu_calculator = GPUPESQCalculator()
        
        # Prepare batch
        reference = self._load_audio(self.test_files['reference_16k'])
        degraded_batch = []
        
        for i in range(10):
            noise = np.random.normal(0, 0.1, len(reference))
            degraded_batch.append(reference + noise)
            
        # Calculate on GPU
        gpu_scores = gpu_calculator.batch_calculate_gpu(
            [reference] * 10,
            degraded_batch
        )
        
        # Verify results
        self.assertEqual(len(gpu_scores), 10)
        
        # All scores should be reasonable
        for score in gpu_scores:
            self.assertGreaterEqual(score, 2.0)
            self.assertLessEqual(score, 4.5)
            
    def test_memory_efficiency(self):
        """Test memory-efficient processing for large batches."""
        if not hasattr(self, 'gpu_calculator'):
            self.skipTest("GPU calculator not available")
            
        # Create large batch
        reference = self._generate_speech_like_signal(16000, 2.0)
        large_batch_ref = [reference] * 100
        large_batch_deg = [reference + np.random.normal(0, 0.1, len(reference)) for _ in range(100)]
        
        # Process with memory-efficient method
        calculator = OptimizedPESQCalculator()
        scores = calculator.memory_efficient_pesq(large_batch_ref, large_batch_deg, chunk_size=10)
        
        # Verify all processed
        self.assertEqual(len(scores), 100)
        
        # Verify reasonable scores
        self.assertGreater(np.mean(scores), 3.0)
        self.assertLess(np.std(scores), 1.0)
        
    def test_itu_compliance(self):
        """Test compliance with ITU-T P.862 standard."""
        # This would test against official ITU test vectors
        # For now, we'll test basic compliance checks
        
        calculator = PESQCalculator()
        
        # Test level alignment
        loud_ref = self._generate_speech_like_signal(16000, 1.0) * 2.0
        quiet_deg = loud_ref * 0.1
        
        result = calculator.calculate_with_details(loud_ref, quiet_deg)
        
        # Should detect significant level difference
        self.assertGreater(abs(result['level_difference']), 10.0)
        
        # Test time alignment
        reference = self._generate_speech_like_signal(16000, 1.0)
        delayed = np.concatenate([np.zeros(800), reference[:-800]])  # 50ms delay
        
        result = calculator.calculate_with_details(reference, delayed)
        
        # Should detect delay
        self.assertGreater(abs(result['delay']), 700)
        self.assertLess(abs(result['delay']), 900)
        
    def test_cross_validation(self):
        """Test cross-validation with pypesq if available."""
        try:
            from pypesq import pesq as pypesq_ref
            pypesq_available = True
        except ImportError:
            pypesq_available = False
            
        # Also try pesq package
        try:
            from pesq import pesq as pesq_ref
            pesq_available = True
        except ImportError:
            pesq_available = False
            
        if not pypesq_available and not pesq_available:
            self.skipTest("No reference PESQ implementation available for cross-validation")
            
        # Test several audio pairs
        reference = self._load_audio(self.test_files['reference_16k'])
        
        for level in ['slight', 'moderate']:
            degraded = self._load_audio(self.test_files[f'degraded_16k_{level}'])
            
            # Our implementation
            our_score = self.calculator.calculate(reference, degraded, sample_rate=16000)
            
            # Reference implementation
            if pesq_available:
                ref_score = pesq_ref(16000, reference, degraded, 'wb')
            else:
                ref_score = pypesq_ref(16000, reference, degraded, 'wb')
            
            # Should be within 0.1 MOS
            self.assertLess(abs(our_score - ref_score), 0.1,
                           f"Score mismatch: {our_score} vs {ref_score}")
    
    def test_performance_targets(self):
        """Test that performance meets specified targets."""
        reference = self._load_audio(self.test_files['reference_16k'])
        degraded = self._load_audio(self.test_files['degraded_16k_moderate'])
        
        # Single calculation should be < 100ms for 2s audio
        start = time.time()
        score = self.calculator.calculate(reference, degraded)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 0.1, f"Single calculation took {elapsed:.3f}s")
        
        # Batch of 100 should be < 5 seconds
        batch_ref = [reference[:16000]] * 100  # 1s clips for faster test
        batch_deg = [degraded[:16000]] * 100
        
        start = time.time()
        scores = self.calculator.batch_calculate(batch_ref, batch_deg)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 5.0, f"Batch processing took {elapsed:.3f}s")
        
    def test_integration_with_pipeline(self):
        """Test integration with audio enhancement pipeline."""
        # Mock audio enhancement pipeline
        from processors.audio_enhancement.core import AudioEnhancer
        
        # Create test audio
        original = self._generate_speech_like_signal(16000, 2.0)
        noisy = original + np.random.normal(0, 0.2, len(original))
        
        # Mock enhancement
        with patch.object(AudioEnhancer, 'enhance') as mock_enhance:
            # Simulate enhancement by reducing noise
            enhanced = original + np.random.normal(0, 0.05, len(original))
            mock_enhance.return_value = enhanced
            
            # Calculate PESQ before and after enhancement
            score_before = self.calculator.calculate(original, noisy)
            score_after = self.calculator.calculate(original, enhanced)
            
            # Enhancement should improve PESQ score
            self.assertGreater(score_after, score_before)
            self.assertGreater(score_after - score_before, 0.5)
            
    def _load_audio(self, file_path):
        """Helper to load audio file."""
        audio, sr = sf.read(file_path)
        return audio


class TestOptimizedPESQImplementation(unittest.TestCase):
    """Test optimized PESQ implementation details."""
    
    def setUp(self):
        """Set up optimization tests."""
        self.calculator = OptimizedPESQCalculator()
        
    def test_filter_bank_initialization(self):
        """Test pre-computed filter bank initialization."""
        # Should have pre-computed filters
        self.assertIsNotNone(self.calculator.narrowband_filters)
        self.assertIsNotNone(self.calculator.wideband_filters)
        
        # Filters should be dictionaries with frame parameters
        self.assertIn('frame_len', self.calculator.narrowband_filters)
        self.assertIn('frame_len', self.calculator.wideband_filters)
        
        # Frame lengths should be different for different sample rates
        self.assertNotEqual(
            self.calculator.narrowband_filters['frame_len'],
            self.calculator.wideband_filters['frame_len']
        )
        
    def test_buffer_preallocation(self):
        """Test buffer pre-allocation for performance."""
        # Process some audio to trigger buffer allocation
        test_audio = np.random.randn(16000)
        self.calculator.calculate(test_audio, test_audio, sample_rate=16000)
        
        # Buffers should be allocated
        self.assertIsNotNone(self.calculator.fft_buffer)
        self.assertIsNotNone(self.calculator.bark_buffer)
        
    def test_level_alignment_algorithm(self):
        """Test active speech level computation (ITU-T P.56)."""
        # Create test signal with speech and silence
        speech = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        silence = np.zeros(8000)
        signal = np.concatenate([silence, speech, silence])
        
        level = self.calculator.compute_active_level(signal, 16000)
        
        # Should compute reasonable level
        self.assertGreater(level, -40)  # Not too quiet
        self.assertLess(level, 0)  # Not clipping
        
    def test_bark_scale_transformation(self):
        """Test frequency to Bark scale transformation."""
        # Test known frequency mappings
        test_frequencies = [100, 500, 1000, 2000, 4000, 8000]
        
        for freq in test_frequencies:
            bark = self.calculator.hz_to_bark(freq)
            
            # Bark scale should be monotonic
            self.assertGreater(bark, 0)
            self.assertLess(bark, 25)  # Max ~24 Bark
            
        # Test monotonicity
        barks = [self.calculator.hz_to_bark(f) for f in test_frequencies]
        for i in range(1, len(barks)):
            self.assertGreater(barks[i], barks[i-1])
            
    def test_critical_band_filtering(self):
        """Test critical band filtering implementation."""
        # Create test spectrum
        sample_rate = 16000
        fft_size = 1024
        spectrum = np.random.randn(fft_size // 2)
        
        critical_bands = self.calculator.apply_critical_band_filtering(
            spectrum, sample_rate
        )
        
        # Should have 24 critical bands
        self.assertEqual(len(critical_bands), 24)
        
        # All bands should have values
        self.assertTrue(all(np.isfinite(critical_bands)))
        
    def test_cognitive_model(self):
        """Test cognitive modeling with asymmetric disturbance."""
        # Create reference and degraded Bark spectra
        ref_bark = np.random.rand(24) * 60 + 20  # 20-80 dB range
        deg_bark = ref_bark + np.random.normal(0, 5, 24)  # Add disturbance
        
        disturbance = self.calculator.calculate_disturbance(ref_bark, deg_bark)
        
        # Disturbance should be computed
        self.assertEqual(len(disturbance), 24)
        
        # Disturbance should be non-negative
        self.assertTrue(all(d >= 0 for d in disturbance))
        
    def test_disturbance_to_mos_mapping(self):
        """Test mapping from disturbance to MOS score."""
        # Test various disturbance levels
        test_disturbances = [0, 10, 50, 100, 200]
        
        mos_scores = []
        for dist in test_disturbances:
            mos = self.calculator.disturbance_to_mos(np.array([dist]))
            mos_scores.append(mos)
            
        # MOS should decrease with increasing disturbance
        for i in range(1, len(mos_scores)):
            self.assertLess(mos_scores[i], mos_scores[i-1])
            
        # All scores should be in valid range
        for mos in mos_scores:
            self.assertGreaterEqual(mos, -0.5)
            self.assertLessEqual(mos, 4.5)


if __name__ == '__main__':
    unittest.main()