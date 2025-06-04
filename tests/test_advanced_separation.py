"""
Test suite for Phase 2: Advanced Separation Models
Tests SepFormer/Conv-TasNet integration, exclusion logic, and post-processing
"""

import unittest
import numpy as np
import tempfile
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# These imports will fail initially (RED phase)
from processors.audio_enhancement.separation import (
    SepFormerEngine,
    ConvTasNetEngine,
    SeparationResult,
    ExclusionCriteria,
    SPEECHBRAIN_AVAILABLE
)
from processors.audio_enhancement.post_processing import (
    ArtifactRemover,
    SpectralSmoother,
    LevelNormalizer
)
from processors.audio_enhancement.core import AudioEnhancer
from utils.audio_metrics import calculate_si_sdr, calculate_pesq, calculate_stoi


class TestAdvancedSeparationModels(unittest.TestCase):
    """Test advanced separation model integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.duration = 2.0  # 2 seconds
        self.samples = int(self.sample_rate * self.duration)
        
        # Create synthetic test audio with two speakers
        t = np.linspace(0, self.duration, self.samples)
        # Primary speaker: 200Hz tone with harmonics
        self.primary = (
            0.5 * np.sin(2 * np.pi * 200 * t) +
            0.3 * np.sin(2 * np.pi * 400 * t) +
            0.2 * np.sin(2 * np.pi * 600 * t)
        )
        # Secondary speaker: 300Hz tone with harmonics
        self.secondary = (
            0.4 * np.sin(2 * np.pi * 300 * t) +
            0.3 * np.sin(2 * np.pi * 600 * t) +
            0.2 * np.sin(2 * np.pi * 900 * t)
        )
        # Mixed audio with overlap
        self.mixed = self.primary + 0.7 * self.secondary
        # Add some noise
        self.mixed += 0.05 * np.random.randn(self.samples)
        
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_sepformer_engine_initialization(self):
        """Test SepFormer engine can be initialized"""
        engine = SepFormerEngine(use_gpu=False)
        self.assertIsNotNone(engine)
        self.assertTrue(engine.is_available())
        self.assertEqual(engine.model_name, "speechbrain/sepformer-wham")
        
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_conv_tasnet_engine_initialization(self):
        """Test Conv-TasNet engine can be initialized"""
        engine = ConvTasNetEngine(use_gpu=False)
        self.assertIsNotNone(engine)
        self.assertTrue(engine.is_available())
        self.assertEqual(engine.model_name, "speechbrain/conv-tasnet-wham")
        
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_sepformer_separation_quality(self):
        """Test SepFormer produces high-quality separation"""
        engine = SepFormerEngine(use_gpu=False)
        result = engine.separate(self.mixed, self.sample_rate)
        
        # Should return SeparationResult
        self.assertIsInstance(result, SeparationResult)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.primary_audio)
        self.assertEqual(len(result.primary_audio), len(self.mixed))
        
        # Check quality metrics
        si_sdr = calculate_si_sdr(self.primary, result.primary_audio)
        self.assertGreater(si_sdr, 10.0)  # Expect >10dB improvement
        
        # Check that secondary speaker is suppressed
        correlation_with_secondary = np.corrcoef(result.primary_audio, self.secondary)[0, 1]
        self.assertLess(abs(correlation_with_secondary), 0.3)  # Low correlation
        
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_conv_tasnet_separation_quality(self):
        """Test Conv-TasNet produces good separation"""
        engine = ConvTasNetEngine(use_gpu=False)
        result = engine.separate(self.mixed, self.sample_rate)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.primary_audio)
        
        # Conv-TasNet might have slightly lower quality than SepFormer
        si_sdr = calculate_si_sdr(self.primary, result.primary_audio)
        self.assertGreater(si_sdr, 8.0)  # Expect >8dB improvement
        
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_fallback_mechanism(self):
        """Test fallback from SepFormer to Conv-TasNet on memory error"""
        # Simulate large audio that causes memory issues
        large_audio = np.random.randn(self.sample_rate * 60)  # 60 seconds
        
        engine = SepFormerEngine(use_gpu=False, max_duration=30.0)
        result = engine.separate(large_audio, self.sample_rate)
        
        # Should fallback to Conv-TasNet
        self.assertTrue(result.success)
        self.assertEqual(result.separation_method, "conv-tasnet")
        self.assertIn("Memory constraint", result.metadata.get("fallback_reason", ""))
        
    def test_exclusion_criteria_initialization(self):
        """Test exclusion criteria configuration"""
        criteria = ExclusionCriteria(
            min_si_sdr=10.0,
            min_pesq=3.0,
            min_stoi=0.85,
            max_attempts=2
        )
        self.assertEqual(criteria.min_si_sdr, 10.0)
        self.assertEqual(criteria.min_pesq, 3.0)
        self.assertEqual(criteria.min_stoi, 0.85)
        self.assertEqual(criteria.max_attempts, 2)
        
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_exclusion_logic_poor_quality(self):
        """Test that poor quality separations are excluded"""
        # Create very noisy mixed signal
        noise = np.random.randn(self.samples)
        very_noisy = 0.1 * self.primary + 0.9 * noise
        
        engine = SepFormerEngine(use_gpu=False)
        criteria = ExclusionCriteria(min_si_sdr=15.0)  # High threshold
        
        result = engine.separate(very_noisy, self.sample_rate)
        
        # Check if it meets exclusion criteria
        should_exclude = criteria.should_exclude(result)
        self.assertTrue(should_exclude)
        self.assertIn("SI-SDR below threshold", result.exclusion_reason)
        
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_exclusion_logic_good_quality(self):
        """Test that good quality separations are not excluded"""
        engine = SepFormerEngine(use_gpu=False)
        criteria = ExclusionCriteria(min_si_sdr=5.0)  # Reasonable threshold
        
        result = engine.separate(self.mixed, self.sample_rate)
        
        should_exclude = criteria.should_exclude(result)
        self.assertFalse(should_exclude)
        self.assertIsNone(result.exclusion_reason)
        
    def test_artifact_remover(self):
        """Test artifact removal post-processing"""
        remover = ArtifactRemover()
        
        # Create audio with artificial clicks/pops
        audio_with_artifacts = self.primary.copy()
        # Add clicks
        audio_with_artifacts[1000] = 2.0
        audio_with_artifacts[5000] = -2.0
        audio_with_artifacts[10000] = 1.5
        
        cleaned = remover.process(audio_with_artifacts, self.sample_rate)
        
        # Check that artifacts are reduced
        self.assertLess(np.max(np.abs(cleaned)), np.max(np.abs(audio_with_artifacts)))
        # Check that overall signal is preserved
        correlation = np.corrcoef(self.primary, cleaned)[0, 1]
        self.assertGreater(correlation, 0.95)
        
    def test_spectral_smoother(self):
        """Test spectral smoothing post-processing"""
        smoother = SpectralSmoother(smoothing_factor=0.8)
        
        # Create audio with harsh spectral content
        harsh_audio = self.primary + 0.1 * np.random.randn(self.samples)
        
        smoothed = smoother.process(harsh_audio, self.sample_rate)
        
        # Check that high frequencies are reduced
        fft_original = np.abs(np.fft.rfft(harsh_audio))
        fft_smoothed = np.abs(np.fft.rfft(smoothed))
        
        # High frequency energy should be reduced
        high_freq_original = np.mean(fft_original[len(fft_original)//2:])
        high_freq_smoothed = np.mean(fft_smoothed[len(fft_smoothed)//2:])
        self.assertLess(high_freq_smoothed, high_freq_original)
        
    def test_level_normalizer(self):
        """Test audio level normalization"""
        normalizer = LevelNormalizer(target_db=-20.0)
        
        # Create quiet audio
        quiet_audio = 0.01 * self.primary
        
        normalized = normalizer.process(quiet_audio, self.sample_rate)
        
        # Check RMS level
        rms_original = np.sqrt(np.mean(quiet_audio**2))
        rms_normalized = np.sqrt(np.mean(normalized**2))
        
        # Normalized should be louder
        self.assertGreater(rms_normalized, rms_original)
        
        # Check target level (approximately -20dB)
        db_normalized = 20 * np.log10(rms_normalized)
        self.assertAlmostEqual(db_normalized, -20.0, delta=3.0)
        
    def test_full_post_processing_pipeline(self):
        """Test complete post-processing pipeline"""
        # Create audio with various issues
        problematic_audio = self.primary.copy()
        problematic_audio[1000] = 1.5  # Click
        problematic_audio *= 0.01  # Too quiet
        problematic_audio += 0.05 * np.random.randn(self.samples)  # Noise
        
        # Apply all post-processing
        remover = ArtifactRemover()
        smoother = SpectralSmoother()
        normalizer = LevelNormalizer()
        
        processed = remover.process(problematic_audio, self.sample_rate)
        processed = smoother.process(processed, self.sample_rate)
        processed = normalizer.process(processed, self.sample_rate)
        
        # Should be clean, smooth, and properly leveled
        self.assertLess(np.max(np.abs(processed)), 1.5)  # No extreme values
        rms = np.sqrt(np.mean(processed**2))
        self.assertGreater(rms, 0.05)  # Not too quiet
        
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_integration_with_audio_enhancer(self):
        """Test integration with main AudioEnhancer class"""
        enhancer = AudioEnhancer(
            use_gpu=False,
            enhancement_level="ultra_aggressive"
        )
        
        # Enable advanced separation
        enhancer.enable_advanced_separation(
            primary_model="sepformer",
            fallback_model="conv-tasnet",
            exclusion_criteria=ExclusionCriteria(min_si_sdr=10.0)
        )
        
        # Process mixed audio
        enhanced, metadata = enhancer.enhance(
            self.mixed,
            self.sample_rate,
            return_metadata=True
        )
        
        # Check metadata
        self.assertTrue(metadata['enhanced'])
        self.assertTrue(metadata.get('advanced_separation_used', False))
        self.assertIn(metadata.get('separation_model'), ['sepformer', 'conv-tasnet'])
        self.assertIsNotNone(metadata.get('si_sdr_improvement'))
        
        # Check quality
        if not metadata.get('excluded', False):
            self.assertGreater(metadata['si_sdr_improvement'], 5.0)
            
    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not installed")
    def test_batch_processing_with_exclusion(self):
        """Test batch processing with some samples excluded"""
        # Create batch with varying quality
        batch = [
            (self.mixed, self.sample_rate, "good_sample"),
            (0.1 * self.primary + 0.9 * np.random.randn(self.samples), 
             self.sample_rate, "noisy_sample"),
            (self.primary + 0.5 * self.secondary, self.sample_rate, "moderate_sample")
        ]
        
        enhancer = AudioEnhancer(use_gpu=False)
        enhancer.enable_advanced_separation(
            exclusion_criteria=ExclusionCriteria(min_si_sdr=12.0)  # Strict
        )
        
        results = enhancer.process_batch(batch)
        
        # Should have 3 results
        self.assertEqual(len(results), 3)
        
        # Check that at least one is excluded
        excluded_count = sum(1 for _, meta in results if meta.get('excluded', False))
        self.assertGreater(excluded_count, 0)
        
        # Non-excluded samples should have good quality
        for enhanced, metadata in results:
            if not metadata.get('excluded', False):
                self.assertGreater(metadata.get('si_sdr_improvement', 0), 10.0)


class TestSeparationResultDataModel(unittest.TestCase):
    """Test SeparationResult data model"""
    
    def test_separation_result_creation(self):
        """Test creating SeparationResult"""
        audio = np.random.randn(16000)
        result = SeparationResult(
            success=True,
            primary_audio=audio,
            metrics={
                'si_sdr': 15.5,
                'pesq': 3.8,
                'stoi': 0.92
            },
            separation_method='sepformer',
            processing_time=1.23,
            excluded_reason=None
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.primary_audio), 16000)
        self.assertEqual(result.metrics['si_sdr'], 15.5)
        self.assertEqual(result.separation_method, 'sepformer')
        self.assertIsNone(result.excluded_reason)
        
    def test_separation_result_failure(self):
        """Test failed separation result"""
        result = SeparationResult(
            success=False,
            primary_audio=None,
            metrics={},
            separation_method='sepformer',
            processing_time=0.5,
            excluded_reason="Model initialization failed"
        )
        
        self.assertFalse(result.success)
        self.assertIsNone(result.primary_audio)
        self.assertEqual(result.excluded_reason, "Model initialization failed")
        
    def test_separation_result_with_metadata(self):
        """Test separation result with additional metadata"""
        result = SeparationResult(
            success=True,
            primary_audio=np.zeros(16000),
            metrics={'si_sdr': 8.0},
            separation_method='conv-tasnet',
            processing_time=0.8,
            metadata={
                'fallback_used': True,
                'original_method': 'sepformer',
                'fallback_reason': 'Memory constraint'
            }
        )
        
        self.assertTrue(result.metadata['fallback_used'])
        self.assertEqual(result.metadata['original_method'], 'sepformer')


if __name__ == '__main__':
    unittest.main()