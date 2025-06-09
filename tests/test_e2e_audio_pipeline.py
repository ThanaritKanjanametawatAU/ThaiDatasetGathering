"""
End-to-End tests for the complete audio analysis pipeline.

This module tests the full pipeline from audio file input to final decision output,
including all intermediate processing stages and data transformations.
"""

import unittest
import pytest
import numpy as np
import tempfile
import os
import json
import time
from pathlib import Path
import soundfile as sf
from unittest.mock import patch, MagicMock

# Import pipeline components
from processors.audio_enhancement.audio_loader import AudioLoader
from utils.enhanced_snr_calculator import EnhancedSNRCalculator
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer
from processors.audio_enhancement.detection.pattern_detector import PatternDetector
from processors.audio_enhancement.issue_categorization import IssueCategorizer
from processors.audio_enhancement.decision_framework import DecisionEngine, DecisionContext


class AudioAnalysisPipeline:
    """Complete audio analysis pipeline for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.components = self._initialize_components()
        self.metrics = {}
        
    def _initialize_components(self):
        """Initialize all pipeline components."""
        return {
            'loader': AudioLoader(),
            'snr_calculator': EnhancedSNRCalculator(),
            'spectral_analyzer': SpectralAnalyzer(),
            'pattern_detector': PatternDetector(),
            'issue_categorizer': IssueCategorizer(),
            'decision_engine': DecisionEngine()
        }
        
    def process(self, audio_path):
        """Process audio file through complete pipeline."""
        start_time = time.time()
        
        # Stage 1: Load and validate audio
        audio_data = self.components['loader'].load(audio_path)
        self.metrics['loading_time'] = time.time() - start_time
        
        # Stage 2: Calculate SNR metrics
        snr_start = time.time()
        snr_metrics = self.components['snr_calculator'].calculate(
            audio_data.samples, 
            audio_data.sample_rate
        )
        self.metrics['snr_calculation_time'] = time.time() - snr_start
        
        # Stage 3: Extract spectral features
        spectral_start = time.time()
        spectral_features = self.components['spectral_analyzer'].analyze(
            audio_data.samples,
            audio_data.sample_rate
        )
        self.metrics['spectral_analysis_time'] = time.time() - spectral_start
        
        # Stage 4: Detect patterns
        pattern_start = time.time()
        patterns = self.components['pattern_detector'].detect(
            audio_data.samples,
            spectral_features
        )
        self.metrics['pattern_detection_time'] = time.time() - pattern_start
        
        # Stage 5: Categorize issues
        categorization_start = time.time()
        issues = self.components['issue_categorizer'].categorize(
            snr_metrics,
            spectral_features,
            patterns
        )
        self.metrics['categorization_time'] = time.time() - categorization_start
        
        # Stage 6: Make decision
        decision_start = time.time()
        context = DecisionContext(
            audio_metrics=snr_metrics,
            spectral_features=spectral_features,
            detected_patterns=patterns
        )
        decision = self.components['decision_engine'].decide(issues, context)
        self.metrics['decision_time'] = time.time() - decision_start
        
        # Total processing time
        self.metrics['total_time'] = time.time() - start_time
        
        return {
            'audio_data': audio_data,
            'snr_metrics': snr_metrics,
            'spectral_features': spectral_features,
            'patterns': patterns,
            'issues': issues,
            'decision': decision,
            'metrics': self.metrics
        }


class TestE2EAudioPipeline(unittest.TestCase):
    """End-to-end tests for audio analysis pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.test_data_dir = tempfile.mkdtemp()
        cls.test_files = cls._create_test_audio_files()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(cls.test_data_dir)
        
    @classmethod
    def _create_test_audio_files(cls):
        """Create various test audio files."""
        files = {}
        sample_rate = 16000
        
        # Clean audio
        clean_path = os.path.join(cls.test_data_dir, 'clean.wav')
        t = np.linspace(0, 2, 2 * sample_rate)
        clean_audio = np.sin(2 * np.pi * 440 * t) * 0.5
        sf.write(clean_path, clean_audio, sample_rate)
        files['clean'] = clean_path
        
        # Noisy audio
        noisy_path = os.path.join(cls.test_data_dir, 'noisy.wav')
        noise = np.random.normal(0, 0.1, len(clean_audio))
        noisy_audio = clean_audio + noise
        sf.write(noisy_path, noisy_audio, sample_rate)
        files['noisy'] = noisy_path
        
        # Very noisy audio
        very_noisy_path = os.path.join(cls.test_data_dir, 'very_noisy.wav')
        heavy_noise = np.random.normal(0, 0.3, len(clean_audio))
        very_noisy_audio = clean_audio + heavy_noise
        sf.write(very_noisy_path, very_noisy_audio, sample_rate)
        files['very_noisy'] = very_noisy_path
        
        # Clipped audio
        clipped_path = os.path.join(cls.test_data_dir, 'clipped.wav')
        clipped_audio = np.clip(clean_audio * 3, -0.9, 0.9)
        sf.write(clipped_path, clipped_audio, sample_rate)
        files['clipped'] = clipped_path
        
        # Silent audio
        silent_path = os.path.join(cls.test_data_dir, 'silent.wav')
        silent_audio = np.zeros(2 * sample_rate)
        sf.write(silent_path, silent_audio, sample_rate)
        files['silent'] = silent_path
        
        # Short audio
        short_path = os.path.join(cls.test_data_dir, 'short.wav')
        short_audio = clean_audio[:sample_rate // 4]  # 0.25 seconds
        sf.write(short_path, short_audio, sample_rate)
        files['short'] = short_path
        
        return files
        
    def setUp(self):
        """Set up for each test."""
        self.pipeline = AudioAnalysisPipeline()
        
    def test_clean_audio_pipeline(self):
        """Test pipeline with clean audio."""
        result = self.pipeline.process(self.test_files['clean'])
        
        # Verify all stages completed
        self.assertIn('audio_data', result)
        self.assertIn('snr_metrics', result)
        self.assertIn('spectral_features', result)
        self.assertIn('patterns', result)
        self.assertIn('issues', result)
        self.assertIn('decision', result)
        
        # Clean audio should have high SNR
        self.assertGreater(result['snr_metrics'].global_snr, 30)
        
        # Should decide to process without enhancement
        self.assertEqual(result['decision'].action, 'PROCESS')
        
        # Should complete quickly
        self.assertLess(result['metrics']['total_time'], 1.0)
        
    def test_noisy_audio_pipeline(self):
        """Test pipeline with noisy audio."""
        result = self.pipeline.process(self.test_files['noisy'])
        
        # Should detect lower SNR
        self.assertLess(result['snr_metrics'].global_snr, 20)
        
        # Should recommend enhancement
        self.assertEqual(result['decision'].action, 'ENHANCE')
        
        # Should identify noise issue
        noise_issues = [
            issue for issue in result['issues'] 
            if issue.type.value == 'noise'
        ]
        self.assertGreater(len(noise_issues), 0)
        
    def test_very_noisy_audio_pipeline(self):
        """Test pipeline with very noisy audio."""
        result = self.pipeline.process(self.test_files['very_noisy'])
        
        # Should detect very low SNR
        self.assertLess(result['snr_metrics'].global_snr, 10)
        
        # Should either enhance or reject
        self.assertIn(result['decision'].action, ['ENHANCE', 'REJECT'])
        
        # Should identify severe noise issue
        severe_issues = [
            issue for issue in result['issues']
            if issue.severity.value == 'severe'
        ]
        self.assertGreater(len(severe_issues), 0)
        
    def test_clipped_audio_pipeline(self):
        """Test pipeline with clipped audio."""
        result = self.pipeline.process(self.test_files['clipped'])
        
        # Should detect clipping patterns
        clipping_patterns = [
            p for p in result['patterns']
            if 'clipping' in p.pattern_type.lower()
        ]
        self.assertGreater(len(clipping_patterns), 0)
        
        # Should recommend enhancement
        self.assertEqual(result['decision'].action, 'ENHANCE')
        
    def test_silent_audio_pipeline(self):
        """Test pipeline with silent audio."""
        result = self.pipeline.process(self.test_files['silent'])
        
        # Should detect silence
        self.assertTrue(result['snr_metrics'].is_silent)
        
        # Should reject silent audio
        self.assertEqual(result['decision'].action, 'REJECT')
        
    def test_short_audio_pipeline(self):
        """Test pipeline with short audio."""
        result = self.pipeline.process(self.test_files['short'])
        
        # Should still process successfully
        self.assertIn('decision', result)
        
        # May have limited frequency resolution
        self.assertIsNotNone(result['spectral_features'])
        
    def test_pipeline_data_flow(self):
        """Test data flows correctly through all stages."""
        result = self.pipeline.process(self.test_files['clean'])
        
        # Audio data should be preserved
        audio_data = result['audio_data']
        self.assertEqual(audio_data.sample_rate, 16000)
        self.assertEqual(len(audio_data.samples), 2 * 16000)
        
        # SNR metrics should use audio data
        snr_metrics = result['snr_metrics']
        self.assertIsNotNone(snr_metrics.global_snr)
        self.assertIsNotNone(snr_metrics.segmental_snr)
        
        # Spectral features should match audio length
        spectral_features = result['spectral_features']
        self.assertGreater(len(spectral_features.frequency_bins), 0)
        
        # Patterns should reference spectral features
        patterns = result['patterns']
        for pattern in patterns:
            self.assertIsNotNone(pattern.confidence)
            
        # Issues should reference metrics
        issues = result['issues']
        for issue in issues:
            self.assertIsNotNone(issue.severity)
            
        # Decision should consider all inputs
        decision = result['decision']
        self.assertIsNotNone(decision.confidence)
        self.assertGreater(len(decision.reasoning), 0)
        
    def test_pipeline_error_recovery(self):
        """Test pipeline handles errors gracefully."""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.pipeline.process('/non/existent/file.wav')
            
        # Test with corrupted file path
        corrupted_path = os.path.join(self.test_data_dir, 'corrupted.wav')
        with open(corrupted_path, 'w') as f:
            f.write('not audio data')
            
        with self.assertRaises(Exception):
            self.pipeline.process(corrupted_path)
            
    def test_pipeline_performance(self):
        """Test pipeline performance meets requirements."""
        # Process multiple files and measure time
        processing_times = []
        
        for audio_type in ['clean', 'noisy', 'clipped']:
            start = time.time()
            result = self.pipeline.process(self.test_files[audio_type])
            processing_times.append(time.time() - start)
            
        # Average processing time should be < 0.5s for 2s audio
        avg_time = np.mean(processing_times)
        self.assertLess(avg_time, 0.5)
        
        # Check component timing
        result = self.pipeline.process(self.test_files['clean'])
        metrics = result['metrics']
        
        # Each component should be fast
        self.assertLess(metrics['snr_calculation_time'], 0.1)
        self.assertLess(metrics['spectral_analysis_time'], 0.2)
        self.assertLess(metrics['pattern_detection_time'], 0.1)
        self.assertLess(metrics['categorization_time'], 0.05)
        self.assertLess(metrics['decision_time'], 0.05)
        
    def test_pipeline_consistency(self):
        """Test pipeline produces consistent results."""
        # Process same file multiple times
        results = []
        for _ in range(3):
            result = self.pipeline.process(self.test_files['clean'])
            results.append(result)
            
        # SNR should be consistent
        snr_values = [r['snr_metrics'].global_snr for r in results]
        self.assertAlmostEqual(snr_values[0], snr_values[1], places=2)
        self.assertAlmostEqual(snr_values[1], snr_values[2], places=2)
        
        # Decision should be consistent
        decisions = [r['decision'].action for r in results]
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(decisions[1], decisions[2])
        
    def test_batch_processing(self):
        """Test processing multiple files in batch."""
        batch_results = {}
        
        for audio_type, audio_path in self.test_files.items():
            batch_results[audio_type] = self.pipeline.process(audio_path)
            
        # Verify all processed
        self.assertEqual(len(batch_results), len(self.test_files))
        
        # Each should have different characteristics
        self.assertGreater(
            batch_results['clean']['snr_metrics'].global_snr,
            batch_results['noisy']['snr_metrics'].global_snr
        )
        
        self.assertGreater(
            batch_results['noisy']['snr_metrics'].global_snr,
            batch_results['very_noisy']['snr_metrics'].global_snr
        )
        
    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        custom_config = {
            'snr_calculator': {
                'frame_length': 0.025,
                'min_speech_duration': 0.3
            },
            'decision_engine': {
                'enhancement_threshold': 25,
                'rejection_threshold': 5
            }
        }
        
        custom_pipeline = AudioAnalysisPipeline(custom_config)
        
        # Process with custom config
        result = custom_pipeline.process(self.test_files['noisy'])
        
        # With higher enhancement threshold, might not enhance
        # (depends on actual SNR of test file)
        self.assertIn(result['decision'].action, ['PROCESS', 'ENHANCE'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test environment."""
        self.pipeline = AudioAnalysisPipeline()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_extremely_long_audio(self):
        """Test with very long audio file."""
        # Create 60 second audio file
        long_path = os.path.join(self.temp_dir, 'long.wav')
        sample_rate = 16000
        duration = 60
        t = np.linspace(0, duration, duration * sample_rate)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        sf.write(long_path, audio, sample_rate)
        
        # Should process successfully
        result = self.pipeline.process(long_path)
        self.assertIn('decision', result)
        
        # Should not take too long
        self.assertLess(result['metrics']['total_time'], 10.0)
        
    def test_very_high_sample_rate(self):
        """Test with high sample rate audio."""
        high_sr_path = os.path.join(self.temp_dir, 'high_sr.wav')
        sample_rate = 48000
        duration = 1
        t = np.linspace(0, duration, duration * sample_rate)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        sf.write(high_sr_path, audio, sample_rate)
        
        # Should handle different sample rate
        result = self.pipeline.process(high_sr_path)
        self.assertEqual(result['audio_data'].sample_rate, sample_rate)
        
    def test_mono_vs_stereo(self):
        """Test mono and stereo audio handling."""
        # Mono audio
        mono_path = os.path.join(self.temp_dir, 'mono.wav')
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        mono_audio = np.sin(2 * np.pi * 440 * t)
        sf.write(mono_path, mono_audio, sample_rate)
        
        # Stereo audio
        stereo_path = os.path.join(self.temp_dir, 'stereo.wav')
        stereo_audio = np.column_stack([mono_audio, mono_audio * 0.8])
        sf.write(stereo_path, stereo_audio, sample_rate)
        
        # Both should process successfully
        mono_result = self.pipeline.process(mono_path)
        stereo_result = self.pipeline.process(stereo_path)
        
        self.assertIn('decision', mono_result)
        self.assertIn('decision', stereo_result)
        
    def test_extreme_amplitude_values(self):
        """Test with extreme amplitude values."""
        # Very quiet audio
        quiet_path = os.path.join(self.temp_dir, 'quiet.wav')
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        quiet_audio = np.sin(2 * np.pi * 440 * t) * 0.001
        sf.write(quiet_path, quiet_audio, sample_rate)
        
        # Very loud audio
        loud_path = os.path.join(self.temp_dir, 'loud.wav')
        loud_audio = np.sin(2 * np.pi * 440 * t) * 0.99
        sf.write(loud_path, loud_audio, sample_rate)
        
        # Both should process
        quiet_result = self.pipeline.process(quiet_path)
        loud_result = self.pipeline.process(loud_path)
        
        # Quiet might be detected as low SNR or silent
        self.assertIn(quiet_result['decision'].action, ['ENHANCE', 'REJECT'])
        
        # Loud should process normally
        self.assertEqual(loud_result['decision'].action, 'PROCESS')


if __name__ == '__main__':
    unittest.main()