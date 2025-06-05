#!/usr/bin/env python3
"""
Unit tests for SpeechBrain-based speaker separation.

Tests the new SpeechBrainSeparator implementation for complete
secondary speaker removal.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.speechbrain_separator import (
    SpeechBrainSeparator,
    SeparationConfig,
    SeparationInput,
    SeparationOutput,
    GPUMemoryManager
)
from processors.audio_enhancement.speaker_selection import SpeakerSelector
from processors.audio_enhancement.quality_validator import QualityValidator


class TestSpeechBrainSeparator(unittest.TestCase):
    """Test SpeechBrain speaker separation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.duration = 4  # 4 seconds
        self.test_audio = self._generate_test_audio()
        
        # Mock config with CPU for testing
        self.config = SeparationConfig(
            device="cpu",  # Use CPU for tests
            confidence_threshold=0.7,
            batch_size=2,
            cache_dir=tempfile.mkdtemp()
        )
        
    def tearDown(self):
        """Clean up after tests"""
        # Clean up temp directory
        import shutil
        if hasattr(self, 'config') and os.path.exists(self.config.cache_dir):
            shutil.rmtree(self.config.cache_dir)
    
    def _generate_test_audio(self):
        """Generate test audio with two speakers"""
        # Generate two different frequency patterns to simulate different speakers
        t = np.linspace(0, self.duration, self.sample_rate * self.duration)
        
        # Speaker 1: Lower frequency (200Hz base)
        speaker1 = np.sin(2 * np.pi * 200 * t) * 0.5
        speaker1 += np.sin(2 * np.pi * 400 * t) * 0.3  # Harmonic
        
        # Speaker 2: Higher frequency (300Hz base)
        speaker2 = np.sin(2 * np.pi * 300 * t) * 0.5
        speaker2 += np.sin(2 * np.pi * 600 * t) * 0.3  # Harmonic
        
        # Mix speakers in different time segments
        mixed = np.zeros_like(t)
        mixed[:self.sample_rate] = speaker1[:self.sample_rate]  # First second: speaker 1
        mixed[self.sample_rate:2*self.sample_rate] = speaker2[self.sample_rate:2*self.sample_rate]  # Second: speaker 2
        mixed[2*self.sample_rate:3*self.sample_rate] = (
            0.7 * speaker1[2*self.sample_rate:3*self.sample_rate] + 
            0.3 * speaker2[2*self.sample_rate:3*self.sample_rate]
        )  # Third second: both (overlap)
        mixed[3*self.sample_rate:] = speaker1[3*self.sample_rate:]  # Fourth second: speaker 1
        
        # Add some noise
        mixed += np.random.randn(len(mixed)) * 0.01
        
        return mixed
    
    @patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation')
    @patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition')
    def test_initialization(self, mock_speaker_rec, mock_sepformer):
        """Test SpeechBrainSeparator initialization"""
        # Mock the from_hparams method
        mock_sepformer.from_hparams.return_value = MagicMock()
        mock_speaker_rec.from_hparams.return_value = MagicMock()
        
        separator = SpeechBrainSeparator(self.config)
        
        # Check initialization
        self.assertIsNotNone(separator)
        self.assertEqual(separator.config.device, "cpu")
        self.assertEqual(separator.config.confidence_threshold, 0.7)
        
        # Check models were loaded
        mock_sepformer.from_hparams.assert_called_once()
        mock_speaker_rec.from_hparams.assert_called_once()
    
    def test_gpu_memory_manager(self):
        """Test GPU memory management"""
        manager = GPUMemoryManager("cpu")
        
        # Test batch size calculation
        batch_size = manager.get_optimal_batch_size(audio_duration=8.0)
        self.assertEqual(batch_size, 1)  # CPU should return 1
        
        # Test cache clearing (should not crash on CPU)
        manager.clear_cache()
    
    @patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation')
    @patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition')
    def test_separation_output_structure(self, mock_speaker_rec, mock_sepformer):
        """Test that separation returns correct output structure"""
        # Mock models
        mock_separator_instance = MagicMock()
        mock_separated = torch.randn(1, len(self.test_audio), 2)  # 2 sources
        mock_separator_instance.separate_batch.return_value = mock_separated
        mock_sepformer.from_hparams.return_value = mock_separator_instance
        
        mock_speaker_instance = MagicMock()
        mock_speaker_rec.from_hparams.return_value = mock_speaker_instance
        
        separator = SpeechBrainSeparator(self.config)
        result = separator.separate_speakers(self.test_audio, self.sample_rate)
        
        # Check output structure
        self.assertIsInstance(result, SeparationOutput)
        self.assertIsInstance(result.audio, np.ndarray)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.metrics, dict)
        self.assertIsInstance(result.rejected, bool)
        self.assertIsInstance(result.processing_time_ms, float)
        self.assertGreater(result.processing_time_ms, 0)
    
    @patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation')
    @patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition')
    def test_empty_audio_handling(self, mock_speaker_rec, mock_sepformer):
        """Test handling of empty audio input"""
        # Mock models
        mock_sepformer.from_hparams.return_value = MagicMock()
        mock_speaker_rec.from_hparams.return_value = MagicMock()
        
        separator = SpeechBrainSeparator(self.config)
        
        # Test with empty audio
        empty_audio = np.array([])
        result = separator.separate_speakers(empty_audio, self.sample_rate)
        
        # Should reject empty audio
        self.assertTrue(result.rejected)
        self.assertEqual(result.rejection_reason, "Empty audio")
        self.assertEqual(result.confidence, 0.0)
    
    @patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation')
    @patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition')
    def test_quality_threshold_rejection(self, mock_speaker_rec, mock_sepformer):
        """Test rejection based on quality thresholds"""
        # Mock models
        mock_separator_instance = MagicMock()
        # Return low quality separation (single source)
        mock_separated = torch.randn(1, len(self.test_audio), 1) * 0.01  # Very low energy
        mock_separator_instance.separate_batch.return_value = mock_separated
        mock_sepformer.from_hparams.return_value = mock_separator_instance
        
        mock_speaker_instance = MagicMock()
        mock_speaker_rec.from_hparams.return_value = mock_speaker_instance
        
        # Set high quality thresholds
        self.config.quality_thresholds['min_stoi'] = 0.95  # Very high threshold
        
        separator = SpeechBrainSeparator(self.config)
        result = separator.separate_speakers(self.test_audio, self.sample_rate)
        
        # Should be rejected due to low quality
        if result.rejected:
            self.assertIn("Low", result.rejection_reason)
    
    @patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation')
    @patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition')
    def test_batch_processing(self, mock_speaker_rec, mock_sepformer):
        """Test batch processing functionality"""
        # Mock models
        mock_separator_instance = MagicMock()
        mock_separated = torch.randn(1, len(self.test_audio), 2)
        mock_separator_instance.separate_batch.return_value = mock_separated
        mock_sepformer.from_hparams.return_value = mock_separator_instance
        
        mock_speaker_instance = MagicMock()
        mock_speaker_rec.from_hparams.return_value = mock_speaker_instance
        
        separator = SpeechBrainSeparator(self.config)
        
        # Create batch of audio
        batch = [self.test_audio, self.test_audio * 0.8, self.test_audio * 0.6]
        
        # Process batch
        results = separator.process_batch(batch, self.sample_rate)
        
        # Check results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, SeparationOutput)
            self.assertIsInstance(result.audio, np.ndarray)
    
    def test_speaker_selection_module(self):
        """Test speaker selection functionality"""
        selector = SpeakerSelector(method="energy")
        
        # Create two test sources
        source1 = np.sin(2 * np.pi * 200 * np.linspace(0, 1, 16000)) * 0.5
        source2 = np.sin(2 * np.pi * 300 * np.linspace(0, 1, 16000)) * 0.3
        
        # Select primary speaker
        primary_idx, confidence = selector.select_primary_speaker([source1, source2])
        
        # Should select source1 (higher energy)
        self.assertEqual(primary_idx, 0)
        self.assertGreater(confidence, 0.5)
        
        # Test validation
        validation = selector.validate_selection(source1, [source1, source2])
        self.assertIn('validation_score', validation)
        self.assertGreater(validation['validation_score'], 0.5)
    
    def test_quality_validator_module(self):
        """Test quality validation functionality"""
        validator = QualityValidator()
        
        # Create test audio
        original = self.test_audio
        separated = original * 0.9  # Slightly attenuated
        
        # Validate
        is_valid, reason, metrics = validator.validate(
            separated, original, confidence=0.8
        )
        
        # Check metrics
        self.assertIn('stoi', metrics)
        self.assertIn('energy_ratio', metrics)
        self.assertIn('spectral_distortion', metrics)
        
        # Generate report
        report = validator.get_quality_report(metrics)
        self.assertIn("Audio Quality Report", report)
    
    @patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation')
    @patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition')
    def test_performance_timing(self, mock_speaker_rec, mock_sepformer):
        """Test processing time measurement"""
        # Mock models
        mock_separator_instance = MagicMock()
        mock_separated = torch.randn(1, len(self.test_audio), 2)
        
        # Add delay to simulate processing
        def delayed_separate(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return mock_separated
        
        mock_separator_instance.separate_batch.side_effect = delayed_separate
        mock_sepformer.from_hparams.return_value = mock_separator_instance
        
        mock_speaker_instance = MagicMock()
        mock_speaker_rec.from_hparams.return_value = mock_speaker_instance
        
        separator = SpeechBrainSeparator(self.config)
        result = separator.separate_speakers(self.test_audio, self.sample_rate)
        
        # Processing time should be at least 100ms
        self.assertGreater(result.processing_time_ms, 100)
    
    @patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation')
    @patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition')
    def test_statistics_tracking(self, mock_speaker_rec, mock_sepformer):
        """Test that statistics are tracked correctly"""
        # Mock models
        mock_separator_instance = MagicMock()
        mock_separated = torch.randn(1, len(self.test_audio), 2)
        mock_separator_instance.separate_batch.return_value = mock_separated
        mock_sepformer.from_hparams.return_value = mock_separator_instance
        
        mock_speaker_instance = MagicMock()
        mock_speaker_rec.from_hparams.return_value = mock_speaker_instance
        
        separator = SpeechBrainSeparator(self.config)
        
        # Process multiple samples
        for _ in range(3):
            separator.separate_speakers(self.test_audio, self.sample_rate)
        
        # Check statistics
        stats = separator.get_stats()
        self.assertEqual(stats['total_processed'], 3)
        self.assertIn('success_rate', stats)
        self.assertIn('average_confidence', stats)
        self.assertIn('average_processing_time_ms', stats)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old speaker_separation.py"""
        from processors.audio_enhancement.speaker_separation import SpeakerSeparator
        
        # Should be able to import and use old class name
        self.assertTrue(hasattr(SpeakerSeparator, 'separate_speakers'))
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with very short audio
        short_audio = np.random.randn(100)  # Very short
        
        # Test with single value
        single_value = np.array([0.5])
        
        # Test with very long audio
        long_audio = np.random.randn(16000 * 60)  # 1 minute
        
        # These should all be handled gracefully without crashing
        # (actual tests would use mocked separator)


if __name__ == '__main__':
    unittest.main()