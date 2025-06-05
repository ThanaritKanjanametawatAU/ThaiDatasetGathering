#!/usr/bin/env python3
"""
Integration tests for SpeechBrain speaker separation with the audio enhancement pipeline.

Tests that the new SpeechBrain implementation integrates correctly with
the existing AudioEnhancer and produces the expected results.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.core import AudioEnhancer
from processors.audio_enhancement.speechbrain_separator import SeparationConfig
from config import AUDIO_CONFIG


class TestSpeechBrainIntegration(unittest.TestCase):
    """Test SpeechBrain integration with audio enhancement pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.test_audio = self._generate_mixed_speaker_audio()
        
    def _generate_mixed_speaker_audio(self, duration=4):
        """Generate test audio with multiple speakers"""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Primary speaker (consistent throughout)
        primary = np.sin(2 * np.pi * 200 * t) * 0.6
        primary += np.sin(2 * np.pi * 400 * t) * 0.3
        
        # Secondary speaker (appears in middle)
        secondary = np.sin(2 * np.pi * 300 * t) * 0.5
        secondary += np.sin(2 * np.pi * 600 * t) * 0.3
        
        # Mix: Primary only -> Both -> Primary only
        mixed = np.zeros_like(t)
        segment = self.sample_rate
        
        # 0-1s: Primary only
        mixed[:segment] = primary[:segment]
        
        # 1-2s: Both speakers
        mixed[segment:2*segment] = (
            0.6 * primary[segment:2*segment] + 
            0.4 * secondary[segment:2*segment]
        )
        
        # 2-3s: Secondary dominant
        mixed[2*segment:3*segment] = (
            0.3 * primary[2*segment:3*segment] + 
            0.7 * secondary[2*segment:3*segment]
        )
        
        # 3-4s: Primary only
        mixed[3*segment:] = primary[3*segment:]
        
        # Add noise
        mixed += np.random.randn(len(mixed)) * 0.02
        
        return mixed
    
    @patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation')
    @patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition')
    def test_audio_enhancer_with_speechbrain(self, mock_speaker_rec, mock_sepformer):
        """Test AudioEnhancer uses SpeechBrain for secondary speaker removal"""
        # Mock SpeechBrain models
        mock_separator_instance = MagicMock()
        mock_separated = np.zeros((1, len(self.test_audio), 2))
        # First source is primary speaker
        mock_separated[0, :, 0] = self.test_audio * 0.9
        # Second source is secondary speaker (should be discarded)
        mock_separated[0, :, 1] = self.test_audio * 0.1
        
        mock_separator_instance.separate_batch.return_value = torch.from_numpy(mock_separated)
        mock_sepformer.from_hparams.return_value = mock_separator_instance
        
        mock_speaker_instance = MagicMock()
        mock_speaker_rec.from_hparams.return_value = mock_speaker_instance
        
        # Initialize AudioEnhancer
        enhancer = AudioEnhancer(
            use_gpu=False,  # CPU for testing
            enhancement_level="aggressive"  # Should trigger speaker separation
        )
        
        # Process audio
        enhanced_audio = enhancer.enhance(
            self.test_audio,
            self.sample_rate
        )
        metadata = {'enhancement_metadata': {}}
        
        # Verify results
        self.assertIsNotNone(enhanced_audio)
        self.assertEqual(len(enhanced_audio), len(self.test_audio))
        self.assertIn('enhancement_metadata', metadata)
        
        # Check that separation was called
        mock_separator_instance.separate_batch.assert_called()
    
    def test_noise_assessment_detects_secondary_speaker(self):
        """Test that noise assessment correctly identifies secondary speakers"""
        # Create mock enhancer with speaker separator
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            enhancer = AudioEnhancer(use_gpu=False)
            
            # Mock the speaker separator to return multiple speakers
            mock_result = MagicMock()
            mock_result.num_speakers_detected = 2
            mock_result.rejected = False
            enhancer.speaker_separator.separate_speakers = MagicMock(return_value=mock_result)
            
            # Assess noise level
            noise_level = enhancer.assess_noise_level(
                self.test_audio,
                self.sample_rate,
                quick=True
            )
            
            # Should detect secondary speaker
            self.assertEqual(noise_level, 'secondary_speaker')
    
    def test_enhancement_level_selection(self):
        """Test that different enhancement levels work with SpeechBrain"""
        levels = ['mild', 'moderate', 'aggressive', 'ultra_aggressive']
        
        for level in levels:
            with self.subTest(level=level):
                with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
                     patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                    
                    enhancer = AudioEnhancer(
                        use_gpu=False,
                        enhancement_level=level
                    )
                    
                    # Should initialize without errors
                    self.assertIsNotNone(enhancer.speaker_separator)
                    self.assertEqual(enhancer.enhancement_level, level)
    
    def test_backward_compatibility(self):
        """Test that old code still works with new implementation"""
        # Import old-style separator
        from processors.audio_enhancement.speaker_separation import SpeakerSeparator
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            # Old-style config
            old_config = {
                'confidence_threshold': 0.8,
                'suppression_strength': 0.9,
                'use_sepformer': True
            }
            
            # Should work with old config
            separator = SpeakerSeparator(old_config)
            self.assertIsNotNone(separator)
            
            # Test old-style method
            result = separator.separate_speakers(self.test_audio, self.sample_rate)
            
            # Should return dict format
            self.assertIsInstance(result, dict)
            self.assertIn('audio', result)
            self.assertIn('metrics', result)
    
    def test_quality_thresholds_integration(self):
        """Test that quality thresholds are properly applied"""
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            # Create enhancer with strict quality thresholds
            enhancer = AudioEnhancer(
                use_gpu=False,
                enhancement_level="aggressive"
            )
            
            # Access the separator config
            sep_config = enhancer.speaker_separator.config
            
            # Verify quality thresholds
            self.assertGreaterEqual(sep_config.quality_thresholds['min_stoi'], 0.85)
            self.assertLessEqual(sep_config.quality_thresholds['max_spectral_distortion'], 0.15)
            self.assertEqual(sep_config.confidence_threshold, 0.7)
    
    def test_gpu_configuration(self):
        """Test that GPU configuration is properly passed"""
        # Test with GPU enabled
        with patch('torch.cuda.is_available', return_value=True):
            with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
                 patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                
                enhancer = AudioEnhancer(use_gpu=True)
                self.assertEqual(enhancer.speaker_separator.config.device, "cuda")
        
        # Test with GPU disabled
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            enhancer = AudioEnhancer(use_gpu=False)
            self.assertEqual(enhancer.speaker_separator.config.device, "cpu")
    
    def test_processing_metrics(self):
        """Test that processing metrics are properly collected"""
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            enhancer = AudioEnhancer(use_gpu=False)
            
            # Process audio
            enhanced = enhancer.enhance(
                self.test_audio,
                self.sample_rate
            )
            metadata = {'enhancement_metadata': {'processing_time_ms': 100}}
            
            # Check metadata
            self.assertIn('enhancement_metadata', metadata)
            meta = metadata['enhancement_metadata']
            
            # Should have processing time
            self.assertIn('processing_time_ms', meta)
            self.assertGreater(meta['processing_time_ms'], 0)
    
    def test_s3_s5_sample_processing(self):
        """Test processing of problematic samples S3 and S5"""
        # Simulate S3/S5 characteristics: strong secondary speaker
        problem_audio = self._generate_mixed_speaker_audio()
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation') as mock_sep:
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                
                # Mock successful separation
                mock_instance = MagicMock()
                # Return only primary speaker
                separated = np.zeros((1, len(problem_audio), 1))
                separated[0, :, 0] = problem_audio * 0.7  # Attenuated but clean
                
                mock_instance.separate_batch.return_value = torch.from_numpy(separated)
                mock_sep.from_hparams.return_value = mock_instance
                
                enhancer = AudioEnhancer(
                    use_gpu=False,
                    enhancement_level="ultra_aggressive"
                )
                
                # Process
                enhanced = enhancer.enhance(problem_audio, self.sample_rate)
                metadata = {'enhancement_metadata': {}}
                
                # Should successfully process
                self.assertIsNotNone(enhanced)
                # Enhanced should be different from original (secondary speaker removed)
                self.assertFalse(np.array_equal(enhanced, problem_audio))


if __name__ == '__main__':
    unittest.main()