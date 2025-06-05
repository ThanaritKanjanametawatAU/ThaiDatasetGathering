#!/usr/bin/env python3
"""
Test-Driven Development: Comprehensive tests for audio enhancement pipeline integration.
These tests will help us understand why secondary speaker removal isn't working in production.
"""

import unittest
import numpy as np
import tempfile
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.gigaspeech2 import GigaSpeech2Processor
from processors.base_processor import BaseProcessor
from processors.audio_enhancement.core import AudioEnhancer
from utils.audio import standardize_audio
import soundfile as sf
import io


class TestEnhancementPipelineIntegration(unittest.TestCase):
    """Test suite to verify audio enhancement is properly integrated in the pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 3.0
        
        # Create test audio with secondary speaker
        self.test_audio = self._create_test_audio_with_secondary_speaker()
        
    def _create_test_audio_with_secondary_speaker(self):
        """Create test audio that simulates primary and secondary speakers."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Primary speaker (first 2 seconds)
        primary = np.sin(2 * np.pi * 200 * t[:int(2 * self.sample_rate)])
        
        # Secondary speaker (overlapping from 1.5 to 2.5 seconds)
        secondary_start = int(1.5 * self.sample_rate)
        secondary_end = int(2.5 * self.sample_rate)
        secondary = np.sin(2 * np.pi * 400 * t[secondary_start:secondary_end]) * 0.7
        
        # Combine
        audio = np.zeros_like(t)
        audio[:int(2 * self.sample_rate)] = primary
        audio[secondary_start:secondary_end] += secondary
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def test_enhancement_config_flows_through_pipeline(self):
        """Test that enhancement configuration properly flows from config to processor."""
        config = {
            "name": "GigaSpeech2",
            "source": "speechcolab/gigaspeech2",
            "streaming": True,
            "audio_enhancement": {
                "enabled": True,
                "level": "aggressive"
            }
        }
        
        processor = GigaSpeech2Processor(config)
        
        # Verify enhancement is enabled
        self.assertTrue(hasattr(processor, 'audio_enhancement'), 
                       "Processor should have audio_enhancement attribute")
        self.assertTrue(processor.audio_enhancement.get('enabled', False),
                       "Audio enhancement should be enabled")
        self.assertEqual(processor.audio_enhancement.get('level'), 'aggressive',
                        "Enhancement level should be aggressive")
    
    def test_preprocess_audio_applies_enhancement(self):
        """Test that preprocess_audio actually applies enhancement when enabled."""
        config = {
            "name": "TestProcessor",
            "audio_enhancement": {
                "enabled": True,
                "enhancer": AudioEnhancer(use_gpu=False, enhancement_level='aggressive')
            },
            "noise_reduction_enabled": True  # This triggers enhancement
        }
        
        # Create a minimal processor instance
        processor = BaseProcessor(config)
        
        # Convert test audio to bytes
        buffer = io.BytesIO()
        sf.write(buffer, self.test_audio, self.sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Apply preprocessing (which should include enhancement)
        enhanced_bytes, metadata = processor.preprocess_audio(audio_bytes, "test_sample")
        
        # Verify enhancement was applied
        self.assertIsNotNone(enhanced_bytes, "Enhanced audio should not be None")
        
        # Convert back to numpy for comparison
        buffer = io.BytesIO(enhanced_bytes)
        enhanced_array, _ = sf.read(buffer)
        
        # Check that audio was modified (not identical to original)
        original_power = np.mean(self.test_audio ** 2)
        enhanced_power = np.mean(enhanced_array ** 2)
        
        # Enhancement should reduce overall power if secondary speaker is removed
        self.assertNotAlmostEqual(original_power, enhanced_power, places=5,
                                msg="Enhanced audio should be different from original")
    
    def test_create_hf_audio_format_uses_enhanced_audio(self):
        """Test that create_hf_audio_format preserves enhanced audio."""
        processor = BaseProcessor({"name": "TestProcessor"})
        
        # Create enhanced audio (simulated by reducing volume)
        enhanced_audio = self.test_audio * 0.5
        
        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, enhanced_audio, self.sample_rate, format='WAV')
        buffer.seek(0)
        enhanced_bytes = buffer.read()
        
        # Create HF format
        hf_format = processor.create_hf_audio_format(enhanced_bytes, "test_sample")
        
        # Verify the audio array matches enhanced audio
        self.assertTrue(np.allclose(hf_format['array'], enhanced_audio, rtol=1e-4),
                       "HF format should contain enhanced audio array")
        self.assertEqual(hf_format['sampling_rate'], self.sample_rate,
                        "Sample rate should be preserved")
    
    def test_streaming_pipeline_applies_enhancement(self):
        """Test that streaming mode properly applies audio enhancement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "name": "GigaSpeech2",
                "source": "speechcolab/gigaspeech2",
                "cache_dir": tmpdir,
                "streaming": True,
                "audio_enhancement": {
                    "enabled": True,
                    "level": "aggressive",
                    "enhancer": AudioEnhancer(use_gpu=False, enhancement_level='aggressive')
                },
                "noise_reduction_enabled": True,
                "audio_config": {
                    "enable_standardization": True
                }
            }
            
            processor = GigaSpeech2Processor(config)
            
            # Mock the dataset to return our test audio
            mock_sample = {
                'audio': {
                    'array': self.test_audio,
                    'sampling_rate': self.sample_rate
                },
                'sentence': 'Test transcript'
            }
            
            with patch('datasets.load_dataset') as mock_load:
                mock_dataset = MagicMock()
                mock_dataset.__iter__ = MagicMock(return_value=iter([mock_sample]))
                mock_load.return_value = mock_dataset
                
                # Process one sample
                samples = list(processor.process_streaming(sample_mode=True, sample_size=1))
                
                self.assertEqual(len(samples), 1, "Should process one sample")
                
                # Verify audio was enhanced
                processed_audio = samples[0]['audio']['array']
                
                # Check that audio is different from original
                self.assertFalse(np.array_equal(processed_audio, self.test_audio),
                               "Processed audio should be different from original")
    
    def test_audio_enhancement_in_full_pipeline(self):
        """Test the complete pipeline from input to output with enhancement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create full config with enhancement
            config = {
                "name": "GigaSpeech2",
                "source": "speechcolab/gigaspeech2",
                "cache_dir": tmpdir,
                "streaming": True,
                "checkpoint_dir": tmpdir,
                "audio_enhancement": {
                    "enabled": True,
                    "level": "aggressive",
                    "enhancer": AudioEnhancer(use_gpu=False, enhancement_level='aggressive')
                },
                "noise_reduction_enabled": True,
                "enable_audio_enhancement": True,
                "enhancement_level": "aggressive"
            }
            
            processor = GigaSpeech2Processor(config)
            
            # Verify processor has enhancement enabled
            self.assertTrue(processor.noise_reduction_enabled,
                          "Noise reduction should be enabled")
            self.assertIsNotNone(processor.audio_enhancer,
                           "Audio enhancer should be initialized")
            
            # Mock dataset
            mock_sample = {
                'audio': {
                    'array': self.test_audio,
                    'sampling_rate': self.sample_rate
                },
                'sentence': 'Test with secondary speaker'
            }
            
            with patch('datasets.load_dataset') as mock_load:
                mock_dataset = MagicMock()
                mock_dataset.__iter__ = MagicMock(return_value=iter([mock_sample]))
                mock_load.return_value = mock_dataset
                
                # Process sample
                samples = list(processor.process_streaming(sample_mode=True, sample_size=1))
                
                # Get processed audio
                processed_audio = samples[0]['audio']['array']
                
                # Measure power reduction (should be significant if secondary speaker removed)
                original_power = np.mean(self.test_audio ** 2)
                processed_power = np.mean(processed_audio ** 2)
                
                if original_power > 0:
                    reduction_db = 10 * np.log10(processed_power / original_power)
                    
                    # Should have at least 3dB reduction if enhancement worked
                    self.assertLess(reduction_db, -3.0,
                                  f"Audio should be reduced by at least 3dB, got {reduction_db:.1f}dB")
    
    def test_enhancement_not_bypassed_in_streaming(self):
        """Test that streaming mode doesn't bypass audio enhancement."""
        # This test verifies that _process_audio_for_streaming includes enhancement
        processor = BaseProcessor({
            "name": "TestProcessor",
            "audio_config": {"enable_standardization": True},
            "noise_reduction_enabled": True,
            "audio_enhancement": {
                "enabled": True,
                "enhancer": AudioEnhancer(use_gpu=False, enhancement_level='aggressive')
            }
        })
        
        # Convert test audio to bytes
        buffer = io.BytesIO()
        sf.write(buffer, self.test_audio, self.sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Process audio for streaming
        hf_audio = processor._process_audio_for_streaming(audio_bytes, "test_sample")
        
        self.assertIsNotNone(hf_audio, "Should return HF audio format")
        
        # Check that audio was enhanced (different from original)
        processed_array = hf_audio['array']
        self.assertFalse(np.array_equal(processed_array, self.test_audio),
                        "Audio should be enhanced, not original")
    
    def test_verify_enhancement_actually_removes_secondary_speaker(self):
        """Test that enhancement actually removes the secondary speaker frequency."""
        enhancer = AudioEnhancer(use_gpu=False, enhancement_level='aggressive')
        
        # Apply enhancement
        enhanced_audio, metadata = enhancer.enhance(
            self.test_audio, 
            self.sample_rate,
            noise_level='aggressive',
            return_metadata=True
        )
        
        # Check metadata
        self.assertTrue(metadata.get('secondary_speaker_detected', False),
                       "Should detect secondary speaker")
        
        # Analyze frequency content
        # The secondary speaker is at 400Hz, primary at 200Hz
        fft_original = np.fft.rfft(self.test_audio)
        fft_enhanced = np.fft.rfft(enhanced_audio)
        
        freqs = np.fft.rfftfreq(len(self.test_audio), 1/self.sample_rate)
        
        # Find power at 400Hz (secondary speaker)
        idx_400 = np.argmin(np.abs(freqs - 400))
        power_400_original = np.abs(fft_original[idx_400])
        power_400_enhanced = np.abs(fft_enhanced[idx_400])
        
        # Secondary speaker frequency should be significantly reduced
        reduction_ratio = power_400_enhanced / (power_400_original + 1e-10)
        self.assertLess(reduction_ratio, 0.1,
                       f"400Hz (secondary speaker) should be reduced by >90%, got {reduction_ratio:.1%}")


if __name__ == '__main__':
    unittest.main()