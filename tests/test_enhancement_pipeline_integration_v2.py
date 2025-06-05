#!/usr/bin/env python3
"""
Test-Driven Development: Updated tests with concrete processor implementation.
"""

import unittest
import numpy as np
import tempfile
import json
import os
import sys
from unittest.mock import patch, MagicMock
from typing import Iterator, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.gigaspeech2 import GigaSpeech2Processor
from processors.base_processor import BaseProcessor
from processors.audio_enhancement.core import AudioEnhancer
import soundfile as sf
import io


class TestProcessor(BaseProcessor):
    """Concrete test processor for testing base functionality."""
    
    def process(self, checkpoint=None, sample_mode=False, sample_size=5):
        """Dummy implementation."""
        return iter([])
    
    def get_dataset_info(self):
        """Dummy implementation."""
        return {"name": "test"}
    
    def estimate_size(self):
        """Dummy implementation."""
        return 100
    
    def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
        """Dummy implementation."""
        return iter([])


class TestEnhancementPipelineIntegrationV2(unittest.TestCase):
    """Updated test suite to find where enhancement is failing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 3.0
        self.test_audio = self._create_test_audio_with_secondary_speaker()
        
    def _create_test_audio_with_secondary_speaker(self):
        """Create test audio that simulates primary and secondary speakers."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Primary speaker (200Hz tone)
        primary = np.sin(2 * np.pi * 200 * t) * 0.8
        
        # Secondary speaker (400Hz tone, overlapping from 1s to 2s)
        secondary = np.zeros_like(t)
        secondary_start = int(1.0 * self.sample_rate)
        secondary_end = int(2.0 * self.sample_rate)
        secondary[secondary_start:secondary_end] = np.sin(2 * np.pi * 400 * t[secondary_start:secondary_end]) * 0.6
        
        # Combine
        audio = primary + secondary
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def test_processor_enhancement_initialization(self):
        """Test that processor correctly initializes enhancement."""
        config = {
            "name": "TestProcessor",
            "noise_reduction_enabled": True,
            "audio_enhancement": {
                "enabled": True,
                "level": "aggressive"
            }
        }
        
        processor = TestProcessor(config)
        
        # Check initialization
        self.assertTrue(processor.noise_reduction_enabled)
        self.assertIsNotNone(processor.audio_enhancer)
        self.assertEqual(processor.enhancement_level, "aggressive")
    
    def test_preprocess_audio_returns_enhanced(self):
        """Test that preprocess_audio returns enhanced audio."""
        # Create processor with enhancement
        processor = TestProcessor({
            "name": "TestProcessor",
            "noise_reduction_enabled": True,
            "audio_enhancement": {
                "enabled": True,
                "level": "aggressive",
                "enhancer": AudioEnhancer(use_gpu=False, enhancement_level='aggressive')
            }
        })
        
        # Convert test audio to bytes
        buffer = io.BytesIO()
        sf.write(buffer, self.test_audio, self.sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Apply preprocessing
        enhanced_bytes, metadata = processor.preprocess_audio(audio_bytes, "test_sample")
        
        # Convert back to array
        buffer = io.BytesIO(enhanced_bytes)
        enhanced_array, _ = sf.read(buffer)
        
        # Verify it's different
        self.assertFalse(np.array_equal(enhanced_array, self.test_audio))
        
        # Check metadata indicates enhancement
        if metadata:
            self.assertIn('secondary_speaker_detected', metadata)
    
    def test_create_hf_format_preserves_array(self):
        """Test that create_hf_audio_format preserves the audio array."""
        processor = TestProcessor({"name": "TestProcessor"})
        
        # Create test audio bytes
        buffer = io.BytesIO()
        sf.write(buffer, self.test_audio, self.sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Create HF format
        hf_format = processor.create_hf_audio_format(audio_bytes, "test")
        
        # Verify array is preserved
        self.assertTrue(np.allclose(hf_format['array'], self.test_audio, rtol=1e-4))
    
    def test_streaming_audio_processing_chain(self):
        """Test the audio processing chain in streaming mode."""
        # Create processor with enhancement
        processor = TestProcessor({
            "name": "TestProcessor",
            "noise_reduction_enabled": True,
            "audio_config": {"enable_standardization": True},
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
        
        # Process through streaming pipeline
        hf_audio = processor._process_audio_for_streaming(audio_bytes, "test")
        
        # Check that audio was processed
        self.assertIsNotNone(hf_audio)
        processed_array = hf_audio['array']
        
        # Should be different if enhancement was applied
        self.assertFalse(np.array_equal(processed_array, self.test_audio),
                        "Audio should be enhanced")
    
    def test_enhancement_strength_verification(self):
        """Test that enhancement is actually strong enough to remove secondary speaker."""
        enhancer = AudioEnhancer(use_gpu=False, enhancement_level='aggressive')
        
        # Apply enhancement
        enhanced_audio, metadata = enhancer.enhance(
            self.test_audio, 
            self.sample_rate,
            noise_level='aggressive',
            return_metadata=True
        )
        
        # Calculate power in secondary speaker frequency band (around 400Hz)
        def get_frequency_power(audio, target_freq, sample_rate):
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
            
            # Find indices around target frequency (±10Hz)
            idx_min = np.argmin(np.abs(freqs - (target_freq - 10)))
            idx_max = np.argmin(np.abs(freqs - (target_freq + 10)))
            
            # Calculate power in that band
            power = np.sum(np.abs(fft[idx_min:idx_max+1])**2)
            return power
        
        # Get power at 400Hz (secondary speaker)
        power_400_original = get_frequency_power(self.test_audio, 400, self.sample_rate)
        power_400_enhanced = get_frequency_power(enhanced_audio, 400, self.sample_rate)
        
        # Calculate reduction
        if power_400_original > 0:
            reduction_db = 10 * np.log10(power_400_enhanced / power_400_original)
        else:
            reduction_db = -60
        
        print(f"\n400Hz power reduction: {reduction_db:.1f} dB")
        print(f"Metadata: {metadata}")
        
        # Should have significant reduction (at least -20dB)
        self.assertLess(reduction_db, -20.0,
                       f"Secondary speaker (400Hz) should be reduced by >20dB, got {reduction_db:.1f}dB")
    
    def test_gigaspeech_processor_enhancement_path(self):
        """Test the actual GigaSpeech2 processor enhancement path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "name": "GigaSpeech2",
                "source": "speechcolab/gigaspeech2",
                "cache_dir": tmpdir,
                "streaming": True,
                "noise_reduction_enabled": True,
                "audio_enhancement": {
                    "enabled": True,
                    "level": "aggressive",
                    "enhancer": AudioEnhancer(use_gpu=False, enhancement_level='aggressive')
                }
            }
            
            processor = GigaSpeech2Processor(config)
            
            # Check that processor._process_audio_for_streaming calls preprocess_audio
            buffer = io.BytesIO()
            sf.write(buffer, self.test_audio, self.sample_rate, format='WAV')
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            # Spy on preprocess_audio
            original_preprocess = processor.preprocess_audio
            preprocess_called = False
            
            def mock_preprocess(audio_data, sample_id):
                nonlocal preprocess_called
                preprocess_called = True
                return original_preprocess(audio_data, sample_id)
            
            processor.preprocess_audio = mock_preprocess
            
            # Process audio
            hf_audio = processor._process_audio_for_streaming(audio_bytes, "test")
            
            # Verify preprocess was called
            self.assertTrue(preprocess_called, "preprocess_audio should be called")
            
            # Verify audio was enhanced
            self.assertFalse(np.array_equal(hf_audio['array'], self.test_audio))


if __name__ == '__main__':
    unittest.main()