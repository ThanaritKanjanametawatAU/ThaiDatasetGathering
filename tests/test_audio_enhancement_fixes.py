#!/usr/bin/env python3
"""
Test-Driven Development: Comprehensive tests for audio enhancement fixes.
Tests for:
1. AudioEnhancer method name fix (enhance_audio → enhance)
2. Float16 audio type support

This test file is created BEFORE verifying the fixes to follow strict TDD methodology.
"""

import unittest
import numpy as np
import io
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.core import AudioEnhancer
from processors.base_processor import BaseProcessor


class TestAudioEnhancementMethodFix(unittest.TestCase):
    """Test that AudioEnhancer has the correct method name 'enhance'"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.duration = 1.0
        self.samples = int(self.sample_rate * self.duration)
        
    def test_audio_enhancer_has_enhance_method(self):
        """Test that AudioEnhancer has 'enhance' method, not 'enhance_audio'"""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Should have 'enhance' method
        self.assertTrue(hasattr(enhancer, 'enhance'), 
                       "AudioEnhancer should have 'enhance' method")
        
        # Should NOT have 'enhance_audio' method
        self.assertFalse(hasattr(enhancer, 'enhance_audio'),
                        "AudioEnhancer should NOT have 'enhance_audio' method")
    
    def test_enhance_method_signature(self):
        """Test that enhance method has correct signature with return_metadata parameter"""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Create test audio
        audio = np.random.randn(self.samples).astype(np.float32) * 0.1
        
        # Test with return_metadata=True
        result = enhancer.enhance(audio, self.sample_rate, return_metadata=True)
        
        # Should return tuple when return_metadata=True
        self.assertIsInstance(result, tuple, 
                            "enhance should return tuple when return_metadata=True")
        self.assertEqual(len(result), 2, 
                        "enhance should return (audio, metadata) tuple")
        
        enhanced_audio, metadata = result
        self.assertIsInstance(enhanced_audio, np.ndarray,
                            "First element should be numpy array")
        self.assertIsInstance(metadata, dict,
                            "Second element should be metadata dict")
    
    def test_enhance_method_metadata_keys(self):
        """Test that enhance method returns correct metadata keys"""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Create test audio
        audio = np.random.randn(self.samples).astype(np.float32) * 0.1
        
        # Get metadata
        _, metadata = enhancer.enhance(audio, self.sample_rate, return_metadata=True)
        
        # Check required keys
        required_keys = ['snr_before', 'snr_after', 'snr_improvement', 'processing_time']
        for key in required_keys:
            self.assertIn(key, metadata, f"Metadata should contain '{key}'")
        
        # Should NOT have old keys
        old_keys = ['original_snr', 'enhanced_snr', 'processing_time_ms']
        for key in old_keys:
            self.assertNotIn(key, metadata, f"Metadata should NOT contain old key '{key}'")
    
    def test_base_processor_calls_correct_method(self):
        """Test that BaseProcessor calls enhance method correctly"""
        # Create a concrete implementation of BaseProcessor for testing
        class TestProcessor(BaseProcessor):
            def __init__(self):
                config = {
                    "dataset_name": "test",
                    "output_dir": ".",
                    "streaming": False,
                    "enable_audio_enhancement": True,
                    "audio_enhancement": {
                        "enabled": True,
                        "enhancer": AudioEnhancer(use_gpu=False),
                        "level": "moderate"
                    }
                }
                super().__init__(config)
                
            def process(self):
                pass
                
            def _process_item(self, item):
                pass
                
            def process_streaming(self):
                pass
                
            def estimate_size(self):
                return 0
                
            def get_dataset_info(self):
                return {"description": "Test dataset"}
        
        processor = TestProcessor()
        
        # Create test audio data
        audio = np.random.randn(self.samples).astype(np.float32) * 0.1
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Test enhancement
        result = processor._apply_noise_reduction_with_metadata(audio_bytes, "test_id")
        
        # Should not raise AttributeError about enhance_audio
        self.assertIsNotNone(result, "Enhancement should return a result")
        
        # Verify result structure
        if result:
            enhanced_bytes, metadata = result
            self.assertIsInstance(enhanced_bytes, bytes, "Should return enhanced audio as bytes")
            self.assertIsInstance(metadata, dict, "Should return metadata dict")


class TestFloat16Support(unittest.TestCase):
    """Test that audio enhancement supports float16 input"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.duration = 1.0
        self.samples = int(self.sample_rate * self.duration)
        
    def test_float16_audio_conversion(self):
        """Test that float16 audio is converted to float32 before enhancement"""
        # Create float16 audio
        audio_float16 = np.random.randn(self.samples).astype(np.float16) * 0.1
        
        # Write to buffer as float16
        buffer = io.BytesIO()
        # Note: soundfile doesn't directly support float16, so we simulate it
        # by saving as float32 then converting when reading
        sf.write(buffer, audio_float16.astype(np.float32), self.sample_rate, format='WAV')
        buffer.seek(0)
        
        # Create processor
        class TestProcessor(BaseProcessor):
            def __init__(self):
                config = {
                    "dataset_name": "test",
                    "output_dir": ".",
                    "streaming": False,
                    "enable_audio_enhancement": True,
                    "audio_enhancement": {
                        "enabled": True,
                        "enhancer": AudioEnhancer(use_gpu=False),
                        "level": "moderate"
                    }
                }
                super().__init__(config)
                
            def process(self):
                pass
                
            def _process_item(self, item):
                pass
                
            def process_streaming(self):
                pass
                
            def estimate_size(self):
                return 0
                
            def get_dataset_info(self):
                return {"description": "Test dataset"}
        
        processor = TestProcessor()
        
        # Mock the audio array to be float16
        original_read = sf.read
        def mock_read(buffer):
            audio, sr = original_read(buffer)
            # Force to float16 to simulate the issue
            return audio.astype(np.float16), sr
        
        with patch('soundfile.read', side_effect=mock_read):
            # This should not raise dtype error
            result = processor._apply_noise_reduction_with_metadata(buffer.read(), "test_float16")
            
            self.assertIsNotNone(result, 
                               "Should successfully enhance float16 audio")
    
    def test_enhancer_handles_float32(self):
        """Test that AudioEnhancer correctly handles float32 input"""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Create float32 audio
        audio_float32 = np.random.randn(self.samples).astype(np.float32) * 0.1
        
        # This should work without errors
        enhanced, metadata = enhancer.enhance(audio_float32, self.sample_rate, return_metadata=True)
        
        self.assertEqual(enhanced.dtype, np.float32,
                        "Output should maintain float32 dtype")
        self.assertIsInstance(metadata, dict,
                        "Should return metadata dict")
    
    def test_enhancer_handles_float64(self):
        """Test that AudioEnhancer correctly handles float64 input"""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Create float64 audio
        audio_float64 = np.random.randn(self.samples).astype(np.float64) * 0.1
        
        # This should work without errors
        enhanced, metadata = enhancer.enhance(audio_float64, self.sample_rate, return_metadata=True)
        
        self.assertIn(enhanced.dtype, [np.float32, np.float64],
                     "Output should be float32 or float64")
        self.assertIsInstance(metadata, dict,
                        "Should return metadata dict")
    
    def test_enhancer_rejects_float16_directly(self):
        """Test that AudioEnhancer cannot handle float16 directly"""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Create float16 audio
        audio_float16 = np.random.randn(self.samples).astype(np.float16) * 0.1
        
        # This should raise an error or convert internally
        # We test that it either handles it or raises a clear error
        try:
            enhanced, metadata = enhancer.enhance(audio_float16, self.sample_rate, return_metadata=True)
            # If it succeeds, the dtype should have been converted
            self.assertIn(enhanced.dtype, [np.float32, np.float64],
                         "If float16 is accepted, it should be converted to float32/64")
        except Exception as e:
            # If it fails, it should be a dtype-related error
            self.assertIn('dtype', str(e).lower(),
                         "Error should mention dtype if float16 is not supported")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete enhancement pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.duration = 2.0
        self.samples = int(self.sample_rate * self.duration)
        
    def test_complete_enhancement_pipeline_with_float16(self):
        """Test the complete pipeline with float16 input"""
        # Create test processor
        class TestProcessor(BaseProcessor):
            def __init__(self):
                config = {
                    "dataset_name": "test",
                    "output_dir": ".",
                    "streaming": False,
                    "enable_audio_enhancement": True,
                    "enhancement_level": "moderate",
                    "audio_enhancement": {
                        "enabled": True,
                        "enhancer": AudioEnhancer(use_gpu=False, enhancement_level="moderate"),
                        "level": "moderate"
                    }
                }
                super().__init__(config)
                
            def process(self):
                pass
                
            def _process_item(self, item):
                pass
                
            def process_streaming(self):
                pass
                
            def estimate_size(self):
                return 0
                
            def get_dataset_info(self):
                return {"description": "Test dataset"}
        
        processor = TestProcessor()
        
        # Create float16 audio
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, self.samples))
        audio = (audio * 0.3).astype(np.float16)
        
        # Add some noise
        noise = np.random.randn(self.samples).astype(np.float16) * 0.05
        noisy_audio = audio + noise
        
        # Convert to bytes
        buffer = io.BytesIO()
        # Simulate float16 storage by converting to float32 for soundfile
        sf.write(buffer, noisy_audio.astype(np.float32), self.sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Mock sf.read to return float16
        original_read = sf.read
        def mock_read(buffer):
            audio, sr = original_read(buffer)
            return audio.astype(np.float16), sr
        
        with patch('soundfile.read', side_effect=mock_read):
            # Process audio
            result = processor._apply_noise_reduction_with_metadata(audio_bytes, "test_integration")
            
            self.assertIsNotNone(result, "Enhancement should succeed")
            
            if result:
                enhanced_bytes, metadata = result
                
                # Verify enhanced audio
                self.assertIsInstance(enhanced_bytes, bytes,
                                    "Should return bytes")
                self.assertIsInstance(metadata, dict,
                                    "Should return metadata dict")
                
                # Check metadata
                self.assertIn('snr_improvement', metadata,
                             "Metadata should contain snr_improvement")
                self.assertGreater(metadata.get('snr_improvement', 0), 0,
                                 "Should show SNR improvement")
    
    def test_error_handling_and_logging(self):
        """Test that errors are properly logged and handled"""
        # Create test processor
        class TestProcessor(BaseProcessor):
            def __init__(self):
                config = {
                    "dataset_name": "test",
                    "output_dir": ".",
                    "streaming": False,
                    "enable_audio_enhancement": True,
                    "audio_enhancement": {
                        "enabled": True,
                        "enhancer": AudioEnhancer(use_gpu=False),
                        "level": "moderate"
                    }
                }
                super().__init__(config)
                self.enhancement_stats = {
                    'enhancement_failures': 0
                }
                
            def process(self):
                pass
                
            def _process_item(self, item):
                pass
                
            def process_streaming(self):
                pass
                
            def estimate_size(self):
                return 0
                
            def get_dataset_info(self):
                return {"description": "Test dataset"}
        
        processor = TestProcessor()
        
        # Test with invalid audio data
        invalid_audio = b"not valid audio data"
        
        # Should handle error gracefully
        result = processor._apply_noise_reduction_with_metadata(invalid_audio, "test_error")
        
        self.assertIsNone(result, "Should return None for invalid audio")
        self.assertEqual(processor.enhancement_stats['enhancement_failures'], 1,
                        "Should increment failure counter")


if __name__ == '__main__':
    unittest.main(verbosity=2)