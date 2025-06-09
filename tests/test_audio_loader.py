"""
Test suite for enhanced audio loader and preprocessor following TDD approach.
"""

import unittest
import tempfile
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

from processors.audio_enhancement.audio_loader import (
    AudioLoader,
    AudioValidator,
    AudioPreprocessor,
    AudioCache,
    UnsupportedFormatError,
    CorruptedFileError,
    PreprocessingError,
    AudioLoadError
)


class TestAudioLoader(unittest.TestCase):
    """Test audio loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = AudioLoader()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_audio(self, duration=1.0, sample_rate=16000, channels=1, format='WAV'):
        """Create a test audio file."""
        samples = int(duration * sample_rate)
        if channels == 1:
            audio = np.sin(2 * np.pi * 440 * np.arange(samples) / sample_rate)
        else:
            audio = np.tile(
                np.sin(2 * np.pi * 440 * np.arange(samples) / sample_rate),
                (channels, 1)
            ).T
            
        filepath = os.path.join(self.temp_dir, f"test.{format.lower()}")
        sf.write(filepath, audio, sample_rate, format=format)
        return filepath
        
    def test_load_wav_file(self):
        """Test loading standard WAV files."""
        # Test mono WAV
        filepath = self.create_test_audio(duration=1.0, channels=1, format='WAV')
        audio, sr = self.loader.load_audio(filepath)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(sr, 16000)
        self.assertEqual(audio.ndim, 1)  # Mono
        
        # Test stereo WAV
        filepath = self.create_test_audio(duration=1.0, channels=2, format='WAV')
        audio, sr = self.loader.load_audio(filepath)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(sr, 16000)
        
    def test_load_compressed_formats(self):
        """Test loading compressed audio formats."""
        # Test FLAC
        filepath = self.create_test_audio(duration=1.0, format='FLAC')
        audio, sr = self.loader.load_audio(filepath)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(sr, 16000)
        
        # Test OGG (if supported)
        try:
            filepath = self.create_test_audio(duration=1.0, format='OGG')
            audio, sr = self.loader.load_audio(filepath)
            self.assertIsInstance(audio, np.ndarray)
        except Exception:
            # OGG might not be supported on all systems
            pass
            
    def test_format_detection(self):
        """Test automatic format detection."""
        # Create WAV file with wrong extension
        filepath = self.create_test_audio(format='WAV')
        wrong_path = os.path.join(self.temp_dir, "test.mp3")
        os.rename(filepath, wrong_path)
        
        # Should still load correctly
        audio, sr = self.loader.load_audio(wrong_path)
        self.assertIsInstance(audio, np.ndarray)
        
    def test_unsupported_format_error(self):
        """Test handling of unsupported formats."""
        # Create a non-audio file
        filepath = os.path.join(self.temp_dir, "test.txt")
        with open(filepath, 'w') as f:
            f.write("This is not an audio file")
            
        with self.assertRaises(UnsupportedFormatError):
            self.loader.load_audio(filepath)
            
    def test_corrupted_file_handling(self):
        """Test handling of corrupted audio files."""
        # Create a corrupted WAV file
        filepath = os.path.join(self.temp_dir, "corrupted.wav")
        with open(filepath, 'wb') as f:
            f.write(b'RIFF' + b'\x00' * 100)  # Invalid WAV header
            
        with self.assertRaises(CorruptedFileError):
            self.loader.load_audio(filepath)
            
    def test_large_file_streaming(self):
        """Test streaming for large files."""
        # Create a large audio file (simulated)
        large_filepath = self.create_test_audio(duration=3600.0)  # 1 hour
        
        # Should use streaming loader
        audio_generator = self.loader.load_audio_streaming(large_filepath, chunk_size=16000)
        
        chunk_count = 0
        for chunk in audio_generator:
            self.assertIsInstance(chunk, np.ndarray)
            chunk_count += 1
            
        self.assertGreater(chunk_count, 1)  # Should have multiple chunks
        
    def test_metadata_extraction(self):
        """Test audio metadata extraction."""
        filepath = self.create_test_audio(duration=2.5, sample_rate=44100, channels=2)
        
        metadata = self.loader.get_metadata(filepath)
        
        self.assertEqual(metadata['duration'], 2.5)
        self.assertEqual(metadata['sample_rate'], 44100)
        self.assertEqual(metadata['channels'], 2)
        self.assertEqual(metadata['format'], 'wav')
        

class TestAudioPreprocessor(unittest.TestCase):
    """Test audio preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = AudioPreprocessor()
        
    def test_sample_rate_conversion(self):
        """Test sample rate conversion."""
        # Test upsampling
        audio = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
        converted = self.preprocessor.convert_sample_rate(audio, 8000, 16000)
        
        self.assertEqual(len(converted), 16000)
        
        # Test downsampling
        audio = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
        converted = self.preprocessor.convert_sample_rate(audio, 48000, 16000)
        
        self.assertEqual(len(converted), 16000)
        
    def test_channel_normalization(self):
        """Test channel normalization."""
        # Test stereo to mono
        stereo = np.random.randn(16000, 2)
        mono = self.preprocessor.normalize_channels(stereo, target_channels=1)
        
        self.assertEqual(mono.ndim, 1)
        self.assertEqual(len(mono), 16000)
        
        # Test multi-channel to mono
        multi = np.random.randn(16000, 5)
        mono = self.preprocessor.normalize_channels(multi, target_channels=1)
        
        self.assertEqual(mono.ndim, 1)
        
    def test_amplitude_normalization(self):
        """Test amplitude normalization."""
        # Create audio with low amplitude
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.1
        
        # Peak normalization
        normalized = self.preprocessor.normalize_amplitude(audio, method='peak')
        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0, places=3)
        
        # RMS normalization
        normalized = self.preprocessor.normalize_amplitude(audio, method='rms', target_rms=0.1)
        rms = np.sqrt(np.mean(normalized**2))
        self.assertAlmostEqual(rms, 0.1, places=2)
        
    def test_silence_trimming(self):
        """Test silence trimming."""
        # Create audio with silence at beginning and end
        signal = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)
        silence = np.zeros(8000)
        audio = np.concatenate([silence, signal, silence])
        
        trimmed = self.preprocessor.trim_silence(audio, 16000)
        
        # Should be significantly shorter
        self.assertLess(len(trimmed), len(audio) * 0.7)
        
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create test audio with non-standard format
        audio = np.random.randn(24000, 2) * 0.1  # 24kHz stereo, low volume
        
        processed = self.preprocessor.process(
            audio,
            source_sr=24000,
            target_sr=16000,
            target_channels=1,
            normalize=True,
            trim_silence=True
        )
        
        # Check all transformations applied
        self.assertEqual(processed.ndim, 1)  # Mono
        self.assertLessEqual(len(processed), 16000 * 1.5)  # Roughly correct length
        self.assertGreater(np.max(np.abs(processed)), 0.5)  # Normalized
        

class TestAudioValidator(unittest.TestCase):
    """Test audio validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = AudioValidator()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_duration_validation(self):
        """Test audio duration validation."""
        # Valid duration
        audio = np.zeros(16000)  # 1 second at 16kHz
        self.assertTrue(self.validator.validate_duration(audio, 16000))
        
        # Too short
        audio = np.zeros(160)  # 0.01 seconds
        self.assertFalse(self.validator.validate_duration(audio, 16000, min_duration=0.1))
        
        # Too long
        audio = np.zeros(16000 * 3700)  # > 1 hour
        self.assertFalse(self.validator.validate_duration(audio, 16000, max_duration=3600))
        
    def test_amplitude_validation(self):
        """Test amplitude validation."""
        # Valid amplitude
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.5
        self.assertTrue(self.validator.validate_amplitude(audio))
        
        # All zeros (silent)
        audio = np.zeros(16000)
        self.assertFalse(self.validator.validate_amplitude(audio))
        
        # Clipping
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 2.0
        audio = np.clip(audio, -1.0, 1.0)
        self.assertFalse(self.validator.validate_amplitude(audio, check_clipping=True))
        
    def test_format_validation(self):
        """Test file format validation."""
        # Valid formats
        for ext in ['wav', 'flac', 'mp3', 'ogg']:
            filepath = os.path.join(self.temp_dir, f"test.{ext}")
            self.assertTrue(self.validator.validate_format(filepath))
            
        # Invalid format
        filepath = os.path.join(self.temp_dir, "test.xyz")
        self.assertFalse(self.validator.validate_format(filepath))
        
    def test_corruption_detection(self):
        """Test corruption detection."""
        # Valid audio
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
        self.assertFalse(self.validator.check_corruption(audio))
        
        # NaN values
        audio = np.ones(16000)
        audio[1000:1100] = np.nan
        self.assertTrue(self.validator.check_corruption(audio))
        
        # Inf values
        audio = np.ones(16000)
        audio[1000] = np.inf
        self.assertTrue(self.validator.check_corruption(audio))
        

class TestAudioCache(unittest.TestCase):
    """Test audio caching functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = AudioCache(max_size=100)  # Small cache for testing
        
    def test_cache_operations(self):
        """Test basic cache operations."""
        # Test set and get
        key = "test_audio.wav"
        audio = np.random.randn(16000)
        sr = 16000
        
        self.cache.set(key, (audio, sr))
        cached = self.cache.get(key)
        
        self.assertIsNotNone(cached)
        np.testing.assert_array_equal(cached[0], audio)
        self.assertEqual(cached[1], sr)
        
        # Test cache miss
        self.assertIsNone(self.cache.get("nonexistent.wav"))
        
    def test_lru_eviction(self):
        """Test LRU cache eviction."""
        # Fill cache
        for i in range(150):  # More than max_size
            key = f"audio_{i}.wav"
            audio = np.random.randn(1000)
            self.cache.set(key, (audio, 16000))
            
        # Early entries should be evicted
        self.assertIsNone(self.cache.get("audio_0.wav"))
        
        # Recent entries should still be cached
        self.assertIsNotNone(self.cache.get("audio_149.wav"))
        
    def test_cache_warming(self):
        """Test cache warming functionality."""
        # Create test files
        temp_dir = tempfile.mkdtemp()
        filepaths = []
        
        for i in range(5):
            audio = np.random.randn(1000)
            filepath = os.path.join(temp_dir, f"audio_{i}.wav")
            sf.write(filepath, audio, 16000)
            filepaths.append(filepath)
            
        # Warm cache
        loader = AudioLoader(cache=self.cache)
        loader.warm_cache(filepaths)
        
        # All files should be cached
        for filepath in filepaths:
            self.assertIsNotNone(self.cache.get(filepath))
            
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        

class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = AudioLoader()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_parallel_loading(self):
        """Test parallel batch loading."""
        # Create multiple test files
        filepaths = []
        for i in range(10):
            audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
            filepath = os.path.join(self.temp_dir, f"audio_{i}.wav")
            sf.write(filepath, audio, 16000)
            filepaths.append(filepath)
            
        # Load in parallel
        results = self.loader.load_batch(filepaths, num_workers=4)
        
        self.assertEqual(len(results), 10)
        for audio, sr in results:
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(sr, 16000)
            
    def test_memory_efficient_loading(self):
        """Test memory-efficient loading for large files."""
        # Create a large file (simulated)
        large_audio = np.random.randn(16000 * 600)  # 10 minutes
        filepath = os.path.join(self.temp_dir, "large.wav")
        sf.write(filepath, large_audio, 16000)
        
        # Test streaming
        total_samples = 0
        for chunk in self.loader.load_audio_streaming(filepath, chunk_size=16000):
            self.assertEqual(len(chunk), 16000)  # Except possibly last chunk
            total_samples += len(chunk)
            
        # Should process entire file
        self.assertGreater(total_samples, len(large_audio) * 0.99)
        

class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = AudioLoader()
        
    def test_fallback_loading_chain(self):
        """Test fallback loading chain for problematic files."""
        # Mock different loaders
        with patch.object(self.loader, '_load_with_librosa', side_effect=Exception("Librosa failed")):
            with patch.object(self.loader, '_load_with_soundfile', side_effect=Exception("Soundfile failed")):
                with patch.object(self.loader, '_load_with_ffmpeg', return_value=(np.zeros(16000), 16000)):
                    # Should fall back to ffmpeg
                    audio, sr = self.loader.load_audio("test.wav")
                    self.assertIsInstance(audio, np.ndarray)
                    
    def test_partial_corruption_recovery(self):
        """Test recovery from partially corrupted files."""
        # This would require more complex mocking or actual corrupted test files
        pass
        
    def test_error_logging(self):
        """Test that errors are properly logged."""
        with patch('logging.Logger.error') as mock_logger:
            try:
                self.loader.load_audio("nonexistent.wav")
            except AudioLoadError:
                pass
                
            # Should have logged the error
            mock_logger.assert_called()


if __name__ == '__main__':
    unittest.main()