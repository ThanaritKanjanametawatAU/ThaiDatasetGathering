"""Test suite for Advanced Edge Case Handler module (S05_T19).

This test suite validates the Advanced Edge Case Handling system with the following requirements:
1. Corrupted audio recovery with FFmpeg fallback
2. Multi-language detection with Thai extraction
3. Intelligent retry systems with error categorization
4. Edge case detection and handling pipeline
5. Automated error categorization with recovery strategies
"""

import unittest
import tempfile
import shutil
import os
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from processors.edge_case_handling.advanced_edge_case_handler import (
    AdvancedEdgeCaseHandler,
    ErrorCategory,
    ErrorInfo,
    RecoveryResult,
    RecoveryMethod,
    RetryStrategy,
    LanguageDetectionResult,
    FFmpegRecoveryEngine,
    LibrosaRecoveryEngine,
    LanguageDetector,
    IntelligentRetryManager
)


class TestErrorCategorization(unittest.TestCase):
    """Test suite for error categorization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retry_manager = IntelligentRetryManager()
    
    def test_network_error_categorization(self):
        """Test 1.1: Network error categorization."""
        # Test various network-related errors
        network_errors = [
            ConnectionError("Connection failed"),
            TimeoutError("Network timeout"),
            Exception("Connection refused"),
            Exception("Network unreachable")
        ]
        
        for error in network_errors:
            category = self.retry_manager.categorize_error(error)
            self.assertEqual(category, ErrorCategory.NETWORK_ERROR)
    
    def test_memory_error_categorization(self):
        """Test 1.2: Memory error categorization."""
        memory_errors = [
            MemoryError("Out of memory"),
            Exception("Memory allocation failed"),
            Exception("Cannot allocate memory")
        ]
        
        for error in memory_errors:
            category = self.retry_manager.categorize_error(error)
            self.assertEqual(category, ErrorCategory.MEMORY_ERROR)
    
    def test_audio_format_error_categorization(self):
        """Test 1.3: Audio format error categorization."""
        # Unsupported format errors
        unsupported_errors = [
            Exception("Unsupported format"),
            Exception("Unknown codec"),
            Exception("Invalid file format")
        ]
        
        for error in unsupported_errors:
            category = self.retry_manager.categorize_error(error)
            self.assertEqual(category, ErrorCategory.UNSUPPORTED_FORMAT)
        
        # Encoding errors
        encoding_errors = [
            Exception("Decoding failed"),
            Exception("Codec error"),
            Exception("Format parsing error")
        ]
        
        for error in encoding_errors:
            category = self.retry_manager.categorize_error(error)
            self.assertEqual(category, ErrorCategory.ENCODING_ERROR)
    
    def test_corrupted_audio_categorization(self):
        """Test 1.4: Corrupted audio error categorization."""
        corrupted_errors = [
            Exception("File is corrupted"),
            Exception("Invalid audio data"),
            Exception("Damaged file header"),
            Exception("Truncated file")
        ]
        
        for error in corrupted_errors:
            category = self.retry_manager.categorize_error(error)
            self.assertEqual(category, ErrorCategory.CORRUPTED_AUDIO)
    
    def test_context_based_categorization(self):
        """Test 1.5: Context-based error categorization."""
        generic_error = Exception("Processing failed")
        
        # Test with different contexts
        contexts = [
            ({"silence_detected": True}, ErrorCategory.SILENCE_DETECTED),
            ({"low_quality": True}, ErrorCategory.LOW_QUALITY),
            ({"language_mismatch": True}, ErrorCategory.LANGUAGE_MISMATCH),
            ({}, ErrorCategory.UNKNOWN_ERROR)
        ]
        
        for context, expected_category in contexts:
            category = self.retry_manager.categorize_error(generic_error, context)
            self.assertEqual(category, expected_category)


class TestRetryMechanism(unittest.TestCase):
    """Test suite for intelligent retry mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retry_manager = IntelligentRetryManager()
    
    def test_retry_decision_logic(self):
        """Test 2.1: Retry decision logic."""
        # Create error info for different categories
        error_categories = [
            (ErrorCategory.CORRUPTED_AUDIO, True),   # Should retry
            (ErrorCategory.NETWORK_ERROR, True),     # Should retry
            (ErrorCategory.LANGUAGE_MISMATCH, False), # Should not retry
            (ErrorCategory.UNKNOWN_ERROR, True)      # Should retry
        ]
        
        for category, should_retry in error_categories:
            error_info = ErrorInfo(
                error_type="TestError",
                error_message="Test error",
                category=category,
                retry_count=0
            )
            
            result = self.retry_manager.should_retry(error_info)
            self.assertEqual(result, should_retry, f"Failed for category {category}")
    
    def test_max_retry_limits(self):
        """Test 2.2: Maximum retry limits."""
        error_info = ErrorInfo(
            error_type="TestError",
            error_message="Test error",
            category=ErrorCategory.CORRUPTED_AUDIO,
            retry_count=0
        )
        
        max_retries = self.retry_manager.max_retries_per_category[ErrorCategory.CORRUPTED_AUDIO]
        
        # Should retry up to max_retries
        for i in range(max_retries):
            error_info.retry_count = i
            self.assertTrue(self.retry_manager.should_retry(error_info))
        
        # Should not retry beyond max_retries
        error_info.retry_count = max_retries
        self.assertFalse(self.retry_manager.should_retry(error_info))
    
    def test_retry_delay_calculation(self):
        """Test 2.3: Retry delay calculation."""
        # Test exponential backoff
        error_info = ErrorInfo(
            error_type="TestError",
            error_message="Test error",
            category=ErrorCategory.NETWORK_ERROR,  # Uses exponential backoff
            retry_count=0
        )
        
        delays = []
        for i in range(3):
            error_info.retry_count = i
            delay = self.retry_manager.calculate_retry_delay(error_info)
            delays.append(delay)
        
        # Exponential backoff should increase delays
        self.assertLess(delays[0], delays[1])
        self.assertLess(delays[1], delays[2])
        
        # Test immediate retry
        error_info.category = ErrorCategory.METADATA_ERROR  # Uses immediate retry
        error_info.retry_count = 0
        delay = self.retry_manager.calculate_retry_delay(error_info)
        self.assertEqual(delay, 0.0)
    
    def test_error_history_tracking(self):
        """Test 2.4: Error history tracking."""
        file_path = "/test/file.wav"
        
        # Record multiple errors
        for i in range(3):
            error_info = ErrorInfo(
                error_type="TestError",
                error_message=f"Test error {i}",
                category=ErrorCategory.CORRUPTED_AUDIO,
                file_path=file_path,
                retry_count=i
            )
            self.retry_manager.record_error(error_info)
        
        # Check history
        self.assertIn(file_path, self.retry_manager.error_history)
        self.assertEqual(len(self.retry_manager.error_history[file_path]), 3)
        
        # Test statistics
        stats = self.retry_manager.get_error_statistics()
        self.assertEqual(stats["total_files_with_errors"], 1)
        self.assertIn("corrupted_audio", stats["errors_by_category"])
        self.assertEqual(stats["errors_by_category"]["corrupted_audio"], 3)


class TestFFmpegRecoveryEngine(unittest.TestCase):
    """Test suite for FFmpeg recovery engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = FFmpegRecoveryEngine()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ffmpeg_availability_check(self):
        """Test 3.1: FFmpeg availability detection."""
        # This test may pass or fail depending on system configuration
        # We'll just verify the check doesn't crash
        ffmpeg_path = self.engine._find_ffmpeg()
        self.assertIsInstance(ffmpeg_path, (str, type(None)))
    
    def test_error_handling_capability(self):
        """Test 3.2: Error handling capability assessment."""
        # Test which errors FFmpeg can handle
        error_categories = [
            (ErrorCategory.CORRUPTED_AUDIO, True),
            (ErrorCategory.UNSUPPORTED_FORMAT, True),
            (ErrorCategory.ENCODING_ERROR, True),
            (ErrorCategory.NETWORK_ERROR, False),
            (ErrorCategory.MEMORY_ERROR, False)
        ]
        
        for category, can_handle in error_categories:
            error_info = ErrorInfo(
                error_type="TestError",
                error_message="Test error",
                category=category
            )
            
            result = self.engine.can_handle(error_info)
            self.assertEqual(result, can_handle and self.engine.ffmpeg_path is not None,
                           f"Failed for category {category}")
    
    def test_duration_extraction(self):
        """Test 3.3: Duration extraction from FFmpeg output."""
        # Mock FFmpeg output
        ffmpeg_output = """
        Input #0, wav, from 'test.wav':
          Duration: 00:01:23.45, bitrate: 1411 kb/s
            Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, stereo, s16, 1411 kb/s
        """
        
        duration = self.engine._extract_duration(ffmpeg_output)
        expected_duration = 1 * 60 + 23.45  # 83.45 seconds
        self.assertAlmostEqual(duration, expected_duration, places=2)
    
    def test_quality_assessment(self):
        """Test 3.4: Audio quality assessment."""
        # Create a simple test audio file
        test_file = os.path.join(self.temp_dir, "test.wav")
        
        # Create minimal WAV file data (this won't be a real audio file)
        with open(test_file, 'wb') as f:
            f.write(b'RIFF' + b'\\x00' * 44)  # Minimal WAV header
        
        # Quality assessment should handle invalid files gracefully
        quality = self.engine._assess_quality(test_file)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)


class TestLibrosaRecoveryEngine(unittest.TestCase):
    """Test suite for Librosa recovery engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = LibrosaRecoveryEngine()
    
    def test_error_handling_capability(self):
        """Test 4.1: Librosa error handling capability."""
        error_categories = [
            (ErrorCategory.CORRUPTED_AUDIO, True),
            (ErrorCategory.LOW_QUALITY, True),
            (ErrorCategory.SILENCE_DETECTED, True),
            (ErrorCategory.NETWORK_ERROR, False),
            (ErrorCategory.UNSUPPORTED_FORMAT, False)
        ]
        
        for category, can_handle in error_categories:
            error_info = ErrorInfo(
                error_type="TestError",
                error_message="Test error",
                category=category
            )
            
            result = self.engine.can_handle(error_info)
            self.assertEqual(result, can_handle, f"Failed for category {category}")
    
    @patch('librosa.load')
    @patch('soundfile.write')
    def test_audio_processing(self, mock_sf_write, mock_librosa_load):
        """Test 4.2: Audio processing and recovery."""
        # Mock librosa.load to return test audio
        test_audio = np.random.randn(16000)  # 1 second of random audio
        mock_librosa_load.return_value = (test_audio, 16000)
        
        # Mock soundfile.write
        mock_sf_write.return_value = None
        
        processed_audio = self.engine._process_audio(test_audio, 16000)
        
        # Verify processing
        self.assertIsInstance(processed_audio, np.ndarray)
        self.assertGreater(len(processed_audio), 0)
    
    def test_quality_assessment(self):
        """Test 4.3: Audio quality assessment."""
        # Test with different audio characteristics
        test_cases = [
            (np.zeros(16000), 0.0),  # Silent audio should get low score
            (np.random.randn(16000) * 0.1, None),  # Random audio with reasonable amplitude
            (np.array([]), 0.0),  # Empty audio should get zero score
        ]
        
        for audio, expected_range in test_cases:
            quality = self.engine._assess_audio_quality(audio, 16000)
            
            self.assertIsInstance(quality, float)
            self.assertGreaterEqual(quality, 0.0)
            self.assertLessEqual(quality, 1.0)
            
            if expected_range is not None:
                self.assertEqual(quality, expected_range)


class TestLanguageDetector(unittest.TestCase):
    """Test suite for language detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = LanguageDetector()
    
    def test_thai_content_detection(self):
        """Test 5.1: Thai content detection in text."""
        test_cases = [
            ("สวัสดีครับ", "th", True),
            ("Hello world", "en", False),
            ("สวัสดี hello world", "th", True),  # Mixed content
            ("", "en", False)
        ]
        
        for text, language, expected in test_cases:
            result = self.detector._is_thai_content(text, language)
            self.assertEqual(result, expected, f"Failed for text: {text}")
    
    def test_thai_confidence_calculation(self):
        """Test 5.2: Thai confidence calculation."""
        # Mock Whisper result
        whisper_result = {
            "language": "th",
            "segments": [
                {"text": "สวัสดีครับ", "language": "th", "start": 0.0, "end": 2.0},
                {"text": "Hello", "language": "en", "start": 2.0, "end": 3.0}
            ]
        }
        
        confidence = self.detector._calculate_thai_confidence(whisper_result, 2.0, 3.0)
        
        # Should have high confidence due to Thai language detection and content
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)
    
    def test_fallback_detection(self):
        """Test 5.3: Fallback language detection."""
        # Test with Thai indicators in filename
        thai_files = [
            "/path/to/thai_audio.wav",
            "/path/to/thailand_recording.mp3",
            "/path/to/ไทย_audio.wav"
        ]
        
        for file_path in thai_files:
            result = self.detector._fallback_detection(file_path)
            
            self.assertIsInstance(result, LanguageDetectionResult)
            self.assertGreater(result.thai_confidence, 0.5)
        
        # Test with non-Thai filename
        result = self.detector._fallback_detection("/path/to/english_audio.wav")
        self.assertLessEqual(result.thai_confidence, 0.5)


class TestAdvancedEdgeCaseHandler(unittest.TestCase):
    """Test suite for the main edge case handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = AdvancedEdgeCaseHandler()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test files
        self.test_file = os.path.join(self.temp_dir, "test_audio.wav")
        with open(self.test_file, 'wb') as f:
            f.write(b'RIFF' + b'\\x00' * 44)  # Minimal WAV header
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('librosa.load')
    def test_normal_processing_success(self, mock_librosa_load):
        """Test 6.1: Successful normal processing."""
        # Mock successful audio loading
        mock_librosa_load.return_value = (np.random.randn(16000), 16000)
        
        result = self.handler._try_normal_processing(self.test_file)
        
        self.assertTrue(result["success"])
        self.assertIsNone(result.get("error_info"))
        self.assertIn("metadata", result)
    
    def test_normal_processing_failure(self):
        """Test 6.2: Normal processing failure handling."""
        # Test with non-existent file
        result = self.handler._try_normal_processing("/non/existent/file.wav")
        
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error_info"])
        self.assertEqual(result["error_info"].category, ErrorCategory.UNKNOWN_ERROR)
    
    @patch('librosa.load')
    def test_file_processing_workflow(self, mock_librosa_load):
        """Test 6.3: Complete file processing workflow."""
        # Mock successful audio loading
        mock_librosa_load.return_value = (np.random.randn(16000), 16000)
        
        result = self.handler.process_file(self.test_file)
        
        # Verify result structure
        required_keys = ["file_path", "success", "processing_time", "metadata"]
        for key in required_keys:
            self.assertIn(key, result)
        
        self.assertEqual(result["file_path"], self.test_file)
        self.assertIsInstance(result["processing_time"], float)
        self.assertGreater(result["processing_time"], 0)
    
    def test_batch_processing(self):
        """Test 6.4: Batch file processing."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"test_{i}.wav")
            with open(file_path, 'wb') as f:
                f.write(b'RIFF' + b'\\x00' * 44)
            test_files.append(file_path)
        
        results = self.handler.batch_process(test_files)
        
        self.assertEqual(len(results), len(test_files))
        for i, result in enumerate(results):
            self.assertEqual(result["file_path"], test_files[i])
            self.assertIn("success", result)
            self.assertIn("processing_time", result)
    
    def test_statistics_tracking(self):
        """Test 6.5: Processing statistics tracking."""
        initial_stats = self.handler.get_statistics()
        
        # Process a file to update statistics
        self.handler.process_file(self.test_file)
        
        updated_stats = self.handler.get_statistics()
        
        # Verify statistics were updated
        self.assertGreater(
            updated_stats["processing_stats"]["files_processed"],
            initial_stats["processing_stats"]["files_processed"]
        )
        
        self.assertGreater(
            updated_stats["processing_stats"]["processing_time_total"],
            initial_stats["processing_stats"]["processing_time_total"]
        )
    
    def test_recovery_engine_integration(self):
        """Test 6.6: Recovery engine integration."""
        # Create error info
        error_info = ErrorInfo(
            error_type="TestError",
            error_message="Test corrupted audio",
            category=ErrorCategory.CORRUPTED_AUDIO,
            file_path=self.test_file
        )
        
        # Attempt recovery
        recovery_result = self.handler._attempt_recovery(self.test_file, error_info)
        
        # Verify recovery attempt
        self.assertIsInstance(recovery_result, RecoveryResult)
        self.assertTrue(error_info.recovery_attempted)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete edge case handling system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = AdvancedEdgeCaseHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_processing(self):
        """Test 7.1: End-to-end processing workflow."""
        # Create various test files with different "issues"
        test_files = []
        
        # Normal file
        normal_file = os.path.join(self.temp_dir, "normal.wav")
        with open(normal_file, 'wb') as f:
            f.write(b'RIFF' + b'\\x00' * 100)  # Larger fake WAV
        test_files.append(normal_file)
        
        # "Corrupted" file (empty)
        corrupted_file = os.path.join(self.temp_dir, "corrupted.wav")
        with open(corrupted_file, 'wb') as f:
            pass  # Empty file
        test_files.append(corrupted_file)
        
        # Non-existent file
        test_files.append("/non/existent/file.wav")
        
        # Process all files
        results = self.handler.batch_process(test_files)
        
        # Verify results
        self.assertEqual(len(results), len(test_files))
        
        # Check that each result has required fields
        for result in results:
            self.assertIn("file_path", result)
            self.assertIn("success", result)
            self.assertIn("processing_time", result)
            self.assertIsInstance(result["processing_time"], float)
    
    def test_error_categorization_integration(self):
        """Test 7.2: Error categorization integration."""
        # Process files that will generate different error types
        test_cases = [
            ("/non/existent/file.wav", ErrorCategory.UNKNOWN_ERROR),
            # Add more test cases as needed
        ]
        
        for file_path, expected_category in test_cases:
            result = self.handler.process_file(file_path)
            
            if not result["success"] and result.get("error_info"):
                # Error categorization should work
                self.assertIsInstance(result["error_info"].category, ErrorCategory)
    
    def test_statistics_aggregation(self):
        """Test 7.3: Statistics aggregation across processing."""
        # Process multiple files
        test_files = [
            os.path.join(self.temp_dir, f"test_{i}.wav") for i in range(5)
        ]
        
        # Create files
        for file_path in test_files:
            with open(file_path, 'wb') as f:
                f.write(b'RIFF' + b'\\x00' * 50)
        
        # Process files
        self.handler.batch_process(test_files)
        
        # Check statistics
        stats = self.handler.get_statistics()
        
        # Verify comprehensive statistics
        self.assertIn("processing_stats", stats)
        self.assertIn("error_stats", stats)
        self.assertIn("recovery_success_rate", stats)
        self.assertIn("average_processing_time", stats)
        
        # Verify numbers make sense
        self.assertEqual(stats["processing_stats"]["files_processed"], len(test_files))
        self.assertGreaterEqual(stats["recovery_success_rate"], 0.0)
        self.assertLessEqual(stats["recovery_success_rate"], 1.0)
        self.assertGreater(stats["average_processing_time"], 0.0)


if __name__ == "__main__":
    unittest.main()