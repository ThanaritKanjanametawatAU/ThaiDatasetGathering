#!/usr/bin/env python3
"""
Comprehensive edge case tests for SpeechBrain implementation.

Tests for potential bugs, edge cases, and error scenarios not covered
by existing test suites.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import gc
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.speechbrain_separator import (
    SpeechBrainSeparator,
    SeparationConfig,
    SeparationOutput,
    GPUMemoryManager
)
from processors.audio_enhancement.core import AudioEnhancer


class TestSpeechBrainEdgeCases(unittest.TestCase):
    """Test edge cases and potential bugs in SpeechBrain implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.config = SeparationConfig(
            device="cpu",
            confidence_threshold=0.7,
            cache_dir=tempfile.mkdtemp()
        )
        
    def tearDown(self):
        """Clean up after tests"""
        # Force garbage collection
        gc.collect()
        
        # Clean up temp directory
        import shutil
        if hasattr(self, 'config') and os.path.exists(self.config.cache_dir):
            shutil.rmtree(self.config.cache_dir)
    
    def test_extremely_short_audio(self):
        """Test with audio shorter than typical frame size"""
        # 10ms of audio (very short)
        short_audio = np.random.randn(160)  # 0.01s at 16kHz
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            separator = SpeechBrainSeparator(self.config)
            result = separator.separate_speakers(short_audio, self.sample_rate)
            
            # Should handle gracefully
            self.assertIsInstance(result, SeparationOutput)
            self.assertEqual(len(result.audio), len(short_audio))
    
    def test_zero_audio_not_empty(self):
        """Test with audio that's all zeros (silence)"""
        silent_audio = np.zeros(16000)  # 1 second of silence
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            separator = SpeechBrainSeparator(self.config)
            result = separator.separate_speakers(silent_audio, self.sample_rate)
            
            # Should process but potentially reject
            self.assertIsInstance(result, SeparationOutput)
            self.assertEqual(len(result.audio), len(silent_audio))
    
    def test_extreme_amplitude_audio(self):
        """Test with audio having extreme amplitudes"""
        # Very loud audio (might cause clipping)
        loud_audio = np.random.randn(16000) * 100  # Extreme amplitude
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation') as mock_sep:
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                
                # Mock separator to handle extreme values
                mock_instance = MagicMock()
                mock_instance.separate_batch.return_value = torch.randn(1, 16000, 2)
                mock_sep.from_hparams.return_value = mock_instance
                
                separator = SpeechBrainSeparator(self.config)
                result = separator.separate_speakers(loud_audio, self.sample_rate)
                
                # Should handle without crashing
                self.assertIsInstance(result, SeparationOutput)
                # Check if audio was normalized/clipped appropriately
                self.assertTrue(np.all(np.isfinite(result.audio)))
    
    def test_nan_inf_in_audio(self):
        """Test with NaN and Inf values in audio"""
        # Audio with NaN and Inf
        bad_audio = np.random.randn(16000)
        bad_audio[100:110] = np.nan
        bad_audio[200:210] = np.inf
        bad_audio[300:310] = -np.inf
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            separator = SpeechBrainSeparator(self.config)
            result = separator.separate_speakers(bad_audio, self.sample_rate)
            
            # Should handle gracefully - likely reject
            self.assertIsInstance(result, SeparationOutput)
            if not result.rejected:
                # If not rejected, output should be finite
                self.assertTrue(np.all(np.isfinite(result.audio)))
    
    def test_memory_leak_batch_processing(self):
        """Test for memory leaks in batch processing"""
        # Create multiple large audio samples
        batch_size = 10
        audio_batch = [np.random.randn(16000 * 5) for _ in range(batch_size)]  # 5s each
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            separator = SpeechBrainSeparator(self.config)
            
            # Get initial memory usage
            initial_memory = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            # Process batch multiple times
            for _ in range(3):
                results = separator.process_batch(audio_batch, self.sample_rate)
                self.assertEqual(len(results), batch_size)
            
            # Check memory wasn't growing unbounded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                # Memory shouldn't grow significantly
                memory_growth = final_memory - initial_memory
                self.assertLess(memory_growth, 100 * 1024 * 1024)  # Less than 100MB growth
    
    def test_concurrent_processing_safety(self):
        """Test thread safety with concurrent processing"""
        from concurrent.futures import ThreadPoolExecutor
        
        audio_samples = [np.random.randn(16000) for _ in range(5)]
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            separator = SpeechBrainSeparator(self.config)
            
            # Process samples concurrently
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(separator.separate_speakers, audio, self.sample_rate)
                    for audio in audio_samples
                ]
                results = [f.result() for f in futures]
            
            # All should complete successfully
            self.assertEqual(len(results), 5)
            for result in results:
                self.assertIsInstance(result, SeparationOutput)
    
    def test_wrong_sample_rate_handling(self):
        """Test with mismatched sample rates"""
        audio = np.random.randn(8000)  # 1 second at 8kHz
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'), \
             patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
            
            separator = SpeechBrainSeparator(self.config)
            
            # Try with wrong sample rate
            result = separator.separate_speakers(audio, 8000)  # Not 16kHz
            
            # Should still process (models might handle resampling internally)
            self.assertIsInstance(result, SeparationOutput)
    
    def test_model_loading_failure(self):
        """Test behavior when model loading fails"""
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation') as mock_sep:
            # Make model loading fail
            mock_sep.from_hparams.side_effect = RuntimeError("Model download failed")
            
            # Should raise during initialization
            with self.assertRaises(RuntimeError):
                separator = SpeechBrainSeparator(self.config)
    
    def test_gpu_out_of_memory(self):
        """Test OOM handling during processing"""
        audio = np.random.randn(16000 * 60)  # 1 minute audio
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation') as mock_sep:
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                
                # Mock OOM error
                mock_instance = MagicMock()
                mock_instance.separate_batch.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM")
                mock_sep.from_hparams.return_value = mock_instance
                
                separator = SpeechBrainSeparator(self.config)
                result = separator.separate_speakers(audio, self.sample_rate)
                
                # Should handle gracefully
                self.assertIsInstance(result, SeparationOutput)
                self.assertTrue(result.rejected)
                self.assertIn("error", result.rejection_reason.lower())
    
    def test_separation_output_dimensions(self):
        """Test handling of unexpected separation output dimensions"""
        audio = np.random.randn(16000)
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation') as mock_sep:
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                
                mock_instance = MagicMock()
                
                # Test 1: Wrong number of dimensions
                mock_instance.separate_batch.return_value = torch.randn(16000)  # 1D instead of 3D
                mock_sep.from_hparams.return_value = mock_instance
                
                separator = SpeechBrainSeparator(self.config)
                result = separator.separate_speakers(audio, self.sample_rate)
                
                # Should handle gracefully
                self.assertIsInstance(result, SeparationOutput)
    
    def test_quality_metrics_calculation_errors(self):
        """Test robustness of quality metrics calculation"""
        # Create audio that might cause metric calculation issues
        audio = np.random.randn(16000)
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation') as mock_sep:
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                # Mock separator to return proper tensor
                mock_instance = MagicMock()
                mock_instance.separate_batch.return_value = torch.randn(1, 16000, 2)
                mock_sep.from_hparams.return_value = mock_instance
                
                with patch('utils.audio_metrics.calculate_stoi') as mock_stoi:
                    # Make STOI calculation fail
                    mock_stoi.side_effect = Exception("STOI calculation error")
                    
                    separator = SpeechBrainSeparator(self.config)
                    result = separator.separate_speakers(audio, self.sample_rate)
                    
                    # Should still return result despite metric errors
                    self.assertIsInstance(result, SeparationOutput)
                    # Metrics should contain error info
                    self.assertIn('error', result.metrics)
    
    def test_cache_directory_permissions(self):
        """Test handling of cache directory permission issues"""
        # Create read-only directory
        import tempfile
        import stat
        
        read_only_dir = tempfile.mkdtemp()
        os.chmod(read_only_dir, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        
        config = SeparationConfig(
            device="cpu",
            cache_dir=os.path.join(read_only_dir, "models")
        )
        
        try:
            with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'):
                with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                    # Should handle permission error gracefully
                    separator = SpeechBrainSeparator(config)
                    # If we get here, it handled the error
                    self.assertIsNotNone(separator)
        except PermissionError:
            # This is also acceptable - error was properly raised
            pass
        finally:
            # Restore permissions and cleanup
            os.chmod(read_only_dir, stat.S_IRWXU)
            shutil.rmtree(read_only_dir)
    
    def test_speaker_selection_edge_cases(self):
        """Test edge cases in speaker selection logic"""
        from processors.audio_enhancement.speaker_selection import SpeakerSelector
        
        selector = SpeakerSelector(method="energy")
        
        # Test 1: All sources have zero energy
        zero_sources = [np.zeros(16000), np.zeros(16000)]
        idx, confidence = selector.select_primary_speaker(zero_sources)
        self.assertEqual(confidence, 0.0)
        
        # Test 2: One source is all NaN
        nan_sources = [np.random.randn(16000), np.full(16000, np.nan)]
        idx, confidence = selector.select_primary_speaker(nan_sources)
        self.assertEqual(idx, 0)  # Should select the valid source
        
        # Test 3: Empty source list
        with self.assertRaises(IndexError):
            selector.select_primary_speaker([])
    
    def test_integration_with_enhancement_pipeline(self):
        """Test integration edge cases with AudioEnhancer"""
        # Test very noisy audio that might trigger aggressive enhancement
        noisy_audio = np.random.randn(16000 * 4)  # 4 seconds
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'):
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                
                enhancer = AudioEnhancer(
                    use_gpu=False,
                    enhancement_level="ultra_aggressive"
                )
                
                # Process with enhancement pipeline
                enhanced = enhancer.enhance(noisy_audio, self.sample_rate)
                
                # Should complete without errors
                self.assertIsNotNone(enhanced)
                self.assertEqual(len(enhanced), len(noisy_audio))
    
    def test_statistics_overflow(self):
        """Test statistics tracking with many processes"""
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation') as mock_sep:
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                # Mock separator to return proper tensors
                mock_instance = MagicMock()
                mock_instance.separate_batch.return_value = torch.randn(1, 1000, 2)
                mock_sep.from_hparams.return_value = mock_instance
                
                separator = SpeechBrainSeparator(self.config)
                
                # Process many samples to test statistics overflow
                for i in range(1000):
                    audio = np.random.randn(1000)  # Small audio
                    separator.separate_speakers(audio, self.sample_rate)
                
                stats = separator.get_stats()
                
                # Stats should be reasonable
                self.assertEqual(stats['total_processed'], 1000)
                self.assertGreaterEqual(stats['average_confidence'], 0)
                self.assertLessEqual(stats['average_confidence'], 1)
                self.assertGreater(stats['average_processing_time_ms'], 0)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios"""
    
    def test_s3_s5_pattern_audio(self):
        """Test with audio patterns similar to problematic S3/S5 samples"""
        # Simulate S3/S5 characteristics
        duration = 8  # 8 seconds typical
        sample_rate = 16000
        t = np.linspace(0, duration, duration * sample_rate)
        
        # Primary speaker (Thai-like tonal patterns)
        primary = np.sin(2 * np.pi * 180 * t) * 0.6  # Lower frequency
        primary += np.sin(2 * np.pi * 360 * t) * 0.3
        primary *= (1 + 0.2 * np.sin(2 * np.pi * 2 * t))  # Amplitude modulation
        
        # Secondary speaker (different tonal pattern)
        secondary = np.sin(2 * np.pi * 250 * t) * 0.5
        secondary += np.sin(2 * np.pi * 500 * t) * 0.3
        
        # Mix with secondary speaker appearing intermittently
        mixed = primary.copy()
        # Add secondary speaker in bursts
        for i in range(1, 7, 2):  # At 1s, 3s, 5s
            start = i * sample_rate
            end = start + sample_rate
            mixed[start:end] = 0.6 * primary[start:end] + 0.4 * secondary[start:end]
        
        # Add realistic noise
        mixed += np.random.randn(len(mixed)) * 0.02
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation') as mock_sep:
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition') as mock_spk:
                with patch('utils.audio_metrics.calculate_stoi') as mock_stoi:
                    # Mock realistic separation
                    mock_instance = MagicMock()
                    separated = torch.zeros(1, len(mixed), 2)
                    # Make primary speaker more dominant
                    separated[0, :, 0] = torch.from_numpy(primary * 1.2)  # Amplified primary
                    separated[0, :, 1] = torch.from_numpy(secondary * 0.4)  # Attenuated secondary
                    
                    mock_instance.separate_batch.return_value = separated
                    mock_sep.from_hparams.return_value = mock_instance
                    
                    # Mock speaker embedding model
                    mock_spk_instance = MagicMock()
                    mock_spk.from_hparams.return_value = mock_spk_instance
                    
                    # Mock STOI to return a good value
                    mock_stoi.return_value = 0.9
                    
                    separator = SpeechBrainSeparator(SeparationConfig(
                        device="cpu",
                        confidence_threshold=0.7,
                        speaker_selection="energy"
                    ))
                    
                    result = separator.separate_speakers(mixed, sample_rate)
                    
                    # Debug information
                    if result.rejected:
                        print(f"Rejection reason: {result.rejection_reason}")
                        print(f"Metrics: {result.metrics}")
                        print(f"Confidence: {result.confidence}")
                    
                    # Should successfully separate
                    self.assertFalse(result.rejected)
                    self.assertGreater(result.confidence, 0.5)
                    self.assertEqual(result.num_speakers_detected, 2)
    
    def test_command_line_integration(self):
        """Test usage patterns from command line integration"""
        # Simulate processing from main.py with typical settings
        config = SeparationConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            confidence_threshold=0.7,
            batch_size=16,
            speaker_selection="energy",
            use_mixed_precision=True,
            quality_thresholds={
                "min_pesq": 3.5,
                "min_stoi": 0.85,
                "max_spectral_distortion": 0.15
            }
        )
        
        # Test configuration validation
        self.assertIn("min_stoi", config.quality_thresholds)
        self.assertEqual(config.confidence_threshold, 0.7)
        
        # Test with typical batch size from command line
        audio_batch = [np.random.randn(16000 * 4) for _ in range(10)]  # 10 samples
        
        with patch('processors.audio_enhancement.speechbrain_separator.SepformerSeparation'):
            with patch('processors.audio_enhancement.speechbrain_separator.SpeakerRecognition'):
                
                separator = SpeechBrainSeparator(config)
                
                # Measure processing time
                import time
                start = time.time()
                results = separator.process_batch(audio_batch[:3], 16000)  # Process 3 samples
                elapsed = time.time() - start
                
                # Should process quickly
                self.assertEqual(len(results), 3)
                # Reasonable time per sample
                time_per_sample = elapsed / 3
                self.assertLess(time_per_sample, 10)  # Less than 10s per sample


if __name__ == '__main__':
    unittest.main()