"""
Comprehensive test suite to verify all PR fixes are properly merged and working.
This test covers:
1. Audio quality metrics calculation (PR #18)
2. Append mode ID continuation (PR #16, #17)
3. SplitInfo error handling (PR #16, #17)
"""
import unittest
import numpy as np
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from datasets import Dataset, Audio, DatasetDict
from datasets.exceptions import DatasetNotFoundError

# Import the modules we're testing
from utils.audio_metrics import calculate_snr, calculate_pesq, calculate_stoi
from utils.huggingface import get_last_id
from processors.audio_enhancement.core import AudioEnhancer


class TestPRMergeVerification(unittest.TestCase):
    """Comprehensive tests to verify all PR functionality is working correctly."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.clean_audio = np.sin(2 * np.pi * 440 * np.arange(self.sample_rate) / self.sample_rate)
        self.noisy_audio = self.clean_audio + 0.1 * np.random.randn(self.sample_rate)
        
    def test_audio_metrics_snr_calculation(self):
        """Test that SNR calculation is properly capped at 40dB (PR #18)."""
        # Test with identical signals (should return 40dB, not infinity)
        snr = calculate_snr(self.clean_audio, self.clean_audio)
        self.assertEqual(snr, 40.0, "SNR should be capped at 40dB for identical signals")
        
        # Test with very clean signal (should also cap at 40dB)
        very_clean = self.clean_audio + 1e-10 * np.random.randn(self.sample_rate)
        snr = calculate_snr(self.clean_audio, very_clean)
        self.assertEqual(snr, 40.0, "SNR should be capped at 40dB for very clean signals")
        
        # Test with normal noisy signal (should return reasonable value)
        snr = calculate_snr(self.clean_audio, self.noisy_audio)
        self.assertLess(snr, 40.0, "SNR should be less than 40dB for noisy signals")
        self.assertGreater(snr, 0.0, "SNR should be positive for moderately noisy signals")
        
    def test_audio_metrics_pesq_calculation(self):
        """Test that PESQ calculation returns valid values (PR #18)."""
        pesq = calculate_pesq(self.clean_audio, self.noisy_audio, self.sample_rate)
        self.assertIsInstance(pesq, float, "PESQ should return a float value")
        self.assertGreater(pesq, -0.5, "PESQ should be greater than -0.5")
        self.assertLess(pesq, 4.5, "PESQ should be less than 4.5")
        
    def test_audio_metrics_stoi_calculation(self):
        """Test that STOI calculation returns valid values (PR #18)."""
        stoi = calculate_stoi(self.clean_audio, self.noisy_audio, self.sample_rate)
        self.assertIsInstance(stoi, float, "STOI should return a float value")
        self.assertGreaterEqual(stoi, 0.0, "STOI should be between 0 and 1")
        self.assertLessEqual(stoi, 1.0, "STOI should be between 0 and 1")
        
    def test_audio_enhancement_metrics_in_metadata(self):
        """Test that audio enhancement properly includes metrics in metadata (PR #18)."""
        enhancer = AudioEnhancer(
            use_gpu=False,  # Use CPU for testing
            fallback_to_cpu=True
        )
        
        # Enhance the audio directly with metadata
        enhanced_audio, metadata = enhancer.enhance(
            self.noisy_audio,
            self.sample_rate,
            noise_level='moderate',  # Force moderate enhancement
            return_metadata=True
        )
        
        # Check that metrics are in metadata
        self.assertIsInstance(enhanced_audio, np.ndarray, "Enhanced audio should be numpy array")
        self.assertIsInstance(metadata, dict, "Metadata should be a dictionary")
        
        # Check metadata contains proper values
        self.assertIn('snr_improvement', metadata, "Metadata should include SNR improvement")
        self.assertIn('pesq', metadata, "Metadata should include PESQ score")
        self.assertIn('stoi', metadata, "Metadata should include STOI score")
        
        # Verify metrics are not 0 or infinity (except for SNR which could be 0)
        if metadata['snr_improvement'] != 0:  # SNR improvement can be 0 for clean audio
            self.assertNotEqual(metadata['snr_improvement'], float('inf'), "SNR improvement should not be infinity")
        self.assertNotEqual(metadata['pesq'], 0, "PESQ should not be 0")
        self.assertNotEqual(metadata['stoi'], 0, "STOI should not be 0")
            
    @patch('utils.huggingface.load_dataset')
    def test_append_mode_id_continuation(self, mock_load_dataset):
        """Test that append mode continues IDs from the last existing ID (PR #16, #17)."""
        # Create a mock dataset with existing data
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        # When accessing dataset["ID"], return a list of IDs
        mock_dataset.__getitem__.side_effect = lambda key: ['S100'] if key == 'ID' else None
        mock_dataset.features = {"ID": None}  # Mock features attribute
        mock_load_dataset.return_value = mock_dataset
        
        # Test get_last_id function
        last_id = get_last_id('test/dataset')
        self.assertEqual(last_id, 100, "Should return the last ID number from existing dataset")
        
    @patch('utils.huggingface.load_dataset')
    def test_split_info_error_handling(self, mock_load_dataset):
        """Test that SplitInfo errors are handled gracefully (PR #16, #17)."""
        # Create a mock dataset for the second call
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 50
        # When accessing dataset["ID"], return a list of IDs
        mock_dataset.__getitem__.side_effect = lambda key: ['S50'] if key == 'ID' else None
        mock_dataset.features = {"ID": None}  # Mock features attribute
        
        # First call raises specific ValueError about SplitInfo
        # Second call returns the mock dataset
        mock_load_dataset.side_effect = [
            ValueError("SplitInfo(name='train', num_bytes=123456, num_examples=100, dataset_name=None) expected but SplitInfo(name='train', num_bytes=654321, num_examples=200, dataset_name=None) recorded"),
            mock_dataset
        ]
        
        # Test that get_last_id handles the error and retries
        last_id = get_last_id('test/dataset')
        
        # Should have been called twice (once failed, once with force_redownload)
        self.assertEqual(mock_load_dataset.call_count, 2, "Should retry with force_redownload=True")
        
        # Check that second call had force_redownload=True
        second_call_kwargs = mock_load_dataset.call_args_list[1][1]
        self.assertTrue(second_call_kwargs.get('download_mode') == 'force_redownload', 
                       "Should retry with force_redownload mode")
        
        # Should return the correct last ID
        self.assertEqual(last_id, 50, "Should return correct last ID after retry")
        
    @patch('utils.huggingface.load_dataset')
    def test_append_mode_no_duplicate_ids(self, mock_load_dataset):
        """Test that append mode doesn't create duplicate IDs (PR #16, #17)."""
        # Mock existing dataset with IDs S1-S100
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        # When accessing dataset["ID"], return a list of IDs S1-S100
        mock_dataset.__getitem__.side_effect = lambda key: [f'S{i+1}' for i in range(100)] if key == 'ID' else None
        mock_dataset.features = {"ID": None}  # Mock features attribute
        mock_load_dataset.return_value = mock_dataset
        
        # Get last ID
        last_id = get_last_id('test/dataset')
        
        # New IDs should start from S101
        new_start_id = last_id + 1
        self.assertEqual(new_start_id, 101, "New IDs should start after the last existing ID")
        
        # Simulate creating new samples
        new_ids = [f'S{new_start_id + i}' for i in range(10)]
        
        # Verify no duplicates with existing IDs
        existing_ids = [f'S{i+1}' for i in range(100)]
        duplicates = set(new_ids) & set(existing_ids)
        self.assertEqual(len(duplicates), 0, "Should be no duplicate IDs between existing and new data")


if __name__ == '__main__':
    unittest.main()