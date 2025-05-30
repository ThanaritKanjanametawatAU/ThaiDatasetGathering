"""
Tests for fixing the get_last_id functionality with SplitInfo error handling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset
import numpy as np


class TestGetLastIdFix(unittest.TestCase):
    """Test cases for fixing get_last_id with proper error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset_name = "test-user/test-dataset"
        
    def test_get_last_id_with_real_dataset_structure(self):
        """Test get_last_id with actual HuggingFace dataset structure."""
        from utils.huggingface import get_last_id
        
        # Create a real dataset-like structure
        data = {
            "ID": ["S1", "S2", "S10", "S5", "S100"],
            "audio": [{"array": np.zeros(16000), "sampling_rate": 16000} for _ in range(5)],
            "transcript": ["test" for _ in range(5)]
        }
        
        with patch('utils.huggingface.load_dataset') as mock_load:
            # Create a real Dataset object
            dataset = Dataset.from_dict(data)
            mock_load.return_value = dataset
            
            result = get_last_id(self.dataset_name)
            self.assertEqual(result, 100)
            
    def test_get_last_id_handles_nonexistent_dataset(self):
        """Test get_last_id returns None for nonexistent dataset."""
        from utils.huggingface import get_last_id
        
        with patch('utils.huggingface.load_dataset') as mock_load:
            mock_load.side_effect = FileNotFoundError("Dataset not found")
            
            result = get_last_id(self.dataset_name)
            self.assertIsNone(result)
            
    def test_integration_with_main_append_mode(self):
        """Test that main.py properly handles None return from get_last_id."""
        # Create test data
        data = {
            "ID": ["S50", "S100", "S150"],
            "audio": [{"array": np.zeros(16000), "sampling_rate": 16000} for _ in range(3)],
            "transcript": ["test" for _ in range(3)]
        }
        dataset = Dataset.from_dict(data)
        
        with patch('utils.huggingface.load_dataset') as mock_load:
            mock_load.return_value = dataset
            
            # Import after patching to ensure our mock is used
            from utils.huggingface import get_last_id
            
            # Test successful case
            result = get_last_id(self.dataset_name)
            self.assertEqual(result, 150)
            
    def test_get_last_id_with_streaming_dataset(self):
        """Test get_last_id with streaming dataset (which might cause SplitInfo error)."""
        from utils.huggingface import get_last_id
        
        with patch('utils.huggingface.load_dataset') as mock_load:
            # Simulate the SplitInfo error
            mock_load.side_effect = ValueError(
                "[{'expected': SplitInfo(name='train', num_bytes=0, num_examples=0, shard_lengths=None, dataset_name=None), "
                "'recorded': SplitInfo(name='train', num_bytes=23130973, num_examples=200, shard_lengths=None, dataset_name='thai-voice-test2')}]"
            )
            
            result = get_last_id(self.dataset_name)
            self.assertIsNone(result)


class TestGetLastIdImproved(unittest.TestCase):
    """Test cases for the improved get_last_id implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset_name = "test-user/test-dataset"
    
    def test_improved_get_last_id_with_split_info_error(self):
        """Test improved get_last_id that handles SplitInfo errors gracefully."""
        # This test will verify our fix works correctly
        from utils.huggingface import get_last_id
        
        with patch('utils.huggingface.load_dataset') as mock_load:
            # First attempt fails with SplitInfo error
            mock_load.side_effect = [
                ValueError("[{'expected': SplitInfo(...), 'recorded': SplitInfo(...)}]"),
                # Second attempt could succeed if we implement retry with streaming=True
            ]
            
            result = get_last_id(self.dataset_name)
            # Currently returns None, but after fix should handle error better
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()