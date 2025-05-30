"""
Tests for append mode ID continuation functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from utils.huggingface import get_last_id


class TestAppendIdContinuation(unittest.TestCase):
    """Test cases for append mode ID continuation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset_name = "test-user/test-dataset"
        
    @patch('utils.huggingface.load_dataset')
    def test_get_last_id_success(self, mock_load_dataset):
        """Test get_last_id returns correct value when dataset loads successfully."""
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.features = {"ID": "string", "audio": "audio", "transcript": "string"}
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["S1", "S2", "S10", "S5", "S100"])
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = get_last_id(self.dataset_name)
        
        # Verify
        self.assertEqual(result, 100)
        mock_load_dataset.assert_called_once_with(self.dataset_name, split="train")
        
    @patch('utils.huggingface.load_dataset')
    def test_get_last_id_no_id_field(self, mock_load_dataset):
        """Test get_last_id returns None when ID field is missing."""
        # Setup mock dataset without ID field
        mock_dataset = Mock()
        mock_dataset.features = {"audio": "audio", "transcript": "string"}
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = get_last_id(self.dataset_name)
        
        # Verify
        self.assertIsNone(result)
        
    @patch('utils.huggingface.load_dataset')
    def test_get_last_id_with_split_info_error(self, mock_load_dataset):
        """Test get_last_id handles SplitInfo mismatch error."""
        # Setup to raise the specific error mentioned in the issue
        error_msg = "[{'expected': SplitInfo(name='train', num_bytes=0, num_examples=0, shard_lengths=None, dataset_name=None), 'recorded': SplitInfo(name='train', num_bytes=23130973, num_examples=200, shard_lengths=None, dataset_name='thai-voice-test2')}]"
        mock_load_dataset.side_effect = Exception(error_msg)
        
        # Test
        result = get_last_id(self.dataset_name)
        
        # Verify
        self.assertIsNone(result)
        
    @patch('utils.huggingface.load_dataset')
    def test_get_last_id_empty_dataset(self, mock_load_dataset):
        """Test get_last_id returns 0 for empty dataset."""
        # Setup mock empty dataset
        mock_dataset = Mock()
        mock_dataset.features = {"ID": "string"}
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: [])
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = get_last_id(self.dataset_name)
        
        # Verify
        self.assertEqual(result, 0)
        
    @patch('utils.huggingface.load_dataset')
    def test_get_last_id_invalid_ids(self, mock_load_dataset):
        """Test get_last_id handles invalid ID formats."""
        # Setup mock dataset with invalid IDs
        mock_dataset = Mock()
        mock_dataset.features = {"ID": "string"}
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["invalid", "T1", "", None, "S", "Sabc"])
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = get_last_id(self.dataset_name)
        
        # Verify
        self.assertEqual(result, 0)
        
    @patch('utils.huggingface.load_dataset')
    def test_get_last_id_mixed_valid_invalid(self, mock_load_dataset):
        """Test get_last_id returns max of valid IDs when mixed with invalid."""
        # Setup mock dataset with mixed IDs
        mock_dataset = Mock()
        mock_dataset.features = {"ID": "string"}
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["S1", "invalid", "S50", "S", "S999", "T100"])
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = get_last_id(self.dataset_name)
        
        # Verify
        self.assertEqual(result, 999)


if __name__ == '__main__':
    unittest.main()