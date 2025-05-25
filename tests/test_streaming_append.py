"""
Tests for StreamingUploader append mode functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
from utils.streaming import StreamingUploader


class TestStreamingUploaderAppendMode(unittest.TestCase):
    """Test cases for StreamingUploader append mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_id = "test-user/test-dataset"
        self.token = "test-token"
        
    @patch('utils.streaming.HfApi')
    def test_init_without_append_mode(self, mock_hf_api):
        """Test that StreamingUploader starts from shard 0 without append mode."""
        # Setup
        mock_api = Mock()
        mock_hf_api.return_value = mock_api
        
        # Create uploader without append mode
        uploader = StreamingUploader(self.repo_id, self.token)
        
        # Verify
        self.assertEqual(uploader.shard_num, 0)
        self.assertFalse(uploader.append_mode)
        
    @patch('utils.streaming.HfApi')
    def test_init_with_append_mode_no_existing_files(self, mock_hf_api):
        """Test append mode when no existing shards are present."""
        # Setup
        mock_api = Mock()
        mock_api.list_repo_files.return_value = []  # No existing files
        mock_hf_api.return_value = mock_api
        
        # Create uploader with append mode
        uploader = StreamingUploader(self.repo_id, self.token, append_mode=True)
        
        # Verify
        self.assertEqual(uploader.shard_num, 0)
        self.assertTrue(uploader.append_mode)
        mock_api.list_repo_files.assert_called_once_with(
            repo_id=self.repo_id,
            repo_type="dataset"
        )
        
    @patch('utils.streaming.HfApi')
    def test_init_with_append_mode_existing_shards(self, mock_hf_api):
        """Test append mode when existing shards are present."""
        # Setup
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "data/train/shard_00000.parquet",
            "data/train/shard_00001.parquet",
            "data/train/shard_00002.parquet",
            "README.md"
        ]
        mock_hf_api.return_value = mock_api
        
        # Create uploader with append mode
        uploader = StreamingUploader(self.repo_id, self.token, append_mode=True)
        
        # Verify - should start from shard 3 (after 0, 1, 2)
        self.assertEqual(uploader.shard_num, 3)
        self.assertTrue(uploader.append_mode)
        
    @patch('utils.streaming.HfApi')
    def test_init_with_append_mode_mixed_format_files(self, mock_hf_api):
        """Test append mode with mixed format parquet files."""
        # Setup
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "data/train/shard_00000.parquet",
            "data/train/shard_00003.parquet",  # Gap in numbering
            "data/train-00000-of-00001.parquet",  # Old format
            "data/train/shard_00007.parquet",
            "README.md",
            "data/.gitkeep"
        ]
        mock_hf_api.return_value = mock_api
        
        # Create uploader with append mode
        uploader = StreamingUploader(self.repo_id, self.token, append_mode=True)
        
        # Verify - should start from 8 (after highest shard 7)
        self.assertEqual(uploader.shard_num, 8)
        
    @patch('utils.streaming.HfApi')
    def test_get_existing_shard_numbers(self, mock_hf_api):
        """Test the _get_existing_shard_numbers method."""
        # Setup
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "data/train/shard_00000.parquet",
            "data/train/shard_00010.parquet",
            "data/train/shard_00005.parquet",
            "data/train/not_a_shard.parquet",
            "data/test/shard_00001.parquet",  # Different split
        ]
        mock_hf_api.return_value = mock_api
        
        # Create uploader
        uploader = StreamingUploader(self.repo_id, self.token, append_mode=True)
        
        # Get shard numbers
        shard_numbers = uploader._get_existing_shard_numbers()
        
        # Verify
        self.assertEqual(sorted(shard_numbers), [0, 5, 10])
        
    @patch('utils.streaming.HfApi')
    @patch('utils.streaming.upload_file')
    @patch('utils.streaming.Dataset')
    def test_upload_batch_preserves_shard_numbering(self, mock_dataset, mock_upload, mock_hf_api):
        """Test that upload_batch preserves correct shard numbering in append mode."""
        # Setup
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "data/train/shard_00002.parquet"
        ]
        mock_hf_api.return_value = mock_api
        
        mock_dataset_instance = Mock()
        mock_dataset.from_list.return_value = mock_dataset_instance
        
        # Create uploader with append mode
        uploader = StreamingUploader(self.repo_id, self.token, append_mode=True)
        
        # Upload first batch
        samples = [{"id": "S1", "data": "test1"}]
        success, shard_name = uploader.upload_batch(samples)
        
        # Verify first upload uses shard 3
        self.assertTrue(success)
        self.assertEqual(shard_name, "shard_00003.parquet")
        self.assertEqual(uploader.shard_num, 4)  # Should increment
        
        # Upload second batch
        samples2 = [{"id": "S2", "data": "test2"}]
        success2, shard_name2 = uploader.upload_batch(samples2)
        
        # Verify second upload uses shard 4
        self.assertTrue(success2)
        self.assertEqual(shard_name2, "shard_00004.parquet")
        self.assertEqual(uploader.shard_num, 5)
        
    @patch('utils.streaming.HfApi')
    def test_append_mode_with_api_error(self, mock_hf_api):
        """Test graceful handling when API fails to list files."""
        # Setup
        mock_api = Mock()
        mock_api.list_repo_files.side_effect = Exception("API Error")
        mock_hf_api.return_value = mock_api
        
        # Create uploader with append mode - should fall back to shard 0
        uploader = StreamingUploader(self.repo_id, self.token, append_mode=True)
        
        # Verify - should default to 0 on error
        self.assertEqual(uploader.shard_num, 0)
        self.assertTrue(uploader.append_mode)


if __name__ == '__main__':
    unittest.main()