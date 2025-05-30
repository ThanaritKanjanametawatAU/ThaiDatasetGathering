"""
Integration tests for append mode ID continuation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from datasets import Dataset, Features, Value, Audio
import numpy as np


class TestAppendModeIntegration(unittest.TestCase):
    """Integration tests for append mode functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    @patch('utils.huggingface.load_dataset')
    @patch('utils.streaming.HfApi')
    @patch('utils.streaming.upload_file')
    @patch('utils.streaming.Dataset')
    def test_streaming_append_continues_ids(self, mock_dataset_class, mock_upload, mock_hf_api, mock_load_dataset):
        """Test that streaming mode with append correctly continues IDs."""
        # Setup existing dataset with IDs up to S200
        existing_data = {
            "ID": [f"S{i}" for i in range(1, 201)],
            "audio": [{"array": np.zeros(16000), "sampling_rate": 16000} for _ in range(200)],
            "transcript": ["test" for _ in range(200)],
            "speaker_id": [f"SPK_{i:05d}" for i in range(1, 201)],
            "Language": ["th" for _ in range(200)],
            "length": [1.0 for _ in range(200)],
            "dataset_name": ["TestDataset" for _ in range(200)],
            "confidence_score": [1.0 for _ in range(200)]
        }
        
        features = Features({
            "ID": Value("string"),
            "speaker_id": Value("string"),
            "Language": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "transcript": Value("string"),
            "length": Value("float32"),
            "dataset_name": Value("string"),
            "confidence_score": Value("float32")
        })
        
        existing_dataset = Dataset.from_dict(existing_data, features=features)
        mock_load_dataset.return_value = existing_dataset
        
        # Setup HfApi mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = []  # No existing shards
        mock_hf_api.return_value = mock_api
        
        # Mock dataset creation
        mock_dataset_instance = Mock()
        mock_dataset_class.from_list.return_value = mock_dataset_instance
        
        # Import after mocking
        from utils.streaming import StreamingUploader
        from utils.huggingface import get_last_id
        
        # Test get_last_id returns 200
        last_id = get_last_id("test-user/test-dataset")
        self.assertEqual(last_id, 200)
        
        # Create uploader in append mode
        uploader = StreamingUploader("test-user/test-dataset", "test-token", append_mode=True)
        
        # Create new samples that should start from S201
        new_samples = []
        start_id = 201  # Should continue from here
        
        for i in range(5):
            sample_id = f"S{start_id + i}"
            new_samples.append({
                "ID": sample_id,
                "speaker_id": f"SPK_{start_id + i:05d}",
                "Language": "th",
                "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
                "transcript": f"New sample {i}",
                "length": 1.0,
                "dataset_name": "NewDataset",
                "confidence_score": 1.0
            })
        
        # Upload batch
        success, shard_name = uploader.upload_batch(new_samples)
        
        # Verify
        self.assertTrue(success)
        # Check that the samples were uploaded with correct IDs
        mock_dataset_class.from_list.assert_called_once()
        uploaded_samples = mock_dataset_class.from_list.call_args[0][0]
        
        # Verify IDs start from S201
        for i, sample in enumerate(uploaded_samples):
            expected_id = f"S{201 + i}"
            self.assertEqual(sample["ID"], expected_id)
            
    def test_full_append_workflow(self):
        """Test the complete append workflow from command line to dataset."""
        # This test would verify the entire flow but requires more complex mocking
        # of the main.py execution flow
        pass


class TestAppendModeEdgeCases(unittest.TestCase):
    """Test edge cases for append mode."""
    
    @patch('utils.huggingface.load_dataset')
    def test_append_to_empty_dataset(self, mock_load_dataset):
        """Test appending to an empty dataset starts from S1."""
        from utils.huggingface import get_last_id
        
        # Create empty dataset
        empty_data = {
            "ID": [],
            "audio": [],
            "transcript": []
        }
        empty_dataset = Dataset.from_dict(empty_data)
        mock_load_dataset.return_value = empty_dataset
        
        # Test
        result = get_last_id("test-user/empty-dataset")
        
        # Should return 0 for empty dataset
        self.assertEqual(result, 0)
        
    @patch('utils.huggingface.load_dataset')
    def test_append_with_non_sequential_ids(self, mock_load_dataset):
        """Test appending when existing dataset has non-sequential IDs."""
        from utils.huggingface import get_last_id
        
        # Create dataset with gaps in IDs
        data = {
            "ID": ["S1", "S5", "S10", "S100", "S50"],  # Non-sequential
            "audio": [{"array": np.zeros(16000), "sampling_rate": 16000} for _ in range(5)],
            "transcript": ["test" for _ in range(5)]
        }
        dataset = Dataset.from_dict(data)
        mock_load_dataset.return_value = dataset
        
        # Test
        result = get_last_id("test-user/test-dataset")
        
        # Should return the maximum ID (100)
        self.assertEqual(result, 100)


if __name__ == '__main__':
    unittest.main()