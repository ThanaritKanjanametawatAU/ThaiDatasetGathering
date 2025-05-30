"""
Comprehensive tests for HuggingFace schema compatibility, including speaker_id field.
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
import numpy as np
from datasets import Dataset, Features, Value, Audio
import pyarrow.parquet as pq

from utils.streaming import StreamingUploader
from huggingface_hub import HfApi
from utils.huggingface import create_hf_dataset


class TestHuggingFaceSchemaComplete(unittest.TestCase):
    """Test complete schema compatibility with HuggingFace, including speaker_id."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {
            "ID": "S1",
            "speaker_id": "SPK_00001",
            "Language": "th",
            "audio": {
                "array": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "sampling_rate": 16000,
                "path": "S1.wav"
            },
            "transcript": "ทดสอบ",
            "length": 1.5,
            "dataset_name": "TestDataset",
            "confidence_score": 1.0
        }
        
    def test_dataset_card_includes_speaker_id(self):
        """Test that the dataset card metadata includes speaker_id field."""
        # Mock both HfApi and upload_file at the module level
        with patch('utils.streaming.HfApi') as mock_hf_api, \
             patch('utils.streaming.upload_file') as mock_upload_file:
            # Mock the API instance and its methods
            mock_api_instance = MagicMock()
            mock_hf_api.return_value = mock_api_instance
            mock_api_instance.create_repo = MagicMock()
            
            uploader = StreamingUploader("test/repo", append_mode=False)
            
            # Mock the upload_file function to capture the content
            uploaded_content = None
            def capture_upload(path_or_fileobj, **kwargs):
                nonlocal uploaded_content
                if kwargs.get('path_in_repo') == 'README.md':
                    if isinstance(path_or_fileobj, bytes):
                        uploaded_content = path_or_fileobj.decode()
                    else:
                        uploaded_content = path_or_fileobj
            
            mock_upload_file.side_effect = capture_upload
            
            dataset_info = {
                'name': 'Test Dataset',
                'total_samples': 100,
                'total_duration_hours': 10.5
            }
            uploader.upload_dataset_card(dataset_info)
            
            # Verify speaker_id is in the dataset card
            self.assertIsNotNone(uploaded_content)
            self.assertIn("- name: speaker_id", uploaded_content)
            self.assertIn("dtype: string", uploaded_content)
            
            # Verify it appears after ID and before Language
            lines = uploaded_content.split('\n')
            id_line = None
            speaker_id_line = None
            language_line = None
            
            for i, line in enumerate(lines):
                if "- name: ID" in line:
                    id_line = i
                elif "- name: speaker_id" in line:
                    speaker_id_line = i
                elif "- name: Language" in line:
                    language_line = i
            
            self.assertIsNotNone(id_line)
            self.assertIsNotNone(speaker_id_line)
            self.assertIsNotNone(language_line)
            self.assertLess(id_line, speaker_id_line)
            self.assertLess(speaker_id_line, language_line)
    
    def test_features_definition_matches_data(self):
        """Test that Features definition matches actual data structure."""
        # Expected features definition
        expected_features = Features({
            "ID": Value("string"),
            "speaker_id": Value("string"),
            "Language": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "transcript": Value("string"),
            "length": Value("float32"),
            "dataset_name": Value("string"),
            "confidence_score": Value("float32")
        })
        
        # Create dataset with sample data
        dataset = create_hf_dataset([self.sample_data])
        
        self.assertIsNotNone(dataset)
        
        # Verify all expected fields are present
        for field in expected_features:
            self.assertIn(field, dataset.features)
    
    def test_parquet_schema_consistency(self):
        """Test that parquet files have consistent schema with Features."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a dataset with our sample data
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
            
            dataset = Dataset.from_list([self.sample_data], features=features)
            
            # Save to parquet
            parquet_path = os.path.join(tmp_dir, "test.parquet")
            dataset.to_parquet(parquet_path)
            
            # Read back the parquet file schema
            parquet_file = pq.ParquetFile(parquet_path)
            parquet_schema = parquet_file.schema
            
            # Verify all fields are present in parquet schema
            field_names = [field.name for field in parquet_schema]
            
            expected_fields = ["ID", "speaker_id", "Language", "audio", "transcript", 
                             "length", "dataset_name", "confidence_score"]
            
            for field in expected_fields:
                self.assertIn(field, field_names)
    
    def test_streaming_uploader_batch_with_speaker_id(self):
        """Test that StreamingUploader correctly handles samples with speaker_id."""
        with patch('huggingface_hub.HfApi'):
            uploader = StreamingUploader("test/repo", append_mode=False)
            
            # Mock the necessary methods
            with patch('datasets.Dataset.to_parquet'), \
                 patch.object(uploader.api, 'upload_file'):
                
                success, filename = uploader.upload_batch([self.sample_data])
                
                self.assertTrue(success)
                self.assertIn("shard_", filename)
    
    def test_complete_schema_in_all_components(self):
        """Test that all components use consistent schema including speaker_id."""
        # Test 1: Features in huggingface.py
        from utils.huggingface import create_hf_dataset
        dataset1 = create_hf_dataset([self.sample_data])
        self.assertIn("speaker_id", dataset1.features)
        
        # Test 2: Features in streaming.py upload_batch
        with patch('huggingface_hub.HfApi'):
            uploader = StreamingUploader("test/repo")
            
            # Capture the dataset creation
            created_dataset = None
            original_from_list = Dataset.from_list
            
            def capture_dataset(*args, **kwargs):
                nonlocal created_dataset
                created_dataset = original_from_list(*args, **kwargs)
                return created_dataset
            
            with patch('datasets.Dataset.from_list', side_effect=capture_dataset), \
                 patch('datasets.Dataset.to_parquet'), \
                 patch.object(uploader.api, 'upload_file'):
                
                uploader.upload_batch([self.sample_data])
                
                self.assertIsNotNone(created_dataset)
                self.assertIn("speaker_id", created_dataset.features)
    
    def test_schema_order_preservation(self):
        """Test that field order is preserved throughout the pipeline."""
        expected_order = ["ID", "speaker_id", "Language", "audio", "transcript", 
                         "length", "dataset_name", "confidence_score"]
        
        # Test dataset creation
        dataset = create_hf_dataset([self.sample_data])
        actual_order = list(dataset.features.keys())
        
        self.assertEqual(expected_order, actual_order)


if __name__ == '__main__':
    unittest.main()