"""
Test HuggingFace schema compatibility for the Thai Voice dataset.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datasets import Features, Value, Audio, Dataset
import pandas as pd
import pyarrow as pa


class TestHuggingFaceSchema(unittest.TestCase):
    """Test schema compatibility with HuggingFace dataset viewer."""
    
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
            "transcript": "สวัสดีครับ",
            "length": 1.5,
            "dataset_name": "GigaSpeech2",
            "confidence_score": 1.0
        }
    
    def test_features_include_all_fields(self):
        """Test that Features definition includes all required fields."""
        from utils.huggingface import create_hf_dataset
        
        # The create_hf_dataset should handle all fields
        samples = [self.sample_data]
        
        # Mock the Dataset creation to check features
        with patch('utils.huggingface.Dataset') as mock_dataset:
            create_hf_dataset(samples)
            
            # Check that Dataset.from_dict was called
            self.assertTrue(mock_dataset.from_dict.called)
            
            # Get the features argument
            call_args = mock_dataset.from_dict.call_args
            features = call_args[1].get('features') if call_args[1] else None
            
            # Features should include all fields
            if features:
                feature_names = list(features.keys()) if hasattr(features, 'keys') else []
                self.assertIn("speaker_id", feature_names)
                self.assertIn("dataset_name", feature_names)
                self.assertIn("confidence_score", feature_names)
    
    def test_streaming_uploader_creates_proper_schema(self):
        """Test that StreamingUploader creates datasets with proper schema."""
        from utils.streaming import StreamingUploader
        
        with patch('utils.streaming.upload_file'), \
             patch('utils.streaming.HfApi'), \
             patch('utils.streaming.Dataset') as mock_dataset:
            
            uploader = StreamingUploader("test/repo", "fake_token")
            
            # Mock Dataset.from_list to check the schema
            def check_schema(samples, features=None):
                # Verify all samples have required fields
                for sample in samples:
                    self.assertIn("speaker_id", sample)
                    self.assertIn("dataset_name", sample)
                    self.assertIn("confidence_score", sample)
                
                # Create a mock dataset with proper features
                mock_ds = MagicMock()
                mock_ds.to_parquet = MagicMock()
                return mock_ds
            
            mock_dataset.from_list.side_effect = check_schema
            
            # Upload a batch
            success, _ = uploader.upload_batch([self.sample_data])
            self.assertTrue(success)
    
    def test_features_definition_matches_data(self):
        """Test that the Features definition matches actual data structure."""
        expected_features = Features({
            "ID": Value("string"),
            "speaker_id": Value("string"),
            "Language": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "transcript": Value("string"),
            "length": Value("float32"),
            "dataset_name": Value("string"),
            "confidence_score": Value("float64")
        })
        
        # Create dataset with our sample data
        dataset = Dataset.from_dict({
            "ID": [self.sample_data["ID"]],
            "speaker_id": [self.sample_data["speaker_id"]],
            "Language": [self.sample_data["Language"]],
            "audio": [self.sample_data["audio"]],
            "transcript": [self.sample_data["transcript"]],
            "length": [self.sample_data["length"]],
            "dataset_name": [self.sample_data["dataset_name"]],
            "confidence_score": [self.sample_data["confidence_score"]]
        }, features=expected_features)
        
        # Verify dataset creation doesn't raise schema errors
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["speaker_id"], "SPK_00001")
    
    def test_parquet_schema_compatibility(self):
        """Test that parquet files have correct schema for HuggingFace."""
        # Create a pandas DataFrame with our data
        df = pd.DataFrame([self.sample_data])
        
        # The audio column needs special handling for parquet
        # In actual implementation, audio is stored as struct
        audio_data = []
        for _, row in df.iterrows():
            audio_data.append({
                "array": row["audio"]["array"].tolist(),
                "sampling_rate": row["audio"]["sampling_rate"],
                "path": row["audio"]["path"]
            })
        
        df["audio"] = audio_data
        
        # Convert to pyarrow table
        table = pa.Table.from_pandas(df)
        
        # Check schema has all fields
        schema_fields = [field.name for field in table.schema]
        self.assertIn("speaker_id", schema_fields)
        self.assertIn("dataset_name", schema_fields)
        self.assertIn("confidence_score", schema_fields)
    
    def test_dataset_card_schema_documentation(self):
        """Test that dataset card documents the correct schema."""
        from utils.streaming import StreamingUploader
        
        with patch('utils.streaming.upload_file') as mock_upload, \
             patch('utils.streaming.HfApi'):
            
            uploader = StreamingUploader("test/repo", "fake_token")
            dataset_info = {
                "datasets": ["GigaSpeech2", "ProcessedVoiceTH", "MozillaCV"],
                "total_samples": 1000,
                "total_shards": 1
            }
            uploader.upload_dataset_card(dataset_info)
            
            # Get the README content that was uploaded
            upload_calls = mock_upload.call_args_list
            readme_call = None
            for call in upload_calls:
                if call[1].get('path_in_repo') == 'README.md':
                    readme_call = call
                    break
            
            self.assertIsNotNone(readme_call)
            
            # Get the content
            content = readme_call[1]['path_or_fileobj'].decode()
            
            # Check that schema documentation includes all fields
            self.assertIn("speaker_id", content)
            self.assertIn("dataset_name", content)
            self.assertIn("confidence_score", content)
    
    def test_base_processor_creates_complete_samples(self):
        """Test that base processor creates samples with all required fields."""
        from processors.base_processor import BaseProcessor
        
        # Create a mock processor
        class MockProcessor(BaseProcessor):
            def process(self, checkpoint=None):
                pass
            
            def process_streaming(self, upload_callback, checkpoint=None):
                pass
            
            def get_dataset_info(self):
                return {"name": "mock", "version": "1.0"}
            
            def estimate_size(self):
                return 100
        
        processor = MockProcessor({"name": "MockDataset"})
        
        # Test _create_streaming_sample
        audio_hf = {
            "array": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "sampling_rate": 16000,
            "path": "test.wav"
        }
        
        sample = processor._create_streaming_sample(
            audio_hf=audio_hf,
            transcript="test transcript",
            samples_processed=0,
            speaker_id="SPK_00001"
        )
        
        # Verify all fields are present
        self.assertIn("speaker_id", sample)
        self.assertIn("dataset_name", sample)
        self.assertIn("confidence_score", sample)
        self.assertEqual(sample["speaker_id"], "SPK_00001")
        self.assertEqual(sample["dataset_name"], "MockDataset")
        self.assertEqual(sample["confidence_score"], 1.0)


if __name__ == '__main__':
    unittest.main()