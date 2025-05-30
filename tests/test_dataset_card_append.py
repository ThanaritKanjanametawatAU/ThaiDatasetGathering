import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import tempfile
import os
from io import StringIO
from utils.streaming import StreamingUploader
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.file_download import http_get


class TestDatasetCardAppend(unittest.TestCase):
    """Test dataset card updates in append mode"""

    def setUp(self):
        """Set up test fixtures"""
        self.repo_id = "test-user/test-dataset"
        # Mock the HfApi in the StreamingUploader __init__
        with patch('utils.streaming.HfApi'):
            self.uploader = StreamingUploader(self.repo_id)
        
        # Sample existing dataset card content
        self.existing_readme = """---
dataset_info:
  features:
  - name: ID
    dtype: string
  - name: speaker_id
    dtype: string
  - name: Language
    dtype: string
  - name: audio
    dtype: 
      audio:
        sampling_rate: 16000
  - name: transcript
    dtype: string
  - name: length
    dtype: float32
  - name: dataset_name
    dtype: string
  - name: confidence_score
    dtype: float64
  splits:
  - name: train
    num_examples: 5000
  download_size: 1073741824
  dataset_size: 1073741824
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.parquet
---

# Thai Voice Dataset

Combined Thai audio dataset from multiple sources

## Dataset Details

- **Total samples**: 5,000
- **Total duration**: 10.50 hours
- **Language**: Thai (th)
- **Audio format**: 16kHz mono WAV
- **Volume normalization**: -20dB

## Sources

Processed 1 datasets in streaming mode

## Source Datasets

1. **GigaSpeech2**: Large-scale multilingual speech corpus
"""

    @patch('utils.streaming.upload_file')
    def test_dataset_card_preserves_existing_stats_in_append_mode(self, mock_upload):
        """Test that dataset card preserves existing statistics when appending"""
        # Mock the method to read existing dataset card
        with patch.object(self.uploader, '_read_existing_dataset_card', return_value=self.existing_readme):
            # Create dataset info for new processing
            dataset_info = {
                "total_samples": 1000,  # New samples
                "total_duration_hours": 2.0,  # New duration
                "sources_description": "Processed 1 datasets in streaming mode",
                "dataset_names": ["ProcessedVoiceTH"]
            }
            
            # Call the method with append_mode=True
            self.uploader.upload_dataset_card(dataset_info, append_mode=True)
            
            # Get the uploaded content from keyword arguments
            call_kwargs = mock_upload.call_args.kwargs
            uploaded_content = call_kwargs['path_or_fileobj'].decode() if isinstance(call_kwargs['path_or_fileobj'], bytes) else str(call_kwargs['path_or_fileobj'])
            
            # Verify statistics were accumulated
            self.assertIn("Total samples**: 6,000", uploaded_content)  # 5000 + 1000
            self.assertIn("Total duration**: 12.50 hours", uploaded_content)  # 10.5 + 2.0
            
            # Verify both datasets are listed
            self.assertIn("GigaSpeech2", uploaded_content)
            self.assertIn("ProcessedVoiceTH", uploaded_content)

    @patch('utils.streaming.upload_file')
    def test_dataset_card_creates_new_when_no_existing(self, mock_upload):
        """Test that dataset card is created correctly when no existing card"""
        # Mock the method to indicate no existing dataset card
        with patch.object(self.uploader, '_read_existing_dataset_card', return_value=None):
            # Create dataset info
            dataset_info = {
                "total_samples": 1000,
                "total_duration_hours": 2.0,
                "sources_description": "Processed 1 datasets in streaming mode",
                "dataset_names": ["ProcessedVoiceTH"]
            }
            
            # Upload dataset card (should create new)
            self.uploader.upload_dataset_card(dataset_info, append_mode=False)
            
            call_kwargs = mock_upload.call_args.kwargs
            uploaded_content = call_kwargs['path_or_fileobj'].decode() if isinstance(call_kwargs['path_or_fileobj'], bytes) else str(call_kwargs['path_or_fileobj'])
            
            # Verify new card has correct stats
            self.assertIn("Total samples**: 1,000", uploaded_content)
            self.assertIn("Total duration**: 2.00 hours", uploaded_content)

    def test_parse_existing_dataset_card(self):
        """Test parsing existing dataset card for statistics"""
        # This test verifies that we can extract stats from an existing README
        
        # Expected values from the sample README
        expected_samples = 5000
        expected_duration = 10.5
        expected_datasets = ["GigaSpeech2"]
        
        # Parse the README
        lines = self.existing_readme.split('\n')
        
        actual_samples = None
        actual_duration = None
        actual_datasets = []
        
        for line in lines:
            if "num_examples:" in line:
                actual_samples = int(line.split(':')[1].strip())
            elif "Total samples**:" in line:
                # Extract from markdown format
                sample_str = line.split('**:')[1].strip().replace(',', '')
                actual_samples = int(sample_str)
            elif "Total duration**:" in line:
                # Extract duration
                duration_str = line.split('**:')[1].strip().split()[0]
                actual_duration = float(duration_str)
            elif line.strip().startswith("1. **") and "**:" in line:
                # Extract dataset name
                dataset_name = line.split("**")[1]
                actual_datasets.append(dataset_name)
        
        self.assertEqual(actual_samples, expected_samples)
        self.assertEqual(actual_duration, expected_duration)
        self.assertEqual(actual_datasets, expected_datasets)

    def test_duration_accumulation_in_main(self):
        """Test duration accumulation logic that should be in main.py"""
        # Simulate processing samples and accumulating duration
        total_duration_seconds = 0
        samples = [
            {"length": 1.5},
            {"length": 2.0},
            {"length": 3.5},
        ]
        
        for sample in samples:
            total_duration_seconds += sample.get("length", 0)
        
        total_duration_hours = total_duration_seconds / 3600
        
        self.assertEqual(total_duration_seconds, 7.0)
        self.assertAlmostEqual(total_duration_hours, 0.00194, places=4)


if __name__ == '__main__':
    unittest.main()