"""
Test suite to verify issue #8 fix: Dataset card should correctly update when using --append flag
Commit c4dccd486ca2c6eeedd41e69a95cd34deaf72652 claims to fix this issue.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import tempfile
import os
import shutil
from io import BytesIO
from utils.streaming import StreamingUploader
from huggingface_hub import HfApi
import numpy as np


class TestIssue8DatasetCardAppendFix(unittest.TestCase):
    """Comprehensive tests to verify issue #8 fix"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.repo_id = "test-user/test-dataset"
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock existing dataset card
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
        
    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_dataset_card_append_accumulates_stats(self):
        """Test that append mode correctly accumulates statistics from existing card"""
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id, append_mode=True)
            
            # Mock reading existing dataset card
            with patch.object(uploader, '_read_existing_dataset_card', return_value=self.existing_readme):
                # Mock upload_file
                with patch('utils.streaming.upload_file') as mock_upload:
                    # New data to append
                    dataset_info = {
                        "total_samples": 3000,  # New samples
                        "total_duration_hours": 6.25,  # New duration
                        "sources_description": "Processed 1 datasets in streaming mode",
                        "dataset_names": ["ProcessedVoiceTH"]
                    }
                    
                    # Upload with append mode
                    uploader.upload_dataset_card(dataset_info, append_mode=True)
                    
                    # Verify upload was called
                    self.assertTrue(mock_upload.called)
                    
                    # Get uploaded content
                    call_kwargs = mock_upload.call_args.kwargs
                    uploaded_content = call_kwargs['path_or_fileobj'].decode('utf-8')
                    
                    # Verify accumulated statistics
                    self.assertIn("num_examples: 8000", uploaded_content)  # 5000 + 3000
                    self.assertIn("**Total samples**: 8,000", uploaded_content)
                    self.assertIn("**Total duration**: 16.75 hours", uploaded_content)  # 10.50 + 6.25
                    
                    # Verify both datasets are listed
                    self.assertIn("GigaSpeech2", uploaded_content)
                    self.assertIn("ProcessedVoiceTH", uploaded_content)
                    self.assertIn("Processed 2 datasets in streaming mode", uploaded_content)
    
    def test_dataset_card_append_with_multiple_datasets(self):
        """Test appending multiple datasets at once"""
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id, append_mode=True)
            
            with patch.object(uploader, '_read_existing_dataset_card', return_value=self.existing_readme):
                with patch('utils.streaming.upload_file') as mock_upload:
                    # Append multiple datasets
                    dataset_info = {
                        "total_samples": 7500,
                        "total_duration_hours": 15.0,
                        "sources_description": "Processed 2 datasets in streaming mode",
                        "dataset_names": ["ProcessedVoiceTH", "MozillaCommonVoice"]
                    }
                    
                    uploader.upload_dataset_card(dataset_info, append_mode=True)
                    
                    uploaded_content = mock_upload.call_args.kwargs['path_or_fileobj'].decode('utf-8')
                    
                    # Should have 3 unique datasets total
                    self.assertIn("GigaSpeech2", uploaded_content)
                    self.assertIn("ProcessedVoiceTH", uploaded_content)
                    self.assertIn("MozillaCommonVoice", uploaded_content)
                    self.assertIn("Processed 3 datasets in streaming mode", uploaded_content)
                    
                    # Stats should accumulate
                    self.assertIn("num_examples: 12500", uploaded_content)  # 5000 + 7500
                    self.assertIn("**Total duration**: 25.50 hours", uploaded_content)  # 10.50 + 15.0
    
    def test_dataset_card_no_append_overwrites(self):
        """Test that non-append mode overwrites existing stats and ignores existing card"""
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id, append_mode=False)
            
            # Even though we mock the method to return existing content,
            # it should not be called when append_mode=False
            mock_read = Mock(return_value=self.existing_readme)
            with patch.object(uploader, '_read_existing_dataset_card', mock_read):
                with patch('utils.streaming.upload_file') as mock_upload:
                    dataset_info = {
                        "total_samples": 1000,
                        "total_duration_hours": 2.0,
                        "sources_description": "Processed 1 datasets in streaming mode",
                        "dataset_names": ["ProcessedVoiceTH"]
                    }
                    
                    uploader.upload_dataset_card(dataset_info, append_mode=False)
                    
                    # Verify _read_existing_dataset_card was NOT called when append_mode=False
                    mock_read.assert_not_called()
                    
                    uploaded_content = mock_upload.call_args.kwargs['path_or_fileobj'].decode('utf-8')
                    
                    # Should NOT accumulate - should show only new values
                    self.assertIn("num_examples: 1000", uploaded_content)
                    self.assertIn("**Total samples**: 1,000", uploaded_content)
                    self.assertIn("**Total duration**: 2.00 hours", uploaded_content)
                    
                    # Should only have new dataset in the sources section
                    # Check that GigaSpeech2 is not in the source datasets list
                    sources_section = uploaded_content.split("## Source Datasets")[1].split("## Usage")[0]
                    self.assertNotIn("**GigaSpeech2**:", sources_section)
                    self.assertIn("**ProcessedVoiceTH**:", sources_section)
                    
                    # Verify it's only listing 1 dataset
                    self.assertIn("Processed 1 datasets in streaming mode", uploaded_content)
    
    def test_parse_existing_stats_method(self):
        """Test the _parse_existing_stats method directly"""
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id)
            
            stats = uploader._parse_existing_stats(self.existing_readme)
            
            self.assertEqual(stats['total_samples'], 5000)
            self.assertEqual(stats['total_duration_hours'], 10.5)
            self.assertEqual(stats['existing_datasets'], ['GigaSpeech2'])
    
    def test_parse_existing_stats_with_multiple_datasets(self):
        """Test parsing stats with multiple datasets"""
        readme_with_multiple = self.existing_readme + """
2. **ProcessedVoiceTH**: Thai voice dataset with processed audio
3. **MozillaCommonVoice**: Mozilla Common Voice Thai dataset
"""
        
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id)
            
            stats = uploader._parse_existing_stats(readme_with_multiple)
            
            self.assertEqual(stats['existing_datasets'], ['GigaSpeech2', 'ProcessedVoiceTH', 'MozillaCommonVoice'])
    
    def test_read_existing_dataset_card_file_not_found(self):
        """Test behavior when no existing dataset card exists"""
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id)
            
            with patch('huggingface_hub.hf_hub_download', side_effect=Exception("File not found")):
                result = uploader._read_existing_dataset_card()
                self.assertIsNone(result)
    
    def test_dataset_card_duplicate_dataset_handling(self):
        """Test that duplicate dataset names are handled correctly"""
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id, append_mode=True)
            
            with patch.object(uploader, '_read_existing_dataset_card', return_value=self.existing_readme):
                with patch('utils.streaming.upload_file') as mock_upload:
                    # Try to append same dataset again
                    dataset_info = {
                        "total_samples": 2000,
                        "total_duration_hours": 4.0,
                        "sources_description": "Processed 1 datasets in streaming mode",
                        "dataset_names": ["GigaSpeech2"]  # Same as existing
                    }
                    
                    uploader.upload_dataset_card(dataset_info, append_mode=True)
                    
                    uploaded_content = mock_upload.call_args.kwargs['path_or_fileobj'].decode('utf-8')
                    
                    # Should not duplicate dataset name
                    gigaspeech_count = uploaded_content.count("**GigaSpeech2**:")
                    self.assertEqual(gigaspeech_count, 1)
                    
                    # But stats should still accumulate
                    self.assertIn("num_examples: 7000", uploaded_content)  # 5000 + 2000
                    self.assertIn("**Total duration**: 14.50 hours", uploaded_content)  # 10.50 + 4.0
    
    def test_empty_existing_readme_handling(self):
        """Test handling of empty existing README"""
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id, append_mode=True)
            
            with patch.object(uploader, '_read_existing_dataset_card', return_value=""):
                with patch('utils.streaming.upload_file') as mock_upload:
                    dataset_info = {
                        "total_samples": 1000,
                        "total_duration_hours": 2.0,
                        "sources_description": "Processed 1 datasets in streaming mode",
                        "dataset_names": ["ProcessedVoiceTH"]
                    }
                    
                    uploader.upload_dataset_card(dataset_info, append_mode=True)
                    
                    uploaded_content = mock_upload.call_args.kwargs['path_or_fileobj'].decode('utf-8')
                    
                    # Should create new card with provided stats
                    self.assertIn("num_examples: 1000", uploaded_content)
                    self.assertIn("**Total samples**: 1,000", uploaded_content)
                    self.assertIn("**Total duration**: 2.00 hours", uploaded_content)
    
    def test_malformed_existing_stats_handling(self):
        """Test handling of malformed stats in existing README"""
        malformed_readme = """---
dataset_info:
  splits:
  - name: train
    num_examples: not_a_number
---

# Thai Voice Dataset

## Dataset Details

- **Total samples**: invalid
- **Total duration**: also invalid hours
"""
        
        with patch('utils.streaming.HfApi'):
            uploader = StreamingUploader(self.repo_id, append_mode=True)
            
            with patch.object(uploader, '_read_existing_dataset_card', return_value=malformed_readme):
                with patch('utils.streaming.upload_file') as mock_upload:
                    dataset_info = {
                        "total_samples": 1000,
                        "total_duration_hours": 2.0,
                        "sources_description": "Processed 1 datasets in streaming mode",
                        "dataset_names": ["ProcessedVoiceTH"]
                    }
                    
                    uploader.upload_dataset_card(dataset_info, append_mode=True)
                    
                    uploaded_content = mock_upload.call_args.kwargs['path_or_fileobj'].decode('utf-8')
                    
                    # Should use only new stats since old ones are invalid
                    self.assertIn("num_examples: 1000", uploaded_content)
                    self.assertIn("**Total samples**: 1,000", uploaded_content)
                    self.assertIn("**Total duration**: 2.00 hours", uploaded_content)


if __name__ == '__main__':
    unittest.main()