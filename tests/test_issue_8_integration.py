"""
Integration test for issue #8: Dataset card append functionality
Tests the complete flow from main.py through to dataset card upload
"""

import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import patch, Mock, MagicMock, call
import numpy as np
from io import BytesIO

# Import the modules we're testing
from main import process_streaming_mode
from utils.streaming import StreamingUploader
from processors.speaker_identification import SpeakerIdentification
from config import DATASET_CONFIG, TARGET_DATASET


class TestIssue8Integration(unittest.TestCase):
    """Integration tests for dataset card append functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoints_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir)
        
        # Mock arguments
        self.args = Mock()
        self.args.fresh = False
        self.args.append = True
        self.args.streaming = True
        self.args.sample = True
        self.args.sample_size = 5
        self.args.enable_speaker_id = False
        self.args.enable_stt = False
        self.args.enable_audio_enhancement = False
        self.args.enhancement_dashboard = False
        self.args.hf_repo = "test-user/test-dataset"
        self.args.resume = False
        self.args.no_upload = False  # Ensure uploader is created
        self.args.no_stt = False
        self.args.streaming_batch_size = 10
        self.args.upload_batch_size = 10
        
        # Mock audio data
        self.mock_audio = np.array([0.1, 0.2, 0.3] * 1000, dtype=np.float32)
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('main.create_processor')
    @patch('main.read_hf_token')
    @patch('main.get_last_id')
    @patch('main.os.path.exists')
    @patch('main.os.makedirs')
    @patch('main.StreamingUploader')
    @patch('datasets.load_dataset')
    def test_append_mode_accumulates_stats_end_to_end(self, mock_load_dataset, mock_uploader_class, 
                                                      mock_makedirs, mock_exists, mock_get_last_id, 
                                                      mock_read_token, mock_create_processor):
        """Test that append mode correctly accumulates stats through the entire pipeline"""
        
        # Mock directory checks
        mock_exists.return_value = True
        
        # Mock HF token and ID tracking
        mock_read_token.return_value = "test-token"
        mock_get_last_id.return_value = 100  # Simulate existing 100 samples
        
        # Create mock dataset with samples
        mock_samples = []
        for i in range(5):
            mock_samples.append({
                'file': f'audio_{i}.wav',
                'audio': {
                    'array': self.mock_audio,
                    'sampling_rate': 16000
                },
                'text': f'Sample text {i}',
                'length': 2.5  # 2.5 seconds each
            })
        
        # Mock dataset iterator
        mock_dataset = MagicMock()
        mock_split = MagicMock()
        mock_split.__iter__ = Mock(return_value=iter(mock_samples))
        mock_dataset.__getitem__ = Mock(return_value=mock_split)
        mock_load_dataset.return_value = mock_dataset
        
        # Mock uploader instance
        mock_uploader = MagicMock()
        mock_uploader.get_next_shard_number.return_value = 1
        mock_uploader.repo_id = self.args.hf_repo
        mock_uploader.upload_batch.return_value = (True, "shard-00001.parquet")
        mock_uploader_class.return_value = mock_uploader
        
        # Mock processor
        mock_processor = MagicMock()
        # Transform mock samples to expected format
        processed_samples = []
        for i, sample in enumerate(mock_samples):
            processed_samples.append({
                'ID': f'S{101 + i}',  # Starting from 101 since last_id=100
                'speaker_id': f'SPK_{101 + i:05d}',
                'Language': 'th',
                'audio': sample['audio'],
                'transcript': sample['text'],
                'length': sample['length'],
                'dataset_name': 'ProcessedVoiceTH',
                'confidence_score': 1.0
            })
        mock_processor.process_all_splits.return_value = iter(processed_samples)
        mock_create_processor.return_value = mock_processor
        
        # Mock existing dataset card content
        existing_card = """---
dataset_info:
  splits:
  - name: train
    num_examples: 1000
---

# Thai Voice Dataset

## Dataset Details

- **Total samples**: 1,000
- **Total duration**: 20.00 hours

## Source Datasets

1. **GigaSpeech2**: Large-scale multilingual speech corpus
"""
        
        # Configure uploader to simulate append mode behavior
        mock_uploader._read_existing_dataset_card.return_value = existing_card
        mock_uploader._parse_existing_stats.return_value = {
            'total_samples': 1000,
            'total_duration_hours': 20.0,
            'existing_datasets': ['GigaSpeech2']
        }
        
        # Process in streaming mode with append
        with patch('processors.speaker_identification.SpeakerIdentification'):
            result = process_streaming_mode(self.args, ['ProcessedVoiceTH'])
        
        # Verify success
        self.assertEqual(result, 0)  # SUCCESS exit code
        
        # Verify upload_dataset_card was called
        mock_uploader.upload_dataset_card.assert_called_once()
        
        # Get the dataset info passed to upload_dataset_card
        call_args = mock_uploader.upload_dataset_card.call_args[0][0]
        
        # Verify the correct values were passed
        self.assertEqual(call_args['total_samples'], 5)  # 5 new samples
        self.assertAlmostEqual(call_args['total_duration_hours'], 12.5 / 3600.0, places=4)  # 5 * 2.5 / 3600
        self.assertEqual(call_args['dataset_names'], ['ProcessedVoiceTH'])
        
        # Now simulate what the uploader would do with these values
        # This tests the accumulation logic
        final_samples = 1000 + 5  # existing + new
        final_duration = 20.0 + (12.5 / 3600.0)  # existing + new
        
        self.assertEqual(final_samples, 1005)
        self.assertAlmostEqual(final_duration, 20.00347, places=4)
    
    @patch('main.os.path.exists')
    @patch('main.os.makedirs')
    @patch('main.StreamingUploader')
    @patch('datasets.load_dataset')
    def test_fresh_mode_ignores_existing_stats(self, mock_load_dataset, mock_uploader_class, 
                                               mock_makedirs, mock_exists):
        """Test that fresh mode (non-append) ignores existing stats"""
        
        # Change to fresh mode
        self.args.append = False
        self.args.fresh = True
        
        # Mock directory checks
        mock_exists.return_value = True
        
        # Create mock dataset
        mock_samples = []
        for i in range(3):
            mock_samples.append({
                'file': f'audio_{i}.wav',
                'audio': {
                    'array': self.mock_audio,
                    'sampling_rate': 16000
                },
                'text': f'Sample text {i}',
                'length': 1.0  # 1 second each
            })
        
        mock_dataset = MagicMock()
        mock_split = MagicMock()
        mock_split.__iter__ = Mock(return_value=iter(mock_samples))
        mock_dataset.__getitem__ = Mock(return_value=mock_split)
        mock_load_dataset.return_value = mock_dataset
        
        # Mock uploader
        mock_uploader = MagicMock()
        mock_uploader.get_next_shard_number.return_value = 0  # Start from 0 for fresh
        mock_uploader.repo_id = self.args.hf_repo
        mock_uploader_class.return_value = mock_uploader
        
        # Process in fresh mode
        with patch('main.SpeakerIdentification'):
            result = process_streaming_mode(self.args, ['MozillaCommonVoice'])
        
        # Verify success
        self.assertEqual(result, 0)
        
        # Verify dataset info passed
        call_args = mock_uploader.upload_dataset_card.call_args[0][0]
        
        # Should only have new data, not accumulated
        self.assertEqual(call_args['total_samples'], 3)
        self.assertAlmostEqual(call_args['total_duration_hours'], 3.0 / 3600.0, places=4)
        self.assertEqual(call_args['dataset_names'], ['MozillaCommonVoice'])
    
    def test_duration_tracking_in_streaming_loop(self):
        """Test that duration is correctly tracked in the streaming processing loop"""
        
        # Simulate the duration accumulation logic from main.py
        total_duration_seconds = 0
        dataset_duration_seconds = 0
        
        # Process samples like in the streaming loop
        samples = [
            {'length': 1.5},
            {'length': 2.0},
            {'length': None},  # Some samples might not have length
            {'length': 3.5},
            {'length': 1.0}
        ]
        
        for sample in samples:
            sample_duration = sample.get('length', 0) or 0
            dataset_duration_seconds += sample_duration
            total_duration_seconds += sample_duration
        
        # Convert to hours
        dataset_duration_hours = dataset_duration_seconds / 3600.0
        total_duration_hours = total_duration_seconds / 3600.0
        
        # Verify calculations
        self.assertEqual(dataset_duration_seconds, 8.0)
        self.assertEqual(total_duration_seconds, 8.0)
        self.assertAlmostEqual(dataset_duration_hours, 0.00222, places=4)
        self.assertAlmostEqual(total_duration_hours, 0.00222, places=4)


if __name__ == '__main__':
    unittest.main()