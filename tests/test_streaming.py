"""
Unit tests for streaming mode functionality.
"""

import unittest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datasets import IterableDataset

from processors.base_processor import BaseProcessor
from utils.streaming import StreamingUploader, StreamingBatchProcessor, create_streaming_dataset
from config import DATASET_CONFIG, STREAMING_CONFIG


class TestStreamingMode(unittest.TestCase):
    """Test streaming mode functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Mock config
        self.test_config = {
            "name": "TestDataset",
            "source": "test/dataset",
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.temp_dir,
            "streaming": True,
            "batch_size": 10,
            "upload_batch_size": 20
        }
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_streaming_checkpoint_save_load(self):
        """Test saving and loading streaming checkpoints."""
        # Create a mock processor
        class MockProcessor(BaseProcessor):
            def process(self, checkpoint=None, sample_mode=False, sample_size=5):
                return []
            
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                return []
            
            def get_dataset_info(self):
                return {"name": "Test"}
            
            def estimate_size(self):
                return 100
        
        processor = MockProcessor(self.test_config)
        
        # Save checkpoint
        checkpoint_file = processor.save_streaming_checkpoint(
            shard_num=5,
            samples_processed=1000,
            last_sample_id="S1000",
            dataset_specific_data={"custom": "data"}
        )
        
        self.assertTrue(os.path.exists(checkpoint_file))
        
        # Load checkpoint
        loaded_data = processor.load_streaming_checkpoint(checkpoint_file)
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["shard_num"], 5)
        self.assertEqual(loaded_data["samples_processed"], 1000)
        self.assertEqual(loaded_data["last_sample_id"], "S1000")
        self.assertEqual(loaded_data["dataset_specific"]["custom"], "data")
    
    @patch('utils.streaming.HfApi')
    def test_streaming_uploader(self, mock_hf_api):
        """Test StreamingUploader functionality."""
        # Mock HfApi
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance
        
        uploader = StreamingUploader(
            repo_id="test/repo",
            token="test_token",
            private=False
        )
        
        # Test batch upload
        samples = [
            {"ID": "S1", "Language": "th", "audio": {"array": [0.1], "sampling_rate": 16000}, "transcript": "test1", "length": 1.0},
            {"ID": "S2", "Language": "th", "audio": {"array": [0.2], "sampling_rate": 16000}, "transcript": "test2", "length": 2.0}
        ]
        
        with patch('utils.streaming.Dataset') as mock_dataset:
            with patch('utils.streaming.upload_file') as mock_upload:
                mock_dataset.from_list.return_value.to_parquet = MagicMock()
                
                success, shard_name = uploader.upload_batch(samples)
                
                self.assertTrue(success)
                self.assertEqual(shard_name, "shard_00000.parquet")
                mock_dataset.from_list.assert_called_once()
                mock_upload.assert_called_once()
    
    def test_streaming_batch_processor(self):
        """Test StreamingBatchProcessor functionality."""
        processor = StreamingBatchProcessor(batch_size=3, checkpoint_dir=self.checkpoint_dir)
        
        # Create mock dataset iterator
        def mock_iterator():
            for i in range(10):
                yield {"id": i, "text": f"sample {i}"}
        
        # Mock process function
        def mock_process_fn(sample):
            return {"ID": f"S{sample['id']}", "processed": True}
        
        # Process with checkpoints
        batches = list(processor.process_with_checkpoints(
            mock_iterator(),
            mock_process_fn
        ))
        
        # Should have 4 batches (3, 3, 3, 1)
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 3)
        self.assertEqual(len(batches[2]), 3)
        self.assertEqual(len(batches[3]), 1)
    
    @patch('processors.base_processor.get_audio_length')
    @patch('processors.gigaspeech2.load_dataset')
    def test_gigaspeech2_streaming(self, mock_load_dataset, mock_get_audio_length):
        """Test GigaSpeech2 processor in streaming mode."""
        from processors.gigaspeech2 import GigaSpeech2Processor
        
        # Mock audio length calculation
        mock_get_audio_length.return_value = 2.5
        
        # Mock streaming dataset
        mock_samples = [
            {
                "audio": {"array": [0.1, 0.2], "sampling_rate": 16000},
                "text": "สวัสดี",
                "language": "th"
            },
            {
                "audio": {"array": [0.3, 0.4], "sampling_rate": 16000},
                "text": "ขอบคุณ",
                "language": "th"
            }
        ]
        
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_samples))
        mock_load_dataset.return_value = mock_dataset
        
        processor = GigaSpeech2Processor(self.test_config)
        
        # Process in streaming mode
        results = list(processor.process_streaming(sample_mode=True, sample_size=2))
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["Language"], "th")
        self.assertEqual(results[0]["transcript"], "สวัสดี")
        self.assertEqual(results[1]["transcript"], "ขอบคุณ")
    
    @patch('processors.base_processor.get_audio_length')
    @patch('processors.mozilla_cv.load_dataset')
    def test_mozilla_cv_streaming(self, mock_load_dataset, mock_get_audio_length):
        """Test Mozilla Common Voice processor in streaming mode."""
        from processors.mozilla_cv import MozillaCommonVoiceProcessor
        
        # Mock audio length calculation
        mock_get_audio_length.return_value = 2.5
        
        # Mock streaming dataset
        mock_samples = [
            {
                "audio": {"array": [0.1, 0.2], "sampling_rate": 16000},
                "sentence": "ทดสอบ",
            }
        ]
        
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_samples))
        mock_load_dataset.return_value = mock_dataset
        
        processor = MozillaCommonVoiceProcessor(self.test_config)
        
        # Process in streaming mode
        results = list(processor.process_streaming(sample_mode=True, sample_size=1))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["Language"], "th")
        self.assertEqual(results[0]["transcript"], "ทดสอบ")
    
    def test_streaming_resume_capability(self):
        """Test ability to resume from checkpoint."""
        processor = StreamingBatchProcessor(batch_size=2, checkpoint_dir=self.checkpoint_dir)
        
        # Save a checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir, "test_checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump({"samples_processed": 5}, f)
        
        # Create mock dataset iterator with 10 items
        def mock_iterator():
            for i in range(10):
                yield {"id": i, "text": f"sample {i}"}
        
        # Mock process function
        def mock_process_fn(sample):
            return {"ID": f"S{sample['id']}", "processed": True}
        
        # Process with checkpoint (should skip first 5)
        batches = list(processor.process_with_checkpoints(
            mock_iterator(),
            mock_process_fn,
            checkpoint_file
        ))
        
        # Should have 3 batches starting from sample 5
        # Batches: [5,6], [7,8], [9]
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0]["ID"], "S5")
        self.assertEqual(batches[0][1]["ID"], "S6")
        self.assertEqual(batches[2][0]["ID"], "S9")
    
    @patch('utils.streaming.load_dataset')
    def test_create_streaming_dataset(self, mock_load_dataset):
        """Test creating combined streaming dataset."""
        # Mock datasets
        mock_ds1 = MagicMock(spec=IterableDataset)
        mock_ds2 = MagicMock(spec=IterableDataset)
        
        mock_load_dataset.side_effect = [mock_ds1, mock_ds2]
        
        configs = [
            {"name": "Dataset1", "source": "source1"},
            {"name": "Dataset2", "source": "source2"}
        ]
        
        with patch('datasets.interleave_datasets') as mock_interleave:
            mock_interleave.return_value = MagicMock(spec=IterableDataset)
            
            result = create_streaming_dataset(configs, interleave=True)
            
            self.assertEqual(mock_load_dataset.call_count, 2)
            mock_interleave.assert_called_once()


class TestStreamingIntegration(unittest.TestCase):
    """Integration tests for streaming mode."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('main.read_hf_token')
    @patch('main.StreamingUploader')
    @patch('main.create_processor')
    def test_main_streaming_mode(self, mock_create_processor, mock_uploader_class, mock_read_token):
        """Test main.py streaming mode execution."""
        from main import process_streaming_mode
        
        # Mock token
        mock_read_token.return_value = "test_token"
        
        # Mock uploader
        mock_uploader = MagicMock()
        mock_uploader.shard_num = 0
        mock_uploader.upload_batch.return_value = (True, "shard_00000.parquet")
        mock_uploader_class.return_value = mock_uploader
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.name = "TestProcessor"
        mock_processor.process_all_splits.return_value = iter([
            {"ID": "temp_1", "Language": "th", "audio": {}, "transcript": "test", "length": 1.0,
             "dataset_name": "TestDataset", "confidence_score": 1.0}
        ])
        mock_processor.save_streaming_checkpoint = MagicMock()
        mock_create_processor.return_value = mock_processor
        
        # Create mock args
        class Args:
            no_upload = False
            append = False
            private = False
            streaming_batch_size = 10
            upload_batch_size = 10
            sample = False
            sample_size = 5
            resume = False
            verbose = False
            no_standardization = False
            sample_rate = 16000
            target_db = -20.0
            no_volume_norm = False
            no_stt = True  # Add missing attribute
            enable_stt = False
            stt_batch_size = 16
        
        args = Args()
        dataset_names = ["TestDataset"]
        
        # Run streaming mode
        result = process_streaming_mode(args, dataset_names)
        
        self.assertEqual(result, 0)  # SUCCESS
        mock_processor.process_all_splits.assert_called_once()
        mock_uploader.upload_dataset_card.assert_called_once()


if __name__ == '__main__':
    unittest.main()