"""Integration tests for streaming mode functionality."""
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main, process_streaming_mode
from utils.streaming import StreamingUploader
import config


class TestStreamingIntegration(unittest.TestCase):
    """Test streaming mode integration with main.py"""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('main.CHECKPOINT_DIR')
    @patch('main.LOG_DIR')
    @patch('main.read_hf_token')
    @patch('main.StreamingUploader')
    @patch('main.create_processor')
    @patch('main.get_last_id')
    def test_streaming_mode_with_sample(self, mock_get_last_id, mock_create_processor, 
                                       mock_uploader_class, mock_read_token, 
                                       mock_log_dir, mock_checkpoint_dir):
        """Test streaming mode with --sample flag."""
        # Set up mocks
        mock_checkpoint_dir.return_value = self.checkpoint_dir
        mock_log_dir.return_value = self.log_dir
        mock_read_token.return_value = "test_token"
        mock_get_last_id.return_value = None
        
        # Mock uploader
        mock_uploader = MagicMock()
        mock_uploader.shard_num = 0
        mock_uploader.upload_batch.return_value = (True, "shard_0000.parquet")
        mock_uploader.upload_dataset_card.return_value = None
        mock_uploader_class.return_value = mock_uploader
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.name = "TestDataset"
        
        # Create sample data generator
        def sample_generator(checkpoint=None, sample_mode=False, sample_size=5):
            samples = []
            for i in range(sample_size if sample_mode else 100):
                samples.append({
                    "language": "th",
                    "audio": {
                        "path": f"sample_{i}.wav",
                        "bytes": b"fake_audio_data"
                    },
                    "transcript": f"Test transcript {i}",
                    "length": 1.5
                })
            return iter(samples)
        
        mock_processor.process_all_splits.side_effect = sample_generator
        mock_processor.save_streaming_checkpoint.return_value = None
        mock_create_processor.return_value = mock_processor
        
        # Create args mock
        args = MagicMock()
        args.fresh = True
        args.streaming = True
        args.sample = True
        args.sample_size = 5
        args.no_upload = False
        args.private = False
        args.append = False
        args.resume = False
        args.streaming_batch_size = 2
        args.upload_batch_size = 3
        args.no_standardization = False
        args.sample_rate = 16000
        args.no_volume_norm = False
        args.target_db = -20
        args.verbose = True
        args.no_stt = True
        args.enable_stt = False
        args.stt_batch_size = 16
        
        # Run streaming mode
        dataset_names = ["TestDataset"]
        exit_code = process_streaming_mode(args, dataset_names)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        
        # Verify processor was created with correct config
        mock_create_processor.assert_called_once()
        processor_config = mock_create_processor.call_args[0][1]
        self.assertTrue(processor_config["streaming"])
        self.assertEqual(processor_config["batch_size"], 2)
        self.assertEqual(processor_config["upload_batch_size"], 3)
        
        # Verify samples were processed
        mock_processor.process_all_splits.assert_called_once_with(
            checkpoint=None,
            sample_mode=True,
            sample_size=5
        )
        
        # Verify uploads happened (5 samples, batch size 3 = 2 uploads)
        self.assertEqual(mock_uploader.upload_batch.call_count, 2)
        
        # Verify checkpoints were saved
        self.assertEqual(mock_processor.save_streaming_checkpoint.call_count, 1)
        
        # Verify dataset card was uploaded
        mock_uploader.upload_dataset_card.assert_called_once()
    
    @patch('main.CHECKPOINT_DIR')
    @patch('main.LOG_DIR')
    @patch('main.read_hf_token')
    @patch('main.StreamingUploader')
    @patch('main.create_processor')
    @patch('main.get_last_id')
    def test_streaming_mode_without_sample(self, mock_get_last_id, mock_create_processor,
                                          mock_uploader_class, mock_read_token,
                                          mock_log_dir, mock_checkpoint_dir):
        """Test streaming mode without --sample flag (full processing)."""
        # Set up mocks
        mock_checkpoint_dir.return_value = self.checkpoint_dir
        mock_log_dir.return_value = self.log_dir
        mock_read_token.return_value = "test_token"
        mock_get_last_id.return_value = None
        
        # Mock uploader
        mock_uploader = MagicMock()
        mock_uploader.shard_num = 0
        mock_uploader.upload_batch.return_value = (True, "shard_0000.parquet")
        mock_uploader.upload_dataset_card.return_value = None
        mock_uploader_class.return_value = mock_uploader
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.name = "TestDataset"
        
        # Create larger data generator (simulating full dataset)
        def full_generator(checkpoint=None, sample_mode=False, sample_size=5):
            num_samples = 15  # More than upload batch size
            for i in range(num_samples):
                yield {
                    "language": "th",
                    "audio": {
                        "path": f"sample_{i}.wav",
                        "bytes": b"fake_audio_data"
                    },
                    "transcript": f"Test transcript {i}",
                    "length": 1.5
                }
        
        mock_processor.process_all_splits.side_effect = full_generator
        mock_processor.save_streaming_checkpoint.return_value = None
        mock_create_processor.return_value = mock_processor
        
        # Create args mock
        args = MagicMock()
        args.fresh = True
        args.streaming = True
        args.sample = False  # Not in sample mode
        args.sample_size = 5  # Should be ignored
        args.no_upload = False
        args.private = False
        args.append = False
        args.resume = False
        args.streaming_batch_size = 100
        args.upload_batch_size = 10
        args.no_standardization = False
        args.sample_rate = 16000
        args.no_volume_norm = False
        args.target_db = -20
        args.verbose = True
        args.no_stt = True
        args.enable_stt = False
        args.stt_batch_size = 16
        
        # Run streaming mode
        dataset_names = ["TestDataset"]
        exit_code = process_streaming_mode(args, dataset_names)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        
        # Verify samples were processed without sample mode
        mock_processor.process_all_splits.assert_called_once_with(
            checkpoint=None,
            sample_mode=False,
            sample_size=5  # Should be passed but ignored
        )
        
        # Verify uploads happened (15 samples, batch size 10 = 2 uploads)
        self.assertEqual(mock_uploader.upload_batch.call_count, 2)
        
        # Verify correct number of samples in each batch
        first_batch = mock_uploader.upload_batch.call_args_list[0][0][0]
        self.assertEqual(len(first_batch), 10)
        second_batch = mock_uploader.upload_batch.call_args_list[1][0][0]
        self.assertEqual(len(second_batch), 5)
        
        # Verify IDs are sequential
        all_ids = []
        for call_args in mock_uploader.upload_batch.call_args_list:
            batch = call_args[0][0]
            all_ids.extend([s["ID"] for s in batch])
        
        expected_ids = [f"S{i}" for i in range(1, 16)]
        self.assertEqual(all_ids, expected_ids)
    
    @patch('main.CHECKPOINT_DIR')
    @patch('main.LOG_DIR')
    @patch('main.read_hf_token')
    @patch('main.StreamingUploader')
    @patch('main.create_processor')
    @patch('main.get_last_id')
    def test_streaming_all_datasets(self, mock_get_last_id, mock_create_processor,
                                   mock_uploader_class, mock_read_token,
                                   mock_log_dir, mock_checkpoint_dir):
        """Test streaming mode with all datasets."""
        # Set up mocks
        mock_checkpoint_dir.return_value = self.checkpoint_dir
        mock_log_dir.return_value = self.log_dir
        mock_read_token.return_value = "test_token"
        mock_get_last_id.return_value = None
        
        # Mock uploader
        mock_uploader = MagicMock()
        mock_uploader.shard_num = 0
        mock_uploader.upload_batch.return_value = (True, "shard_0000.parquet")
        mock_uploader_class.return_value = mock_uploader
        
        # Track created processors
        created_processors = []
        
        def create_mock_processor(dataset_name, config):
            mock_processor = MagicMock()
            mock_processor.name = dataset_name
            
            # Each dataset provides 3 samples
            def dataset_generator(checkpoint=None, sample_mode=False, sample_size=5):
                for i in range(3):
                    yield {
                        "language": "th",
                        "audio": {
                            "path": f"{dataset_name}_sample_{i}.wav",
                            "bytes": b"fake_audio_data"
                        },
                        "transcript": f"{dataset_name} transcript {i}",
                        "length": 1.5
                    }
            
            mock_processor.process_all_splits.side_effect = dataset_generator
            created_processors.append(mock_processor)
            return mock_processor
        
        mock_create_processor.side_effect = create_mock_processor
        
        # Create args mock
        args = MagicMock()
        args.fresh = True
        args.streaming = True
        args.sample = True
        args.sample_size = 5
        args.no_upload = False
        args.private = False
        args.append = False
        args.resume = False
        args.streaming_batch_size = 100
        args.upload_batch_size = 5
        args.no_standardization = False
        args.sample_rate = 16000
        args.no_volume_norm = False
        args.target_db = -20
        args.verbose = True
        args.no_stt = True
        args.enable_stt = False
        args.stt_batch_size = 16
        
        # Run with all datasets
        dataset_names = list(config.DATASET_CONFIG.keys())
        exit_code = process_streaming_mode(args, dataset_names)
        
        # Verify success
        self.assertEqual(exit_code, 0)
        
        # Verify all datasets were processed (3 datasets now after VISTEC removal)
        self.assertEqual(len(created_processors), 3)
        for processor in created_processors:
            processor.process_all_splits.assert_called_once()
        
        # Verify total samples (3 datasets * 3 samples = 9)
        total_samples = 0
        for call_args in mock_uploader.upload_batch.call_args_list:
            batch = call_args[0][0]
            total_samples += len(batch)
        self.assertEqual(total_samples, 9)
    
    @patch('sys.argv', ['main.py', '--fresh', '--all', '--streaming', '--sample', '--sample-size', '5'])
    @patch('main.process_streaming_mode')
    def test_main_with_streaming_args(self, mock_process_streaming):
        """Test main() function with streaming arguments."""
        mock_process_streaming.return_value = 0
        
        # Run main
        exit_code = main()
        
        # Verify streaming mode was called
        mock_process_streaming.assert_called_once()
        args = mock_process_streaming.call_args[0][0]
        
        # Verify args
        self.assertTrue(args.streaming)
        self.assertTrue(args.sample)
        self.assertEqual(args.sample_size, 5)
        self.assertTrue(args.fresh)
        self.assertTrue(args.all)
        
        # Verify dataset names
        dataset_names = mock_process_streaming.call_args[0][1]
        self.assertEqual(set(dataset_names), set(config.DATASET_CONFIG.keys()))


if __name__ == "__main__":
    unittest.main()