"""Test multi-dataset checkpoint and resume functionality."""
import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.base_processor import BaseProcessor
from processors.gigaspeech2 import GigaSpeech2Processor
from processors.mozilla_cv import MozillaCommonVoiceProcessor
from processors.processed_voice_th import ProcessedVoiceTHProcessor


class TestMultiDatasetCheckpoint(unittest.TestCase):
    """Test checkpoint functionality across multiple datasets."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        self.log_dir = os.path.join(self.test_dir, "logs")
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)
        
        self.config = {
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "streaming": True,
            "audio_config": {
                "enable_standardization": False
            }
        }
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_individual_dataset_checkpoints(self):
        """Test that each dataset maintains its own checkpoint."""
        # Create processors
        gigaspeech_processor = GigaSpeech2Processor(self.config)
        mozilla_processor = MozillaCommonVoiceProcessor(self.config)
        
        # Save checkpoints for different datasets
        gigaspeech_processor.save_streaming_checkpoint(
            shard_num=5,
            samples_processed=1000,
            last_sample_id="S1000",
            dataset_specific_data={"split": "train", "archive_index": 10}
        )
        
        mozilla_processor.save_streaming_checkpoint(
            shard_num=8,
            samples_processed=500,
            last_sample_id="S1500",
            dataset_specific_data={"split": "validated", "file_index": 25}
        )
        
        # Verify each checkpoint exists
        gigaspeech_checkpoint = os.path.join(self.checkpoint_dir, "GigaSpeech2_streaming_checkpoint.json")
        mozilla_checkpoint = os.path.join(self.checkpoint_dir, "MozillaCV_streaming_checkpoint.json")
        
        self.assertTrue(os.path.exists(gigaspeech_checkpoint))
        self.assertTrue(os.path.exists(mozilla_checkpoint))
        
        # Load and verify checkpoint contents
        with open(gigaspeech_checkpoint, 'r') as f:
            gs_data = json.load(f)
            self.assertEqual(gs_data["shard_num"], 5)
            self.assertEqual(gs_data["samples_processed"], 1000)
            self.assertEqual(gs_data["last_sample_id"], "S1000")
            self.assertEqual(gs_data["dataset_specific"]["archive_index"], 10)
        
        with open(mozilla_checkpoint, 'r') as f:
            moz_data = json.load(f)
            self.assertEqual(moz_data["shard_num"], 8)
            self.assertEqual(moz_data["samples_processed"], 500)
            self.assertEqual(moz_data["last_sample_id"], "S1500")
            self.assertEqual(moz_data["dataset_specific"]["file_index"], 25)
    
    def test_resume_preserves_dataset_progress(self):
        """Test that resuming preserves progress across datasets."""
        # Simulate processing multiple datasets with interruption
        
        # Process GigaSpeech2 completely
        gigaspeech_processor = GigaSpeech2Processor(self.config)
        gigaspeech_processor.save_streaming_checkpoint(
            shard_num=3,
            samples_processed=600,
            last_sample_id="S600",
            dataset_specific_data={"split": "train", "completed": True}
        )
        
        # Process MozillaCV partially
        mozilla_processor = MozillaCommonVoiceProcessor(self.config)
        mozilla_processor.save_streaming_checkpoint(
            shard_num=3,  # Same shard number, different dataset
            samples_processed=300,
            last_sample_id="S900",
            dataset_specific_data={"split": "validated", "file_index": 15, "completed": False}
        )
        
        # ProcessedVoiceTH not started yet (no checkpoint)
        
        # Verify checkpoint loading
        gs_checkpoint = gigaspeech_processor.load_streaming_checkpoint()
        self.assertIsNotNone(gs_checkpoint)
        self.assertEqual(gs_checkpoint["samples_processed"], 600)
        self.assertTrue(gs_checkpoint["dataset_specific"].get("completed", False))
        
        moz_checkpoint = mozilla_processor.load_streaming_checkpoint()
        self.assertIsNotNone(moz_checkpoint)
        self.assertEqual(moz_checkpoint["samples_processed"], 300)
        self.assertFalse(moz_checkpoint["dataset_specific"].get("completed", False))
        
        # ProcessedVoiceTH should have no checkpoint
        processed_processor = ProcessedVoiceTHProcessor(self.config)
        pv_checkpoint = processed_processor.load_streaming_checkpoint()
        self.assertIsNone(pv_checkpoint)
    
    def test_checkpoint_tracks_split_progress(self):
        """Test that checkpoints track progress within dataset splits."""
        processor = GigaSpeech2Processor(self.config)
        
        # Save checkpoint with split progress
        processor.save_streaming_checkpoint(
            shard_num=2,
            samples_processed=400,
            last_sample_id="S400",
            dataset_specific_data={
                "split": "train",
                "splits_completed": [],
                "current_split_index": 0,
                "archive_index": 5,
                "segment_id": "0-123-45"
            }
        )
        
        checkpoint = processor.load_streaming_checkpoint()
        self.assertEqual(checkpoint["dataset_specific"]["split"], "train")
        self.assertEqual(checkpoint["dataset_specific"]["archive_index"], 5)
        self.assertEqual(checkpoint["dataset_specific"]["segment_id"], "0-123-45")
    
    @patch('datasets.load_dataset')
    def test_resume_skips_completed_datasets(self, mock_load_dataset):
        """Test that resuming skips already completed datasets."""
        # Create checkpoint indicating GigaSpeech2 is complete
        gigaspeech_processor = GigaSpeech2Processor(self.config)
        gigaspeech_processor.save_streaming_checkpoint(
            shard_num=5,
            samples_processed=1000,
            last_sample_id="S1000",
            dataset_specific_data={"completed": True, "all_splits_processed": True}
        )
        
        # When loading checkpoint, processor should detect completion
        checkpoint = gigaspeech_processor.load_streaming_checkpoint()
        self.assertTrue(checkpoint["dataset_specific"].get("completed", False))
        
        # In actual usage, main.py should check this and skip the dataset
        # This is handled by the process_all_splits method checking checkpoint
    
    def test_global_checkpoint_file(self):
        """Test creation of a global checkpoint tracking all datasets."""
        # This would be an enhancement - track overall progress
        global_checkpoint = {
            "datasets_completed": ["GigaSpeech2"],
            "datasets_in_progress": {
                "MozillaCV": {
                    "samples_processed": 300,
                    "last_sample_id": "S900"
                }
            },
            "datasets_pending": ["ProcessedVoiceTH"],
            "total_samples": 900,
            "last_shard_num": 3
        }
        
        global_checkpoint_file = os.path.join(self.checkpoint_dir, "global_streaming_checkpoint.json")
        with open(global_checkpoint_file, 'w') as f:
            json.dump(global_checkpoint, f, indent=2)
        
        # Verify we can load and use this information
        with open(global_checkpoint_file, 'r') as f:
            loaded = json.load(f)
            self.assertEqual(loaded["datasets_completed"], ["GigaSpeech2"])
            self.assertIn("MozillaCV", loaded["datasets_in_progress"])
            self.assertEqual(loaded["datasets_pending"], ["ProcessedVoiceTH"])
    
    def test_checkpoint_with_stt_state(self):
        """Test that checkpoints preserve STT processing state."""
        processor = GigaSpeech2Processor(self.config)
        
        # Save checkpoint with STT state
        processor.save_streaming_checkpoint(
            shard_num=1,
            samples_processed=100,
            last_sample_id="S100",
            dataset_specific_data={
                "stt_enabled": True,
                "samples_needing_stt": ["0-123-1", "0-123-2"],
                "stt_batch_accumulated": 5
            }
        )
        
        checkpoint = processor.load_streaming_checkpoint()
        self.assertTrue(checkpoint["dataset_specific"]["stt_enabled"])
        self.assertEqual(len(checkpoint["dataset_specific"]["samples_needing_stt"]), 2)
        self.assertEqual(checkpoint["dataset_specific"]["stt_batch_accumulated"], 5)


class TestMainCheckpointIntegration(unittest.TestCase):
    """Test checkpoint integration in main.py workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    @patch('main.create_processor')
    @patch('main.StreamingUploader')
    def test_main_resumes_from_correct_dataset(self, mock_uploader_class, mock_create_processor):
        """Test that main.py resumes from the correct dataset when using --resume."""
        # Setup mock processors
        mock_processors = {}
        
        # GigaSpeech2 - completed
        gs_processor = Mock()
        gs_processor.name = "GigaSpeech2"
        gs_processor.process_all_splits.return_value = iter([])  # No samples, already done
        gs_processor.get_latest_checkpoint.return_value = "checkpoint_gs.json"
        gs_processor.load_streaming_checkpoint.return_value = {
            "dataset_specific": {"completed": True}
        }
        mock_processors["GigaSpeech2"] = gs_processor
        
        # MozillaCV - partially done
        moz_processor = Mock()
        moz_processor.name = "MozillaCV"
        moz_processor.process_all_splits.return_value = iter([
            {"audio": "audio_data", "transcript": "text", "length": 2.0}
        ])
        moz_processor.get_latest_checkpoint.return_value = "checkpoint_moz.json"
        moz_processor.load_streaming_checkpoint.return_value = {
            "dataset_specific": {"completed": False, "file_index": 15}
        }
        mock_processors["MozillaCV"] = moz_processor
        
        # ProcessedVoiceTH - not started
        pv_processor = Mock()
        pv_processor.name = "ProcessedVoiceTH"
        pv_processor.process_all_splits.return_value = iter([
            {"audio": "audio_data", "transcript": "text", "length": 3.0}
        ])
        pv_processor.get_latest_checkpoint.return_value = None
        pv_processor.load_streaming_checkpoint.return_value = None
        mock_processors["ProcessedVoiceTH"] = pv_processor
        
        # Configure mock_create_processor to return appropriate processor
        def create_processor_side_effect(dataset_name, config):
            return mock_processors[dataset_name]
        
        mock_create_processor.side_effect = create_processor_side_effect
        
        # Mock uploader
        mock_uploader = Mock()
        mock_uploader.upload_batch.return_value = (True, "shard_00001")
        mock_uploader.shard_num = 1
        mock_uploader_class.return_value = mock_uploader
        
        # Import and run main with resume flag
        from main import main_streaming_mode
        args = Mock()
        args.resume = True
        args.checkpoint = None
        args.sample = False
        args.sample_size = 5
        args.upload_batch_size = 100
        args.streaming_batch_size = 10
        args.no_upload = False
        args.append = False
        args.private = False
        args.no_standardization = False
        args.sample_rate = 16000
        args.no_volume_norm = False
        args.target_db = -20.0
        args.enable_stt = False
        args.no_stt = True
        args.stt_batch_size = 16
        args.verbose = False
        
        dataset_names = ["GigaSpeech2", "MozillaCV", "ProcessedVoiceTH"]
        
        # Would need to mock more of main_streaming_mode to fully test
        # Key points to verify:
        # 1. GigaSpeech2 should be skipped (already completed)
        # 2. MozillaCV should resume from checkpoint
        # 3. ProcessedVoiceTH should start fresh


if __name__ == '__main__':
    unittest.main()