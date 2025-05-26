#!/usr/bin/env python3
"""
Comprehensive tests for the checkpoint/resume system.
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.base_processor import BaseProcessor
from processors.gigaspeech2 import GigaSpeech2Processor


class TestCheckpointSystem(unittest.TestCase):
    """Test the unified checkpoint system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)
        
        # Create a mock processor
        self.config = {
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": os.path.join(self.temp_dir, "logs"),
            "dataset_name": "TestDataset",
            "enable_stt": False,
            "audio_config": {
                "enable_standardization": False
            }
        }
        
        # Create a concrete test processor class
        class TestProcessor(BaseProcessor):
            def __init__(self, config):
                super().__init__(config)
                self.source = "test_dataset"  # Add source for get_available_splits
            
            def process(self, checkpoint=None, sample_mode=False, sample_size=5):
                for i in range(sample_size):
                    yield {
                        "ID": f"S{i+1}",
                        "speaker_id": f"SPK_{i+1:05d}",
                        "Language": "th",
                        "audio": {"path": f"test_{i}.wav", "bytes": b"test"},
                        "transcript": f"Test transcript {i}",
                        "length": 1.0,
                        "dataset_name": "TestDataset",
                        "confidence_score": 1.0
                    }
            
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                return self.process(checkpoint, sample_mode, sample_size)
            
            def get_dataset_info(self):
                return {"name": "Test", "size": 100}
            
            def estimate_size(self):
                return 100
            
            def get_available_splits(self):
                # Override to avoid trying to load test_dataset
                return ["train"]
            
            def _process_single_split(self, split, checkpoint=None, sample_mode=False, sample_size=5):
                for i in range(sample_size):
                    yield {
                        "ID": f"S{split}_{i+1}",
                        "speaker_id": f"SPK_{i+1:05d}",
                        "Language": "th",
                        "audio": {"path": f"test_{split}_{i}.wav", "bytes": b"test"},
                        "transcript": f"Test transcript {split} {i}",
                        "length": 1.0,
                        "dataset_name": "TestDataset",
                        "confidence_score": 1.0
                    }
        
        self.processor = TestProcessor(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_unified_checkpoint_save_load(self):
        """Test saving and loading unified checkpoints."""
        # Save checkpoint
        checkpoint_file = self.processor.save_unified_checkpoint(
            samples_processed=100,
            current_split="train",
            split_index=50,
            shard_num=2,
            last_sample_id="S100",
            processed_ids=["S98", "S99", "S100"],
            dataset_specific_data={"custom": "data"}
        )
        
        self.assertTrue(os.path.exists(checkpoint_file))
        
        # Load checkpoint
        loaded_data = self.processor.load_unified_checkpoint(checkpoint_file)
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["mode"], "unified")
        self.assertEqual(loaded_data["samples_processed"], 100)
        self.assertEqual(loaded_data["current_split"], "train")
        self.assertEqual(loaded_data["split_index"], 50)
        self.assertEqual(loaded_data["shard_num"], 2)
        self.assertEqual(loaded_data["last_sample_id"], "S100")
        self.assertEqual(loaded_data["processed_ids"], ["S98", "S99", "S100"])
        self.assertEqual(loaded_data["dataset_specific"]["custom"], "data")
    
    def test_backward_compatibility_regular_checkpoint(self):
        """Test loading old format regular checkpoints."""
        # Create old format checkpoint
        old_checkpoint = {
            "processed_count": 50,
            "current_index": 49,
            "processed_ids": ["S1", "S2", "S3"],
            "timestamp": 1234567890,
            "metadata": {
                "processor": "TestProcessor",
                "timestamp": 1234567890,
                "version": "1.0"
            }
        }
        
        checkpoint_file = os.path.join(self.checkpoint_dir, "old_checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(old_checkpoint, f)
        
        # Load with unified loader
        loaded_data = self.processor.load_unified_checkpoint(checkpoint_file)
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["mode"], "unified")
        self.assertEqual(loaded_data["samples_processed"], 50)
        self.assertEqual(loaded_data["processed_ids"], ["S1", "S2", "S3"])
        self.assertEqual(loaded_data["shard_num"], 0)  # Default value
    
    def test_backward_compatibility_streaming_checkpoint(self):
        """Test loading old format streaming checkpoints."""
        # Create old format streaming checkpoint
        old_checkpoint = {
            "mode": "streaming",
            "shard_num": 5,
            "samples_processed": 1000,
            "last_sample_id": "S1000",
            "dataset_specific": {"key": "value"},
            "timestamp": 1234567890,
            "processor": "TestProcessor"
        }
        
        checkpoint_file = os.path.join(self.checkpoint_dir, "streaming_checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(old_checkpoint, f)
        
        # Load with unified loader
        loaded_data = self.processor.load_unified_checkpoint(checkpoint_file)
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["mode"], "unified")
        self.assertEqual(loaded_data["samples_processed"], 1000)
        self.assertEqual(loaded_data["shard_num"], 5)
        self.assertEqual(loaded_data["last_sample_id"], "S1000")
    
    def test_process_all_splits_with_checkpoint(self):
        """Test process_all_splits with checkpoint resume."""
        # Override available splits
        self.processor.get_available_splits = lambda: ["train", "test", "val"]
        
        # Process first batch
        samples = []
        for i, sample in enumerate(self.processor.process_all_splits(sample_mode=True, sample_size=6)):
            samples.append(sample)
            if i == 3:  # Stop after 4 samples
                break
        
        self.assertEqual(len(samples), 4)
        
        # Save checkpoint manually
        # Since we processed 4 samples total: 2 from train, 2 from test
        # We should resume from val split (next split after test)
        checkpoint_file = self.processor.save_unified_checkpoint(
            samples_processed=4,
            current_split="test",
            split_index=2,  # We've processed 2 samples from the test split
            processed_ids=[s["ID"] for s in samples]
        )
        
        # Resume from checkpoint
        resumed_samples = []
        for sample in self.processor.process_all_splits(
            checkpoint=checkpoint_file,
            sample_mode=True,
            sample_size=6
        ):
            resumed_samples.append(sample)
        
        # Check no duplicates
        all_ids = [s["ID"] for s in samples] + [s["ID"] for s in resumed_samples]
        unique_ids = set(all_ids)
        self.assertEqual(len(all_ids), len(unique_ids), "Duplicate samples found")
    
    def test_periodic_checkpoint_saving(self):
        """Test that checkpoints are saved periodically."""
        # Mock save_unified_checkpoint to track calls
        original_save = self.processor.save_unified_checkpoint
        save_calls = []
        
        def mock_save(*args, **kwargs):
            save_calls.append(kwargs)
            return original_save(*args, **kwargs)
        
        self.processor.save_unified_checkpoint = mock_save
        
        # Process with small checkpoint interval
        samples = list(self.processor.process_all_splits(
            sample_mode=True,
            sample_size=10,
            checkpoint_interval=3  # Save every 3 samples
        ))
        
        # Check that periodic saves happened
        # Should have saves at 3, 6, 9, and final
        self.assertGreaterEqual(len(save_calls), 3)
        
        # Check save at sample 3
        save_at_3 = next((s for s in save_calls if s.get("samples_processed") == 3), None)
        self.assertIsNotNone(save_at_3)
        
        # Check save at sample 6
        save_at_6 = next((s for s in save_calls if s.get("samples_processed") == 6), None)
        self.assertIsNotNone(save_at_6)
    
    def test_checkpoint_on_error(self):
        """Test that checkpoint is saved on error."""
        # Create processor that errors after 3 samples
        class ErrorProcessor(BaseProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sample_count = 0
                self.source = "test_dataset"
            
            def process(self, checkpoint=None, sample_mode=False, sample_size=5):
                pass
            
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                pass
            
            def get_dataset_info(self):
                return {"name": "Error", "size": 100}
            
            def estimate_size(self):
                return 100
            
            def get_available_splits(self):
                return ["train"]
            
            def _process_single_split(self, split, checkpoint=None, sample_mode=False, sample_size=5):
                for i in range(sample_size):
                    if self.sample_count == 3:
                        raise Exception("Test error")
                    self.sample_count += 1
                    yield {
                        "ID": f"S{i+1}",
                        "speaker_id": f"SPK_{i+1:05d}",
                        "Language": "th",
                        "audio": {"path": f"test_{i}.wav", "bytes": b"test"},
                        "transcript": f"Test transcript {i}",
                        "length": 1.0,
                        "dataset_name": "TestDataset",
                        "confidence_score": 1.0
                    }
        
        error_processor = ErrorProcessor(self.config)
        
        # Process until error
        samples = []
        try:
            for sample in error_processor.process_all_splits(sample_mode=False, sample_size=10):
                samples.append(sample)
        except Exception:
            pass
        
        # Check that checkpoint was saved
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.json')]
        self.assertGreater(len(checkpoint_files), 0)
        
        # Load the checkpoint and verify it has error info
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_files[0])
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        self.assertIn("error", checkpoint_data.get("dataset_specific", {}))
    
    def test_split_tracking_in_checkpoint(self):
        """Test that split information is properly tracked in checkpoints."""
        # Override available splits
        self.processor.get_available_splits = lambda: ["train", "test"]
        
        # Mock save to track split info
        save_calls = []
        original_save = self.processor.save_unified_checkpoint
        
        def mock_save(*args, **kwargs):
            save_calls.append(kwargs)
            return original_save(*args, **kwargs)
        
        self.processor.save_unified_checkpoint = mock_save
        
        # Process all splits
        list(self.processor.process_all_splits(
            sample_mode=True,
            sample_size=4,
            checkpoint_interval=2
        ))
        
        # Check that splits were tracked correctly
        train_saves = [s for s in save_calls if s.get("current_split") == "train"]
        test_saves = [s for s in save_calls if s.get("current_split") == "test"]
        
        self.assertGreater(len(train_saves), 0)
        self.assertGreater(len(test_saves), 0)


class TestGigaSpeech2CheckpointIntegration(unittest.TestCase):
    """Integration test for GigaSpeech2 processor checkpoint functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)
        
        self.config = {
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": os.path.join(self.temp_dir, "logs"),
            "dataset_name": "GigaSpeech2",
            "enable_stt": False,
            "audio_config": {
                "enable_standardization": False
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('processors.gigaspeech2.load_dataset')
    def test_gigaspeech2_checkpoint_resume(self, mock_load_dataset):
        """Test GigaSpeech2 processor checkpoint resume functionality."""
        # Create a proper iterable dataset mock
        class MockIterableDataset:
            def __init__(self, samples):
                self.samples = samples
                self.filter_called = False
                
            def __iter__(self):
                # Return fresh iterator each time
                return iter(self.samples)
                
            def filter(self, *args, **kwargs):
                self.filter_called = True
                return self
        
        # Mock samples
        mock_samples = [
            {
                "id": f"sample_{i}",
                "language": "th",
                "audio": {"array": [0.1] * 16000, "sampling_rate": 16000},
                "sentence": f"Thai text {i}",
                "duration": 1.0
            }
            for i in range(10)
        ]
        
        # Create mock dataset
        mock_dataset = MockIterableDataset(mock_samples)
        
        # Mock load_dataset to return our mock for any split
        def mock_load_fn(*args, **kwargs):
            if kwargs.get('streaming', False):
                # Return fresh dataset instance for each call
                return MockIterableDataset(mock_samples)
            return mock_dataset
            
        mock_load_dataset.side_effect = mock_load_fn
        
        processor = GigaSpeech2Processor(self.config)
        
        # Process first batch using streaming mode directly
        samples_before = []
        sample_count = 0
        checkpoint_file = None
        
        for sample in processor.process_streaming(sample_mode=True, sample_size=10):
            samples_before.append(sample["ID"])
            sample_count += 1
            if sample_count == 5:  # Stop after 5 samples
                # Save checkpoint
                checkpoint_file = processor.save_unified_checkpoint(
                    samples_processed=5,
                    current_split="train",
                    split_index=5,
                    last_sample_id=sample["ID"],
                    processed_ids=samples_before
                )
                break
        
        self.assertEqual(len(samples_before), 5)
        self.assertIsNotNone(checkpoint_file)
        
        # Resume from checkpoint
        samples_after = []
        for sample in processor.process_streaming(
            checkpoint=checkpoint_file,
            sample_mode=True,
            sample_size=10
        ):
            samples_after.append(sample["ID"])
        
        # Should get 5 more samples (10 total minus 5 already processed)
        self.assertEqual(len(samples_after), 5)
        
        # Check no duplicates
        all_samples = samples_before + samples_after
        unique_samples = set(all_samples)
        self.assertEqual(len(all_samples), len(unique_samples), 
                        f"Found duplicates: {[s for s in all_samples if all_samples.count(s) > 1]}")


if __name__ == "__main__":
    unittest.main()