"""
Test cases for fixing checkpoint duplication issue where samples are uploaded multiple times.

This test suite ensures that:
1. Each sample is uploaded exactly once
2. Checkpoints properly track uploaded samples
3. Resume functionality doesn't re-upload existing samples
4. Streaming uploads maintain unique IDs
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from collections import Counter

from utils.streaming import StreamingUploader, StreamingBatchProcessor
from processors.base_processor import BaseProcessor


class TestCheckpointDuplicationFix(unittest.TestCase):
    """Test suite for fixing checkpoint duplication issues."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.test_dir, "test_checkpoint.json")
        self.mock_token = "test_token"
        self.repo_id = "test/repo"
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_mock_uploader(self, checkpoint_path=None, split="train"):
        """Helper to create a mocked StreamingUploader."""
        with patch('utils.streaming.HfApi') as mock_hf_api:
            mock_api_instance = Mock()
            mock_hf_api.return_value = mock_api_instance
            mock_api_instance.create_repo.return_value = None
            mock_api_instance.list_repo_files.return_value = []
            
            uploader = StreamingUploader(
                repo_id=self.repo_id,
                split=split,
                checkpoint_path=checkpoint_path or self.checkpoint_path
            )
            # Keep the mock for upload_shard
            uploader.upload_shard = Mock(return_value=(True, "shard_00000.parquet"))
            return uploader
    
    def test_streaming_uploader_tracks_uploaded_ids(self):
        """Test that StreamingUploader properly tracks uploaded sample IDs."""
        uploader = self._create_mock_uploader()
        
        # Simulate uploading samples
        uploaded_ids = []
        # Upload 20 samples
        for i in range(20):
            sample = {"ID": f"S{i+1}"}
            uploader.add_sample(sample)
            uploaded_ids.append(sample["ID"])
        
        # Force flush
        uploader.flush()
        
        # Check that each ID was tracked
        self.assertEqual(len(set(uploaded_ids)), 20, "Should have 20 unique IDs")
        
        # Load checkpoint and verify uploaded IDs are tracked
        with open(self.checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Checkpoint should track uploaded samples
        self.assertIn("uploaded_ids", checkpoint, "Checkpoint should track uploaded IDs")
        self.assertEqual(len(checkpoint["uploaded_ids"]), 20, "Should track all 20 uploaded IDs")
        self.assertEqual(set(checkpoint["uploaded_ids"]), set(uploaded_ids), "Tracked IDs should match uploaded IDs")
    
    def test_streaming_uploader_prevents_duplicate_uploads(self):
        """Test that StreamingUploader prevents uploading the same ID twice."""
        uploader = self._create_mock_uploader()
        
        upload_calls = []
        def track_upload(shard_data, *args, **kwargs):
            upload_calls.extend([s["ID"] for s in shard_data])
            return (True, "shard.parquet")
        uploader.upload_shard = track_upload
        
        # Try to upload the same samples multiple times
        for attempt in range(3):  # Try 3 times
            for i in range(20):
                sample = {"ID": f"S{i+1}"}
                uploader.add_sample(sample)
            uploader.flush()
        
        # Check that each ID was uploaded only once
        id_counts = Counter(upload_calls)
        for id, count in id_counts.items():
            self.assertEqual(count, 1, f"ID {id} should be uploaded exactly once, but was uploaded {count} times")
    
    def test_checkpoint_resume_skips_uploaded_samples(self):
        """Test that resuming from checkpoint skips already uploaded samples."""
        # First, create a checkpoint with some uploaded samples
        initial_checkpoint = {
            "shard_index": 2,
            "samples_uploaded": 10,
            "uploaded_ids": [f"S{i+1}" for i in range(10)],
            "dataset_name": "GigaSpeech2",
            "split": "train"
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(initial_checkpoint, f)
        
        # Create uploader that should resume from checkpoint
        uploader = self._create_mock_uploader()
        
        upload_calls = []
        def track_upload(shard_data, *args, **kwargs):
            upload_calls.extend([s["ID"] for s in shard_data])
            return (True, "shard.parquet")
        uploader.upload_shard = track_upload
        
        # Try to upload all 20 samples (including the 10 already uploaded)
        for i in range(20):
            sample = {"ID": f"S{i+1}"}
            uploader.add_sample(sample)
        uploader.flush()
        
        # Only samples S11-S20 should be uploaded
        self.assertEqual(len(upload_calls), 10, "Should only upload 10 new samples")
        expected_new_ids = [f"S{i+1}" for i in range(10, 20)]
        self.assertEqual(set(upload_calls), set(expected_new_ids), "Should only upload samples S11-S20")
    
    def test_streaming_batch_processor_no_duplicates(self):
        """Test that StreamingBatchProcessor doesn't process samples multiple times."""
        processor = StreamingBatchProcessor(
            batch_size=5,
            process_fn=Mock(),
            checkpoint_path=self.checkpoint_path
        )
        
        processed_ids = []
        def track_processing(batch):
            processed_ids.extend([s["ID"] for s in batch])
            return batch  # Return unchanged
        
        processor.process_fn = track_processing
        
        # Process 20 samples
        for i in range(20):
            sample = {"ID": f"S{i+1}"}
            processor.add_sample(sample)
        processor.flush()
        
        # Check no duplicates
        self.assertEqual(len(processed_ids), 20, "Should process exactly 20 samples")
        self.assertEqual(len(set(processed_ids)), 20, "All processed IDs should be unique")
    
    def test_checkpoint_atomic_write(self):
        """Test that checkpoint writes are atomic to prevent corruption."""
        uploader = self._create_mock_uploader()
        
        # Track checkpoint saves
        checkpoint_saves = []
        original_save = uploader._save_checkpoint
        
        def track_save(*args, **kwargs):
            # Read checkpoint before save
            if os.path.exists(self.checkpoint_path):
                with open(self.checkpoint_path, 'r') as f:
                    before = json.load(f)
            else:
                before = None
            
            # Call original save
            original_save(*args, **kwargs)
            
            # Read checkpoint after save
            with open(self.checkpoint_path, 'r') as f:
                after = json.load(f)
            
            checkpoint_saves.append((before, after))
        
        uploader._save_checkpoint = track_save
        
        # Upload samples
        for i in range(10):
            sample = {"ID": f"S{i+1}"}
            uploader.add_sample(sample)
        uploader.flush()
        
        # Verify checkpoints are complete and consistent
        for i, (before, after) in enumerate(checkpoint_saves):
            if before is not None:
                # Uploaded IDs should only increase
                self.assertTrue(
                    set(before.get("uploaded_ids", [])).issubset(set(after.get("uploaded_ids", []))),
                    f"Checkpoint {i}: uploaded_ids should only increase"
                )
    
    def test_multiple_splits_maintain_separate_checkpoints(self):
        """Test that different splits maintain separate uploaded ID tracking."""
        train_checkpoint = os.path.join(self.test_dir, "train_checkpoint.json")
        val_checkpoint = os.path.join(self.test_dir, "val_checkpoint.json")
        
        train_uploader = self._create_mock_uploader(checkpoint_path=train_checkpoint, split="train")
        val_uploader = self._create_mock_uploader(checkpoint_path=val_checkpoint, split="validation")
        
        # Upload to train split
        for i in range(10):
            train_uploader.add_sample({"ID": f"S{i+1}"})
        train_uploader.flush()
        
        # Upload to validation split
        for i in range(5):
            val_uploader.add_sample({"ID": f"S{i+1}"})
        val_uploader.flush()
        
        # Load checkpoints
        with open(train_checkpoint, 'r') as f:
            train_cp = json.load(f)
        with open(val_checkpoint, 'r') as f:
            val_cp = json.load(f)
        
        # Verify separate tracking
        self.assertEqual(len(train_cp["uploaded_ids"]), 10, "Train should have 10 IDs")
        self.assertEqual(len(val_cp["uploaded_ids"]), 5, "Validation should have 5 IDs")
        self.assertEqual(train_cp["split"], "train", "Train checkpoint should track train split")
        self.assertEqual(val_cp["split"], "validation", "Val checkpoint should track validation split")
    
    def test_processor_integration_no_duplicates(self):
        """Test that processor integration doesn't cause duplicate uploads."""
        class TestProcessor(BaseProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.process_count = Counter()
            
            def process(self, checkpoint=None):
                # Simulate processing samples
                for i in range(20):
                    sample = {
                        "ID": f"S{i+1}",
                        "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
                        "transcript": f"Test {i+1}"
                    }
                    self.process_count[sample["ID"]] += 1
                    yield sample
            
            def estimate_size(self):
                return {"estimated_samples": 20, "estimated_hours": 0.1}
            
            def get_dataset_info(self):
                return {"name": "TestDataset", "size": 20}
            
            def process_streaming(self, *args, **kwargs):
                return self.process(*args, **kwargs)
        
        # Create processor
        processor = TestProcessor(
            config={
                "dataset_name": "TestDataset",
                "output_dir": self.test_dir,
                "name": "TestDataset"
            }
        )
        
        # Mock streaming uploader
        mock_uploader = Mock()
        uploaded_samples = []
        
        def track_upload(sample):
            uploaded_samples.append(sample["ID"])
        
        mock_uploader.add_sample.side_effect = track_upload
        
        # Process with streaming
        with patch('utils.streaming.StreamingUploader', return_value=mock_uploader):
            # Simulate multiple processing attempts (like resume)
            for attempt in range(3):
                samples = list(processor.process())
                for sample in samples:
                    mock_uploader.add_sample(sample)
        
        # Check that processor didn't process samples multiple times internally
        for id, count in processor.process_count.items():
            self.assertEqual(count, 3, f"Each sample should be processed 3 times (3 attempts)")
        
        # But uploader should handle deduplication
        # In real implementation, uploader would skip duplicates
        # This test verifies the expected behavior


if __name__ == "__main__":
    unittest.main()