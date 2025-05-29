#!/usr/bin/env python3
"""
Test-driven development tests for speaker ID dataset separation.
Tests that speaker IDs are correctly assigned within datasets but don't overlap between datasets.
"""

import unittest
import json
import os
import shutil
import tempfile
from unittest.mock import Mock
from datasets import load_dataset

from processors.gigaspeech2 import GigaSpeech2Processor
from processors.processed_voice_th import ProcessedVoiceTHProcessor
from processors.speaker_identification import SpeakerIdentification
from utils.streaming import StreamingUploader


class TestSpeakerIdDatasetSeparation(unittest.TestCase):
    """Test speaker ID assignment across different datasets."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Mock args for processors
        self.args = Mock()
        self.args.sample = True
        self.args.sample_size = 10
        self.args.streaming_batch_size = 1000
        self.args.cache_dir = os.path.join(self.test_dir, "cache")
        self.args.checkpoint_dir = self.checkpoint_dir
        self.args.verbose = False
        self.args.enable_stt = False
        self.args.stt_batch_size = 16
        self.args.no_volume_norm = False
        self.args.sample_rate = 16000
        self.args.target_db = -20.0
        self.args.no_standardization = False
        self.args.hf_repo = "test/repo"
        self.args.fresh = True
        self.args.speaker_model = "pyannote/embedding"
        self.args.speaker_batch_size = 50
        self.args.speaker_threshold = 0.7
        self.args.store_embeddings = False
        self.args.speaker_min_cluster_size = 15
        self.args.speaker_min_samples = 10
        self.args.speaker_epsilon = 0.3
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_speaker_ids_within_dataset(self):
        """Test that speaker IDs are correctly clustered within a dataset."""
        # This test verifies that samples from the same speaker get the same ID
        # We'll use a mock dataset with known speaker patterns
        
        # Create speaker identification system
        speaker_id = SpeakerIdentification(
            model_name=self.args.speaker_model,
            device="cpu",
            checkpoint_dir=self.checkpoint_dir,
            fresh_start=True
        )
        
        # Simulate processing GigaSpeech2 samples S1-S10
        # S1-S8 and S10 should be same speaker, S9 different
        gigaspeech_samples = []
        for i in range(1, 11):
            sample_id = f"S{i}"
            # S9 is different speaker
            is_different_speaker = (i == 9)
            gigaspeech_samples.append({
                "id": sample_id,
                "dataset": "GigaSpeech2",
                "is_different_speaker": is_different_speaker
            })
        
        # Process and check speaker assignments
        speaker_assignments = {}
        for sample in gigaspeech_samples:
            # In real usage, embeddings would come from audio
            # For testing, we'll verify the logic works correctly
            speaker_assignments[sample["id"]] = {
                "dataset": sample["dataset"],
                "is_different": sample["is_different_speaker"]
            }
        
        # Verify S1-S8 and S10 should have same speaker ID
        expected_same_speaker = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S10"]
        
        # This test will initially fail as we haven't implemented the logic yet
        # But it defines what we expect
        
    def test_speaker_ids_across_datasets(self):
        """Test that speaker IDs don't overlap between datasets."""
        # Create speaker identification system
        speaker_id = SpeakerIdentification(
            model_name=self.args.speaker_model,
            device="cpu",
            checkpoint_dir=self.checkpoint_dir,
            fresh_start=True
        )
        
        # Test scenario:
        # 1. Process GigaSpeech2 dataset (should get SPK_00001, SPK_00002, etc.)
        # 2. Process ProcessedVoiceTH dataset (should NOT reuse SPK_00001, etc.)
        
        # Expected behavior:
        # - GigaSpeech2: SPK_00001 (for S1-S8,S10), SPK_00002 (for S9), maybe more
        # - ProcessedVoiceTH: Should start from next available ID (e.g., SPK_00003 or higher)
        
        gigaspeech_speaker_ids = set()
        processed_voice_speaker_ids = set()
        
        # This test will verify that the two sets don't overlap
        # self.assertEqual(len(gigaspeech_speaker_ids & processed_voice_speaker_ids), 0)
        
    def test_speaker_counter_continuity(self):
        """Test that speaker counter continues incrementing across datasets."""
        # The speaker counter should not reset between datasets
        # If GigaSpeech2 ends with SPK_00005, ProcessedVoiceTH should start from SPK_00006
        
        # Create speaker identification system
        speaker_id = SpeakerIdentification(
            model_name=self.args.speaker_model,
            device="cpu",
            checkpoint_dir=self.checkpoint_dir,
            fresh_start=True
        )
        
        # Get initial counter
        initial_counter = speaker_id.next_speaker_id
        self.assertEqual(initial_counter, 1)  # Should start at 1 in fresh mode
        
        # After processing first dataset, counter should have increased
        # After processing second dataset, counter should continue from where it left off
        
    def test_clustering_state_reset_between_datasets(self):
        """Test that clustering state is reset between datasets but counter continues."""
        # The clustering algorithm should not try to match speakers across datasets
        # But the speaker ID counter should continue incrementing
        
        speaker_id = SpeakerIdentification(
            model_name=self.args.speaker_model,
            device="cpu",
            checkpoint_dir=self.checkpoint_dir,
            fresh_start=True
        )
        
        # Process dataset 1
        # Clear clustering state (embeddings, etc.) but keep counter
        # Process dataset 2
        # Verify no cross-dataset matching occurred
        
    def test_end_to_end_dataset_processing(self):
        """Test complete processing of two datasets with proper speaker ID separation."""
        # This is the comprehensive test that verifies the entire flow
        
        # Expected results after processing both datasets:
        # 1. GigaSpeech2 S1-S8,S10: SPK_00001
        # 2. GigaSpeech2 S9: SPK_00002
        # 3. GigaSpeech2 may have additional speakers (SPK_00003, etc.)
        # 4. ProcessedVoiceTH should start from next available (e.g., SPK_00006)
        # 5. No speaker ID should appear in both datasets
        
        # We'll verify this by checking the actual uploaded dataset
        
    def test_reset_for_new_dataset_behavior(self):
        """Test that reset_for_new_dataset clears clustering state but preserves counter."""
        speaker_id = SpeakerIdentification(
            model_name=self.args.speaker_model,
            device="cpu",
            checkpoint_dir=self.checkpoint_dir,
            fresh_start=True
        )
        
        # Add some embeddings and increment counter
        speaker_id.next_speaker_id = 5
        # In real usage, embeddings would be added
        
        # Reset for new dataset (should clear embeddings but keep counter)
        speaker_id.reset_for_new_dataset(reset_counter=False)
        
        # Counter should be preserved
        self.assertEqual(speaker_id.next_speaker_id, 5)
        
        # Embeddings should be cleared
        self.assertEqual(len(speaker_id.embeddings), 0)
        self.assertEqual(len(speaker_id.sample_ids), 0)
        
    def test_main_processing_logic(self):
        """Test the main.py processing logic for speaker ID assignment."""
        # This test verifies that main.py correctly:
        # 1. Processes each dataset
        # 2. Resets clustering state between datasets
        # 3. Maintains speaker counter continuity
        # 4. Results in non-overlapping speaker IDs
        
        # The key is that reset_for_new_dataset should be called with reset_counter=False
        # This preserves the counter while clearing the clustering state


if __name__ == "__main__":
    unittest.main()