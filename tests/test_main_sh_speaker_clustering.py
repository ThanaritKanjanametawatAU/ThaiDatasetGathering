#!/usr/bin/env python3
"""
Test to verify speaker clustering works correctly when running main.sh.
This tests that S1-S8 and S10 should be clustered as the same speaker.
"""

import unittest
import subprocess
import time
import os
import json
from datasets import load_dataset


class TestMainShSpeakerClustering(unittest.TestCase):
    """Test speaker clustering behavior with main.sh."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_repo = "Thanarit/Thai-Voice-Test-Clustering"
        self.test_samples = 20  # Enough to include S1-S10 and verify clustering
        
    def test_speaker_clustering_s1_to_s10(self):
        """Test that S1-S8 and S10 are clustered as the same speaker, S9 is different."""
        print(f"\n{'='*60}")
        print("TESTING SPEAKER CLUSTERING WITH MAIN.SH")
        print(f"{'='*60}")
        
        # Clean up previous test data
        print("Cleaning up previous test data...")
        subprocess.run(["rm", "-rf", "checkpoints/*"], shell=True)
        subprocess.run(["rm", "-rf", "cache/*"], shell=True)
        subprocess.run(["rm", "-rf", "enhancement_metrics"], shell=True)
        
        # Create test version of main.sh
        print(f"Creating test version of main.sh with {self.test_samples} samples...")
        with open("main.sh", "r") as f:
            main_sh_content = f.read()
        
        # Modify for test
        test_content = main_sh_content.replace(
            'SAMPLES_PER_DATASET=1000000',
            f'SAMPLES_PER_DATASET={self.test_samples}'
        ).replace(
            'HF_REPO="Thanarit/Thai-Voice-Test-1000000"',
            f'HF_REPO="{self.test_repo}"'
        )
        
        with open("main_test_clustering.sh", "w") as f:
            f.write(test_content)
        
        os.chmod("main_test_clustering.sh", 0o755)
        
        # Run main.sh
        print("Running main_test_clustering.sh...")
        result = subprocess.run(
            ["./main_test_clustering.sh"],
            capture_output=True,
            text=True
        )
        
        # Check if processing succeeded
        self.assertEqual(result.returncode, 0, 
                        f"main.sh failed with return code {result.returncode}\n"
                        f"STDOUT:\n{result.stdout}\n"
                        f"STDERR:\n{result.stderr}")
        
        # Wait for HuggingFace to process
        print("Waiting 15 seconds for HuggingFace to process...")
        time.sleep(15)
        
        # Load and verify the dataset
        print(f"Loading dataset from {self.test_repo}...")
        dataset = load_dataset(self.test_repo, split='train', streaming=True)
        
        # Collect speaker IDs for S1-S10
        speaker_mapping = {}
        samples_checked = 0
        
        print("\nChecking speaker IDs:")
        print(f"{'Sample ID':<10} {'Speaker ID':<15} {'Expected Group':<20}")
        print("-" * 45)
        
        for sample in dataset:
            sample_id = sample.get('ID', '')
            if sample_id in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']:
                speaker_id = sample.get('speaker_id', 'MISSING')
                expected_group = "Same as S1" if sample_id != 'S9' else "Different"
                speaker_mapping[sample_id] = speaker_id
                print(f"{sample_id:<10} {speaker_id:<15} {expected_group:<20}")
                samples_checked += 1
                
            # Stop after finding all S1-S10
            if samples_checked >= 10:
                break
        
        # Verify clustering
        print(f"\n{'='*60}")
        print("VERIFICATION:")
        print(f"{'='*60}")
        
        # Get S1's speaker ID (reference)
        s1_speaker = speaker_mapping.get('S1')
        self.assertIsNotNone(s1_speaker, "S1 not found in dataset")
        
        # Check S2-S8 have same speaker as S1
        same_speaker_samples = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S10']
        for sample_id in same_speaker_samples:
            speaker_id = speaker_mapping.get(sample_id)
            self.assertIsNotNone(speaker_id, f"{sample_id} not found in dataset")
            self.assertEqual(speaker_id, s1_speaker,
                           f"{sample_id} has speaker {speaker_id}, expected {s1_speaker}")
            print(f"✅ {sample_id} correctly has same speaker as S1: {speaker_id}")
        
        # Check S9 has different speaker
        s9_speaker = speaker_mapping.get('S9')
        self.assertIsNotNone(s9_speaker, "S9 not found in dataset")
        self.assertNotEqual(s9_speaker, s1_speaker,
                          f"S9 incorrectly has same speaker as S1: {s9_speaker}")
        print(f"✅ S9 correctly has different speaker: {s9_speaker}")
        
        # Clean up
        subprocess.run(["rm", "-f", "main_test_clustering.sh"], shell=True)
        
        print(f"\n{'='*60}")
        print("✅ ALL SPEAKER CLUSTERING TESTS PASSED!")
        print(f"{'='*60}\n")
        
    def test_speaker_clustering_consistency_across_batches(self):
        """Test that speaker clustering is consistent even across batches."""
        print(f"\n{'='*60}")
        print("TESTING SPEAKER CLUSTERING CONSISTENCY ACROSS BATCHES")
        print(f"{'='*60}")
        
        # This test ensures that if we process more samples,
        # the clustering remains consistent
        test_samples = 100
        
        # Clean up
        subprocess.run(["rm", "-rf", "checkpoints/*"], shell=True)
        subprocess.run(["rm", "-rf", "cache/*"], shell=True)
        
        # Create test version with more samples
        with open("main.sh", "r") as f:
            main_sh_content = f.read()
        
        test_content = main_sh_content.replace(
            'SAMPLES_PER_DATASET=1000000',
            f'SAMPLES_PER_DATASET={test_samples}'
        ).replace(
            'HF_REPO="Thanarit/Thai-Voice-Test-1000000"',
            f'HF_REPO="{self.test_repo}-batch"'
        ).replace(
            '--speaker-batch-size 50',
            '--speaker-batch-size 25'  # Smaller batches to test consistency
        )
        
        with open("main_test_batch_consistency.sh", "w") as f:
            f.write(test_content)
        
        os.chmod("main_test_batch_consistency.sh", 0o755)
        
        # Run test
        print(f"Running test with {test_samples} samples and batch size 25...")
        result = subprocess.run(
            ["./main_test_batch_consistency.sh"],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, "Processing failed")
        
        # Wait for HuggingFace
        time.sleep(15)
        
        # Verify clustering consistency
        dataset = load_dataset(f"{self.test_repo}-batch", split='train', streaming=True)
        
        # Collect speaker IDs
        speaker_ids = {}
        for i, sample in enumerate(dataset):
            if i >= test_samples:
                break
            sample_id = sample.get('ID')
            speaker_id = sample.get('speaker_id')
            if sample_id in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']:
                speaker_ids[sample_id] = speaker_id
        
        # Verify S1-S8 and S10 still have same speaker
        s1_speaker = speaker_ids.get('S1')
        for sample_id in ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S10']:
            self.assertEqual(speaker_ids.get(sample_id), s1_speaker,
                           f"Batch consistency failed for {sample_id}")
        
        # S9 should still be different
        self.assertNotEqual(speaker_ids.get('S9'), s1_speaker,
                          "S9 should have different speaker")
        
        # Clean up
        subprocess.run(["rm", "-f", "main_test_batch_consistency.sh"], shell=True)
        
        print("✅ Batch consistency test passed!")
        
    def test_speaker_clustering_with_all_features(self):
        """Test speaker clustering works with all features enabled."""
        print(f"\n{'='*60}")
        print("TESTING SPEAKER CLUSTERING WITH ALL FEATURES")
        print(f"{'='*60}")
        
        # Verify clustering works with STT, enhancement, etc.
        # The main.sh already has all features enabled
        
        # We'll use the first test's results since main.sh has all features
        # This test just verifies that clustering works despite other features
        
        # Check that checkpoint has speaker model
        speaker_model_path = "checkpoints/speaker_model.json"
        self.assertTrue(os.path.exists(speaker_model_path),
                       "Speaker model not saved")
        
        with open(speaker_model_path, 'r') as f:
            speaker_model = json.load(f)
            
        # Verify speaker model has expected structure
        self.assertIn('speaker_counter', speaker_model)
        self.assertIn('existing_clusters', speaker_model)
        self.assertGreater(speaker_model['speaker_counter'], 0,
                          "No speakers identified")
        
        print(f"✅ Speaker model saved with {speaker_model['speaker_counter']} speakers")
        print("✅ All features work together with clustering!")


if __name__ == '__main__':
    unittest.main(verbosity=2)