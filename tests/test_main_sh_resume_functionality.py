#!/usr/bin/env python3
"""
Comprehensive tests for main.sh resume functionality.

This test suite verifies:
1. First run with ./main.sh creates 10 samples (5 from each dataset)
2. Second run with ./main.sh --resume appends and creates 20 total samples
3. No duplicates between the two runs
4. Resume does NOT overwrite existing data
5. IDs continue correctly (S11-S20 after S1-S10)
6. Verification on actual HuggingFace repository
"""

import unittest
import subprocess
import os
import json
import time
import shutil
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from huggingface_hub import HfApi

class TestMainShResumeFeature(unittest.TestCase):
    """Test suite for main.sh resume functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.project_root = os.path.dirname(cls.test_dir)
        cls.main_sh_path = os.path.join(cls.project_root, "main.sh")
        cls.checkpoints_dir = os.path.join(cls.project_root, "checkpoints")
        cls.backup_dir = os.path.join(cls.project_root, "test_checkpoints_backup")
        cls.hf_repo = "Thanarit/Thai-Voice"
        cls.hf_api = HfApi()
        
        # Ensure main.sh is executable
        if os.path.exists(cls.main_sh_path):
            os.chmod(cls.main_sh_path, 0o755)
        
        # Backup existing checkpoints
        if os.path.exists(cls.checkpoints_dir):
            shutil.copytree(cls.checkpoints_dir, cls.backup_dir, dirs_exist_ok=True)
            
    @classmethod
    def tearDownClass(cls):
        """Restore original state after all tests."""
        # Restore checkpoints
        if os.path.exists(cls.backup_dir):
            if os.path.exists(cls.checkpoints_dir):
                shutil.rmtree(cls.checkpoints_dir)
            shutil.move(cls.backup_dir, cls.checkpoints_dir)
    
    def setUp(self):
        """Set up before each test."""
        # Clear checkpoints for fresh test
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Clear cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        thai_voice_cache = os.path.join(cache_dir, "Thanarit___thai-voice")
        if os.path.exists(thai_voice_cache):
            shutil.rmtree(thai_voice_cache)
    
    def _run_main_sh(self, args: List[str] = None) -> Tuple[int, str, str]:
        """Run main.sh with given arguments."""
        cmd = [self.main_sh_path]
        if args:
            cmd.extend(args)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.project_root
        )
        
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr
    
    def _get_checkpoint_data(self) -> Dict[str, dict]:
        """Load all checkpoint data."""
        checkpoints = {}
        if os.path.exists(self.checkpoints_dir):
            for filename in os.listdir(self.checkpoints_dir):
                if filename.endswith("_unified_checkpoint.json"):
                    filepath = os.path.join(self.checkpoints_dir, filename)
                    with open(filepath, 'r') as f:
                        dataset_name = filename.replace("_unified_checkpoint.json", "")
                        checkpoints[dataset_name] = json.load(f)
        return checkpoints
    
    def _wait_for_hf_processing(self, expected_count: int, max_wait: int = 300):
        """Wait for HuggingFace to process uploaded data."""
        print(f"\nWaiting for HuggingFace to process data (expecting {expected_count} samples)...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Try streaming first (faster)
                dataset = load_dataset(self.hf_repo, split="train", streaming=True)
                samples = list(dataset.take(expected_count + 5))  # Take a few extra to ensure we got all
                
                if len(samples) >= expected_count:
                    print(f"✓ Found {len(samples)} samples on HuggingFace")
                    return True
                    
            except Exception as e:
                print(f"  Streaming failed: {e}, trying regular load...")
                try:
                    # Fallback to regular load
                    dataset = load_dataset(self.hf_repo, split="train")
                    if len(dataset) >= expected_count:
                        print(f"✓ Found {len(dataset)} samples on HuggingFace")
                        return True
                except:
                    pass
            
            print(f"  Current samples: {len(samples) if 'samples' in locals() else 0}, waiting...")
            time.sleep(10)
        
        return False
    
    def _verify_samples_on_hf(self, expected_ids: List[str]) -> bool:
        """Verify specific sample IDs exist on HuggingFace."""
        try:
            dataset = load_dataset(self.hf_repo, split="train", streaming=True)
            found_ids = set()
            
            for sample in dataset:
                if sample['ID'] in expected_ids:
                    found_ids.add(sample['ID'])
                if len(found_ids) == len(expected_ids):
                    break
            
            missing_ids = set(expected_ids) - found_ids
            if missing_ids:
                print(f"Missing IDs on HuggingFace: {missing_ids}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error verifying samples: {e}")
            return False
    
    def test_01_prerequisites(self):
        """Test that all prerequisites are in place."""
        # Check main.sh exists
        self.assertTrue(os.path.exists(self.main_sh_path), 
                       f"main.sh not found at {self.main_sh_path}")
        
        # Check it's executable
        self.assertTrue(os.access(self.main_sh_path, os.X_OK),
                       "main.sh is not executable")
        
        # Test basic execution
        returncode, stdout, stderr = self._run_main_sh(["--help"])
        self.assertEqual(returncode, 0, f"main.sh --help failed: {stderr}")
        
        # Check HuggingFace connectivity
        try:
            repo_info = self.hf_api.repo_info(self.hf_repo, repo_type="dataset")
            self.assertIsNotNone(repo_info, "Cannot access HuggingFace repository")
        except Exception as e:
            self.fail(f"Cannot connect to HuggingFace: {e}")
    
    def test_02_full_resume_workflow(self):
        """Test the complete resume workflow - ensure resume APPENDS, not overwrites."""
        print("\n=== Testing Full Resume Workflow - Resume Must APPEND ===")
        
        # Step 1: First run - should create 10 samples
        print("\n1. Running first batch (expecting 10 samples)...")
        returncode, stdout, stderr = self._run_main_sh()
        
        self.assertEqual(returncode, 0, f"First run failed: {stderr}")
        
        # Verify checkpoints were created
        checkpoints = self._get_checkpoint_data()
        self.assertEqual(len(checkpoints), 2, "Should have 2 checkpoint files")
        self.assertIn("GigaSpeech2", checkpoints)
        self.assertIn("ProcessedVoiceTH", checkpoints)
        
        # Verify each checkpoint shows 5 samples processed and marked as completed
        for dataset, data in checkpoints.items():
            self.assertEqual(data.get("samples_processed", 0), 5,
                           f"{dataset} should have processed 5 samples")
            self.assertTrue(data.get("completed", False),
                           f"{dataset} should be marked as completed")
        
        # Wait for HuggingFace processing
        print("Waiting for HuggingFace to process first batch...")
        self.assertTrue(self._wait_for_hf_processing(10),
                       "HuggingFace didn't process first batch in time")
        
        # Verify samples S1-S10 exist
        first_batch_ids = [f"S{i}" for i in range(1, 11)]
        self.assertTrue(self._verify_samples_on_hf(first_batch_ids),
                       "Not all samples from first batch found on HuggingFace")
        
        # Capture first batch data for later comparison
        dataset = load_dataset(self.hf_repo, split="train")
        first_batch_samples = {}
        for sample in dataset:
            first_batch_samples[sample['ID']] = {
                'transcript': sample.get('transcript'),
                'dataset_name': sample.get('dataset_name'),
                'speaker_id': sample.get('speaker_id')
            }
        
        print(f"First batch: {len(first_batch_samples)} samples captured")
        
        # Step 2: DELETE checkpoints to simulate new data processing
        print("\n2. Deleting checkpoints to force new processing...")
        for checkpoint_file in self.checkpoints_dir.glob("*_unified_checkpoint.json"):
            checkpoint_file.unlink()
            print(f"  Deleted: {checkpoint_file.name}")
        
        # Step 3: Resume run - should APPEND 10 more samples (total 20)
        print("\n3. Running resume (expecting to APPEND 10 more for 20 total)...")
        returncode, stdout, stderr = self._run_main_sh(["--resume"])
        
        self.assertEqual(returncode, 0, f"Resume run failed: {stderr}")
        
        # Wait for HuggingFace processing
        print("Waiting for HuggingFace to process resume batch...")
        self.assertTrue(self._wait_for_hf_processing(20),
                       "HuggingFace didn't process resume batch in time")
        
        # Step 4: CRITICAL - Verify we have 20 samples, not 10
        print("\n4. Verifying dataset has 20 samples (not overwritten to 10)...")
        dataset = load_dataset(self.hf_repo, split="train")
        
        # Check total count
        actual_count = len(dataset)
        self.assertEqual(actual_count, 20, 
                        f"CRITICAL: Expected 20 total samples after resume, but got {actual_count}. "
                        f"Resume appears to have OVERWRITTEN instead of APPENDING!")
        
        # Verify all original samples are still present
        for sample in dataset:
            sample_id = sample['ID']
            if sample_id in first_batch_samples:
                # This is an original sample - verify it wasn't changed
                original = first_batch_samples[sample_id]
                self.assertEqual(sample.get('transcript'), original['transcript'],
                               f"Sample {sample_id} transcript was modified!")
                self.assertEqual(sample.get('dataset_name'), original['dataset_name'],
                               f"Sample {sample_id} dataset_name was modified!")
        
        # Check for duplicates
        ids = [sample['ID'] for sample in dataset]
        self.assertEqual(len(ids), len(set(ids)), "Found duplicate IDs in dataset")
        
        # Verify ID sequence includes both batches
        expected_ids = [f"S{i}" for i in range(1, 21)]
        self.assertEqual(sorted(ids), sorted(expected_ids),
                        f"Sample IDs don't match expected sequence. Got: {sorted(ids)}")
        
        # Verify samples S11-S20 exist (new batch)
        second_batch_ids = [f"S{i}" for i in range(11, 21)]
        for expected_id in second_batch_ids:
            self.assertIn(expected_id, ids, f"Missing expected ID {expected_id} from resume batch")
        
        # Verify dataset distribution (10 from each dataset total)
        dataset_counts = {}
        for sample in dataset:
            dataset_name = sample.get('dataset_name', 'unknown')
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        
        print(f"\nDataset distribution: {dataset_counts}")
        self.assertEqual(dataset_counts.get('GigaSpeech2', 0), 10,
                        "Expected 10 total samples from GigaSpeech2")
        self.assertEqual(dataset_counts.get('ProcessedVoiceTH', 0), 10,
                        "Expected 10 total samples from ProcessedVoiceTH")
        
        print("\n✓ Resume correctly APPENDED data without overwriting!")
    
    def test_03_checkpoint_persistence(self):
        """Test that checkpoints persist correctly between runs."""
        print("\n=== Testing Checkpoint Persistence ===")
        
        # Run first batch
        returncode, _, _ = self._run_main_sh()
        self.assertEqual(returncode, 0, "First run failed")
        
        # Save checkpoint state
        original_checkpoints = self._get_checkpoint_data()
        self.assertEqual(len(original_checkpoints), 2, "Should have 2 checkpoints")
        
        # Modify one checkpoint to simulate partial processing
        gs2_checkpoint_path = os.path.join(self.checkpoints_dir, "GigaSpeech2_unified_checkpoint.json")
        with open(gs2_checkpoint_path, 'r') as f:
            gs2_data = json.load(f)
        
        gs2_data['samples_processed'] = 3  # Simulate only 3 processed
        with open(gs2_checkpoint_path, 'w') as f:
            json.dump(gs2_data, f, indent=2)
        
        # Run resume
        returncode, stdout, _ = self._run_main_sh(["--resume"])
        self.assertEqual(returncode, 0, "Resume run failed")
        
        # Verify checkpoint was updated correctly
        final_checkpoints = self._get_checkpoint_data()
        self.assertEqual(
            final_checkpoints['GigaSpeech2']['samples_processed'], 8,
            "GigaSpeech2 should have 8 samples (3 original + 5 resumed)"
        )
        self.assertEqual(
            final_checkpoints['ProcessedVoiceTH']['samples_processed'], 10,
            "ProcessedVoiceTH should have 10 samples (5 original + 5 resumed)"
        )
    
    def test_04_interrupt_and_resume(self):
        """Test resume behavior when run is interrupted."""
        print("\n=== Testing Interrupt and Resume ===")
        
        # Simulate interrupted run by creating partial checkpoints
        gs2_checkpoint = {
            "samples_processed": 3,
            "current_split": "train",
            "split_index": 3,
            "version": "2.0"
        }
        
        gs2_path = os.path.join(self.checkpoints_dir, "GigaSpeech2_unified_checkpoint.json")
        with open(gs2_path, 'w') as f:
            json.dump(gs2_checkpoint, f, indent=2)
        
        # Run resume
        returncode, stdout, stderr = self._run_main_sh(["--resume"])
        self.assertEqual(returncode, 0, f"Resume after interrupt failed: {stderr}")
        
        # Should process remaining samples
        checkpoints = self._get_checkpoint_data()
        
        # GigaSpeech2 should complete to 5 samples
        self.assertEqual(
            checkpoints.get('GigaSpeech2', {}).get('samples_processed', 0), 5,
            "GigaSpeech2 didn't complete to 5 samples"
        )
        
        # ProcessedVoiceTH should process its 5 samples
        self.assertEqual(
            checkpoints.get('ProcessedVoiceTH', {}).get('samples_processed', 0), 5,
            "ProcessedVoiceTH didn't process 5 samples"
        )
    
    def test_05_dataset_continuity(self):
        """Test that dataset content is preserved across resume."""
        print("\n=== Testing Dataset Content Continuity ===")
        
        # First run
        self._run_main_sh()
        self._wait_for_hf_processing(10)
        
        # Load first batch samples
        dataset = load_dataset(self.hf_repo, split="train")
        first_batch_samples = {sample['ID']: sample for sample in dataset}
        
        # Resume run
        self._run_main_sh(["--resume"])
        self._wait_for_hf_processing(20)
        
        # Load complete dataset
        dataset = load_dataset(self.hf_repo, split="train")
        all_samples = {sample['ID']: sample for sample in dataset}
        
        # Verify first batch samples are unchanged
        for sample_id, original_sample in first_batch_samples.items():
            self.assertIn(sample_id, all_samples, f"Sample {sample_id} missing after resume")
            
            resumed_sample = all_samples[sample_id]
            
            # Check key fields match
            self.assertEqual(
                original_sample.get('transcript'), resumed_sample.get('transcript'),
                f"Transcript changed for {sample_id}"
            )
            self.assertEqual(
                original_sample.get('speaker_id'), resumed_sample.get('speaker_id'),
                f"Speaker ID changed for {sample_id}"
            )
            self.assertEqual(
                original_sample.get('dataset_name'), resumed_sample.get('dataset_name'),
                f"Dataset name changed for {sample_id}"
            )
    
    def test_06_speaker_id_continuity(self):
        """Test that speaker IDs continue correctly across resume."""
        print("\n=== Testing Speaker ID Continuity ===")
        
        # First run
        self._run_main_sh()
        self._wait_for_hf_processing(10)
        
        # Check speaker IDs from first batch
        dataset = load_dataset(self.hf_repo, split="train")
        first_batch_speaker_ids = set()
        for sample in dataset:
            if sample.get('speaker_id'):
                first_batch_speaker_ids.add(sample['speaker_id'])
        
        print(f"First batch speaker IDs: {sorted(first_batch_speaker_ids)}")
        
        # Resume run
        self._run_main_sh(["--resume"])
        self._wait_for_hf_processing(20)
        
        # Check speaker IDs from complete dataset
        dataset = load_dataset(self.hf_repo, split="train")
        all_speaker_ids = set()
        speaker_id_counts = {}
        
        for sample in dataset:
            speaker_id = sample.get('speaker_id')
            if speaker_id:
                all_speaker_ids.add(speaker_id)
                speaker_id_counts[speaker_id] = speaker_id_counts.get(speaker_id, 0) + 1
        
        print(f"All speaker IDs: {sorted(all_speaker_ids)}")
        print(f"Speaker ID distribution: {speaker_id_counts}")
        
        # Verify speaker IDs are consistent
        self.assertTrue(
            first_batch_speaker_ids.issubset(all_speaker_ids),
            "Some speaker IDs from first batch are missing"
        )
        
        # Verify expected speaker ID pattern (based on CLAUDE.md)
        # S1-S8 and S10 should have SPK_00001, S9 should have SPK_00002
        expected_spk_00001_count = 9  # In first 10 samples
        if 'SPK_00001' in speaker_id_counts:
            self.assertGreaterEqual(
                speaker_id_counts['SPK_00001'], expected_spk_00001_count,
                f"SPK_00001 should appear at least {expected_spk_00001_count} times"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)