"""
Test suite to ensure main.sh produces exactly the expected number of samples.
This is a TDD test to fix the issue where 20 samples are created instead of 10.
"""

import unittest
import subprocess
import os
import sys
import time
from typing import List, Dict
from datasets import load_dataset
from huggingface_hub import HfApi

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HF_TOKEN_FILE
from utils.huggingface import read_hf_token, authenticate_hf


class TestMainShSampleCount(unittest.TestCase):
    """Test that main.sh produces exactly the expected number of samples."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_repo = "Thanarit/Thai-Voice-Test-1000000"
        cls.samples_per_dataset = 5
        cls.expected_datasets = ["GigaSpeech2", "ProcessedVoiceTH"]
        cls.expected_total_samples = cls.samples_per_dataset * len(cls.expected_datasets)
        cls.main_sh_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.sh")
        
        # Authenticate with HuggingFace
        token = read_hf_token(HF_TOKEN_FILE)
        if token:
            authenticate_hf(token)
        else:
            raise Exception("No HuggingFace token found")
        
        cls.hf_api = HfApi()
    
    def test_01_clean_environment(self):
        """Test that we can clean the environment properly."""
        # Clean checkpoints
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
        
        # Remove all checkpoint files
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.json'):
                file_path = os.path.join(checkpoint_dir, file)
                os.remove(file_path)
        
        # Verify checkpoints are removed
        remaining_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.json')]
        self.assertEqual(len(remaining_files), 0, 
                        f"Checkpoint files still exist: {remaining_files}")
    
    def test_02_main_sh_executable(self):
        """Test that main.sh exists and is executable."""
        self.assertTrue(os.path.exists(self.main_sh_path), "main.sh does not exist")
        self.assertTrue(os.access(self.main_sh_path, os.X_OK), "main.sh is not executable")
    
    def test_03_fresh_mode_produces_exact_count(self):
        """Test that fresh mode produces exactly 10 samples, not more."""
        # Run main.sh in fresh mode
        result = subprocess.run(
            [self.main_sh_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(self.main_sh_path)
        )
        
        # Check execution succeeded
        self.assertEqual(result.returncode, 0,
                        f"main.sh failed with return code {result.returncode}\\n"
                        f"STDOUT:\\n{result.stdout}\\n"
                        f"STDERR:\\n{result.stderr}")
        
        # Wait for HuggingFace to process
        time.sleep(30)
        
        # Load dataset and count samples
        dataset = load_dataset(self.test_repo, split="train", streaming=True)
        
        samples = []
        for sample in dataset:
            samples.append(sample)
            if len(samples) > 20:  # Safety limit
                break
        
        # CRITICAL TEST: Must have exactly 10 samples
        self.assertEqual(len(samples), self.expected_total_samples,
                        f"Expected exactly {self.expected_total_samples} samples, "
                        f"but got {len(samples)}")
    
    def test_04_sample_ids_start_from_one(self):
        """Test that sample IDs start from S1, not S11."""
        # Load dataset
        dataset = load_dataset(self.test_repo, split="train", streaming=True)
        
        samples = []
        for sample in dataset:
            samples.append(sample)
            if len(samples) >= self.expected_total_samples:
                break
        
        # Extract IDs
        ids = [sample["ID"] for sample in samples]
        
        # Test that IDs are S1-S10
        expected_ids = [f"S{i}" for i in range(1, self.expected_total_samples + 1)]
        self.assertEqual(ids, expected_ids,
                        f"Expected IDs {expected_ids}, but got {ids}")
    
    def test_05_no_duplicate_samples(self):
        """Test that there are no duplicate audio samples."""
        # Load dataset
        dataset = load_dataset(self.test_repo, split="train", streaming=True)
        
        samples = []
        for sample in dataset:
            samples.append(sample)
            if len(samples) >= 20:  # Check up to 20 to catch duplicates
                break
        
        # Check audio lengths for duplicates
        audio_signatures = []
        for sample in samples:
            # Create a signature from length and dataset name
            signature = (sample["length"], sample["dataset_name"])
            audio_signatures.append(signature)
        
        # Count occurrences of each signature
        from collections import Counter
        signature_counts = Counter(audio_signatures)
        
        # Check for duplicates
        duplicates = [(sig, count) for sig, count in signature_counts.items() if count > 1]
        
        # Allow some duplicates by chance, but not systematic duplication
        self.assertLess(len(duplicates), 3,
                       f"Found too many duplicate signatures: {duplicates}")
    
    def test_06_dataset_distribution_correct(self):
        """Test that we have correct distribution of samples per dataset."""
        # Load dataset
        dataset = load_dataset(self.test_repo, split="train", streaming=True)
        
        samples = []
        for sample in dataset:
            samples.append(sample)
            if len(samples) >= self.expected_total_samples:
                break
        
        # Count samples per dataset
        dataset_counts = {}
        for sample in samples:
            ds_name = sample["dataset_name"]
            dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1
        
        # Verify each dataset has exactly 5 samples
        for dataset_name in self.expected_datasets:
            self.assertEqual(dataset_counts.get(dataset_name, 0), self.samples_per_dataset,
                           f"Expected {self.samples_per_dataset} samples from {dataset_name}, "
                           f"got {dataset_counts.get(dataset_name, 0)}")
    
    def test_07_fresh_mode_overwrites_existing(self):
        """Test that running fresh mode again still produces exactly 10 samples."""
        # Run main.sh again
        result = subprocess.run(
            [self.main_sh_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(self.main_sh_path)
        )
        
        self.assertEqual(result.returncode, 0, "Second run of main.sh failed")
        
        # Wait for processing
        time.sleep(30)
        
        # Load dataset and count
        dataset = load_dataset(self.test_repo, split="train", streaming=True)
        
        samples = []
        for sample in dataset:
            samples.append(sample)
            if len(samples) > 20:
                break
        
        # Must still have exactly 10 samples
        self.assertEqual(len(samples), self.expected_total_samples,
                        f"After second run, expected {self.expected_total_samples} samples, "
                        f"but got {len(samples)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)