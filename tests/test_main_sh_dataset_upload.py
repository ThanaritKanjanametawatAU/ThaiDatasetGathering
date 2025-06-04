"""
Test suite for main.sh execution and HuggingFace dataset upload verification.
Tests both fresh and append modes with 5 samples per dataset.
"""

import unittest
import subprocess
import time
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.huggingface import authenticate_hf, read_hf_token
from config import HF_TOKEN_FILE


class TestMainShDatasetUpload(unittest.TestCase):
    """Test main.sh execution and dataset upload to HuggingFace."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_repo = "Thanarit/Thai-Voice-Test-1000000"
        cls.samples_per_dataset = 5
        cls.expected_datasets = ["GigaSpeech2", "ProcessedVoiceTH"]
        cls.hf_api = HfApi()
        
        # Authenticate with HuggingFace
        token = read_hf_token(HF_TOKEN_FILE)
        if token:
            authenticate_hf(token)
        else:
            raise Exception("No HuggingFace token found. Please add your token to hf_token.txt")
    
    def test_01_main_sh_exists_and_executable(self):
        """Test that main.sh exists and is executable."""
        main_sh_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.sh")
        self.assertTrue(os.path.exists(main_sh_path), "main.sh does not exist")
        self.assertTrue(os.access(main_sh_path, os.X_OK), "main.sh is not executable")
    
    def test_02_fresh_mode_execution(self):
        """Test main.sh execution in fresh mode with 5 samples."""
        # Create temporary main.sh for testing with 5 samples
        test_main_sh = self._create_test_main_sh(samples=5, mode="fresh")
        
        # Execute main.sh
        result = subprocess.run(
            ["bash", test_main_sh],
            capture_output=True,
            text=True
        )
        
        # Check execution succeeded
        self.assertEqual(result.returncode, 0, 
                        f"main.sh failed with return code {result.returncode}\n"
                        f"STDOUT:\n{result.stdout}\n"
                        f"STDERR:\n{result.stderr}")
        
        # Verify output contains success indicators
        self.assertIn("Processing complete", result.stdout, 
                      "Processing did not complete successfully")
        
        # Clean up
        os.remove(test_main_sh)
    
    def test_03_dataset_uploaded_to_huggingface(self):
        """Test that dataset was uploaded to HuggingFace repository."""
        # Wait for HuggingFace to process the upload
        time.sleep(30)
        
        # Check if dataset exists on HuggingFace
        try:
            dataset_info = self.hf_api.dataset_info(self.test_repo)
            self.assertIsNotNone(dataset_info, f"Dataset {self.test_repo} not found on HuggingFace")
        except Exception as e:
            self.fail(f"Failed to access dataset {self.test_repo}: {str(e)}")
    
    def test_04_dataset_viewer_shows_correct_data(self):
        """Test that HuggingFace dataset viewer shows the correct data."""
        # Load the dataset from HuggingFace
        try:
            # Load in streaming mode first to gather data
            dataset_stream = load_dataset(self.test_repo, split="train", streaming=True)
            
            # Collect all samples (we know there are only 10)
            samples = []
            for sample in dataset_stream:
                samples.append(sample)
                if len(samples) >= 10:  # Safety limit
                    break
            
            # Verify we got samples
            self.assertGreater(len(samples), 0, "No samples loaded from dataset")
            
            # Check total samples (5 per dataset)
            expected_total = self.samples_per_dataset * len(self.expected_datasets)
            self.assertEqual(len(samples), expected_total,
                           f"Expected {expected_total} samples, got {len(samples)}")
            
            # Verify dataset schema
            expected_columns = ["ID", "speaker_id", "Language", "audio", "transcript", 
                              "length", "dataset_name", "confidence_score"]
            first_sample = samples[0]
            for col in expected_columns:
                self.assertIn(col, first_sample.keys(),
                            f"Missing expected column: {col}")
            
            # Verify IDs are sequential
            ids = [row["ID"] for row in samples]
            expected_ids = [f"S{i}" for i in range(1, expected_total + 1)]
            self.assertEqual(ids, expected_ids,
                           f"IDs are not sequential. Expected {expected_ids}, got {ids}")
            
            # Verify dataset_name distribution
            dataset_names = [row["dataset_name"] for row in samples]
            for expected_dataset in self.expected_datasets:
                count = dataset_names.count(expected_dataset)
                self.assertEqual(count, self.samples_per_dataset,
                               f"Expected {self.samples_per_dataset} samples from {expected_dataset}, got {count}")
            
            # Verify all samples have Thai language
            languages = [row["Language"] for row in samples]
            self.assertTrue(all(lang == "th" for lang in languages),
                          "Not all samples have Thai language")
            
            # Verify audio format
            for i, row in enumerate(samples):
                audio = row["audio"]
                self.assertIsInstance(audio, dict, f"Sample {i+1}: Audio is not a dict")
                self.assertIn("array", audio, f"Sample {i+1}: Audio missing 'array' key")
                self.assertIn("sampling_rate", audio, f"Sample {i+1}: Audio missing 'sampling_rate' key")
                self.assertIn("path", audio, f"Sample {i+1}: Audio missing 'path' key")
                self.assertEqual(audio["sampling_rate"], 16000, 
                               f"Sample {i+1}: Expected 16kHz sampling rate, got {audio['sampling_rate']}")
            
        except Exception as e:
            self.fail(f"Failed to load and verify dataset: {str(e)}")
    
    def test_05_append_mode_execution(self):
        """Test main.sh execution in append mode."""
        # Record current dataset state
        initial_dataset = load_dataset(self.test_repo, split="train", streaming=False)
        initial_count = len(initial_dataset)
        last_id = initial_dataset[-1]["ID"] if initial_count > 0 else "S0"
        
        # Create append mode main.sh
        test_main_sh = self._create_test_main_sh(samples=5, mode="append")
        
        # Execute in append mode
        result = subprocess.run(
            ["bash", test_main_sh],
            capture_output=True,
            text=True
        )
        
        # Check execution succeeded
        self.assertEqual(result.returncode, 0,
                        f"Append mode failed with return code {result.returncode}\n"
                        f"STDOUT:\n{result.stdout}\n"
                        f"STDERR:\n{result.stderr}")
        
        # Clean up
        os.remove(test_main_sh)
    
    def test_06_append_mode_preserves_and_continues_data(self):
        """Test that append mode preserves existing data and continues IDs correctly."""
        # Wait for HuggingFace to process the append
        time.sleep(30)
        
        # Load updated dataset
        try:
            # Load in streaming mode to gather all data
            dataset_stream = load_dataset(self.test_repo, split="train", streaming=True)
            
            # Collect all samples
            samples = []
            for sample in dataset_stream:
                samples.append(sample)
                if len(samples) >= 20:  # Safety limit
                    break
            
            # Calculate expected counts
            samples_per_append = self.samples_per_dataset * len(self.expected_datasets)
            expected_total = samples_per_append * 2  # Fresh + append
            
            # Verify total count increased
            self.assertEqual(len(samples), expected_total,
                           f"Expected {expected_total} total samples after append, got {len(samples)}")
            
            # Verify IDs are continuous
            ids = [row["ID"] for row in samples]
            expected_ids = [f"S{i}" for i in range(1, expected_total + 1)]
            self.assertEqual(ids, expected_ids,
                           f"IDs are not continuous after append. Expected {expected_ids}, got {ids}")
            
            # Verify original data is preserved
            # Check first half matches original pattern
            first_half = samples[:samples_per_append]
            for i, row in enumerate(first_half):
                self.assertEqual(row["ID"], f"S{i+1}",
                               f"Original data ID changed at position {i}")
            
            # Verify appended data starts with correct ID
            second_half = samples[samples_per_append:expected_total]
            for i, row in enumerate(second_half):
                expected_id = f"S{samples_per_append + i + 1}"
                self.assertEqual(row["ID"], expected_id,
                               f"Appended data has wrong ID at position {i}. Expected {expected_id}, got {row['ID']}")
            
            # Verify dataset distribution in appended data
            appended_dataset_names = [row["dataset_name"] for row in second_half]
            for expected_dataset in self.expected_datasets:
                count = appended_dataset_names.count(expected_dataset)
                self.assertEqual(count, self.samples_per_dataset,
                               f"Expected {self.samples_per_dataset} appended samples from {expected_dataset}, got {count}")
            
        except Exception as e:
            self.fail(f"Failed to verify append mode: {str(e)}")
    
    def test_07_audio_enhancement_and_speaker_id_features(self):
        """Test that audio enhancement and speaker ID features are working."""
        # Load the final dataset
        try:
            dataset = load_dataset(self.test_repo, split="train", streaming=False)
            
            # Verify speaker IDs are assigned
            speaker_ids = [row["speaker_id"] for row in dataset]
            self.assertTrue(all(sid.startswith("SPK_") for sid in speaker_ids),
                          "Not all samples have proper speaker IDs")
            
            # Verify speaker ID pattern (should have at least 2 different speakers)
            unique_speakers = list(set(speaker_ids))
            self.assertGreaterEqual(len(unique_speakers), 2,
                                  f"Expected at least 2 unique speakers, got {len(unique_speakers)}: {unique_speakers}")
            
            # Verify confidence scores
            confidence_scores = [row["confidence_score"] for row in dataset]
            self.assertTrue(all(0.0 <= score <= 1.0 for score in confidence_scores),
                          "Confidence scores not in valid range [0, 1]")
            
            # Verify transcripts exist (either original or STT-generated)
            transcripts = [row["transcript"] for row in dataset]
            non_empty_transcripts = [t for t in transcripts if t and t.strip()]
            self.assertGreater(len(non_empty_transcripts), 0,
                             "No non-empty transcripts found")
            
        except Exception as e:
            self.fail(f"Failed to verify features: {str(e)}")
    
    def _create_test_main_sh(self, samples: int, mode: str) -> str:
        """Create a temporary main.sh configured for testing."""
        # Read the original main.sh
        main_sh_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.sh")
        with open(main_sh_path, 'r') as f:
            content = f.read()
        
        # Modify for testing
        # Replace SAMPLES_PER_DATASET value
        content = content.replace('SAMPLES_PER_DATASET=500000', f'SAMPLES_PER_DATASET={samples}')
        
        # Ensure correct HF_REPO
        content = content.replace('HF_REPO="Thanarit/Thai-Voice-Test-1000000"', 
                                 f'HF_REPO="{self.test_repo}"')
        
        # Adjust mode
        if mode == "fresh":
            content = content.replace('MODE="--append"', 'MODE="--fresh"')
        else:
            content = content.replace('MODE="--fresh"', 'MODE="--append"')
        
        # Ensure we only process our expected datasets
        content = content.replace('DATASETS="--all"', 
                                 f'DATASETS="{" ".join(self.expected_datasets)}"')
        
        # Create temporary file
        test_sh_path = "/tmp/test_main_sh_" + str(int(time.time())) + ".sh"
        with open(test_sh_path, 'w') as f:
            f.write(content)
        
        # Make executable
        os.chmod(test_sh_path, 0o755)
        
        return test_sh_path


if __name__ == "__main__":
    unittest.main(verbosity=2)