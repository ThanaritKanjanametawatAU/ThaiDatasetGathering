#!/usr/bin/env python3
"""
Comprehensive tests for main.sh with all features enabled.
Tests the full pipeline including STT, audio enhancement, and streaming upload.
"""

import unittest
import os
import json
import subprocess
import time
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.huggingface import read_hf_token, get_last_id
from huggingface_hub import HfApi


class TestMainShAllFeatures(unittest.TestCase):
    """Test main.sh with all features enabled."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_repo = "Thanarit/Thai-Voice-Test6"
        cls.expected_datasets = ["GigaSpeech2", "ProcessedVoiceTH"]
        cls.samples_per_dataset = 100
        cls.total_expected_samples = cls.samples_per_dataset * len(cls.expected_datasets)
        
        # Read HF token
        cls.hf_token = read_hf_token('.hf_token')
        if cls.hf_token:
            cls.api = HfApi()
        else:
            cls.api = None
            
        # Paths
        cls.enhancement_metrics_dir = Path("enhancement_metrics")
        cls.checkpoints_dir = Path("checkpoints")
        
    def test_01_prerequisites(self):
        """Test that all prerequisites are in place."""
        # Check main.sh exists and is executable
        self.assertTrue(os.path.exists("main.sh"), "main.sh not found")
        self.assertTrue(os.access("main.sh", os.X_OK), "main.sh is not executable")
        
        # Check main.py exists
        self.assertTrue(os.path.exists("main.py"), "main.py not found")
        
        # Check HF token exists
        self.assertIsNotNone(self.hf_token, "HuggingFace token not found")
        
        # Check conda environment
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True
        )
        self.assertIn("thaidataset", result.stdout, "thaidataset conda environment not found")
        
    def test_02_feature_integration(self):
        """Test that all features are properly integrated in main.py."""
        # Check audio enhancement is integrated
        with open("main.py", 'r') as f:
            main_content = f.read()
            
        # Check for audio enhancement imports
        self.assertIn("from processors.audio_enhancement.core import AudioEnhancer", main_content,
                     "AudioEnhancer import not found")
        self.assertIn("from monitoring.metrics_collector import MetricsCollector", main_content,
                     "MetricsCollector import not found")
        
        # Check for enhancement arguments
        self.assertIn("--enable-audio-enhancement", main_content,
                     "Audio enhancement flag not found")
        self.assertIn("--enhancement-level", main_content,
                     "Enhancement level flag not found")
        
        # Check for enhancement processing in streaming mode
        self.assertIn("audio_enhancer", main_content,
                     "Audio enhancer variable not found")
        self.assertIn("enhancement_buffer", main_content,
                     "Enhancement buffer not found")
        
    def test_03_audio_enhancement_modules(self):
        """Test that audio enhancement modules exist and are properly structured."""
        # Check core module
        audio_enhancement_path = Path("processors/audio_enhancement")
        self.assertTrue(audio_enhancement_path.exists(),
                       "audio_enhancement directory not found")
        
        # Check required files
        required_files = [
            "__init__.py",
            "core.py",
            "engines/__init__.py",
            "engines/denoiser.py",
            "engines/spectral_gating.py"
        ]
        
        for file in required_files:
            file_path = audio_enhancement_path / file
            self.assertTrue(file_path.exists(), f"Required file {file} not found")
            
    def test_04_monitoring_modules(self):
        """Test that monitoring modules exist."""
        monitoring_path = Path("monitoring")
        self.assertTrue(monitoring_path.exists(), "monitoring directory not found")
        
        # Check required files
        required_files = [
            "__init__.py",
            "metrics_collector.py",
            "dashboard.py"
        ]
        
        for file in required_files:
            file_path = monitoring_path / file
            self.assertTrue(file_path.exists(), f"Required file {file} not found")
            
    def test_05_run_main_sh(self):
        """Test running main.sh with all features."""
        # Clean up before test
        subprocess.run(["rm", "-rf", "enhancement_metrics"], check=False)
        subprocess.run(["rm", "-rf", "checkpoints/*test*"], check=False)
        
        # Record start time
        start_time = time.time()
        
        # Run main.sh
        result = subprocess.run(
            ["bash", "main.sh"],
            capture_output=True,
            text=True
        )
        
        # Record end time
        end_time = time.time()
        duration = end_time - start_time
        
        # Print output for debugging
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print(f"Execution time: {duration:.2f} seconds")
        
        # Check execution was successful
        self.assertEqual(result.returncode, 0, 
                        f"main.sh failed with return code {result.returncode}\nSTDERR: {result.stderr}")
        
        # Check output contains expected messages
        self.assertIn("Successfully completed dataset processing!", result.stdout,
                     "Success message not found in output")
        self.assertIn("Streaming processing completed:", result.stdout,
                     "Streaming completion message not found")
        
    def test_06_enhancement_metrics(self):
        """Test that enhancement metrics were generated."""
        # Check enhancement metrics directory exists
        self.assertTrue(self.enhancement_metrics_dir.exists(),
                       "Enhancement metrics directory not created")
        
        # Check summary.json exists
        summary_file = self.enhancement_metrics_dir / "summary.json"
        self.assertTrue(summary_file.exists(), "Enhancement summary.json not created")
        
        # Load and validate summary
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            
        # Check structure
        self.assertIn("samples", summary, "samples field missing from summary")
        self.assertIn("summary", summary, "summary statistics missing")
        self.assertIn("trends", summary, "trends missing from summary")
        
        # Check we have metrics for expected number of samples
        samples = summary.get("samples", {})
        self.assertGreaterEqual(len(samples), self.total_expected_samples * 0.9,  # Allow 10% tolerance
                               f"Expected at least {self.total_expected_samples * 0.9} samples, got {len(samples)}")
        
        # Check sample metrics structure
        if samples:
            sample_id = list(samples.keys())[0]
            sample_metrics = samples[sample_id]
            
            required_fields = [
                "snr_improvement",
                "pesq_score", 
                "stoi_score",
                "processing_time",
                "noise_level",
                "audio_file"
            ]
            
            for field in required_fields:
                self.assertIn(field, sample_metrics, f"Field {field} missing from sample metrics")
                
        # Check summary statistics
        summary_stats = summary.get("summary", {})
        self.assertIn("processing_time", summary_stats, "processing_time stats missing")
        self.assertIn("noise_level", summary_stats, "noise_level stats missing")
        
    def test_07_checkpoint_files(self):
        """Test that checkpoint files were created."""
        # Check for dataset checkpoints
        checkpoint_files = list(self.checkpoints_dir.glob("*_unified_checkpoint.json"))
        self.assertGreaterEqual(len(checkpoint_files), len(self.expected_datasets),
                               f"Expected at least {len(self.expected_datasets)} checkpoint files")
        
        # Check checkpoint content
        for checkpoint_file in checkpoint_files:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                
            # Validate checkpoint structure
            self.assertIn("mode", checkpoint, "mode field missing from checkpoint")
            self.assertIn("samples_processed", checkpoint, "samples_processed missing")
            self.assertIn("processor", checkpoint, "processor field missing")
            
            # Check samples were processed
            self.assertGreater(checkpoint["samples_processed"], 0,
                             f"No samples processed in {checkpoint_file.name}")
                             
    def test_08_huggingface_upload(self):
        """Test that data was uploaded to HuggingFace."""
        if not self.api:
            self.skipTest("HuggingFace API not available")
            
        try:
            # Get dataset info
            dataset_info = self.api.dataset_info(self.test_repo, token=self.hf_token)
            
            # Check dataset exists
            self.assertIsNotNone(dataset_info, "Dataset info could not be retrieved")
            
            # Check for parquet files
            files = self.api.list_repo_files(
                repo_id=self.test_repo,
                repo_type="dataset",
                token=self.hf_token
            )
            
            parquet_files = [f for f in files if f.endswith('.parquet')]
            self.assertGreater(len(parquet_files), 0, "No parquet files found in repository")
            
            # Check README.md exists
            self.assertIn("README.md", files, "README.md not found in repository")
            
        except Exception as e:
            self.fail(f"Failed to check HuggingFace repository: {str(e)}")
            
    def test_09_sample_validation(self):
        """Test that uploaded samples have correct structure."""
        if not self.api:
            self.skipTest("HuggingFace API not available")
            
        try:
            from datasets import load_dataset
            
            # Load a few samples to validate
            dataset = load_dataset(self.test_repo, split="train", streaming=True)
            
            # Check first few samples
            samples_checked = 0
            for sample in dataset:
                # Check required fields
                required_fields = [
                    "ID", "speaker_id", "Language", "audio", 
                    "transcript", "length", "dataset_name", "confidence_score"
                ]
                
                for field in required_fields:
                    self.assertIn(field, sample, f"Field {field} missing from sample")
                    
                # Validate field types and values
                self.assertTrue(sample["ID"].startswith("S"), "ID should start with S")
                self.assertTrue(sample["speaker_id"].startswith("SPK_"), "speaker_id should start with SPK_")
                self.assertEqual(sample["Language"], "th", "Language should be 'th'")
                self.assertIsInstance(sample["audio"], dict, "audio should be a dict")
                self.assertIn("array", sample["audio"], "audio should have 'array' field")
                self.assertIn("sampling_rate", sample["audio"], "audio should have 'sampling_rate' field")
                self.assertEqual(sample["audio"]["sampling_rate"], 16000, "Sampling rate should be 16000")
                self.assertIsInstance(sample["transcript"], str, "transcript should be string")
                self.assertGreater(sample["length"], 0, "length should be positive")
                self.assertIn(sample["dataset_name"], self.expected_datasets, 
                            f"dataset_name should be one of {self.expected_datasets}")
                self.assertGreaterEqual(sample["confidence_score"], 0.0, "confidence_score should be >= 0")
                self.assertLessEqual(sample["confidence_score"], 1.0, "confidence_score should be <= 1")
                
                samples_checked += 1
                if samples_checked >= 10:  # Check first 10 samples
                    break
                    
            self.assertGreater(samples_checked, 0, "No samples could be loaded from dataset")
            
        except Exception as e:
            self.fail(f"Failed to validate samples: {str(e)}")
            
    def test_10_stt_functionality(self):
        """Test that STT was applied to samples without transcripts."""
        # This would require checking specific samples that originally had no transcript
        # For now, we just verify STT was enabled in the command
        with open("main.sh", 'r') as f:
            main_sh_content = f.read()
            
        self.assertIn("--enable-stt", main_sh_content, "STT flag not found in main.sh")
        self.assertIn("--stt-batch-size", main_sh_content, "STT batch size not found")
        
    def test_11_performance_metrics(self):
        """Test performance and resource usage."""
        # Check if processing completed in reasonable time
        # This is more of a monitoring test
        
        # Check enhancement processing times
        if self.enhancement_metrics_dir.exists():
            summary_file = self.enhancement_metrics_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                # Check average processing time
                if "summary" in summary and "processing_time" in summary["summary"]:
                    avg_time = summary["summary"]["processing_time"].get("mean", 0)
                    self.assertLess(avg_time, 5.0,  # Should process each sample in less than 5 seconds
                                   f"Average processing time too high: {avg_time:.2f}s")
                                   

if __name__ == "__main__":
    unittest.main(verbosity=2)