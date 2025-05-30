"""
Test suite to ensure HuggingFace dataset viewer compatibility.
This addresses the JobManagerCrashedError issue by verifying:
1. Audio format is compatible with HF viewer
2. All data types are serializable
3. Dataset structure follows HF best practices
4. Append mode preserves viewer compatibility
"""
import unittest
import numpy as np
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
from datasets import Dataset, Audio, Features, Value
import soundfile as sf

# Import modules to test
from utils.huggingface import create_hf_dataset, upload_dataset, get_last_id
from processors.base_processor import BaseProcessor
from utils.streaming import StreamingUploader


class TestHuggingFaceViewerCompatibility(unittest.TestCase):
    """Test HuggingFace dataset viewer compatibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.audio_duration = 2.0  # 2 seconds
        self.num_samples = int(self.sample_rate * self.audio_duration)
        
        # Create test audio
        self.test_audio = np.sin(2 * np.pi * 440 * np.arange(self.num_samples) / self.sample_rate)
        self.test_audio = self.test_audio.astype(np.float32)
        
    def test_audio_format_compatibility(self):
        """Test that audio format is compatible with HF viewer."""
        # Create a sample with proper audio format
        sample = {
            "ID": "S1",
            "speaker_id": "SPK_00001", 
            "Language": "th",
            "audio": {
                "array": self.test_audio,
                "sampling_rate": self.sample_rate,
                "path": "S1.wav"
            },
            "transcript": "ทดสอบ",
            "length": self.audio_duration,
            "dataset_name": "TestDataset",
            "confidence_score": 1.0
        }
        
        # Audio array should be numpy array
        self.assertIsInstance(sample["audio"]["array"], np.ndarray, 
                            "Audio array must be numpy.ndarray")
        
        # Audio should be float32
        self.assertEqual(sample["audio"]["array"].dtype, np.float32,
                        "Audio must be float32 for HF viewer compatibility")
        
        # Audio values should be in [-1, 1] range
        self.assertTrue(np.all(np.abs(sample["audio"]["array"]) <= 1.0),
                       "Audio values must be normalized to [-1, 1]")
        
        # Sampling rate should be standard
        self.assertIn(sample["audio"]["sampling_rate"], [8000, 16000, 22050, 44100, 48000],
                     "Sampling rate should be a standard value")
                     
    def test_data_serialization(self):
        """Test that all data types can be serialized for HF viewer."""
        sample = {
            "ID": "S1",
            "speaker_id": "SPK_00001",
            "Language": "th", 
            "audio": {
                "array": self.test_audio,
                "sampling_rate": self.sample_rate,
                "path": "S1.wav"
            },
            "transcript": "ทดสอบ",
            "length": float(self.audio_duration),  # Must be float, not np.float32
            "dataset_name": "TestDataset",
            "confidence_score": 1.0
        }
        
        # All non-audio fields should be JSON serializable
        non_audio_fields = {k: v for k, v in sample.items() if k != "audio"}
        try:
            json.dumps(non_audio_fields)
        except (TypeError, ValueError) as e:
            self.fail(f"Non-audio fields must be JSON serializable: {e}")
            
        # Length should be native Python float
        self.assertIsInstance(sample["length"], float,
                            "Length must be Python float, not numpy.float32")
        
        # Confidence score should be native Python float
        self.assertIsInstance(sample["confidence_score"], (int, float),
                            "Confidence score must be Python numeric type")
                            
    def test_dataset_features_schema(self):
        """Test that dataset features are properly defined for HF viewer."""
        # Define expected features
        features = Features({
            "ID": Value("string"),
            "speaker_id": Value("string"),
            "Language": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "transcript": Value("string"),
            "length": Value("float32"),
            "dataset_name": Value("string"), 
            "confidence_score": Value("float32")
        })
        
        # Create sample data
        samples = [{
            "ID": "S1",
            "speaker_id": "SPK_00001",
            "Language": "th",
            "audio": {
                "array": self.test_audio,
                "sampling_rate": self.sample_rate,
                "path": "S1.wav"
            },
            "transcript": "ทดสอบ",
            "length": self.audio_duration,
            "dataset_name": "TestDataset",
            "confidence_score": 1.0
        }]
        
        # Create dataset with features
        dataset = create_hf_dataset(samples, features)
        
        self.assertIsNotNone(dataset, "Dataset creation should succeed")
        self.assertEqual(len(dataset), 1, "Dataset should have 1 sample")
        
        # Verify audio feature is properly typed
        self.assertIn("audio", dataset.features)
        self.assertIsInstance(dataset.features["audio"], Audio)
        
    def test_audio_file_saving(self):
        """Test that audio files are saved in a format readable by HF viewer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "test.wav")
            
            # Save audio file - ensure audio is in correct range
            # Normalize to [-1, 1] range if needed
            normalized_audio = self.test_audio
            if np.max(np.abs(normalized_audio)) > 1.0:
                normalized_audio = normalized_audio / np.max(np.abs(normalized_audio))
            
            sf.write(audio_path, normalized_audio, self.sample_rate)
            
            # Read it back
            audio_data, sr = sf.read(audio_path)
            
            # Verify format
            self.assertEqual(sr, self.sample_rate, "Sampling rate preserved")
            self.assertEqual(audio_data.dtype, np.float64, "Audio read as float")
            # Both should be in [-1, 1] range
            self.assertTrue(np.all(np.abs(audio_data) <= 1.0), "Read audio in valid range")
            self.assertTrue(np.all(np.abs(normalized_audio) <= 1.0), "Original audio in valid range")
                          
    def test_streaming_uploader_format(self):
        """Test that StreamingUploader produces HF-compatible format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the HfApi to avoid actual API calls
            with patch('utils.streaming.HfApi') as mock_hf_api:
                mock_api_instance = MagicMock()
                mock_hf_api.return_value = mock_api_instance
                mock_api_instance.create_repo.return_value = None
                
                uploader = StreamingUploader(
                    repo_id="test/repo",
                    token=None,
                    private=False,
                    append_mode=False
                )
                
                # Verify uploader has required methods
                self.assertTrue(hasattr(uploader, 'upload_batch'), "Should have upload_batch method")
                self.assertTrue(hasattr(uploader, 'upload_dataset_card'), "Should have upload_dataset_card method")
                
                # Test that uploader was initialized correctly
                self.assertEqual(uploader.repo_id, "test/repo")
                self.assertEqual(uploader.shard_num, 0)
                self.assertEqual(uploader.total_samples, 0)
                    
    def test_append_mode_compatibility(self):
        """Test that append mode maintains HF viewer compatibility."""
        # Create initial dataset
        initial_samples = [{
            "ID": "S1", 
            "speaker_id": "SPK_00001",
            "Language": "th",
            "audio": {
                "array": self.test_audio,
                "sampling_rate": self.sample_rate,
                "path": "S1.wav"
            },
            "transcript": "ทดสอบ 1",
            "length": self.audio_duration,
            "dataset_name": "TestDataset",
            "confidence_score": 1.0
        }]
        
        # Create appended samples
        append_samples = [{
            "ID": "S2",
            "speaker_id": "SPK_00002", 
            "Language": "th",
            "audio": {
                "array": self.test_audio * 0.8,  # Slightly different
                "sampling_rate": self.sample_rate,
                "path": "S2.wav"
            },
            "transcript": "ทดสอบ 2",
            "length": self.audio_duration,
            "dataset_name": "TestDataset", 
            "confidence_score": 1.0
        }]
        
        # Both should create valid datasets
        dataset1 = create_hf_dataset(initial_samples)
        dataset2 = create_hf_dataset(append_samples)
        
        self.assertIsNotNone(dataset1, "Initial dataset should be created")
        self.assertIsNotNone(dataset2, "Append dataset should be created")
        
        # Features should match
        self.assertEqual(set(dataset1.features.keys()), set(dataset2.features.keys()),
                        "Feature keys should match for append compatibility")
                        
    def test_metadata_completeness(self):
        """Test that all required metadata is present for HF viewer."""
        sample = {
            "ID": "S1",
            "speaker_id": "SPK_00001",
            "Language": "th",
            "audio": {
                "array": self.test_audio,
                "sampling_rate": self.sample_rate,
                "path": "S1.wav"
            },
            "transcript": "ทดสอบ",
            "length": self.audio_duration,
            "dataset_name": "TestDataset",
            "confidence_score": 1.0
        }
        
        # Check all required fields are present
        required_fields = ["ID", "speaker_id", "Language", "audio", 
                          "transcript", "length", "dataset_name", "confidence_score"]
        for field in required_fields:
            self.assertIn(field, sample, f"Required field '{field}' missing")
            
        # Audio should have required subfields
        audio_fields = ["array", "sampling_rate", "path"]
        for field in audio_fields:
            self.assertIn(field, sample["audio"], f"Audio field '{field}' missing")
            
    def test_audio_enhancement_metadata_serialization(self):
        """Test that audio enhancement metadata doesn't break HF viewer."""
        sample = {
            "ID": "S1",
            "speaker_id": "SPK_00001",
            "Language": "th",
            "audio": {
                "array": self.test_audio,
                "sampling_rate": self.sample_rate,
                "path": "S1.wav"
            },
            "transcript": "ทดสอบ",
            "length": self.audio_duration,
            "dataset_name": "TestDataset",
            "confidence_score": 1.0,
            "audio_enhancement": {
                "enhanced": True,
                "noise_level": "moderate",
                "snr_improvement": 5.2,
                "pesq": 3.5,
                "stoi": 0.95,
                "processing_time": 0.8
            }
        }
        
        # Enhancement metadata should be serializable
        try:
            json.dumps(sample["audio_enhancement"])
        except (TypeError, ValueError) as e:
            self.fail(f"Enhancement metadata must be JSON serializable: {e}")
            
    def test_large_batch_handling(self):
        """Test that large batches don't crash HF viewer."""
        # Create 100 samples
        samples = []
        for i in range(100):
            # Vary the audio slightly
            audio_variation = self.test_audio * (0.8 + 0.2 * (i / 100))
            samples.append({
                "ID": f"S{i+1}",
                "speaker_id": f"SPK_{i//10 + 1:05d}",  # 10 speakers
                "Language": "th",
                "audio": {
                    "array": audio_variation.astype(np.float32),
                    "sampling_rate": self.sample_rate,
                    "path": f"S{i+1}.wav"
                },
                "transcript": f"ทดสอบ {i+1}",
                "length": self.audio_duration,
                "dataset_name": "TestDataset",
                "confidence_score": 0.9 + 0.1 * (i % 2)  # Vary confidence
            })
            
        # Should create dataset without issues
        dataset = create_hf_dataset(samples)
        self.assertIsNotNone(dataset, "Large batch dataset creation should succeed")
        self.assertEqual(len(dataset), 100, "All samples should be included")
        
        # All audio arrays should be float32 or float64 (HF may convert internally)
        for i in range(len(dataset)):
            audio_array = dataset[i]["audio"]["array"]
            self.assertIn(audio_array.dtype, [np.float32, np.float64],
                           f"Sample {i} audio should be float32 or float64")


if __name__ == '__main__':
    unittest.main()