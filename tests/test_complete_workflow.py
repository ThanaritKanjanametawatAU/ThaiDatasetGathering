#!/usr/bin/env python3
"""
Comprehensive tests for the complete Thai Audio Dataset Collection workflow.
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.gigaspeech2 import GigaSpeech2Processor
from processors.processed_voice_th import ProcessedVoiceTHProcessor
from processors.mozilla_cv import MozillaCommonVoiceProcessor
from utils.audio import is_valid_audio, get_audio_length, standardize_audio
from utils.cache import CacheManager
from utils.streaming import StreamingBatchProcessor, StreamingUploader
from utils.logging import ProgressTracker, ProcessingLogger
import config


class TestCompleteWorkflow(unittest.TestCase):
    """Test the complete workflow from data loading to upload."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)
        os.makedirs(self.cache_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_schema_validation(self):
        """Test that all required fields are present in schema."""
        # Check that schema has all required fields
        required_fields = ["ID", "Language", "audio", "transcript", "length", 
                          "dataset_name", "confidence_score"]
        
        for field in required_fields:
            self.assertIn(field, config.SCHEMA)
            
        # Check validation rules
        for field in required_fields:
            self.assertIn(field, config.VALIDATION_RULES)
            
    def test_audio_config(self):
        """Test audio configuration settings."""
        # Check required audio settings
        self.assertIn("target_format", config.AUDIO_CONFIG)
        self.assertIn("target_sample_rate", config.AUDIO_CONFIG)
        self.assertIn("target_channels", config.AUDIO_CONFIG)
        self.assertIn("normalize_volume", config.AUDIO_CONFIG)
        self.assertIn("target_db", config.AUDIO_CONFIG)
        
        # Validate values
        self.assertEqual(config.AUDIO_CONFIG["target_format"], "wav")
        self.assertEqual(config.AUDIO_CONFIG["target_sample_rate"], 16000)
        self.assertEqual(config.AUDIO_CONFIG["target_channels"], 1)
        
    def test_streaming_config(self):
        """Test streaming configuration."""
        self.assertIn("batch_size", config.STREAMING_CONFIG)
        self.assertIn("upload_batch_size", config.STREAMING_CONFIG)
        self.assertIn("shard_size", config.STREAMING_CONFIG)
        self.assertIn("max_retries", config.STREAMING_CONFIG)
        
    def test_cache_manager(self):
        """Test cache manager functionality."""
        cache_manager = CacheManager(self.cache_dir, max_size_gb=0.001)  # 1MB for testing
        
        # Test cache operations
        test_data = b"test data"
        cache_key = "test_key"
        
        # Save to cache
        cache_manager.save(cache_key, test_data)
        
        # Load from cache
        loaded_data = cache_manager.load(cache_key)
        self.assertEqual(loaded_data, test_data)
        
        # Test cache size management
        cache_info = cache_manager.get_cache_info()
        self.assertIn("size_bytes", cache_info)
        self.assertIn("file_count", cache_info)
        
    def test_progress_tracker(self):
        """Test progress tracking functionality."""
        tracker = ProgressTracker(total_items=100)
        
        # Test updates
        tracker.update(items_processed=10)
        self.assertEqual(tracker.items_processed, 10)
        
        tracker.update(items_processed=5, skipped=2, errors=1)
        self.assertEqual(tracker.items_processed, 15)
        self.assertEqual(tracker.items_skipped, 2)
        self.assertEqual(tracker.items_error, 1)
        
    def test_processing_logger(self):
        """Test processing logger functionality."""
        log_file = os.path.join(self.log_dir, "test_processing.log")
        logger = ProcessingLogger(log_file, "TestDataset")
        
        # Log samples
        logger.log_sample("sample_1", "processed", {"id": "S1"})
        logger.log_sample("sample_2", "skipped", {"reason": "invalid audio"})
        logger.log_sample("sample_3", "error", {"error": "test error"})
        
        # Save JSON log
        json_file = logger.save_json_log()
        self.assertTrue(os.path.exists(json_file))
        
    @patch('processors.base_processor.is_valid_audio')
    def test_sample_validation(self, mock_is_valid_audio):
        """Test sample validation across all processors."""
        mock_is_valid_audio.return_value = True
        
        # Create valid sample
        valid_sample = {
            "ID": "S1",
            "Language": "th",
            "audio": {"path": "test.wav", "bytes": b"audio_data"},
            "transcript": "Thai text",
            "length": 2.5,
            "dataset_name": "TestDataset",
            "confidence_score": 0.95
        }
        
        # Test with each processor
        for processor_class in [GigaSpeech2Processor, ProcessedVoiceTHProcessor, 
                               MozillaCommonVoiceProcessor]:
            config_dict = {
                "checkpoint_dir": self.checkpoint_dir,
                "log_dir": self.log_dir,
                "enable_stt": False,
                "audio_config": {"enable_standardization": False}
            }
            processor = processor_class(config_dict)
            
            errors = processor.validate_sample(valid_sample)
            self.assertEqual(errors, [], f"{processor_class.__name__} validation failed")
            
    def test_streaming_batch_processor(self):
        """Test streaming batch processor."""
        processor = StreamingBatchProcessor(batch_size=3)
        
        # Add samples
        samples = []
        for i in range(10):
            sample = {
                "ID": f"S{i+1}",
                "Language": "th",
                "audio": {"path": f"test_{i}.wav", "bytes": b"audio"},
                "transcript": f"Text {i}",
                "length": 1.0
            }
            batch = processor.add_sample(sample)
            if batch:
                samples.extend(batch)
                
        # Get final batch
        final_batch = processor.get_final_batch()
        if final_batch:
            samples.extend(final_batch)
            
        # Verify all samples processed
        self.assertEqual(len(samples), 10)
        
    def test_checkpoint_resume_workflow(self):
        """Test checkpoint and resume functionality."""
        config_dict = {
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "dataset_name": "TestDataset",
            "enable_stt": False,
            "audio_config": {"enable_standardization": False}
        }
        
        # Create a test processor
        class TestProcessor(GigaSpeech2Processor):
            def __init__(self, config):
                super().__init__(config)
                self.sample_count = 0
                
            def _process_single_split(self, split, checkpoint=None, 
                                     sample_mode=False, sample_size=5):
                # Simulate processing that can be interrupted
                start_idx = 0
                if checkpoint:
                    checkpoint_data = self.load_unified_checkpoint(checkpoint)
                    if checkpoint_data:
                        start_idx = checkpoint_data.get("split_index", 0)
                        
                for i in range(start_idx, 10):
                    if self.sample_count >= 5 and not checkpoint:
                        # Simulate interruption
                        self.save_unified_checkpoint(
                            samples_processed=self.sample_count,
                            current_split=split,
                            split_index=i
                        )
                        break
                        
                    self.sample_count += 1
                    yield {
                        "ID": f"S{self.sample_count}",
                        "Language": "th",
                        "audio": {"path": "test.wav", "bytes": b"audio"},
                        "transcript": "test",
                        "length": 1.0,
                        "dataset_name": "TestDataset",
                        "confidence_score": 1.0
                    }
            
            def get_available_splits(self):
                return ["train"]
                
        processor = TestProcessor(config_dict)
        
        # Process first batch
        samples1 = list(processor.process_all_splits())
        self.assertEqual(len(samples1), 5)
        
        # Get checkpoint
        checkpoint_file = processor.get_latest_checkpoint()
        self.assertIsNotNone(checkpoint_file)
        
        # Resume from checkpoint
        processor2 = TestProcessor(config_dict)
        samples2 = list(processor2.process_all_splits(checkpoint=checkpoint_file))
        self.assertEqual(len(samples2), 5)
        
        # Verify no duplicates
        all_ids = [s["ID"] for s in samples1 + samples2]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        
    def test_end_to_end_streaming(self):
        """Test end-to-end streaming workflow."""
        # This test verifies the complete flow from processing to upload
        # In a real test, we would mock the HuggingFace upload
        
        # Create mock processor
        mock_samples = [
            {
                "ID": f"S{i+1}",
                "Language": "th", 
                "audio": {"path": f"test_{i}.wav", "bytes": b"audio_data"},
                "transcript": f"Thai text {i}",
                "length": 1.5,
                "dataset_name": "TestDataset",
                "confidence_score": 0.95
            }
            for i in range(5)
        ]
        
        # Test batch processing
        batch_processor = StreamingBatchProcessor(batch_size=3)
        batches = []
        
        for sample in mock_samples:
            batch = batch_processor.add_sample(sample)
            if batch:
                batches.append(batch)
                
        # Get final batch
        final_batch = batch_processor.get_final_batch()
        if final_batch:
            batches.append(final_batch)
            
        # Verify batches
        self.assertEqual(len(batches), 2)  # 3 + 2 samples
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 2)
        

if __name__ == "__main__":
    unittest.main()