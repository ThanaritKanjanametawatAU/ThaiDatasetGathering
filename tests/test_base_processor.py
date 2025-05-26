"""
Tests for base processor.
"""

import unittest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processors.base_processor import BaseProcessor, ValidationError
from config import ErrorCategory

class TestProcessor(BaseProcessor):
    """Test processor implementation."""
    
    def process(self, checkpoint=None):
        """Process the dataset."""
        yield {"ID": "S1", "Language": "th", "audio": b'test', "transcript": "test", "length": 1.0}
    
    def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
        """Process the dataset in streaming mode."""
        yield {"ID": "temp_0", "Language": "th", "audio": {"path": "test.wav", "bytes": b'test'}, "transcript": "test", "length": 1.0}
    
    def get_dataset_info(self):
        """Return dataset information."""
        return {"name": self.name, "total_samples": 1}
    
    def estimate_size(self):
        """Estimate dataset size."""
        return 1

class TestBaseProcessor(unittest.TestCase):
    """Test BaseProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            "name": "TestProcessor",
            "checkpoint_dir": os.path.join(self.temp_dir.name, "checkpoints"),
            "log_dir": os.path.join(self.temp_dir.name, "logs")
        }
        self.processor = TestProcessor(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_validate_sample_valid(self):
        """Test validate_sample with valid sample."""
        sample = {
            "ID": "S1",
            "speaker_id": "SPK_00001",
            "Language": "th",
            "audio": {"path": "test.wav", "bytes": b'test'},
            "transcript": "test",
            "length": 1.0,
            "dataset_name": "TestDataset",
            "confidence_score": 1.0
        }
        
        with patch('processors.base_processor.is_valid_audio', return_value=True):
            errors = self.processor.validate_sample(sample)
        
        self.assertEqual(errors, [])
    
    def test_validate_sample_invalid(self):
        """Test validate_sample with invalid sample."""
        sample = {
            "ID": "invalid",
            "Language": "en",
            "audio": b'',
            "length": -1.0
        }
        
        with patch('processors.base_processor.is_valid_audio', return_value=False):
            errors = self.processor.validate_sample(sample)
        
        self.assertGreater(len(errors), 0)
    
    def test_save_load_checkpoint(self):
        """Test save_checkpoint and load_checkpoint."""
        checkpoint_data = {
            "processed_count": 10,
            "current_index": 11,
            "processed_ids": ["1", "2", "3"]
        }
        
        # Save checkpoint
        checkpoint_file = self.processor.save_checkpoint(checkpoint_data)
        
        # Load checkpoint
        loaded_data = self.processor.load_checkpoint(checkpoint_file)
        
        # Check data - the unified format converts old keys
        self.assertEqual(loaded_data["samples_processed"], checkpoint_data["processed_count"])
        self.assertEqual(loaded_data["split_index"], checkpoint_data["current_index"])
        self.assertEqual(loaded_data["processed_ids"], checkpoint_data["processed_ids"])
        # In unified format, processor is at top level
        self.assertEqual(loaded_data["processor"], "TestProcessor")
        self.assertEqual(loaded_data["mode"], "unified")
    
    def test_load_checkpoint_invalid(self):
        """Test load_checkpoint with invalid file."""
        # Test with non-existent file
        loaded_data = self.processor.load_checkpoint("non_existent_file.json")
        self.assertIsNone(loaded_data)
        
        # Test with invalid content - checkpoint from different processor
        invalid_file = os.path.join(self.processor.checkpoint_dir, "invalid.json")
        os.makedirs(self.processor.checkpoint_dir, exist_ok=True)
        with open(invalid_file, 'w') as f:
            json.dump({
                "metadata": {"processor": "OtherProcessor", "version": "2.0"},
                "samples_processed": 10,
                "mode": "unified"
            }, f)
        
        loaded_data = self.processor.load_checkpoint(invalid_file)
        self.assertIsNone(loaded_data)
    
    def test_get_latest_checkpoint(self):
        """Test get_latest_checkpoint."""
        # Create multiple checkpoints
        checkpoint_data = {"test": "data"}
        
        # No checkpoints
        latest = self.processor.get_latest_checkpoint()
        self.assertIsNone(latest)
        
        # Create checkpoints
        self.processor.save_checkpoint(checkpoint_data)
        import time
        time.sleep(0.1)  # Ensure different modification times
        checkpoint2 = self.processor.save_checkpoint(checkpoint_data)
        
        # Get latest
        latest = self.processor.get_latest_checkpoint()
        self.assertEqual(latest, checkpoint2)
    
    def test_generate_id(self):
        """Test generate_id."""
        id_str = self.processor.generate_id(123)
        self.assertEqual(id_str, "S123")
    
    def test_extract_id_number(self):
        """Test extract_id_number."""
        # Valid ID
        num = self.processor.extract_id_number("S123")
        self.assertEqual(num, 123)
        
        # Invalid IDs
        self.assertIsNone(self.processor.extract_id_number("123"))
        self.assertIsNone(self.processor.extract_id_number("S"))
        self.assertIsNone(self.processor.extract_id_number("Sabc"))
    
    def test_get_next_id(self):
        """Test get_next_id."""
        # Empty list
        next_id = self.processor.get_next_id([])
        self.assertEqual(next_id, "S1")
        
        # Valid IDs
        next_id = self.processor.get_next_id(["S1", "S5", "S3"])
        self.assertEqual(next_id, "S6")
        
        # Mixed IDs
        next_id = self.processor.get_next_id(["S1", "invalid", "S5"])
        self.assertEqual(next_id, "S6")

if __name__ == '__main__':
    unittest.main()
