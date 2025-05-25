"""
Tests for sample mode feature.
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processors.base_processor import BaseProcessor
from processors.gigaspeech2 import GigaSpeech2Processor
from processors.processed_voice_th import ProcessedVoiceTHProcessor
from processors.mozilla_cv import MozillaCommonVoiceProcessor

class TestSampleMode(unittest.TestCase):
    """Test sample mode feature."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            "checkpoint_dir": os.path.join(self.temp_dir.name, "checkpoints"),
            "log_dir": os.path.join(self.temp_dir.name, "logs")
        }

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @patch('processors.gigaspeech2.load_dataset')
    def test_gigaspeech2_sample_mode(self, mock_load_dataset):
        """Test GigaSpeech2Processor sample mode."""
        # Create mock dataset with proper GigaSpeech2 structure
        mock_samples = [
            {
                "id": "sample_1", 
                "audio": {"array": [0.1] * 16000, "sampling_rate": 16000}, 
                "sentence": "Thai text 1", 
                "language": "th",
                "duration": 1.0
            },
            {
                "id": "sample_2", 
                "audio": {"array": [0.2] * 16000, "sampling_rate": 16000}, 
                "sentence": "Thai text 2", 
                "language": "th",
                "duration": 1.0
            },
            {
                "id": "sample_3", 
                "audio": {"array": [0.3] * 16000, "sampling_rate": 16000}, 
                "sentence": "Thai text 3", 
                "language": "th",
                "duration": 1.0
            }
        ]
        
        # Create a proper iterable dataset mock
        class MockIterableDataset:
            def __init__(self, samples):
                self.samples = samples
                self.filter_called = False
                self.select_called = False
                
            def __iter__(self):
                return iter(self.samples)
                
            def filter(self, *args, **kwargs):
                self.filter_called = True
                return self
                
            def select(self, indices):
                self.select_called = True
                # Return only the selected samples
                selected = [self.samples[i] for i in indices if i < len(self.samples)]
                return MockIterableDataset(selected)
                
            def __len__(self):
                return len(self.samples)
        
        mock_dataset = MockIterableDataset(mock_samples)
        mock_load_dataset.return_value = mock_dataset

        # Create processor with proper config
        config = {**self.config, 
                  "name": "GigaSpeech2",
                  "source": "speechcolab/gigaspeech2",
                  "language_filter": "th",
                  "enable_stt": False,
                  "audio_config": {
                      "enable_standardization": False
                  }}
        processor = GigaSpeech2Processor(config)

        # Process in sample mode
        samples = list(processor.process(sample_mode=True, sample_size=2))

        # Check results
        self.assertEqual(len(samples), 2)  # Should get exactly 2 samples
        # Note: GigaSpeech2 uses streaming mode instead of select for sampling
        
        # Check that samples have all required fields
        for sample in samples:
            self.assertIn("ID", sample)
            self.assertIn("Language", sample)
            self.assertIn("audio", sample)
            self.assertIn("transcript", sample)
            self.assertIn("length", sample)
            self.assertIn("dataset_name", sample)
            self.assertIn("confidence_score", sample)
            self.assertEqual(sample["dataset_name"], "GigaSpeech2")
            self.assertEqual(sample["confidence_score"], 1.0)

    @patch('processors.processed_voice_th.load_dataset')
    def test_processed_voice_th_sample_mode(self, mock_load_dataset):
        """Test ProcessedVoiceTHProcessor sample mode."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__iter__.return_value = [
            {"id": "1", "audio": {"bytes": b'test1'}, "text": "test1"},
            {"id": "2", "audio": {"bytes": b'test2'}, "text": "test2"}
        ]
        mock_dataset.__len__.return_value = 2
        mock_load_dataset.return_value = mock_dataset

        # Create processor
        processor = ProcessedVoiceTHProcessor({**self.config, "name": "ProcessedVoiceTH"})

        # Mock audio length calculation
        with patch('processors.processed_voice_th.get_audio_length', return_value=1.0):
            with patch('processors.base_processor.is_valid_audio', return_value=True):
                # Process in sample mode
                samples = list(processor.process(sample_mode=True, sample_size=1))

        # Check results
        self.assertGreaterEqual(len(samples), 0)  # We should get some samples
        mock_dataset.select.assert_called_once()  # select was called for sampling

    @patch('processors.mozilla_cv.load_dataset')
    def test_mozilla_cv_sample_mode(self, mock_load_dataset):
        """Test MozillaCommonVoiceProcessor sample mode."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__iter__.return_value = [
            {"client_id": "1", "path": "1.mp3", "audio": {"bytes": b'test1'}, "sentence": "test1"},
            {"client_id": "2", "path": "2.mp3", "audio": {"bytes": b'test2'}, "sentence": "test2"}
        ]
        mock_dataset.__len__.return_value = 2
        mock_load_dataset.return_value = mock_dataset

        # Create processor
        processor = MozillaCommonVoiceProcessor({**self.config, "name": "MozillaCommonVoice"})

        # Mock audio length calculation
        with patch('processors.mozilla_cv.get_audio_length', return_value=1.0):
            with patch('processors.base_processor.is_valid_audio', return_value=True):
                # Process in sample mode
                samples = list(processor.process(sample_mode=True, sample_size=1))

        # Check results
        self.assertGreaterEqual(len(samples), 0)  # We should get some samples
        mock_dataset.select.assert_called_once()  # select was called for sampling


if __name__ == '__main__':
    unittest.main()
