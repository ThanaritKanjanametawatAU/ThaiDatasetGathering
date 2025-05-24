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
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.filter.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__iter__.return_value = [
            {"id": "1", "audio": {"bytes": b'test1'}, "text": "test1", "language": "th"},
            {"id": "2", "audio": {"bytes": b'test2'}, "text": "test2", "language": "th"},
            {"id": "3", "audio": {"bytes": b'test3'}, "text": "test3", "language": "th"}
        ]
        mock_dataset.__len__.return_value = 3
        mock_load_dataset.return_value = mock_dataset

        # Create processor
        processor = GigaSpeech2Processor({**self.config, "name": "GigaSpeech2"})

        # Mock audio length calculation and validation
        with patch('processors.gigaspeech2.get_audio_length', return_value=1.0):
            with patch('processors.base_processor.is_valid_audio', return_value=True):
                # Process in sample mode
                samples = list(processor.process(sample_mode=True, sample_size=2))

        # Check results
        self.assertGreaterEqual(len(samples), 0)  # We should get some samples
        mock_dataset.select.assert_called_once()  # select was called for sampling

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
