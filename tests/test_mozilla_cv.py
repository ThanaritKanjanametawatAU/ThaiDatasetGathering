"""
Tests for Mozilla Common Voice processor.
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processors.mozilla_cv import MozillaCommonVoiceProcessor
from processors.base_processor import ValidationError

class TestMozillaCommonVoiceProcessor(unittest.TestCase):
    """Test MozillaCommonVoiceProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            "name": "MozillaCommonVoice",
            "source": "mozilla-foundation/common_voice_11_0",
            "language_filter": "th",
            "checkpoint_dir": os.path.join(self.temp_dir.name, "checkpoints"),
            "log_dir": os.path.join(self.temp_dir.name, "logs")
        }
        self.processor = MozillaCommonVoiceProcessor(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    @patch('processors.mozilla_cv.load_dataset')
    def test_get_dataset_info(self, mock_load_dataset):
        """Test get_dataset_info."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.features = {"audio": None, "sentence": None}
        mock_dataset.__len__.return_value = 100
        mock_load_dataset.return_value = mock_dataset
        
        # Test function
        info = self.processor.get_dataset_info()
        
        # Check result
        self.assertEqual(info["name"], "MozillaCommonVoice")
        self.assertEqual(info["source"], "mozilla-foundation/common_voice_11_0")
        self.assertEqual(info["language"], "th")
        self.assertEqual(info["total_samples"], 100)
        self.assertIn("features", info)
        mock_load_dataset.assert_called_once_with(
            "mozilla-foundation/common_voice_11_0", "th", split="train"
        )
    
    @patch('processors.mozilla_cv.load_dataset')
    def test_get_dataset_info_error(self, mock_load_dataset):
        """Test get_dataset_info with error."""
        # Mock load_dataset to raise an exception
        mock_load_dataset.side_effect = Exception("Test error")
        
        # Test function
        info = self.processor.get_dataset_info()
        
        # Check result
        self.assertEqual(info["name"], "MozillaCommonVoice")
        self.assertEqual(info["source"], "mozilla-foundation/common_voice_11_0")
        self.assertEqual(info["language"], "th")
        self.assertIn("error", info)
        mock_load_dataset.assert_called_once()
    
    @patch('processors.mozilla_cv.load_dataset')
    def test_estimate_size(self, mock_load_dataset):
        """Test estimate_size."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_load_dataset.return_value = mock_dataset
        
        # Test function
        size = self.processor.estimate_size()
        
        # Check result
        self.assertEqual(size, 100)
        mock_load_dataset.assert_called_once()
    
    @patch('processors.mozilla_cv.load_dataset')
    def test_estimate_size_error(self, mock_load_dataset):
        """Test estimate_size with error."""
        # Mock load_dataset to raise an exception
        mock_load_dataset.side_effect = Exception("Test error")
        
        # Test function
        size = self.processor.estimate_size()
        
        # Check result
        self.assertEqual(size, 0)
        mock_load_dataset.assert_called_once()
    
    @patch('processors.mozilla_cv.get_audio_length')
    @patch.object(MozillaCommonVoiceProcessor, 'preprocess_audio')
    @patch.object(MozillaCommonVoiceProcessor, 'create_hf_audio_format')
    def test_convert_sample(self, mock_create_hf_audio, mock_preprocess_audio, mock_get_audio_length):
        """Test _convert_sample."""
        # Mock audio preprocessing
        mock_preprocess_audio.return_value = b'preprocessed_audio'
        
        # Mock HF audio format creation
        mock_hf_audio = {
            "array": [0.1, 0.2, 0.3],
            "sampling_rate": 16000,
            "path": "S123.wav"
        }
        mock_create_hf_audio.return_value = mock_hf_audio
        
        # Mock get_audio_length
        mock_get_audio_length.return_value = 2.5
        
        # Create sample
        sample = {
            "client_id": "test_client",
            "path": "test_path",
            "audio": {"bytes": b'test_audio'},
            "sentence": "test sentence"
        }
        
        # Test function
        result = self.processor._convert_sample(sample, 123)
        
        # Check result
        self.assertEqual(result["ID"], "S123")
        self.assertEqual(result["Language"], "th")
        self.assertEqual(result["audio"], mock_hf_audio)
        self.assertEqual(result["transcript"], "test sentence")
        self.assertEqual(result["length"], 2.5)
        
        # Verify method calls
        mock_preprocess_audio.assert_called_once_with(b'test_audio', 'S123')
        mock_create_hf_audio.assert_called_once_with(b'preprocessed_audio', 'S123')
        mock_get_audio_length.assert_called_once_with(mock_hf_audio)
    
    @patch('processors.mozilla_cv.get_audio_length')
    def test_convert_sample_missing_audio(self, mock_get_audio_length):
        """Test _convert_sample with missing audio."""
        # Create sample with missing audio
        sample = {
            "client_id": "test_client",
            "path": "test_path",
            "sentence": "test sentence"
        }
        
        # Test function
        with self.assertRaises(ValidationError):
            self.processor._convert_sample(sample, 123)
        
        # Check that get_audio_length was not called
        mock_get_audio_length.assert_not_called()
    
    @patch('processors.mozilla_cv.get_audio_length')
    @patch.object(MozillaCommonVoiceProcessor, 'preprocess_audio')
    @patch.object(MozillaCommonVoiceProcessor, 'create_hf_audio_format')
    def test_convert_sample_missing_length(self, mock_create_hf_audio, mock_preprocess_audio, mock_get_audio_length):
        """Test _convert_sample with missing length."""
        # Mock audio preprocessing
        mock_preprocess_audio.return_value = b'preprocessed_audio'
        
        # Mock HF audio format creation
        mock_hf_audio = {
            "array": [0.1, 0.2, 0.3],
            "sampling_rate": 16000,
            "path": "S123.wav"
        }
        mock_create_hf_audio.return_value = mock_hf_audio
        
        # Mock get_audio_length to return None
        mock_get_audio_length.return_value = None
        
        # Create sample
        sample = {
            "client_id": "test_client",
            "path": "test_path",
            "audio": {"bytes": b'test_audio'},
            "sentence": "test sentence"
        }
        
        # Test function
        with self.assertRaises(ValidationError):
            self.processor._convert_sample(sample, 123)
        
        # Check that methods were called in order
        mock_preprocess_audio.assert_called_once_with(b'test_audio', 'S123')
        mock_create_hf_audio.assert_called_once_with(b'preprocessed_audio', 'S123')
        mock_get_audio_length.assert_called_once_with(mock_hf_audio)

if __name__ == '__main__':
    unittest.main()
