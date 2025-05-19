"""
Tests for audio utilities.
"""

import unittest
import os
import io
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.audio import is_valid_audio, get_audio_length, convert_audio_format

class TestAudioUtils(unittest.TestCase):
    """Test audio utility functions."""
    
    @patch('utils.audio.librosa.load')
    def test_is_valid_audio_valid(self, mock_load):
        """Test is_valid_audio with valid audio data."""
        # Mock librosa.load to return valid audio data
        mock_load.return_value = (MagicMock(), 16000)
        
        # Create dummy audio data
        audio_data = b'dummy audio data'
        
        # Test function
        result = is_valid_audio(audio_data)
        
        # Check result
        self.assertTrue(result)
        mock_load.assert_called_once()
    
    @patch('utils.audio.librosa.load')
    def test_is_valid_audio_invalid(self, mock_load):
        """Test is_valid_audio with invalid audio data."""
        # Mock librosa.load to raise an exception
        mock_load.side_effect = Exception("Invalid audio data")
        
        # Create dummy audio data
        audio_data = b'invalid audio data'
        
        # Test function
        result = is_valid_audio(audio_data)
        
        # Check result
        self.assertFalse(result)
        mock_load.assert_called_once()
    
    def test_is_valid_audio_empty(self):
        """Test is_valid_audio with empty audio data."""
        # Test function with empty data
        result = is_valid_audio(b'')
        
        # Check result
        self.assertFalse(result)
    
    @patch('utils.audio.librosa.load')
    def test_get_audio_length(self, mock_load):
        """Test get_audio_length."""
        # Mock librosa.load to return audio data with known length
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 16000
        mock_load.return_value = (mock_audio, 16000)
        
        # Create dummy audio data
        audio_data = b'dummy audio data'
        
        # Test function
        result = get_audio_length(audio_data)
        
        # Check result
        self.assertEqual(result, 1.0)  # 16000 samples / 16000 Hz = 1.0 seconds
        mock_load.assert_called_once()
    
    @patch('utils.audio.librosa.load')
    def test_get_audio_length_error(self, mock_load):
        """Test get_audio_length with error."""
        # Mock librosa.load to raise an exception
        mock_load.side_effect = Exception("Error loading audio")
        
        # Create dummy audio data
        audio_data = b'invalid audio data'
        
        # Test function
        result = get_audio_length(audio_data)
        
        # Check result
        self.assertIsNone(result)
        mock_load.assert_called_once()
    
    @patch('utils.audio.librosa.load')
    @patch('utils.audio.sf.write')
    def test_convert_audio_format(self, mock_write, mock_load):
        """Test convert_audio_format."""
        # Mock librosa.load to return audio data
        mock_audio = MagicMock()
        mock_load.return_value = (mock_audio, 16000)
        
        # Mock soundfile.write to write to buffer
        def mock_sf_write(buf, data, sr, format):
            buf.write(b'converted audio data')
            return None
        
        mock_write.side_effect = mock_sf_write
        
        # Create dummy audio data
        audio_data = b'dummy audio data'
        
        # Test function
        result = convert_audio_format(audio_data, target_format="wav")
        
        # Check result
        self.assertEqual(result, b'converted audio data')
        mock_load.assert_called_once()
        mock_write.assert_called_once()
    
    @patch('utils.audio.librosa.load')
    def test_convert_audio_format_error(self, mock_load):
        """Test convert_audio_format with error."""
        # Mock librosa.load to raise an exception
        mock_load.side_effect = Exception("Error loading audio")
        
        # Create dummy audio data
        audio_data = b'invalid audio data'
        
        # Test function
        result = convert_audio_format(audio_data, target_format="wav")
        
        # Check result
        self.assertIsNone(result)
        mock_load.assert_called_once()

if __name__ == '__main__':
    unittest.main()
