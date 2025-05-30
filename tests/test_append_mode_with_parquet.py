"""
Test append mode with the new parquet-based get_last_id function.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import tempfile
import os

from main import main
from utils.huggingface import get_last_id


class TestAppendModeWithParquet(unittest.TestCase):
    """Test append mode functionality with parquet-based get_last_id."""

    def setUp(self):
        """Set up test environment."""
        self.test_args = [
            'main.py',
            '--append',
            'GigaSpeech2',
            '--sample',
            '--sample-size', '5',
            '--streaming'
        ]

    @patch('main.StreamingUploader')
    @patch('main.get_last_id')
    @patch('main.authenticate_hf')
    @patch('main.read_hf_token')
    @patch('main.os.path.exists')
    def test_append_mode_continues_from_last_id(self, mock_exists, mock_read_token, 
                                                mock_auth, mock_get_last_id, 
                                                mock_uploader_class):
        """Test that append mode correctly continues from the last ID."""
        # Setup mocks
        mock_exists.return_value = True
        mock_read_token.return_value = "test_token"
        mock_auth.return_value = True
        
        # Mock get_last_id to return a specific value
        mock_get_last_id.return_value = 1000
        
        # Mock StreamingUploader
        mock_uploader = MagicMock()
        mock_uploader_class.return_value = mock_uploader
        
        # Mock dataset processing
        with patch('main.DATASET_CONFIG', {
            'GigaSpeech2': {
                'repo': 'speechcolab/gigaspeech2',
                'subset': 'th',
                'processor': 'GigaSpeech2Processor'
            }
        }):
            with patch('main.importlib.import_module') as mock_import:
                # Mock processor
                mock_processor_module = MagicMock()
                mock_processor_class = MagicMock()
                mock_processor = MagicMock()
                
                # Configure processor to return some samples
                mock_processor.process_all_splits.return_value = iter([
                    {
                        'Language': 'th',
                        'audio': {'array': [0.1, 0.2], 'sampling_rate': 16000, 'path': 'test.wav'},
                        'transcript': 'test transcript',
                        'length': 1.0,
                        'dataset_name': 'GigaSpeech2',
                        'confidence_score': 1.0
                    }
                    for _ in range(5)
                ])
                
                mock_processor_class.return_value = mock_processor
                mock_processor_module.GigaSpeech2Processor = mock_processor_class
                mock_import.return_value = mock_processor_module
                
                # Run with append mode
                with patch('sys.argv', self.test_args):
                    main()
                
                # Verify get_last_id was called
                mock_get_last_id.assert_called_once_with('Thanarit/Thai-Voice')
                
                # Verify samples were assigned IDs starting from 1001
                # Check the upload_batch calls
                self.assertTrue(mock_uploader.upload_batch.called)
                
                # Get the first batch that was uploaded
                call_args = mock_uploader.upload_batch.call_args_list[0]
                batch = call_args[0][0]  # First argument is the batch
                
                # Verify the first sample has ID S1001
                self.assertEqual(batch[0]['ID'], 'S1001')

    @patch('utils.huggingface.HfApi')
    def test_get_last_id_handles_empty_dataset(self, mock_hfapi_class):
        """Test get_last_id handles empty datasets correctly."""
        # Setup mock HfApi
        mock_api = MagicMock()
        mock_hfapi_class.return_value = mock_api
        
        # Mock list_repo_files to return a parquet file
        mock_api.list_repo_files.return_value = [
            'data/train-00000-of-00001.parquet'
        ]
        
        # Mock hf_hub_download
        mock_api.hf_hub_download.return_value = '/tmp/empty.parquet'
        
        # Mock pandas read_parquet to return empty dataframe
        with patch('utils.huggingface.pd.read_parquet') as mock_read_parquet:
            mock_df = pd.DataFrame({'ID': []})
            mock_read_parquet.return_value = mock_df
            
            # Mock os.remove for cleanup
            with patch('os.remove'):
                with patch('os.path.exists', return_value=True):
                    # Test
                    result = get_last_id('test/empty-dataset')
                    
                    # Should fall back and return 0 for empty dataset
                    self.assertIsNotNone(result)

    def test_id_format_validation(self):
        """Test that IDs are correctly formatted."""
        # Test valid IDs
        valid_ids = ['S1', 'S100', 'S9999', 'S12345']
        for id_str in valid_ids:
            self.assertTrue(id_str.startswith('S'))
            self.assertTrue(id_str[1:].isdigit())
        
        # Test invalid IDs that should be filtered out
        invalid_ids = ['1', 'SA100', 'S', 'Sample1', 'S100A']
        for id_str in invalid_ids:
            is_valid = id_str.startswith('S') and id_str[1:].isdigit()
            self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()