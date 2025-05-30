"""
Test the new parquet-based get_last_id function.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import tempfile
import os

from utils.huggingface import get_last_id


class TestGetLastIdParquet(unittest.TestCase):
    """Test the parquet-based get_last_id function."""

    @patch('utils.huggingface.HfApi')
    @patch('utils.huggingface.pd.read_parquet')
    @patch('utils.huggingface.load_dataset')
    def test_parquet_approach_success(self, mock_load_dataset, mock_read_parquet, mock_hfapi_class):
        """Test successful parquet file reading."""
        # Setup mock HfApi
        mock_api = MagicMock()
        mock_hfapi_class.return_value = mock_api
        
        # Mock list_repo_files to return parquet files
        mock_api.list_repo_files.return_value = [
            'data/train-00000-of-00002.parquet',
            'data/train-00001-of-00002.parquet',
            'other/file.txt'
        ]
        
        # Mock hf_hub_download to return a temp file path
        temp_path = '/tmp/test.parquet'
        mock_api.hf_hub_download.return_value = temp_path
        
        # Mock pandas read_parquet to return dataframes with IDs
        mock_df1 = pd.DataFrame({
            'ID': ['S1', 'S2', 'S3', 'S100'],
            'other_col': ['data1', 'data2', 'data3', 'data4']
        })
        mock_df2 = pd.DataFrame({
            'ID': ['S101', 'S102', 'S200', 'S250'],
            'other_col': ['data5', 'data6', 'data7', 'data8']
        })
        
        # Return different dataframes for each call
        mock_read_parquet.side_effect = [mock_df1, mock_df2]
        
        # Test
        result = get_last_id('test/dataset')
        
        # Verify
        self.assertEqual(result, 250)  # Max ID should be 250
        
        # Verify API calls
        mock_api.list_repo_files.assert_called_once_with(
            repo_id='test/dataset',
            repo_type='dataset'
        )
        
        # Verify that load_dataset was not called (parquet approach succeeded)
        mock_load_dataset.assert_not_called()

    @patch('utils.huggingface.HfApi')
    @patch('utils.huggingface.load_dataset')
    def test_fallback_to_load_dataset_no_parquet_files(self, mock_load_dataset, mock_hfapi_class):
        """Test fallback when no parquet files are found."""
        # Setup mock HfApi
        mock_api = MagicMock()
        mock_hfapi_class.return_value = mock_api
        
        # Mock list_repo_files to return no parquet files
        mock_api.list_repo_files.return_value = [
            'README.md',
            'config.json'
        ]
        
        # Mock load_dataset
        mock_dataset = MagicMock()
        mock_dataset.features = {'ID': 'string'}
        mock_dataset.__getitem__.return_value = ['S1', 'S2', 'S50']
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = get_last_id('test/dataset')
        
        # Verify
        self.assertEqual(result, 50)
        
        # Verify load_dataset was called as fallback
        mock_load_dataset.assert_called_once()

    @patch('utils.huggingface.HfApi')
    @patch('utils.huggingface.pd.read_parquet')
    @patch('utils.huggingface.load_dataset')
    def test_parquet_invalid_ids_fallback(self, mock_load_dataset, mock_read_parquet, mock_hfapi_class):
        """Test fallback when parquet files have no valid IDs."""
        # Setup mock HfApi
        mock_api = MagicMock()
        mock_hfapi_class.return_value = mock_api
        
        # Mock list_repo_files
        mock_api.list_repo_files.return_value = [
            'data/train-00000-of-00001.parquet'
        ]
        
        # Mock hf_hub_download
        mock_api.hf_hub_download.return_value = '/tmp/test.parquet'
        
        # Mock pandas read_parquet with invalid IDs
        mock_df = pd.DataFrame({
            'ID': ['invalid1', 'invalid2', None, ''],
            'other_col': ['data1', 'data2', 'data3', 'data4']
        })
        mock_read_parquet.return_value = mock_df
        
        # Mock load_dataset for fallback
        mock_dataset = MagicMock()
        mock_dataset.features = {'ID': 'string'}
        mock_dataset.__getitem__.return_value = ['S1', 'S2', 'S30']
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = get_last_id('test/dataset')
        
        # Verify
        self.assertEqual(result, 30)
        
        # Verify load_dataset was called as fallback
        mock_load_dataset.assert_called_once()

    @patch('utils.huggingface.HfApi')
    @patch('utils.huggingface.pd.read_parquet')
    @patch('utils.huggingface.load_dataset')
    def test_mixed_valid_invalid_ids(self, mock_load_dataset, mock_read_parquet, mock_hfapi_class):
        """Test with a mix of valid and invalid IDs in parquet files."""
        # Setup mock HfApi
        mock_api = MagicMock()
        mock_hfapi_class.return_value = mock_api
        
        # Mock list_repo_files
        mock_api.list_repo_files.return_value = [
            'data/train-00000-of-00001.parquet'
        ]
        
        # Mock hf_hub_download
        mock_api.hf_hub_download.return_value = '/tmp/test.parquet'
        
        # Mock pandas read_parquet with mixed IDs
        mock_df = pd.DataFrame({
            'ID': ['S1', 'invalid', 'S150', None, 'S99', 'ABC123', 'S5'],
            'other_col': ['data'] * 7
        })
        mock_read_parquet.return_value = mock_df
        
        # Test
        result = get_last_id('test/dataset')
        
        # Verify - should return 150 (max valid ID)
        self.assertEqual(result, 150)
        
        # Verify load_dataset was not called
        mock_load_dataset.assert_not_called()

    @patch('utils.huggingface.HfApi')
    @patch('utils.huggingface.load_dataset')
    @patch('os.remove')
    def test_file_cleanup(self, mock_remove, mock_load_dataset, mock_hfapi_class):
        """Test that downloaded parquet files are cleaned up."""
        # Setup mock HfApi
        mock_api = MagicMock()
        mock_hfapi_class.return_value = mock_api
        
        # Mock list_repo_files
        mock_api.list_repo_files.return_value = [
            'data/train-00000-of-00001.parquet'
        ]
        
        # Mock hf_hub_download
        temp_path = '/tmp/test_cleanup.parquet'
        mock_api.hf_hub_download.return_value = temp_path
        
        # Mock pandas read_parquet
        with patch('utils.huggingface.pd.read_parquet') as mock_read_parquet:
            mock_df = pd.DataFrame({'ID': ['S10']})
            mock_read_parquet.return_value = mock_df
            
            # Mock os.path.exists to return True
            with patch('os.path.exists', return_value=True):
                # Test
                result = get_last_id('test/dataset')
                
                # Verify cleanup was called
                mock_remove.assert_called_once_with(temp_path)


if __name__ == '__main__':
    unittest.main()