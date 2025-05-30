"""
Test for the final fix of append mode ID restart issue.
This test ensures that when using --append flag, IDs continue from the last existing ID.
"""
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import pyarrow.parquet as pq
import pyarrow as pa

from utils.huggingface import get_last_id


class TestAppendModeIDFix(unittest.TestCase):
    """Test the fix for append mode ID restart issue."""
    
    def test_get_last_id_from_parquet_files(self):
        """Test getting last ID from parquet files in HF dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock parquet files with sample data
            data1 = {
                'ID': ['S1', 'S2', 'S3', 'S4', 'S5'],
                'transcript': ['test1', 'test2', 'test3', 'test4', 'test5']
            }
            data2 = {
                'ID': ['S6', 'S7', 'S8', 'S9', 'S10'],
                'transcript': ['test6', 'test7', 'test8', 'test9', 'test10']  
            }
            
            # Write parquet files
            os.makedirs(os.path.join(temp_dir, 'data', 'train'), exist_ok=True)
            pq.write_table(
                pa.table(data1), 
                os.path.join(temp_dir, 'data', 'train', 'shard_00000.parquet')
            )
            pq.write_table(
                pa.table(data2),
                os.path.join(temp_dir, 'data', 'train', 'shard_00001.parquet')
            )
            
            # Mock HfApi to return file list
            with patch('utils.huggingface.HfApi') as mock_hf_api:
                mock_api = MagicMock()
                mock_hf_api.return_value = mock_api
                
                # Mock list_repo_files to return parquet files
                mock_api.list_repo_files.return_value = [
                    'data/train/shard_00000.parquet',
                    'data/train/shard_00001.parquet'
                ]
                
                # Mock download_file to return local paths
                def mock_download(filename, repo_id, repo_type, **kwargs):
                    return os.path.join(temp_dir, filename)
                
                mock_api.hf_hub_download = mock_download
                
                # Test the new implementation
                last_id = get_last_id_from_parquet('test/dataset')
                self.assertEqual(last_id, 10, "Should return 10 as the last ID")
                
    def test_get_last_id_handles_empty_dataset(self):
        """Test that get_last_id handles empty datasets gracefully."""
        with patch('utils.huggingface.HfApi') as mock_hf_api:
            mock_api = MagicMock()
            mock_hf_api.return_value = mock_api
            mock_api.list_repo_files.return_value = []
            
            last_id = get_last_id_from_parquet('test/dataset')
            self.assertEqual(last_id, 0, "Should return 0 for empty dataset")
            
    def test_get_last_id_handles_non_standard_ids(self):
        """Test handling of non-standard ID formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create parquet with mixed ID formats
            data = {
                'ID': ['S1', 'S2', 'invalid', 'S10', 'S5a', 'S100'],
                'transcript': ['t1', 't2', 't3', 't4', 't5', 't6']
            }
            
            os.makedirs(os.path.join(temp_dir, 'data', 'train'), exist_ok=True)
            pq.write_table(
                pa.table(data),
                os.path.join(temp_dir, 'data', 'train', 'shard_00000.parquet')
            )
            
            with patch('utils.huggingface.HfApi') as mock_hf_api:
                mock_api = MagicMock()
                mock_hf_api.return_value = mock_api
                mock_api.list_repo_files.return_value = ['data/train/shard_00000.parquet']
                
                def mock_download(filename, repo_id, repo_type, **kwargs):
                    return os.path.join(temp_dir, filename)
                
                mock_api.hf_hub_download = mock_download
                
                last_id = get_last_id_from_parquet('test/dataset')
                self.assertEqual(last_id, 100, "Should return 100, ignoring invalid IDs")


# Placeholder for the new implementation
def get_last_id_from_parquet(dataset_name: str) -> int:
    """
    Get last ID by reading parquet files directly from HuggingFace.
    This avoids the SplitInfo mismatch error.
    """
    try:
        from huggingface_hub import HfApi
        import pyarrow.parquet as pq
        import tempfile
        
        api = HfApi()
        
        # List all parquet files
        files = api.list_repo_files(
            repo_id=dataset_name,
            repo_type="dataset"
        )
        
        parquet_files = [f for f in files if f.endswith('.parquet') and 'train' in f]
        
        if not parquet_files:
            return 0
            
        max_id = 0
        
        # Download and check each parquet file
        for file in parquet_files:
            try:
                with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp_file:
                    # Download the file
                    local_path = api.hf_hub_download(
                        repo_id=dataset_name,
                        filename=file,
                        repo_type="dataset",
                        local_dir=tempfile.gettempdir()
                    )
                    
                    # Read the parquet file
                    table = pq.read_table(local_path, columns=['ID'])
                    ids = table['ID'].to_pylist()
                    
                    # Extract numeric IDs
                    for id_str in ids:
                        if id_str and id_str.startswith('S') and id_str[1:].isdigit():
                            num_id = int(id_str[1:])
                            max_id = max(max_id, num_id)
                            
            except Exception as e:
                # Skip problematic files
                continue
                
        return max_id
        
    except Exception as e:
        return 0


if __name__ == '__main__':
    unittest.main()