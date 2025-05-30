"""
Final verification test for issue #8 fix
Tests that duration tracking works correctly in streaming mode
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import shutil


class TestIssue8FinalVerification(unittest.TestCase):
    """Final verification that issue #8 is fixed"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_main_py_has_duration_tracking(self):
        """Verify main.py has proper duration tracking"""
        
        # Read main.py
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for duration variable initialization
        self.assertIn('total_duration_seconds = 0', content, 
                     "total_duration_seconds must be initialized")
        self.assertIn('dataset_duration_seconds = 0', content,
                     "dataset_duration_seconds must be initialized")
        
        # Check for duration accumulation
        self.assertIn('dataset_duration_seconds += sample_duration', content,
                     "dataset duration must be accumulated")
        self.assertIn('total_duration_seconds += sample_duration', content,
                     "total duration must be accumulated")
        
        # Check for duration in logging
        self.assertIn('dataset_duration_hours = dataset_duration_seconds / 3600', content,
                     "dataset duration must be converted to hours")
        
        # Check for total duration calculation
        self.assertIn('total_duration_hours = total_duration_seconds / 3600', content,
                     "total duration must be converted to hours")
        
        # Check that duration is passed to dataset card
        self.assertIn('"total_duration_hours": total_duration_hours', content,
                     "total duration hours must be passed to dataset card")
    
    @patch('main.StreamingUploader')
    @patch('main.read_hf_token')
    def test_duration_tracking_simple_case(self, mock_read_token, mock_uploader_class):
        """Test duration tracking with a simple mock scenario"""
        
        # Import after patching to avoid import errors
        from main import process_streaming_mode
        
        # Mock token
        mock_read_token.return_value = "test-token"
        
        # Mock uploader
        mock_uploader = MagicMock()
        mock_uploader.upload_batch.return_value = (True, "shard.parquet")
        mock_uploader_class.return_value = mock_uploader
        
        # Create test args
        args = Mock()
        args.append = False
        args.no_upload = False
        args.hf_repo = "test/repo"
        args.private = False
        args.streaming = True
        args.sample = True
        args.sample_size = 3
        args.resume = False
        args.enable_speaker_id = False
        args.enable_stt = False
        args.no_stt = True
        args.enable_audio_enhancement = False
        args.streaming_batch_size = 10
        args.upload_batch_size = 10
        
        # Mock processor
        with patch('main.create_processor') as mock_create_processor:
            mock_processor = MagicMock()
            
            # Create test samples with duration
            test_samples = [
                {'ID': 'S1', 'transcript': 'test1', 'length': 1.5, 'audio': {'array': [0.1], 'sampling_rate': 16000}},
                {'ID': 'S2', 'transcript': 'test2', 'length': 2.0, 'audio': {'array': [0.2], 'sampling_rate': 16000}},
                {'ID': 'S3', 'transcript': 'test3', 'length': 1.0, 'audio': {'array': [0.3], 'sampling_rate': 16000}},
            ]
            
            mock_processor.process_all_splits.return_value = iter(test_samples)
            mock_create_processor.return_value = mock_processor
            
            # Run processing
            result = process_streaming_mode(args, ['TestDataset'])
            
            # Verify success
            self.assertEqual(result, 0)
            
            # Verify dataset card was called with duration
            mock_uploader.upload_dataset_card.assert_called_once()
            dataset_info = mock_uploader.upload_dataset_card.call_args[0][0]
            
            # Check duration was calculated correctly
            # 1.5 + 2.0 + 1.0 = 4.5 seconds = 0.00125 hours
            expected_hours = 4.5 / 3600.0
            self.assertAlmostEqual(dataset_info['total_duration_hours'], expected_hours, places=5)
            self.assertEqual(dataset_info['total_samples'], 3)


if __name__ == '__main__':
    unittest.main()