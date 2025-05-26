#!/usr/bin/env python3
"""
Integration tests for --fresh flag behavior in main.py
"""

import unittest
import os
import tempfile
import shutil
import json
import subprocess
import sys
import numpy as np
from unittest.mock import patch, MagicMock, call

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from processors.speaker_identification import SpeakerIdentification


class TestMainFreshIntegration(unittest.TestCase):
    """Test integration of --fresh flag in main.py."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)
        
        # Create existing speaker model
        self.model_path = os.path.join(self.checkpoint_dir, "speaker_model.json")
        existing_model = {
            "speaker_counter": 75,
            "existing_clusters": {"0": "SPK_00001", "1": "SPK_00002"},
            "cluster_centroids": [[0.1] * 512, [0.4] * 512]  # Proper 512-dimensional centroids
        }
        
        with open(self.model_path, 'w') as f:
            json.dump(existing_model, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('main.CHECKPOINT_DIR')
    @patch('main.process_streaming_mode')
    def test_fresh_flag_passed_to_speaker_id(self, mock_streaming, mock_checkpoint_dir):
        """Test that --fresh flag is passed to SpeakerIdentification config."""
        mock_checkpoint_dir.__str__.return_value = self.checkpoint_dir
        mock_streaming.return_value = 0
        
        # Simulate command line args with --fresh
        test_args = [
            'main.py',
            '--fresh',
            'GigaSpeech2',
            '--enable-speaker-id',
            '--streaming',
            '--no-upload',
            '--sample',
            '--sample-size', '5'
        ]
        
        with patch('sys.argv', test_args):
            with patch('processors.speaker_identification.SpeakerIdentification') as mock_speaker_class:
                # Run main
                try:
                    main.main()
                except SystemExit:
                    pass
                
                # Check that SpeakerIdentification was called with fresh=True
                if mock_speaker_class.called:
                    args, kwargs = mock_speaker_class.call_args
                    config = args[0] if args else kwargs.get('config', {})
                    
                    # Config should have fresh=True
                    self.assertTrue(config.get('fresh', False), 
                                  "SpeakerIdentification config should have fresh=True")
    
    @patch('main.CHECKPOINT_DIR')
    def test_append_flag_no_fresh(self, mock_checkpoint_dir):
        """Test that --append flag does not set fresh=True."""
        mock_checkpoint_dir.__str__.return_value = self.checkpoint_dir
        
        # Simulate command line args with --append (no --fresh)
        test_args = [
            'main.py',
            '--append',
            'GigaSpeech2',
            '--enable-speaker-id',
            '--streaming',
            '--no-upload',
            '--sample',
            '--sample-size', '5'
        ]
        
        with patch('sys.argv', test_args):
            with patch('processors.speaker_identification.SpeakerIdentification') as mock_speaker_class:
                with patch('main.process_streaming_mode') as mock_streaming:
                    mock_streaming.return_value = 0
                    
                    # Run main
                    try:
                        main.main()
                    except SystemExit:
                        pass
                
                    # Check that SpeakerIdentification was called without fresh=True
                    if mock_speaker_class.called:
                        args, kwargs = mock_speaker_class.call_args
                        config = args[0] if args else kwargs.get('config', {})
                        
                        # Config should not have fresh=True (or have fresh=False)
                        self.assertFalse(config.get('fresh', False), 
                                       "SpeakerIdentification config should not have fresh=True with --append")
    
    def test_command_line_fresh_behavior(self):
        """Test actual command line behavior with --fresh flag."""
        # Create a simple test script that checks speaker counter
        test_script = os.path.join(self.temp_dir, "test_fresh.py")
        
        script_content = '''
import sys
import os
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.speaker_identification import SpeakerIdentification

checkpoint_dir = sys.argv[1]
fresh = sys.argv[2] == "True"

config = {
    'model_path': os.path.join(checkpoint_dir, 'speaker_model.json'),
    'clustering': {},
    'fresh': fresh
}

speaker_id = SpeakerIdentification(config)
print(f"speaker_counter:{speaker_id.speaker_counter}")
'''
        
        with open(test_script, 'w') as f:
            f.write(script_content)
        
        # Test with fresh=False (should load counter=75)
        result = subprocess.run(
            [sys.executable, test_script, self.checkpoint_dir, "False"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(self.temp_dir)  # Run from parent directory
        )
        
        if result.returncode != 0:
            print(f"Script error: {result.stderr}")
        
        self.assertIn("speaker_counter:75", result.stdout,
                     f"Without fresh flag, should load counter from saved model. Got: {result.stdout}")
        
        # Test with fresh=True (should reset counter=1)
        result = subprocess.run(
            [sys.executable, test_script, self.checkpoint_dir, "True"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(self.temp_dir)  # Run from parent directory
        )
        
        if result.returncode != 0:
            print(f"Script error: {result.stderr}")
            
        self.assertIn("speaker_counter:1", result.stdout,
                     f"With fresh flag, should reset counter to 1. Got: {result.stdout}")
    
    def test_fresh_deletes_model_before_processing(self):
        """Test that fresh flag deletes the model file before processing starts."""
        # The model file should be deleted when SpeakerIdentification is initialized
        # with fresh=True, not after processing
        
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        # Model exists before
        self.assertTrue(os.path.exists(self.model_path))
        
        # Initialize with fresh
        speaker_id = SpeakerIdentification(config)
        
        # Model should be deleted immediately
        self.assertFalse(os.path.exists(self.model_path))
        
        # Counter should be 1
        self.assertEqual(speaker_id.speaker_counter, 1)
    
    def test_multiple_datasets_with_fresh(self):
        """Test that fresh flag works correctly when processing multiple datasets."""
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # First dataset
        embeddings1 = np.array([[0.1] * 512, [0.2] * 512])
        sample_ids1 = ["dataset1_s1", "dataset1_s2"]
        
        with patch('processors.speaker_identification.hdbscan.HDBSCAN') as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = [0, 0]
            mock_hdbscan.return_value = mock_clusterer
            
            speaker_ids1 = speaker_id.cluster_embeddings(embeddings1, sample_ids1)
        
        # Should start from SPK_00001
        self.assertIn("SPK_00001", speaker_ids1)
        
        # Reset state for next dataset
        speaker_id.reset_state()
        
        # Second dataset
        embeddings2 = np.array([[0.3] * 512, [0.4] * 512])
        sample_ids2 = ["dataset2_s1", "dataset2_s2"]
        
        with patch('processors.speaker_identification.hdbscan.HDBSCAN') as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = [0, 0]
            mock_hdbscan.return_value = mock_clusterer
            
            speaker_ids2 = speaker_id.cluster_embeddings(embeddings2, sample_ids2)
        
        # Should continue from SPK_00002
        self.assertIn("SPK_00002", speaker_ids2)
        self.assertNotIn("SPK_00075", speaker_ids2)
        self.assertNotIn("SPK_00076", speaker_ids2)


if __name__ == "__main__":
    unittest.main()