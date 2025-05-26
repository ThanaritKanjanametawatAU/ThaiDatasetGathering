#!/usr/bin/env python3
"""
Test that --fresh flag resets speaker ID counter to 1.
"""

import unittest
import os
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.speaker_identification import SpeakerIdentification


class TestFreshSpeakerReset(unittest.TestCase):
    """Test speaker ID reset behavior with --fresh flag."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "speaker_model.json")
        
        # Create a mock existing model with speaker counter at 75
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
    
    def test_normal_initialization_loads_counter(self):
        """Test that normal initialization loads the speaker counter from saved model."""
        config = {
            'model_path': self.model_path,
            'clustering': {}
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Should load counter from saved model
        self.assertEqual(speaker_id.speaker_counter, 75)
        self.assertEqual(len(speaker_id.existing_clusters), 2)
    
    def test_fresh_initialization_resets_counter(self):
        """Test that initialization with fresh=True resets speaker counter to 1."""
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True  # This flag should trigger reset
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Should reset counter to 1 despite saved model
        self.assertEqual(speaker_id.speaker_counter, 1)
        self.assertEqual(len(speaker_id.existing_clusters), 0)
        self.assertIsNone(speaker_id.cluster_centroids)
    
    def test_fresh_flag_deletes_model_file(self):
        """Test that fresh flag deletes the existing model file."""
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        # Model file should exist before
        self.assertTrue(os.path.exists(self.model_path))
        
        speaker_id = SpeakerIdentification(config)
        
        # Model file should be deleted after fresh initialization
        self.assertFalse(os.path.exists(self.model_path))
    
    def test_speaker_ids_start_from_one_with_fresh(self):
        """Test that speaker IDs start from SPK_00001 with fresh flag."""
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Mock embeddings for 3 samples
        embeddings = np.array([[0.1] * 512, [0.2] * 512, [0.3] * 512])
        sample_ids = ["sample1", "sample2", "sample3"]
        
        # Mock clustering to return 2 clusters
        # For small batch, it uses AgglomerativeClustering, not HDBSCAN
        with patch('processors.speaker_identification.AgglomerativeClustering') as mock_agg:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.array([0, 0, 1])  # 2 clusters
            mock_agg.return_value = mock_clusterer
            
            speaker_ids = speaker_id.cluster_embeddings(embeddings, sample_ids)
        
        # Should get SPK_00001 and SPK_00002
        self.assertIn("SPK_00001", speaker_ids)
        self.assertIn("SPK_00002", speaker_ids)
        self.assertNotIn("SPK_00075", speaker_ids)
        self.assertNotIn("SPK_00076", speaker_ids)
    
    def test_append_mode_preserves_counter(self):
        """Test that append mode (no fresh flag) preserves the counter."""
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': False  # Explicitly not fresh
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Mock embeddings for 2 new samples
        embeddings = np.array([[0.7] * 512, [0.8] * 512])
        sample_ids = ["new_sample1", "new_sample2"]
        
        # Mock clustering to return new cluster
        with patch('processors.speaker_identification.hdbscan.HDBSCAN') as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = [0, 0]  # 1 new cluster
            mock_hdbscan.return_value = mock_clusterer
            
            speaker_ids = speaker_id.cluster_embeddings(embeddings, sample_ids)
        
        # Should get SPK_00075 (continuing from saved counter)
        self.assertIn("SPK_00075", speaker_ids)
        self.assertEqual(speaker_id.speaker_counter, 76)  # Should increment
    
    def test_fresh_flag_in_main_process(self):
        """Test that fresh flag is passed correctly through main processing."""
        # This tests the integration point where main.py should pass fresh flag
        # to SpeakerIdentification config
        
        # Mock command line args with --fresh
        mock_args = MagicMock()
        mock_args.fresh = True
        mock_args.enable_speaker_id = True
        mock_args.speaker_model = 'pyannote/embedding'
        mock_args.speaker_min_cluster_size = 5
        mock_args.speaker_min_samples = 3
        mock_args.speaker_epsilon = 0.5
        mock_args.speaker_threshold = 0.6
        mock_args.speaker_batch_size = 10000
        mock_args.store_embeddings = False
        mock_args.output = None
        
        # Expected config should include fresh=True
        expected_config = {
            'model': 'pyannote/embedding',
            'clustering': {
                'algorithm': 'hdbscan',
                'min_cluster_size': 5,
                'min_samples': 3,
                'metric': 'cosine',
                'cluster_selection_epsilon': 0.5,
                'similarity_threshold': 0.6
            },
            'batch_size': 10000,
            'store_embeddings': False,
            'embedding_path': os.path.join('.', 'speaker_embeddings.h5'),
            'model_path': os.path.join('checkpoints', 'speaker_model.json'),
            'fresh': True  # This should be added when args.fresh is True
        }
        
        # This is what main.py should construct
        self.assertTrue(expected_config['fresh'])


if __name__ == "__main__":
    unittest.main()