"""
Test clustering accuracy for specific samples.

Specifically tests that S1-S8 and S10 should have the same speaker ID.
"""

import unittest
import numpy as np
import json
import os
import tempfile
from unittest.mock import MagicMock, patch
from processors.speaker_identification import SpeakerIdentification


class TestClusteringSpecificSamples(unittest.TestCase):
    """Test clustering accuracy for specific samples."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
            'clustering': {
                'algorithm': 'hdbscan',
                'min_cluster_size': 5,  # Original value
                'min_samples': 3,       # Original value
                'metric': 'cosine',
                'cluster_selection_epsilon': 0.5,  # Original value
                'similarity_threshold': 0.6        # Original value
            },
            'model_path': os.path.join(self.temp_dir, 'speaker_model.json')
        }
        
        # Also create config with current config.py parameters for testing
        self.current_config = {
            'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
            'clustering': {
                'algorithm': 'adaptive',  # Uses different algorithms based on batch size
                'min_cluster_size': 2,    # Current config.py value
                'min_samples': 1,         # Current config.py value
                'metric': 'cosine',
                'cluster_selection_epsilon': 0.3,  # Current config.py value
                'similarity_threshold': 0.7        # Current config.py value (too high!)
            },
            'model_path': os.path.join(self.temp_dir, 'speaker_model_current.json')
        }
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('processors.speaker_identification.Model')
    @patch('processors.speaker_identification.Inference')
    def test_s1_to_s8_and_s10_same_speaker(self, mock_inference, mock_model):
        """Test that S1-S8 and S10 get the same speaker ID."""
        # Create embeddings that are similar for S1-S8 and S10
        # These should be 256-dimensional embeddings based on the model
        
        # Create a base embedding for the main speaker
        base_embedding = np.random.randn(256)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        # Create slightly varied embeddings for S1-S8 and S10
        embeddings = []
        sample_ids = []
        
        for i in range(1, 9):  # S1 to S8
            # Add small random noise to create variation
            noise = np.random.randn(256) * 0.02  # Very small noise for same speaker
            varied_embedding = base_embedding + noise
            varied_embedding = varied_embedding / np.linalg.norm(varied_embedding)
            embeddings.append(varied_embedding)
            sample_ids.append(f'S{i}')
        
        # S9 should be a different speaker - create a very different embedding
        different_embedding = np.random.randn(256)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)
        # Ensure it's sufficiently different
        while np.dot(base_embedding, different_embedding) > 0.3:
            different_embedding = np.random.randn(256)
            different_embedding = different_embedding / np.linalg.norm(different_embedding)
        embeddings.append(different_embedding)
        sample_ids.append('S9')
        
        # S10 should be same as S1-S8
        noise = np.random.randn(256) * 0.02  # Very small noise for same speaker
        varied_embedding = base_embedding + noise
        varied_embedding = varied_embedding / np.linalg.norm(varied_embedding)
        embeddings.append(varied_embedding)
        sample_ids.append('S10')
        
        # Mock the model and inference
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        
        # Create speaker identification instance
        speaker_id = SpeakerIdentification(self.config)
        
        # Test clustering
        speaker_ids = speaker_id.cluster_embeddings(np.array(embeddings), sample_ids)
        
        # Verify S1-S8 and S10 have the same speaker ID
        s1_to_s8_ids = [speaker_ids[i] for i in range(8)]  # S1 to S8
        s10_id = speaker_ids[9]  # S10
        
        # All should be the same
        self.assertEqual(len(set(s1_to_s8_ids + [s10_id])), 1,
                        f"S1-S8 and S10 should have the same speaker ID. Got: S1-S8={s1_to_s8_ids}, S10={s10_id}")
        
        # S9 should be different
        s9_id = speaker_ids[8]
        self.assertNotEqual(s9_id, s1_to_s8_ids[0],
                           f"S9 should have a different speaker ID. Got S9={s9_id}, others={s1_to_s8_ids[0]}")
        
    @patch('processors.speaker_identification.Model')
    @patch('processors.speaker_identification.Inference')
    def test_cosine_similarity_threshold(self, mock_inference, mock_model):
        """Test that cosine similarity threshold is properly respected."""
        # Create embeddings with specific cosine similarities
        base_embedding = np.random.randn(256)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        embeddings = []
        sample_ids = []
        
        # Create 5 similar embeddings (should cluster together)
        for i in range(5):
            # Create embedding with cosine similarity > 0.7 to base
            noise = np.random.randn(256) * 0.1
            similar_embedding = base_embedding * 0.8 + noise * 0.2
            similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
            
            # Verify similarity
            similarity = np.dot(base_embedding, similar_embedding)
            self.assertGreater(similarity, 0.7, f"Similar embedding {i} should have > 0.7 similarity")
            
            embeddings.append(similar_embedding)
            sample_ids.append(f'Similar_{i}')
        
        # Create 5 different embeddings (should be in different clusters)
        for i in range(5):
            # Create embedding with cosine similarity < 0.3 to base
            different_embedding = np.random.randn(256)
            different_embedding = different_embedding / np.linalg.norm(different_embedding)
            
            # Ensure low similarity
            while np.dot(base_embedding, different_embedding) > 0.3:
                different_embedding = np.random.randn(256)
                different_embedding = different_embedding / np.linalg.norm(different_embedding)
            
            embeddings.append(different_embedding)
            sample_ids.append(f'Different_{i}')
        
        # Mock the model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        
        # Create speaker identification instance
        speaker_id = SpeakerIdentification(self.config)
        
        # Test clustering
        speaker_ids = speaker_id.cluster_embeddings(np.array(embeddings), sample_ids)
        
        # Similar samples should have the same speaker ID
        similar_ids = [speaker_ids[i] for i in range(5)]
        self.assertEqual(len(set(similar_ids)), 1,
                        f"Similar samples should have the same speaker ID. Got: {similar_ids}")
        
        # Different samples should have different speaker IDs (or at least not all the same as similar)
        different_ids = [speaker_ids[i] for i in range(5, 10)]
        self.assertNotIn(similar_ids[0], different_ids,
                        f"Different samples should not have the same speaker ID as similar samples")
    
    @patch('processors.speaker_identification.Model')
    @patch('processors.speaker_identification.Inference')
    def test_s1_to_s8_and_s10_same_speaker_with_current_config(self, mock_inference, mock_model):
        """Test that S1-S8 and S10 get the same speaker ID with current config.py parameters."""
        # Same test as above but with current config.py parameters
        
        # Create a base embedding for the main speaker
        base_embedding = np.random.randn(256)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        # Create slightly varied embeddings for S1-S8 and S10
        embeddings = []
        sample_ids = []
        
        for i in range(1, 9):  # S1 to S8
            # Add small random noise to create variation
            noise = np.random.randn(256) * 0.02  # Very small noise for same speaker
            varied_embedding = base_embedding + noise
            varied_embedding = varied_embedding / np.linalg.norm(varied_embedding)
            embeddings.append(varied_embedding)
            sample_ids.append(f'S{i}')
        
        # S9 should be a different speaker - create a very different embedding
        different_embedding = np.random.randn(256)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)
        # Ensure it's sufficiently different
        while np.dot(base_embedding, different_embedding) > 0.3:
            different_embedding = np.random.randn(256)
            different_embedding = different_embedding / np.linalg.norm(different_embedding)
        embeddings.append(different_embedding)
        sample_ids.append('S9')
        
        # S10 should be same as S1-S8
        noise = np.random.randn(256) * 0.02  # Very small noise for same speaker
        varied_embedding = base_embedding + noise
        varied_embedding = varied_embedding / np.linalg.norm(varied_embedding)
        embeddings.append(varied_embedding)
        sample_ids.append('S10')
        
        # Mock the model and inference
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        
        # Create speaker identification instance with CURRENT config
        speaker_id = SpeakerIdentification(self.current_config)
        
        # Test clustering
        speaker_ids = speaker_id.cluster_embeddings(np.array(embeddings), sample_ids)
        
        # Verify S1-S8 and S10 have the same speaker ID
        s1_to_s8_ids = [speaker_ids[i] for i in range(8)]  # S1 to S8
        s10_id = speaker_ids[9]  # S10
        
        # All should be the same
        self.assertEqual(len(set(s1_to_s8_ids + [s10_id])), 1,
                        f"S1-S8 and S10 should have the same speaker ID. Got: S1-S8={s1_to_s8_ids}, S10={s10_id}")
        
        # S9 should be different
        s9_id = speaker_ids[8]
        self.assertNotEqual(s9_id, s1_to_s8_ids[0],
                           f"S9 should have a different speaker ID. Got S9={s9_id}, others={s1_to_s8_ids[0]}")


if __name__ == '__main__':
    unittest.main()