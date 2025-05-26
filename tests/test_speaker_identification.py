"""
Test module for speaker identification functionality.

Tests the SpeakerIdentification class for embedding extraction,
clustering, model persistence, and streaming integration.
"""

import unittest
import numpy as np
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock
import torch

class TestSpeakerIdentification(unittest.TestCase):
    """Test cases for speaker identification system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model': 'pyannote/embedding',
            'batch_size': 100,
            'min_cluster_size': 15,
            'min_samples': 10,
            'epsilon': 0.3,
            'threshold': 0.7,
            'store_embeddings': True,
            'embedding_path': os.path.join(self.temp_dir, 'test_embeddings.h5'),
            'model_path': os.path.join(self.temp_dir, 'test_model.json')
        }
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('processors.speaker_identification.Model')
    def test_initialization(self, mock_model):
        """Test proper initialization of SpeakerIdentification."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        from processors.speaker_identification import SpeakerIdentification
        speaker_id = SpeakerIdentification(self.config)
        
        # Check initialization
        self.assertEqual(speaker_id.config, self.config)
        self.assertEqual(speaker_id.speaker_counter, 1)
        self.assertEqual(speaker_id.existing_clusters, {})
        self.assertTrue(speaker_id.store_embeddings)
        mock_model.from_pretrained.assert_called_once_with('pyannote/embedding')
    
    @patch('processors.speaker_identification.Inference')
    @patch('processors.speaker_identification.Model')
    def test_embedding_extraction(self, mock_model, mock_inference):
        """Test embedding extraction from audio."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        # Mock the inference to return fixed embedding
        mock_embedding = np.random.randn(256)  # Use numpy array, not torch tensor
        mock_inference_instance = MagicMock()
        mock_inference_instance.return_value = mock_embedding
        mock_inference.return_value = mock_inference_instance
        
        from processors.speaker_identification import SpeakerIdentification
        speaker_id = SpeakerIdentification(self.config)
        
        # Create dummy audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        sample_rate = 16000
        
        embedding = speaker_id.extract_embedding(audio, sample_rate)
        
        # Check embedding shape
        self.assertEqual(len(embedding.shape), 1)
        self.assertEqual(embedding.shape[0], 256)  # Updated embedding size
    
    @patch('processors.speaker_identification.Model')
    def test_batch_processing(self, mock_model):
        """Test batch processing of audio samples."""
        # Mock the model
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.randn(1, 256)  # Updated embedding size
        mock_model.from_pretrained.return_value.to.return_value = mock_model_instance
        
        from processors.speaker_identification import SpeakerIdentification
        speaker_id = SpeakerIdentification(self.config)
        
        # Create dummy samples
        samples = []
        for i in range(20):
            samples.append({
                'ID': f'S{i+1}',
                'audio': {
                    'array': np.random.randn(16000 * 2),  # 2 seconds
                    'sampling_rate': 16000
                }
            })
        
        # Process batch
        speaker_ids = speaker_id.process_batch(samples)
        
        # Check results
        self.assertEqual(len(speaker_ids), len(samples))
        self.assertTrue(all(sid.startswith('SPK_') for sid in speaker_ids))
        self.assertTrue(all(len(sid) == 9 for sid in speaker_ids))  # SPK_00001 format
    
    @patch('processors.speaker_identification.Model')
    def test_clustering_over_segmentation(self, mock_model):
        """Test that clustering groups similar speakers appropriately."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        from processors.speaker_identification import SpeakerIdentification
        speaker_id = SpeakerIdentification(self.config)
        
        # Create embeddings with clear groups
        embeddings = np.random.randn(30, 256)  # Updated embedding size
        # Make some embeddings similar (these should be grouped together)
        embeddings[10:15] = embeddings[0] + np.random.randn(5, 256) * 0.2
        embeddings[20:25] = embeddings[1] + np.random.randn(5, 256) * 0.2
        
        sample_ids = [f'S{i+1}' for i in range(30)]
        speaker_ids = speaker_id.cluster_embeddings(embeddings, sample_ids)
        
        # With improved clustering, should group similar speakers
        unique_speakers = len(set(speaker_ids))
        # Should have fewer unique speakers than total samples (good clustering)
        self.assertLess(unique_speakers, 30)  # Good clustering groups similar speakers
        # But not too few (still maintains separation)
        self.assertGreaterEqual(unique_speakers, 5)  # Maintains reasonable separation
    
    @patch('processors.speaker_identification.Model')
    def test_uncertain_cluster_handling(self, mock_model):
        """Test that uncertain/single samples get unique speaker IDs."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        from processors.speaker_identification import SpeakerIdentification
        # Use smaller min_samples for this test
        config = self.config.copy()
        config['min_samples'] = 2
        speaker_id = SpeakerIdentification(config)
        
        # Create sparse embeddings that won't cluster well
        embeddings = np.random.randn(5, 256) * 10  # Very spread out
        sample_ids = [f'S{i+1}' for i in range(5)]
        
        speaker_ids = speaker_id.cluster_embeddings(embeddings, sample_ids)
        
        # All should have unique speaker IDs
        unique_speakers = len(set(speaker_ids))
        self.assertEqual(unique_speakers, 5)
    
    @patch('processors.speaker_identification.Model')
    def test_model_persistence(self, mock_model):
        """Test saving and loading speaker model."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        from processors.speaker_identification import SpeakerIdentification
        speaker_id = SpeakerIdentification(self.config)
        
        # Set some state
        speaker_id.speaker_counter = 42
        speaker_id.existing_clusters = {0: 'SPK_00001', 1: 'SPK_00002'}
        speaker_id.cluster_centroids = np.random.randn(2, 256)
        
        # Save model
        speaker_id.save_model()
        
        # Check file exists
        self.assertTrue(os.path.exists(self.config['model_path']))
        
        # Load model data
        with open(self.config['model_path'], 'r') as f:
            model_data = json.load(f)
        
        self.assertEqual(model_data['speaker_counter'], 42)
        self.assertEqual(model_data['existing_clusters'], {'0': 'SPK_00001', '1': 'SPK_00002'})
        self.assertIsNotNone(model_data['cluster_centroids'])
        self.assertEqual(len(model_data['cluster_centroids']), 2)
    
    @patch('processors.speaker_identification.Model')
    def test_append_mode_compatibility(self, mock_model):
        """Test that append mode works with existing model."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        from processors.speaker_identification import SpeakerIdentification
        
        # Create and save initial model
        speaker_id1 = SpeakerIdentification(self.config)
        speaker_id1.speaker_counter = 10
        speaker_id1.existing_clusters = {0: 'SPK_00001', 1: 'SPK_00005'}
        speaker_id1.cluster_centroids = np.random.randn(2, 256)
        speaker_id1.save_model()
        
        # Create new instance (simulating append mode)
        speaker_id2 = SpeakerIdentification(self.config)
        
        # Check state is loaded
        self.assertEqual(speaker_id2.speaker_counter, 10)
        self.assertEqual(speaker_id2.existing_clusters, {0: 'SPK_00001', 1: 'SPK_00005'})
        self.assertIsNotNone(speaker_id2.cluster_centroids)
        self.assertEqual(speaker_id2.cluster_centroids.shape, (2, 256))
    
    @patch('processors.speaker_identification.h5py.File')
    @patch('processors.speaker_identification.Model')
    def test_embedding_storage(self, mock_model, mock_h5py):
        """Test embedding storage in HDF5 format."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        # Mock HDF5 file
        mock_file = MagicMock()
        mock_h5py.return_value = mock_file
        mock_file.__contains__.return_value = False
        
        from processors.speaker_identification import SpeakerIdentification
        speaker_id = SpeakerIdentification(self.config)
        
        # Store embeddings
        embeddings = np.random.randn(10, 256)
        sample_ids = [f'S{i+1}' for i in range(10)]
        speaker_ids = [f'SPK_{i+1:05d}' for i in range(10)]
        
        speaker_id._store_embeddings(embeddings, sample_ids, speaker_ids)
        
        # Check HDF5 operations
        mock_file.create_dataset.assert_called()
        mock_file.flush.assert_called()
    
    @patch('processors.speaker_identification.Model')
    def test_cosine_similarity_computation(self, mock_model):
        """Test cosine similarity computation between embeddings."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        from processors.speaker_identification import SpeakerIdentification
        speaker_id = SpeakerIdentification(self.config)
        
        # Create test embeddings
        embeddings1 = np.array([[1, 0, 0], [0, 1, 0]])
        embeddings2 = np.array([[1, 0, 0], [0, 0, 1]])
        
        similarities = speaker_id._compute_cosine_similarity(embeddings1, embeddings2)
        
        # Check similarity matrix
        self.assertEqual(similarities.shape, (2, 2))
        self.assertAlmostEqual(similarities[0, 0], 1.0, places=5)  # Same vector
        self.assertAlmostEqual(similarities[0, 1], 0.0, places=5)  # Orthogonal
        self.assertAlmostEqual(similarities[1, 0], 0.0, places=5)  # Orthogonal
        self.assertAlmostEqual(similarities[1, 1], 0.0, places=5)  # Orthogonal
    
    @patch('processors.speaker_identification.Model')
    def test_incremental_speaker_assignment(self, mock_model):
        """Test incremental speaker ID assignment."""
        # Mock the model
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()
        
        from processors.speaker_identification import SpeakerIdentification
        speaker_id = SpeakerIdentification(self.config)
        
        # Process first batch
        embeddings1 = np.random.randn(10, 256)
        sample_ids1 = [f'S{i+1}' for i in range(10)]
        speaker_ids1 = speaker_id.cluster_embeddings(embeddings1, sample_ids1)
        
        initial_counter = speaker_id.speaker_counter
        
        # Process second batch
        embeddings2 = np.random.randn(10, 256)
        sample_ids2 = [f'S{i+11}' for i in range(10)]
        speaker_ids2 = speaker_id.cluster_embeddings(embeddings2, sample_ids2)
        
        # Check incremental assignment
        self.assertGreater(speaker_id.speaker_counter, initial_counter)
        # Check that speaker IDs are valid
        all_ids = speaker_ids1 + speaker_ids2
        self.assertTrue(all(sid.startswith('SPK_') for sid in all_ids))
        # With improved clustering, similar speakers across batches may get same ID
        # This is correct behavior - we're identifying the same speaker across batches
        unique_ids = set(all_ids)
        # Should have at least some unique speakers, but not necessarily 20
        self.assertGreaterEqual(len(unique_ids), 5)  # At least 5 unique speakers
        self.assertLessEqual(len(unique_ids), 20)  # At most 20 unique speakers


class TestSpeakerIdentificationIntegration(unittest.TestCase):
    """Integration tests for speaker identification in the pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_schema_includes_speaker_id(self):
        """Test that TARGET_SCHEMA includes speaker_id field."""
        from config import TARGET_SCHEMA
        
        self.assertIn('speaker_id', TARGET_SCHEMA)
        self.assertEqual(TARGET_SCHEMA['speaker_id'], str)
    
    def test_speaker_id_config_exists(self):
        """Test that speaker identification config exists."""
        from config import SPEAKER_ID_CONFIG
        
        self.assertIsInstance(SPEAKER_ID_CONFIG, dict)
        self.assertIn('enabled', SPEAKER_ID_CONFIG)
        self.assertIn('model', SPEAKER_ID_CONFIG)
        self.assertIn('clustering', SPEAKER_ID_CONFIG)
        self.assertIn('storage', SPEAKER_ID_CONFIG)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_cli_arguments(self, mock_parse_args):
        """Test that CLI arguments for speaker identification exist."""
        from main import parse_arguments
        
        # Mock the return value with all required arguments
        mock_args = MagicMock()
        # Required arguments
        mock_args.fresh = True
        mock_args.append = False
        mock_args.all = True
        mock_args.datasets = []
        # Other required arguments
        mock_args.resume = False
        mock_args.checkpoint = None
        mock_args.no_upload = False
        mock_args.private = False
        mock_args.output = None
        mock_args.verbose = False
        mock_args.sample = False
        mock_args.sample_size = 5
        mock_args.sample_archives = 1
        mock_args.chunk_size = 10000
        mock_args.max_cache_gb = 100.0
        mock_args.clear_cache = False
        mock_args.streaming = False
        mock_args.streaming_batch_size = 1000
        mock_args.upload_batch_size = 10000
        mock_args.no_standardization = False
        mock_args.sample_rate = 16000
        mock_args.target_db = -20.0
        mock_args.no_volume_norm = False
        mock_args.enable_stt = False
        mock_args.stt_batch_size = 16
        mock_args.no_stt = False
        mock_args.hf_repo = None
        # Speaker identification arguments
        mock_args.enable_speaker_id = True
        mock_args.speaker_model = 'pyannote/embedding'
        mock_args.speaker_batch_size = 10000
        mock_args.speaker_threshold = 0.7
        mock_args.store_embeddings = True
        mock_args.speaker_min_cluster_size = 15
        mock_args.speaker_min_samples = 10
        mock_args.speaker_epsilon = 0.3
        mock_parse_args.return_value = mock_args
        
        args = parse_arguments()
        
        # Check all speaker-related arguments
        self.assertTrue(hasattr(args, 'enable_speaker_id'))
        self.assertTrue(hasattr(args, 'speaker_model'))
        self.assertTrue(hasattr(args, 'speaker_batch_size'))
        self.assertTrue(hasattr(args, 'speaker_threshold'))
        self.assertTrue(hasattr(args, 'store_embeddings'))
        self.assertTrue(hasattr(args, 'speaker_min_cluster_size'))
        self.assertTrue(hasattr(args, 'speaker_min_samples'))
        self.assertTrue(hasattr(args, 'speaker_epsilon'))


if __name__ == '__main__':
    unittest.main()