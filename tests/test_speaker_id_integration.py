"""
Integration test for speaker identification with real pyannote model.
Tests that similar voices get clustered together.
"""

import unittest
import numpy as np
import tempfile
import os
import json

from processors.speaker_identification import SpeakerIdentification


class TestSpeakerIdIntegration(unittest.TestCase):
    """Integration test with real pyannote model."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_speaker_model.json')
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_real_model_extracts_embeddings(self):
        """Test that the real pyannote model extracts embeddings successfully."""
        config = {
            'model': 'pyannote/embedding',
            'batch_size': 10,
            'min_cluster_size': 2,
            'min_samples': 1,
            'model_path': self.model_path
        }
        
        # Initialize with real model
        speaker_id = SpeakerIdentification(config)
        
        # Create test audio - 3 seconds of random noise
        audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        sample_rate = 16000
        
        # Extract embedding
        try:
            embedding = speaker_id.extract_embedding(audio, sample_rate)
            
            # Verify embedding properties
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (512,))
            self.assertFalse(np.all(embedding == 0))  # Not all zeros
            self.assertTrue(np.all(np.isfinite(embedding)))  # No NaN/inf
            
            print(f"✓ Successfully extracted embedding with shape {embedding.shape}")
            print(f"✓ Embedding stats: min={embedding.min():.3f}, max={embedding.max():.3f}, mean={embedding.mean():.3f}")
            
        except Exception as e:
            self.fail(f"Failed to extract embedding: {e}")
    
    def test_similar_audio_gets_clustered(self):
        """Test that similar audio samples get assigned the same speaker ID."""
        config = {
            'model': 'pyannote/embedding',
            'batch_size': 10,
            'min_cluster_size': 2,
            'min_samples': 1,
            'epsilon': 0.5,  # More lenient for test
            'model_path': self.model_path
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Create base audio pattern
        base_frequency = 440  # A4 note
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, sample_rate * duration)
        
        # Create similar audio samples (same frequency with slight variations)
        samples = []
        for i in range(6):
            # Add slight frequency variation
            freq_variation = 1 + (i % 2) * 0.01  # Two groups with slight freq difference
            audio = np.sin(2 * np.pi * base_frequency * freq_variation * t)
            
            # Add small amount of noise
            audio += np.random.randn(len(audio)) * 0.01
            
            samples.append({
                'ID': f'S{i+1}',
                'audio': {
                    'array': audio.astype(np.float32),
                    'sampling_rate': sample_rate
                }
            })
        
        # Process batch
        try:
            speaker_ids = speaker_id.process_batch(samples)
            
            # Verify results
            self.assertEqual(len(speaker_ids), 6)
            
            # Count unique speaker IDs
            unique_ids = set(speaker_ids)
            print(f"\n✓ Assigned speaker IDs: {speaker_ids}")
            print(f"✓ Number of unique speakers: {len(unique_ids)}")
            
            # We expect fewer unique IDs than samples (clustering happened)
            self.assertLess(len(unique_ids), 6, 
                          "Expected some clustering, but all samples got unique IDs")
            
            # Check saved model
            self.assertTrue(os.path.exists(self.model_path))
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
                print(f"✓ Model saved with speaker_counter: {model_data['speaker_counter']}")
                
        except Exception as e:
            self.fail(f"Failed to process batch: {e}")
    
    def test_no_crash_on_edge_cases(self):
        """Test that the system handles edge cases gracefully."""
        config = {
            'model': 'pyannote/embedding',
            'batch_size': 10,
            'min_cluster_size': 5,
            'min_samples': 3,
            'model_path': self.model_path
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Test with single sample (too few for clustering)
        samples = [{
            'ID': 'S1',
            'audio': {
                'array': np.random.randn(16000).astype(np.float32),
                'sampling_rate': 16000
            }
        }]
        
        try:
            speaker_ids = speaker_id.process_batch(samples)
            self.assertEqual(len(speaker_ids), 1)
            self.assertTrue(speaker_ids[0].startswith('SPK_'))
            print(f"✓ Single sample handled correctly: {speaker_ids[0]}")
        except Exception as e:
            self.fail(f"Failed on single sample: {e}")
        
        # Test with empty batch
        try:
            speaker_ids = speaker_id.process_batch([])
            self.assertEqual(len(speaker_ids), 0)
            print("✓ Empty batch handled correctly")
        except Exception as e:
            self.fail(f"Failed on empty batch: {e}")


if __name__ == '__main__':
    unittest.main(verbose=2)