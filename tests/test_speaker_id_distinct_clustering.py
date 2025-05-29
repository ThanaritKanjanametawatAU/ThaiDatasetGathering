#!/usr/bin/env python3
"""
Tests for speaker ID clustering to ensure distinct speakers get different IDs.
"""

import unittest
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.speaker_identification import SpeakerIdentification
from config import SPEAKER_ID_CONFIG
import soundfile as sf
import tempfile
import shutil


class TestSpeakerIDDistinctClustering(unittest.TestCase):
    """Test that speaker identification creates distinct IDs for different speakers."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Configure speaker ID with test settings
        self.config = SPEAKER_ID_CONFIG.copy()
        self.config['model_path'] = os.path.join(self.test_dir, 'test_speaker_model.json')
        self.config['fresh'] = True  # Always start fresh for tests
        
        # Adjust clustering parameters for better separation
        self.config['clustering'] = {
            'algorithm': 'adaptive',
            'min_cluster_size': 2,
            'min_samples': 1,
            'metric': 'cosine',
            'cluster_selection_epsilon': 0.3,
            'similarity_threshold': 0.6
        }
        
        self.speaker_id = SpeakerIdentification(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'speaker_id'):
            self.speaker_id.cleanup()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_distinct_audio_samples(self, num_speakers=3, samples_per_speaker=2):
        """Create synthetic audio samples with distinct characteristics."""
        audio_samples = []
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        
        # Create more distinct frequency patterns for each speaker
        # Simulate different formant frequencies for different speakers
        speaker_profiles = [
            # Speaker 1: Low voice (male-like)
            {'f0': 120, 'f1': 700, 'f2': 1220, 'f3': 2600},
            # Speaker 2: Medium voice
            {'f0': 200, 'f1': 850, 'f2': 2100, 'f3': 3000},
            # Speaker 3: High voice (female-like)
            {'f0': 250, 'f1': 920, 'f2': 2400, 'f3': 3300}
        ]
        
        for speaker_idx in range(num_speakers):
            profile = speaker_profiles[speaker_idx % len(speaker_profiles)]
            
            for sample_idx in range(samples_per_speaker):
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                # Create more realistic audio with formants
                # Fundamental frequency
                audio = 0.3 * np.sin(2 * np.pi * profile['f0'] * t)
                
                # Add formants (resonant frequencies)
                audio += 0.25 * np.sin(2 * np.pi * profile['f1'] * t)
                audio += 0.2 * np.sin(2 * np.pi * profile['f2'] * t)
                audio += 0.15 * np.sin(2 * np.pi * profile['f3'] * t)
                
                # Add some harmonics for richness
                audio += 0.1 * np.sin(2 * np.pi * profile['f0'] * 2 * t)
                audio += 0.05 * np.sin(2 * np.pi * profile['f0'] * 3 * t)
                
                # Add amplitude modulation to simulate speech rhythm
                mod_freq = 4 + sample_idx * 0.5  # Slight variation between samples
                amplitude_mod = 0.8 + 0.2 * np.sin(2 * np.pi * mod_freq * t)
                audio *= amplitude_mod
                
                # Add slight pitch variation (vibrato)
                vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)
                audio_vibrato = np.sin(2 * np.pi * profile['f0'] * (1 + vibrato) * t)
                audio = 0.7 * audio + 0.3 * audio_vibrato
                
                # Add some colored noise to make it more realistic
                noise = np.random.randn(len(audio))
                # Low-pass filter the noise
                from scipy.signal import butter, filtfilt
                b, a = butter(4, 3000 / (sample_rate / 2), 'low')
                filtered_noise = filtfilt(b, a, noise)
                audio += 0.02 * filtered_noise
                
                # Normalize
                audio = audio / np.max(np.abs(audio))
                
                # Create sample dict
                sample_id = f"S{speaker_idx * samples_per_speaker + sample_idx + 1}"
                audio_sample = {
                    'ID': sample_id,
                    'audio': {
                        'array': audio,
                        'sampling_rate': sample_rate
                    },
                    'expected_speaker': speaker_idx  # For testing
                }
                audio_samples.append(audio_sample)
        
        return audio_samples
    
    def test_clustering_with_known_embeddings(self):
        """Test clustering with manually created distinct embeddings."""
        # Create distinct embeddings manually
        # Each row is an embedding vector (256 dimensions)
        
        # Create 3 distinct clusters of embeddings
        embedding_dim = 256
        embeddings = []
        expected_labels = []
        
        # Cluster 1: centered around [1, 0, 0, ...]
        for _ in range(2):
            emb = np.zeros(embedding_dim)
            emb[0] = 1.0 + np.random.normal(0, 0.05)
            emb += np.random.normal(0, 0.01, embedding_dim)
            embeddings.append(emb)
            expected_labels.append(0)
        
        # Cluster 2: centered around [0, 1, 0, ...]
        for _ in range(2):
            emb = np.zeros(embedding_dim)
            emb[1] = 1.0 + np.random.normal(0, 0.05)
            emb += np.random.normal(0, 0.01, embedding_dim)
            embeddings.append(emb)
            expected_labels.append(1)
            
        # Cluster 3: centered around [0, 0, 1, ...]
        for _ in range(2):
            emb = np.zeros(embedding_dim)
            emb[2] = 1.0 + np.random.normal(0, 0.05)
            emb += np.random.normal(0, 0.01, embedding_dim)
            embeddings.append(emb)
            expected_labels.append(2)
        
        embeddings = np.array(embeddings)
        sample_ids = [f"S{i+1}" for i in range(6)]
        
        # Test clustering directly
        speaker_ids = self.speaker_id.cluster_embeddings(embeddings, sample_ids)
        
        # Check that we get 3 distinct clusters
        unique_ids = set(speaker_ids)
        self.assertEqual(len(unique_ids), 3,
                        f"Should detect 3 distinct clusters, got {len(unique_ids)}: {unique_ids}")
        
        # Check that samples from same expected cluster get same ID
        for i in range(0, 6, 2):
            self.assertEqual(speaker_ids[i], speaker_ids[i+1],
                           f"Samples {i} and {i+1} should have same speaker ID")
    
    def test_distinct_speakers_get_different_ids(self):
        """Test that distinctly different speakers get different IDs."""
        # Create 3 distinct speakers with 2 samples each
        audio_samples = self.create_distinct_audio_samples(
            num_speakers=3, 
            samples_per_speaker=2
        )
        
        # Process all samples
        speaker_ids = self.speaker_id.process_batch(audio_samples)
        
        # Verify we have the right number of results
        self.assertEqual(len(speaker_ids), 6, "Should have 6 speaker IDs")
        
        # Group by expected speaker
        speaker_groups = {}
        for i, sample in enumerate(audio_samples):
            expected_speaker = sample['expected_speaker']
            if expected_speaker not in speaker_groups:
                speaker_groups[expected_speaker] = []
            speaker_groups[expected_speaker].append(speaker_ids[i])
        
        # Check that each speaker group has consistent IDs
        for expected_speaker, ids in speaker_groups.items():
            unique_ids = set(ids)
            self.assertEqual(len(unique_ids), 1, 
                           f"Speaker {expected_speaker} should have consistent ID, got {unique_ids}")
        
        # Check that different speakers have different IDs
        all_unique_ids = [list(set(ids))[0] for ids in speaker_groups.values()]
        self.assertEqual(len(set(all_unique_ids)), 3,
                        f"Should have 3 distinct speaker IDs, got {set(all_unique_ids)}")
        
        # Verify ID format
        for speaker_id in speaker_ids:
            self.assertRegex(speaker_id, r'^SPK_\d{5}$',
                           f"Speaker ID {speaker_id} should match format SPK_XXXXX")
    
    def test_clustering_parameters_affect_separation(self):
        """Test that clustering parameters properly affect speaker separation."""
        # Create samples with varying similarity
        audio_samples = []
        sample_rate = 16000
        
        # Create two very similar speakers (should be clustered together with high threshold)
        for i in range(2):
            t = np.linspace(0, 2.0, int(sample_rate * 2.0))
            audio = np.sin(2 * np.pi * 300 * t)
            audio += 0.5 * np.sin(2 * np.pi * 600 * t)
            audio += 0.01 * np.random.randn(len(audio))  # Very little variation
            audio = audio / np.max(np.abs(audio))
            
            audio_samples.append({
                'ID': f'S{i+1}',
                'audio': {'array': audio, 'sampling_rate': sample_rate},
                'group': 'similar'
            })
        
        # Create two very different speakers
        for i in range(2):
            t = np.linspace(0, 2.0, int(sample_rate * 2.0))
            freq = 500 + i * 300  # Very different frequencies
            audio = np.sin(2 * np.pi * freq * t)
            audio = audio / np.max(np.abs(audio))
            
            audio_samples.append({
                'ID': f'S{i+3}',
                'audio': {'array': audio, 'sampling_rate': sample_rate},
                'group': f'different_{i}'
            })
        
        # Test with high similarity threshold (lenient clustering)
        self.speaker_id.config['clustering']['similarity_threshold'] = 0.9
        speaker_ids_lenient = self.speaker_id.process_batch(audio_samples)
        
        # Reset for next test
        self.speaker_id.reset_for_new_dataset()
        
        # Test with low similarity threshold (strict clustering)
        self.speaker_id.config['clustering']['similarity_threshold'] = 0.3
        speaker_ids_strict = self.speaker_id.process_batch(audio_samples)
        
        # With lenient threshold, similar speakers might be grouped
        unique_lenient = len(set(speaker_ids_lenient))
        # With strict threshold, all should be separate
        unique_strict = len(set(speaker_ids_strict))
        
        self.assertLessEqual(unique_lenient, unique_strict,
                            "Lenient clustering should create same or fewer clusters")
        self.assertGreaterEqual(unique_strict, 3,
                               "Strict clustering should separate most speakers")
    
    def test_small_batch_clustering(self):
        """Test clustering with small batches (uses AgglomerativeClustering)."""
        # Create small batch of 10 samples (5 speakers, 2 samples each)
        audio_samples = self.create_distinct_audio_samples(
            num_speakers=5,
            samples_per_speaker=2
        )
        
        # Process batch
        speaker_ids = self.speaker_id.process_batch(audio_samples)
        
        # Should use AgglomerativeClustering for small batch
        unique_ids = set(speaker_ids)
        self.assertGreaterEqual(len(unique_ids), 3,
                               f"Should detect at least 3 distinct speakers in small batch, got {len(unique_ids)}")
        self.assertLessEqual(len(unique_ids), 5,
                            "Should not over-segment speakers")
        
        # Check consistency within speaker
        for i in range(5):
            id1 = speaker_ids[i * 2]
            id2 = speaker_ids[i * 2 + 1]
            self.assertEqual(id1, id2,
                           f"Same speaker samples should have same ID: {id1} vs {id2}")
    
    def test_large_batch_clustering(self):
        """Test clustering with large batches (uses HDBSCAN)."""
        # Create large batch of 100 samples (10 speakers, 10 samples each)
        audio_samples = self.create_distinct_audio_samples(
            num_speakers=10,
            samples_per_speaker=10
        )
        
        # Process batch
        speaker_ids = self.speaker_id.process_batch(audio_samples)
        
        # Should detect multiple distinct speakers
        unique_ids = set(speaker_ids)
        self.assertGreaterEqual(len(unique_ids), 7,
                               f"Should detect at least 7 distinct speakers in large batch, got {len(unique_ids)}")
        
        # Log clustering results
        print(f"\nLarge batch clustering results:")
        print(f"  Total samples: {len(audio_samples)}")
        print(f"  Unique speaker IDs: {len(unique_ids)}")
        print(f"  Speaker distribution: {dict((id, speaker_ids.count(id)) for id in unique_ids)}")
    
    def test_noise_point_handling(self):
        """Test handling of noise points (outliers) in clustering."""
        audio_samples = []
        sample_rate = 16000
        
        # Create regular speakers
        regular_samples = self.create_distinct_audio_samples(
            num_speakers=3,
            samples_per_speaker=3
        )
        audio_samples.extend(regular_samples)
        
        # Add some outlier samples (very different from others)
        for i in range(3):
            t = np.linspace(0, 2.0, int(sample_rate * 2.0))
            # Random frequencies and noise
            audio = np.random.randn(len(t)) * 0.1
            audio += 0.5 * np.sin(2 * np.pi * np.random.randint(1000, 2000) * t)
            audio = audio / np.max(np.abs(audio))
            
            audio_samples.append({
                'ID': f'OUTLIER_{i+1}',
                'audio': {'array': audio, 'sampling_rate': sample_rate}
            })
        
        # Process all samples
        speaker_ids = self.speaker_id.process_batch(audio_samples)
        
        # Check that outliers get assigned IDs (not rejected)
        for i, sample in enumerate(audio_samples):
            if sample['ID'].startswith('OUTLIER'):
                self.assertNotEqual(speaker_ids[i], 'UNKNOWN',
                                  "Outliers should still get speaker IDs")
                self.assertRegex(speaker_ids[i], r'^SPK_\d{5}$',
                               "Outlier should get valid speaker ID format")
    
    def test_incremental_speaker_assignment(self):
        """Test that speaker counter increments properly."""
        # First batch
        batch1 = self.create_distinct_audio_samples(num_speakers=2, samples_per_speaker=1)
        ids1 = self.speaker_id.process_batch(batch1)
        
        # Second batch with new speakers
        batch2 = self.create_distinct_audio_samples(num_speakers=2, samples_per_speaker=1)
        ids2 = self.speaker_id.process_batch(batch2)
        
        # All IDs should be unique if speakers are different
        all_ids = ids1 + ids2
        unique_ids = set(all_ids)
        
        self.assertEqual(len(unique_ids), 4,
                        f"Should have 4 unique speaker IDs, got {unique_ids}")
        
        # Check that IDs increment
        id_numbers = []
        for speaker_id in unique_ids:
            # Extract number from SPK_XXXXX
            num = int(speaker_id.split('_')[1])
            id_numbers.append(num)
        
        id_numbers.sort()
        # Should be consecutive
        for i in range(len(id_numbers) - 1):
            self.assertEqual(id_numbers[i+1], id_numbers[i] + 1,
                           "Speaker IDs should increment consecutively")
    
    def test_dataset_separation(self):
        """Test that reset_for_new_dataset prevents cross-dataset contamination."""
        # Process first dataset
        dataset1 = self.create_distinct_audio_samples(num_speakers=3, samples_per_speaker=2)
        ids1 = self.speaker_id.process_batch(dataset1)
        
        # Reset for new dataset (but keep counter)
        self.speaker_id.reset_for_new_dataset(reset_counter=False)
        
        # Process second dataset with similar audio
        dataset2 = self.create_distinct_audio_samples(num_speakers=3, samples_per_speaker=2)
        ids2 = self.speaker_id.process_batch(dataset2)
        
        # IDs should not overlap
        set1 = set(ids1)
        set2 = set(ids2)
        overlap = set1.intersection(set2)
        
        self.assertEqual(len(overlap), 0,
                        f"Dataset IDs should not overlap, but found: {overlap}")
        
        # All IDs should still be unique
        all_unique = set1.union(set2)
        self.assertEqual(len(all_unique), len(set1) + len(set2),
                        "All speaker IDs should be globally unique")


if __name__ == "__main__":
    unittest.main(verbosity=2)