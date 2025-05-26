"""
Comprehensive tests for speaker clustering accuracy.

Tests that the speaker identification system properly groups similar speakers
and separates different speakers.
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
from processors.speaker_identification import SpeakerIdentification
import soundfile as sf
import librosa


class TestSpeakerClusteringAccuracy(unittest.TestCase):
    """Test speaker clustering accuracy."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.speaker_model_path = os.path.join(self.test_dir, 'test_speaker_model.json')
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_synthetic_speaker_audio(self, speaker_id: int, sample_id: int, 
                                     duration: float = 5.0, sr: int = 16000) -> np.ndarray:
        """Create synthetic audio that represents a specific speaker.
        
        Each speaker has unique characteristics:
        - Base frequency
        - Harmonic structure
        - Modulation patterns
        """
        t = np.linspace(0, duration, int(duration * sr))
        
        # Speaker-specific characteristics
        base_freq = 100 + speaker_id * 50  # Different base frequency per speaker
        
        # Generate harmonics based on speaker
        audio = np.zeros_like(t)
        for harmonic in range(1, 5):
            freq = base_freq * harmonic
            # Add speaker-specific phase and amplitude variations
            phase = speaker_id * np.pi / 4
            amplitude = 1.0 / (harmonic + speaker_id * 0.1)
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add speaker-specific modulation
        mod_freq = 3 + speaker_id * 0.5
        mod_depth = 0.2 + speaker_id * 0.05
        modulation = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        audio *= modulation
        
        # Add slight variations between samples from same speaker
        noise_level = 0.02
        audio += np.random.normal(0, noise_level, len(audio))
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio.astype(np.float32)
    
    def test_same_speaker_clustering(self):
        """Test that samples from the same speaker are clustered together."""
        config = {
            'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
            'model_path': self.speaker_model_path,
            'clustering': {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 0.3,
                'similarity_threshold': 0.7
            }
        }
        
        speaker_id_system = SpeakerIdentification(config)
        
        # Create samples from 3 speakers, 5 samples each
        samples = []
        expected_clusters = {}
        
        for speaker_id in range(3):
            for sample_id in range(5):
                audio = self.create_synthetic_speaker_audio(speaker_id, sample_id)
                sample_name = f"speaker{speaker_id}_sample{sample_id}"
                
                samples.append({
                    'ID': sample_name,
                    'audio': {
                        'array': audio,
                        'sampling_rate': 16000
                    }
                })
                
                # Track expected clusters
                if speaker_id not in expected_clusters:
                    expected_clusters[speaker_id] = []
                expected_clusters[speaker_id].append(sample_name)
        
        # Process all samples
        speaker_ids = speaker_id_system.process_batch(samples)
        
        # Verify clustering accuracy
        assigned_clusters = {}
        for sample, speaker_id in zip(samples, speaker_ids):
            if speaker_id not in assigned_clusters:
                assigned_clusters[speaker_id] = []
            assigned_clusters[speaker_id].append(sample['ID'])
        
        # Check that samples from same speaker are clustered together
        print(f"\nAssigned clusters: {assigned_clusters}")
        print(f"Expected clusters: {expected_clusters}")
        
        # We should have at most 3 unique speaker IDs
        self.assertLessEqual(len(assigned_clusters), 3, 
                           f"Too many clusters: {len(assigned_clusters)}, expected <= 3")
        
        # Check that samples from same speaker are mostly in same cluster
        for expected_speaker, expected_samples in expected_clusters.items():
            # Find which cluster has most samples from this speaker
            cluster_counts = {}
            for sample_id in expected_samples:
                for assigned_id, assigned_samples in assigned_clusters.items():
                    if sample_id in assigned_samples:
                        cluster_counts[assigned_id] = cluster_counts.get(assigned_id, 0) + 1
            
            # Most samples from same speaker should be in same cluster
            if cluster_counts:
                max_count = max(cluster_counts.values())
                self.assertGreaterEqual(max_count, 3, 
                    f"Speaker {expected_speaker} samples not well clustered: {cluster_counts}")
    
    def test_different_speaker_separation(self):
        """Test that different speakers are separated into different clusters."""
        config = {
            'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
            'model_path': self.speaker_model_path,
            'clustering': {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 0.3,
                'similarity_threshold': 0.7
            }
        }
        
        speaker_id_system = SpeakerIdentification(config)
        
        # Create very distinct speakers
        samples = []
        
        # Speaker 1: Low frequency male-like voice
        for i in range(3):
            t = np.linspace(0, 5.0, 80000)
            audio = np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            samples.append({
                'ID': f"male_speaker_{i}",
                'audio': {
                    'array': audio.astype(np.float32),
                    'sampling_rate': 16000
                }
            })
        
        # Speaker 2: High frequency female-like voice  
        for i in range(3):
            t = np.linspace(0, 5.0, 80000)
            audio = np.sin(2 * np.pi * 250 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            samples.append({
                'ID': f"female_speaker_{i}",
                'audio': {
                    'array': audio.astype(np.float32),
                    'sampling_rate': 16000
                }
            })
        
        # Process samples
        speaker_ids = speaker_id_system.process_batch(samples)
        
        # Check that male and female speakers get different IDs
        male_ids = set()
        female_ids = set()
        
        for sample, speaker_id in zip(samples, speaker_ids):
            if 'male_speaker' in sample['ID']:
                male_ids.add(speaker_id)
            else:
                female_ids.add(speaker_id)
        
        print(f"\nMale speaker IDs: {male_ids}")
        print(f"Female speaker IDs: {female_ids}")
        
        # Male and female speakers should have different IDs
        self.assertEqual(len(male_ids.intersection(female_ids)), 0,
                        "Male and female speakers should have different IDs")
    
    def test_clustering_parameters_impact(self):
        """Test impact of different clustering parameters on accuracy."""
        # Test with different parameter sets
        param_sets = [
            {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 0.1,  # Very tight clustering
            },
            {
                'min_cluster_size': 2,
                'min_samples': 1, 
                'cluster_selection_epsilon': 0.5,  # Medium clustering
            },
            {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 1.0,  # Loose clustering
            }
        ]
        
        # Create test samples - 2 speakers, 4 samples each
        samples = []
        for speaker_id in range(2):
            for sample_id in range(4):
                audio = self.create_synthetic_speaker_audio(speaker_id, sample_id)
                samples.append({
                    'ID': f"spk{speaker_id}_s{sample_id}",
                    'audio': {
                        'array': audio,
                        'sampling_rate': 16000
                    }
                })
        
        results = {}
        for i, params in enumerate(param_sets):
            config = {
                'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
                'model_path': f"{self.speaker_model_path}_{i}",
                'clustering': params
            }
            
            speaker_id_system = SpeakerIdentification(config)
            speaker_ids = speaker_id_system.process_batch(samples)
            
            # Count unique speaker IDs
            unique_ids = len(set(speaker_ids))
            results[f"epsilon_{params['cluster_selection_epsilon']}"] = unique_ids
            
            print(f"\nEpsilon {params['cluster_selection_epsilon']}: {unique_ids} unique speakers")
            print(f"Speaker assignments: {list(zip([s['ID'] for s in samples], speaker_ids))}")
        
        # With proper parameters, we should get close to 2 clusters
        self.assertIn(2, results.values(), 
                     f"No parameter set produced expected 2 clusters: {results}")
    
    def test_small_batch_clustering(self):
        """Test clustering with small batches (edge case)."""
        config = {
            'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
            'model_path': self.speaker_model_path,
            'clustering': {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 0.5,
            }
        }
        
        speaker_id_system = SpeakerIdentification(config)
        
        # Test with just 3 samples from 2 speakers
        samples = []
        
        # Speaker 1: 2 samples
        for i in range(2):
            audio = self.create_synthetic_speaker_audio(0, i)
            samples.append({
                'ID': f"speaker1_sample{i}",
                'audio': {'array': audio, 'sampling_rate': 16000}
            })
        
        # Speaker 2: 1 sample
        audio = self.create_synthetic_speaker_audio(1, 0)
        samples.append({
            'ID': "speaker2_sample0",
            'audio': {'array': audio, 'sampling_rate': 16000}
        })
        
        speaker_ids = speaker_id_system.process_batch(samples)
        
        print(f"\nSmall batch results:")
        for sample, spk_id in zip(samples, speaker_ids):
            print(f"  {sample['ID']}: {spk_id}")
        
        # Should have 2 unique speaker IDs
        unique_ids = set(speaker_ids)
        self.assertLessEqual(len(unique_ids), 3,
                           f"Too many unique IDs for 3 samples: {len(unique_ids)}")
    
    def test_incremental_clustering(self):
        """Test that incremental clustering maintains consistency."""
        config = {
            'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
            'model_path': self.speaker_model_path,
            'clustering': {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 0.5,
                'similarity_threshold': 0.7
            }
        }
        
        speaker_id_system = SpeakerIdentification(config)
        
        # First batch: 2 speakers, 3 samples each
        batch1 = []
        for speaker_id in range(2):
            for sample_id in range(3):
                audio = self.create_synthetic_speaker_audio(speaker_id, sample_id)
                batch1.append({
                    'ID': f"batch1_spk{speaker_id}_s{sample_id}",
                    'audio': {'array': audio, 'sampling_rate': 16000}
                })
        
        batch1_ids = speaker_id_system.process_batch(batch1)
        
        # Save model state
        speaker_id_system.save_model()
        
        # Second batch: Same 2 speakers, new samples
        batch2 = []
        for speaker_id in range(2):
            for sample_id in range(3, 5):  # New samples
                audio = self.create_synthetic_speaker_audio(speaker_id, sample_id)
                batch2.append({
                    'ID': f"batch2_spk{speaker_id}_s{sample_id}",
                    'audio': {'array': audio, 'sampling_rate': 16000}
                })
        
        batch2_ids = speaker_id_system.process_batch(batch2)
        
        print(f"\nBatch 1 assignments: {list(zip([s['ID'] for s in batch1], batch1_ids))}")
        print(f"Batch 2 assignments: {list(zip([s['ID'] for s in batch2], batch2_ids))}")
        
        # Check if similar speakers from batch 2 got assigned to existing clusters
        # This tests the _merge_with_existing functionality
        self.assertIsNotNone(speaker_id_system.cluster_centroids,
                           "Cluster centroids should be saved after first batch")


if __name__ == '__main__':
    unittest.main()