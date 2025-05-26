"""Test that speaker clustering state is reset between datasets in streaming mode."""

import unittest
import numpy as np
from processors.speaker_identification import SpeakerIdentification


class TestStreamingSpeakerReset(unittest.TestCase):
    """Test speaker clustering reset between datasets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
            'clustering': {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 0.3,  # Lower epsilon for tighter clusters
                'similarity_threshold': 0.8  # Higher threshold for merging
            },
            'model_path': 'test_speaker_model_reset.json'
        }
    
    def create_similar_audio_samples(self, num_samples, base_freq, sample_id_prefix):
        """Create audio samples that will produce similar embeddings."""
        samples = []
        for i in range(num_samples):
            # Create similar audio within a dataset
            audio = np.sin(2 * np.pi * base_freq * np.arange(16000 * 5) / 16000)
            # Add slight variation
            audio += 0.05 * np.random.randn(16000 * 5)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            samples.append({
                'ID': f'{sample_id_prefix}_{i}',
                'audio': {
                    'array': audio.astype(np.float32),
                    'sampling_rate': 16000
                }
            })
        return samples
    
    def test_no_cross_dataset_merging(self):
        """Test that similar speakers from different datasets get different IDs."""
        speaker_id = SpeakerIdentification(self.config)
        
        # Dataset 1: Create samples with base frequency 440Hz (similar to A note)
        dataset1_samples = self.create_similar_audio_samples(5, 440, 'dataset1')
        speaker_ids_1 = speaker_id.process_batch(dataset1_samples)
        
        # All samples in dataset 1 should have the same speaker ID
        unique_ids_1 = set(speaker_ids_1)
        self.assertEqual(len(unique_ids_1), 1, f"Expected 1 speaker in dataset 1, got {len(unique_ids_1)}")
        dataset1_speaker = list(unique_ids_1)[0]
        
        print(f"Dataset 1 speaker: {dataset1_speaker}")
        print(f"Existing clusters before reset: {speaker_id.existing_clusters}")
        print(f"Has centroids: {speaker_id.cluster_centroids is not None}")
        
        # Simulate dataset boundary - reset clustering state
        speaker_id.existing_clusters = {}
        speaker_id.cluster_centroids = None
        
        print(f"Existing clusters after reset: {speaker_id.existing_clusters}")
        print(f"Has centroids after reset: {speaker_id.cluster_centroids is not None}")
        
        # Dataset 2: Create similar samples (same frequency)
        dataset2_samples = self.create_similar_audio_samples(4, 440, 'dataset2')
        speaker_ids_2 = speaker_id.process_batch(dataset2_samples)
        
        # All samples in dataset 2 should have the same speaker ID
        unique_ids_2 = set(speaker_ids_2)
        self.assertEqual(len(unique_ids_2), 1, f"Expected 1 speaker in dataset 2, got {len(unique_ids_2)}")
        dataset2_speaker = list(unique_ids_2)[0]
        
        print(f"Dataset 2 speaker: {dataset2_speaker}")
        
        # The speaker IDs should be different even though audio is similar
        self.assertNotEqual(dataset1_speaker, dataset2_speaker,
                           f"Speakers from different datasets should have different IDs, but both got {dataset1_speaker}")
    
    def test_speaker_counter_continues(self):
        """Test that speaker counter continues incrementing across datasets."""
        speaker_id = SpeakerIdentification(self.config)
        
        initial_counter = speaker_id.speaker_counter
        
        # Process first dataset
        dataset1_samples = self.create_similar_audio_samples(3, 500, 'dataset1')
        speaker_ids_1 = speaker_id.process_batch(dataset1_samples)
        
        counter_after_dataset1 = speaker_id.speaker_counter
        
        # Reset clustering state
        speaker_id.existing_clusters = {}
        speaker_id.cluster_centroids = None
        
        # Process second dataset
        dataset2_samples = self.create_similar_audio_samples(3, 600, 'dataset2')
        speaker_ids_2 = speaker_id.process_batch(dataset2_samples)
        
        counter_after_dataset2 = speaker_id.speaker_counter
        
        # Counter should have increased
        self.assertGreater(counter_after_dataset1, initial_counter)
        self.assertGreater(counter_after_dataset2, counter_after_dataset1)
        
        print(f"Speaker counter progression: {initial_counter} -> {counter_after_dataset1} -> {counter_after_dataset2}")
        print(f"All speaker IDs: {set(speaker_ids_1)} and {set(speaker_ids_2)}")


if __name__ == '__main__':
    unittest.main()