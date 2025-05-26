"""Test speaker ID assignment across different datasets."""

import unittest
import numpy as np
from processors.speaker_identification import SpeakerIdentification


class TestSpeakerIdDatasetSeparation(unittest.TestCase):
    """Test that speaker IDs are properly separated across datasets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'model': 'pyannote/wespeaker-voxceleb-resnet34-LM',
            'clustering': {
                'min_cluster_size': 2,
                'min_samples': 1,
                'cluster_selection_epsilon': 0.5,
                'similarity_threshold': 0.6
            },
            'model_path': 'test_speaker_model.json'
        }
    
    def create_mock_audio_samples(self, num_samples, dataset_name, speaker_offset=0):
        """Create mock audio samples with embeddings that cluster together."""
        samples = []
        for i in range(num_samples):
            # Create audio that will produce similar embeddings within dataset
            # but different embeddings across datasets
            base_audio = np.random.randn(16000 * 5) * 0.1  # 5 seconds at 16kHz
            
            # Add dataset-specific and speaker-specific patterns
            dataset_pattern = np.sin(2 * np.pi * (100 + dataset_name.__hash__() % 100) * np.arange(16000 * 5) / 16000)
            speaker_pattern = np.sin(2 * np.pi * (200 + speaker_offset * 50) * np.arange(16000 * 5) / 16000)
            
            audio = base_audio + 0.3 * dataset_pattern + 0.2 * speaker_pattern
            audio = audio / (np.max(np.abs(audio)) + 1e-8)  # Normalize
            
            samples.append({
                'ID': f'{dataset_name}_{i}',
                'audio': {
                    'array': audio.astype(np.float32),
                    'sampling_rate': 16000
                },
                'dataset_name': dataset_name
            })
        return samples
    
    def test_speaker_ids_not_reused_across_datasets(self):
        """Test that speaker IDs from one dataset are not reused in another dataset."""
        # Mark test as in progress
        # Initialize speaker identifier
        speaker_id = SpeakerIdentification(self.config)
        
        # Process first dataset (GigaSpeech2) - 2 different speakers
        dataset1_samples = []
        # Speaker 1 in dataset 1
        dataset1_samples.extend(self.create_mock_audio_samples(5, 'GigaSpeech2', speaker_offset=0))
        # Speaker 2 in dataset 1  
        dataset1_samples.extend(self.create_mock_audio_samples(5, 'GigaSpeech2', speaker_offset=1))
        
        speaker_ids_1 = speaker_id.process_batch(dataset1_samples)
        
        # Process second dataset (ProcessedVoiceTH) - 2 different speakers
        dataset2_samples = []
        # Speaker 3 in dataset 2
        dataset2_samples.extend(self.create_mock_audio_samples(4, 'ProcessedVoiceTH', speaker_offset=2))
        # Speaker 4 in dataset 2
        dataset2_samples.extend(self.create_mock_audio_samples(4, 'ProcessedVoiceTH', speaker_offset=3))
        
        speaker_ids_2 = speaker_id.process_batch(dataset2_samples)
        
        # Extract unique speaker IDs from each dataset
        unique_ids_1 = set(speaker_ids_1)
        unique_ids_2 = set(speaker_ids_2)
        
        print(f"Dataset 1 speaker IDs: {unique_ids_1}")
        print(f"Dataset 2 speaker IDs: {unique_ids_2}")
        print(f"Speaker ID assignments:")
        for i, (sample, spk_id) in enumerate(zip(dataset1_samples + dataset2_samples, 
                                                  speaker_ids_1 + speaker_ids_2)):
            print(f"  {sample['ID']}: {spk_id}")
        
        # Check that speaker IDs don't overlap between datasets
        overlapping_ids = unique_ids_1.intersection(unique_ids_2)
        self.assertEqual(len(overlapping_ids), 0, 
                        f"Speaker IDs are reused across datasets: {overlapping_ids}")
        
        # Verify we have the expected number of unique speakers
        all_unique_ids = unique_ids_1.union(unique_ids_2)
        self.assertGreaterEqual(len(all_unique_ids), 4,
                               f"Expected at least 4 unique speakers, got {len(all_unique_ids)}")
    
    def test_streaming_mode_speaker_separation(self):
        """Test speaker ID assignment in streaming mode simulation."""
        speaker_id = SpeakerIdentification(self.config)
        
        all_speaker_ids = []
        dataset_boundaries = []
        
        # Simulate streaming mode processing
        for dataset_idx, dataset_name in enumerate(['GigaSpeech2', 'ProcessedVoiceTH', 'MozillaCV']):
            dataset_start = len(all_speaker_ids)
            
            # Process samples in small batches as in streaming
            for batch_idx in range(2):  # 2 batches per dataset
                batch_samples = self.create_mock_audio_samples(
                    3,  # 3 samples per batch
                    dataset_name,
                    speaker_offset=dataset_idx * 2 + batch_idx
                )
                
                batch_speaker_ids = speaker_id.process_batch(batch_samples)
                all_speaker_ids.extend(batch_speaker_ids)
            
            dataset_boundaries.append((dataset_start, len(all_speaker_ids), dataset_name))
        
        # Check each dataset's speaker IDs
        print("\nStreaming mode results:")
        for start, end, name in dataset_boundaries:
            dataset_ids = set(all_speaker_ids[start:end])
            print(f"{name}: {dataset_ids}")
            
            # Check overlap with other datasets
            for other_start, other_end, other_name in dataset_boundaries:
                if other_name != name:
                    other_ids = set(all_speaker_ids[other_start:other_end])
                    overlap = dataset_ids.intersection(other_ids)
                    if overlap:
                        print(f"  WARNING: Overlap with {other_name}: {overlap}")


if __name__ == '__main__':
    unittest.main()