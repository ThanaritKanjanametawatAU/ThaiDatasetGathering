#!/usr/bin/env python3
"""
Test speaker identification with audio enhancement enabled.
This test mimics the actual processing pipeline used in main.sh.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.speaker_identification import SpeakerIdentification
from processors.audio_enhancement.core import AudioEnhancer
from config import SPEAKER_ID_CONFIG
import soundfile as sf


class TestSpeakerIDWithEnhancement(unittest.TestCase):
    """Test speaker ID clustering with audio enhancement preprocessing."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 16000
        
        # Use the actual config from config.py
        self.config = SPEAKER_ID_CONFIG.copy()
        self.config['model_path'] = os.path.join(self.temp_dir, 'test_speaker_model.json')
        self.config['fresh'] = True
        
        # Initialize audio enhancer
        self.enhancer = AudioEnhancer(use_gpu=False)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_realistic_audio_samples(self):
        """Create more realistic audio samples that mimic actual audio data."""
        samples = []
        
        # Create base frequency profiles for two speakers
        speaker1_profile = {
            'f0': 150,  # Fundamental frequency
            'formants': [700, 1220, 2600, 3300],  # Formant frequencies
            'jitter': 0.02,  # Voice stability
            'shimmer': 0.03
        }
        
        speaker2_profile = {
            'f0': 250,  # Higher pitch
            'formants': [850, 1850, 2800, 3800],
            'jitter': 0.025,
            'shimmer': 0.035
        }
        
        duration = 2.0  # 2 seconds per sample
        
        # S1-S8: Speaker 1 with variations
        for i in range(1, 9):
            audio = self._synthesize_voice(speaker1_profile, duration, variation=i*0.01)
            # Add some realistic noise
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
            
            samples.append({
                'ID': f'S{i}',
                'audio': {'array': audio, 'sampling_rate': self.sample_rate}
            })
        
        # S9: Speaker 2 (different speaker)
        audio = self._synthesize_voice(speaker2_profile, duration, variation=0.02)
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise
        
        samples.append({
            'ID': 'S9',
            'audio': {'array': audio, 'sampling_rate': self.sample_rate}
        })
        
        # S10: Speaker 1 again
        audio = self._synthesize_voice(speaker1_profile, duration, variation=0.09)
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise
        
        samples.append({
            'ID': 'S10',
            'audio': {'array': audio, 'sampling_rate': self.sample_rate}
        })
        
        return samples
    
    def _synthesize_voice(self, profile, duration, variation=0.0):
        """Synthesize voice with given profile."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Base frequency with variation
        f0 = profile['f0'] * (1 + variation)
        
        # Generate harmonics
        audio = np.zeros_like(t)
        
        # Fundamental and harmonics
        for harmonic in range(1, 6):
            freq = f0 * harmonic
            amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
            
            # Add jitter (frequency variation)
            jitter = profile['jitter'] * np.random.randn(len(t))
            phase_mod = np.cumsum(jitter) * 2 * np.pi * freq / self.sample_rate
            
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase_mod)
        
        # Add formants (resonances)
        for i, formant_freq in enumerate(profile['formants']):
            # Simple formant simulation
            formant_width = 50 + i * 20  # Bandwidth increases for higher formants
            amplitude = 0.3 / (i + 1)
            
            # Create formant envelope
            envelope = np.exp(-((t - duration/2)**2) / (2 * (duration/4)**2))
            formant = amplitude * envelope * np.sin(2 * np.pi * formant_freq * t)
            
            audio += formant
        
        # Apply amplitude modulation (shimmer)
        shimmer_freq = 3 + np.random.rand()  # 3-4 Hz modulation
        shimmer = 1 + profile['shimmer'] * np.sin(2 * np.pi * shimmer_freq * t)
        audio *= shimmer
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    
    def test_speaker_clustering_with_enhancement(self):
        """Test speaker clustering with audio enhancement preprocessing."""
        # Create realistic audio samples
        samples = self.create_realistic_audio_samples()
        
        # Process samples with audio enhancement (like main.sh does)
        enhanced_samples = []
        for sample in samples:
            audio = sample['audio']['array']
            
            # Apply audio enhancement
            enhanced_audio, metadata = self.enhancer.enhance(
                audio,
                self.sample_rate,
                noise_level='moderate',  # Same as main.sh
                return_metadata=True
            )
            
            enhanced_sample = {
                'ID': sample['ID'],
                'audio': {
                    'array': enhanced_audio,
                    'sampling_rate': self.sample_rate
                }
            }
            enhanced_samples.append(enhanced_sample)
            
            print(f"\n{sample['ID']} enhancement metadata:")
            print(f"  - Noise detected: {metadata.get('noise_detected', False)}")
            print(f"  - Secondary speaker: {metadata.get('secondary_speaker_detected', False)}")
            print(f"  - Processing time: {metadata.get('processing_time', 0):.3f}s")
        
        # Initialize speaker identification
        speaker_id = SpeakerIdentification(self.config)
        
        # Process batch for speaker identification
        speaker_ids = speaker_id.process_batch(enhanced_samples)
        
        # Print results for debugging
        print("\nSpeaker ID results:")
        for i, (sample, spk_id) in enumerate(zip(enhanced_samples, speaker_ids)):
            print(f"  {sample['ID']}: {spk_id}")
        
        # Group by speaker ID
        id_groups = {}
        for i, spk_id in enumerate(speaker_ids):
            if spk_id not in id_groups:
                id_groups[spk_id] = []
            id_groups[spk_id].append(enhanced_samples[i]['ID'])
        
        print(f"\nSpeaker groups: {id_groups}")
        print(f"Number of unique speakers detected: {len(id_groups)}")
        
        # Verify clustering
        # S1-S8 and S10 should have the same speaker ID
        s1_to_s8_ids = speaker_ids[:8]  # S1 to S8
        s10_id = speaker_ids[9]  # S10
        s9_id = speaker_ids[8]   # S9
        
        # Check if S1-S8 and S10 are clustered together
        main_speaker_ids = s1_to_s8_ids + [s10_id]
        unique_main_ids = set(main_speaker_ids)
        
        self.assertEqual(len(unique_main_ids), 1,
                        f"S1-S8 and S10 should have the same speaker ID. Got: {unique_main_ids}")
        
        # S9 should be different
        self.assertNotIn(s9_id, unique_main_ids,
                        f"S9 should have a different speaker ID. Got S9={s9_id}, main speaker={list(unique_main_ids)[0]}")
    
    def test_clustering_parameters_effect(self):
        """Test how different clustering parameters affect results."""
        samples = self.create_realistic_audio_samples()
        
        # Test different parameter sets
        param_sets = [
            {
                'name': 'Current config.py',
                'params': {
                    'min_cluster_size': 2,
                    'min_samples': 1,
                    'cluster_selection_epsilon': 0.3,
                    'similarity_threshold': 0.7
                }
            },
            {
                'name': 'Working commit ed80885',
                'params': {
                    'min_cluster_size': 5,
                    'min_samples': 3,
                    'cluster_selection_epsilon': 0.5,
                    'similarity_threshold': 0.6
                }
            },
            {
                'name': 'Adjusted for small batches',
                'params': {
                    'min_cluster_size': 3,
                    'min_samples': 2,
                    'cluster_selection_epsilon': 0.4,
                    'similarity_threshold': 0.65
                }
            }
        ]
        
        for param_set in param_sets:
            print(f"\n\nTesting with {param_set['name']}:")
            print(f"Parameters: {param_set['params']}")
            
            # Update config
            test_config = self.config.copy()
            test_config['clustering'].update(param_set['params'])
            test_config['fresh'] = True
            
            # Initialize speaker identification
            speaker_id = SpeakerIdentification(test_config)
            
            # Process samples
            speaker_ids = speaker_id.process_batch(samples)
            
            # Analyze results
            id_groups = {}
            for i, spk_id in enumerate(speaker_ids):
                if spk_id not in id_groups:
                    id_groups[spk_id] = []
                id_groups[spk_id].append(samples[i]['ID'])
            
            print(f"Speaker groups: {id_groups}")
            print(f"Number of unique speakers: {len(id_groups)}")
            
            # Check if S1-S8,S10 are grouped correctly
            s1_to_s8_s10 = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S10']
            main_group = None
            for spk_id, group in id_groups.items():
                if any(s in group for s in s1_to_s8_s10):
                    if main_group is None:
                        main_group = spk_id
                    matching = [s for s in s1_to_s8_s10 if s in group]
                    print(f"  Speaker {spk_id} has {len(matching)} samples from main group: {matching}")
            
            # Clean up for next test
            if hasattr(speaker_id, 'cleanup'):
                speaker_id.cleanup()


if __name__ == '__main__':
    unittest.main(verbosity=2)