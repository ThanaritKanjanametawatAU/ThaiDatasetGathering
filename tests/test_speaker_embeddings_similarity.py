#!/usr/bin/env python3
"""
Test to analyze how audio enhancement affects speaker embeddings.
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


class TestSpeakerEmbeddingsSimilarity(unittest.TestCase):
    """Test to analyze embedding similarities before and after enhancement."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 16000
        
        # Use the actual config from config.py
        self.config = SPEAKER_ID_CONFIG.copy()
        self.config['model_path'] = os.path.join(self.temp_dir, 'test_speaker_model.json')
        self.config['fresh'] = True
        
        # Initialize components
        self.speaker_id = SpeakerIdentification(self.config)
        self.enhancer = AudioEnhancer(use_gpu=False)
        
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'speaker_id'):
            self.speaker_id.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_distinct_speakers(self):
        """Create two very distinct speaker audio samples."""
        duration = 2.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        samples = []
        
        # Speaker 1: Low pitch male-like voice (120 Hz)
        # S1-S8 and S10
        for i in range(1, 9):
            # Base frequency with slight variation
            f0 = 120 + i * 2  # 122-136 Hz
            
            # Generate harmonics
            audio = np.zeros_like(t)
            for harmonic in range(1, 8):
                freq = f0 * harmonic
                amp = 1.0 / (harmonic ** 0.8)
                audio += amp * np.sin(2 * np.pi * freq * t)
            
            # Add formants
            formants = [700, 1220, 2600]
            for j, formant in enumerate(formants):
                formant_varied = formant * (1 + i * 0.01)  # Slight variation
                audio += 0.3 / (j + 1) * np.sin(2 * np.pi * formant_varied * t)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            samples.append({
                'ID': f'S{i}',
                'audio': {'array': audio, 'sampling_rate': self.sample_rate},
                'expected_speaker': 1
            })
        
        # Speaker 2: High pitch female-like voice (250 Hz)
        # S9
        f0 = 250
        audio = np.zeros_like(t)
        for harmonic in range(1, 8):
            freq = f0 * harmonic
            amp = 1.0 / (harmonic ** 0.7)  # Different decay
            audio += amp * np.sin(2 * np.pi * freq * t)
        
        # Different formants
        formants = [850, 1850, 2800]
        for j, formant in enumerate(formants):
            audio += 0.4 / (j + 1) * np.sin(2 * np.pi * formant * t)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        samples.append({
            'ID': 'S9',
            'audio': {'array': audio, 'sampling_rate': self.sample_rate},
            'expected_speaker': 2
        })
        
        # S10: Same as speaker 1
        f0 = 125  # Slight variation
        audio = np.zeros_like(t)
        for harmonic in range(1, 8):
            freq = f0 * harmonic
            amp = 1.0 / (harmonic ** 0.8)
            audio += amp * np.sin(2 * np.pi * freq * t)
        
        formants = [700, 1220, 2600]
        for j, formant in enumerate(formants):
            audio += 0.3 / (j + 1) * np.sin(2 * np.pi * formant * t)
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        samples.append({
            'ID': 'S10',
            'audio': {'array': audio, 'sampling_rate': self.sample_rate},
            'expected_speaker': 1
        })
        
        return samples
    
    def test_embedding_similarity_analysis(self):
        """Analyze embedding similarities before and after enhancement."""
        samples = self.create_distinct_speakers()
        
        # Extract embeddings before enhancement
        print("\n=== BEFORE ENHANCEMENT ===")
        embeddings_before = []
        
        for sample in samples:
            audio = sample['audio']['array']
            embedding = self.speaker_id.extract_embedding(audio, self.sample_rate)
            embeddings_before.append(embedding)
        
        embeddings_before = np.array(embeddings_before)
        
        # Normalize for cosine similarity
        normalized_before = embeddings_before / (np.linalg.norm(embeddings_before, axis=1, keepdims=True) + 1e-8)
        
        # Calculate similarity matrix
        similarities_before = np.dot(normalized_before, normalized_before.T)
        
        print("Similarity matrix (before enhancement):")
        print("     ", "  ".join([f"  {s['ID']}" for s in samples]))
        for i, sample in enumerate(samples):
            print(f"{sample['ID']:>4}:", " ".join([f"{similarities_before[i, j]:5.3f}" for j in range(len(samples))]))
        
        # Calculate average similarities
        s1_to_s8_s10_mask = np.array([s['expected_speaker'] == 1 for s in samples])
        within_speaker1 = similarities_before[s1_to_s8_s10_mask][:, s1_to_s8_s10_mask]
        within_speaker1_avg = np.mean(within_speaker1[np.triu_indices_from(within_speaker1, k=1)])
        
        between_speakers = []
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                if samples[i]['expected_speaker'] != samples[j]['expected_speaker']:
                    between_speakers.append(similarities_before[i, j])
        between_speakers_avg = np.mean(between_speakers) if between_speakers else 0
        
        print(f"\nAverage similarity within speaker 1 (S1-S8,S10): {within_speaker1_avg:.3f}")
        print(f"Average similarity between speakers: {between_speakers_avg:.3f}")
        print(f"Separation: {within_speaker1_avg - between_speakers_avg:.3f}")
        
        # Now with enhancement
        print("\n=== AFTER ENHANCEMENT (moderate level) ===")
        embeddings_after = []
        
        for sample in samples:
            audio = sample['audio']['array']
            
            # Apply enhancement
            enhanced_audio, metadata = self.enhancer.enhance(
                audio,
                self.sample_rate,
                noise_level='moderate',
                return_metadata=True
            )
            
            embedding = self.speaker_id.extract_embedding(enhanced_audio, self.sample_rate)
            embeddings_after.append(embedding)
        
        embeddings_after = np.array(embeddings_after)
        
        # Normalize for cosine similarity
        normalized_after = embeddings_after / (np.linalg.norm(embeddings_after, axis=1, keepdims=True) + 1e-8)
        
        # Calculate similarity matrix
        similarities_after = np.dot(normalized_after, normalized_after.T)
        
        print("Similarity matrix (after enhancement):")
        print("     ", "  ".join([f"  {s['ID']}" for s in samples]))
        for i, sample in enumerate(samples):
            print(f"{sample['ID']:>4}:", " ".join([f"{similarities_after[i, j]:5.3f}" for j in range(len(samples))]))
        
        # Calculate average similarities after enhancement
        within_speaker1_after = similarities_after[s1_to_s8_s10_mask][:, s1_to_s8_s10_mask]
        within_speaker1_avg_after = np.mean(within_speaker1_after[np.triu_indices_from(within_speaker1_after, k=1)])
        
        between_speakers_after = []
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                if samples[i]['expected_speaker'] != samples[j]['expected_speaker']:
                    between_speakers_after.append(similarities_after[i, j])
        between_speakers_avg_after = np.mean(between_speakers_after) if between_speakers_after else 0
        
        print(f"\nAverage similarity within speaker 1 (S1-S8,S10): {within_speaker1_avg_after:.3f}")
        print(f"Average similarity between speakers: {between_speakers_avg_after:.3f}")
        print(f"Separation: {within_speaker1_avg_after - between_speakers_avg_after:.3f}")
        
        # Test clustering on enhanced embeddings
        print("\n=== CLUSTERING TEST ===")
        speaker_ids = self.speaker_id.cluster_embeddings(embeddings_after, [s['ID'] for s in samples])
        
        print("Speaker ID assignments:")
        for sample, spk_id in zip(samples, speaker_ids):
            print(f"  {sample['ID']}: {spk_id}")
        
        # Check clustering accuracy
        id_groups = {}
        for i, spk_id in enumerate(speaker_ids):
            if spk_id not in id_groups:
                id_groups[spk_id] = []
            id_groups[spk_id].append(samples[i]['ID'])
        
        print(f"\nSpeaker groups: {id_groups}")
        
        # Verify clustering
        s1_to_s8_s10 = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S10']
        s9 = ['S9']
        
        # Check if S1-S8,S10 are in the same group
        main_group_id = None
        for spk_id, group in id_groups.items():
            if any(s in group for s in s1_to_s8_s10):
                main_group_id = spk_id
                break
        
        if main_group_id:
            main_group = id_groups[main_group_id]
            correct_main = all(s in main_group for s in s1_to_s8_s10)
            s9_different = 'S9' not in main_group
            
            print(f"\nClustering result:")
            print(f"  S1-S8,S10 clustered together: {correct_main}")
            print(f"  S9 in different cluster: {s9_different}")
            
            if correct_main and s9_different:
                print("  ✓ CORRECT CLUSTERING")
            else:
                print("  ✗ INCORRECT CLUSTERING")
                print(f"\nDEBUG: Enhancement may be making samples too similar")
                print(f"  Mean similarity after enhancement: {np.mean(similarities_after[np.triu_indices_from(similarities_after, k=1)]):.3f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)