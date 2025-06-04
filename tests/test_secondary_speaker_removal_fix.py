#!/usr/bin/env python3
"""
Test-Driven Development tests for secondary speaker removal fix.

Tests verify:
1. Secondary speaker detection works properly
2. Ultra aggressive mode triggers secondary speaker removal
3. Speaker ID functionality continues to work
4. Resume functionality still works
5. Performance is optimized
"""

import unittest
import numpy as np
import tempfile
import json
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch, MagicMock
from processors.audio_enhancement.core import AudioEnhancer
from processors.audio_enhancement.detection.secondary_speaker import AdaptiveSecondaryDetection, DetectionResult
from processors.audio_enhancement.speaker_separation import SpeakerSeparator, SeparationConfig
from processors.speaker_identification import SpeakerIdentification


class TestSecondSpeakerRemovalFix(unittest.TestCase):
    """TDD tests for secondary speaker removal fix"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.duration = 3  # 3 seconds
        self.audio_length = self.sample_rate * self.duration
        
        # Create test audio with two speakers
        # Main speaker (0-1s and 2-3s)
        self.main_speaker_audio = self._generate_voice_signal(frequency=200, duration=1)
        
        # Secondary speaker (1-2s)
        self.secondary_speaker_audio = self._generate_voice_signal(frequency=300, duration=1)
        
        # Combined audio
        self.mixed_audio = np.zeros(self.audio_length)
        self.mixed_audio[:self.sample_rate] = self.main_speaker_audio  # 0-1s
        self.mixed_audio[self.sample_rate:2*self.sample_rate] = self.secondary_speaker_audio  # 1-2s
        self.mixed_audio[2*self.sample_rate:] = self.main_speaker_audio  # 2-3s
        
    def _generate_voice_signal(self, frequency=200, duration=1):
        """Generate a simple voice-like signal"""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        # Fundamental frequency
        signal = np.sin(2 * np.pi * frequency * t)
        # Add harmonics
        signal += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
        signal += 0.3 * np.sin(2 * np.pi * frequency * 3 * t)
        # Add some noise
        signal += 0.1 * np.random.randn(len(t))
        return signal * 0.5
    
    def test_secondary_speaker_detection_works(self):
        """Test that secondary speaker detection correctly identifies second speaker"""
        detector = AdaptiveSecondaryDetection(
            min_duration=0.5,
            max_duration=2.0,
            confidence_threshold=0.5
        )
        
        # Detect secondary speakers
        detections = detector.detect(self.mixed_audio, self.sample_rate)
        
        # Should detect secondary speaker between 1-2 seconds
        self.assertGreater(len(detections), 0, "Should detect at least one secondary speaker")
        
        # Check detection is in the right time range
        found_correct_detection = False
        for detection in detections:
            if 0.5 < detection.start_time < 1.5 and 1.5 < detection.end_time < 2.5:
                found_correct_detection = True
                break
        
        self.assertTrue(found_correct_detection, 
                       f"Should detect secondary speaker around 1-2s, got: {[(d.start_time, d.end_time) for d in detections]}")
    
    def test_ultra_aggressive_triggers_secondary_speaker_removal(self):
        """Test that ultra_aggressive enhancement level triggers secondary speaker removal"""
        enhancer = AudioEnhancer(use_gpu=False, workers=2)
        
        # Process with ultra_aggressive mode
        enhanced_audio, metadata = enhancer.enhance(
            self.mixed_audio,
            self.sample_rate,
            noise_level='ultra_aggressive',
            return_metadata=True
        )
        
        # Check that secondary speaker removal was applied
        # This should be true in the fixed version
        self.assertTrue(
            metadata.get('use_speaker_separation', False),
            f"Ultra aggressive mode should use speaker separation, got metadata: {metadata}"
        )
        
        # Verify audio is actually modified
        self.assertFalse(np.array_equal(self.mixed_audio, enhanced_audio),
                        "Audio should be modified by enhancement")
        
        # Even if no secondary speaker is detected, the pipeline should run
        self.assertIn('secondary_speaker_detected', metadata,
                     "Metadata should include secondary_speaker_detected flag")
    
    def test_speaker_separation_directly(self):
        """Test speaker separation module directly"""
        config = SeparationConfig(
            suppression_strength=0.6,
            confidence_threshold=0.5,
            use_sepformer=False  # Don't require external model
        )
        separator = SpeakerSeparator(config)
        
        # Process audio
        result = separator.separate_speakers(self.mixed_audio, self.sample_rate)
        
        # Check results
        self.assertIn('audio', result)
        self.assertIn('detections', result)
        self.assertIn('metrics', result)
        
        # Should detect secondary speakers
        self.assertGreater(len(result['detections']), 0,
                          "Should detect secondary speakers")
        
        # Audio should be modified
        self.assertFalse(np.array_equal(self.mixed_audio, result['audio']),
                        "Audio should be modified")
        
        # Check suppression metrics
        metrics = result['metrics']
        self.assertIn('secondary_speaker_count', metrics)
        self.assertGreater(metrics['secondary_speaker_count'], 0)
    
    def test_performance_optimization(self):
        """Test that processing is optimized for speed"""
        enhancer = AudioEnhancer(use_gpu=False, workers=4)
        
        # Time single sample processing
        start_time = time.time()
        enhanced_single, _ = enhancer.enhance(
            self.mixed_audio,
            self.sample_rate,
            noise_level='ultra_aggressive',
            return_metadata=True
        )
        single_time = time.time() - start_time
        
        # Time batch processing
        batch_size = 10
        audio_batch = [
            (self.mixed_audio.copy(), self.sample_rate, f"sample_{i}")
            for i in range(batch_size)
        ]
        
        start_time = time.time()
        batch_results = enhancer.process_batch(audio_batch, max_workers=4)
        batch_time = time.time() - start_time
        
        # Batch should be faster per sample
        time_per_sample_single = single_time
        time_per_sample_batch = batch_time / batch_size
        
        # Batch processing should be at least 50% faster per sample
        self.assertLess(time_per_sample_batch, time_per_sample_single * 0.7,
                       f"Batch processing should be faster: {time_per_sample_batch:.3f}s vs {time_per_sample_single:.3f}s per sample")
        
        # All results should be valid
        self.assertEqual(len(batch_results), batch_size)
        for enhanced, metadata in batch_results:
            self.assertIsInstance(enhanced, np.ndarray)
            self.assertIsInstance(metadata, dict)
    
    def test_speaker_id_still_works(self):
        """Test that speaker ID functionality continues to work correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'model': 'pyannote/embedding',
                'model_path': os.path.join(tmpdir, 'speaker_model.json'),
                'fresh': True,
                'clustering': {
                    'algorithm': 'agglomerative',
                    'similarity_threshold': 0.9
                }
            }
            
            speaker_id = SpeakerIdentification(config)
            
            # Create test samples with more realistic variation
            samples = []
            for i in range(10):
                if i == 8:  # S9 is different speaker
                    # Generate significantly different audio with more variation
                    audio = self._generate_voice_signal(frequency=350, duration=1)
                    # Add different harmonics pattern
                    t = np.linspace(0, 1, self.sample_rate)
                    audio += 0.3 * np.sin(2 * np.pi * 700 * t)  # Different harmonics
                    audio += 0.2 * np.random.randn(len(audio)) * 0.5  # Different noise pattern
                else:  # S1-S8 and S10 are same speaker
                    # Generate consistent audio with slight variations
                    audio = self._generate_voice_signal(frequency=200 + i*2, duration=1)  # Slight variation
                    # Add consistent harmonics
                    t = np.linspace(0, 1, self.sample_rate)
                    audio += 0.2 * np.sin(2 * np.pi * 400 * t)
                
                # Normalize
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                
                samples.append({
                    'ID': f'S{i+1}',
                    'audio': {
                        'array': audio,
                        'sampling_rate': self.sample_rate
                    }
                })
            
            # Process batch
            speaker_ids = speaker_id.process_batch(samples)
            
            # Verify clustering
            self.assertEqual(len(speaker_ids), 10)
            
            # S1-S8 and S10 should have same speaker ID
            main_speaker_ids = [speaker_ids[i] for i in [0,1,2,3,4,5,6,7,9]]
            self.assertEqual(len(set(main_speaker_ids)), 1,
                           f"S1-S8 and S10 should have same speaker ID, got: {speaker_ids}")
            
            # S9 should have different speaker ID
            self.assertNotEqual(speaker_ids[8], speaker_ids[0],
                              f"S9 should have different speaker ID, got: {speaker_ids}")
    
    def test_resume_functionality_preserved(self):
        """Test that resume functionality still works with speaker ID"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'speaker_model.json')
            
            # First run - fresh
            config1 = {
                'model': 'pyannote/embedding',
                'model_path': checkpoint_path,
                'fresh': True
            }
            
            speaker_id1 = SpeakerIdentification(config1)
            
            # Process some samples
            samples1 = [{
                'ID': f'S{i+1}',
                'audio': {
                    'array': self._generate_voice_signal(frequency=200),
                    'sampling_rate': self.sample_rate
                }
            } for i in range(5)]
            
            speaker_ids1 = speaker_id1.process_batch(samples1)
            
            # Save checkpoint
            speaker_id1.save_model()
            
            # Verify checkpoint was saved
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Second run - resume
            config2 = {
                'model': 'pyannote/embedding',
                'model_path': checkpoint_path,
                'fresh': False  # Resume mode
            }
            
            speaker_id2 = SpeakerIdentification(config2)
            
            # Counter should continue from checkpoint
            self.assertEqual(speaker_id2.speaker_counter, speaker_id1.speaker_counter,
                           "Speaker counter should be loaded from checkpoint")
            
            # Process more samples
            samples2 = [{
                'ID': f'S{i+6}',
                'audio': {
                    'array': self._generate_voice_signal(frequency=200),
                    'sampling_rate': self.sample_rate
                }
            } for i in range(5)]
            
            speaker_ids2 = speaker_id2.process_batch(samples2)
            
            # New speaker IDs should continue from where we left off
            max_id1 = max(int(sid.split('_')[1]) for sid in speaker_ids1)
            min_id2 = min(int(sid.split('_')[1]) for sid in speaker_ids2)
            
            self.assertGreater(min_id2, max_id1,
                             f"Resume should continue speaker IDs: {speaker_ids1} -> {speaker_ids2}")
    
    def test_enhancement_config_includes_speaker_separation(self):
        """Test that enhancement config properly includes speaker separation"""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Check that ultra_aggressive config exists
        self.assertIn('ultra_aggressive', enhancer.ENHANCEMENT_LEVELS)
        
        # In the fixed version, ultra_aggressive should enable speaker separation
        config = enhancer.ENHANCEMENT_LEVELS['ultra_aggressive']
        
        # This test will fail before fix and pass after fix
        self.assertTrue(
            config.get('use_speaker_separation', False) or 
            config.get('check_secondary_speaker', False),
            "Ultra aggressive mode should enable secondary speaker handling"
        )
    
    def test_secondary_speaker_detection_in_assess_noise(self):
        """Test that assess_noise_level can detect secondary speakers"""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Assess noise level on mixed audio
        noise_level = enhancer.assess_noise_level(
            self.mixed_audio,
            self.sample_rate,
            quick=True
        )
        
        # In fixed version, this might return 'secondary_speaker'
        # Or at least not 'clean' for audio with multiple speakers
        self.assertIn(noise_level, 
                     ['mild', 'moderate', 'aggressive', 'ultra_aggressive', 'secondary_speaker'],
                     f"Mixed speaker audio should not be assessed as 'clean', got: {noise_level}")
    
    def test_end_to_end_processing_with_enhancement(self):
        """Test end-to-end processing with audio enhancement enabled"""
        # Simulate processing pipeline
        enhancer = AudioEnhancer(use_gpu=False, workers=2)
        
        # Create a batch of samples with mixed audio
        samples = []
        for i in range(10):
            if i == 8:  # S9 has different mix
                audio = self.mixed_audio.copy()
                audio = np.roll(audio, self.sample_rate // 2)  # Shift secondary speaker
            else:
                audio = self.mixed_audio.copy()
            
            samples.append({
                'ID': f'S{i+1}',
                'audio': {
                    'array': audio,
                    'sampling_rate': self.sample_rate
                }
            })
        
        # Process with enhancement
        enhanced_samples = []
        for sample in samples:
            enhanced_audio, metadata = enhancer.enhance(
                sample['audio']['array'],
                sample['audio']['sampling_rate'],
                noise_level='ultra_aggressive',
                return_metadata=True
            )
            
            enhanced_sample = sample.copy()
            enhanced_sample['audio']['array'] = enhanced_audio
            enhanced_sample['enhancement_metadata'] = metadata
            enhanced_samples.append(enhanced_sample)
        
        # Verify all samples were enhanced
        self.assertEqual(len(enhanced_samples), 10)
        
        # Check that enhancement was applied
        for i, (original, enhanced) in enumerate(zip(samples, enhanced_samples)):
            self.assertFalse(
                np.array_equal(original['audio']['array'], enhanced['audio']['array']),
                f"Sample {i+1} should be modified by enhancement"
            )
            self.assertIn('enhancement_metadata', enhanced)
            self.assertTrue(enhanced['enhancement_metadata']['enhanced'])


if __name__ == '__main__':
    unittest.main()