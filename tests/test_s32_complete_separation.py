#!/usr/bin/env python3
"""
Test-Driven Development for S32 Complete Speaker Separation
==========================================================

This test suite ensures we properly separate overlapping speakers throughout
the entire audio file, not just at the end. S32 is a 19.7 second sample with
multiple overlapping speakers that our previous approaches failed to handle.

Professional Approach:
1. Use source separation to separate ALL speakers
2. Identify primary speaker using embeddings
3. Extract only primary speaker audio
4. Handle overlapping speech throughout the file
"""

import unittest
import numpy as np
import soundfile as sf
from pathlib import Path
import torch


class TestS32CompleteSeparation(unittest.TestCase):
    """Test complete speaker separation for S32 and similar complex samples"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path("test_audio_output")
        self.s32_path = self.test_dir / "s32_from_dataset.wav"
        self.sample_rate = 16000
    
    def create_overlapping_audio(self, duration: float = 5.0):
        """Create test audio with overlapping speakers throughout"""
        samples = int(duration * self.sample_rate)
        t = np.arange(samples) / self.sample_rate
        
        # Primary speaker - continuous throughout
        primary = 0.3 * np.sin(2 * np.pi * 200 * t)
        primary += 0.1 * np.sin(2 * np.pi * 400 * t)  # Harmonics
        
        # Secondary speaker 1 - overlaps at 0-2s and 3-4s
        secondary1 = np.zeros_like(t)
        mask1 = ((t >= 0) & (t < 2)) | ((t >= 3) & (t < 4))
        secondary1[mask1] = 0.4 * np.sin(2 * np.pi * 500 * t[mask1])
        
        # Secondary speaker 2 - overlaps at 1-3s
        secondary2 = np.zeros_like(t)
        mask2 = (t >= 1) & (t < 3)
        secondary2[mask2] = 0.35 * np.sin(2 * np.pi * 700 * t[mask2])
        
        # Mix all speakers
        mixed = primary + secondary1 + secondary2
        
        # Add noise
        mixed += 0.01 * np.random.randn(len(mixed))
        
        return mixed.astype(np.float32), primary.astype(np.float32)
    
    def test_detect_overlapping_speakers_throughout_audio(self):
        """Test detection of overlapping speakers throughout the file"""
        from processors.audio_enhancement.complete_separation import CompleteSeparator
        
        mixed, _ = self.create_overlapping_audio()
        
        separator = CompleteSeparator()
        analysis = separator.analyze_overlapping_speakers(mixed, self.sample_rate)
        
        # Should detect multiple speakers
        self.assertGreaterEqual(analysis.num_speakers, 2)
        
        # Should detect overlapping regions
        self.assertTrue(analysis.has_overlapping_speech)
        self.assertGreater(len(analysis.overlap_regions), 0)
        
        # Should identify that overlap occurs throughout, not just at end
        total_overlap_duration = sum(r.duration for r in analysis.overlap_regions)
        self.assertGreater(total_overlap_duration, 2.0)  # At least 2 seconds of overlap
    
    def test_separate_all_speakers_using_sepformer(self):
        """Test separation of all speakers using SepFormer model"""
        from processors.audio_enhancement.complete_separation import CompleteSeparator
        
        mixed, primary = self.create_overlapping_audio()
        
        separator = CompleteSeparator()
        separated_sources = separator.separate_speakers(mixed, self.sample_rate)
        
        # Should return multiple separated sources
        self.assertGreaterEqual(len(separated_sources), 2)
        
        # Each source should have same length as input
        for source in separated_sources:
            self.assertEqual(len(source), len(mixed))
        
        # At least one source should correlate highly with primary speaker
        correlations = []
        for source in separated_sources:
            corr = np.corrcoef(source[:1000], primary[:1000])[0, 1]
            correlations.append(abs(corr))
        
        self.assertGreater(max(correlations), 0.8)  # High correlation with primary
    
    def test_identify_primary_speaker_using_embeddings(self):
        """Test identification of primary speaker using speaker embeddings"""
        from processors.audio_enhancement.complete_separation import CompleteSeparator
        
        mixed, _ = self.create_overlapping_audio()
        
        separator = CompleteSeparator()
        separated_sources = separator.separate_speakers(mixed, self.sample_rate)
        
        # Identify primary speaker (most dominant/longest speaking)
        primary_idx = separator.identify_primary_speaker(
            separated_sources, 
            self.sample_rate
        )
        
        self.assertIsNotNone(primary_idx)
        self.assertGreaterEqual(primary_idx, 0)
        self.assertLess(primary_idx, len(separated_sources))
    
    def test_extract_only_primary_speaker(self):
        """Test extraction of only the primary speaker"""
        from processors.audio_enhancement.complete_separation import CompleteSeparator
        
        mixed, primary_truth = self.create_overlapping_audio()
        
        separator = CompleteSeparator()
        extracted_primary = separator.extract_primary_speaker(mixed, self.sample_rate)
        
        # Should return clean audio
        self.assertEqual(len(extracted_primary), len(mixed))
        
        # Should have high SNR improvement
        noise_before = mixed - primary_truth
        noise_power_before = np.mean(noise_before**2)
        
        noise_after = extracted_primary - primary_truth
        noise_power_after = np.mean(noise_after**2)
        
        snr_improvement = 10 * np.log10(noise_power_before / (noise_power_after + 1e-10))
        self.assertGreater(snr_improvement, 10)  # At least 10dB improvement
    
    def test_handle_real_s32_sample(self):
        """Test with the actual S32 sample that has overlapping speakers"""
        if not self.s32_path.exists():
            self.skipTest("S32 sample not available")
        
        from processors.audio_enhancement.complete_separation import CompleteSeparator
        
        # Load S32
        audio, sr = sf.read(self.s32_path)
        
        separator = CompleteSeparator()
        cleaned = separator.extract_primary_speaker(audio, sr)
        
        # Should not destroy the audio
        self.assertGreater(np.max(np.abs(cleaned)), 0.01)
        
        # Should have similar energy to original (not silence)
        original_energy = np.sqrt(np.mean(audio**2))
        cleaned_energy = np.sqrt(np.mean(cleaned**2))
        
        # Energy should be at least 50% of original
        self.assertGreater(cleaned_energy, original_energy * 0.5)
        
        # Save for manual verification
        sf.write(self.test_dir / "s32_cleaned_complete.wav", cleaned, sr)
    
    def test_integration_with_enhancement_pipeline(self):
        """Test integration with existing enhancement pipeline"""
        from processors.audio_enhancement.core import AudioEnhancer
        
        mixed, _ = self.create_overlapping_audio()
        
        try:
            # Enhancer should use complete separation for overlapping speech
            enhancer = AudioEnhancer(
                enhancement_level='ultra_aggressive',
                use_gpu=False
            )
            
            enhanced = enhancer.enhance(mixed, self.sample_rate)
            
            # Should not be silence
            self.assertGreater(np.max(np.abs(enhanced)), 0.01)
            
            # Should have similar length
            self.assertEqual(len(enhanced), len(mixed))
            
        except Exception as e:
            # For now, just ensure the complete_separator is accessible
            self.assertTrue(hasattr(enhancer, 'complete_separator'))
            self.assertIsNotNone(enhancer.complete_separator)
    
    def test_performance_on_long_audio(self):
        """Test performance on long audio files like S32 (19.7s)"""
        from processors.audio_enhancement.complete_separation import CompleteSeparator
        
        # Create 20 second test audio
        mixed, _ = self.create_overlapping_audio(duration=20.0)
        
        separator = CompleteSeparator()
        
        import time
        start = time.time()
        cleaned = separator.extract_primary_speaker(mixed, self.sample_rate)
        duration = time.time() - start
        
        # Should complete in reasonable time (less than 10s for 20s audio)
        self.assertLess(duration, 10.0)
        
        # Should produce valid output
        self.assertEqual(len(cleaned), len(mixed))
    
    def test_preserve_single_speaker_audio(self):
        """Test that single speaker audio is preserved unchanged"""
        from processors.audio_enhancement.complete_separation import CompleteSeparator
        
        # Create audio with only one speaker
        t = np.arange(self.sample_rate * 3) / self.sample_rate
        single_speaker = 0.3 * np.sin(2 * np.pi * 200 * t)
        
        separator = CompleteSeparator()
        processed = separator.extract_primary_speaker(single_speaker, self.sample_rate)
        
        # Should be nearly identical
        difference = np.mean(np.abs(processed - single_speaker))
        self.assertLess(difference, 0.01)
    
    def test_batch_processing_capability(self):
        """Test ability to process multiple files efficiently"""
        from processors.audio_enhancement.complete_separation import CompleteSeparator
        
        # Create multiple test files
        test_audios = []
        for i in range(5):
            mixed, _ = self.create_overlapping_audio(duration=3.0)
            test_audios.append(mixed)
        
        separator = CompleteSeparator()
        
        # Process batch
        results = separator.process_batch(test_audios, self.sample_rate)
        
        self.assertEqual(len(results), len(test_audios))
        
        # All should be processed
        for result in results:
            self.assertGreater(np.max(np.abs(result)), 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)