#!/usr/bin/env python3
"""
Test-Driven Development: Loudness Normalization Tests

Requirements:
1. After preprocessing, audio should be as loud as the original
2. Quality should be maintained (same as original)
3. All noise removal should remain effective
4. No clipping or distortion should occur
"""

import unittest
import numpy as np
import librosa
import torch
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the processor (will fail initially as loudness normalization doesn't exist yet)
from processors.audio_enhancement.loudness_normalizer import LoudnessNormalizer
from process_gigaspeech2_thai_50_samples import PatternMetricGANProcessor


class TestLoudnessNormalization(unittest.TestCase):
    """Comprehensive tests for loudness normalization after preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sr = 16000
        self.duration = 3.0
        self.processor = PatternMetricGANProcessor()
        self.normalizer = LoudnessNormalizer()
        
        # Create test audio samples
        self.create_test_samples()
    
    def create_test_samples(self):
        """Create various test audio samples"""
        t = np.linspace(0, self.duration, int(self.duration * self.sr))
        
        # Normal speech-like signal
        self.normal_audio = 0.3 * np.sin(2 * np.pi * 440 * t) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
        
        # Quiet audio
        self.quiet_audio = 0.05 * np.sin(2 * np.pi * 440 * t)
        
        # Loud audio (near clipping)
        self.loud_audio = 0.9 * np.sin(2 * np.pi * 440 * t)
        
        # Audio with noise
        noise = 0.1 * np.random.randn(len(t))
        self.noisy_audio = 0.3 * np.sin(2 * np.pi * 440 * t) + noise
        
        # Silent audio
        self.silent_audio = np.zeros(len(t))
        
        # Audio with interruptions
        self.interrupted_audio = self.normal_audio.copy()
        # Add loud burst
        burst_start = int(1.0 * self.sr)
        burst_end = int(1.1 * self.sr)
        self.interrupted_audio[burst_start:burst_end] += 0.5 * np.random.randn(burst_end - burst_start)
    
    def test_loudness_matches_original(self):
        """Test that processed audio loudness matches the original"""
        # Process with pattern suppression and MetricGAN+
        processed, _ = self.processor.process_audio(self.normal_audio, self.sr)
        
        # Apply loudness normalization
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.normal_audio, 
            self.sr
        )
        
        # Calculate RMS of both
        original_rms = np.sqrt(np.mean(self.normal_audio**2))
        normalized_rms = np.sqrt(np.mean(normalized**2))
        
        # They should be very close (within 5%)
        ratio = normalized_rms / original_rms
        self.assertAlmostEqual(ratio, 1.0, delta=0.05, 
                              msg=f"Loudness ratio {ratio:.3f} not close to 1.0")
    
    def test_peak_normalization_option(self):
        """Test peak-based normalization as an alternative"""
        processed, _ = self.processor.process_audio(self.normal_audio, self.sr)
        
        # Apply peak normalization
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.normal_audio, 
            self.sr,
            method='peak'
        )
        
        # Check peaks match
        original_peak = np.max(np.abs(self.normal_audio))
        normalized_peak = np.max(np.abs(normalized))
        
        ratio = normalized_peak / original_peak
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)
    
    def test_lufs_normalization_option(self):
        """Test LUFS-based normalization for broadcast standard"""
        processed, _ = self.processor.process_audio(self.normal_audio, self.sr)
        
        # Apply LUFS normalization
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.normal_audio, 
            self.sr,
            method='lufs'
        )
        
        # Calculate LUFS for both
        original_lufs = self.normalizer.calculate_lufs(self.normal_audio, self.sr)
        normalized_lufs = self.normalizer.calculate_lufs(normalized, self.sr)
        
        # Should be within 1 dB
        self.assertAlmostEqual(original_lufs, normalized_lufs, delta=1.0)
    
    def test_no_clipping_occurs(self):
        """Test that normalization doesn't cause clipping"""
        # Process loud audio
        processed, _ = self.processor.process_audio(self.loud_audio, self.sr)
        
        # Apply normalization
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.loud_audio, 
            self.sr
        )
        
        # Check no samples exceed [-1, 1]
        self.assertTrue(np.all(np.abs(normalized) <= 1.0), 
                       "Normalized audio contains clipping")
        
        # Check that limiting was applied if needed
        if np.max(np.abs(processed)) * (np.max(np.abs(self.loud_audio)) / np.max(np.abs(processed))) > 1.0:
            # Soft limiting should have been applied
            self.assertTrue(self.normalizer.soft_limiting_applied)
    
    def test_quality_preservation(self):
        """Test that audio quality is preserved"""
        processed, _ = self.processor.process_audio(self.normal_audio, self.sr)
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.normal_audio, 
            self.sr
        )
        
        # Check spectral similarity
        orig_stft = np.abs(librosa.stft(self.normal_audio))
        norm_stft = np.abs(librosa.stft(normalized))
        
        # Normalize magnitudes for shape comparison
        orig_stft_norm = orig_stft / (np.max(orig_stft) + 1e-8)
        norm_stft_norm = norm_stft / (np.max(norm_stft) + 1e-8)
        
        # Spectral shapes should be similar
        correlation = np.corrcoef(orig_stft_norm.flatten(), norm_stft_norm.flatten())[0, 1]
        self.assertGreater(correlation, 0.9, "Spectral shape correlation too low")
    
    def test_noise_removal_preserved(self):
        """Test that noise removal from preprocessing is preserved"""
        # Process noisy audio
        processed, _ = self.processor.process_audio(self.noisy_audio, self.sr)
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.noisy_audio, 
            self.sr
        )
        
        # Calculate noise floor
        orig_noise_floor = np.percentile(np.abs(self.noisy_audio), 10)
        proc_noise_floor = np.percentile(np.abs(processed), 10)
        norm_noise_floor = np.percentile(np.abs(normalized), 10)
        
        # Normalized should maintain the noise reduction
        noise_reduction_ratio = proc_noise_floor / orig_noise_floor
        norm_reduction_ratio = norm_noise_floor / orig_noise_floor
        
        # The normalized audio should have similar noise reduction
        self.assertAlmostEqual(noise_reduction_ratio, norm_reduction_ratio, delta=0.1)
    
    def test_handles_silent_audio(self):
        """Test handling of silent or near-silent audio"""
        processed, _ = self.processor.process_audio(self.silent_audio, self.sr)
        
        # Should handle gracefully without division by zero
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.silent_audio, 
            self.sr
        )
        
        # Output should also be silent
        self.assertTrue(np.all(np.abs(normalized) < 1e-6))
    
    def test_handles_quiet_audio(self):
        """Test handling of very quiet audio"""
        processed, _ = self.processor.process_audio(self.quiet_audio, self.sr)
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.quiet_audio, 
            self.sr
        )
        
        # Should amplify to match original quiet level
        original_rms = np.sqrt(np.mean(self.quiet_audio**2))
        normalized_rms = np.sqrt(np.mean(normalized**2))
        
        if original_rms > 1e-6:  # Not silent
            ratio = normalized_rms / original_rms
            self.assertAlmostEqual(ratio, 1.0, delta=0.1)
    
    def test_batch_processing(self):
        """Test batch processing of multiple audio files"""
        audios = [self.normal_audio, self.quiet_audio, self.loud_audio]
        
        # Process all
        processed_audios = []
        for audio in audios:
            proc, _ = self.processor.process_audio(audio, self.sr)
            processed_audios.append(proc)
        
        # Normalize all
        normalized_audios = self.normalizer.normalize_batch(
            processed_audios, 
            audios, 
            self.sr
        )
        
        # Check each matches its original
        for i, (normalized, original) in enumerate(zip(normalized_audios, audios)):
            orig_rms = np.sqrt(np.mean(original**2))
            norm_rms = np.sqrt(np.mean(normalized**2))
            
            if orig_rms > 1e-6:
                ratio = norm_rms / orig_rms
                self.assertAlmostEqual(ratio, 1.0, delta=0.1, 
                                      msg=f"Batch item {i} loudness mismatch")
    
    def test_interrupted_audio_handling(self):
        """Test that interruption removal is preserved with proper loudness"""
        # Process audio with interruptions
        processed, num_interruptions = self.processor.process_audio(
            self.interrupted_audio, self.sr
        )
        
        # Should detect interruptions
        self.assertGreater(num_interruptions, 0, "No interruptions detected")
        
        # Normalize
        normalized = self.normalizer.normalize_loudness(
            processed, 
            self.interrupted_audio, 
            self.sr
        )
        
        # The overall loudness should match, but interruptions should still be suppressed
        # Check non-interruption parts
        clean_part_orig = self.interrupted_audio[:int(0.9 * self.sr)]
        clean_part_norm = normalized[:int(0.9 * self.sr)]
        
        orig_rms = np.sqrt(np.mean(clean_part_orig**2))
        norm_rms = np.sqrt(np.mean(clean_part_norm**2))
        
        ratio = norm_rms / orig_rms
        self.assertAlmostEqual(ratio, 1.0, delta=0.15)
    
    def test_preserves_dynamic_range(self):
        """Test that dynamic range is preserved"""
        # Create audio with varying dynamics
        t = np.linspace(0, self.duration, int(self.duration * self.sr))
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Slow envelope
        dynamic_audio = envelope * np.sin(2 * np.pi * 440 * t)
        
        processed, _ = self.processor.process_audio(dynamic_audio, self.sr)
        normalized = self.normalizer.normalize_loudness(
            processed, 
            dynamic_audio, 
            self.sr
        )
        
        # Calculate dynamic range (ratio of loud to quiet parts)
        orig_loud = np.percentile(np.abs(dynamic_audio), 90)
        orig_quiet = np.percentile(np.abs(dynamic_audio), 10)
        orig_dynamic_range = orig_loud / (orig_quiet + 1e-8)
        
        norm_loud = np.percentile(np.abs(normalized), 90)
        norm_quiet = np.percentile(np.abs(normalized), 10)
        norm_dynamic_range = norm_loud / (norm_quiet + 1e-8)
        
        # Dynamic range should be preserved (within 20%)
        range_ratio = norm_dynamic_range / orig_dynamic_range
        self.assertAlmostEqual(range_ratio, 1.0, delta=0.2)
    
    def test_configuration_options(self):
        """Test various configuration options"""
        processed, _ = self.processor.process_audio(self.normal_audio, self.sr)
        
        # Test with different configs
        configs = [
            {'method': 'rms', 'headroom_db': -1},
            {'method': 'peak', 'headroom_db': -3},
            {'method': 'lufs', 'target_lufs': -16},
            {'method': 'rms', 'soft_limit': True},
            {'method': 'rms', 'soft_limit': False}
        ]
        
        for config in configs:
            normalized = self.normalizer.normalize_loudness(
                processed, 
                self.normal_audio, 
                self.sr,
                **config
            )
            
            # Should not crash and produce valid audio
            self.assertEqual(len(normalized), len(self.normal_audio))
            self.assertTrue(np.all(np.isfinite(normalized)))
            self.assertTrue(np.all(np.abs(normalized) <= 1.0))


class TestIntegrationWithPipeline(unittest.TestCase):
    """Test integration with existing Pattern→MetricGAN+ pipeline"""
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline: Pattern→MetricGAN+→Loudness Normalization"""
        # Load a real sample if available
        sample_path = Path("sample_01_original.wav")
        if sample_path.exists():
            audio, sr = librosa.load(sample_path, sr=None)
        else:
            # Create synthetic test audio
            sr = 16000
            t = np.linspace(0, 2, 2 * sr)
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        # Full pipeline
        processor = PatternMetricGANProcessor()
        normalizer = LoudnessNormalizer()
        
        # Process
        processed, num_interruptions = processor.process_audio(audio, sr)
        
        # Normalize
        normalized = normalizer.normalize_loudness(processed, audio, sr)
        
        # Final audio should have same loudness as original
        orig_rms = np.sqrt(np.mean(audio**2))
        final_rms = np.sqrt(np.mean(normalized**2))
        
        if orig_rms > 1e-6:
            ratio = final_rms / orig_rms
            self.assertAlmostEqual(ratio, 1.0, delta=0.05)
        
        # Should preserve sample rate and duration
        self.assertEqual(len(normalized), len(audio))
    
    def test_preserves_metricgan_benefits(self):
        """Test that MetricGAN+ noise reduction benefits are preserved"""
        # Create noisy audio
        sr = 16000
        t = np.linspace(0, 2, 2 * sr)
        clean = 0.3 * np.sin(2 * np.pi * 440 * t)
        noise = 0.1 * np.random.randn(len(t))
        noisy = clean + noise
        
        processor = PatternMetricGANProcessor()
        normalizer = LoudnessNormalizer()
        
        # Process
        processed, _ = processor.process_audio(noisy, sr)
        normalized = normalizer.normalize_loudness(processed, noisy, sr)
        
        # Calculate SNR improvement
        def calculate_snr(clean_ref, noisy_sig):
            """Estimate SNR assuming clean_ref is the signal component"""
            noise_est = noisy_sig - clean_ref
            signal_power = np.mean(clean_ref**2)
            noise_power = np.mean(noise_est**2)
            return 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        # The normalized version should maintain the SNR improvement from MetricGAN+
        # (We can't calculate exact SNR without ground truth, but we can check noise floor)
        orig_noise_floor = np.percentile(np.abs(noisy), 5)
        proc_noise_floor = np.percentile(np.abs(processed), 5)
        norm_noise_floor = np.percentile(np.abs(normalized), 5)
        
        # Noise reduction should be maintained
        proc_reduction = proc_noise_floor / orig_noise_floor
        norm_reduction = norm_noise_floor / orig_noise_floor
        
        # Should be similar (within 20%)
        self.assertAlmostEqual(proc_reduction, norm_reduction, delta=0.2)


if __name__ == '__main__':
    unittest.main(verbosity=2)