#!/usr/bin/env python3
"""
Final test to verify the complete secondary speaker removal solution.
"""

import unittest
import numpy as np
import os
import sys
import io
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.gigaspeech2 import GigaSpeech2Processor
from processors.audio_enhancement.core import AudioEnhancer


class TestCompleteSecondaryRemovalSolution(unittest.TestCase):
    """Test the complete solution with full audio removal."""
    
    def test_s5_with_new_solution(self):
        """Test S5 with the new full audio removal solution."""
        print("\n=== Testing S5 with Complete Solution ===")
        
        # Get S5
        config = {
            "name": "GigaSpeech2",
            "source": "speechcolab/gigaspeech2",
            "cache_dir": "./cache",
            "streaming": True,
        }
        
        processor = GigaSpeech2Processor(config)
        samples = list(processor.process_streaming(sample_mode=True, sample_size=5))
        
        self.assertGreaterEqual(len(samples), 5, "Need at least 5 samples")
        
        s5 = samples[4]
        audio_data = s5.get('audio', {})
        self.assertIsInstance(audio_data, dict)
        self.assertIn('array', audio_data)
        
        audio = audio_data['array']
        sr = audio_data.get('sampling_rate', 16000)
        
        print(f"S5 audio length: {len(audio)/sr:.2f}s")
        
        # Test with aggressive enhancement (should use full audio removal)
        enhancer = AudioEnhancer(
            use_gpu=False,
            fallback_to_cpu=True,
            enhancement_level='aggressive'
        )
        
        # Apply enhancement
        enhanced, metadata = enhancer.enhance(
            audio, sr,
            noise_level='aggressive',
            return_metadata=True
        )
        
        print(f"\nEnhancement metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Check that secondary speaker was detected and removed
        self.assertTrue(metadata.get('secondary_speaker_detected', False),
                       "Should detect secondary speaker")
        
        # Measure reduction
        original_power = np.mean(audio**2)
        enhanced_power = np.mean(enhanced**2)
        
        if original_power > 0:
            reduction_db = 10 * np.log10(enhanced_power / original_power)
            print(f"\nOverall power reduction: {reduction_db:.1f} dB")
            
            # Should have significant reduction
            self.assertLess(reduction_db, -5,
                          f"Should have >5dB reduction, got {reduction_db:.1f}dB")
        
        # Save results
        os.makedirs("test_audio_output", exist_ok=True)
        sf.write("test_audio_output/s5_complete_solution.wav", enhanced, sr)
        print("Saved: test_audio_output/s5_complete_solution.wav")
        
        # Also save original for comparison
        sf.write("test_audio_output/s5_original_comparison.wav", audio, sr)
        print("Saved: test_audio_output/s5_original_comparison.wav")
    
    def test_synthetic_audio_removal(self):
        """Test with synthetic audio to verify frequency-specific removal."""
        print("\n=== Testing Synthetic Audio ===")
        
        # Create test audio
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Main speaker (300Hz + harmonics)
        main = np.sin(2 * np.pi * 300 * t) * 0.6
        main += np.sin(2 * np.pi * 600 * t) * 0.3
        main += np.sin(2 * np.pi * 900 * t) * 0.15
        
        # Secondary speaker (500Hz + 1000Hz, from 1-2s)
        secondary = np.zeros_like(t)
        start_idx = int(1.0 * sr)
        end_idx = int(2.0 * sr)
        secondary[start_idx:end_idx] = np.sin(2 * np.pi * 500 * t[start_idx:end_idx]) * 0.5
        secondary[start_idx:end_idx] += np.sin(2 * np.pi * 1000 * t[start_idx:end_idx]) * 0.25
        
        # Combine
        combined = main + secondary
        combined = combined / np.max(np.abs(combined)) * 0.8
        
        # Apply enhancement
        enhancer = AudioEnhancer(
            use_gpu=False,
            fallback_to_cpu=True,
            enhancement_level='aggressive'
        )
        
        enhanced, metadata = enhancer.enhance(
            combined, sr,
            noise_level='aggressive',
            return_metadata=True
        )
        
        print(f"Secondary speaker detected: {metadata.get('secondary_speaker_detected')}")
        
        # Analyze frequency content
        def analyze_freq_power(audio, freq, sample_rate):
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
            idx = np.argmin(np.abs(freqs - freq))
            window = slice(max(0, idx-2), min(len(fft), idx+3))
            return np.mean(np.abs(fft[window])**2)
        
        # Check main frequencies (should be preserved)
        for freq in [300, 600, 900]:
            power_before = analyze_freq_power(combined, freq, sr)
            power_after = analyze_freq_power(enhanced, freq, sr)
            if power_before > 0:
                reduction = 10 * np.log10(power_after / power_before)
                print(f"{freq}Hz (main): {reduction:+.1f} dB")
                self.assertGreater(reduction, -10,
                                 f"Main frequency {freq}Hz should not be reduced by >10dB")
        
        # Check secondary frequencies (should be reduced)
        for freq in [500, 1000]:
            power_before = analyze_freq_power(combined, freq, sr)
            power_after = analyze_freq_power(enhanced, freq, sr)
            if power_before > 0:
                reduction = 10 * np.log10(power_after / power_before)
                print(f"{freq}Hz (secondary): {reduction:+.1f} dB")
                self.assertLess(reduction, -10,
                              f"Secondary frequency {freq}Hz should be reduced by >10dB")
    
    def test_processing_preserves_audio_quality(self):
        """Test that processing preserves overall audio quality."""
        print("\n=== Testing Audio Quality Preservation ===")
        
        # Create clean speech-like audio
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Simulate speech with formants
        speech = np.zeros_like(t)
        # F0 (fundamental)
        speech += np.sin(2 * np.pi * 150 * t) * 0.3
        # F1 (first formant)
        speech += np.sin(2 * np.pi * 700 * t) * 0.4
        # F2 (second formant) 
        speech += np.sin(2 * np.pi * 1200 * t) * 0.3
        # F3 (third formant)
        speech += np.sin(2 * np.pi * 2500 * t) * 0.2
        
        # Add slight modulation for realism
        speech *= (1 + 0.1 * np.sin(2 * np.pi * 5 * t))
        
        # Normalize
        speech = speech / np.max(np.abs(speech)) * 0.7
        
        # Process with enhancer (should not damage clean audio much)
        enhancer = AudioEnhancer(
            use_gpu=False,
            fallback_to_cpu=True,
            enhancement_level='aggressive'
        )
        
        enhanced, metadata = enhancer.enhance(
            speech, sr,
            noise_level='aggressive',
            return_metadata=True
        )
        
        # Check if it was processed
        if metadata.get('secondary_speaker_detected', False):
            print("Secondary speaker detected in clean audio (false positive)")
        else:
            print("No secondary speaker detected in clean audio (correct)")
        
        # Measure distortion
        original_power = np.mean(speech**2)
        enhanced_power = np.mean(enhanced**2)
        
        if original_power > 0:
            power_change_db = 10 * np.log10(enhanced_power / original_power)
            print(f"Power change: {power_change_db:+.1f} dB")
            
            # Should not change clean audio too much
            self.assertGreater(power_change_db, -6,
                            "Clean audio should not be reduced by >6dB")
            self.assertLess(power_change_db, 3,
                         "Clean audio should not be amplified by >3dB")


if __name__ == '__main__':
    unittest.main()