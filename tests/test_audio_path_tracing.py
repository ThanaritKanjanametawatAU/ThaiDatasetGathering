#!/usr/bin/env python3
"""
Test to trace the audio path through the pipeline and find where enhancement might be weakened.
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
from processors.audio_enhancement.simple_secondary_removal import SimpleSecondaryRemoval
from processors.audio_enhancement.speaker_separation import SpeakerSeparator, SeparationConfig


class TestAudioPathTracing(unittest.TestCase):
    """Trace audio through the enhancement pipeline to find issues."""
    
    def setUp(self):
        """Set up test audio."""
        self.sample_rate = 16000
        self.duration = 3.0
        
        # Create clear test audio with distinct frequencies
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Primary speaker: 200Hz
        self.primary = np.sin(2 * np.pi * 200 * t) * 0.8
        
        # Secondary speaker: 600Hz (more distinct), from 1-2 seconds
        self.secondary = np.zeros_like(t)
        start_idx = int(1.0 * self.sample_rate)
        end_idx = int(2.0 * self.sample_rate)
        self.secondary[start_idx:end_idx] = np.sin(2 * np.pi * 600 * t[start_idx:end_idx]) * 0.7
        
        # Combined audio
        self.test_audio = self.primary + self.secondary
        self.test_audio = self.test_audio / np.max(np.abs(self.test_audio)) * 0.8
    
    def _measure_frequency_power(self, audio, freq, sample_rate):
        """Measure power at a specific frequency."""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        idx = np.argmin(np.abs(freqs - freq))
        power = np.abs(fft[idx])**2
        return power
    
    def test_simple_secondary_removal_directly(self):
        """Test SimpleSecondaryRemoval in isolation."""
        print("\n=== Testing SimpleSecondaryRemoval Directly ===")
        
        remover = SimpleSecondaryRemoval(
            energy_threshold=0.5,
            min_silence_duration=0.05,
            suppression_db=-60,  # Strong suppression
            use_spectral_masking=True
        )
        
        # Apply removal
        cleaned = remover.remove_secondary_speakers(self.test_audio, self.sample_rate)
        
        # Measure power at 600Hz (secondary speaker)
        power_600_before = self._measure_frequency_power(self.test_audio, 600, self.sample_rate)
        power_600_after = self._measure_frequency_power(cleaned, 600, self.sample_rate)
        
        if power_600_before > 0:
            reduction_db = 10 * np.log10(power_600_after / power_600_before)
            print(f"SimpleSecondaryRemoval: 600Hz reduction = {reduction_db:.1f} dB")
            
            self.assertLess(reduction_db, -20,
                          f"SimpleSecondaryRemoval should reduce by >20dB, got {reduction_db:.1f}dB")
    
    def test_speaker_separator_directly(self):
        """Test SpeakerSeparator in isolation."""
        print("\n=== Testing SpeakerSeparator Directly ===")
        
        config = SeparationConfig(
            suppression_strength=0.95,
            confidence_threshold=0.3,
            detection_methods=["embedding", "vad", "energy", "spectral"]
        )
        
        separator = SpeakerSeparator(config)
        result = separator.separate_speakers(self.test_audio, self.sample_rate)
        
        print(f"Detections: {len(result['detections'])}")
        for d in result['detections']:
            print(f"  {d.start_time:.2f}s - {d.end_time:.2f}s (conf: {d.confidence:.2f})")
        
        # Measure reduction
        power_600_before = self._measure_frequency_power(self.test_audio, 600, self.sample_rate)
        power_600_after = self._measure_frequency_power(result['audio'], 600, self.sample_rate)
        
        if power_600_before > 0:
            reduction_db = 10 * np.log10(power_600_after / power_600_before)
            print(f"SpeakerSeparator: 600Hz reduction = {reduction_db:.1f} dB")
            
            self.assertLess(reduction_db, -20,
                          f"SpeakerSeparator should reduce by >20dB, got {reduction_db:.1f}dB")
    
    def test_audio_enhancer_full_pipeline(self):
        """Test the full AudioEnhancer pipeline."""
        print("\n=== Testing Full AudioEnhancer Pipeline ===")
        
        enhancer = AudioEnhancer(
            use_gpu=False,
            enhancement_level='aggressive'
        )
        
        # Apply enhancement
        enhanced, metadata = enhancer.enhance(
            self.test_audio,
            self.sample_rate,
            noise_level='aggressive',
            return_metadata=True
        )
        
        print(f"Metadata: {metadata}")
        
        # Measure reduction
        power_600_before = self._measure_frequency_power(self.test_audio, 600, self.sample_rate)
        power_600_after = self._measure_frequency_power(enhanced, 600, self.sample_rate)
        
        if power_600_before > 0:
            reduction_db = 10 * np.log10(power_600_after / power_600_before)
            print(f"Full pipeline: 600Hz reduction = {reduction_db:.1f} dB")
            
            # Also check overall power
            overall_before = np.mean(self.test_audio**2)
            overall_after = np.mean(enhanced**2)
            overall_reduction_db = 10 * np.log10(overall_after / overall_before)
            print(f"Overall power reduction = {overall_reduction_db:.1f} dB")
            
            self.assertLess(reduction_db, -20,
                          f"Full pipeline should reduce by >20dB, got {reduction_db:.1f}dB")
    
    def test_check_if_simple_remover_is_called(self):
        """Check if SimpleSecondaryRemoval is actually called in aggressive mode."""
        print("\n=== Testing if SimpleSecondaryRemoval is Called ===")
        
        enhancer = AudioEnhancer(
            use_gpu=False,
            enhancement_level='aggressive'
        )
        
        # Monkey patch to track calls
        original_remove = enhancer.simple_remover.remove_secondary_speakers
        simple_remover_called = False
        
        def tracked_remove(audio, sr):
            nonlocal simple_remover_called
            simple_remover_called = True
            print("SimpleSecondaryRemoval.remove_secondary_speakers was called!")
            return original_remove(audio, sr)
        
        enhancer.simple_remover.remove_secondary_speakers = tracked_remove
        
        # Apply enhancement
        enhanced, metadata = enhancer.enhance(
            self.test_audio,
            self.sample_rate,
            noise_level='aggressive',
            return_metadata=True
        )
        
        self.assertTrue(simple_remover_called,
                       "SimpleSecondaryRemoval should be called for aggressive mode")
    
    def test_trace_audio_modifications(self):
        """Trace audio through each step of enhancement."""
        print("\n=== Tracing Audio Through Enhancement Steps ===")
        
        enhancer = AudioEnhancer(
            use_gpu=False,
            enhancement_level='aggressive'
        )
        
        # Patch various methods to trace audio
        audio_trace = []
        
        # Patch speaker separator
        original_separate = enhancer.speaker_separator.separate_speakers
        def traced_separate(audio, sr):
            result = original_separate(audio, sr)
            power_600 = self._measure_frequency_power(result['audio'], 600, sr)
            audio_trace.append(('after_speaker_separator', power_600))
            return result
        enhancer.speaker_separator.separate_speakers = traced_separate
        
        # Patch simple remover
        original_simple = enhancer.simple_remover.remove_secondary_speakers
        def traced_simple(audio, sr):
            power_before = self._measure_frequency_power(audio, 600, sr)
            audio_trace.append(('before_simple_remover', power_before))
            
            result = original_simple(audio, sr)
            
            power_after = self._measure_frequency_power(result, 600, sr)
            audio_trace.append(('after_simple_remover', power_after))
            return result
        enhancer.simple_remover.remove_secondary_speakers = traced_simple
        
        # Initial power
        initial_power = self._measure_frequency_power(self.test_audio, 600, self.sample_rate)
        audio_trace.append(('initial', initial_power))
        
        # Apply enhancement
        enhanced, metadata = enhancer.enhance(
            self.test_audio,
            self.sample_rate,
            noise_level='aggressive',
            return_metadata=True
        )
        
        # Final power
        final_power = self._measure_frequency_power(enhanced, 600, self.sample_rate)
        audio_trace.append(('final', final_power))
        
        # Print trace
        print("\nAudio power trace at 600Hz:")
        for i, (stage, power) in enumerate(audio_trace):
            if i > 0 and audio_trace[i-1][1] > 0:
                reduction_db = 10 * np.log10(power / audio_trace[i-1][1])
                print(f"  {stage}: {power:.6f} ({reduction_db:+.1f} dB)")
            else:
                print(f"  {stage}: {power:.6f}")


if __name__ == '__main__':
    unittest.main()