#!/usr/bin/env python3
"""
Test a new approach: If secondary speaker is detected, process the entire audio.
"""

import unittest
import numpy as np
import os
import sys
import io
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.simple_secondary_removal import SimpleSecondaryRemoval


class FullAudioSecondaryRemoval:
    """
    New approach: If secondary speaker is detected anywhere, 
    apply aggressive filtering to the entire audio.
    """
    
    def __init__(self):
        self.detector = SimpleSecondaryRemoval(
            energy_threshold=0.5,
            min_silence_duration=0.05,
            suppression_db=-40,
            use_spectral_masking=True
        )
    
    def remove_secondary_speakers_full(self, audio, sample_rate=16000):
        """
        Remove secondary speakers from entire audio if any are detected.
        """
        # First, detect if there are any secondary speakers
        segments = self.detector.detect_speaker_changes(audio, sample_rate)
        
        if not segments:
            return audio
        
        # If secondary speakers detected, apply aggressive filtering to entire audio
        print(f"Secondary speakers detected in {len(segments)} segments")
        print("Applying full-audio secondary speaker removal...")
        
        # Use frequency-based separation on entire audio
        from scipy.signal import butter, sosfilt, stft, istft
        
        # Design a filter to preserve main speech frequencies (300-3000 Hz)
        # but aggressively reduce everything else
        nyquist = sample_rate / 2
        low_freq = 300 / nyquist
        high_freq = 3000 / nyquist
        
        # Bandpass filter
        sos = butter(6, [low_freq, high_freq], btype='band', output='sos')
        filtered = sosfilt(sos, audio)
        
        # Apply spectral subtraction to remove overlapping frequencies
        f, t, Zxx = stft(audio, fs=sample_rate, nperseg=512)
        
        # Find dominant frequency bins (likely main speaker)
        magnitude = np.abs(Zxx)
        mean_magnitude = np.mean(magnitude, axis=1, keepdims=True)
        
        # Create mask: keep only frequencies with consistent energy
        # (main speaker tends to be more consistent)
        std_magnitude = np.std(magnitude, axis=1, keepdims=True)
        consistency_ratio = std_magnitude / (mean_magnitude + 1e-10)
        
        # Mask: keep consistent frequencies, suppress variable ones
        mask = consistency_ratio < 0.5  # Lower ratio = more consistent
        
        # Apply mask more aggressively
        Zxx_masked = Zxx * mask * 0.8  # Reduce even "good" frequencies slightly
        
        # Reconstruct
        _, cleaned = istft(Zxx_masked, fs=sample_rate, nperseg=512)
        
        # Ensure same length
        if len(cleaned) > len(audio):
            cleaned = cleaned[:len(audio)]
        elif len(cleaned) < len(audio):
            cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))
        
        return cleaned


class TestFullAudioRemoval(unittest.TestCase):
    """Test the new full-audio removal approach."""
    
    def setUp(self):
        """Create test audio."""
        self.sample_rate = 16000
        self.duration = 3.0
        
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Main speaker (300Hz fundamental)
        self.main = np.sin(2 * np.pi * 300 * t) * 0.8
        # Add harmonics for realism
        self.main += np.sin(2 * np.pi * 600 * t) * 0.4
        self.main += np.sin(2 * np.pi * 900 * t) * 0.2
        
        # Secondary speaker (500Hz, from 1-2s)
        self.secondary = np.zeros_like(t)
        start_idx = int(1.0 * self.sample_rate)
        end_idx = int(2.0 * self.sample_rate)
        self.secondary[start_idx:end_idx] = np.sin(2 * np.pi * 500 * t[start_idx:end_idx]) * 0.7
        self.secondary[start_idx:end_idx] += np.sin(2 * np.pi * 1000 * t[start_idx:end_idx]) * 0.3
        
        self.combined = self.main + self.secondary
        self.combined = self.combined / np.max(np.abs(self.combined)) * 0.8
    
    def test_full_audio_removal(self):
        """Test removing secondary speaker from entire audio."""
        remover = FullAudioSecondaryRemoval()
        
        cleaned = remover.remove_secondary_speakers_full(self.combined, self.sample_rate)
        
        # Measure power at secondary speaker frequency
        def get_power_at_freq(audio, freq, sr):
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sr)
            idx = np.argmin(np.abs(freqs - freq))
            return np.abs(fft[idx])**2
        
        # Check 500Hz (secondary speaker)
        power_500_before = get_power_at_freq(self.combined, 500, self.sample_rate)
        power_500_after = get_power_at_freq(cleaned, 500, self.sample_rate)
        
        if power_500_before > 0:
            reduction_500 = 10 * np.log10(power_500_after / power_500_before)
            print(f"500Hz (secondary) reduction: {reduction_500:.1f} dB")
            
            self.assertLess(reduction_500, -15,
                          "Secondary frequency should be reduced by >15dB")
        
        # Check 300Hz (main speaker) is preserved
        power_300_before = get_power_at_freq(self.combined, 300, self.sample_rate)
        power_300_after = get_power_at_freq(cleaned, 300, self.sample_rate)
        
        if power_300_before > 0:
            reduction_300 = 10 * np.log10(power_300_after / power_300_before)
            print(f"300Hz (main) reduction: {reduction_300:.1f} dB")
            
            self.assertGreater(reduction_300, -6,
                          "Main frequency should not be reduced by >6dB")
    
    def test_on_real_s5(self):
        """Test on real S5 audio."""
        from processors.gigaspeech2 import GigaSpeech2Processor
        
        config = {
            "name": "GigaSpeech2",
            "source": "speechcolab/gigaspeech2",
            "cache_dir": "./cache",
            "streaming": True,
        }
        
        processor = GigaSpeech2Processor(config)
        samples = list(processor.process_streaming(sample_mode=True, sample_size=5))
        
        if len(samples) >= 5:
            s5 = samples[4]
            audio_data = s5.get('audio', {})
            
            if isinstance(audio_data, dict) and 'array' in audio_data:
                audio = audio_data['array']
                sr = audio_data.get('sampling_rate', 16000)
                
                # Apply full audio removal
                remover = FullAudioSecondaryRemoval()
                cleaned = remover.remove_secondary_speakers_full(audio, sr)
                
                # Measure overall reduction
                original_power = np.mean(audio**2)
                cleaned_power = np.mean(cleaned**2)
                
                if original_power > 0:
                    reduction_db = 10 * np.log10(cleaned_power / original_power)
                    print(f"\nS5 overall reduction: {reduction_db:.1f} dB")
                
                # Save for listening
                os.makedirs("test_audio_output", exist_ok=True)
                sf.write("test_audio_output/s5_full_removal.wav", cleaned, sr)
                print("Saved: test_audio_output/s5_full_removal.wav")


if __name__ == '__main__':
    unittest.main()