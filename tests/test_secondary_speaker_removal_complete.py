"""
Comprehensive Test Suite for Secondary Speaker Removal
Tests complete removal of secondary speakers from audio
"""
import numpy as np
import pytest
import torch
from scipy import signal
from unittest.mock import patch, MagicMock
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.core import AudioEnhancer
from processors.audio_enhancement.complete_separation import CompleteSeparator
from processors.audio_enhancement.dominant_speaker_separation import DominantSpeakerSeparator
from utils.audio_metrics import calculate_snr, calculate_energy_db


class TestSecondaryRemovalComplete:
    """Test complete secondary speaker removal"""
    
    @pytest.fixture
    def sample_rate(self):
        return 16000
    
    @pytest.fixture
    def enhancer(self):
        """Create audio enhancer with ultra_aggressive settings"""
        enhancer = AudioEnhancer(
            use_gpu=torch.cuda.is_available(),
            enhancement_level='ultra_aggressive'
        )
        # Ensure secondary speaker removal is enabled
        enhancer.enable_secondary_speaker_removal = True
        return enhancer
    
    def create_multi_speaker_audio(self, sample_rate, duration=5.0):
        """Create synthetic audio with multiple speakers"""
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Primary speaker (speaks 60% of the time)
        primary = np.zeros(samples)
        primary_freq = 150  # Male voice fundamental
        
        # Speak from 0.5s to 3.5s (3 seconds total)
        primary_start = int(0.5 * sample_rate)
        primary_end = int(3.5 * sample_rate)
        primary[primary_start:primary_end] = 0.3 * np.sin(2 * np.pi * primary_freq * t[primary_start:primary_end])
        
        # Add harmonics for realism
        primary[primary_start:primary_end] += 0.15 * np.sin(2 * np.pi * primary_freq * 2 * t[primary_start:primary_end])
        primary[primary_start:primary_end] += 0.1 * np.sin(2 * np.pi * primary_freq * 3 * t[primary_start:primary_end])
        
        # Secondary speaker 1 (speaks at the beginning)
        secondary1 = np.zeros(samples)
        secondary1_freq = 250  # Female voice
        
        # Speak from 0s to 0.8s
        sec1_end = int(0.8 * sample_rate)
        secondary1[:sec1_end] = 0.4 * np.sin(2 * np.pi * secondary1_freq * t[:sec1_end])
        
        # Secondary speaker 2 (speaks at the end - critical test case)
        secondary2 = np.zeros(samples)
        secondary2_freq = 180  # Another male voice
        
        # Speak from 4.0s to 5.0s (last second)
        sec2_start = int(4.0 * sample_rate)
        secondary2[sec2_start:] = 0.5 * np.sin(2 * np.pi * secondary2_freq * t[sec2_start:])
        
        # Mix all speakers
        mixed = primary + secondary1 + secondary2
        
        # Add some noise for realism
        noise = 0.01 * np.random.randn(samples)
        mixed += noise
        
        # Normalize
        mixed = mixed / np.max(np.abs(mixed)) * 0.8
        
        return mixed.astype(np.float32), {
            'primary_start': primary_start,
            'primary_end': primary_end,
            'secondary1_end': sec1_end,
            'secondary2_start': sec2_start,
            'primary_duration': 3.0,  # seconds
            'secondary1_duration': 0.8,
            'secondary2_duration': 1.0
        }
    
    def test_complete_secondary_removal(self, enhancer, sample_rate):
        """Test that secondary speakers are completely removed"""
        # Create test audio
        mixed_audio, speaker_info = self.create_multi_speaker_audio(sample_rate)
        
        # Process with enhancer
        enhanced = enhancer.enhance(mixed_audio, sample_rate)
        
        # Test 1: Check energy at secondary speaker regions
        # Beginning (secondary speaker 1)
        beginning_segment = enhanced[:speaker_info['secondary1_end']]
        beginning_energy = calculate_energy_db(beginning_segment)
        
        # End (secondary speaker 2) - Most critical
        end_segment = enhanced[speaker_info['secondary2_start']:]
        end_energy = calculate_energy_db(end_segment)
        
        # Both should be very low (below -50dB)
        assert beginning_energy < -50, f"Secondary speaker 1 not removed: {beginning_energy:.1f}dB"
        assert end_energy < -50, f"Secondary speaker 2 at end not removed: {end_energy:.1f}dB"
        
        # Test 2: Check primary speaker is preserved
        primary_segment = enhanced[speaker_info['primary_start']:speaker_info['primary_end']]
        primary_energy = calculate_energy_db(primary_segment)
        
        # Primary should have significant energy
        assert primary_energy > -30, f"Primary speaker damaged: {primary_energy:.1f}dB"
    
    def test_dominant_speaker_identification(self, sample_rate):
        """Test that dominant speaker (most speaking time) is correctly identified"""
        separator = DominantSpeakerSeparator()
        
        # Create audio where first speaker is louder but speaks less
        mixed_audio, speaker_info = self.create_multi_speaker_audio(sample_rate)
        
        # The primary speaker speaks for 3 seconds (60% of 5 seconds)
        # Secondary speakers speak for 0.8s + 1.0s = 1.8s total
        
        # Analyze speaker activity
        activities = separator.analyze_speaker_activity(mixed_audio, sample_rate)
        
        # Should identify speaker with longest duration
        assert len(activities) >= 1
        dominant = activities[0]  # Should be sorted by duration
        
        # Dominant speaker should have ~3 seconds of activity
        assert dominant.total_duration > 2.5, f"Dominant speaker duration wrong: {dominant.total_duration}"
    
    def test_end_audio_silence(self, enhancer, sample_rate):
        """Test that the end of audio is properly silenced when secondary speaker present"""
        # Create audio with secondary speaker at end
        duration = 3.0
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Primary speaker (first 2 seconds)
        audio = np.zeros(samples)
        primary_end = int(2.0 * sample_rate)
        audio[:primary_end] = 0.3 * np.sin(2 * np.pi * 200 * t[:primary_end])
        
        # Secondary speaker (last 0.5 seconds) - should be removed
        secondary_start = int(2.5 * sample_rate)
        audio[secondary_start:] = 0.5 * np.sin(2 * np.pi * 300 * t[secondary_start:])
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio = audio.astype(np.float32)
        
        # Process
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Check last 500ms is silent
        last_500ms = enhanced[secondary_start:]
        max_amplitude = np.max(np.abs(last_500ms))
        
        # Should be nearly silent (below 0.01)
        assert max_amplitude < 0.01, f"End not silenced: max amplitude {max_amplitude}"
    
    def test_real_world_scenario(self, enhancer, sample_rate):
        """Test with realistic multi-speaker scenario"""
        # Create complex multi-speaker audio
        duration = 6.0
        samples = int(duration * sample_rate)
        
        # Speaker timings (simulating real conversation)
        # Speaker A (dominant): 0.5-2.5s, 3.0-4.5s (3.5s total)
        # Speaker B: 0-0.7s, 2.3-3.2s (1.6s total)  
        # Speaker C: 4.8-6.0s (1.2s total)
        
        audio = np.zeros(samples)
        t = np.linspace(0, duration, samples)
        
        # Speaker A (dominant - should be kept)
        a_seg1 = (int(0.5 * sample_rate), int(2.5 * sample_rate))
        a_seg2 = (int(3.0 * sample_rate), int(4.5 * sample_rate))
        
        for start, end in [a_seg1, a_seg2]:
            segment_t = t[start:end]
            audio[start:end] += 0.3 * np.sin(2 * np.pi * 180 * segment_t)
            audio[start:end] += 0.15 * np.sin(2 * np.pi * 360 * segment_t)
        
        # Speaker B (should be removed)
        b_seg1 = (0, int(0.7 * sample_rate))
        b_seg2 = (int(2.3 * sample_rate), int(3.2 * sample_rate))
        
        for start, end in [b_seg1, b_seg2]:
            segment_t = t[start:end]
            audio[start:end] += 0.4 * np.sin(2 * np.pi * 250 * segment_t)
        
        # Speaker C at end (should be removed)
        c_start = int(4.8 * sample_rate)
        audio[c_start:] += 0.5 * np.sin(2 * np.pi * 220 * t[c_start:])
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio = audio.astype(np.float32)
        
        # Process
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Verify Speaker B regions are silent
        b1_energy = calculate_energy_db(enhanced[b_seg1[0]:b_seg1[1]])
        b2_energy = calculate_energy_db(enhanced[b_seg2[0]:b_seg2[1]])
        
        assert b1_energy < -45, f"Speaker B segment 1 not removed: {b1_energy:.1f}dB"
        assert b2_energy < -45, f"Speaker B segment 2 not removed: {b2_energy:.1f}dB"
        
        # Verify Speaker C (end speaker) is removed
        c_energy = calculate_energy_db(enhanced[c_start:])
        assert c_energy < -45, f"End speaker C not removed: {c_energy:.1f}dB"
        
        # Verify Speaker A is preserved
        a1_energy = calculate_energy_db(enhanced[a_seg1[0]:a_seg1[1]])
        a2_energy = calculate_energy_db(enhanced[a_seg2[0]:a_seg2[1]])
        
        assert a1_energy > -30, f"Dominant speaker A segment 1 damaged: {a1_energy:.1f}dB"
        assert a2_energy > -30, f"Dominant speaker A segment 2 damaged: {a2_energy:.1f}dB"
    
    def test_separation_quality_metrics(self, enhancer, sample_rate):
        """Test separation quality using objective metrics"""
        # Create test audio
        mixed_audio, speaker_info = self.create_multi_speaker_audio(sample_rate)
        
        # Process
        enhanced = enhancer.enhance(mixed_audio, sample_rate)
        
        # Calculate cross-talk ratio (energy in secondary regions vs primary)
        secondary_energy = 0
        primary_energy = 0
        
        # Secondary regions
        sec1 = enhanced[:speaker_info['secondary1_end']]
        sec2 = enhanced[speaker_info['secondary2_start']:]
        secondary_energy = np.sum(sec1**2) + np.sum(sec2**2)
        
        # Primary region
        primary = enhanced[speaker_info['primary_start']:speaker_info['primary_end']]
        primary_energy = np.sum(primary**2)
        
        # Cross-talk ratio should be very low
        cross_talk_ratio = secondary_energy / (primary_energy + 1e-10)
        assert cross_talk_ratio < 0.001, f"High cross-talk: {cross_talk_ratio:.4f}"
    
    def test_no_speaker_mixing_back(self, enhancer, sample_rate):
        """Test that secondary speakers are not mixed back in post-processing"""
        # Create audio with clear secondary speaker at end
        duration = 2.0
        samples = int(duration * sample_rate)
        
        # Primary: 0-1.5s
        # Secondary: 1.5-2.0s (last 500ms)
        audio = np.zeros(samples)
        t = np.linspace(0, duration, samples)
        
        primary_end = int(1.5 * sample_rate)
        secondary_start = int(1.5 * sample_rate)
        
        # Create distinct speakers
        audio[:primary_end] = 0.3 * signal.chirp(t[:primary_end], 100, 1.5, 200)
        audio[secondary_start:] = 0.5 * signal.chirp(t[secondary_start:] - 1.5, 250, 0.5, 350)
        
        audio = audio.astype(np.float32)
        
        # Process
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # The last 500ms should be completely silent
        final_segment = enhanced[secondary_start:]
        
        # Check RMS energy
        rms = np.sqrt(np.mean(final_segment**2))
        assert rms < 0.001, f"Secondary speaker not removed, RMS: {rms:.4f}"
        
        # Check peak amplitude
        peak = np.max(np.abs(final_segment))
        assert peak < 0.01, f"Secondary speaker peaks remain: {peak:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])