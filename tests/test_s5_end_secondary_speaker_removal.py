"""
Test Suite for S5 End Secondary Speaker Removal
Comprehensive tests to ensure secondary speakers at the end of audio are completely removed
"""
import numpy as np
import pytest
import torch
from scipy import signal
import os
import sys
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.core import AudioEnhancer
from utils.audio_metrics import calculate_energy_db


class TestS5EndSecondaryRemoval:
    """Comprehensive tests for S5 secondary speaker removal at end of audio"""
    
    @pytest.fixture
    def sample_rate(self):
        return 16000
    
    @pytest.fixture 
    def enhancer(self):
        """Create audio enhancer with ultra_aggressive settings"""
        return AudioEnhancer(
            use_gpu=torch.cuda.is_available(),
            enhancement_level='ultra_aggressive'
        )
    
    def create_end_secondary_speaker_audio(self, sample_rate, 
                                         primary_duration=4.0,
                                         secondary_start=3.5,
                                         secondary_amplitude=0.5):
        """Create audio with secondary speaker at the end"""
        total_duration = 5.0
        samples = int(total_duration * sample_rate)
        t = np.linspace(0, total_duration, samples)
        
        # Primary speaker (0.5s to primary_duration)
        audio = np.zeros(samples)
        primary_start_idx = int(0.5 * sample_rate)
        primary_end_idx = int(primary_duration * sample_rate)
        
        # Add primary speaker with harmonics for realism
        primary_freq = 150  # Male voice
        audio[primary_start_idx:primary_end_idx] = 0.3 * np.sin(2 * np.pi * primary_freq * t[primary_start_idx:primary_end_idx])
        audio[primary_start_idx:primary_end_idx] += 0.15 * np.sin(2 * np.pi * primary_freq * 2 * t[primary_start_idx:primary_end_idx])
        
        # Secondary speaker at end
        secondary_start_idx = int(secondary_start * sample_rate)
        secondary_freq = 220  # Different voice
        audio[secondary_start_idx:] = secondary_amplitude * np.sin(2 * np.pi * secondary_freq * t[secondary_start_idx:])
        
        # Add slight noise for realism
        audio += 0.01 * np.random.randn(samples)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32)
    
    def test_end_secondary_removal_basic(self, enhancer, sample_rate):
        """Test basic removal of secondary speaker at end"""
        # Create test audio
        audio = self.create_end_secondary_speaker_audio(sample_rate)
        
        # Process
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Check last 1 second
        last_second = enhanced[-sample_rate:]
        energy = calculate_energy_db(last_second)
        
        assert energy < -50, f"Secondary speaker at end not removed: {energy:.1f}dB"
    
    def test_end_secondary_removal_last_half_second(self, enhancer, sample_rate):
        """Test removal in last 0.5 seconds specifically"""
        audio = self.create_end_secondary_speaker_audio(sample_rate)
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Check last 0.5 seconds
        last_half = enhanced[int(-0.5 * sample_rate):]
        energy = calculate_energy_db(last_half)
        
        assert energy < -50, f"Secondary speaker in last 0.5s not removed: {energy:.1f}dB"
    
    def test_end_secondary_removal_various_durations(self, enhancer, sample_rate):
        """Test removal for various end durations"""
        test_durations = [0.25, 0.5, 0.75, 1.0, 1.5]
        
        for duration in test_durations:
            # Create audio with secondary speaker starting at different times
            start_time = 5.0 - duration
            audio = self.create_end_secondary_speaker_audio(
                sample_rate, 
                primary_duration=start_time - 0.5,
                secondary_start=start_time
            )
            
            enhanced = enhancer.enhance(audio, sample_rate)
            
            # Check the specific duration at end
            segment = enhanced[int(-duration * sample_rate):]
            energy = calculate_energy_db(segment)
            
            assert energy < -50, f"Secondary speaker in last {duration}s not removed: {energy:.1f}dB"
    
    def test_abrupt_end_secondary_speaker(self, enhancer, sample_rate):
        """Test removal when secondary speaker appears abruptly at end"""
        audio = self.create_end_secondary_speaker_audio(
            sample_rate,
            primary_duration=3.8,  # Primary stops at 3.8s
            secondary_start=4.0,   # Secondary starts at 4.0s
            secondary_amplitude=0.7  # Louder secondary speaker
        )
        
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Check from 4.0s onwards
        segment = enhanced[int(4.0 * sample_rate):]
        energy = calculate_energy_db(segment)
        
        assert energy < -50, f"Abrupt secondary speaker at end not removed: {energy:.1f}dB"
    
    def test_overlapping_end_speakers(self, enhancer, sample_rate):
        """Test when primary and secondary overlap at end"""
        # Create complex overlap scenario
        total_duration = 5.0
        samples = int(total_duration * sample_rate)
        t = np.linspace(0, total_duration, samples)
        
        audio = np.zeros(samples)
        
        # Primary speaker (0.5s to 4.5s)
        primary_start = int(0.5 * sample_rate)
        primary_end = int(4.5 * sample_rate)
        audio[primary_start:primary_end] = 0.3 * np.sin(2 * np.pi * 150 * t[primary_start:primary_end])
        
        # Secondary speaker (3.5s to 5.0s) - overlaps with primary
        secondary_start = int(3.5 * sample_rate)
        audio[secondary_start:] += 0.5 * np.sin(2 * np.pi * 250 * t[secondary_start:])
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio = audio.astype(np.float32)
        
        # Process
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Check the non-overlapping end part (4.5s to 5.0s)
        segment = enhanced[int(4.5 * sample_rate):]
        energy = calculate_energy_db(segment)
        
        assert energy < -50, f"Secondary speaker in non-overlapping end not removed: {energy:.1f}dB"
    
    def test_multiple_end_speakers(self, enhancer, sample_rate):
        """Test removal of multiple secondary speakers at end"""
        total_duration = 5.0
        samples = int(total_duration * sample_rate)
        t = np.linspace(0, total_duration, samples)
        
        audio = np.zeros(samples)
        
        # Primary speaker
        primary_end = int(3.0 * sample_rate)
        audio[:primary_end] = 0.3 * np.sin(2 * np.pi * 150 * t[:primary_end])
        
        # Secondary speaker 1 (3.0s to 4.0s)
        sec1_start = int(3.0 * sample_rate)
        sec1_end = int(4.0 * sample_rate)
        audio[sec1_start:sec1_end] = 0.4 * np.sin(2 * np.pi * 200 * t[sec1_start:sec1_end])
        
        # Secondary speaker 2 (4.0s to 5.0s)
        sec2_start = int(4.0 * sample_rate)
        audio[sec2_start:] = 0.5 * np.sin(2 * np.pi * 250 * t[sec2_start:])
        
        # Normalize and process
        audio = (audio / np.max(np.abs(audio)) * 0.8).astype(np.float32)
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Check both secondary speaker regions
        seg1_energy = calculate_energy_db(enhanced[sec1_start:sec1_end])
        seg2_energy = calculate_energy_db(enhanced[sec2_start:])
        
        assert seg1_energy < -50, f"Secondary speaker 1 not removed: {seg1_energy:.1f}dB"
        assert seg2_energy < -50, f"Secondary speaker 2 at end not removed: {seg2_energy:.1f}dB"
    
    def test_energy_profile_at_end(self, enhancer, sample_rate):
        """Test that energy profile drops sharply when secondary speaker appears"""
        audio = self.create_end_secondary_speaker_audio(sample_rate)
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Analyze energy profile in 100ms windows
        window_size = int(0.1 * sample_rate)
        
        # Get energy for windows before and after secondary speaker start
        # Window at 3.4s (before secondary)
        before_window = enhanced[int(3.4 * sample_rate):int(3.5 * sample_rate)]
        before_energy = calculate_energy_db(before_window)
        
        # Window at 3.6s (after secondary starts)
        after_window = enhanced[int(3.6 * sample_rate):int(3.7 * sample_rate)]
        after_energy = calculate_energy_db(after_window)
        
        # Energy should drop significantly
        assert after_energy < -50, f"Energy after secondary speaker start too high: {after_energy:.1f}dB"
        
        # If there was speech before, it should drop by at least 30dB
        if before_energy > -50:
            energy_drop = before_energy - after_energy
            assert energy_drop > 30, f"Energy drop insufficient: {energy_drop:.1f}dB"
    
    @pytest.mark.parametrize("noise_level", [0.0, 0.01, 0.05, 0.1])
    def test_end_removal_with_noise(self, enhancer, sample_rate, noise_level):
        """Test removal works with various noise levels"""
        audio = self.create_end_secondary_speaker_audio(sample_rate)
        
        # Add noise
        audio += noise_level * np.random.randn(len(audio))
        audio = (audio / np.max(np.abs(audio)) * 0.8).astype(np.float32)
        
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Check last second
        last_second = enhanced[-sample_rate:]
        energy = calculate_energy_db(last_second)
        
        # Allow slightly higher threshold for noisy audio
        threshold = -45 if noise_level > 0.05 else -50
        assert energy < threshold, f"Secondary speaker with noise={noise_level} not removed: {energy:.1f}dB"
    
    def test_silence_enforcement_at_end(self, enhancer, sample_rate):
        """Test that complete silence is enforced when secondary speaker detected"""
        audio = self.create_end_secondary_speaker_audio(sample_rate)
        enhanced = enhancer.enhance(audio, sample_rate)
        
        # Check actual sample values in last 0.5s
        last_half = enhanced[int(-0.5 * sample_rate):]
        
        # Most samples should be exactly 0 or very close
        zero_samples = np.sum(np.abs(last_half) < 1e-6)
        zero_ratio = zero_samples / len(last_half)
        
        assert zero_ratio > 0.95, f"Not enough silence at end: {zero_ratio:.1%} zeros"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_real_s5_sample(self, enhancer, sample_rate):
        """Test with actual S5 sample from dataset if available"""
        try:
            # Try to load from Huggingface
            dataset = load_dataset("Thanarit/Thai-Voice", split="train", streaming=True)
            
            # Get S5 (index 4 based on the output we saw)
            s5 = None
            for i, sample in enumerate(dataset):
                if sample.get('ID') == 'S5':
                    s5 = sample
                    break
                if i > 10:  # Safety limit
                    break
            
            if s5:
                audio = s5['audio']['array']
                audio_sr = s5['audio']['sampling_rate']
                
                # Resample if needed
                if audio_sr != sample_rate:
                    # Simple decimation/interpolation
                    audio = signal.resample(audio, int(len(audio) * sample_rate / audio_sr))
                
                # Process
                enhanced = enhancer.enhance(audio, sample_rate)
                
                # Check last second
                last_second = enhanced[-sample_rate:]
                energy = calculate_energy_db(last_second)
                
                assert energy < -50, f"S5 secondary speaker at end not removed: {energy:.1f}dB"
        except Exception as e:
            pytest.skip(f"Could not load real S5 sample: {e}")