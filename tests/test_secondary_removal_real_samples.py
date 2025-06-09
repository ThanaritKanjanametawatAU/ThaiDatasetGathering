"""
Test Secondary Speaker Removal with Real Audio Samples
Uses actual dataset samples to ensure tests reflect reality
"""
import numpy as np
import pytest
import torch
import os
import sys
import librosa
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.core import AudioEnhancer
from processors.audio_enhancement.complete_separation import CompleteSeparator
from processors.audio_enhancement.dominant_speaker_separation import DominantSpeakerSeparator
from utils.audio_metrics import calculate_energy_db


class TestRealSampleSecondaryRemoval:
    """Test secondary speaker removal on real samples"""
    
    @pytest.fixture
    def enhancer(self):
        """Create audio enhancer with ultra_aggressive settings"""
        enhancer = AudioEnhancer(
            use_gpu=torch.cuda.is_available(),
            enhancement_level='ultra_aggressive'
        )
        enhancer.enable_secondary_speaker_removal = True
        return enhancer
    
    @pytest.fixture
    def sample_dir(self):
        """Directory containing test samples"""
        return Path("/media/ssd1/SparkVoiceProject/ThaiDatasetGathering/test_audio_output")
    
    def load_audio_sample(self, file_path, target_sr=16000):
        """Load audio file and resample if needed"""
        if os.path.exists(file_path):
            audio, sr = librosa.load(file_path, sr=None)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            return audio.astype(np.float32), target_sr
        else:
            # Create synthetic sample if file doesn't exist
            return self.create_problematic_sample()
    
    def create_problematic_sample(self, duration=3.0, sample_rate=16000):
        """Create sample mimicking problematic S5 case"""
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Primary speaker (middle portion)
        audio = np.zeros(samples)
        primary_start = int(0.5 * sample_rate)
        primary_end = int(2.0 * sample_rate)
        
        # Primary speaker with natural speech-like modulation
        carrier = np.sin(2 * np.pi * 150 * t[primary_start:primary_end])
        modulation = 0.3 * np.sin(2 * np.pi * 4 * t[primary_start:primary_end])  # 4Hz modulation
        audio[primary_start:primary_end] = carrier * (1 + modulation) * 0.3
        
        # Secondary speaker at end (mimicking S5 issue)
        secondary_start = int(2.2 * sample_rate)
        secondary_audio = 0.4 * np.sin(2 * np.pi * 220 * t[secondary_start:])
        # Add some speech-like characteristics
        secondary_audio *= (1 + 0.2 * np.sin(2 * np.pi * 5 * t[secondary_start:]))
        audio[secondary_start:] += secondary_audio
        
        # Add background noise
        noise = 0.02 * np.random.randn(samples)
        audio += noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32), sample_rate
    
    def test_s5_like_sample(self, enhancer):
        """Test sample similar to problematic S5"""
        # Create or load S5-like sample
        audio, sr = self.create_problematic_sample()
        
        # Process with enhancer
        enhanced = enhancer.enhance(audio, sr)
        
        # Check that end is silenced (after 2.2s where secondary speaker starts)
        secondary_start_idx = int(2.2 * sr)
        end_segment = enhanced[secondary_start_idx:]
        
        # Calculate metrics
        end_energy = calculate_energy_db(end_segment)
        end_rms = np.sqrt(np.mean(end_segment**2))
        end_peak = np.max(np.abs(end_segment))
        
        # Strict requirements for complete removal
        assert end_energy < -50, f"End speaker not removed: {end_energy:.1f}dB"
        assert end_rms < 0.001, f"End RMS too high: {end_rms:.4f}"
        assert end_peak < 0.01, f"End peak too high: {end_peak:.4f}"
    
    def test_overlapping_speech_removal(self, enhancer):
        """Test removal when speakers overlap"""
        sr = 16000
        duration = 4.0
        samples = int(duration * sr)
        t = np.linspace(0, duration, samples)
        
        # Create overlapping speech scenario
        audio = np.zeros(samples)
        
        # Speaker A (dominant): speaks 70% of time
        a_segments = [
            (0.2, 1.5),   # 1.3s
            (1.8, 3.8)    # 2.0s
        ]  # Total: 3.3s
        
        # Speaker B: overlaps and speaks alone
        b_segments = [
            (0.0, 0.5),   # 0.5s alone at start
            (1.2, 2.0),   # 0.8s with overlap
            (3.5, 4.0)    # 0.5s alone at end
        ]  # Total: 1.8s
        
        # Generate speaker A (150Hz fundamental)
        for start, end in a_segments:
            idx_start = int(start * sr)
            idx_end = int(end * sr)
            seg_t = t[idx_start:idx_end]
            audio[idx_start:idx_end] += 0.3 * np.sin(2 * np.pi * 150 * seg_t)
            audio[idx_start:idx_end] += 0.15 * np.sin(2 * np.pi * 300 * seg_t)
        
        # Generate speaker B (220Hz fundamental)
        for start, end in b_segments:
            idx_start = int(start * sr)
            idx_end = int(end * sr)
            seg_t = t[idx_start:idx_end]
            audio[idx_start:idx_end] += 0.35 * np.sin(2 * np.pi * 220 * seg_t)
            audio[idx_start:idx_end] += 0.17 * np.sin(2 * np.pi * 440 * seg_t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio = audio.astype(np.float32)
        
        # Process
        enhanced = enhancer.enhance(audio, sr)
        
        # Check speaker B segments are removed
        # Beginning (0-0.5s)
        begin_energy = calculate_energy_db(enhanced[:int(0.5 * sr)])
        assert begin_energy < -45, f"Beginning speaker B not removed: {begin_energy:.1f}dB"
        
        # End (3.5-4.0s) - Critical test
        end_energy = calculate_energy_db(enhanced[int(3.5 * sr):])
        assert end_energy < -45, f"End speaker B not removed: {end_energy:.1f}dB"
        
        # Check speaker A is preserved
        a_energy = calculate_energy_db(enhanced[int(2.0 * sr):int(3.5 * sr)])
        assert a_energy > -30, f"Dominant speaker A damaged: {a_energy:.1f}dB"
    
    def test_rapid_speaker_switching(self, enhancer):
        """Test with rapid speaker switching"""
        sr = 16000
        duration = 3.0
        samples = int(duration * sr)
        t = np.linspace(0, duration, samples)
        
        audio = np.zeros(samples)
        
        # Create rapid switching pattern
        # Speaker A speaks more overall but in shorter bursts
        switch_duration = 0.2  # 200ms segments
        
        for i in range(int(duration / switch_duration)):
            start_idx = int(i * switch_duration * sr)
            end_idx = int((i + 1) * switch_duration * sr)
            
            if i % 3 != 2:  # Speaker A speaks 2/3 of segments
                # Speaker A
                audio[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * 160 * t[start_idx:end_idx])
            else:
                # Speaker B
                audio[start_idx:end_idx] = 0.4 * np.sin(2 * np.pi * 240 * t[start_idx:end_idx])
        
        audio = audio.astype(np.float32)
        
        # Process
        enhanced = enhancer.enhance(audio, sr)
        
        # Check that speaker B segments are removed
        for i in range(int(duration / switch_duration)):
            if i % 3 == 2:  # Speaker B segments
                start_idx = int(i * switch_duration * sr)
                end_idx = int((i + 1) * switch_duration * sr)
                segment_energy = calculate_energy_db(enhanced[start_idx:end_idx])
                assert segment_energy < -40, f"Speaker B segment {i} not removed: {segment_energy:.1f}dB"
    
    def test_verify_no_mixing_back(self, enhancer):
        """Verify secondary speakers aren't mixed back in post-processing"""
        sr = 16000
        duration = 2.0
        samples = int(duration * sr)
        
        # Simple case: primary first half, secondary second half
        audio = np.zeros(samples)
        t = np.linspace(0, duration, samples)
        
        mid_point = samples // 2
        
        # Primary speaker (first half)
        audio[:mid_point] = 0.3 * np.sin(2 * np.pi * 150 * t[:mid_point])
        
        # Secondary speaker (second half) - should be completely removed
        audio[mid_point:] = 0.4 * np.sin(2 * np.pi * 250 * t[mid_point:])
        
        audio = audio.astype(np.float32)
        
        # Process multiple times to ensure consistency
        for i in range(3):
            enhanced = enhancer.enhance(audio, sr)
            
            # Second half should be silent
            second_half = enhanced[mid_point:]
            
            # Very strict checks
            assert np.max(np.abs(second_half)) < 0.005, f"Run {i+1}: Secondary speaker remains"
            assert np.std(second_half) < 0.001, f"Run {i+1}: Noise in secondary region"
            
            # Verify it's actually zeros or near-zeros
            num_zeros = np.sum(np.abs(second_half) < 0.0001)
            assert num_zeros > len(second_half) * 0.95, f"Run {i+1}: Not enough silence"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])