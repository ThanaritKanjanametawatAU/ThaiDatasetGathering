"""
Test-Driven Development for Selective Secondary Speaker Removal
This test suite ensures secondary speakers are removed while preserving primary speaker quality
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys
import soundfile as sf

sys.path.append('.')

from processors.audio_enhancement.core import AudioEnhancer
from utils.audio_metrics import calculate_energy_db, calculate_snr, calculate_pesq, calculate_stoi


class TestSelectiveSecondaryRemoval:
    """Test suite for selective secondary speaker removal that preserves primary speaker quality"""
    
    @pytest.fixture
    def audio_enhancer(self):
        """Create audio enhancer with selective secondary removal"""
        return AudioEnhancer(
            use_gpu=torch.cuda.is_available(),
            enhancement_level="selective_secondary_removal"  # New mode we'll implement
        )
    
    @pytest.fixture
    def real_s5_audio(self):
        """Load real S5 sample that has secondary speaker at end"""
        # This would be the actual S5 sample from GigaSpeech2
        # For now, we'll simulate it
        sr = 16000
        duration = 3.22  # S5 duration
        
        # Simulate primary speaker (first 2.2s)
        t1 = np.linspace(0, 2.2, int(2.2 * sr))
        primary = 0.3 * np.sin(2 * np.pi * 200 * t1) * (1 + 0.1 * np.sin(2 * np.pi * 3 * t1))
        
        # Simulate secondary speaker (last 1s, overlapping slightly)
        t2 = np.linspace(0, 1.0, int(1.0 * sr))
        secondary = 0.2 * np.sin(2 * np.pi * 400 * t2) * (1 + 0.1 * np.sin(2 * np.pi * 5 * t2))
        
        # Combine (secondary starts at 2.22s)
        audio = np.zeros(int(duration * sr))
        audio[:len(primary)] = primary
        audio[-len(secondary):] += secondary
        
        return audio.astype(np.float32), sr
    
    def test_secondary_speaker_removed_at_end(self, audio_enhancer, real_s5_audio):
        """Test that secondary speaker at end is removed"""
        audio, sr = real_s5_audio
        
        # Process audio
        enhanced, metadata = audio_enhancer.enhance(
            audio, sr, 
            noise_level="selective_secondary_removal",
            return_metadata=True
        )
        
        # Check last 1s for secondary speaker removal
        last_1s = enhanced[-sr:]
        end_energy = calculate_energy_db(last_1s)
        
        # Should be significantly reduced but not completely silent
        assert end_energy < -50, f"Secondary speaker still present: {end_energy:.1f}dB"
        assert metadata.get('secondary_speaker_removed', False), "Secondary speaker removal not reported"
    
    def test_primary_speaker_preserved(self, audio_enhancer, real_s5_audio):
        """Test that primary speaker quality is preserved"""
        audio, sr = real_s5_audio
        
        # Get original primary speaker segment (first 2s)
        original_primary = audio[:2*sr]
        
        # Process audio
        enhanced, metadata = audio_enhancer.enhance(
            audio, sr,
            noise_level="selective_secondary_removal", 
            return_metadata=True
        )
        
        # Get enhanced primary speaker segment
        enhanced_primary = enhanced[:2*sr]
        
        # Check that primary speaker is preserved
        # 1. Energy should be similar
        orig_energy = calculate_energy_db(original_primary)
        enh_energy = calculate_energy_db(enhanced_primary)
        energy_diff = abs(orig_energy - enh_energy)
        assert energy_diff < 3.0, f"Primary speaker energy changed too much: {energy_diff:.1f}dB"
        
        # 2. Should not be silent
        assert enh_energy > -30, f"Primary speaker too quiet: {enh_energy:.1f}dB"
        
        # 3. Correlation should be high (similar waveform)
        correlation = np.corrcoef(original_primary, enhanced_primary)[0, 1]
        assert correlation > 0.8, f"Primary speaker waveform changed too much: correlation={correlation:.2f}"
    
    def test_no_quality_degradation(self, audio_enhancer):
        """Test that audio quality metrics are preserved for primary speaker"""
        # Create clean speech signal
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(duration * sr))
        
        # Primary speaker only (clean speech)
        primary = 0.3 * np.sin(2 * np.pi * 200 * t) * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))
        
        # Add secondary speaker at end
        secondary = 0.2 * np.sin(2 * np.pi * 400 * t[-sr:]) 
        audio = primary.copy()
        audio[-sr:] += secondary
        
        # Process
        enhanced, metadata = audio_enhancer.enhance(
            audio.astype(np.float32), sr,
            noise_level="selective_secondary_removal",
            return_metadata=True
        )
        
        # Check quality metrics for the primary speaker portion (first 2s)
        primary_orig = primary[:2*sr]
        primary_enh = enhanced[:2*sr]
        
        # PESQ should be reasonable (good quality for synthetic signals)
        # Note: PESQ on synthetic sine waves is typically lower than real speech
        pesq = calculate_pesq(primary_enh, primary_orig, sr)
        assert pesq > 2.5, f"PESQ too low: {pesq:.2f}"
        
        # STOI should be high (good intelligibility) 
        stoi = calculate_stoi(primary_enh, primary_orig, sr)
        assert stoi > 0.85, f"STOI too low: {stoi:.2f}"
    
    def test_overlapping_speech_handling(self, audio_enhancer):
        """Test handling of overlapping speech regions"""
        sr = 16000
        
        # Create overlapping speech
        t = np.linspace(0, 3, 3 * sr)
        primary = 0.3 * np.sin(2 * np.pi * 200 * t)
        secondary = np.zeros_like(t)
        
        # Secondary speaker appears in segments
        secondary[int(0.5*sr):int(1.0*sr)] = 0.2 * np.sin(2 * np.pi * 400 * t[int(0.5*sr):int(1.0*sr)])
        secondary[int(2.0*sr):] = 0.2 * np.sin(2 * np.pi * 400 * t[int(2.0*sr):])
        
        audio = (primary + secondary).astype(np.float32)
        
        # Process
        enhanced, metadata = audio_enhancer.enhance(
            audio, sr,
            noise_level="selective_secondary_removal",
            return_metadata=True
        )
        
        # Check that secondary regions are reduced
        # Region 1: 0.5-1.0s
        region1 = enhanced[int(0.5*sr):int(1.0*sr)]
        region1_energy = calculate_energy_db(region1)
        
        # Region 2: 2.0s-end
        region2 = enhanced[int(2.0*sr):]
        region2_energy = calculate_energy_db(region2)
        
        # Both should be reduced. Region 1 has primary speaker so won't be silent
        # Region 2 at end should be much more reduced
        assert region1_energy < -10, f"Region 1 secondary not reduced: {region1_energy:.1f}dB"
        assert region2_energy < -50, f"Region 2 secondary not reduced: {region2_energy:.1f}dB"
        
        # Primary regions should be preserved
        primary_region = enhanced[int(1.2*sr):int(1.8*sr)]
        primary_energy = calculate_energy_db(primary_region)
        assert primary_energy > -25, f"Primary speaker affected: {primary_energy:.1f}dB"
    
    def test_intelligent_speaker_identification(self, audio_enhancer):
        """Test that the system correctly identifies primary vs secondary speakers"""
        sr = 16000
        
        # Scenario: Secondary speaker speaks first briefly, then primary speaks longer
        t = np.linspace(0, 3, 3 * sr)
        
        # Secondary speaker: 0-0.3s (brief)
        secondary = np.zeros_like(t)
        secondary[:int(0.3*sr)] = 0.3 * np.sin(2 * np.pi * 400 * t[:int(0.3*sr)])
        
        # Primary speaker: 0.5s-2.8s (dominant)
        primary = np.zeros_like(t)
        primary[int(0.5*sr):int(2.8*sr)] = 0.3 * np.sin(2 * np.pi * 200 * t[int(0.5*sr):int(2.8*sr)])
        
        # Secondary returns at end: 2.8s-3s
        secondary[int(2.8*sr):] = 0.3 * np.sin(2 * np.pi * 400 * t[int(2.8*sr):])
        
        audio = (primary + secondary).astype(np.float32)
        
        # Process
        enhanced, metadata = audio_enhancer.enhance(
            audio, sr,
            noise_level="selective_secondary_removal",
            return_metadata=True
        )
        
        # Check that both secondary segments are removed
        # First segment (0-0.3s)
        first_segment = enhanced[:int(0.3*sr)]
        first_energy = calculate_energy_db(first_segment)
        assert first_energy < -50, f"First secondary segment not removed: {first_energy:.1f}dB"
        
        # Last segment (2.8s-3s)
        last_segment = enhanced[int(2.8*sr):]
        last_energy = calculate_energy_db(last_segment)
        assert last_energy < -50, f"Last secondary segment not removed: {last_energy:.1f}dB"
        
        # Primary segment preserved (1s-2s for testing)
        primary_segment = enhanced[sr:2*sr]
        primary_energy = calculate_energy_db(primary_segment)
        assert primary_energy > -25, f"Primary speaker removed: {primary_energy:.1f}dB"
        
        # Metadata should indicate correct identification
        assert metadata.get('dominant_speaker_identified', False), "Dominant speaker not identified"
        assert metadata.get('dominant_speaker_duration', 0) > 2.0, "Dominant speaker duration incorrect"
    
    def test_no_removal_when_single_speaker(self, audio_enhancer):
        """Test that nothing is removed when there's only one speaker"""
        sr = 16000
        duration = 3.0
        
        # Single speaker throughout
        t = np.linspace(0, duration, int(duration * sr))
        audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        
        # Process
        enhanced, metadata = audio_enhancer.enhance(
            audio, sr,
            noise_level="selective_secondary_removal",
            return_metadata=True
        )
        
        # Audio should be unchanged
        correlation = np.corrcoef(audio, enhanced)[0, 1]
        assert correlation > 0.99, f"Single speaker audio modified: correlation={correlation:.2f}"
        
        # Energy should be preserved
        orig_energy = calculate_energy_db(audio)
        enh_energy = calculate_energy_db(enhanced)
        energy_diff = abs(orig_energy - enh_energy)
        assert energy_diff < 0.5, f"Single speaker energy changed: {energy_diff:.1f}dB"
        
        # Metadata should indicate no secondary speaker
        assert not metadata.get('secondary_speaker_detected', True), "False positive secondary speaker"


class TestRealAudioSamples:
    """Test with real audio samples if available"""
    
    @pytest.fixture
    def s5_path(self):
        """Path to real S5 sample"""
        # Try multiple possible locations
        paths = [
            "s5_from_hf.wav",
            "/tmp/s5_test_output/s5_final.wav",
            "test_sets/gigaspeech2/S5.wav"
        ]
        
        for path in paths:
            if Path(path).exists():
                return path
        
        # Skip if no real sample available
        pytest.skip("No real S5 sample found")
    
    def test_real_s5_sample(self, s5_path):
        """Test with real S5 sample"""
        # Load audio
        audio, sr = sf.read(s5_path)
        
        # Create enhancer
        enhancer = AudioEnhancer(
            use_gpu=torch.cuda.is_available(),
            enhancement_level="selective_secondary_removal"
        )
        
        # Process
        enhanced, metadata = enhancer.enhance(
            audio.astype(np.float32), sr,
            noise_level="selective_secondary_removal",
            return_metadata=True
        )
        
        # Verify secondary speaker removed at end
        last_1s = enhanced[-sr:]
        end_energy = calculate_energy_db(last_1s)
        assert end_energy < -45, f"Secondary speaker still present in S5: {end_energy:.1f}dB"
        
        # Verify primary speaker preserved (check middle section)
        middle_2s = enhanced[sr:3*sr]  # 1s-3s
        middle_energy = calculate_energy_db(middle_2s)
        assert middle_energy > -25, f"Primary speaker degraded in S5: {middle_energy:.1f}dB"
        
        # Save result for manual verification
        sf.write("s5_selective_removal_result.wav", enhanced, sr)
        print(f"Saved processed S5 to s5_selective_removal_result.wav")