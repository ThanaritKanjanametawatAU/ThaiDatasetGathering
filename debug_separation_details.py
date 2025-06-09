"""
Debug speaker separation details
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processors.audio_enhancement.speechbrain_separator import SpeechBrainSeparator, SeparationConfig
from processors.audio_enhancement.dominant_speaker_separation import DominantSpeakerSeparator
from utils.audio_metrics import calculate_energy_db
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_test_audio(sample_rate=16000):
    """Create test audio with clear speaker separation"""
    duration = 5.0
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Primary speaker: 0.5s to 3.5s (3 seconds, 60% of audio)
    primary = np.zeros(samples)
    primary_start = int(0.5 * sample_rate)
    primary_end = int(3.5 * sample_rate)
    primary[primary_start:primary_end] = 0.3 * np.sin(2 * np.pi * 150 * t[primary_start:primary_end])
    
    # Secondary speaker 1: 0s to 0.8s
    secondary1 = np.zeros(samples)
    sec1_end = int(0.8 * sample_rate)
    secondary1[:sec1_end] = 0.4 * np.sin(2 * np.pi * 250 * t[:sec1_end])
    
    # Secondary speaker 2: 4.0s to 5.0s
    secondary2 = np.zeros(samples)
    sec2_start = int(4.0 * sample_rate)
    secondary2[sec2_start:] = 0.5 * np.sin(2 * np.pi * 300 * t[sec2_start:])
    
    # Mix
    mixed = primary + secondary1 + secondary2
    mixed = mixed / np.max(np.abs(mixed)) * 0.8
    
    return mixed.astype(np.float32), {
        'primary': primary,
        'secondary1': secondary1,
        'secondary2': secondary2,
        'primary_start': primary_start,
        'primary_end': primary_end,
        'sec1_end': sec1_end,
        'sec2_start': sec2_start
    }

def test_separation_details():
    """Debug separation process in detail"""
    audio, components = create_test_audio()
    sample_rate = 16000
    
    print("Original audio components:")
    print(f"  Primary speaker energy: {calculate_energy_db(components['primary']):.1f}dB")
    print(f"  Secondary 1 energy: {calculate_energy_db(components['secondary1']):.1f}dB")
    print(f"  Secondary 2 energy: {calculate_energy_db(components['secondary2']):.1f}dB")
    print(f"  Mixed audio energy: {calculate_energy_db(audio):.1f}dB")
    
    # Test SpeechBrain separation
    config = SeparationConfig(device='cpu')
    separator = SpeechBrainSeparator(config=config)
    
    print("\n1. Testing SpeechBrain separation...")
    separated_sources = separator.separate_all_speakers(audio, sample_rate)
    
    print(f"Number of separated sources: {len(separated_sources)}")
    for i, source in enumerate(separated_sources):
        print(f"\nSource {i}:")
        print(f"  Total energy: {calculate_energy_db(source):.1f}dB")
        print(f"  Beginning energy: {calculate_energy_db(source[:components['sec1_end']]):.1f}dB")
        print(f"  Middle energy: {calculate_energy_db(source[components['primary_start']:components['primary_end']]):.1f}dB")
        print(f"  End energy: {calculate_energy_db(source[components['sec2_start']:]):.1f}dB")
    
    # Test dominant speaker identification
    print("\n2. Testing dominant speaker identification...")
    dominant_separator = DominantSpeakerSeparator(device='cpu')
    dominant_idx = dominant_separator.identify_dominant_speaker(separated_sources, sample_rate)
    
    print(f"\nDominant speaker identified: Source {dominant_idx}")
    
    # Test which source best matches the primary speaker
    print("\n3. Checking which source matches primary speaker...")
    for i, source in enumerate(separated_sources):
        # Calculate correlation with primary speaker in middle region
        middle_source = source[components['primary_start']:components['primary_end']]
        middle_primary = components['primary'][components['primary_start']:components['primary_end']]
        
        if len(middle_source) > 0 and len(middle_primary) > 0:
            correlation = np.corrcoef(middle_source, middle_primary)[0, 1]
            print(f"Source {i} correlation with primary: {correlation:.3f}")

if __name__ == "__main__":
    test_separation_details()