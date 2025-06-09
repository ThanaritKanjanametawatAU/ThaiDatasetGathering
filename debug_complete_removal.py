"""
Debug complete removal to see what's happening
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processors.audio_enhancement.core import AudioEnhancer
from utils.audio_metrics import calculate_energy_db
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def create_test_audio(sample_rate=16000):
    """Create test audio with secondary speakers"""
    duration = 5.0
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Primary speaker: middle (0.5s to 3.5s)
    # Secondary speakers: beginning (0-0.8s) and end (4.0-5.0s)
    audio = np.zeros(samples)
    
    # Secondary at beginning
    sec1_end = int(0.8 * sample_rate)
    audio[:sec1_end] = 0.4 * np.sin(2 * np.pi * 250 * t[:sec1_end])
    
    # Primary speaker 
    primary_start = int(0.5 * sample_rate)
    primary_end = int(3.5 * sample_rate)
    audio[primary_start:primary_end] = 0.3 * np.sin(2 * np.pi * 150 * t[primary_start:primary_end])
    
    # Secondary at end
    sec2_start = int(4.0 * sample_rate)
    audio[sec2_start:] = 0.5 * np.sin(2 * np.pi * 300 * t[sec2_start:])
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    return audio.astype(np.float32)

def test_removal_debug():
    """Debug test for complete removal"""
    enhancer = AudioEnhancer(
        use_gpu=False,
        enhancement_level='ultra_aggressive'
    )
    
    # Check if complete_secondary_removal is initialized
    print(f"complete_secondary_removal initialized: {enhancer.complete_secondary_removal is not None}")
    print(f"enhancement_level: {enhancer.enhancement_level}")
    
    audio = create_test_audio()
    sample_rate = 16000
    
    print(f"\nOriginal audio:")
    print(f"  Beginning (0-0.8s): {calculate_energy_db(audio[:int(0.8*sample_rate)]):.1f}dB")
    print(f"  Middle (0.5-3.5s): {calculate_energy_db(audio[int(0.5*sample_rate):int(3.5*sample_rate)]):.1f}dB")
    print(f"  End (4.0-5.0s): {calculate_energy_db(audio[int(4.0*sample_rate):]):.1f}dB")
    
    # Process
    enhanced = enhancer.enhance(audio, sample_rate)
    
    print(f"\nEnhanced audio:")
    print(f"  Beginning: {calculate_energy_db(enhanced[:int(0.8*sample_rate)]):.1f}dB")
    print(f"  Middle: {calculate_energy_db(enhanced[int(0.5*sample_rate):int(3.5*sample_rate)]):.1f}dB")
    print(f"  End: {calculate_energy_db(enhanced[int(4.0*sample_rate):]):.1f}dB")
    
    # Test if removal was effective
    begin_energy = calculate_energy_db(enhanced[:int(0.8*sample_rate)])
    end_energy = calculate_energy_db(enhanced[int(4.0*sample_rate):])
    
    if begin_energy < -50 and end_energy < -50:
        print("\n✅ SUCCESS: Secondary speakers removed!")
    else:
        print(f"\n❌ FAILED: Secondary speakers still present")
        if begin_energy >= -50:
            print(f"   Beginning: {begin_energy:.1f}dB (should be < -50dB)")
        if end_energy >= -50:
            print(f"   End: {end_energy:.1f}dB (should be < -50dB)")

if __name__ == "__main__":
    test_removal_debug()