#!/usr/bin/env python3
"""Debug script to understand why selective secondary removal isn't working"""

import numpy as np
import logging
from processors.audio_enhancement.selective_secondary_removal import SelectiveSecondaryRemoval

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_selective_removal():
    """Test selective secondary removal with detailed debugging"""
    
    # Create test audio
    sr = 16000
    duration = 3.0
    
    # Primary speaker (first 2.5s) - 440Hz sine wave
    t1 = np.linspace(0, 2.5, int(2.5 * sr))
    primary = np.sin(2 * np.pi * 440 * t1) * 0.5
    
    # Secondary speaker (last 0.5s) - 880Hz sine wave
    t2 = np.linspace(0, 0.5, int(0.5 * sr))
    secondary = np.sin(2 * np.pi * 880 * t2) * 0.3
    
    # Combine
    audio = np.zeros(int(duration * sr))
    audio[:len(primary)] = primary
    audio[len(primary):] = secondary
    
    print(f"Created test audio: {len(audio)} samples")
    print(f"Primary speaker: 0-2.5s (440Hz)")
    print(f"Secondary speaker: 2.5-3.0s (880Hz)")
    
    # Calculate initial energies
    last_1s = audio[-sr:]
    initial_energy = 20 * np.log10(np.sqrt(np.mean(last_1s ** 2)) + 1e-10)
    print(f"\nInitial end energy (last 1s): {initial_energy:.1f}dB")
    
    # Process with selective removal
    remover = SelectiveSecondaryRemoval()
    processed, metadata = remover.process(audio, sr)
    
    print(f"\nProcessing metadata: {metadata}")
    
    # Check results
    last_1s_processed = processed[-sr:]
    final_energy = 20 * np.log10(np.sqrt(np.mean(last_1s_processed ** 2)) + 1e-10)
    print(f"\nFinal end energy (last 1s): {final_energy:.1f}dB")
    
    # Check if actually zeroed
    last_0_5s = processed[int(2.5 * sr):]
    max_val = np.max(np.abs(last_0_5s))
    mean_val = np.mean(np.abs(last_0_5s))
    print(f"\nLast 0.5s analysis:")
    print(f"  Max absolute value: {max_val}")
    print(f"  Mean absolute value: {mean_val}")
    print(f"  Is all zeros: {np.allclose(last_0_5s, 0)}")
    
    # Check a few samples
    print(f"\nSample values from last 0.5s:")
    for i in range(0, min(10, len(last_0_5s)), 1):
        print(f"  Sample {i}: {last_0_5s[i]}")
    
    # Save for inspection
    import soundfile as sf
    sf.write('debug_original.wav', audio, sr)
    sf.write('debug_processed.wav', processed, sr)
    print("\nSaved debug_original.wav and debug_processed.wav for inspection")

if __name__ == "__main__":
    test_selective_removal()