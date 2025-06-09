#!/usr/bin/env python
"""Diagnose the complete enhancement flow for S5"""

import numpy as np
import soundfile as sf
from processors.audio_enhancement import AudioEnhancer
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Load S5 sample
print("Loading S5 sample...")
audio, sr = sf.read("test_audio_output/s5_from_dataset.wav")
print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
print(f"Original max amplitude: {np.max(np.abs(audio)):.6f}")
print(f"Original last second max: {np.max(np.abs(audio[-sr:])):.6f}")

# Create enhancer
enhancer = AudioEnhancer(
    enhancement_level='ultra_aggressive',
    enable_35db_enhancement=False
)

# Test with forced noise level to ensure it uses speaker separation
print("\n=== Testing with ultra_aggressive ===")
enhanced, metadata = enhancer.enhance(
    audio, sr, 
    noise_level='ultra_aggressive',
    return_metadata=True
)

print(f"\nEnhanced max amplitude: {np.max(np.abs(enhanced)):.6f}")
print(f"Enhanced last second max: {np.max(np.abs(enhanced[-sr:])):.6f}")

# Save for listening
sf.write("test_audio_output/s5_flow_test.wav", enhanced, sr)

# Now test what happens with secondary_speaker noise level
print("\n\n=== Testing with secondary_speaker noise level ===")
enhanced2, metadata2 = enhancer.enhance(
    audio, sr,
    noise_level='secondary_speaker',
    return_metadata=True
)

print(f"\nMetadata: {metadata2}")
print(f"Enhanced2 max amplitude: {np.max(np.abs(enhanced2)):.6f}")
print(f"Enhanced2 last second max: {np.max(np.abs(enhanced2[-sr:])):.6f}")

sf.write("test_audio_output/s5_secondary_speaker_mode.wav", enhanced2, sr)