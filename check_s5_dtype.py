#!/usr/bin/env python
"""Check dtype of S5 audio"""

import numpy as np
import soundfile as sf

# Load S5 sample
print("Loading S5 sample...")
audio, sr = sf.read("test_audio_output/s5_from_dataset.wav")
print(f"Audio shape: {audio.shape}")
print(f"Audio dtype: {audio.dtype}")
print(f"Sample rate: {sr}")

# Check if it's float16
if audio.dtype == np.float16:
    print("\nWARNING: Audio is float16! Converting to float32...")
    audio = audio.astype(np.float32)
    print(f"New dtype: {audio.dtype}")
    
    # Save as float32
    sf.write("test_audio_output/s5_float32.wav", audio, sr)
    print("Saved float32 version to: test_audio_output/s5_float32.wav")
else:
    print("\nAudio is already in a compatible format")