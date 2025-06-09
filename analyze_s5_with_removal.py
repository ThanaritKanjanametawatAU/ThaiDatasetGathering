#!/usr/bin/env python3
"""Analyze S5 sample with secondary speaker removal"""

import numpy as np
import soundfile as sf
from processors.audio_enhancement.secondary_removal import SecondaryRemover

# Load S5 sample
audio, sr = sf.read("test_audio_output/s5_from_dataset.wav")
print(f"S5 audio: duration={len(audio)/sr:.2f}s, sample_rate={sr}")

# Detect secondary speakers
remover = SecondaryRemover()
detection = remover.detect_secondary_speakers(audio, sr)

print(f"\nDetection result:")
print(f"  Has secondary at end: {detection.has_secondary_at_end}")
print(f"  Secondary start time: {detection.secondary_start_time:.2f}s")
print(f"  Number of speakers: {detection.num_speakers}")
print(f"  Confidence: {detection.confidence:.2f}")

# Apply removal
processed = remover.remove_secondary_speakers(audio, sr)

# Analyze results
last_500ms_orig = audio[-int(0.5 * sr):]
last_500ms_proc = processed[-int(0.5 * sr):]

print(f"\nOriginal last 500ms:")
print(f"  Max amplitude: {np.max(np.abs(last_500ms_orig)):.4f}")
print(f"  RMS energy: {np.sqrt(np.mean(last_500ms_orig**2)):.4f}")

print(f"\nProcessed last 500ms:")
print(f"  Max amplitude: {np.max(np.abs(last_500ms_proc)):.4f}")
print(f"  RMS energy: {np.sqrt(np.mean(last_500ms_proc**2)):.4f}")

# Check what's happening in the last 200ms
last_200ms_proc = processed[-int(0.2 * sr):]
print(f"\nProcessed last 200ms:")
print(f"  Max amplitude: {np.max(np.abs(last_200ms_proc)):.4f}")
print(f"  RMS energy: {np.sqrt(np.mean(last_200ms_proc**2)):.4f}")

# Check the fade region
fade_start_sample = int(detection.secondary_start_time * sr)
fade_region = processed[fade_start_sample-800:fade_start_sample]
print(f"\nFade region (50ms before {detection.secondary_start_time:.2f}s):")
print(f"  Max amplitude: {np.max(np.abs(fade_region)):.4f}")

# Save processed audio
sf.write("test_audio_output/s5_processed_removal.wav", processed, sr)
print(f"\nSaved processed audio to test_audio_output/s5_processed_removal.wav")