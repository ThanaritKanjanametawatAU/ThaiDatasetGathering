#!/usr/bin/env python
"""Diagnose speaker separation issue for S5"""

import numpy as np
import soundfile as sf
from processors.audio_enhancement.speechbrain_separator import SpeakerSeparator, SeparationConfig

# Load S5 sample
print("Loading S5 sample...")
audio, sr = sf.read("test_audio_output/s5_from_dataset.wav")
print(f"Audio shape: {audio.shape}, Sample rate: {sr}")

# Create separator with same config as AudioEnhancer
separation_config = SeparationConfig(
    confidence_threshold=0.7,
    device="cuda",
    batch_size=16,
    speaker_selection="energy",
    use_mixed_precision=True,
    quality_thresholds={
        "min_pesq": 3.5,
        "min_stoi": 0.85,
        "max_spectral_distortion": 0.15
    }
)

separator = SpeakerSeparator(separation_config)

print("\n=== Testing Speaker Separation ===")
# Ensure float32
if audio.dtype == np.float16:
    audio = audio.astype(np.float32)

# Test separation
result = separator.separate_speakers(audio, sr)

print(f"\nSeparation result type: {type(result)}")
print(f"Result attributes: {dir(result)}")

if hasattr(result, 'num_speakers_detected'):
    # New format
    print(f"Number of speakers detected: {result.num_speakers_detected}")
    print(f"Rejected: {result.rejected}")
    print(f"Rejection reason: {result.rejection_reason}")
    print(f"Audio shape: {result.audio.shape}")
    print(f"Metrics: {result.metrics}")
    
    # Check if audio was modified
    audio_diff = np.mean(np.abs(audio - result.audio))
    print(f"\nAverage difference from original: {audio_diff}")
    print(f"Max amplitude of result: {np.max(np.abs(result.audio))}")
else:
    # Old format
    print(f"Result keys: {result.keys()}")
    print(f"Audio shape: {result['audio'].shape}")
    print(f"Detections: {result.get('detections', [])}")
    print(f"Metrics: {result.get('metrics', {})}")

# Save separated audio
output_audio = result.audio if hasattr(result, 'audio') else result['audio']
sf.write("test_audio_output/s5_separated.wav", output_audio, sr)
print(f"\nSaved separated audio to: test_audio_output/s5_separated.wav")

# Analyze what happened
print("\n=== Analysis ===")
print(f"Original audio energy: {np.sqrt(np.mean(audio**2)):.6f}")
print(f"Separated audio energy: {np.sqrt(np.mean(output_audio**2)):.6f}")

# Check last second
last_second_orig = audio[-sr:]
last_second_sep = output_audio[-sr:]

print(f"\nLast second analysis:")
print(f"  Original max amplitude: {np.max(np.abs(last_second_orig)):.6f}")
print(f"  Separated max amplitude: {np.max(np.abs(last_second_sep)):.6f}")

# Try with different settings
print("\n\n=== Testing with different settings ===")
# Try with lower confidence threshold
config2 = SeparationConfig(
    confidence_threshold=0.5,  # Lower threshold
    device="cuda",
    speaker_selection="correlation",  # Try correlation instead of energy
    skip_on_single_speaker=True  # Skip processing if only one speaker
)

separator2 = SpeakerSeparator(config2)
result2 = separator2.separate_speakers(audio, sr)

if hasattr(result2, 'num_speakers_detected'):
    print(f"Speakers detected (lower threshold): {result2.num_speakers_detected}")
    print(f"Rejected: {result2.rejected}")
    if not result2.rejected:
        sf.write("test_audio_output/s5_separated_v2.wav", result2.audio, sr)
        print("Saved alternative separation")
else:
    print("Alternative separation failed")