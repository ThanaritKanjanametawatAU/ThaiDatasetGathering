#!/usr/bin/env python
"""Diagnose why S5 secondary speaker is not being removed"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from processors.audio_enhancement import AudioEnhancer
# Load S5 sample
print("Loading S5 sample...")
audio, sr = sf.read("test_audio_output/s5_from_dataset.wav")
print(f"Audio shape: {audio.shape}, Sample rate: {sr}")

# Create enhancer with ultra_aggressive mode
print("\n=== Testing Enhancement ===")
enhancer = AudioEnhancer(
    enhancement_level='ultra_aggressive',
    enable_35db_enhancement=False
)

# Check configuration
print(f"\nEnhancement config for ultra_aggressive:")
config = enhancer.ENHANCEMENT_LEVELS.get('ultra_aggressive')
for key, value in config.items():
    print(f"  {key}: {value}")

# Test noise assessment
print("\n=== Noise Assessment ===")
noise_level = enhancer.assess_noise_level(audio, sr, quick=False)
print(f"Assessed noise level: {noise_level}")

# Apply enhancement
print("\n=== Applying Enhancement ===")
enhanced, metadata = enhancer.enhance(
    audio, sr, 
    noise_level='ultra_aggressive',
    return_metadata=True
)

print("\nEnhancement metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")

# Save enhanced audio
sf.write("test_audio_output/s5_enhanced_diagnostic.wav", enhanced, sr)
print("\nSaved enhanced audio to: test_audio_output/s5_enhanced_diagnostic.wav")

# Analyze last second of enhanced audio
last_second = enhanced[-sr:]
energy = np.sqrt(np.mean(last_second**2))
energy_db = 20 * np.log10(energy + 1e-10)
max_amp = np.max(np.abs(last_second))

print(f"\nEnhanced audio - Last 1 second analysis:")
print(f"  Energy: {energy:.6f} ({energy_db:.1f} dB)")
print(f"  Max amplitude: {max_amp:.6f}")

# Visual check if secondary speaker is still there
print(f"\nVisual check: Is secondary speaker still audible at max_amp={max_amp:.6f}? {'YES' if max_amp > 0.01 else 'NO'}")

# Plot comparison
plt.figure(figsize=(12, 8))

# Original audio
plt.subplot(2, 2, 1)
time_orig = np.arange(len(audio)) / sr
plt.plot(time_orig, audio)
plt.title('Original Audio - Full')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 2)
last_3s_orig = audio[-3*sr:]
time_3s = np.arange(len(last_3s_orig)) / sr
plt.plot(time_3s, last_3s_orig)
plt.title('Original Audio - Last 3 seconds')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Enhanced audio
plt.subplot(2, 2, 3)
time_enh = np.arange(len(enhanced)) / sr
plt.plot(time_enh, enhanced, color='green')
plt.title('Enhanced Audio - Full')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 4)
last_3s_enh = enhanced[-3*sr:]
time_3s = np.arange(len(last_3s_enh)) / sr
plt.plot(time_3s, last_3s_enh, color='green')
plt.title('Enhanced Audio - Last 3 seconds')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.savefig('test_audio_output/s5_enhancement_comparison.png')
print("\nSaved comparison plot to: test_audio_output/s5_enhancement_comparison.png")

# Check if the main processing path is using enhancement
print("\n=== Checking Main Processing Path ===")
from processors.gigaspeech2 import GigaSpeech2Processor
from config import NOISE_REDUCTION_CONFIG

config = {
    "name": "GigaSpeech2",
    "source": "speechcolab/gigaspeech2",
    "dataset_name": "GigaSpeech2",
    "language_filter": "th",
    "audio_enhancement": {
        "enabled": True,
        "level": "ultra_aggressive",
        "enhancer": enhancer
    }
}

processor = GigaSpeech2Processor(config)
print(f"Processor noise reduction enabled: {processor.noise_reduction_enabled}")
print(f"Processor audio enhancement config: {processor.audio_enhancement}")

# Test preprocessing
print("\n=== Testing Preprocessing ===")
# Convert audio to bytes for testing
import io
buffer = io.BytesIO()
sf.write(buffer, audio, sr, format='WAV')
audio_bytes = buffer.getvalue()

processed_bytes, enhancement_metadata = processor.preprocess_audio(audio_bytes, "S5")
print(f"Enhancement metadata from preprocessing: {enhancement_metadata}")

# Convert back to array to check
buffer_out = io.BytesIO(processed_bytes)
processed_audio, _ = sf.read(buffer_out)

# Check last second
last_second_proc = processed_audio[-sr:]
energy_proc = np.sqrt(np.mean(last_second_proc**2))
energy_db_proc = 20 * np.log10(energy_proc + 1e-10)

print(f"\nProcessed audio - Last 1 second analysis:")
print(f"  Energy: {energy_proc:.6f} ({energy_db_proc:.1f} dB)")
print(f"  Max amplitude: {np.max(np.abs(last_second_proc)):.6f}")

# Visual check on processed audio
print(f"Visual check after preprocessing: Is secondary speaker still audible? {'YES' if np.max(np.abs(last_second_proc)) > 0.01 else 'NO'}")