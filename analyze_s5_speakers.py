#!/usr/bin/env python
"""Analyze S5 to understand the speaker pattern"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

# Load S5 sample
print("Loading S5 sample...")
audio, sr = sf.read("test_audio_output/s5_from_dataset.wav")
print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
print(f"Duration: {len(audio)/sr:.2f} seconds")

# Analyze energy over time
frame_length = int(0.025 * sr)  # 25ms frames
hop_length = int(0.010 * sr)    # 10ms hop

# Calculate energy for each frame
energies = []
times = []
for i in range(0, len(audio) - frame_length, hop_length):
    frame = audio[i:i+frame_length]
    energy = np.sqrt(np.mean(frame**2))
    energies.append(energy)
    times.append(i / sr)

energies = np.array(energies)
times = np.array(times)

# Find silence regions
silence_threshold = np.percentile(energies, 20)  # Bottom 20% energy
silence_mask = energies < silence_threshold

# Identify speaker segments
speaking_mask = ~silence_mask
speaking_changes = np.diff(np.concatenate(([False], speaking_mask, [False])).astype(int))
starts = np.where(speaking_changes == 1)[0]
ends = np.where(speaking_changes == -1)[0]

print(f"\nFound {len(starts)} speaking segments:")
for i, (start, end) in enumerate(zip(starts, ends)):
    start_time = times[start] if start < len(times) else len(audio)/sr
    end_time = times[end-1] if end-1 < len(times) else len(audio)/sr
    duration = end_time - start_time
    print(f"  Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")

# Analyze the last 3 seconds in detail
last_3s_start = max(0, len(audio) - 3*sr)
last_3s_audio = audio[last_3s_start:]
last_3s_time_offset = last_3s_start / sr

print(f"\n=== Last 3 seconds analysis (starting at {last_3s_time_offset:.2f}s) ===")

# Find speaking segments in last 3 seconds
last_3s_energies = []
last_3s_times = []
for i in range(0, len(last_3s_audio) - frame_length, hop_length):
    frame = last_3s_audio[i:i+frame_length]
    energy = np.sqrt(np.mean(frame**2))
    last_3s_energies.append(energy)
    last_3s_times.append(last_3s_time_offset + i / sr)

last_3s_energies = np.array(last_3s_energies)
last_3s_times = np.array(last_3s_times)

# Identify distinct speakers based on energy patterns
# Secondary speakers often have different energy patterns
energy_mean = np.mean(last_3s_energies[last_3s_energies > silence_threshold])
energy_std = np.std(last_3s_energies[last_3s_energies > silence_threshold])

print(f"Energy statistics in last 3s:")
print(f"  Mean energy (non-silent): {energy_mean:.4f}")
print(f"  Std deviation: {energy_std:.4f}")

# Look for abrupt changes that might indicate speaker change
energy_diff = np.abs(np.diff(last_3s_energies))
change_threshold = np.percentile(energy_diff, 95)
potential_speaker_changes = np.where(energy_diff > change_threshold)[0]

print(f"\nPotential speaker changes in last 3s:")
for idx in potential_speaker_changes[-5:]:  # Show last 5
    if idx < len(last_3s_times):
        print(f"  At {last_3s_times[idx]:.2f}s (energy change: {energy_diff[idx]:.4f})")

# Plot analysis
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Full audio waveform with energy
ax1 = axes[0]
ax1.plot(np.arange(len(audio))/sr, audio, alpha=0.6)
ax1_twin = ax1.twinx()
ax1_twin.plot(times, energies, 'r-', alpha=0.8, label='Energy')
ax1_twin.axhline(y=silence_threshold, color='g', linestyle='--', label='Silence threshold')
ax1.set_title('Full Audio with Energy Envelope')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1_twin.set_ylabel('Energy', color='r')
ax1_twin.legend()
ax1.grid(True, alpha=0.3)

# Last 3 seconds detailed
ax2 = axes[1]
ax2.plot(last_3s_time_offset + np.arange(len(last_3s_audio))/sr, last_3s_audio)
ax2.set_title('Last 3 Seconds - Waveform')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.grid(True, alpha=0.3)

# Last 3 seconds energy
ax3 = axes[2]
ax3.plot(last_3s_times, last_3s_energies, 'b-', label='Energy')
ax3.axhline(y=silence_threshold, color='g', linestyle='--', label='Silence threshold')
ax3.axhline(y=energy_mean, color='r', linestyle=':', label='Mean energy')
# Mark potential speaker changes
for idx in potential_speaker_changes:
    if idx < len(last_3s_times):
        ax3.axvline(x=last_3s_times[idx], color='orange', alpha=0.5)
ax3.set_title('Last 3 Seconds - Energy Analysis')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Energy')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_audio_output/s5_speaker_analysis.png', dpi=150)
print(f"\nSaved analysis plot to: test_audio_output/s5_speaker_analysis.png")

# Listen to specific segments
print("\n=== Creating audio segments for listening ===")

# Extract last 1 second
last_1s = audio[-sr:]
sf.write("test_audio_output/s5_last_1s.wav", last_1s, sr)
print("Saved last 1 second to: test_audio_output/s5_last_1s.wav")

# Extract last 2 seconds
last_2s = audio[-2*sr:]
sf.write("test_audio_output/s5_last_2s.wav", last_2s, sr)
print("Saved last 2 seconds to: test_audio_output/s5_last_2s.wav")

# Extract segment before the last speaker
if len(potential_speaker_changes) > 0:
    last_change_idx = potential_speaker_changes[-1]
    last_change_time = last_3s_times[last_change_idx] if last_change_idx < len(last_3s_times) else len(audio)/sr
    change_sample = int(last_change_time * sr)
    
    # Get 1 second before and after the change
    before_change = audio[max(0, change_sample-sr):change_sample]
    after_change = audio[change_sample:min(len(audio), change_sample+sr)]
    
    sf.write("test_audio_output/s5_before_speaker_change.wav", before_change, sr)
    sf.write("test_audio_output/s5_after_speaker_change.wav", after_change, sr)
    print(f"Saved audio around last speaker change at {last_change_time:.2f}s")

print("\nPlease listen to these audio files to understand the speaker pattern.")