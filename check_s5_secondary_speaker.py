#!/usr/bin/env python
"""Check if S5 sample has secondary speaker removed"""

from datasets import load_dataset
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Load the dataset
print("Loading dataset...")
ds = load_dataset('Thanarit/Thai-Voice-10000000', split='train', streaming=True)

# Find sample 5
for i, sample in enumerate(ds):
    if sample['ID'] == 'S5':
        print(f'Found sample S5')
        
        # Extract audio
        audio = np.array(sample["audio"]["array"])
        sr = sample["audio"]["sampling_rate"]
        
        # Save the audio
        output_path = "test_audio_output/s5_from_dataset.wav"
        sf.write(output_path, audio, sr)
        print(f"Saved audio to: {output_path}")
        
        # Analyze the last 1 second
        last_second = audio[-sr:]
        
        # Calculate energy in last second
        energy = np.sqrt(np.mean(last_second**2))
        energy_db = 20 * np.log10(energy + 1e-10)
        
        print(f"\nAnalysis of last 1 second:")
        print(f"Energy: {energy:.6f} ({energy_db:.1f} dB)")
        print(f"Max amplitude: {np.max(np.abs(last_second)):.6f}")
        print(f"RMS: {np.sqrt(np.mean(last_second**2)):.6f}")
        
        # Check if last 0.5s is mostly silent
        last_half_second = audio[-int(sr/2):]
        silence_threshold = 0.001
        is_silent = np.max(np.abs(last_half_second)) < silence_threshold
        
        print(f"\nLast 0.5 seconds analysis:")
        print(f"Max amplitude: {np.max(np.abs(last_half_second)):.6f}")
        print(f"Is silent (< {silence_threshold})?: {is_silent}")
        
        # Plot waveform
        plt.figure(figsize=(12, 6))
        
        # Full waveform
        plt.subplot(2, 1, 1)
        time = np.arange(len(audio)) / sr
        plt.plot(time, audio)
        plt.title('S5 - Full Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Last 2 seconds zoomed
        plt.subplot(2, 1, 2)
        last_2s = audio[-2*sr:]
        time_2s = np.arange(len(last_2s)) / sr
        plt.plot(time_2s, last_2s)
        plt.title('S5 - Last 2 Seconds (Zoomed)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.axvline(x=1.5, color='r', linestyle='--', label='Last 0.5s')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('test_audio_output/s5_waveform_analysis.png')
        print(f"\nSaved waveform plot to: test_audio_output/s5_waveform_analysis.png")
        
        break
        
    if i >= 10:
        print("Sample S5 not found in first 10 samples")
        break