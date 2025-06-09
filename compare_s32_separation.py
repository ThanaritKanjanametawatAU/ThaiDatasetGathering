#!/usr/bin/env python3
"""
Compare S32 before and after complete speaker separation
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import os

def analyze_audio(audio, sample_rate, title):
    """Analyze and plot audio characteristics"""
    
    # Calculate energy over time
    window_size = int(0.1 * sample_rate)  # 100ms windows
    hop_size = int(0.05 * sample_rate)    # 50ms hop
    
    energy = []
    times = []
    
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        energy.append(rms)
        times.append(i / sample_rate)
    
    # Calculate spectral characteristics
    freqs, power = signal.periodogram(audio, sample_rate)
    
    # Find dominant frequencies
    peaks, _ = signal.find_peaks(power, height=np.max(power) * 0.1)
    dominant_freqs = freqs[peaks[:5]]
    
    print(f"\n{title}:")
    print(f"  Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"  RMS Energy: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"  Peak amplitude: {np.max(np.abs(audio)):.4f}")
    print(f"  Dynamic range: {20 * np.log10(np.max(np.abs(audio)) / (np.min(np.abs(audio[np.abs(audio) > 0.001])) + 1e-10)):.1f} dB")
    print(f"  Dominant frequencies: {dominant_freqs[:3]} Hz")
    
    return times, energy

def compare_s32_files():
    """Compare different versions of S32"""
    
    files = {
        'Uploaded': 'test_audio_output/s32_uploaded.wav',
        'Re-separated': 'test_audio_output/s32_reseparated.wav',
        'Original (if exists)': 'test_audio_output/s32_from_dataset.wav'
    }
    
    fig, axes = plt.subplots(len(files), 2, figsize=(15, 10))
    fig.suptitle('S32 Audio Analysis - Complete Speaker Separation', fontsize=16)
    
    valid_files = 0
    
    for idx, (label, filepath) in enumerate(files.items()):
        if os.path.exists(filepath):
            audio, sr = sf.read(filepath)
            times, energy = analyze_audio(audio, sr, label)
            
            # Plot waveform
            ax1 = axes[idx, 0]
            time_axis = np.arange(len(audio)) / sr
            ax1.plot(time_axis, audio, alpha=0.7)
            ax1.set_title(f'{label} - Waveform')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            
            # Plot energy over time
            ax2 = axes[idx, 1]
            ax2.plot(times, energy, color='red', alpha=0.8)
            ax2.set_title(f'{label} - Energy over Time')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('RMS Energy')
            ax2.grid(True, alpha=0.3)
            
            # Mark potential speaker changes
            if 'separated' in label.lower():
                # Detect significant energy drops
                energy_array = np.array(energy)
                threshold = np.mean(energy_array) * 0.3
                low_energy_regions = energy_array < threshold
                
                # Find transitions
                transitions = np.diff(low_energy_regions.astype(int))
                silence_starts = np.where(transitions == 1)[0]
                silence_ends = np.where(transitions == -1)[0]
                
                for start_idx in silence_starts:
                    if start_idx < len(times):
                        ax2.axvline(x=times[start_idx], color='green', linestyle='--', alpha=0.5, label='Silence start')
                
            valid_files += 1
        else:
            axes[idx, 0].text(0.5, 0.5, f'{label}\nFile not found', ha='center', va='center')
            axes[idx, 1].text(0.5, 0.5, f'{label}\nFile not found', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('test_audio_output/s32_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: test_audio_output/s32_comparison.png")
    
    # If we have both uploaded and re-separated, calculate difference
    if os.path.exists(files['Uploaded']) and os.path.exists(files['Re-separated']):
        uploaded, sr1 = sf.read(files['Uploaded'])
        separated, sr2 = sf.read(files['Re-separated'])
        
        if sr1 == sr2 and len(uploaded) == len(separated):
            # Calculate correlation
            correlation = np.corrcoef(uploaded, separated)[0, 1]
            print(f"\nCorrelation between uploaded and re-separated: {correlation:.4f}")
            
            # Calculate difference
            diff = uploaded - separated
            diff_rms = np.sqrt(np.mean(diff**2))
            print(f"RMS difference: {diff_rms:.6f}")
            
            # Check if they're identical
            if np.allclose(uploaded, separated, rtol=1e-5):
                print("WARNING: Uploaded and re-separated audio are nearly identical!")
                print("This suggests the complete separation might not have been applied during main.sh")
            else:
                print("Audio files are different, suggesting separation was applied")

if __name__ == "__main__":
    compare_s32_files()