#!/usr/bin/env python3
"""Analyze why energy calculation shows -14.4dB when audio is all zeros"""

import numpy as np

def analyze_energy():
    # Create all zeros array (like the silenced region)
    sr = 16000
    zeros = np.zeros(sr)  # 1 second of zeros
    
    # Calculate energy the same way as the code
    energy_linear = np.sqrt(np.mean(zeros ** 2))
    energy_db = 20 * np.log10(energy_linear + 1e-10)
    
    print(f"All zeros array:")
    print(f"  Linear energy: {energy_linear}")
    print(f"  Energy in dB: {energy_db:.1f}dB")
    print(f"  Expected: around -200dB or lower")
    
    # Try with small epsilon
    small_val = 1e-10
    energy_db_small = 20 * np.log10(small_val)
    print(f"\nWith epsilon only (1e-10):")
    print(f"  Energy in dB: {energy_db_small:.1f}dB")
    
    # Create array with small non-zero values
    small_noise = np.random.normal(0, 1e-3, sr)
    energy_noise = np.sqrt(np.mean(small_noise ** 2))
    energy_db_noise = 20 * np.log10(energy_noise + 1e-10)
    print(f"\nSmall noise (std=0.001):")
    print(f"  Linear energy: {energy_noise}")
    print(f"  Energy in dB: {energy_db_noise:.1f}dB")
    
    # The issue might be that the last 1s includes non-zero audio
    # Let's simulate: 0.5s zeros + 0.5s of primary speaker
    mixed = np.zeros(sr)
    mixed[sr//2:] = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sr//2)) * 0.1
    energy_mixed = np.sqrt(np.mean(mixed ** 2))
    energy_db_mixed = 20 * np.log10(energy_mixed + 1e-10)
    print(f"\nMixed (half zeros, half 0.1 amplitude sine):")
    print(f"  Linear energy: {energy_mixed}")
    print(f"  Energy in dB: {energy_db_mixed:.1f}dB")

if __name__ == "__main__":
    analyze_energy()