#!/usr/bin/env python3
"""Debug what's actually in the last 1 second of processed audio"""

import numpy as np
import soundfile as sf

# Load the processed audio we just created
processed, sr = sf.read('debug_processed.wav')
original, _ = sf.read('debug_original.wav')

print(f"Audio shape: {processed.shape}, Sample rate: {sr}")
print(f"Total duration: {len(processed)/sr:.2f}s")

# Analyze last 1 second
last_1s = processed[-sr:]
print(f"\nLast 1 second analysis:")
print(f"  Shape: {last_1s.shape}")
print(f"  Min value: {np.min(last_1s)}")
print(f"  Max value: {np.max(last_1s)}")
print(f"  Mean absolute: {np.mean(np.abs(last_1s))}")
print(f"  Non-zero count: {np.count_nonzero(last_1s)}")

# Check sections
print("\nBreaking down last 1s into quarters:")
for i in range(4):
    start = i * sr // 4
    end = (i + 1) * sr // 4
    section = last_1s[start:end]
    energy = 20 * np.log10(np.sqrt(np.mean(section ** 2)) + 1e-10)
    print(f"  Quarter {i+1} (t={2.0 + i*0.25:.2f}s to {2.0 + (i+1)*0.25:.2f}s): {energy:.1f}dB, non-zeros: {np.count_nonzero(section)}")

# Check where non-zeros start/end
non_zero_indices = np.where(last_1s != 0)[0]
if len(non_zero_indices) > 0:
    first_nonzero = non_zero_indices[0]
    last_nonzero = non_zero_indices[-1]
    print(f"\nNon-zero range in last 1s:")
    print(f"  First non-zero at sample {first_nonzero} ({first_nonzero/sr:.3f}s from start of last 1s)")
    print(f"  Last non-zero at sample {last_nonzero} ({last_nonzero/sr:.3f}s from start of last 1s)")
else:
    print("\nAll samples in last 1s are zero!")

# Check the actual secondary speaker region (2.5s-3.0s)
secondary_start = int(2.5 * sr)
secondary_region = processed[secondary_start:]
print(f"\nSecondary speaker region (2.5s-3.0s):")
print(f"  All zeros: {np.allclose(secondary_region, 0)}")
print(f"  Max absolute value: {np.max(np.abs(secondary_region))}")

# The issue: last 1s includes 2.0s-3.0s, not just 2.5s-3.0s!
print("\n!!! KEY INSIGHT !!!")
print("Last 1s includes audio from 2.0s to 3.0s")
print("Secondary speaker is only from 2.5s to 3.0s")
print("So last 1s includes 0.5s of PRIMARY speaker (2.0s-2.5s)!")