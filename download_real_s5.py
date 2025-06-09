#!/usr/bin/env python3
"""Download real S5 sample from GigaSpeech2"""

import os
import sys
from datasets import load_dataset
import soundfile as sf
import numpy as np

sys.path.append('.')

# Set Hugging Face token
os.environ['HF_TOKEN'] = open('.hf_token', 'r').read().strip()

print("Loading GigaSpeech2 dataset...")
dataset = load_dataset(
    "speechcolab/gigaspeech2",
    'default',
    split='th_train',  # Thai train split
    streaming=True,
    trust_remote_code=True,
    token=os.environ['HF_TOKEN']
)

print("Searching for S5 (5th sample)...")
# Get the 5th sample (S5)
samples = []
for i, sample in enumerate(dataset):
    samples.append(sample)
    if i >= 4:  # 0-indexed, so 4 is the 5th sample
        break

if len(samples) >= 5:
    s5_sample = samples[4]  # 5th sample (0-indexed)
    
    # Extract audio
    audio = np.array(s5_sample['audio']['array'], dtype=np.float32)
    sr = s5_sample['audio']['sampling_rate']
    
    print(f"\nFound S5:")
    print(f"  ID: {s5_sample.get('sid', 'unknown')}")
    print(f"  Duration: {len(audio) / sr:.2f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Transcript: {s5_sample.get('text', 'N/A')}")
    
    # Save audio
    sf.write("s5_original_from_gigaspeech2.wav", audio, sr)
    print(f"\nSaved to: s5_original_from_gigaspeech2.wav")
    
    # Also save metadata
    with open("s5_metadata.txt", "w") as f:
        f.write(f"Sample ID: {s5_sample.get('sid', 'unknown')}\n")
        f.write(f"Duration: {len(audio) / sr:.2f}s\n")
        f.write(f"Sample rate: {sr}Hz\n")
        f.write(f"Transcript: {s5_sample.get('text', 'N/A')}\n")
        f.write(f"Shape: {audio.shape}\n")
    
    print("Metadata saved to: s5_metadata.txt")
else:
    print("Could not find 5 samples in the dataset")