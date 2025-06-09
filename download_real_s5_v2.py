#!/usr/bin/env python3
"""Download real S5 sample from GigaSpeech2 Thai data"""

import os
import sys
from datasets import load_dataset
import soundfile as sf
import numpy as np

sys.path.append('.')

# Set Hugging Face token
os.environ['HF_TOKEN'] = open('.hf_token', 'r').read().strip()

print("Loading GigaSpeech2 Thai dataset...")

# Load Thai data specifically
dataset = load_dataset(
    "speechcolab/gigaspeech2",
    data_files={'train': 'data/th/train/*.tar.gz'},
    split='train',
    streaming=True,
    trust_remote_code=True,
    token=os.environ['HF_TOKEN']
)

print("Searching for S5 (5th Thai sample)...")
# Get the 5th sample (S5)
samples = []
thai_count = 0

for i, sample in enumerate(dataset):
    # Check if this is a Thai sample by URL
    url = sample.get('__url__', '')
    if '/data/th/' in url:
        samples.append(sample)
        thai_count += 1
        print(f"Found Thai sample {thai_count}: {sample.get('sid', 'unknown')}")
        
        if thai_count >= 5:  # We want the 5th Thai sample
            break
    
    # Safety limit
    if i > 100:
        print("Examined 100 samples, stopping search")
        break

if len(samples) >= 5:
    s5_sample = samples[4]  # 5th sample (0-indexed)
    
    # Extract audio
    audio_data = s5_sample.get('wav', s5_sample.get('audio', {}))
    if isinstance(audio_data, dict):
        audio = np.array(audio_data['array'], dtype=np.float32)
        sr = audio_data['sampling_rate']
    else:
        # Direct array
        audio = np.array(audio_data, dtype=np.float32)
        sr = 16000  # Default
    
    # Get transcript
    transcript = s5_sample.get('text', s5_sample.get('transcript', 'N/A'))
    
    print(f"\nFound S5:")
    print(f"  ID: {s5_sample.get('sid', 'unknown')}")
    print(f"  Duration: {len(audio) / sr:.2f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Transcript: {transcript}")
    
    # Save audio
    sf.write("s5_original_from_gigaspeech2.wav", audio, sr)
    print(f"\nSaved to: s5_original_from_gigaspeech2.wav")
    
    # Also save metadata
    with open("s5_metadata.txt", "w") as f:
        f.write(f"Sample ID: {s5_sample.get('sid', 'unknown')}\n")
        f.write(f"Duration: {len(audio) / sr:.2f}s\n")
        f.write(f"Sample rate: {sr}Hz\n")
        f.write(f"Transcript: {transcript}\n")
        f.write(f"Shape: {audio.shape}\n")
        f.write(f"URL: {s5_sample.get('__url__', 'N/A')}\n")
    
    print("Metadata saved to: s5_metadata.txt")
else:
    print(f"Could not find 5 Thai samples (only found {len(samples)})")