#!/usr/bin/env python3
"""Download S32 sample from HuggingFace dataset"""

from datasets import load_dataset
import soundfile as sf
import numpy as np

print("Downloading S32 from HuggingFace...")
ds = load_dataset('Thanarit/Thai-Voice-10000000', split='train', streaming=True)

for sample in ds:
    if sample['ID'] == 'S32':
        print(f"Found S32: duration={sample['length']:.2f}s")
        print(f"Transcript: {sample['transcript']}")
        
        # Extract audio
        audio = np.array(sample["audio"]["array"])
        sr = sample["audio"]["sampling_rate"]
        
        # Save
        output_path = "test_audio_output/s32_from_dataset.wav"
        sf.write(output_path, audio, sr)
        print(f"Saved to: {output_path}")
        break