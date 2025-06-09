#!/usr/bin/env python
"""Verify if S5 was uploaded with secondary speaker removed"""

from datasets import load_dataset
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import time

# Wait a bit for upload to complete
print("Waiting for upload to complete...")
time.sleep(5)

# Load the dataset
print("\nLoading dataset...")
try:
    ds = load_dataset('Thanarit/Thai-Voice-10000000', split='train', streaming=True)
    
    # Find sample 5
    found = False
    for i, sample in enumerate(ds):
        if sample['ID'] == 'S5':
            print(f'Found sample S5')
            
            # Extract audio
            audio = np.array(sample["audio"]["array"])
            sr = sample["audio"]["sampling_rate"]
            
            # Save the audio
            output_path = "test_audio_output/s5_final_check.wav"
            sf.write(output_path, audio, sr)
            print(f"Saved audio to: {output_path}")
            
            # Analyze the last 1 second
            last_second = audio[-sr:]
            
            # Calculate energy in last second
            energy = np.sqrt(np.mean(last_second**2))
            energy_db = 20 * np.log10(energy + 1e-10)
            max_amp = np.max(np.abs(last_second))
            
            print(f"\nAnalysis of last 1 second:")
            print(f"Energy: {energy:.6f} ({energy_db:.1f} dB)")
            print(f"Max amplitude: {max_amp:.6f}")
            
            # Check if secondary speaker is removed
            if max_amp < 0.01:
                print("\n✓ SUCCESS: Secondary speaker has been removed!")
            else:
                print("\n✗ FAILED: Secondary speaker is still present!")
                
            found = True
            break
            
        if i >= 10:
            break
            
    if not found:
        print("Sample S5 not found in first 10 samples")
        
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("The dataset might still be processing. Try again in a few seconds.")