#!/usr/bin/env python3
"""
Extract a 5-second segment from S32 for detailed comparison
"""

import numpy as np
import soundfile as sf

def extract_segment():
    """Extract a 5-second segment from the middle of S32"""
    
    files = {
        'uploaded': 'test_audio_output/s32_uploaded.wav',
        'separated': 'test_audio_output/s32_reseparated.wav',
        'original': 'test_audio_output/s32_from_dataset.wav'
    }
    
    # Extract from 7-12 seconds (middle of the 19.74s file)
    start_time = 7.0
    duration = 5.0
    
    for name, filepath in files.items():
        try:
            audio, sr = sf.read(filepath)
            start_sample = int(start_time * sr)
            end_sample = int((start_time + duration) * sr)
            
            segment = audio[start_sample:end_sample]
            
            output_path = f'test_audio_output/s32_{name}_segment.wav'
            sf.write(output_path, segment, sr)
            print(f"Extracted {name} segment: {output_path}")
            
            # Calculate some quick stats
            rms = np.sqrt(np.mean(segment**2))
            max_amp = np.max(np.abs(segment))
            print(f"  RMS: {rms:.4f}, Max amplitude: {max_amp:.4f}")
            
        except Exception as e:
            print(f"Could not process {name}: {e}")
    
    print("\nYou can now listen to these segments to hear the difference:")
    print("- s32_uploaded_segment.wav (what was uploaded to HuggingFace)")
    print("- s32_separated_segment.wav (re-separated with complete separator)")
    print("- s32_original_segment.wav (original before any processing)")

if __name__ == "__main__":
    extract_segment()