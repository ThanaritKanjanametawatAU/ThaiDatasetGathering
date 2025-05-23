#!/usr/bin/env python3
"""
Test script to verify audio playback functionality works with the correct format.
"""

import tempfile
import os
from datasets import load_dataset
import soundfile as sf
import numpy as np

def test_audio_playback():
    """Test that the audio can be played back correctly."""
    try:
        # Load the dataset
        dataset = load_dataset("Thanarit/Thai-Voice", split="train")
        
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Get first sample
        sample = dataset[0]
        audio_data = sample['audio']
        
        print(f"Testing audio playback for sample: {sample['ID']}")
        print(f"Original sampling rate: {audio_data['sampling_rate']}")
        print(f"Audio array length: {len(audio_data['array'])}")
        print(f"Audio duration: {len(audio_data['array']) / audio_data['sampling_rate']:.2f} seconds")
        
        # Convert to numpy array
        audio_array = np.array(audio_data['array'], dtype=np.float32)
        
        # Create a temporary file to test audio writing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            # Write audio to temporary file
            sf.write(temp_path, audio_array, audio_data['sampling_rate'])
            
            # Read it back to verify
            read_audio, read_sr = sf.read(temp_path)
            
            print(f"✅ Successfully wrote and read audio file:")
            print(f"   Written file size: {os.path.getsize(temp_path)} bytes")
            print(f"   Read audio shape: {read_audio.shape}")
            print(f"   Read sampling rate: {read_sr}")
            print(f"   Audio values range: {read_audio.min():.6f} to {read_audio.max():.6f}")
            
            # Clean up
            os.unlink(temp_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error writing/reading audio file: {str(e)}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return False
            
    except Exception as e:
        print(f"❌ Error testing audio playback: {str(e)}")
        return False

if __name__ == "__main__":
    test_audio_playback()