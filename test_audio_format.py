#!/usr/bin/env python3
"""
Test script to verify the audio format is correct for HuggingFace preview.
"""

import logging
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_format():
    """Test that the uploaded dataset has proper audio format."""
    try:
        # Load the dataset
        dataset = load_dataset("Thanarit/Thai-Voice", split="train")
        
        print(f"Dataset loaded with {len(dataset)} samples")
        print(f"Features: {list(dataset.features.keys())}")
        print(f"Audio feature type: {dataset.features['audio']}")
        
        # Check first sample
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"ID: {sample['ID']}")
        print(f"Language: {sample['Language']}")
        print(f"Transcript: {sample['transcript']}")
        print(f"Length: {sample['length']}")
        
        # Check audio format
        audio = sample['audio']
        print(f"\nAudio format:")
        print(f"Audio type: {type(audio)}")
        if isinstance(audio, dict):
            print(f"Audio keys: {list(audio.keys())}")
            if 'array' in audio:
                array = audio['array']
                print(f"Array type: {type(array)}")
                print(f"Array shape: {array.shape if hasattr(array, 'shape') else len(array)}")
                print(f"Array dtype: {array.dtype if hasattr(array, 'dtype') else type(array[0])}")
            if 'sampling_rate' in audio:
                print(f"Sampling rate: {audio['sampling_rate']}")
            if 'path' in audio:
                print(f"Path: {audio['path']}")
        
        print("\n✅ Audio format test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing audio format: {str(e)}")
        return False

if __name__ == "__main__":
    test_audio_format()