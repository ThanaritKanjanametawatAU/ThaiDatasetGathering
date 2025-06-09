"""
Get S5 sample using main.py for analysis
"""
import subprocess
import json
import numpy as np
import soundfile as sf
from datasets import load_dataset
import sys
import os

def get_s5_using_main():
    """Run main.py to get S5 and save it"""
    print("Getting S5 using main.py...")
    
    # First, run main.py in sample mode to get S5
    cmd = [
        "python", "main.py",
        "--fresh",
        "--sample",
        "--sample-size", "10",  # Get S0-S9
        "--no-upload",  # Don't upload, just process locally
        "--output", "s5_analysis_dataset",
        "--enable-audio-enhancement",
        "--enhancement-level", "ultra_aggressive",
        "--enable-secondary-speaker-removal",
        "GigaSpeech2"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running main.py: {result.stderr}")
        return False
    
    print("Successfully processed samples")
    return True

def analyze_processed_data():
    """Load and analyze the processed dataset"""
    print("\nLoading processed dataset...")
    
    try:
        # Load the saved dataset
        dataset = load_dataset("s5_analysis_dataset")
        
        # Get S5 (index 5)
        if len(dataset['train']) > 5:
            s5 = dataset['train'][5]
            
            # Save audio files
            original_audio = s5['audio']['array']
            sample_rate = s5['audio']['sampling_rate']
            
            print(f"S5 info:")
            print(f"  ID: {s5.get('ID', 'unknown')}")
            print(f"  Transcript: {s5.get('transcript', 'N/A')}")
            print(f"  Duration: {len(original_audio) / sample_rate:.2f}s")
            
            # Save the audio
            sf.write('s5_processed.wav', original_audio, sample_rate)
            print(f"Saved S5 to s5_processed.wav")
            
            return True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

def main():
    if get_s5_using_main():
        analyze_processed_data()
    else:
        print("Failed to get S5")

if __name__ == "__main__":
    main()