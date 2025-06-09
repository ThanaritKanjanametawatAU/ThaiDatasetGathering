"""
Analyze S5 from JSON output
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, '.')
from utils.audio_metrics import calculate_energy_db
import soundfile as sf

def analyze_s5_json():
    """Analyze S5 from the JSON output"""
    json_path = "/tmp/s5_test_output/combined_dataset.json"
    
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return False
    
    print(f"Loading data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Find S5
    s5_found = False
    for i, sample in enumerate(data):
        if sample.get('ID') == 'S5' or i == 5:
            print(f"\n=== S5 ANALYSIS ===")
            print(f"Index: {i}")
            print(f"ID: {sample.get('ID', 'N/A')}")
            print(f"Transcript: {sample.get('transcript', 'N/A')[:50]}...")
            print(f"Dataset: {sample.get('dataset_name', 'N/A')}")
            
            # Check audio structure
            audio_info = sample.get('audio', {})
            print(f"Audio info type: {type(audio_info)}")
            
            if isinstance(audio_info, str):
                print(f"Audio string (serialized dict): {audio_info[:100]}...")
                # Try to parse it as a serialized dict
                try:
                    import ast
                    # This is a serialized numpy array, we need to handle it differently
                    # For now, let's skip the complex parsing
                    print("Audio data is serialized, cannot analyze directly")
                except:
                    pass
            elif isinstance(audio_info, dict):
                print(f"Audio info keys: {audio_info.keys()}")
            
            # Check if audio data is embedded
            if isinstance(audio_info, dict) and 'array' in audio_info:
                audio = np.array(audio_info['array'])
                sr = audio_info.get('sampling_rate', 16000)
                print(f"Audio array found: {len(audio)/sr:.2f}s at {sr}Hz")
                
                # Analyze end energy
                print("\nEnd energy analysis:")
                for window in [0.25, 0.5, 1.0]:
                    if len(audio) / sr > window:
                        segment = audio[int(-window * sr):]
                        energy = calculate_energy_db(segment)
                        print(f"  Last {window}s: {energy:.1f}dB", end="")
                        
                        if energy > -50:
                            print(" ❌ SECONDARY SPEAKER PRESENT!")
                        else:
                            print(" ✅ Secondary speaker removed")
                
                # Save a copy for manual inspection
                output_path = "/tmp/s5_test_output/s5_final.wav"
                sf.write(output_path, audio, sr)
                print(f"\nSaved to: {output_path}")
            else:
                print("No audio path in sample")
            
            # Check enhancement metadata
            if 'enhancement_metadata' in sample:
                meta = sample['enhancement_metadata']
                print(f"\nEnhancement metadata:")
                print(f"  Enhanced: {meta.get('enhanced', False)}")
                print(f"  Level: {meta.get('enhancement_level', 'N/A')}")
                print(f"  Secondary detected: {meta.get('secondary_speaker_detected', False)}")
                print(f"  Secondary removed: {meta.get('secondary_speaker_removed', False)}")
                
            s5_found = True
            break
    
    if not s5_found:
        print("\nS5 not found in dataset!")
        # Show all IDs
        ids = [s.get('ID', f'idx_{i}') for i, s in enumerate(data)]
        print(f"Available IDs: {ids}")
    
    return True

if __name__ == "__main__":
    analyze_s5_json()