"""
Simple test to demonstrate the speaker ID bug in streaming mode.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Simulate the bug from main.py lines 447-448
def demonstrate_bug():
    """Demonstrate the speaker ID assignment bug."""
    print("\n=== Demonstrating Speaker ID Bug ===\n")
    
    # Simulate processing two datasets
    current_id = 1
    all_samples = []
    
    # Dataset 1: GigaSpeech2
    print("Processing GigaSpeech2:")
    for i in range(5):
        sample = {
            'ID': f'S{current_id}',
            'dataset_name': 'GigaSpeech2',
            'speaker_id': f'SPK_{current_id:05d}'  # BUG: Using current_id instead of actual speaker ID
        }
        print(f"  Sample {sample['ID']}: speaker_id = {sample['speaker_id']}")
        all_samples.append(sample)
        current_id += 1
    
    # Dataset 2: ProcessedVoiceTH
    print("\nProcessing ProcessedVoiceTH:")
    for i in range(5):
        sample = {
            'ID': f'S{current_id}',
            'dataset_name': 'ProcessedVoiceTH',
            'speaker_id': f'SPK_{current_id:05d}'  # BUG: Using current_id instead of actual speaker ID
        }
        print(f"  Sample {sample['ID']}: speaker_id = {sample['speaker_id']}")
        all_samples.append(sample)
        current_id += 1
    
    print("\n=== Analysis ===")
    print("The bug is that speaker IDs are assigned based on the sample ID counter (current_id)")
    print("instead of actual speaker clustering results.")
    print("\nThis means:")
    print("1. Every sample gets a unique speaker ID (no clustering)")
    print("2. Speaker IDs are just sequential numbers")
    print("3. The actual speaker identification system is bypassed")
    print("\nExpected behavior:")
    print("- Speaker IDs should be assigned by the SpeakerIdentification.process_batch() method")
    print("- Similar speakers should get the same ID across datasets")
    print("- The speaker_identifier should maintain a global counter across all datasets")
    
    # Show the problematic code from main.py
    print("\n=== Problematic Code (main.py:447-448) ===")
    print("elif not speaker_identifier:")
    print("    # No speaker identification - assign unique ID")
    print("    sample['speaker_id'] = f'SPK_{current_id:05d}'")
    print("\nThis condition is ALWAYS executed when speaker_identifier exists!")
    print("The correct logic should use the speaker_ids from process_batch()")


if __name__ == '__main__':
    demonstrate_bug()