#!/usr/bin/env python3
"""
Test script to load HuggingFace dataset and verify speaker ID continuation.

This script:
1. Loads the dataset from HuggingFace
2. Checks speaker IDs to ensure they don't reset to SPK_00001
3. Verifies proper continuation after resume
"""

import sys
import json
from datasets import load_dataset
from collections import Counter


def verify_speaker_ids(dataset_name: str = "Thanarit/Thai-Voice-Test-1000000"):
    """Load dataset and verify speaker IDs."""
    print(f"Loading dataset from {dataset_name}...")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        # Get first 100 samples to analyze speaker IDs
        samples = []
        speaker_ids = []
        
        for i, sample in enumerate(dataset):
            if i >= 100:  # Limit to first 100 samples
                break
            
            samples.append(sample)
            speaker_id = sample.get('speaker_id', 'UNKNOWN')
            speaker_ids.append(speaker_id)
            
            # Print first 20 samples for inspection
            if i < 20:
                print(f"Sample {i+1}: ID={sample.get('ID', 'N/A')}, Speaker={speaker_id}")
        
        # Analyze speaker ID distribution
        speaker_counts = Counter(speaker_ids)
        print(f"\n=== Speaker ID Analysis ===")
        print(f"Total samples analyzed: {len(samples)}")
        print(f"Unique speakers: {len(speaker_counts)}")
        print(f"\nSpeaker ID distribution:")
        for speaker_id, count in sorted(speaker_counts.items()):
            print(f"  {speaker_id}: {count} samples")
        
        # Check for speaker ID reset issue
        min_speaker_id = min(speaker_ids) if speaker_ids else None
        max_speaker_id = max(speaker_ids) if speaker_ids else None
        
        print(f"\nSpeaker ID range: {min_speaker_id} to {max_speaker_id}")
        
        # Verify no reset to SPK_00001 if we have checkpoint data
        try:
            with open('checkpoints/speaker_model.json', 'r') as f:
                checkpoint_data = json.load(f)
                checkpoint_counter = checkpoint_data.get('speaker_counter', 0)
                print(f"\nCheckpoint speaker counter: {checkpoint_counter}")
                
                # If checkpoint shows counter > 1, we shouldn't see SPK_00001
                if checkpoint_counter > 1 and 'SPK_00001' in speaker_ids:
                    print("⚠️  WARNING: Found SPK_00001 despite checkpoint counter > 1")
                    print("   This indicates speaker ID reset issue!")
                else:
                    print("✓ Speaker IDs appear to continue correctly from checkpoint")
        except FileNotFoundError:
            print("\nNo checkpoint file found for comparison")
        
        # Check for clustering (samples with same speaker ID)
        clustered_speakers = [sid for sid, count in speaker_counts.items() if count > 1]
        if clustered_speakers:
            print(f"\n✓ Found clustering: {len(clustered_speakers)} speakers have multiple samples")
        else:
            print("\n⚠️  No clustering detected - each sample has unique speaker ID")
        
        return samples, speaker_counts
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None


def check_sequential_processing():
    """Check if samples were processed in expected order."""
    print("\n=== Sequential Processing Check ===")
    
    # For fresh run, we expect:
    # - S1-S10: Early samples, should have low speaker IDs
    # - Resume should continue from where left off
    
    try:
        dataset = load_dataset("Thanarit/Thai-Voice-Test-1000000", split="train", streaming=True)
        
        # Get S1-S20 to check pattern
        s_samples = {}
        for i, sample in enumerate(dataset):
            if i >= 20:
                break
            
            sample_id = sample.get('ID', '')
            if sample_id.startswith('S'):
                try:
                    s_num = int(sample_id[1:])
                    if 1 <= s_num <= 20:
                        s_samples[s_num] = sample.get('speaker_id', 'UNKNOWN')
                except ValueError:
                    pass
        
        if s_samples:
            print("Sample ID to Speaker ID mapping:")
            for s_num in sorted(s_samples.keys()):
                print(f"  S{s_num}: {s_samples[s_num]}")
                
            # Check S1-S8 and S10 clustering
            s1_to_s8_and_s10 = [s_samples.get(i) for i in [1,2,3,4,5,6,7,8,10] if i in s_samples]
            s9_speaker = s_samples.get(9)
            
            if len(set(s1_to_s8_and_s10)) == 1 and s9_speaker and s9_speaker not in s1_to_s8_and_s10:
                print("\n✓ S1-S8 and S10 properly clustered together, S9 is different")
            else:
                print("\n⚠️  S1-S10 clustering pattern not as expected")
                
    except Exception as e:
        print(f"Error in sequential check: {str(e)}")


if __name__ == "__main__":
    # Run verification
    samples, speaker_counts = verify_speaker_ids()
    
    # Run sequential check
    check_sequential_processing()
    
    print("\n=== Verification Complete ===")