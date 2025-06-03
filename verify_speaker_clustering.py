#!/usr/bin/env python3
"""Verify speaker clustering in Thai-Voice-Test-1000 dataset"""

from datasets import load_dataset
import pandas as pd
from collections import Counter

def verify_speaker_clustering():
    print("Loading Thai-Voice-Test-1000 dataset...")
    
    # Load the dataset - use streaming to avoid the split size mismatch error
    dataset = load_dataset("Thanarit/Thai-Voice-Test-1000", split="train", streaming=True)
    
    # Convert streaming dataset to list
    print("Loading samples...")
    data = []
    for i, sample in enumerate(dataset):
        data.append(sample)
        if i % 100 == 0:
            print(f"  Loaded {i} samples...")
    
    # Convert to pandas for easier analysis
    df = pd.DataFrame(data)
    
    print(f"\nTotal samples in dataset: {len(df)}")
    
    # Check specific samples S1-S10
    print("\n=== Verification of S1-S10 Speaker IDs ===")
    for i in range(1, 11):
        sample_id = f"S{i}"
        sample = df[df['ID'] == sample_id]
        if not sample.empty:
            speaker_id = sample.iloc[0]['speaker_id']
            print(f"{sample_id}: speaker_id = {speaker_id}")
        else:
            print(f"{sample_id}: NOT FOUND in dataset")
    
    # Analyze overall speaker clustering
    print("\n=== Overall Speaker Clustering Analysis ===")
    speaker_counts = Counter(df['speaker_id'])
    
    print(f"Total unique speakers: {len(speaker_counts)}")
    print(f"Total samples: {len(df)}")
    print(f"Clustering ratio: {len(speaker_counts) / len(df) * 100:.2f}%")
    
    # Show top 10 speakers by sample count
    print("\nTop 10 speakers by sample count:")
    for speaker, count in speaker_counts.most_common(10):
        percentage = count / len(df) * 100
        print(f"  {speaker}: {count} samples ({percentage:.1f}%)")
    
    # Verify requirements
    print("\n=== Requirement Verification ===")
    
    # Check S1-S8 and S10 have same speaker_id
    s1_8_10_ids = []
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        sample = df[df['ID'] == f"S{i}"]
        if not sample.empty:
            s1_8_10_ids.append(sample.iloc[0]['speaker_id'])
    
    if s1_8_10_ids and all(id == s1_8_10_ids[0] for id in s1_8_10_ids):
        print(f"✅ S1-S8 and S10 all have the same speaker_id: {s1_8_10_ids[0]}")
    else:
        print("❌ S1-S8 and S10 do NOT have the same speaker_id")
        print(f"   Found IDs: {s1_8_10_ids}")
    
    # Check S9 has different speaker_id
    s9_sample = df[df['ID'] == 'S9']
    if not s9_sample.empty:
        s9_speaker = s9_sample.iloc[0]['speaker_id']
        if s1_8_10_ids and s9_speaker != s1_8_10_ids[0]:
            print(f"✅ S9 has a different speaker_id: {s9_speaker}")
        else:
            print(f"❌ S9 does NOT have a different speaker_id: {s9_speaker}")
    
    # Check clustering effectiveness
    if len(speaker_counts) / len(df) < 0.5:  # Less than 50% unique speakers
        print(f"✅ Good clustering: Only {len(speaker_counts)} unique speakers for {len(df)} samples")
    else:
        print(f"❌ Poor clustering: {len(speaker_counts)} unique speakers for {len(df)} samples")

if __name__ == "__main__":
    verify_speaker_clustering()