#!/usr/bin/env python3
"""Verify dataset functionality and speaker IDs."""

from datasets import load_dataset
from collections import Counter
import pandas as pd

def main():
    print("Loading dataset from HuggingFace...")
    try:
        dataset = load_dataset("Thanarit/Thai-Voice", split="train", streaming=True)
        
        # Take first 150 samples (what we just uploaded)
        samples = list(dataset.take(150))
        
        print(f"\nTotal samples loaded: {len(samples)}")
        
        # Check basic fields
        if samples:
            print(f"\nSample fields: {list(samples[0].keys())}")
            
            # Check speaker distribution
            speaker_counts = Counter(s['speaker_id'] for s in samples)
            print(f"\nSpeaker distribution:")
            for speaker, count in sorted(speaker_counts.items()):
                print(f"  {speaker}: {count} samples")
            
            # Check dataset distribution
            dataset_counts = Counter(s['dataset_name'] for s in samples)
            print(f"\nDataset distribution:")
            for dataset, count in sorted(dataset_counts.items()):
                print(f"  {dataset}: {count} samples")
            
            # Verify audio format
            print(f"\nAudio format check:")
            sample = samples[0]
            if 'audio' in sample:
                audio = sample['audio']
                print(f"  Audio type: {type(audio)}")
                if isinstance(audio, dict):
                    print(f"  Audio keys: {list(audio.keys())}")
                    if 'array' in audio:
                        print(f"  Array shape: {audio['array'].shape if hasattr(audio['array'], 'shape') else len(audio['array'])}")
                    if 'sampling_rate' in audio:
                        print(f"  Sampling rate: {audio['sampling_rate']}")
                    if 'path' in audio:
                        print(f"  Path: {audio['path']}")
            
            # Check for any missing transcripts
            missing_transcripts = sum(1 for s in samples if not s.get('transcript'))
            print(f"\nSamples with missing transcripts: {missing_transcripts}")
            
            # Verify all required fields
            required_fields = ['ID', 'speaker_id', 'Language', 'audio', 'transcript', 'length', 'dataset_name', 'confidence_score']
            missing_fields = []
            for field in required_fields:
                if field not in samples[0]:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"\nMissing required fields: {missing_fields}")
            else:
                print(f"\nAll required fields present ✓")
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    print("\n✓ Dataset verification complete")
    return 0

if __name__ == "__main__":
    exit(main())