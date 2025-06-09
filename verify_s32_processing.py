#!/usr/bin/env python3
"""
Verify S32 processing with complete speaker separation
"""

import numpy as np
import soundfile as sf
from datasets import load_dataset
from utils.huggingface import read_hf_token
from processors.audio_enhancement.complete_separation import CompleteSeparator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_s32_processing():
    """Download and verify S32 from the uploaded dataset"""
    
    # Load the dataset
    token = read_hf_token('.hf_token')
    print("Loading dataset from HuggingFace...")
    
    try:
        dataset = load_dataset(
            "Thanarit/Thai-Voice-10000000",
            split="train",
            token=token,
            streaming=True
        )
        
        # Find S32
        print("Searching for S32...")
        s32_found = False
        
        for sample in dataset:
            if sample['ID'] == 'S32':
                s32_found = True
                print(f"\nFound S32!")
                print(f"Speaker ID: {sample['speaker_id']}")
                print(f"Length: {sample['length']:.2f} seconds")
                print(f"Dataset: {sample['dataset_name']}")
                print(f"Enhancement metadata: {sample.get('enhancement_metadata', {})}")
                
                # Save the audio
                audio_data = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']
                
                print(f"\nAudio shape: {audio_data.shape}")
                print(f"Sample rate: {sample_rate}")
                print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
                
                # Save original
                sf.write('test_audio_output/s32_uploaded.wav', audio_data, sample_rate)
                print("Saved uploaded S32 to: test_audio_output/s32_uploaded.wav")
                
                # Check enhancement metadata
                if 'enhancement_metadata' in sample and sample['enhancement_metadata']:
                    meta = sample['enhancement_metadata']
                    print(f"\nEnhancement applied: {meta.get('enhanced', False)}")
                    print(f"Enhancement level: {meta.get('enhancement_level', 'N/A')}")
                    print(f"Secondary speaker detected: {meta.get('secondary_speaker_detected', False)}")
                    print(f"Use speaker separation: {meta.get('use_speaker_separation', False)}")
                    print(f"Engine used: {meta.get('engine_used', 'N/A')}")
                
                # Test with complete separator locally
                print("\nTesting local complete separator on uploaded S32...")
                separator = CompleteSeparator()
                
                # Analyze for overlapping speakers
                analysis = separator.analyze_overlapping_speakers(audio_data, sample_rate)
                print(f"Overlapping speech detected: {analysis.has_overlapping_speech}")
                print(f"Number of speakers: {analysis.num_speakers}")
                print(f"Overlap regions: {len(analysis.overlap_regions)}")
                
                if analysis.overlap_regions:
                    print("\nOverlap regions:")
                    for i, region in enumerate(analysis.overlap_regions):
                        print(f"  Region {i+1}: {region.start_time:.2f}s - {region.end_time:.2f}s (duration: {region.duration:.2f}s)")
                
                # If overlapping detected, try separation
                if analysis.has_overlapping_speech:
                    print("\nApplying complete speaker separation...")
                    cleaned = separator.extract_primary_speaker(audio_data, sample_rate)
                    sf.write('test_audio_output/s32_reseparated.wav', cleaned, sample_rate)
                    print("Saved re-separated S32 to: test_audio_output/s32_reseparated.wav")
                    
                    # Compare energy
                    orig_energy = np.sqrt(np.mean(audio_data**2))
                    clean_energy = np.sqrt(np.mean(cleaned**2))
                    print(f"\nOriginal RMS energy: {orig_energy:.4f}")
                    print(f"Cleaned RMS energy: {clean_energy:.4f}")
                    print(f"Energy ratio: {clean_energy/orig_energy:.2f}")
                
                break
        
        if not s32_found:
            print("S32 not found in the dataset!")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying to check local files...")
        
        # Check if we have S32 locally
        import os
        if os.path.exists('test_audio_output/s32_from_dataset.wav'):
            print("\nFound local S32 file, testing with complete separator...")
            audio_data, sample_rate = sf.read('test_audio_output/s32_from_dataset.wav')
            
            separator = CompleteSeparator()
            analysis = separator.analyze_overlapping_speakers(audio_data, sample_rate)
            
            print(f"Overlapping speech detected: {analysis.has_overlapping_speech}")
            print(f"Number of speakers: {analysis.num_speakers}")
            print(f"Overlap regions: {len(analysis.overlap_regions)}")
            
            if analysis.has_overlapping_speech:
                print("\nApplying complete speaker separation...")
                cleaned = separator.extract_primary_speaker(audio_data, sample_rate)
                sf.write('test_audio_output/s32_local_separated.wav', cleaned, sample_rate)
                print("Saved separated S32 to: test_audio_output/s32_local_separated.wav")

if __name__ == "__main__":
    verify_s32_processing()