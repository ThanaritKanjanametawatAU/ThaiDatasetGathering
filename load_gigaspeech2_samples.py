#!/usr/bin/env python3
"""
Load GigaSpeech2 Dataset Samples
Properly loads audio from the GigaSpeech2 dataset
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from datasets import load_dataset
import torch

print("Loading GigaSpeech2 Dataset Samples")
print("=" * 80)

def load_gigaspeech2_samples(num_samples=50):
    """Load samples from GigaSpeech2 dataset"""
    
    print(f"Loading {num_samples} samples from GigaSpeech2 dataset...")
    
    try:
        # Load GigaSpeech2 dataset
        # The dataset ID might be different, let's try various options
        dataset_names = [
            "speechcolab/gigaspeech2",
            "speechcolab/gigaspeech",
            "gigaspeech/gigaspeech",
            "esb/datasets",  # ESB has GigaSpeech
        ]
        
        dataset = None
        dataset_name_used = None
        
        for dataset_name in dataset_names:
            try:
                print(f"  Trying {dataset_name}...")
                
                # Try loading with different configurations
                if dataset_name == "esb/datasets":
                    # ESB dataset has GigaSpeech as a subset
                    dataset = load_dataset(dataset_name, "gigaspeech", split="test", streaming=True)
                else:
                    # Try default configuration first
                    dataset = load_dataset(dataset_name, split="train", streaming=True)
                
                dataset_name_used = dataset_name
                print(f"  ✓ Successfully connected to {dataset_name}")
                break
                
            except Exception as e:
                print(f"  ✗ {dataset_name} failed: {str(e)[:100]}...")
                continue
        
        if dataset is None:
            print("\nTrying alternative approach - loading specific language subset...")
            # Try language-specific subsets
            languages = ["th", "thai", "en", "zh", None]
            
            for lang in languages:
                try:
                    if lang:
                        print(f"  Trying speechcolab/gigaspeech2 with language '{lang}'...")
                        dataset = load_dataset("speechcolab/gigaspeech2", lang, split="train", streaming=True)
                    else:
                        print(f"  Trying speechcolab/gigaspeech2 with no language specified...")
                        dataset = load_dataset("speechcolab/gigaspeech2", split="train", streaming=True)
                    
                    dataset_name_used = f"speechcolab/gigaspeech2 ({lang or 'default'})"
                    print(f"  ✓ Successfully connected!")
                    break
                    
                except Exception as e:
                    continue
        
        if dataset is None:
            raise Exception("Could not load GigaSpeech2 dataset with any configuration")
        
        # Process samples
        samples = []
        print(f"\nProcessing samples from {dataset_name_used}...")
        
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            
            try:
                # Debug: print the structure of the first item
                if i == 0:
                    print(f"\nDataset item structure:")
                    print(f"  Type: {type(item)}")
                    if isinstance(item, dict):
                        print(f"  Keys: {list(item.keys())}")
                        # Print audio field structure
                        if 'audio' in item:
                            print(f"  Audio type: {type(item['audio'])}")
                            if isinstance(item['audio'], dict):
                                print(f"  Audio keys: {list(item['audio'].keys())}")
                        if 'wav' in item:
                            print(f"  Wav type: {type(item['wav'])}")
                            if isinstance(item['wav'], bytes):
                                print(f"  Wav is bytes, length: {len(item['wav'])}")
                            elif isinstance(item['wav'], dict):
                                print(f"  Wav keys: {list(item['wav'].keys())}")
                
                # Extract audio data - handle different possible formats
                audio_data = None
                sample_rate = None
                
                # Try different field names and structures
                if isinstance(item, dict):
                    # GigaSpeech2 uses 'wav' field
                    if 'wav' in item:
                        wav_data = item['wav']
                        if isinstance(wav_data, dict):
                            audio_data = wav_data.get('array', wav_data.get('data'))
                            sample_rate = wav_data.get('sampling_rate', wav_data.get('sample_rate', 16000))
                        elif isinstance(wav_data, (list, np.ndarray)):
                            audio_data = wav_data
                            sample_rate = 16000  # Default sample rate
                        elif isinstance(wav_data, bytes):
                            # Decode bytes to audio
                            import io
                            audio_data, sample_rate = sf.read(io.BytesIO(wav_data))
                    
                    # Standard format with 'audio' field
                    elif 'audio' in item:
                        if isinstance(item['audio'], dict):
                            # Nested structure
                            audio_data = item['audio'].get('array', item['audio'].get('data'))
                            sample_rate = item['audio'].get('sampling_rate', item['audio'].get('sample_rate', 16000))
                        else:
                            # Direct array
                            audio_data = item['audio']
                            sample_rate = item.get('sampling_rate', item.get('sample_rate', 16000))
                    
                    # Alternative field names
                    elif 'waveform' in item:
                        audio_data = item['waveform']
                        sample_rate = item.get('sample_rate', 16000)
                    
                    elif 'speech' in item:
                        audio_data = item['speech']
                        sample_rate = item.get('sampling_rate', 16000)
                
                # Handle torch tensors
                if torch.is_tensor(audio_data):
                    audio_data = audio_data.numpy()
                
                # Ensure we have audio data
                if audio_data is None:
                    print(f"  Warning: No audio data found in sample {i+1}")
                    continue
                
                # Convert to numpy array if needed
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data)
                
                # Ensure 1D audio
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.squeeze()
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data[0]  # Take first channel
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    print(f"  Resampling from {sample_rate}Hz to 16000Hz...")
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                
                # Normalize audio
                audio_data = audio_data.astype(np.float32)
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.95
                
                # Get metadata
                text = item.get('text', item.get('transcript', item.get('sentence', '')))
                sample_id = item.get('id', item.get('audio_id', f'gigaspeech_{i+1:04d}'))
                
                samples.append({
                    'audio': audio_data,
                    'sr': sample_rate,
                    'text': text,
                    'id': sample_id,
                    'duration': len(audio_data) / sample_rate
                })
                
                print(f"  Loaded sample {i+1}: {len(audio_data)/sample_rate:.2f}s, id={sample_id}")
                
            except Exception as e:
                print(f"  Error processing sample {i+1}: {e}")
                continue
        
        print(f"\n✓ Successfully loaded {len(samples)} GigaSpeech2 samples")
        return samples
        
    except Exception as e:
        print(f"\n✗ Failed to load GigaSpeech2 dataset: {e}")
        
        # Try to get more information about available datasets
        print("\nChecking available datasets...")
        try:
            from huggingface_hub import list_datasets
            
            print("Searching for GigaSpeech datasets on HuggingFace...")
            datasets = list(list_datasets(filter="gigaspeech", limit=10))
            
            if datasets:
                print("\nFound these GigaSpeech-related datasets:")
                for ds in datasets:
                    print(f"  - {ds.id}")
            else:
                print("No GigaSpeech datasets found in search")
                
        except:
            pass
        
        return None

def save_samples(samples, output_dir="gigaspeech2_samples"):
    """Save samples to disk"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nSaving {len(samples)} samples to {output_dir}/...")
    
    # Save metadata
    metadata = []
    
    for i, sample in enumerate(samples):
        # Save audio file
        filename = f"gigaspeech2_{i+1:04d}.wav"
        filepath = output_path / filename
        
        sf.write(filepath, sample['audio'], sample['sr'], subtype='PCM_16')
        
        # Add to metadata
        metadata.append({
            'filename': filename,
            'id': sample['id'],
            'text': sample['text'],
            'duration': sample['duration'],
            'sample_rate': sample['sr']
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(samples)} samples")
    
    # Save metadata as JSON
    import json
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved {len(samples)} samples to {output_dir}/")
    print(f"✓ Metadata saved to {metadata_path}")

def main():
    """Main function"""
    
    # Try to load samples
    samples = load_gigaspeech2_samples(num_samples=50)
    
    if samples:
        # Save to disk
        save_samples(samples)
        
        print("\n" + "="*80)
        print("GIGASPEECH2 SAMPLES LOADED SUCCESSFULLY!")
        print("="*80)
        print(f"\n✓ {len(samples)} samples loaded from GigaSpeech2")
        print("✓ All samples resampled to 16kHz")
        print("✓ Audio normalized to prevent clipping")
        print("✓ Samples saved to gigaspeech2_samples/")
        
    else:
        print("\n" + "="*80)
        print("FAILED TO LOAD GIGASPEECH2")
        print("="*80)
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Verify HuggingFace credentials if the dataset is gated")
        print("3. Try a different dataset configuration")
        print("4. Check if the dataset name has changed")
        
        # Suggest alternative approach
        print("\nAlternative: You can manually download GigaSpeech2 samples and place them in a folder")

if __name__ == "__main__":
    main()