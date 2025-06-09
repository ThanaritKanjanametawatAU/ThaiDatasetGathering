"""
Verify S5 on Hugging Face dataset
"""
from datasets import load_dataset
import numpy as np
from utils.audio_metrics import calculate_energy_db
import soundfile as sf

def verify_s5_on_hf():
    """Check S5 on the uploaded Hugging Face dataset"""
    print("Loading dataset from Hugging Face...")
    
    try:
        # Load the dataset in streaming mode
        dataset = load_dataset("Thanarit/Thai-Voice-10000000", split="train", streaming=True)
        
        # Get first 10 samples
        samples = []
        for i, sample in enumerate(dataset):
            samples.append(sample)
            if i >= 9:  # Get S0-S9
                break
        
        print(f"Loaded {len(samples)} samples")
        
        # Find S5
        s5 = None
        for i, sample in enumerate(samples):
            if sample.get('ID') == 'S5' or i == 5:
                s5 = sample
                print(f"\nFound S5 at index {i}")
                break
        
        if s5:
            print("\n=== S5 ANALYSIS ===")
            print(f"ID: {s5.get('ID', 'N/A')}")
            print(f"Transcript: {s5.get('transcript', 'N/A')[:50]}...")
            print(f"Dataset: {s5.get('dataset_name', 'N/A')}")
            
            # Get audio
            audio = s5['audio']['array']
            sr = s5['audio']['sampling_rate']
            
            print(f"\nAudio: {len(audio)/sr:.2f}s at {sr}Hz")
            
            # Analyze end energy
            print("\nEnd energy analysis:")
            for window in [0.25, 0.5, 1.0]:
                if len(audio) / sr > window:
                    segment = audio[int(-window * sr):]
                    energy = calculate_energy_db(segment)
                    print(f"  Last {window}s: {energy:.1f}dB", end="")
                    
                    if energy > -50:
                        print(" ❌ SECONDARY SPEAKER STILL PRESENT!")
                    else:
                        print(" ✅ Secondary speaker removed")
            
            # Save for manual inspection
            sf.write('s5_from_hf.wav', audio, sr)
            print(f"\nSaved S5 to: s5_from_hf.wav")
            
            # Check if enhancement metadata exists
            if 'enhancement_metadata' in s5:
                print(f"\nEnhancement metadata found:")
                meta = s5['enhancement_metadata']
                for key, value in meta.items():
                    print(f"  {key}: {value}")
            else:
                print("\nNo enhancement metadata found")
                
            return True
        else:
            print("S5 not found in first 10 samples")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = verify_s5_on_hf()
    
    if success:
        print("\n✅ Successfully analyzed S5 from Hugging Face")
    else:
        print("\n❌ Failed to analyze S5")