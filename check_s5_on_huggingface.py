"""
Check S5 on the uploaded Huggingface dataset
"""
from datasets import load_dataset
import numpy as np
import soundfile as sf
from utils.audio_metrics import calculate_energy_db
import matplotlib.pyplot as plt

def analyze_huggingface_s5():
    """Download and analyze S5 from Huggingface"""
    print("Loading Thai-Voice dataset from Huggingface...")
    
    try:
        # Load the dataset
        dataset = load_dataset("Thanarit/Thai-Voice", split="train", streaming=True)
        
        # Get first 10 samples
        samples = []
        for i, sample in enumerate(dataset):
            samples.append(sample)
            print(f"Sample {i}: {sample.get('ID', 'unknown')}")
            if i >= 9:  # S0-S9
                break
        
        if len(samples) > 5:
            s5 = samples[5]
            print(f"\nAnalyzing S5: {s5.get('ID', 'unknown')}")
            
            # Extract audio
            audio = s5['audio']['array']
            sample_rate = s5['audio']['sampling_rate']
            
            # Save audio
            sf.write('s5_from_huggingface.wav', audio, sample_rate)
            print(f"Saved S5 to s5_from_huggingface.wav")
            
            # Analyze the end of audio
            duration = len(audio) / sample_rate
            print(f"\nS5 Analysis:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Sample rate: {sample_rate}Hz")
            
            # Check last segments
            for window in [0.25, 0.5, 1.0]:
                if duration > window:
                    segment = audio[int(-window * sample_rate):]
                    energy = calculate_energy_db(segment)
                    print(f"  Last {window}s energy: {energy:.1f}dB")
                    
                    # Check if it's above threshold
                    if energy > -50:
                        print(f"    ⚠️  SECONDARY SPEAKER DETECTED (>{-50}dB)")
            
            # Plot energy profile
            window_size = 0.1  # 100ms windows
            hop_size = 0.05   # 50ms hop
            
            window_samples = int(window_size * sample_rate)
            hop_samples = int(hop_size * sample_rate)
            
            energies = []
            times = []
            
            for i in range(0, len(audio) - window_samples, hop_samples):
                window = audio[i:i+window_samples]
                energy = calculate_energy_db(window)
                energies.append(energy)
                times.append(i / sample_rate)
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(times, energies, 'b-', linewidth=1)
            plt.axhline(y=-50, color='r', linestyle='--', label='Threshold (-50dB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy (dB)')
            plt.title(f'S5 Energy Profile (ID: {s5.get("ID", "unknown")})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Highlight the end region
            if duration > 1.0:
                plt.axvspan(duration - 1.0, duration, alpha=0.2, color='red', label='Last 1s')
            
            plt.tight_layout()
            plt.savefig('s5_huggingface_profile.png')
            print(f"\nSaved energy profile to s5_huggingface_profile.png")
            
            return audio, sample_rate
            
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    analyze_huggingface_s5()