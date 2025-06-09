"""
Deep analysis of S5 secondary speaker issue
"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from datasets import load_dataset
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.audio_metrics import calculate_energy_db
from processors.audio_enhancement.core import AudioEnhancer
import logging

logging.basicConfig(level=logging.INFO)

def download_s5_sample():
    """Download S5 from GigaSpeech2"""
    print("Downloading S5 sample from GigaSpeech2...")
    
    # Load dataset in streaming mode (use default config and filter for Thai)
    dataset = load_dataset("speechcolab/gigaspeech2", split="train", streaming=True)
    
    # Get first 10 samples to find S5
    samples = []
    for i, sample in enumerate(dataset):
        samples.append(sample)
        if i >= 9:  # S0-S9
            break
    
    if len(samples) > 5:
        s5 = samples[5]
        print(f"Found S5: ID={s5.get('id', 'unknown')}")
        return s5['audio']['array'], s5['audio']['sampling_rate']
    else:
        raise ValueError("Could not find S5 in dataset")

def analyze_audio_profile(audio, sample_rate, window_size=0.5):
    """Analyze energy profile of audio in windows"""
    window_samples = int(window_size * sample_rate)
    hop_samples = int(0.1 * sample_rate)  # 100ms hop
    
    profile = []
    timestamps = []
    
    for i in range(0, len(audio) - window_samples, hop_samples):
        window = audio[i:i+window_samples]
        energy = calculate_energy_db(window)
        profile.append(energy)
        timestamps.append(i / sample_rate)
    
    # Also analyze the last segment
    if len(audio) > window_samples:
        last_window = audio[-window_samples:]
        last_energy = calculate_energy_db(last_window)
        profile.append(last_energy)
        timestamps.append((len(audio) - window_samples) / sample_rate)
    
    return np.array(timestamps), np.array(profile)

def detect_speaker_changes(audio, sample_rate):
    """Detect potential speaker changes using spectral analysis"""
    from scipy import signal
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024, noverlap=512)
    
    # Focus on speech frequencies (100-4000 Hz)
    speech_freq_idx = np.where((f >= 100) & (f <= 4000))[0]
    speech_power = Sxx[speech_freq_idx, :].mean(axis=0)
    
    # Detect significant changes in spectral profile
    changes = []
    for i in range(1, len(speech_power)):
        if i > 10:  # Look at longer windows
            before = speech_power[i-10:i].mean()
            after = speech_power[i:min(i+10, len(speech_power))].mean()
            
            if before > 0 and after > 0:
                ratio = after / before
                if ratio > 2.0 or ratio < 0.5:  # Significant change
                    time = t[i]
                    changes.append(time)
    
    return changes

def main():
    """Analyze S5 secondary speaker issue"""
    try:
        # Download S5
        original_audio, sample_rate = download_s5_sample()
        print(f"S5 audio shape: {original_audio.shape}, sample rate: {sample_rate}")
        
        # Save original for reference
        sf.write('s5_original.wav', original_audio, sample_rate)
        print("Saved original S5 to s5_original.wav")
        
        # Analyze original audio profile
        print("\n1. Analyzing original audio profile...")
        timestamps, energy_profile = analyze_audio_profile(original_audio, sample_rate)
        
        # Find high energy regions
        high_energy_mask = energy_profile > -30
        high_energy_times = timestamps[high_energy_mask]
        
        print(f"High energy regions: {len(high_energy_times)} windows")
        if len(high_energy_times) > 0:
            print(f"  First high energy: {high_energy_times[0]:.1f}s")
            print(f"  Last high energy: {high_energy_times[-1]:.1f}s")
        
        # Specific analysis of the end
        duration = len(original_audio) / sample_rate
        last_second = original_audio[int(-1.0 * sample_rate):]
        last_half_second = original_audio[int(-0.5 * sample_rate):]
        
        print(f"\n2. End of audio analysis (duration: {duration:.1f}s):")
        print(f"  Last 1.0s energy: {calculate_energy_db(last_second):.1f}dB")
        print(f"  Last 0.5s energy: {calculate_energy_db(last_half_second):.1f}dB")
        
        # Detect speaker changes
        print("\n3. Detecting potential speaker changes...")
        changes = detect_speaker_changes(original_audio, sample_rate)
        print(f"Found {len(changes)} potential speaker changes")
        for i, change_time in enumerate(changes[-3:]):  # Last 3 changes
            print(f"  Change {i+1}: {change_time:.2f}s")
        
        # Process with current enhancement
        print("\n4. Processing with ultra_aggressive enhancement...")
        enhancer = AudioEnhancer(
            use_gpu=False,
            enhancement_level='ultra_aggressive'
        )
        enhanced_audio = enhancer.enhance(original_audio, sample_rate)
        
        # Save enhanced
        sf.write('s5_enhanced.wav', enhanced_audio, sample_rate)
        print("Saved enhanced S5 to s5_enhanced.wav")
        
        # Analyze enhanced audio
        print("\n5. Analyzing enhanced audio...")
        enhanced_last_second = enhanced_audio[int(-1.0 * sample_rate):]
        enhanced_last_half = enhanced_audio[int(-0.5 * sample_rate):]
        
        print(f"Enhanced end analysis:")
        print(f"  Last 1.0s energy: {calculate_energy_db(enhanced_last_second):.1f}dB")
        print(f"  Last 0.5s energy: {calculate_energy_db(enhanced_last_half):.1f}dB")
        
        # Check if secondary speaker still present
        if calculate_energy_db(enhanced_last_half) > -50:
            print("\n❌ SECONDARY SPEAKER STILL PRESENT AT END!")
            print("   Energy should be < -50dB but is", calculate_energy_db(enhanced_last_half))
        else:
            print("\n✅ Secondary speaker successfully removed")
        
        # Plot energy profiles
        print("\n6. Creating visualization...")
        plt.figure(figsize=(12, 8))
        
        # Original profile
        plt.subplot(2, 1, 1)
        orig_t, orig_e = analyze_audio_profile(original_audio, sample_rate, 0.1)
        plt.plot(orig_t, orig_e, 'b-', label='Original')
        plt.axhline(y=-50, color='r', linestyle='--', label='Target (-50dB)')
        plt.title('S5 Original Audio Energy Profile')
        plt.ylabel('Energy (dB)')
        plt.legend()
        plt.grid(True)
        
        # Enhanced profile
        plt.subplot(2, 1, 2)
        enh_t, enh_e = analyze_audio_profile(enhanced_audio, sample_rate, 0.1)
        plt.plot(enh_t, enh_e, 'g-', label='Enhanced')
        plt.axhline(y=-50, color='r', linestyle='--', label='Target (-50dB)')
        plt.title('S5 Enhanced Audio Energy Profile')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (dB)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('s5_analysis.png')
        print("Saved analysis plot to s5_analysis.png")
        
    except Exception as e:
        print(f"Error analyzing S5: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()