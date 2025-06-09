#!/usr/bin/env python
"""Smart secondary speaker removal that preserves primary speaker"""

import numpy as np
import soundfile as sf

class SmartSecondaryRemoval:
    """Remove only the actual secondary speaker, not the primary speaker"""
    
    def __init__(self):
        self.min_segment_duration = 0.1  # Minimum duration to consider as separate speaker
        
    def detect_final_speaker_change(self, audio, sample_rate):
        """Detect where the final speaker change occurs"""
        # Work with last 1 second or full audio if shorter
        analysis_duration = min(1.0, len(audio) / sample_rate)
        analysis_samples = int(analysis_duration * sample_rate)
        analysis_audio = audio[-analysis_samples:]
        
        # Calculate energy in small windows
        window_size = int(0.025 * sample_rate)  # 25ms
        hop_size = int(0.010 * sample_rate)     # 10ms
        
        energies = []
        for i in range(0, len(analysis_audio) - window_size, hop_size):
            window = analysis_audio[i:i+window_size]
            energy = np.sqrt(np.mean(window**2))
            energies.append(energy)
            
        energies = np.array(energies)
        
        # Find the last significant energy increase
        if len(energies) < 3:
            return None
            
        # Calculate energy changes
        energy_diff = np.diff(energies)
        
        # Find the last significant increase
        threshold = np.std(energy_diff) * 2
        significant_increases = np.where(energy_diff > threshold)[0]
        
        if len(significant_increases) == 0:
            return None
            
        # Get the last significant increase
        last_increase_idx = significant_increases[-1]
        
        # Convert to sample position
        change_position = last_increase_idx * hop_size
        change_position_in_full = len(audio) - analysis_samples + change_position
        
        # Verify this is actually a different speaker by checking characteristics
        before_change = analysis_audio[:change_position]
        after_change = analysis_audio[change_position:]
        
        if len(before_change) < window_size or len(after_change) < window_size:
            return None
            
        # Check if energy pattern is different
        before_energy = np.sqrt(np.mean(before_change**2))
        after_energy = np.sqrt(np.mean(after_change**2))
        
        # If after is significantly louder/different, it's likely a different speaker
        if after_energy > before_energy * 1.5 or after_energy < before_energy * 0.5:
            return change_position_in_full / sample_rate
            
        return None
        
    def remove_secondary_speaker(self, audio, sample_rate):
        """Remove only the secondary speaker while preserving primary"""
        # Detect where the secondary speaker starts
        change_time = self.detect_final_speaker_change(audio, sample_rate)
        
        if change_time is None:
            print("No clear secondary speaker detected")
            return audio
            
        print(f"Secondary speaker detected at {change_time:.2f}s")
        
        # Only remove if it's in the last 0.5 seconds
        audio_duration = len(audio) / sample_rate
        if audio_duration - change_time > 0.5:
            print("Secondary speaker segment too long, might be primary speaker")
            return audio
            
        # Apply removal
        change_sample = int(change_time * sample_rate)
        result = audio.copy()
        
        # Fade out from the change point
        fade_duration = 0.05  # 50ms fade
        fade_samples = int(fade_duration * sample_rate)
        
        if change_sample > fade_samples:
            # Apply fade
            fade = np.linspace(1.0, 0.0, fade_samples)
            result[change_sample-fade_samples:change_sample] *= fade
            
        # Silence after the change
        result[change_sample:] = 0.0
        
        return result

# Test on S5
print("Loading S5 sample...")
audio, sr = sf.read("test_audio_output/s5_from_dataset.wav")
print(f"Original duration: {len(audio)/sr:.2f}s")

# Apply smart removal
remover = SmartSecondaryRemoval()
cleaned = remover.remove_secondary_speaker(audio, sr)

# Save result
sf.write("test_audio_output/s5_smart_removed.wav", cleaned, sr)
print("Saved to: test_audio_output/s5_smart_removed.wav")

# Compare
print(f"\nOriginal last 0.5s max amplitude: {np.max(np.abs(audio[-int(0.5*sr):])):.6f}")
print(f"Cleaned last 0.5s max amplitude: {np.max(np.abs(cleaned[-int(0.5*sr):])):.6f}")

# Also test a more selective approach
class SelectiveEndSilencer:
    """Only silence if there's actual speech at the very end"""
    
    def __init__(self, end_check_duration=0.2):
        self.end_check_duration = end_check_duration
        
    def process(self, audio, sample_rate):
        """Only silence if there's significant audio at the very end"""
        check_samples = int(self.end_check_duration * sample_rate)
        
        if len(audio) < check_samples:
            return audio
            
        # Check the very end
        end_segment = audio[-check_samples:]
        end_energy = np.sqrt(np.mean(end_segment**2))
        
        # Check the segment just before
        before_segment = audio[-2*check_samples:-check_samples]
        before_energy = np.sqrt(np.mean(before_segment**2))
        
        # Only silence if end has significant energy
        silence_threshold = 0.01
        if end_energy > silence_threshold:
            # Check if it's different from before (potential speaker change)
            if end_energy > before_energy * 1.5 or end_energy < before_energy * 0.5:
                print(f"Silencing last {self.end_check_duration}s due to speaker change")
                result = audio.copy()
                # Fade out
                fade_samples = int(0.05 * sample_rate)
                fade = np.linspace(1.0, 0.0, fade_samples)
                result[-check_samples-fade_samples:-check_samples] *= fade
                result[-check_samples:] = 0.0
                return result
        
        return audio

# Test selective approach
selective = SelectiveEndSilencer(end_check_duration=0.15)
selective_result = selective.process(audio, sr)
sf.write("test_audio_output/s5_selective_removed.wav", selective_result, sr)
print("\nSaved selective removal to: test_audio_output/s5_selective_removed.wav")