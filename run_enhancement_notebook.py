#!/usr/bin/env python3
"""
Run the intelligent audio enhancement notebook code
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Audio, display, HTML

# Add project root to path
project_root = Path('/media/ssd1/SparkVoiceProject/ThaiDatasetGathering')
sys.path.append(str(project_root))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Running Intelligent Audio Enhancement on 5 Samples")
print("=" * 80)

# Initialize enhancement methods
print("\n1. Initializing enhancement methods...")

# Method 1: SpeechBrain MetricGAN+
try:
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    speechbrain_enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus"
    )
    SPEECHBRAIN_AVAILABLE = True
    print("✓ SpeechBrain MetricGAN+ loaded successfully")
except Exception as e:
    print(f"❌ SpeechBrain not available: {e}")
    SPEECHBRAIN_AVAILABLE = False

# Method 2: NoiseReduce
try:
    import noisereduce as nr
    from scipy import signal
    from scipy.signal import butter, filtfilt, wiener
    NOISEREDUCE_AVAILABLE = True
    print("✓ NoiseReduce + Advanced Spectral loaded successfully")
except Exception as e:
    print(f"❌ NoiseReduce not available: {e}")
    NOISEREDUCE_AVAILABLE = False

# Method 3: Custom methods
try:
    import librosa
    from scipy.signal import medfilt
    CUSTOM_METHODS_AVAILABLE = True
    print("✓ Custom spectral methods loaded successfully")
except Exception as e:
    print(f"❌ Custom methods not available: {e}")
    CUSTOM_METHODS_AVAILABLE = False

# Enhancement class
class IntelligentAudioEnhancer:
    """Three proven audio enhancement methods"""
    
    def __init__(self):
        self.device = device
        
    def method1_speechbrain_metricgan(self, audio, sr):
        if not SPEECHBRAIN_AVAILABLE:
            return audio, {"success": False, "error": "SpeechBrain not available"}
        
        try:
            start_time = time.time()
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, sr)
                enhanced = speechbrain_enhancer.enhance_file(tmp.name)
                enhanced = enhanced.squeeze().cpu().numpy()
                
                if len(enhanced) != len(audio):
                    enhanced = librosa.util.fix_length(enhanced, size=len(audio))
                
                os.unlink(tmp.name)
            
            processing_time = time.time() - start_time
            
            return enhanced, {
                "success": True,
                "method": "SpeechBrain MetricGAN+",
                "processing_time": processing_time,
                "rtf": processing_time / (len(audio) / sr)
            }
        except Exception as e:
            return audio, {"success": False, "error": str(e)}
    
    def method2_noisereduce_advanced(self, audio, sr):
        if not NOISEREDUCE_AVAILABLE:
            return audio, {"success": False, "error": "NoiseReduce not available"}
        
        try:
            start_time = time.time()
            
            # NoiseReduce
            enhanced = nr.reduce_noise(
                y=audio, sr=sr, stationary=False,
                prop_decrease=0.8, time_constant_s=2.0, freq_mask_smooth_hz=500
            )
            
            # Wiener filter
            if len(enhanced) > 3:
                enhanced = wiener(enhanced, mysize=5)
            
            # Bandpass filter
            nyquist = sr / 2
            low = max(0.001, 80 / nyquist)
            high = min(0.999, min(8000, nyquist - 100) / nyquist)
            
            if low < high:
                b, a = butter(4, [low, high], btype='band')
                enhanced = filtfilt(b, a, enhanced)
            
            # Spectral subtraction
            stft = librosa.stft(enhanced, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            noise_profile = np.percentile(magnitude, 20, axis=1, keepdims=True)
            enhanced_magnitude = magnitude - 1.5 * noise_profile
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced = librosa.istft(enhanced_stft, hop_length=512)
            
            if len(enhanced) != len(audio):
                enhanced = librosa.util.fix_length(enhanced, size=len(audio))
            
            processing_time = time.time() - start_time
            
            return enhanced, {
                "success": True,
                "method": "NoiseReduce + Advanced",
                "processing_time": processing_time,
                "rtf": processing_time / (len(audio) / sr)
            }
        except Exception as e:
            return audio, {"success": False, "error": str(e)}
    
    def method3_spectral_gating(self, audio, sr):
        if not CUSTOM_METHODS_AVAILABLE:
            return audio, {"success": False, "error": "Custom methods not available"}
        
        try:
            start_time = time.time()
            
            # STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # Noise profile
            frame_energy = np.mean(magnitude_db, axis=0)
            noise_frames = magnitude_db[:, frame_energy < np.percentile(frame_energy, 10)]
            
            if noise_frames.shape[1] > 0:
                noise_profile_db = np.mean(noise_frames, axis=1, keepdims=True)
            else:
                noise_profile_db = np.percentile(magnitude_db, 10, axis=1, keepdims=True)
            
            # Gate
            gate_threshold_db = noise_profile_db + 6
            gate = np.zeros_like(magnitude_db)
            
            for i in range(magnitude_db.shape[0]):
                for j in range(magnitude_db.shape[1]):
                    if magnitude_db[i, j] > gate_threshold_db[i, 0] + 3:
                        gate[i, j] = 1.0
                    elif magnitude_db[i, j] > gate_threshold_db[i, 0]:
                        gate[i, j] = (magnitude_db[i, j] - gate_threshold_db[i, 0]) / 3
                    else:
                        gate[i, j] = 0.1
            
            # Smooth gate
            for i in range(gate.shape[0]):
                gate[i, :] = medfilt(gate[i, :], kernel_size=5)
            
            # Apply gate
            gated_magnitude = magnitude * gate
            
            # Temporal smoothing
            for i in range(1, gated_magnitude.shape[1]):
                gated_magnitude[:, i] = 0.7 * gated_magnitude[:, i] + 0.3 * gated_magnitude[:, i-1]
            
            # Reconstruct
            gated_stft = gated_magnitude * np.exp(1j * phase)
            enhanced = librosa.istft(gated_stft, hop_length=512)
            
            # High-pass filter
            if sr > 1000:
                b, a = butter(2, 60 / (sr/2), btype='high')
                enhanced = filtfilt(b, a, enhanced)
            
            if len(enhanced) != len(audio):
                enhanced = librosa.util.fix_length(enhanced, size=len(audio))
            
            processing_time = time.time() - start_time
            
            return enhanced, {
                "success": True,
                "method": "Advanced Spectral Gating",
                "processing_time": processing_time,
                "rtf": processing_time / (len(audio) / sr)
            }
        except Exception as e:
            return audio, {"success": False, "error": str(e)}

# Initialize enhancer
enhancer = IntelligentAudioEnhancer()
print("✓ Enhancer initialized")

# Load samples
print("\n2. Loading GigaSpeech2 samples...")

def load_test_samples(num_samples=5):
    """Load test samples from GigaSpeech2"""
    try:
        from processors.gigaspeech2 import GigaSpeech2Processor
        
        config = {
            "name": "GigaSpeech2",
            "source": "speechcolab/gigaspeech2", 
            "language_filter": "th",
            "description": "GigaSpeech2 Thai language dataset"
        }
        
        processor = GigaSpeech2Processor(config)
        samples = []
        
        for sample in processor._process_split_streaming('train', sample_mode=True, sample_size=num_samples):
            if len(samples) >= num_samples:
                break
            samples.append(sample)
            print(f"  Loaded sample {len(samples)}/{num_samples}")
        
        print(f"✓ Successfully loaded {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"❌ Error loading samples: {e}")
        # Create synthetic samples
        return create_synthetic_samples(num_samples)

def create_synthetic_samples(num_samples=5):
    """Create synthetic test samples"""
    print("Creating synthetic test samples...")
    samples = []
    sr = 16000
    
    noise_types = ["Wind noise", "Background speaker", "Static/hum", "White noise", "Mixed noise"]
    
    for i in range(num_samples):
        duration = np.random.uniform(3.0, 5.0)
        t = np.linspace(0, duration, int(duration * sr))
        
        # Speech-like signal
        speech = np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
        speech *= np.exp(-0.3 * t)
        
        # Add different noise types
        if i == 0:  # Wind
            noise = 0.1 * np.random.normal(0, 1, len(t))
            noise += 0.05 * np.sin(2 * np.pi * 50 * t)
        elif i == 1:  # Background speaker
            background = 0.3 * np.sin(2 * np.pi * 300 * t) * (1 + np.sin(2 * np.pi * 2 * t))
            noise = background + 0.05 * np.random.normal(0, 1, len(t))
        elif i == 2:  # Static/hum
            noise = 0.1 * np.sin(2 * np.pi * 60 * t)
            noise += 0.05 * np.random.normal(0, 1, len(t))
        elif i == 3:  # White noise
            noise = 0.15 * np.random.normal(0, 1, len(t))
        else:  # Mixed
            noise = 0.05 * np.random.normal(0, 1, len(t))
            noise += 0.05 * np.sin(2 * np.pi * 100 * t)
        
        audio = speech + noise
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        sample = {
            'audio': {'array': audio.astype(np.float32), 'sampling_rate': sr},
            'transcript': f'Test Thai speech sample {i+1}',
            'speaker_id': f'TEST_{i+1:03d}',
            'ID': f'test_{i+1}',
            'length': duration,
            'noise_type': noise_types[i]
        }
        
        samples.append(sample)
    
    print(f"✓ Created {len(samples)} synthetic samples")
    return samples

# Load samples
test_samples = load_test_samples(5)

# Process samples
print("\n3. Processing samples with enhancement methods...")
enhancement_results = []

for i, sample in enumerate(test_samples):
    print(f"\n{'='*60}")
    print(f"Processing Sample {i+1}/5")
    print(f"{'='*60}")
    
    # Extract audio
    if isinstance(sample['audio'], dict) and 'array' in sample['audio']:
        original_audio = sample['audio']['array']
        sample_rate = sample['audio'].get('sampling_rate', 16000)
    else:
        continue
    
    original_audio = original_audio.astype(np.float32)
    duration = len(original_audio) / sample_rate
    noise_type = sample.get('noise_type', 'Real audio')
    
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Type: {noise_type}")
    
    result = {
        'sample_id': i + 1,
        'original_audio': original_audio,
        'sample_rate': sample_rate,
        'duration': duration,
        'transcript': sample.get('transcript', 'N/A'),
        'noise_type': noise_type,
        'enhancements': {}
    }
    
    # Test each method
    methods = [
        ('SpeechBrain MetricGAN+', enhancer.method1_speechbrain_metricgan),
        ('NoiseReduce + Advanced', enhancer.method2_noisereduce_advanced),
        ('Advanced Spectral Gating', enhancer.method3_spectral_gating)
    ]
    
    for method_name, method_func in methods:
        print(f"\n  {method_name}:", end=' ')
        enhanced_audio, metadata = method_func(original_audio, sample_rate)
        
        if metadata['success']:
            print(f"✓ RTF: {metadata['rtf']:.3f}")
            
            # Calculate metrics
            noise_original = np.std(original_audio[:int(0.1 * len(original_audio))])
            noise_enhanced = np.std(enhanced_audio[:int(0.1 * len(enhanced_audio))])
            
            if noise_original > 0 and noise_enhanced > 0:
                noise_reduction = 20 * np.log10(noise_original / noise_enhanced)
            else:
                noise_reduction = 0.0
            
            result['enhancements'][method_name] = {
                'audio': enhanced_audio,
                'metadata': metadata,
                'noise_reduction_db': noise_reduction
            }
        else:
            print(f"❌ Failed")
    
    enhancement_results.append(result)

# Save results
print("\n4. Saving enhanced audio files...")
output_dir = Path('intelligent_enhancement_results')
output_dir.mkdir(exist_ok=True)

for result in enhancement_results:
    sample_id = result['sample_id']
    
    # Save original
    orig_path = output_dir / f'sample_{sample_id:02d}_original.wav'
    sf.write(orig_path, result['original_audio'], result['sample_rate'])
    
    # Save enhanced versions
    for method_name, enhancement in result['enhancements'].items():
        method_clean = method_name.replace(' ', '_').replace('+', 'plus').lower()
        enh_path = output_dir / f'sample_{sample_id:02d}_{method_clean}.wav'
        sf.write(enh_path, enhancement['audio'], result['sample_rate'])
    
    print(f"  ✓ Saved Sample {sample_id} files")

# Save report
print("\n5. Creating comparison report...")
report_data = []
for result in enhancement_results:
    for method_name, enhancement in result['enhancements'].items():
        report_data.append({
            'sample_id': result['sample_id'],
            'noise_type': result['noise_type'],
            'duration': result['duration'],
            'method': method_name,
            'rtf': enhancement['metadata']['rtf'],
            'noise_reduction_db': enhancement['noise_reduction_db']
        })

df = pd.DataFrame(report_data)
csv_path = output_dir / 'enhancement_comparison_results.csv'
df.to_csv(csv_path, index=False)

# Create summary
print("\n" + "="*80)
print("ENHANCEMENT COMPLETE!")
print("="*80)
print(f"\n✓ Processed 5 samples with 3 enhancement methods")
print(f"✓ Saved {len(os.listdir(output_dir))} files to: {output_dir.absolute()}")
print(f"\nFiles saved:")
for file in sorted(os.listdir(output_dir)):
    print(f"  - {file}")

print(f"\nAverage performance by method:")
for method in df['method'].unique():
    method_df = df[df['method'] == method]
    avg_rtf = method_df['rtf'].mean()
    avg_nr = method_df['noise_reduction_db'].mean()
    print(f"  {method}: RTF={avg_rtf:.3f}, Noise Reduction={avg_nr:.1f}dB")

print(f"\n✓ All results saved in: {output_dir.absolute()}")
print("\nYou can now listen to the enhanced audio files!")