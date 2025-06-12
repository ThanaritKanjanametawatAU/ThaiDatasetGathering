#!/usr/bin/env python3
"""
Process 50 GigaSpeech2 Thai samples with SpeechBrain MetricGAN+
The winning enhancement method based on human judgment
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import soundfile as sf
import librosa
import pandas as pd
from pathlib import Path
import tempfile
from tqdm import tqdm

# Add project root to path
project_root = Path('/media/ssd1/SparkVoiceProject/ThaiDatasetGathering')
sys.path.append(str(project_root))

print("SpeechBrain MetricGAN+ Enhancement - 50 Sample Batch")
print("=" * 80)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MetricGANPlusEnhancer:
    """SpeechBrain MetricGAN+ Enhancer - The winning approach"""
    
    def __init__(self):
        self.device = device
        self.load_model()
        
    def load_model(self):
        """Load MetricGAN+ model"""
        print("\n1. Loading SpeechBrain MetricGAN+ model...")
        try:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
            self.enhancer = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus"
            )
            print("✓ MetricGAN+ loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load MetricGAN+: {e}")
            self.enhancer = None
            return False
    
    def enhance_audio(self, audio, sr):
        """Enhance audio using MetricGAN+"""
        if self.enhancer is None:
            return audio, {"success": False, "error": "Model not loaded"}
        
        try:
            start_time = time.time()
            
            # Save to temporary file (MetricGAN+ expects file input)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio, sr)
                
                # Apply enhancement
                enhanced_tensor = self.enhancer.enhance_file(tmp.name)
                enhanced = enhanced_tensor.squeeze().cpu().numpy()
                
                # Clean up
                os.unlink(tmp.name)
            
            # Ensure same length
            if len(enhanced) != len(audio):
                enhanced = librosa.util.fix_length(enhanced, size=len(audio))
            
            processing_time = time.time() - start_time
            
            # Calculate simple metrics
            noise_reduction = self._estimate_noise_reduction(audio, enhanced)
            
            return enhanced, {
                "success": True,
                "processing_time": processing_time,
                "rtf": processing_time / (len(audio) / sr),
                "noise_reduction_db": noise_reduction
            }
            
        except Exception as e:
            return audio, {"success": False, "error": str(e)}
    
    def _estimate_noise_reduction(self, original, enhanced):
        """Estimate noise reduction in dB"""
        try:
            # Use first 10% of signal (usually contains more noise)
            noise_len = int(0.1 * len(original))
            orig_noise_std = np.std(original[:noise_len])
            enh_noise_std = np.std(enhanced[:noise_len])
            
            if orig_noise_std > 0 and enh_noise_std > 0:
                return 20 * np.log10(orig_noise_std / enh_noise_std)
            return 0.0
        except:
            return 0.0

def load_gigaspeech2_samples(num_samples=50):
    """Load samples from GigaSpeech2"""
    print("\n2. Loading 50 GigaSpeech2 Thai samples...")
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
        
        print("Loading samples...")
        with tqdm(total=num_samples, desc="Loading") as pbar:
            for sample in processor._process_split_streaming('train', sample_mode=True, sample_size=num_samples):
                if len(samples) >= num_samples:
                    break
                samples.append(sample)
                pbar.update(1)
        
        print(f"✓ Successfully loaded {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"❌ Error loading samples: {e}")
        return []

def main():
    # Initialize enhancer
    enhancer = MetricGANPlusEnhancer()
    
    if enhancer.enhancer is None:
        print("❌ Cannot proceed without MetricGAN+ model")
        return
    
    # Load 50 samples
    samples = load_gigaspeech2_samples(50)
    
    if not samples:
        print("❌ No samples loaded")
        return
    
    # Create output directory
    output_dir = Path('metricgan_50_samples')
    output_dir.mkdir(exist_ok=True)
    
    # Process samples
    print("\n3. Processing 50 samples with MetricGAN+...")
    results = []
    
    with tqdm(total=len(samples), desc="Processing") as pbar:
        for i, sample in enumerate(samples):
            # Extract audio
            if isinstance(sample['audio'], dict) and 'array' in sample['audio']:
                original_audio = sample['audio']['array']
                sample_rate = sample['audio'].get('sampling_rate', 16000)
            else:
                pbar.update(1)
                continue
            
            original_audio = original_audio.astype(np.float32)
            duration = len(original_audio) / sample_rate
            
            # Enhance audio
            enhanced_audio, metadata = enhancer.enhance_audio(original_audio, sample_rate)
            
            if metadata['success']:
                # Save files
                sample_id = i + 1
                
                # Save original
                orig_path = output_dir / f'sample_{sample_id:03d}_original.wav'
                sf.write(orig_path, original_audio, sample_rate)
                
                # Save enhanced
                enh_path = output_dir / f'sample_{sample_id:03d}_enhanced.wav'
                sf.write(enh_path, enhanced_audio, sample_rate)
                
                # Store results
                results.append({
                    'sample_id': sample_id,
                    'duration': duration,
                    'transcript': sample.get('transcript', 'N/A')[:100],  # First 100 chars
                    'processing_time': metadata['processing_time'],
                    'rtf': metadata['rtf'],
                    'noise_reduction_db': metadata['noise_reduction_db']
                })
            else:
                print(f"\n❌ Sample {i+1} failed: {metadata['error']}")
            
            pbar.update(1)
    
    # Save results summary
    print("\n4. Creating results summary...")
    
    if results:
        df = pd.DataFrame(results)
        csv_path = output_dir / 'enhancement_results_50_samples.csv'
        df.to_csv(csv_path, index=False)
        
        # Calculate statistics
        avg_duration = df['duration'].mean()
        avg_processing_time = df['processing_time'].mean()
        avg_rtf = df['rtf'].mean()
        avg_noise_reduction = df['noise_reduction_db'].mean()
        total_duration = df['duration'].sum()
        total_processing_time = df['processing_time'].sum()
        
        # Create summary report
        summary_path = output_dir / 'ENHANCEMENT_SUMMARY.md'
        with open(summary_path, 'w') as f:
            f.write("# SpeechBrain MetricGAN+ Enhancement Results - 50 Samples\n\n")
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total samples processed**: {len(results)}\n")
            f.write(f"- **Total audio duration**: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)\n")
            f.write(f"- **Total processing time**: {total_processing_time:.1f} seconds ({total_processing_time/60:.1f} minutes)\n")
            f.write(f"- **Average sample duration**: {avg_duration:.2f} seconds\n")
            f.write(f"- **Average processing time**: {avg_processing_time:.3f} seconds\n")
            f.write(f"- **Average RTF**: {avg_rtf:.3f} ({int(1/avg_rtf)}x faster than real-time)\n")
            f.write(f"- **Average noise reduction**: {avg_noise_reduction:.1f} dB\n\n")
            
            f.write("## Performance Projection for 10M Samples\n\n")
            f.write("Based on 50-sample performance:\n")
            f.write(f"- **Estimated processing time**: {(avg_processing_time * 10_000_000 / 3600):.0f} hours ")
            f.write(f"({(avg_processing_time * 10_000_000 / 3600 / 24):.1f} days)\n")
            f.write(f"- **Processing speed**: {int(1/avg_rtf)}x faster than real-time\n")
            f.write(f"- **Expected noise reduction**: {avg_noise_reduction:.1f} dB average\n\n")
            
            f.write("## Sample Duration Distribution\n\n")
            f.write(f"- Shortest: {df['duration'].min():.2f}s\n")
            f.write(f"- Longest: {df['duration'].max():.2f}s\n")
            f.write(f"- Median: {df['duration'].median():.2f}s\n")
            f.write(f"- Std Dev: {df['duration'].std():.2f}s\n\n")
            
            f.write("## Files Generated\n\n")
            f.write(f"- Original audio files: {len(results)} files (*_original.wav)\n")
            f.write(f"- Enhanced audio files: {len(results)} files (*_enhanced.wav)\n")
            f.write(f"- Results CSV: enhancement_results_50_samples.csv\n")
        
        print(f"\n" + "="*80)
        print("METRICGAN+ ENHANCEMENT COMPLETE!")
        print("="*80)
        print(f"\n✓ Processed {len(results)} samples successfully")
        print(f"✓ Saved to: {output_dir.absolute()}/")
        print(f"\n📊 Performance Summary:")
        print(f"  - Average RTF: {avg_rtf:.3f} ({int(1/avg_rtf)}x faster than real-time)")
        print(f"  - Average noise reduction: {avg_noise_reduction:.1f} dB")
        print(f"  - Total processing time: {total_processing_time:.1f}s for {total_duration:.1f}s of audio")
        print(f"\n📁 Output files:")
        print(f"  - {len(results)} original + {len(results)} enhanced audio files")
        print(f"  - enhancement_results_50_samples.csv")
        print(f"  - ENHANCEMENT_SUMMARY.md")
        
        print(f"\n🎧 You can now:")
        print(f"  1. Listen to the 50 enhanced samples")
        print(f"  2. Compare with originals")
        print(f"  3. Review the summary statistics")
        print(f"  4. Make decision for full dataset processing")
    
    else:
        print("❌ No results to save")

if __name__ == "__main__":
    main()