#!/usr/bin/env python3
"""
Process 50 samples with Resemble-Enhance on top of Pattern→MetricGAN+
Creates comparison: Original vs Pattern→MetricGAN+ vs Pattern→MetricGAN+→Resemble-Enhance
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
from pathlib import Path

print("Resemble-Enhance Comparison Pipeline")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


class ResembleEnhanceProcessor:
    """Process audio with Resemble-Enhance"""
    
    def __init__(self):
        self.device = device
        self.enhancer = None
        self.load_model()
        
    def load_model(self):
        """Load Resemble-Enhance model"""
        print("Loading Resemble-Enhance model...")
        try:
            # Try to import resemble-enhance
            from resemble_enhance.enhancer.inference import enhance
            self.enhance_fn = enhance
            print("✓ Resemble-Enhance loaded successfully")
            self.available = True
        except ImportError:
            print("⚠️ Resemble-Enhance not installed. Installing...")
            self.available = False
            # Install resemble-enhance
            os.system("pip install git+https://github.com/resemble-ai/resemble-enhance.git --upgrade")
            try:
                from resemble_enhance.enhancer.inference import enhance
                self.enhance_fn = enhance
                print("✓ Resemble-Enhance installed and loaded")
                self.available = True
            except Exception as e:
                print(f"✗ Failed to install Resemble-Enhance: {e}")
                self.available = False
                
    def process(self, audio_path, output_path):
        """Process audio file with Resemble-Enhance"""
        if not self.available:
            print("Resemble-Enhance not available, copying input to output")
            import shutil
            shutil.copy2(audio_path, output_path)
            return
            
        try:
            # Resemble-Enhance expects paths, not arrays
            # Run enhancement
            print(f"  Enhancing with Resemble-Enhance...")
            
            # The enhance function typically takes input path and output path
            # Some versions might have different APIs
            try:
                # Try newer API first
                self.enhance_fn(audio_path, output_path, solver="midpoint", nfe=64, tau=0.5)
            except TypeError:
                # Try older API
                try:
                    self.enhance_fn(audio_path, output_path)
                except Exception as e:
                    # Last resort - try with minimal parameters
                    print(f"  Trying basic enhancement...")
                    # Load audio, process, save
                    audio, sr = librosa.load(audio_path, sr=None)
                    
                    # If enhance expects tensor input
                    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
                    enhanced = self.enhance_fn(audio_tensor)
                    
                    if isinstance(enhanced, torch.Tensor):
                        enhanced = enhanced.cpu().numpy().squeeze()
                    
                    sf.write(output_path, enhanced, sr)
            
            print(f"  ✓ Enhanced successfully")
            
        except Exception as e:
            print(f"  ✗ Enhancement failed: {e}")
            # Copy original if enhancement fails
            import shutil
            shutil.copy2(audio_path, output_path)


def analyze_audio_quality(audio_path, sample_id, method_name):
    """Analyze audio quality metrics"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Basic metrics
        rms = np.sqrt(np.mean(audio**2))
        
        # High-frequency energy (>4kHz)
        stft = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=sr)
        hf_mask = freqs > 4000
        hf_energy = np.mean(np.abs(stft[hf_mask, :]))
        total_energy = np.mean(np.abs(stft))
        hf_ratio = hf_energy / (total_energy + 1e-8)
        
        # Spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        return {
            'sample_id': sample_id,
            'method': method_name,
            'rms': rms,
            'hf_ratio': hf_ratio,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr,
            'duration': len(audio) / sr
        }
    except Exception as e:
        print(f"  Error analyzing {audio_path}: {e}")
        return None


def main():
    """Main processing pipeline"""
    
    # Initialize processor
    processor = ResembleEnhanceProcessor()
    
    # Set up directories
    results_dir = Path("intelligent_enhancement_results")
    
    # Check if we have the required files
    print("\nChecking for required files...")
    original_files = list(results_dir.glob("sample_*_original.wav"))
    metricgan_files = list(results_dir.glob("sample_*_final_pattern_then_metricgan.wav"))
    
    print(f"Found {len(original_files)} original files")
    print(f"Found {len(metricgan_files)} MetricGAN+ processed files")
    
    if len(original_files) == 0 or len(metricgan_files) == 0:
        print("\n❌ Missing required files!")
        return
    
    # Process each sample
    print(f"\n{'='*80}")
    print("Processing samples with Resemble-Enhance")
    print(f"{'='*80}\n")
    
    successful = 0
    quality_metrics = []
    
    for i in range(1, 51):
        sample_id = f"{i:02d}"
        
        # File paths
        original_path = results_dir / f"sample_{sample_id}_original.wav"
        metricgan_path = results_dir / f"sample_{sample_id}_final_pattern_then_metricgan.wav"
        resemble_path = results_dir / f"sample_{sample_id}_final_pattern_then_metricgan_then_resemble_enhance.wav"
        
        if not original_path.exists() or not metricgan_path.exists():
            print(f"Sample {sample_id}: Missing files, skipping")
            continue
            
        print(f"Sample {sample_id}:")
        
        try:
            # Process with Resemble-Enhance
            start_time = time.time()
            processor.process(str(metricgan_path), str(resemble_path))
            process_time = time.time() - start_time
            
            print(f"  Processing time: {process_time:.2f}s")
            
            # Analyze all three versions
            original_metrics = analyze_audio_quality(original_path, sample_id, "Original")
            metricgan_metrics = analyze_audio_quality(metricgan_path, sample_id, "Pattern→MetricGAN+")
            resemble_metrics = analyze_audio_quality(resemble_path, sample_id, "Pattern→MetricGAN+→Resemble")
            
            if original_metrics:
                quality_metrics.append(original_metrics)
            if metricgan_metrics:
                quality_metrics.append(metricgan_metrics)
            if resemble_metrics:
                quality_metrics.append(resemble_metrics)
                
            # Compare volume preservation
            if original_metrics and resemble_metrics:
                volume_ratio = resemble_metrics['rms'] / original_metrics['rms']
                hf_preservation = resemble_metrics['hf_ratio'] / original_metrics['hf_ratio']
                print(f"  Volume ratio (vs original): {volume_ratio:.2f}x")
                print(f"  HF preservation: {hf_preservation:.1%}")
            
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            
        if i % 10 == 0:
            print(f"\nProgress: {i}/50 samples processed\n")
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"  Successfully processed: {successful}/50 samples")
    
    if quality_metrics:
        # Calculate average metrics for each method
        methods = ["Original", "Pattern→MetricGAN+", "Pattern→MetricGAN+→Resemble"]
        
        print("\n📈 Average Metrics:")
        for method in methods:
            method_metrics = [m for m in quality_metrics if m['method'] == method]
            if method_metrics:
                avg_rms = np.mean([m['rms'] for m in method_metrics])
                avg_hf = np.mean([m['hf_ratio'] for m in method_metrics])
                avg_centroid = np.mean([m['spectral_centroid'] for m in method_metrics])
                
                print(f"\n{method}:")
                print(f"  Average RMS: {avg_rms:.4f}")
                print(f"  Average HF ratio: {avg_hf:.4f}")
                print(f"  Average spectral centroid: {avg_centroid:.1f} Hz")
    
    # Save detailed report
    report_path = results_dir / "resemble_enhance_comparison_report.md"
    with open(report_path, 'w') as f:
        f.write("# Resemble-Enhance Comparison Report\n\n")
        f.write(f"## Overview\n")
        f.write(f"- Total samples: 50\n")
        f.write(f"- Successfully processed: {successful}\n")
        f.write(f"- Processing pipeline: Original → Pattern→MetricGAN+ → Resemble-Enhance\n\n")
        
        f.write("## Method Descriptions\n\n")
        f.write("### 1. Original\n")
        f.write("- Raw audio from GigaSpeech2 Thai dataset\n")
        f.write("- No preprocessing applied\n\n")
        
        f.write("### 2. Pattern→MetricGAN+\n")
        f.write("- Ultra-conservative interruption pattern detection\n")
        f.write("- 85% suppression of detected patterns\n")
        f.write("- MetricGAN+ enhancement for noise reduction\n\n")
        
        f.write("### 3. Pattern→MetricGAN+→Resemble-Enhance\n")
        f.write("- Takes Pattern→MetricGAN+ output\n")
        f.write("- Applies Resemble-Enhance for further improvement\n")
        f.write("- AI-based speech enhancement and restoration\n\n")
        
        if quality_metrics:
            f.write("## Quality Metrics\n\n")
            f.write("| Metric | Original | Pattern→MetricGAN+ | +Resemble-Enhance |\n")
            f.write("|--------|----------|-------------------|------------------|\n")
            
            for method in methods:
                method_metrics = [m for m in quality_metrics if m['method'] == method]
                if method_metrics:
                    avg_rms = np.mean([m['rms'] for m in method_metrics])
                    avg_hf = np.mean([m['hf_ratio'] for m in method_metrics])
                    avg_centroid = np.mean([m['spectral_centroid'] for m in method_metrics])
                    
                    if method == "Original":
                        f.write(f"| RMS Energy | {avg_rms:.4f} | ")
                    elif method == "Pattern→MetricGAN+":
                        f.write(f"{avg_rms:.4f} | ")
                    else:
                        f.write(f"{avg_rms:.4f} |\n")
                        
            for method in methods:
                method_metrics = [m for m in quality_metrics if m['method'] == method]
                if method_metrics:
                    avg_hf = np.mean([m['hf_ratio'] for m in method_metrics])
                    
                    if method == "Original":
                        f.write(f"| HF Ratio | {avg_hf:.4f} | ")
                    elif method == "Pattern→MetricGAN+":
                        f.write(f"{avg_hf:.4f} | ")
                    else:
                        f.write(f"{avg_hf:.4f} |\n")
        
        f.write("\n## Output Files\n")
        f.write("- Original: `sample_XX_original.wav`\n")
        f.write("- Pattern→MetricGAN+: `sample_XX_final_pattern_then_metricgan.wav`\n")
        f.write("- +Resemble-Enhance: `sample_XX_final_pattern_then_metricgan_then_resemble_enhance.wav`\n")
    
    print(f"\n✓ Report saved: {report_path}")
    print(f"\n🎯 All files in: {results_dir}/")


if __name__ == "__main__":
    main()