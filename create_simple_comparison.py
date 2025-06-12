#!/usr/bin/env python3
"""
Simple comparison script: Original vs Pattern→MetricGAN+
Since Resemble-Enhance is having issues, let's analyze what we have
"""

import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import json

print("Audio Enhancement Comparison Analysis")
print("=" * 80)


def analyze_audio_comprehensive(audio_path, sample_id, method_name):
    """Comprehensive audio analysis"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Time-domain metrics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        # Frequency-domain metrics
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # High-frequency energy (>4kHz)
        hf_mask = freqs > 4000
        hf_energy = np.mean(magnitude[hf_mask, :])
        total_energy = np.mean(magnitude)
        hf_ratio = hf_energy / (total_energy + 1e-8)
        
        # Mid-frequency energy (1-4kHz) - important for speech
        mf_mask = (freqs > 1000) & (freqs <= 4000)
        mf_energy = np.mean(magnitude[mf_mask, :])
        mf_ratio = mf_energy / (total_energy + 1e-8)
        
        # Low-frequency energy (<1kHz)
        lf_mask = freqs <= 1000
        lf_energy = np.mean(magnitude[lf_mask, :])
        lf_ratio = lf_energy / (total_energy + 1e-8)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        
        # Zero crossing rate (noise indicator)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Estimate SNR using silent parts vs speech parts
        energy_db = librosa.amplitude_to_db(librosa.feature.rms(y=audio, hop_length=512)[0])
        noise_floor = np.percentile(energy_db, 10)  # Bottom 10% as noise
        signal_peak = np.percentile(energy_db, 90)  # Top 10% as signal
        estimated_snr = signal_peak - noise_floor
        
        return {
            'sample_id': sample_id,
            'method': method_name,
            'rms': float(rms),
            'peak': float(peak),
            'hf_ratio': float(hf_ratio),
            'mf_ratio': float(mf_ratio),
            'lf_ratio': float(lf_ratio),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_bandwidth': float(spectral_bandwidth),
            'zcr': float(zcr),
            'estimated_snr': float(estimated_snr),
            'duration': len(audio) / sr
        }
    except Exception as e:
        print(f"  Error analyzing {audio_path}: {e}")
        return None


def create_waveform_comparison(sample_id, results_dir):
    """Create waveform comparison plot"""
    try:
        # Load audio files
        original_path = results_dir / f"sample_{sample_id}_original.wav"
        metricgan_path = results_dir / f"sample_{sample_id}_final_pattern_then_metricgan.wav"
        
        original, sr = librosa.load(original_path, sr=None)
        metricgan, _ = librosa.load(metricgan_path, sr=None)
        
        # Time axis
        time_orig = np.arange(len(original)) / sr
        time_mg = np.arange(len(metricgan)) / sr
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # Original waveform
        axes[0].plot(time_orig, original, color='blue', alpha=0.7)
        axes[0].set_title(f'Sample {sample_id}: Original', fontsize=12)
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-1, 1)
        
        # MetricGAN+ waveform
        axes[1].plot(time_mg, metricgan, color='green', alpha=0.7)
        axes[1].set_title(f'Sample {sample_id}: Pattern→MetricGAN+', fontsize=12)
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-1, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = results_dir / f"waveform_comparison_sample_{sample_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  Error creating waveform plot: {e}")
        return False


def create_spectrogram_comparison(sample_id, results_dir):
    """Create spectrogram comparison"""
    try:
        # Load audio files
        original_path = results_dir / f"sample_{sample_id}_original.wav"
        metricgan_path = results_dir / f"sample_{sample_id}_final_pattern_then_metricgan.wav"
        
        original, sr = librosa.load(original_path, sr=None)
        metricgan, _ = librosa.load(metricgan_path, sr=None)
        
        # Compute spectrograms
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
        D_mg = librosa.amplitude_to_db(np.abs(librosa.stft(metricgan)), ref=np.max)
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original spectrogram
        img1 = librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz', ax=axes[0])
        axes[0].set_title(f'Sample {sample_id}: Original Spectrogram', fontsize=12)
        axes[0].set_ylim(0, 8000)  # Focus on 0-8kHz
        
        # MetricGAN+ spectrogram
        img2 = librosa.display.specshow(D_mg, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title(f'Sample {sample_id}: Pattern→MetricGAN+ Spectrogram', fontsize=12)
        axes[1].set_ylim(0, 8000)
        
        # Add colorbars
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = results_dir / f"spectrogram_comparison_sample_{sample_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  Error creating spectrogram plot: {e}")
        return False


def main():
    """Main analysis pipeline"""
    
    results_dir = Path("intelligent_enhancement_results")
    
    # Check files
    print("\nChecking available files...")
    original_files = list(results_dir.glob("sample_*_original.wav"))
    metricgan_files = list(results_dir.glob("sample_*_final_pattern_then_metricgan.wav"))
    
    print(f"Found {len(original_files)} original files")
    print(f"Found {len(metricgan_files)} Pattern→MetricGAN+ files")
    
    # Analyze all samples
    print(f"\n{'='*80}")
    print("Analyzing audio quality metrics")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    for i in range(1, 51):
        sample_id = f"{i:02d}"
        
        original_path = results_dir / f"sample_{sample_id}_original.wav"
        metricgan_path = results_dir / f"sample_{sample_id}_final_pattern_then_metricgan.wav"
        
        if not original_path.exists() or not metricgan_path.exists():
            continue
            
        print(f"Sample {sample_id}:", end=" ")
        
        # Analyze both versions
        orig_metrics = analyze_audio_comprehensive(original_path, sample_id, "Original")
        mg_metrics = analyze_audio_comprehensive(metricgan_path, sample_id, "Pattern→MetricGAN+")
        
        if orig_metrics and mg_metrics:
            all_metrics.extend([orig_metrics, mg_metrics])
            
            # Quick comparison
            volume_change = (mg_metrics['rms'] / orig_metrics['rms'] - 1) * 100
            hf_change = (mg_metrics['hf_ratio'] / orig_metrics['hf_ratio'] - 1) * 100
            snr_improvement = mg_metrics['estimated_snr'] - orig_metrics['estimated_snr']
            
            print(f"Volume: {volume_change:+.1f}%, HF: {hf_change:+.1f}%, SNR: {snr_improvement:+.1f}dB")
        else:
            print("Analysis failed")
    
    # Save detailed metrics
    metrics_path = results_dir / "detailed_comparison_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create visualizations for a few samples
    print(f"\n{'='*80}")
    print("Creating visualizations")
    print(f"{'='*80}\n")
    
    sample_ids = ["01", "10", "25", "40", "50"]  # Representative samples
    
    for sample_id in sample_ids:
        print(f"Creating plots for sample {sample_id}...")
        create_waveform_comparison(sample_id, results_dir)
        create_spectrogram_comparison(sample_id, results_dir)
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("Generating summary report")
    print(f"{'='*80}\n")
    
    # Calculate averages
    orig_metrics_list = [m for m in all_metrics if m['method'] == "Original"]
    mg_metrics_list = [m for m in all_metrics if m['method'] == "Pattern→MetricGAN+"]
    
    if orig_metrics_list and mg_metrics_list:
        # Average metrics
        avg_orig_rms = np.mean([m['rms'] for m in orig_metrics_list])
        avg_mg_rms = np.mean([m['rms'] for m in mg_metrics_list])
        
        avg_orig_hf = np.mean([m['hf_ratio'] for m in orig_metrics_list])
        avg_mg_hf = np.mean([m['hf_ratio'] for m in mg_metrics_list])
        
        avg_orig_snr = np.mean([m['estimated_snr'] for m in orig_metrics_list])
        avg_mg_snr = np.mean([m['estimated_snr'] for m in mg_metrics_list])
        
        # Write report
        report_path = results_dir / "pattern_metricgan_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write("# Pattern→MetricGAN+ Enhancement Analysis\n\n")
            
            f.write("## Overview\n")
            f.write(f"- Analyzed {len(orig_metrics_list)} Thai audio samples\n")
            f.write("- Comparison: Original vs Pattern→MetricGAN+ enhancement\n\n")
            
            f.write("## Average Metrics\n\n")
            f.write("| Metric | Original | Pattern→MetricGAN+ | Change |\n")
            f.write("|--------|----------|-------------------|--------|\n")
            f.write(f"| RMS Energy | {avg_orig_rms:.4f} | {avg_mg_rms:.4f} | {(avg_mg_rms/avg_orig_rms-1)*100:+.1f}% |\n")
            f.write(f"| HF Ratio | {avg_orig_hf:.4f} | {avg_mg_hf:.4f} | {(avg_mg_hf/avg_orig_hf-1)*100:+.1f}% |\n")
            f.write(f"| Est. SNR | {avg_orig_snr:.1f} dB | {avg_mg_snr:.1f} dB | {avg_mg_snr-avg_orig_snr:+.1f} dB |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Volume analysis
            volume_ratio = avg_mg_rms / avg_orig_rms
            if volume_ratio < 0.9:
                f.write(f"1. **Volume Reduction Issue**: Average {(1-volume_ratio)*100:.1f}% quieter\n")
            else:
                f.write(f"1. **Volume**: {'Increased' if volume_ratio > 1.1 else 'Maintained'} ({volume_ratio:.2f}x)\n")
            
            # HF analysis  
            hf_preservation = avg_mg_hf / avg_orig_hf
            f.write(f"2. **High-Frequency Content**: {hf_preservation*100:.1f}% preserved\n")
            
            # SNR analysis
            snr_improvement = avg_mg_snr - avg_orig_snr
            f.write(f"3. **SNR Improvement**: {snr_improvement:+.1f} dB\n")
            
            f.write("\n## Detailed Metrics\n\n")
            
            # Frequency distribution
            avg_orig_lf = np.mean([m['lf_ratio'] for m in orig_metrics_list])
            avg_mg_lf = np.mean([m['lf_ratio'] for m in mg_metrics_list])
            avg_orig_mf = np.mean([m['mf_ratio'] for m in orig_metrics_list])  
            avg_mg_mf = np.mean([m['mf_ratio'] for m in mg_metrics_list])
            
            f.write("### Frequency Distribution\n")
            f.write("| Band | Original | Enhanced | Change |\n")
            f.write("|------|----------|----------|--------|\n")
            f.write(f"| Low (<1kHz) | {avg_orig_lf:.3f} | {avg_mg_lf:.3f} | {(avg_mg_lf/avg_orig_lf-1)*100:+.1f}% |\n")
            f.write(f"| Mid (1-4kHz) | {avg_orig_mf:.3f} | {avg_mg_mf:.3f} | {(avg_mg_mf/avg_orig_mf-1)*100:+.1f}% |\n")
            f.write(f"| High (>4kHz) | {avg_orig_hf:.3f} | {avg_mg_hf:.3f} | {(avg_mg_hf/avg_orig_hf-1)*100:+.1f}% |\n")
            
            f.write("\n## Visualizations\n")
            f.write("- Waveform comparisons: `waveform_comparison_sample_XX.png`\n")
            f.write("- Spectrogram comparisons: `spectrogram_comparison_sample_XX.png`\n")
            f.write("- Detailed metrics: `detailed_comparison_metrics.json`\n")
        
        print(f"✓ Report saved: {report_path}")
        
        # Print summary
        print("\n📊 Summary:")
        print(f"  Volume change: {(volume_ratio-1)*100:+.1f}%")
        print(f"  HF preservation: {hf_preservation*100:.1f}%")
        print(f"  SNR improvement: {snr_improvement:+.1f} dB")
    
    print(f"\n🎯 All files in: {results_dir}/")


if __name__ == "__main__":
    main()