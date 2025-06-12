#!/usr/bin/env python3
"""
Comprehensive comparison script for all audio enhancement methods.
Compares Pattern→MetricGAN+ baseline with three advanced methods across 50 samples.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AudioMetricsCalculator:
    """Calculate various audio quality metrics"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS energy"""
        return np.sqrt(np.mean(audio**2))
    
    def calculate_rms_ratio(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate volume preservation ratio"""
        rms_orig = self.calculate_rms(original)
        rms_proc = self.calculate_rms(processed)
        return rms_proc / rms_orig if rms_orig > 0 else 0
    
    def calculate_snr(self, signal: np.ndarray, noise_floor: float = 0.001) -> float:
        """Estimate SNR using silent regions as noise reference"""
        # Find quiet regions (bottom 10% of energy)
        frame_size = int(0.02 * self.sr)  # 20ms frames
        hop_size = frame_size // 2
        
        energies = []
        for i in range(0, len(signal) - frame_size, hop_size):
            frame = signal[i:i+frame_size]
            energies.append(np.mean(frame**2))
        
        energies = np.array(energies)
        noise_energy = np.percentile(energies, 10)
        signal_energy = np.mean(energies)
        
        if noise_energy > 0:
            snr = 10 * np.log10(signal_energy / noise_energy)
            return snr
        return 0
    
    def calculate_hf_preservation(self, original: np.ndarray, processed: np.ndarray, 
                                  cutoff: int = 4000) -> float:
        """Calculate high-frequency preservation ratio"""
        # Apply FFT
        orig_fft = np.fft.rfft(original)
        proc_fft = np.fft.rfft(processed)
        
        # Get frequency bins
        freqs = np.fft.rfftfreq(len(original), 1/self.sr)
        
        # Calculate energy above cutoff
        hf_mask = freqs >= cutoff
        
        orig_hf_energy = np.sum(np.abs(orig_fft[hf_mask])**2)
        proc_hf_energy = np.sum(np.abs(proc_fft[hf_mask])**2)
        
        if orig_hf_energy > 0:
            return proc_hf_energy / orig_hf_energy
        return 0
    
    def detect_interruptions(self, audio: np.ndarray, threshold: float = 0.02) -> List[Tuple[float, float]]:
        """Detect potential interruption regions"""
        interruptions = []
        
        # Use short-term energy with adaptive threshold
        frame_size = int(0.05 * self.sr)  # 50ms frames
        hop_size = frame_size // 4
        
        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energies.append(np.sqrt(np.mean(frame**2)))
        
        energies = np.array(energies)
        
        # Adaptive threshold based on statistics
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        adaptive_threshold = mean_energy + 2 * std_energy
        
        # Find sudden changes
        energy_diff = np.diff(energies)
        
        # Detect interruptions as sudden increases followed by decreases
        in_interruption = False
        start_idx = 0
        
        for i in range(1, len(energy_diff)-1):
            # Sudden increase
            if not in_interruption and energy_diff[i] > threshold and energies[i+1] > adaptive_threshold:
                in_interruption = True
                start_idx = i
            # Sudden decrease
            elif in_interruption and energy_diff[i] < -threshold:
                in_interruption = False
                start_time = start_idx * hop_size / self.sr
                end_time = i * hop_size / self.sr
                interruptions.append((start_time, end_time))
        
        return interruptions
    
    def calculate_interruption_suppression(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate how well interruptions were suppressed"""
        orig_interruptions = self.detect_interruptions(original)
        proc_interruptions = self.detect_interruptions(processed)
        
        if len(orig_interruptions) == 0:
            return 1.0  # No interruptions to suppress
        
        # Calculate total interruption duration
        orig_duration = sum(end - start for start, end in orig_interruptions)
        proc_duration = sum(end - start for start, end in proc_interruptions)
        
        # Suppression ratio (1.0 = perfect suppression)
        suppression = 1.0 - (proc_duration / orig_duration) if orig_duration > 0 else 0
        return max(0, suppression)
    
    def calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid as a measure of brightness"""
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        # Weighted average of frequencies
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return centroid
        return 0


class MethodComparison:
    """Compare all enhancement methods"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.metrics_calc = AudioMetricsCalculator()
        self.methods = {
            'baseline': 'final_pattern_then_metricgan',
            'method1': 'method1_coherence_vad',
            'method2': 'method2_dualpath',
            'method3': 'method3_attention'
        }
        self.method_names = {
            'baseline': 'Pattern→MetricGAN+',
            'method1': 'Spectral Coherence + VAD',
            'method2': 'Dual-Path Separation',
            'method3': 'Attention-Based Filtering'
        }
        
    def load_audio_pair(self, sample_id: int, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load original and processed audio for a sample"""
        orig_path = self.results_dir / f'sample_{sample_id:02d}_original.wav'
        proc_path = self.results_dir / f'sample_{sample_id:02d}_{self.methods[method]}.wav'
        
        if not orig_path.exists() or not proc_path.exists():
            return None, None
            
        original, sr1 = sf.read(orig_path)
        processed, sr2 = sf.read(proc_path)
        
        # Ensure same length
        min_len = min(len(original), len(processed))
        return original[:min_len], processed[:min_len]
    
    def analyze_sample(self, sample_id: int) -> Dict[str, Dict[str, float]]:
        """Analyze all methods for a single sample"""
        results = {}
        
        for method in self.methods:
            original, processed = self.load_audio_pair(sample_id, method)
            
            if original is None or processed is None:
                continue
                
            metrics = {
                'volume_preservation': self.metrics_calc.calculate_rms_ratio(original, processed),
                'hf_preservation': self.metrics_calc.calculate_hf_preservation(original, processed),
                'snr_improvement': self.metrics_calc.calculate_snr(processed) - self.metrics_calc.calculate_snr(original),
                'interruption_suppression': self.metrics_calc.calculate_interruption_suppression(original, processed),
                'spectral_centroid_shift': self.metrics_calc.calculate_spectral_centroid(processed) - self.metrics_calc.calculate_spectral_centroid(original)
            }
            
            results[method] = metrics
            
        return results
    
    def analyze_all_samples(self) -> pd.DataFrame:
        """Analyze all 50 samples and compile results"""
        all_results = []
        
        print("Analyzing samples...")
        for sample_id in range(1, 51):
            if sample_id % 10 == 0:
                print(f"  Progress: {sample_id}/50")
                
            sample_results = self.analyze_sample(sample_id)
            
            for method, metrics in sample_results.items():
                row = {
                    'sample_id': sample_id,
                    'method': method,
                    'method_name': self.method_names[method],
                    **metrics
                }
                all_results.append(row)
        
        return pd.DataFrame(all_results)
    
    def plot_metric_comparison(self, df: pd.DataFrame, metric: str, title: str, ylabel: str):
        """Create box plot comparing methods for a specific metric"""
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        df_plot = df[['method_name', metric]].copy()
        df_plot = df_plot.dropna()
        
        # Order methods
        method_order = ['Pattern→MetricGAN+', 'Spectral Coherence + VAD', 
                       'Dual-Path Separation', 'Attention-Based Filtering']
        
        sns.boxplot(data=df_plot, x='method_name', y=metric, order=method_order)
        plt.xticks(rotation=15, ha='right')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel)
        plt.xlabel('Enhancement Method')
        
        # Add reference line for ideal values
        if metric == 'volume_preservation' or metric == 'hf_preservation':
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal (1.0)')
            plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_waveform_comparison(self, sample_id: int):
        """Plot waveform comparison for a specific sample"""
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        
        # Load original
        orig_path = self.results_dir / f'sample_{sample_id:02d}_original.wav'
        original, sr = sf.read(orig_path)
        time = np.arange(len(original)) / sr
        
        # Plot original
        axes[0].plot(time, original, color='gray', alpha=0.7, linewidth=0.5)
        axes[0].set_ylabel('Original')
        axes[0].set_ylim(-1, 1)
        axes[0].grid(True, alpha=0.3)
        
        # Plot each method
        colors = ['blue', 'green', 'orange', 'red']
        for idx, (method, color) in enumerate(zip(self.methods, colors)):
            _, processed = self.load_audio_pair(sample_id, method)
            if processed is not None:
                axes[idx+1].plot(time[:len(processed)], processed, color=color, alpha=0.7, linewidth=0.5)
                axes[idx+1].set_ylabel(self.method_names[method])
                axes[idx+1].set_ylim(-1, 1)
                axes[idx+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle(f'Waveform Comparison - Sample {sample_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_spectrogram_comparison(self, sample_id: int):
        """Plot spectrogram comparison for a specific sample"""
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        
        # Load original
        orig_path = self.results_dir / f'sample_{sample_id:02d}_original.wav'
        original, sr = sf.read(orig_path)
        
        # Compute and plot spectrograms
        nperseg = 512
        noverlap = 256
        
        # Original
        f, t, Sxx = signal.spectrogram(original, sr, nperseg=nperseg, noverlap=noverlap)
        im = axes[0].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        axes[0].set_ylabel('Original\nFreq (Hz)')
        axes[0].set_ylim(0, 8000)
        
        # Each method
        for idx, method in enumerate(self.methods):
            _, processed = self.load_audio_pair(sample_id, method)
            if processed is not None:
                f, t, Sxx = signal.spectrogram(processed, sr, nperseg=nperseg, noverlap=noverlap)
                axes[idx+1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
                axes[idx+1].set_ylabel(f'{self.method_names[method]}\nFreq (Hz)')
                axes[idx+1].set_ylim(0, 8000)
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle(f'Spectrogram Comparison - Sample {sample_id}', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.01)
        cbar.set_label('Power (dB)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_energy_envelope_comparison(self, sample_id: int):
        """Plot energy envelope comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Load original
        orig_path = self.results_dir / f'sample_{sample_id:02d}_original.wav'
        original, sr = sf.read(orig_path)
        
        # Calculate energy envelopes
        frame_size = int(0.02 * sr)  # 20ms frames
        hop_size = frame_size // 2
        
        # Original envelope
        orig_envelope = []
        for i in range(0, len(original) - frame_size, hop_size):
            frame = original[i:i+frame_size]
            orig_envelope.append(np.sqrt(np.mean(frame**2)))
        
        time_frames = np.arange(len(orig_envelope)) * hop_size / sr
        
        # Plot original
        ax.plot(time_frames, orig_envelope, 'k-', alpha=0.5, linewidth=2, label='Original')
        
        # Plot each method
        colors = ['blue', 'green', 'orange', 'red']
        linestyles = ['-', '--', '-.', ':']
        
        for method, color, ls in zip(self.methods, colors, linestyles):
            _, processed = self.load_audio_pair(sample_id, method)
            if processed is not None:
                proc_envelope = []
                for i in range(0, len(processed) - frame_size, hop_size):
                    frame = processed[i:i+frame_size]
                    proc_envelope.append(np.sqrt(np.mean(frame**2)))
                
                ax.plot(time_frames[:len(proc_envelope)], proc_envelope, 
                       color=color, linestyle=ls, linewidth=1.5, 
                       label=self.method_names[method])
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('RMS Energy')
        ax.set_title(f'Energy Envelope Comparison - Sample {sample_id}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive summary report"""
        report = []
        report.append("# Comprehensive Audio Enhancement Method Comparison Report\n")
        report.append("## Executive Summary\n")
        
        # Overall statistics
        report.append("### Dataset Overview")
        report.append(f"- Total samples analyzed: {df['sample_id'].nunique()}")
        report.append(f"- Methods compared: {', '.join(self.method_names.values())}\n")
        
        # Best method by metric
        report.append("### Best Performing Methods by Metric\n")
        
        metrics_info = {
            'volume_preservation': ('Volume Preservation', 'closest to 1.0'),
            'hf_preservation': ('High-Frequency Preservation', 'closest to 1.0'),
            'snr_improvement': ('SNR Improvement', 'highest value'),
            'interruption_suppression': ('Interruption Suppression', 'highest value')
        }
        
        for metric, (name, criteria) in metrics_info.items():
            metric_means = df.groupby('method_name')[metric].mean()
            
            if 'closest to 1.0' in criteria:
                best_method = metric_means.sub(1.0).abs().idxmin()
                best_value = metric_means[best_method]
            else:
                best_method = metric_means.idxmax()
                best_value = metric_means[best_method]
            
            report.append(f"- **{name}**: {best_method} ({criteria}: {best_value:.3f})")
        
        report.append("\n## Detailed Analysis by Method\n")
        
        # Detailed stats for each method
        for method in self.methods:
            method_name = self.method_names[method]
            method_data = df[df['method'] == method]
            
            report.append(f"### {method_name}\n")
            
            # Calculate statistics
            stats = method_data[list(metrics_info.keys())].describe()
            
            report.append("#### Performance Metrics (mean ± std)")
            for metric, (name, _) in metrics_info.items():
                mean_val = stats.loc['mean', metric]
                std_val = stats.loc['std', metric]
                report.append(f"- {name}: {mean_val:.3f} ± {std_val:.3f}")
            
            # Identify strengths and weaknesses
            report.append("\n#### Key Characteristics")
            
            # Volume preservation
            vol_pres = stats.loc['mean', 'volume_preservation']
            if abs(vol_pres - 1.0) < 0.05:
                report.append("- ✅ Excellent volume preservation")
            elif abs(vol_pres - 1.0) < 0.15:
                report.append("- ⚠️ Moderate volume preservation")
            else:
                report.append("- ❌ Poor volume preservation")
            
            # HF preservation
            hf_pres = stats.loc['mean', 'hf_preservation']
            if hf_pres > 0.8:
                report.append("- ✅ Good high-frequency preservation")
            elif hf_pres > 0.6:
                report.append("- ⚠️ Moderate high-frequency loss")
            else:
                report.append("- ❌ Significant high-frequency loss")
            
            # SNR improvement
            snr_imp = stats.loc['mean', 'snr_improvement']
            if snr_imp > 5:
                report.append("- ✅ Strong noise reduction")
            elif snr_imp > 2:
                report.append("- ⚠️ Moderate noise reduction")
            else:
                report.append("- ❌ Limited noise reduction")
            
            # Interruption suppression
            int_supp = stats.loc['mean', 'interruption_suppression']
            if int_supp > 0.7:
                report.append("- ✅ Effective interruption removal")
            elif int_supp > 0.4:
                report.append("- ⚠️ Partial interruption removal")
            else:
                report.append("- ❌ Limited interruption removal")
            
            report.append("")
        
        # Recommendations section
        report.append("## Recommendations by Use Case\n")
        
        # Analyze correlations
        report.append("### Optimal Method Selection\n")
        
        # Find samples with detected interruptions
        interruption_samples = []
        for sample_id in df['sample_id'].unique():
            sample_data = df[df['sample_id'] == sample_id]
            baseline_supp = sample_data[sample_data['method'] == 'baseline']['interruption_suppression'].values
            if len(baseline_supp) > 0 and baseline_supp[0] < 0.5:
                interruption_samples.append(sample_id)
        
        report.append(f"#### Samples with Detected Interruptions: {len(interruption_samples)}")
        if interruption_samples:
            report.append(f"Sample IDs: {', '.join(map(str, interruption_samples[:10]))}")
            if len(interruption_samples) > 10:
                report.append(f"... and {len(interruption_samples) - 10} more")
        
        # Best method for interruption cases
        if interruption_samples:
            int_df = df[df['sample_id'].isin(interruption_samples)]
            int_performance = int_df.groupby('method_name')['interruption_suppression'].mean()
            best_int_method = int_performance.idxmax()
            report.append(f"\n**Best for interruption removal**: {best_int_method}")
        
        # General recommendations
        report.append("\n### General Recommendations\n")
        
        # Calculate overall scores
        overall_scores = {}
        for method in self.methods:
            method_name = self.method_names[method]
            method_data = df[df['method'] == method]
            
            # Weighted score
            score = 0
            score += abs(1.0 - method_data['volume_preservation'].mean()) * -2  # Penalty for volume change
            score += method_data['hf_preservation'].mean() * 1.5  # Reward HF preservation
            score += method_data['snr_improvement'].mean() * 0.5  # Reward SNR improvement
            score += method_data['interruption_suppression'].mean() * 2  # Reward interruption removal
            
            overall_scores[method_name] = score
        
        # Sort by score
        sorted_methods = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        report.append("#### Overall Performance Ranking")
        for rank, (method, score) in enumerate(sorted_methods, 1):
            report.append(f"{rank}. **{method}** (score: {score:.2f})")
        
        # Specific use case recommendations
        report.append("\n#### Use Case Recommendations\n")
        
        report.append("1. **For general enhancement with good quality preservation**:")
        report.append(f"   - Recommended: {sorted_methods[0][0]}")
        
        report.append("\n2. **For aggressive interruption removal**:")
        int_scores = df.groupby('method_name')['interruption_suppression'].mean()
        report.append(f"   - Recommended: {int_scores.idxmax()}")
        
        report.append("\n3. **For maintaining original audio characteristics**:")
        vol_scores = df.groupby('method_name')['volume_preservation'].apply(lambda x: 1 - abs(x - 1).mean())
        report.append(f"   - Recommended: {vol_scores.idxmax()}")
        
        report.append("\n4. **For maximum noise reduction**:")
        snr_scores = df.groupby('method_name')['snr_improvement'].mean()
        report.append(f"   - Recommended: {snr_scores.idxmax()}")
        
        # Technical details
        report.append("\n## Technical Implementation Notes\n")
        
        report.append("### Method Characteristics Summary\n")
        report.append("- **Pattern→MetricGAN+**: Conservative approach, good for clean audio")
        report.append("- **Spectral Coherence + VAD**: Advanced detection, good for consistent speakers")
        report.append("- **Dual-Path Separation**: Harmonic/percussive split, good for music/claps")
        report.append("- **Attention-Based Filtering**: ML-based approach, good overall quality")
        
        return '\n'.join(report)


def main():
    """Main execution function"""
    results_dir = Path('/media/ssd1/SparkVoiceProject/ThaiDatasetGathering/intelligent_enhancement_results')
    
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        return
    
    print("="*60)
    print("Comprehensive Audio Enhancement Method Comparison")
    print("="*60)
    
    # Initialize comparison
    comparison = MethodComparison(results_dir)
    
    # Analyze all samples
    print("\nAnalyzing all 50 samples across 4 methods...")
    df_results = comparison.analyze_all_samples()
    
    # Save raw results
    csv_path = results_dir / 'method_comparison_metrics.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nRaw metrics saved to: {csv_path}")
    
    # Generate plots
    print("\nGenerating comparison visualizations...")
    
    # 1. Metric comparison plots
    metrics_to_plot = [
        ('volume_preservation', 'Volume Preservation Comparison', 'Volume Ratio (processed/original)'),
        ('hf_preservation', 'High-Frequency Preservation Comparison', 'HF Energy Ratio (>4kHz)'),
        ('snr_improvement', 'Signal-to-Noise Ratio Improvement', 'SNR Improvement (dB)'),
        ('interruption_suppression', 'Interruption Suppression Effectiveness', 'Suppression Ratio (0-1)')
    ]
    
    for metric, title, ylabel in metrics_to_plot:
        fig = comparison.plot_metric_comparison(df_results, metric, title, ylabel)
        fig.savefig(results_dir / f'comparison_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # 2. Sample-specific visualizations (for samples with interruptions)
    # Find samples with significant interruptions
    interruption_samples = []
    for sample_id in range(1, 51):
        sample_data = df_results[df_results['sample_id'] == sample_id]
        baseline_data = sample_data[sample_data['method'] == 'baseline']
        if len(baseline_data) > 0:
            if baseline_data['interruption_suppression'].values[0] < 0.8:
                interruption_samples.append(sample_id)
    
    print(f"\nGenerating detailed plots for {len(interruption_samples[:5])} samples with interruptions...")
    
    for sample_id in interruption_samples[:5]:  # Limit to 5 samples for space
        # Waveform comparison
        fig = comparison.plot_waveform_comparison(sample_id)
        fig.savefig(results_dir / f'waveform_comparison_sample_{sample_id:02d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Spectrogram comparison
        fig = comparison.plot_spectrogram_comparison(sample_id)
        fig.savefig(results_dir / f'spectrogram_comparison_sample_{sample_id:02d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Energy envelope comparison
        fig = comparison.plot_energy_envelope_comparison(sample_id)
        fig.savefig(results_dir / f'energy_envelope_sample_{sample_id:02d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # 3. Generate summary report
    print("\nGenerating comprehensive summary report...")
    report = comparison.generate_summary_report(df_results)
    
    report_path = results_dir / 'COMPREHENSIVE_METHOD_COMPARISON_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nSummary report saved to: {report_path}")
    
    # 4. Create a summary visualization grid
    print("\nCreating summary visualization grid...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot each metric
    for idx, (metric, title, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        # Create box plot
        df_plot = df_results[['method_name', metric]].dropna()
        method_order = ['Pattern→MetricGAN+', 'Spectral Coherence + VAD', 
                       'Dual-Path Separation', 'Attention-Based Filtering']
        
        # Create violin plot for more detail
        parts = ax.violinplot([df_plot[df_plot['method_name'] == m][metric].values 
                              for m in method_order], 
                             positions=range(len(method_order)),
                             showmeans=True, showmedians=True)
        
        # Customize colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(method_order)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in method_order], fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Add reference line for ideal values
        if metric in ['volume_preservation', 'hf_preservation']:
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Audio Enhancement Methods - Performance Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(results_dir / 'method_comparison_overview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - Raw metrics: {csv_path}")
    print(f"  - Summary report: {report_path}")
    print(f"  - Comparison plots: {results_dir}/comparison_*.png")
    print(f"  - Sample visualizations: {results_dir}/*_sample_*.png")
    print(f"  - Overview plot: {results_dir}/method_comparison_overview.png")
    
    # Print quick summary
    print("\n" + "="*60)
    print("Quick Summary of Results")
    print("="*60)
    
    for method in comparison.methods:
        method_name = comparison.method_names[method]
        method_data = df_results[df_results['method'] == method]
        
        print(f"\n{method_name}:")
        print(f"  Volume preservation: {method_data['volume_preservation'].mean():.3f} ± {method_data['volume_preservation'].std():.3f}")
        print(f"  HF preservation: {method_data['hf_preservation'].mean():.3f} ± {method_data['hf_preservation'].std():.3f}")
        print(f"  SNR improvement: {method_data['snr_improvement'].mean():.1f} ± {method_data['snr_improvement'].std():.1f} dB")
        print(f"  Interruption suppression: {method_data['interruption_suppression'].mean():.3f} ± {method_data['interruption_suppression'].std():.3f}")


if __name__ == "__main__":
    main()