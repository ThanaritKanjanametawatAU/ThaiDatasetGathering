"""Before/after comparison UI for audio enhancement."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from pathlib import Path

# Optional imports
try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ComparisonAnalyzer:
    """Analyze and visualize before/after audio comparisons."""
    
    def __init__(self, output_dir: str = "./comparisons"):
        """Initialize comparison analyzer.
        
        Args:
            output_dir: Directory for comparison outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid') if 'seaborn-v0_8-darkgrid' in plt.style.available else None
        
    def generate_waveform_comparison(self, original: np.ndarray, enhanced: np.ndarray,
                                   audio_id: str) -> Dict[str, str]:
        """Generate waveform comparison visualization.
        
        Args:
            original: Original audio signal
            enhanced: Enhanced audio signal
            audio_id: Unique identifier for the audio
            
        Returns:
            Dictionary with paths to generated visualizations
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Time axis
        time_axis = np.arange(len(original)) / 16000  # Assuming 16kHz
        
        # Original waveform
        ax1.plot(time_axis, original, color='blue', alpha=0.7, linewidth=0.5)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Original Audio')
        ax1.grid(True, alpha=0.3)
        
        # Enhanced waveform
        ax2.plot(time_axis, enhanced, color='green', alpha=0.7, linewidth=0.5)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Enhanced Audio')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f"{audio_id}_waveform.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"waveform_path": plot_path}
        
    def generate_spectrogram_comparison(self, original: np.ndarray, enhanced: np.ndarray,
                                      audio_id: str, sample_rate: int = 16000) -> Dict[str, str]:
        """Generate spectrogram comparison.
        
        Args:
            original: Original audio signal
            enhanced: Enhanced audio signal
            audio_id: Audio identifier
            sample_rate: Sample rate
            
        Returns:
            Dictionary with visualization paths
        """
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Compute spectrograms
        nperseg = int(0.025 * sample_rate)  # 25ms window
        noverlap = int(0.020 * sample_rate)  # 20ms overlap
        
        # Original spectrogram
        f, t, Sxx_orig = signal.spectrogram(original, sample_rate, 
                                           nperseg=nperseg, noverlap=noverlap)
        
        # Enhanced spectrogram
        f, t, Sxx_enh = signal.spectrogram(enhanced, sample_rate,
                                          nperseg=nperseg, noverlap=noverlap)
        
        # Plot in dB scale
        vmin, vmax = -80, 0  # dB range
        
        # Original
        im1 = ax1.pcolormesh(t, f, 10 * np.log10(Sxx_orig + 1e-10), 
                            shading='gouraud', cmap='viridis',
                            vmin=vmin, vmax=vmax)
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('Original Audio Spectrogram')
        ax1.set_ylim([0, 8000])  # Focus on speech frequencies
        
        # Enhanced
        im2 = ax2.pcolormesh(t, f, 10 * np.log10(Sxx_enh + 1e-10),
                            shading='gouraud', cmap='viridis',
                            vmin=vmin, vmax=vmax)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Enhanced Audio Spectrogram')
        ax2.set_ylim([0, 8000])
        
        # Add colorbar
        cbar = plt.colorbar(im2, ax=[ax1, ax2])
        cbar.set_label('Power (dB)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f"{audio_id}_spectrogram.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"spectrogram_path": plot_path}
        
    def generate_metrics_chart(self, metrics: Dict[str, Dict[str, float]], 
                             audio_id: str) -> Dict[str, str]:
        """Generate metrics comparison chart.
        
        Args:
            metrics: Dictionary of metrics with before/after values
            audio_id: Audio identifier
            
        Returns:
            Dictionary with chart path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        metric_names = list(metrics.keys())
        before_values = [metrics[m].get("before", 0) for m in metric_names]
        after_values = [metrics[m].get("after", 0) for m in metric_names]
        improvements = [metrics[m].get("improvement", 0) for m in metric_names]
        
        # Bar positions
        x = np.arange(len(metric_names))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, before_values, width, label='Before', 
                       color='lightcoral', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_values, width, label='After',
                       color='lightgreen', alpha=0.8)
        
        # Add improvement values on top
        for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvements)):
            if imp > 0:
                ax.text(i, max(b1.get_height(), b2.get_height()) + 0.1,
                       f'+{imp:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Customize chart
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title(f'Audio Enhancement Metrics Comparison - {audio_id}')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f"{audio_id}_metrics.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"metrics_chart_path": plot_path}
        
    def generate_frequency_response_comparison(self, original: np.ndarray, 
                                            enhanced: np.ndarray,
                                            audio_id: str,
                                            sample_rate: int = 16000) -> Dict[str, str]:
        """Generate frequency response comparison.
        
        Args:
            original: Original audio
            enhanced: Enhanced audio
            audio_id: Audio identifier
            sample_rate: Sample rate
            
        Returns:
            Dictionary with plot path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compute frequency response
        freqs_orig, response_orig = signal.periodogram(original, sample_rate)
        freqs_enh, response_enh = signal.periodogram(enhanced, sample_rate)
        
        # Convert to dB
        response_orig_db = 10 * np.log10(response_orig + 1e-10)
        response_enh_db = 10 * np.log10(response_enh + 1e-10)
        
        # Plot
        ax.semilogx(freqs_orig, response_orig_db, 'b-', alpha=0.7, 
                   label='Original', linewidth=1.5)
        ax.semilogx(freqs_enh, response_enh_db, 'g-', alpha=0.7,
                   label='Enhanced', linewidth=1.5)
        
        # Focus on speech frequencies
        ax.set_xlim([50, 8000])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.set_title(f'Frequency Response Comparison - {audio_id}')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f"{audio_id}_frequency_response.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"frequency_response_path": plot_path}
        
    def generate_comparison_report(self, original_audio: np.ndarray,
                                 enhanced_audio: np.ndarray,
                                 audio_id: str,
                                 metrics: Dict[str, float],
                                 sample_rate: int = 16000) -> Dict[str, Any]:
        """Generate comprehensive comparison report.
        
        Args:
            original_audio: Original audio signal
            enhanced_audio: Enhanced audio signal
            audio_id: Audio identifier
            metrics: Enhancement metrics
            sample_rate: Sample rate
            
        Returns:
            Dictionary with report data and paths
        """
        report = {
            "audio_id": audio_id,
            "timestamp": datetime.now().isoformat(),
            "sample_rate": sample_rate,
            "duration": len(original_audio) / sample_rate,
            "metrics": metrics,
            "visualizations": {}
        }
        
        # Generate all visualizations
        try:
            # Waveform comparison
            waveform_result = self.generate_waveform_comparison(
                original_audio, enhanced_audio, audio_id
            )
            report["visualizations"].update(waveform_result)
            
            # Spectrogram comparison
            spectrogram_result = self.generate_spectrogram_comparison(
                original_audio, enhanced_audio, audio_id, sample_rate
            )
            report["visualizations"].update(spectrogram_result)
            
            # Metrics chart
            metrics_data = {
                "SNR": {
                    "before": metrics.get("snr_before", 10),
                    "after": metrics.get("snr_after", 18),
                    "improvement": metrics.get("snr_improvement", 8)
                },
                "PESQ": {
                    "before": metrics.get("pesq_before", 2.5),
                    "after": metrics.get("pesq", 3.4),
                    "improvement": metrics.get("pesq", 3.4) - metrics.get("pesq_before", 2.5)
                },
                "STOI": {
                    "before": metrics.get("stoi_before", 0.75),
                    "after": metrics.get("stoi", 0.91),
                    "improvement": metrics.get("stoi", 0.91) - metrics.get("stoi_before", 0.75)
                }
            }
            
            metrics_result = self.generate_metrics_chart(metrics_data, audio_id)
            report["visualizations"].update(metrics_result)
            
            # Frequency response
            freq_result = self.generate_frequency_response_comparison(
                original_audio, enhanced_audio, audio_id, sample_rate
            )
            report["visualizations"].update(freq_result)
            
        except Exception as e:
            report["error"] = str(e)
            
        # Generate recommendation
        report["recommendation"] = self._generate_recommendation(metrics)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        # Save JSON report
        json_path = os.path.join(self.output_dir, f"{audio_id}_report.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
        
    def _generate_recommendation(self, metrics: Dict[str, float]) -> str:
        """Generate enhancement recommendation based on metrics.
        
        Args:
            metrics: Enhancement metrics
            
        Returns:
            Recommendation text
        """
        snr_imp = metrics.get("snr_improvement", 0)
        pesq = metrics.get("pesq", 0)
        stoi = metrics.get("stoi", 0)
        
        if snr_imp < 3:
            return "Enhancement had minimal effect. Consider using more aggressive settings."
        elif snr_imp > 15:
            return "Significant enhancement achieved. Monitor for potential artifacts."
        elif pesq < 3.0:
            return "Quality below target. Consider adjusting enhancement parameters."
        elif stoi < 0.85:
            return "Intelligibility below target. May need different enhancement approach."
        else:
            return "Enhancement successful. All metrics within target ranges."
            
    def _generate_html_report(self, report: Dict[str, Any]) -> None:
        """Generate HTML comparison report.
        
        Args:
            report: Report data
        """
        audio_id = report["audio_id"]
        
        # Build image HTML
        images_html = ""
        for key, path in report.get("visualizations", {}).items():
            if path and isinstance(path, str) and os.path.exists(path):
                rel_path = os.path.relpath(path, self.output_dir)
                images_html += f'<img src="{rel_path}" alt="{key}" style="max-width: 100%; margin: 10px 0;">\n'
                
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Audio Enhancement Comparison - {audio_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        .metrics {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .recommendation {{ background: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .warning {{ background: #fff3cd; border-left-color: #ffc107; }}
        img {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Audio Enhancement Comparison Report</h1>
        <p><strong>Audio ID:</strong> {audio_id}</p>
        <p><strong>Generated:</strong> {report['timestamp']}</p>
        <p><strong>Duration:</strong> {report['duration']:.2f} seconds</p>
        
        <div class="metrics">
            <h2>Enhancement Metrics</h2>
            <ul>
                <li>SNR Improvement: +{report['metrics'].get('snr_improvement', 0):.1f} dB</li>
                <li>PESQ Score: {report['metrics'].get('pesq', 0):.2f}/4.5</li>
                <li>STOI Score: {report['metrics'].get('stoi', 0):.3f}/1.0</li>
            </ul>
        </div>
        
        <div class="recommendation">
            <h3>Recommendation</h3>
            <p>{report['recommendation']}</p>
        </div>
        
        <h2>Visual Comparisons</h2>
        {images_html}
    </div>
</body>
</html>"""
        
        html_path = os.path.join(self.output_dir, f"{audio_id}_report.html")
        with open(html_path, 'w') as f:
            f.write(html)
            
    def batch_compare(self, audio_pairs: List[Tuple[np.ndarray, np.ndarray, str]],
                     sample_rate: int = 16000) -> List[Dict[str, Any]]:
        """Compare multiple audio pairs.
        
        Args:
            audio_pairs: List of (original, enhanced, id) tuples
            sample_rate: Sample rate
            
        Returns:
            List of comparison reports
        """
        reports = []
        
        for original, enhanced, audio_id in audio_pairs:
            # Basic metrics (mock for now)
            metrics = {
                "snr_improvement": np.random.uniform(5, 10),
                "pesq": np.random.uniform(3.0, 4.0),
                "stoi": np.random.uniform(0.85, 0.95)
            }
            
            report = self.generate_comparison_report(
                original, enhanced, audio_id, metrics, sample_rate
            )
            reports.append(report)
            
        return reports