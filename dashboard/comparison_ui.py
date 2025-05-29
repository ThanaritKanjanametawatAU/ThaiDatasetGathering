"""Before/after comparison analysis and visualization."""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import logging
from scipy import signal
import librosa
import librosa.display


logger = logging.getLogger(__name__)


class ComparisonAnalyzer:
    """Analyzes and visualizes before/after audio comparisons."""
    
    def __init__(self, output_dir: str = "comparison_output"):
        """Initialize comparison analyzer.
        
        Args:
            output_dir: Directory for saving comparison outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for comparisons
        self.comparisons: List[Dict] = []
        
    def analyze_audio_pair(self, original_audio: np.ndarray, 
                          enhanced_audio: np.ndarray,
                          sample_rate: int,
                          audio_id: str) -> Dict[str, Any]:
        """Analyze a pair of original and enhanced audio.
        
        Args:
            original_audio: Original audio samples
            enhanced_audio: Enhanced audio samples
            sample_rate: Sample rate
            audio_id: Identifier for this audio pair
            
        Returns:
            Dictionary with analysis results
        """
        # Basic validation
        if len(original_audio) != len(enhanced_audio):
            # Pad or trim to match lengths
            min_len = min(len(original_audio), len(enhanced_audio))
            original_audio = original_audio[:min_len]
            enhanced_audio = enhanced_audio[:min_len]
            
        # Calculate metrics
        metrics = self._calculate_metrics(original_audio, enhanced_audio, sample_rate)
        
        # Extract waveform data for visualization
        waveform_data = {
            'original': original_audio.tolist() if len(original_audio) < 10000 else original_audio[::len(original_audio)//10000].tolist(),
            'enhanced': enhanced_audio.tolist() if len(enhanced_audio) < 10000 else enhanced_audio[::len(enhanced_audio)//10000].tolist()
        }
        
        # Extract spectrogram data
        spectrogram_data = self._calculate_spectrograms(original_audio, enhanced_audio, sample_rate)
        
        # Generate quality verdict
        quality_verdict = self.generate_verdict(metrics)
        
        comparison = {
            'audio_id': audio_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'waveform_data': waveform_data,
            'spectrogram_data': spectrogram_data,
            'quality_verdict': quality_verdict
        }
        
        self.comparisons.append(comparison)
        return comparison
        
    def _calculate_metrics(self, original: np.ndarray, 
                          enhanced: np.ndarray,
                          sample_rate: int) -> Dict[str, float]:
        """Calculate comparison metrics between audio pairs.
        
        Args:
            original: Original audio
            enhanced: Enhanced audio
            sample_rate: Sample rate
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # SNR calculation (simplified)
        signal_power_orig = np.mean(original ** 2)
        noise_estimate_orig = np.std(original[np.abs(original) < np.percentile(np.abs(original), 10)])
        snr_orig = 10 * np.log10(signal_power_orig / (noise_estimate_orig ** 2 + 1e-10))
        
        signal_power_enh = np.mean(enhanced ** 2)
        noise_estimate_enh = np.std(enhanced[np.abs(enhanced) < np.percentile(np.abs(enhanced), 10)])
        snr_enh = 10 * np.log10(signal_power_enh / (noise_estimate_enh ** 2 + 1e-10))
        
        metrics['snr_before'] = float(snr_orig)
        metrics['snr_after'] = float(snr_enh)
        metrics['snr_improvement'] = float(snr_enh - snr_orig)
        
        # Spectral distortion
        orig_fft = np.fft.rfft(original)
        enh_fft = np.fft.rfft(enhanced)
        spectral_distortion = np.sqrt(np.mean((np.abs(orig_fft) - np.abs(enh_fft)) ** 2))
        metrics['spectral_distortion'] = float(spectral_distortion)
        
        # Energy ratio
        energy_ratio = np.sum(enhanced ** 2) / (np.sum(original ** 2) + 1e-10)
        metrics['energy_ratio'] = float(energy_ratio)
        
        # Zero crossing rate change
        zcr_orig = np.sum(np.abs(np.diff(np.sign(original)))) / len(original)
        zcr_enh = np.sum(np.abs(np.diff(np.sign(enhanced)))) / len(enhanced)
        metrics['zcr_change'] = float(zcr_enh - zcr_orig)
        
        return metrics
        
    def _calculate_spectrograms(self, original: np.ndarray,
                               enhanced: np.ndarray,
                               sample_rate: int) -> Dict[str, Any]:
        """Calculate spectrogram data for visualization.
        
        Args:
            original: Original audio
            enhanced: Enhanced audio
            sample_rate: Sample rate
            
        Returns:
            Dictionary with spectrogram data
        """
        # Calculate spectrograms
        f_orig, t_orig, Sxx_orig = signal.spectrogram(original, sample_rate, nperseg=512)
        f_enh, t_enh, Sxx_enh = signal.spectrogram(enhanced, sample_rate, nperseg=512)
        
        # Convert to dB
        Sxx_orig_db = 10 * np.log10(Sxx_orig + 1e-10)
        Sxx_enh_db = 10 * np.log10(Sxx_enh + 1e-10)
        
        # Downsample for JSON storage
        stride = max(1, len(t_orig) // 100)
        
        return {
            'frequencies': f_orig[::2].tolist(),  # Every other frequency bin
            'times': t_orig[::stride].tolist(),
            'original_spectrogram': Sxx_orig_db[:, ::stride].tolist(),
            'enhanced_spectrogram': Sxx_enh_db[:, ::stride].tolist()
        }
        
    def generate_comparison_plot(self, original_audio: np.ndarray,
                               enhanced_audio: np.ndarray,
                               audio_id: str,
                               metrics: Dict[str, float],
                               sample_rate: int = 16000) -> str:
        """Generate comprehensive comparison plot.
        
        Args:
            original_audio: Original audio samples
            enhanced_audio: Enhanced audio samples
            audio_id: Audio identifier
            metrics: Calculated metrics
            sample_rate: Sample rate
            
        Returns:
            Path to saved plot
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Time axis
        time_orig = np.arange(len(original_audio)) / sample_rate
        time_enh = np.arange(len(enhanced_audio)) / sample_rate
        
        # 1. Waveform comparison
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_orig, original_audio, alpha=0.7, label='Original', linewidth=0.5)
        ax1.plot(time_enh, enhanced_audio, alpha=0.7, label='Enhanced', linewidth=0.5)
        ax1.set_title(f'Waveform Comparison - {audio_id}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Original spectrogram
        ax2 = fig.add_subplot(gs[1, 0])
        D_orig = librosa.stft(original_audio, n_fft=2048, hop_length=512)
        S_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
        img1 = librosa.display.specshow(S_orig, sr=sample_rate, x_axis='time', 
                                       y_axis='hz', ax=ax2, cmap='viridis')
        ax2.set_title('Original Spectrogram')
        ax2.set_ylim(0, 8000)  # Focus on speech frequencies
        
        # 3. Enhanced spectrogram
        ax3 = fig.add_subplot(gs[1, 1])
        D_enh = librosa.stft(enhanced_audio, n_fft=2048, hop_length=512)
        S_enh = librosa.amplitude_to_db(np.abs(D_enh), ref=np.max)
        img2 = librosa.display.specshow(S_enh, sr=sample_rate, x_axis='time', 
                                       y_axis='hz', ax=ax3, cmap='viridis')
        ax3.set_title('Enhanced Spectrogram')
        ax3.set_ylim(0, 8000)
        
        # 4. Difference spectrogram
        ax4 = fig.add_subplot(gs[1, 2])
        S_diff = S_enh - S_orig
        img3 = librosa.display.specshow(S_diff, sr=sample_rate, x_axis='time', 
                                       y_axis='hz', ax=ax4, cmap='coolwarm',
                                       vmin=-20, vmax=20)
        ax4.set_title('Difference (Enhanced - Original)')
        ax4.set_ylim(0, 8000)
        plt.colorbar(img3, ax=ax4, format='%+2.0f dB')
        
        # 5. Metrics visualization
        ax5 = fig.add_subplot(gs[2, 0])
        metric_names = ['SNR\nImprovement', 'Spectral\nDistortion', 'Energy\nRatio']
        metric_values = [
            metrics.get('snr_improvement', 0),
            -metrics.get('spectral_distortion', 0) / 1000,  # Normalize and invert
            metrics.get('energy_ratio', 1)
        ]
        colors = ['green' if v > 0 else 'red' for v in metric_values]
        bars = ax5.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax5.set_title('Key Metrics')
        ax5.set_ylabel('Value')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 6. Frequency response comparison
        ax6 = fig.add_subplot(gs[2, 1])
        freq_orig = np.fft.rfftfreq(len(original_audio), 1/sample_rate)
        fft_orig = np.abs(np.fft.rfft(original_audio))
        fft_enh = np.abs(np.fft.rfft(enhanced_audio))
        
        # Smooth for better visualization
        from scipy.ndimage import gaussian_filter1d
        fft_orig_smooth = gaussian_filter1d(20 * np.log10(fft_orig + 1e-10), sigma=5)
        fft_enh_smooth = gaussian_filter1d(20 * np.log10(fft_enh + 1e-10), sigma=5)
        
        ax6.plot(freq_orig, fft_orig_smooth, alpha=0.7, label='Original')
        ax6.plot(freq_orig, fft_enh_smooth, alpha=0.7, label='Enhanced')
        ax6.set_xlim(0, 4000)  # Focus on speech frequencies
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Magnitude (dB)')
        ax6.set_title('Frequency Response')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Quality verdict
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        verdict = self.generate_verdict(metrics)
        verdict_color = {
            'improved': 'green',
            'degraded': 'red',
            'unchanged': 'orange'
        }.get(verdict['status'], 'black')
        
        ax7.text(0.5, 0.7, verdict['status'].upper(), 
                ha='center', va='center', fontsize=20, 
                weight='bold', color=verdict_color)
        ax7.text(0.5, 0.3, verdict['summary'], 
                ha='center', va='center', fontsize=12, 
                wrap=True, multialignment='center')
        
        # Overall title
        fig.suptitle(f'Audio Enhancement Analysis - {audio_id}', fontsize=16)
        
        # Save plot
        plot_path = self.output_dir / f"{audio_id}_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
        
    def generate_verdict(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate quality verdict based on metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary with status and summary
        """
        # Analyze metrics
        snr_imp = metrics.get('snr_improvement', 0)
        spectral_dist = metrics.get('spectral_distortion', 0)
        energy_ratio = metrics.get('energy_ratio', 1)
        
        # Scoring
        score = 0
        reasons = []
        
        if snr_imp > 3:
            score += 2
            reasons.append(f"SNR improved by {snr_imp:.1f} dB")
        elif snr_imp < -1:
            score -= 2
            reasons.append(f"SNR degraded by {abs(snr_imp):.1f} dB")
            
        if spectral_dist < 1000:
            score += 1
            reasons.append("Low spectral distortion")
        elif spectral_dist > 5000:
            score -= 1
            reasons.append("High spectral distortion")
            
        if 0.7 < energy_ratio < 1.3:
            score += 1
            reasons.append("Energy well preserved")
        elif energy_ratio < 0.5 or energy_ratio > 2:
            score -= 1
            reasons.append("Significant energy change")
            
        # Determine status
        if score >= 2:
            status = 'improved'
            summary = "Audio successfully enhanced. " + ". ".join(reasons[:2])
        elif score <= -1:
            status = 'degraded'
            summary = "Audio quality degraded. " + ". ".join(reasons[:2])
        else:
            status = 'unchanged'
            summary = "Minimal change detected. " + ". ".join(reasons[:2])
            
        return {
            'status': status,
            'summary': summary,
            'score': score,
            'reasons': reasons
        }
        
    def generate_batch_report(self, comparisons: List[Dict]) -> Dict[str, Any]:
        """Generate report for batch of comparisons.
        
        Args:
            comparisons: List of comparison results
            
        Returns:
            Summary report
        """
        if not comparisons:
            return {
                'total_samples': 0,
                'improved_count': 0,
                'degraded_count': 0,
                'unchanged_count': 0
            }
            
        # Count by status
        status_counts = {
            'improved': 0,
            'degraded': 0,
            'unchanged': 0
        }
        
        # Aggregate metrics
        snr_improvements = []
        pesq_scores = []
        
        for comp in comparisons:
            metrics = comp.get('metrics', {})
            
            # Count status
            if 'quality_improved' in metrics:
                if metrics['quality_improved']:
                    status_counts['improved'] += 1
                else:
                    status_counts['degraded'] += 1
            
            # Collect metrics
            if 'snr_improvement' in metrics:
                snr_improvements.append(metrics['snr_improvement'])
            if 'pesq' in metrics:
                pesq_scores.append(metrics['pesq'])
                
        report = {
            'total_samples': len(comparisons),
            'improved_count': status_counts['improved'],
            'degraded_count': status_counts['degraded'],
            'unchanged_count': status_counts['unchanged'],
            'avg_snr_improvement': np.mean(snr_improvements) if snr_improvements else 0,
            'avg_pesq': np.mean(pesq_scores) if pesq_scores else 0,
            'improvement_rate': (status_counts['improved'] / len(comparisons) * 100) if comparisons else 0
        }
        
        return report
        
    def generate_interactive_ui(self, comparisons: List[Dict]) -> str:
        """Generate interactive HTML UI for comparisons.
        
        Args:
            comparisons: List of comparison data
            
        Returns:
            Path to HTML file
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Audio Comparison Interface</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .comparison-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .comparison-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .audio-controls {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        .audio-player {
            width: 100%;
            height: 40px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .metric {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
        }
        .metric-value {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        .status-improved { color: #27ae60; }
        .status-degraded { color: #e74c3c; }
        .status-unchanged { color: #f39c12; }
        .filter-controls {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .summary-stats {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-around;
        }
        .stat-box {
            text-align: center;
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Audio Comparison Interface</h1>
            <p>Interactive before/after comparison of enhanced audio samples</p>
        </div>
        
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-value">{total_samples}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-box">
                <div class="stat-value status-improved">{improved_count}</div>
                <div class="stat-label">Improved</div>
            </div>
            <div class="stat-box">
                <div class="stat-value status-degraded">{degraded_count}</div>
                <div class="stat-label">Degraded</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{avg_improvement:.1f} dB</div>
                <div class="stat-label">Avg SNR Improvement</div>
            </div>
        </div>
        
        <div class="filter-controls">
            <label>
                Filter by status:
                <select id="statusFilter" onchange="filterCards()">
                    <option value="all">All</option>
                    <option value="improved">Improved</option>
                    <option value="degraded">Degraded</option>
                    <option value="unchanged">Unchanged</option>
                </select>
            </label>
            <label>
                Sort by:
                <select id="sortBy" onchange="sortCards()">
                    <option value="id">Sample ID</option>
                    <option value="snr">SNR Improvement</option>
                    <option value="status">Status</option>
                </select>
            </label>
        </div>
        
        <div class="comparison-grid" id="comparisonGrid">
            {comparison_cards}
        </div>
    </div>
    
    <script>
        function filterCards() {
            const filter = document.getElementById('statusFilter').value;
            const cards = document.querySelectorAll('.comparison-card');
            
            cards.forEach(card => {
                if (filter === 'all' || card.dataset.status === filter) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
        
        function sortCards() {
            const sortBy = document.getElementById('sortBy').value;
            const grid = document.getElementById('comparisonGrid');
            const cards = Array.from(grid.children);
            
            cards.sort((a, b) => {
                if (sortBy === 'id') {
                    return a.dataset.id.localeCompare(b.dataset.id);
                } else if (sortBy === 'snr') {
                    return parseFloat(b.dataset.snr) - parseFloat(a.dataset.snr);
                } else if (sortBy === 'status') {
                    return a.dataset.status.localeCompare(b.dataset.status);
                }
            });
            
            cards.forEach(card => grid.appendChild(card));
        }
        
        function playComparison(audioId) {
            // Toggle between original and enhanced audio
            console.log('Playing comparison for', audioId);
        }
    </script>
</body>
</html>
        """
        
        # Generate comparison cards
        comparison_cards = []
        improved_count = 0
        degraded_count = 0
        total_snr_improvement = 0
        
        for comp in comparisons[:50]:  # Limit to 50 for performance
            metrics = comp.get('metrics', {})
            snr_imp = metrics.get('snr_improvement', 0)
            
            # Determine status
            if snr_imp > 3:
                status = 'improved'
                improved_count += 1
            elif snr_imp < -1:
                status = 'degraded'
                degraded_count += 1
            else:
                status = 'unchanged'
                
            total_snr_improvement += snr_imp
            
            card_html = f"""
            <div class="comparison-card" data-id="{comp['audio_id']}" 
                 data-status="{status}" data-snr="{snr_imp}">
                <h3>{comp['audio_id']}</h3>
                
                <div class="audio-controls">
                    <button onclick="playComparison('{comp['audio_id']}')">
                        ▶️ Play Comparison
                    </button>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">SNR Improvement</div>
                        <div class="metric-value status-{status}">{snr_imp:.1f} dB</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Status</div>
                        <div class="metric-value status-{status}">{status.upper()}</div>
                    </div>
                </div>
                
                <div style="margin-top: 10px; font-size: 14px; color: #666;">
                    Original: {comp.get('original_path', 'N/A')}<br>
                    Enhanced: {comp.get('enhanced_path', 'N/A')}
                </div>
            </div>
            """
            comparison_cards.append(card_html)
            
        # Fill template
        html_content = html_template.format(
            total_samples=len(comparisons),
            improved_count=improved_count,
            degraded_count=degraded_count,
            avg_improvement=total_snr_improvement / len(comparisons) if comparisons else 0,
            comparison_cards='\n'.join(comparison_cards)
        )
        
        # Save HTML
        html_path = self.output_dir / "comparison_interface.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return str(html_path)
        
    def save_comparison_data(self, filepath: Optional[str] = None) -> str:
        """Save all comparison data to file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = self.output_dir / f"comparisons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_comparisons': len(self.comparisons),
            'comparisons': self.comparisons
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(self.comparisons)} comparisons to {filepath}")
        return str(filepath)