"""Real-time monitoring dashboard for audio enhancement pipeline."""

import os
import json
import time
import sys
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from threading import Lock
from pathlib import Path

# For terminal UI
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import metrics collector
from .metrics_collector import MetricsCollector, get_gpu_metrics


class EnhancementDashboard:
    """Real-time audio enhancement monitoring dashboard with terminal UI."""
    
    def __init__(self, update_interval: int = 100, output_dir: str = "./dashboard_output"):
        """Initialize the dashboard.
        
        Args:
            update_interval: Number of files between dashboard updates
            output_dir: Directory for dashboard output files
        """
        self.update_interval = update_interval
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.processed_count = 0
        self.total_count = 0
        self.failed_count = 0
        self.low_quality_count = 0
        
        self.current_metrics = {}
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 updates
        self.progress_percent = 0.0
        self.processing_rate = 0.0
        self.eta_seconds = 0
        
        # Quality thresholds
        self.quality_thresholds = {
            'snr_improvement': 3.0,
            'pesq': 3.0,
            'stoi': 0.85
        }
        
        # Issue tracking
        self.issues = {
            'failed_files': [],
            'low_quality_files': [],
            'warnings': []
        }
        
        # Terminal UI components
        self.console = Console() if RICH_AVAILABLE else None
        self.live_display = None
        self.layout = None
        
        # Thread safety
        self.lock = Lock()
        
        # Metrics collector
        self.metrics_collector = MetricsCollector(output_dir=output_dir)
        
        # Initialize terminal UI
        if RICH_AVAILABLE:
            self._init_terminal_ui()
    
    def _init_terminal_ui(self):
        """Initialize terminal UI layout."""
        self.layout = Layout()
        
        # Define layout structure
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Body sections
        self.layout["body"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="metrics", ratio=1),
            Layout(name="gpu", ratio=1)
        )
        
        # Update header
        self.layout["header"].update(
            Panel(
                Text("🎵 Audio Enhancement Dashboard", style="bold white on blue", justify="center"),
                border_style="blue"
            )
        )
    
    def update(self, processed_count: int, total_count: int, 
               current_metrics: Dict[str, float], 
               sample_id: Optional[str] = None,
               failed: bool = False,
               error: Optional[str] = None) -> None:
        """Update dashboard with new metrics.
        
        Args:
            processed_count: Number of files processed
            total_count: Total number of files
            current_metrics: Current aggregate metrics
            sample_id: Optional sample identifier
            failed: Whether processing failed
            error: Error message if failed
        """
        with self.lock:
            self.processed_count = processed_count
            self.total_count = total_count
            
            if failed:
                self.failed_count += 1
                if sample_id:
                    self.issues['failed_files'].append({
                        'sample_id': sample_id,
                        'error': error,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Update metrics
            self.current_metrics = current_metrics
            self.progress_percent = (processed_count / total_count * 100) if total_count > 0 else 0
            
            # Calculate processing rate and ETA
            elapsed = time.time() - self.start_time
            self.processing_rate = (processed_count / elapsed * 60) if elapsed > 0 else 0
            
            remaining_files = total_count - processed_count
            if self.processing_rate > 0:
                self.eta_seconds = (remaining_files / self.processing_rate) * 60
            
            # Check for quality issues
            if not failed and current_metrics:
                if current_metrics.get('snr_improvement', 0) < self.quality_thresholds['snr_improvement']:
                    self.low_quality_count += 1
                    if sample_id:
                        self.issues['low_quality_files'].append({
                            'sample_id': sample_id,
                            'metrics': current_metrics,
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Add to history
            self.metrics_history.append({
                "timestamp": datetime.now().isoformat(),
                "processed": processed_count,
                "metrics": current_metrics.copy() if current_metrics else {},
                "gpu_metrics": get_gpu_metrics()
            })
            
            # Update display
            if processed_count % self.update_interval == 0 or processed_count == total_count:
                self._update_display()
                self._save_snapshot()
    
    def _update_display(self):
        """Update the terminal display."""
        if not RICH_AVAILABLE or not self.layout:
            # Fallback to simple print
            self._print_simple_update()
            return
        
        # Update stats section
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Label", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Progress", f"{self.processed_count}/{self.total_count} ({self.progress_percent:.1f}%)")
        stats_table.add_row("Processing Rate", f"{self.processing_rate:.0f} files/min")
        stats_table.add_row("ETA", self._format_eta())
        stats_table.add_row("Failed Files", f"{self.failed_count}")
        stats_table.add_row("Low Quality", f"{self.low_quality_count}")
        
        self.layout["stats"].update(Panel(stats_table, title="Processing Stats", border_style="green"))
        
        # Update metrics section
        metrics_table = Table(show_header=False, box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        
        metrics_table.add_row("SNR Improvement", f"+{self.current_metrics.get('avg_snr_improvement', 0):.1f} dB")
        metrics_table.add_row("PESQ Score", f"{self.current_metrics.get('avg_pesq', 0):.2f}/4.5")
        metrics_table.add_row("STOI Score", f"{self.current_metrics.get('avg_stoi', 0):.3f}/1.0")
        metrics_table.add_row("Processing Time", f"{self.current_metrics.get('avg_processing_time', 0):.2f}s")
        
        self.layout["metrics"].update(Panel(metrics_table, title="Quality Metrics", border_style="yellow"))
        
        # Update GPU section
        gpu_metrics = get_gpu_metrics()
        gpu_table = Table(show_header=False, box=None)
        gpu_table.add_column("Resource", style="cyan")
        gpu_table.add_column("Value", style="white")
        
        gpu_table.add_row("Memory Used", f"{gpu_metrics['memory_used']:.0f} MB")
        gpu_table.add_row("Memory Total", f"{gpu_metrics['memory_total']:.0f} MB")
        gpu_table.add_row("Utilization", f"{gpu_metrics['utilization']:.0f}%")
        gpu_table.add_row("Temperature", f"{gpu_metrics['temperature']:.0f}°C")
        
        self.layout["gpu"].update(Panel(gpu_table, title="GPU Status", border_style="red"))
        
        # Update footer
        self.layout["footer"].update(
            Panel(
                Text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", justify="center"),
                border_style="blue"
            )
        )
        
        # Display
        if self.console:
            self.console.clear()
            self.console.print(self.layout)
    
    def _print_simple_update(self):
        """Simple console output for when Rich is not available."""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("=" * 80)
        print(f"Audio Enhancement Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"Progress: {self.processed_count}/{self.total_count} ({self.progress_percent:.1f}%)")
        print(f"Processing Rate: {self.processing_rate:.0f} files/min")
        print(f"ETA: {self._format_eta()}")
        print(f"Failed Files: {self.failed_count}")
        print(f"Low Quality: {self.low_quality_count}")
        print("-" * 80)
        print(f"SNR Improvement: +{self.current_metrics.get('avg_snr_improvement', 0):.1f} dB")
        print(f"PESQ Score: {self.current_metrics.get('avg_pesq', 0):.2f}/4.5")
        print(f"STOI Score: {self.current_metrics.get('avg_stoi', 0):.3f}/1.0")
        print("=" * 80)
    
    def _format_eta(self) -> str:
        """Format ETA as human-readable string."""
        if self.eta_seconds <= 0:
            return "N/A"
        
        delta = timedelta(seconds=int(self.eta_seconds))
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")
        
        return " ".join(parts)
    
    def _save_snapshot(self):
        """Save current dashboard state as snapshot."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'processed_count': self.processed_count,
            'total_count': self.total_count,
            'progress_percent': self.progress_percent,
            'processing_rate': self.processing_rate,
            'eta_seconds': self.eta_seconds,
            'failed_count': self.failed_count,
            'low_quality_count': self.low_quality_count,
            'current_metrics': self.current_metrics,
            'issues': self.issues
        }
        
        snapshot_file = os.path.join(self.output_dir, 'dashboard_snapshot.json')
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self.lock:
            return {
                'percentage': self.progress_percent,
                'processed': self.processed_count,
                'total': self.total_count,
                'processing_rate': self.processing_rate,
                'eta_seconds': self.eta_seconds,
                'eta_formatted': self._format_eta()
            }
    
    def get_quality_trend(self) -> str:
        """Analyze quality trend from recent metrics."""
        if len(self.metrics_history) < 10:
            return "Insufficient data"
        
        # Get recent SNR improvements
        recent_snrs = [
            entry['metrics'].get('avg_snr_improvement', 0) 
            for entry in list(self.metrics_history)[-10:]
            if 'avg_snr_improvement' in entry.get('metrics', {})
        ]
        
        if len(recent_snrs) < 2:
            return "Insufficient data"
        
        # Calculate trend
        trend = np.mean(recent_snrs[-5:]) - np.mean(recent_snrs[:5])
        
        if trend > 0.5:
            return "Improving - Quality metrics are trending upward"
        elif trend < -0.5:
            return "Degrading - Quality metrics are trending downward"
        else:
            return "Stable - Quality metrics are consistent"
    
    def save_metrics_snapshot(self, metrics: Dict[str, float]) -> None:
        """Save metrics snapshot to file."""
        snapshot_file = os.path.join(self.output_dir, 'metrics_snapshots.json')
        
        # Load existing snapshots
        snapshots = []
        if os.path.exists(snapshot_file):
            with open(snapshot_file, 'r') as f:
                snapshots = json.load(f)
        
        # Add new snapshot
        snapshots.append({
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
        
        # Save back
        with open(snapshot_file, 'w') as f:
            json.dump(snapshots, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate comprehensive processing report."""
        report_data = {
            'generation_time': datetime.now().isoformat(),
            'processing_duration': time.time() - self.start_time,
            'total_files': self.total_count,
            'processed_files': self.processed_count,
            'failed_files': self.failed_count,
            'low_quality_files': self.low_quality_count,
            'success_rate': ((self.processed_count - self.failed_count) / self.processed_count * 100) 
                          if self.processed_count > 0 else 0,
            'metrics': self.current_metrics,
            'quality_trend': self.get_quality_trend(),
            'issues': self.issues
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, f'enhancement_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_file
    
    def get_gpu_memory(self) -> float:
        """Get GPU memory usage percentage."""
        metrics = get_gpu_metrics()
        if metrics['memory_total'] > 0:
            return (metrics['memory_used'] / metrics['memory_total']) * 100
        return 0.0
    
    def get_gpu_temp(self) -> float:
        """Get GPU temperature."""
        return get_gpu_metrics()['temperature']