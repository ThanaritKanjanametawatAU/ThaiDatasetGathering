"""Audio Enhancement Dashboard - Real-time monitoring and visualization."""

import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import curses
import logging
import psutil
import GPUtil

from .metrics_collector import MetricsCollector, EnhancementMetrics


logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Real-time processing statistics."""
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    current_file: str = ""
    processing_speed: float = 0.0  # files per minute
    avg_processing_time: float = 0.0  # seconds per file
    estimated_time_remaining: timedelta = timedelta()
    start_time: datetime = None
    last_update: datetime = None
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    gpu_temperature: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


class EnhancementDashboard:
    """Real-time monitoring dashboard for audio enhancement processing."""
    
    def __init__(self, metrics_collector: MetricsCollector, update_interval: int = 100):
        """Initialize the dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
            update_interval: Update dashboard every N processed files
        """
        self.metrics_collector = metrics_collector
        self.update_interval = update_interval
        self.stats = ProcessingStats()
        self.is_running = False
        self.update_thread = None
        self.screen = None
        self.last_metrics_window = []
        self.window_size = 1000
        self.alerts = []
        self.quality_trends = []
        self.processing_history = []
        
    def start(self, total_files: int):
        """Start the dashboard monitoring."""
        self.stats.total_files = total_files
        self.stats.start_time = datetime.now()
        self.stats.last_update = datetime.now()
        self.is_running = True
        
        # Start system monitoring thread
        self.update_thread = threading.Thread(target=self._update_system_stats)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def stop(self):
        """Stop the dashboard monitoring."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
            
    def update_progress(self, file_path: str, success: bool = True):
        """Update processing progress."""
        self.stats.processed_files += 1
        self.stats.current_file = os.path.basename(file_path)
        
        if success:
            self.stats.successful_files += 1
        else:
            self.stats.failed_files += 1
            self.alerts.append({
                'time': datetime.now(),
                'type': 'error',
                'file': file_path,
                'message': 'Processing failed'
            })
            
        # Update processing speed
        elapsed = (datetime.now() - self.stats.start_time).total_seconds()
        if elapsed > 0:
            self.stats.processing_speed = (self.stats.processed_files / elapsed) * 60
            self.stats.avg_processing_time = elapsed / self.stats.processed_files
            
            # Calculate ETA
            remaining_files = self.stats.total_files - self.stats.processed_files
            if self.stats.processing_speed > 0:
                eta_seconds = (remaining_files / self.stats.processing_speed) * 60
                self.stats.estimated_time_remaining = timedelta(seconds=int(eta_seconds))
                
        self.stats.last_update = datetime.now()
        
        # Update metrics window
        if self.stats.processed_files % self.update_interval == 0:
            self._update_metrics_window()
            
    def _update_metrics_window(self):
        """Update the rolling window of metrics."""
        recent_metrics = self.metrics_collector.get_recent_metrics(self.window_size)
        if recent_metrics:
            self.last_metrics_window = recent_metrics
            
            # Check for quality degradation
            avg_snr_improvement = np.mean([m.snr_improvement for m in recent_metrics])
            if avg_snr_improvement < 3.0:  # Alert if SNR improvement is low
                self.alerts.append({
                    'time': datetime.now(),
                    'type': 'warning',
                    'message': f'Low average SNR improvement: {avg_snr_improvement:.2f}dB'
                })
                
            # Update quality trends
            self.quality_trends.append({
                'timestamp': datetime.now(),
                'avg_snr': avg_snr_improvement,
                'avg_pesq': np.mean([m.pesq_score for m in recent_metrics if m.pesq_score]),
                'avg_stoi': np.mean([m.stoi_score for m in recent_metrics if m.stoi_score])
            })
            
    def _update_system_stats(self):
        """Update system resource statistics."""
        while self.is_running:
            try:
                # CPU and memory
                self.stats.cpu_usage = psutil.cpu_percent(interval=1)
                self.stats.memory_usage = psutil.virtual_memory().percent
                
                # GPU stats
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self.stats.gpu_usage = gpu.load * 100
                    self.stats.gpu_memory = gpu.memoryUtil * 100
                    self.stats.gpu_temperature = gpu.temperature
                    
            except Exception as e:
                logger.error(f"Error updating system stats: {e}")
                
            time.sleep(5)  # Update every 5 seconds
            
    def run_terminal_ui(self):
        """Run the interactive terminal UI."""
        try:
            curses.wrapper(self._terminal_ui)
        except KeyboardInterrupt:
            pass
            
    def _terminal_ui(self, stdscr):
        """Terminal UI implementation using curses."""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        
        # Define color pairs
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
        while self.is_running:
            try:
                stdscr.clear()
                height, width = stdscr.getmaxyx()
                
                # Header
                header = "Audio Enhancement Dashboard"
                stdscr.addstr(0, (width - len(header)) // 2, header, 
                            curses.A_BOLD | curses.color_pair(4))
                
                # Progress section
                y = 2
                self._draw_progress_section(stdscr, y, width)
                
                # System stats section
                y = 10
                self._draw_system_stats(stdscr, y, width)
                
                # Quality metrics section
                y = 17
                self._draw_quality_metrics(stdscr, y, width)
                
                # Alerts section
                y = 25
                self._draw_alerts(stdscr, y, width, height)
                
                # Footer
                footer = "Press 'q' to quit | 'r' to refresh | 's' to save report"
                stdscr.addstr(height - 1, 0, footer, curses.color_pair(4))
                
                stdscr.refresh()
                
                # Handle input
                key = stdscr.getch()
                if key == ord('q'):
                    self.is_running = False
                elif key == ord('s'):
                    self._save_report()
                    
                time.sleep(0.5)  # Refresh rate
                
            except curses.error:
                pass  # Ignore resize errors
                
    def _draw_progress_section(self, stdscr, y, width):
        """Draw the progress section."""
        # Title
        stdscr.addstr(y, 0, "Processing Progress", curses.A_BOLD)
        y += 2
        
        # Progress bar
        progress = self.stats.processed_files / max(self.stats.total_files, 1)
        bar_width = width - 20
        filled = int(bar_width * progress)
        bar = "[" + "=" * filled + " " * (bar_width - filled) + "]"
        
        stdscr.addstr(y, 0, "Progress: ")
        stdscr.addstr(y, 10, bar)
        stdscr.addstr(y, 10 + bar_width + 2, f"{progress*100:.1f}%")
        y += 1
        
        # Stats
        stdscr.addstr(y, 0, f"Files: {self.stats.processed_files}/{self.stats.total_files}")
        stdscr.addstr(y, 25, f"Success: {self.stats.successful_files}", curses.color_pair(1))
        stdscr.addstr(y, 45, f"Failed: {self.stats.failed_files}", curses.color_pair(3))
        y += 1
        
        # Speed and ETA
        stdscr.addstr(y, 0, f"Speed: {self.stats.processing_speed:.1f} files/min")
        stdscr.addstr(y, 25, f"Avg Time: {self.stats.avg_processing_time:.2f}s/file")
        stdscr.addstr(y, 50, f"ETA: {self.stats.estimated_time_remaining}")
        y += 1
        
        # Current file
        stdscr.addstr(y, 0, f"Current: {self.stats.current_file[:width-10]}")
        
    def _draw_system_stats(self, stdscr, y, width):
        """Draw system statistics."""
        stdscr.addstr(y, 0, "System Resources", curses.A_BOLD)
        y += 2
        
        # CPU and Memory
        cpu_color = self._get_usage_color(self.stats.cpu_usage)
        mem_color = self._get_usage_color(self.stats.memory_usage)
        
        stdscr.addstr(y, 0, f"CPU Usage: {self.stats.cpu_usage:5.1f}%", cpu_color)
        stdscr.addstr(y, 25, f"Memory: {self.stats.memory_usage:5.1f}%", mem_color)
        y += 1
        
        # GPU stats
        if self.stats.gpu_usage > 0:
            gpu_color = self._get_usage_color(self.stats.gpu_usage)
            gmem_color = self._get_usage_color(self.stats.gpu_memory)
            temp_color = self._get_temp_color(self.stats.gpu_temperature)
            
            stdscr.addstr(y, 0, f"GPU Usage: {self.stats.gpu_usage:5.1f}%", gpu_color)
            stdscr.addstr(y, 25, f"GPU Memory: {self.stats.gpu_memory:5.1f}%", gmem_color)
            stdscr.addstr(y, 50, f"Temp: {self.stats.gpu_temperature:3.0f}°C", temp_color)
            
    def _draw_quality_metrics(self, stdscr, y, width):
        """Draw quality metrics section."""
        stdscr.addstr(y, 0, "Quality Metrics (Last 1000 files)", curses.A_BOLD)
        y += 2
        
        if self.last_metrics_window:
            # Calculate averages
            avg_snr = np.mean([m.snr_improvement for m in self.last_metrics_window])
            avg_pesq = np.mean([m.pesq_score for m in self.last_metrics_window if m.pesq_score])
            avg_stoi = np.mean([m.stoi_score for m in self.last_metrics_window if m.stoi_score])
            
            # SNR improvement
            snr_color = curses.color_pair(1) if avg_snr > 5 else curses.color_pair(2)
            stdscr.addstr(y, 0, f"Avg SNR Improvement: {avg_snr:6.2f} dB", snr_color)
            y += 1
            
            # PESQ score
            pesq_color = curses.color_pair(1) if avg_pesq > 3.0 else curses.color_pair(2)
            stdscr.addstr(y, 0, f"Avg PESQ Score:      {avg_pesq:6.2f}", pesq_color)
            y += 1
            
            # STOI score
            stoi_color = curses.color_pair(1) if avg_stoi > 0.8 else curses.color_pair(2)
            stdscr.addstr(y, 0, f"Avg STOI Score:      {avg_stoi:6.2f}", stoi_color)
            
            # Trend indicator
            if len(self.quality_trends) > 1:
                trend = "↑" if self.quality_trends[-1]['avg_snr'] > self.quality_trends[-2]['avg_snr'] else "↓"
                trend_color = curses.color_pair(1) if trend == "↑" else curses.color_pair(3)
                stdscr.addstr(y - 2, 35, trend, trend_color)
                
    def _draw_alerts(self, stdscr, y, width, height):
        """Draw alerts section."""
        stdscr.addstr(y, 0, "Recent Alerts", curses.A_BOLD)
        y += 2
        
        # Show last 5 alerts
        max_alerts = min(5, height - y - 2)
        recent_alerts = self.alerts[-max_alerts:] if self.alerts else []
        
        for alert in recent_alerts:
            alert_color = curses.color_pair(3) if alert['type'] == 'error' else curses.color_pair(2)
            time_str = alert['time'].strftime("%H:%M:%S")
            msg = f"[{time_str}] {alert['message']}"
            stdscr.addstr(y, 0, msg[:width-1], alert_color)
            y += 1
            
    def _get_usage_color(self, usage):
        """Get color based on usage percentage."""
        if usage > 90:
            return curses.color_pair(3)  # Red
        elif usage > 70:
            return curses.color_pair(2)  # Yellow
        else:
            return curses.color_pair(1)  # Green
            
    def _get_temp_color(self, temp):
        """Get color based on temperature."""
        if temp > 80:
            return curses.color_pair(3)  # Red
        elif temp > 70:
            return curses.color_pair(2)  # Yellow
        else:
            return curses.color_pair(1)  # Green
            
    def _save_report(self):
        """Save current dashboard state to report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'stats': asdict(self.stats),
            'quality_metrics': {
                'window_size': self.window_size,
                'metrics': [asdict(m) for m in self.last_metrics_window[-100:]]  # Last 100
            },
            'alerts': self.alerts[-50:],  # Last 50 alerts
            'quality_trends': self.quality_trends[-100:]  # Last 100 trend points
        }
        
        report_path = Path("enhancement_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.alerts.append({
            'time': datetime.now(),
            'type': 'info',
            'message': f'Report saved to {report_path}'
        })
        

class BatchProcessingMonitor:
    """Monitor batch processing across multiple datasets."""
    
    def __init__(self):
        self.dataset_stats = {}
        self.global_stats = ProcessingStats()
        self.batch_start_time = None
        self.quality_thresholds = {
            'min_snr_improvement': 3.0,
            'min_pesq_score': 2.5,
            'min_stoi_score': 0.7
        }
        
    def start_batch(self, datasets: List[str]):
        """Start monitoring a batch of datasets."""
        self.batch_start_time = datetime.now()
        for dataset in datasets:
            self.dataset_stats[dataset] = {
                'stats': ProcessingStats(),
                'quality_issues': [],
                'start_time': None,
                'end_time': None
            }
            
    def start_dataset(self, dataset: str, total_files: int):
        """Start monitoring a specific dataset."""
        if dataset in self.dataset_stats:
            self.dataset_stats[dataset]['stats'].total_files = total_files
            self.dataset_stats[dataset]['start_time'] = datetime.now()
            
    def update_dataset_progress(self, dataset: str, file_path: str, 
                              metrics: EnhancementMetrics, success: bool = True):
        """Update progress for a specific dataset."""
        if dataset not in self.dataset_stats:
            return
            
        stats = self.dataset_stats[dataset]['stats']
        stats.processed_files += 1
        stats.current_file = os.path.basename(file_path)
        
        if success:
            stats.successful_files += 1
        else:
            stats.failed_files += 1
            
        # Check quality thresholds
        if metrics:
            if metrics.snr_improvement < self.quality_thresholds['min_snr_improvement']:
                self.dataset_stats[dataset]['quality_issues'].append({
                    'file': file_path,
                    'issue': 'low_snr',
                    'value': metrics.snr_improvement
                })
                
            if metrics.pesq_score and metrics.pesq_score < self.quality_thresholds['min_pesq_score']:
                self.dataset_stats[dataset]['quality_issues'].append({
                    'file': file_path,
                    'issue': 'low_pesq',
                    'value': metrics.pesq_score
                })
                
        # Update global stats
        self.global_stats.processed_files = sum(
            d['stats'].processed_files for d in self.dataset_stats.values()
        )
        self.global_stats.successful_files = sum(
            d['stats'].successful_files for d in self.dataset_stats.values()
        )
        self.global_stats.failed_files = sum(
            d['stats'].failed_files for d in self.dataset_stats.values()
        )
        
    def end_dataset(self, dataset: str):
        """Mark dataset processing as complete."""
        if dataset in self.dataset_stats:
            self.dataset_stats[dataset]['end_time'] = datetime.now()
            
    def generate_batch_report(self) -> Dict[str, Any]:
        """Generate comprehensive batch processing report."""
        total_time = (datetime.now() - self.batch_start_time).total_seconds()
        
        report = {
            'batch_summary': {
                'start_time': self.batch_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(timedelta(seconds=int(total_time))),
                'total_files': self.global_stats.processed_files,
                'successful_files': self.global_stats.successful_files,
                'failed_files': self.global_stats.failed_files,
                'success_rate': (self.global_stats.successful_files / 
                               max(self.global_stats.processed_files, 1)) * 100
            },
            'dataset_reports': {}
        }
        
        for dataset, data in self.dataset_stats.items():
            stats = data['stats']
            duration = (data['end_time'] - data['start_time']).total_seconds() if data['end_time'] else 0
            
            report['dataset_reports'][dataset] = {
                'stats': {
                    'total_files': stats.total_files,
                    'processed_files': stats.processed_files,
                    'successful_files': stats.successful_files,
                    'failed_files': stats.failed_files,
                    'success_rate': (stats.successful_files / max(stats.processed_files, 1)) * 100,
                    'duration': str(timedelta(seconds=int(duration))),
                    'processing_speed': (stats.processed_files / duration * 60) if duration > 0 else 0
                },
                'quality_issues': {
                    'total': len(data['quality_issues']),
                    'by_type': self._group_quality_issues(data['quality_issues'])
                }
            }
            
        return report
        
    def _group_quality_issues(self, issues: List[Dict]) -> Dict[str, int]:
        """Group quality issues by type."""
        grouped = {}
        for issue in issues:
            issue_type = issue['issue']
            grouped[issue_type] = grouped.get(issue_type, 0) + 1
        return grouped
        
    def export_to_csv(self, output_path: str):
        """Export batch statistics to CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', 'Total Files', 'Processed', 'Successful', 
                           'Failed', 'Success Rate', 'Quality Issues'])
            
            for dataset, data in self.dataset_stats.items():
                stats = data['stats']
                writer.writerow([
                    dataset,
                    stats.total_files,
                    stats.processed_files,
                    stats.successful_files,
                    stats.failed_files,
                    f"{(stats.successful_files / max(stats.processed_files, 1)) * 100:.1f}%",
                    len(data['quality_issues'])
                ])
                

class ConfigurationManager:
    """Manage enhancement configurations with UI."""
    
    def __init__(self, config_path: str = "enhancement_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.presets = {
            'conservative': {
                'enhancement_level': 0.3,
                'noise_reduction': 0.5,
                'batch_size': 16,
                'gpu_memory_fraction': 0.7
            },
            'balanced': {
                'enhancement_level': 0.5,
                'noise_reduction': 0.7,
                'batch_size': 32,
                'gpu_memory_fraction': 0.8
            },
            'aggressive': {
                'enhancement_level': 0.8,
                'noise_reduction': 0.9,
                'batch_size': 64,
                'gpu_memory_fraction': 0.9
            }
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return self.get_default_config()
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enhancement': {
                'level': 0.5,
                'noise_reduction': 0.7,
                'preserve_original': True,
                'adaptive_processing': True
            },
            'processing': {
                'batch_size': 32,
                'num_workers': 4,
                'gpu_memory_fraction': 0.8,
                'cache_size': 1000
            },
            'quality': {
                'min_snr_improvement': 3.0,
                'target_pesq': 3.0,
                'target_stoi': 0.8
            },
            'monitoring': {
                'update_interval': 100,
                'metrics_window': 1000,
                'enable_web_dashboard': True,
                'web_port': 8080
            }
        }
        
    def apply_preset(self, preset_name: str):
        """Apply a configuration preset."""
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            self.config['enhancement']['level'] = preset['enhancement_level']
            self.config['enhancement']['noise_reduction'] = preset['noise_reduction']
            self.config['processing']['batch_size'] = preset['batch_size']
            self.config['processing']['gpu_memory_fraction'] = preset['gpu_memory_fraction']
            self.save_config()
            
    def optimize_for_gpu(self, gpu_memory_gb: float):
        """Optimize configuration based on available GPU memory."""
        if gpu_memory_gb < 4:
            self.config['processing']['batch_size'] = 8
            self.config['processing']['gpu_memory_fraction'] = 0.7
        elif gpu_memory_gb < 8:
            self.config['processing']['batch_size'] = 16
            self.config['processing']['gpu_memory_fraction'] = 0.8
        elif gpu_memory_gb < 16:
            self.config['processing']['batch_size'] = 32
            self.config['processing']['gpu_memory_fraction'] = 0.85
        else:
            self.config['processing']['batch_size'] = 64
            self.config['processing']['gpu_memory_fraction'] = 0.9
            
        self.save_config()
        
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate configuration settings."""
        errors = []
        
        # Validate enhancement settings
        if not 0 <= self.config['enhancement']['level'] <= 1:
            errors.append("Enhancement level must be between 0 and 1")
            
        if not 0 <= self.config['enhancement']['noise_reduction'] <= 1:
            errors.append("Noise reduction must be between 0 and 1")
            
        # Validate processing settings
        if self.config['processing']['batch_size'] < 1:
            errors.append("Batch size must be at least 1")
            
        if not 0 < self.config['processing']['gpu_memory_fraction'] <= 1:
            errors.append("GPU memory fraction must be between 0 and 1")
            
        return len(errors) == 0, errors
        

class ReportGenerator:
    """Generate comprehensive enhancement reports."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
    def generate_html_report(self, output_path: str, 
                           batch_monitor: Optional[BatchProcessingMonitor] = None):
        """Generate HTML report with visualizations."""
        from datetime import datetime
        import matplotlib.pyplot as plt
        import seaborn as sns
        from io import BytesIO
        import base64
        
        # Get all metrics
        all_metrics = self.metrics_collector.get_all_metrics()
        summary = self.metrics_collector.get_summary_statistics()
        
        # Generate plots
        plots = {}
        
        # SNR improvement distribution
        plt.figure(figsize=(10, 6))
        sns.histplot([m.snr_improvement for m in all_metrics], bins=50, kde=True)
        plt.xlabel('SNR Improvement (dB)')
        plt.ylabel('Count')
        plt.title('SNR Improvement Distribution')
        plots['snr_dist'] = self._fig_to_base64()
        plt.close()
        
        # Quality scores over time
        if len(all_metrics) > 100:
            plt.figure(figsize=(12, 6))
            window = 100
            snr_trend = pd.Series([m.snr_improvement for m in all_metrics]).rolling(window).mean()
            plt.plot(snr_trend, label='SNR Improvement')
            plt.xlabel('Sample Number')
            plt.ylabel('SNR Improvement (dB)')
            plt.title(f'Quality Trend (Rolling Average, Window={window})')
            plt.legend()
            plots['quality_trend'] = self._fig_to_base64()
            plt.close()
        
        # Generate HTML
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Audio Enhancement Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: #4CAF50; }}
                .warning {{ color: #FF9800; }}
                .error {{ color: #F44336; }}
            </style>
        </head>
        <body>
            <h1>Audio Enhancement Report</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <div class="metric">
                    <div>Total Files</div>
                    <div class="metric-value">{total_files}</div>
                </div>
                <div class="metric">
                    <div>Average SNR Improvement</div>
                    <div class="metric-value">{avg_snr:.2f} dB</div>
                </div>
                <div class="metric">
                    <div>Average PESQ Score</div>
                    <div class="metric-value">{avg_pesq:.2f}</div>
                </div>
                <div class="metric">
                    <div>Average STOI Score</div>
                    <div class="metric-value">{avg_stoi:.2f}</div>
                </div>
            </div>
            
            <div class="plot">
                <h2>SNR Improvement Distribution</h2>
                <img src="data:image/png;base64,{snr_dist_plot}" />
            </div>
            
            {quality_trend_section}
            
            {batch_report_section}
            
        </body>
        </html>
        """
        
        # Prepare quality trend section
        quality_trend_section = ""
        if 'quality_trend' in plots:
            quality_trend_section = f"""
            <div class="plot">
                <h2>Quality Trend Over Time</h2>
                <img src="data:image/png;base64,{plots['quality_trend']}" />
            </div>
            """
            
        # Prepare batch report section
        batch_report_section = ""
        if batch_monitor:
            batch_report = batch_monitor.generate_batch_report()
            batch_report_section = self._generate_batch_report_html(batch_report)
            
        # Fill template
        html = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_files=summary['total_files'],
            avg_snr=summary['avg_snr_improvement'],
            avg_pesq=summary['avg_pesq_score'],
            avg_stoi=summary['avg_stoi_score'],
            snr_dist_plot=plots['snr_dist'],
            quality_trend_section=quality_trend_section,
            batch_report_section=batch_report_section
        )
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html)
            
    def _fig_to_base64(self) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
        
    def _generate_batch_report_html(self, batch_report: Dict) -> str:
        """Generate HTML section for batch report."""
        html = """
        <h2>Batch Processing Report</h2>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Total Files</th>
                <th>Processed</th>
                <th>Success Rate</th>
                <th>Processing Speed</th>
                <th>Quality Issues</th>
            </tr>
        """
        
        for dataset, report in batch_report['dataset_reports'].items():
            stats = report['stats']
            success_class = 'success' if stats['success_rate'] > 95 else 'warning'
            quality_class = 'error' if report['quality_issues']['total'] > 10 else 'success'
            
            html += f"""
            <tr>
                <td>{dataset}</td>
                <td>{stats['total_files']}</td>
                <td>{stats['processed_files']}</td>
                <td class="{success_class}">{stats['success_rate']:.1f}%</td>
                <td>{stats['processing_speed']:.1f} files/min</td>
                <td class="{quality_class}">{report['quality_issues']['total']}</td>
            </tr>
            """
            
        html += "</table>"
        return html
        

class DashboardCheckpointHandler:
    """Handle dashboard state persistence."""
    
    def __init__(self, checkpoint_dir: str = "dashboard_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_dashboard_state(self, dashboard: EnhancementDashboard, 
                           checkpoint_name: str = "dashboard_state.json"):
        """Save dashboard state to checkpoint."""
        state = {
            'stats': asdict(dashboard.stats),
            'alerts': dashboard.alerts[-100:],  # Last 100 alerts
            'quality_trends': dashboard.quality_trends[-1000:],  # Last 1000 trend points
            'processing_history': dashboard.processing_history[-1000:],  # Last 1000 files
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
    def load_dashboard_state(self, dashboard: EnhancementDashboard,
                           checkpoint_name: str = "dashboard_state.json") -> bool:
        """Load dashboard state from checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            return False
            
        try:
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
                
            # Restore stats
            for key, value in state['stats'].items():
                if hasattr(dashboard.stats, key):
                    setattr(dashboard.stats, key, value)
                    
            # Restore other state
            dashboard.alerts = state.get('alerts', [])
            dashboard.quality_trends = state.get('quality_trends', [])
            dashboard.processing_history = state.get('processing_history', [])
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading dashboard state: {e}")
            return False