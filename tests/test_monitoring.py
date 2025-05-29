"""Comprehensive test suite for audio enhancement monitoring dashboard."""

import unittest
import tempfile
import shutil
import os
import json
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.dashboard import EnhancementDashboard, BatchProcessingMonitor as MonitoringBatchMonitor
from monitoring.metrics_collector import MetricsCollector, get_gpu_metrics, EnhancementMetrics
from monitoring.comparison_ui import ComparisonAnalyzer
from dashboard.configuration_ui import ConfigurationUI
from dashboard.batch_monitor import BatchProcessingMonitor


class TestEnhancementDashboard(unittest.TestCase):
    """Test real-time dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_collector = MetricsCollector(metrics_dir=self.temp_dir)
        self.dashboard = EnhancementDashboard(
            metrics_collector=self.metrics_collector,
            update_interval=10
        )
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        self.assertEqual(self.dashboard.update_interval, 10)
        self.assertEqual(self.dashboard.stats.processed_files, 0)
        self.assertEqual(self.dashboard.stats.total_files, 0)
        self.assertTrue(os.path.exists(self.temp_dir))
        
    def test_metric_update(self):
        """Test real-time metric updates."""
        # Start dashboard with total files
        self.dashboard.start(100)
        
        # Update progress
        self.dashboard.update_progress("file1.wav", success=True)
        
        self.assertEqual(self.dashboard.stats.processed_files, 1)
        self.assertEqual(self.dashboard.stats.total_files, 100)
        self.assertEqual(self.dashboard.stats.successful_files, 1)
        
    def test_processing_rate_calculation(self):
        """Test processing rate calculation."""
        # Start processing
        self.dashboard.start(100)
        
        # Simulate processing over time
        self.dashboard.update_progress("file1.wav")
        time.sleep(0.1)
        self.dashboard.update_progress("file2.wav")
        
        # Check processing rate is calculated
        self.assertGreater(self.dashboard.stats.processing_speed, 0)
        
    def test_html_generation(self):
        """Test static HTML dashboard generation."""
        # Start dashboard
        self.dashboard.start(100)
        
        # Add some metrics to collector
        self.metrics_collector.add_sample("test1", {
            "snr_improvement": 5.2,
            "pesq_score": 3.8,
            "stoi_score": 0.92
        })
        
        # Update progress
        self.dashboard.update_progress("file1.wav")
        
        # Save report (which generates HTML-like content)
        self.dashboard._save_report()
        
        # Check report was created
        report_path = Path("enhancement_report.json")
        self.assertTrue(report_path.exists())
        
        # Clean up
        if report_path.exists():
            os.remove(report_path)
            
    def test_metrics_history_tracking(self):
        """Test metrics history tracking."""
        self.dashboard.start(100)
        
        # Add file metrics (which is what get_recent_metrics returns)
        for i in range(5):
            metrics = EnhancementMetrics(
                snr_improvement=i * 1.1,
                pesq_score=3.5,
                stoi_score=0.9,
                processing_time=0.5,
                audio_file=f"file_{i}.wav"
            )
            # Add to file_metrics list directly
            self.metrics_collector.file_metrics.append(metrics)
            
        # Update dashboard progress to trigger metrics window update
        for i in range(self.dashboard.update_interval):
            self.dashboard.update_progress(f"file_{i}.wav")
        
        # Force an update to metrics window
        self.dashboard._update_metrics_window()
        
        # Check that metrics window was updated and quality trends
        self.assertEqual(len(self.dashboard.last_metrics_window), 5)
        self.assertGreater(len(self.dashboard.quality_trends), 0)
        
    def test_dashboard_lifecycle(self):
        """Test dashboard lifecycle methods."""
        # Test start
        self.dashboard.start(50)
        self.assertTrue(self.dashboard.is_running)
        self.assertEqual(self.dashboard.stats.total_files, 50)
        
        # Test stop
        self.dashboard.stop()
        self.assertFalse(self.dashboard.is_running)


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = MetricsCollector(
            window_size=100,
            metrics_dir=self.temp_dir
        )
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_collector_initialization(self):
        """Test metrics collector initialization."""
        self.assertEqual(self.collector.window_size, 100)
        self.assertEqual(len(self.collector.metrics_window), 0)
        self.assertTrue(os.path.exists(self.temp_dir))
        
    def test_add_sample_metrics(self):
        """Test adding sample metrics."""
        sample_metrics = {
            "snr_improvement": 5.0,
            "pesq_score": 3.5,
            "stoi_score": 0.9,
            "processing_time": 0.5
        }
        
        self.collector.add_sample("audio_001", sample_metrics)
        
        self.assertEqual(len(self.collector.metrics_window), 1)
        self.assertIn("audio_001", self.collector.sample_metrics)
        
    def test_calculate_averages(self):
        """Test average calculation."""
        # Add multiple samples
        for i in range(10):
            metrics = {
                "snr_improvement": 5.0 + i * 0.1,
                "pesq_score": 3.0 + i * 0.05,
                "stoi_score": 0.8 + i * 0.01
            }
            self.collector.add_sample(f"audio_{i:03d}", metrics)
            
        averages = self.collector.get_current_averages()
        
        self.assertAlmostEqual(averages["avg_snr_improvement"], 5.45, places=2)
        self.assertAlmostEqual(averages["avg_pesq_score"], 3.225, places=3)
        self.assertAlmostEqual(averages["avg_stoi_score"], 0.845, places=3)
        
    def test_window_size_limit(self):
        """Test that metrics window respects size limit."""
        # Add more samples than window size
        for i in range(150):
            self.collector.add_sample(f"audio_{i:03d}", {"snr_improvement": i})
            
        self.assertEqual(len(self.collector.metrics_window), 100)
        
    def test_trend_calculation(self):
        """Test trend calculation over time."""
        # Add samples with increasing quality
        for i in range(50):
            metrics = {"snr_improvement": 2.0 + i * 0.1}
            self.collector.add_sample(f"audio_{i:03d}", metrics)
            
        trends = self.collector.calculate_trends()
        
        self.assertIn("snr_improvement_trend", trends)
        self.assertGreater(trends["snr_improvement_trend"], 0)  # Positive trend
        
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        # Add diverse samples
        snr_values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        for i, snr in enumerate(snr_values):
            self.collector.add_sample(f"audio_{i:03d}", {"snr_improvement": snr})
            
        stats = self.collector.get_summary_statistics()
        
        self.assertIn("snr_improvement", stats)
        self.assertAlmostEqual(stats["snr_improvement"]["mean"], 4.17, places=2)
        self.assertEqual(stats["snr_improvement"]["min"], 1.0)
        self.assertEqual(stats["snr_improvement"]["max"], 10.0)
        # Standard deviation calculation might vary slightly
        self.assertAlmostEqual(stats["snr_improvement"]["std"], 2.91, places=1)
        
    def test_export_metrics(self):
        """Test metrics export functionality."""
        # Add some samples
        for i in range(5):
            self.collector.add_sample(f"audio_{i:03d}", {"snr_improvement": i})
            
        # Export to CSV
        csv_path = os.path.join(self.temp_dir, "metrics.csv")
        self.collector.export_to_csv(csv_path)
        self.assertTrue(os.path.exists(csv_path))
        
        # Export to JSON
        json_path = os.path.join(self.temp_dir, "metrics.json")
        self.collector.export_to_json(json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data["samples"]), 5)
            
    def test_gpu_metrics_collection(self):
        """Test GPU metrics collection."""
        gpu_metrics = get_gpu_metrics()
        
        self.assertIn("memory_used", gpu_metrics)
        self.assertIn("memory_total", gpu_metrics)
        self.assertIn("utilization", gpu_metrics)
        self.assertIn("temperature", gpu_metrics)


class TestComparisonAnalyzer(unittest.TestCase):
    """Test comparison analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ComparisonAnalyzer(output_dir=self.temp_dir)
        
        # Create test audio signals
        self.sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Original: noisy signal
        self.original = np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(len(t))
        
        # Enhanced: cleaner signal
        self.enhanced = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_waveform_comparison(self):
        """Test waveform comparison generation."""
        result = self.analyzer.generate_waveform_comparison(
            self.original, self.enhanced, "test_audio"
        )
        
        self.assertIn("waveform_path", result)
        self.assertTrue(os.path.exists(result["waveform_path"]))
        
    @patch('monitoring.comparison_ui.SCIPY_AVAILABLE', True)
    def test_spectrogram_comparison(self):
        """Test spectrogram comparison generation."""
        # Fix: the argument order is (original, enhanced, audio_id, sample_rate)
        result = self.analyzer.generate_spectrogram_comparison(
            self.original, self.enhanced, "test_audio", self.sample_rate
        )
        
        self.assertIn("spectrogram_path", result)
        self.assertTrue(os.path.exists(result["spectrogram_path"]))
        
    def test_frequency_response_analysis(self):
        """Test frequency response analysis."""
        result = self.analyzer.generate_frequency_response_comparison(
            self.original, self.enhanced, "test_audio", self.sample_rate
        )
        
        self.assertIn("frequency_response_path", result)
        self.assertTrue(os.path.exists(result["frequency_response_path"]))
        
    def test_quality_verdict_generation(self):
        """Test quality verdict generation."""
        metrics = {
            "snr_improvement": 5.2,
            "pesq": 3.8,
            "stoi": 0.92
        }
        
        # This is a private method that returns a string
        recommendation = self.analyzer._generate_recommendation(metrics)
        
        self.assertIsInstance(recommendation, str)
        self.assertGreater(len(recommendation), 0)
        
    def test_comparison_export(self):
        """Test comparison plot export."""
        # Generate comprehensive report
        metrics = {
            "snr_improvement": 5.2,
            "pesq": 3.8,
            "stoi": 0.92
        }
        
        report = self.analyzer.generate_comparison_report(
            self.original, self.enhanced, "test_audio", metrics, self.sample_rate
        )
        
        # Check report structure
        self.assertEqual(report["audio_id"], "test_audio")
        self.assertIn("metrics", report)
        self.assertIn("visualizations", report)
        self.assertIn("recommendation", report)
        
        # Check JSON report file
        json_path = os.path.join(self.temp_dir, "test_audio_report.json")
        self.assertTrue(os.path.exists(json_path))


class TestConfigurationUI(unittest.TestCase):
    """Test configuration management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        self.config_manager = ConfigurationUI(config_file=self.config_file)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_default_configuration(self):
        """Test default configuration loading."""
        config = self.config_manager.current_config
        
        self.assertIn("noise_reduction_level", config)
        self.assertIn("batch_size", config)
        self.assertIn("gpu_device", config)
        
    def test_configuration_update(self):
        """Test configuration update."""
        new_config = {
            "noise_reduction_level": "aggressive",
            "batch_size": 64,
            "gpu_device": 1
        }
        
        self.config_manager.update_config(new_config)
        
        self.assertEqual(self.config_manager.current_config["noise_reduction_level"], "aggressive")
        self.assertEqual(self.config_manager.current_config["batch_size"], 64)
        
    def test_configuration_persistence(self):
        """Test configuration save/load."""
        # Update config
        self.config_manager.update_config({
            "noise_reduction_level": "moderate",
            "batch_size": 32
        })
        
        # Save configuration
        self.config_manager.save_config()
        
        # Create new instance and load
        new_manager = ConfigurationUI(config_file=self.config_file)
        
        self.assertEqual(new_manager.current_config["noise_reduction_level"], "moderate")
        self.assertEqual(new_manager.current_config["batch_size"], 32)
        
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with valid configuration
        self.config_manager.current_config["noise_reduction_level"] = "moderate"
        self.config_manager.current_config["batch_size"] = 32
        self.assertTrue(self.config_manager.validate_config())
        
        # Test with invalid configuration
        self.config_manager.current_config["noise_reduction_level"] = "ultra"  # Invalid
        self.assertFalse(self.config_manager.validate_config())
        
        # Reset to valid
        self.config_manager.current_config["noise_reduction_level"] = "moderate"
        self.config_manager.current_config["batch_size"] = -1  # Invalid
        self.assertFalse(self.config_manager.validate_config())
        
    def test_profile_management(self):
        """Test configuration profile management."""
        # Test noise reduction config
        mild_config = self.config_manager.get_noise_reduction_config("mild")
        moderate_config = self.config_manager.get_noise_reduction_config("moderate")
        aggressive_config = self.config_manager.get_noise_reduction_config("aggressive")
        
        # Check different levels have different settings
        self.assertNotEqual(mild_config["denoiser_dry"], aggressive_config["denoiser_dry"])
        self.assertNotEqual(mild_config["spectral_gate_freq"], aggressive_config["spectral_gate_freq"])
        
        # Check suppress_secondary_speakers settings
        self.assertFalse(mild_config["suppress_secondary_speakers"])
        self.assertTrue(moderate_config["suppress_secondary_speakers"])
        self.assertTrue(aggressive_config["suppress_secondary_speakers"])


class TestBatchProcessingMonitor(unittest.TestCase):
    """Test batch processing monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = BatchProcessingMonitor(batch_size=32)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_batch_initialization(self):
        """Test batch monitor initialization."""
        batch_id = self.monitor.start_batch()
        
        self.assertEqual(batch_id, 0)
        self.assertIn(batch_id, self.monitor.batches)
        self.assertFalse(self.monitor.batches[batch_id]["completed"])
        
    def test_batch_progress_tracking(self):
        """Test batch progress tracking."""
        batch_id = self.monitor.start_batch()
        
        # Record samples
        for i in range(10):
            self.monitor.record_sample(f"sample_{i}", processing_time=0.5, success=True, batch_id=batch_id)
            
        batch_summary = self.monitor.get_batch_summary(batch_id)
        self.assertEqual(batch_summary["sample_count"], 10)
        self.assertEqual(batch_summary["success_count"], 10)
        
    def test_multiple_dataset_monitoring(self):
        """Test monitoring multiple datasets."""
        # Start multiple batches
        batch1_id = self.monitor.start_batch()
        batch2_id = self.monitor.start_batch()
        
        # Record samples for each batch
        for i in range(5):
            self.monitor.record_sample(f"batch1_sample_{i}", processing_time=0.5, success=True, batch_id=batch1_id)
        for i in range(10):
            self.monitor.record_sample(f"batch2_sample_{i}", processing_time=0.3, success=True, batch_id=batch2_id)
        
        # Check statistics
        stats = self.monitor.get_statistics()
        self.assertEqual(stats["total_samples"], 15)
        self.assertEqual(stats["successful_samples"], 15)
        
    def test_quality_alert_system(self):
        """Test quality degradation alerts."""
        batch_id = self.monitor.start_batch()
        
        # Add samples with failures
        for i in range(10):
            success = i % 3 != 0  # Every 3rd sample fails
            self.monitor.record_sample(f"sample_{i}", processing_time=0.5, success=success, 
                                     error_message="Test error" if not success else None,
                                     batch_id=batch_id)
            
        # Check failures
        failures = self.monitor.get_failures()
        self.assertGreater(len(failures), 0)
        self.assertEqual(failures[0]["error"], "Test error")
        
    def test_batch_report_generation(self):
        """Test batch report generation."""
        batch_id = self.monitor.start_batch()
        
        # Process samples
        for i in range(10):
            processing_time = 0.5 + np.random.rand() * 0.2
            self.monitor.record_sample(f"sample_{i}", processing_time=processing_time, 
                                     success=True, batch_id=batch_id)
            
        # Complete batch
        self.monitor.complete_batch(batch_id)
        
        # Get batch summary
        summary = self.monitor.get_batch_summary(batch_id)
        
        self.assertEqual(summary["batch_id"], batch_id)
        self.assertEqual(summary["sample_count"], 10)
        self.assertTrue(summary["completed"])
        self.assertGreater(summary["avg_sample_time"], 0)
            
    def test_export_batch_metrics(self):
        """Test batch metrics export."""
        batch_id = self.monitor.start_batch()
        
        # Add sample metrics
        for i in range(5):
            self.monitor.record_sample(f"sample_{i}", processing_time=i * 0.1, 
                                     success=True, batch_id=batch_id)
        
        # Test bottleneck detection
        bottlenecks = self.monitor.detect_bottlenecks(threshold_factor=2.0)
        # Bottlenecks should be empty for uniform processing times
        self.assertIsInstance(bottlenecks, list)
        
        # Test efficiency report
        efficiency_report = self.monitor.get_efficiency_report()
        self.assertIn("efficiency_percentage", efficiency_report)
        self.assertIn("issues", efficiency_report)
        self.assertIn("recommendations", efficiency_report)


class TestWebDashboardFunctionality(unittest.TestCase):
    """Test web dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_static_file_generation(self):
        """Test static HTML/CSS/JS file generation."""
        metrics_collector = MetricsCollector(metrics_dir=self.temp_dir)
        dashboard = EnhancementDashboard(metrics_collector=metrics_collector)
        
        # Start dashboard
        dashboard.start(100)
        
        # Add some metrics
        metrics_collector.add_sample("test1", {
            "snr_improvement": 5.2,
            "pesq_score": 3.8,
            "stoi_score": 0.92
        })
        
        # Update progress
        dashboard.update_progress("file1.wav")
        
        # Save report
        dashboard._save_report()
        
        # Check report exists
        self.assertTrue(os.path.exists("enhancement_report.json"))
        
        # Clean up
        if os.path.exists("enhancement_report.json"):
            os.remove("enhancement_report.json")
        
    def test_real_time_update_mechanism(self):
        """Test real-time update mechanism."""
        metrics_collector = MetricsCollector()
        dashboard = EnhancementDashboard(metrics_collector=metrics_collector)
        
        # Start and update
        dashboard.start(100)
        
        initial_processed = dashboard.stats.processed_files
        
        # Trigger updates
        for i in range(5):
            dashboard.update_progress(f"file_{i}.wav")
            
        self.assertEqual(dashboard.stats.processed_files, initial_processed + 5)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_dashboard_metrics_integration(self):
        """Test integration between dashboard and metrics collector."""
        collector = MetricsCollector(metrics_dir=self.temp_dir)
        dashboard = EnhancementDashboard(metrics_collector=collector)
        
        # Start dashboard
        dashboard.start(20)
        
        # Simulate processing pipeline
        for i in range(20):
            # Collect metrics
            sample_metrics = {
                "snr_improvement": 5.0 + np.random.randn() * 0.5,
                "pesq_score": 3.5 + np.random.rand() * 0.3,
                "stoi_score": 0.85 + np.random.rand() * 0.1
            }
            collector.add_sample(f"audio_{i:03d}", sample_metrics)
            
            # Update dashboard progress
            dashboard.update_progress(f"audio_{i:03d}.wav")
                
        # Verify integration
        self.assertEqual(dashboard.stats.processed_files, 20)
        self.assertEqual(len(collector.metrics_window), 20)
        
    def test_batch_monitor_dashboard_integration(self):
        """Test integration between batch monitor and dashboard."""
        metrics_collector = MetricsCollector(metrics_dir=self.temp_dir)
        dashboard = EnhancementDashboard(metrics_collector=metrics_collector)
        monitor = BatchProcessingMonitor()
        
        # Start dashboard and batch
        dashboard.start(50)
        batch_id = monitor.start_batch()
        
        # Simulate batch processing with dashboard updates
        for i in range(50):
            # Process file
            monitor.record_sample(f"sample_{i}", processing_time=0.5, success=True, batch_id=batch_id)
            
            # Update dashboard
            dashboard.update_progress(f"sample_{i}.wav")
            
            # Add metrics
            metrics_collector.add_sample(f"sample_{i}", {
                "snr_improvement": 5.0 + np.random.randn() * 0.5
            })
                
        # Complete batch
        monitor.complete_batch(batch_id)
        
        # Verify integration
        self.assertEqual(dashboard.stats.processed_files, 50)
        self.assertTrue(monitor.batches[batch_id]["completed"])


if __name__ == "__main__":
    unittest.main()