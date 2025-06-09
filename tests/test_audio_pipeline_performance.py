"""
Performance benchmark tests for the audio analysis pipeline.

This module tests performance characteristics including:
- Processing speed for various audio lengths
- Memory usage patterns
- CPU/GPU utilization
- Scalability with concurrent processing
- Component-level performance profiling
"""

import unittest
import pytest
import numpy as np
import tempfile
import os
import time
import psutil
import gc
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import soundfile as sf
from memory_profiler import profile
import matplotlib.pyplot as plt
from datetime import datetime

# Import pipeline components
from processors.audio_enhancement.audio_loader import AudioLoader
from utils.enhanced_snr_calculator import EnhancedSNRCalculator
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer
from processors.audio_enhancement.detection.pattern_detector import PatternDetector
from processors.audio_enhancement.issue_categorization import IssueCategorizer
from processors.audio_enhancement.decision_framework import DecisionEngine


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.results = {
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'component_times': {}
        }
        
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure function execution time."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
        
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure peak memory usage during function execution."""
        process = psutil.Process()
        
        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - baseline_memory
        
        return result, memory_used
        
    def profile_cpu_usage(self, func, *args, **kwargs):
        """Profile CPU usage during execution."""
        process = psutil.Process()
        
        # Start CPU monitoring
        cpu_percentages = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_percentages.append(process.cpu_percent(interval=0.1))
                
        import threading
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Stop monitoring
        monitoring = False
        monitor_thread.join()
        
        avg_cpu = np.mean(cpu_percentages) if cpu_percentages else 0
        peak_cpu = max(cpu_percentages) if cpu_percentages else 0
        
        return result, {'average': avg_cpu, 'peak': peak_cpu}
        
    def generate_report(self, output_path=None):
        """Generate performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'avg_processing_time': np.mean(self.results['processing_times']),
                'max_processing_time': np.max(self.results['processing_times']),
                'avg_memory_usage': np.mean(self.results['memory_usage']),
                'peak_memory_usage': np.max(self.results['memory_usage']),
                'avg_cpu_usage': np.mean([c['average'] for c in self.results['cpu_usage']])
            },
            'detailed_results': self.results
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report


class TestAudioPipelinePerformance(unittest.TestCase):
    """Performance tests for audio analysis pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up performance test environment."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.benchmark = PerformanceBenchmark()
        cls.test_files = cls._create_performance_test_files()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up and generate report."""
        import shutil
        shutil.rmtree(cls.temp_dir)
        
        # Generate performance report
        report_path = 'performance_report.json'
        cls.benchmark.generate_report(report_path)
        print(f"Performance report saved to {report_path}")
        
    @classmethod
    def _create_performance_test_files(cls):
        """Create test files of various sizes."""
        files = {}
        sample_rate = 16000
        
        # Different durations for testing
        durations = {
            'tiny': 0.5,      # 0.5 seconds
            'small': 2,       # 2 seconds
            'medium': 10,     # 10 seconds
            'large': 30,      # 30 seconds
            'huge': 60        # 1 minute
        }
        
        for size_name, duration in durations.items():
            file_path = os.path.join(cls.temp_dir, f'{size_name}.wav')
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Create realistic audio with speech-like characteristics
            audio = np.zeros_like(t)
            
            # Add multiple frequency components
            for f in [200, 400, 600, 800, 1000]:
                audio += 0.2 * np.sin(2 * np.pi * f * t)
                
            # Add amplitude modulation
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
            audio *= modulation
            
            # Add some noise
            audio += np.random.normal(0, 0.05, len(audio))
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            sf.write(file_path, audio, sample_rate)
            files[size_name] = {
                'path': file_path,
                'duration': duration,
                'samples': len(audio)
            }
            
        return files
        
    def setUp(self):
        """Set up for each test."""
        self.loader = AudioLoader()
        self.snr_calculator = EnhancedSNRCalculator()
        self.spectral_analyzer = SpectralAnalyzer()
        self.pattern_detector = PatternDetector()
        self.issue_categorizer = IssueCategorizer()
        self.decision_engine = DecisionEngine()
        
    def test_processing_speed_scaling(self):
        """Test how processing speed scales with audio length."""
        results = {}
        
        for size_name, file_info in self.test_files.items():
            # Measure processing time
            start_time = time.perf_counter()
            
            # Load audio
            audio = self.loader.load(file_info['path'])
            
            # Process through pipeline
            snr_metrics = self.snr_calculator.calculate(audio.samples, audio.sample_rate)
            spectral_features = self.spectral_analyzer.analyze(audio.samples, audio.sample_rate)
            patterns = self.pattern_detector.detect(audio.samples, spectral_features)
            issues = self.issue_categorizer.categorize(snr_metrics, spectral_features, patterns)
            decision = self.decision_engine.decide(issues)
            
            processing_time = time.perf_counter() - start_time
            
            results[size_name] = {
                'duration': file_info['duration'],
                'processing_time': processing_time,
                'realtime_factor': file_info['duration'] / processing_time
            }
            
            # Performance requirement: process faster than real-time
            self.assertGreater(
                results[size_name]['realtime_factor'], 
                1.0,
                f"Processing {size_name} audio slower than real-time"
            )
            
        # Check linear scaling (processing time should scale linearly with duration)
        durations = [results[k]['duration'] for k in ['tiny', 'small', 'medium']]
        times = [results[k]['processing_time'] for k in ['tiny', 'small', 'medium']]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(durations, times)[0, 1]
        self.assertGreater(correlation, 0.9, "Processing time doesn't scale linearly")
        
        # Store results
        self.benchmark.results['processing_times'].extend(times)
        
    def test_memory_usage_patterns(self):
        """Test memory usage patterns for different audio sizes."""
        memory_results = {}
        
        for size_name, file_info in self.test_files.items():
            # Force garbage collection
            gc.collect()
            
            # Get baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process audio
            audio = self.loader.load(file_info['path'])
            snr_metrics = self.snr_calculator.calculate(audio.samples, audio.sample_rate)
            spectral_features = self.spectral_analyzer.analyze(audio.samples, audio.sample_rate)
            patterns = self.pattern_detector.detect(audio.samples, spectral_features)
            issues = self.issue_categorizer.categorize(snr_metrics, spectral_features, patterns)
            decision = self.decision_engine.decide(issues)
            
            # Get peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - baseline_memory
            
            memory_results[size_name] = {
                'file_size_mb': file_info['samples'] * 4 / 1024 / 1024,  # float32
                'memory_used_mb': memory_used,
                'memory_ratio': memory_used / (file_info['samples'] * 4 / 1024 / 1024)
            }
            
            # Memory should not exceed 10x the audio file size
            self.assertLess(
                memory_results[size_name]['memory_ratio'],
                10.0,
                f"Excessive memory usage for {size_name} audio"
            )
            
        # Store results
        self.benchmark.results['memory_usage'].extend(
            [m['memory_used_mb'] for m in memory_results.values()]
        )
        
    def test_component_level_performance(self):
        """Test performance of individual components."""
        # Use medium-sized file for component testing
        test_file = self.test_files['medium']
        audio = self.loader.load(test_file['path'])
        
        component_times = {}
        
        # Test each component
        components = [
            ('snr_calculation', 
             lambda: self.snr_calculator.calculate(audio.samples, audio.sample_rate)),
            ('spectral_analysis',
             lambda: self.spectral_analyzer.analyze(audio.samples, audio.sample_rate)),
            ('pattern_detection',
             lambda: self.pattern_detector.detect(audio.samples, None)),
            ('issue_categorization',
             lambda: self.issue_categorizer.categorize(None, None, None)),
            ('decision_making',
             lambda: self.decision_engine.decide([]))
        ]
        
        for component_name, component_func in components:
            _, exec_time = self.benchmark.measure_execution_time(component_func)
            component_times[component_name] = exec_time
            
        # No single component should take more than 50% of total time
        total_time = sum(component_times.values())
        for component, time_taken in component_times.items():
            percentage = (time_taken / total_time) * 100
            self.assertLess(
                percentage,
                50,
                f"{component} takes {percentage:.1f}% of processing time"
            )
            
        self.benchmark.results['component_times'] = component_times
        
    def test_concurrent_processing_performance(self):
        """Test performance with concurrent processing."""
        # Test files for concurrent processing
        test_files = [self.test_files['small']['path'] for _ in range(10)]
        
        # Sequential processing
        sequential_start = time.time()
        for file_path in test_files:
            audio = self.loader.load(file_path)
            self.snr_calculator.calculate(audio.samples, audio.sample_rate)
        sequential_time = time.time() - sequential_start
        
        # Concurrent processing with threads
        thread_start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file_path in test_files:
                future = executor.submit(self._process_file, file_path)
                futures.append(future)
            
            # Wait for all to complete
            for future in futures:
                future.result()
        thread_time = time.time() - thread_start
        
        # Concurrent processing should be faster
        speedup = sequential_time / thread_time
        self.assertGreater(speedup, 1.5, "Insufficient speedup from concurrent processing")
        
        print(f"Concurrent processing speedup: {speedup:.2f}x")
        
    def test_stress_test_1000_files(self):
        """Stress test with 1000 file batch processing."""
        # Create lightweight test data
        batch_size = 1000
        sample_rate = 16000
        duration = 0.5  # 500ms clips
        
        # Generate batch of audio data in memory
        audio_batch = []
        for i in range(batch_size):
            t = np.linspace(0, duration, int(duration * sample_rate))
            audio = 0.3 * np.sin(2 * np.pi * 440 * t)
            audio += np.random.normal(0, 0.05, len(audio))
            audio_batch.append(audio)
            
        # Process batch
        start_time = time.time()
        results = []
        
        for audio in audio_batch:
            snr = self.snr_calculator.calculate(audio, sample_rate)
            results.append(snr.global_snr)
            
        total_time = time.time() - start_time
        
        # Should process 1000 files in under 5 minutes
        self.assertLess(total_time, 300, "Batch processing too slow")
        
        # Calculate throughput
        files_per_second = batch_size / total_time
        print(f"Throughput: {files_per_second:.2f} files/second")
        
        # Should process at least 10 files per second
        self.assertGreater(files_per_second, 10)
        
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Process same file multiple times
        test_file = self.test_files['small']['path']
        memory_readings = []
        
        for i in range(50):
            audio = self.loader.load(test_file)
            snr = self.snr_calculator.calculate(audio.samples, audio.sample_rate)
            
            if i % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)
                
        # Memory should stabilize (not continuously increase)
        memory_growth = memory_readings[-1] - memory_readings[0]
        self.assertLess(
            memory_growth,
            50,  # Less than 50MB growth
            f"Potential memory leak detected: {memory_growth:.1f}MB growth"
        )
        
    def test_cpu_gpu_utilization(self):
        """Test CPU and GPU utilization patterns."""
        test_file = self.test_files['large']['path']
        
        # Profile CPU usage
        audio = self.loader.load(test_file)
        _, cpu_stats = self.benchmark.profile_cpu_usage(
            self._process_file, test_file
        )
        
        # CPU usage should be reasonable
        self.assertLess(
            cpu_stats['average'],
            80,
            f"Average CPU usage too high: {cpu_stats['average']:.1f}%"
        )
        
        # Store results
        self.benchmark.results['cpu_usage'].append(cpu_stats)
        
    def _process_file(self, file_path):
        """Helper method to process a single file."""
        audio = self.loader.load(file_path)
        snr = self.snr_calculator.calculate(audio.samples, audio.sample_rate)
        return snr
        
    def test_performance_regression(self):
        """Test for performance regression against baseline."""
        # Load baseline if exists
        baseline_path = 'performance_baseline.json'
        baseline = None
        
        try:
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)
        except FileNotFoundError:
            print("No baseline found, creating new baseline")
            
        # Run standard benchmark
        test_file = self.test_files['medium']['path']
        
        start_time = time.perf_counter()
        audio = self.loader.load(test_file)
        snr = self.snr_calculator.calculate(audio.samples, audio.sample_rate)
        current_time = time.perf_counter() - start_time
        
        if baseline:
            # Compare with baseline
            baseline_time = baseline.get('medium_file_time', current_time)
            regression = (current_time - baseline_time) / baseline_time
            
            # Allow up to 10% regression
            self.assertLess(
                regression,
                0.1,
                f"Performance regression detected: {regression*100:.1f}% slower"
            )
        else:
            # Save as new baseline
            baseline = {'medium_file_time': current_time}
            with open(baseline_path, 'w') as f:
                json.dump(baseline, f)


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization strategies."""
    
    def setUp(self):
        """Set up optimization tests."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_caching_performance(self):
        """Test performance improvements from caching."""
        from processors.audio_enhancement.audio_loader import AudioCache
        
        # Create test file
        test_path = os.path.join(self.temp_dir, 'test.wav')
        audio = np.random.randn(16000 * 2)  # 2 seconds
        sf.write(test_path, audio, 16000)
        
        # Without cache
        loader_no_cache = AudioLoader()
        start = time.time()
        for _ in range(10):
            loader_no_cache.load(test_path)
        no_cache_time = time.time() - start
        
        # With cache
        cache = AudioCache(max_size=100)
        loader_with_cache = AudioLoader(cache=cache)
        start = time.time()
        for _ in range(10):
            loader_with_cache.load(test_path)
        cache_time = time.time() - start
        
        # Cache should provide significant speedup
        speedup = no_cache_time / cache_time
        self.assertGreater(speedup, 5.0, "Insufficient speedup from caching")
        
    def test_vectorization_performance(self):
        """Test performance of vectorized operations."""
        # Large array for testing
        size = 1000000
        data = np.random.randn(size)
        
        # Non-vectorized operation
        def non_vectorized():
            result = []
            for x in data:
                result.append(x * 2 + 3)
            return np.array(result)
            
        # Vectorized operation
        def vectorized():
            return data * 2 + 3
            
        # Measure performance
        start = time.time()
        non_vec_result = non_vectorized()
        non_vec_time = time.time() - start
        
        start = time.time()
        vec_result = vectorized()
        vec_time = time.time() - start
        
        # Vectorized should be much faster
        speedup = non_vec_time / vec_time
        self.assertGreater(speedup, 10.0, "Insufficient speedup from vectorization")
        
        # Results should be identical
        np.testing.assert_allclose(non_vec_result, vec_result)


def generate_performance_plots(report_path='performance_report.json'):
    """Generate performance visualization plots."""
    with open(report_path, 'r') as f:
        report = json.load(f)
        
    # Create plots directory
    plots_dir = 'performance_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Processing time distribution
    plt.figure(figsize=(10, 6))
    processing_times = report['detailed_results']['processing_times']
    plt.hist(processing_times, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Processing Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Processing Time Distribution')
    plt.savefig(os.path.join(plots_dir, 'processing_time_dist.png'))
    plt.close()
    
    # Plot 2: Component performance breakdown
    if 'component_times' in report['detailed_results']:
        plt.figure(figsize=(10, 6))
        components = list(report['detailed_results']['component_times'].keys())
        times = list(report['detailed_results']['component_times'].values())
        
        plt.bar(components, times, color='green', alpha=0.7)
        plt.xlabel('Component')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Component Performance Breakdown')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'component_breakdown.png'))
        plt.close()
    
    print(f"Performance plots saved to {plots_dir}/")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    # Generate plots if report exists
    if os.path.exists('performance_report.json'):
        generate_performance_plots()