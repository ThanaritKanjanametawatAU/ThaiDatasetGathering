"""
Comprehensive integration tests for the audio analysis pipeline.

This module implements the integration test framework as specified in S01_T08.
It validates the entire audio analysis pipeline from loading to decision making,
ensuring all components work together seamlessly.
"""

import unittest
import pytest
import numpy as np
import tempfile
import os
import time
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from concurrent.futures import ProcessPoolExecutor, as_completed
import soundfile as sf
import threading
from typing import Dict, List, Tuple, Any

# Import all S01 components
from processors.audio_enhancement.audio_loader import AudioLoader
from utils.enhanced_snr_calculator import EnhancedSNRCalculator
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer
from processors.audio_enhancement.detection.pattern_detector import PatternDetector
from processors.audio_enhancement.issue_categorization import IssueCategorizer
from processors.audio_enhancement.decision_framework import DecisionEngine, DecisionContext


class IntegrationTestFramework:
    """Enhanced test framework for integration testing."""
    
    def __init__(self):
        self.components = self._initialize_components()
        self.test_data = TestDataGenerator()
        self.validators = ValidationSuite()
        self.profiler = PerformanceProfiler()
        self.failure_handler = TestFailureHandler()
        
    def _initialize_components(self):
        """Initialize all pipeline components."""
        return {
            'audio_loader': AudioLoader(),
            'snr_calculator': EnhancedSNRCalculator(),
            'spectral_analyzer': SpectralAnalyzer(),
            'pattern_detector': PatternDetector(),
            'issue_categorizer': IssueCategorizer(),
            'decision_engine': DecisionEngine()
        }
        
    def run_integration_test(self, scenario):
        """Run a complete integration test scenario."""
        # Setup
        env = self.setup_test_environment(scenario)
        
        try:
            # Execute with profiling
            with self.profiler.measure():
                results = self.execute_scenario(env, scenario)
                
            # Validate
            validation = self.validators.validate_all(results, scenario.expected)
            
            # Generate report
            return TestReport(
                scenario=scenario,
                results=results,
                validation=validation,
                performance=self.profiler.get_metrics()
            )
            
        except Exception as e:
            return self.failure_handler.handle_failure(scenario, e)
            
        finally:
            # Cleanup
            self.cleanup_environment(env)
            
    def setup_test_environment(self, scenario):
        """Set up test environment for scenario."""
        env = TestEnvironment()
        env.temp_dir = tempfile.mkdtemp()
        env.config = scenario.config
        
        # Initialize components with test config
        for name, component in self.components.items():
            if hasattr(component, 'configure'):
                component.configure(env.config.get(name, {}))
                
        return env
        
    def execute_scenario(self, env, scenario):
        """Execute test scenario through pipeline."""
        results = {}
        
        # Stage 1: Audio Loading
        audio_data = self.test_data.generate_audio(scenario.audio_spec)
        loaded_audio = self.components['audio_loader'].load(audio_data)
        results['audio_loading'] = loaded_audio
        
        # Stage 2: SNR Calculation
        snr_metrics = self.components['snr_calculator'].calculate(loaded_audio.samples)
        results['snr_metrics'] = snr_metrics
        
        # Stage 3: Spectral Analysis
        spectral_features = self.components['spectral_analyzer'].analyze(
            loaded_audio.samples, loaded_audio.sample_rate
        )
        results['spectral_features'] = spectral_features
        
        # Stage 4: Pattern Detection
        patterns = self.components['pattern_detector'].detect(
            loaded_audio.samples, spectral_features
        )
        results['patterns'] = patterns
        
        # Stage 5: Issue Categorization
        issues = self.components['issue_categorizer'].categorize(
            snr_metrics, spectral_features, patterns
        )
        results['issues'] = issues
        
        # Stage 6: Decision Making
        decision = self.components['decision_engine'].decide(
            issues, DecisionContext(audio_metrics=snr_metrics)
        )
        results['decision'] = decision
        
        return results
        
    def cleanup_environment(self, env):
        """Clean up test environment."""
        import shutil
        if hasattr(env, 'temp_dir') and os.path.exists(env.temp_dir):
            shutil.rmtree(env.temp_dir)


class TestDataGenerator:
    """Generate test data for integration tests."""
    
    def __init__(self):
        self.sample_rate = 16000
        
    def generate_audio(self, spec):
        """Generate audio based on specification."""
        duration = spec.get('duration', 1.0)
        audio_type = spec.get('type', 'clean')
        
        if audio_type == 'clean':
            return self._generate_clean_audio(duration)
        elif audio_type == 'noisy':
            return self._generate_noisy_audio(duration, spec.get('snr', 10))
        elif audio_type == 'corrupted':
            return self._generate_corrupted_audio(duration)
        elif audio_type == 'multi_issue':
            return self._generate_multi_issue_audio(duration)
        else:
            raise ValueError(f"Unknown audio type: {audio_type}")
            
    def _generate_clean_audio(self, duration):
        """Generate clean speech-like audio."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        # Simulate speech with multiple harmonic components
        audio = np.zeros_like(t)
        for f in [200, 400, 600, 800]:  # Fundamental + harmonics
            audio += 0.25 * np.sin(2 * np.pi * f * t)
        
        # Add envelope
        envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
        audio *= envelope
        
        return AudioData(audio, self.sample_rate)
        
    def _generate_noisy_audio(self, duration, target_snr):
        """Generate audio with specified SNR."""
        clean = self._generate_clean_audio(duration)
        noise = np.random.normal(0, 0.1, len(clean.samples))
        
        # Adjust noise level for target SNR
        signal_power = np.mean(clean.samples ** 2)
        noise_power = np.mean(noise ** 2)
        noise_scaling = np.sqrt(signal_power / (noise_power * (10 ** (target_snr / 10))))
        
        noisy_audio = clean.samples + noise * noise_scaling
        return AudioData(noisy_audio, self.sample_rate)
        
    def _generate_corrupted_audio(self, duration):
        """Generate corrupted audio with artifacts."""
        audio = self._generate_clean_audio(duration)
        
        # Add clipping
        audio.samples = np.clip(audio.samples, -0.5, 0.5)
        
        # Add dropouts
        dropout_mask = np.random.random(len(audio.samples)) > 0.1
        audio.samples *= dropout_mask
        
        return audio
        
    def _generate_multi_issue_audio(self, duration):
        """Generate audio with multiple issues."""
        audio = self._generate_noisy_audio(duration, 5)  # Low SNR
        
        # Add frequency distortion
        from scipy import signal
        b, a = signal.butter(4, 0.1)
        audio.samples = signal.filtfilt(b, a, audio.samples)
        
        # Add echo
        delay_samples = int(0.1 * self.sample_rate)
        echo = np.pad(audio.samples, (delay_samples, 0))[:-delay_samples] * 0.3
        audio.samples += echo
        
        return audio


class ValidationSuite:
    """Validate integration test results."""
    
    def validate_all(self, results, expected):
        """Validate all aspects of results."""
        validations = {
            'data_integrity': self._validate_data_integrity(results),
            'pipeline_flow': self._validate_pipeline_flow(results),
            'output_correctness': self._validate_output_correctness(results, expected),
            'performance': self._validate_performance(results)
        }
        
        return ValidationReport(validations)
        
    def _validate_data_integrity(self, results):
        """Validate data integrity through pipeline."""
        issues = []
        
        # Check audio data preserved
        if 'audio_loading' in results:
            audio = results['audio_loading']
            if not isinstance(audio.samples, np.ndarray):
                issues.append("Audio samples not numpy array")
            if audio.sample_rate != 16000:
                issues.append(f"Sample rate mismatch: {audio.sample_rate}")
                
        # Check metrics are numeric
        if 'snr_metrics' in results:
            snr = results['snr_metrics']
            if not isinstance(snr.global_snr, (int, float)):
                issues.append("SNR not numeric")
                
        return issues
        
    def _validate_pipeline_flow(self, results):
        """Validate data flows correctly through pipeline."""
        required_stages = [
            'audio_loading', 'snr_metrics', 'spectral_features',
            'patterns', 'issues', 'decision'
        ]
        
        missing = [stage for stage in required_stages if stage not in results]
        return missing
        
    def _validate_output_correctness(self, results, expected):
        """Validate outputs match expectations."""
        issues = []
        
        if 'decision' in results and 'decision' in expected:
            actual_decision = results['decision']
            expected_decision = expected['decision']
            
            if actual_decision.action != expected_decision:
                issues.append(
                    f"Decision mismatch: {actual_decision.action} != {expected_decision}"
                )
                
        return issues
        
    def _validate_performance(self, results):
        """Validate performance metrics."""
        # Performance validation handled by profiler
        return []


class PerformanceProfiler:
    """Profile performance of integration tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []
        
    def measure(self):
        """Context manager for performance measurement."""
        return self
        
    def __enter__(self):
        self.start_time = time.time()
        self._start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self._stop_monitoring()
        
    def _start_monitoring(self):
        """Start monitoring resources."""
        import psutil
        self.process = psutil.Process()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring = True
        self.monitor_thread.start()
        
    def _monitor_resources(self):
        """Monitor resource usage in background."""
        while self.monitoring:
            self.memory_usage.append(self.process.memory_info().rss / 1024 / 1024)
            self.cpu_usage.append(self.process.cpu_percent())
            time.sleep(0.1)
            
    def _stop_monitoring(self):
        """Stop monitoring resources."""
        self.monitoring = False
        self.monitor_thread.join()
        
    def get_metrics(self):
        """Get performance metrics."""
        return {
            'duration': self.end_time - self.start_time,
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0
        }


class TestFailureHandler:
    """Handle test failures intelligently."""
    
    def __init__(self):
        self.failure_log = []
        self.recovery_strategies = {}
        
    def handle_failure(self, test_case, error):
        """Handle test failure with context."""
        failure = TestFailure(
            test_case=test_case,
            error=error,
            timestamp=time.time(),
            context=self._capture_context()
        )
        self.failure_log.append(failure)
        
        # Attempt recovery
        if recovery := self.recovery_strategies.get(type(error)):
            return recovery.attempt_recovery(test_case, error)
            
        # Default handling
        return self._default_handling(test_case, error)
        
    def _capture_context(self):
        """Capture debugging context."""
        return {
            'system_state': self._get_system_state(),
            'component_states': self._get_component_states(),
            'recent_logs': self._get_recent_logs()
        }
        
    def _get_system_state(self):
        """Get current system state."""
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
    def _get_component_states(self):
        """Get component states."""
        # Would query actual component states in production
        return {}
        
    def _get_recent_logs(self):
        """Get recent log entries."""
        # Would fetch actual logs in production
        return []
        
    def _default_handling(self, test_case, error):
        """Default error handling."""
        return TestReport(
            scenario=test_case,
            results={},
            validation=ValidationReport({'error': str(error)}),
            performance={}
        )


# Data classes for test framework
class AudioData:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate


class TestEnvironment:
    def __init__(self):
        self.temp_dir = None
        self.config = {}


class TestReport:
    def __init__(self, scenario, results, validation, performance):
        self.scenario = scenario
        self.results = results
        self.validation = validation
        self.performance = performance


class ValidationReport:
    def __init__(self, validations):
        self.validations = validations
        self.passed = all(
            not issues if isinstance(issues, list) else issues
            for issues in validations.values()
        )


class TestFailure:
    def __init__(self, test_case, error, timestamp, context):
        self.test_case = test_case
        self.error = error
        self.timestamp = timestamp
        self.context = context


class TestScenario:
    def __init__(self, name, description, audio_spec, config, expected):
        self.name = name
        self.description = description
        self.audio_spec = audio_spec
        self.config = config
        self.expected = expected


# Actual test class
class TestAudioAnalysisIntegration(unittest.TestCase):
    """Integration tests for audio analysis pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.framework = IntegrationTestFramework()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_complete_pipeline_clean_audio(self):
        """Test complete pipeline with clean audio."""
        scenario = TestScenario(
            name="clean_audio_pipeline",
            description="Process clean audio through entire pipeline",
            audio_spec={'type': 'clean', 'duration': 2.0},
            config={},
            expected={'decision': 'PROCESS'}
        )
        
        report = self.framework.run_integration_test(scenario)
        
        self.assertTrue(report.validation.passed)
        self.assertEqual(report.results['decision'].action, 'PROCESS')
        self.assertGreater(report.results['snr_metrics'].global_snr, 30)
        
    def test_complete_pipeline_noisy_audio(self):
        """Test complete pipeline with noisy audio."""
        scenario = TestScenario(
            name="noisy_audio_pipeline",
            description="Process noisy audio requiring enhancement",
            audio_spec={'type': 'noisy', 'duration': 2.0, 'snr': 5},
            config={},
            expected={'decision': 'ENHANCE'}
        )
        
        report = self.framework.run_integration_test(scenario)
        
        self.assertTrue(report.validation.passed)
        self.assertEqual(report.results['decision'].action, 'ENHANCE')
        self.assertLess(report.results['snr_metrics'].global_snr, 10)
        
    def test_complete_pipeline_corrupted_audio(self):
        """Test complete pipeline with corrupted audio."""
        scenario = TestScenario(
            name="corrupted_audio_pipeline",
            description="Process corrupted audio with artifacts",
            audio_spec={'type': 'corrupted', 'duration': 2.0},
            config={},
            expected={'decision': 'ENHANCE'}
        )
        
        report = self.framework.run_integration_test(scenario)
        
        self.assertTrue(report.validation.passed)
        self.assertIn(report.results['decision'].action, ['ENHANCE', 'REJECT'])
        
    def test_complete_pipeline_multi_issue(self):
        """Test complete pipeline with multiple issues."""
        scenario = TestScenario(
            name="multi_issue_pipeline",
            description="Process audio with multiple quality issues",
            audio_spec={'type': 'multi_issue', 'duration': 2.0},
            config={},
            expected={'decision': 'ENHANCE'}
        )
        
        report = self.framework.run_integration_test(scenario)
        
        self.assertTrue(report.validation.passed)
        self.assertEqual(report.results['decision'].action, 'ENHANCE')
        self.assertGreater(len(report.results['issues']), 1)
        
    def test_component_communication(self):
        """Test that components communicate correctly."""
        scenario = TestScenario(
            name="component_communication",
            description="Verify data flows correctly between components",
            audio_spec={'type': 'clean', 'duration': 1.0},
            config={},
            expected={}
        )
        
        report = self.framework.run_integration_test(scenario)
        
        # Verify all stages executed
        required_stages = [
            'audio_loading', 'snr_metrics', 'spectral_features',
            'patterns', 'issues', 'decision'
        ]
        
        for stage in required_stages:
            self.assertIn(stage, report.results)
            
        # Verify data types match between stages
        self.assertIsInstance(report.results['audio_loading'], AudioData)
        self.assertTrue(hasattr(report.results['snr_metrics'], 'global_snr'))
        self.assertTrue(hasattr(report.results['decision'], 'action'))
        
    def test_error_handling_across_modules(self):
        """Test error propagation through pipeline."""
        # Test with invalid audio data
        with self.assertRaises(Exception):
            scenario = TestScenario(
                name="error_handling",
                description="Test error handling",
                audio_spec={'type': 'invalid'},  # Will cause error
                config={},
                expected={}
            )
            
            self.framework.run_integration_test(scenario)
            
    def test_performance_benchmarks(self):
        """Test performance meets requirements."""
        scenario = TestScenario(
            name="performance_benchmark",
            description="Benchmark pipeline performance",
            audio_spec={'type': 'clean', 'duration': 10.0},
            config={},
            expected={}
        )
        
        report = self.framework.run_integration_test(scenario)
        
        # Check processing time (should process 10s audio in < 2s)
        self.assertLess(report.performance['duration'], 2.0)
        
        # Check memory usage (should use < 500MB)
        self.assertLess(report.performance['peak_memory_mb'], 500)
        
    def test_concurrent_processing(self):
        """Test concurrent processing of multiple files."""
        scenarios = [
            TestScenario(
                name=f"concurrent_{i}",
                description=f"Concurrent test {i}",
                audio_spec={'type': 'clean', 'duration': 1.0},
                config={},
                expected={'decision': 'PROCESS'}
            )
            for i in range(5)
        ]
        
        # Process scenarios concurrently
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self.framework.run_integration_test, scenario)
                for scenario in scenarios
            ]
            
            reports = [future.result() for future in as_completed(futures)]
            
        # Verify all succeeded
        for report in reports:
            self.assertTrue(report.validation.passed)
            
    def test_configuration_propagation(self):
        """Test configuration propagates through pipeline."""
        custom_config = {
            'snr_calculator': {'min_speech_duration': 0.5},
            'decision_engine': {'enhancement_threshold': 20}
        }
        
        scenario = TestScenario(
            name="config_propagation",
            description="Test configuration propagation",
            audio_spec={'type': 'noisy', 'duration': 2.0, 'snr': 15},
            config=custom_config,
            expected={}
        )
        
        report = self.framework.run_integration_test(scenario)
        
        # With custom threshold of 20dB, 15dB SNR should trigger enhancement
        self.assertEqual(report.results['decision'].action, 'ENHANCE')


# Parallel test runner for faster execution
class ParallelTestRunner:
    """Run integration tests in parallel."""
    
    def __init__(self, n_workers=None):
        import multiprocessing
        self.n_workers = n_workers or multiprocessing.cpu_count()
        self.executor = ProcessPoolExecutor(self.n_workers)
        
    def run_tests_parallel(self, test_suite):
        """Run integration tests in parallel."""
        # Group tests by resource requirements
        test_groups = self._group_tests_by_resources(test_suite)
        
        # Execute groups in parallel
        futures = []
        for group in test_groups:
            future = self.executor.submit(self._run_test_group, group)
            futures.append(future)
            
        # Collect results
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
            
        return results
        
    def _group_tests_by_resources(self, test_suite):
        """Group tests by resource requirements."""
        # Simple grouping - could be enhanced
        tests = list(test_suite)
        group_size = max(1, len(tests) // self.n_workers)
        
        groups = []
        for i in range(0, len(tests), group_size):
            groups.append(tests[i:i + group_size])
            
        return groups
        
    def _run_test_group(self, test_group):
        """Run a group of tests."""
        results = []
        for test in test_group:
            result = test.run()
            results.append(result)
        return results


if __name__ == '__main__':
    unittest.main()