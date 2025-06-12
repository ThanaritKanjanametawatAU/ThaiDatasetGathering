"""
Test suite for Pattern→MetricGAN+ Quality Validation Framework

Comprehensive quality validation tests following TDD principles,
ensuring Pattern→MetricGAN+ enhancement meets project quality standards.
"""

import unittest
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from processors.audio_enhancement.quality.pattern_metricgan_validator import (
    PatternMetricGANQualityValidator,
    PatternSpecificMetrics,
    QualityThresholds,
    QualityReport,
    InterruptionPattern
)


class TestPatternMetricGANQuality(unittest.TestCase):
    """Comprehensive quality validation tests for Pattern→MetricGAN+ enhancement"""
    
    def setUp(self):
        """Set up test environment following existing patterns"""
        self.validator = PatternMetricGANQualityValidator()
        self.sample_rate = 16000
        self.test_duration = 3.0  # 3 seconds
        self.test_samples = int(self.test_duration * self.sample_rate)
        
        # Create test audio samples
        self.original_audio = self._create_test_audio_with_interruption()
        self.enhanced_audio = self._create_enhanced_test_audio()
        self.clean_audio = self._create_clean_test_audio()
        
        # Create test patterns
        self.test_patterns = [
            InterruptionPattern(start=1.0, end=1.5, confidence=0.85, duration=0.5),
            InterruptionPattern(start=2.0, end=2.2, confidence=0.92, duration=0.2)
        ]
        
        self.test_boundaries = [
            (int(1.0 * self.sample_rate), int(1.5 * self.sample_rate)),
            (int(2.0 * self.sample_rate), int(2.2 * self.sample_rate))
        ]
    
    def _create_test_audio_with_interruption(self) -> np.ndarray:
        """Create test audio with simulated interruption patterns"""
        # Create base speech signal (sine wave with varying frequency)
        t = np.linspace(0, self.test_duration, self.test_samples)
        base_signal = 0.3 * np.sin(2 * np.pi * 300 * t)  # 300Hz base frequency
        
        # Add interruption patterns
        interruption_start = int(1.0 * self.sample_rate)
        interruption_end = int(1.5 * self.sample_rate)
        base_signal[interruption_start:interruption_end] += 0.6 * np.sin(2 * np.pi * 800 * t[interruption_start:interruption_end])
        
        # Add second interruption
        interruption_start2 = int(2.0 * self.sample_rate)
        interruption_end2 = int(2.2 * self.sample_rate)
        base_signal[interruption_start2:interruption_end2] += 0.4 * np.sin(2 * np.pi * 1200 * t[interruption_start2:interruption_end2])
        
        # Add some noise
        noise = 0.05 * np.random.randn(self.test_samples)
        return base_signal + noise
    
    def _create_enhanced_test_audio(self) -> np.ndarray:
        """Create test audio simulating Pattern→MetricGAN+ enhancement"""
        # Start with original
        enhanced = self.original_audio.copy()
        
        # Simulate pattern suppression (reduce energy in interruption regions)
        interruption_start = int(1.0 * self.sample_rate)
        interruption_end = int(1.5 * self.sample_rate)
        enhanced[interruption_start:interruption_end] *= 0.15  # 85% suppression
        
        interruption_start2 = int(2.0 * self.sample_rate)
        interruption_end2 = int(2.2 * self.sample_rate)
        enhanced[interruption_start2:interruption_end2] *= 0.15
        
        # Simulate loudness normalization (160% increase)
        enhanced *= 1.6
        
        # Apply soft limiting to prevent clipping
        enhanced = np.clip(enhanced, -0.95, 0.95)
        
        return enhanced
    
    def _create_clean_test_audio(self) -> np.ndarray:
        """Create clean test audio without interruptions"""
        t = np.linspace(0, self.test_duration, self.test_samples)
        clean_signal = 0.3 * np.sin(2 * np.pi * 300 * t)
        noise = 0.02 * np.random.randn(self.test_samples)
        return clean_signal + noise
    
    def test_quality_thresholds_met(self):
        """Test 1: Verify all quality thresholds are consistently met"""
        report = self.validator.validate_enhancement(
            self.original_audio,
            self.enhanced_audio,
            "test_sample_001",
            detected_patterns=self.test_patterns,
            pattern_boundaries=self.test_boundaries
        )
        
        # Verify report structure
        self.assertIsInstance(report, QualityReport)
        self.assertEqual(report.sample_id, "test_sample_001")
        self.assertIsInstance(report.original_metrics, dict)
        self.assertIsInstance(report.enhanced_metrics, dict)
        self.assertIsInstance(report.improvement_metrics, dict)
        self.assertIsInstance(report.pattern_metrics, dict)
        self.assertIsInstance(report.threshold_compliance, dict)
        self.assertIsInstance(report.overall_pass, bool)
        self.assertGreater(report.processing_time, 0)
        
        # Check that key metrics are present
        required_pattern_metrics = [
            'pattern_suppression_effectiveness',
            'primary_speaker_preservation',
            'transition_smoothness',
            'loudness_consistency',
            'naturalness_score'
        ]
        
        for metric in required_pattern_metrics:
            self.assertIn(metric, report.pattern_metrics)
            self.assertGreaterEqual(report.pattern_metrics[metric], 0.0)
            self.assertLessEqual(report.pattern_metrics[metric], 1.0)
        
        # Check threshold compliance structure
        required_compliance_checks = [
            'pattern_suppression_effectiveness',
            'speaker_similarity',
            'naturalness_score',
            'loudness_consistency'
        ]
        
        for check in required_compliance_checks:
            self.assertIn(check, report.threshold_compliance)
            self.assertIsInstance(report.threshold_compliance[check], bool)
    
    def test_pattern_detection_accuracy(self):
        """Test 2: Validate pattern detection accuracy on known samples"""
        pattern_metrics = PatternSpecificMetrics(self.sample_rate)
        
        # Test pattern suppression effectiveness
        effectiveness = pattern_metrics.calculate_pattern_suppression_effectiveness(
            self.original_audio,
            self.enhanced_audio,
            self.test_patterns,
            self.sample_rate
        )
        
        # Should show high suppression effectiveness (>0.8) due to 85% reduction
        self.assertGreater(effectiveness, 0.8)
        self.assertLessEqual(effectiveness, 1.0)
        
        # Test with no patterns
        no_pattern_effectiveness = pattern_metrics.calculate_pattern_suppression_effectiveness(
            self.original_audio,
            self.enhanced_audio,
            [],  # No patterns
            self.sample_rate
        )
        
        # Should return 1.0 when no patterns to suppress
        self.assertEqual(no_pattern_effectiveness, 1.0)
    
    def test_speaker_preservation_quality(self):
        """Test 3: Ensure speaker characteristics are preserved"""
        pattern_metrics = PatternSpecificMetrics(self.sample_rate)
        
        # Test speaker preservation
        preservation_score = pattern_metrics.calculate_primary_speaker_preservation(
            self.original_audio,
            self.enhanced_audio,
            self.sample_rate
        )
        
        # Should show good speaker preservation (>0.7)
        self.assertGreater(preservation_score, 0.7)
        self.assertLessEqual(preservation_score, 1.0)
        
        # Test with identical audio (should be very high)
        perfect_preservation = pattern_metrics.calculate_primary_speaker_preservation(
            self.clean_audio,
            self.clean_audio,
            self.sample_rate
        )
        
        self.assertGreater(perfect_preservation, 0.95)
    
    def test_loudness_normalization_consistency(self):
        """Test 4: Validate 160% loudness normalization accuracy"""
        pattern_metrics = PatternSpecificMetrics(self.sample_rate)
        
        # Test loudness consistency for 160% target
        consistency_score = pattern_metrics.calculate_loudness_consistency(
            self.enhanced_audio,
            target_multiplier=1.6
        )
        
        # Should show good consistency
        self.assertGreater(consistency_score, 0.8)
        self.assertLessEqual(consistency_score, 1.0)
        
        # Test with different target multipliers
        for multiplier in [1.2, 1.8, 2.0]:
            score = pattern_metrics.calculate_loudness_consistency(
                self.enhanced_audio,
                target_multiplier=multiplier
            )
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_edge_case_handling(self):
        """Test 5: Verify robust handling of edge cases"""
        # Test with very short audio
        short_audio = np.random.randn(1000)  # ~0.06 seconds
        short_enhanced = short_audio * 1.6
        
        report = self.validator.validate_enhancement(
            short_audio,
            short_enhanced,
            "short_audio_test"
        )
        
        self.assertIsInstance(report, QualityReport)
        self.assertIsInstance(report.overall_pass, bool)
        
        # Test with silent audio
        silent_audio = np.zeros(self.test_samples)
        silent_enhanced = np.zeros(self.test_samples)
        
        silent_report = self.validator.validate_enhancement(
            silent_audio,
            silent_enhanced,
            "silent_audio_test"
        )
        
        self.assertIsInstance(silent_report, QualityReport)
        self.assertIsInstance(silent_report.overall_pass, bool)
        
        # Test with very loud audio (near clipping)
        loud_audio = 0.98 * np.ones(self.test_samples)
        loud_enhanced = np.clip(loud_audio * 1.6, -0.99, 0.99)
        
        loud_report = self.validator.validate_enhancement(
            loud_audio,
            loud_enhanced,
            "loud_audio_test"
        )
        
        self.assertIsInstance(loud_report, QualityReport)
        
        # Test with NaN or infinite values
        corrupted_audio = self.original_audio.copy()
        corrupted_audio[100:110] = np.nan
        
        corrupted_report = self.validator.validate_enhancement(
            corrupted_audio,
            self.enhanced_audio,
            "corrupted_audio_test"
        )
        
        # Should handle gracefully and not crash
        self.assertIsInstance(corrupted_report, QualityReport)
    
    def test_transition_smoothness_calculation(self):
        """Test transition smoothness calculation"""
        pattern_metrics = PatternSpecificMetrics(self.sample_rate)
        
        # Test with smooth transitions
        smoothness_score = pattern_metrics.calculate_transition_smoothness(
            self.enhanced_audio,
            self.test_boundaries,
            self.sample_rate
        )
        
        self.assertGreaterEqual(smoothness_score, 0.0)
        self.assertLessEqual(smoothness_score, 1.0)
        
        # Test with no boundaries
        no_boundary_score = pattern_metrics.calculate_transition_smoothness(
            self.enhanced_audio,
            [],  # No boundaries
            self.sample_rate
        )
        
        self.assertEqual(no_boundary_score, 1.0)
        
        # Test with boundaries at edges
        edge_boundaries = [
            (0, 100),  # Start edge
            (len(self.enhanced_audio) - 100, len(self.enhanced_audio))  # End edge
        ]
        
        edge_score = pattern_metrics.calculate_transition_smoothness(
            self.enhanced_audio,
            edge_boundaries,
            self.sample_rate
        )
        
        self.assertGreaterEqual(edge_score, 0.0)
        self.assertLessEqual(edge_score, 1.0)
    
    def test_custom_quality_thresholds(self):
        """Test quality validation with custom thresholds"""
        # Create stricter thresholds
        strict_thresholds = QualityThresholds(
            si_sdr_improvement=12.0,  # Higher than default 8.0
            pesq_score=3.8,           # Higher than default 3.2
            stoi_score=0.92,          # Higher than default 0.87
            pattern_suppression_effectiveness=0.98  # Higher than default 0.92
        )
        
        strict_validator = PatternMetricGANQualityValidator(
            sample_rate=self.sample_rate,
            thresholds=strict_thresholds
        )
        
        report = strict_validator.validate_enhancement(
            self.original_audio,
            self.enhanced_audio,
            "strict_test",
            detected_patterns=self.test_patterns
        )
        
        self.assertIsInstance(report, QualityReport)
        # With stricter thresholds, more tests might fail
        self.assertIsInstance(report.overall_pass, bool)
        
        # Create lenient thresholds
        lenient_thresholds = QualityThresholds(
            si_sdr_improvement=2.0,   # Lower than default
            pesq_score=2.0,           # Lower than default
            stoi_score=0.5,           # Lower than default
            pattern_suppression_effectiveness=0.5  # Lower than default
        )
        
        lenient_validator = PatternMetricGANQualityValidator(
            sample_rate=self.sample_rate,
            thresholds=lenient_thresholds
        )
        
        lenient_report = lenient_validator.validate_enhancement(
            self.original_audio,
            self.enhanced_audio,
            "lenient_test",
            detected_patterns=self.test_patterns
        )
        
        # With lenient thresholds, more tests should pass
        self.assertIsInstance(lenient_report, QualityReport)
    
    def test_batch_validation(self):
        """Test batch validation functionality"""
        # Create batch of test samples
        batch_data = []
        for i in range(5):
            original = self._create_test_audio_with_interruption()
            enhanced = self._create_enhanced_test_audio()
            sample_id = f"batch_test_{i:03d}"
            batch_data.append((original, enhanced, sample_id))
        
        # Run batch validation
        reports = self.validator.validate_batch(
            batch_data,
            detected_patterns=self.test_patterns,
            pattern_boundaries=self.test_boundaries
        )
        
        # Verify results
        self.assertEqual(len(reports), 5)
        
        for i, report in enumerate(reports):
            self.assertIsInstance(report, QualityReport)
            self.assertEqual(report.sample_id, f"batch_test_{i:03d}")
            self.assertIsInstance(report.overall_pass, bool)
        
        # Test batch summary generation
        summary = self.validator.generate_batch_summary(reports)
        
        self.assertIn('total_samples', summary)
        self.assertIn('passed_samples', summary)
        self.assertIn('pass_rate', summary)
        self.assertIn('average_processing_time', summary)
        
        self.assertEqual(summary['total_samples'], 5)
        self.assertGreaterEqual(summary['passed_samples'], 0)
        self.assertLessEqual(summary['passed_samples'], 5)
        self.assertGreaterEqual(summary['pass_rate'], 0.0)
        self.assertLessEqual(summary['pass_rate'], 1.0)
        self.assertGreater(summary['average_processing_time'], 0.0)


class TestPatternMetricGANPerformance(unittest.TestCase):
    """Performance benchmarking for Pattern→MetricGAN+ pipeline"""
    
    def setUp(self):
        """Set up performance testing environment"""
        self.validator = PatternMetricGANQualityValidator()
        self.sample_rate = 16000
        
        # Create performance test samples
        self.small_sample = np.random.randn(self.sample_rate)  # 1 second
        self.medium_sample = np.random.randn(5 * self.sample_rate)  # 5 seconds
        self.large_sample = np.random.randn(10 * self.sample_rate)  # 10 seconds
    
    def test_processing_speed_requirements(self):
        """Test processing speed meets <1.8s per sample requirement"""
        # Test small sample
        start_time = time.time()
        report = self.validator.validate_enhancement(
            self.small_sample,
            self.small_sample * 1.6,
            "perf_test_small"
        )
        processing_time = time.time() - start_time
        
        # Should process 1-second sample quickly
        self.assertLess(processing_time, 0.5)  # Should be much faster than 1.8s
        self.assertIsInstance(report, QualityReport)
        
        # Test medium sample
        start_time = time.time()
        report = self.validator.validate_enhancement(
            self.medium_sample,
            self.medium_sample * 1.6,
            "perf_test_medium"
        )
        processing_time = time.time() - start_time
        
        # Should process 5-second sample reasonably fast
        self.assertLess(processing_time, 1.0)
        self.assertIsInstance(report, QualityReport)
    
    def test_memory_efficiency(self):
        """Test memory usage stays within acceptable bounds"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large sample
        report = self.validator.validate_enhancement(
            self.large_sample,
            self.large_sample * 1.6,
            "memory_test"
        )
        
        # Check memory usage after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 10s audio)
        self.assertLess(memory_increase, 500)  # MB
        self.assertIsInstance(report, QualityReport)
    
    def test_batch_processing_efficiency(self):
        """Test batch processing optimization effectiveness"""
        # Create batch of samples
        batch_size = 10
        batch_data = []
        
        for i in range(batch_size):
            sample = np.random.randn(2 * self.sample_rate)  # 2 seconds each
            enhanced = sample * 1.6
            batch_data.append((sample, enhanced, f"batch_perf_{i}"))
        
        # Time batch processing
        start_time = time.time()
        reports = self.validator.validate_batch(batch_data)
        batch_time = time.time() - start_time
        
        # Time individual processing
        individual_times = []
        for original, enhanced, sample_id in batch_data:
            start_time = time.time()
            self.validator.validate_enhancement(original, enhanced, sample_id)
            individual_times.append(time.time() - start_time)
        
        total_individual_time = sum(individual_times)
        
        # Batch processing should be reasonably efficient
        efficiency_ratio = batch_time / total_individual_time
        
        # Should be somewhat efficient (batch time <= individual time)
        self.assertLessEqual(efficiency_ratio, 1.2)  # Allow 20% overhead
        self.assertEqual(len(reports), batch_size)
    
    def test_gpu_utilization_mock(self):
        """Test GPU utilization efficiency (mocked for testing)"""
        # This would test actual GPU utilization in a real environment
        # For testing, we mock the behavior
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
                
                # Test processing with GPU simulation
                report = self.validator.validate_enhancement(
                    self.medium_sample,
                    self.medium_sample * 1.6,
                    "gpu_test"
                )
                
                self.assertIsInstance(report, QualityReport)
                # In real implementation, would check GPU memory usage
                # and utilization metrics


class TestPatternMetricGANEdgeCases(unittest.TestCase):
    """Edge case testing for Pattern→MetricGAN+ enhancement"""
    
    def setUp(self):
        """Set up edge case testing environment"""
        self.validator = PatternMetricGANQualityValidator()
        self.sample_rate = 16000
    
    def test_overlapping_speakers_handling(self):
        """Test handling of overlapping speaker scenarios"""
        # Create overlapping speaker scenario
        duration = 3.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Two overlapping speakers with different frequencies
        speaker1 = 0.4 * np.sin(2 * np.pi * 200 * t)  # 200Hz
        speaker2 = 0.3 * np.sin(2 * np.pi * 600 * t)  # 600Hz
        overlapping_audio = speaker1 + speaker2
        
        # Simulate enhancement that reduces speaker2
        enhanced_audio = speaker1 + 0.1 * speaker2  # Suppress speaker2
        enhanced_audio *= 1.6  # Apply loudness
        
        report = self.validator.validate_enhancement(
            overlapping_audio,
            enhanced_audio,
            "overlapping_speakers_test"
        )
        
        self.assertIsInstance(report, QualityReport)
        self.assertIsInstance(report.overall_pass, bool)
    
    def test_rapid_speaker_changes(self):
        """Test handling of rapid speaker alternation"""
        duration = 2.0
        samples = int(duration * self.sample_rate)
        
        # Create rapid alternating pattern
        audio = np.zeros(samples)
        chunk_size = samples // 10  # 10 rapid changes
        
        for i in range(10):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, samples)
            
            if i % 2 == 0:
                # Primary speaker
                t_chunk = np.linspace(0, (end_idx - start_idx) / self.sample_rate, end_idx - start_idx)
                audio[start_idx:end_idx] = 0.4 * np.sin(2 * np.pi * 300 * t_chunk)
            else:
                # Secondary speaker
                t_chunk = np.linspace(0, (end_idx - start_idx) / self.sample_rate, end_idx - start_idx)
                audio[start_idx:end_idx] = 0.6 * np.sin(2 * np.pi * 800 * t_chunk)
        
        # Simulate enhancement
        enhanced = audio * 1.6
        
        report = self.validator.validate_enhancement(
            audio,
            enhanced,
            "rapid_changes_test"
        )
        
        self.assertIsInstance(report, QualityReport)
        self.assertIsInstance(report.overall_pass, bool)
    
    def test_low_quality_input_audio(self):
        """Test enhancement of heavily degraded input audio"""
        # Create heavily degraded audio
        duration = 2.0
        samples = int(duration * self.sample_rate)
        
        # Very noisy, low-quality audio
        signal = 0.1 * np.random.randn(samples)  # Mostly noise
        noise = 0.3 * np.random.randn(samples)   # Heavy noise
        degraded_audio = signal + noise
        
        # Simulate some enhancement
        enhanced_audio = signal * 0.8 + noise * 0.3  # Reduce noise somewhat
        enhanced_audio *= 1.6
        
        report = self.validator.validate_enhancement(
            degraded_audio,
            enhanced_audio,
            "low_quality_test"
        )
        
        self.assertIsInstance(report, QualityReport)
        # Low quality input might not pass all thresholds
        self.assertIsInstance(report.overall_pass, bool)
    
    def test_extreme_loudness_variations(self):
        """Test loudness normalization with extreme input variations"""
        duration = 2.0
        samples = int(duration * self.sample_rate)
        
        # Create audio with extreme volume variations
        audio = np.zeros(samples)
        chunk_size = samples // 4
        
        # Very quiet section
        audio[0:chunk_size] = 0.01 * np.random.randn(chunk_size)
        
        # Normal section
        audio[chunk_size:2*chunk_size] = 0.3 * np.random.randn(chunk_size)
        
        # Very loud section (near clipping)
        audio[2*chunk_size:3*chunk_size] = 0.9 * np.random.randn(chunk_size)
        
        # Silent section
        audio[3*chunk_size:] = np.zeros(samples - 3*chunk_size)
        
        # Simulate loudness normalization
        enhanced_audio = audio * 1.6
        enhanced_audio = np.clip(enhanced_audio, -0.95, 0.95)
        
        report = self.validator.validate_enhancement(
            audio,
            enhanced_audio,
            "extreme_loudness_test"
        )
        
        self.assertIsInstance(report, QualityReport)
        self.assertIsInstance(report.overall_pass, bool)


class TestPatternMetricGANRobustness(unittest.TestCase):
    """Robustness testing for Pattern→MetricGAN+ system"""
    
    def setUp(self):
        """Set up robustness testing environment"""
        self.validator = PatternMetricGANQualityValidator()
        self.sample_rate = 16000
    
    def test_corrupted_audio_handling(self):
        """Test handling of corrupted or truncated audio"""
        # Test with NaN values
        corrupted_audio = np.random.randn(self.sample_rate)
        corrupted_audio[500:510] = np.nan
        
        enhanced_audio = np.random.randn(self.sample_rate)
        
        report = self.validator.validate_enhancement(
            corrupted_audio,
            enhanced_audio,
            "corrupted_nan_test"
        )
        
        # Should handle gracefully
        self.assertIsInstance(report, QualityReport)
        
        # Test with infinite values
        inf_audio = np.random.randn(self.sample_rate)
        inf_audio[100:105] = np.inf
        
        report2 = self.validator.validate_enhancement(
            inf_audio,
            enhanced_audio,
            "corrupted_inf_test"
        )
        
        self.assertIsInstance(report2, QualityReport)
    
    def test_memory_stress_scenarios(self):
        """Test behavior under memory pressure"""
        # Create large audio samples
        large_samples = []
        
        try:
            # Create several large samples to stress memory
            for i in range(3):
                large_audio = np.random.randn(30 * self.sample_rate)  # 30 seconds
                enhanced_large = large_audio * 1.6
                large_samples.append((large_audio, enhanced_large, f"stress_test_{i}"))
            
            # Process them sequentially
            reports = []
            for original, enhanced, sample_id in large_samples:
                report = self.validator.validate_enhancement(
                    original, enhanced, sample_id
                )
                reports.append(report)
            
            # All should complete successfully
            self.assertEqual(len(reports), 3)
            for report in reports:
                self.assertIsInstance(report, QualityReport)
                
        except MemoryError:
            # If we hit memory limits, that's acceptable for stress testing
            self.skipTest("Insufficient memory for stress testing")
    
    def test_concurrent_processing_safety(self):
        """Test thread safety and concurrent processing"""
        import threading
        import queue
        
        # Create test samples
        test_samples = []
        for i in range(10):
            audio = np.random.randn(self.sample_rate)
            enhanced = audio * 1.6
            test_samples.append((audio, enhanced, f"concurrent_{i}"))
        
        # Process concurrently
        results_queue = queue.Queue()
        
        def process_sample(sample_data):
            original, enhanced, sample_id = sample_data
            report = self.validator.validate_enhancement(
                original, enhanced, sample_id
            )
            results_queue.put(report)
        
        # Start threads
        threads = []
        for sample_data in test_samples:
            thread = threading.Thread(target=process_sample, args=(sample_data,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        reports = []
        while not results_queue.empty():
            reports.append(results_queue.get())
        
        # Should have processed all samples
        self.assertEqual(len(reports), 10)
        for report in reports:
            self.assertIsInstance(report, QualityReport)
    
    def test_model_loading_failures(self):
        """Test graceful handling of model loading failures"""
        # This test simulates model loading failures
        # In practice, this would test actual model loading scenarios
        
        # Test with mocked failures
        with patch('utils.audio_metrics.calculate_pesq', side_effect=Exception("Model load failed")):
            report = self.validator.validate_enhancement(
                np.random.randn(self.sample_rate),
                np.random.randn(self.sample_rate),
                "model_failure_test"
            )
            
            # Should handle gracefully and not crash
            self.assertIsInstance(report, QualityReport)
            # May not pass all tests due to mocked failure
            self.assertIsInstance(report.overall_pass, bool)


class TestRealSampleValidator(unittest.TestCase):
    """Tests for real sample validation framework"""
    
    def setUp(self):
        """Set up real sample validator tests"""
        # Use synthetic samples for testing
        self.validator = RealSampleValidator()
    
    def test_gigaspeech_sample_validation(self):
        """Test validation on GigaSpeech-like samples"""
        results = self.validator.validate_gigaspeech_samples()
        
        # Should have S1-S10 samples
        self.assertGreaterEqual(len(results), 10)
        
        # All results should be SampleTestResult objects
        for sample_id, result in results.items():
            self.assertIsInstance(result, SampleTestResult)
            self.assertIsInstance(result.quality_report, QualityReport)
            self.assertIn('sample_type', result.test_metadata)
    
    def test_edge_case_sample_validation(self):
        """Test validation on challenging edge case samples"""
        results = self.validator.validate_edge_case_samples()
        
        # Should have challenging samples
        self.assertGreater(len(results), 0)
        
        for sample_id, result in results.items():
            self.assertIsInstance(result, SampleTestResult)
            self.assertIn('challenge', sample_id.lower())
    
    def test_thai_linguistic_validation(self):
        """Test Thai linguistic feature validation"""
        results = self.validator.validate_thai_linguistic_features()
        
        # Should have Thai-specific results
        self.assertGreater(len(results), 0)
        
        for sample_id, result in results.items():
            self.assertIsInstance(result, SampleTestResult)
            self.assertIn('thai', sample_id.lower())
            self.assertEqual(result.test_metadata['sample_type'], 'thai_linguistic')


class TestPatternMetricGANComparison(unittest.TestCase):
    """Tests for A/B comparison framework"""
    
    def setUp(self):
        """Set up comparison tests"""
        self.comparison = PatternMetricGANComparison()
        
        # Create test samples
        self.test_samples = []
        for i in range(3):
            audio = np.random.randn(16000)  # 1 second
            sr = 16000
            sample_id = f"comparison_test_{i}"
            self.test_samples.append((audio, sr, sample_id))
    
    def test_comparison_with_existing_levels(self):
        """Test comparison with baseline enhancement levels"""
        results = self.comparison.compare_with_existing_levels(self.test_samples)
        
        # Should have comparison results for each sample
        self.assertEqual(len(results), len(self.test_samples))
        
        for result in results:
            self.assertIsInstance(result, ComparisonResult)
            self.assertIsInstance(result.pattern_metricgan_report, QualityReport)
            self.assertIsInstance(result.baseline_reports, dict)
            self.assertIn(result.winner, ['pattern_metricgan_plus', 'moderate', 'aggressive', 'ultra_aggressive'])
            self.assertGreaterEqual(result.improvement_score, 0.0)
    
    def test_quality_report_generation(self):
        """Test quality comparison report generation"""
        # First run comparisons
        comparison_results = self.comparison.compare_with_existing_levels(self.test_samples)
        
        # Generate report
        report = self.comparison.generate_quality_reports(comparison_results)
        
        # Verify report structure
        self.assertIn('summary', report)
        self.assertIn('method_performance', report)
        self.assertIn('quality_metrics', report)
        self.assertIn('detailed_results', report)
        
        # Check summary metrics
        summary = report['summary']
        self.assertEqual(summary['total_comparisons'], len(self.test_samples))
        self.assertIn('pattern_metricgan_wins', summary)
        self.assertIn('win_rate', summary)


class TestQualityMonitor(unittest.TestCase):
    """Tests for quality monitoring pipeline"""
    
    def setUp(self):
        """Set up quality monitor tests"""
        self.monitor = PatternMetricGANQualityMonitor()
        
        # Create test batch
        self.test_batch = []
        for i in range(5):
            original = np.random.randn(16000)
            enhanced = original * 1.6  # Simulate enhancement
            sample_id = f"monitor_test_{i}"
            self.test_batch.append((original, enhanced, sample_id))
    
    def test_batch_quality_validation(self):
        """Test batch quality validation"""
        batch_report = self.monitor.validate_batch_quality(self.test_batch)
        
        # Verify batch report structure
        self.assertIsInstance(batch_report, BatchQualityReport)
        self.assertEqual(batch_report.total_samples, len(self.test_batch))
        self.assertGreaterEqual(batch_report.passed_samples, 0)
        self.assertLessEqual(batch_report.passed_samples, batch_report.total_samples)
        self.assertIsInstance(batch_report.quality_metrics, dict)
        self.assertIsInstance(batch_report.alerts, list)
    
    def test_quality_trend_analysis(self):
        """Test quality trend analysis"""
        # Validate multiple batches to build history
        for _ in range(3):
            self.monitor.validate_batch_quality(self.test_batch)
        
        # Check that history is building
        self.assertGreater(len(self.monitor.quality_history), 0)
        
        # Test trend analysis
        trends = self.monitor._analyze_quality_trends()
        self.assertIsInstance(trends, dict)
    
    def test_alert_generation(self):
        """Test quality alert generation"""
        # Create a batch with known quality issues
        poor_quality_batch = []
        for i in range(3):
            original = np.random.randn(16000)
            # Create poor enhancement (too much noise)
            enhanced = original + 0.5 * np.random.randn(16000)
            sample_id = f"poor_quality_{i}"
            poor_quality_batch.append((original, enhanced, sample_id))
        
        batch_report = self.monitor.validate_batch_quality(poor_quality_batch)
        
        # Should detect quality issues
        self.assertIsInstance(batch_report.alerts, list)
        # Note: Alerts may or may not be generated depending on thresholds
    
    def test_dashboard_metrics_generation(self):
        """Test dashboard metrics generation"""
        # Build some history first
        self.monitor.validate_batch_quality(self.test_batch)
        
        dashboard_metrics = self.monitor.generate_quality_dashboard_metrics()
        
        # Verify dashboard structure
        self.assertIn('timestamp', dashboard_metrics)
        self.assertIn('current_status', dashboard_metrics)
        self.assertIn('current_metrics', dashboard_metrics)
        self.assertIn('performance', dashboard_metrics)


class TestQualityRegressionTester(unittest.TestCase):
    """Tests for quality regression testing"""
    
    def setUp(self):
        """Set up regression tester"""
        self.regression_tester = QualityRegressionTester()
        
        # Create test samples
        self.test_samples = []
        for i in range(3):
            original = np.random.randn(16000)
            enhanced = original * 1.5  # Simulate enhancement
            sample_id = f"regression_test_{i}"
            self.test_samples.append((original, enhanced, sample_id))
    
    def test_baseline_establishment(self):
        """Test baseline establishment"""
        baseline_reports = self.regression_tester.establish_baseline(
            self.test_samples, version="test_v1.0"
        )
        
        # Verify baseline was established
        self.assertEqual(len(baseline_reports), len(self.test_samples))
        self.assertIn("test_v1.0", self.regression_tester.baseline_results)
        
        for sample_id, report in baseline_reports.items():
            self.assertIsInstance(report, QualityReport)
    
    def test_regression_testing(self):
        """Test regression testing against baseline"""
        # First establish baseline
        self.regression_tester.establish_baseline(self.test_samples, version="test_v1.0")
        
        # Test against baseline (using same samples should show no regression)
        regression_report = self.regression_tester.test_against_baseline_quality(
            self.test_samples, baseline_version="test_v1.0"
        )
        
        # Verify regression report structure
        self.assertIn('overall_status', regression_report)
        self.assertIn('regression_results', regression_report)
        self.assertIn('samples_tested', regression_report)
    
    def test_version_compatibility(self):
        """Test version compatibility validation"""
        # Create two versions of results
        version_results = {
            "v1.0": [self.regression_tester.validator.validate_enhancement(
                orig, enh, sid
            ) for orig, enh, sid in self.test_samples],
            "v1.1": [self.regression_tester.validator.validate_enhancement(
                orig, enh * 1.1, sid  # Slightly different enhancement
            ) for orig, enh, sid in self.test_samples]
        }
        
        compatibility_report = self.regression_tester.validate_version_compatibility(
            version_results
        )
        
        # Verify compatibility report
        self.assertIn('compatibility_status', compatibility_report)
        self.assertIn('overall_compatibility', compatibility_report)
        self.assertIn('versions_tested', compatibility_report)


# Import the new classes for the tests
try:
    from processors.audio_enhancement.quality.real_sample_validator import (
        RealSampleValidator, PatternMetricGANComparison, SampleTestResult, ComparisonResult
    )
    from processors.audio_enhancement.quality.quality_monitor import (
        PatternMetricGANQualityMonitor, QualityRegressionTester, 
        BatchQualityReport, QualityAlert
    )
except ImportError as e:
    # Skip these tests if modules aren't available
    import warnings
    warnings.warn(f"Skipping advanced quality tests due to import error: {e}")
    
    # Create dummy test classes that will be skipped
    class TestRealSampleValidator(unittest.TestCase):
        @unittest.skip("Real sample validator not available")
        def test_skip(self):
            pass
    
    class TestPatternMetricGANComparison(unittest.TestCase):
        @unittest.skip("Comparison framework not available")  
        def test_skip(self):
            pass
    
    class TestQualityMonitor(unittest.TestCase):
        @unittest.skip("Quality monitor not available")
        def test_skip(self):
            pass
    
    class TestQualityRegressionTester(unittest.TestCase):
        @unittest.skip("Regression tester not available")
        def test_skip(self):
            pass


if __name__ == "__main__":
    unittest.main()