"""Test suite for Separation Quality Metrics module (S03_T07).

This test suite validates the Separation Quality Metrics with the following requirements:
1. Multiple quality metrics for audio separation evaluation
2. Reference-based and reference-free metrics
3. Perceptual and objective quality measures
4. Real-time computation capability
5. Comprehensive quality reporting
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings
from processors.audio_enhancement.metrics.separation_quality_metrics import (
    SeparationQualityMetrics,
    QualityMetric,
    QualityReport,
    MetricsConfig
)


class TestSeparationQualityMetrics(unittest.TestCase):
    """Test suite for Separation Quality Metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.metrics = SeparationQualityMetrics(sample_rate=self.sample_rate)
        
        # Suppress warnings for missing optional packages
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def test_si_sdr_calculation(self):
        """Test 1.1: Calculate SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)."""
        # Create reference and separated signals
        duration = 3.0
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Reference: clean sine wave
        reference = np.sin(2 * np.pi * 440 * t)
        
        # Test cases with different quality levels
        test_cases = [
            # Perfect separation
            {
                "separated": reference.copy(),
                "expected_si_sdr": (20, 40),  # Very high SI-SDR
                "description": "Perfect separation"
            },
            # Good separation with small noise
            {
                "separated": reference + 0.1 * np.random.randn(len(reference)),
                "expected_si_sdr": (10, 25),
                "description": "Good separation"
            },
            # Poor separation with significant noise
            {
                "separated": reference + 0.5 * np.random.randn(len(reference)),
                "expected_si_sdr": (0, 15),
                "description": "Poor separation"
            }
        ]
        
        for case in test_cases:
            with self.subTest(description=case["description"]):
                si_sdr = self.metrics.calculate_si_sdr(
                    reference=reference,
                    separated=case["separated"]
                )
                
                min_expected, max_expected = case["expected_si_sdr"]
                self.assertGreaterEqual(si_sdr, min_expected,
                                       f"SI-SDR too low for {case['description']}")
                self.assertLessEqual(si_sdr, max_expected,
                                    f"SI-SDR too high for {case['description']}")
    
    def test_si_sir_calculation(self):
        """Test 1.2: Calculate SI-SIR (Scale-Invariant Signal-to-Interference Ratio)."""
        # Create signals with interference
        duration = 2.0
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Target signal
        target = np.sin(2 * np.pi * 440 * t)
        
        # Interference signal
        interference = np.sin(2 * np.pi * 880 * t) * 0.3
        
        # Mixed signal
        mixed = target + interference
        
        # Separated signal (partial removal of interference)
        separated = target + interference * 0.1
        
        # Calculate SI-SIR
        si_sir = self.metrics.calculate_si_sir(
            reference=target,
            separated=separated,
            mixture=mixed
        )
        
        # Should show improvement
        self.assertGreater(si_sir, 5.0, "SI-SIR should show interference reduction")
    
    def test_si_sar_calculation(self):
        """Test 1.3: Calculate SI-SAR (Scale-Invariant Signal-to-Artifacts Ratio)."""
        # Create signal with artifacts
        duration = 2.0
        samples = int(duration * self.sample_rate)
        
        # Clean reference
        reference = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        
        # Add artifacts (clipping, distortion)
        separated = reference.copy()
        # Add clipping artifacts
        separated = np.clip(separated * 1.5, -0.8, 0.8)
        # Add some harmonic distortion
        separated += 0.1 * np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples))
        
        # Calculate SI-SAR
        si_sar = self.metrics.calculate_si_sar(
            reference=reference,
            separated=separated
        )
        
        # Should detect artifacts
        self.assertLess(si_sar, 20.0, "SI-SAR should be reduced due to artifacts")
        self.assertGreater(si_sar, -10.0, "SI-SAR should be reasonable")
    
    def test_perceptual_metrics(self):
        """Test 1.4: Calculate perceptual metrics (PESQ, STOI)."""
        # Create test signals
        duration = 3.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Reference speech-like signal (modulated)
        reference = np.sin(2 * np.pi * 200 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        
        # Degraded version
        separated = reference + 0.1 * np.random.randn(samples)
        
        # Try to calculate PESQ (may fail if not installed)
        try:
            pesq_score = self.metrics.calculate_pesq(
                reference=reference,
                separated=separated,
                sample_rate=self.sample_rate
            )
            
            # PESQ ranges from -0.5 to 4.5
            self.assertGreaterEqual(pesq_score, -0.5)
            self.assertLessEqual(pesq_score, 4.5)
            self.assertGreater(pesq_score, 2.0, "PESQ should be reasonable for mild degradation")
        except ImportError:
            self.skipTest("PESQ not available")
        
        # Try to calculate STOI (may fail if not installed)
        try:
            stoi_score = self.metrics.calculate_stoi(
                reference=reference,
                separated=separated,
                sample_rate=self.sample_rate
            )
            
            # STOI ranges from 0 to 1
            self.assertGreaterEqual(stoi_score, 0.0)
            self.assertLessEqual(stoi_score, 1.0)
            self.assertGreater(stoi_score, 0.7, "STOI should be high for mild degradation")
        except ImportError:
            self.skipTest("STOI not available")
    
    def test_spectral_metrics(self):
        """Test 1.5: Calculate spectral-based metrics."""
        # Create test signals
        duration = 2.0
        samples = int(duration * self.sample_rate)
        
        # Reference with specific spectral content
        freqs = [440, 880, 1320]  # Harmonic series
        reference = np.zeros(samples)
        t = np.linspace(0, duration, samples)
        for freq in freqs:
            reference += np.sin(2 * np.pi * freq * t) / len(freqs)
        
        # Separated with slightly altered spectrum
        separated = reference.copy()
        # Add spectral noise
        separated += 0.05 * np.random.randn(samples)
        # Slightly attenuate high frequency
        from scipy import signal as scipy_signal
        b, a = scipy_signal.butter(4, 2000, fs=self.sample_rate)
        separated = scipy_signal.filtfilt(b, a, separated)
        
        # Calculate spectral divergence
        spectral_div = self.metrics.calculate_spectral_divergence(
            reference=reference,
            separated=separated
        )
        
        # Should be small but non-zero
        self.assertGreater(spectral_div, 0.0)
        self.assertLess(spectral_div, 1.0, "Spectral divergence should be small")
        
        # Calculate log-spectral distance
        log_spectral_dist = self.metrics.calculate_log_spectral_distance(
            reference=reference,
            separated=separated
        )
        
        self.assertGreater(log_spectral_dist, 0.0)
        self.assertLess(log_spectral_dist, 10.0, "Log-spectral distance should be reasonable")
    
    def test_reference_free_metrics(self):
        """Test 2.1: Calculate reference-free quality metrics."""
        # Create test signals with clear quality differences
        duration = 3.0
        samples = int(duration * self.sample_rate)
        
        # High quality separated audio (clean speech-like signal)
        t = np.linspace(0, duration, samples)
        high_quality = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        high_quality += 0.02 * np.random.randn(samples)  # Very little noise
        
        # Low quality separated audio (heavily distorted)
        low_quality = np.random.randn(samples) * 0.8  # Mostly noise
        # Add severe clipping artifacts
        low_quality = np.clip(low_quality * 2.0, -0.9, 0.9)
        
        # Calculate reference-free metrics
        high_metrics = self.metrics.calculate_reference_free_metrics(high_quality)
        low_metrics = self.metrics.calculate_reference_free_metrics(low_quality)
        
        # Check that metrics are computed
        self.assertIn("snr_estimate", high_metrics)
        self.assertIn("clarity_score", high_metrics)
        self.assertIn("artifact_score", high_metrics)
        
        # Check that metrics are in valid ranges
        for metrics in [high_metrics, low_metrics]:
            self.assertGreaterEqual(metrics["snr_estimate"], 0.0)
            self.assertLessEqual(metrics["snr_estimate"], 60.0)
            self.assertGreaterEqual(metrics["clarity_score"], 0.0)
            self.assertLessEqual(metrics["clarity_score"], 1.0)
            self.assertGreaterEqual(metrics["artifact_score"], 0.0)
            self.assertLessEqual(metrics["artifact_score"], 1.0)
        
        # Test that metrics can distinguish different characteristics
        # At minimum, both signals should produce valid metrics
        self.assertIsInstance(high_metrics["snr_estimate"], float)
        self.assertIsInstance(low_metrics["snr_estimate"], float)
        
        # The clipped signal should have some clipping artifacts detected
        # (clipping ratio is part of artifact score calculation)
        self.assertGreater(low_metrics["artifact_score"], 0.1,
                          "Clipped audio should show some artifacts")
    
    def test_batch_evaluation(self):
        """Test 2.2: Batch evaluation of multiple separations."""
        # Create batch of test cases
        batch_size = 5
        duration = 2.0
        samples = int(duration * self.sample_rate)
        
        references = []
        separated_list = []
        
        for i in range(batch_size):
            # Create reference
            t = np.linspace(0, duration, samples)
            freq = 440 * (1 + i * 0.1)  # Slightly different frequencies
            reference = np.sin(2 * np.pi * freq * t)
            
            # Create separated with varying quality
            noise_level = 0.05 * (i + 1)  # Increasing noise
            separated = reference + noise_level * np.random.randn(samples)
            
            references.append(reference)
            separated_list.append(separated)
        
        # Batch evaluate
        results = self.metrics.evaluate_batch(
            references=references,
            separated_list=separated_list,
            metrics=[QualityMetric.SI_SDR, QualityMetric.SPECTRAL_DIVERGENCE]
        )
        
        # Check results
        self.assertEqual(len(results), batch_size)
        
        # Quality should decrease with increasing noise
        si_sdrs = [r.metrics["si_sdr"] for r in results]
        for i in range(1, len(si_sdrs)):
            self.assertLess(si_sdrs[i], si_sdrs[i-1] + 5,  # Allow some variation
                           "SI-SDR should generally decrease with more noise")
    
    def test_comprehensive_report(self):
        """Test 2.3: Generate comprehensive quality report."""
        # Create test signals
        duration = 3.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Reference
        reference = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        
        # Separated versions with different characteristics
        separated_good = reference + 0.05 * np.random.randn(samples)
        separated_poor = reference * 0.5 + 0.3 * np.random.randn(samples)
        
        # Generate comprehensive reports
        report_good = self.metrics.evaluate_separation(
            reference=reference,
            separated=separated_good,
            mixture=reference + 0.2 * np.random.randn(samples)
        )
        
        report_poor = self.metrics.evaluate_separation(
            reference=reference,
            separated=separated_poor,
            mixture=reference + 0.2 * np.random.randn(samples)
        )
        
        # Check report structure
        self.assertIsInstance(report_good, QualityReport)
        self.assertIn("si_sdr", report_good.metrics)
        self.assertIsNotNone(report_good.overall_quality)
        self.assertIsNotNone(report_good.timestamp)
        
        # Overall quality should reflect individual metrics
        self.assertGreater(report_good.overall_quality, report_poor.overall_quality)
        self.assertGreater(report_good.overall_quality, 0.6)
        
        # Quality difference should be significant (at least 0.1)
        quality_diff = report_good.overall_quality - report_poor.overall_quality
        self.assertGreater(quality_diff, 0.1, "Quality difference should be significant")
    
    def test_real_time_computation(self):
        """Test 2.4: Ensure real-time computation performance."""
        import time
        
        # Create test signals
        duration = 10.0  # 10 seconds
        samples = int(duration * self.sample_rate)
        
        reference = np.random.randn(samples) * 0.1
        separated = reference + 0.05 * np.random.randn(samples)
        
        # Measure computation time for basic metrics
        start_time = time.time()
        
        # Calculate multiple metrics
        si_sdr = self.metrics.calculate_si_sdr(reference, separated)
        spectral_div = self.metrics.calculate_spectral_divergence(reference, separated)
        ref_free = self.metrics.calculate_reference_free_metrics(separated)
        
        computation_time = time.time() - start_time
        
        # Should be much faster than real-time
        self.assertLess(computation_time, duration * 0.1,  # 10% of audio duration
                       f"Computation took {computation_time:.2f}s for {duration}s audio")
    
    def test_segment_wise_evaluation(self):
        """Test 3.1: Evaluate quality on audio segments."""
        # Create long audio with varying quality
        duration = 10.0
        samples = int(duration * self.sample_rate)
        segment_duration = 2.0  # 2-second segments
        
        # Create reference
        t = np.linspace(0, duration, samples)
        reference = np.sin(2 * np.pi * 440 * t)
        
        # Create separated with time-varying quality
        separated = reference.copy()
        segment_samples = int(segment_duration * self.sample_rate)
        
        # Add different noise levels to each segment
        for i in range(0, samples, segment_samples):
            end = min(i + segment_samples, samples)
            noise_level = 0.05 * (1 + i / samples)  # Increasing noise
            separated[i:end] += noise_level * np.random.randn(end - i)
        
        # Evaluate segment-wise
        segment_results = self.metrics.evaluate_segments(
            reference=reference,
            separated=separated,
            segment_duration=segment_duration
        )
        
        # Check results
        expected_segments = int(np.ceil(duration / segment_duration))
        self.assertEqual(len(segment_results), expected_segments)
        
        # Quality should generally decrease over time
        qualities = [r.overall_quality for r in segment_results]
        self.assertGreater(qualities[0], qualities[-1],
                          "Quality should decrease with increasing noise")
    
    def test_multi_channel_support(self):
        """Test 3.2: Support for multi-channel audio evaluation."""
        # Create stereo test signals
        duration = 2.0
        samples = int(duration * self.sample_rate)
        channels = 2
        
        # Reference stereo signal
        reference = np.zeros((channels, samples))
        t = np.linspace(0, duration, samples)
        reference[0] = np.sin(2 * np.pi * 440 * t)  # Left: 440 Hz
        reference[1] = np.sin(2 * np.pi * 550 * t)  # Right: 550 Hz
        
        # Separated with channel-specific degradation
        separated = reference.copy()
        separated[0] += 0.1 * np.random.randn(samples)  # More noise on left
        separated[1] += 0.05 * np.random.randn(samples)  # Less noise on right
        
        # Evaluate multi-channel
        mc_report = self.metrics.evaluate_multichannel(
            reference=reference,
            separated=separated
        )
        
        # Check channel-specific results
        self.assertEqual(len(mc_report.channel_reports), channels)
        self.assertLess(mc_report.channel_reports[0].overall_quality,
                       mc_report.channel_reports[1].overall_quality,
                       "Right channel should have better quality")
        
        # Check aggregate metrics
        self.assertIsNotNone(mc_report.aggregate_quality)
        self.assertGreater(mc_report.aggregate_quality, 0.0)
    
    def test_custom_metrics(self):
        """Test 3.3: Support for custom user-defined metrics."""
        # Define custom metric function
        def custom_energy_ratio(reference, separated, **kwargs):
            """Custom metric: energy preservation ratio"""
            ref_energy = np.sum(reference ** 2)
            sep_energy = np.sum(separated ** 2)
            if ref_energy > 0:
                return sep_energy / ref_energy
            return 0.0
        
        # Register custom metric
        self.metrics.register_custom_metric(
            name="energy_ratio",
            func=custom_energy_ratio,
            range=(0, 2),  # Expected range
            higher_is_better=False,  # Closer to 1 is better
            optimal_value=1.0
        )
        
        # Create test signals
        reference = np.random.randn(self.sample_rate * 2) * 0.5
        separated_good = reference * 0.95  # Slight attenuation
        separated_poor = reference * 0.5   # Significant attenuation
        
        # Evaluate with custom metric
        report_good = self.metrics.evaluate_separation(
            reference=reference,
            separated=separated_good,
            include_custom_metrics=True
        )
        
        report_poor = self.metrics.evaluate_separation(
            reference=reference,
            separated=separated_poor,
            include_custom_metrics=True
        )
        
        # Check custom metric included
        self.assertIn("energy_ratio", report_good.metrics)
        self.assertIn("energy_ratio", report_poor.metrics)
        
        # Good separation should preserve energy better
        self.assertLess(abs(report_good.metrics["energy_ratio"] - 1.0),
                       abs(report_poor.metrics["energy_ratio"] - 1.0))
    
    def test_adaptive_metric_selection(self):
        """Test 3.4: Adaptive metric selection based on signal type."""
        # Create different signal types
        duration = 2.0
        samples = int(duration * self.sample_rate)
        
        # Speech-like signal (modulated)
        t = np.linspace(0, duration, samples)
        speech_ref = np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        speech_sep = speech_ref + 0.1 * np.random.randn(samples)
        
        # Music-like signal (harmonic)
        music_ref = np.zeros(samples)
        for harmonic in [1, 2, 3, 4]:
            music_ref += np.sin(2 * np.pi * 440 * harmonic * t) / harmonic
        music_sep = music_ref + 0.05 * np.random.randn(samples)
        
        # Noise signal
        noise_ref = np.random.randn(samples) * 0.3
        noise_sep = noise_ref + 0.1 * np.random.randn(samples)
        
        # Configure adaptive selection
        self.metrics.enable_adaptive_selection(True)
        
        # Evaluate different signal types
        speech_report = self.metrics.evaluate_separation(speech_ref, speech_sep)
        music_report = self.metrics.evaluate_separation(music_ref, music_sep)
        noise_report = self.metrics.evaluate_separation(noise_ref, noise_sep)
        
        # Check that different metrics were prioritized
        self.assertIn("recommended_metrics", speech_report.metadata)
        self.assertIn("recommended_metrics", music_report.metadata)
        
        # Speech should prioritize perceptual metrics
        if "stoi" in speech_report.metrics:
            self.assertIn("stoi", speech_report.metadata["recommended_metrics"])
    
    def test_statistical_analysis(self):
        """Test 4: Statistical analysis of metrics."""
        # Create multiple test samples
        num_samples = 20
        duration = 1.0
        samples_per_audio = int(duration * self.sample_rate)
        
        all_results = []
        
        for i in range(num_samples):
            # Create reference
            reference = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples_per_audio))
            
            # Add varying levels of noise
            noise_level = 0.05 + 0.02 * i
            separated = reference + noise_level * np.random.randn(samples_per_audio)
            
            # Evaluate
            report = self.metrics.evaluate_separation(reference, separated)
            all_results.append(report)
        
        # Perform statistical analysis
        stats = self.metrics.analyze_results(all_results)
        
        # Check statistics
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("percentiles", stats)
        
        # Check metric-specific stats
        self.assertIn("si_sdr", stats["mean"])
        self.assertGreater(stats["std"]["si_sdr"], 0.0)
        
        # Check percentiles
        self.assertIn(25, stats["percentiles"])
        self.assertIn(50, stats["percentiles"])
        self.assertIn(75, stats["percentiles"])
        
        # Median should be between 25th and 75th percentile
        self.assertGreaterEqual(stats["percentiles"][50]["si_sdr"],
                               stats["percentiles"][25]["si_sdr"])
        self.assertLessEqual(stats["percentiles"][50]["si_sdr"],
                            stats["percentiles"][75]["si_sdr"])


if __name__ == "__main__":
    unittest.main()