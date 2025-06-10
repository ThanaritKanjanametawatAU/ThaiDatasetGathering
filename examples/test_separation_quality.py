#!/usr/bin/env python3
"""Example usage of the Separation Quality Metrics module (S03_T07).

This example demonstrates how to use the Separation Quality Metrics for:
1. Basic quality evaluation
2. Comprehensive quality reporting
3. Batch evaluation
4. Multi-channel assessment
5. Custom metrics
6. Real-world scenario simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from processors.audio_enhancement.metrics.separation_quality_metrics import (
    SeparationQualityMetrics,
    QualityMetric,
    MetricsConfig
)


def generate_test_audio(sample_rate=16000, duration=5.0, scenario="clean"):
    """Generate test audio for different scenarios."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    if scenario == "clean":
        # Clean speech-like signal
        signal = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        signal += 0.2 * np.sin(2 * np.pi * 600 * t) * (1 + 0.2 * np.sin(2 * np.pi * 7 * t))
        noise = 0.02 * np.random.randn(samples)
        return signal + noise
    
    elif scenario == "noisy":
        # Noisy speech
        signal = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        noise = 0.15 * np.random.randn(samples)
        return signal + noise
    
    elif scenario == "distorted":
        # Distorted with artifacts
        signal = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        # Add clipping
        signal = np.clip(signal * 1.5, -0.8, 0.8)
        # Add harmonic distortion
        signal += 0.1 * np.sin(2 * np.pi * 900 * t)
        noise = 0.05 * np.random.randn(samples)
        return signal + noise
    
    elif scenario == "mixed":
        # Mixed sources
        source1 = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        source2 = np.sin(2 * np.pi * 500 * t) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
        return 0.7 * source1 + 0.3 * source2
    
    else:
        return np.random.randn(samples) * 0.1


def example_basic_evaluation():
    """Example 1: Basic separation quality evaluation."""
    print("=" * 60)
    print("Example 1: Basic Separation Quality Evaluation")
    print("=" * 60)
    
    # Create metrics calculator
    metrics = SeparationQualityMetrics(sample_rate=16000)
    
    # Generate test signals
    reference = generate_test_audio(scenario="clean")
    separated_good = reference + 0.05 * np.random.randn(len(reference))
    separated_poor = reference * 0.6 + 0.2 * np.random.randn(len(reference))
    
    # Evaluate both versions
    report_good = metrics.evaluate_separation(reference, separated_good)
    report_poor = metrics.evaluate_separation(reference, separated_poor)
    
    print("Good Separation Results:")
    print(f"  Overall Quality: {report_good.overall_quality:.3f}")
    print(f"  SI-SDR: {report_good.metrics['si_sdr']:.2f} dB")
    print(f"  SI-SAR: {report_good.metrics['si_sar']:.2f} dB")
    
    print("\nPoor Separation Results:")
    print(f"  Overall Quality: {report_poor.overall_quality:.3f}")
    print(f"  SI-SDR: {report_poor.metrics['si_sdr']:.2f} dB")
    print(f"  SI-SAR: {report_poor.metrics['si_sar']:.2f} dB")
    
    print(f"\nQuality Improvement: {report_good.overall_quality - report_poor.overall_quality:.3f}")
    print()


def example_comprehensive_evaluation():
    """Example 2: Comprehensive evaluation with all metrics."""
    print("=" * 60)
    print("Example 2: Comprehensive Quality Assessment")
    print("=" * 60)
    
    # Configure for comprehensive evaluation
    config = MetricsConfig(
        enable_perceptual=True,
        enable_spectral=True,
        enable_reference_free=True,
        adaptive_selection=True
    )
    
    metrics = SeparationQualityMetrics(sample_rate=16000, config=config)
    
    # Generate test case
    reference = generate_test_audio(scenario="clean", duration=3.0)
    mixture = reference + generate_test_audio(scenario="mixed", duration=3.0) * 0.4
    separated = reference + 0.1 * np.random.randn(len(reference))
    
    # Comprehensive evaluation
    report = metrics.evaluate_separation(
        reference=reference,
        separated=separated,
        mixture=mixture
    )
    
    print("Comprehensive Quality Report:")
    print(f"Overall Quality: {report.overall_quality:.3f}")
    print("\nDetailed Metrics:")
    for metric_name, value in report.metrics.items():
        print(f"  {metric_name}: {value:.3f}")
    
    print(f"\nMetadata:")
    print(f"  Signal length: {report.metadata['signal_length']} samples")
    print(f"  Metrics computed: {len(report.metadata['metrics_computed'])}")
    
    if "recommended_metrics" in report.metadata:
        print(f"  Recommended metrics: {report.metadata['recommended_metrics']}")
    
    if report.warnings:
        print(f"\nWarnings: {len(report.warnings)}")
    
    print()


def example_batch_evaluation():
    """Example 3: Batch evaluation of multiple separations."""
    print("=" * 60)
    print("Example 3: Batch Evaluation")
    print("=" * 60)
    
    metrics = SeparationQualityMetrics(sample_rate=16000)
    
    # Generate batch of test cases with varying quality
    batch_size = 5
    references = []
    separated_list = []
    scenarios = ["clean", "noisy", "distorted", "mixed", "clean"]
    noise_levels = [0.02, 0.08, 0.15, 0.25, 0.35]
    
    for i in range(batch_size):
        ref = generate_test_audio(scenario=scenarios[i], duration=2.0)
        sep = ref + noise_levels[i] * np.random.randn(len(ref))
        
        references.append(ref)
        separated_list.append(sep)
    
    # Batch evaluation
    reports = metrics.evaluate_batch(
        references=references,
        separated_list=separated_list,
        metrics=[QualityMetric.SI_SDR, QualityMetric.SPECTRAL_DIVERGENCE]
    )
    
    # Display results
    print(f"{'Case':<8} {'Scenario':<12} {'Overall':<10} {'SI-SDR':<10} {'Spectral Div':<12}")
    print("-" * 60)
    
    for i, (report, scenario) in enumerate(zip(reports, scenarios)):
        si_sdr = report.metrics.get('si_sdr', 0)
        spec_div = report.metrics.get('spectral_divergence', 0)
        print(f"{i+1:<8} {scenario:<12} {report.overall_quality:<10.3f} "
              f"{si_sdr:<10.2f} {spec_div:<12.3f}")
    
    # Statistical analysis
    stats = metrics.analyze_results(reports)
    print(f"\nStatistical Summary:")
    print(f"Mean Overall Quality: {stats['overall_quality']['mean']:.3f} ± {stats['overall_quality']['std']:.3f}")
    print(f"SI-SDR Range: {stats['min']['si_sdr']:.2f} to {stats['max']['si_sdr']:.2f} dB")
    print()


def example_multichannel_evaluation():
    """Example 4: Multi-channel audio evaluation."""
    print("=" * 60)
    print("Example 4: Multi-Channel Audio Evaluation")
    print("=" * 60)
    
    metrics = SeparationQualityMetrics(sample_rate=16000)
    
    # Generate stereo test signals
    duration = 3.0
    samples = int(duration * 16000)
    
    # Reference stereo (different content on each channel)
    ref_left = generate_test_audio(scenario="clean", duration=duration)
    ref_right = generate_test_audio(scenario="mixed", duration=duration)
    reference_stereo = np.array([ref_left, ref_right])
    
    # Separated stereo (with different degradation levels)
    sep_left = ref_left + 0.05 * np.random.randn(samples)  # Good quality
    sep_right = ref_right + 0.15 * np.random.randn(samples)  # Poor quality
    separated_stereo = np.array([sep_left, sep_right])
    
    # Multi-channel evaluation
    mc_report = metrics.evaluate_multichannel(
        reference=reference_stereo,
        separated=separated_stereo
    )
    
    print("Multi-Channel Quality Report:")
    print(f"Aggregate Quality: {mc_report.aggregate_quality:.3f}")
    if mc_report.channel_correlation is not None:
        print(f"Channel Correlation: {mc_report.channel_correlation:.3f}")
    
    print(f"\nPer-Channel Results:")
    for i, ch_report in enumerate(mc_report.channel_reports):
        channel_name = "Left" if i == 0 else "Right"
        print(f"  {channel_name} Channel:")
        print(f"    Overall Quality: {ch_report.overall_quality:.3f}")
        print(f"    SI-SDR: {ch_report.metrics['si_sdr']:.2f} dB")
    
    print()


def example_custom_metrics():
    """Example 5: Using custom metrics."""
    print("=" * 60)
    print("Example 5: Custom Metrics")
    print("=" * 60)
    
    metrics = SeparationQualityMetrics(sample_rate=16000)
    
    # Define custom metrics
    def energy_preservation(reference, separated, **kwargs):
        """Energy preservation ratio (should be close to 1.0)"""
        ref_energy = np.sum(reference ** 2)
        sep_energy = np.sum(separated ** 2)
        return sep_energy / (ref_energy + 1e-8)
    
    def frequency_balance(reference, separated, **kwargs):
        """Spectral balance preservation"""
        ref_spectrum = np.abs(np.fft.rfft(reference))
        sep_spectrum = np.abs(np.fft.rfft(separated))
        
        # Normalize
        ref_spectrum = ref_spectrum / (np.sum(ref_spectrum) + 1e-8)
        sep_spectrum = sep_spectrum / (np.sum(sep_spectrum) + 1e-8)
        
        # Calculate correlation
        correlation = np.corrcoef(ref_spectrum, sep_spectrum)[0, 1]
        return max(0, correlation)
    
    # Register custom metrics
    metrics.register_custom_metric(
        name="energy_preservation",
        func=energy_preservation,
        range=(0, 2),
        higher_is_better=False,
        optimal_value=1.0
    )
    
    metrics.register_custom_metric(
        name="frequency_balance",
        func=frequency_balance,
        range=(0, 1),
        higher_is_better=True,
        optimal_value=1.0
    )
    
    # Test with different scenarios
    reference = generate_test_audio(scenario="clean", duration=2.0)
    
    # Test cases
    test_cases = [
        ("Good separation", reference + 0.05 * np.random.randn(len(reference))),
        ("Energy loss", reference * 0.7),
        ("Spectral change", reference + 0.1 * np.sin(2 * np.pi * 1000 * np.linspace(0, 2.0, len(reference))))
    ]
    
    print("Custom Metrics Evaluation:")
    print(f"{'Case':<20} {'Energy Pres.':<12} {'Freq. Balance':<12} {'Overall':<10}")
    print("-" * 60)
    
    for case_name, separated in test_cases:
        report = metrics.evaluate_separation(
            reference=reference,
            separated=separated,
            include_custom_metrics=True
        )
        
        energy_pres = report.metrics.get('energy_preservation', 0)
        freq_bal = report.metrics.get('frequency_balance', 0)
        
        print(f"{case_name:<20} {energy_pres:<12.3f} {freq_bal:<12.3f} {report.overall_quality:<10.3f}")
    
    print()


def example_segment_analysis():
    """Example 6: Segment-wise quality analysis."""
    print("=" * 60)
    print("Example 6: Segment-wise Quality Analysis")
    print("=" * 60)
    
    metrics = SeparationQualityMetrics(sample_rate=16000)
    
    # Generate long audio with varying quality
    duration = 10.0
    samples = int(duration * 16000)
    
    # Reference signal
    reference = generate_test_audio(scenario="clean", duration=duration)
    
    # Create separated signal with time-varying quality
    separated = reference.copy()
    segment_duration = 2.0
    segment_samples = int(segment_duration * 16000)
    
    # Add increasing noise levels over time
    for i in range(0, samples, segment_samples):
        end = min(i + segment_samples, samples)
        noise_level = 0.02 + 0.03 * (i / samples)  # Increasing noise
        separated[i:end] += noise_level * np.random.randn(end - i)
    
    # Segment-wise evaluation
    segment_reports = metrics.evaluate_segments(
        reference=reference,
        separated=separated,
        segment_duration=segment_duration
    )
    
    print("Segment-wise Quality Analysis:")
    print(f"{'Segment':<8} {'Time Range':<12} {'Overall':<10} {'SI-SDR':<10}")
    print("-" * 45)
    
    qualities = []
    for i, report in enumerate(segment_reports):
        start_time = report.metadata['segment_start']
        end_time = report.metadata['segment_end']
        time_range = f"{start_time:.1f}-{end_time:.1f}s"
        si_sdr = report.metrics.get('si_sdr', 0)
        
        print(f"{i+1:<8} {time_range:<12} {report.overall_quality:<10.3f} {si_sdr:<10.2f}")
        qualities.append(report.overall_quality)
    
    print(f"\nQuality Trend:")
    print(f"  Initial quality: {qualities[0]:.3f}")
    print(f"  Final quality: {qualities[-1]:.3f}")
    print(f"  Quality degradation: {qualities[0] - qualities[-1]:.3f}")
    
    # Plot quality over time
    try:
        plt.figure(figsize=(10, 4))
        time_points = [r.metadata['segment_start'] + 1.0 for r in segment_reports]  # Mid-point
        plt.plot(time_points, qualities, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Time (s)')
        plt.ylabel('Quality Score')
        plt.title('Quality Degradation Over Time')
        plt.grid(True, alpha=0.3)
        plt.savefig('segment_quality_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Quality trend plot saved to: segment_quality_analysis.png")
    except Exception as e:
        print(f"  Could not create plot: {e}")
    
    print()


def example_real_world_scenario():
    """Example 7: Real-world scenario simulation."""
    print("=" * 60)
    print("Example 7: Real-World Scenario Simulation")
    print("=" * 60)
    
    metrics = SeparationQualityMetrics(sample_rate=16000)
    
    # Simulate real-world audio separation scenario
    duration = 5.0
    samples = int(duration * 16000)
    t = np.linspace(0, duration, samples)
    
    # Target speaker (what we want to extract)
    target_speaker = np.sin(2 * np.pi * 250 * t) * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))
    
    # Interfering speaker
    interferer = np.sin(2 * np.pi * 400 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
    
    # Background noise
    background_noise = 0.05 * np.random.randn(samples)
    
    # Original mixture
    mixture = target_speaker + 0.6 * interferer + background_noise
    
    # Simulate different separation algorithms
    separation_algorithms = {
        "Perfect": target_speaker + 0.01 * np.random.randn(samples),
        "Good": target_speaker + 0.1 * interferer + 0.02 * np.random.randn(samples),
        "Average": target_speaker + 0.3 * interferer + 0.03 * np.random.randn(samples),
        "Poor": target_speaker + 0.5 * interferer + 0.05 * np.random.randn(samples),
        "Bad": 0.7 * target_speaker + 0.7 * interferer + 0.1 * np.random.randn(samples)
    }
    
    print("Real-World Separation Algorithm Comparison:")
    print(f"{'Algorithm':<12} {'Overall':<10} {'SI-SDR':<10} {'SI-SIR':<10} {'SI-SAR':<10}")
    print("-" * 55)
    
    results = {}
    for alg_name, separated in separation_algorithms.items():
        report = metrics.evaluate_separation(
            reference=target_speaker,
            separated=separated,
            mixture=mixture
        )
        
        si_sdr = report.metrics.get('si_sdr', 0)
        si_sir = report.metrics.get('si_sir', 0)
        si_sar = report.metrics.get('si_sar', 0)
        
        print(f"{alg_name:<12} {report.overall_quality:<10.3f} {si_sdr:<10.2f} "
              f"{si_sir:<10.2f} {si_sar:<10.2f}")
        
        results[alg_name] = report.overall_quality
    
    # Find best algorithm
    best_algorithm = max(results, key=results.get)
    print(f"\nBest performing algorithm: {best_algorithm} "
          f"(Quality: {results[best_algorithm]:.3f})")
    
    print()


if __name__ == "__main__":
    print("\nSeparation Quality Metrics Examples\n")
    
    # Run all examples
    example_basic_evaluation()
    example_comprehensive_evaluation()
    example_batch_evaluation()
    example_multichannel_evaluation()
    example_custom_metrics()
    example_segment_analysis()
    example_real_world_scenario()
    
    print("All examples completed successfully!")