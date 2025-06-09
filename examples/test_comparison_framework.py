"""
Example script demonstrating the Comparison Framework usage.

This script shows how to use the comparison framework to evaluate
different audio enhancement methods.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.evaluation.comparison_framework import (
    ComparisonFramework,
    ComparisonConfig
)


def simulate_enhancement_methods(reference_audio, noise_level=0.1):
    """Simulate different enhancement methods for demonstration."""
    
    # Method 1: Simple denoising (reduce noise by 50%)
    noise = np.random.normal(0, noise_level, len(reference_audio))
    noisy_audio = reference_audio + noise
    denoised_v1 = reference_audio + noise * 0.5
    
    # Method 2: Aggressive denoising (reduce noise by 80%)
    denoised_v2 = reference_audio + noise * 0.2
    
    # Method 3: Over-processed (reduce noise but also signal)
    overprocessed = reference_audio * 0.8 + noise * 0.1
    
    return {
        'noisy': noisy_audio,
        'denoised_v1': denoised_v1,
        'denoised_v2': denoised_v2,
        'overprocessed': overprocessed
    }


def main():
    """Run comparison framework example."""
    
    # Generate test audio
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a reference signal (combination of tones)
    reference = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
        0.1 * np.sin(2 * np.pi * 659.25 * t)  # E5
    )
    
    # Simulate enhancement methods
    enhanced_versions = simulate_enhancement_methods(reference, noise_level=0.15)
    
    # Initialize comparison framework
    config = ComparisonConfig(
        metrics=['snr'],  # Only use SNR since we don't have actual PESQ/STOI calculators
        statistical_tests=True,
        confidence_level=0.95,
        multiple_comparison_correction='bonferroni'
    )
    
    framework = ComparisonFramework(config=config)
    
    # Add methods for comparison
    for name, audio in enhanced_versions.items():
        framework.add_method(
            name=name,
            audio_data=audio,
            reference=reference,
            sample_rate=sample_rate
        )
    
    # Perform comparison
    print("Performing comparison...")
    results = framework.compare(statistical_tests=True)
    
    # Display results
    print("\n=== Comparison Results ===")
    for method, method_results in results.items():
        if method != 'statistical_analysis':
            print(f"\nMethod: {method}")
            print(f"  SNR: {method_results['metrics']['snr']:.2f} dB")
            print(f"  Rank: {method_results['rank']}")
            print(f"  Overall Score: {method_results['overall_score']:.2f}")
    
    # Show statistical analysis
    if 'statistical_analysis' in results:
        print("\n=== Statistical Analysis ===")
        stats = results['statistical_analysis']
        
        if 'pairwise_comparisons' in stats:
            print("\nPairwise Comparisons:")
            for comparison, result in stats['pairwise_comparisons'].items():
                print(f"  {comparison}:")
                print(f"    p-value: {result['p_value']:.4f}")
                print(f"    Significant: {result['significant']}")
                print(f"    Effect size: {result['effect_size']} (d={result['cohens_d']:.3f})")
    
    # Get best method
    best_method = framework.get_best_method()
    print(f"\nBest method: {best_method['name']} (score: {best_method['score']:.2f})")
    
    # Generate reports
    print("\nGenerating reports...")
    framework.generate_report('comparison_report.html', format='html')
    framework.generate_report('comparison_report.json', format='json')
    print("Reports generated: comparison_report.html, comparison_report.json")
    
    # Export results for later use
    framework.export_results('comparison_results_export.json')
    print("Results exported to: comparison_results_export.json")


if __name__ == '__main__':
    main()