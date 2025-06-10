#!/usr/bin/env python3
"""
Example script demonstrating Enhancement Scoring System usage
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.scoring import EnhancementScorer


def demonstrate_basic_scoring():
    """Demonstrate basic scoring functionality"""
    print("=== Basic Enhancement Scoring Demo ===\n")
    
    # Initialize scorer with balanced profile
    scorer = EnhancementScorer(profile='balanced')
    
    # Example metrics from an enhanced audio
    metrics = {
        'pesq': 3.5,        # Good perceptual quality
        'stoi': 0.85,       # High intelligibility
        'si_sdr': 15.2,     # Good source separation
        'snr': 25.0,        # Good signal-to-noise ratio
        'naturalness': 0.8  # Natural sounding
    }
    
    # Calculate score
    score = scorer.calculate_score(metrics)
    
    print(f"Enhancement Score: {score.value:.2f}/100")
    print(f"Grade: {score.grade}")
    print(f"Category: {score.category}")
    print(f"Profile Used: {score.profile}")
    
    # Get detailed explanation
    explanation = scorer.explain_score(score)
    print(f"\nOverall Assessment: {explanation['overall_assessment']}")
    
    print("\nMetric Contributions:")
    for metric, contrib in explanation['metric_contributions'].items():
        print(f"  {metric}: {contrib['value']:.2f} → {contrib['contribution']:.1f} points")
    
    print("\nStrengths:")
    for strength in explanation['strengths']:
        print(f"  - {strength}")
    
    print("\nRecommendations:")
    for rec in explanation['recommendations']:
        print(f"  - {rec}")


def demonstrate_profile_comparison():
    """Compare different scoring profiles"""
    print("\n\n=== Scoring Profile Comparison ===\n")
    
    # Test metrics with high intelligibility but lower quality
    metrics = {
        'pesq': 2.8,        # Fair quality
        'stoi': 0.92,       # Excellent intelligibility
        'si_sdr': 12.0,     # Moderate separation
        'snr': 22.0,        # Good SNR
        'naturalness': 0.7  # Fair naturalness
    }
    
    profiles = ['balanced', 'intelligibility', 'quality', 'naturalness']
    
    print("Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    print("\nScores by Profile:")
    for profile in profiles:
        scorer = EnhancementScorer(profile=profile)
        score = scorer.calculate_score(metrics)
        print(f"  {profile:15s}: {score.value:5.1f} (Grade: {score.grade})")


def demonstrate_batch_ranking():
    """Demonstrate ranking multiple samples"""
    print("\n\n=== Batch Sample Ranking ===\n")
    
    # Generate batch of samples with varied quality
    np.random.seed(42)
    batch_samples = []
    
    for i in range(10):
        sample = {
            'id': f'sample_{i+1:02d}',
            'metrics': {
                'pesq': np.random.uniform(2.0, 4.2),
                'stoi': np.random.uniform(0.6, 0.95),
                'si_sdr': np.random.uniform(5.0, 20.0),
                'snr': np.random.uniform(15.0, 35.0),
                'naturalness': np.random.uniform(0.5, 0.9)
            }
        }
        batch_samples.append(sample)
    
    # Rank samples
    scorer = EnhancementScorer(profile='balanced')
    ranked_results = scorer.rank_samples(batch_samples)
    
    print("Top 5 Samples:")
    print("Rank | Sample ID | Score | Grade | Key Metrics")
    print("-" * 60)
    
    for result in ranked_results[:5]:
        metrics = result.metrics
        print(f" {result.rank:2d}  | {result.id:9s} | {result.score:5.1f} | {scorer._calculate_grade(result.score):5s} | "
              f"PESQ:{metrics['pesq']:.1f} STOI:{metrics['stoi']:.2f}")


def demonstrate_custom_weights():
    """Demonstrate custom weight configuration"""
    print("\n\n=== Custom Weight Configuration ===\n")
    
    # Create custom weights emphasizing PESQ and STOI
    custom_weights = {
        'pesq': 0.35,       # 35% weight on perceptual quality
        'stoi': 0.30,       # 30% weight on intelligibility
        'si_sdr': 0.15,     # 15% weight on separation
        'snr': 0.10,        # 10% weight on SNR
        'naturalness': 0.10 # 10% weight on naturalness
    }
    
    scorer = EnhancementScorer(profile='custom', custom_weights=custom_weights)
    
    # Test with different metric combinations
    test_cases = [
        {
            'name': 'High PESQ/STOI',
            'metrics': {'pesq': 4.0, 'stoi': 0.9, 'si_sdr': 10.0, 'snr': 20.0, 'naturalness': 0.7}
        },
        {
            'name': 'Low PESQ/STOI',
            'metrics': {'pesq': 2.5, 'stoi': 0.7, 'si_sdr': 20.0, 'snr': 30.0, 'naturalness': 0.9}
        }
    ]
    
    print(f"Custom Weights: {custom_weights}")
    print("\nComparison:")
    
    for case in test_cases:
        score = scorer.calculate_score(case['metrics'])
        print(f"\n{case['name']}:")
        print(f"  Score: {score.value:.1f}")
        print(f"  Grade: {score.grade}")


def demonstrate_percentile_ranking():
    """Demonstrate percentile ranking with historical data"""
    print("\n\n=== Percentile Ranking ===\n")
    
    scorer = EnhancementScorer()
    
    # Simulate historical scores
    print("Building score history...")
    np.random.seed(42)
    for _ in range(1000):
        metrics = {
            'pesq': np.random.normal(3.0, 0.6),
            'stoi': np.random.normal(0.8, 0.1),
            'si_sdr': np.random.normal(12.0, 5.0),
            'snr': np.random.normal(22.0, 5.0),
            'naturalness': np.random.normal(0.75, 0.15)
        }
        scorer.calculate_score(metrics)
    
    # Test specific scores
    test_scores = [30, 50, 70, 85, 95]
    
    print("\nPercentile Rankings:")
    print("Score | Percentile | Interpretation")
    print("-" * 50)
    
    for test_score in test_scores:
        percentile = scorer.get_percentile(test_score)
        
        if percentile >= 90:
            interpretation = "Top 10% - Excellent"
        elif percentile >= 75:
            interpretation = "Top 25% - Very Good"
        elif percentile >= 50:
            interpretation = "Above Average"
        elif percentile >= 25:
            interpretation = "Below Average"
        else:
            interpretation = "Bottom 25% - Needs Improvement"
        
        print(f" {test_score:3d}  | {percentile:6.1f}%    | {interpretation}")
    
    # Show historical statistics
    stats = scorer.get_historical_stats()
    print(f"\nHistorical Statistics (n={stats['count']}):")
    print(f"  Mean Score: {stats['mean']:.1f}")
    print(f"  Std Dev: {stats['std']:.1f}")
    print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")


def main():
    """Run all demonstrations"""
    print("Audio Enhancement Scoring System Demo")
    print("=" * 50)
    
    demonstrate_basic_scoring()
    demonstrate_profile_comparison()
    demonstrate_batch_ranking()
    demonstrate_custom_weights()
    demonstrate_percentile_ranking()
    
    print("\n\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()