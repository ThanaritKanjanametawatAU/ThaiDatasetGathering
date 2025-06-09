"""
Example script demonstrating the Quality Threshold Manager.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from processors.audio_enhancement.quality import QualityThresholdManager
import numpy as np


def main():
    """Demonstrate Quality Threshold Manager usage."""
    
    # Initialize with production profile
    print("=== Quality Threshold Manager Demo ===\n")
    
    manager = QualityThresholdManager(profile='production')
    
    # Example 1: Check quality of good audio
    print("1. Checking high-quality audio:")
    good_metrics = {
        'pesq': 4.2,
        'stoi': 0.95,
        'si_sdr': 18.5,
        'snr': 32.0
    }
    
    result = manager.check_quality(good_metrics)
    print(f"   Passed: {result.passed}")
    print(f"   Score: {result.score:.2f}")
    print(f"   Severity: {result.severity}")
    print()
    
    # Example 2: Check quality of poor audio
    print("2. Checking low-quality audio:")
    poor_metrics = {
        'pesq': 2.5,
        'stoi': 0.72,
        'si_sdr': 8.0,
        'snr': 15.0
    }
    
    result = manager.check_quality(poor_metrics)
    print(f"   Passed: {result.passed}")
    print(f"   Score: {result.score:.2f}")
    print(f"   Severity: {result.severity}")
    print(f"   Failures: {list(result.failures.keys())}")
    print(f"   Recovery suggestions:")
    for suggestion in result.recovery_suggestions:
        print(f"     - {suggestion}")
    print()
    
    # Example 3: Using weighted metrics
    print("3. Using weighted quality assessment:")
    weights = {
        'pesq': 0.4,  # Most important
        'stoi': 0.3,  # Second most important
        'si_sdr': 0.2,
        'snr': 0.1    # Least important
    }
    
    weighted_manager = QualityThresholdManager(
        profile='production',
        metric_weights=weights
    )
    
    mixed_metrics = {
        'pesq': 3.8,  # Good
        'stoi': 0.65,  # Poor
        'si_sdr': 14.0,  # OK
        'snr': 28.0   # Good
    }
    
    result = weighted_manager.check_quality(mixed_metrics)
    print(f"   Weighted score: {result.score:.2f}")
    print(f"   Individual scores: {result.metric_scores}")
    print()
    
    # Example 4: Dynamic threshold adjustment
    print("4. Dynamic threshold adjustment:")
    
    # Simulate historical data
    historical_data = []
    for i in range(50):
        # Generate synthetic historical metrics
        quality = np.random.normal(0.8, 0.1)
        historical_data.append({
            'pesq': 2.5 + quality * 2.0,
            'stoi': 0.6 + quality * 0.35,
            'si_sdr': 8.0 + quality * 15.0,
            'snr': 15.0 + quality * 20.0
        })
    
    print(f"   Original PESQ threshold: {manager.thresholds['pesq']}")
    manager.update_thresholds(historical_data)
    print(f"   Updated PESQ threshold: {manager.thresholds['pesq']}")
    print()
    
    # Example 5: Multi-stage quality gates
    print("5. Multi-stage quality gates:")
    
    # Create manager with stage-specific thresholds
    stage_config = {
        'production': {
            'pre_enhancement': {
                'snr': {'min': 10.0, 'target': 15.0, 'max': 40.0}
            },
            'post_enhancement': {
                'pesq': {'min': 3.0, 'target': 3.5, 'max': 5.0},
                'stoi': {'min': 0.8, 'target': 0.9, 'max': 1.0},
                'si_sdr': {'min': 10.0, 'target': 15.0, 'max': 30.0},
                'snr': {'min': 20.0, 'target': 25.0, 'max': 40.0}
            }
        }
    }
    
    stage_manager = QualityThresholdManager(profile='production', config=stage_config)
    
    # Check pre-enhancement
    pre_gate = stage_manager.get_quality_gate('pre_enhancement')
    pre_result = pre_gate.check({'snr': 12.0})
    print(f"   Pre-enhancement gate passed: {pre_result.passed}")
    
    # Check post-enhancement
    post_gate = stage_manager.get_quality_gate('post_enhancement')
    post_result = post_gate.check(good_metrics)
    print(f"   Post-enhancement gate passed: {post_result.passed}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()