#!/usr/bin/env python3
"""
Demo script for Issue Categorization Engine.

Shows how to use the categorization system to analyze audio quality issues.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.issue_categorization import IssueCategorizer
from utils.audio_metrics import calculate_snr
import librosa


def create_test_audio_scenarios():
    """Create different audio scenarios for testing."""
    sample_rate = 16000
    duration = 3.0
    samples = int(sample_rate * duration)
    
    scenarios = {}
    
    # 1. Clean audio
    clean = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
    scenarios['clean'] = {
        'audio': clean,
        'description': 'Clean 440Hz sine wave'
    }
    
    # 2. Noisy audio
    noise = 0.3 * np.random.randn(samples)
    noisy = clean + noise
    scenarios['noisy'] = {
        'audio': noisy,
        'description': 'Sine wave with background noise'
    }
    
    # 3. Clipped audio
    clipped = clean.copy()
    clipped[clipped > 0.8] = 0.8
    clipped[clipped < -0.8] = -0.8
    scenarios['clipped'] = {
        'audio': clipped,
        'description': 'Clipped sine wave'
    }
    
    # 4. Audio with silence
    silent = clean.copy()
    silent[samples//3:2*samples//3] = 0
    scenarios['silent'] = {
        'audio': silent,
        'description': 'Audio with excessive silence'
    }
    
    # 5. Multiple issues
    multi_issue = noisy.copy()
    multi_issue[multi_issue > 0.7] = 0.7  # Clipping
    multi_issue[samples//2:] *= 0.1  # Reduced volume
    scenarios['multi_issue'] = {
        'audio': multi_issue,
        'description': 'Audio with multiple issues'
    }
    
    return scenarios


def analyze_audio(audio, sample_rate, description):
    """Analyze audio and categorize issues."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {description}")
    print('='*60)
    
    # Calculate basic metrics
    metrics = {}
    
    # Energy and SNR
    signal_power = np.mean(audio ** 2)
    noise_estimate = np.percentile(np.abs(audio), 10)
    if noise_estimate > 0:
        metrics['snr'] = 10 * np.log10(signal_power / (noise_estimate ** 2))
    else:
        metrics['snr'] = 40.0
    
    # Voice activity
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    frame_energy = np.sum(frames ** 2, axis=0)
    energy_threshold = np.percentile(frame_energy, 30)
    voice_frames = frame_energy > energy_threshold
    
    metrics['voice_activity_ratio'] = np.sum(voice_frames) / len(voice_frames)
    metrics['silence_ratio'] = 1 - metrics['voice_activity_ratio']
    metrics['speech_duration'] = len(audio) / sample_rate * metrics['voice_activity_ratio']
    
    # Spectral metrics
    stft = np.abs(librosa.stft(audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
    metrics['spectral_centroid'] = spectral_centroid
    
    # Clipping detection
    clipping_threshold = 0.95 * np.max(np.abs(audio))
    clipped_samples = np.sum(np.abs(audio) >= clipping_threshold)
    clip_ratio = clipped_samples / len(audio)
    
    # Pattern detection
    patterns = []
    if clip_ratio > 0.001:  # More than 0.1% samples clipped
        clip_locations = np.where(np.abs(audio) >= clipping_threshold)[0] / sample_rate
        patterns.append({
            'type': 'clipping',
            'severity': min(1.0, clip_ratio * 100),
            'locations': clip_locations[:10].tolist()  # First 10 locations
        })
    
    # Print metrics
    print("\nMetrics:")
    print(f"  SNR: {metrics['snr']:.1f} dB")
    print(f"  Voice Activity: {metrics['voice_activity_ratio']:.1%}")
    print(f"  Silence Ratio: {metrics['silence_ratio']:.1%}")
    print(f"  Speech Duration: {metrics['speech_duration']:.1f}s")
    print(f"  Clipping Ratio: {clip_ratio:.2%}")
    
    return metrics, patterns


def main():
    """Main demo function."""
    print("Issue Categorization Engine Demo")
    print("================================\n")
    
    # Initialize categorizer
    categorizer = IssueCategorizer()
    print("✓ Categorizer initialized")
    
    # Create test scenarios
    scenarios = create_test_audio_scenarios()
    print(f"✓ Created {len(scenarios)} test scenarios")
    
    # Analyze each scenario
    for name, scenario in scenarios.items():
        audio = scenario['audio']
        description = scenario['description']
        
        # Analyze audio
        metrics, patterns = analyze_audio(audio, 16000, description)
        
        # Categorize issues
        report = categorizer.categorize_with_explanation(patterns, metrics)
        
        # Display results
        print(f"\nCategorization Report:")
        print(f"  Summary: {report.summary}")
        
        if report.categories:
            print(f"\n  Detected Issues ({len(report.categories)}):")
            for i, category in enumerate(report.categories, 1):
                print(f"    {i}. {category.name.replace('_', ' ').title()}")
                print(f"       - Confidence: {category.confidence:.0%}")
                print(f"       - Severity: {category.severity.level} ({category.severity.score:.2f})")
                print(f"       - Priority: {category.priority_score:.2f}")
                
                # Show explanation
                if category.name in report.explanations:
                    explanation = report.explanations[category.name]
                    print(f"       - Explanation: {explanation['explanation']}")
        
        if report.recommendations:
            print(f"\n  Top Recommendations:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"    {i}. {rec.method}")
                print(f"       - Effectiveness: {rec.effectiveness:.0%}")
                print(f"       - Complexity: {rec.complexity}")
                if rec.parameters:
                    print(f"       - Parameters: {rec.parameters}")
    
    # Test real-time mode
    print(f"\n{'='*60}")
    print("Real-time Mode Demo")
    print('='*60)
    
    rt_categorizer = IssueCategorizer(real_time_mode=True)
    
    # Simulate streaming chunks
    chunk_size = 16000  # 1 second chunks
    audio = scenarios['noisy']['audio']
    
    print("\nProcessing audio in 1-second chunks...")
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
            
        report = rt_categorizer.process_chunk(chunk)
        print(f"  Chunk {i//chunk_size + 1}: {len(report.categories)} issues detected")
    
    # Test batch mode
    print(f"\n{'='*60}")
    print("Batch Mode Demo")
    print('='*60)
    
    batch_categorizer = IssueCategorizer(batch_mode=True)
    
    # Create batch
    batch_data = []
    for name, scenario in list(scenarios.items())[:3]:
        audio = scenario['audio']
        metrics, patterns = analyze_audio(audio, 16000, f"Batch: {scenario['description']}")
        batch_data.append((metrics, patterns))
    
    # Process batch
    print(f"\nProcessing batch of {len(batch_data)} samples...")
    results = batch_categorizer.categorize_batch(batch_data)
    
    for i, (result, (name, _)) in enumerate(zip(results, list(scenarios.items())[:3])):
        print(f"\n  Sample {i+1} ({name}): {len(result.categories)} issues")
        print(f"    Summary: {result.summary}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == '__main__':
    main()