#!/usr/bin/env python3
"""Example usage of the Enhanced SNR Calculator module (S01_T03).

This example demonstrates how to use the Enhanced SNR Calculator for:
1. Basic SNR calculation
2. Detailed analysis with VAD segments
3. Different VAD backends
4. Batch processing
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.enhanced_snr_calculator import EnhancedSNRCalculator, calculate_snr


def example_basic_usage():
    """Example 1: Basic SNR calculation."""
    print("=" * 60)
    print("Example 1: Basic SNR Calculation")
    print("=" * 60)
    
    # Generate a simple test signal
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Signal: 440 Hz tone
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Add noise
    noise = 0.1 * np.random.randn(len(signal))
    mixed = signal + noise
    
    # Calculate SNR using convenience function
    snr_db = calculate_snr(mixed, sample_rate)
    print(f"SNR: {snr_db:.2f} dB")
    
    print()


def example_detailed_analysis():
    """Example 2: Detailed SNR analysis with VAD segments."""
    print("=" * 60)
    print("Example 2: Detailed Analysis with VAD")
    print("=" * 60)
    
    # Create calculator instance
    calculator = EnhancedSNRCalculator(vad_backend='energy')
    
    # Generate test signal with speech and silence
    sample_rate = 16000
    duration = 4.0
    samples = int(duration * sample_rate)
    
    # Create signal with silence-speech-silence pattern
    audio = np.zeros(samples)
    
    # Add speech in the middle 2 seconds
    speech_start = int(1.0 * sample_rate)
    speech_end = int(3.0 * sample_rate)
    t_speech = np.linspace(0, 2.0, speech_end - speech_start)
    
    # Modulated tone to simulate speech
    speech = np.sin(2 * np.pi * 200 * t_speech) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t_speech))
    audio[speech_start:speech_end] = speech
    
    # Add background noise
    noise = 0.05 * np.random.randn(samples)
    audio += noise
    
    # Calculate SNR with detailed results
    result = calculator.calculate_snr(audio, sample_rate)
    
    print(f"SNR: {result['snr_db']:.2f} dB")
    print(f"Signal Power: {result['signal_power']:.6f}")
    print(f"Noise Power: {result['noise_power']:.6f}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nVAD Segments:")
    for i, (start, end) in enumerate(result['vad_segments']):
        print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    print()


def example_vad_backends():
    """Example 3: Compare different VAD backends."""
    print("=" * 60)
    print("Example 3: Different VAD Backends")
    print("=" * 60)
    
    # Generate test audio
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Complex signal
    signal = np.sin(2 * np.pi * 300 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    noise = 0.1 * np.random.randn(len(signal))
    audio = signal + noise
    
    # Test different backends
    backends = ['energy', 'silero', 'pyannote']
    
    for backend in backends:
        try:
            calculator = EnhancedSNRCalculator(vad_backend=backend)
            result = calculator.calculate_snr(audio, sample_rate)
            print(f"{backend:8s} backend: SNR = {result['snr_db']:6.2f} dB, "
                  f"Segments = {len(result['vad_segments'])}, "
                  f"Confidence = {result['confidence']:.2f}")
        except Exception as e:
            print(f"{backend:8s} backend: Not available ({str(e)[:50]}...)")
    
    print()


def example_batch_processing():
    """Example 4: Batch processing of multiple audio files."""
    print("=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)
    
    # Create calculator once for efficiency
    calculator = EnhancedSNRCalculator()
    
    # Simulate multiple audio files with different characteristics
    test_cases = [
        {"name": "Clean speech", "snr_target": 40},
        {"name": "Moderate noise", "snr_target": 20},
        {"name": "Heavy noise", "snr_target": 5},
        {"name": "Very noisy", "snr_target": -5},
    ]
    
    sample_rate = 16000
    duration = 1.0
    
    results = []
    for case in test_cases:
        # Generate synthetic audio with target SNR
        t = np.linspace(0, duration, int(duration * sample_rate))
        signal = np.sin(2 * np.pi * 440 * t)
        
        # Calculate noise level for target SNR
        signal_power = np.mean(signal ** 2)
        target_snr_linear = 10 ** (case["snr_target"] / 10)
        noise_power = signal_power / target_snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        
        audio = signal + noise
        
        # Calculate SNR
        result = calculator.calculate_snr(audio, sample_rate)
        results.append({
            "name": case["name"],
            "target_snr": case["snr_target"],
            "measured_snr": result["snr_db"],
            "confidence": result["confidence"]
        })
    
    # Display results
    print(f"{'Audio Type':<20} {'Target SNR':>10} {'Measured SNR':>12} {'Confidence':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<20} {r['target_snr']:>10.1f} {r['measured_snr']:>12.2f} {r['confidence']:>10.2f}")
    
    print()


def example_real_world_audio():
    """Example 5: Real-world audio simulation."""
    print("=" * 60)
    print("Example 5: Real-World Audio Simulation")
    print("=" * 60)
    
    calculator = EnhancedSNRCalculator()
    sample_rate = 16000
    duration = 5.0
    
    # Simulate real-world audio with multiple speakers and background noise
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Primary speaker (intermittent)
    audio = np.zeros_like(t)
    
    # Add speech segments
    speech_segments = [
        (0.5, 1.5, 300),   # First utterance
        (2.0, 3.0, 350),   # Second utterance
        (3.5, 4.5, 280),   # Third utterance
    ]
    
    for start, end, freq in speech_segments:
        mask = (t >= start) & (t <= end)
        t_segment = t[mask] - start
        # Modulated tone to simulate speech
        speech = np.sin(2 * np.pi * freq * t_segment) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t_segment))
        # Add envelope
        envelope = np.sin(np.pi * t_segment / (end - start))
        audio[mask] += speech * envelope
    
    # Add background noise (traffic, air conditioning, etc.)
    # Low-frequency rumble
    rumble = 0.05 * np.sin(2 * np.pi * 50 * t + np.random.rand() * 2 * np.pi)
    # Mid-frequency noise
    background = 0.03 * np.random.randn(len(t))
    # High-frequency hiss
    hiss = 0.02 * np.random.randn(len(t))
    hiss = np.convolve(hiss, np.ones(10)/10, mode='same')  # Smooth it
    
    # Combine
    audio += rumble + background + hiss
    
    # Calculate SNR
    result = calculator.calculate_snr(audio, sample_rate)
    
    print(f"Real-world audio analysis:")
    print(f"  SNR: {result['snr_db']:.2f} dB")
    print(f"  Speech segments detected: {len(result['vad_segments'])}")
    print(f"  Confidence: {result['confidence']:.2f}")
    
    # Calculate speech percentage
    if result['vad_segments']:
        speech_duration = sum(end - start for start, end in result['vad_segments'])
        speech_percentage = (speech_duration / duration) * 100
        print(f"  Speech percentage: {speech_percentage:.1f}%")
    
    print()


if __name__ == "__main__":
    print("\nEnhanced SNR Calculator Examples\n")
    
    # Run all examples
    example_basic_usage()
    example_detailed_analysis()
    example_vad_backends()
    example_batch_processing()
    example_real_world_audio()
    
    print("All examples completed successfully!")