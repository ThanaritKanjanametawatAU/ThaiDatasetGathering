#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) Demonstration

This script demonstrates the Voice Activity Detection module with various
detection methods and real-time processing capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from processors.audio_enhancement.detection.voice_activity_detector import (
    VoiceActivityDetector, StreamingVAD, VADMethod
)


def create_test_signal(duration=3.0, sample_rate=16000):
    """Create test signal with speech and silence segments"""
    samples = int(duration * sample_rate)
    signal = np.zeros(samples)
    
    # Add speech segments
    speech_segments = [
        (0.5, 1.0),   # First speech segment
        (1.5, 2.0),   # Second speech segment
        (2.3, 2.8),   # Third speech segment
    ]
    
    for start, end in speech_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        
        # Simulate speech with filtered noise
        speech = np.random.randn(end_idx - start_idx)
        # Add some harmonic content
        t = np.arange(end_idx - start_idx) / sample_rate
        speech += 0.5 * np.sin(2 * np.pi * 200 * t)  # Fundamental
        speech += 0.3 * np.sin(2 * np.pi * 400 * t)  # Harmonic
        
        signal[start_idx:end_idx] = speech * 0.3
    
    # Add background noise
    signal += np.random.randn(samples) * 0.01
    
    return signal, speech_segments


def demonstrate_vad_methods():
    """Demonstrate different VAD methods"""
    print("Voice Activity Detection Demonstration")
    print("=" * 50)
    
    # Create test signal
    signal, true_segments = create_test_signal()
    sample_rate = 16000
    
    # Test different methods
    methods = ['energy', 'spectral', 'neural', 'hybrid']
    
    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(12, 10))
    
    # Plot original signal
    time = np.arange(len(signal)) / sample_rate
    axes[0].plot(time, signal)
    axes[0].set_title('Original Signal with True Speech Segments')
    axes[0].set_ylabel('Amplitude')
    
    # Highlight true speech segments
    for start, end in true_segments:
        axes[0].axvspan(start, end, alpha=0.3, color='green', label='Speech')
    axes[0].legend()
    
    # Test each method
    for i, method in enumerate(methods):
        print(f"\nTesting {method.upper()} VAD:")
        
        detector = VoiceActivityDetector(method=method, sample_rate=sample_rate)
        result = detector.detect(signal)
        
        print(f"  Speech ratio: {result.speech_ratio:.2%}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Number of frames: {result.num_frames}")
        print(f"  Speech frames: {len(result.speech_frames)}")
        
        # Plot results
        ax = axes[i + 1]
        ax.plot(time, signal, alpha=0.5)
        
        # Convert frame decisions to time
        frame_time = np.arange(len(result.frame_decisions)) * result.frame_duration
        
        # Plot VAD decisions
        for j, is_speech in enumerate(result.frame_decisions):
            if is_speech and j < len(frame_time):
                ax.axvspan(frame_time[j], 
                          min(frame_time[j] + result.frame_duration, time[-1]),
                          alpha=0.3, color='red')
        
        ax.set_title(f'{method.upper()} VAD (Speech ratio: {result.speech_ratio:.2%})')
        ax.set_ylabel('Amplitude')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('vad_comparison.png', dpi=150)
    print("\nSaved VAD comparison plot to 'vad_comparison.png'")
    

def demonstrate_streaming_vad():
    """Demonstrate streaming VAD processing"""
    print("\n\nStreaming VAD Demonstration")
    print("=" * 50)
    
    # Create test signal
    signal, _ = create_test_signal(duration=5.0)
    sample_rate = 16000
    
    # Create streaming VAD
    streaming_vad = StreamingVAD(
        method='energy',
        sample_rate=sample_rate,
        chunk_duration=0.032  # 32ms chunks
    )
    
    # Process in chunks
    chunk_size = int(0.032 * sample_rate)
    results = []
    
    print(f"Processing {len(signal) // chunk_size} chunks...")
    
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        if len(chunk) == chunk_size:
            result = streaming_vad.process_chunk(chunk)
            results.append(result)
            
            if result['is_speech']:
                print(f"  Chunk {len(results)}: SPEECH detected "
                      f"(prob: {result['probability']:.3f}, "
                      f"time: {result['timestamp']:.3f}s)")
    
    # Calculate overall statistics
    speech_chunks = sum(1 for r in results if r['is_speech'])
    print(f"\nStreaming results:")
    print(f"  Total chunks: {len(results)}")
    print(f"  Speech chunks: {speech_chunks}")
    print(f"  Speech ratio: {speech_chunks/len(results):.2%}")
    

def demonstrate_speech_segments():
    """Demonstrate speech segment extraction"""
    print("\n\nSpeech Segment Extraction")
    print("=" * 50)
    
    # Create test signal
    signal, true_segments = create_test_signal()
    sample_rate = 16000
    
    # Detect speech segments
    detector = VoiceActivityDetector(method='hybrid', sample_rate=sample_rate)
    segments = detector.get_speech_segments(signal, min_speech_duration=0.1)
    
    print(f"Found {len(segments)} speech segments:")
    for i, (start, end) in enumerate(segments):
        duration = (end - start) / sample_rate
        print(f"  Segment {i+1}: {start/sample_rate:.3f}s - {end/sample_rate:.3f}s "
              f"(duration: {duration:.3f}s)")
    
    print(f"\nTrue segments were:")
    for i, (start, end) in enumerate(true_segments):
        print(f"  Segment {i+1}: {start:.3f}s - {end:.3f}s")


def demonstrate_aggressiveness_levels():
    """Demonstrate different aggressiveness levels"""
    print("\n\nAggressiveness Level Comparison")
    print("=" * 50)
    
    # Create noisy signal
    signal, _ = create_test_signal()
    signal += np.random.randn(len(signal)) * 0.05  # Add more noise
    sample_rate = 16000
    
    aggressiveness_levels = [0, 1, 2, 3]
    
    for level in aggressiveness_levels:
        detector = VoiceActivityDetector(
            method='energy', 
            sample_rate=sample_rate,
            aggressiveness=level
        )
        result = detector.detect(signal)
        
        print(f"Aggressiveness {level}: Speech ratio = {result.speech_ratio:.2%}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_vad_methods()
    demonstrate_streaming_vad()
    demonstrate_speech_segments()
    demonstrate_aggressiveness_levels()
    
    print("\n\nVAD demonstration completed!")