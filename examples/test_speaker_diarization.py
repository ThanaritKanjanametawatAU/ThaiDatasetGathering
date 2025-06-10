#!/usr/bin/env python3
"""
Speaker Diarization Demonstration

This script demonstrates the speaker diarization module that segments audio
into speaker turns and identifies different speakers in a conversation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from processors.audio_enhancement.detection.speaker_diarization import (
    SpeakerDiarization, StreamingDiarizer, ClusteringMethod
)


def create_conversation(duration=20.0, sample_rate=16000):
    """Create synthetic conversation with multiple speakers"""
    samples = int(duration * sample_rate)
    t = np.arange(samples) / sample_rate
    
    # Create 3 distinct speakers with different characteristics
    speakers = []
    
    # Speaker 1 - Low pitched male voice
    speaker1 = np.zeros(samples)
    for f in [85, 170, 255]:  # Fundamental and harmonics
        speaker1 += (0.3 / (f/85)) * np.sin(2 * np.pi * f * t)
    speaker1 += np.random.randn(samples) * 0.02
    speakers.append(speaker1)
    
    # Speaker 2 - Female voice
    speaker2 = np.zeros(samples)
    for f in [165, 330, 495]:
        speaker2 += (0.3 / (f/165)) * np.sin(2 * np.pi * f * t + 0.5)
    speaker2 += np.random.randn(samples) * 0.02
    speakers.append(speaker2)
    
    # Speaker 3 - Higher pitched voice
    speaker3 = np.zeros(samples)
    for f in [220, 440, 660]:
        speaker3 += (0.3 / (f/220)) * np.sin(2 * np.pi * f * t + 1.0)
    speaker3 += np.random.randn(samples) * 0.02
    speakers.append(speaker3)
    
    # Create conversation timeline
    conversation = np.zeros(samples)
    speaker_timeline = []
    
    # Define speaking turns
    turns = [
        (0, 3, 0),      # Speaker 1: 0-3s
        (2.5, 5, 1),    # Speaker 2: 2.5-5s (overlap)
        (5, 8, 0),      # Speaker 1: 5-8s
        (8, 10, 2),     # Speaker 3: 8-10s
        (10, 12, 1),    # Speaker 2: 10-12s
        (12, 15, 0),    # Speaker 1: 12-15s
        (14, 17, 2),    # Speaker 3: 14-17s (overlap)
        (17, 20, 1),    # Speaker 2: 17-20s
    ]
    
    for start, end, speaker_id in turns:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        
        # Apply envelope for natural speech
        segment_len = end_idx - start_idx
        envelope = np.hanning(segment_len)
        
        # Add to conversation
        conversation[start_idx:end_idx] += speakers[speaker_id][start_idx:end_idx] * envelope
        speaker_timeline.append((start, end, speaker_id))
    
    # Normalize
    conversation = conversation / np.max(np.abs(conversation)) * 0.5
    
    return conversation, speaker_timeline


def visualize_diarization(audio, diarization_result, ground_truth=None, 
                         sample_rate=16000, title="Speaker Diarization"):
    """Visualize diarization results"""
    duration = len(audio) / sample_rate
    time = np.linspace(0, duration, len(audio))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot audio waveform
    ax1 = axes[0]
    ax1.plot(time, audio, 'b-', linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{title} - Audio Waveform')
    ax1.grid(True, alpha=0.3)
    
    # Plot ground truth if available
    if ground_truth:
        ax2 = axes[1]
        ax2.set_ylabel('Speaker')
        ax2.set_title('Ground Truth')
        
        for start, end, speaker_id in ground_truth:
            ax2.axhspan(speaker_id - 0.4, speaker_id + 0.4, 
                       xmin=start/duration, xmax=end/duration,
                       alpha=0.7, color=f'C{speaker_id}')
        
        ax2.set_ylim(-0.5, 2.5)
        ax2.set_yticks([0, 1, 2])
        ax2.grid(True, alpha=0.3)
    
    # Plot diarization results
    ax3 = axes[2]
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speaker')
    ax3.set_title(f'Diarization Results (Detected {diarization_result.num_speakers} speakers)')
    
    for segment in diarization_result.segments:
        ax3.axhspan(segment.speaker_id - 0.4, segment.speaker_id + 0.4,
                   xmin=segment.start_time/duration, xmax=segment.end_time/duration,
                   alpha=segment.confidence, color=f'C{segment.speaker_id}',
                   label=f'Speaker {segment.speaker_id}')
    
    # Set y-axis limits based on number of speakers
    max_speaker = max(seg.speaker_id for seg in diarization_result.segments) if diarization_result.segments else 0
    ax3.set_ylim(-0.5, max_speaker + 0.5)
    ax3.set_yticks(range(max_speaker + 1))
    ax3.grid(True, alpha=0.3)
    
    # Add overlap indicator
    if diarization_result.overlap_ratio > 0:
        ax3.text(0.02, 0.98, f'Overlap: {diarization_result.overlap_ratio:.1%}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    return fig


def demonstrate_basic_diarization():
    """Demonstrate basic speaker diarization"""
    print("Basic Speaker Diarization Demonstration")
    print("=" * 50)
    
    # Create conversation
    audio, ground_truth = create_conversation()
    sample_rate = 16000
    
    # Perform diarization
    diarizer = SpeakerDiarization(sample_rate=sample_rate)
    result = diarizer.diarize(audio)
    
    print(f"Detected {result.num_speakers} speakers")
    print(f"Overlap ratio: {result.overlap_ratio:.1%}")
    print(f"Number of segments: {len(result.segments)}")
    
    # Show first few segments
    print("\nFirst 5 segments:")
    for i, segment in enumerate(result.segments[:5]):
        print(f"  {segment.start_time:.2f}s - {segment.end_time:.2f}s: "
              f"Speaker {segment.speaker_id} (confidence: {segment.confidence:.2f})")
    
    # Visualize
    fig = visualize_diarization(audio, result, ground_truth, sample_rate)
    plt.savefig('diarization_basic.png', dpi=150)
    print("\nSaved visualization to 'diarization_basic.png'")


def demonstrate_clustering_methods():
    """Compare different clustering methods"""
    print("\n\nClustering Methods Comparison")
    print("=" * 50)
    
    # Create conversation
    audio, _ = create_conversation(duration=10.0)
    sample_rate = 16000
    
    methods = ['spectral', 'agglomerative', 'kmeans']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} clustering:")
        diarizer = SpeakerDiarization(
            sample_rate=sample_rate,
            clustering_method=method
        )
        result = diarizer.diarize(audio)
        results[method] = result
        
        print(f"  Speakers: {result.num_speakers}")
        print(f"  Segments: {len(result.segments)}")
        print(f"  Overlap: {result.overlap_ratio:.1%}")


def demonstrate_streaming_diarization():
    """Demonstrate streaming/online diarization"""
    print("\n\nStreaming Diarization Demonstration")
    print("=" * 50)
    
    # Create conversation
    audio, _ = create_conversation(duration=10.0)
    sample_rate = 16000
    
    # Create streaming diarizer
    streaming = StreamingDiarizer(
        sample_rate=sample_rate,
        window_duration=2.0,
        step_duration=0.5
    )
    
    # Process in chunks
    chunk_duration = 0.5  # 500ms chunks
    chunk_size = int(chunk_duration * sample_rate)
    
    print(f"Processing {len(audio) // chunk_size} chunks...")
    all_segments = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) == chunk_size:
            segments = streaming.process_chunk(chunk)
            all_segments.extend(segments)
            
            if segments:
                for seg in segments:
                    print(f"  Chunk {i//chunk_size}: Speaker {seg.speaker_id} "
                          f"@ {seg.start_time:.2f}s (conf: {seg.confidence:.2f})")
    
    print(f"\nTotal segments detected: {len(all_segments)}")


def demonstrate_export_formats():
    """Demonstrate different export formats"""
    print("\n\nExport Formats Demonstration")
    print("=" * 50)
    
    # Create simple conversation
    audio, _ = create_conversation(duration=5.0)
    sample_rate = 16000
    
    # Diarize
    diarizer = SpeakerDiarization(sample_rate=sample_rate)
    result = diarizer.diarize(audio)
    
    # Export to different formats
    print("\nRTTM Format:")
    print("-" * 30)
    rttm = result.to_rttm()
    print(rttm[:200] + "..." if len(rttm) > 200 else rttm)
    
    print("\n\nJSON Format:")
    print("-" * 30)
    json_str = result.to_json()
    print(json_str[:300] + "..." if len(json_str) > 300 else json_str)
    
    print("\n\nDataFrame Format:")
    print("-" * 30)
    df = result.to_dataframe()
    print(df.head())
    print(f"... ({len(df)} total segments)")


def demonstrate_speaker_similarity():
    """Demonstrate speaker embedding similarity"""
    print("\n\nSpeaker Similarity Demonstration")
    print("=" * 50)
    
    sample_rate = 16000
    duration = 2.0
    samples = int(duration * sample_rate)
    t = np.arange(samples) / sample_rate
    
    # Create similar speakers
    speaker_a1 = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.random.randn(samples)
    speaker_a2 = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.random.randn(samples)
    
    # Create different speaker
    speaker_b = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.random.randn(samples)
    
    # Extract embeddings
    diarizer = SpeakerDiarization(sample_rate=sample_rate)
    
    emb_a1 = diarizer.extract_embedding(speaker_a1)
    emb_a2 = diarizer.extract_embedding(speaker_a2)
    emb_b = diarizer.extract_embedding(speaker_b)
    
    # Compute similarities
    sim_same = diarizer.compute_similarity(emb_a1, emb_a2)
    sim_diff1 = diarizer.compute_similarity(emb_a1, emb_b)
    sim_diff2 = diarizer.compute_similarity(emb_a2, emb_b)
    
    print(f"Similarity between same speaker (A1 vs A2): {sim_same:.3f}")
    print(f"Similarity between different speakers (A1 vs B): {sim_diff1:.3f}")
    print(f"Similarity between different speakers (A2 vs B): {sim_diff2:.3f}")
    
    if sim_same > max(sim_diff1, sim_diff2):
        print("\n✓ Correctly identified same speaker has higher similarity")
    else:
        print("\n✗ Warning: Similarity metrics may need tuning")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_basic_diarization()
    demonstrate_clustering_methods()
    demonstrate_streaming_diarization()
    demonstrate_export_formats()
    demonstrate_speaker_similarity()
    
    print("\n\nDiarization demonstration completed!")
    plt.show()