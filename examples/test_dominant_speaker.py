#!/usr/bin/env python3
"""Example usage of the Dominant Speaker Identifier module (S03_T05).

This example demonstrates how to use the Dominant Speaker Identifier for:
1. Basic dominant speaker identification
2. Different dominance methods
3. Multi-speaker scenarios
4. Streaming mode
"""

import numpy as np
import matplotlib.pyplot as plt
from processors.audio_enhancement.identification.dominant_speaker_identifier import (
    DominantSpeakerIdentifier,
    DominanceMethod,
    DominanceConfig
)


def generate_multi_speaker_audio(sample_rate=16000, duration=10.0):
    """Generate synthetic multi-speaker audio with diarization."""
    samples = int(duration * sample_rate)
    audio = np.zeros(samples)
    
    # Define speaker segments
    diarization = [
        {"speaker": "SPK_001", "start": 0.0, "end": 4.0},     # 40%
        {"speaker": "SPK_002", "start": 4.0, "end": 7.0},     # 30%
        {"speaker": "SPK_003", "start": 7.0, "end": 8.5},     # 15%
        {"speaker": "SPK_001", "start": 8.5, "end": 10.0},    # 15% more for SPK_001
    ]
    
    # Generate different audio characteristics for each speaker
    for segment in diarization:
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)
        
        if segment["speaker"] == "SPK_001":
            # Moderate energy speaker
            audio[start_sample:end_sample] = np.random.randn(end_sample - start_sample) * 0.3
        elif segment["speaker"] == "SPK_002":
            # High energy speaker
            audio[start_sample:end_sample] = np.random.randn(end_sample - start_sample) * 0.6
        else:
            # Low energy speaker
            audio[start_sample:end_sample] = np.random.randn(end_sample - start_sample) * 0.1
    
    # Generate mock embeddings
    embeddings = {
        "SPK_001": np.random.randn(192),
        "SPK_002": np.random.randn(192),
        "SPK_003": np.random.randn(192)
    }
    
    return audio, diarization, embeddings


def example_basic_identification():
    """Example 1: Basic dominant speaker identification."""
    print("=" * 60)
    print("Example 1: Basic Dominant Speaker Identification")
    print("=" * 60)
    
    # Generate test data
    audio, diarization, embeddings = generate_multi_speaker_audio()
    
    # Create identifier
    identifier = DominantSpeakerIdentifier(
        sample_rate=16000,
        dominance_method=DominanceMethod.DURATION
    )
    
    # Identify dominant speaker
    result = identifier.identify_dominant(audio, diarization, embeddings)
    
    print(f"Dominant speaker: {result.dominant_speaker}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Total speakers: {result.total_speakers}")
    print("\nSpeaker statistics:")
    for speaker in sorted(result.speaker_durations.keys()):
        print(f"  {speaker}: {result.speaker_durations[speaker]:.1f}s "
              f"({result.speaker_ratios[speaker]:.1%})")
    
    print()


def example_different_methods():
    """Example 2: Compare different dominance methods."""
    print("=" * 60)
    print("Example 2: Different Dominance Methods")
    print("=" * 60)
    
    # Generate test data
    audio, diarization, embeddings = generate_multi_speaker_audio()
    
    methods = [
        DominanceMethod.DURATION,
        DominanceMethod.ENERGY,
        DominanceMethod.HYBRID
    ]
    
    for method in methods:
        identifier = DominantSpeakerIdentifier(
            sample_rate=16000,
            dominance_method=method
        )
        
        result = identifier.identify_dominant(audio, diarization, embeddings)
        
        print(f"\n{method.value.upper()} method:")
        print(f"  Dominant: {result.dominant_speaker}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        if method == DominanceMethod.ENERGY:
            print("  Speaker energies:")
            for speaker, energy in sorted(result.speaker_energies.items()):
                print(f"    {speaker}: {energy:.4f}")
    
    print()


def example_balanced_scenario():
    """Example 3: Balanced speaker scenario."""
    print("=" * 60)
    print("Example 3: Balanced Speaker Scenario")
    print("=" * 60)
    
    # Create balanced diarization (33% each)
    sample_rate = 16000
    duration = 9.0
    audio = np.random.randn(int(duration * sample_rate)) * 0.2
    
    diarization = [
        {"speaker": "SPK_001", "start": 0.0, "end": 3.0},
        {"speaker": "SPK_002", "start": 3.0, "end": 6.0},
        {"speaker": "SPK_003", "start": 6.0, "end": 9.0}
    ]
    
    embeddings = {
        "SPK_001": np.random.randn(192),
        "SPK_002": np.random.randn(192),
        "SPK_003": np.random.randn(192)
    }
    
    identifier = DominantSpeakerIdentifier(sample_rate=sample_rate)
    result = identifier.identify_dominant(audio, diarization, embeddings)
    
    print(f"Is balanced: {result.is_balanced}")
    print(f"Dominant speaker: {result.dominant_speaker}")
    print(f"Confidence: {result.confidence:.2f}")
    print("\nSpeaker distribution:")
    for speaker, ratio in sorted(result.speaker_ratios.items()):
        print(f"  {speaker}: {ratio:.1%}")
    
    print()


def example_overlapping_speech():
    """Example 4: Overlapping speech handling."""
    print("=" * 60)
    print("Example 4: Overlapping Speech")
    print("=" * 60)
    
    sample_rate = 16000
    duration = 5.0
    audio = np.random.randn(int(duration * sample_rate)) * 0.2
    
    # Create diarization with overlaps
    diarization = [
        {"speaker": "SPK_001", "start": 0.0, "end": 3.0},
        {"speaker": "SPK_002", "start": 2.0, "end": 4.0},  # Overlap 2-3s
        {"speaker": "SPK_001", "start": 3.5, "end": 5.0}
    ]
    
    embeddings = {
        "SPK_001": np.random.randn(192),
        "SPK_002": np.random.randn(192)
    }
    
    identifier = DominantSpeakerIdentifier(sample_rate=sample_rate)
    result = identifier.identify_dominant(audio, diarization, embeddings)
    
    print(f"Dominant speaker: {result.dominant_speaker}")
    print(f"Overlap regions: {result.overlap_regions}")
    print(f"Total overlap duration: {sum(end-start for start, end in result.overlap_regions):.1f}s")
    
    print()


def example_similarity_analysis():
    """Example 5: Speaker similarity analysis."""
    print("=" * 60)
    print("Example 5: Speaker Similarity Analysis")
    print("=" * 60)
    
    sample_rate = 16000
    audio, diarization, _ = generate_multi_speaker_audio(duration=5.0)
    
    # Create embeddings with SPK_001 and SPK_002 being similar
    base_embedding = np.random.randn(192)
    embeddings = {
        "SPK_001": base_embedding + np.random.randn(192) * 0.05,  # Very similar
        "SPK_002": base_embedding + np.random.randn(192) * 0.1,   # Similar
        "SPK_003": np.random.randn(192)  # Different
    }
    
    identifier = DominantSpeakerIdentifier(sample_rate=sample_rate)
    result = identifier.identify_dominant(
        audio, diarization, embeddings,
        analyze_similarity=True
    )
    
    print(f"Dominant speaker: {result.dominant_speaker}")
    print("\nSpeaker similarities:")
    if result.speaker_similarities:
        for (spk1, spk2), similarity in sorted(result.speaker_similarities.items()):
            if spk1 < spk2:  # Avoid duplicates
                print(f"  {spk1} <-> {spk2}: {similarity:.3f}")
    
    print()


def example_streaming_mode():
    """Example 6: Streaming mode for continuous processing."""
    print("=" * 60)
    print("Example 6: Streaming Mode")
    print("=" * 60)
    
    # Create streaming identifier
    identifier = DominantSpeakerIdentifier(
        sample_rate=16000,
        dominance_method=DominanceMethod.DURATION,
        streaming_mode=True,
        window_duration=5.0,
        update_interval=1.0
    )
    
    # Simulate streaming with 1-second chunks
    total_duration = 10.0
    chunk_duration = 1.0
    sample_rate = 16000
    
    # Define continuous diarization
    full_diarization = [
        {"speaker": "SPK_001", "start": 0.0, "end": 6.0},
        {"speaker": "SPK_002", "start": 6.0, "end": 10.0}
    ]
    
    embeddings = {
        "SPK_001": np.random.randn(192),
        "SPK_002": np.random.randn(192)
    }
    
    print("Processing streaming audio...")
    
    # Process chunks
    for i in range(int(total_duration / chunk_duration)):
        chunk_start = i * chunk_duration
        chunk_end = (i + 1) * chunk_duration
        
        # Get relevant diarization for this chunk
        chunk_diarization = []
        for seg in full_diarization:
            if seg["start"] < chunk_end and seg["end"] > chunk_start:
                chunk_seg = {
                    "speaker": seg["speaker"],
                    "start": max(0, seg["start"] - chunk_start),
                    "end": min(chunk_duration, seg["end"] - chunk_start)
                }
                chunk_diarization.append(chunk_seg)
        
        # Generate chunk audio
        audio_chunk = np.random.randn(int(chunk_duration * sample_rate)) * 0.2
        
        # Update streaming
        result = identifier.update_streaming(
            audio_chunk=audio_chunk,
            chunk_diarization=chunk_diarization,
            embeddings=embeddings,
            timestamp=chunk_start
        )
        
        if result.dominant_speaker:
            print(f"  Time {chunk_start:.1f}s: Dominant = {result.dominant_speaker} "
                  f"(confidence: {result.confidence:.2f})")
    
    print()


def visualize_dominance():
    """Visualize speaker dominance over time."""
    print("=" * 60)
    print("Visualization: Speaker Dominance Timeline")
    print("=" * 60)
    
    # Generate data
    audio, diarization, embeddings = generate_multi_speaker_audio(duration=20.0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    # Plot diarization
    colors = {'SPK_001': 'blue', 'SPK_002': 'red', 'SPK_003': 'green'}
    for segment in diarization:
        speaker = segment["speaker"]
        ax1.barh(0, segment["end"] - segment["start"], 
                left=segment["start"], height=0.8,
                color=colors[speaker], alpha=0.7, label=speaker)
    
    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Speaker Diarization")
    ax1.set_yticks([])
    
    # Calculate dominance for sliding windows
    window_size = 5.0
    step_size = 0.5
    identifier = DominantSpeakerIdentifier(sample_rate=16000)
    
    times = []
    dominance_scores = {spk: [] for spk in embeddings.keys()}
    
    for start_time in np.arange(0, 20 - window_size, step_size):
        end_time = start_time + window_size
        
        # Get segments in window
        window_segments = []
        for seg in diarization:
            if seg["start"] < end_time and seg["end"] > start_time:
                window_seg = {
                    "speaker": seg["speaker"],
                    "start": max(0, seg["start"] - start_time),
                    "end": min(window_size, seg["end"] - start_time)
                }
                window_segments.append(window_seg)
        
        if window_segments:
            # Extract window audio
            start_sample = int(start_time * 16000)
            end_sample = int(end_time * 16000)
            window_audio = audio[start_sample:end_sample]
            
            # Calculate dominance
            result = identifier.identify_dominant(window_audio, window_segments, embeddings)
            
            times.append(start_time + window_size / 2)
            for speaker in embeddings.keys():
                score = result.speaker_ratios.get(speaker, 0.0)
                dominance_scores[speaker].append(score)
    
    # Plot dominance scores
    for speaker, scores in dominance_scores.items():
        ax2.plot(times, scores, color=colors[speaker], label=speaker, linewidth=2)
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Dominance Score")
    ax2.set_title("Speaker Dominance Over Time (5s sliding window)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("dominance_visualization.png", dpi=150)
    print("Visualization saved to dominance_visualization.png")
    plt.close()


if __name__ == "__main__":
    print("\nDominant Speaker Identifier Examples\n")
    
    # Run all examples
    example_basic_identification()
    example_different_methods()
    example_balanced_scenario()
    example_overlapping_speech()
    example_similarity_analysis()
    example_streaming_mode()
    visualize_dominance()
    
    print("All examples completed successfully!")