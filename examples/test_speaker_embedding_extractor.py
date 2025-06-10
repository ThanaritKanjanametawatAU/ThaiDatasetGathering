#!/usr/bin/env python3
"""Example usage of the Speaker Embedding Extractor module (S03_T03).

This example demonstrates:
1. Basic embedding extraction
2. Speaker verification
3. Batch processing
4. Quality assessment
5. Model comparison
"""

import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from processors.audio_enhancement.embeddings import (
    SpeakerEmbeddingExtractor,
    AudioTooShortError
)


def create_synthetic_speaker(speaker_id: int, duration: float = 3.0,
                           sample_rate: int = 16000) -> np.ndarray:
    """Create synthetic audio for a specific speaker."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Base frequency for speaker (different for each speaker)
    base_freq = 100 + speaker_id * 50
    
    # Add harmonics to simulate speech
    signal = np.sin(2 * np.pi * base_freq * t)
    signal += 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)
    signal += 0.3 * np.sin(2 * np.pi * base_freq * 3 * t)
    
    # Add formant-like modulation
    signal *= (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
    
    # Add some noise
    signal += 0.05 * np.random.randn(len(signal))
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return signal.astype(np.float32)


def example_basic_extraction():
    """Demonstrate basic embedding extraction."""
    print("\n" + "="*50)
    print("Example 1: Basic Embedding Extraction")
    print("="*50)
    
    # Create extractor
    extractor = SpeakerEmbeddingExtractor(model='mock', device='cpu')
    
    # Create test audio
    audio = create_synthetic_speaker(speaker_id=1)
    sample_rate = 16000
    
    # Extract embedding
    result = extractor.extract(audio, sample_rate)
    
    print(f"Model: {result.model_name}")
    print(f"Embedding shape: {result.vector.shape}")
    print(f"Quality score: {result.quality_score:.3f}")
    print(f"Extraction time: {result.extraction_time:.3f}s")
    print(f"Embedding norm: {np.linalg.norm(result.vector):.3f}")


def example_speaker_verification():
    """Demonstrate speaker verification."""
    print("\n" + "="*50)
    print("Example 2: Speaker Verification")
    print("="*50)
    
    extractor = SpeakerEmbeddingExtractor(model='mock')
    sample_rate = 16000
    
    # Create audio from same speaker (with variations)
    speaker1_audio1 = create_synthetic_speaker(1, duration=3.0)
    speaker1_audio2 = create_synthetic_speaker(1, duration=2.5)
    speaker1_audio2 += 0.1 * np.random.randn(len(speaker1_audio2))  # Add noise
    
    # Create audio from different speaker
    speaker2_audio = create_synthetic_speaker(2, duration=3.0)
    
    # Verify same speaker
    is_same = extractor.verify(speaker1_audio1, speaker1_audio2, sample_rate, threshold=0.7)
    print(f"Same speaker verification: {is_same}")
    
    # Extract embeddings for detailed analysis
    emb1 = extractor.extract(speaker1_audio1, sample_rate)
    emb2 = extractor.extract(speaker1_audio2, sample_rate)
    similarity_same = extractor.compute_similarity(emb1.vector, emb2.vector)
    print(f"Same speaker similarity: {similarity_same:.3f}")
    
    # Verify different speakers
    is_same = extractor.verify(speaker1_audio1, speaker2_audio, sample_rate, threshold=0.7)
    print(f"Different speaker verification: {is_same}")
    
    emb3 = extractor.extract(speaker2_audio, sample_rate)
    similarity_diff = extractor.compute_similarity(emb1.vector, emb3.vector)
    print(f"Different speaker similarity: {similarity_diff:.3f}")


def example_batch_processing():
    """Demonstrate batch processing efficiency."""
    print("\n" + "="*50)
    print("Example 3: Batch Processing")
    print("="*50)
    
    extractor = SpeakerEmbeddingExtractor(model='mock')
    sample_rate = 16000
    
    # Create batch of audio samples
    n_samples = 20
    audio_batch = []
    for i in range(n_samples):
        speaker_id = i % 5  # 5 different speakers
        audio = create_synthetic_speaker(speaker_id)
        audio_batch.append(audio)
    
    # Time individual processing
    start = time.time()
    individual_results = []
    for audio in audio_batch:
        result = extractor.extract(audio, sample_rate, return_quality=False)
        individual_results.append(result)
    individual_time = time.time() - start
    
    # Time batch processing
    start = time.time()
    batch_results = extractor.extract_batch(audio_batch, sample_rate, return_quality=False)
    batch_time = time.time() - start
    
    print(f"Individual processing time: {individual_time:.3f}s")
    print(f"Batch processing time: {batch_time:.3f}s")
    print(f"Speedup: {individual_time/batch_time:.2f}x")
    print(f"Average time per sample (individual): {individual_time/n_samples:.3f}s")
    print(f"Average time per sample (batch): {batch_time/n_samples:.3f}s")


def example_quality_assessment():
    """Demonstrate quality assessment features."""
    print("\n" + "="*50)
    print("Example 4: Quality Assessment")
    print("="*50)
    
    extractor = SpeakerEmbeddingExtractor(model='mock')
    sample_rate = 16000
    
    # Create different quality audio samples
    print("\nTesting different audio qualities:")
    
    # 1. High quality audio
    good_audio = create_synthetic_speaker(1, duration=5.0)
    result = extractor.extract(good_audio, sample_rate, return_quality=True, return_uncertainty=True)
    print(f"\nHigh quality audio (5s, clean):")
    print(f"  Quality score: {result.quality_score:.3f}")
    print(f"  Uncertainty: {result.uncertainty:.3f}")
    
    # 2. Noisy audio
    noisy_audio = good_audio + 0.3 * np.random.randn(len(good_audio))
    result = extractor.extract(noisy_audio, sample_rate, return_quality=True, return_uncertainty=True)
    print(f"\nNoisy audio:")
    print(f"  Quality score: {result.quality_score:.3f}")
    print(f"  Uncertainty: {result.uncertainty:.3f}")
    
    # 3. Short audio (minimum duration)
    short_audio = create_synthetic_speaker(1, duration=1.5)
    result = extractor.extract(short_audio, sample_rate, return_quality=True, return_uncertainty=True)
    print(f"\nShort audio (1.5s):")
    print(f"  Quality score: {result.quality_score:.3f}")
    print(f"  Uncertainty: {result.uncertainty:.3f}")
    
    # 4. Very short audio (should fail)
    very_short_audio = create_synthetic_speaker(1, duration=0.5)
    try:
        extractor.extract(very_short_audio, sample_rate)
    except AudioTooShortError as e:
        print(f"\nVery short audio (0.5s): {e}")


def example_embedding_visualization():
    """Visualize embeddings from different speakers."""
    print("\n" + "="*50)
    print("Example 5: Embedding Visualization")
    print("="*50)
    
    extractor = SpeakerEmbeddingExtractor(model='mock')
    sample_rate = 16000
    
    # Create embeddings for multiple speakers
    n_speakers = 5
    n_samples_per_speaker = 4
    embeddings = []
    labels = []
    
    for speaker_id in range(n_speakers):
        for sample in range(n_samples_per_speaker):
            # Add slight variations for same speaker
            audio = create_synthetic_speaker(speaker_id, duration=3.0)
            if sample > 0:
                audio += 0.05 * np.random.randn(len(audio))
            
            result = extractor.extract(audio, sample_rate)
            embeddings.append(result.vector)
            labels.append(speaker_id)
    
    embeddings = np.array(embeddings)
    
    # Compute similarity matrix
    similarity_matrix = 1 - cdist(embeddings, embeddings, metric='cosine')
    
    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Speaker Embedding Similarity Matrix')
    
    # Add speaker boundaries
    for i in range(1, n_speakers):
        pos = i * n_samples_per_speaker - 0.5
        plt.axhline(pos, color='black', linewidth=2)
        plt.axvline(pos, color='black', linewidth=2)
    
    # Add labels
    tick_positions = [i * n_samples_per_speaker + n_samples_per_speaker//2 
                     for i in range(n_speakers)]
    tick_labels = [f'Speaker {i+1}' for i in range(n_speakers)]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.yticks(tick_positions, tick_labels)
    
    plt.tight_layout()
    plt.savefig('speaker_embedding_similarity.png')
    print("Similarity matrix saved to 'speaker_embedding_similarity.png'")
    
    # Compute average within-speaker and between-speaker similarities
    within_speaker_sims = []
    between_speaker_sims = []
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            if labels[i] == labels[j]:
                within_speaker_sims.append(similarity_matrix[i, j])
            else:
                between_speaker_sims.append(similarity_matrix[i, j])
    
    print(f"\nAverage within-speaker similarity: {np.mean(within_speaker_sims):.3f}")
    print(f"Average between-speaker similarity: {np.mean(between_speaker_sims):.3f}")
    print(f"Discrimination ratio: {np.mean(within_speaker_sims)/np.mean(between_speaker_sims):.2f}")


def example_dimension_reduction():
    """Demonstrate embedding dimension reduction."""
    print("\n" + "="*50)
    print("Example 6: Dimension Reduction")
    print("="*50)
    
    extractor = SpeakerEmbeddingExtractor(model='mock')
    sample_rate = 16000
    
    audio = create_synthetic_speaker(1)
    
    # Extract full embedding
    full_result = extractor.extract(audio, sample_rate)
    print(f"Full embedding dimension: {full_result.vector.shape[0]}")
    
    # Extract reduced embeddings
    for target_dim in [128, 64, 32]:
        reduced_result = extractor.extract(audio, sample_rate, reduce_dim=target_dim)
        print(f"Reduced embedding dimension: {reduced_result.vector.shape[0]}")
        
        # Check that reduced embedding still captures speaker information
        audio2 = create_synthetic_speaker(1)  # Same speaker
        reduced_result2 = extractor.extract(audio2, sample_rate, reduce_dim=target_dim)
        
        similarity = extractor.compute_similarity(
            reduced_result.vector,
            reduced_result2.vector
        )
        print(f"  Same speaker similarity (dim={target_dim}): {similarity:.3f}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Speaker Embedding Extractor Examples")
    print("="*70)
    
    # Run examples
    example_basic_extraction()
    example_speaker_verification()
    example_batch_processing()
    example_quality_assessment()
    example_embedding_visualization()
    example_dimension_reduction()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()