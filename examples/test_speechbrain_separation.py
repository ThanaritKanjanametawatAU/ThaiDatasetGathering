#!/usr/bin/env python3
"""
Example script to test SpeechBrain speaker separation.

This demonstrates how to use the new SpeechBrain-based implementation
for complete secondary speaker removal.
"""

import numpy as np
import soundfile as sf
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.speechbrain_separator import (
    SpeechBrainSeparator,
    SeparationConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_audio(duration=5, sample_rate=16000):
    """Generate test audio with two speakers"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Speaker 1: Lower frequency (male-like)
    speaker1 = np.sin(2 * np.pi * 150 * t) * 0.5
    speaker1 += np.sin(2 * np.pi * 300 * t) * 0.3
    speaker1 += np.sin(2 * np.pi * 450 * t) * 0.2
    
    # Speaker 2: Higher frequency (female-like)
    speaker2 = np.sin(2 * np.pi * 250 * t) * 0.5
    speaker2 += np.sin(2 * np.pi * 500 * t) * 0.3
    speaker2 += np.sin(2 * np.pi * 750 * t) * 0.2
    
    # Mix speakers in different segments
    mixed = np.zeros_like(t)
    segment_length = sample_rate
    
    # 0-1s: Speaker 1 only
    mixed[:segment_length] = speaker1[:segment_length]
    
    # 1-2s: Speaker 2 only
    mixed[segment_length:2*segment_length] = speaker2[segment_length:2*segment_length]
    
    # 2-3s: Both speakers (overlapping)
    mixed[2*segment_length:3*segment_length] = (
        0.6 * speaker1[2*segment_length:3*segment_length] + 
        0.4 * speaker2[2*segment_length:3*segment_length]
    )
    
    # 3-4s: Speaker 1 with background speaker 2
    mixed[3*segment_length:4*segment_length] = (
        0.8 * speaker1[3*segment_length:4*segment_length] + 
        0.2 * speaker2[3*segment_length:4*segment_length]
    )
    
    # 4-5s: Speaker 1 only
    mixed[4*segment_length:] = speaker1[4*segment_length:]
    
    # Add some noise
    mixed += np.random.randn(len(mixed)) * 0.02
    
    return mixed


def test_speechbrain_separation():
    """Test the SpeechBrain separator"""
    
    # Configuration
    config = SeparationConfig(
        device="cuda",  # Use GPU if available
        confidence_threshold=0.7,
        batch_size=16,
        speaker_selection="energy",
        use_mixed_precision=True,
        quality_thresholds={
            "min_pesq": 3.5,
            "min_stoi": 0.85,
            "max_spectral_distortion": 0.15
        }
    )
    
    # Initialize separator
    logger.info("Initializing SpeechBrain separator...")
    separator = SpeechBrainSeparator(config)
    
    # Generate or load test audio
    logger.info("Generating test audio with mixed speakers...")
    test_audio = generate_test_audio()
    sample_rate = 16000
    
    # Save original audio
    sf.write("test_mixed_speakers.wav", test_audio, sample_rate)
    logger.info("Saved mixed audio to test_mixed_speakers.wav")
    
    # Perform separation
    logger.info("Performing speaker separation...")
    result = separator.separate_speakers(test_audio, sample_rate)
    
    # Check results
    logger.info(f"\nSeparation Results:")
    logger.info(f"- Confidence: {result.confidence:.2f}")
    logger.info(f"- Rejected: {result.rejected}")
    logger.info(f"- Rejection reason: {result.rejection_reason}")
    logger.info(f"- Processing time: {result.processing_time_ms:.1f}ms")
    logger.info(f"- Speakers detected: {result.num_speakers_detected}")
    
    # Print metrics
    logger.info(f"\nQuality Metrics:")
    for metric, value in result.metrics.items():
        logger.info(f"- {metric}: {value:.3f}")
    
    # Save separated audio
    if not result.rejected:
        sf.write("test_primary_speaker.wav", result.audio, sample_rate)
        logger.info("\nSaved separated audio to test_primary_speaker.wav")
    else:
        logger.warning(f"\nAudio was rejected: {result.rejection_reason}")
    
    # Test batch processing
    logger.info("\nTesting batch processing...")
    batch = [test_audio, test_audio * 0.8, test_audio * 0.6]
    batch_results = separator.process_batch(batch, sample_rate)
    
    logger.info(f"Batch processing results:")
    for i, result in enumerate(batch_results):
        logger.info(f"- Sample {i+1}: Confidence={result.confidence:.2f}, Rejected={result.rejected}")
    
    # Print overall statistics
    stats = separator.get_stats()
    logger.info(f"\nOverall Statistics:")
    logger.info(f"- Total processed: {stats['total_processed']}")
    logger.info(f"- Success rate: {stats.get('success_rate', 0):.2%}")
    logger.info(f"- Average confidence: {stats['average_confidence']:.2f}")
    logger.info(f"- Average processing time: {stats.get('average_processing_time_ms', 0):.1f}ms")


def test_real_audio(audio_path):
    """Test with real audio file"""
    
    # Load audio
    audio, sample_rate = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Configuration for real audio
    config = SeparationConfig(
        device="cuda",
        confidence_threshold=0.7,
        speaker_selection="energy",  # Can also try "embedding" for better accuracy
        quality_thresholds={
            "min_stoi": 0.80,  # Slightly lower for real audio
            "max_spectral_distortion": 0.20
        }
    )
    
    # Process
    separator = SpeechBrainSeparator(config)
    result = separator.separate_speakers(audio, sample_rate)
    
    # Save result
    if not result.rejected:
        output_path = audio_path.replace('.wav', '_separated.wav')
        sf.write(output_path, result.audio, sample_rate)
        logger.info(f"Saved separated audio to {output_path}")
        
        # Print detailed report
        from processors.audio_enhancement.quality_validator import QualityValidator
        validator = QualityValidator()
        report = validator.get_quality_report(result.metrics)
        print("\n" + report)
    else:
        logger.error(f"Separation failed: {result.rejection_reason}")


if __name__ == "__main__":
    # Test with synthetic audio
    test_speechbrain_separation()
    
    # Test with real audio if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            logger.info(f"\nTesting with real audio: {audio_file}")
            test_real_audio(audio_file)
        else:
            logger.error(f"Audio file not found: {audio_file}")
    else:
        print("\nUsage: python test_speechbrain_separation.py [audio_file.wav]")
        print("If no audio file is provided, synthetic test audio will be used.")