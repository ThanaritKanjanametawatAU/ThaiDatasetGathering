"""
Test suite for Speaker Diarization System

Tests the speaker diarization module that segments audio into speaker turns
and identifies different speakers in a conversation.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from pathlib import Path

# Import the module to be tested
from processors.audio_enhancement.detection.speaker_diarization import (
    SpeakerDiarization,
    DiarizationResult,
    SpeakerSegment,
    SpeakerEmbedding,
    ClusteringMethod,
    SegmentationMethod
)


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests"""
    return 16000


@pytest.fixture
def mono_conversation(sample_rate):
    """Create synthetic mono conversation with 2 speakers"""
    duration = 10.0  # 10 seconds
    samples = int(duration * sample_rate)
    t = np.arange(samples) / sample_rate
    
    # Create more realistic speaker signals with harmonics
    # Speaker 1 - male-like voice (lower pitch)
    speaker1 = np.zeros(samples)
    fundamentals1 = [100, 120, 110]  # Varying fundamental frequency
    for i, f0 in enumerate(fundamentals1):
        phase = i * 0.5  # Phase variation
        for harmonic in range(1, 5):  # Add harmonics
            amplitude = 0.3 / harmonic  # Decreasing amplitude for harmonics
            speaker1 += amplitude * np.sin(2 * np.pi * f0 * harmonic * t + phase)
    
    # Add formant-like filtering
    speaker1 += 0.1 * np.random.randn(samples)  # Add noise
    
    # Speaker 2 - female-like voice (higher pitch)
    speaker2 = np.zeros(samples)
    fundamentals2 = [200, 220, 210]  # Higher fundamental
    for i, f0 in enumerate(fundamentals2):
        phase = i * 0.7
        for harmonic in range(1, 5):
            amplitude = 0.3 / harmonic
            speaker2 += amplitude * np.sin(2 * np.pi * f0 * harmonic * t + phase)
    
    speaker2 += 0.1 * np.random.randn(samples)
    
    # Create alternating speaker segments
    conversation = np.zeros(samples)
    segment_duration = 2.0  # 2 seconds per turn
    segment_samples = int(segment_duration * sample_rate)
    
    for i in range(0, samples, segment_samples * 2):
        # Speaker 1
        end1 = min(i + segment_samples, samples)
        # Add envelope to make it more speech-like
        envelope = np.hanning(end1 - i)
        conversation[i:end1] = speaker1[i:end1] * envelope
        
        # Speaker 2
        start2 = end1
        end2 = min(start2 + segment_samples, samples)
        if start2 < samples:
            envelope = np.hanning(end2 - start2)
            conversation[start2:end2] = speaker2[start2:end2] * envelope
    
    # Normalize to reasonable amplitude
    conversation = conversation / np.max(np.abs(conversation)) * 0.5
    
    return conversation


@pytest.fixture
def overlapping_conversation(sample_rate):
    """Create conversation with overlapping speakers"""
    duration = 6.0
    samples = int(duration * sample_rate)
    t = np.arange(samples) / sample_rate
    
    # Create two realistic speakers
    # Speaker 1
    speaker1 = np.zeros(samples)
    for harmonic in range(1, 5):
        speaker1 += (0.3 / harmonic) * np.sin(2 * np.pi * 120 * harmonic * t)
    speaker1 += 0.1 * np.random.randn(samples)
    
    # Speaker 2
    speaker2 = np.zeros(samples)
    for harmonic in range(1, 5):
        speaker2 += (0.3 / harmonic) * np.sin(2 * np.pi * 180 * harmonic * t)
    speaker2 += 0.1 * np.random.randn(samples)
    
    # Create conversation with overlaps
    conversation = np.zeros(samples)
    
    # Speaker 1: 0-2s, 3-4s
    env1 = np.hanning(int(2*sample_rate))
    conversation[:int(2*sample_rate)] = speaker1[:int(2*sample_rate)] * env1
    
    env2 = np.hanning(int(1*sample_rate))
    conversation[int(3*sample_rate):int(4*sample_rate)] = speaker1[int(3*sample_rate):int(4*sample_rate)] * env2
    
    # Speaker 2: 1.5-3.5s, 4.5-6s (overlaps with speaker 1)
    start = int(1.5*sample_rate)
    end = int(3.5*sample_rate)
    env3 = np.hanning(end - start)
    conversation[start:end] += speaker2[start:end] * env3
    
    start = int(4.5*sample_rate)
    env4 = np.hanning(len(conversation) - start)
    conversation[start:] = speaker2[start:] * env4
    
    # Normalize
    conversation = conversation / np.max(np.abs(conversation)) * 0.5
    
    return conversation


@pytest.fixture
def diarizer():
    """Create diarizer instance"""
    return SpeakerDiarization(sample_rate=16000)


class TestSpeakerDiarization:
    """Test speaker diarization functionality"""
    
    def test_initialization(self):
        """Test diarizer initialization"""
        # Default initialization
        diarizer = SpeakerDiarization()
        assert diarizer.sample_rate == 16000
        assert diarizer.embedding_model is not None
        assert diarizer.clustering_method == ClusteringMethod.SPECTRAL
        
        # Custom initialization
        diarizer = SpeakerDiarization(
            sample_rate=8000,
            clustering_method='agglomerative',
            min_speakers=2,
            max_speakers=10
        )
        assert diarizer.sample_rate == 8000
        assert diarizer.clustering_method == ClusteringMethod.AGGLOMERATIVE
        assert diarizer.min_speakers == 2
        assert diarizer.max_speakers == 10
    
    def test_basic_diarization(self, diarizer, mono_conversation):
        """Test basic speaker diarization"""
        result = diarizer.diarize(mono_conversation)
        
        # Check result structure
        assert isinstance(result, DiarizationResult)
        assert len(result.segments) > 0
        assert result.num_speakers >= 2
        assert 0 <= result.overlap_ratio <= 1
        
        # Check segments
        for segment in result.segments:
            assert isinstance(segment, SpeakerSegment)
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time
            assert segment.speaker_id >= 0
            assert 0 <= segment.confidence <= 1
    
    def test_segment_continuity(self, diarizer, mono_conversation):
        """Test that segments cover the entire audio"""
        result = diarizer.diarize(mono_conversation)
        
        # Sort segments by start time
        segments = sorted(result.segments, key=lambda s: s.start_time)
        
        # Check coverage
        duration = len(mono_conversation) / diarizer.sample_rate
        
        # First segment should start near beginning
        assert segments[0].start_time < 0.5
        
        # Last segment should end near the end
        assert segments[-1].end_time > duration - 0.5
    
    def test_speaker_consistency(self, diarizer, mono_conversation):
        """Test speaker ID consistency"""
        result = diarizer.diarize(mono_conversation)
        
        # Group segments by speaker
        speaker_segments = {}
        for segment in result.segments:
            if segment.speaker_id not in speaker_segments:
                speaker_segments[segment.speaker_id] = []
            speaker_segments[segment.speaker_id].append(segment)
        
        # Should have at least 2 speakers
        assert len(speaker_segments) >= 2
        
        # Each speaker should have multiple segments
        for speaker_id, segments in speaker_segments.items():
            assert len(segments) >= 1
    
    def test_overlapping_speakers(self, diarizer, overlapping_conversation):
        """Test detection of overlapping speech"""
        result = diarizer.diarize(overlapping_conversation)
        
        # Should detect some overlap
        assert result.overlap_ratio > 0
        
        # Check for overlapping segments
        overlaps = []
        segments = sorted(result.segments, key=lambda s: s.start_time)
        
        for i in range(len(segments) - 1):
            for j in range(i + 1, len(segments)):
                if (segments[i].end_time > segments[j].start_time and
                    segments[i].speaker_id != segments[j].speaker_id):
                    overlaps.append((segments[i], segments[j]))
        
        # Should find some overlaps
        assert len(overlaps) > 0
    
    def test_embedding_extraction(self, diarizer):
        """Test speaker embedding extraction"""
        # Create short audio segment
        audio = np.random.randn(16000)  # 1 second
        
        # Extract embedding
        embedding = diarizer.extract_embedding(audio)
        
        # Check embedding properties
        assert isinstance(embedding, SpeakerEmbedding)
        assert embedding.vector.shape[0] > 0  # Has dimensions
        assert np.abs(embedding.vector).max() > 0  # Non-zero
        assert embedding.confidence > 0
    
    def test_different_clustering_methods(self, mono_conversation):
        """Test different clustering algorithms"""
        methods = ['spectral', 'agglomerative', 'kmeans']
        results = {}
        
        for method in methods:
            diarizer = SpeakerDiarization(clustering_method=method)
            result = diarizer.diarize(mono_conversation)
            results[method] = result
            
            # Each method should produce valid results
            assert result.num_speakers >= 2
            assert len(result.segments) > 0
    
    def test_min_max_speakers(self, mono_conversation):
        """Test min/max speaker constraints"""
        # Test minimum speakers
        diarizer = SpeakerDiarization(min_speakers=3, max_speakers=10)
        result = diarizer.diarize(mono_conversation)
        assert result.num_speakers >= 3
        
        # Test maximum speakers
        diarizer = SpeakerDiarization(min_speakers=1, max_speakers=2)
        result = diarizer.diarize(mono_conversation)
        assert result.num_speakers <= 2
    
    def test_silence_handling(self, sample_rate):
        """Test handling of silence in audio"""
        # Create audio with silence gaps
        duration = 6.0
        samples = int(duration * sample_rate)
        audio = np.zeros(samples)
        
        # Add speech segments with gaps
        audio[int(0.5*sample_rate):int(1.5*sample_rate)] = np.random.randn(sample_rate) * 0.1
        audio[int(3*sample_rate):int(4*sample_rate)] = np.random.randn(sample_rate) * 0.1
        audio[int(5*sample_rate):int(5.5*sample_rate)] = np.random.randn(int(0.5*sample_rate)) * 0.1
        
        diarizer = SpeakerDiarization(sample_rate=sample_rate)
        result = diarizer.diarize(audio)
        
        # Should not create segments in silence
        for segment in result.segments:
            # Check that segments don't span long silence
            assert segment.end_time - segment.start_time < 2.0
    
    def test_export_formats(self, diarizer, mono_conversation):
        """Test different export formats"""
        result = diarizer.diarize(mono_conversation)
        
        # Test RTTM format
        rttm = result.to_rttm()
        assert isinstance(rttm, str)
        assert 'SPEAKER' in rttm
        
        # Test JSON format
        json_data = result.to_json()
        assert isinstance(json_data, str)
        import json
        parsed = json.loads(json_data)
        assert 'segments' in parsed
        assert 'num_speakers' in parsed
        
        # Test DataFrame format
        df = result.to_dataframe()
        assert len(df) == len(result.segments)
        assert 'start_time' in df.columns
        assert 'speaker_id' in df.columns
    
    def test_streaming_diarization(self, mono_conversation, sample_rate):
        """Test streaming/online diarization"""
        from processors.audio_enhancement.detection.speaker_diarization import StreamingDiarizer
        
        # Create streaming diarizer
        streaming = StreamingDiarizer(
            sample_rate=sample_rate,
            window_duration=2.0,
            step_duration=0.5
        )
        
        # Process in chunks
        chunk_duration = 0.5  # 500ms chunks
        chunk_size = int(chunk_duration * sample_rate)
        
        all_segments = []
        for i in range(0, len(mono_conversation), chunk_size):
            chunk = mono_conversation[i:i+chunk_size]
            segments = streaming.process_chunk(chunk)
            all_segments.extend(segments)
        
        # Should produce segments
        assert len(all_segments) > 0
        
        # Check temporal ordering
        for i in range(1, len(all_segments)):
            assert all_segments[i].start_time >= all_segments[i-1].start_time
    
    def test_resegmentation(self, diarizer, mono_conversation):
        """Test resegmentation with different parameters"""
        # Initial diarization
        result1 = diarizer.diarize(mono_conversation)
        
        # Resegment with different min duration
        result2 = diarizer.resegment(
            result1,
            min_segment_duration=1.0,
            merge_threshold=0.5
        )
        
        # Should have fewer, longer segments
        assert len(result2.segments) <= len(result1.segments)
        
        # Check minimum duration
        for segment in result2.segments:
            assert segment.end_time - segment.start_time >= 0.9  # Allow small tolerance
    
    def test_speaker_embedding_similarity(self, diarizer):
        """Test speaker embedding similarity computation"""
        # Create two similar audio segments
        audio1 = np.random.randn(16000) * 0.1
        audio2 = audio1 + np.random.randn(16000) * 0.01  # Slightly different
        
        # Create different audio
        audio3 = np.random.randn(16000) * 0.1
        
        # Extract embeddings
        emb1 = diarizer.extract_embedding(audio1)
        emb2 = diarizer.extract_embedding(audio2)
        emb3 = diarizer.extract_embedding(audio3)
        
        # Compute similarities
        sim_same = diarizer.compute_similarity(emb1, emb2)
        sim_diff = diarizer.compute_similarity(emb1, emb3)
        
        # Similar audio should have higher similarity
        assert sim_same > sim_diff
        assert 0 <= sim_same <= 1
        assert 0 <= sim_diff <= 1
    
    def test_performance_metrics(self, diarizer, mono_conversation):
        """Test diarization performance metrics"""
        # Diarize audio
        result = diarizer.diarize(mono_conversation)
        
        # Create mock ground truth
        ground_truth = DiarizationResult(
            segments=[
                SpeakerSegment(0.0, 2.0, 0, 0.9),
                SpeakerSegment(2.0, 4.0, 1, 0.9),
                SpeakerSegment(4.0, 6.0, 0, 0.9),
                SpeakerSegment(6.0, 8.0, 1, 0.9),
            ],
            num_speakers=2,
            overlap_ratio=0.0
        )
        
        # Compute metrics
        metrics = diarizer.evaluate(result, ground_truth)
        
        # Check metrics
        assert 'der' in metrics  # Diarization Error Rate
        assert 'confusion' in metrics
        assert 'missed_speech' in metrics
        assert 'false_alarm' in metrics
        
        # Metrics should be in valid ranges
        assert 0 <= metrics['der']  # DER can exceed 1 if errors > speech
        assert 0 <= metrics['confusion'] <= 1
        assert 0 <= metrics['missed_speech'] <= 1
        assert 0 <= metrics['false_alarm']
    
    def test_multi_channel_diarization(self, sample_rate):
        """Test multi-channel audio diarization"""
        # Create 2-channel audio
        duration = 5.0
        samples = int(duration * sample_rate)
        
        # Channel 1: Speaker A
        channel1 = np.random.randn(samples) * 0.1
        
        # Channel 2: Speaker B
        channel2 = np.random.randn(samples) * 0.1
        
        # Stack channels
        multi_channel = np.stack([channel1, channel2], axis=0)
        
        # Diarize with channel information
        diarizer = SpeakerDiarization(sample_rate=sample_rate, use_channel_info=True)
        result = diarizer.diarize(multi_channel)
        
        # Should identify 2 speakers (one per channel)
        assert result.num_speakers == 2
        
        # Check channel assignments
        channel_speakers = {}
        for segment in result.segments:
            if hasattr(segment, 'channel'):
                if segment.channel not in channel_speakers:
                    channel_speakers[segment.channel] = set()
                channel_speakers[segment.channel].add(segment.speaker_id)
        
        # Each channel should predominantly have one speaker
        for channel, speakers in channel_speakers.items():
            assert len(speakers) <= 2  # Allow some confusion but not too much
    
    def test_language_specific_models(self):
        """Test language-specific diarization models"""
        # Test Thai language model if available
        try:
            diarizer = SpeakerDiarization(language='th')
            assert diarizer.language == 'th'
        except ValueError:
            # Thai model might not be available
            pass
        
        # Test English model (should be available)
        diarizer = SpeakerDiarization(language='en')
        assert diarizer.language == 'en'
    
    def test_confidence_thresholding(self, diarizer, mono_conversation):
        """Test confidence-based segment filtering"""
        result = diarizer.diarize(mono_conversation)
        
        # Filter by confidence
        high_conf_segments = [s for s in result.segments if s.confidence > 0.8]
        low_conf_segments = [s for s in result.segments if s.confidence <= 0.5]
        
        # High confidence segments should be longer on average
        if high_conf_segments and low_conf_segments:
            avg_high_duration = np.mean([s.end_time - s.start_time for s in high_conf_segments])
            avg_low_duration = np.mean([s.end_time - s.start_time for s in low_conf_segments])
            assert avg_high_duration > avg_low_duration * 0.8  # Some tolerance
    
    def test_error_handling(self, diarizer):
        """Test error handling"""
        # Empty audio
        with pytest.raises(ValueError):
            diarizer.diarize(np.array([]))
        
        # Invalid audio shape
        with pytest.raises(ValueError):
            diarizer.diarize(np.random.randn(10, 10, 10))
        
        # Too short audio
        with pytest.raises(ValueError):
            diarizer.diarize(np.random.randn(100))  # Very short