"""Test suite for Dominant Speaker Identifier module (S03_T05).

This test suite validates the Dominant Speaker Identifier with the following requirements:
1. Accurate identification of dominant speaker in multi-speaker segments
2. Robust handling of various speaker configurations
3. Integration with speaker embeddings and diarization
4. Performance optimization for real-time processing
5. Confidence scoring for identification results
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch
from processors.audio_enhancement.identification.dominant_speaker_identifier import (
    DominantSpeakerIdentifier,
    DominanceMethod,
    SpeakerDominance,
    DominanceConfig
)


class TestDominantSpeakerIdentifier(unittest.TestCase):
    """Test suite for Dominant Speaker Identifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.identifier = DominantSpeakerIdentifier(
            sample_rate=self.sample_rate,
            dominance_method=DominanceMethod.DURATION
        )
    
    def test_single_speaker_identification(self):
        """Test 1.1: Handle single speaker case correctly."""
        # Create single speaker diarization
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 5.0}
        ]
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192)  # Mock embedding
        }
        
        # Create 5 seconds of audio
        audio = np.random.randn(5 * self.sample_rate) * 0.1
        
        # Identify dominant speaker
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        # Verify results
        self.assertEqual(result.dominant_speaker, "SPK_001")
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.total_speakers, 1)
        self.assertEqual(result.speaker_durations["SPK_001"], 5.0)
    
    def test_two_speaker_identification(self):
        """Test 1.2: Identify dominant speaker with two speakers."""
        # Create two speaker diarization (70-30 split)
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 3.5},
            {"speaker": "SPK_002", "start": 3.5, "end": 5.0}
        ]
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192)
        }
        
        # Create audio
        audio = np.random.randn(5 * self.sample_rate) * 0.1
        
        # Identify dominant speaker
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        # Verify results
        self.assertEqual(result.dominant_speaker, "SPK_001")
        self.assertAlmostEqual(result.confidence, 0.7, places=1)
        self.assertEqual(result.total_speakers, 2)
        self.assertAlmostEqual(result.speaker_durations["SPK_001"], 3.5, places=1)
        self.assertAlmostEqual(result.speaker_durations["SPK_002"], 1.5, places=1)
    
    def test_multiple_speaker_identification(self):
        """Test 1.3: Handle multiple speakers (3+) correctly."""
        # Create multi-speaker diarization
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 2.0},
            {"speaker": "SPK_002", "start": 2.0, "end": 3.0},
            {"speaker": "SPK_003", "start": 3.0, "end": 3.5},
            {"speaker": "SPK_001", "start": 3.5, "end": 5.0},  # SPK_001 speaks again
        ]
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192),
            "SPK_003": np.random.randn(192)
        }
        
        # Create audio
        audio = np.random.randn(5 * self.sample_rate) * 0.1
        
        # Identify dominant speaker
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        # Verify results (SPK_001 has 3.5s total)
        self.assertEqual(result.dominant_speaker, "SPK_001")
        self.assertAlmostEqual(result.confidence, 0.7, places=1)  # 3.5/5.0
        self.assertEqual(result.total_speakers, 3)
        self.assertAlmostEqual(result.speaker_durations["SPK_001"], 3.5, places=1)
    
    def test_energy_based_dominance(self):
        """Test 1.4: Energy-based dominance identification."""
        # Create identifier with energy method
        identifier = DominantSpeakerIdentifier(
            sample_rate=self.sample_rate,
            dominance_method=DominanceMethod.ENERGY
        )
        
        # Create diarization with equal durations
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 2.5},
            {"speaker": "SPK_002", "start": 2.5, "end": 5.0}
        ]
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192)
        }
        
        # Create audio with different energy levels
        audio = np.zeros(5 * self.sample_rate)
        # SPK_001: lower energy
        audio[:int(2.5 * self.sample_rate)] = np.random.randn(int(2.5 * self.sample_rate)) * 0.1
        # SPK_002: higher energy
        audio[int(2.5 * self.sample_rate):] = np.random.randn(int(2.5 * self.sample_rate)) * 0.5
        
        # Identify dominant speaker
        result = identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        # SPK_002 should be dominant due to higher energy
        self.assertEqual(result.dominant_speaker, "SPK_002")
        self.assertGreater(result.speaker_energies["SPK_002"], 
                          result.speaker_energies["SPK_001"])
    
    def test_hybrid_dominance(self):
        """Test 1.5: Hybrid dominance method combining duration and energy."""
        # Create identifier with hybrid method
        identifier = DominantSpeakerIdentifier(
            sample_rate=self.sample_rate,
            dominance_method=DominanceMethod.HYBRID
        )
        
        # Create diarization
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 3.0},  # 60% duration
            {"speaker": "SPK_002", "start": 3.0, "end": 5.0}   # 40% duration
        ]
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192)
        }
        
        # Create audio with SPK_002 having much higher energy
        audio = np.zeros(5 * self.sample_rate)
        # SPK_001: very low energy
        audio[:int(3.0 * self.sample_rate)] = np.random.randn(int(3.0 * self.sample_rate)) * 0.05
        # SPK_002: very high energy
        audio[int(3.0 * self.sample_rate):] = np.random.randn(int(2.0 * self.sample_rate)) * 0.8
        
        # Identify dominant speaker
        result = identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        # Result should balance duration and energy
        self.assertIsNotNone(result.dominant_speaker)
        self.assertGreater(result.confidence, 0.0)
    
    def test_overlapping_speech_handling(self):
        """Test 1.6: Handle overlapping speech regions."""
        # Create diarization with overlaps
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 3.0},
            {"speaker": "SPK_002", "start": 2.0, "end": 4.0},  # Overlap from 2-3s
            {"speaker": "SPK_001", "start": 3.5, "end": 5.0}
        ]
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192)
        }
        
        # Create audio
        audio = np.random.randn(5 * self.sample_rate) * 0.1
        
        # Identify dominant speaker
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        # Should handle overlaps gracefully
        self.assertIsNotNone(result.dominant_speaker)
        self.assertEqual(result.total_speakers, 2)
        self.assertGreater(len(result.overlap_regions), 0)
    
    def test_empty_diarization(self):
        """Test 2.1: Handle empty diarization gracefully."""
        # Empty diarization
        diarization = []
        embeddings = {}
        
        audio = np.random.randn(5 * self.sample_rate) * 0.1
        
        # Should handle gracefully
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        self.assertIsNone(result.dominant_speaker)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.total_speakers, 0)
    
    def test_no_clear_dominant(self):
        """Test 2.2: Handle case with no clear dominant speaker."""
        # Create balanced diarization (50-50 split)
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 2.5},
            {"speaker": "SPK_002", "start": 2.5, "end": 5.0}
        ]
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192)
        }
        
        # Create audio with equal energy
        audio = np.random.randn(5 * self.sample_rate) * 0.1
        
        # Identify dominant speaker
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        # Should still pick one but with low confidence
        self.assertIsNotNone(result.dominant_speaker)
        self.assertLess(result.confidence, 0.6)  # Low confidence
        self.assertTrue(result.is_balanced)
    
    def test_speaker_similarity_analysis(self):
        """Test 2.3: Analyze speaker similarity using embeddings."""
        # Create diarization
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 2.0},
            {"speaker": "SPK_002", "start": 2.0, "end": 4.0},
            {"speaker": "SPK_003", "start": 4.0, "end": 5.0}
        ]
        
        # Create embeddings with SPK_001 and SPK_002 being similar
        base_embedding = np.random.randn(192)
        embeddings = {
            "SPK_001": base_embedding + np.random.randn(192) * 0.1,  # Similar to base
            "SPK_002": base_embedding + np.random.randn(192) * 0.1,  # Similar to base
            "SPK_003": np.random.randn(192)  # Different
        }
        
        # Create audio
        audio = np.random.randn(5 * self.sample_rate) * 0.1
        
        # Identify dominant speaker with similarity analysis
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings,
            analyze_similarity=True
        )
        
        # Should detect similarity
        self.assertIsNotNone(result.speaker_similarities)
        self.assertGreater(result.speaker_similarities[("SPK_001", "SPK_002")], 0.8)
        self.assertLess(result.speaker_similarities[("SPK_001", "SPK_003")], 0.5)
    
    def test_min_duration_threshold(self):
        """Test 2.4: Apply minimum duration threshold for dominance."""
        # Configure with minimum duration threshold
        config = DominanceConfig(
            min_duration_ratio=0.3,  # Speaker must have at least 30% of total
            energy_weight=0.3,
            duration_weight=0.7
        )
        identifier = DominantSpeakerIdentifier(
            sample_rate=self.sample_rate,
            dominance_method=DominanceMethod.DURATION,
            config=config
        )
        
        # Create diarization with one speaker below threshold
        diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 4.0},  # 80%
            {"speaker": "SPK_002", "start": 4.0, "end": 4.5},  # 10% - below threshold
            {"speaker": "SPK_003", "start": 4.5, "end": 5.0}   # 10% - below threshold
        ]
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192),
            "SPK_003": np.random.randn(192)
        }
        
        # Create audio
        audio = np.random.randn(5 * self.sample_rate) * 0.1
        
        # Identify dominant speaker
        result = identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        
        # Only SPK_001 should be considered for dominance
        self.assertEqual(result.dominant_speaker, "SPK_001")
        self.assertEqual(len(result.qualified_speakers), 1)
    
    def test_real_time_processing(self):
        """Test 3.1: Ensure real-time processing performance."""
        import time
        
        # Create realistic diarization (10 seconds, multiple speakers)
        diarization = []
        for i in range(20):  # 20 segments
            speaker = f"SPK_{(i % 3) + 1:03d}"
            start = i * 0.5
            end = (i + 1) * 0.5
            diarization.append({"speaker": speaker, "start": start, "end": end})
        
        # Mock embeddings
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192),
            "SPK_003": np.random.randn(192)
        }
        
        # Create 10 seconds of audio
        audio = np.random.randn(10 * self.sample_rate) * 0.1
        
        # Measure processing time
        start_time = time.time()
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        processing_time = time.time() - start_time
        
        # Should process in real-time (less than audio duration)
        self.assertLess(processing_time, 10.0)  # Less than audio duration
        self.assertLess(processing_time, 1.0)   # Actually should be much faster
        self.assertIsNotNone(result.dominant_speaker)
    
    def test_batch_processing(self):
        """Test 3.2: Batch processing of multiple audio segments."""
        # Create multiple audio segments
        batch_size = 5
        audio_segments = []
        diarizations = []
        embeddings_list = []
        
        for i in range(batch_size):
            # Create audio
            audio = np.random.randn(3 * self.sample_rate) * 0.1
            audio_segments.append(audio)
            
            # Create diarization
            diarization = [
                {"speaker": "SPK_001", "start": 0.0, "end": 2.0},
                {"speaker": "SPK_002", "start": 2.0, "end": 3.0}
            ]
            diarizations.append(diarization)
            
            # Create embeddings
            embeddings = {
                "SPK_001": np.random.randn(192),
                "SPK_002": np.random.randn(192)
            }
            embeddings_list.append(embeddings)
        
        # Process batch
        results = self.identifier.process_batch(
            audio_segments=audio_segments,
            diarizations=diarizations,
            embeddings_list=embeddings_list
        )
        
        # Verify results
        self.assertEqual(len(results), batch_size)
        for result in results:
            self.assertIsNotNone(result.dominant_speaker)
            self.assertGreater(result.confidence, 0.0)
    
    def test_streaming_mode(self):
        """Test 3.3: Streaming mode for continuous audio."""
        # Configure for streaming
        identifier = DominantSpeakerIdentifier(
            sample_rate=self.sample_rate,
            dominance_method=DominanceMethod.DURATION,
            streaming_mode=True,
            window_duration=5.0,
            update_interval=1.0
        )
        
        # Simulate streaming with chunks
        total_duration = 10.0
        chunk_duration = 1.0
        num_chunks = int(total_duration / chunk_duration)
        
        # Create continuous diarization
        full_diarization = [
            {"speaker": "SPK_001", "start": 0.0, "end": 6.0},
            {"speaker": "SPK_002", "start": 6.0, "end": 10.0}
        ]
        
        embeddings = {
            "SPK_001": np.random.randn(192),
            "SPK_002": np.random.randn(192)
        }
        
        # Process chunks
        for i in range(num_chunks):
            chunk_start = i * chunk_duration
            chunk_end = (i + 1) * chunk_duration
            
            # Extract relevant diarization for chunk
            chunk_diarization = [
                seg for seg in full_diarization
                if seg["start"] < chunk_end and seg["end"] > chunk_start
            ]
            
            # Create audio chunk
            audio_chunk = np.random.randn(int(chunk_duration * self.sample_rate)) * 0.1
            
            # Update streaming identifier
            result = identifier.update_streaming(
                audio_chunk=audio_chunk,
                chunk_diarization=chunk_diarization,
                embeddings=embeddings,
                timestamp=chunk_start
            )
            
            # After 5 seconds, should identify SPK_001 as dominant
            if i >= 4:  # After 5 seconds
                self.assertIsNotNone(result.dominant_speaker)
                if i < 8:  # Before SPK_002 becomes dominant
                    self.assertEqual(result.dominant_speaker, "SPK_001")
    
    def test_confidence_calibration(self):
        """Test 3.4: Verify confidence score calibration."""
        test_cases = [
            # (durations, expected_confidence_range)
            ([5.0], (0.95, 1.0)),  # Single speaker - very high confidence
            ([4.0, 1.0], (0.75, 0.85)),  # 80-20 split - high confidence
            ([3.0, 2.0], (0.55, 0.65)),  # 60-40 split - medium confidence
            ([2.5, 2.5], (0.45, 0.55)),  # 50-50 split - low confidence
            ([2.0, 2.0, 1.0], (0.35, 0.45)),  # 40-40-20 split - very low confidence
        ]
        
        for durations, (min_conf, max_conf) in test_cases:
            # Create diarization based on durations
            diarization = []
            embeddings = {}
            current_time = 0.0
            
            for i, duration in enumerate(durations):
                speaker = f"SPK_{i+1:03d}"
                diarization.append({
                    "speaker": speaker,
                    "start": current_time,
                    "end": current_time + duration
                })
                embeddings[speaker] = np.random.randn(192)
                current_time += duration
            
            # Create audio
            total_duration = sum(durations)
            audio = np.random.randn(int(total_duration * self.sample_rate)) * 0.1
            
            # Identify dominant speaker
            result = self.identifier.identify_dominant(
                audio=audio,
                diarization=diarization,
                embeddings=embeddings
            )
            
            # Check confidence calibration
            self.assertGreaterEqual(result.confidence, min_conf,
                                   f"Confidence too low for {durations}")
            self.assertLessEqual(result.confidence, max_conf,
                                f"Confidence too high for {durations}")
    
    def test_edge_cases(self):
        """Test 4: Edge cases and error handling."""
        # Test 4.1: Very short audio
        short_audio = np.random.randn(int(0.1 * self.sample_rate)) * 0.1
        diarization = [{"speaker": "SPK_001", "start": 0.0, "end": 0.1}]
        embeddings = {"SPK_001": np.random.randn(192)}
        
        result = self.identifier.identify_dominant(
            audio=short_audio,
            diarization=diarization,
            embeddings=embeddings
        )
        self.assertIsNotNone(result)
        
        # Test 4.2: Mismatched embeddings
        diarization = [{"speaker": "SPK_001", "start": 0.0, "end": 1.0}]
        embeddings = {"SPK_002": np.random.randn(192)}  # Wrong speaker
        audio = np.random.randn(self.sample_rate) * 0.1
        
        with self.assertRaises(ValueError):
            self.identifier.identify_dominant(
                audio=audio,
                diarization=diarization,
                embeddings=embeddings
            )
        
        # Test 4.3: Invalid diarization times
        diarization = [{"speaker": "SPK_001", "start": 0.0, "end": 10.0}]
        embeddings = {"SPK_001": np.random.randn(192)}
        audio = np.random.randn(5 * self.sample_rate) * 0.1  # Only 5s audio
        
        # Should handle gracefully by clipping
        result = self.identifier.identify_dominant(
            audio=audio,
            diarization=diarization,
            embeddings=embeddings
        )
        self.assertEqual(result.speaker_durations["SPK_001"], 5.0)


if __name__ == "__main__":
    unittest.main()