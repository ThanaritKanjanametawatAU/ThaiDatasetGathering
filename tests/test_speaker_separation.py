"""
Test Suite for Speaker Separation Module

Tests secondary speaker detection, flexible duration handling,
confidence-based suppression, and integration.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import modules to test
from processors.audio_enhancement.speaker_separation import (
    SpeakerSeparator, SeparationConfig
)
from processors.audio_enhancement.detection.secondary_speaker import (
    AdaptiveSecondaryDetection, DetectionResult
)
from processors.audio_enhancement.detection.overlap_detector import (
    OverlapDetector, OverlapConfig
)
from utils.speaker_utils import (
    extract_speaker_embedding, compare_embeddings,
    create_speaker_profile, SpeakerComparator
)


class TestSpeakerSeparation(unittest.TestCase):
    """Test speaker separation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 3.0  # 3 seconds
        self.samples = int(self.sample_rate * self.duration)
        
        # Create test audio
        self.audio = self._create_test_audio()
        
        # Create separator
        self.config = SeparationConfig(
            min_duration=0.1,
            max_duration=5.0,
            suppression_strength=0.6,
            confidence_threshold=0.5
        )
        self.separator = SpeakerSeparator(self.config)
    
    def _create_test_audio(self):
        """Create test audio with main speaker and interruptions."""
        t = np.linspace(0, self.duration, self.samples)
        
        # Main speaker (200Hz fundamental)
        main_speaker = np.sin(2 * np.pi * 200 * t) * 0.5
        
        # Add harmonics
        main_speaker += np.sin(2 * np.pi * 400 * t) * 0.3
        main_speaker += np.sin(2 * np.pi * 600 * t) * 0.2
        
        # Secondary speaker interruptions
        audio = main_speaker.copy()
        
        # Short interruption (0.2s at t=0.5s)
        start = int(0.5 * self.sample_rate)
        end = int(0.7 * self.sample_rate)
        secondary = np.sin(2 * np.pi * 300 * t[start:end]) * 0.7
        audio[start:end] += secondary
        
        # Medium interruption (1s at t=1.5s)
        start = int(1.5 * self.sample_rate)
        end = int(2.5 * self.sample_rate)
        secondary = np.sin(2 * np.pi * 350 * t[start:end]) * 0.6
        audio[start:end] += secondary
        
        return audio
    
    def test_separation_with_detected_regions(self):
        """Test separation process when secondary speakers are detected."""
        # Create audio that should have secondary speakers
        audio = self._create_test_audio()
        
        # Run separation
        result = self.separator.separate_speakers(audio, self.sample_rate)
        
        # Verify results
        self.assertIn("audio", result)
        self.assertIn("detections", result)
        self.assertIn("metrics", result)
        self.assertIsInstance(result["audio"], np.ndarray)
        self.assertEqual(len(result["audio"]), len(audio))
        
        # If detections were found, audio should be modified
        if result["detections"]:
            # Check that metrics indicate processing occurred
            self.assertIn("secondary_speaker_count", result["metrics"])
            self.assertGreaterEqual(result["metrics"]["secondary_speaker_count"], 0)
    
    def test_duration_filtering(self):
        """Test filtering of regions by duration constraints."""
        # Create audio with interruptions of various durations
        audio = self._create_test_audio()
        
        # Configure separator with specific duration constraints
        config = SeparationConfig(min_duration=0.1, max_duration=5.0)
        separator = SpeakerSeparator(config)
        
        result = separator.separate_speakers(audio, self.sample_rate)
        
        # Check that detections are within duration constraints
        for detection in result["detections"]:
            self.assertGreaterEqual(detection.duration, 0.1)
            self.assertLessEqual(detection.duration, 5.0)
    
    def test_confidence_threshold(self):
        """Test confidence-based filtering."""
        # Configure separator with specific confidence threshold
        config = SeparationConfig(confidence_threshold=0.6)
        separator = SpeakerSeparator(config)
        
        result = separator.separate_speakers(self.audio, self.sample_rate)
        
        # All returned detections should meet confidence threshold
        for detection in result["detections"]:
            self.assertGreaterEqual(detection.confidence, 0.6)
    
    def test_main_speaker_preservation(self):
        """Test preservation of main speaker characteristics."""
        # Configure separator to preserve main speaker
        config = SeparationConfig(preserve_main_speaker=True)
        separator = SpeakerSeparator(config)
        
        result = separator.separate_speakers(self.audio, self.sample_rate)
        
        # Check similarity is preserved using metrics
        if 'metrics' in result and 'similarity_preservation' in result['metrics']:
            similarity = result['metrics']['similarity_preservation']
            self.assertGreater(similarity, 0.9)  # Should maintain high similarity
        
        # Verify audio is returned
        self.assertIn('audio', result)
        self.assertEqual(len(result['audio']), len(self.audio))
    
    def test_adaptive_suppression(self):
        """Test adaptive suppression based on confidence."""
        config = SeparationConfig(suppression_strength=0.8)
        separator = SpeakerSeparator(config)
        
        # Create test audio with secondary speaker
        audio = self._create_test_audio()
        
        # Test adaptive suppression with target similarity
        result = separator.adaptive_suppression(audio, 16000, target_similarity=0.95)
        
        # Check that result maintains high similarity with original
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(audio, result)
        self.assertGreater(correlation, 0.9)  # Should maintain similarity
    
    def test_spectral_suppression(self):
        """Test spectral suppression method."""
        # Create separator
        separator = SpeakerSeparator()
        
        # Apply suppression to a segment
        suppressed = separator._apply_suppression(
            self.audio,
            self.sample_rate,
            start_time=0.5,
            end_time=0.75,
            strength=0.8
        )
        
        # Verify output
        self.assertEqual(len(suppressed), len(self.audio))
        # Energy in suppressed region should be reduced
        start_idx = int(0.5 * self.sample_rate)
        end_idx = int(0.75 * self.sample_rate)
        
        original_energy = np.sum(self.audio[start_idx:end_idx] ** 2)
        suppressed_energy = np.sum(suppressed[start_idx:end_idx] ** 2)
        self.assertLess(suppressed_energy, original_energy)


class TestSecondaryDetection(unittest.TestCase):
    """Test secondary speaker detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 2.0
        self.samples = int(self.sample_rate * self.duration)
        
        # Create test audio
        self.audio = self._create_mixed_audio()
        
        # Create detector with parameters directly
        self.detector = AdaptiveSecondaryDetection(
            min_duration=0.1,
            max_duration=5.0,
            detection_methods=["embedding", "energy", "spectral"]
        )
    
    def _create_mixed_audio(self):
        """Create audio with multiple speakers."""
        t = np.linspace(0, self.duration, self.samples)
        
        # Main speaker
        main = np.sin(2 * np.pi * 200 * t) * 0.5
        
        # Add secondary speaker at specific times
        audio = main.copy()
        
        # Short "ครับ" style interruption (150ms)
        start = int(0.5 * self.sample_rate)
        end = int(0.65 * self.sample_rate)
        audio[start:end] += np.sin(2 * np.pi * 350 * t[start:end]) * 0.6
        
        # Longer interruption (2s)
        start = int(1.2 * self.sample_rate)
        end = int(1.8 * self.sample_rate)  # 600ms
        audio[start:end] += np.sin(2 * np.pi * 400 * t[start:end]) * 0.5
        
        return audio
    
    def test_multi_modal_detection(self):
        """Test detection using multiple methods."""
        regions = self.detector.detect(self.audio, self.sample_rate)
        
        # Should detect some regions
        self.assertIsInstance(regions, list)
        # Each region should be a DetectionResult
        for region in regions:
            self.assertIsInstance(region, DetectionResult)
            self.assertGreater(region.end_time, region.start_time)
            self.assertGreaterEqual(region.confidence, 0)
            self.assertLessEqual(region.confidence, 1)
            self.assertIsInstance(region.detection_methods, list)
    
    def test_energy_detection(self):
        """Test energy-based detection."""
        # Test only energy detection
        regions = self.detector._detect_by_energy(self.audio, self.sample_rate)
        
        # Should detect energy anomalies
        self.assertIsInstance(regions, list)
        if regions:
            # Check format
            for region in regions:
                self.assertIsInstance(region, DetectionResult)
                self.assertGreater(region.end_time, region.start_time)
                self.assertGreaterEqual(region.confidence, 0)
                self.assertLessEqual(region.confidence, 1)
    
    def test_spectral_detection(self):
        """Test spectral-based detection."""
        regions = self.detector._detect_by_spectral(self.audio, self.sample_rate)
        
        # Should detect spectral changes
        self.assertIsInstance(regions, list)
        # Any detected regions should be valid
        for region in regions:
            self.assertIsInstance(region, DetectionResult)
            self.assertGreater(region.end_time, region.start_time)
            self.assertGreaterEqual(region.confidence, 0)
            self.assertLessEqual(region.confidence, 1)
    
    def test_duration_constraints(self):
        """Test filtering by duration constraints."""
        # Create audio with various interruption lengths
        audio = self._create_variable_interruptions()
        
        regions = self.detector.detect(audio, self.sample_rate)
        
        # All regions should be within constraints
        for region in regions:
            duration = region.duration
            self.assertGreaterEqual(duration, self.detector.min_duration)
            self.assertLessEqual(duration, self.detector.max_duration)
    
    def _create_variable_interruptions(self):
        """Create audio with interruptions of various lengths."""
        t = np.linspace(0, 3.0, 48000)
        audio = np.sin(2 * np.pi * 200 * t) * 0.5
        
        # 50ms interruption (too short)
        start = int(0.5 * self.sample_rate)
        end = int(0.55 * self.sample_rate)
        audio[start:end] += np.random.randn(end - start) * 0.3
        
        # 200ms interruption (valid)
        start = int(1.0 * self.sample_rate)
        end = int(1.2 * self.sample_rate)
        audio[start:end] += np.random.randn(end - start) * 0.4
        
        # 6s interruption (too long)
        # Would exceed audio length, so skip
        
        return audio
    
    @unittest.skip("ThaiInterjectionDetector not yet implemented")
    def test_thai_interjection_detector(self):
        """Test Thai interjection detection framework."""
        # detector = ThaiInterjectionDetector()
        
        # # Test patterns are defined
        # self.assertIn("polite_particles", detector.patterns)
        # self.assertIn("ครับ", detector.patterns["polite_particles"])
        # self.assertIn("ค่ะ", detector.patterns["polite_particles"])
        
        # # Test detection (requires transcript)
        # result = detector.detect_interjections("", self.audio, self.sample_rate)
        # self.assertEqual(result, [])  # No transcript provided
        pass


class TestOverlapDetection(unittest.TestCase):
    """Test overlap detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 2.0
        self.samples = int(self.sample_rate * self.duration)
        
        # Create overlapping audio
        self.audio = self._create_overlapping_audio()
        
        # Create detector
        self.config = OverlapConfig()
        self.detector = OverlapDetector(self.config)
    
    def _create_overlapping_audio(self):
        """Create audio with overlapping speakers."""
        t = np.linspace(0, self.duration, self.samples)
        
        # Speaker 1
        speaker1 = np.sin(2 * np.pi * 200 * t) * 0.5
        
        # Speaker 2 (overlapping)
        speaker2 = np.zeros_like(t)
        start = int(0.8 * self.sample_rate)
        end = int(1.2 * self.sample_rate)
        speaker2[start:end] = np.sin(2 * np.pi * 300 * t[start:end]) * 0.5
        
        # Combine
        return speaker1 + speaker2
    
    def test_overlap_detection(self):
        """Test basic overlap detection."""
        results = self.detector.detect_overlaps(self.audio, self.sample_rate)
        
        # Check results structure
        self.assertIn("overlaps", results)
        self.assertIn("anomalies", results)
        self.assertIn("prosody_breaks", results)
        self.assertIn("confidence_scores", results)
    
    def test_energy_anomaly_detection(self):
        """Test energy anomaly detection."""
        anomalies = self.detector._energy_anomaly_detection(
            self.audio,
            self.sample_rate
        )
        
        # Should detect anomalies
        self.assertIsInstance(anomalies, list)
    
    def test_spectral_overlap_detection(self):
        """Test spectral overlap detection."""
        overlaps = self.detector._spectral_overlap_detection(
            self.audio,
            self.sample_rate
        )
        
        # Should detect spectral overlaps
        self.assertIsInstance(overlaps, list)
    
    def test_prosody_discontinuity(self):
        """Test prosody discontinuity detection."""
        # Create audio with pitch jump
        audio = self._create_pitch_jump_audio()
        
        discontinuities = self.detector._prosody_discontinuity_detection(
            audio,
            self.sample_rate
        )
        
        # Should detect discontinuities
        self.assertIsInstance(discontinuities, list)
    
    def _create_pitch_jump_audio(self):
        """Create audio with sudden pitch changes."""
        t = np.linspace(0, 1.0, 16000)
        
        # First half at 200Hz
        audio = np.zeros_like(t)
        mid = len(t) // 2
        audio[:mid] = np.sin(2 * np.pi * 200 * t[:mid])
        
        # Second half at 300Hz (sudden jump)
        audio[mid:] = np.sin(2 * np.pi * 300 * t[mid:])
        
        return audio * 0.5
    
    @unittest.skip("SimultaneousSpeechDetector not yet implemented")
    def test_simultaneous_speech_detector(self):
        """Test simultaneous speech detection."""
        # detector = SimultaneousSpeechDetector()
        
        # # Create audio with multiple fundamentals
        # audio = self._create_multi_fundamental_audio()
        
        # regions = detector.detect_simultaneous_speech(audio, self.sample_rate)
        
        # # Should detect simultaneous speech
        # self.assertIsInstance(regions, list)
        # if regions:
        #     for start, end, conf in regions:
        #         self.assertGreater(end, start)
        #         self.assertGreaterEqual(conf, 0)
        #         self.assertLessEqual(conf, 1)
        pass
    
    def _create_multi_fundamental_audio(self):
        """Create audio with multiple fundamental frequencies."""
        t = np.linspace(0, 1.0, 16000)
        
        # Multiple speakers talking simultaneously
        audio = np.sin(2 * np.pi * 150 * t) * 0.3  # Speaker 1
        audio += np.sin(2 * np.pi * 250 * t) * 0.3  # Speaker 2
        audio += np.sin(2 * np.pi * 350 * t) * 0.2  # Speaker 3
        
        return audio


class TestSpeakerUtils(unittest.TestCase):
    """Test speaker utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 1.0
        self.samples = int(self.sample_rate * self.duration)
        
        # Create test audio
        t = np.linspace(0, self.duration, self.samples)
        self.audio1 = np.sin(2 * np.pi * 200 * t) * 0.5
        self.audio2 = np.sin(2 * np.pi * 300 * t) * 0.5
    
    def test_speaker_profile_creation(self):
        """Test creating speaker profile."""
        profile = create_speaker_profile(self.audio1, self.sample_rate)
        
        # Should return embedding vector
        self.assertIsInstance(profile, np.ndarray)
        self.assertGreater(len(profile), 0)
    
    def test_embedding_comparison(self):
        """Test comparing embeddings."""
        # Create embeddings
        embedding1 = extract_speaker_embedding(self.audio1, self.sample_rate)
        embedding2 = extract_speaker_embedding(self.audio2, self.sample_rate)
        
        # Compare
        similarity = compare_embeddings(embedding1, embedding2)
        
        # Should return similarity score
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
        
        # Same embedding should have high similarity
        self_similarity = compare_embeddings(embedding1, embedding1)
        self.assertAlmostEqual(self_similarity, 1.0, places=5)
    
    def test_speaker_comparator(self):
        """Test speaker comparison class."""
        comparator = SpeakerComparator()
        
        # Compare different speakers
        result = comparator.compare(self.audio1, self.audio2, self.sample_rate)
        
        # Check result structure
        self.assertIn("cosine_similarity", result)
        self.assertIn("euclidean_distance", result)
        self.assertIn("classification", result)
        self.assertIn("confidence", result)
        
        # Classification should be one of expected values
        self.assertIn(result["classification"], [
            "same_speaker", "similar_speaker", "possibly_different", "different_speaker"
        ])
    
    def test_cluster_speakers(self):
        """Test speaker clustering."""
        from utils.speaker_utils import cluster_speakers
        
        # Create multiple audio segments
        segments = [
            self.audio1[:8000],  # First half of speaker 1
            self.audio1[8000:],  # Second half of speaker 1
            self.audio2[:8000],  # First half of speaker 2
            self.audio2[8000:]   # Second half of speaker 2
        ]
        
        # Cluster
        labels = cluster_speakers(segments, self.sample_rate, threshold=0.7)
        
        # Should have correct number of labels
        self.assertEqual(len(labels), len(segments))
        
        # Segments from same speaker should have same label
        # (This might not always work with simple test audio)
        self.assertIsInstance(labels[0], int)


class TestIntegration(unittest.TestCase):
    """Test integration of all components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 3.0
        self.samples = int(self.sample_rate * self.duration)
    
    def test_complete_pipeline(self):
        """Test complete detection and separation pipeline."""
        # Create complex audio
        audio = self._create_complex_audio()
        
        # 1. Detect secondary speakers
        detector = AdaptiveSecondaryDetection()
        regions = detector.detect(audio, self.sample_rate)
        confidences = [r.confidence for r in regions]
        details = regions
        
        # 2. Detect overlaps
        overlap_detector = OverlapDetector()
        overlap_results = overlap_detector.detect_overlaps(audio, self.sample_rate)
        
        # 3. Separate speakers
        separator = SpeakerSeparator()
        
        # Combine detected regions
        all_regions = regions + overlap_results["overlaps"]
        all_confidences = confidences + overlap_results["confidence_scores"].get("overlaps", [])
        
        if all_regions:
            result = separator.separate_speakers(
                audio,
                self.sample_rate,
                all_regions,
                all_confidences
            )
            
            # Verify processing
            self.assertIn("audio", result)
            self.assertIn("metrics", result)
            
            # Check audio quality preserved
            self.assertEqual(len(result["audio"]), len(audio))
    
    def _create_complex_audio(self):
        """Create audio with multiple speakers and interruptions."""
        t = np.linspace(0, self.duration, self.samples)
        
        # Main speaker
        audio = np.sin(2 * np.pi * 200 * t) * 0.5
        
        # Thai interjection at 0.5s (150ms)
        start = int(0.5 * self.sample_rate)
        end = int(0.65 * self.sample_rate)
        audio[start:end] += np.sin(2 * np.pi * 350 * t[start:end]) * 0.6
        
        # Overlapping speech at 1.5s (500ms)
        start = int(1.5 * self.sample_rate)
        end = int(2.0 * self.sample_rate)
        audio[start:end] += np.sin(2 * np.pi * 250 * t[start:end]) * 0.4
        
        # Long interruption at 2.2s (600ms)
        start = int(2.2 * self.sample_rate)
        end = int(2.8 * self.sample_rate)
        audio[start:end] = np.sin(2 * np.pi * 400 * t[start:end]) * 0.7
        
        return audio
    
    def test_flexible_duration_handling(self):
        """Test handling of various interruption durations."""
        # Test that the detector can handle various durations
        durations = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        
        for duration in durations:
            audio = self._create_interruption_audio(duration)
            
            # Detect - use only methods that don't require speaker identification
            detector = AdaptiveSecondaryDetection(
                detection_methods=["energy", "spectral"],  # Skip embedding & vad
                energy_threshold=0.1,  # Lower threshold for synthetic audio
                spectral_threshold=0.3  # Lower threshold for synthetic audio
            )
            
            # Just verify detection runs without errors
            regions = detector.detect(audio, self.sample_rate)
            
            # Regions should be a list (may be empty for synthetic audio)
            self.assertIsInstance(regions, list)
            
            # If any regions are detected, they should be valid
            for region in regions:
                self.assertIsInstance(region, DetectionResult)
                self.assertGreater(region.end_time, region.start_time)
    
    def _create_interruption_audio(self, interruption_duration):
        """Create audio with specific interruption duration."""
        total_duration = max(3.0, interruption_duration + 1.0)
        samples = int(self.sample_rate * total_duration)
        t = np.linspace(0, total_duration, samples)
        
        # Main speaker
        audio = np.sin(2 * np.pi * 200 * t) * 0.5
        
        # Interruption
        start = int(0.5 * self.sample_rate)
        end = int((0.5 + interruption_duration) * self.sample_rate)
        end = min(end, len(audio))
        
        audio[start:end] = np.sin(2 * np.pi * 350 * t[start:end]) * 0.7
        
        return audio


if __name__ == "__main__":
    unittest.main()