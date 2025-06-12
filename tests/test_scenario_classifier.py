"""Test suite for Scenario Classifier module (S03_T08).

This test suite validates the Scenario Classifier with the following requirements:
1. Audio scenario detection and classification
2. Multi-speaker scenario identification
3. Acoustic environment classification
4. Signal quality assessment
5. Real-time classification capability
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings
from processors.audio_enhancement.classification.scenario_classifier import (
    ScenarioClassifier,
    AudioScenario,
    ScenarioFeatures,
    ClassificationResult,
    ClassifierConfig
)


class TestScenarioClassifier(unittest.TestCase):
    """Test suite for Scenario Classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.classifier = ScenarioClassifier(sample_rate=self.sample_rate)
        
        # Suppress warnings for optional dependencies
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def test_single_speaker_detection(self):
        """Test 1.1: Detect single speaker scenarios."""
        # Create single speaker audio
        duration = 5.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Single modulated tone (speech-like)
        audio = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        audio += 0.05 * np.random.randn(samples)  # Light background noise
        
        # Classify scenario
        result = self.classifier.classify_scenario(audio)
        
        # Verify single speaker detection
        self.assertEqual(result.scenario, AudioScenario.SINGLE_SPEAKER)
        self.assertEqual(result.estimated_speakers, 1)
        self.assertGreater(result.confidence, 0.7)
        self.assertGreater(result.features.speech_ratio, 0.6)
    
    def test_multi_speaker_detection(self):
        """Test 1.2: Detect multi-speaker scenarios."""
        # Create multi-speaker audio
        duration = 5.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Two speakers with different characteristics
        speaker1 = np.sin(2 * np.pi * 250 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        speaker2 = np.sin(2 * np.pi * 400 * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))
        
        # Simulate alternating speakers
        audio = np.zeros_like(t)
        segment_size = len(t) // 4
        audio[:segment_size] = speaker1[:segment_size]
        audio[segment_size:2*segment_size] = speaker2[:segment_size]
        audio[2*segment_size:3*segment_size] = speaker1[:segment_size]
        audio[3*segment_size:] = speaker2[:len(t)-3*segment_size]
        
        # Classify scenario
        result = self.classifier.classify_scenario(audio)
        
        # Verify multi-speaker detection
        self.assertEqual(result.scenario, AudioScenario.MULTI_SPEAKER)
        self.assertGreaterEqual(result.estimated_speakers, 2)
        self.assertGreater(result.confidence, 0.6)
        self.assertGreater(result.features.speaker_change_rate, 0.5)
    
    def test_overlapping_speech_detection(self):
        """Test 1.3: Detect overlapping speech scenarios."""
        # Create overlapping speech
        duration = 4.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Two speakers speaking simultaneously
        speaker1 = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        speaker2 = np.sin(2 * np.pi * 450 * t) * (1 + 0.2 * np.sin(2 * np.pi * 6 * t))
        
        # Mix speakers with overlap
        audio = 0.7 * speaker1 + 0.5 * speaker2
        audio += 0.03 * np.random.randn(samples)
        
        # Classify scenario
        result = self.classifier.classify_scenario(audio)
        
        # Verify overlapping speech detection
        self.assertEqual(result.scenario, AudioScenario.OVERLAPPING_SPEECH)
        self.assertGreaterEqual(result.estimated_speakers, 2)
        self.assertGreater(result.features.spectral_complexity, 0.7)
    
    def test_music_detection(self):
        """Test 1.4: Detect music scenarios."""
        # Create music-like audio (harmonic content)
        duration = 4.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Music with harmonics
        fundamental = 220  # A3
        music = np.zeros_like(t)
        for harmonic in [1, 2, 3, 4, 5]:
            amplitude = 1.0 / harmonic
            music += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)
        
        # Add some rhythm
        rhythm = 1 + 0.3 * np.sin(2 * np.pi * 2 * t)  # 2 Hz rhythm
        music = music * rhythm
        
        # Classify scenario
        result = self.classifier.classify_scenario(music)
        
        # Verify music detection
        self.assertEqual(result.scenario, AudioScenario.MUSIC)
        self.assertGreater(result.features.harmonic_ratio, 0.2)
        self.assertGreater(result.features.rhythm_strength, 0.6)
    
    def test_noise_detection(self):
        """Test 1.5: Detect noise-only scenarios."""
        # Create different types of noise
        duration = 3.0
        samples = int(duration * self.sample_rate)
        
        # White noise
        white_noise = np.random.randn(samples) * 0.5
        
        # Classify white noise
        result = self.classifier.classify_scenario(white_noise)
        
        # Verify noise detection
        self.assertEqual(result.scenario, AudioScenario.NOISE)
        self.assertEqual(result.estimated_speakers, 0)
        self.assertLess(result.features.speech_ratio, 0.2)
        self.assertGreater(result.features.noise_ratio, 0.7)
    
    def test_silence_detection(self):
        """Test 1.6: Detect silence scenarios."""
        # Create near-silence
        duration = 2.0
        samples = int(duration * self.sample_rate)
        silence = np.random.randn(samples) * 0.001  # Very quiet noise
        
        # Classify silence
        result = self.classifier.classify_scenario(silence)
        
        # Verify silence detection
        self.assertEqual(result.scenario, AudioScenario.SILENCE)
        self.assertEqual(result.estimated_speakers, 0)
        self.assertLess(result.features.energy_level, 0.01)
        self.assertGreater(result.features.silence_ratio, 0.9)
    
    def test_mixed_content_detection(self):
        """Test 1.7: Detect mixed content scenarios."""
        # Create mixed content (speech + music + noise)
        duration = 6.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Speech component
        speech = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        
        # Music component (background)
        music = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
        
        # Noise component
        noise = 0.1 * np.random.randn(samples)
        
        # Mix components
        audio = speech + music + noise
        
        # Classify scenario
        result = self.classifier.classify_scenario(audio)
        
        # Verify mixed content detection
        self.assertEqual(result.scenario, AudioScenario.MIXED_CONTENT)
        self.assertGreaterEqual(result.estimated_speakers, 1)
        self.assertGreater(result.features.spectral_complexity, 0.6)
    
    def test_feature_extraction(self):
        """Test 2.1: Comprehensive feature extraction."""
        # Create test audio with known characteristics
        duration = 4.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Speech-like signal with clear characteristics
        audio = np.sin(2 * np.pi * 300 * t) * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))
        audio += 0.05 * np.random.randn(samples)
        
        # Extract features
        features = self.classifier.extract_features(audio)
        
        # Verify feature types and ranges
        self.assertIsInstance(features, ScenarioFeatures)
        
        # Energy features
        self.assertGreaterEqual(features.energy_level, 0.0)
        self.assertLessEqual(features.energy_level, 1.0)
        
        # Speech features
        self.assertGreaterEqual(features.speech_ratio, 0.0)
        self.assertLessEqual(features.speech_ratio, 1.0)
        
        # Spectral features
        self.assertGreaterEqual(features.spectral_centroid, 0.0)
        self.assertGreaterEqual(features.spectral_bandwidth, 0.0)
        self.assertGreaterEqual(features.spectral_complexity, 0.0)
        self.assertLessEqual(features.spectral_complexity, 1.0)
        
        # Temporal features
        self.assertGreaterEqual(features.zero_crossing_rate, 0.0)
        self.assertGreaterEqual(features.temporal_variation, 0.0)
        
        # Speaker features
        self.assertGreaterEqual(features.speaker_change_rate, 0.0)
        self.assertGreaterEqual(features.pitch_variation, 0.0)
    
    def test_confidence_scoring(self):
        """Test 2.2: Confidence scoring accuracy."""
        # Test clear scenarios with high expected confidence
        duration = 3.0
        samples = int(duration * self.sample_rate)
        
        # Clear single speaker
        clear_speech = np.sin(2 * np.pi * 300 * np.linspace(0, duration, samples))
        clear_speech *= (1 + 0.3 * np.sin(2 * np.pi * 4 * np.linspace(0, duration, samples)))
        result_clear = self.classifier.classify_scenario(clear_speech)
        
        # Clear noise
        clear_noise = np.random.randn(samples) * 0.8
        result_noise = self.classifier.classify_scenario(clear_noise)
        
        # Clear silence
        clear_silence = np.random.randn(samples) * 0.001
        result_silence = self.classifier.classify_scenario(clear_silence)
        
        # Verify high confidence for clear scenarios
        self.assertGreater(result_clear.confidence, 0.7,
                          "Clear speech should have high confidence")
        self.assertGreater(result_noise.confidence, 0.7,
                          "Clear noise should have high confidence")
        self.assertGreater(result_silence.confidence, 0.8,
                          "Clear silence should have very high confidence")
        
        # Test ambiguous scenario with lower expected confidence
        ambiguous = clear_speech * 0.5 + clear_noise * 0.3
        result_ambiguous = self.classifier.classify_scenario(ambiguous)
        
        self.assertLess(result_ambiguous.confidence, result_clear.confidence,
                       "Ambiguous scenario should have lower confidence")
    
    def test_real_time_performance(self):
        """Test 2.3: Real-time classification performance."""
        import time
        
        # Test various audio lengths
        test_durations = [1.0, 5.0, 10.0, 30.0]
        
        for duration in test_durations:
            samples = int(duration * self.sample_rate)
            audio = np.random.randn(samples) * 0.3
            
            # Measure processing time
            start_time = time.time()
            result = self.classifier.classify_scenario(audio)
            processing_time = time.time() - start_time
            
            # Should be much faster than real-time
            max_allowed_time = duration * 0.1  # 10% of audio duration
            self.assertLess(processing_time, max_allowed_time,
                           f"Processing {duration}s audio took {processing_time:.3f}s "
                           f"(max allowed: {max_allowed_time:.3f}s)")
            
            # Verify result validity
            self.assertIsInstance(result.scenario, AudioScenario)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
    
    def test_batch_classification(self):
        """Test 2.4: Batch classification of multiple audio segments."""
        # Create batch of different scenarios
        duration = 2.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Generate different scenario types
        scenarios = []
        expected_types = []
        
        # Single speaker
        speech = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
        scenarios.append(speech)
        expected_types.append(AudioScenario.SINGLE_SPEAKER)
        
        # Noise
        noise = np.random.randn(samples) * 0.5
        scenarios.append(noise)
        expected_types.append(AudioScenario.NOISE)
        
        # Music
        music = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        scenarios.append(music)
        expected_types.append(AudioScenario.MUSIC)
        
        # Batch classify
        results = self.classifier.classify_batch(scenarios)
        
        # Verify batch results
        self.assertEqual(len(results), len(scenarios))
        
        for i, (result, expected) in enumerate(zip(results, expected_types)):
            self.assertEqual(result.scenario, expected,
                           f"Batch item {i}: expected {expected}, got {result.scenario}")
            self.assertGreater(result.confidence, 0.5,
                             f"Batch item {i}: confidence too low")
    
    def test_streaming_classification(self):
        """Test 2.5: Streaming classification for continuous audio."""
        # Create streaming classifier
        streaming_classifier = ScenarioClassifier(
            sample_rate=self.sample_rate,
            streaming_mode=True,
            window_duration=3.0,
            update_interval=1.0
        )
        
        # Simulate streaming audio with changing scenarios
        chunk_duration = 1.0
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        # Scenario sequence: speech -> music -> noise
        scenarios = ["speech", "music", "noise"]
        
        for i, scenario_type in enumerate(scenarios):
            t = np.linspace(0, chunk_duration, chunk_samples)
            
            if scenario_type == "speech":
                chunk = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
            elif scenario_type == "music":
                chunk = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
            else:  # noise
                chunk = np.random.randn(chunk_samples) * 0.5
            
            # Update streaming classifier
            result = streaming_classifier.update_streaming(
                audio_chunk=chunk,
                timestamp=i * chunk_duration
            )
            
            # After sufficient context, should detect scenario
            if i >= 2:  # After 3 chunks
                self.assertIsNotNone(result)
                self.assertIsInstance(result.scenario, AudioScenario)
    
    def test_scenario_transitions(self):
        """Test 2.6: Detection of scenario transitions."""
        # Create audio with clear transitions
        duration = 9.0
        samples = int(duration * self.sample_rate)
        
        # Create three segments with different scenarios
        segment_samples = samples // 3
        audio = np.zeros(samples)
        
        # Segment 1: Speech
        t1 = np.linspace(0, 3.0, segment_samples)
        speech = np.sin(2 * np.pi * 300 * t1) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t1))
        audio[:segment_samples] = speech
        
        # Segment 2: Music
        t2 = np.linspace(0, 3.0, segment_samples)
        music = np.sin(2 * np.pi * 440 * t2) + 0.5 * np.sin(2 * np.pi * 880 * t2)
        audio[segment_samples:2*segment_samples] = music
        
        # Segment 3: Noise
        noise = np.random.randn(segment_samples) * 0.5
        audio[2*segment_samples:] = noise
        
        # Detect transitions
        transitions = self.classifier.detect_transitions(
            audio=audio,
            segment_duration=1.5  # 1.5 second windows
        )
        
        # Should detect at least one transition
        self.assertGreater(len(transitions), 0)
        
        # Verify transition timing
        for transition in transitions:
            self.assertIn("timestamp", transition)
            self.assertIn("from_scenario", transition)
            self.assertIn("to_scenario", transition)
            self.assertIn("confidence", transition)
    
    def test_acoustic_environment_classification(self):
        """Test 3.1: Acoustic environment classification."""
        duration = 4.0
        samples = int(duration * self.sample_rate)
        
        # Test different acoustic environments
        environments = {
            "clean": lambda: 0.01 * np.random.randn(samples),
            "reverberant": lambda: self._add_reverb(np.random.randn(samples) * 0.3),
            "noisy": lambda: np.random.randn(samples) * 0.8,
            "outdoor": lambda: self._simulate_outdoor_noise(samples)
        }
        
        for env_name, env_generator in environments.items():
            # Add speech to environment
            t = np.linspace(0, duration, samples)
            speech = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
            
            # Generate environment
            env_audio = speech + env_generator()
            
            # Classify with environment analysis
            result = self.classifier.classify_scenario(
                env_audio,
                analyze_environment=True
            )
            
            # Check environment information is included
            self.assertIn("acoustic_environment", result.metadata)
            env_info = result.metadata["acoustic_environment"]
            
            self.assertIn("reverb_level", env_info)
            self.assertIn("noise_level", env_info)
            self.assertIn("snr_estimate", env_info)
    
    def test_configuration_options(self):
        """Test 3.2: Different configuration options."""
        # Test with different sensitivity settings
        configs = [
            ClassifierConfig(sensitivity="high", min_confidence=0.6),
            ClassifierConfig(sensitivity="medium", min_confidence=0.5),
            ClassifierConfig(sensitivity="low", min_confidence=0.3)
        ]
        
        # Create borderline speech signal
        duration = 3.0
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Weak speech signal
        weak_speech = 0.3 * np.sin(2 * np.pi * 300 * t) * (1 + 0.2 * np.sin(2 * np.pi * 4 * t))
        weak_speech += 0.2 * np.random.randn(samples)
        
        results = []
        for config in configs:
            classifier = ScenarioClassifier(
                sample_rate=self.sample_rate,
                config=config
            )
            result = classifier.classify_scenario(weak_speech)
            results.append(result)
        
        # High sensitivity should be more likely to detect speech
        # Low sensitivity should be more conservative
        high_sens_result = results[0]
        low_sens_result = results[2]
        
        # At minimum, results should be valid
        for result in results:
            self.assertIsInstance(result.scenario, AudioScenario)
            self.assertGreaterEqual(result.confidence, 0.0)
    
    def test_edge_cases(self):
        """Test 4: Edge cases and error handling."""
        # Test 4.1: Very short audio
        short_audio = np.random.randn(self.sample_rate // 10)  # 0.1 seconds
        result = self.classifier.classify_scenario(short_audio)
        self.assertIsNotNone(result)
        
        # Test 4.2: Empty audio
        with self.assertRaises(ValueError):
            self.classifier.classify_scenario(np.array([]))
        
        # Test 4.3: NaN values in audio
        nan_audio = np.full(self.sample_rate, np.nan)
        with self.assertRaises(ValueError):
            self.classifier.classify_scenario(nan_audio)
        
        # Test 4.4: Extremely loud audio
        loud_audio = np.random.randn(self.sample_rate) * 100
        result = self.classifier.classify_scenario(loud_audio)
        self.assertIsNotNone(result)  # Should handle with normalization
        
        # Test 4.5: DC offset
        dc_audio = np.random.randn(self.sample_rate) + 5.0  # Large DC offset
        result = self.classifier.classify_scenario(dc_audio)
        self.assertIsNotNone(result)  # Should handle with DC removal
    
    # Helper methods
    
    def _add_reverb(self, audio, decay=0.5, delay_samples=800):
        """Add simple reverb effect."""
        reverb_audio = audio.copy()
        for i in range(1, 4):  # Multiple reflections
            delay = delay_samples * i
            if delay < len(audio):
                amplitude = decay ** i
                reverb_audio[delay:] += amplitude * audio[:-delay]
        return reverb_audio
    
    def _simulate_outdoor_noise(self, samples):
        """Simulate outdoor acoustic environment."""
        # Low-frequency wind noise
        wind = np.random.randn(samples) * 0.1
        wind = np.convolve(wind, np.ones(50)/50, mode='same')  # Smooth
        
        # Occasional higher frequency events
        events = np.random.randn(samples) * 0.05
        events[np.random.rand(samples) > 0.95] *= 10  # Random spikes
        
        return wind + events


if __name__ == "__main__":
    unittest.main()