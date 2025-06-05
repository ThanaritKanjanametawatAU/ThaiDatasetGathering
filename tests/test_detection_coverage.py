#!/usr/bin/env python3
"""
Test to check detection coverage of secondary speakers.
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.speaker_separation import SpeakerSeparator, SeparationConfig
from processors.audio_enhancement.detection.secondary_speaker import SecondarySpeakerDetector, AdaptiveSecondaryDetection


class TestDetectionCoverage(unittest.TestCase):
    """Test detection coverage of secondary speakers."""
    
    def setUp(self):
        """Create test audio with clear secondary speaker."""
        self.sample_rate = 16000
        self.duration = 3.0
        
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Primary speaker throughout
        self.audio = np.sin(2 * np.pi * 200 * t) * 0.8
        
        # Add secondary speaker from 1.0 to 2.0 seconds
        self.secondary_start = 1.0
        self.secondary_end = 2.0
        start_idx = int(self.secondary_start * self.sample_rate)
        end_idx = int(self.secondary_end * self.sample_rate)
        
        # Make secondary speaker distinct (600Hz, slightly louder)
        secondary = np.sin(2 * np.pi * 600 * t[start_idx:end_idx]) * 0.9
        self.audio[start_idx:end_idx] += secondary
        
        # Normalize
        self.audio = self.audio / np.max(np.abs(self.audio)) * 0.8
    
    def test_detection_coverage(self):
        """Test how much of the secondary speaker is detected."""
        config = SeparationConfig(
            suppression_strength=0.95,
            confidence_threshold=0.3,
            detection_methods=["embedding", "vad", "energy", "spectral"]
        )
        
        separator = SpeakerSeparator(config)
        result = separator.separate_speakers(self.audio, self.sample_rate)
        
        print(f"\nExpected secondary speaker: {self.secondary_start:.1f}s - {self.secondary_end:.1f}s")
        print(f"Detected {len(result['detections'])} regions:")
        
        total_detected_duration = 0
        for i, det in enumerate(result['detections']):
            duration = det.end_time - det.start_time
            total_detected_duration += duration
            print(f"  {i+1}: {det.start_time:.3f}s - {det.end_time:.3f}s "
                  f"(duration: {duration:.3f}s, confidence: {det.confidence:.2f})")
        
        expected_duration = self.secondary_end - self.secondary_start
        coverage = total_detected_duration / expected_duration * 100
        
        print(f"\nTotal detected: {total_detected_duration:.3f}s")
        print(f"Expected: {expected_duration:.3f}s")
        print(f"Coverage: {coverage:.1f}%")
        
        # Check if detections actually cover the secondary speaker region
        covered_samples = np.zeros(len(self.audio), dtype=bool)
        for det in result['detections']:
            start_idx = int(det.start_time * self.sample_rate)
            end_idx = int(det.end_time * self.sample_rate)
            covered_samples[start_idx:end_idx] = True
        
        # Check coverage in the actual secondary speaker region
        secondary_start_idx = int(self.secondary_start * self.sample_rate)
        secondary_end_idx = int(self.secondary_end * self.sample_rate)
        secondary_covered = covered_samples[secondary_start_idx:secondary_end_idx]
        actual_coverage = np.mean(secondary_covered) * 100
        
        print(f"Actual coverage of secondary speaker region: {actual_coverage:.1f}%")
        
        self.assertGreater(coverage, 50, "Should detect at least 50% of secondary speaker")
    
    def test_detector_types(self):
        """Test different detector types individually."""
        print("\n=== Testing Individual Detectors ===")
        
        # Test base detector
        detector = SecondarySpeakerDetector()
        try:
            detections = detector.detect(self.audio, self.sample_rate)
            print(f"Base detector: {len(detections)} detections")
        except NotImplementedError:
            print("Base detector: Not implemented")
        
        # Test adaptive detector
        adaptive = AdaptiveSecondaryDetection()
        detections = adaptive.detect(self.audio, self.sample_rate)
        print(f"Adaptive detector: {len(detections)} detections")
        for i, det in enumerate(detections[:5]):
            print(f"  {det.start_time:.3f}s - {det.end_time:.3f}s "
                  f"(confidence: {det.confidence:.2f}, method: {det.detection_method})")
    
    def test_real_s5_detection(self):
        """Test detection on real S5 audio."""
        print("\n=== Testing Real S5 Detection ===")
        
        # Load S5
        from processors.gigaspeech2 import GigaSpeech2Processor
        
        config = {
            "name": "GigaSpeech2",
            "source": "speechcolab/gigaspeech2",
            "cache_dir": "./cache",
            "streaming": True,
        }
        
        processor = GigaSpeech2Processor(config)
        samples = list(processor.process_streaming(sample_mode=True, sample_size=5))
        
        if len(samples) >= 5:
            s5 = samples[4]
            audio_data = s5.get('audio', {})
            
            if isinstance(audio_data, dict) and 'array' in audio_data:
                audio = audio_data['array']
                sr = audio_data.get('sampling_rate', 16000)
                
                # Test detection
                separator = SpeakerSeparator(SeparationConfig(
                    confidence_threshold=0.2,  # Even lower threshold
                    min_duration=0.05,  # Shorter segments
                    detection_methods=["embedding", "vad", "energy", "spectral"]
                ))
                
                result = separator.separate_speakers(audio, sr)
                
                print(f"S5 detections: {len(result['detections'])}")
                total_duration = 0
                for det in result['detections']:
                    duration = det.end_time - det.start_time
                    total_duration += duration
                    print(f"  {det.start_time:.3f}s - {det.end_time:.3f}s "
                          f"(duration: {duration:.3f}s, confidence: {det.confidence:.2f})")
                
                print(f"Total detected duration: {total_duration:.3f}s")
                print(f"Audio duration: {len(audio)/sr:.3f}s")
                print(f"Coverage: {total_duration/(len(audio)/sr)*100:.1f}%")


if __name__ == '__main__':
    unittest.main()