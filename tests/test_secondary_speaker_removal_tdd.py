"""
TDD Test Suite for Secondary Speaker Removal
============================================

This test suite follows Test-Driven Development principles to ensure
secondary speakers are properly removed from audio files.

Based on best practices:
- Voice Activity Detection (VAD) for speech/silence detection
- Energy-based speaker change detection
- Spectral analysis for speaker identification
- End-of-utterance detection techniques
"""

import unittest
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import shutil


class TestSecondaryRemoval(unittest.TestCase):
    """Test secondary speaker removal functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.sample_rate = 16000
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def create_test_audio(self, duration: float, speaker_changes: list = None) -> np.ndarray:
        """
        Create test audio with multiple speakers.
        
        Args:
            duration: Total duration in seconds
            speaker_changes: List of (start_time, end_time, speaker_id, frequency)
        
        Returns:
            Audio array
        """
        samples = int(duration * self.sample_rate)
        audio = np.zeros(samples, dtype=np.float32)
        
        if speaker_changes:
            for start_time, end_time, speaker_id, freq in speaker_changes:
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                t = np.arange(end_sample - start_sample) / self.sample_rate
                
                # Create different frequency patterns for different speakers
                if speaker_id == 1:
                    # Primary speaker - lower frequency, consistent amplitude
                    signal = 0.3 * np.sin(2 * np.pi * freq * t)
                    # Add some harmonics for realism
                    signal += 0.1 * np.sin(2 * np.pi * freq * 2 * t)
                elif speaker_id == 2:
                    # Secondary speaker - higher frequency, different amplitude
                    signal = 0.5 * np.sin(2 * np.pi * freq * t)
                    # Different harmonic pattern
                    signal += 0.15 * np.sin(2 * np.pi * freq * 1.5 * t)
                else:
                    # Silence (speaker_id == 0)
                    signal = np.zeros(len(t))
                
                audio[start_sample:end_sample] = signal
        
        # Add small amount of noise for realism
        audio += 0.001 * np.random.randn(len(audio))
        
        return audio
    
    def test_detect_secondary_speaker_at_end(self):
        """Test detection of secondary speaker at the end of audio"""
        # Create audio with primary speaker for 3s, then secondary for 0.3s
        audio = self.create_test_audio(3.3, [
            (0.0, 3.0, 1, 200),    # Primary speaker
            (3.0, 3.3, 2, 400)     # Secondary speaker at end
        ])
        
        # Import the module we're testing (will fail initially)
        from processors.audio_enhancement.secondary_removal import SecondaryRemover
        
        remover = SecondaryRemover()
        result = remover.detect_secondary_speakers(audio, self.sample_rate)
        
        # Debug print
        print(f"\nDetection result: has_secondary={result.has_secondary_at_end}, "
              f"start_time={result.secondary_start_time:.2f}, "
              f"num_speakers={result.num_speakers}, "
              f"confidence={result.confidence:.2f}")
        
        # Should detect secondary speaker at end
        self.assertIsNotNone(result)
        self.assertTrue(result.has_secondary_at_end)
        # Allow some tolerance in detection timing (within 0.3s)
        self.assertGreaterEqual(result.secondary_start_time, 2.7)
        self.assertLessEqual(result.secondary_start_time, 3.3)
        self.assertEqual(result.num_speakers, 2)
    
    def test_remove_secondary_speaker_completely(self):
        """Test complete removal of secondary speaker"""
        # Create audio with secondary speaker at end
        audio = self.create_test_audio(3.3, [
            (0.0, 3.0, 1, 200),    # Primary speaker
            (3.0, 3.3, 2, 400)     # Secondary speaker
        ])
        
        from processors.audio_enhancement.secondary_removal import SecondaryRemover
        
        remover = SecondaryRemover()
        processed = remover.remove_secondary_speakers(audio, self.sample_rate)
        
        # Check that secondary speaker is removed
        last_300ms = processed[-int(0.3 * self.sample_rate):]
        max_amplitude = np.max(np.abs(last_300ms))
        
        # Should be completely silent (or very close to silent)
        self.assertLess(max_amplitude, 0.001, 
                       f"Secondary speaker not removed. Max amplitude: {max_amplitude}")
    
    def test_preserve_primary_speaker(self):
        """Test that primary speaker content is preserved"""
        # Create audio with only primary speaker
        audio = self.create_test_audio(3.0, [
            (0.0, 3.0, 1, 200)     # Only primary speaker
        ])
        
        from processors.audio_enhancement.secondary_removal import SecondaryRemover
        
        remover = SecondaryRemover()
        processed = remover.remove_secondary_speakers(audio, self.sample_rate)
        
        # Primary speaker should be preserved
        # Check energy is maintained
        original_energy = np.sqrt(np.mean(audio**2))
        processed_energy = np.sqrt(np.mean(processed**2))
        
        # Should maintain at least 90% of original energy
        self.assertGreater(processed_energy / original_energy, 0.9,
                          "Primary speaker content was damaged")
    
    def test_vad_based_detection(self):
        """Test Voice Activity Detection for speech/silence segments"""
        # Create audio with speech and silence
        audio = self.create_test_audio(4.0, [
            (0.0, 1.0, 1, 200),    # Speech
            (1.0, 2.0, 0, 0),      # Silence (speaker_id=0)
            (2.0, 3.5, 1, 200),    # Speech
            (3.5, 4.0, 2, 400)     # Different speaker
        ])
        
        from processors.audio_enhancement.secondary_removal import VoiceActivityDetector
        
        vad = VoiceActivityDetector()
        segments = vad.detect_speech_segments(audio, self.sample_rate)
        
        # Should detect 3 speech segments
        self.assertEqual(len(segments), 3)
        
        # Check segment timings
        self.assertAlmostEqual(segments[0].start_time, 0.0, places=1)
        self.assertAlmostEqual(segments[0].end_time, 1.0, places=1)
        self.assertAlmostEqual(segments[1].start_time, 2.0, places=1)
        self.assertAlmostEqual(segments[2].start_time, 3.5, places=1)
    
    def test_energy_based_speaker_change_detection(self):
        """Test energy-based detection of speaker changes"""
        # Create audio with significant energy change
        audio = self.create_test_audio(2.0, [
            (0.0, 1.5, 1, 200),    # Primary speaker (lower energy)
            (1.5, 2.0, 2, 400)     # Secondary speaker (higher energy)
        ])
        
        from processors.audio_enhancement.secondary_removal import EnergyAnalyzer
        
        analyzer = EnergyAnalyzer()
        changes = analyzer.detect_energy_changes(audio, self.sample_rate)
        
        # Should detect energy change at 1.5s
        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(changes[0].time, 1.5, places=1)
        self.assertGreater(changes[0].energy_ratio, 1.5)  # Higher energy after change
    
    def test_handle_multiple_secondary_speakers(self):
        """Test handling of multiple secondary speakers"""
        # Create complex audio with multiple speakers
        audio = self.create_test_audio(5.0, [
            (0.0, 2.0, 1, 200),    # Primary
            (2.0, 2.5, 2, 400),    # Secondary 1
            (2.5, 4.0, 1, 200),    # Primary again
            (4.0, 4.5, 3, 600),    # Secondary 2
            (4.5, 5.0, 2, 400)     # Secondary 1 again
        ])
        
        from processors.audio_enhancement.secondary_removal import SecondaryRemover
        
        remover = SecondaryRemover()
        processed = remover.remove_secondary_speakers(audio, self.sample_rate)
        
        # Check that all secondary speakers are removed
        # Time ranges 2.0-2.5, 4.0-5.0 should be silent
        segment1 = processed[int(2.0 * self.sample_rate):int(2.5 * self.sample_rate)]
        segment2 = processed[int(4.0 * self.sample_rate):]
        
        self.assertLess(np.max(np.abs(segment1)), 0.001)
        self.assertLess(np.max(np.abs(segment2)), 0.001)
        
        # Primary speaker segments should be preserved
        primary1 = processed[:int(2.0 * self.sample_rate)]
        primary2 = processed[int(2.5 * self.sample_rate):int(4.0 * self.sample_rate)]
        
        self.assertGreater(np.max(np.abs(primary1)), 0.1)
        self.assertGreater(np.max(np.abs(primary2)), 0.1)
    
    def test_smart_end_detection(self):
        """Test smart detection of actual speech end vs secondary speaker"""
        # Create audio where primary speaker trails off, then secondary speaker
        samples = int(3.5 * self.sample_rate)
        audio = np.zeros(samples, dtype=np.float32)
        
        # Primary speaker with trailing off
        t1 = np.arange(int(2.8 * self.sample_rate)) / self.sample_rate
        primary_signal = 0.3 * np.sin(2 * np.pi * 200 * t1)
        # Apply fade out over last 0.3 seconds
        fade_samples = int(0.3 * self.sample_rate)
        primary_signal[-fade_samples:] *= np.linspace(1.0, 0.1, fade_samples)
        audio[:len(primary_signal)] = primary_signal
        
        # Secondary speaker after brief pause
        t2 = np.arange(int(0.5 * self.sample_rate)) / self.sample_rate
        secondary_signal = 0.5 * np.sin(2 * np.pi * 400 * t2)
        audio[int(3.0 * self.sample_rate):] = secondary_signal
        
        from processors.audio_enhancement.secondary_removal import SmartEndDetector
        
        detector = SmartEndDetector()
        end_info = detector.analyze_end(audio, self.sample_rate)
        
        # Should identify the trailing primary speech vs new secondary speaker
        self.assertTrue(end_info.has_secondary_speaker)
        self.assertAlmostEqual(end_info.primary_end_time, 2.8, places=1)
        self.assertAlmostEqual(end_info.secondary_start_time, 3.0, places=1)
    
    def test_spectral_analysis_for_speaker_identification(self):
        """Test spectral analysis to distinguish speakers"""
        # Create audio with speakers having different spectral characteristics
        audio = self.create_test_audio(3.0, [
            (0.0, 1.5, 1, 200),    # Primary - lower frequency
            (1.5, 3.0, 2, 600)     # Secondary - higher frequency
        ])
        
        from processors.audio_enhancement.secondary_removal import SpectralAnalyzer
        
        analyzer = SpectralAnalyzer()
        speaker_info = analyzer.analyze_speakers(audio, self.sample_rate)
        
        # Should identify different spectral characteristics
        self.assertEqual(len(speaker_info.speakers), 2)
        self.assertLess(speaker_info.speakers[0].dominant_frequency, 300)
        self.assertGreater(speaker_info.speakers[1].dominant_frequency, 500)
    
    def test_real_world_s5_sample(self):
        """Test with the actual S5 sample that has issues"""
        # Check if S5 sample exists
        s5_path = Path("test_audio_output/s5_from_dataset.wav")
        if not s5_path.exists():
            self.skipTest("S5 sample not available")
        
        # Load S5
        audio, sr = sf.read(s5_path)
        
        from processors.audio_enhancement.secondary_removal import SecondaryRemover
        
        remover = SecondaryRemover()
        processed = remover.remove_secondary_speakers(audio, sr)
        
        # Check that end is cleaned
        last_500ms = processed[-int(0.5 * sr):]
        max_amplitude = np.max(np.abs(last_500ms))
        
        self.assertLess(max_amplitude, 0.01,
                       f"S5 secondary speaker not removed. Max amplitude: {max_amplitude}")
        
        # Check that most of the audio is preserved
        total_energy_original = np.sqrt(np.mean(audio**2))
        total_energy_processed = np.sqrt(np.mean(processed**2))
        
        # Should preserve at least 80% of total energy
        self.assertGreater(total_energy_processed / total_energy_original, 0.8)
    
    def test_integration_with_enhancement_pipeline(self):
        """Test integration with the main enhancement pipeline"""
        # Create test audio
        audio = self.create_test_audio(3.3, [
            (0.0, 3.0, 1, 200),
            (3.0, 3.3, 2, 400)
        ])
        
        # Test through main enhancement pipeline
        from processors.audio_enhancement import AudioEnhancer
        
        enhancer = AudioEnhancer(enhancement_level='ultra_aggressive')
        enhanced = enhancer.enhance(audio, self.sample_rate)
        
        # Secondary speaker should be removed
        last_300ms = enhanced[-int(0.3 * self.sample_rate):]
        self.assertLess(np.max(np.abs(last_300ms)), 0.01)


if __name__ == '__main__':
    unittest.main(verbosity=2)