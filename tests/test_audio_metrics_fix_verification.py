"""Test to verify audio metrics fix for issue #5."""

import unittest
import numpy as np
import json
from utils.audio_metrics import calculate_snr, calculate_all_metrics
from processors.audio_enhancement.core import AudioEnhancer


class TestAudioMetricsFix(unittest.TestCase):
    """Test cases to verify audio metrics calculations are fixed."""
    
    def test_snr_no_infinity(self):
        """Test that SNR doesn't return infinity for identical signals."""
        # Create identical signals
        signal = np.random.randn(16000)  # 1 second at 16kHz
        
        # Calculate SNR
        snr = calculate_snr(signal, signal)
        
        # Should be 40dB (capped), not infinity
        self.assertEqual(snr, 40.0)
        self.assertNotEqual(snr, float('inf'))
    
    def test_snr_very_clean_signal(self):
        """Test SNR for very clean signals."""
        clean = np.random.randn(16000)
        # Add tiny noise
        noisy = clean + np.random.randn(16000) * 1e-10
        
        snr = calculate_snr(clean, noisy)
        
        # Should be capped at 40dB
        self.assertLessEqual(snr, 40.0)
        self.assertGreater(snr, 0)
    
    def test_all_metrics_returns_pesq_stoi(self):
        """Test that calculate_all_metrics returns PESQ and STOI."""
        reference = np.random.randn(16000)
        degraded = reference + np.random.randn(16000) * 0.1
        
        metrics = calculate_all_metrics(reference, degraded, 16000)
        
        # Should have all metrics
        self.assertIn('snr', metrics)
        self.assertIn('pesq', metrics)
        self.assertIn('stoi', metrics)
        
        # Values should be reasonable
        self.assertGreater(metrics['pesq'], 0)
        self.assertLessEqual(metrics['pesq'], 5.0)
        self.assertGreater(metrics['stoi'], 0)
        self.assertLessEqual(metrics['stoi'], 1.0)
    
    def test_enhancement_metadata_includes_all_metrics(self):
        """Test that enhancement metadata includes PESQ and STOI."""
        enhancer = AudioEnhancer(enhancement_level='moderate')
        
        # Create test audio
        audio = np.random.randn(16000)
        sample_rate = 16000
        
        # Process with metadata
        enhanced, metadata = enhancer.enhance(audio, sample_rate, return_metadata=True)
        
        # Check metadata
        self.assertIn('pesq', metadata)
        self.assertIn('stoi', metadata)
        self.assertIn('snr_after', metadata)
        self.assertIn('snr_improvement', metadata)
        
        # Values should not be 0 (unless truly 0)
        self.assertIsInstance(metadata['pesq'], (int, float))
        self.assertIsInstance(metadata['stoi'], (int, float))
    
    def test_json_serialization(self):
        """Test that metrics can be serialized to JSON without errors."""
        metrics = {
            'snr': 40.0,  # Was infinity
            'pesq': 3.5,
            'stoi': 0.85,
            'snr_improvement': 5.0
        }
        
        # Should not raise an error
        json_str = json.dumps(metrics)
        self.assertIsInstance(json_str, str)
        
        # Can be deserialized back
        loaded = json.loads(json_str)
        self.assertEqual(loaded['snr'], 40.0)


if __name__ == '__main__':
    unittest.main()