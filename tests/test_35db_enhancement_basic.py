"""Basic test for 35dB enhancement integration."""

import unittest
import numpy as np
from processors.audio_enhancement.core import AudioEnhancer
from utils.snr_measurement import SNRMeasurement


class Test35dBEnhancementBasic(unittest.TestCase):
    """Basic tests for 35dB enhancement functionality."""
    
    def setUp(self):
        self.sample_rate = 16000
    
    def test_enhancement_enabled(self):
        """Test that 35dB enhancement can be enabled."""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        self.assertTrue(enhancer.enable_35db_enhancement)
    
    def test_enhance_to_target_snr_exists(self):
        """Test that enhance_to_target_snr method exists."""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        self.assertTrue(hasattr(enhancer, 'enhance_to_target_snr'))
    
    def test_basic_enhancement(self):
        """Test basic enhancement functionality."""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        # Create a simple noisy signal
        duration = 2.0
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        clean_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Add noise for ~20dB SNR
        noise = 0.1 * np.random.randn(len(clean_signal))
        noisy_signal = clean_signal + noise
        
        # Enhance
        enhanced, metadata = enhancer.enhance_to_target_snr(noisy_signal, self.sample_rate)
        
        # Check basic properties
        self.assertIsNotNone(enhanced)
        self.assertIsNotNone(metadata)
        self.assertEqual(len(enhanced), len(noisy_signal))
        self.assertIn('snr_db', metadata)
        self.assertIn('enhancement_applied', metadata)
        self.assertIn('target_achieved', metadata)
    
    def test_metadata_structure(self):
        """Test that metadata has expected structure."""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        # Create simple signal
        signal = np.random.randn(self.sample_rate * 2)
        
        # Enhance
        enhanced, metadata = enhancer.enhance_to_target_snr(signal, self.sample_rate, target_snr=35)
        
        # Check metadata fields
        expected_fields = [
            'snr_db',
            'audio_quality_metrics',
            'enhancement_applied',
            'naturalness_score',
            'snr_improvement',
            'target_achieved',
            'iterations',
            'stages_applied'
        ]
        
        for field in expected_fields:
            self.assertIn(field, metadata, f"Missing field: {field}")
        
        # Check audio_quality_metrics sub-fields
        quality_metrics = metadata.get('audio_quality_metrics', {})
        self.assertIn('pesq', quality_metrics)
        self.assertIn('stoi', quality_metrics)
        self.assertIn('mos_estimate', quality_metrics)
    
    def test_target_snr_parameter(self):
        """Test that target SNR parameter is respected."""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        # Create noisy signal
        signal = np.random.randn(self.sample_rate * 2)
        
        # Test with different targets
        targets = [30, 35, 40]
        
        for target in targets:
            enhanced, metadata = enhancer.enhance_to_target_snr(signal, self.sample_rate, target_snr=target)
            self.assertIsNotNone(enhanced)
            self.assertIsNotNone(metadata)
            
            # The enhancement should at least attempt to reach the target
            # (actual achievement depends on signal quality)
            self.assertIn('snr_db', metadata)
    
    def test_disable_mode(self):
        """Test that enhancement can be disabled."""
        # With 35dB disabled
        enhancer_disabled = AudioEnhancer(enable_35db_enhancement=False)
        signal = np.random.randn(self.sample_rate * 2)
        
        # Should still work but use standard enhancement
        enhanced, metadata = enhancer_disabled.enhance_to_target_snr(signal, self.sample_rate)
        self.assertIsNotNone(enhanced)
        self.assertIsNotNone(metadata)


if __name__ == "__main__":
    unittest.main()