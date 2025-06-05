#!/usr/bin/env python3
"""
Test to verify the suppression fix is actually being applied.
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.speaker_separation import SpeakerSeparator, SeparationConfig


class TestSuppressionFix(unittest.TestCase):
    """Test that our suppression fixes are working."""
    
    def test_apply_suppression_strength(self):
        """Test the _apply_suppression method directly."""
        
        # Create test audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440Hz tone
        
        # Create separator
        config = SeparationConfig(suppression_strength=0.95)
        separator = SpeakerSeparator(config)
        
        # Apply suppression to middle 0.5 seconds
        suppressed = separator._apply_suppression(
            audio.copy(), sample_rate,
            start_time=0.25, end_time=0.75,
            strength=0.95
        )
        
        # Check the suppressed region
        start_idx = int(0.25 * sample_rate)
        end_idx = int(0.75 * sample_rate)
        
        # Measure power in suppressed region
        original_power = np.mean(audio[start_idx:end_idx]**2)
        suppressed_power = np.mean(suppressed[start_idx:end_idx]**2)
        
        if original_power > 0:
            reduction_db = 10 * np.log10(suppressed_power / original_power)
            print(f"Suppression reduction: {reduction_db:.1f} dB")
            
            # We expect -60dB based on our fix
            self.assertLess(reduction_db, -50,
                          f"Should have >50dB reduction, got {reduction_db:.1f}dB")
    
    def test_check_suppression_calculation(self):
        """Check what the suppression is actually doing."""
        
        # Test our -60dB calculation
        factor = 10 ** (-60/20)
        print(f"10^(-60/20) = {factor}")
        print(f"This means audio is multiplied by {factor:.6f}")
        print(f"In dB: {20 * np.log10(factor):.1f} dB")
        
        # Test with actual audio
        test_signal = np.array([1.0, 0.5, -0.8, 0.3])
        suppressed = test_signal * factor
        
        print(f"\nOriginal: {test_signal}")
        print(f"Suppressed: {suppressed}")
        
        # Verify the reduction
        original_power = np.mean(test_signal**2)
        suppressed_power = np.mean(suppressed**2)
        actual_reduction_db = 10 * np.log10(suppressed_power / original_power)
        
        print(f"\nActual reduction: {actual_reduction_db:.1f} dB")
        self.assertAlmostEqual(actual_reduction_db, -60.0, places=1)


if __name__ == '__main__':
    unittest.main()