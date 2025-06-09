"""
Comprehensive test suite for Enhanced SNR Calculator Module.
Following Test-Driven Development (TDD) approach as specified in S01_T03.
"""

import pytest
import numpy as np
import os
import sys
from scipy import signal

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.enhanced_snr_calculator import (
    EnhancedSNRCalculator, NoiseEstimator, SNRError, 
    InsufficientDataError, NoSpeechDetectedError, InvalidSignalError
)


class TestBasicSNRCalculation:
    """Test basic SNR calculation functionality."""
    
    def test_clean_signal_snr(self):
        """Test with synthetic clean signal - expect high SNR (>40dB)."""
        # Create clean sine wave
        fs = 16000
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(fs * duration))
        clean_signal = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add minimal noise floor
        noise = np.random.normal(0, 0.001, len(clean_signal))
        signal_with_noise = clean_signal + noise
        
        calculator = EnhancedSNRCalculator()
        snr = calculator.calculate_snr(signal_with_noise, fs)
        
        # Expect high SNR for clean signal
        assert snr > 40, f"Expected SNR > 40dB for clean signal, got {snr}dB"
        assert snr < 60, f"SNR unrealistically high: {snr}dB"
    
    def test_known_snr_signals(self):
        """Test with signals of known SNR - verify accuracy within 1dB."""
        fs = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(fs * duration))
        
        # Create speech-like signal (multiple frequencies)
        speech = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.2 * np.sin(2 * np.pi * 400 * t) +
            0.1 * np.sin(2 * np.pi * 800 * t)
        )
        
        # Test various known SNR levels
        known_snrs = [0, 10, 20, 30]
        calculator = EnhancedSNRCalculator()
        
        for target_snr in known_snrs:
            # Calculate noise level for target SNR
            signal_power = np.mean(speech ** 2)
            noise_power = signal_power / (10 ** (target_snr / 10))
            noise_std = np.sqrt(noise_power)
            
            # Add calibrated noise
            noise = np.random.normal(0, noise_std, len(speech))
            noisy_signal = speech + noise
            
            # Calculate SNR
            calculated_snr = calculator.calculate_snr(noisy_signal, fs, method='waveform')
            
            # Verify accuracy within 1dB
            error = abs(calculated_snr - target_snr)
            assert error < 1.0, f"SNR error {error}dB exceeds 1dB threshold for {target_snr}dB signal"
    
    def test_various_noise_types(self):
        """Test SNR calculation with different noise types."""
        fs = 16000
        duration = 2.0
        samples = int(fs * duration)
        
        # Create speech signal
        t = np.linspace(0, duration, samples)
        speech = 0.5 * np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        
        calculator = EnhancedSNRCalculator()
        
        # Test with white noise
        white_noise = np.random.normal(0, 0.1, samples)
        snr_white = calculator.calculate_snr(speech + white_noise, fs)
        assert 10 < snr_white < 30, f"White noise SNR out of expected range: {snr_white}dB"
        
        # Test with pink noise
        pink_noise = self._generate_pink_noise(samples, 0.1)
        snr_pink = calculator.calculate_snr(speech + pink_noise, fs)
        assert 10 < snr_pink < 30, f"Pink noise SNR out of expected range: {snr_pink}dB"
        
        # Test with babble noise (simulated)
        babble = sum(0.05 * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi) 
                    for f in [150, 250, 350, 450, 550])
        snr_babble = calculator.calculate_snr(speech + babble, fs)
        assert 5 < snr_babble < 25, f"Babble noise SNR out of expected range: {snr_babble}dB"
    
    def _generate_pink_noise(self, samples, amplitude):
        """Generate pink noise using 1/f spectrum."""
        # Generate white noise
        white = np.random.randn(samples)
        
        # Apply 1/f filter in frequency domain
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(samples)
        
        # Avoid division by zero
        freqs[0] = 1
        fft = fft / np.sqrt(freqs)
        
        # Convert back to time domain
        pink = np.fft.irfft(fft, n=samples)
        
        # Normalize amplitude
        pink = amplitude * pink / np.std(pink)
        return pink


class TestNoiseEstimation:
    """Test noise estimation functionality."""
    
    def test_silence_detection(self):
        """Test silence segment identification."""
        fs = 16000
        
        # Create signal with clear silence regions
        speech_duration = 0.5
        silence_duration = 0.3
        
        speech = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, speech_duration, int(fs * speech_duration)))
        silence = np.zeros(int(fs * silence_duration))
        
        # Pattern: silence - speech - silence - speech - silence
        signal = np.concatenate([silence, speech, silence, speech, silence])
        
        estimator = NoiseEstimator()
        silence_segments = estimator.detect_silence_segments(signal, fs)
        
        # Should detect at least 3 silence segments
        assert len(silence_segments) >= 3, f"Expected at least 3 silence segments, got {len(silence_segments)}"
        
        # First segment should start at 0
        assert silence_segments[0][0] == 0, "First silence segment should start at beginning"
    
    def test_noise_floor_estimation_stationary(self):
        """Test noise floor estimation with stationary noise."""
        fs = 16000
        duration = 3.0
        samples = int(fs * duration)
        
        # Create signal with known noise floor
        noise_level = 0.05
        noise = np.random.normal(0, noise_level, samples)
        
        # Add speech segments
        t = np.linspace(0, duration, samples)
        speech_mask = np.zeros(samples, dtype=bool)
        # Speech from 0.5-1.0s and 1.5-2.0s
        speech_mask[int(0.5*fs):int(1.0*fs)] = True
        speech_mask[int(1.5*fs):int(2.0*fs)] = True
        
        speech = np.zeros(samples)
        speech[speech_mask] = 0.5 * np.sin(2 * np.pi * 300 * t[speech_mask])
        
        signal = speech + noise
        
        estimator = NoiseEstimator()
        estimated_noise = estimator.estimate_noise_floor(signal, fs)
        
        # Should estimate within 20% of actual noise power
        actual_noise_power = noise_level ** 2
        error = abs(estimated_noise - actual_noise_power) / actual_noise_power
        assert error < 0.2, f"Noise estimation error {error*100}% exceeds 20%"
    
    def test_noise_floor_estimation_nonstationary(self):
        """Test noise floor estimation with non-stationary noise."""
        fs = 16000
        duration = 3.0
        samples = int(fs * duration)
        t = np.linspace(0, duration, samples)
        
        # Create time-varying noise
        noise_envelope = 0.02 + 0.03 * np.sin(2 * np.pi * 0.5 * t)  # Slowly varying
        noise = noise_envelope * np.random.normal(0, 1, samples)
        
        # Add speech
        speech = np.zeros(samples)
        speech_sections = [(0.5, 0.8), (1.2, 1.5), (2.0, 2.3)]
        for start, end in speech_sections:
            mask = (t >= start) & (t < end)
            speech[mask] = 0.5 * np.sin(2 * np.pi * 300 * t[mask])
        
        signal = speech + noise
        
        estimator = NoiseEstimator()
        estimated_noise = estimator.estimate_noise_floor(signal, fs)
        
        # For non-stationary noise, check if estimate is reasonable
        min_expected = np.min(noise_envelope) ** 2
        max_expected = np.mean(noise_envelope) ** 2
        assert min_expected < estimated_noise < max_expected, \
            f"Noise estimate {estimated_noise} outside expected range [{min_expected}, {max_expected}]"
    
    def test_robustness_to_speech_presence(self):
        """Test noise estimation robustness when speech is present."""
        fs = 16000
        duration = 2.0
        samples = int(fs * duration)
        
        # Create mostly speech signal with little silence
        t = np.linspace(0, duration, samples)
        speech = 0.5 * np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
        
        # Add small silence gaps
        speech[int(0.4*fs):int(0.5*fs)] = 0
        speech[int(1.4*fs):int(1.5*fs)] = 0
        
        # Add known noise
        noise_level = 0.03
        noise = np.random.normal(0, noise_level, samples)
        signal = speech + noise
        
        estimator = NoiseEstimator()
        estimated_noise = estimator.estimate_noise_floor(signal, fs)
        
        # Should still estimate noise reasonably despite limited silence
        actual_noise_power = noise_level ** 2
        error = abs(estimated_noise - actual_noise_power) / actual_noise_power
        assert error < 0.5, f"Noise estimation error {error*100}% too high with limited silence"


class TestAdvancedFeatures:
    """Test advanced SNR calculation features."""
    
    def test_vad_integration(self):
        """Test SNR calculation with Voice Activity Detection."""
        fs = 16000
        duration = 2.0
        samples = int(fs * duration)
        t = np.linspace(0, duration, samples)
        
        # Create signal with clear speech/silence pattern
        speech = np.zeros(samples)
        speech_periods = [(0.2, 0.7), (1.0, 1.5), (1.7, 1.9)]
        
        for start, end in speech_periods:
            mask = (t >= start) & (t < end)
            speech[mask] = 0.5 * np.sin(2 * np.pi * 300 * t[mask])
        
        # Add noise
        noise = 0.05 * np.random.normal(0, 1, samples)
        signal = speech + noise
        
        calculator = EnhancedSNRCalculator()
        
        # Calculate with and without VAD
        snr_with_vad = calculator.calculate_snr(signal, fs, method='vad_enhanced')
        snr_without_vad = calculator.calculate_snr(signal, fs, method='waveform')
        
        # VAD should give more accurate SNR
        assert snr_with_vad != snr_without_vad, "VAD should affect SNR calculation"
        # In this case, VAD should give higher SNR as it excludes silence
        assert snr_with_vad > snr_without_vad, "VAD should improve SNR by excluding silence"
    
    def test_perceptual_weighting(self):
        """Test A-weighting implementation for perceptual relevance."""
        fs = 16000
        duration = 1.0
        samples = int(fs * duration)
        
        # Create signals at different frequencies
        frequencies = [100, 1000, 4000, 8000]
        calculator = EnhancedSNRCalculator()
        
        for freq in frequencies:
            t = np.linspace(0, duration, samples)
            signal = 0.5 * np.sin(2 * np.pi * freq * t)
            noise = 0.05 * np.random.normal(0, 1, samples)
            noisy_signal = signal + noise
            
            # Calculate weighted and unweighted SNR
            snr_unweighted = calculator.calculate_snr(noisy_signal, fs, method='spectral', weighting=None)
            snr_weighted = calculator.calculate_snr(noisy_signal, fs, method='spectral', weighting='A')
            
            # A-weighting should affect the result
            assert snr_weighted != snr_unweighted, f"A-weighting should affect SNR at {freq}Hz"
    
    def test_multiband_snr_analysis(self):
        """Test multi-band SNR analysis for detailed assessment."""
        fs = 16000
        duration = 1.0
        samples = int(fs * duration)
        t = np.linspace(0, duration, samples)
        
        # Create signal with energy in different bands
        low_freq = 0.3 * np.sin(2 * np.pi * 200 * t)
        mid_freq = 0.3 * np.sin(2 * np.pi * 1000 * t)
        high_freq = 0.3 * np.sin(2 * np.pi * 4000 * t)
        signal = low_freq + mid_freq + high_freq
        
        # Add frequency-dependent noise
        noise = np.random.normal(0, 0.05, samples)
        # High-pass filter the noise
        b, a = signal.butter(4, 2000, fs=fs, btype='high')
        filtered_noise = signal.filtfilt(b, a, noise)
        
        noisy_signal = signal + filtered_noise
        
        calculator = EnhancedSNRCalculator()
        multiband_snr = calculator.calculate_multiband_snr(noisy_signal, fs, bands=[(0, 500), (500, 2000), (2000, 8000)])
        
        # Should return SNR for each band
        assert len(multiband_snr) == 3, "Should return SNR for 3 bands"
        
        # Low band should have higher SNR (less noise)
        assert multiband_snr[0] > multiband_snr[2], "Low band should have higher SNR than high band"
    
    def test_confidence_scoring(self):
        """Test confidence scoring for SNR estimates."""
        fs = 16000
        calculator = EnhancedSNRCalculator()
        
        # Test 1: High confidence - long, stationary signal
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration))
        clean_signal = 0.5 * np.sin(2 * np.pi * 300 * t)
        noise = 0.05 * np.random.normal(0, 1, len(clean_signal))
        
        snr, confidence = calculator.calculate_snr_with_confidence(clean_signal + noise, fs)
        assert confidence > 0.8, f"Expected high confidence for long stationary signal, got {confidence}"
        
        # Test 2: Low confidence - short, non-stationary signal
        duration = 0.5
        t = np.linspace(0, duration, int(fs * duration))
        varying_signal = 0.5 * np.sin(2 * np.pi * 300 * t) * np.sin(2 * np.pi * 2 * t)
        varying_noise = 0.05 * np.random.normal(0, 1, len(varying_signal)) * (1 + 0.5 * np.sin(2 * np.pi * 1 * t))
        
        snr, confidence = calculator.calculate_snr_with_confidence(varying_signal + varying_noise, fs)
        assert confidence < 0.7, f"Expected lower confidence for short non-stationary signal, got {confidence}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_low_snr(self):
        """Test with very low SNR (<0dB)."""
        fs = 16000
        duration = 1.0
        samples = int(fs * duration)
        
        # Create weak signal with strong noise
        signal = 0.1 * np.sin(2 * np.pi * 300 * np.linspace(0, duration, samples))
        noise = 0.5 * np.random.normal(0, 1, samples)  # 5x stronger than signal
        
        calculator = EnhancedSNRCalculator()
        snr = calculator.calculate_snr(signal + noise, fs)
        
        # Should handle negative SNR
        assert snr < 0, f"Expected negative SNR for noise-dominated signal, got {snr}dB"
        assert snr > -20, f"SNR unrealistically low: {snr}dB"
    
    def test_very_high_snr(self):
        """Test with very high SNR (>60dB)."""
        fs = 16000
        duration = 1.0
        samples = int(fs * duration)
        
        # Create strong signal with minimal noise
        signal = 0.9 * np.sin(2 * np.pi * 300 * np.linspace(0, duration, samples))
        noise = 0.0001 * np.random.normal(0, 1, samples)
        
        calculator = EnhancedSNRCalculator()
        snr = calculator.calculate_snr(signal + noise, fs)
        
        # Should cap at reasonable maximum
        assert snr > 40, f"Expected high SNR for clean signal, got {snr}dB"
        assert snr <= 60, f"SNR should be capped at 60dB, got {snr}dB"
    
    def test_clipped_signals(self):
        """Test with clipped/saturated signals."""
        fs = 16000
        duration = 1.0
        samples = int(fs * duration)
        t = np.linspace(0, duration, samples)
        
        # Create signal that clips
        signal = 2.0 * np.sin(2 * np.pi * 300 * t)  # Amplitude > 1
        clipped_signal = np.clip(signal, -1.0, 1.0)
        
        # Add noise
        noise = 0.05 * np.random.normal(0, 1, samples)
        noisy_clipped = clipped_signal + noise
        
        calculator = EnhancedSNRCalculator()
        snr = calculator.calculate_snr(noisy_clipped, fs)
        
        # Should still provide reasonable estimate despite clipping
        assert 10 < snr < 40, f"SNR estimate unreasonable for clipped signal: {snr}dB"
    
    def test_dc_offset(self):
        """Test with DC offset in signal."""
        fs = 16000
        duration = 1.0
        samples = int(fs * duration)
        t = np.linspace(0, duration, samples)
        
        # Create signal with DC offset
        dc_offset = 0.3
        signal = dc_offset + 0.5 * np.sin(2 * np.pi * 300 * t)
        noise = 0.05 * np.random.normal(0, 1, samples)
        
        calculator = EnhancedSNRCalculator()
        snr = calculator.calculate_snr(signal + noise, fs)
        
        # Should handle DC offset correctly
        assert 15 < snr < 35, f"SNR affected by DC offset: {snr}dB"
    
    def test_all_silence(self):
        """Test with all silence signal."""
        fs = 16000
        duration = 1.0
        samples = int(fs * duration)
        
        # Pure silence with tiny noise floor
        signal = 1e-6 * np.random.normal(0, 1, samples)
        
        calculator = EnhancedSNRCalculator()
        snr = calculator.calculate_snr(signal, fs)
        
        # Should handle gracefully
        assert isinstance(snr, float), "Should return float even for silence"
        # SNR might be undefined or very high for pure silence
    
    def test_insufficient_data(self):
        """Test with insufficient data for reliable estimation."""
        fs = 16000
        
        # Very short signal (10ms)
        signal = np.random.normal(0, 0.1, int(fs * 0.01))
        
        calculator = EnhancedSNRCalculator()
        with pytest.raises(InsufficientDataError) as excinfo:
            calculator.calculate_snr(signal, fs, min_duration=0.1)
        
        assert excinfo.value.required == 0.1
        assert excinfo.value.actual < 0.1
    
    def test_invalid_signal(self):
        """Test with invalid signal values."""
        fs = 16000
        samples = fs  # 1 second
        
        calculator = EnhancedSNRCalculator()
        
        # Test with NaN
        signal_nan = np.ones(samples)
        signal_nan[100] = np.nan
        with pytest.raises(InvalidSignalError):
            calculator.calculate_snr(signal_nan, fs)
        
        # Test with Inf
        signal_inf = np.ones(samples)
        signal_inf[200] = np.inf
        with pytest.raises(InvalidSignalError):
            calculator.calculate_snr(signal_inf, fs)
    
    def test_no_speech_detected(self):
        """Test when no speech is detected but VAD is required."""
        fs = 16000
        duration = 1.0
        
        # Pure noise, no speech
        signal = 0.1 * np.random.normal(0, 1, int(fs * duration))
        
        calculator = EnhancedSNRCalculator()
        # Should fall back gracefully when no speech detected
        snr = calculator.calculate_snr(signal, fs, method='vad_enhanced')
        assert isinstance(snr, float), "Should return SNR even without detected speech"


class TestPerformance:
    """Test performance requirements."""
    
    def test_processing_speed(self):
        """Test that 1-hour audio processes in <5 seconds."""
        import time
        
        fs = 16000
        duration = 3600  # 1 hour
        
        # Create realistic signal (don't need full complexity for speed test)
        # Use smaller chunks to avoid memory issues
        chunk_size = fs * 60  # 1 minute chunks
        
        calculator = EnhancedSNRCalculator()
        
        start_time = time.time()
        
        # Process in chunks (simulating streaming)
        for i in range(60):  # 60 minutes
            chunk = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 60, chunk_size))
            noise = 0.05 * np.random.normal(0, 1, chunk_size)
            snr = calculator.calculate_snr(chunk + noise, fs)
        
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0, f"Processing took {elapsed}s, exceeding 5s requirement"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        fs = 16000
        calculator = EnhancedSNRCalculator()
        
        # Test with very small values
        tiny_signal = 1e-10 * np.ones(fs)
        snr = calculator.calculate_snr(tiny_signal, fs)
        assert np.isfinite(snr), "SNR should be finite for tiny signals"
        
        # Test with very large values
        large_signal = 1e10 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, fs))
        # Normalize to valid range
        large_signal = large_signal / np.max(np.abs(large_signal))
        snr = calculator.calculate_snr(large_signal, fs)
        assert np.isfinite(snr), "SNR should be finite for large signals"


class TestIntegration:
    """Test integration with existing codebase."""
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing calculate_snr function."""
        from utils.snr_measurement import SNRMeasurement as LegacySNR
        
        fs = 16000
        duration = 2.0
        samples = int(fs * duration)
        t = np.linspace(0, duration, samples)
        
        # Create test signal
        signal = 0.5 * np.sin(2 * np.pi * 300 * t)
        noise = 0.05 * np.random.normal(0, 1, samples)
        noisy_signal = signal + noise
        
        # Compare legacy and enhanced
        legacy = LegacySNR()
        legacy_snr = legacy.measure_snr(noisy_signal, fs)
        
        calculator = EnhancedSNRCalculator()
        enhanced_snr = calculator.calculate_snr(noisy_signal, fs, method='legacy')
        
        # Should match legacy implementation when using legacy method
        assert abs(legacy_snr - enhanced_snr) < 0.1, \
            f"Legacy mode mismatch: {legacy_snr} vs {enhanced_snr}"
    
    def test_integration_with_audio_metrics(self):
        """Test integration with AudioMetrics class."""
        from utils.audio_metrics import AudioQualityMetrics
        
        fs = 16000
        duration = 1.0
        samples = int(fs * duration)
        
        # Create reference and degraded signals
        reference = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, duration, samples))
        noise = 0.05 * np.random.normal(0, 1, samples)
        degraded = reference + noise
        
        # Test that enhanced SNR can be used within metrics framework
        metrics = AudioQualityMetrics()
        calculator = EnhancedSNRCalculator()
        
        # Should be able to use enhanced calculator for metrics
        enhanced_snr = calculator.calculate_snr(degraded, fs)
        basic_snr = metrics.calculate_snr(reference, degraded)
        
        # Both should give reasonable results
        assert 10 < enhanced_snr < 40, f"Enhanced SNR out of range: {enhanced_snr}"
        assert 10 < basic_snr < 40, f"Basic SNR out of range: {basic_snr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])