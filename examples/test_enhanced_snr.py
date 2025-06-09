"""
Example usage of the Enhanced SNR Calculator Module.
Demonstrates various SNR calculation methods and configuration options.
"""

import numpy as np
from utils.enhanced_snr_calculator import EnhancedSNRCalculator, SNRConfig

def test_basic_usage():
    """Basic SNR calculation example."""
    print("=== Basic SNR Calculation ===")
    
    # Create a test signal
    fs = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Speech-like signal with modulation
    speech = 0.5 * np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    
    # Add noise
    noise = 0.05 * np.random.normal(0, 1, len(speech))
    noisy_signal = speech + noise
    
    # Calculate SNR
    calculator = EnhancedSNRCalculator()
    snr = calculator.calculate_snr(noisy_signal, fs)
    print(f"Automatic method SNR: {snr:.2f} dB")
    
    # Try different methods
    methods = ['waveform', 'spectral', 'segmental', 'vad_enhanced']
    for method in methods:
        snr = calculator.calculate_snr(noisy_signal, fs, method=method)
        print(f"{method.capitalize()} SNR: {snr:.2f} dB")


def test_multiband_snr():
    """Multiband SNR analysis example."""
    print("\n=== Multiband SNR Analysis ===")
    
    fs = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create signal with multiple frequency components
    low_freq = 0.3 * np.sin(2 * np.pi * 200 * t)
    mid_freq = 0.3 * np.sin(2 * np.pi * 1000 * t)
    high_freq = 0.3 * np.sin(2 * np.pi * 4000 * t)
    signal = low_freq + mid_freq + high_freq
    
    # Add frequency-dependent noise
    noise = np.random.normal(0, 0.03, len(signal))
    # Emphasize high frequency noise
    b, a = np.array([1, -0.95]), np.array([1])  # High-pass filter
    filtered_noise = np.convolve(noise, b, mode='same')
    
    noisy_signal = signal + filtered_noise
    
    calculator = EnhancedSNRCalculator()
    bands = [(0, 500), (500, 2000), (2000, 8000)]
    multiband_snr = calculator.calculate_multiband_snr(noisy_signal, fs, bands)
    
    for i, (low, high) in enumerate(bands):
        print(f"Band {low}-{high} Hz: {multiband_snr[i]:.2f} dB")


def test_confidence_scoring():
    """SNR calculation with confidence scores."""
    print("\n=== SNR with Confidence Scoring ===")
    
    fs = 16000
    calculator = EnhancedSNRCalculator()
    
    # Test 1: Long, stationary signal (high confidence)
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))
    clean = 0.5 * np.sin(2 * np.pi * 300 * t)
    noise = 0.05 * np.random.normal(0, 1, len(clean))
    signal1 = clean + noise
    
    snr1, conf1 = calculator.calculate_snr_with_confidence(signal1, fs)
    print(f"Long stationary signal: SNR={snr1:.2f} dB, Confidence={conf1:.2f}")
    
    # Test 2: Short, non-stationary signal (lower confidence)
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration))
    varying = 0.5 * np.sin(2 * np.pi * 300 * t) * np.sin(2 * np.pi * 2 * t)
    varying_noise = 0.05 * np.random.normal(0, 1, len(varying)) * (1 + 0.5 * np.sin(2 * np.pi * 1 * t))
    signal2 = varying + varying_noise
    
    snr2, conf2 = calculator.calculate_snr_with_confidence(signal2, fs)
    print(f"Short varying signal: SNR={snr2:.2f} dB, Confidence={conf2:.2f}")


def test_custom_configuration():
    """Using custom configuration."""
    print("\n=== Custom Configuration ===")
    
    # Create custom config
    config = SNRConfig(
        frame_size=0.030,  # 30ms frames
        frame_shift=0.015,  # 15ms shift
        noise_estimation_method="minimum_statistics",
        noise_bias_compensation=1.2,  # Less aggressive
        vad_enabled=True,
        vad_backend="energy",
        vad_energy_threshold=-35,  # More sensitive
        min_duration=0.3  # Shorter minimum
    )
    
    calculator = EnhancedSNRCalculator(config)
    
    # Test with speech signal
    fs = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create speech with pauses
    speech = np.zeros(len(t))
    speech[int(0.2*fs):int(0.5*fs)] = 0.5 * np.sin(2 * np.pi * 300 * t[int(0.2*fs):int(0.5*fs)])
    speech[int(0.6*fs):int(0.9*fs)] = 0.5 * np.sin(2 * np.pi * 400 * t[int(0.6*fs):int(0.9*fs)])
    
    noise = 0.03 * np.random.normal(0, 1, len(speech))
    noisy_signal = speech + noise
    
    snr = calculator.calculate_snr(noisy_signal, fs)
    print(f"Custom config SNR: {snr:.2f} dB")


def test_real_world_scenarios():
    """Test with more realistic audio scenarios."""
    print("\n=== Real-World Scenarios ===")
    
    fs = 16000
    calculator = EnhancedSNRCalculator()
    
    # Scenario 1: Clean recording with room noise
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration))
    speech = 0.7 * np.sin(2 * np.pi * 250 * t) * (1 + 0.2 * np.sin(2 * np.pi * 4 * t))
    room_noise = 0.02 * np.random.normal(0, 1, len(speech))
    signal = speech + room_noise
    
    snr = calculator.calculate_snr(signal, fs)
    print(f"Clean recording with room noise: {snr:.2f} dB")
    
    # Scenario 2: Noisy environment
    street_noise = 0.1 * np.random.normal(0, 1, len(speech))
    # Add some low-frequency rumble
    rumble = 0.05 * np.sin(2 * np.pi * 50 * t)
    signal = speech + street_noise + rumble
    
    snr = calculator.calculate_snr(signal, fs)
    print(f"Noisy street recording: {snr:.2f} dB")
    
    # Scenario 3: Phone quality
    # Bandlimit the signal
    from scipy import signal as scipy_signal
    b, a = scipy_signal.butter(4, [300, 3400], fs=fs, btype='band')
    phone_speech = scipy_signal.filtfilt(b, a, speech)
    phone_noise = 0.04 * np.random.normal(0, 1, len(phone_speech))
    signal = phone_speech + phone_noise
    
    snr = calculator.calculate_snr(signal, fs)
    print(f"Phone quality audio: {snr:.2f} dB")


def test_spectral_weighting():
    """Test A-weighting for perceptual SNR."""
    print("\n=== Perceptual SNR with A-weighting ===")
    
    fs = 16000
    duration = 1.0
    calculator = EnhancedSNRCalculator()
    
    # Test at different frequencies
    frequencies = [100, 500, 1000, 4000]
    
    for freq in frequencies:
        t = np.linspace(0, duration, int(fs * duration))
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
        noise = 0.05 * np.random.normal(0, 1, len(signal))
        noisy = signal + noise
        
        # Unweighted
        snr_unweighted = calculator.calculate_snr(noisy, fs, method='spectral', weighting=None)
        
        # A-weighted
        snr_weighted = calculator.calculate_snr(noisy, fs, method='spectral', weighting='A')
        
        print(f"{freq} Hz: Unweighted={snr_unweighted:.2f} dB, A-weighted={snr_weighted:.2f} dB")


if __name__ == "__main__":
    # Run all examples
    test_basic_usage()
    test_multiband_snr()
    test_confidence_scoring()
    test_custom_configuration()
    test_real_world_scenarios()
    test_spectral_weighting()
    
    print("\n=== Integration with Existing Code ===")
    print("The EnhancedSNRCalculator is backward compatible.")
    print("Use method='legacy' to match existing SNRMeasurement behavior.")
    print("See utils/enhanced_snr_calculator.py for full API documentation.")