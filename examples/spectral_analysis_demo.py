"""
Demo script for the Spectral Analysis Module.
Shows how to use the spectral analyzer for audio quality assessment.
"""

import numpy as np
import matplotlib.pyplot as plt
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer
import librosa
import librosa.display


def demonstrate_spectral_analysis():
    """Demonstrate spectral analysis capabilities."""
    
    # Generate test signals
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 1. Pure tone
    print("1. Analyzing pure tone (440 Hz)...")
    pure_tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # 2. Complex harmonic signal
    print("2. Analyzing complex harmonic signal...")
    harmonic = np.zeros_like(t)
    for h in range(1, 6):
        harmonic += (1.0 / h) * np.sin(2 * np.pi * 200 * h * t)
    harmonic = harmonic.astype(np.float32)
    
    # 3. Noisy signal
    print("3. Analyzing noisy signal...")
    noisy = pure_tone + 0.1 * np.random.randn(len(pure_tone))
    noisy = noisy.astype(np.float32)
    
    # Initialize analyzer
    analyzer = SpectralAnalyzer()
    
    # Analyze each signal
    signals = {
        'Pure Tone': pure_tone,
        'Harmonic': harmonic,
        'Noisy': noisy
    }
    
    results = {}
    for name, signal in signals.items():
        print(f"\nAnalyzing {name}...")
        result = analyzer.analyze(signal, sr)
        results[name] = result
        
        # Print summary
        print(f"  Quality Score: {result['quality_score']:.3f}")
        print(f"  Anomalies detected: {len(result['anomalies'])}")
        
        # Print spectral features
        features = result['features']
        print(f"  Spectral Centroid: {np.mean(features['spectral_centroid']):.3f}")
        print(f"  Spectral Flatness: {np.mean(features['spectral_flatness']):.3f}")
        
        # Print anomaly types
        anomaly_types = {}
        for anomaly in result['anomalies']:
            atype = anomaly['type']
            anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
        
        if anomaly_types:
            print("  Anomaly breakdown:")
            for atype, count in anomaly_types.items():
                print(f"    - {atype}: {count}")
    
    # Visualize results
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for idx, (name, result) in enumerate(results.items()):
        # Spectrogram
        ax = axes[idx, 0]
        spec = result['visualization']['spectrogram']
        librosa.display.specshow(spec, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(f'{name} - Spectrogram')
        ax.set_ylim(0, 8000)
        
        # Spectral features over time
        ax = axes[idx, 1]
        features = result['features']
        time_frames = np.arange(features['spectral_centroid'].shape[0]) * 512 / sr
        
        # Plot normalized features
        ax.plot(time_frames, features['spectral_centroid'], label='Centroid', alpha=0.8)
        ax.plot(time_frames, features['spectral_rolloff'], label='Rolloff', alpha=0.8)
        ax.plot(time_frames, features['spectral_flatness'], label='Flatness', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'{name} - Spectral Features')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Quality metrics
        ax = axes[idx, 2]
        metrics = analyzer.compute_quality_metrics(signals[name], sr)
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values)
        ax.set_title(f'{name} - Quality Metrics')
        ax.set_ylabel('Value')
        
        # Color bars based on value
        for bar, val in zip(bars, metric_values):
            if val > 0.8:
                bar.set_color('green')
            elif val > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add quality score as text
        ax.text(0.5, 0.95, f"Overall Quality: {result['quality_score']:.3f}", 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('spectral_analysis_demo.png', dpi=150)
    print("\nVisualization saved to 'spectral_analysis_demo.png'")
    
    # Demonstrate harmonic analysis
    print("\n4. Harmonic Analysis Demo...")
    harmonic_features = analyzer.compute_harmonic_features(harmonic, sr)
    
    print(f"  Fundamental Frequency: {harmonic_features['fundamental_frequency']:.1f} Hz")
    print(f"  Harmonic-to-Noise Ratio: {harmonic_features['harmonic_to_noise_ratio']:.1f} dB")
    print(f"  Total Harmonic Distortion: {harmonic_features['total_harmonic_distortion']:.2f}%")
    print(f"  Detected Harmonics: {len(harmonic_features['harmonics'])}")
    
    # Demonstrate formant analysis (for speech-like signal)
    print("\n5. Formant Analysis Demo...")
    # Generate a simple vowel-like signal
    vowel = librosa.tone(700, sr=sr, duration=1.0)  # Simplified vowel
    formants = analyzer.analyze_formants(vowel, sr)
    
    print(f"  Detected {len(formants)} formants:")
    for i, formant in enumerate(formants[:3]):  # Show first 3
        print(f"    F{i+1}: {formant['frequency']:.0f} Hz (bandwidth: {formant['bandwidth']:.0f} Hz)")


def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection capabilities."""
    print("\n\n=== Anomaly Detection Demo ===")
    
    sr = 16000
    analyzer = SpectralAnalyzer()
    
    # 1. Create signal with spectral hole
    print("\n1. Detecting spectral holes...")
    t = np.linspace(0, 1, sr)
    
    # White noise with notch filter
    noise = np.random.randn(sr)
    # Create notch at 2kHz
    from scipy import signal
    b, a = signal.iirnotch(2000 / (sr/2), 30)
    notched = signal.filtfilt(b, a, noise).astype(np.float32)
    
    result = analyzer.analyze(notched, sr)
    holes = [a for a in result['anomalies'] if a['type'] == 'spectral_hole']
    print(f"  Found {len(holes)} spectral holes")
    if holes:
        print(f"  Example: {holes[0]['frequency_range']} Hz")
    
    # 2. Detect harmonic distortion
    print("\n2. Detecting harmonic distortion...")
    clean = np.sin(2 * np.pi * 500 * t)
    # Add clipping distortion
    distorted = np.clip(clean, -0.5, 0.5).astype(np.float32)
    
    result = analyzer.analyze(distorted, sr)
    distortions = [a for a in result['anomalies'] if a['type'] == 'harmonic_distortion']
    print(f"  Found {len(distortions)} harmonic distortion anomalies")
    
    # 3. Real-world example with speech
    print("\n3. Analyzing real-world speech signal...")
    # Load a speech sample (you would use a real file here)
    # For demo, we'll synthesize a simple speech-like signal
    speech = generate_speech_like_signal(sr, 2.0)
    
    result = analyzer.analyze(speech, sr)
    print(f"  Quality Score: {result['quality_score']:.3f}")
    print(f"  Total anomalies: {len(result['anomalies'])}")
    
    # Group anomalies by type
    anomaly_summary = {}
    for anomaly in result['anomalies']:
        atype = anomaly['type']
        anomaly_summary[atype] = anomaly_summary.get(atype, 0) + 1
    
    print("  Anomaly summary:")
    for atype, count in anomaly_summary.items():
        print(f"    - {atype}: {count}")


def generate_speech_like_signal(sr, duration):
    """Generate a synthetic speech-like signal."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Combine multiple formants
    signal = np.zeros_like(t)
    
    # Vowel formants (simplified)
    formants = [(700, 130), (1220, 70), (2600, 160)]  # (freq, bandwidth)
    
    for freq, bw in formants:
        # Create formant using filtered noise
        noise = np.random.randn(len(t)) * 0.1
        # Simple resonant filter
        from scipy import signal as sig
        w0 = 2 * np.pi * freq / sr
        r = np.exp(-np.pi * bw / sr)
        b = [1 - r]
        a = [1, -2 * r * np.cos(w0), r**2]
        
        formant = sig.lfilter(b, a, noise)
        signal += formant
    
    # Add some pitch modulation
    pitch = 120 + 20 * np.sin(2 * np.pi * 3 * t)  # 3 Hz vibrato
    glottal = np.zeros_like(t)
    phase = 0
    for i in range(len(t)):
        phase += 2 * np.pi * pitch[i] / sr
        glottal[i] = np.sin(phase)
    
    # Modulate formants with glottal source
    speech = signal * (0.5 + 0.5 * glottal)
    
    # Normalize
    speech = speech / (np.max(np.abs(speech)) + 1e-8)
    
    return speech.astype(np.float32)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_spectral_analysis()
    demonstrate_anomaly_detection()
    
    print("\n✅ Spectral analysis demonstration complete!")
    print("Check 'spectral_analysis_demo.png' for visualizations.")