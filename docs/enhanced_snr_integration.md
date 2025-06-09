# Enhanced SNR Calculator Integration Guide

## Overview

The Enhanced SNR Calculator Module provides comprehensive Signal-to-Noise Ratio calculation capabilities for the autonomous audio quality assessment system. It extends the existing SNR functionality with multiple calculation methods, VAD integration, and advanced features.

## Key Features

- **Multiple SNR Methods**: Waveform, spectral, segmental, and VAD-enhanced calculations
- **Noise Estimation**: Minimum statistics, percentile-based, and adaptive tracking
- **VAD Integration**: Energy-based VAD with hooks for WebRTC and PyAnnote
- **Advanced Features**: Multiband analysis, confidence scoring, perceptual weighting
- **Backward Compatible**: Works with existing codebase through legacy mode

## Installation

No additional dependencies required beyond the existing project requirements:
```bash
# Optional for enhanced VAD support
pip install webrtcvad  # For WebRTC VAD
```

## Basic Usage

```python
from utils.enhanced_snr_calculator import EnhancedSNRCalculator

# Basic usage with automatic method selection
calculator = EnhancedSNRCalculator()
snr = calculator.calculate_snr(audio, sample_rate)

# Specify method
snr = calculator.calculate_snr(audio, sample_rate, method='spectral')

# With confidence score
snr, confidence = calculator.calculate_snr_with_confidence(audio, sample_rate)
```

## Integration Points

### 1. Audio Enhancement Pipeline

```python
# In processors/audio_enhancement/quality_validator.py
from utils.enhanced_snr_calculator import EnhancedSNRCalculator

class QualityValidator:
    def __init__(self):
        self.snr_calculator = EnhancedSNRCalculator()
    
    def validate_enhancement(self, original, enhanced, sr):
        # Use multiband SNR for detailed assessment
        bands = [(0, 500), (500, 2000), (2000, 8000)]
        snr_improvement = []
        
        for band in bands:
            orig_snr = self.snr_calculator.calculate_multiband_snr(original, sr, [band])[0]
            enh_snr = self.snr_calculator.calculate_multiband_snr(enhanced, sr, [band])[0]
            snr_improvement.append(enh_snr - orig_snr)
        
        return snr_improvement
```

### 2. Audio Metrics Integration

```python
# In utils/audio_metrics.py
from utils.enhanced_snr_calculator import EnhancedSNRCalculator

class AudioQualityMetrics:
    def __init__(self):
        self.enhanced_snr = EnhancedSNRCalculator()
    
    def calculate_comprehensive_snr(self, clean, noisy, fs):
        """Calculate SNR using multiple methods for robustness."""
        methods = ['waveform', 'spectral', 'vad_enhanced']
        snr_values = []
        
        for method in methods:
            snr = self.enhanced_snr.calculate_snr(noisy, fs, method=method)
            snr_values.append(snr)
        
        # Return median for robustness
        return np.median(snr_values)
```

### 3. Main Processing Pipeline

```python
# In main.py or processors
from utils.enhanced_snr_calculator import EnhancedSNRCalculator, SNRConfig

# Configure for production use
snr_config = SNRConfig(
    vad_enabled=True,
    vad_backend="energy",  # or "webrtcvad" if available
    noise_estimation_method="minimum_statistics",
    use_cache=True  # Enable caching for repeated calculations
)

calculator = EnhancedSNRCalculator(snr_config)

# Use in processing loop
for audio_file in audio_files:
    audio, sr = load_audio(audio_file)
    
    # Get SNR with confidence
    snr, confidence = calculator.calculate_snr_with_confidence(audio, sr)
    
    # Make decisions based on SNR and confidence
    if snr < 15 and confidence > 0.8:
        # Low SNR with high confidence - needs enhancement
        enhanced = enhance_audio(audio, sr)
```

## Configuration Examples

### High-Quality Voice Cloning

```python
config = SNRConfig(
    frame_size=0.025,
    frame_shift=0.010,
    vad_enabled=True,
    vad_backend="webrtcvad",
    vad_aggressiveness=2,
    noise_estimation_method="minimum_statistics",
    noise_bias_compensation=1.5
)
```

### Real-Time Processing

```python
config = SNRConfig(
    frame_size=0.020,  # Smaller frames
    frame_shift=0.010,
    vad_enabled=False,  # Disable for speed
    noise_estimation_method="percentile",  # Faster
    use_cache=False  # No caching for streaming
)
```

### Noisy Environment

```python
config = SNRConfig(
    vad_enabled=True,
    vad_energy_threshold=-45,  # More sensitive
    noise_estimation_method="minimum_statistics",
    noise_window_size=2.0,  # Longer window for better estimation
    noise_bias_compensation=2.0  # More aggressive
)
```

## Performance Optimization

### Caching

The calculator includes built-in caching for repeated calculations:

```python
calculator = EnhancedSNRCalculator(SNRConfig(use_cache=True))

# First call computes
snr1 = calculator.calculate_snr(audio, sr)  # Computed

# Second call uses cache
snr2 = calculator.calculate_snr(audio, sr)  # From cache
```

### Batch Processing

For processing multiple files:

```python
# Process in parallel (future enhancement)
from concurrent.futures import ProcessPoolExecutor

def process_file(file_path):
    audio, sr = load_audio(file_path)
    calculator = EnhancedSNRCalculator()
    return calculator.calculate_snr(audio, sr)

with ProcessPoolExecutor() as executor:
    snr_values = list(executor.map(process_file, file_paths))
```

## Troubleshooting

### Issue: SNR values seem too low

```python
# Check noise estimation
calculator = EnhancedSNRCalculator()
noise_power = calculator.noise_estimator.estimate_noise_floor(audio, sr)
print(f"Estimated noise power: {noise_power}")
print(f"Signal power: {np.mean(audio**2)}")

# Try different methods
for method in ['waveform', 'spectral', 'percentile']:
    snr = calculator.calculate_snr(audio, sr, method=method)
    print(f"{method}: {snr} dB")
```

### Issue: Inconsistent results

```python
# Use confidence scoring
snr, confidence = calculator.calculate_snr_with_confidence(audio, sr)
if confidence < 0.7:
    print("Low confidence - results may be unreliable")
    # Try alternative approach
```

### Issue: Performance problems

```python
# Profile the calculation
import time

start = time.time()
snr = calculator.calculate_snr(audio, sr)
print(f"Time: {time.time() - start:.3f}s")

# Optimize configuration
config = SNRConfig(
    vad_enabled=False,  # Disable VAD
    noise_estimation_method="percentile",  # Faster
    use_cache=True  # Enable caching
)
```

## Migration from Legacy SNR

To migrate from the existing `SNRMeasurement`:

```python
# Old code
from utils.snr_measurement import SNRMeasurement
snr_calc = SNRMeasurement()
snr = snr_calc.measure_snr(audio, sr)

# New code (backward compatible)
from utils.enhanced_snr_calculator import EnhancedSNRCalculator
snr_calc = EnhancedSNRCalculator()
snr = snr_calc.calculate_snr(audio, sr, method='legacy')

# Or use enhanced features
snr = snr_calc.calculate_snr(audio, sr, method='auto')
```

## Future Enhancements

1. **GPU Acceleration**: For large-scale processing
2. **Streaming Mode**: For real-time applications
3. **Machine Learning**: Learned noise estimation
4. **Additional VAD Backends**: Silero, custom models

## References

- Task specification: S01_T03_Build_SNR_Calculator_Module.md
- Implementation: utils/enhanced_snr_calculator.py
- Tests: tests/test_enhanced_snr_calculator.py
- Examples: examples/test_enhanced_snr.py