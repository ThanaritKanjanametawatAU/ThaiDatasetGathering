# Task: Build SNR Calculator Module

## Task ID
S01_T03

## Description
Develop a comprehensive Signal-to-Noise Ratio (SNR) calculation module that can accurately assess audio quality across different noise conditions. This module will be central to the autonomous quality assessment system, providing reliable metrics for decision-making.

## Status
**Status**: 🔴 Not Started  
**Assigned To**: Unassigned  
**Created**: 2025-06-09  
**Updated**: 2025-06-09

## Technical Requirements

### SNR Calculation Methods
1. **Multiple SNR Algorithms**
   - Waveform-based SNR (time domain)
   - Spectral SNR (frequency domain)
   - Segmental SNR for temporal analysis
   - Weighted SNR for perceptual relevance

2. **Noise Estimation Techniques**
   ```python
   class NoiseEstimator:
       def estimate_noise_floor(self, audio: np.ndarray, sr: int) -> float
       def detect_silence_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]
       def spectral_noise_estimation(self, stft: np.ndarray) -> np.ndarray
       def adaptive_noise_tracking(self, audio: np.ndarray) -> np.ndarray
   ```

3. **Advanced Features**
   - Voice Activity Detection (VAD) integration
   - Dynamic noise floor tracking
   - Multi-band SNR analysis
   - Perceptual weighting (A-weighting, ITU-R BS.1387)

4. **Quality Metrics**
   - Global SNR
   - Segmental SNR (frame-by-frame)
   - Frequency-weighted SNR
   - Peak SNR for transient detection

### Implementation Steps
1. Implement basic waveform SNR calculation
2. Add spectral domain SNR computation
3. Integrate VAD for accurate speech/noise separation
4. Implement noise floor estimation algorithms
5. Add perceptual weighting filters
6. Create multi-band analysis for detailed assessment
7. Implement confidence scoring for SNR estimates
8. Add visualization capabilities for SNR profiles

## Test Requirements (TDD)

### Test First Approach
1. **Basic SNR Tests**
   ```python
   def test_clean_signal_snr():
       # Test with synthetic clean signal
       # Expect high SNR (>40dB)
       # Verify numerical stability
   
   def test_known_snr_signals():
       # Test with signals of known SNR
       # Verify accuracy within 1dB
       # Test various noise types
   ```

2. **Noise Estimation Tests**
   ```python
   def test_silence_detection():
       # Test silence segment identification
       # Test with various threshold levels
       # Test edge cases (all silence, no silence)
   
   def test_noise_floor_estimation():
       # Test with stationary noise
       # Test with non-stationary noise
       # Test robustness to speech presence
   ```

3. **Advanced Feature Tests**
   ```python
   def test_vad_integration():
       # Test SNR calculation with VAD
       # Compare with/without VAD
       # Test VAD failure cases
   
   def test_perceptual_weighting():
       # Test A-weighting implementation
       # Test frequency response
       # Compare with reference implementations
   ```

4. **Edge Case Tests**
   ```python
   def test_extreme_conditions():
       # Test very low SNR (<0dB)
       # Test very high SNR (>60dB)
       # Test with clipped signals
       # Test with DC offset
   ```

## Acceptance Criteria
- [ ] Calculates SNR with <1dB error for known test signals
- [ ] Supports multiple SNR calculation methods
- [ ] Integrates VAD for improved accuracy
- [ ] Handles edge cases gracefully (silence, clipping)
- [ ] Processes 1-hour audio in <5 seconds
- [ ] Provides confidence scores for estimates
- [ ] 95% test coverage with comprehensive edge cases
- [ ] Clear documentation with mathematical foundations

## Dependencies
- numpy for numerical operations
- scipy.signal for signal processing
- S01_T02 (Audio loader for input data)
- webrtcvad or pyannote.audio for VAD

## Estimated Effort
**Duration**: 1-2 days  
**Complexity**: Medium-High

## Detailed Algorithm Specifications

### Waveform-based SNR Calculation
```
1. Signal Power Estimation:
   a. Compute signal energy: E_sig = Σ(x[n]²)
   b. Calculate RMS: RMS_sig = sqrt(E_sig / N)
   c. Convert to dB: P_sig = 20 * log10(RMS_sig)

2. Noise Floor Estimation (Minimum Statistics):
   a. Divide signal into frames (20-50ms)
   b. Compute power spectrum for each frame
   c. Track minimum power over sliding window
   d. Apply bias compensation: α = 1.5
   e. Noise floor = min_power * α

3. SNR Calculation:
   SNR_dB = 10 * log10(P_signal / P_noise)
```

### Spectral SNR Algorithm
```
1. STFT Computation:
   a. Window size: n_fft = 2048 (128ms @ 16kHz)
   b. Hop length: hop = n_fft // 4
   c. Window function: Hann window
   d. Apply STFT: S = stft(x, n_fft, hop)

2. Frequency-band SNR:
   For each frequency bin k:
   a. Signal power: P_s[k] = mean(|S[k, voiced_frames]|²)
   b. Noise power: P_n[k] = mean(|S[k, noise_frames]|²)
   c. SNR[k] = 10 * log10(P_s[k] / P_n[k])

3. Weighted Average:
   SNR_total = Σ(w[k] * SNR[k]) / Σ(w[k])
   where w[k] = A_weighting(freq[k])
```

### VAD-Enhanced SNR Calculation
```
1. Voice Activity Detection:
   a. Energy-based VAD: E > E_threshold
   b. Zero-crossing rate: ZCR < ZCR_threshold
   c. Spectral flatness: SF < SF_threshold
   d. Combine features: VAD = E_vad AND ZCR_vad AND SF_vad

2. Segment Classification:
   a. Speech segments: VAD == 1
   b. Noise segments: VAD == 0 for > 200ms
   c. Transition segments: excluded

3. Refined SNR:
   SNR = 10 * log10(mean(P_speech) / mean(P_noise))
```

### Mathematical Formulations
- **Signal Power**: P_s = (1/N) * Σ|x[n]|²
- **Noise Power (Welch's method)**: P_n = (1/K) * Σ|X_k(f)|²
- **Segmental SNR**: SNR_seg = (1/M) * Σ SNR_frame[m]
- **A-weighting**: A(f) = 12194² * f⁴ / ((f² + 20.6²) * sqrt((f² + 107.7²) * (f² + 737.9²)) * (f² + 12194²))
- **Perceptual SNR**: PSNR = SNR + 10 * log10(Σ A(f) * |S(f)|² / Σ |S(f)|²)

## Integration with Existing Codebase

### Files to Interface With
1. **utils/snr_measurement.py**
   - Extend `calculate_snr()` function
   - Add new SNR calculation methods
   - Maintain backward compatibility

2. **utils/audio_metrics.py**
   - Integrate with `AudioMetrics` class
   - Add SNR to metric collection
   - Use existing metric infrastructure

3. **processors/audio_enhancement/quality_validator.py**
   - Provide SNR for quality decisions
   - Interface with validation thresholds
   - Support enhancement targeting

4. **config.py**
   - Read SNR_THRESHOLD settings
   - Use FRAME_SIZE for segmentation
   - Apply VAD_PARAMS configuration

### Integration Example
```python
# Extend existing SNR measurement
from utils.snr_measurement import calculate_snr as legacy_snr

class EnhancedSNRCalculator:
    def __init__(self, config=None):
        self.vad = VoiceActivityDetector()
        self.noise_estimator = NoiseEstimator()
        self.config = config or self._default_config()
    
    def calculate_snr(self, audio, sr, method='auto'):
        if method == 'legacy':
            return legacy_snr(audio, sr)
        
        # Enhanced calculation
        vad_mask = self.vad.detect(audio, sr)
        noise_profile = self.noise_estimator.estimate(audio, vad_mask)
        
        return self._compute_snr_with_vad(audio, noise_profile, vad_mask)
```

## Configuration Examples

### SNR Calculator Configuration (snr_config.yaml)
```yaml
snr_calculator:
  methods:
    waveform:
      enabled: true
      frame_size: 0.025  # seconds
      frame_shift: 0.010  # seconds
      
    spectral:
      enabled: true
      n_fft: 2048
      hop_length: 512
      frequency_weighting: "A"  # A, C, or None
      
    segmental:
      enabled: true
      segment_length: 0.2  # seconds
      overlap: 0.5
      aggregation: "mean"  # mean, median, or percentile
      
  noise_estimation:
    method: "minimum_statistics"
    window_size: 1.5  # seconds
    bias_compensation: 1.5
    smoothing_factor: 0.98
    
  vad_integration:
    enabled: true
    backend: "webrtcvad"  # webrtcvad, pyannote, or energy
    aggressiveness: 2  # 0-3 for webrtcvad
    energy_threshold: -40  # dB
    min_speech_duration: 0.1  # seconds
    
  quality_thresholds:
    excellent: 35  # dB
    good: 25
    fair: 15
    poor: 5
    
  performance:
    cache_enabled: true
    parallel_processing: true
    gpu_acceleration: false
    batch_size: 100
```

### Advanced Configuration (advanced_snr.json)
```json
{
  "snr_profiles": {
    "voice_cloning": {
      "target_snr": 35,
      "method": "spectral",
      "frequency_range": [80, 8000],
      "perceptual_weighting": true
    },
    "general_speech": {
      "target_snr": 25,
      "method": "segmental",
      "vad_required": true
    },
    "noisy_environment": {
      "target_snr": 15,
      "method": "adaptive",
      "noise_tracking": "continuous"
    }
  },
  "algorithms": {
    "noise_estimation": {
      "minimum_statistics": {
        "window_ms": 1500,
        "bias": 1.5
      },
      "mcra": {
        "smoothing": 0.7,
        "prob_threshold": 0.8
      }
    }
  }
}
```

## Error Handling Strategy

### Exception Hierarchy
```python
class SNRError(Exception):
    """Base exception for SNR calculation errors"""
    pass

class InsufficientDataError(SNRError):
    """Not enough data for reliable SNR estimation"""
    def __init__(self, required_duration, actual_duration):
        self.required = required_duration
        self.actual = actual_duration
        super().__init__(f"Need {required_duration}s, got {actual_duration}s")

class NoSpeechDetectedError(SNRError):
    """No speech segments found for SNR calculation"""
    pass

class InvalidSignalError(SNRError):
    """Signal contains invalid values (NaN, Inf)"""
    pass
```

### Robust Calculation Strategy
```python
def calculate_snr_robust(audio, sr, config):
    try:
        # Validate input
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise InvalidSignalError("Signal contains NaN or Inf")
        
        # Check duration
        duration = len(audio) / sr
        if duration < config.min_duration:
            raise InsufficientDataError(config.min_duration, duration)
        
        # Try primary method
        return primary_snr_method(audio, sr)
        
    except NoSpeechDetectedError:
        # Fallback to energy-based method
        logger.warning("No speech detected, using energy-based SNR")
        return energy_based_snr(audio, sr)
        
    except Exception as e:
        # Final fallback
        logger.error(f"SNR calculation failed: {e}")
        return estimate_snr_simple(audio)
```

### Validation Pipeline
```python
class SNRValidator:
    def validate_snr_result(self, snr_value, audio_stats):
        checks = [
            (self.check_range, "SNR out of valid range"),
            (self.check_consistency, "SNR inconsistent with audio stats"),
            (self.check_confidence, "Low confidence in SNR estimate")
        ]
        
        for check, error_msg in checks:
            if not check(snr_value, audio_stats):
                logger.warning(f"SNR validation: {error_msg}")
                return False
        return True
```

## Performance Optimization

### GPU Acceleration
```python
import cupy as cp  # GPU arrays

class GPUSNRCalculator:
    def calculate_snr_gpu(self, audio, sr):
        # Transfer to GPU
        audio_gpu = cp.asarray(audio)
        
        # GPU-accelerated STFT
        stft_gpu = cp.fft.fft(audio_gpu.reshape(-1, self.n_fft), axis=1)
        
        # Parallel noise estimation
        noise_power = self._estimate_noise_gpu(stft_gpu)
        signal_power = self._estimate_signal_gpu(stft_gpu)
        
        # Calculate SNR
        snr = 10 * cp.log10(signal_power / noise_power)
        
        return cp.asnumpy(snr)  # Transfer back to CPU
```

### Streaming SNR Calculation
```python
class StreamingSNRCalculator:
    def __init__(self, sr, frame_size=0.025):
        self.sr = sr
        self.frame_samples = int(sr * frame_size)
        self.noise_tracker = OnlineNoiseTracker()
        self.signal_buffer = RingBuffer(capacity=sr * 2)  # 2 seconds
        
    def update(self, audio_chunk):
        self.signal_buffer.append(audio_chunk)
        
        # Update noise estimate
        self.noise_tracker.update(audio_chunk)
        
        # Calculate instantaneous SNR
        signal_power = np.mean(audio_chunk ** 2)
        noise_power = self.noise_tracker.get_noise_power()
        
        instant_snr = 10 * np.log10(signal_power / noise_power)
        
        # Smooth with history
        return self._smooth_snr(instant_snr)
```

### Caching System
```python
class SNRCache:
    def __init__(self, max_size=1000):
        self.cache = LRUCache(max_size)
        self.hash_func = xxhash.xxh64
        
    def get_or_compute(self, audio, sr, calculator_func):
        # Create cache key
        audio_hash = self.hash_func(audio.tobytes()).hexdigest()
        cache_key = f"{audio_hash}_{sr}"
        
        # Check cache
        if cached := self.cache.get(cache_key):
            return cached
        
        # Compute and cache
        result = calculator_func(audio, sr)
        self.cache.put(cache_key, result)
        return result
```

## Production Considerations

### Monitoring Metrics
```python
# Prometheus metrics
snr_calculation_time = Histogram(
    'snr_calculation_seconds',
    'Time to calculate SNR',
    ['method', 'audio_length']
)

snr_values = Histogram(
    'snr_values_db',
    'Distribution of calculated SNR values',
    buckets=[-10, 0, 10, 20, 30, 40, 50]
)

snr_errors = Counter(
    'snr_errors_total',
    'SNR calculation errors',
    ['error_type']
)
```

### Real-time Dashboard
```python
class SNRMonitor:
    def __init__(self):
        self.metrics = {
            'current_snr': Gauge('current_snr_db', 'Current SNR value'),
            'avg_snr': Gauge('average_snr_db', 'Average SNR over window'),
            'min_snr': Gauge('min_snr_db', 'Minimum SNR in window')
        }
        
    def update(self, snr_value):
        self.metrics['current_snr'].set(snr_value)
        self.update_statistics(snr_value)
```

### Configuration Management
```yaml
# Kubernetes ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: snr-calculator-config
data:
  config.yaml: |
    snr:
      default_method: "spectral"
      target_snr: 35
      min_acceptable_snr: 15
      vad:
        enabled: true
        backend: "webrtcvad"
      caching:
        enabled: true
        ttl: 3600
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Incorrect SNR Values (Too High)**
   - **Symptom**: SNR > 50dB for normal speech
   - **Cause**: Incorrect noise estimation
   - **Solution**:
     ```python
     # Adjust noise estimation parameters
     calculator = SNRCalculator(
         noise_bias=2.0,  # Increase bias
         min_noise_duration=0.5  # Require longer silence
     )
     ```

2. **Negative SNR for Clean Audio**
   - **Symptom**: Clean audio shows negative SNR
   - **Cause**: VAD failing to detect speech
   - **Solution**:
     ```python
     # Lower VAD threshold
     vad_config = {
         'energy_threshold': -45,  # More sensitive
         'aggressiveness': 1  # Less aggressive
     }
     ```

3. **Inconsistent Results**
   - **Symptom**: Same audio gives different SNR
   - **Cause**: Non-deterministic noise estimation
   - **Solution**:
     ```python
     # Use deterministic algorithm
     calculator = SNRCalculator(
         method='fixed_percentile',
         noise_percentile=10,
         random_seed=42
     )
     ```

4. **Performance Issues**
   - **Symptom**: SNR calculation takes >1s
   - **Cause**: Inefficient implementation
   - **Solution**:
     ```python
     # Enable optimizations
     calculator = SNRCalculator(
         use_gpu=True,
         parallel=True,
         cache_enabled=True
     )
     ```

### Debug Utilities
```python
class SNRDebugger:
    def debug_snr_calculation(self, audio, sr):
        """Detailed SNR analysis with intermediate results"""
        results = {
            'input_stats': self.analyze_input(audio, sr),
            'vad_results': self.debug_vad(audio, sr),
            'noise_profile': self.analyze_noise(audio, sr),
            'snr_methods': {
                'waveform': self.calc_waveform_snr(audio, sr),
                'spectral': self.calc_spectral_snr(audio, sr),
                'segmental': self.calc_segmental_snr(audio, sr)
            },
            'visualizations': self.generate_plots(audio, sr)
        }
        return results

# Usage
debugger = SNRDebugger()
debug_info = debugger.debug_snr_calculation(audio, sr)
debug_info.save_report('snr_debug_report.html')
```

### Visualization Tools
```python
def visualize_snr_analysis(audio, sr, snr_results):
    """Create comprehensive SNR visualization"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Waveform with VAD
    axes[0, 0].plot(audio)
    axes[0, 0].fill_between(range(len(audio)), 
                           audio * vad_mask, 
                           alpha=0.3)
    axes[0, 0].set_title('Waveform with VAD')
    
    # Spectrogram with noise floor
    axes[0, 1].specgram(audio, Fs=sr)
    axes[0, 1].plot(noise_floor, 'r--')
    axes[0, 1].set_title('Spectrogram with Noise Floor')
    
    # SNR over time
    axes[1, 0].plot(time_axis, segmental_snr)
    axes[1, 0].set_title('SNR over Time')
    
    # Frequency-dependent SNR
    axes[1, 1].plot(frequencies, spectral_snr)
    axes[1, 1].set_title('SNR by Frequency')
    
    # Noise power spectrum
    axes[2, 0].plot(noise_spectrum)
    axes[2, 0].set_title('Noise Power Spectrum')
    
    # SNR distribution
    axes[2, 1].hist(snr_values, bins=50)
    axes[2, 1].set_title('SNR Distribution')
    
    plt.tight_layout()
    return fig
```

## Notes
- Consider GPU acceleration for large-scale processing
- Implement streaming SNR calculation for real-time analysis
- Design for integration with existing utils/snr_measurement.py
- Consider compatibility with 35dB enhancement requirements
- Cache computed SNR values for efficiency

## References
- [ITU-T P.563 Standard](https://www.itu.int/rec/T-REC-P.563)
- [Speech Enhancement Theory](https://www.springer.com/gp/book/9783642002953)
- Existing SNR implementation in utils/snr_measurement.py