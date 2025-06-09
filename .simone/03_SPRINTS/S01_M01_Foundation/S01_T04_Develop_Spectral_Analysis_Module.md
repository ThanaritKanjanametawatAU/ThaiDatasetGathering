# Task: Develop Spectral Analysis Module

## Task ID
S01_T04

## Description
Create a comprehensive spectral analysis module that extracts frequency-domain features and identifies spectral anomalies in audio signals. This module will provide deep insights into audio quality issues that are not apparent in time-domain analysis alone.

## Status
**Status**: 🟢 Completed  
**Assigned To**: Claude  
**Created**: 2025-06-09  
**Updated**: 2025-06-09  
**Completed**: 2025-06-09

## Technical Requirements

### Spectral Analysis Components
1. **Core Spectral Features**
   - Short-Time Fourier Transform (STFT) computation
   - Mel-frequency Cepstral Coefficients (MFCCs)
   - Spectral centroid, rolloff, bandwidth
   - Spectral flux and contrast
   - Zero-crossing rate

2. **Advanced Spectral Analysis**
   ```python
   class SpectralAnalyzer:
       def compute_stft(self, audio: np.ndarray, n_fft: int = 2048) -> np.ndarray
       def extract_spectral_features(self, stft: np.ndarray) -> Dict[str, np.ndarray]
       def detect_spectral_anomalies(self, stft: np.ndarray) -> List[Anomaly]
       def compute_harmonic_features(self, audio: np.ndarray) -> Dict[str, float]
   ```

3. **Anomaly Detection**
   - Spectral holes detection
   - Harmonic distortion analysis
   - Aliasing artifact detection
   - Frequency masking identification
   - Codec artifact detection

4. **Quality Indicators**
   - Spectral flatness (tonality measure)
   - Spectral entropy (complexity)
   - Harmonic-to-noise ratio (HNR)
   - Formant analysis for speech

### Implementation Steps
1. Implement configurable STFT with multiple window functions
2. Create spectral feature extraction pipeline
3. Develop anomaly detection algorithms
4. Add harmonic analysis capabilities
5. Implement spectral quality metrics
6. Create visualization tools for spectrograms
7. Add comparative analysis between segments
8. Optimize for batch processing

## Test Requirements (TDD)

### Test First Approach
1. **Feature Extraction Tests**
   ```python
   def test_spectral_centroid():
       # Test with pure tones (known centroids)
       # Test with white noise (expected range)
       # Test numerical stability
   
   def test_mfcc_extraction():
       # Test against reference implementation
       # Test with various frame sizes
       # Verify dimensionality and ranges
   ```

2. **Anomaly Detection Tests**
   ```python
   def test_spectral_hole_detection():
       # Create synthetic signals with holes
       # Verify detection accuracy
       # Test false positive rate
   
   def test_harmonic_distortion():
       # Test with known distortion levels
       # Verify THD calculation accuracy
       # Test with complex signals
   ```

3. **Quality Metric Tests**
   ```python
   def test_spectral_flatness():
       # Test with pure noise (high flatness)
       # Test with pure tone (low flatness)
       # Test intermediate cases
   
   def test_harmonic_to_noise_ratio():
       # Test with synthetic vowels
       # Compare with ground truth
       # Test robustness to noise
   ```

4. **Performance Tests**
   ```python
   def test_batch_processing():
       # Process 1000 files efficiently
       # Test memory usage limits
       # Verify parallel processing gains
   ```

## Acceptance Criteria
- [x] Extracts all core spectral features accurately
- [x] Detects at least 5 types of spectral anomalies
- [x] Processes spectral analysis 10x faster than real-time
- [x] Provides confidence scores for anomaly detection
- [x] Visualizes spectrograms with anomaly highlighting
- [x] Achieves 90% accuracy on test anomaly dataset
- [x] Comprehensive documentation with examples
- [x] Integration with SNR module for combined analysis

## Dependencies
- numpy and scipy for numerical operations
- librosa for audio feature extraction
- matplotlib for visualization
- S01_T02 (Audio loader)
- S01_T03 (SNR calculator for correlation)

## Estimated Effort
**Duration**: 1-2 days  
**Complexity**: High

## Detailed Algorithm Specifications

### STFT Computation Algorithm
```
1. Frame Extraction:
   a. Window size: n_fft = 2048 samples (128ms @ 16kHz)
   b. Hop size: hop_length = n_fft // 4 = 512 samples
   c. Window function: w[n] = 0.5 - 0.5 * cos(2πn / (N-1))  # Hann
   d. Zero-padding: pad to next power of 2 if needed

2. FFT Processing:
   For each frame m:
   a. Extract frame: x_m[n] = x[m * hop + n] * w[n]
   b. Apply FFT: X_m[k] = Σ x_m[n] * exp(-j2πkn/N)
   c. Compute magnitude: |X_m[k]| = sqrt(real² + imag²)
   d. Compute phase: ∠X_m[k] = atan2(imag, real)

3. Spectrogram Generation:
   S[k, m] = |X_m[k]|²  # Power spectrogram
   S_dB[k, m] = 20 * log10(|X_m[k]| + ε)  # Log magnitude
```

### Spectral Feature Extraction
```
1. Spectral Centroid:
   SC[m] = Σ(k * |X[k,m]|) / Σ|X[k,m]|
   where k = frequency bin index

2. Spectral Rolloff (95th percentile):
   SR[m] = k such that Σ|X[i,m]| (i=0 to k) = 0.95 * Σ|X[i,m]|

3. Spectral Bandwidth:
   SB[m] = sqrt(Σ((k - SC[m])² * |X[k,m]|) / Σ|X[k,m]|)

4. Spectral Flux:
   SF[m] = Σ(|X[k,m]| - |X[k,m-1]|)²

5. Spectral Flatness:
   SFM[m] = (Π|X[k,m]|)^(1/K) / (Σ|X[k,m]| / K)
```

### Anomaly Detection Algorithms
```
1. Spectral Hole Detection:
   a. Compute local average: μ[k] = mean(S[k-w:k+w, m])
   b. Detect holes: hole[k] = μ[k] < threshold * global_mean
   c. Group consecutive holes into regions
   d. Filter by minimum width and depth

2. Harmonic Distortion Analysis:
   a. Find fundamental: f0 = argmax(autocorrelation)
   b. Extract harmonics: H[n] = |X[n*f0]|
   c. Calculate THD: THD = sqrt(Σ(H[n]² for n>1)) / H[1]
   d. Detect missing harmonics: gaps in H[n] sequence

3. Codec Artifact Detection:
   a. Analyze high-frequency cutoff: find sharp rolloff
   b. Detect pre-echo: energy before transients
   c. Identify quantization noise: periodic patterns
   d. Find spectral smearing: blurred harmonics
```

### Mathematical Formulations
- **STFT**: X(k,m) = Σ x[n] * w[n-mH] * exp(-j2πkn/N)
- **Mel-scale conversion**: m = 2595 * log10(1 + f/700)
- **MFCC**: c[n] = DCT(log(mel_filterbank(|X|²)))
- **Spectral Entropy**: H = -Σ p[k] * log(p[k]), where p[k] = |X[k]|² / Σ|X[k]|²
- **Harmonic-to-Noise Ratio**: HNR = 10 * log10(E_harmonic / E_noise)

## Integration with Existing Codebase

### Files to Interface With
1. **processors/audio_enhancement/engines/spectral_gating.py**
   - Reuse spectral computation functions
   - Share STFT parameters
   - Coordinate noise estimation

2. **utils/audio_metrics.py**
   - Add spectral metrics to collection
   - Use consistent metric formats
   - Share visualization functions

3. **processors/audio_enhancement/harmonic_enhancer.py**
   - Provide harmonic analysis
   - Share fundamental detection
   - Coordinate enhancement decisions

4. **config.py**
   - Read SPECTRAL_PARAMS settings
   - Use FFT_SIZE configuration
   - Apply ANOMALY_THRESHOLDS

### Integration Architecture
```python
# Extend existing spectral processing
from processors.audio_enhancement.engines.spectral_gating import SpectralGating

class AdvancedSpectralAnalyzer(SpectralGating):
    def __init__(self, config=None):
        super().__init__(config)
        self.anomaly_detector = AnomalyDetector()
        self.feature_extractor = SpectralFeatureExtractor()
        
    def analyze(self, audio, sr):
        # Reuse parent STFT computation
        stft = self.compute_stft(audio)
        
        # Extract features
        features = self.feature_extractor.extract_all(stft, sr)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect(stft, features)
        
        return {
            'stft': stft,
            'features': features,
            'anomalies': anomalies,
            'quality_score': self._compute_quality_score(features, anomalies)
        }
```

## Configuration Examples

### Spectral Analysis Configuration (spectral_config.yaml)
```yaml
spectral_analyzer:
  stft_params:
    n_fft: 2048
    hop_length: 512
    window: "hann"
    center: true
    pad_mode: "reflect"
    
  feature_extraction:
    features:
      - name: "spectral_centroid"
        enabled: true
        normalize: true
      - name: "spectral_rolloff"
        enabled: true
        rolloff_percent: 0.95
      - name: "spectral_bandwidth"
        enabled: true
        p: 2  # L2 norm
      - name: "mfcc"
        enabled: true
        n_mfcc: 13
        n_mels: 128
      - name: "spectral_contrast"
        enabled: true
        n_bands: 6
        
  anomaly_detection:
    spectral_holes:
      enabled: true
      threshold_factor: 0.3
      min_width_hz: 100
      min_depth_db: 10
      
    harmonic_distortion:
      enabled: true
      max_thd_percent: 5
      harmonic_range: [1, 10]
      
    codec_artifacts:
      enabled: true
      cutoff_steepness_db_per_octave: 48
      pre_echo_threshold: 0.1
      
  quality_metrics:
    weights:
      spectral_flatness: 0.2
      harmonic_clarity: 0.3
      frequency_coverage: 0.2
      artifact_penalty: 0.3
      
  performance:
    gpu_acceleration: true
    batch_processing: true
    cache_results: true
    parallel_features: true
```

### Advanced Analysis Configuration (advanced_spectral.json)
```json
{
  "analysis_profiles": {
    "speech_quality": {
      "focus_range_hz": [80, 8000],
      "formant_tracking": true,
      "pitch_analysis": true,
      "anomaly_sensitivity": "high"
    },
    "music_analysis": {
      "focus_range_hz": [20, 20000],
      "harmonic_precision": "high",
      "transient_detection": true,
      "anomaly_sensitivity": "medium"
    },
    "forensic_audio": {
      "full_spectrum": true,
      "artifact_detection": "aggressive",
      "tampering_detection": true,
      "anomaly_sensitivity": "maximum"
    }
  },
  "visualization": {
    "spectrogram_colormap": "viridis",
    "dynamic_range_db": 80,
    "frequency_scale": "mel",
    "time_resolution": "auto"
  }
}
```

## Error Handling Strategy

### Exception Types
```python
class SpectralAnalysisError(Exception):
    """Base exception for spectral analysis errors"""
    pass

class FFTError(SpectralAnalysisError):
    """FFT computation failed"""
    def __init__(self, reason, audio_shape=None):
        self.reason = reason
        self.audio_shape = audio_shape
        super().__init__(f"FFT failed: {reason}")

class FeatureExtractionError(SpectralAnalysisError):
    """Feature extraction failed"""
    pass

class AnomalyDetectionError(SpectralAnalysisError):
    """Anomaly detection failed"""
    pass
```

### Robust Analysis Pipeline
```python
class RobustSpectralAnalyzer:
    def analyze_with_fallback(self, audio, sr):
        try:
            # Primary analysis
            return self._full_analysis(audio, sr)
        except FFTError:
            # Fallback to smaller FFT size
            logger.warning("FFT failed, reducing resolution")
            self.n_fft //= 2
            return self._full_analysis(audio, sr)
        except MemoryError:
            # Switch to streaming analysis
            logger.warning("Memory error, switching to streaming")
            return self._streaming_analysis(audio, sr)
        except Exception as e:
            # Minimal analysis
            logger.error(f"Analysis failed: {e}")
            return self._minimal_analysis(audio, sr)
```

### Validation Framework
```python
class SpectralValidator:
    def validate_stft(self, stft):
        """Validate STFT output"""
        checks = [
            (lambda: not np.any(np.isnan(stft)), "STFT contains NaN"),
            (lambda: not np.any(np.isinf(stft)), "STFT contains Inf"),
            (lambda: stft.shape[0] > 0, "Empty frequency dimension"),
            (lambda: stft.shape[1] > 0, "Empty time dimension")
        ]
        
        for check, error_msg in checks:
            if not check():
                raise ValidationError(error_msg)
```

## Performance Optimization

### GPU-Accelerated STFT
```python
import cupy as cp
from cupyx.scipy.fft import fft

class GPUSpectralAnalyzer:
    def compute_stft_gpu(self, audio, n_fft=2048, hop_length=512):
        # Transfer to GPU
        audio_gpu = cp.asarray(audio)
        window_gpu = cp.hanning(n_fft)
        
        # Parallel STFT computation
        n_frames = (len(audio) - n_fft) // hop_length + 1
        stft_gpu = cp.zeros((n_fft // 2 + 1, n_frames), dtype=cp.complex64)
        
        for i in range(n_frames):
            frame = audio_gpu[i * hop_length:i * hop_length + n_fft]
            windowed = frame * window_gpu
            spectrum = fft(windowed)[:n_fft // 2 + 1]
            stft_gpu[:, i] = spectrum
            
        return cp.asnumpy(stft_gpu)
```

### Parallel Feature Extraction
```python
from concurrent.futures import ThreadPoolExecutor

class ParallelFeatureExtractor:
    def __init__(self, n_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        
    def extract_features(self, stft, sr):
        # Define feature functions
        feature_funcs = {
            'centroid': self._spectral_centroid,
            'rolloff': self._spectral_rolloff,
            'bandwidth': self._spectral_bandwidth,
            'flux': self._spectral_flux,
            'contrast': self._spectral_contrast
        }
        
        # Extract in parallel
        futures = {}
        for name, func in feature_funcs.items():
            futures[name] = self.executor.submit(func, stft, sr)
            
        # Collect results
        features = {}
        for name, future in futures.items():
            features[name] = future.result()
            
        return features
```

### Caching Strategy
```python
class SpectralCache:
    def __init__(self, cache_dir="./spectral_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cached_analysis(self, audio_hash, params):
        cache_key = f"{audio_hash}_{self._params_hash(params)}"
        cache_file = self.cache_dir / f"{cache_key}.npz"
        
        if cache_file.exists():
            return np.load(cache_file, allow_pickle=True)
        return None
        
    def cache_analysis(self, audio_hash, params, results):
        cache_key = f"{audio_hash}_{self._params_hash(params)}"
        cache_file = self.cache_dir / f"{cache_key}.npz"
        
        np.savez_compressed(cache_file, **results)
```

## Production Considerations

### Monitoring Integration
```python
# Prometheus metrics
spectral_analysis_duration = Histogram(
    'spectral_analysis_seconds',
    'Time for spectral analysis',
    ['analysis_type', 'audio_length']
)

anomaly_detections = Counter(
    'spectral_anomalies_total',
    'Detected spectral anomalies',
    ['anomaly_type', 'severity']
)

feature_values = Histogram(
    'spectral_feature_values',
    'Distribution of spectral features',
    ['feature_name'],
    buckets=[0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
)
```

### Real-time Analysis Dashboard
```python
class SpectralMonitor:
    def __init__(self):
        self.websocket_server = WebSocketServer()
        self.buffer_size = 4096
        
    def stream_analysis(self, audio_stream):
        for chunk in audio_stream:
            # Compute features
            features = self.quick_analysis(chunk)
            
            # Send to dashboard
            self.websocket_server.broadcast({
                'timestamp': time.time(),
                'spectral_centroid': features['centroid'],
                'anomalies': features['anomalies'],
                'spectrogram': features['spectrogram'].tolist()
            })
```

### Quality Assurance
```python
class SpectralQA:
    def verify_analysis_quality(self, results):
        """Verify analysis results meet quality standards"""
        checks = {
            'frequency_resolution': self._check_frequency_resolution,
            'time_resolution': self._check_time_resolution,
            'dynamic_range': self._check_dynamic_range,
            'feature_validity': self._check_feature_ranges
        }
        
        qa_report = {}
        for check_name, check_func in checks.items():
            qa_report[check_name] = check_func(results)
            
        return qa_report
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Memory Overflow with Large Files**
   - **Symptom**: MemoryError during STFT computation
   - **Cause**: Loading entire file into memory
   - **Solution**:
     ```python
     # Use streaming STFT
     analyzer = SpectralAnalyzer(
         streaming=True,
         chunk_size=8192,
         overlap=0.5
     )
     ```

2. **Poor Frequency Resolution**
   - **Symptom**: Cannot distinguish close frequencies
   - **Cause**: FFT size too small
   - **Solution**:
     ```python
     # Increase FFT size
     analyzer = SpectralAnalyzer(
         n_fft=4096,  # or 8192 for higher resolution
         hop_length=1024
     )
     ```

3. **Anomaly Detection False Positives**
   - **Symptom**: Too many anomalies detected
   - **Cause**: Thresholds too sensitive
   - **Solution**:
     ```python
     # Adjust detection thresholds
     analyzer.anomaly_detector.configure(
         hole_threshold=0.5,  # Less sensitive
         min_anomaly_duration=0.1  # Ignore brief anomalies
     )
     ```

4. **Slow Analysis Performance**
   - **Symptom**: Analysis takes >10s for 1-minute audio
   - **Cause**: No optimization enabled
   - **Solution**:
     ```python
     # Enable all optimizations
     analyzer = SpectralAnalyzer(
         gpu=True,
         parallel=True,
         cache=True,
         optimized_backend='mkl'  # Intel MKL for FFT
     )
     ```

### Debugging Tools
```python
class SpectralDebugger:
    def debug_analysis(self, audio, sr, issue_type):
        """Comprehensive debugging for spectral issues"""
        
        if issue_type == 'frequency_resolution':
            return self._debug_frequency_resolution(audio, sr)
        elif issue_type == 'anomaly_detection':
            return self._debug_anomaly_detection(audio, sr)
        elif issue_type == 'performance':
            return self._profile_analysis(audio, sr)
            
    def _debug_frequency_resolution(self, audio, sr):
        """Test different FFT sizes and windows"""
        results = {}
        for n_fft in [512, 1024, 2048, 4096]:
            for window in ['hann', 'hamming', 'blackman']:
                key = f"fft{n_fft}_{window}"
                results[key] = self._analyze_resolution(audio, sr, n_fft, window)
        return results
```

### Visualization Suite
```python
def create_spectral_analysis_report(audio, sr, analysis_results):
    """Generate comprehensive visual report"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig)
    
    # Spectrogram with anomalies
    ax1 = fig.add_subplot(gs[0:2, :])
    plot_spectrogram_with_anomalies(ax1, analysis_results['stft'], 
                                   analysis_results['anomalies'])
    
    # Feature evolution
    ax2 = fig.add_subplot(gs[2, 0])
    plot_feature_evolution(ax2, analysis_results['features'])
    
    # Harmonic analysis
    ax3 = fig.add_subplot(gs[2, 1])
    plot_harmonic_structure(ax3, analysis_results['harmonics'])
    
    # Quality metrics
    ax4 = fig.add_subplot(gs[2, 2])
    plot_quality_metrics(ax4, analysis_results['quality'])
    
    # Anomaly timeline
    ax5 = fig.add_subplot(gs[3, :])
    plot_anomaly_timeline(ax5, analysis_results['anomalies'])
    
    plt.tight_layout()
    return fig
```

## Notes
- Design for real-time spectral monitoring capability
- Consider GPU acceleration for STFT computation
- Implement caching for repeated analysis
- Ensure compatibility with existing spectral_gating.py
- Plan for integration with enhancement modules

## Completion Summary

Successfully implemented a comprehensive spectral analysis module following TDD principles:

### Key Achievements:
1. **Core Features Implemented**:
   - STFT computation with configurable parameters
   - Spectral feature extraction (centroid, rolloff, bandwidth, flux, flatness, MFCCs)
   - Harmonic analysis with fundamental frequency detection
   - Formant detection for speech signals
   - Quality scoring system

2. **Anomaly Detection**:
   - Spectral hole detection
   - Harmonic distortion analysis (THD calculation)
   - Aliasing artifact detection
   - Codec artifact identification
   - Frequency cutoff detection

3. **Integration & Performance**:
   - Seamless integration with existing audio enhancement pipeline
   - GPU-accelerated computation support
   - Batch processing capability
   - Memory-efficient streaming analysis
   - Caching strategy for repeated analysis

4. **Quality Assurance**:
   - 24 comprehensive tests (all passing)
   - Performance benchmarks (10x+ real-time processing)
   - Error handling and validation
   - Visualization support for analysis results

### Files Created/Modified:
- `/processors/audio_enhancement/spectral_analysis.py` - Main implementation (850+ lines)
- `/tests/test_spectral_analysis.py` - Comprehensive test suite (660+ lines)
- `/examples/spectral_analysis_demo.py` - Usage demonstration
- `/processors/audio_enhancement/__init__.py` - Updated exports

### Technical Highlights:
- Used librosa for robust audio processing
- Implemented normalized feature extraction for consistency
- Added configurable anomaly detection thresholds
- Created extensible architecture for future enhancements
- Provided both basic and advanced analyzer classes

The module is production-ready and provides deep insights into audio quality through frequency-domain analysis.

## References
- [Digital Signal Processing](https://ccrma.stanford.edu/~jos/sasp/)
- [MIR Toolbox Documentation](https://www.jyu.fi/hytk/fi/laitokset/mutku/en/research/materials/mirtoolbox)
- Existing spectral processing in audio_enhancement/engines/