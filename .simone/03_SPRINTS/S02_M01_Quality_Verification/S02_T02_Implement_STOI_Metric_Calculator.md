# Task S02_T02: Implement STOI Metric Calculator

## Task Overview
Implement a Short-Time Objective Intelligibility (STOI) metric calculator that measures speech intelligibility for enhanced audio, providing objective assessments of how understandable the speech is.

## Technical Requirements

### Core Implementation
- **STOI Calculator Module** (`processors/audio_enhancement/metrics/stoi_calculator.py`)
  - Standard STOI and extended STOI (ESTOI) implementations
  - Frame-based processing with proper windowing
  - Correlation-based intelligibility measurement
  - Support for various sampling rates

### Key Features
1. **Dual Mode Support**
   - Standard STOI for general assessment
   - Extended STOI for better correlation with subjective scores
   - Automatic mode selection based on use case

2. **Robust Analysis**
   - Third-octave band analysis
   - Dynamic range compression modeling
   - Proper normalization and clipping handling

3. **Performance Optimization**
   - Vectorized operations for speed
   - Memory-efficient frame processing
   - Parallel computation support

## TDD Requirements

### Test Structure
```
tests/test_stoi_calculator.py
- test_standard_stoi_calculation()
- test_extended_stoi_calculation()
- test_different_sample_rates()
- test_frame_processing()
- test_intelligibility_ranges()
- test_noise_conditions()
```

### Test Data Requirements
- Clean speech samples
- Speech with various noise levels
- Pre-calculated STOI scores for validation
- Edge cases (silence, extreme noise)

## Implementation Approach

### Phase 1: Core Calculator
```python
class STOICalculator:
    def __init__(self, fs=16000, extended=False):
        self.fs = fs
        self.extended = extended
        self.frame_len = int(fs * 0.025)  # 25ms frames
        self.frame_shift = int(fs * 0.010)  # 10ms shift
    
    def calculate(self, clean, processed):
        # STOI calculation logic
        pass
    
    def calculate_extended(self, clean, processed):
        # Extended STOI with better correlation
        pass
```

### Phase 2: Advanced Features
- Multi-resolution analysis
- Frequency-weighted intelligibility
- Temporal dynamics consideration

### Phase 3: Integration
- Real-time monitoring dashboard
- Batch evaluation system
- Threshold-based quality gating

## Acceptance Criteria
1. ✅ STOI scores correlate with subjective intelligibility (r > 0.9)
2. ✅ Support for 8kHz, 16kHz, and 48kHz audio
3. ✅ Extended STOI implementation available
4. ✅ Processing speed > 100x real-time
5. ✅ Integration with quality verification pipeline

## Example Usage
```python
from processors.audio_enhancement.metrics import STOICalculator

# Standard STOI
calculator = STOICalculator(fs=16000)
score = calculator.calculate(clean_speech, enhanced_speech)
print(f"STOI Score: {score:.3f}")

# Extended STOI for better accuracy
ext_calculator = STOICalculator(fs=16000, extended=True)
ext_score = ext_calculator.calculate_extended(clean_speech, enhanced_speech)
print(f"Extended STOI: {ext_score:.3f}")
```

## Dependencies
- NumPy for numerical computations
- SciPy for signal processing
- `pystoi` library or custom implementation
- Audio utilities for preprocessing

## Performance Targets
- Single calculation: < 50ms for 10s audio
- Batch of 100: < 2 seconds
- Memory usage: < 200MB for batch processing
- Accuracy: Within 0.01 of reference implementation

## Notes
- STOI ranges from 0 to 1 (higher is better)
- Designed specifically for speech intelligibility
- More robust to certain distortions than PESQ
- Consider ESTOI for non-linear processing evaluation

## Mathematical Formulation

### STOI Algorithm Details

1. **Frame-Based Processing**
   ```python
   def compute_stoi(clean, degraded, fs=16000):
       # Frame parameters
       N = 512  # DFT size
       frame_len = 256  # ~16ms at 16kHz
       frame_shift = 128  # 50% overlap
       
       # Third-octave band analysis
       obm = thirdoct(fs, N, numBands=15)  # 15 bands from 150-4000 Hz
       
       # Segmentation
       clean_frames = segment_signal(clean, frame_len, frame_shift)
       degraded_frames = segment_signal(degraded, frame_len, frame_shift)
       
       # Process each frame
       d_values = []
       for clean_frame, deg_frame in zip(clean_frames, degraded_frames):
           # Apply window
           clean_win = clean_frame * np.hanning(frame_len)
           deg_win = deg_frame * np.hanning(frame_len)
           
           # DFT
           clean_dft = np.fft.rfft(clean_win, N)
           deg_dft = np.fft.rfft(deg_win, N)
           
           # Third-octave bands
           clean_bands = obm @ np.abs(clean_dft)**2
           deg_bands = obm @ np.abs(deg_dft)**2
           
           # Compute correlation coefficient
           d = compute_correlation(clean_bands, deg_bands)
           d_values.append(d)
       
       return np.mean(d_values)
   ```

2. **Third-Octave Band Filterbank**
   ```python
   def thirdoct(fs, N_fft, numBands=15):
       """Generate third-octave band filterbank matrix"""
       f = np.linspace(0, fs/2, N_fft//2 + 1)
       
       # Center frequencies from 150Hz to 4kHz
       cf = 150 * (10**(0.1 * np.arange(numBands)))
       
       # Create filterbank
       H = np.zeros((numBands, N_fft//2 + 1))
       
       for i in range(numBands):
           # Lower and upper cutoff frequencies
           fl = cf[i] / (10**0.05)
           fu = cf[i] * (10**0.05)
           
           # Triangular filter
           H[i, :] = (f >= fl) & (f <= fu)
           H[i, :] = H[i, :] / np.sum(H[i, :])
       
       return H
   ```

3. **Extended STOI (ESTOI)**
   ```python
   def compute_estoi(clean, degraded, fs=16000):
       """Extended STOI with better correlation to subjective scores"""
       # Standard STOI processing
       clean_bands, deg_bands = process_bands(clean, degraded, fs)
       
       # Intermediate intelligibility measure
       d_values = []
       for j in range(len(clean_bands)):
           # Normalize energy
           clean_norm = normalize_bands(clean_bands[j])
           deg_norm = normalize_bands(deg_bands[j])
           
           # Compute intermediate measure
           num = np.sum(clean_norm * deg_norm)
           denom = np.sqrt(np.sum(clean_norm**2) * np.sum(deg_norm**2))
           
           d_values.append(num / (denom + 1e-10))
       
       # Apply sigmoid mapping
       d_mean = np.mean(d_values)
       estoi = 1 / (1 + np.exp(-17.4906 * (d_mean - 0.5)))
       
       return estoi
   ```

### GPU Acceleration Strategies

1. **Batch FFT Processing**
   ```python
   import cupy as cp
   from cupy.cuda import Device
   
   class GPUSTOICalculator:
       def __init__(self, fs=16000, device_id=0):
           self.fs = fs
           self.device = Device(device_id)
           
           # Pre-allocate GPU arrays
           with self.device:
               self.fft_plan = cp.fft.rfft
               self.window = cp.hanning(256)
               self.obm = cp.asarray(self._create_filterbank())
       
       def batch_compute(self, clean_batch, degraded_batch):
           """Compute STOI for batch of audio pairs"""
           with self.device:
               # Transfer to GPU
               clean_gpu = cp.asarray(clean_batch)
               deg_gpu = cp.asarray(degraded_batch)
               
               # Batch segmentation
               clean_frames = self._batch_segment(clean_gpu)
               deg_frames = self._batch_segment(deg_gpu)
               
               # Parallel windowing
               clean_win = clean_frames * self.window[None, None, :]
               deg_win = deg_frames * self.window[None, None, :]
               
               # Batch FFT
               clean_fft = self.fft_plan(clean_win, axis=-1)
               deg_fft = self.fft_plan(deg_win, axis=-1)
               
               # Third-octave band analysis
               clean_bands = cp.abs(clean_fft)**2 @ self.obm.T
               deg_bands = cp.abs(deg_fft)**2 @ self.obm.T
               
               # Compute correlations
               stoi_scores = self._batch_correlation(clean_bands, deg_bands)
               
               return cp.asnumpy(stoi_scores)
   ```

2. **Memory-Efficient Streaming**
   ```python
   def streaming_stoi(audio_stream, reference, chunk_size=16000):
       """Compute STOI in streaming fashion"""
       calculator = StreamingSTOICalculator(reference)
       
       scores = []
       for chunk in audio_stream:
           # Update internal buffers
           calculator.update(chunk)
           
           # Compute STOI for current window
           if calculator.has_enough_data():
               score = calculator.compute_current()
               scores.append(score)
       
       return scores
   ```

### Calibration and Validation Procedures

1. **Correlation with Subjective Scores**
   ```python
   def validate_correlation(stoi_calculator, test_database):
       """Validate STOI correlation with MOS scores"""
       stoi_scores = []
       mos_scores = []
       
       for sample in test_database:
           stoi = stoi_calculator.calculate(sample['clean'], sample['degraded'])
           mos = sample['subjective_score']
           
           stoi_scores.append(stoi)
           mos_scores.append(mos)
       
       # Compute correlation
       r_pearson = np.corrcoef(stoi_scores, mos_scores)[0, 1]
       r_spearman = spearmanr(stoi_scores, mos_scores)[0]
       
       print(f"Pearson correlation: {r_pearson:.3f}")
       print(f"Spearman correlation: {r_spearman:.3f}")
       
       # Should exceed 0.9 for good performance
       assert r_pearson > 0.9, "Insufficient correlation with subjective scores"
   ```

2. **Noise Robustness Testing**
   ```python
   def test_noise_robustness(calculator):
       """Test STOI behavior under various noise conditions"""
       clean_speech = load_clean_speech()
       
       noise_types = ['white', 'babble', 'traffic', 'music']
       snr_levels = [-5, 0, 5, 10, 15, 20]
       
       results = {}
       for noise_type in noise_types:
           noise = load_noise(noise_type)
           results[noise_type] = []
           
           for snr in snr_levels:
               # Mix at specified SNR
               noisy = mix_at_snr(clean_speech, noise, snr)
               
               # Compute STOI
               stoi = calculator.calculate(clean_speech, noisy)
               results[noise_type].append((snr, stoi))
       
       # Verify monotonic increase with SNR
       for noise_type, scores in results.items():
           snrs, stois = zip(*scores)
           assert all(s1 <= s2 for s1, s2 in zip(stois, stois[1:])), \
               f"Non-monotonic STOI for {noise_type} noise"
   ```

### Industry Standard Compliance

1. **Reference Implementation Compatibility**
   ```python
   def ensure_compatibility():
       """Ensure compatibility with reference STOI implementation"""
       # Test against official STOI implementation
       from pystoi import stoi as reference_stoi
       
       test_cases = load_standard_test_cases()
       
       for case in test_cases:
           our_result = our_stoi(case['clean'], case['degraded'], case['fs'])
           ref_result = reference_stoi(case['clean'], case['degraded'], case['fs'])
           
           # Must match within numerical precision
           assert np.abs(our_result - ref_result) < 1e-4, \
               f"Mismatch: {our_result} vs {ref_result}"
   ```

2. **Performance Benchmarks**
   ```python
   def benchmark_performance():
       """Benchmark against performance targets"""
       import time
       
       # Test data
       audio_10s = np.random.randn(16000 * 10)  # 10 seconds
       batch_100 = [(np.random.randn(16000 * 10), np.random.randn(16000 * 10)) 
                    for _ in range(100)]
       
       calculator = OptimizedSTOICalculator()
       
       # Single calculation
       start = time.time()
       score = calculator.calculate(audio_10s, audio_10s)
       single_time = (time.time() - start) * 1000
       print(f"Single calculation: {single_time:.1f}ms")
       assert single_time < 50  # Target: < 50ms
       
       # Batch processing
       start = time.time()
       scores = calculator.batch_calculate(batch_100)
       batch_time = time.time() - start
       print(f"Batch of 100: {batch_time:.1f}s")
       assert batch_time < 2  # Target: < 2s
   ```

### Real-Time Monitoring Integration

```python
class RealTimeSTOIMonitor:
    def __init__(self, reference_audio, window_size=3.0, update_rate=10):
        self.reference = reference_audio
        self.window_size = window_size
        self.update_rate = update_rate
        self.calculator = OptimizedSTOICalculator()
        
        # Circular buffer for streaming
        self.buffer_size = int(window_size * 16000)
        self.buffer = np.zeros(self.buffer_size)
        self.buffer_idx = 0
        
    def update(self, audio_chunk):
        """Update with new audio chunk and compute STOI"""
        chunk_len = len(audio_chunk)
        
        # Update circular buffer
        if self.buffer_idx + chunk_len <= self.buffer_size:
            self.buffer[self.buffer_idx:self.buffer_idx + chunk_len] = audio_chunk
            self.buffer_idx += chunk_len
        else:
            # Wrap around
            first_part = self.buffer_size - self.buffer_idx
            self.buffer[self.buffer_idx:] = audio_chunk[:first_part]
            self.buffer[:chunk_len - first_part] = audio_chunk[first_part:]
            self.buffer_idx = chunk_len - first_part
        
        # Compute STOI on window
        if self.buffer_idx == 0:  # Full buffer
            return self.calculator.calculate(
                self.reference[:self.buffer_size], 
                self.buffer
            )
        
        return None
```