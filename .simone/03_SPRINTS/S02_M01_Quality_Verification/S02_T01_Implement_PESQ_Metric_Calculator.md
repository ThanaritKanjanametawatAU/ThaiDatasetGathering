# Task S02_T01: Implement PESQ Metric Calculator

## Task Overview
Implement a Perceptual Evaluation of Speech Quality (PESQ) metric calculator that provides ITU-T P.862 compliant speech quality measurements for the audio enhancement pipeline.

## Technical Requirements

### Core Implementation
- **PESQ Calculator Module** (`processors/audio_enhancement/metrics/pesq_calculator.py`)
  - ITU-T P.862 compliant implementation
  - Support for both wideband (P.862.2) and narrowband modes
  - Batch processing capabilities for efficiency
  - Proper handling of sample rate conversions

### Key Features
1. **Multi-Mode Support**
   - Narrowband mode (8 kHz sampling)
   - Wideband mode (16 kHz sampling)
   - Automatic mode detection based on input

2. **Robust Processing**
   - Handle various audio formats and bit depths
   - Graceful degradation for edge cases
   - Memory-efficient batch processing

3. **Integration Points**
   - Quality verification module
   - Enhancement evaluation system
   - Automated testing framework

## Mathematical Formulation

### PESQ Algorithm Overview
The PESQ algorithm consists of several stages:

1. **Level Alignment**
   ```python
   # Compute active speech level using ITU-T P.56
   def compute_active_level(signal, fs):
       # Frame-based energy calculation
       frame_len = int(0.02 * fs)  # 20ms frames
       hop_len = int(0.01 * fs)    # 10ms hop
       
       # Activity detection threshold
       activity_threshold = compute_activity_threshold(signal)
       
       # Compute RMS for active frames
       active_rms = []
       for i in range(0, len(signal) - frame_len, hop_len):
           frame = signal[i:i+frame_len]
           if np.sqrt(np.mean(frame**2)) > activity_threshold:
               active_rms.append(np.sqrt(np.mean(frame**2)))
       
       return 20 * np.log10(np.mean(active_rms))
   ```

2. **Perceptual Model**
   ```python
   # Bark scale transformation
   def hz_to_bark(freq):
       return 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)
   
   # Critical band filtering
   def apply_critical_band_filtering(spectrum, fs):
       freqs = np.fft.fftfreq(len(spectrum), 1/fs)[:len(spectrum)//2]
       bark_scale = hz_to_bark(freqs)
       
       # Group into critical bands
       critical_bands = []
       for b in range(24):  # 24 critical bands
           mask = (bark_scale >= b) & (bark_scale < b+1)
           critical_bands.append(np.mean(spectrum[mask]))
       
       return np.array(critical_bands)
   ```

3. **Cognitive Model**
   ```python
   # Asymmetric disturbance calculation
   def calculate_disturbance(ref_bark, deg_bark):
       # Auditory masking patterns
       masking_slope = 27  # dB/Bark
       
       disturbance_frame = []
       for i in range(len(ref_bark)):
           # Forward masking
           forward_mask = ref_bark[i] - masking_slope * np.arange(len(ref_bark) - i)
           
           # Calculate local disturbance
           local_dist = np.maximum(0, deg_bark[i:] - forward_mask)
           disturbance_frame.append(np.sum(local_dist))
       
       return np.array(disturbance_frame)
   ```

### GPU Acceleration Strategies

1. **Batch Processing with CuPy**
   ```python
   import cupy as cp
   
   class GPUPESQCalculator:
       def __init__(self, device_id=0):
           cp.cuda.Device(device_id).use()
           
       def batch_calculate_gpu(self, ref_batch, deg_batch):
           # Transfer to GPU
           ref_gpu = cp.asarray(ref_batch)
           deg_gpu = cp.asarray(deg_batch)
           
           # Parallel FFT computation
           ref_fft = cp.fft.fft(ref_gpu, axis=1)
           deg_fft = cp.fft.fft(deg_gpu, axis=1)
           
           # Parallel critical band filtering
           critical_bands_ref = self._gpu_critical_bands(ref_fft)
           critical_bands_deg = self._gpu_critical_bands(deg_fft)
           
           # Compute disturbances in parallel
           disturbances = self._gpu_disturbance(critical_bands_ref, critical_bands_deg)
           
           # Map to MOS scores
           mos_scores = self._disturbance_to_mos_gpu(disturbances)
           
           return cp.asnumpy(mos_scores)
   ```

2. **Memory Optimization**
   ```python
   def memory_efficient_pesq(self, ref_audio, deg_audio, chunk_size=10):
       """Process large batches in chunks to manage GPU memory"""
       n_samples = len(ref_audio)
       scores = []
       
       for i in range(0, n_samples, chunk_size):
           chunk_ref = ref_audio[i:i+chunk_size]
           chunk_deg = deg_audio[i:i+chunk_size]
           
           # Process chunk on GPU
           chunk_scores = self.batch_calculate_gpu(chunk_ref, chunk_deg)
           scores.extend(chunk_scores)
           
           # Clear GPU cache periodically
           if i % (chunk_size * 10) == 0:
               cp.get_default_memory_pool().free_all_blocks()
       
       return scores
   ```

### Calibration and Validation

1. **Reference Implementation Validation**
   ```python
   def validate_against_reference(self):
       """Validate against ITU-T test vectors"""
       test_vectors = load_itu_test_vectors()
       
       for vector in test_vectors:
           our_score = self.calculate(vector['reference'], vector['degraded'])
           ref_score = vector['expected_mos']
           
           # Must be within 0.1 MOS
           assert abs(our_score - ref_score) < 0.1, \
               f"Score mismatch: {our_score} vs {ref_score}"
   ```

2. **Cross-Validation with pypesq**
   ```python
   def cross_validate_implementation(self, test_set):
       """Compare with established pypesq library"""
       from pypesq import pesq as reference_pesq
       
       differences = []
       for ref, deg in test_set:
           our_score = self.calculate(ref, deg)
           ref_score = reference_pesq(16000, ref, deg, 'wb')
           differences.append(abs(our_score - ref_score))
       
       # 95% of differences should be < 0.05 MOS
       assert np.percentile(differences, 95) < 0.05
   ```

### Industry Standard Compliance

1. **ITU-T P.862 Compliance Checklist**
   - ✅ Level alignment per P.56
   - ✅ Time alignment with cross-correlation
   - ✅ Perceptual frequency transformation
   - ✅ Cognitive modeling with asymmetry
   - ✅ Disturbance to MOS mapping

2. **Extended Standards Support**
   - P.862.1: Mapping to subjective test results
   - P.862.2: Wideband extension
   - P.862.3: Application guide for VoIP

### Optimized Implementation Example

```python
class OptimizedPESQCalculator:
    def __init__(self, mode='auto', use_gpu=True):
        self.mode = mode
        self.use_gpu = use_gpu and cp.cuda.is_available()
        
        # Pre-compute filter banks
        self.narrowband_filters = self._init_filters(8000)
        self.wideband_filters = self._init_filters(16000)
        
        # Pre-allocate buffers
        self.fft_buffer = None
        self.bark_buffer = None
        
    def calculate_with_details(self, reference, degraded):
        """Calculate PESQ with intermediate results for debugging"""
        results = {
            'mos': None,
            'level_difference': None,
            'delay': None,
            'disturbance_profile': None
        }
        
        # Level alignment
        ref_level = self.compute_active_level(reference)
        deg_level = self.compute_active_level(degraded)
        results['level_difference'] = ref_level - deg_level
        
        # Time alignment
        delay = self.find_delay(reference, degraded)
        results['delay'] = delay
        degraded_aligned = self.apply_delay(degraded, delay)
        
        # Perceptual transform
        ref_bark = self.perceptual_transform(reference)
        deg_bark = self.perceptual_transform(degraded_aligned)
        
        # Cognitive model
        disturbance = self.cognitive_model(ref_bark, deg_bark)
        results['disturbance_profile'] = disturbance
        
        # Map to MOS
        results['mos'] = self.disturbance_to_mos(disturbance)
        
        return results
```

## TDD Requirements

### Test Structure
```
tests/test_pesq_calculator.py
- test_narrowband_calculation()
- test_wideband_calculation()
- test_batch_processing()
- test_sample_rate_handling()
- test_edge_cases()
- test_invalid_inputs()
```

### Test Data Requirements
- Reference audio samples at various quality levels
- Pre-calculated PESQ scores for validation
- Edge case audio (silence, clipping, noise)

## Implementation Approach

### Phase 1: Core Calculator
```python
class PESQCalculator:
    def __init__(self, mode='auto'):
        self.mode = mode
        self.supported_rates = [8000, 16000]
    
    def calculate(self, reference, degraded):
        # PESQ calculation logic
        pass
    
    def batch_calculate(self, reference_batch, degraded_batch):
        # Efficient batch processing
        pass
```

### Phase 2: Integration
- Connect to enhancement pipeline
- Add to quality verification system
- Enable real-time monitoring

### Phase 3: Optimization
- GPU acceleration where possible
- Caching for repeated calculations
- Parallel processing for large batches

## Acceptance Criteria
1. ✅ PESQ scores match reference implementation within 0.1 MOS
2. ✅ Support for both narrowband and wideband modes
3. ✅ Batch processing faster than sequential by 3x
4. ✅ Comprehensive error handling and logging
5. ✅ Integration with existing quality metrics

## Example Usage
```python
from processors.audio_enhancement.metrics import PESQCalculator

calculator = PESQCalculator(mode='wideband')
score = calculator.calculate(reference_audio, enhanced_audio)
print(f"PESQ Score: {score:.2f} MOS")

# Batch processing
scores = calculator.batch_calculate(ref_batch, enhanced_batch)
print(f"Average PESQ: {np.mean(scores):.2f} MOS")
```

## Dependencies
- `pypesq` or custom PESQ implementation
- NumPy for numerical operations
- SciPy for signal processing
- Audio processing utilities

## Performance Targets
- Single calculation: < 100ms for 10s audio
- Batch of 100: < 5 seconds
- Memory usage: < 500MB for batch of 100

## Notes
- PESQ is designed for speech quality assessment
- Not suitable for music or non-speech audio
- Consider PESQ-WB for wideband speech
- Cache results for repeated evaluations