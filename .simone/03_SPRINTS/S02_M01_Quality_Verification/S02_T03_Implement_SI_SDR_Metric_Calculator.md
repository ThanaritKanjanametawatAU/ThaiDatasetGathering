# Task S02_T03: Implement SI-SDR Metric Calculator

## Task Overview
Implement a Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric calculator for evaluating source separation quality and enhancement effectiveness in the audio pipeline.

## Technical Requirements

### Core Implementation
- **SI-SDR Calculator Module** (`processors/audio_enhancement/metrics/si_sdr_calculator.py`)
  - Scale-invariant SDR computation
  - Support for multi-source evaluation
  - Permutation-invariant SI-SDR (PIT-SDR)
  - Segmental SI-SDR for temporal analysis

### Key Features
1. **Comprehensive Metrics**
   - Standard SI-SDR calculation
   - SI-SDR improvement (SI-SDRi)
   - Permutation handling for multi-speaker
   - Frame-wise SI-SDR tracking

2. **Advanced Analysis**
   - Source-to-noise ratio decomposition
   - Artifact-to-signal ratio measurement
   - Time-frequency domain SI-SDR

3. **Optimization Features**
   - Vectorized operations
   - GPU acceleration support
   - Efficient permutation search

## TDD Requirements

### Test Structure
```
tests/test_si_sdr_calculator.py
- test_basic_si_sdr_calculation()
- test_scale_invariance_property()
- test_multi_source_separation()
- test_permutation_invariant_sdr()
- test_segmental_analysis()
- test_improvement_calculation()
```

### Test Data Requirements
- Single and multi-source audio
- Various mixing scenarios
- Known SI-SDR ground truth values
- Edge cases (silence, identical signals)

## Implementation Approach

### Phase 1: Core Calculator
```python
class SISDRCalculator:
    def __init__(self, eps=1e-8):
        self.eps = eps
    
    def calculate(self, reference, estimate):
        # Scale-invariant SDR calculation
        scaling = np.dot(estimate, reference) / (np.dot(reference, reference) + self.eps)
        target = scaling * reference
        noise = estimate - target
        
        si_sdr = 10 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + self.eps))
        return si_sdr
    
    def calculate_improvement(self, mixture, estimate, reference):
        # SI-SDR improvement calculation
        pass
```

### Phase 2: Multi-Source Support
- Permutation-invariant calculation
- Optimal assignment algorithms
- Batch processing capabilities

### Phase 3: Advanced Features
- Segmental SI-SDR analysis
- Frequency-domain SI-SDR
- Real-time monitoring integration

## Acceptance Criteria
1. ✅ SI-SDR calculation matches reference within 0.01 dB
2. ✅ Scale invariance property verified
3. ✅ Support for multi-source evaluation
4. ✅ Efficient permutation search (< 100ms for 4 sources)
5. ✅ Integration with separation quality assessment

## Example Usage
```python
from processors.audio_enhancement.metrics import SISDRCalculator

calculator = SISDRCalculator()

# Basic SI-SDR
si_sdr = calculator.calculate(reference_audio, separated_audio)
print(f"SI-SDR: {si_sdr:.2f} dB")

# SI-SDR improvement
si_sdri = calculator.calculate_improvement(
    mixture_audio, separated_audio, reference_audio
)
print(f"SI-SDR Improvement: {si_sdri:.2f} dB")

# Multi-source with permutation
results = calculator.calculate_permutation_invariant(
    reference_sources, estimated_sources
)
print(f"Best permutation SI-SDR: {results['si_sdr']:.2f} dB")
```

## Dependencies
- NumPy for numerical operations
- SciPy for optimization (permutation search)
- Optional: CuPy for GPU acceleration
- Audio processing utilities

## Performance Targets
- Single calculation: < 10ms for 10s audio
- Permutation search (4 sources): < 100ms
- Batch of 100: < 1 second
- Memory usage: < 100MB for batch processing

## Notes
- SI-SDR is scale-invariant unlike standard SDR
- Better suited for source separation evaluation
- Negative values indicate poor separation
- Consider using with other metrics for complete assessment

## Mathematical Formulation

### SI-SDR Mathematical Foundation

1. **Scale-Invariant SDR Definition**
   ```python
   def si_sdr(reference, estimate, eps=1e-8):
       """
       Scale-Invariant Signal-to-Distortion Ratio
       
       SI-SDR = 10 * log10(||s_target||² / ||e_noise||²)
       
       where:
       s_target = <s, s_hat> / ||s||² * s
       e_noise = s_hat - s_target
       """
       # Ensure zero mean
       reference = reference - np.mean(reference)
       estimate = estimate - np.mean(estimate)
       
       # Compute scaling factor (projection)
       alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
       
       # Target signal component
       s_target = alpha * reference
       
       # Noise/interference component
       e_noise = estimate - s_target
       
       # Compute SI-SDR
       target_energy = np.dot(s_target, s_target)
       noise_energy = np.dot(e_noise, e_noise)
       
       si_sdr_value = 10 * np.log10(target_energy / (noise_energy + eps))
       
       return si_sdr_value
   ```

2. **SI-SDR Improvement (SI-SDRi)**
   ```python
   def si_sdr_improvement(mixture, estimate, reference, eps=1e-8):
       """
       SI-SDR improvement: improvement over baseline (mixture)
       
       SI-SDRi = SI-SDR(estimate, reference) - SI-SDR(mixture, reference)
       """
       # Baseline SI-SDR (mixture vs reference)
       baseline_si_sdr = si_sdr(reference, mixture, eps)
       
       # Enhanced SI-SDR (estimate vs reference)
       enhanced_si_sdr = si_sdr(reference, estimate, eps)
       
       # Improvement
       improvement = enhanced_si_sdr - baseline_si_sdr
       
       return improvement, enhanced_si_sdr, baseline_si_sdr
   ```

3. **Permutation-Invariant SI-SDR (PIT-SDR)**
   ```python
   from itertools import permutations
   import numpy as np
   
   def pit_si_sdr(references, estimates, eps=1e-8):
       """
       Permutation-Invariant SI-SDR for multi-source scenarios
       
       Finds optimal assignment between estimates and references
       """
       n_sources = len(references)
       assert len(estimates) == n_sources, "Number of sources must match"
       
       # Try all permutations
       all_perms = list(permutations(range(n_sources)))
       si_sdrs = []
       
       for perm in all_perms:
           perm_si_sdr = 0
           for ref_idx, est_idx in enumerate(perm):
               perm_si_sdr += si_sdr(references[ref_idx], estimates[est_idx], eps)
           si_sdrs.append(perm_si_sdr / n_sources)
       
       # Find best permutation
       best_idx = np.argmax(si_sdrs)
       best_perm = all_perms[best_idx]
       best_si_sdr = si_sdrs[best_idx]
       
       # Return individual SI-SDRs with best assignment
       individual_si_sdrs = []
       for ref_idx, est_idx in enumerate(best_perm):
           individual_si_sdrs.append(
               si_sdr(references[ref_idx], estimates[est_idx], eps)
           )
       
       return {
           'mean_si_sdr': best_si_sdr,
           'best_permutation': best_perm,
           'individual_si_sdrs': individual_si_sdrs,
           'all_permutation_scores': si_sdrs
       }
   ```

### GPU Acceleration Strategies

1. **Vectorized Batch Processing**
   ```python
   import cupy as cp
   
   class GPUSISDRCalculator:
       def __init__(self, device_id=0):
           self.device = cp.cuda.Device(device_id)
           
       def batch_si_sdr_gpu(self, references, estimates, eps=1e-8):
           """Compute SI-SDR for batches on GPU"""
           with self.device:
               # Transfer to GPU
               ref_gpu = cp.asarray(references)
               est_gpu = cp.asarray(estimates)
               
               # Remove mean (vectorized)
               ref_gpu = ref_gpu - cp.mean(ref_gpu, axis=1, keepdims=True)
               est_gpu = est_gpu - cp.mean(est_gpu, axis=1, keepdims=True)
               
               # Compute scaling factors for all pairs
               num = cp.sum(est_gpu * ref_gpu, axis=1)
               den = cp.sum(ref_gpu * ref_gpu, axis=1) + eps
               alpha = num / den
               
               # Target signals
               s_target = alpha[:, None] * ref_gpu
               
               # Noise signals
               e_noise = est_gpu - s_target
               
               # SI-SDR computation
               target_energy = cp.sum(s_target * s_target, axis=1)
               noise_energy = cp.sum(e_noise * e_noise, axis=1)
               
               si_sdr_values = 10 * cp.log10(target_energy / (noise_energy + eps))
               
               return cp.asnumpy(si_sdr_values)
   ```

2. **Efficient Permutation Search**
   ```python
   def gpu_pit_si_sdr(references, estimates, max_sources=4):
       """GPU-accelerated permutation-invariant SI-SDR"""
       n_sources = len(references)
       
       if n_sources > max_sources:
           # Use Hungarian algorithm for large number of sources
           return hungarian_pit_si_sdr(references, estimates)
       
       # Pre-compute all pairwise SI-SDRs on GPU
       with cp.cuda.Device():
           ref_gpu = cp.asarray(references)
           est_gpu = cp.asarray(estimates)
           
           # Compute SI-SDR matrix
           si_sdr_matrix = cp.zeros((n_sources, n_sources))
           
           for i in range(n_sources):
               for j in range(n_sources):
                   si_sdr_matrix[i, j] = gpu_si_sdr_single(
                       ref_gpu[i], est_gpu[j]
                   )
           
           # Transfer back to CPU for permutation search
           si_sdr_matrix_cpu = cp.asnumpy(si_sdr_matrix)
       
       # Find optimal assignment
       from scipy.optimize import linear_sum_assignment
       row_ind, col_ind = linear_sum_assignment(-si_sdr_matrix_cpu)
       
       best_si_sdr = si_sdr_matrix_cpu[row_ind, col_ind].mean()
       
       return {
           'mean_si_sdr': best_si_sdr,
           'assignment': list(zip(row_ind, col_ind)),
           'si_sdr_matrix': si_sdr_matrix_cpu
       }
   ```

### Batch Processing Optimizations

1. **Memory-Efficient Chunking**
   ```python
   class MemoryEfficientSISDR:
       def __init__(self, chunk_size=1000):
           self.chunk_size = chunk_size
           self.gpu_calculator = GPUSISDRCalculator()
           
       def process_large_dataset(self, ref_list, est_list):
           """Process large datasets in chunks"""
           n_samples = len(ref_list)
           all_scores = []
           
           for i in range(0, n_samples, self.chunk_size):
               # Get chunk
               ref_chunk = ref_list[i:i + self.chunk_size]
               est_chunk = est_list[i:i + self.chunk_size]
               
               # Process on GPU
               chunk_scores = self.gpu_calculator.batch_si_sdr_gpu(
                   ref_chunk, est_chunk
               )
               
               all_scores.extend(chunk_scores)
               
               # Clear GPU memory periodically
               if i % (self.chunk_size * 10) == 0:
                   cp.get_default_memory_pool().free_all_blocks()
           
           return np.array(all_scores)
   ```

2. **Parallel Multi-GPU Processing**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   def multi_gpu_si_sdr(references, estimates, n_gpus=4):
       """Distribute SI-SDR computation across multiple GPUs"""
       n_samples = len(references)
       chunk_size = n_samples // n_gpus
       
       def process_on_gpu(gpu_id, start_idx, end_idx):
           calculator = GPUSISDRCalculator(device_id=gpu_id)
           return calculator.batch_si_sdr_gpu(
               references[start_idx:end_idx],
               estimates[start_idx:end_idx]
           )
       
       with ThreadPoolExecutor(max_workers=n_gpus) as executor:
           futures = []
           for gpu_id in range(n_gpus):
               start = gpu_id * chunk_size
               end = start + chunk_size if gpu_id < n_gpus - 1 else n_samples
               
               future = executor.submit(process_on_gpu, gpu_id, start, end)
               futures.append(future)
           
           # Collect results
           all_results = []
           for future in futures:
               all_results.extend(future.result())
       
       return np.array(all_results)
   ```

### Calibration and Validation Procedures

1. **Numerical Accuracy Validation**
   ```python
   def validate_numerical_accuracy():
       """Validate SI-SDR implementation accuracy"""
       # Test cases with known results
       test_cases = [
           {
               'reference': np.array([1, 0, -1, 0]),
               'estimate': np.array([1, 0, -1, 0]),
               'expected': float('inf')  # Perfect reconstruction
           },
           {
               'reference': np.array([1, 0, -1, 0]),
               'estimate': np.array([0.5, 0, -0.5, 0]),
               'expected': float('inf')  # Scaled version
           },
           {
               'reference': np.array([1, 0, -1, 0]),
               'estimate': np.array([0, 1, 0, -1]),
               'expected': -float('inf')  # Orthogonal
           }
       ]
       
       for case in test_cases:
           result = si_sdr(case['reference'], case['estimate'])
           if np.isinf(case['expected']):
               assert np.isinf(result) and np.sign(result) == np.sign(case['expected'])
           else:
               assert np.abs(result - case['expected']) < 1e-6
   ```

2. **Scale Invariance Verification**
   ```python
   def verify_scale_invariance():
       """Verify scale invariance property"""
       reference = np.random.randn(16000)
       estimate = np.random.randn(16000)
       
       # Original SI-SDR
       original_si_sdr = si_sdr(reference, estimate)
       
       # Test different scales
       scales = [0.1, 0.5, 2.0, 10.0, 100.0]
       
       for scale in scales:
           scaled_estimate = scale * estimate
           scaled_si_sdr = si_sdr(reference, scaled_estimate)
           
           # Should be identical (within numerical precision)
           assert np.abs(scaled_si_sdr - original_si_sdr) < 1e-6, \
               f"Scale invariance violated for scale={scale}"
   ```

### Industry Standard Compliance

1. **BSS Eval Compatibility**
   ```python
   def ensure_bss_eval_compatibility():
       """Ensure compatibility with BSS Eval toolbox"""
       from mir_eval.separation import bss_eval_sources
       
       # Test signals
       reference = np.random.randn(2, 16000)
       estimate = reference + 0.1 * np.random.randn(2, 16000)
       
       # Our implementation
       our_result = pit_si_sdr(reference, estimate)
       
       # BSS Eval reference
       sdr, sir, sar, perm = bss_eval_sources(reference, estimate)
       
       # Compare results
       assert np.abs(our_result['mean_si_sdr'] - np.mean(sdr)) < 0.1, \
           "Incompatible with BSS Eval"
   ```

2. **Performance Benchmarking**
   ```python
   def benchmark_si_sdr_performance():
       """Benchmark against performance requirements"""
       import time
       
       # Test configurations
       configs = [
           ('Single 10s', 1, 160000),
           ('Batch 100', 100, 160000),
           ('4-source PIT', 4, 160000)
       ]
       
       calculator = OptimizedSISDRCalculator()
       
       for name, n_sources, length in configs:
           if n_sources == 1:
               ref = np.random.randn(length)
               est = np.random.randn(length)
               
               start = time.time()
               result = calculator.calculate(ref, est)
               elapsed = (time.time() - start) * 1000
               
               print(f"{name}: {elapsed:.1f}ms")
               assert elapsed < 10  # < 10ms target
               
           elif n_sources == 100:
               refs = np.random.randn(n_sources, length)
               ests = np.random.randn(n_sources, length)
               
               start = time.time()
               results = calculator.batch_calculate(refs, ests)
               elapsed = time.time() - start
               
               print(f"{name}: {elapsed:.2f}s")
               assert elapsed < 1  # < 1s target
               
           else:  # PIT-SDR
               refs = [np.random.randn(length) for _ in range(n_sources)]
               ests = [np.random.randn(length) for _ in range(n_sources)]
               
               start = time.time()
               result = calculator.calculate_pit(refs, ests)
               elapsed = (time.time() - start) * 1000
               
               print(f"{name}: {elapsed:.1f}ms")
               assert elapsed < 100  # < 100ms target
   ```

### Real-Time Integration Example

```python
class RealTimeSISDRTracker:
    """Track SI-SDR in real-time for streaming applications"""
    
    def __init__(self, reference, window_size=3.0, hop_size=0.5):
        self.reference = reference
        self.window_samples = int(window_size * 16000)
        self.hop_samples = int(hop_size * 16000)
        
        # Pre-compute reference windows
        self.ref_windows = self._extract_windows(reference)
        
        # Circular buffer for streaming
        self.buffer = np.zeros(len(reference))
        self.write_idx = 0
        
        # SI-SDR history
        self.si_sdr_history = []
        self.timestamps = []
        
    def update(self, audio_chunk):
        """Update with new audio and compute windowed SI-SDR"""
        # Update buffer
        chunk_len = len(audio_chunk)
        if self.write_idx + chunk_len <= len(self.buffer):
            self.buffer[self.write_idx:self.write_idx + chunk_len] = audio_chunk
            self.write_idx += chunk_len
        
        # Compute SI-SDR for completed windows
        while self.write_idx >= self.window_samples:
            window_start = len(self.si_sdr_history) * self.hop_samples
            window_end = window_start + self.window_samples
            
            if window_end <= self.write_idx:
                # Extract window
                est_window = self.buffer[window_start:window_end]
                ref_window = self.reference[window_start:window_end]
                
                # Compute SI-SDR
                si_sdr_value = si_sdr(ref_window, est_window)
                
                # Store results
                self.si_sdr_history.append(si_sdr_value)
                self.timestamps.append(window_start / 16000)
            else:
                break
        
        return self.get_current_stats()
    
    def get_current_stats(self):
        """Get current SI-SDR statistics"""
        if not self.si_sdr_history:
            return None
            
        return {
            'current_si_sdr': self.si_sdr_history[-1],
            'mean_si_sdr': np.mean(self.si_sdr_history),
            'std_si_sdr': np.std(self.si_sdr_history),
            'min_si_sdr': np.min(self.si_sdr_history),
            'max_si_sdr': np.max(self.si_sdr_history),
            'timestamp': self.timestamps[-1]
        }
```