# Audio Quality Enhancement to 35dB SNR Implementation Plan

## 1. Executive Summary

### Brief Description
Enhance the audio preprocessing pipeline to achieve a minimum 35 dB SNR for voice cloning TTS training data, with at least 90% of samples meeting this threshold while preserving naturalness and avoiding over-denoising artifacts.

### Key Business Objectives
- Improve voice cloning quality by ensuring clean training data (35 dB SNR minimum)
- Maintain audio naturalness to prevent synthetic-sounding voice clones
- Enable quality-based filtering with SNR metadata
- Process efficiently within hardware constraints (32GB VRAM, <1.8s per sample)

### Success Criteria
- 90% of audio samples achieve ≥35 dB SNR after enhancement
- Processing speed: <3 minutes for 100 samples (1.8s per sample)
- Minimal perceptual artifacts (validated by PESQ, STOI metrics)
- Successful integration with existing pipeline features

### Estimated Timeline
- Phase 1 (Analysis): 2 days - Analyze current audio quality on 1000 samples
- Phase 2 (Implementation): 3-4 days - Enhance preprocessing pipeline
- Phase 3 (Validation): 2 days - Test and generate comprehensive report
- Total: 7-8 days

## 2. Business Requirements

### Problem Statement
Current audio preprocessing produces samples with background noise levels unsuitable for high-quality voice cloning. At 25 dB SNR, there is still noticeable background noise that affects TTS model training quality.

### User Personas and Use Cases
- **Primary User**: ML Engineer training voice cloning models
- **Use Case 1**: Preprocess Thai audio datasets for TTS training
- **Use Case 2**: Filter dataset by quality metrics post-processing
- **Use Case 3**: Monitor enhancement effectiveness across datasets

### Functional Requirements
1. **SNR Enhancement**: Achieve 35 dB SNR for 90% of samples
2. **Quality Preservation**: Maintain natural speech characteristics
3. **Metadata Addition**: Add SNR column to dataset schema
4. **Batch Processing**: Handle multiple datasets uniformly
5. **Failure Handling**: Include sub-35dB samples with SNR metadata

### Non-Functional Requirements
- **Performance**: Process 100 samples in <3 minutes
- **GPU Utilization**: Efficiently use 32GB VRAM
- **Compatibility**: Integrate with existing enhancement pipeline
- **Monitoring**: Generate comprehensive quality reports

### Acceptance Criteria
1. 1000-sample analysis shows current SNR distribution
2. Enhanced samples reach 35 dB SNR for 90%+ cases
3. PESQ scores remain >3.5 (good perceptual quality)
4. STOI scores remain >0.85 (good intelligibility)
5. Processing time meets performance requirements

## 3. Technical Architecture

### System Architecture
```
Input Audio → SNR Measurement → Multi-Stage Enhancement → Quality Validation → Output
                     ↓                    ↓                      ↓
                  Metrics DB       Enhancement Engine      Quality Metrics
                                   - Adaptive Denoising
                                   - Spectral Refinement
                                   - Harmonic Preservation
```

### Technology Stack
- **Core Framework**: PyTorch with CUDA support
- **Enhancement Libraries**:
  - Facebook Denoiser (already integrated)
  - Enhanced spectral gating algorithms
  - Wiener filtering for gentle enhancement
  - Perceptual weighting filters
- **Metrics Libraries**:
  - pypesq for PESQ calculation
  - pystoi for STOI calculation
  - Custom SNR measurement
- **GPU Optimization**: torch.cuda for 32GB VRAM utilization

### Enhancement Strategy
1. **Multi-Stage Approach** (based on research):
   - Stage 1: Gentle spectral subtraction (preserve naturalness)
   - Stage 2: Adaptive Wiener filtering
   - Stage 3: Harmonic enhancement
   - Stage 4: Perceptual post-processing

2. **Adaptive Enhancement**:
   - Measure initial SNR
   - Apply progressively stronger enhancement only if needed
   - Stop when 35 dB reached or naturalness threshold hit

### Integration Points
- Integrate after audio standardization in existing pipeline
- Before speaker identification and STT processing
- Parallel processing with existing enhancement features
- Add SNR measurement to metadata collection

## 4. Data Models & Schemas

### Updated Dataset Schema
```python
{
    "ID": str,                    # Sequential identifier
    "speaker_id": str,            # Speaker identifier
    "Language": str,              # "th"
    "audio": {                    # HuggingFace audio format
        "array": np.ndarray,
        "sampling_rate": int,
        "path": str
    },
    "transcript": str,            # Transcript if available
    "length": float,              # Duration in seconds
    "dataset_name": str,          # Source dataset
    "confidence_score": float,    # Transcript confidence
    "snr_db": float,             # NEW: Signal-to-Noise Ratio in dB
    "audio_quality_metrics": {    # NEW: Additional quality metrics
        "pesq": float,            # Perceptual quality (1.0-4.5)
        "stoi": float,            # Intelligibility (0-1)
        "mos_estimate": float     # Estimated Mean Opinion Score
    }
}
```

### Quality Thresholds Configuration
```python
QUALITY_THRESHOLDS = {
    "target_snr_db": 35.0,
    "min_acceptable_snr_db": 30.0,
    "target_success_rate": 0.90,
    "max_enhancement_passes": 3,
    "naturalness_weights": {
        "preserve_harmonics": 0.8,
        "suppress_noise": 0.2
    },
    "perceptual_limits": {
        "min_pesq": 3.5,
        "min_stoi": 0.85,
        "max_spectral_distortion": 0.15
    }
}
```

## 5. Detailed Implementation Plan

### Development Phases

#### Phase 1: Current State Analysis (2 days)

##### Day 1: Sample Selection and Initial Analysis
**Morning (4 hours)**:
1. **Create Sample Extraction Script** (`scripts/analysis/extract_quality_samples.py`):
   ```python
   def extract_samples():
       # Step 1: Calculate proportional sampling
       # - GigaSpeech2: 40% (400 samples)
       # - ProcessedVoiceTH: 30% (300 samples)  
       # - Mozilla CV: 30% (300 samples)
       
       # Step 2: Implement stratified sampling
       # - Short clips (<2s): 20%
       # - Medium clips (2-5s): 50%
       # - Long clips (>5s): 30%
       
       # Step 3: Save sample metadata
       # - Original file path
       # - Duration, sample rate
       # - Dataset source
   ```

2. **Implement SNR Measurement Tool** (`utils/snr_measurement.py`):
   ```python
   class SNRMeasurement:
       def measure_snr(self, audio, sample_rate):
           # Step 1: Voice Activity Detection (VAD)
           # - Use pyannote.audio VAD
           # - Identify speech vs silence segments
           
           # Step 2: Noise floor estimation
           # - Calculate RMS of silence segments
           # - Apply percentile filtering (5th percentile)
           
           # Step 3: Signal power calculation
           # - Calculate RMS of speech segments
           # - Apply peak normalization
           
           # Step 4: SNR calculation
           # - SNR_dB = 20 * log10(RMS_signal / RMS_noise)
           # - Handle edge cases (all silence, all speech)
   ```

**Afternoon (4 hours)**:
3. **Run Initial Analysis**:
   ```python
   # Create analysis pipeline
   for sample in samples:
       # Load audio
       # Measure SNR
       # Calculate PESQ (using reference = enhanced with mild settings)
       # Calculate STOI
       # Measure frequency spectrum
       # Save results to CSV
   ```

4. **Generate Baseline Report**:
   - SNR distribution histogram
   - Percentile analysis (10th, 25th, 50th, 75th, 90th)
   - Correlation with duration, dataset source
   - Noise frequency analysis

##### Day 2: Noise Profiling and Characterization
**Morning (4 hours)**:
1. **Implement Noise Profiler** (`utils/noise_profiler.py`):
   ```python
   class NoiseProfiler:
       def analyze_noise(self, audio, sample_rate):
           # Step 1: Extract noise segments
           # - Use VAD to find non-speech regions
           # - Ensure minimum 500ms of noise
           
           # Step 2: Frequency analysis
           # - FFT with 2048 samples, 50% overlap
           # - Calculate power spectral density
           # - Identify peak frequencies
           
           # Step 3: Temporal analysis
           # - Check for periodic noise (fans, AC)
           # - Detect impulse noise (clicks, pops)
           # - Measure noise stationarity
           
           # Step 4: Classify noise type
           # - White noise (flat spectrum)
           # - Pink noise (1/f spectrum)
           # - Narrow-band (specific frequencies)
           # - Non-stationary (varying)
   ```

2. **Create Noise Database**:
   ```python
   # Structure: noise_profiles.json
   {
       "stationary": {
           "white_noise": {"samples": [...], "characteristics": {...}},
           "pink_noise": {"samples": [...], "characteristics": {...}},
           "ac_hum": {"samples": [...], "characteristics": {...}}
       },
       "non_stationary": {
           "traffic": {"samples": [...], "characteristics": {...}},
           "crowd": {"samples": [...], "characteristics": {...}}
       }
   }
   ```

**Afternoon (4 hours)**:
3. **Identify Enhancement Challenges**:
   - List top 10 most common noise types
   - Identify samples with SNR < 20 dB
   - Find samples with non-stationary noise
   - Document edge cases (very short, clipping, etc.)

4. **Create Test Set Categories**:
   ```
   test_sets/
   ├── easy/          # SNR 25-30 dB, stationary noise
   ├── moderate/      # SNR 20-25 dB, mixed noise
   ├── challenging/   # SNR 15-20 dB, non-stationary
   └── edge_cases/    # Special cases requiring careful handling
   ```

#### Phase 2: Enhancement Implementation (3-4 days)

##### Day 3: Core Enhancement Modules
**Morning (4 hours)**:
1. **Implement Adaptive Spectral Subtraction** (`processors/audio_enhancement/adaptive_spectral.py`):
   ```python
   class AdaptiveSpectralSubtraction:
       def __init__(self):
           self.frame_size = 2048
           self.hop_size = 512
           self.oversubtraction_factor = 1.5  # Start gentle
           
       def process(self, audio, sample_rate, target_snr=35):
           # Step 1: STFT analysis
           # - Window with Hann window
           # - 75% overlap for smoothness
           
           # Step 2: Noise estimation
           # - Use first 100ms or VAD-based
           # - Update noise profile adaptively
           
           # Step 3: Spectral subtraction
           # - Wiener gain function
           # - Musical noise reduction
           # - Preserve speech harmonics
           
           # Step 4: Reconstruction
           # - Overlap-add synthesis
           # - Phase preservation
   ```

2. **Implement Wiener Filter** (`processors/audio_enhancement/wiener_filter.py`):
   ```python
   class AdaptiveWienerFilter:
       def __init__(self):
           self.alpha = 0.98  # Smoothing factor
           self.beta = 0.8    # Overestimation factor
           
       def estimate_clean_speech(self, noisy_spectrum, noise_spectrum):
           # Step 1: A priori SNR estimation
           # - Decision-directed approach
           # - Smooth across time and frequency
           
           # Step 2: Wiener gain calculation
           # - G = SNR_priori / (1 + SNR_priori)
           # - Apply lower bound for naturalness
           
           # Step 3: Apply gain
           # - Preserve phase information
           # - Smooth transitions
   ```

**Afternoon (4 hours)**:
3. **Implement Harmonic Enhancement** (`processors/audio_enhancement/harmonic_enhancer.py`):
   ```python
   class HarmonicEnhancer:
       def enhance(self, audio, sample_rate):
           # Step 1: Pitch detection
           # - Use CREPE or YIN algorithm
           # - Track pitch contour
           
           # Step 2: Harmonic extraction
           # - Comb filter at pitch harmonics
           # - Preserve formant structure
           
           # Step 3: Harmonic boosting
           # - Amplify harmonic components
           # - Suppress inter-harmonic noise
           
           # Step 4: Blend with original
           # - Adaptive mixing based on voicing
   ```

4. **Implement Perceptual Post-Processing** (`processors/audio_enhancement/perceptual_post.py`):
   ```python
   class PerceptualPostProcessor:
       def process(self, audio, sample_rate):
           # Step 1: Pre-emphasis
           # - Boost high frequencies slightly
           # - Compensate for spectral tilt
           
           # Step 2: Dynamic range adjustment
           # - Gentle compression
           # - Preserve natural dynamics
           
           # Step 3: Comfort noise injection
           # - Add minimal pink noise
           # - Prevent "dead air" feeling
           
           # Step 4: Smoothing filter
           # - Remove any remaining artifacts
           # - Ensure smooth spectral envelope
   ```

##### Day 4: Integration and Quality Control
**Morning (4 hours)**:
1. **Create Enhancement Orchestrator** (`processors/audio_enhancement/enhancement_orchestrator.py`):
   ```python
   class EnhancementOrchestrator:
       def __init__(self, target_snr=35, max_iterations=3):
           self.target_snr = target_snr
           self.stages = [
               AdaptiveSpectralSubtraction(),
               AdaptiveWienerFilter(),
               HarmonicEnhancer(),
               PerceptualPostProcessor()
           ]
           
       def enhance(self, audio, sample_rate):
           # Step 1: Initial assessment
           current_snr = measure_snr(audio, sample_rate)
           if current_snr >= self.target_snr:
               return audio, {"snr": current_snr, "enhanced": False}
           
           # Step 2: Progressive enhancement
           for iteration in range(self.max_iterations):
               # Apply each stage
               for stage in self.stages:
                   audio = stage.process(audio, sample_rate)
                   
                   # Check quality metrics
                   metrics = self.measure_quality(audio, original)
                   if metrics["naturalness"] < 0.85:
                       # Rollback if too aggressive
                       audio = previous_audio
                       break
               
               # Check if target reached
               current_snr = measure_snr(audio, sample_rate)
               if current_snr >= self.target_snr:
                   break
           
           return audio, metrics
   ```

2. **Implement Quality Monitoring** (`utils/quality_monitor.py`):
   ```python
   class QualityMonitor:
       def __init__(self):
           self.thresholds = {
               "spectral_distortion": 0.15,
               "phase_coherence": 0.9,
               "harmonic_preservation": 0.85
           }
           
       def check_naturalness(self, original, enhanced):
           # Step 1: Spectral distortion
           # - Compare mel-spectrograms
           # - Calculate MSE in log domain
           
           # Step 2: Phase coherence
           # - Check phase derivative
           # - Detect phase jumps
           
           # Step 3: Harmonic structure
           # - Compare harmonic peaks
           # - Check formant preservation
           
           # Return naturalness score 0-1
   ```

**Afternoon (4 hours)**:
3. **Modify AudioEnhancer Integration** (`processors/audio_enhancement/core.py`):
   ```python
   # Add to existing AudioEnhancer class
   def enhance_to_target_snr(self, audio, sample_rate, target_snr=35):
       # Step 1: Check if 35dB mode enabled
       if not self.config.get('enable_35db_enhancement', False):
           return self.enhance(audio, sample_rate)  # Use existing
       
       # Step 2: Create orchestrator
       orchestrator = EnhancementOrchestrator(target_snr)
       
       # Step 3: Process with monitoring
       enhanced, metrics = orchestrator.enhance(audio, sample_rate)
       
       # Step 4: Update metadata
       metadata = {
           "snr_db": metrics["final_snr"],
           "audio_quality_metrics": {
               "pesq": metrics.get("pesq", None),
               "stoi": metrics.get("stoi", None),
               "mos_estimate": metrics.get("mos", None)
           },
           "enhancement_applied": metrics["enhanced"],
           "naturalness_score": metrics["naturalness"]
       }
       
       return enhanced, metadata
   ```

4. **Create Configuration Interface**:
   ```python
   # config.py additions
   ENHANCEMENT_35DB_CONFIG = {
       "enabled": True,
       "target_snr_db": 35.0,
       "quality_thresholds": {
           "min_naturalness": 0.85,
           "max_spectral_distortion": 0.15,
           "min_harmonic_preservation": 0.85
       },
       "processing": {
           "max_iterations": 3,
           "batch_size": 32,
           "use_gpu": True
       },
       "fallback": {
           "include_failed": True,  # Include <35dB samples
           "min_acceptable_snr": 25.0
       }
   }
   ```

##### Day 5-6: GPU Optimization and Batch Processing
**Morning Day 5 (4 hours)**:
1. **GPU-Optimized Enhancement** (`processors/audio_enhancement/gpu_enhancement.py`):
   ```python
   class GPUEnhancementBatch:
       def __init__(self, batch_size=32, device='cuda'):
           self.batch_size = batch_size
           self.device = device
           
       def process_batch(self, audio_batch):
           # Step 1: Prepare batch tensors
           # - Pad to same length
           # - Convert to torch tensors
           # - Move to GPU
           
           # Step 2: Batch STFT
           # - Use torch.stft for GPU processing
           # - Process all frames simultaneously
           
           # Step 3: Batch enhancement
           # - Vectorized operations
           # - Minimize CPU-GPU transfers
           
           # Step 4: Batch synthesis
           # - torch.istft for reconstruction
           # - Parallel overlap-add
   ```

2. **Memory Management**:
   ```python
   class MemoryEfficientProcessor:
       def __init__(self, max_memory_gb=30):  # Leave 2GB headroom
           self.max_memory = max_memory_gb * 1024**3
           
       def estimate_batch_size(self, audio_length, sample_rate):
           # Calculate memory per sample
           # Adjust batch size dynamically
           # Handle variable length inputs
   ```

**Afternoon Day 5 + Day 6**:
3. **Complete Integration Testing**
4. **Performance Optimization**
5. **Edge Case Handling**

#### Phase 3: Validation & Reporting (2 days)

##### Day 7: Comprehensive Testing
1. **Process 1000 Test Samples**:
   - Run through complete pipeline
   - Collect all metrics
   - Measure processing times
   - Track GPU memory usage

2. **Statistical Analysis**:
   - SNR improvement distribution
   - Success rate calculation
   - Quality metric correlations
   - Performance benchmarks

##### Day 8: Report Generation
1. **Create Detailed Report**:
   - Executive summary
   - Technical metrics
   - Visual comparisons
   - Recommendations

2. **Generate Sample Comparisons**:
   - Before/after spectrograms
   - Waveform comparisons
   - Quality metric tables

### Risk Assessment
1. **Over-denoising Risk**: Mitigated by multi-stage approach and naturalness checks
2. **Performance Risk**: GPU optimization and batch processing
3. **Integration Risk**: Extensive testing with existing features
4. **Quality Risk**: Multiple metrics validation

## 6. Comprehensive Testing Strategy

### Unit Test Suite Structure
```
tests/
├── test_snr_measurement.py
├── test_enhancement_modules.py
├── test_quality_metrics.py
├── test_integration_35db.py
├── test_edge_cases.py
├── test_performance.py
└── fixtures/
    ├── synthetic_signals/
    ├── real_samples/
    └── noise_profiles/
```

### Detailed Unit Tests

#### 1. SNR Measurement Tests (`test_snr_measurement.py`)

```python
import unittest
import numpy as np
from utils.snr_measurement import SNRMeasurement

class TestSNRMeasurement(unittest.TestCase):
    """Comprehensive SNR measurement validation"""
    
    def setUp(self):
        self.snr_calc = SNRMeasurement()
        self.sample_rate = 16000
        
    def test_known_snr_accuracy(self):
        """Test 1.1: Validate SNR calculation with synthetic signals of known SNR"""
        test_cases = [
            {"true_snr": 0, "tolerance": 0.5},   # Equal signal and noise
            {"true_snr": 10, "tolerance": 0.5},  # 10 dB SNR
            {"true_snr": 20, "tolerance": 0.5},  # 20 dB SNR
            {"true_snr": 35, "tolerance": 0.5},  # Target SNR
            {"true_snr": 40, "tolerance": 0.5},  # High SNR
        ]
        
        for case in test_cases:
            # Generate clean signal (sine wave)
            duration = 3.0
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            
            # Generate white noise at specific SNR
            signal_power = np.mean(signal ** 2)
            target_snr_linear = 10 ** (case["true_snr"] / 10)
            noise_power = signal_power / target_snr_linear
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
            
            # Mix signal and noise
            mixed = signal + noise
            
            # Measure SNR
            measured_snr = self.snr_calc.measure_snr(mixed, self.sample_rate)
            
            # Assert within tolerance
            self.assertAlmostEqual(
                measured_snr, case["true_snr"], 
                delta=case["tolerance"],
                msg=f"SNR measurement failed for {case['true_snr']} dB"
            )
    
    def test_vad_integration(self):
        """Test 1.2: Ensure VAD correctly identifies speech vs silence"""
        # Create signal with clear speech and silence segments
        duration = 4.0
        samples = int(duration * self.sample_rate)
        
        # First 1s: silence
        # Next 2s: speech
        # Last 1s: silence
        signal = np.zeros(samples)
        speech_start = int(1.0 * self.sample_rate)
        speech_end = int(3.0 * self.sample_rate)
        
        # Add speech in middle
        t_speech = np.linspace(0, 2.0, speech_end - speech_start)
        signal[speech_start:speech_end] = np.sin(2 * np.pi * 200 * t_speech)
        
        # Add noise throughout
        noise = 0.01 * np.random.randn(samples)
        mixed = signal + noise
        
        # Get VAD segments
        vad_segments = self.snr_calc._get_vad_segments(mixed, self.sample_rate)
        
        # Verify VAD detected speech correctly
        self.assertEqual(len(vad_segments), 1, "Should detect one speech segment")
        self.assertAlmostEqual(vad_segments[0][0], 1.0, delta=0.1)
        self.assertAlmostEqual(vad_segments[0][1], 3.0, delta=0.1)
    
    def test_edge_case_all_silence(self):
        """Test 1.3: Handle audio that is all silence"""
        # Create pure silence with minimal noise
        silence = 0.0001 * np.random.randn(self.sample_rate * 2)
        
        # Should return very low SNR or handle gracefully
        snr = self.snr_calc.measure_snr(silence, self.sample_rate)
        self.assertIsNotNone(snr)
        self.assertLess(snr, 0, "All silence should have negative SNR")
    
    def test_edge_case_all_speech(self):
        """Test 1.4: Handle audio with no silence segments"""
        # Create continuous speech
        t = np.linspace(0, 3.0, 3 * self.sample_rate)
        speech = np.sin(2 * np.pi * 200 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        
        # Should estimate noise floor from quietest parts
        snr = self.snr_calc.measure_snr(speech, self.sample_rate)
        self.assertIsNotNone(snr)
        self.assertGreater(snr, 20, "Clean speech should have high SNR")
    
    def test_short_audio_handling(self):
        """Test 1.5: Handle very short audio clips (<1 second)"""
        short_durations = [0.1, 0.3, 0.5, 0.8]
        
        for duration in short_durations:
            samples = int(duration * self.sample_rate)
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
            audio += 0.1 * np.random.randn(samples)
            
            # Should handle without crashing
            snr = self.snr_calc.measure_snr(audio, self.sample_rate)
            self.assertIsNotNone(snr, f"Failed for {duration}s audio")
    
    def test_different_noise_types(self):
        """Test 1.6: Accurate measurement with different noise types"""
        duration = 3.0
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        signal = np.sin(2 * np.pi * 440 * t)
        
        noise_types = {
            "white": np.random.randn(len(t)),
            "pink": self._generate_pink_noise(len(t)),
            "brown": self._generate_brown_noise(len(t)),
            "periodic": np.sin(2 * np.pi * 60 * t),  # 60 Hz hum
        }
        
        for noise_type, noise in noise_types.items():
            # Normalize noise for 20 dB SNR
            noise = noise / np.std(noise) * 0.1
            mixed = signal + noise
            
            snr = self.snr_calc.measure_snr(mixed, self.sample_rate)
            self.assertAlmostEqual(snr, 20, delta=2, 
                                 msg=f"Failed for {noise_type} noise")

#### 2. Enhancement Module Tests (`test_enhancement_modules.py`)

class TestEnhancementModules(unittest.TestCase):
    """Test individual enhancement stages"""
    
    def test_spectral_subtraction_gentle(self):
        """Test 2.1: Spectral subtraction preserves speech quality"""
        # Create test signal with known characteristics
        signal = self._create_speech_like_signal()
        noise = 0.1 * np.random.randn(len(signal))
        noisy = signal + noise
        
        # Apply spectral subtraction
        enhancer = AdaptiveSpectralSubtraction()
        enhanced = enhancer.process(noisy, 16000)
        
        # Verify noise reduction
        residual_noise = enhanced - signal
        original_noise_power = np.mean(noise ** 2)
        residual_noise_power = np.mean(residual_noise ** 2)
        
        self.assertLess(residual_noise_power, original_noise_power * 0.5,
                       "Should reduce noise by at least 50%")
        
        # Verify speech preservation
        speech_distortion = np.mean((enhanced - signal) ** 2) / np.mean(signal ** 2)
        self.assertLess(speech_distortion, 0.1, "Speech distortion should be <10%")
    
    def test_wiener_filter_adaptation(self):
        """Test 2.2: Wiener filter adapts to changing noise"""
        # Create signal with varying noise levels
        signal = self._create_speech_like_signal()
        
        # First half: low noise, second half: high noise
        noise = np.zeros_like(signal)
        half = len(signal) // 2
        noise[:half] = 0.05 * np.random.randn(half)
        noise[half:] = 0.2 * np.random.randn(len(signal) - half)
        
        noisy = signal + noise
        
        # Apply Wiener filter
        wiener = AdaptiveWienerFilter()
        enhanced = wiener.process(noisy, 16000)
        
        # Check adaptation: should preserve more in low noise region
        distortion_low = np.mean((enhanced[:half] - signal[:half]) ** 2)
        distortion_high = np.mean((enhanced[half:] - signal[half:]) ** 2)
        
        # Low noise region should have less distortion
        self.assertLess(distortion_low, distortion_high * 0.7)
    
    def test_harmonic_enhancement_pitch_tracking(self):
        """Test 2.3: Harmonic enhancer correctly tracks pitch"""
        # Create signal with varying pitch
        duration = 2.0
        sr = 16000
        t = np.linspace(0, duration, int(duration * sr))
        
        # Pitch glide from 100 Hz to 200 Hz
        pitch_curve = 100 + 50 * t / duration
        phase = 2 * np.pi * np.cumsum(pitch_curve) / sr
        signal = np.sin(phase)
        
        # Add inter-harmonic noise
        noise = 0.1 * np.random.randn(len(signal))
        noisy = signal + noise
        
        # Apply harmonic enhancement
        enhancer = HarmonicEnhancer()
        enhanced = enhancer.enhance(noisy, sr)
        
        # Verify harmonic structure preserved
        # Check power at harmonic frequencies increased
        fft_enhanced = np.abs(np.fft.rfft(enhanced))
        fft_noisy = np.abs(np.fft.rfft(noisy))
        
        # Sample some harmonic frequencies
        test_freqs = [100, 150, 200]  # Hz
        for freq in test_freqs:
            bin_idx = int(freq * len(fft_enhanced) / (sr / 2))
            harmonic_gain = fft_enhanced[bin_idx] / (fft_noisy[bin_idx] + 1e-10)
            self.assertGreater(harmonic_gain, 1.2, 
                             f"Harmonic at {freq}Hz not enhanced")
    
    def test_perceptual_post_processing(self):
        """Test 2.4: Perceptual post-processing improves quality"""
        # Create processed signal with some artifacts
        signal = self._create_speech_like_signal()
        
        # Add some processing artifacts
        # 1. Harsh high frequencies
        signal_harsh = signal + 0.05 * np.random.randn(len(signal)) * self._high_pass(8000, 16000, len(signal))
        
        # 2. Unnaturally silent gaps
        signal_harsh[1000:1100] = 0
        signal_harsh[5000:5100] = 0
        
        # Apply perceptual post-processing
        processor = PerceptualPostProcessor()
        processed = processor.process(signal_harsh, 16000)
        
        # Check comfort noise added to gaps
        gap_power = np.mean(processed[1000:1100] ** 2)
        self.assertGreater(gap_power, 1e-6, "Should add comfort noise to gaps")
        
        # Check high frequency harshness reduced
        fft_original = np.abs(np.fft.rfft(signal_harsh))
        fft_processed = np.abs(np.fft.rfft(processed))
        
        high_freq_idx = len(fft_original) * 3 // 4  # Upper quarter of spectrum
        hf_reduction = np.mean(fft_processed[high_freq_idx:]) / np.mean(fft_original[high_freq_idx:])
        self.assertLess(hf_reduction, 0.8, "Should reduce harsh high frequencies")

#### 3. Quality Metrics Tests (`test_quality_metrics.py`)

class TestQualityMetrics(unittest.TestCase):
    """Test quality assessment metrics"""
    
    def test_naturalness_detection(self):
        """Test 3.1: Naturalness score detects over-processing"""
        monitor = QualityMonitor()
        signal = self._create_speech_like_signal()
        
        # Test cases with varying degrees of processing
        test_cases = [
            {
                "name": "mild_enhancement",
                "process": lambda x: x * 0.95 + 0.05 * self._gentle_denoise(x),
                "expected_naturalness": 0.95,
                "tolerance": 0.05
            },
            {
                "name": "moderate_enhancement", 
                "process": lambda x: self._moderate_denoise(x),
                "expected_naturalness": 0.85,
                "tolerance": 0.05
            },
            {
                "name": "over_processed",
                "process": lambda x: self._aggressive_denoise(x),
                "expected_naturalness": 0.65,
                "tolerance": 0.1
            },
            {
                "name": "destroyed",
                "process": lambda x: np.sign(x) * 0.5,  # Extreme clipping
                "expected_naturalness": 0.2,
                "tolerance": 0.1
            }
        ]
        
        for case in test_cases:
            processed = case["process"](signal)
            naturalness = monitor.check_naturalness(signal, processed)
            
            self.assertAlmostEqual(
                naturalness, 
                case["expected_naturalness"],
                delta=case["tolerance"],
                msg=f"Failed for {case['name']}"
            )
    
    def test_spectral_distortion_measurement(self):
        """Test 3.2: Spectral distortion metric accuracy"""
        monitor = QualityMonitor()
        signal = self._create_speech_like_signal()
        
        # Add known spectral distortions
        distortions = [
            {"type": "lowpass", "cutoff": 4000, "expected_distortion": 0.15},
            {"type": "highpass", "cutoff": 300, "expected_distortion": 0.10},
            {"type": "notch", "freq": 1000, "expected_distortion": 0.08},
            {"type": "amplify_band", "band": [2000, 3000], "expected_distortion": 0.12},
        ]
        
        for dist in distortions:
            distorted = self._apply_spectral_distortion(signal, dist)
            measured = monitor._measure_spectral_distortion(signal, distorted)
            
            self.assertAlmostEqual(
                measured,
                dist["expected_distortion"],
                delta=0.03,
                msg=f"Failed for {dist['type']}"
            )
    
    def test_phase_coherence_detection(self):
        """Test 3.3: Phase coherence detects phase artifacts"""
        monitor = QualityMonitor()
        signal = self._create_speech_like_signal()
        
        # Test phase manipulations
        test_cases = [
            {
                "name": "original",
                "process": lambda x: x,
                "expected_coherence": 1.0
            },
            {
                "name": "minor_phase_shift",
                "process": lambda x: self._add_phase_shift(x, 0.1),
                "expected_coherence": 0.95
            },
            {
                "name": "major_phase_jumps",
                "process": lambda x: self._add_phase_jumps(x, 10),
                "expected_coherence": 0.7
            },
            {
                "name": "random_phase",
                "process": lambda x: self._randomize_phase(x),
                "expected_coherence": 0.3
            }
        ]
        
        for case in test_cases:
            processed = case["process"](signal)
            coherence = monitor._measure_phase_coherence(signal, processed)
            
            self.assertAlmostEqual(
                coherence,
                case["expected_coherence"],
                delta=0.1,
                msg=f"Failed for {case['name']}"
            )

#### 4. Integration Tests (`test_integration_35db.py`)

class TestIntegration35dB(unittest.TestCase):
    """Test complete 35dB enhancement pipeline"""
    
    def test_pipeline_achieves_target_snr(self):
        """Test 4.1: Pipeline achieves 35dB for various input SNRs"""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        input_snrs = [15, 20, 25, 30, 33]
        success_count = 0
        
        for input_snr in input_snrs:
            # Create signal with specific SNR
            signal = self._create_realistic_speech()
            noisy = self._add_noise_at_snr(signal, input_snr)
            
            # Enhance
            enhanced, metadata = enhancer.enhance_to_target_snr(noisy, 16000)
            
            # Check result
            output_snr = metadata["snr_db"]
            
            if output_snr >= 35.0:
                success_count += 1
            elif output_snr >= 33.0:  # Close enough considering measurement error
                success_count += 0.5
            
            # Verify quality preserved
            self.assertGreater(metadata["naturalness_score"], 0.85,
                             f"Naturalness too low for input SNR {input_snr}")
        
        # At least 90% success rate
        success_rate = success_count / len(input_snrs)
        self.assertGreaterEqual(success_rate, 0.9)
    
    def test_quality_preservation_throughout_pipeline(self):
        """Test 4.2: Each stage preserves quality metrics"""
        orchestrator = EnhancementOrchestrator(target_snr=35)
        
        # Create test signal
        signal = self._create_realistic_speech()
        noisy = self._add_noise_at_snr(signal, 25)
        
        # Track metrics through each stage
        metrics_history = []
        
        # Hook into orchestrator to capture intermediate results
        def capture_metrics(audio, stage_name):
            metrics = {
                "stage": stage_name,
                "snr": measure_snr(audio, 16000),
                "pesq": calculate_pesq(signal, audio, 16000),
                "stoi": calculate_stoi(signal, audio, 16000),
                "naturalness": self._quick_naturalness_check(signal, audio)
            }
            metrics_history.append(metrics)
        
        # Process with metric capture
        orchestrator._metric_callback = capture_metrics
        enhanced, final_metrics = orchestrator.enhance(noisy, 16000)
        
        # Verify quality never drops below threshold
        for metrics in metrics_history:
            self.assertGreater(metrics["naturalness"], 0.80,
                             f"Quality dropped at {metrics['stage']}")
            self.assertGreater(metrics["pesq"], 3.0,
                             f"PESQ too low at {metrics['stage']}")
    
    def test_batch_processing_consistency(self):
        """Test 4.3: Batch processing produces consistent results"""
        gpu_processor = GPUEnhancementBatch(batch_size=8)
        
        # Create batch of similar samples
        batch = []
        for i in range(8):
            signal = self._create_realistic_speech()
            # Add similar noise levels
            noisy = self._add_noise_at_snr(signal, 25 + np.random.randn() * 2)
            batch.append(noisy)
        
        # Process batch
        enhanced_batch = gpu_processor.process_batch(batch)
        
        # Check consistency
        snrs = [measure_snr(audio, 16000) for audio in enhanced_batch]
        
        # All should be close to target
        for snr in snrs:
            self.assertGreater(snr, 33.0, "Batch processing inconsistent")
        
        # Standard deviation should be low
        self.assertLess(np.std(snrs), 2.0, "Too much variation in batch")

#### 5. Edge Case Tests (`test_edge_cases.py`)

class TestEdgeCases(unittest.TestCase):
    """Test handling of edge cases and difficult scenarios"""
    
    def test_clipped_audio_handling(self):
        """Test 5.1: Handle clipped/saturated audio"""
        # Create clipped signal
        signal = self._create_realistic_speech() * 3.0  # Amplify to cause clipping
        clipped = np.clip(signal, -1.0, 1.0)
        
        # Add noise
        noisy_clipped = clipped + 0.1 * np.random.randn(len(clipped))
        
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        enhanced, metadata = enhancer.enhance_to_target_snr(noisy_clipped, 16000)
        
        # Should handle without crashing
        self.assertIsNotNone(enhanced)
        # Should detect clipping and handle appropriately
        self.assertIn("clipping_detected", metadata)
    
    def test_very_low_snr_input(self):
        """Test 5.2: Handle extremely noisy input (SNR < 10dB)"""
        signal = self._create_realistic_speech()
        
        # Create very noisy signal (5dB SNR)
        very_noisy = self._add_noise_at_snr(signal, 5)
        
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        enhanced, metadata = enhancer.enhance_to_target_snr(very_noisy, 16000)
        
        # Should improve SNR significantly even if not reaching 35dB
        output_snr = metadata["snr_db"]
        self.assertGreater(output_snr, 20.0, "Should achieve significant improvement")
        
        # Should maintain some quality
        self.assertGreater(metadata["naturalness_score"], 0.7)
    
    def test_silence_and_speech_transitions(self):
        """Test 5.3: Handle audio with frequent silence-speech transitions"""
        # Create audio with alternating speech and silence
        duration = 4.0
        sr = 16000
        audio = np.zeros(int(duration * sr))
        
        # Add 0.5s speech segments with 0.5s gaps
        for i in range(4):
            start = int(i * sr)
            end = int((i + 0.5) * sr)
            t = np.linspace(0, 0.5, end - start)
            audio[start:end] = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 2)
        
        # Add noise
        noisy = audio + 0.05 * np.random.randn(len(audio))
        
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        enhanced, metadata = enhancer.enhance_to_target_snr(noisy, sr)
        
        # Check that transitions are preserved
        # Detect zero crossings as proxy for transitions
        transitions_original = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
        transitions_enhanced = np.sum(np.abs(np.diff(np.sign(enhanced))) > 0)
        
        # Should preserve transition structure
        self.assertAlmostEqual(transitions_enhanced, transitions_original, 
                             delta=transitions_original * 0.2)
    
    def test_multi_speaker_audio(self):
        """Test 5.4: Handle audio with multiple speakers"""
        # Simulate two speakers
        speaker1 = self._create_realistic_speech(pitch=120)
        speaker2 = self._create_realistic_speech(pitch=200)
        
        # Mix speakers with overlap
        mixed = speaker1 * 0.7 + speaker2 * 0.3
        
        # Add noise
        noisy = mixed + 0.1 * np.random.randn(len(mixed))
        
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        enhanced, metadata = enhancer.enhance_to_target_snr(noisy, 16000)
        
        # Should preserve both speakers
        # Check spectral peaks at both fundamental frequencies
        fft = np.abs(np.fft.rfft(enhanced))
        freqs = np.fft.rfftfreq(len(enhanced), 1/16000)
        
        # Find peaks near expected frequencies
        peak1 = np.max(fft[np.abs(freqs - 120) < 10])
        peak2 = np.max(fft[np.abs(freqs - 200) < 10])
        
        # Both speakers should be preserved
        self.assertGreater(peak1, np.median(fft) * 5)
        self.assertGreater(peak2, np.median(fft) * 3)
    
    def test_varying_sample_rates(self):
        """Test 5.5: Handle different sample rates correctly"""
        sample_rates = [8000, 16000, 22050, 44100, 48000]
        
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        for sr in sample_rates:
            # Create appropriate signal for sample rate
            duration = 2.0
            t = np.linspace(0, duration, int(duration * sr))
            signal = np.sin(2 * np.pi * 200 * t)
            
            # Add noise
            noisy = signal + 0.1 * np.random.randn(len(signal))
            
            # Should handle any sample rate
            enhanced, metadata = enhancer.enhance_to_target_snr(noisy, sr)
            
            self.assertIsNotNone(enhanced)
            self.assertEqual(len(enhanced), len(noisy))
            self.assertIn("snr_db", metadata)

#### 6. Performance Tests (`test_performance.py`)

class TestPerformance(unittest.TestCase):
    """Test performance requirements"""
    
    def test_processing_speed_requirement(self):
        """Test 6.1: Process 100 samples in under 3 minutes"""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        # Create 100 test samples
        samples = []
        for i in range(100):
            # Varying durations 1-5 seconds
            duration = 1 + np.random.rand() * 4
            signal = self._create_realistic_speech(duration=duration)
            noisy = self._add_noise_at_snr(signal, 20 + np.random.rand() * 10)
            samples.append((noisy, 16000))
        
        # Time processing
        import time
        start_time = time.time()
        
        for audio, sr in samples:
            enhanced, metadata = enhancer.enhance_to_target_snr(audio, sr)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Must be under 180 seconds (3 minutes)
        self.assertLess(total_time, 180, 
                       f"Processing took {total_time:.1f}s, exceeds 3 minute limit")
        
        # Calculate per-sample time
        per_sample = total_time / 100
        self.assertLess(per_sample, 1.8, 
                       f"Per-sample time {per_sample:.2f}s exceeds 1.8s limit")
    
    def test_gpu_memory_usage(self):
        """Test 6.2: Stay within 32GB VRAM limit"""
        import torch
        
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        
        gpu_processor = GPUEnhancementBatch(batch_size=32)
        
        # Create large batch
        batch = []
        for i in range(32):
            # 10 second audio
            audio = np.random.randn(10 * 16000).astype(np.float32)
            batch.append(audio)
        
        # Monitor GPU memory
        torch.cuda.reset_peak_memory_stats()
        
        # Process
        enhanced_batch = gpu_processor.process_batch(batch)
        
        # Check peak memory
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        
        # Should use less than 30GB (leaving headroom)
        self.assertLess(peak_memory_gb, 30.0,
                       f"Used {peak_memory_gb:.1f}GB, too close to 32GB limit")
    
    def test_batch_size_optimization(self):
        """Test 6.3: Optimal batch size for different audio lengths"""
        memory_manager = MemoryEfficientProcessor(max_memory_gb=30)
        
        test_cases = [
            {"duration": 1.0, "expected_batch": 64},
            {"duration": 5.0, "expected_batch": 32},
            {"duration": 10.0, "expected_batch": 16},
            {"duration": 30.0, "expected_batch": 4},
        ]
        
        for case in test_cases:
            batch_size = memory_manager.estimate_batch_size(
                case["duration"] * 16000, 16000
            )
            
            # Should be close to expected
            self.assertAlmostEqual(batch_size, case["expected_batch"], 
                                 delta=case["expected_batch"] * 0.3)

### Test Fixtures and Utilities

def _create_realistic_speech(duration=3.0, pitch=150):
    """Create speech-like signal with formants and modulation"""
    sr = 16000
    t = np.linspace(0, duration, int(duration * sr))
    
    # Fundamental frequency with vibrato
    f0 = pitch * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    
    # Generate harmonics with formant structure
    signal = np.zeros_like(t)
    formants = [700, 1220, 2600]  # Typical formant frequencies
    
    for i in range(1, 10):  # First 10 harmonics
        harmonic_freq = i * f0
        harmonic = np.sin(2 * np.pi * harmonic_freq * t)
        
        # Apply formant filtering
        for formant in formants:
            if abs(i * pitch - formant) < 200:
                harmonic *= 2.0  # Boost near formants
        
        signal += harmonic / i  # Natural harmonic rolloff
    
    # Add amplitude modulation (speech envelope)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    signal *= envelope
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal

def _add_noise_at_snr(signal, target_snr_db):
    """Add noise to achieve specific SNR"""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

## 7. Deployment & Operations

### Deployment Strategy
1. Feature flag for 35dB enhancement mode
2. Gradual rollout with monitoring
3. Backward compatibility maintenance
4. Configuration management

### Configuration
```python
# In config.py
AUDIO_ENHANCEMENT_35DB = {
    "enabled": True,
    "target_snr": 35.0,
    "max_processing_time": 1.8,  # seconds per sample
    "gpu_batch_size": 32,
    "quality_thresholds": {
        "min_pesq": 3.5,
        "min_stoi": 0.85
    }
}
```

### Monitoring Requirements
- Real-time SNR achievement tracking
- Processing time monitoring
- GPU memory usage tracking
- Quality metrics dashboard

### Operational Procedures
1. Regular quality audits
2. Performance optimization reviews
3. Noise profile updates
4. Enhancement algorithm tuning

## 8. Future Considerations

### Potential Enhancements
1. **Adaptive Thresholds**: Different SNR targets for different content types
2. **Advanced Models**: Integration of deep learning denoisers
3. **Real-time Adaptation**: Dynamic enhancement based on content
4. **Multi-modal Enhancement**: Using visual cues for better enhancement

### Scalability Planning
- Distributed processing for large datasets
- Model quantization for faster inference
- Streaming enhancement for real-time applications

### Technical Debt Considerations
- Regular refactoring of enhancement pipeline
- Performance optimization iterations
- Documentation updates
- Test coverage expansion

### Research Opportunities
1. Custom denoising models for Thai speech
2. Perceptual loss functions for naturalness
3. Zero-shot noise adaptation
4. Self-supervised enhancement learning