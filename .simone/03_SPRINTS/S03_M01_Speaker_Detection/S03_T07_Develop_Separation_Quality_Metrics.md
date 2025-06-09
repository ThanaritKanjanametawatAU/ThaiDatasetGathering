# Task S03_T07: Develop Separation Quality Metrics

## Task Overview
Develop a comprehensive metrics system specifically designed to evaluate the quality of speaker separation and secondary speaker removal processes.

## Technical Requirements

### Core Implementation
- **Separation Metrics** (`processors/audio_enhancement/metrics/separation_metrics.py`)
  - Source-specific metrics
  - Interference measures
  - Artifact detection
  - Perceptual quality scores

### Key Features
1. **Objective Metrics**
   - Source-to-Interference Ratio (SIR)
   - Source-to-Artifact Ratio (SAR)
   - Source-to-Distortion Ratio (SDR)
   - Perceptual metrics adaptation

2. **Quality Assessment**
   - Leakage detection
   - Cross-talk measurement
   - Naturalness scoring
   - Intelligibility impact

3. **Comparative Analysis**
   - Before/after comparison
   - Method benchmarking
   - Consistency tracking
   - Performance trends

## TDD Requirements

### Test Structure
```
tests/test_separation_metrics.py
- test_sir_calculation()
- test_sar_calculation()
- test_artifact_detection()
- test_perceptual_scoring()
- test_metric_consistency()
- test_edge_cases()
```

### Test Data Requirements
- Perfect separation examples
- Various separation qualities
- Known metric values
- Artifact samples

## Implementation Approach

### Phase 1: Core Metrics
```python
class SeparationMetrics:
    def __init__(self):
        self.metrics = ['sir', 'sar', 'sdr', 'perceptual']
        self.artifact_detector = ArtifactDetector()
        
    def evaluate(self, original, separated, reference=None):
        # Comprehensive evaluation
        pass
    
    def calculate_sir(self, target, interference):
        # Source-to-Interference Ratio
        pass
    
    def detect_artifacts(self, separated_audio):
        # Artifact detection and quantification
        pass
```

### Phase 2: Advanced Metrics
- Perceptual quality metrics
- Time-frequency analysis
- Psychoacoustic modeling
- Machine learning scores

### Phase 3: Integration
- Real-time monitoring
- Automated reporting
- Threshold management
- Visualization tools

## Acceptance Criteria
1. ✅ Metric accuracy validated against benchmarks
2. ✅ Support for multi-source evaluation
3. ✅ Artifact detection sensitivity > 90%
4. ✅ Real-time calculation capability
5. ✅ Comprehensive reporting

## Example Usage
```python
from processors.audio_enhancement.metrics import SeparationMetrics

# Initialize metrics
metrics = SeparationMetrics()

# Evaluate separation quality
results = metrics.evaluate(
    original=mixed_audio,
    separated=separated_audio,
    reference=clean_target
)

print(f"SIR: {results.sir:.1f} dB")
print(f"SAR: {results.sar:.1f} dB")
print(f"SDR: {results.sdr:.1f} dB")
print(f"Perceptual Score: {results.perceptual:.2f}/5")

# Detect artifacts
artifacts = metrics.detect_artifacts(separated_audio)
print(f"Artifact severity: {artifacts.severity}")
print(f"Artifact locations: {artifacts.timestamps}")

# Compare methods
comparison = metrics.compare_methods({
    'method1': separated_audio1,
    'method2': separated_audio2,
    'method3': separated_audio3
})
print(f"Best method: {comparison.best_method}")
```

## Dependencies
- mir_eval for BSS metrics
- NumPy for calculations
- SciPy for signal analysis
- Librosa for audio features
- Matplotlib for visualization

## Performance Targets
- Metric calculation: < 100ms per file
- Batch evaluation: > 50 files/second
- Memory usage: < 200MB
- Accuracy: Within 0.1 dB of reference

## Notes
- Consider perceptual importance
- Support for incomplete separation
- Enable custom metric weights
- Provide actionable insights