# Programmatic Verification Guide

## Overview
This guide provides comprehensive approaches for implementing zero-human-intervention verification systems for audio quality and primary speaker preservation. All verification must be done programmatically through code and commands.

## Core Principles

### 1. No Human in the Loop
- Every quality metric must be computable
- All pass/fail decisions must be automated
- Ground truth must be generated programmatically
- Validation reports must be machine-readable

### 2. Statistical Rigor
- Use proper hypothesis testing
- Calculate effect sizes, not just p-values
- Bootstrap confidence intervals
- Account for multiple comparisons

### 3. Comprehensive Coverage
- Test across diverse scenarios
- Verify both positive and negative cases
- Check edge cases automatically
- Validate across noise types and levels

## Implementation Architecture

### Test Generation Pipeline
```python
# Automated test case generation
test_generator = AutomatedTestGenerator()
test_suite = test_generator.generate_test_suite()

# Each test case includes:
# - Clean reference audio (synthetic)
# - Noisy input (controlled mixing)
# - Ground truth metadata
# - Expected quality bounds
```

### Verification Components

#### 1. Primary Speaker Preservation
- **Speaker Embedding Similarity**: Must be > 0.85
- **F0 Correlation**: Track pitch consistency
- **Formant Preservation**: < 10% deviation
- **Temporal Alignment**: No time stretching

#### 2. Quality Metrics
- **Pseudo-PESQ**: Perceptual quality without ITU license
- **STOI Alternative**: Intelligibility measurement
- **SNR Improvement**: Measurable noise reduction
- **Harmonic Preservation**: Maintain voice characteristics

#### 3. Test Oracles
- **Invariant Checking**: Properties that must always hold
- **Metamorphic Testing**: Relations between transformations
- **Property-Based Testing**: Generate tests from specifications

## Practical Examples

### Example 1: Speaker Preservation Test
```python
def test_speaker_preservation():
    # Generate test audio
    clean = generate_synthetic_speech(duration=3.0, f0=120)
    noisy = add_noise(clean, snr=10)
    
    # Process
    enhanced = audio_processor.process(noisy)
    
    # Verify
    preservation_score = verify_speaker_preservation(clean, enhanced)
    assert preservation_score > 0.85, f"Speaker not preserved: {preservation_score}"
```

### Example 2: Quality Improvement Test
```python
def test_quality_improvement():
    results = []
    
    for snr in [-5, 0, 5, 10, 15, 20]:
        clean = generate_speech()
        noisy = mix_at_snr(clean, generate_noise(), snr)
        enhanced = enhance_audio(noisy)
        
        improvement = calculate_snr(enhanced) - snr
        results.append(improvement)
    
    # Statistical validation
    assert np.mean(results) > 10, "Insufficient improvement"
    assert all(r > 0 for r in results), "Regression detected"
```

### Example 3: CI/CD Integration
```python
class AudioValidationPipeline:
    def run(self):
        # 1. Generate tests
        tests = self.generate_test_cases(n=100)
        
        # 2. Execute processing
        results = self.process_all_tests(tests)
        
        # 3. Validate results
        validation = self.validate_results(results)
        
        # 4. Generate report
        report = self.generate_report(validation)
        
        # 5. Enforce quality gates
        if not self.passes_quality_gates(report):
            raise ValidationError(f"Quality gates failed: {report}")
        
        return report
```

## Quality Gates

### Mandatory Criteria
1. **Speaker Preservation**: > 85% similarity
2. **SNR Improvement**: > 10dB average
3. **No Regressions**: All tests must improve
4. **Processing Speed**: < 100ms latency
5. **Statistical Significance**: p < 0.05

### Nice-to-Have Criteria
1. **Perceptual Quality**: PESQ > 3.5
2. **Intelligibility**: STOI > 0.85
3. **Consistency**: Std dev < 10%

## Automated Reporting

### JSON Report Format
```json
{
  "timestamp": "2025-06-09T10:00:00Z",
  "summary": {
    "total_tests": 100,
    "passed": 95,
    "failed": 5,
    "pass_rate": 0.95
  },
  "metrics": {
    "speaker_preservation": {
      "mean": 0.87,
      "std": 0.05,
      "min": 0.75,
      "max": 0.95
    },
    "snr_improvement": {
      "mean": 12.5,
      "std": 3.2,
      "min": 5.0,
      "max": 20.0
    }
  },
  "statistical_validation": {
    "effect_size": 1.2,
    "p_value": 0.001,
    "confidence_interval": [11.8, 13.2]
  }
}
```

## Integration with pytest

```python
# tests/test_audio_quality_programmatic.py
import pytest
from verification import ProgrammaticVerifier

class TestAudioQualityProgrammatic:
    @pytest.fixture
    def verifier(self):
        return ProgrammaticVerifier()
    
    def test_speaker_preservation_all_scenarios(self, verifier):
        """Test speaker preservation across all noise types"""
        results = verifier.test_speaker_preservation()
        assert results['pass_rate'] > 0.95
        assert results['mean_score'] > 0.85
    
    def test_quality_improvement_statistical(self, verifier):
        """Statistically validate quality improvement"""
        results = verifier.test_quality_improvement()
        assert results['p_value'] < 0.05
        assert results['effect_size'] > 0.8
    
    @pytest.mark.parametrize("noise_type,snr", [
        ("white", -5), ("pink", 0), ("babble", 5), ("traffic", 10)
    ])
    def test_specific_scenarios(self, verifier, noise_type, snr):
        """Test specific challenging scenarios"""
        result = verifier.test_scenario(noise_type, snr)
        assert result['improved'], f"Failed for {noise_type} at {snr}dB"
```

## Best Practices

1. **Use Synthetic Data**: Generate test data with known properties
2. **Multiple Metrics**: Don't rely on single metric
3. **Statistical Power**: Ensure sufficient test cases
4. **Reproducibility**: Set random seeds
5. **Continuous Monitoring**: Track metrics over time
6. **Automated Alerts**: Notify on regression

## Troubleshooting

### Common Issues
1. **Flaky Tests**: Increase test stability with larger samples
2. **False Positives**: Tighten thresholds gradually
3. **Performance**: Use parallel processing
4. **Memory**: Stream large test sets

### Debug Commands
```bash
# Run single test with verbose output
pytest tests/test_audio_quality_programmatic.py::test_speaker_preservation -v

# Generate detailed report
python scripts/generate_validation_report.py --detailed

# Check specific metric
python scripts/check_metric.py speaker_preservation --threshold 0.85
```

## Conclusion
Programmatic verification enables continuous validation without human intervention. By combining synthetic test generation, comprehensive metrics, and statistical validation, we can ensure audio quality and primary speaker preservation at scale.