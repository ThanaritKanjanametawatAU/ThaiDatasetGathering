# Quality Threshold Manager

## Overview

The Quality Threshold Manager is a dynamic quality threshold management system that defines, monitors, and enforces quality standards throughout the audio enhancement pipeline. It provides flexible threshold configuration, multi-level quality gates, and adaptive threshold adjustment based on historical performance.

## Features

### 1. **Dynamic Threshold Configuration**
- Load thresholds from YAML/JSON files or Python dictionaries
- Support for multiple profiles (production, development, dataset-specific)
- Profile inheritance for easy configuration management

### 2. **Multi-Level Quality Gates**
- Pre-enhancement validation
- Post-enhancement verification
- Multi-stage checkpoints with stage-specific thresholds
- Fail-fast mechanisms to save processing time

### 3. **Adaptive Threshold Adjustment**
- Percentile-based threshold updates from historical data
- Machine learning-based threshold optimization
- Automatic threshold tuning while respecting safety limits

### 4. **Comprehensive Quality Assessment**
- Support for multiple metrics (PESQ, STOI, SI-SDR, SNR, etc.)
- Weighted metric importance for overall scoring
- Severity classification (minor, major, critical)
- Recovery suggestions for failed quality checks

## Usage

### Basic Example

```python
from processors.audio_enhancement.quality import QualityThresholdManager

# Initialize with default production profile
manager = QualityThresholdManager(profile='production')

# Check audio quality
metrics = {
    'pesq': 3.5,
    'stoi': 0.85,
    'si_sdr': 15.2,
    'snr': 25.0
}

result = manager.check_quality(metrics)
if not result.passed:
    print(f"Quality check failed: {result.failures}")
    print(f"Suggestions: {result.recovery_suggestions}")
```

### Advanced Configuration

```python
# Custom threshold configuration
config = {
    'strict': {
        'pesq': {'min': 3.5, 'target': 4.0, 'max': 5.0},
        'stoi': {'min': 0.85, 'target': 0.92, 'max': 1.0},
        'si_sdr': {'min': 15.0, 'target': 20.0, 'max': 30.0},
        'snr': {'min': 25.0, 'target': 30.0, 'max': 40.0}
    }
}

# Initialize with weighted metrics
weights = {
    'pesq': 0.4,   # Most important
    'stoi': 0.3,   # Second most important
    'si_sdr': 0.2,
    'snr': 0.1     # Least important
}

manager = QualityThresholdManager(
    profile='strict',
    config=config,
    metric_weights=weights
)
```

### Multi-Stage Quality Gates

```python
# Configure stage-specific thresholds
stage_config = {
    'production': {
        'pre_enhancement': {
            'snr': {'min': 10.0, 'target': 15.0, 'max': 40.0}
        },
        'post_enhancement': {
            'pesq': {'min': 3.0, 'target': 3.5, 'max': 5.0},
            'stoi': {'min': 0.8, 'target': 0.9, 'max': 1.0},
            'si_sdr': {'min': 10.0, 'target': 15.0, 'max': 30.0},
            'snr': {'min': 20.0, 'target': 25.0, 'max': 40.0}
        }
    }
}

manager = QualityThresholdManager(profile='production', config=stage_config)

# Get stage-specific gates
pre_gate = manager.get_quality_gate('pre_enhancement')
post_gate = manager.get_quality_gate('post_enhancement')
```

### Adaptive Learning

```python
# Enable adaptive learning
manager = QualityThresholdManager(
    profile='production',
    enable_adaptive_learning=True
)

# Add samples with success indicators
for sample in processing_results:
    metrics_with_success = {
        'pesq': sample['pesq'],
        'stoi': sample['stoi'],
        'si_sdr': sample['si_sdr'],
        'snr': sample['snr'],
        'enhancement_success': sample['user_approved']
    }
    manager.add_sample(metrics_with_success)

# Optimize thresholds based on success patterns
manager.optimize_thresholds()

# Predict success for new samples
success_prob = manager.predict_enhancement_success(new_metrics)
```

## Integration with Enhancement Pipeline

The Quality Threshold Manager integrates seamlessly with the existing audio enhancement pipeline:

```python
from processors.audio_enhancement.enhancement_orchestrator import EnhancementOrchestrator
from processors.audio_enhancement.quality import QualityThresholdManager

# Initialize components
orchestrator = EnhancementOrchestrator()
quality_manager = QualityThresholdManager(profile='production')

# Pre-enhancement check
pre_metrics = {'snr': calculate_snr(audio)}
pre_check = quality_manager.check_quality(pre_metrics)

if pre_check.passed:
    # Proceed with enhancement
    enhanced = orchestrator.enhance(audio)
    
    # Post-enhancement validation
    post_metrics = calculate_all_metrics(original, enhanced)
    post_check = quality_manager.check_quality(post_metrics)
    
    if not post_check.passed:
        # Apply recovery strategies based on suggestions
        for suggestion in post_check.recovery_suggestions:
            # Implement recovery logic
            pass
```

## Performance

- Threshold check: < 1ms per metric
- Profile loading: < 10ms
- Adaptive update: < 100ms
- Memory usage: < 50MB

## Configuration Files

### YAML Example

```yaml
production:
  pesq:
    min: 3.0
    target: 3.5
    max: 5.0
  stoi:
    min: 0.8
    target: 0.9
    max: 1.0
    
development:
  _inherit: production
  pesq:
    min: 2.5  # Override production value
```

### JSON Example

```json
{
  "production": {
    "pesq": {"min": 3.0, "target": 3.5, "max": 5.0},
    "stoi": {"min": 0.8, "target": 0.9, "max": 1.0}
  }
}
```

## API Reference

### QualityThresholdManager

- `__init__(profile='production', config=None, config_path=None, metric_weights=None, enable_adaptive_learning=False)`
- `check_quality(metrics)` → QualityCheckResult
- `update_thresholds(performance_data)` → None
- `get_quality_gate(stage)` → QualityGate
- `switch_profile(new_profile)` → None
- `add_sample(sample)` → None
- `optimize_thresholds()` → None
- `predict_enhancement_success(metrics)` → float
- `detect_outliers(metrics, z_threshold=2.0)` → Tuple[bool, Dict]

### QualityCheckResult

- `passed`: bool - Whether all metrics meet thresholds
- `score`: float - Overall quality score (0.0 to 1.0)
- `failures`: Dict - Details of failed metrics
- `severity`: str - Severity level (none, minor, major, critical)
- `metric_scores`: Dict - Individual metric scores
- `recovery_suggestions`: List[str] - Suggestions for improvement

### QualityGate

- `name`: str - Gate identifier
- `thresholds`: Dict - Metric thresholds for this gate
- `check(metrics)` → QualityCheckResult