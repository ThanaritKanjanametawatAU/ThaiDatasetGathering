# Audio Analysis Pipeline Developer Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Quick Start Examples](#quick-start-examples)
3. [Component Usage](#component-usage)
4. [Integration Patterns](#integration-patterns)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/thai-dataset-gathering.git
cd thai-dataset-gathering

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements_dev.txt
```

### Basic Setup

```python
# Import required components
from processors.audio_enhancement.audio_loader import AudioLoader
from utils.enhanced_snr_calculator import EnhancedSNRCalculator
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer
from processors.audio_enhancement.detection.pattern_detector import PatternDetector
from processors.audio_enhancement.issue_categorization import IssueCategorizer
from processors.audio_enhancement.decision_framework import DecisionEngine
```

## Quick Start Examples

### Example 1: Basic Audio Quality Check

```python
import numpy as np
import soundfile as sf

# Create a simple pipeline
def check_audio_quality(audio_path):
    # Load audio
    loader = AudioLoader()
    audio_data = loader.load(audio_path)
    
    # Calculate SNR
    snr_calculator = EnhancedSNRCalculator()
    snr_metrics = snr_calculator.calculate(
        audio_data.samples, 
        audio_data.sample_rate
    )
    
    # Make decision based on SNR
    if snr_metrics.global_snr > 30:
        return "High quality audio"
    elif snr_metrics.global_snr > 20:
        return "Good quality audio"
    elif snr_metrics.global_snr > 10:
        return "Moderate quality - enhancement recommended"
    else:
        return "Poor quality - consider rejection"

# Usage
result = check_audio_quality("path/to/audio.wav")
print(f"Audio quality: {result}")
```

### Example 2: Complete Pipeline Analysis

```python
from tests.test_audio_analysis_integration import AudioAnalysisPipeline

def analyze_audio_file(audio_path):
    """Complete audio analysis with all components."""
    pipeline = AudioAnalysisPipeline()
    
    # Process audio through pipeline
    results = pipeline.process(audio_path)
    
    # Extract key information
    print(f"Audio file: {audio_path}")
    print(f"SNR: {results['snr_metrics'].global_snr:.2f} dB")
    print(f"Issues found: {len(results['issues'])}")
    
    for issue in results['issues']:
        print(f"  - {issue.type.value}: {issue.severity.value} "
              f"(confidence: {issue.confidence:.2f})")
    
    print(f"Decision: {results['decision'].action}")
    print(f"Reasoning: {', '.join(results['decision'].reasoning)}")
    
    return results

# Usage
results = analyze_audio_file("sample_audio.wav")
```

### Example 3: Batch Processing

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json

def batch_analyze_directory(directory_path, output_file="analysis_results.json"):
    """Analyze all audio files in a directory."""
    pipeline = AudioAnalysisPipeline()
    audio_files = list(Path(directory_path).glob("*.wav"))
    
    print(f"Found {len(audio_files)} audio files to analyze")
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(pipeline.process, audio_files))
    
    # Compile summary
    summary = {
        "total_files": len(audio_files),
        "processed": len(results),
        "decisions": {
            "PROCESS": sum(1 for r in results if r['decision'].action == 'PROCESS'),
            "ENHANCE": sum(1 for r in results if r['decision'].action == 'ENHANCE'),
            "REJECT": sum(1 for r in results if r['decision'].action == 'REJECT'),
        },
        "average_snr": np.mean([r['snr_metrics'].global_snr for r in results]),
        "detailed_results": [
            {
                "file": str(audio_files[i]),
                "snr": results[i]['snr_metrics'].global_snr,
                "issues": len(results[i]['issues']),
                "decision": results[i]['decision'].action
            }
            for i in range(len(results))
        ]
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_file}")
    return summary

# Usage
summary = batch_analyze_directory("/path/to/audio/files")
```

## Component Usage

### Audio Loader

The AudioLoader handles file loading, validation, and preprocessing.

```python
from processors.audio_enhancement.audio_loader import AudioLoader, AudioCache

# Basic usage
loader = AudioLoader()
audio_data = loader.load("audio.wav")

# With caching
cache = AudioCache(max_size=1000)
loader = AudioLoader(cache=cache)

# Multiple loads of same file will use cache
audio1 = loader.load("audio.wav")  # Loads from disk
audio2 = loader.load("audio.wav")  # Loads from cache

# Custom preprocessing
class CustomLoader(AudioLoader):
    def preprocess(self, audio_data):
        # Apply custom preprocessing
        audio_data.samples = self.apply_filter(audio_data.samples)
        return audio_data
        
    def apply_filter(self, samples):
        # Your filter implementation
        return samples
```

### SNR Calculator

Calculate various SNR metrics for audio quality assessment.

```python
from utils.enhanced_snr_calculator import EnhancedSNRCalculator

calculator = EnhancedSNRCalculator()

# Basic SNR calculation
snr_metrics = calculator.calculate(audio_samples, sample_rate)
print(f"Global SNR: {snr_metrics.global_snr:.2f} dB")

# Segmental SNR for temporal analysis
segmental_snr = snr_metrics.segmental_snr
print(f"Average segmental SNR: {np.mean(segmental_snr):.2f} dB")
print(f"SNR variation: {np.std(segmental_snr):.2f} dB")

# Access noise profile
noise_profile = snr_metrics.noise_profile
print(f"Noise floor: {noise_profile.noise_floor:.2f} dB")

# Voice Activity Detection results
vad_segments = snr_metrics.vad_segments
speech_ratio = sum(s.duration for s in vad_segments) / audio_duration
print(f"Speech ratio: {speech_ratio:.2%}")
```

### Spectral Analyzer

Extract spectral features and perform time-frequency analysis.

```python
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer

analyzer = SpectralAnalyzer()

# Full spectral analysis
features = analyzer.analyze(audio_samples, sample_rate)

# Access individual features
print(f"Spectral centroid: {features.spectral_centroid:.2f} Hz")
print(f"Spectral rolloff: {features.spectral_rolloff:.2f} Hz")
print(f"Spectral flux: {features.spectral_flux:.4f}")
print(f"Zero crossing rate: {features.zero_crossing_rate:.4f}")

# Compute spectrogram
spectrogram = analyzer.compute_spectrogram(audio_samples)

# Custom feature extraction
custom_features = analyzer.extract_features(
    features.magnitude_spectrum,
    features.frequency_bins
)
```

### Pattern Detector

Detect various patterns and anomalies in audio.

```python
from processors.audio_enhancement.detection.pattern_detector import PatternDetector

detector = PatternDetector()

# Detect all patterns
patterns = detector.detect(audio_samples, spectral_features)

# Filter by pattern type
clipping_patterns = [p for p in patterns if p.pattern_type == 'clipping']
silence_patterns = [p for p in patterns if p.pattern_type == 'silence']

# Check specific anomalies
anomaly_detector = detector.anomaly_detector
anomalies = anomaly_detector.detect(audio_samples)

for anomaly in anomalies:
    print(f"Anomaly at {anomaly.timestamp}s: {anomaly.description}")

# Temporal pattern analysis
temporal_analyzer = detector.temporal_analyzer
temporal_patterns = temporal_analyzer.analyze(audio_samples)
```

### Issue Categorizer

Categorize and prioritize detected issues.

```python
from processors.audio_enhancement.issue_categorization import (
    IssueCategorizer, IssueType, IssueSeverity
)

categorizer = IssueCategorizer()

# Categorize issues from analysis
issues = categorizer.categorize(snr_metrics, spectral_features, patterns)

# Group by type
issues_by_type = {}
for issue in issues:
    issue_type = issue.type.value
    if issue_type not in issues_by_type:
        issues_by_type[issue_type] = []
    issues_by_type[issue_type].append(issue)

# Filter by severity
severe_issues = [i for i in issues if i.severity == IssueSeverity.SEVERE]
moderate_issues = [i for i in issues if i.severity == IssueSeverity.MODERATE]

# Custom categorization rules
categorizer.add_rule(
    name="custom_noise_rule",
    condition=lambda m, s, p: m.global_snr < 15,
    issue_type=IssueType.NOISE,
    severity=IssueSeverity.SEVERE,
    confidence=0.9
)
```

### Decision Engine

Make intelligent decisions based on analysis results.

```python
from processors.audio_enhancement.decision_framework import (
    DecisionEngine, DecisionContext, DecisionRule
)

engine = DecisionEngine()

# Basic decision making
decision = engine.decide(issues)
print(f"Action: {decision.action}")
print(f"Confidence: {decision.confidence:.2f}")
print("Reasoning:")
for reason in decision.reasoning:
    print(f"  - {reason}")

# With context
context = DecisionContext(
    audio_metrics=snr_metrics,
    spectral_features=spectral_features,
    detected_patterns=patterns,
    metadata={'duration': 10.5, 'source': 'microphone'}
)
decision = engine.decide(issues, context)

# Custom decision rules
custom_rule = DecisionRule(
    name="low_snr_rejection",
    condition=lambda issues, ctx: ctx.audio_metrics.global_snr < 5,
    action="REJECT",
    confidence=0.95,
    reasoning="SNR too low for any enhancement"
)
engine.add_rule(custom_rule)
```

## Integration Patterns

### Integration with Main Pipeline

```python
# In your main processing script
from main import process_audio_file

def process_with_quality_check(audio_path, quality_threshold=20):
    """Process audio with quality checking."""
    
    # Analyze audio quality first
    pipeline = AudioAnalysisPipeline()
    analysis = pipeline.process(audio_path)
    
    # Check if quality meets threshold
    if analysis['snr_metrics'].global_snr < quality_threshold:
        if analysis['decision'].action == 'ENHANCE':
            # Apply enhancement before processing
            enhanced_path = enhance_audio(audio_path, analysis['issues'])
            return process_audio_file(enhanced_path)
        else:
            # Skip poor quality audio
            print(f"Skipping {audio_path}: {analysis['decision'].reasoning}")
            return None
    
    # Process good quality audio directly
    return process_audio_file(audio_path)
```

### Integration with Enhancement Pipeline

```python
from processors.audio_enhancement.enhancement_orchestrator import EnhancementOrchestrator

def enhance_based_on_analysis(audio_path, analysis_results):
    """Enhance audio based on analysis results."""
    
    orchestrator = EnhancementOrchestrator()
    
    # Configure enhancement based on issues
    enhancement_config = {
        'denoise': any(i.type == IssueType.NOISE for i in analysis_results['issues']),
        'declip': any(i.type == IssueType.DISTORTION for i in analysis_results['issues']),
        'deecho': any(i.type == IssueType.ECHO for i in analysis_results['issues']),
        'target_snr': max(35, analysis_results['snr_metrics'].global_snr + 10)
    }
    
    # Apply enhancement
    enhanced_audio = orchestrator.enhance(
        audio_path,
        config=enhancement_config,
        issues=analysis_results['issues']
    )
    
    return enhanced_audio
```

### Custom Pipeline Configuration

```python
def create_custom_pipeline(config_dict):
    """Create pipeline with custom configuration."""
    
    # Create components with custom config
    loader = AudioLoader(
        target_sample_rate=config_dict.get('sample_rate', 16000),
        cache_size=config_dict.get('cache_size', 100)
    )
    
    snr_calc = EnhancedSNRCalculator(
        frame_length=config_dict.get('frame_length', 0.025),
        min_speech_duration=config_dict.get('min_speech', 0.3)
    )
    
    # Build custom pipeline
    class CustomPipeline:
        def __init__(self):
            self.loader = loader
            self.snr_calculator = snr_calc
            # ... other components
            
        def process(self, audio_path):
            # Custom processing logic
            pass
    
    return CustomPipeline()
```

## Advanced Features

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def parallel_analysis(audio_files, max_workers=None):
    """Analyze multiple files in parallel."""
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    pipeline = AudioAnalysisPipeline()
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(pipeline.process, f): f 
            for f in audio_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            audio_file = future_to_file[future]
            try:
                result = future.result()
                results[audio_file] = result
                print(f"Completed: {audio_file}")
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results[audio_file] = {'error': str(e)}
    
    return results
```

### Streaming Analysis

```python
def streaming_analysis(audio_stream, chunk_size=16000):
    """Analyze audio in streaming fashion."""
    
    snr_calculator = EnhancedSNRCalculator()
    pattern_detector = PatternDetector()
    
    chunk_results = []
    
    for chunk in audio_stream:
        # Analyze chunk
        snr = snr_calculator.calculate(chunk, 16000)
        patterns = pattern_detector.detect(chunk, None)
        
        chunk_results.append({
            'snr': snr.global_snr,
            'patterns': len(patterns),
            'timestamp': len(chunk_results) * chunk_size / 16000
        })
        
        # Real-time decision
        if snr.global_snr < 10:
            print("Warning: Low quality audio detected")
    
    return chunk_results
```

### Custom Component Development

```python
from abc import ABC, abstractmethod

class BaseAudioAnalyzer(ABC):
    """Base class for custom analyzers."""
    
    @abstractmethod
    def analyze(self, audio_data):
        """Analyze audio data."""
        pass
        
class CustomFrequencyAnalyzer(BaseAudioAnalyzer):
    """Custom analyzer for specific frequency bands."""
    
    def __init__(self, frequency_bands):
        self.frequency_bands = frequency_bands
        
    def analyze(self, audio_data):
        results = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Analyze specific frequency band
            band_energy = self._calculate_band_energy(
                audio_data, low_freq, high_freq
            )
            results[band_name] = band_energy
            
        return results
        
    def _calculate_band_energy(self, audio_data, low_freq, high_freq):
        # Implementation details
        pass

# Usage
analyzer = CustomFrequencyAnalyzer({
    'bass': (20, 250),
    'midrange': (250, 4000),
    'treble': (4000, 20000)
})
```

### Machine Learning Integration

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

class MLQualityPredictor:
    """ML-based audio quality prediction."""
    
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = None
            
    def extract_features(self, audio_analysis):
        """Extract features for ML model."""
        features = [
            audio_analysis['snr_metrics'].global_snr,
            np.mean(audio_analysis['snr_metrics'].segmental_snr),
            audio_analysis['spectral_features'].spectral_centroid,
            audio_analysis['spectral_features'].spectral_rolloff,
            len(audio_analysis['patterns']),
            len(audio_analysis['issues'])
        ]
        return np.array(features).reshape(1, -1)
        
    def predict_quality(self, audio_analysis):
        """Predict audio quality using ML model."""
        if self.model is None:
            raise ValueError("No model loaded")
            
        features = self.extract_features(audio_analysis)
        quality_score = self.model.predict_proba(features)[0, 1]
        
        return {
            'quality_score': quality_score,
            'quality_label': 'good' if quality_score > 0.7 else 'poor'
        }
```

## Best Practices

### 1. Error Handling

Always implement proper error handling:

```python
from processors.audio_enhancement.audio_loader import AudioLoadError

def safe_audio_analysis(audio_path):
    """Analyze audio with comprehensive error handling."""
    try:
        pipeline = AudioAnalysisPipeline()
        result = pipeline.process(audio_path)
        return {'success': True, 'result': result}
        
    except AudioLoadError as e:
        return {'success': False, 'error': f"Failed to load audio: {e}"}
        
    except MemoryError:
        # Try with reduced settings
        return analyze_with_reduced_memory(audio_path)
        
    except Exception as e:
        # Log unexpected errors
        import logging
        logging.error(f"Unexpected error analyzing {audio_path}: {e}")
        return {'success': False, 'error': str(e)}
```

### 2. Resource Management

Properly manage resources:

```python
class AudioAnalysisContext:
    """Context manager for audio analysis."""
    
    def __init__(self, cache_size=100):
        self.cache_size = cache_size
        self.pipeline = None
        
    def __enter__(self):
        self.pipeline = AudioAnalysisPipeline()
        return self.pipeline
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources
        if hasattr(self.pipeline, 'cleanup'):
            self.pipeline.cleanup()
        # Clear cache
        import gc
        gc.collect()

# Usage
with AudioAnalysisContext() as pipeline:
    result = pipeline.process("audio.wav")
```

### 3. Configuration Management

Use configuration files:

```python
import yaml

def load_pipeline_config(config_path):
    """Load pipeline configuration from file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    required_keys = ['audio_loader', 'snr_calculator', 'decision_engine']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration: {key}")
    
    return config

# Usage
config = load_pipeline_config('config/audio_analysis.yaml')
pipeline = AudioAnalysisPipeline(config)
```

### 4. Logging and Monitoring

Implement comprehensive logging:

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('audio_analysis')

def analyze_with_logging(audio_path):
    """Analyze audio with detailed logging."""
    start_time = datetime.now()
    logger.info(f"Starting analysis of {audio_path}")
    
    try:
        pipeline = AudioAnalysisPipeline()
        result = pipeline.process(audio_path)
        
        # Log results
        logger.info(f"Analysis complete: SNR={result['snr_metrics'].global_snr:.2f}dB, "
                   f"Issues={len(result['issues'])}, "
                   f"Decision={result['decision'].action}")
        
        # Log performance
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing time: {duration:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise
```

### 5. Testing Your Implementation

Always test your code:

```python
import unittest
import tempfile
import numpy as np

class TestCustomAnalysis(unittest.TestCase):
    """Test custom analysis implementation."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def test_quality_detection(self):
        """Test quality detection accuracy."""
        # Create test audio
        clean_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        noisy_audio = clean_audio + np.random.normal(0, 0.3, 16000)
        
        # Test clean audio
        clean_result = analyze_audio_array(clean_audio, 16000)
        self.assertGreater(clean_result['snr'], 20)
        
        # Test noisy audio
        noisy_result = analyze_audio_array(noisy_audio, 16000)
        self.assertLess(noisy_result['snr'], 10)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors with Large Files

**Problem**: `MemoryError` when processing large audio files

**Solution**:
```python
def process_large_file(audio_path, chunk_duration=30):
    """Process large files in chunks."""
    audio_info = sf.info(audio_path)
    total_duration = audio_info.duration
    
    results = []
    for start in range(0, int(total_duration), chunk_duration):
        end = min(start + chunk_duration, total_duration)
        
        # Load chunk
        chunk, sr = sf.read(audio_path, start=start*audio_info.samplerate, 
                           stop=end*audio_info.samplerate)
        
        # Analyze chunk
        chunk_result = analyze_audio_array(chunk, sr)
        results.append(chunk_result)
    
    # Aggregate results
    return aggregate_chunk_results(results)
```

#### 2. Slow Processing Speed

**Problem**: Analysis takes too long

**Solution**:
```python
# Enable caching
cache = AudioCache(max_size=1000)

# Use parallel processing
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(analyze_audio, audio_files))

# Reduce analysis resolution
config = {
    'spectral_analyzer': {
        'fft_size': 1024,  # Reduced from 2048
        'hop_size': 512    # Increased from 256
    }
}
```

#### 3. Inconsistent Results

**Problem**: Same file gives different results

**Solution**:
```python
# Set random seed for reproducibility
np.random.seed(42)

# Disable adaptive features
config = {
    'decision_engine': {
        'enable_adaptive_learning': False
    }
}

# Use deterministic algorithms
snr_calculator = EnhancedSNRCalculator(use_random=False)
```

### Debugging Tips

#### Enable Debug Mode

```python
import logging

# Set debug level
logging.getLogger('audio_analysis').setLevel(logging.DEBUG)

# Enable component debugging
pipeline = AudioAnalysisPipeline(debug=True)
```

#### Profile Performance

```python
import cProfile
import pstats

def profile_analysis(audio_path):
    """Profile analysis performance."""
    profiler = cProfile.Profile()
    
    profiler.enable()
    result = analyze_audio_file(audio_path)
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

#### Visualize Intermediate Results

```python
import matplotlib.pyplot as plt

def debug_analysis_visually(audio_path):
    """Visualize analysis steps for debugging."""
    # Load and analyze
    loader = AudioLoader()
    audio = loader.load(audio_path)
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Plot waveform
    axes[0, 0].plot(audio.samples[:10000])
    axes[0, 0].set_title('Waveform')
    
    # Plot spectrum
    analyzer = SpectralAnalyzer()
    spectrum = analyzer.analyze(audio.samples, audio.sample_rate)
    axes[0, 1].plot(spectrum.frequency_bins, spectrum.magnitude_spectrum)
    axes[0, 1].set_title('Spectrum')
    
    # Plot SNR over time
    calculator = EnhancedSNRCalculator()
    metrics = calculator.calculate(audio.samples, audio.sample_rate)
    axes[1, 0].plot(metrics.segmental_snr)
    axes[1, 0].set_title('Segmental SNR')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('debug_analysis.png')
    plt.close()
```

## Contributing

### Adding New Features

1. **Create feature branch**
```bash
git checkout -b feature/new-analyzer
```

2. **Implement with tests**
```python
# Write tests first (TDD)
class TestNewAnalyzer(unittest.TestCase):
    def test_basic_functionality(self):
        analyzer = NewAnalyzer()
        result = analyzer.analyze(test_audio)
        self.assertIsNotNone(result)
```

3. **Update documentation**
```python
class NewAnalyzer:
    """
    Brief description of the analyzer.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Example:
        >>> analyzer = NewAnalyzer()
        >>> result = analyzer.analyze(audio_data)
    """
```

4. **Submit pull request**
```bash
git add .
git commit -m "feat: Add new analyzer for X functionality"
git push origin feature/new-analyzer
```

### Code Style Guidelines

- Follow PEP 8
- Use type hints
- Write comprehensive docstrings
- Include unit tests
- Add integration tests for new components

### Testing Requirements

- Maintain >80% test coverage
- All tests must pass
- Include edge case tests
- Add performance benchmarks for new features

## Additional Resources

- [Project Repository](https://github.com/your-repo/thai-dataset-gathering)
- [API Documentation](./api_reference.md)
- [Architecture Overview](./audio_analysis_architecture.md)
- [Issue Tracker](https://github.com/your-repo/thai-dataset-gathering/issues)

For questions or support, please open an issue on GitHub or contact the development team.