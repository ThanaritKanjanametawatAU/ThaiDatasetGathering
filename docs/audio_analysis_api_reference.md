# Audio Analysis Pipeline API Reference

## Table of Contents

- [Audio Loader](#audio-loader)
- [SNR Calculator](#snr-calculator)
- [Spectral Analyzer](#spectral-analyzer)
- [Pattern Detector](#pattern-detector)
- [Issue Categorizer](#issue-categorizer)
- [Decision Framework](#decision-framework)
- [Data Types](#data-types)
- [Exceptions](#exceptions)
- [Configuration](#configuration)

---

## Audio Loader

### Class: `AudioLoader`

Handles audio file loading, validation, and preprocessing.

```python
from processors.audio_enhancement.audio_loader import AudioLoader
```

#### Constructor

```python
AudioLoader(
    target_sample_rate: int = 16000,
    cache: Optional[AudioCache] = None,
    validate_on_load: bool = True
)
```

**Parameters:**
- `target_sample_rate` (int): Target sample rate for resampling. Default: 16000
- `cache` (AudioCache, optional): Cache instance for storing loaded audio
- `validate_on_load` (bool): Whether to validate audio on load. Default: True

#### Methods

##### `load(file_path: str) -> AudioData`

Load audio file and return processed audio data.

**Parameters:**
- `file_path` (str): Path to the audio file

**Returns:**
- `AudioData`: Loaded and preprocessed audio data

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `UnsupportedFormatError`: If audio format is not supported
- `CorruptedFileError`: If file is corrupted
- `AudioLoadError`: For other loading errors

**Example:**
```python
loader = AudioLoader()
audio_data = loader.load("speech.wav")
print(f"Loaded {len(audio_data.samples)} samples at {audio_data.sample_rate}Hz")
```

##### `validate(audio_data: AudioData) -> bool`

Validate audio data for processing.

**Parameters:**
- `audio_data` (AudioData): Audio data to validate

**Returns:**
- `bool`: True if valid, False otherwise

##### `preprocess(audio_data: AudioData) -> AudioData`

Preprocess audio data (resampling, normalization, etc.).

**Parameters:**
- `audio_data` (AudioData): Audio data to preprocess

**Returns:**
- `AudioData`: Preprocessed audio data

### Class: `AudioCache`

LRU cache for audio files.

```python
AudioCache(max_size: int = 100)
```

**Parameters:**
- `max_size` (int): Maximum number of files to cache. Default: 100

#### Methods

##### `get(key: str) -> Optional[AudioData]`

Get audio data from cache.

##### `set(key: str, value: AudioData) -> None`

Store audio data in cache.

##### `clear() -> None`

Clear all cached entries.

---

## SNR Calculator

### Class: `EnhancedSNRCalculator`

Calculate Signal-to-Noise Ratio metrics.

```python
from utils.enhanced_snr_calculator import EnhancedSNRCalculator
```

#### Constructor

```python
EnhancedSNRCalculator(
    frame_length: float = 0.025,
    frame_shift: float = 0.010,
    min_speech_duration: float = 0.3,
    noise_estimation_method: str = 'mmse'
)
```

**Parameters:**
- `frame_length` (float): Frame length in seconds. Default: 0.025
- `frame_shift` (float): Frame shift in seconds. Default: 0.010
- `min_speech_duration` (float): Minimum speech duration. Default: 0.3
- `noise_estimation_method` (str): Method for noise estimation. Options: 'mmse', 'percentile', 'minimum_statistics'. Default: 'mmse'

#### Methods

##### `calculate(audio: np.ndarray, sample_rate: int) -> SNRMetrics`

Calculate comprehensive SNR metrics.

**Parameters:**
- `audio` (np.ndarray): Audio samples
- `sample_rate` (int): Sample rate in Hz

**Returns:**
- `SNRMetrics`: Calculated SNR metrics

**Raises:**
- `InsufficientDataError`: If audio is too short
- `NoSpeechDetectedError`: If no speech is detected
- `InvalidSignalError`: If signal contains invalid values
- `SNRError`: For other SNR calculation errors

**Example:**
```python
calculator = EnhancedSNRCalculator()
metrics = calculator.calculate(audio_samples, 16000)
print(f"Global SNR: {metrics.global_snr:.2f} dB")
print(f"Average Segmental SNR: {np.mean(metrics.segmental_snr):.2f} dB")
```

##### `calculate_segmental(audio: np.ndarray, frame_length: float) -> List[float]`

Calculate frame-by-frame SNR values.

**Parameters:**
- `audio` (np.ndarray): Audio samples
- `frame_length` (float): Frame length in seconds

**Returns:**
- `List[float]`: SNR values for each frame

##### `estimate_noise(audio: np.ndarray, sample_rate: int) -> NoiseProfile`

Estimate noise characteristics.

**Parameters:**
- `audio` (np.ndarray): Audio samples
- `sample_rate` (int): Sample rate

**Returns:**
- `NoiseProfile`: Estimated noise profile

### Class: `NoiseEstimator`

Advanced noise estimation algorithms.

#### Methods

##### `estimate_mmse(audio: np.ndarray) -> np.ndarray`

Minimum Mean Square Error noise estimation.

##### `estimate_percentile(audio: np.ndarray, percentile: float = 5.0) -> float`

Percentile-based noise estimation.

##### `estimate_minimum_statistics(audio: np.ndarray) -> np.ndarray`

Minimum statistics noise estimation.

---

## Spectral Analyzer

### Class: `SpectralAnalyzer`

Perform spectral analysis and feature extraction.

```python
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer
```

#### Constructor

```python
SpectralAnalyzer(
    fft_size: int = 2048,
    hop_size: int = 512,
    window: str = 'hann',
    n_mels: int = 128
)
```

**Parameters:**
- `fft_size` (int): FFT size. Default: 2048
- `hop_size` (int): Hop size for STFT. Default: 512
- `window` (str): Window function. Default: 'hann'
- `n_mels` (int): Number of mel bands. Default: 128

#### Methods

##### `analyze(audio: np.ndarray, sample_rate: int) -> SpectralFeatures`

Perform complete spectral analysis.

**Parameters:**
- `audio` (np.ndarray): Audio samples
- `sample_rate` (int): Sample rate

**Returns:**
- `SpectralFeatures`: Extracted spectral features

**Example:**
```python
analyzer = SpectralAnalyzer()
features = analyzer.analyze(audio, 16000)
print(f"Spectral Centroid: {features.spectral_centroid:.2f} Hz")
print(f"Spectral Rolloff: {features.spectral_rolloff:.2f} Hz")
```

##### `extract_features(spectrum: np.ndarray, frequencies: np.ndarray) -> Dict[str, float]`

Extract spectral features from magnitude spectrum.

**Parameters:**
- `spectrum` (np.ndarray): Magnitude spectrum
- `frequencies` (np.ndarray): Frequency bins

**Returns:**
- `Dict[str, float]`: Dictionary of features

##### `compute_spectrogram(audio: np.ndarray, sample_rate: int) -> np.ndarray`

Compute STFT spectrogram.

**Parameters:**
- `audio` (np.ndarray): Audio samples
- `sample_rate` (int): Sample rate

**Returns:**
- `np.ndarray`: Spectrogram (time x frequency)

### Class: `SpectralFeatureExtractor`

Extract specific spectral features.

#### Methods

##### `spectral_centroid(magnitude: np.ndarray, frequencies: np.ndarray) -> float`

Calculate spectral centroid.

##### `spectral_rolloff(magnitude: np.ndarray, frequencies: np.ndarray, threshold: float = 0.85) -> float`

Calculate spectral rolloff frequency.

##### `spectral_flux(spectrogram: np.ndarray) -> np.ndarray`

Calculate spectral flux over time.

##### `zero_crossing_rate(audio: np.ndarray) -> float`

Calculate zero crossing rate.

---

## Pattern Detector

### Class: `PatternDetector`

Detect patterns and anomalies in audio.

```python
from processors.audio_enhancement.detection.pattern_detector import PatternDetector
```

#### Constructor

```python
PatternDetector(
    clipping_threshold: float = 0.99,
    silence_threshold: float = -60.0,
    anomaly_method: str = 'isolation_forest'
)
```

**Parameters:**
- `clipping_threshold` (float): Threshold for clipping detection. Default: 0.99
- `silence_threshold` (float): Threshold for silence detection in dB. Default: -60.0
- `anomaly_method` (str): Anomaly detection method. Default: 'isolation_forest'

#### Methods

##### `detect(audio: np.ndarray, spectral_features: Optional[SpectralFeatures] = None) -> List[Pattern]`

Detect all patterns in audio.

**Parameters:**
- `audio` (np.ndarray): Audio samples
- `spectral_features` (SpectralFeatures, optional): Pre-computed spectral features

**Returns:**
- `List[Pattern]`: Detected patterns

**Example:**
```python
detector = PatternDetector()
patterns = detector.detect(audio_samples, spectral_features)
for pattern in patterns:
    print(f"{pattern.pattern_type} at {pattern.location.start:.2f}s "
          f"(confidence: {pattern.confidence:.2f})")
```

##### `detect_clipping(audio: np.ndarray, threshold: float = None) -> List[ClippingEvent]`

Detect clipping events.

**Parameters:**
- `audio` (np.ndarray): Audio samples
- `threshold` (float, optional): Custom threshold

**Returns:**
- `List[ClippingEvent]`: Detected clipping events

##### `detect_silence(audio: np.ndarray, threshold_db: float = None) -> List[SilenceSegment]`

Detect silence segments.

**Parameters:**
- `audio` (np.ndarray): Audio samples
- `threshold_db` (float, optional): Custom threshold in dB

**Returns:**
- `List[SilenceSegment]`: Detected silence segments

### Class: `AnomalyDetector`

Detect anomalies using machine learning.

#### Methods

##### `detect(features: np.ndarray) -> List[Anomaly]`

Detect anomalies in feature space.

##### `train(normal_features: np.ndarray) -> None`

Train anomaly detector on normal data.

---

## Issue Categorizer

### Class: `IssueCategorizer`

Categorize audio quality issues.

```python
from processors.audio_enhancement.issue_categorization import IssueCategorizer
```

#### Constructor

```python
IssueCategorizer(
    severity_thresholds: Dict[str, float] = None,
    confidence_threshold: float = 0.5
)
```

**Parameters:**
- `severity_thresholds` (Dict[str, float], optional): Custom severity thresholds
- `confidence_threshold` (float): Minimum confidence threshold. Default: 0.5

#### Methods

##### `categorize(snr_metrics: SNRMetrics, spectral_features: SpectralFeatures, patterns: List[Pattern]) -> List[Issue]`

Categorize issues based on analysis results.

**Parameters:**
- `snr_metrics` (SNRMetrics): SNR analysis results
- `spectral_features` (SpectralFeatures): Spectral analysis results
- `patterns` (List[Pattern]): Detected patterns

**Returns:**
- `List[Issue]`: Categorized issues

**Example:**
```python
categorizer = IssueCategorizer()
issues = categorizer.categorize(snr_metrics, spectral_features, patterns)

for issue in issues:
    print(f"{issue.type.value}: {issue.severity.value} severity "
          f"(confidence: {issue.confidence:.2f})")
    print(f"  Description: {issue.description}")
```

##### `assess_severity(issue_type: IssueType, metrics: Dict[str, float]) -> IssueSeverity`

Assess severity of an issue.

**Parameters:**
- `issue_type` (IssueType): Type of issue
- `metrics` (Dict[str, float]): Relevant metrics

**Returns:**
- `IssueSeverity`: Assessed severity level

##### `rank_issues(issues: List[Issue]) -> List[Issue]`

Rank issues by priority.

**Parameters:**
- `issues` (List[Issue]): Issues to rank

**Returns:**
- `List[Issue]`: Ranked issues (highest priority first)

### Class: `CategoryRule`

Define custom categorization rules.

```python
CategoryRule(
    name: str,
    condition: Callable,
    issue_type: IssueType,
    severity: IssueSeverity,
    confidence: float,
    description: str
)
```

---

## Decision Framework

### Class: `DecisionEngine`

Make decisions based on audio analysis.

```python
from processors.audio_enhancement.decision_framework import DecisionEngine
```

#### Constructor

```python
DecisionEngine(
    enhancement_threshold: float = 20.0,
    rejection_threshold: float = 5.0,
    confidence_threshold: float = 0.7
)
```

**Parameters:**
- `enhancement_threshold` (float): SNR threshold for enhancement. Default: 20.0
- `rejection_threshold` (float): SNR threshold for rejection. Default: 5.0
- `confidence_threshold` (float): Minimum confidence for decisions. Default: 0.7

#### Methods

##### `decide(issues: List[Issue], context: Optional[DecisionContext] = None) -> Decision`

Make decision based on issues and context.

**Parameters:**
- `issues` (List[Issue]): Detected issues
- `context` (DecisionContext, optional): Additional context

**Returns:**
- `Decision`: Processing decision

**Example:**
```python
engine = DecisionEngine()
decision = engine.decide(issues, context)

print(f"Decision: {decision.action}")
print(f"Confidence: {decision.confidence:.2f}")
print("Reasoning:")
for reason in decision.reasoning:
    print(f"  - {reason}")
```

##### `explain_decision(decision: Decision) -> List[str]`

Generate detailed explanation for decision.

**Parameters:**
- `decision` (Decision): Decision to explain

**Returns:**
- `List[str]`: Explanation points

##### `add_rule(rule: DecisionRule) -> None`

Add custom decision rule.

**Parameters:**
- `rule` (DecisionRule): Custom rule to add

### Class: `DecisionContext`

Context information for decision making.

```python
DecisionContext(
    audio_metrics: Optional[SNRMetrics] = None,
    spectral_features: Optional[SpectralFeatures] = None,
    detected_patterns: Optional[List[Pattern]] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

### Class: `DecisionRule`

Custom decision rule.

```python
DecisionRule(
    name: str,
    condition: Callable[[List[Issue], DecisionContext], bool],
    action: str,
    confidence: float,
    reasoning: str
)
```

---

## Data Types

### `AudioData`

Container for audio data.

```python
@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `SNRMetrics`

SNR calculation results.

```python
@dataclass
class SNRMetrics:
    global_snr: float
    segmental_snr: List[float]
    noise_profile: NoiseProfile
    vad_segments: List[VADSegment]
    is_silent: bool = False
    has_speech: bool = True
```

### `SpectralFeatures`

Spectral analysis results.

```python
@dataclass
class SpectralFeatures:
    frequency_bins: np.ndarray
    magnitude_spectrum: np.ndarray
    phase_spectrum: np.ndarray
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flux: float
    zero_crossing_rate: float
    spectral_bandwidth: float
    mfcc: np.ndarray
```

### `Pattern`

Detected pattern information.

```python
@dataclass
class Pattern:
    pattern_type: str
    location: TimeRange
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `Issue`

Categorized issue.

```python
@dataclass
class Issue:
    type: IssueType
    severity: IssueSeverity
    confidence: float
    description: str
    affected_range: Optional[TimeRange] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `Decision`

Processing decision.

```python
@dataclass
class Decision:
    action: str  # 'PROCESS', 'ENHANCE', 'REJECT', 'MANUAL_REVIEW'
    confidence: float
    reasoning: List[str]
    recommendations: List[str] = field(default_factory=list)
```

### Enums

#### `IssueType`

```python
class IssueType(Enum):
    NOISE = "noise"
    DISTORTION = "distortion"
    ECHO = "echo"
    BANDWIDTH = "bandwidth"
    SILENCE = "silence"
    UNKNOWN = "unknown"
```

#### `IssueSeverity`

```python
class IssueSeverity(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
```

#### `DecisionType`

```python
class DecisionType(Enum):
    PROCESS = "PROCESS"
    ENHANCE = "ENHANCE"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"
```

---

## Exceptions

### Audio Loading Exceptions

```python
class AudioLoadError(Exception):
    """Base exception for audio loading errors."""

class FileNotFoundError(AudioLoadError):
    """Audio file not found."""

class UnsupportedFormatError(AudioLoadError):
    """Audio format not supported."""

class CorruptedFileError(AudioLoadError):
    """Audio file is corrupted."""

class PreprocessingError(AudioLoadError):
    """Error during preprocessing."""
```

### SNR Calculation Exceptions

```python
class SNRError(Exception):
    """Base exception for SNR calculation errors."""

class InsufficientDataError(SNRError):
    """Not enough audio data for analysis."""

class NoSpeechDetectedError(SNRError):
    """No speech detected in audio."""

class InvalidSignalError(SNRError):
    """Signal contains invalid values (NaN, Inf)."""
```

### Analysis Exceptions

```python
class AnalysisError(Exception):
    """Base exception for analysis errors."""

class SpectralAnalysisError(AnalysisError):
    """Error in spectral analysis."""

class PatternDetectionError(AnalysisError):
    """Error in pattern detection."""

class CategorizationError(AnalysisError):
    """Error in issue categorization."""
```

---

## Configuration

### Configuration Schema

```yaml
audio_analysis:
  # Audio Loader Configuration
  audio_loader:
    target_sample_rate: 16000
    supported_formats: ['wav', 'mp3', 'flac', 'ogg']
    max_duration: 3600  # seconds
    cache_size: 100
    validate_on_load: true
    
  # SNR Calculator Configuration
  snr_calculator:
    frame_length: 0.025  # seconds
    frame_shift: 0.010   # seconds
    min_speech_duration: 0.3  # seconds
    noise_estimation_method: 'mmse'  # 'mmse', 'percentile', 'minimum_statistics'
    vad_threshold: 0.1
    
  # Spectral Analyzer Configuration
  spectral_analyzer:
    fft_size: 2048
    hop_size: 512
    window: 'hann'  # 'hann', 'hamming', 'blackman'
    n_mels: 128
    n_mfcc: 13
    
  # Pattern Detector Configuration
  pattern_detector:
    clipping_threshold: 0.99
    silence_threshold: -60  # dB
    dropout_min_duration: 0.01  # seconds
    anomaly_detection_method: 'isolation_forest'
    anomaly_contamination: 0.1
    
  # Issue Categorizer Configuration
  issue_categorizer:
    severity_thresholds:
      minor: 0.3
      moderate: 0.6
      severe: 0.8
    confidence_threshold: 0.5
    min_issue_duration: 0.1  # seconds
    
  # Decision Engine Configuration
  decision_engine:
    enhancement_threshold: 20  # dB
    rejection_threshold: 5     # dB
    confidence_threshold: 0.7
    enable_adaptive_learning: false
    decision_weights:
      snr: 0.4
      issues: 0.3
      patterns: 0.2
      spectral: 0.1
```

### Loading Configuration

```python
import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Usage
config = load_config('config/audio_analysis.yaml')
loader = AudioLoader(**config['audio_analysis']['audio_loader'])
```

### Environment Variables

Override configuration with environment variables:

```bash
# Override sample rate
export AUDIO_ANALYSIS_SAMPLE_RATE=48000

# Override enhancement threshold
export AUDIO_ANALYSIS_ENHANCEMENT_THRESHOLD=25

# Enable debug mode
export AUDIO_ANALYSIS_DEBUG=true
```

---

## Usage Examples

### Complete Pipeline Example

```python
from processors.audio_enhancement.audio_loader import AudioLoader
from utils.enhanced_snr_calculator import EnhancedSNRCalculator
from processors.audio_enhancement.spectral_analysis import SpectralAnalyzer
from processors.audio_enhancement.detection.pattern_detector import PatternDetector
from processors.audio_enhancement.issue_categorization import IssueCategorizer
from processors.audio_enhancement.decision_framework import DecisionEngine, DecisionContext

# Initialize components
loader = AudioLoader()
snr_calculator = EnhancedSNRCalculator()
spectral_analyzer = SpectralAnalyzer()
pattern_detector = PatternDetector()
issue_categorizer = IssueCategorizer()
decision_engine = DecisionEngine()

# Process audio file
audio_data = loader.load("speech.wav")
snr_metrics = snr_calculator.calculate(audio_data.samples, audio_data.sample_rate)
spectral_features = spectral_analyzer.analyze(audio_data.samples, audio_data.sample_rate)
patterns = pattern_detector.detect(audio_data.samples, spectral_features)
issues = issue_categorizer.categorize(snr_metrics, spectral_features, patterns)

# Make decision with context
context = DecisionContext(
    audio_metrics=snr_metrics,
    spectral_features=spectral_features,
    detected_patterns=patterns
)
decision = decision_engine.decide(issues, context)

# Print results
print(f"SNR: {snr_metrics.global_snr:.2f} dB")
print(f"Issues: {len(issues)}")
print(f"Decision: {decision.action} (confidence: {decision.confidence:.2f})")
```

### Error Handling Example

```python
from processors.audio_enhancement.audio_loader import (
    AudioLoader, AudioLoadError, UnsupportedFormatError
)

loader = AudioLoader()

try:
    audio_data = loader.load("audio.mp3")
except UnsupportedFormatError as e:
    print(f"Format not supported: {e}")
    # Try converting or use alternative
except AudioLoadError as e:
    print(f"Failed to load audio: {e}")
    # Handle error appropriately
```

### Custom Configuration Example

```python
# Create custom configuration
custom_config = {
    'frame_length': 0.030,  # 30ms frames
    'noise_estimation_method': 'percentile',
    'min_speech_duration': 0.5
}

# Initialize with custom config
snr_calculator = EnhancedSNRCalculator(**custom_config)
```

---

## Version Information

- **API Version**: 1.0.0
- **Last Updated**: 2025-06-09
- **Python Compatibility**: 3.7+
- **Dependencies**: numpy, scipy, librosa, soundfile, sklearn