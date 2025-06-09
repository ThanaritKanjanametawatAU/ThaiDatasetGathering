# Thai Audio Dataset Collection - Architecture Document

## Project Overview

The Thai Audio Dataset Collection is a modular system designed to gather Thai audio data from multiple sources and combine them into a single standardized dataset hosted on Huggingface (Thanarit/Thai-Voice). This system implements state-of-the-art audio processing, speaker identification, and quality enhancement features while maintaining scalability and modularity.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐│
│  │ Command-Line│   │  Dashboard  │   │ Monitoring  │   │ Config   ││
│  │   Parser    │   │     UI      │   │   Tools     │   │  Files   ││
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      Processing Pipeline Layer                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐│
│  │   Dataset   │   │    Audio    │   │   Speaker   │   │   STT    ││
│  │ Processors  │   │ Enhancement │   │     ID      │   │ Engine   ││
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                         Core Services Layer                          │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐│
│  │   Audio     │   │  Streaming  │   │ Checkpoint  │   │HuggingFace│
│  │  Utilities  │   │   Engine    │   │   System    │   │Integration│
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
ThaiDatasetGathering/
├── main.py                            # Main entry point and orchestrator
├── config.py                          # Central configuration management
│
├── processors/                        # Dataset processing modules
│   ├── base_processor.py             # Abstract base class for all processors
│   ├── gigaspeech2.py               # GigaSpeech2 dataset processor
│   ├── processed_voice_th.py        # Processed Voice TH processor
│   ├── mozilla_cv.py                # Mozilla Common Voice processor
│   ├── speaker_identification.py     # Speaker clustering system
│   │
│   ├── audio_enhancement/            # Audio quality enhancement modules
│   │   ├── core.py                  # Core enhancement orchestrator
│   │   ├── detection/               # Audio issue detection
│   │   │   ├── overlap_detector.py # Detect overlapping speech
│   │   │   └── secondary_speaker.py # Detect secondary speakers
│   │   ├── engines/                 # Enhancement engines
│   │   │   ├── denoiser.py         # Facebook Denoiser integration
│   │   │   └── spectral_gating.py  # Spectral noise gating
│   │   ├── speaker_separation.py    # PyAnnote-based separation
│   │   ├── simple_secondary_removal.py # Energy-based removal
│   │   ├── separation.py            # Advanced models (SepFormer/Conv-TasNet)
│   │   ├── post_processing.py       # Post-processing pipeline
│   │   └── evaluation.py            # Quality metrics and evaluation
│   │
│   └── stt/                         # Speech-to-text processors
│       └── ensemble_stt.py          # Ensemble STT with multiple models
│
├── utils/                           # Utility modules
│   ├── audio.py                    # Audio processing utilities
│   ├── audio_metrics.py            # Audio quality metrics (SNR, PESQ, STOI)
│   ├── speaker_utils.py            # Speaker processing utilities
│   ├── streaming.py                # Streaming mode utilities
│   ├── cache.py                    # Cache management
│   ├── huggingface.py              # HuggingFace API integration
│   ├── logging.py                  # Logging utilities
│   ├── noise_profiler.py           # Noise profiling
│   └── snr_measurement.py          # SNR calculation
│
├── monitoring/                      # Real-time monitoring
│   ├── dashboard.py                # Enhancement monitoring dashboard
│   ├── metrics_collector.py        # Metrics collection
│   └── comparison_ui.py            # Before/after comparison UI
│
└── tests/                          # Comprehensive test suite
    ├── test_*.py                   # Unit and integration tests
    └── fixtures/                   # Test data and fixtures
```

## Core Components

### 1. Base Processor Pattern

All dataset processors inherit from `BaseProcessor`, providing:

```python
class BaseProcessor(ABC):
    # Common functionality for all processors
    - Audio preprocessing and standardization
    - Checkpoint management (unified v2.0 format)
    - Streaming and cached mode support
    - Validation and error handling
    - Progress tracking and logging
    
    # Abstract methods for dataset-specific logic
    @abstractmethod
    def process(checkpoint, sample_mode, sample_size)
    @abstractmethod
    def get_dataset_info()
    @abstractmethod
    def estimate_size()
```

### 2. Processing Pipeline

The system implements a sophisticated multi-stage processing pipeline:

```
Input Audio → Validation → Preprocessing → Enhancement → Speaker ID → STT → Output
                    ↓            ↓             ↓            ↓         ↓
                Checkpoint   Checkpoint   Checkpoint   Checkpoint Checkpoint
```

#### Audio Enhancement Pipeline
1. **Detection Phase**
   - Secondary speaker detection (PyAnnote embeddings)
   - Overlap detection
   - Noise profiling

2. **Enhancement Phase**
   - Speaker separation (SpeechBrain models - optional)
   - Noise reduction (Facebook Denoiser)
   - Spectral gating
   - Secondary speaker removal

3. **Post-Processing Phase**
   - Artifact removal
   - Spectral smoothing
   - Level normalization
   - Quality validation

### 3. Data Schema

Standardized schema for all audio samples:

```python
{
    "ID": "S12345",                    # Sequential ID
    "speaker_id": "SPK_00001",         # Speaker identifier
    "Language": "th",                  # Thai language
    "audio": {                         # HuggingFace audio format
        "array": np.array,
        "sampling_rate": 16000,
        "path": "path/to/audio.wav"
    },
    "transcript": "Thai text...",      # Transcript (100% coverage)
    "length": 3.45,                    # Duration in seconds
    "dataset_name": "GigaSpeech2",     # Source dataset
    "confidence_score": 0.95,          # STT confidence (1.0 for original)
    "enhancement_metadata": {          # Optional enhancement data
        "original_snr": 15.2,
        "enhanced_snr": 35.1,
        "enhancement_level": "moderate",
        "processing_time_ms": 234
    }
}
```

## Key Features

### 1. Streaming Mode Architecture

Handles datasets larger than available storage:

```
Dataset Stream → Batch Processing → Shard Creation → Progressive Upload
       ↓               ↓                  ↓                 ↓
   No Storage     Memory Efficient    5GB Shards      Immediate Upload
```

Benefits:
- Process 10TB datasets on 2TB disk
- Resumable with checkpoint support
- Progressive upload during processing

### 2. Speaker Identification System

Adaptive clustering system that separates speakers by dataset:

```
Audio → Embeddings → Clustering → Speaker Assignment
          ↓              ↓              ↓
      PyAnnote      Adaptive Algo   SPK_XXXXX Format
                    (Agglomerative/HDBSCAN)
```

Key features:
- Dataset-separated clustering (prevents cross-dataset merging)
- Adaptive algorithm selection based on batch size
- Persistent speaker models across runs
- Configurable similarity thresholds

### 3. Audio Enhancement Architecture

Multi-engine enhancement system:

```
┌─────────────────────────────────────────────────────────┐
│                Enhancement Orchestrator                  │
├─────────────┬──────────────┬──────────────┬────────────┤
│  Detection  │  Separation  │   Reduction  │    Post    │
│   Engines   │   Engines    │   Engines    │ Processing │
├─────────────┼──────────────┼──────────────┼────────────┤
│ PyAnnote    │ SpeechBrain  │ Denoiser     │ Smoothing  │
│ Energy-based│ PyAnnote     │ Spectral     │ Artifact   │
│ Overlap     │ Simple       │ Wiener       │ Level Norm │
└─────────────┴──────────────┴──────────────┴────────────┘
```

Enhancement levels:
- **Mild**: Light noise reduction, preserve naturalness
- **Moderate**: Balanced enhancement (default)
- **Aggressive**: Strong noise/secondary speaker removal
- **Ultra-aggressive**: Maximum enhancement, all methods

### 4. STT Integration

Ensemble approach for 100% transcript coverage:

```
Missing Transcript → Wav2Vec2 Thai → Confidence Score
                  ↘               ↗
                    Whisper V3 → Select Best Result
```

Fallback hierarchy:
1. Original transcript (confidence = 1.0)
2. Ensemble STT result (confidence < 1.0)
3. "[INAUDIBLE]" marker
4. "[STT_ERROR]" for failures

## Design Patterns

### 1. Factory Pattern
Dynamic processor creation based on dataset configuration:

```python
processor = ProcessorFactory.create(dataset_name, config)
```

### 2. Strategy Pattern
Each processor implements dataset-specific strategies while sharing common interface.

### 3. Observer Pattern
Real-time monitoring dashboard observes processing pipeline:

```python
metrics_collector.observe(enhancement_event)
dashboard.update(metrics)
```

### 4. Chain of Responsibility
Enhancement modules form a processing chain, each handling specific tasks.

## Performance Considerations

### Memory Management
- Streaming-first approach (never load full dataset)
- Micro-batches for processing (16-32 samples)
- GPU cache clearing after batches
- Automatic fallback to CPU

### Scalability
- Horizontal scaling via dataset parallelization
- Checkpoint system for fault tolerance
- Progressive upload for network efficiency
- Configurable batch sizes

### Optimization Targets
- Processing speed: ~30 samples/minute
- GPU utilization: ~95%
- Memory usage: <100GB storage, <100MB RAM/sample
- VRAM usage: ~8GB for all models

## Recent Enhancements (2025)

### Secondary Speaker Removal (February 2025)
- Test-driven development implementation
- Advanced separation models (SepFormer/Conv-TasNet)
- Quality-based exclusion logic
- Comprehensive evaluation dashboard

### Code Quality Improvements (January 2025)
- Major refactoring following Power of 10 rules
- Removed ~600 lines of duplicate code
- Fixed ~100 linting issues
- Enhanced documentation

### TDD Revert Implementation (January 2025)
- Successfully reverted secondary speaker removal
- Fixed omegaconf compatibility
- Restored speaker clustering accuracy

## Configuration Management

Central configuration in `config.py`:

```python
# Audio Processing
AUDIO_CONFIG = {
    "target_sample_rate": 16000,
    "target_channels": 1,
    "normalize_volume": True,
    "target_db": -20.0
}

# Enhancement Levels
ENHANCEMENT_CONFIG = {
    "mild": {...},
    "moderate": {...},
    "aggressive": {...},
    "ultra_aggressive": {...}
}

# Speaker Clustering
SPEAKER_ID_CONFIG = {
    "algorithm": "adaptive",
    "similarity_threshold": 0.7,
    "min_cluster_size": 5
}
```

## Testing Strategy

Comprehensive test coverage:

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Pipeline testing
3. **End-to-End Tests**: Full workflow validation
4. **Performance Tests**: Speed and resource usage
5. **Quality Tests**: Audio enhancement validation

Test command:
```bash
python main.py --fresh --all --sample --sample-size 5 --enable-speaker-id --enable-stt --streaming
```

## Deployment and Usage

### Basic Usage
```bash
# Process all datasets
python main.py --fresh --all --streaming

# Test with samples
python main.py --fresh GigaSpeech2 --sample --sample-size 10

# Enable all features
python main.py --fresh --all --enable-speaker-id --enable-stt --enable-audio-enhancement
```

### Production Deployment
1. Set up HuggingFace token in `hf_token.txt`
2. Configure enhancement levels in `main.sh`
3. Run with appropriate batch sizes for available resources
4. Monitor using dashboard (`--enhancement-dashboard`)

## Future Roadmap

1. **Advanced Features**
   - Real-time processing capabilities
   - Multi-language support
   - Advanced diarization
   - Quality-aware sampling

2. **Performance Improvements**
   - Distributed processing
   - Model quantization
   - Streaming upload optimization

3. **Integration Enhancements**
   - API endpoints
   - Cloud deployment
   - Automated quality monitoring

## Conclusion

The Thai Audio Dataset Collection system represents a sophisticated, modular architecture designed for scalability, quality, and extensibility. Its test-driven development approach, comprehensive feature set, and robust error handling make it suitable for production-scale audio dataset creation and enhancement.