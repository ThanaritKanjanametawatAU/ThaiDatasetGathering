# Audio Analysis Architecture

## Overview

This module provides a modular, plugin-based architecture for autonomous audio processing. It implements the core framework specified in Sprint S01 Task T01 for the Thai Audio Dataset Collection project.

## Key Features

- **Plugin-Based Architecture**: Dynamic loading of audio processors with version compatibility
- **Pipeline Execution**: Stage-based processing with dependency management
- **Factory Pattern**: Easy component instantiation with dependency injection
- **Configuration System**: YAML/JSON configuration with validation
- **Memory Management**: Efficient memory pools and resource monitoring
- **Error Handling**: Graceful degradation and recovery mechanisms

## Core Components

### 1. Abstract Base Classes

- `AudioProcessor`: Base class for all audio processing modules
- `QualityMetric`: Base class for quality assessment metrics
- `DecisionEngine`: Base class for autonomous decision making
- `AudioAnalyzer`: Base class for audio analysis modules

### 2. Plugin System

```python
from processors.audio_analysis import PluginManager

# Initialize plugin manager
manager = PluginManager(
    plugin_paths=['./plugins'],
    auto_discover=True
)

# Get a processor
processor = manager.get_plugin('NoiseReducer')
```

### 3. Pipeline Execution

```python
from processors.audio_analysis import Pipeline

# Create pipeline
pipeline = Pipeline()
pipeline.add_stage('loader', AudioLoader())
pipeline.add_stage('enhancer', NoiseReducer(), dependencies=['loader'])

# Execute
result = pipeline.execute(audio, sample_rate)
```

### 4. Configuration

```yaml
# pipeline_config.yaml
pipeline:
  name: "audio_enhancement"
  version: "1.0.0"
  stages:
    - name: "loader"
      processor: "AudioLoader"
      config:
        formats: ["wav", "mp3"]
    - name: "enhancer"
      processor: "NoiseReducer"
      dependencies: ["loader"]
      config:
        reduction_strength: 0.7
```

## Creating a Custom Processor

```python
from processors.audio_analysis import AudioProcessor, ProcessResult, PluginInfo

class MyProcessor(AudioProcessor):
    def process(self, audio: np.ndarray, sr: int, **kwargs) -> ProcessResult:
        # Your processing logic here
        processed_audio = audio * 0.5  # Example
        
        return ProcessResult(
            audio=processed_audio,
            metadata={"gain_applied": -6.0}
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "processing_type": "gain",
            "real_time_capable": True
        }
    
    def validate_input(self, audio: np.ndarray, sr: int) -> bool:
        return audio.ndim in [1, 2] and sr > 0
    
    @classmethod
    def get_plugin_info(cls) -> PluginInfo:
        return PluginInfo(
            name="MyProcessor",
            version="1.0.0",
            author="Your Name",
            description="Custom gain processor"
        )
```

## Memory Management

```python
from processors.audio_analysis import MemoryPool

# Create memory pool
pool = MemoryPool(max_size="2GB")

# Allocate arrays
audio_buffer = pool.allocate(shape=(44100,), dtype=np.float32)

# Process...

# Release when done
pool.release(audio_buffer)
```

## Testing

```bash
# Run architecture tests
python -m pytest tests/test_audio_analysis_architecture.py -v

# Run specific test
python -m pytest tests/test_audio_analysis_architecture.py::TestAudioProcessorInterface -v
```

## Dependencies

- numpy: For audio array processing
- Optional:
  - PyYAML: For YAML configuration support
  - jsonschema: For configuration validation
  - psutil: For resource monitoring

## Architecture Benefits

1. **Extensibility**: Easy to add new processors without modifying core code
2. **Maintainability**: Clear separation of concerns with well-defined interfaces
3. **Scalability**: Designed for processing 10M+ audio samples
4. **Flexibility**: Configure pipelines via code or configuration files
5. **Performance**: Memory pools and parallel execution support

## Integration with Existing Code

This architecture integrates with the existing codebase through:

- Extends `BaseProcessor` for compatibility
- Uses existing `config.py` for configuration
- Integrates with `utils/logging.py` for logging
- Compatible with `main.py` processing flow

## Next Steps

1. Implement Silero VAD integration (S01_T02)
2. Build SNR calculator module (S01_T03)
3. Develop spectral analysis module (S01_T04)
4. Create pattern detection system (S01_T05)