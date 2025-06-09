# Task: Implement Basic Audio Loader and Preprocessor

## Task ID
S01_T02

## Description
Implement a robust audio loading and preprocessing module that handles various audio formats, sample rates, and encoding types. This module will serve as the entry point for all audio data into the analysis pipeline, ensuring consistent format and quality before processing.

## Status
**Status**: 🔴 Not Started  
**Assigned To**: Unassigned  
**Created**: 2025-06-09  
**Updated**: 2025-06-09

## Technical Requirements

### Core Components
1. **Audio Loader**
   - Support for multiple formats: WAV, MP3, FLAC, OGG
   - Automatic format detection
   - Error handling for corrupted files
   - Memory-efficient loading for large files

2. **Preprocessor**
   - Sample rate conversion (target: 16kHz for speech)
   - Channel normalization (mono conversion)
   - Bit depth standardization (16-bit or float32)
   - Audio trimming and padding

3. **Validation System**
   ```python
   class AudioValidator:
       def validate_duration(self, audio: np.ndarray, sr: int) -> bool
       def validate_amplitude(self, audio: np.ndarray) -> bool
       def validate_format(self, filepath: str) -> bool
       def check_corruption(self, audio: np.ndarray) -> bool
   ```

4. **Caching Mechanism**
   - LRU cache for frequently accessed files
   - Preprocessed audio caching
   - Cache invalidation on parameter changes

### Implementation Steps
1. Implement format-agnostic audio loader using librosa/soundfile
2. Create preprocessing pipeline with configurable stages
3. Add validation checks at each stage
4. Implement streaming loader for large files
5. Create caching layer with configurable size
6. Add metadata extraction (duration, channels, sample rate)
7. Implement batch loading capabilities
8. Add progress tracking for long operations

## Test Requirements (TDD)

### Test First Approach
1. **Format Support Tests**
   ```python
   def test_load_wav_file():
       # Test loading standard WAV files
       # Test various sample rates and bit depths
       # Test mono and stereo files
   
   def test_load_compressed_formats():
       # Test MP3, FLAC, OGG loading
       # Test quality preservation
       # Test metadata extraction
   ```

2. **Preprocessing Tests**
   ```python
   def test_sample_rate_conversion():
       # Test upsampling and downsampling
       # Test quality metrics after conversion
       # Test edge cases (very low/high rates)
   
   def test_channel_normalization():
       # Test stereo to mono conversion
       # Test multi-channel handling
       # Test silent channel detection
   ```

3. **Validation Tests**
   ```python
   def test_corruption_detection():
       # Test detection of corrupted files
       # Test partial file handling
       # Test invalid format rejection
   
   def test_duration_validation():
       # Test minimum/maximum duration checks
       # Test silence detection
       # Test clipping detection
   ```

4. **Performance Tests**
   ```python
   def test_large_file_handling():
       # Test streaming for files > 1GB
       # Test memory usage limits
       # Test processing speed benchmarks
   ```

## Acceptance Criteria
- [ ] Supports WAV, MP3, FLAC, OGG formats with automatic detection
- [ ] Converts all audio to consistent format (16kHz, mono, float32)
- [ ] Validates audio files for corruption and quality issues
- [ ] Implements caching with < 100ms retrieval for cached files
- [ ] Handles files up to 2GB without memory issues
- [ ] Achieves 95% test coverage
- [ ] Documentation includes usage examples and API reference

## Dependencies
- librosa >= 0.10.0 for audio loading
- soundfile for additional format support
- numpy for array operations
- pydub for format conversion fallback
- S01_T01 (Core architecture must be complete)

## Estimated Effort
**Duration**: 1-2 days  
**Complexity**: Medium

## Detailed Algorithm Specifications

### Audio Loading Algorithm
```
1. File Format Detection:
   a. Read file header (first 512 bytes)
   b. Match magic numbers for format identification
   c. Fallback to extension-based detection
   d. Validate format support

2. Efficient Loading Strategy:
   a. For small files (<100MB):
      - Load entire file to memory
      - Decode in single pass
   b. For large files (>100MB):
      - Use memory-mapped I/O
      - Decode in chunks
      - Stream processing pipeline

3. Multi-format Decoding:
   a. WAV: Direct numpy array conversion
   b. MP3/OGG: FFmpeg backend decoding
   c. FLAC: libflac streaming decoder
   d. Fallback: soundfile universal loader
```

### Preprocessing Pipeline Algorithm
```
1. Sample Rate Conversion:
   a. Calculate resampling ratio: r = target_sr / original_sr
   b. Choose resampling method:
      - r > 1: Polyphase filtering (upsampling)
      - r < 1: Anti-aliasing + decimation (downsampling)
   c. Apply Kaiser window (β = 14.769656459)
   d. Resample using scipy.signal.resample_poly

2. Channel Normalization:
   a. Detect channel configuration
   b. For stereo to mono:
      - Method 1: Average channels: mono = (L + R) / 2
      - Method 2: Weighted: mono = 0.5 * L + 0.5 * R
      - Method 3: Max energy: mono = max(L, R)
   c. Preserve phase relationships

3. Amplitude Normalization:
   a. Calculate signal statistics
   b. Apply normalization:
      - Peak normalization: x_norm = x / max(|x|)
      - RMS normalization: x_norm = x * (target_rms / current_rms)
      - LUFS normalization for perceptual loudness
```

### Mathematical Formulations
- **Resampling Quality**: SNR_resample = 20 * log10(signal_power / aliasing_power)
- **Anti-aliasing Filter**: H(f) = sinc(2πf_c) * kaiser_window(β)
- **RMS Calculation**: RMS = sqrt(mean(x²))
- **LUFS Loudness**: LUFS = -0.691 + 10 * log10(mean(weighted_power))

## Integration with Existing Codebase

### Files to Interface With
1. **utils/audio.py**
   - Reuse `load_audio()` function
   - Extend with new format support
   - Add streaming capabilities

2. **processors/base_processor.py**
   - Inherit preprocessing interface
   - Implement `_process_audio()` method
   - Add format validation

3. **config.py**
   - Read AUDIO_SAMPLE_RATE
   - Use AUDIO_CHANNELS setting
   - Apply MAX_AUDIO_LENGTH

4. **utils/cache.py**
   - Integrate with existing cache
   - Add audio-specific caching
   - Implement cache warming

### Integration Points
```python
# Extend existing audio utilities
from utils.audio import load_audio as base_load_audio

class EnhancedAudioLoader:
    def __init__(self):
        self.cache = AudioCache()
        self.validator = AudioValidator()
    
    def load_audio(self, path: str, **kwargs):
        # Check cache first
        if cached := self.cache.get(path):
            return cached
        
        # Validate before loading
        if not self.validator.validate_format(path):
            raise UnsupportedFormatError(path)
        
        # Load with enhanced features
        audio, sr = self._load_with_fallback(path)
        
        # Cache result
        self.cache.set(path, (audio, sr))
        return audio, sr
```

## Configuration Examples

### Audio Loader Configuration (audio_config.yaml)
```yaml
audio_loader:
  supported_formats:
    - ext: "wav"
      priority: 1
      decoder: "scipy"
      options:
        dtype: "float32"
    - ext: "mp3"
      priority: 2
      decoder: "librosa"
      options:
        res_type: "kaiser_best"
    - ext: "flac"
      priority: 1
      decoder: "soundfile"
      options:
        always_2d: false
  
  preprocessing:
    target_sample_rate: 16000
    target_channels: 1
    target_bit_depth: 32
    normalize_method: "peak"
    trim_silence: true
    silence_threshold: -40  # dB
    
  validation:
    min_duration: 0.1  # seconds
    max_duration: 3600  # seconds
    max_file_size: 2048  # MB
    check_corruption: true
    
  cache:
    enabled: true
    max_size: "4GB"
    ttl: 3600  # seconds
    backend: "lru"  # or "redis"
```

### Batch Processing Configuration (batch_config.json)
```json
{
  "batch_processor": {
    "chunk_size": 1000,
    "parallel_workers": 8,
    "memory_limit": "16GB",
    "progress_reporting": {
      "enabled": true,
      "interval": 100,
      "webhook": "http://localhost:5000/progress"
    },
    "error_handling": {
      "max_retries": 3,
      "skip_corrupted": true,
      "log_errors": true,
      "error_dir": "./errors/"
    }
  }
}
```

## Error Handling Strategy

### Exception Types
```python
class AudioLoadError(Exception):
    """Base exception for audio loading errors"""
    pass

class UnsupportedFormatError(AudioLoadError):
    """File format not supported"""
    def __init__(self, filepath, detected_format=None):
        self.filepath = filepath
        self.format = detected_format
        super().__init__(f"Unsupported format: {detected_format}")

class CorruptedFileError(AudioLoadError):
    """Audio file is corrupted"""
    pass

class PreprocessingError(AudioLoadError):
    """Error during preprocessing"""
    pass
```

### Recovery Strategies
1. **Format Fallback Chain**
   ```python
   loaders = [librosa_load, soundfile_load, audioread_load, ffmpeg_load]
   for loader in loaders:
       try:
           return loader(filepath)
       except Exception as e:
           last_error = e
   raise AudioLoadError(f"All loaders failed: {last_error}")
   ```

2. **Corruption Handling**
   - Attempt partial file recovery
   - Extract readable segments
   - Log corruption details
   - Quarantine corrupted files

### Validation Pipeline
```python
def validate_audio_file(filepath: Path) -> ValidationResult:
    checks = [
        check_file_exists,
        check_file_size,
        check_format_header,
        check_audio_integrity,
        check_duration_limits
    ]
    
    for check in checks:
        result = check(filepath)
        if not result.passed:
            return result
    
    return ValidationResult(passed=True)
```

## Performance Optimization

### Caching Architecture
```python
class HierarchicalAudioCache:
    def __init__(self):
        self.l1_cache = LRUCache(maxsize=100)  # Hot cache
        self.l2_cache = DiskCache(max_size="4GB")  # Warm cache
        self.l3_cache = RedisCache()  # Distributed cache
    
    def get(self, key: str) -> Optional[Audio]:
        # Check caches in order
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            if value := cache.get(key):
                # Promote to higher cache
                self._promote(key, value)
                return value
        return None
```

### Parallel Loading Strategy
```python
class ParallelAudioLoader:
    def __init__(self, num_workers=8):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.semaphore = Semaphore(num_workers * 2)  # Prevent overload
    
    def load_batch(self, filepaths: List[str]) -> List[Audio]:
        futures = []
        for filepath in filepaths:
            future = self.executor.submit(self._load_single, filepath)
            futures.append(future)
        
        return [f.result() for f in as_completed(futures)]
```

### Memory Management
```python
# Memory-efficient streaming loader
class StreamingAudioLoader:
    def __init__(self, chunk_size=1024*1024):  # 1MB chunks
        self.chunk_size = chunk_size
    
    def load_large_file(self, filepath: str):
        with soundfile.SoundFile(filepath) as f:
            for chunk in f.blocks(self.chunk_size):
                yield self.preprocess_chunk(chunk)
```

## Production Considerations

### Monitoring Integration
```python
# Prometheus metrics
audio_load_duration = Histogram(
    'audio_load_seconds',
    'Time to load audio file',
    ['format', 'size_category']
)

preprocessing_errors = Counter(
    'preprocessing_errors_total',
    'Preprocessing error count',
    ['error_type', 'stage']
)

cache_metrics = {
    'hits': Counter('audio_cache_hits_total', 'Cache hit count'),
    'misses': Counter('audio_cache_misses_total', 'Cache miss count'),
    'size': Gauge('audio_cache_size_bytes', 'Current cache size')
}
```

### Health Checks
```python
class AudioLoaderHealthCheck:
    def check_format_support(self) -> bool:
        """Test all format decoders"""
        test_files = {
            'wav': 'test.wav',
            'mp3': 'test.mp3',
            'flac': 'test.flac'
        }
        for fmt, file in test_files.items():
            if not self._test_load(file):
                return False
        return True
    
    def check_cache_availability(self) -> bool:
        """Verify cache backend connectivity"""
        return self.cache.ping()
```

### Deployment Configuration
```yaml
# kubernetes configmap
apiVersion: v1
kind: ConfigMap
metadata:
  name: audio-loader-config
data:
  audio.yaml: |
    loader:
      timeout: 30s
      max_file_size: 2GB
      temp_dir: /tmp/audio
    cache:
      redis_url: redis://redis-service:6379
      ttl: 1h
```

## Troubleshooting Guide

### Common Issues

1. **Slow Loading Performance**
   - **Symptom**: Files take >5s to load
   - **Diagnosis**: Check cache hit rate
   - **Solution**: 
     ```bash
     # Increase cache size
     export AUDIO_CACHE_SIZE=8GB
     # Enable cache warming
     python -m audio_loader.warm_cache --dir /audio/files
     ```

2. **Memory Errors with Large Files**
   - **Symptom**: OOM errors on files >1GB
   - **Diagnosis**: Monitor memory usage
   - **Solution**: Enable streaming mode
     ```python
     loader = AudioLoader(streaming=True, chunk_size=1024*1024)
     ```

3. **Format Compatibility Issues**
   - **Symptom**: "Unsupported format" errors
   - **Diagnosis**: Check installed codecs
   - **Solution**: 
     ```bash
     # Install additional codecs
     apt-get install ffmpeg libsndfile1
     pip install audioread
     ```

4. **Preprocessing Artifacts**
   - **Symptom**: Clicking/aliasing after resampling
   - **Diagnosis**: Analyze frequency spectrum
   - **Solution**: Adjust resampling parameters
     ```python
     preprocessor = AudioPreprocessor(
         resample_method='kaiser_best',
         anti_alias_filter=True
     )
     ```

### Debug Tools
```python
# Audio file inspector
class AudioInspector:
    def inspect(self, filepath: str):
        """Detailed file analysis"""
        info = {
            'format': self.detect_format(filepath),
            'duration': self.get_duration(filepath),
            'channels': self.get_channels(filepath),
            'sample_rate': self.get_sample_rate(filepath),
            'bit_depth': self.get_bit_depth(filepath),
            'file_size': os.path.getsize(filepath),
            'corruption': self.check_corruption(filepath)
        }
        return info

# Usage
python -m audio_loader.inspect --file problematic_audio.mp3
```

## Notes
- Consider lazy loading for metadata extraction
- Implement progress callbacks for UI integration
- Design with batch processing in mind
- Consider GPU acceleration for resampling operations
- Ensure thread-safety for parallel processing

## References
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Audio Format Specifications](https://www.loc.gov/preservation/digital/formats/fdd/browse_list.shtml)
- Thai Audio Dataset requirements from existing codebase