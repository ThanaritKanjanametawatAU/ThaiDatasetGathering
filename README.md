# Thai Audio Dataset Collection

A modular system to gather Thai audio data from multiple sources and combine them into a single Huggingface dataset (Thanarit/Thai-Voice) with a standardized schema.

## Dataset Schema

- **ID**: Sequential identifiers (S1, S2, S3, ...) globally unique across all datasets
- **speaker_id**: Speaker identifier (SPK_00001, SPK_00002, ...) for speaker tracking
- **Language**: "th" for Thai
- **audio**: Audio data in HuggingFace format (dict with array, sampling_rate, and path)
- **transcript**: Transcript of the audio if available
- **length**: Duration of the audio in seconds
- **dataset_name**: Name of the source dataset (e.g., "GigaSpeech2", "ProcessedVoiceTH")
- **confidence_score**: Confidence score of the transcript (1.0 for original transcripts)

## Features

- **Modular Design**: Easily extensible to add new dataset sources
- **HuggingFace Integration**: Direct upload to HuggingFace Hub
- **Audio Standardization**: Converts all audio to consistent format (16kHz mono)
- **Volume Normalization**: Normalizes audio to -20dB for consistent volume
- **Batch Processing**: Efficient processing of large datasets
- **Progress Tracking**: Visual progress bars for processing
- **Checkpoint System**: Resume interrupted processing with unified checkpoint system (v2.0)
- **Schema Validation**: Ensures data integrity across all sources
- **Streaming Mode**: Process datasets without downloading everything
- **Custom Repository Support**: Push datasets to any HuggingFace repository with --hf-repo flag
- **Speaker Identification**: Automatic speaker clustering using pyannote embeddings
  - Adaptive clustering that adjusts based on similarity distribution
  - AgglomerativeClustering for small batches (<50 samples) for better accuracy
  - HDBSCAN for larger batches with configurable parameters
  - Persistent speaker models across processing runs
  - **Dataset-Separated Speaker IDs**: Each dataset's speakers are clustered independently to prevent cross-dataset merging
  - Supports fresh mode (`--fresh`) to reset speaker IDs
- **Audio Enhancement**: Advanced audio quality improvement
  - Secondary speaker detection and removal
  - Noise reduction with denoiser and spectral gating
  - Configurable enhancement levels (mild, moderate, aggressive)
  - Real-time monitoring dashboard for enhancement progress
  - Batch processing for efficient enhancement
  - GPU acceleration support

## Data Sources

1. **GigaSpeech2** (GitHub: SpeechColab/GigaSpeech2) - Thai refined only
   - Source: https://github.com/SpeechColab/GigaSpeech2
   - Huggingface: https://huggingface.co/datasets/speechcolab/gigaspeech2
   - Filter: Only Thai language content

2. **Processed Voice TH** (Huggingface: Porameht/processed-voice-th-169k)
   - Source: https://huggingface.co/datasets/Porameht/processed-voice-th-169k

3. **Mozilla Common Voice** (Huggingface: mozilla-foundation/common_voice_11_0) - Thai only
   - Source: https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
   - Filter: Only Thai language content

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ThaiDatasetGathering.git
   cd ThaiDatasetGathering
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Huggingface token:
   - Create a file named `hf_token.txt` in the project root
   - Add your Huggingface token to this file

## Usage


### Main Command (Test with this ALWAYS)
```bash
python main.py --fresh GigaSpeech2 ProcessedVoiceTH --sample --sample-size 10 --enable-speaker-id --enable-stt --streaming --speaker-min-cluster-size 5 --speaker-min-samples 3 --speaker-epsilon 0.5 --speaker-threshold 0.6 --enable-audio-enhancement --enhancement-level moderate

```

### Streaming Mode (NEW!)

Process large datasets without downloading everything to disk. Perfect for datasets that exceed your available storage:

```bash
# Process all datasets in streaming mode
python main.py --fresh --all --streaming

# Process specific datasets with streaming
python main.py --fresh GigaSpeech2 ProcessedVoiceTH --streaming

# Test streaming with small samples
python main.py --fresh --all --streaming --sample --sample-size 10

# Resume interrupted streaming processing
python main.py --fresh --all --streaming --resume

# Customize batch sizes for streaming
python main.py --fresh --all --streaming --streaming-batch-size 1000 --upload-batch-size 10000
```


### Basic Usage (Cached Mode)

```bash
# Create new Conda Env
conda create -n thaidataset python=3.10

# Always activate the env
conda activate thaidataset

# Process all datasets and create a fresh dataset
python main.py --fresh --all

# Process only GigaSpeech2 and ProcessedVoiceTH, create fresh
python main.py --fresh GigaSpeech2 ProcessedVoiceTH

# Append MozillaCV dataset to existing dataset
python main.py --append MozillaCommonVoice

# Resume processing from latest checkpoint
python main.py --fresh --all --resume

# Process a small sample (5 entries) from each dataset for testing
python main.py --fresh --all --sample

# Process a specific number of samples from each dataset
python main.py --fresh --all --sample --sample-size 10

# Push to a custom HuggingFace repository
python main.py --fresh --all --hf-repo "myusername/my-thai-dataset"

# Append to a custom repository in streaming mode
python main.py --append MozillaCV --streaming --hf-repo "myorg/thai-audio-collection"
```

### Command-Line Options

**Mode Selection:**
- `--fresh`: Create a new dataset from scratch
- `--append`: Append to an existing dataset
- `--streaming`: Use streaming mode to process without full download (NEW!)

**Dataset Selection:**
- `--all`: Process all available datasets
- Dataset names as positional arguments (e.g., `GigaSpeech2 ProcessedVoiceTH`)

**Processing Options:**
- `--resume`: Resume processing from latest checkpoint
- `--checkpoint CHECKPOINT`: Resume processing from specific checkpoint file
- `--sample`: Process only a small sample from each dataset
- `--sample-size N`: Number of samples to process in sample mode (default: 5)
- `--chunk-size N`: Number of samples per chunk in cached mode (default: 10000)

**Streaming Options:**
- `--streaming-batch-size N`: Batch size for streaming mode (default: 1000)
- `--upload-batch-size N`: Number of samples before uploading a shard (default: 10000)

**Upload Options:**
- `--no-upload`: Skip uploading to Huggingface
- `--private`: Make the Huggingface dataset private
- `--hf-repo REPO`: Specify custom HuggingFace repository to push dataset to (e.g., 'username/dataset-name'). Defaults to 'Thanarit/Thai-Voice' if not specified

**Audio Processing:**
- `--no-standardization`: Disable audio standardization
- `--sample-rate N`: Target sample rate in Hz (default: 16000)
- `--target-db F`: Target volume level in dB (default: -20.0)
- `--no-volume-norm`: Disable volume normalization

**Speech-to-Text Options:**
- `--no-stt`: Disable Speech-to-Text processing
- `--enable-stt`: Enable Speech-to-Text processing for missing transcripts (automatically fills blank transcripts)
- `--stt-batch-size N`: Batch size for STT processing (default: 16)

**Speaker Identification Options:**
- `--enable-speaker-id`: Enable automatic speaker identification and clustering
- `--speaker-model MODEL`: Speaker embedding model to use (default: 'pyannote/embedding')
- `--speaker-batch-size N`: Batch size for speaker processing (default: 10000)
- `--speaker-threshold F`: Similarity threshold for speaker clustering (default: 0.6)
- `--speaker-min-cluster-size N`: Minimum cluster size for HDBSCAN (default: 5)
- `--speaker-min-samples N`: Minimum samples for HDBSCAN (default: 3)
- `--speaker-epsilon F`: Epsilon for HDBSCAN (default: 0.5)
- `--store-embeddings`: Store speaker embeddings (increases storage)

**Audio Enhancement Options:**
- `--enable-audio-enhancement`: Enable audio quality enhancement (noise reduction, clarity improvement)
- `--enhancement-batch-size N`: Batch size for audio enhancement processing (default: 32)
- `--enhancement-dashboard`: Enable real-time dashboard for monitoring enhancement progress
- `--enhancement-level LEVEL`: Enhancement level: mild, moderate, or aggressive (default: moderate)
- `--enhancement-gpu`: Use GPU for audio enhancement (if available)

**Other Options:**
- `--output DIR`: Output directory for local dataset
- `--verbose`: Enable verbose logging
- `--max-cache-gb F`: Maximum cache size in GB for cached mode (default: 100)
- `--clear-cache`: Clear cache before processing

### Streaming vs Cached Mode

**Streaming Mode** (`--streaming`):
- ✅ No storage requirements - processes data on-the-fly
- ✅ Can handle datasets larger than available disk space
- ✅ Uploads data in shards as it processes
- ✅ Fully resumable with checkpoint support
- ❌ Slower processing due to network streaming
- ❌ Requires stable internet connection throughout

**Cached Mode** (default):
- ✅ Faster processing after initial download
- ✅ Can work offline after download
- ✅ Better for smaller datasets
- ❌ Requires enough disk space for entire dataset
- ❌ Initial download can be very large

Choose streaming mode when:
- Working with datasets larger than available storage (e.g., 10TB dataset on 2TB disk)
- You want to start uploading results immediately
- You have a stable, fast internet connection

Choose cached mode when:
- You have sufficient disk space
- You need to process the same dataset multiple times
- You want maximum processing speed

## Project Structure

```
ThaiDatasetGathering/
├── main.py                  # Main entry point
├── config.py                # Configuration settings
├── utils/
│   ├── __init__.py
│   ├── audio.py             # Audio processing utilities
│   ├── logging.py           # Logging utilities
│   ├── huggingface.py       # Huggingface interaction utilities
│   ├── streaming.py         # Streaming mode utilities
│   ├── cache.py             # Cache management utilities
│   ├── audio_metrics.py     # Audio quality metrics
│   └── speaker_utils.py     # Speaker processing utilities
├── monitoring/              # Real-time monitoring components
│   ├── __init__.py
│   ├── dashboard.py         # Enhancement monitoring dashboard
│   ├── metrics_collector.py # Metrics collection and aggregation
│   └── comparison_ui.py     # Before/after comparison UI
├── processors/
│   ├── __init__.py
│   ├── base_processor.py    # Base class for dataset processors
│   ├── gigaspeech2.py       # GigaSpeech2 processor
│   ├── processed_voice_th.py # Processed Voice TH processor
│   ├── mozilla_cv.py        # Mozilla Common Voice processor
│   ├── speaker_identification.py # Speaker ID clustering system
│   ├── audio_enhancement/   # Audio enhancement modules
│   │   ├── __init__.py
│   │   ├── core.py          # Core enhancement orchestrator
│   │   ├── detection/       # Audio issue detection
│   │   │   ├── __init__.py
│   │   │   ├── overlap_detector.py
│   │   │   └── secondary_speaker.py
│   │   ├── engines/         # Enhancement engines
│   │   │   ├── __init__.py
│   │   │   ├── denoiser.py
│   │   │   └── spectral_gating.py
│   │   └── speaker_separation.py
│   └── stt/                 # Speech-to-text processors
│       ├── __init__.py
│       └── ensemble_stt.py
├── tests/
│   ├── __init__.py
│   ├── test_audio.py        # Tests for audio utilities
│   ├── test_base_processor.py # Tests for base processor
│   ├── test_mozilla_cv.py   # Tests for Mozilla Common Voice processor
│   ├── test_sample_mode.py  # Tests for sample mode feature
│   ├── test_streaming.py    # Tests for streaming mode
│   ├── test_streaming_integration.py # Integration tests for streaming
│   ├── test_checkpoint_system.py # Tests for unified checkpoint system
│   └── test_complete_workflow.py # End-to-end workflow tests
├── docs/
│   └── architecture.md      # Architecture documentation
├── data/                    # Data directory (created at runtime)
├── checkpoints/             # Checkpoint directory (created at runtime)
├── logs/                    # Log directory (created at runtime)
├── README.md                # This file
├── requirements.txt         # Dependencies
└── hf_token.txt             # Huggingface token (not included in repo)
```

## Architecture Documentation

For detailed information about the system architecture, design patterns, and component interactions, see [Architecture Documentation](docs/architecture.md).

### Code Organization

The codebase follows a modular architecture with clear separation of concerns:

1. **Base Processor Pattern**: All dataset processors inherit from `BaseProcessor` which provides:
   - Common audio preprocessing functionality
   - Checkpoint management for both cached and streaming modes
   - Validation and error handling
   - Helper methods to reduce code duplication in streaming mode

2. **Streaming Mode Helpers**: The base processor includes reusable methods for streaming:
   - `_initialize_streaming_state()`: Initialize checkpoint and skip count
   - `_should_skip_sample()`: Check if sample should be skipped or processing should stop
   - `_extract_audio_bytes()`: Extract audio bytes from various formats
   - `_process_audio_for_streaming()`: Validate, preprocess, and convert audio
   - `_create_streaming_sample()`: Create standardized samples
   - `_log_streaming_progress()`: Log progress at regular intervals

3. **Audio Processing Pipeline**: Consistent audio handling across all processors:
   - Automatic format detection and conversion
   - Standardization to 16kHz mono WAV
   - Volume normalization to -20dB
   - HuggingFace-compatible audio format output

## Adding New Dataset Processors

To add a new dataset processor:

1. Create a new file in the `processors` directory (e.g., `new_dataset.py`)
2. Implement a class that inherits from `BaseProcessor`
3. Implement the required methods:
   - `process(self, checkpoint=None)`
   - `get_dataset_info(self)`
   - `estimate_size(self)`
4. Add the dataset configuration to `DATASET_CONFIG` in `config.py`

Example:

```python
from processors.base_processor import BaseProcessor

class NewDatasetProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        # Initialize dataset-specific settings

    def process(self, checkpoint=None):
        # Process the dataset
        # Convert to standard schema
        # Yield processed samples

    def get_dataset_info(self):
        # Return dataset information

    def estimate_size(self):
        # Estimate dataset size
```

## Code Quality & Best Practices

This project follows clean code principles and ML pipeline best practices:

### Code Organization
- **Modular Design**: Each dataset processor inherits from `BaseProcessor` with common functionality
- **Separation of Concerns**: Clear separation between data processing, audio handling, and HuggingFace integration
- **DRY Principle**: Common streaming processing logic has been extracted to `BaseProcessor._process_split_streaming_generic()`
- **Type Hints**: Comprehensive type annotations throughout the codebase

### Testing
- Run all tests: `python -m unittest discover`
- Test specific modules: `python -m unittest tests.test_module_name`
- Always test after making changes with: 
  ```bash
  python main.py --fresh --all --sample --sample-size 5 --enable-speaker-id --enable-stt --streaming
  ```

### Development Guidelines
1. **Follow PEP 8**: Use `flake8` for linting (`flake8 . --max-line-length=120`)
2. **Write Tests**: Add tests for new functionality in the `tests/` directory
3. **Document Changes**: Update documentation when adding features
4. **Use Version Control**: Commit logical units of work with clear messages

### Directory Structure
```
ThaiDatasetGathering/
├── main.py                    # Entry point
├── config.py                  # Configuration
├── processors/                # Dataset processors
├── utils/                     # Utility modules
├── tests/                     # Test suite
├── examples/                  # Example scripts
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
└── plan/                      # Project planning
```

## License

[MIT License](LICENSE)
