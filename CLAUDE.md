# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains code for the Thai Audio Dataset Collection project - a modular system designed to gather Thai audio data from multiple sources and combine them into a single standardized dataset hosted on Huggingface (Thanarit/Thai-Voice).

The project collects, processes, and normalizes Thai language audio from various sources including:
- GigaSpeech2
- Processed Voice TH
- VISTEC Common Voice TH
- Mozilla Common Voice

## Dataset Verification

You can Check the pushed dataset on https://huggingface.co/datasets/Thanarit/Thai-Voice

## Recent Updates (January 2025)

### Code Cleanup and Refactoring (January 30, 2025)
- **Major Code Cleanup**: Following Power of 10 rules and ML pipeline best practices
- **Reduced Code Duplication**: 
  - Moved common `_save_processing_checkpoint` method to base processor
  - Removed ~600 lines of duplicate code across processors
  - Identified `_process_split_streaming_generic()` method for future refactoring
- **Improved File Organization**: 
  - Moved analysis scripts to `scripts/analysis/` directory
  - Moved debug scripts to `scripts/debug/` directory  
  - Root directory now only contains main.py and config.py
- **Fixed All Linting Issues**: ~100 issues resolved including:
  - Import ordering (E402)
  - Whitespace issues (W293, W291)
  - Line length (E501)
  - Missing f-string placeholders (F541)
  - Unused imports (F401)
- **Enhanced Documentation**: 
  - Added comprehensive code quality section to README
  - Created CHANGELOG.md with detailed version history
  - Added code_quality_report_2025.md documenting all improvements
- **Test Suite Updates**: All tests pass (except one pre-existing test bug)

### Custom HuggingFace Repository Support
- Added `--hf-repo` command-line flag to specify custom target repository
- Allows pushing datasets to any HuggingFace repository you have access to
- Defaults to `Thanarit/Thai-Voice` if not specified
- Works with both cached and streaming modes
- Supports append mode to custom repositories

### GigaSpeech2 Native Transcript Support
- Implemented native TSV transcript loading from GigaSpeech2's `text` column
- No longer requires external transcript mapping files
- Transcripts are loaded directly from the dataset's TSV files
- Improved reliability and reduced external dependencies

### Audio Format Standardization
The project has been updated to use HuggingFace's native audio format for better compatibility with the platform's audio preview features. All processors now:
- Output audio as a dictionary with `{'array': np.array, 'sampling_rate': int, 'path': str}` format
- Preprocess audio to 16kHz mono WAV before encoding  
- Include proper validation for audio data integrity
- All audio is normalized to -20dB for consistent volume levels
- Added `create_hf_audio_format()` utility method in base processor for consistent format conversion

### Unified Checkpoint System (v2.0)
- Single checkpoint format for both streaming and cached modes
- Backward compatibility with legacy checkpoint formats
- Version tracking for format evolution
- Consistent field naming across all processing modes

### Schema Enhancements
- Added `dataset_name` field to track source dataset
- Added `confidence_score` field for transcript reliability (1.0 for original transcripts)
- Added `speaker_id` field for speaker tracking (SPK_00001, SPK_00002, ...)
- Full schema validation across all processors

### Speech-to-Text Integration
- Optional STT processing for samples without transcripts
- Command-line flags: `--enable-stt`, `--no-stt`, `--stt-batch-size`
- Ensemble STT model support for improved accuracy
- **Fixed (Jan 2025)**: STT now properly fills empty transcripts when `--enable-stt` flag is used
- Automatic detection and processing of samples with blank transcripts

### Streaming Mode Append Functionality (Jan 2025)
- **Fixed**: `--append` flag now correctly appends to existing datasets without overwriting
- StreamingUploader detects existing shard numbers and continues from the next available shard
- Preserves existing data when adding new samples
- Maintains sequential ID numbering across append operations

### Multi-Dataset Checkpoint System
- Each dataset maintains independent checkpoint files (`{dataset_name}_unified_checkpoint.json`)
- Supports interruption and resumption across multiple datasets with `--resume` flag
- Automatically skips completed datasets when resuming
- Preserves progress within partially processed datasets
- Checkpoint includes: samples processed, current split, split index, and dataset-specific data
- No global checkpoint needed - each dataset tracks its own progress independently

### Speaker Identification (January 2025)
- Automatic speaker identification and clustering across all audio samples
- Uses pyannote/embedding model for speaker embeddings
- **Improved Clustering Algorithm (January 26, 2025)**:
  - Adaptive clustering that adjusts based on similarity distribution
  - Uses AgglomerativeClustering for small batches (<50 samples) for better accuracy
  - Dynamic distance threshold based on mean similarity
  - HDBSCAN for larger batches with configurable parameters
- Generates consistent speaker IDs (SPK_00001, SPK_00002, etc.)
- Each processor implements speaker ID generation
- Speaker IDs are preserved across dataset processing runs
- Supports both cached and streaming modes
- Persistent speaker models stored in checkpoints directory
- **Fresh Mode Support (January 26, 2025)**:
  - `--fresh` flag now resets speaker ID counter to 1 instead of continuing from saved state
  - Deletes existing speaker model file when fresh mode is enabled
  - Ensures clean start for new dataset collections
- **Dataset Separation (January 27, 2025)**:
  - Speaker identification now processes each dataset independently
  - Prevents cross-dataset speaker merging by resetting clustering state between datasets
  - Each dataset maintains its own speaker clusters while keeping globally unique IDs
  - Added `reset_for_new_dataset()` method to SpeakerIdentification class
- **Fixed Dataset Speaker ID Overlap (January 29, 2025)**:
  - Resolved issue where ProcessedVoiceTH would incorrectly use SPK_00001 from GigaSpeech2
  - Now properly calls `reset_for_new_dataset(reset_counter=False)` between datasets
  - Ensures no speaker ID overlap between different datasets while maintaining unique IDs

### Audio Enhancement (January 2025)
- **Secondary Speaker Detection and Removal**:
  - Detects overlapping speech and secondary speakers in audio
  - Uses spectral subtraction to remove secondary speakers
  - Configurable suppression strength (0.0-1.0)
- **Noise Reduction**:
  - Denoiser engine using Facebook's denoiser model
  - Spectral gating for background noise removal
  - Adaptive noise profiling for better results
- **Enhancement Levels**:
  - Mild: Light noise reduction, preserves audio character
  - Moderate: Balanced enhancement (default)
  - Aggressive: Maximum noise removal, may affect voice quality
- **Real-time Monitoring**:
  - Live dashboard for tracking enhancement progress
  - Before/after audio comparison
  - Metrics visualization (SNR, PESQ, STOI)
- **Batch Processing**:
  - Efficient parallel processing of audio samples
  - GPU acceleration support
  - Automatic fallback to CPU if GPU unavailable

## Architecture

### Core Components

1. **Main Controller** (`main.py`):
   - Entry point for the application
   - Handles command-line arguments
   - Orchestrates dataset processing workflow
   - Manages dataset creation and upload to Huggingface

2. **Configuration System** (`config.py`):
   - Defines dataset sources and configurations
   - Specifies schema for the target dataset
   - Contains validation rules for data integrity
   - Sets audio processing parameters

3. **Processor System**:
   - `base_processor.py`: Abstract base class for all processors
     - Provides `create_hf_audio_format()` for consistent audio formatting
     - Implements unified checkpoint system (v2.0) with backward compatibility
     - Provides `process_all_splits()` for streaming mode entry point
     - Helper methods for streaming mode processing
   - Specialized processors for each dataset source:
     - `gigaspeech2.py`: GigaSpeech2 Thai subset
     - `mozilla_cv.py`: Mozilla Common Voice Thai
     - `processed_voice_th.py`: Processed Voice TH dataset
   - Handles data extraction, transformation, and validation

4. **Utility Modules**:
   - `audio.py`: Audio processing utilities (normalization, resampling, format conversion)
   - `huggingface.py`: Integration with Huggingface APIs and dataset management
   - `logging.py`: Specialized logging and progress tracking
   - `cache.py`: Caching utilities for efficient data processing
   - `streaming.py`: Streaming mode infrastructure (StreamingUploader, StreamingBatchProcessor)
   - `audio_metrics.py`: Audio quality metrics calculation (SNR, PESQ, STOI)
   - `speaker_utils.py`: Speaker processing utilities

5. **STT System** (`processors/stt/`):
   - `ensemble_stt.py`: Ensemble Speech-to-Text processor for missing transcripts

6. **Audio Enhancement System** (`processors/audio_enhancement/`):
   - `core.py`: Enhancement orchestrator managing all enhancement engines
   - `detection/secondary_speaker.py`: Secondary speaker detection algorithm
   - `detection/overlap_detector.py`: Speech overlap detection
   - `engines/denoiser.py`: Deep learning-based noise reduction
   - `engines/spectral_gating.py`: Spectral noise gate implementation
   - `speaker_separation.py`: Speaker separation utilities

7. **Monitoring System** (`monitoring/`):
   - `dashboard.py`: Real-time web dashboard for enhancement monitoring
   - `metrics_collector.py`: Collects and aggregates audio quality metrics
   - `comparison_ui.py`: Before/after audio comparison interface

### Data Flow

1. User invokes `main.py` with command-line arguments
2. System loads configurations and validates arguments
3. For each specified dataset, a processor is instantiated
4. Each processor:
   - Loads source data from respective repository
   - Applies audio preprocessing (normalization, resampling)
   - Converts audio to HuggingFace format using `create_hf_audio_format()`
   - Transforms data into standardized schema
   - Validates data integrity
   - Creates checkpoints for resumable processing
5. Processed data is combined into a unified dataset
6. Combined dataset is saved locally and/or uploaded to Huggingface

## Common Commands

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ThaiDatasetGathering.git
cd ThaiDatasetGathering

# Install dependencies
pip install -r requirements.txt

# Optional for faster downloads
pip install hf_transfer
```

### Running the Application

```bash
# Process all datasets and create a fresh dataset
python main.py --fresh --all

# Process only specific datasets
python main.py --fresh GigaSpeech2 ProcessedVoiceTH

# Append to existing dataset
python main.py --append MozillaCV

# Resume from latest checkpoint
python main.py --fresh --all --resume

# Process sample data (for testing)
python main.py --fresh --all --sample
python main.py --fresh --all --sample --sample-size 10

# Streaming mode (process without downloading entire dataset)
python main.py --fresh --all --streaming
python main.py --fresh --all --streaming --resume

# Enable Speech-to-Text for missing transcripts
python main.py --fresh --all --enable-stt --stt-batch-size 32

# Enable Speaker Identification
python main.py --fresh --all --enable-speaker-id --streaming

# Speaker ID with custom parameters
python main.py --fresh --all --enable-speaker-id --speaker-min-cluster-size 5 --speaker-min-samples 3 --speaker-threshold 0.6

# Full test with all features
python main.py --fresh --all --sample --sample-size 50 --enable-speaker-id --enable-stt --streaming

# Enable audio enhancement with speaker ID
python main.py --fresh --all --enable-speaker-id --enable-audio-enhancement --enhancement-level moderate --streaming

# Full test with enhancement and dashboard
python main.py --fresh GigaSpeech2 ProcessedVoiceTH --sample --sample-size 10 --enable-speaker-id --enable-audio-enhancement --enhancement-dashboard --streaming

# Performance optimization for high-bandwidth connections
export HF_HUB_ENABLE_HF_TRANSFER=1
python main.py --fresh --all

# Push to a custom HuggingFace repository
python main.py --fresh --all --hf-repo "myusername/my-thai-dataset"

# Append to a custom repository
python main.py --append MozillaCV --hf-repo "myorg/thai-audio-collection"
```

### Testing

```bash
# Run all tests
python -m unittest discover

# Run specific test modules
python -m unittest tests.test_audio
python -m unittest tests.test_base_processor
python -m unittest tests.test_mozilla_cv
python -m unittest tests.test_sample_mode
python -m unittest tests.test_streaming
python -m unittest tests.test_streaming_integration
python -m unittest tests.test_checkpoint_system
python -m unittest tests.test_complete_workflow
python -m unittest tests.test_streaming_append
python -m unittest tests.test_multi_dataset_checkpoint
python -m unittest tests.test_speaker_identification
python -m unittest tests.test_speaker_id_streaming
python -m unittest tests.test_huggingface_schema
python -m unittest tests.test_huggingface_schema_complete
python -m unittest tests.test_clustering_specific_samples
python -m unittest tests.test_speaker_clustering_accuracy
```

## Dataset Schema

- **ID**: Sequential identifiers (S1, S2, S3, ...) globally unique across all datasets
- **speaker_id**: Speaker identifier (SPK_00001, SPK_00002, ...) for speaker tracking
- **Language**: "th" for Thai
- **audio**: Audio data in HuggingFace format:
  ```python
  {
      "array": np.array([...]),      # Audio samples as numpy array
      "sampling_rate": 16000,        # Sample rate (always 16kHz)
      "path": "S1.wav"              # File path
  }
  ```
- **transcript**: Transcript of the audio if available
- **length**: Duration of the audio in seconds
- **dataset_name**: Name of the source dataset (e.g., "GigaSpeech2", "ProcessedVoiceTH")
- **confidence_score**: Confidence score of the transcript (1.0 for original transcripts)

## Adding New Dataset Processors

To add a new dataset processor:

1. Create a new file in the `processors` directory (e.g., `new_dataset.py`)
2. Implement a class that inherits from `BaseProcessor`
3. Implement the required methods:
   - `process(self, checkpoint=None)`
   - `get_dataset_info(self)`
   - `estimate_size(self)`
4. Use `self.create_hf_audio_format(audio_bytes, audio_id)` to create audio entries
5. Add the dataset configuration to `DATASET_CONFIG` in `config.py`

## Testing and Verification

### Audio Format Verification
```bash
# Verify HuggingFace audio format compatibility
python verify_hf_audio_format.py

# Test audio playback functionality
python test_audio_playback.py

# Test audio format conversion
python test_audio_format.py
```

### Lint and Type Checking
```bash
# Run linting (if available)
python -m flake8 .

# Run type checking (if available)
python -m mypy .
```

## Quick Verification Checklist

- Always run `python main.py --fresh GigaSpeech2 ProcessedVoiceTH --sample --sample-size 10 --enable-speaker-id --enable-stt --streaming --enable-audio-enhancement` to test and check huggingface page when the command finishes every time new feature is implementation to make completely sure that it will work as expected.
- Verify that:
  - S1-S8 and S10 have the same speaker ID (SPK_00001)
  - S9 has a different speaker ID (SPK_00002)  
  - ProcessedVoiceTH samples do NOT use SPK_00001 or SPK_00002
  - No speaker ID overlap between datasets
  - Audio enhancement removes secondary speakers effectively
  - Enhancement metrics are properly collected
```

</invoke>