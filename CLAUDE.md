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

5. **STT System** (`processors/stt/`):
   - `ensemble_stt.py`: Ensemble Speech-to-Text processor for missing transcripts

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

# Performance optimization for high-bandwidth connections
export HF_HUB_ENABLE_HF_TRANSFER=1
python main.py --fresh --all
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
```

## Dataset Schema

- **ID**: Sequential identifiers (S1, S2, S3, ...) globally unique across all datasets
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

- Always check that everything is running fine and the dataset sample is uploaded to huggingface with the command python main.py --fresh --all --streaming --sample