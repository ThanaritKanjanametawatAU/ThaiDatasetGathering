# Thai Audio Dataset Collection

A modular system to gather Thai audio data from multiple sources and combine them into a single Huggingface dataset (Thanarit/Thai-Voice) with a standardized schema.

## Dataset Schema

- **ID**: Sequential identifiers (S1, S2, S3, ...) globally unique across all datasets
- **Language**: "th" for Thai
- **audio**: Audio file data
- **transcript**: Transcript of the audio if available
- **length**: Duration of the audio in seconds

## Data Sources

1. **GigaSpeech2** (GitHub: SpeechColab/GigaSpeech2) - Thai refined only
   - Source: https://github.com/SpeechColab/GigaSpeech2
   - Huggingface: https://huggingface.co/datasets/speechcolab/gigaspeech2
   - Filter: Only Thai language content

2. **Processed Voice TH** (Huggingface: Porameht/processed-voice-th-169k)
   - Source: https://huggingface.co/datasets/Porameht/processed-voice-th-169k

3. **VISTEC Common Voice TH** (GitHub: vistec-AI/commonvoice-th)
   - Source: https://github.com/vistec-AI/commonvoice-th

4. **Mozilla Common Voice** (Huggingface: mozilla-foundation/common_voice_11_0) - Thai only
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

### Basic Usage

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
python main.py --append MozillaCV

# Resume processing from latest checkpoint
python main.py --fresh --all --resume

# Process a small sample (5 entries) from each dataset for testing
python main.py --fresh --all --sample

# Process a specific number of samples from each dataset
python main.py --fresh --all --sample --sample-size 10
```

### Command-Line Options

- `--fresh`: Create a new dataset from scratch
- `--append`: Append to an existing dataset
- `--all`: Process all available datasets
- Dataset names as positional arguments (e.g., `GigaSpeech2 ProcessedVoiceTH`)
- `--resume`: Resume processing from latest checkpoint
- `--checkpoint CHECKPOINT`: Resume processing from specific checkpoint file
- `--no-upload`: Skip uploading to Huggingface
- `--private`: Make the Huggingface dataset private
- `--output DIR`: Output directory for local dataset
- `--verbose`: Enable verbose logging
- `--sample`: Process only a small sample from each dataset
- `--sample-size N`: Number of samples to process in sample mode (default: 5)

## Project Structure

```
ThaiDatasetGathering/
├── main.py                  # Main entry point
├── config.py                # Configuration settings
├── utils/
│   ├── __init__.py
│   ├── audio.py             # Audio processing utilities
│   ├── logging.py           # Logging utilities
│   └── huggingface.py       # Huggingface interaction utilities
├── processors/
│   ├── __init__.py
│   ├── base_processor.py    # Base class for dataset processors
│   ├── gigaspeech2.py       # GigaSpeech2 processor
│   ├── processed_voice_th.py # Processed Voice TH processor
│   ├── vistec_cv_th.py      # VISTEC Common Voice TH processor
│   └── mozilla_cv.py        # Mozilla Common Voice processor
├── tests/
│   ├── __init__.py
│   ├── test_audio.py        # Tests for audio utilities
│   ├── test_base_processor.py # Tests for base processor
│   ├── test_mozilla_cv.py   # Tests for Mozilla Common Voice processor
│   └── test_sample_mode.py  # Tests for sample mode feature
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

## License

[MIT License](LICENSE)
