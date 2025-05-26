# Change Log

## 2025-05-26

### Fixed (Speaker Identification Improvements)
- Fixed AttributeError in speaker identification when cluster_centroids was None
- Changed default embedding model from 'pyannote/embedding' to 'pyannote/wespeaker-voxceleb-resnet34-LM'
- Improved clustering parameters for better speaker separation:
  - Reduced min_cluster_size from 15 to 5
  - Reduced min_samples from 10 to 3
  - Increased cluster_selection_epsilon from 0.3 to 0.5
  - Reduced similarity_threshold from 0.7 to 0.6
- Added proper audio resampling to 16kHz for speaker embedding extraction
- Fixed embedding extraction to use proper model inference pipeline
- Successfully tested full pipeline with all features enabled

### Improved (Speaker Clustering Algorithm - Later in Day)
- Fixed speaker identification accuracy issues that were causing poor clustering
- Reverted to original 'pyannote/embedding' model (more stable)
- Implemented adaptive clustering algorithm that adjusts based on similarity distribution
- Uses AgglomerativeClustering for small batches (<50 samples) for better accuracy
- Dynamic distance threshold based on mean similarity:
  - High similarity (>0.9): threshold = 0.2 (min similarity 0.8)
  - Medium similarity (>0.7): threshold = 0.4 (min similarity 0.6)
  - Low similarity: threshold = 0.6 (min similarity 0.4)
- HDBSCAN still used for larger batches with configurable parameters
- Fixed JSON serialization issue with numpy int64 types in speaker model saving
- Restored original clustering parameters as defaults (min_cluster_size=5, min_samples=3)

## 2025-05-25

### Added (Speaker Identification Feature)
- Implemented speaker identification and clustering system
- Added `speaker_id` field to dataset schema (format: SPK_00001, SPK_00002, etc.)
- Created `processors/speaker_identification.py` module with:
  - Speaker embedding extraction using pyannote/embedding model
  - HDBSCAN clustering for identifying unique speakers
  - Support for both cached and streaming modes
- Updated all dataset processors to generate speaker IDs
- Added comprehensive tests for speaker identification
- Updated HuggingFace dataset card metadata to include speaker_id field
- Full schema validation for speaker_id across all processors

## 2023-05-19

### Added (Project Summary)
- Created project summary in `docs/project_summary.md`
- Documented completed tasks
- Documented remaining tasks
- Outlined next steps

## 2023-05-19

### Added (Architecture Documentation)
- Created comprehensive architecture documentation in `docs/architecture.md`
- Added component relationship diagrams
- Added data flow diagrams
- Added sequence diagrams
- Defined API specifications
- Documented design patterns and architecture decisions

## 2023-05-19

### Added (Sample Testing Feature)
- Implemented sample testing feature to process a small number of samples from each dataset
- Added command-line options `--sample` and `--sample-size` to enable sample mode
- Updated all dataset processors to support sample mode
- Added unit tests for sample mode functionality
- Updated documentation to include sample mode usage

## 2023-05-19 (Initial Implementation)

### Added
- Initial project setup
- Created project directory structure
- Implemented configuration settings in `config.py`
- Created utility modules:
  - `utils/audio.py` for audio processing
  - `utils/logging.py` for logging and progress tracking
  - `utils/huggingface.py` for Huggingface interaction
- Implemented base processor class in `processors/base_processor.py`
- Implemented dataset processors:
  - `processors/gigaspeech2.py` for GigaSpeech2 dataset
  - `processors/processed_voice_th.py` for Processed Voice TH dataset
  - `processors/vistec_cv_th.py` for VISTEC Common Voice TH dataset
  - `processors/mozilla_cv.py` for Mozilla Common Voice dataset
- Created main entry point in `main.py`
- Added unit tests:
  - `tests/test_audio.py` for audio utilities
  - `tests/test_base_processor.py` for base processor
  - `tests/test_mozilla_cv.py` for Mozilla Common Voice processor
  - `tests/test_sample_mode.py` for sample testing feature
- Created README.md with usage instructions
- Added requirements.txt

### Technical Decisions
- Used abstract base class for dataset processors to ensure consistent interface
- Implemented checkpointing system for resuming interrupted processing
- Created modular architecture to support adding new dataset processors
- Used streaming processing to minimize memory usage
- Implemented detailed logging and progress tracking