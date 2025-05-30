# Changelog

All notable changes to the Thai Audio Dataset Collection project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Moved common `_save_processing_checkpoint` method to base processor
- Created scripts directory structure for better organization

### Changed
- Refactored processors to reduce code duplication (~600 lines removed)
- Moved analysis and debug scripts from root to scripts/ directory
- Improved code organization following ML pipeline best practices

### Fixed
- Fixed ~100 linting issues across the codebase
- Fixed trailing whitespace and blank line issues
- Fixed missing placeholders in f-strings
- Fixed import ordering issues
- Removed unused imports in processors (time, os, json, Path)
- Removed duplicate methods in base_processor.py (_initialize_audio_enhancer, _apply_noise_reduction)

## [2.3.0] - 2025-01-29

### Added
- Audio enhancement system with secondary speaker detection and removal
- Real-time monitoring dashboard for enhancement progress
- Batch processing for efficient audio enhancement
- GPU acceleration support with automatic CPU fallback
- Audio quality metrics (SNR, PESQ, STOI) calculation
- Configurable enhancement levels (mild, moderate, aggressive)
- Integration of denoiser and spectral gating engines

### Fixed
- Speaker ID dataset separation - ProcessedVoiceTH no longer uses SPK_00001 from GigaSpeech2
- Proper implementation of `reset_for_new_dataset()` between datasets
- Import errors in audio enhancement modules
- Module integration issues from parallel development branches

### Changed
- Updated documentation to reflect audio enhancement features
- Enhanced CLAUDE.md with verification checklist
- Updated requirements.txt with enhancement dependencies

## [2.2.0] - 2025-01-26

### Added
- Adaptive speaker clustering algorithm based on similarity distribution
- AgglomerativeClustering for small batches (<50 samples)
- Dynamic distance threshold adjustment for better clustering accuracy
- Fresh mode support for speaker identification

### Fixed
- Speaker identification accuracy issues causing poor clustering
- JSON serialization issue with numpy int64 types
- AttributeError when cluster_centroids was None

### Changed
- Reverted to original 'pyannote/embedding' model for stability
- Improved clustering parameters for better speaker separation

## [2.1.0] - 2025-01-25

### Added
- Speaker identification and clustering system
- Speaker ID field (SPK_00001, SPK_00002, etc.) to dataset schema
- Support for both cached and streaming modes
- Persistent speaker models in checkpoints directory
- Comprehensive tests for speaker identification

## [2.0.0] - 2025-01-20

### Added
- Streaming mode for processing datasets without full download
- Unified checkpoint system v2.0 with backward compatibility
- Multi-dataset checkpoint support with independent progress tracking
- Custom HuggingFace repository support with --hf-repo flag
- Native TSV transcript loading for GigaSpeech2
- Speech-to-Text (STT) integration for missing transcripts
- Audio format standardization to HuggingFace native format

### Changed
- Audio output format to dictionary with array, sampling_rate, and path
- All audio normalized to -20dB for consistent volume
- Checkpoint format unified across streaming and cached modes

### Fixed
- Streaming mode append functionality
- STT properly fills empty transcripts when enabled

## [1.0.0] - 2023-05-19

### Added
- Initial project implementation
- Support for GigaSpeech2, ProcessedVoiceTH, and Mozilla Common Voice datasets
- Modular processor architecture
- HuggingFace integration
- Audio standardization (16kHz mono)
- Volume normalization
- Batch processing with progress tracking
- Checkpoint system for resumable processing
- Sample mode for testing
- Comprehensive test suite