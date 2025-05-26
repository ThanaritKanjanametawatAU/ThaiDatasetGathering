# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Generic streaming split processor in `BaseProcessor` to reduce code duplication
- `get_available_splits()` method in `BaseProcessor` for consistent split detection
- `_extract_additional_fields()` hook for dataset-specific field extraction
- Code quality documentation in README.md
- Best practices section following ML pipeline guidelines
- `scripts/` directory for utility scripts
- This CHANGELOG.md file

### Changed
- Refactored duplicate split processing methods across processors
- Moved `test_multi_dataset_resume.py` to `tests/` directory
- Moved `verify_dataset.py` to `scripts/` directory
- Improved code organization and reduced root directory clutter
- Fixed linting issues (spacing, imports, line lengths)
- Removed unused imports and dead code

### Fixed
- Fixed missing double blank lines between functions (PEP 8)
- Fixed long lines exceeding 120 characters
- Fixed unused variable `sample_archives` in main.py
- Fixed whitespace issues in example scripts

## [2.0.0] - 2025-01-26

### Added
- Improved speaker clustering algorithm with adaptive thresholds
- Better handling of small batches (<50 samples) using AgglomerativeClustering
- Dynamic distance threshold based on similarity distribution

### Changed
- Speaker clustering now adjusts parameters based on data characteristics
- HDBSCAN parameters are dynamically adjusted for better accuracy

## [1.9.0] - 2025-01-25

### Added
- Multi-dataset checkpoint system with independent progress tracking
- Streaming mode append functionality
- Custom HuggingFace repository support with `--hf-repo` flag
- Speech-to-Text integration for missing transcripts
- Speaker identification and clustering across audio samples

### Changed
- Unified checkpoint format with backward compatibility
- Native TSV transcript loading for GigaSpeech2
- Audio format standardization to HuggingFace native format

### Fixed
- Streaming mode append now correctly continues from existing shards
- STT properly fills empty transcripts when enabled