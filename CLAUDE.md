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

### TDD Revert of Secondary Speaker Removal (January 31, 2025)
- **Successfully Reverted Secondary Speaker Removal**: Using Test-Driven Development methodology
- **Fixed Issues**:
  - Resolved omegaconf compatibility errors
  - Restored correct speaker clustering (S1-S8,S10 same speaker, S9 different)
  - Fixed import errors after removing wrapper files
- **Files Removed**:
  - `processors/speaker_identification_wrapper.py`
  - `processors/speaker_identification_simple.py`
  - `utils/omegaconf_fix.py`
  - `run_with_fixes.py`
- **Configuration Changes**:
  - SPEAKER_THRESHOLD: 0.9 → 0.7 (restored original)
  - ENHANCEMENT_LEVEL: ultra_aggressive → moderate (restored original)
- **Verification**: Dataset uploads successfully with all required fields

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

### Verification Guidelines
- **Speaker ID Testing Requirement**: 
  - When testing speaker ID with at least 10 samples on GigaSpeech, S1-S8 and S10 must have the same speaker ID (e.g., SPK_00001) and S9 must have a different speaker ID (e.g., SPK_00002)
  - This clustering pattern verifies that speaker identification is working correctly
- **Dataset Fields**:
  - The Huggingface dataset uses `transcript` (not `transcription`) for the text
  - The dataset source field is `dataset_name` (not `dataset`)
  - All other fields follow the standard schema: ID, speaker_id, audio, Language

[Rest of the file remains unchanged]