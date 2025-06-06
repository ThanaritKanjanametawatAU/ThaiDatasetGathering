# Thai STT Implementation Plan - Summary

## Overview
This document summarizes the updated implementation plan for integrating Speech-to-Text (STT) capabilities into the Thai Dataset Gathering project. The goal is to ensure 100% transcript coverage for all audio samples in the Thanarit/Thai-Voice dataset.

## Key Requirements (Updated January 2025)

### 1. Single Split Output
- **All data from every source dataset** (train/test/validation) will be combined into **one single 'train' split** for Thanarit/Thai-Voice
- No separate test/validation splits in the final output

### 2. 100% Transcript Coverage
- **Every audio sample must have a transcript**
- Either from the source dataset or generated by our STT models
- No empty transcripts allowed - use fallback markers like "[INAUDIBLE]" or "[STT_ERROR]" if needed

### 3. Preserve Existing Logic
- Keep processor-specific filtering (e.g., GigaSpeech2's 'th' folder filtering)
- Maintain existing audio preprocessing pipeline
- Continue using streaming mode for memory efficiency

### 4. Enhanced Schema
Add two new fields to the dataset schema:
- `dataset_name` (str): Source dataset identifier
- `confidence_score` (float): 0.0-1.0, where 1.0 = original transcript, <1.0 = STT-generated

### 5. Test-Driven Development
- Test with 5 samples from each dataset
- Iterate until all validation criteria pass
- Verify schema compliance before full processing

## Technical Architecture

### STT Ensemble Models
1. **Primary**: Wav2Vec2 Thai (`airesearch/wav2vec2-large-xlsr-53-th`) - ~3GB VRAM
2. **Secondary**: Whisper Large V3 (`openai/whisper-large-v3`) - ~5GB VRAM
3. **Strategy**: Use highest confidence result from both models

### Memory Management
- **Streaming-first approach** - Never load full dataset into memory
- Process in micro-batches of 16 samples
- Clear GPU cache after each batch
- Maximum 100GB cache for audio buffering

### Processing Pipeline
1. Load dataset in streaming mode (all splits)
2. Apply processor-specific filtering
3. Check if transcript exists
4. If missing, run ensemble STT
5. Add metadata fields (dataset_name, confidence_score)
6. Validate sample completeness
7. Yield to final dataset

## Implementation Checklist

### Pre-Implementation
- [ ] Test with 5 samples from each dataset
- [ ] Verify split discovery and combination logic
- [ ] Validate STT models produce non-empty transcripts
- [ ] Check schema compliance

### During Implementation
- [ ] Implement split combination logic
- [ ] Add dataset_name and confidence_score fields
- [ ] Ensure 100% transcript coverage
- [ ] Implement fallback mechanisms for failed STT
- [ ] Add comprehensive error handling

### Post-Implementation
- [ ] Run full validation suite
- [ ] Upload test batch to HuggingFace
- [ ] Verify audio preview functionality
- [ ] Check all metadata fields are present
- [ ] Document any issues or limitations

## Testing Commands

```bash
# Test with 5 samples from each dataset
python main.py --fresh --all --sample --sample-size 5 --streaming

# Validate results
python validate_samples.py --check-transcripts --check-metadata

# Run full processing (after validation)
python main.py --fresh --all --streaming
```

## Expected Outcomes

1. **Single unified dataset** on Thanarit/Thai-Voice with only 'train' split
2. **100% of samples have transcripts** (no empty values)
3. **Enhanced metadata** for quality tracking and source attribution
4. **Reliable STT pipeline** with fallback mechanisms
5. **Test-driven confidence** in the implementation

## Performance Metrics

- Processing speed: ~30 samples/minute
- GPU utilization: ~95%
- Memory usage: <100GB storage, <100MB RAM per sample
- VRAM usage: ~8GB for both models
- Expected accuracy: >90% for high-quality audio

## Next Steps

1. Review and confirm this plan
2. Begin implementation with test-driven approach
3. Validate with 5-sample tests
4. Iterate based on test results
5. Deploy full processing pipeline

---

**Status**: Ready for implementation pending final confirmation