# GigaSpeech2 TSV Transcript Loading Update

## Overview
Updated the GigaSpeech2 processor to load native transcripts from TSV files instead of relying solely on Speech-to-Text (STT) for Thai language samples.

## Changes Made

### 1. Added TSV Loading Functionality
- Added `_load_transcript_mapping()` method to download and parse TSV files from HuggingFace
- Supports all splits: train_refined, train_raw, dev, and test
- Implements caching to avoid re-downloading TSV files
- TSV format: `<segment_id>\t<text>`

### 2. Segment ID Extraction
- Added `_extract_segment_id()` method to extract segment IDs from webdataset samples
- Handles the `__key__` field format: `0/685/0-685-9` → extracts `0-685-9`
- Falls back to other fields if `__key__` is not available

### 3. Transcript Lookup Integration
- Modified `_convert_sample()` to first try loading transcript from TSV mapping
- Falls back to sample data or STT if transcript not found in TSV
- Sets confidence_score to 1.0 for native transcripts, 0.0 for missing ones

### 4. Streaming Mode Support
- Updated `process_streaming()` and `_process_split_streaming()` to use TSV transcripts
- Loads TSV mappings on-demand for each split
- Maintains all existing streaming functionality

### 5. Multi-Split Support
- Added `process_all_splits()` method for processing multiple splits
- Ensures proper TSV loading for each split (train, dev, test)

## Benefits

1. **Higher Quality Transcripts**: Native transcripts are more accurate than STT-generated ones
2. **Better Coverage**: ~9.9M transcripts available for training data
3. **Professional Annotations**: Dev and test sets have human-annotated transcripts
4. **Fallback Support**: STT can still be used for samples without TSV transcripts

## TSV Files Available

- `th/train_refined.tsv`: ~9.9M high-quality training transcripts
- `th/train_raw.tsv`: Additional training transcripts (fallback)
- `th/dev.tsv`: ~8K development set transcripts (human-annotated)
- `th/test.tsv`: ~8K test set transcripts (human-annotated)

## Testing Results

Successfully tested with:
- TSV file downloading and parsing
- Transcript matching with audio samples
- Streaming mode processing
- Main pipeline integration

Example output showing successful transcript loading:
```
Sample 2 (0-685-25): "พังยับแบบนี้หลังคายู่ห้องโดยสารยู่เข้าไปเพราะอะไร..."
Sample 4 (0-685-54): "อย่างที่บอกว่าต้องสอบปากคําคนขับแท็กซี่..."
Sample 5 (0-685-58): "แล้วก็ดําเนินการตามขั้นตอนของกฎหมายอีกครั้งหนึ่งนะครับ..."
```

## Backward Compatibility

All existing functionality is preserved:
- Streaming mode still works
- Checkpoint/resume functionality intact
- STT can still be enabled for samples without TSV transcripts
- Cache management unchanged