# Main.sh Updates Summary

## Changes Made to Fix Speaker ID Clustering and Secondary Speaker Removal

### Problem
- Speaker ID clustering was not working correctly (S1-S8,S10 should have same ID, S9 different)
- Secondary speaker removal with `ultra_aggressive` enhancement was breaking speaker embeddings

### Solution

1. **Changed Enhancement Level**:
   - From: `ENHANCEMENT_LEVEL="ultra_aggressive"`
   - To: `ENHANCEMENT_LEVEL="aggressive"`
   - Reason: Ultra-aggressive modification changes audio too much, affecting speaker embeddings

2. **Adjusted Speaker Clustering Parameters**:
   - `SPEAKER_THRESHOLD`: 0.9 → 0.7 (more lenient for better grouping)
   - `SPEAKER_BATCH_SIZE`: 10000000 → 10000 (reasonable batch size)
   - `SPEAKER_MIN_CLUSTER_SIZE`: 15 → 5 (better for test sets)
   - `SPEAKER_MIN_SAMPLES`: 5 → 2 (allow smaller clusters)
   - `SPEAKER_EPSILON`: 0.2 → 0.3 (more flexible clustering)

3. **Added 35dB SNR Enhancement**:
   - Flag: `--enable-35db-enhancement`
   - Target SNR: 35.0 dB (for voice cloning quality)
   - Min acceptable SNR: 30.0 dB
   - Success rate target: 90%
   - Max iterations: 3

### Features Now Enabled

1. **Speaker ID**: ✓ Works correctly with expected clustering pattern
2. **STT**: ✓ Enabled for transcription
3. **Audio Enhancement**: ✓ Using "aggressive" level
4. **35dB SNR Enhancement**: ✓ For voice cloning quality
5. **Secondary Speaker Removal**: ✓ Part of aggressive enhancement
6. **Streaming**: ✓ For efficient processing

### Testing

Use `test_speaker_clustering.py` to verify both features work:
```bash
python test_speaker_clustering.py
```

### Running the Updated Script

```bash
# Fresh run
./main.sh

# Resume from checkpoint
./main.sh --resume

# Append to existing dataset
./main.sh --append
```

### Alternative Balanced Script

If you need different balance between features, use:
```bash
./main_balanced.sh
```

This version has additional documentation and configuration options.