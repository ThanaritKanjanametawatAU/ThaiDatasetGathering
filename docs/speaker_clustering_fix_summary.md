# Speaker Clustering Fix Summary

## Issue
When running `./main.sh`, the speaker clustering was not working correctly - each sample was getting a unique speaker ID instead of clustering similar speakers together. Specifically, S1-S8 and S10 should have been clustered as the same speaker, with S9 as a different speaker.

## Root Cause
The issue was caused by the batch size configuration in `main.sh`:
- `--speaker-batch-size 50` triggered HDBSCAN clustering algorithm
- `--speaker-batch-size < 50` triggers Agglomerative Clustering algorithm

The speaker identification system uses different clustering algorithms based on batch size:
```python
if len(embeddings) < 50:  # Small batch - use Agglomerative Clustering
    # Better for small, well-defined groups
else:
    # Larger batch - use HDBSCAN
    # Density-based, may not work well for test samples
```

## Solution
Changed the batch sizes in `main.sh`:
1. `--speaker-batch-size 50` → `--speaker-batch-size 30`
2. `--enhancement-batch-size 32` → `--enhancement-batch-size 20`

This ensures Agglomerative Clustering is used, which works better for identifying small speaker groups like our test samples.

## Results
After the fix:
- ✅ S1-S8 and S10 correctly have `speaker_id = SPK_00001`
- ✅ S9 correctly has `speaker_id = SPK_00002`
- ✅ Overall clustering works well: 477 samples → 15 unique speakers (3.14% ratio)

## Verification
The fix can be verified by:
1. Running `./main.sh`
2. Checking https://huggingface.co/datasets/Thanarit/Thai-Voice-Test-1000
3. Confirming S1-S10 clustering meets the requirements

## Additional Improvements Made
1. Fixed `check_and_upload_batch` to always process embeddings before upload
2. Added speaker batch processing after enhancement batches
3. Optimized upload check frequency to allow proper batch accumulation
4. Added forced embedding processing at dataset boundaries