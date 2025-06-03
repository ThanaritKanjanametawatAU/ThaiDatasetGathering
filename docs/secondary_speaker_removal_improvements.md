# Secondary Speaker Removal Improvements

## Summary
The secondary speaker removal feature has been significantly enhanced with more aggressive detection and suppression techniques.

## Improvements Made

### 1. Enhanced Detection Sensitivity
- **Speaker similarity threshold**: Reduced from 0.7 to 0.5 for more sensitive detection
- **Confidence threshold**: Lowered from 0.5 to 0.3 to detect more secondary speakers
- **Detection window**: Reduced from 500ms to 300ms for better granularity
- **Minimum duration**: Reduced from 0.1s to 0.05s to catch shorter segments

### 2. Stronger Suppression Algorithm
- **Suppression strength**: Increased from 0.6 to 0.95 (95% suppression)
- **Multi-technique approach** for strong suppression (>0.8):
  - Aggressive amplitude reduction (90% reduction)
  - Spectral suppression across all frequency bands
  - Noise floor addition to mask residual signals
  - Gate-like effect for low amplitude signals
- **Window expansion**: Added 50ms expansion around detected segments
- **No blending** for very strong suppression (>0.9) to avoid mixing

### 3. Simple Secondary Removal
Added a new `SimpleSecondaryRemoval` module that:
- Uses energy-based detection for fast processing
- Detects speakers after silence periods
- Identifies sudden energy changes
- Applies -60dB suppression (0.1% of original signal)
- Works as additional processing after main detection

### 4. Combined Processing Pipeline
For ultra_aggressive mode:
1. Advanced speaker separation (embedding-based)
2. Simple secondary removal (energy-based)
3. Spectral gating/denoising
4. Reduced preserve ratio (25% instead of 50%)

## Configuration in main.sh

The current settings optimize for maximum secondary speaker removal:
```bash
ENHANCEMENT_LEVEL="ultra_aggressive"  # Triggers all removal methods
SPEAKER_THRESHOLD="0.9"              # High threshold for main speaker
```

## Testing Results
- Secondary speaker energy reduced to ~40-44% of original
- Detection working for segments as short as 50ms
- Multiple detection methods increase coverage

## Recommendations

If secondary speakers are still audible:

1. **Install advanced models**:
   ```bash
   pip install asteroid  # For SepFormer model
   pip install denoiser  # For Facebook Denoiser
   ```

2. **Adjust thresholds in main.sh**:
   - Reduce SPEAKER_THRESHOLD to 0.8 for more aggressive clustering
   - Increase SPEAKER_MIN_CLUSTER_SIZE to 20 for stricter grouping

3. **Use batch processing**:
   - Process multiple files to allow better speaker clustering
   - Larger batches improve detection accuracy

## Limitations
- Works best with clear speech segments
- May struggle with overlapping simultaneous speech
- Synthetic test audio shows ~60% reduction
- Real audio results may vary based on recording quality

## Future Improvements
1. Implement SepFormer for advanced source separation
2. Add overlap detection and separation
3. Use speaker diarization for better segmentation
4. Implement adaptive thresholds based on audio characteristics