# Float16 and Secondary Speaker Removal Fixes

## Date: January 31, 2025

## Issues Fixed

### 1. Float16 Support for GigaSpeech2
**Problem**: GigaSpeech2 dataset uses float16 audio arrays, but AudioEnhancer couldn't handle them, causing:
```
Audio enhancement failed for gigaspeech2_train_XX: array type dtype('float16') not supported
```

**Solutions Applied**:
1. Added float16 to float32 conversion in `AudioEnhancer.enhance()` method (line 405-406)
2. Added float16 handling in `EnhancementOrchestrator.enhance()` method (line 53-55)
3. Added dtype preservation to maintain original dtype after processing

### 2. Secondary Speaker Removal Not Working
**Problem**: Secondary speaker removal was not working in ultra_aggressive mode because:
- `preserve_ratio: 0.5` was mixing back 50% of the original audio containing the secondary speaker

**Solution**: Removed `preserve_ratio` from ultra_aggressive configuration, allowing full secondary speaker removal

### 3. Missing SNR Improvement Key
**Problem**: When audio already meets target SNR, the enhancement orchestrator wasn't including `snr_improvement` in metadata

**Solution**: Added `snr_improvement: 0` when audio already meets target SNR

## Files Modified

1. `/media/ssd1/SparkVoiceProject/ThaiDatasetGathering/processors/audio_enhancement/core.py`
   - Added float16 support in `enhance()` method
   - Removed `preserve_ratio` from ultra_aggressive config

2. `/media/ssd1/SparkVoiceProject/ThaiDatasetGathering/processors/audio_enhancement/enhancement_orchestrator.py`
   - Added float16 support
   - Added missing `snr_improvement` key
   - Added dtype preservation

## Verification
These fixes ensure:
- GigaSpeech2 audio with float16 can be processed
- Secondary speaker removal works properly in ultra_aggressive mode
- No metadata key errors occur during processing