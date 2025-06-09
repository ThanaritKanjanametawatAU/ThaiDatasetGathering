# Aggressive Secondary Speaker Removal Fix

## Date: 2025-06-06

## Issue
The secondary speaker removal was not removing secondary speakers. The dominant speaker separator was incorrectly determining that the "dominant speaker was already clear" and returning the original audio without performing separation.

## Root Cause
In `dominant_speaker_separation.py`, the `extract_dominant_speaker` method had a check that prevented separation:
```python
if activities[0].total_duration / activities[0].speaking_ratio > 0.8:
    logger.info("Dominant speaker already clear, minimal overlap")
    return audio
```

This check was too lenient and caused the system to skip separation for most audio files.

## Fixes Applied

### 1. Fixed Dominant Speaker Separator
- Removed the check that prevented separation when multiple speakers were detected
- Now always performs separation when multiple speakers are present
- Added logging to show number of speakers detected and dominant speaker information

### 2. Made Overlap Detection More Sensitive
In `complete_separation.py`, lowered thresholds for detecting multiple speakers:
- Spectral peaks threshold: 4 → 3
- Fundamental frequencies: 2 → 1  
- Spectral entropy: 3.5 → 3.0
- Modulation index: 0.5 → 0.3

### 3. Continued Processing After Separation
In `core.py`, removed the code that skipped additional secondary speaker removal after dominant speaker extraction:
- Removed `check_secondary = False` after dominant speaker extraction
- Removed `use_separation = False` after dominant speaker extraction
- This ensures the full secondary speaker removal pipeline is applied

## Results
- Test run with 10 samples shows all samples now have "Multiple speakers detected, performing separation"
- The system is now actively removing secondary speakers instead of returning original audio
- Ready for production run with 100 samples

## Files Modified
1. `/media/ssd1/SparkVoiceProject/ThaiDatasetGathering/processors/audio_enhancement/dominant_speaker_separation.py`
2. `/media/ssd1/SparkVoiceProject/ThaiDatasetGathering/processors/audio_enhancement/complete_separation.py`
3. `/media/ssd1/SparkVoiceProject/ThaiDatasetGathering/processors/audio_enhancement/core.py`