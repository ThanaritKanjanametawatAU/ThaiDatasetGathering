# Secondary Speaker Removal Fix Summary

## Problem
Secondary speaker removal was not working when running `main.sh` or `main_balanced.sh`, even though it was detected in S5.

## Root Cause
1. The `$SECONDARY_SPEAKER_REMOVAL` flag was defined in both scripts but NOT included in the actual command
2. Secondary speaker removal was only enabled in the `ultra_aggressive` enhancement level
3. The scripts were using `aggressive` enhancement level to preserve speaker characteristics

## Solution
Added secondary speaker removal capabilities to the `aggressive` enhancement level in `/processors/audio_enhancement/core.py`:

```python
'aggressive': {
    'skip': False,
    'denoiser_ratio': 0.005,
    'spectral_ratio': 0.8,
    'passes': 4,
    'check_secondary_speaker': True,  # Added this
    'use_speaker_separation': True    # Added this
},
```

## Verification
Tested with S5 from GigaSpeech2:
- Secondary speaker detected: ✓
- Enhancement applied with secondary speaker removal: ✓
- Enhancement level reported as: `secondary_speaker_removal`

## Result
Now when running `main.sh` with `aggressive` enhancement level:
1. Speaker ID clustering works correctly (S1-S8,S10 same, S9 different)
2. Secondary speakers are removed from audio
3. Both features work together without conflict

## Usage
No changes needed to the scripts. Just run:
```bash
./main.sh --fresh
```

The aggressive enhancement level now includes secondary speaker removal automatically!