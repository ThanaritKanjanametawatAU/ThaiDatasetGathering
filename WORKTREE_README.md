# Secondary Speaker Detection Worktree

## Assignment
This worktree implements flexible secondary speaker detection and suppression.

## Modules to Implement
- `processors/speaker_detection/secondary_detector.py` - Multi-modal detection
- `processors/speaker_detection/speaker_separator.py` - Speaker separation engine
- `processors/speaker_detection/voice_activity.py` - VAD for speaker overlap
- `utils/vad.py` - Voice activity detection utilities

## Key Responsibilities
1. Implement flexible speaker detection (0.1s-10s+ duration)
2. NOT limited to specific words - detect any secondary speaker
3. Multi-modal detection methods:
   - Speaker embedding comparison
   - Energy pattern analysis
   - Spectral change detection
   - Prosody discontinuity detection
4. Confidence-based suppression
5. Integration with pyannote.audio and Asteroid/SepFormer

## Interfaces to Coordinate
- Detection results format for enhancement engine
- Confidence scoring system
- Speaker profile format

## Testing
Run specific tests:
```bash
python -m pytest tests/test_audio_enhancement_comprehensive.py -k "test_07\|test_08\|test_09\|test_10"
```