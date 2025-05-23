# Audio Format Update Changelog

## Overview
Updated the Thai Audio Dataset Collection project to use proper HuggingFace Audio format that supports preview functionality in the HuggingFace dataset viewer.

## Changes Made

### 1. Schema Update (`config.py`)
- **Changed**: Audio schema from `bytes` to `dict`
- **Reason**: HuggingFace Audio feature requires dictionary format with `array`, `sampling_rate`, and `path` fields
- **Impact**: Enables proper audio preview in HuggingFace dataset viewer

### 2. Base Processor Enhancement (`processors/base_processor.py`)
- **Added**: `create_hf_audio_format()` method to convert audio bytes to HuggingFace-compatible format
- **Features**:
  - Converts raw audio bytes to numpy arrays
  - Ensures float32 format for consistency
  - Includes sampling rate and virtual path
  - Graceful error handling with fallback

### 3. Utility Functions Update (`utils/audio.py`)
- **Updated**: `is_valid_audio()`, `get_audio_length()`, and `get_audio_info()` functions
- **Enhancement**: Now support both bytes and HuggingFace dictionary formats
- **Backward Compatibility**: Maintains support for legacy bytes format

### 4. Processor Updates
Updated all dataset processors to use the new audio format:

#### GigaSpeech2 Processor (`processors/gigaspeech2.py`)
- Apply audio preprocessing (standardization)
- Convert to HuggingFace audio format
- Calculate length from the standardized format

#### Mozilla Common Voice Processor (`processors/mozilla_cv.py`)
- Same audio format conversion pattern
- Maintains preprocessing pipeline

#### Processed Voice TH Processor (`processors/processed_voice_th.py`)
- Updated to use HuggingFace audio format
- Consistent with other processors

#### VISTEC Common Voice TH Processor (`processors/vistec_cv_th.py`)
- Updated for file-based audio loading
- Applies preprocessing and format conversion

## Audio Format Structure

### New Format (HuggingFace Compatible)
```python
{
    "array": numpy_array,          # Float32 audio samples
    "sampling_rate": int,          # Sample rate (e.g., 16000)
    "path": str                    # Virtual path (e.g., "S1.wav")
}
```

### Benefits
1. **Preview Support**: Audio can be played directly in HuggingFace dataset viewer
2. **Standardization**: Consistent 16kHz, mono, -20dB normalized audio
3. **Compatibility**: Works with HuggingFace Audio() feature
4. **Quality**: Maintains high audio quality with proper float32 format

## Verification

### Tests Created
1. `test_audio_format.py` - Verifies correct HuggingFace format structure
2. `test_audio_playback.py` - Tests audio can be written/read correctly
3. `verify_hf_audio_format.py` - Comprehensive format validation

### Validation Results
- ✅ Correct dictionary structure with array, sampling_rate, path
- ✅ Audio data is in float32 format suitable for playback
- ✅ Sampling rates are consistent at 16kHz
- ✅ Audio contains meaningful signal data (not silent)
- ✅ No clipping detected in processed audio
- ✅ Proper volume normalization to -20dB
- ✅ Format is compatible with HuggingFace audio preview widgets

## Usage

The updated format works transparently with existing commands:

```bash
# Process and upload with proper audio format
python main.py --fresh --all

# Sample processing (now with preview-compatible audio)
python main.py --fresh GigaSpeech2 --sample --sample-size 5

# Audio preprocessing is still available
python main.py --fresh --all --target-db -20.0 --sample-rate 16000
```

## Dataset Preview

The uploaded datasets now support:
- 🎵 Audio playback in HuggingFace dataset viewer
- 📊 Proper audio metadata display
- 🔄 Seamless integration with HuggingFace Transformers
- 📱 Cross-platform audio compatibility

## Technical Notes

- Audio preprocessing (normalization, resampling) still occurs before format conversion
- Length calculation is performed on the final HuggingFace format for consistency
- Error handling includes fallback to ensure dataset creation doesn't fail
- All processors use the shared `create_hf_audio_format()` method for consistency