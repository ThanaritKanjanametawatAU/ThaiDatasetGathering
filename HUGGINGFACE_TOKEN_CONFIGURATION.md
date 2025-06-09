# HuggingFace Token Configuration - Complete

## Summary

Successfully configured the system to use HuggingFace token from `.env` file instead of `.hf_token` or `hf_token.txt`.

## Changes Made

### 1. **Updated `utils/huggingface.py`**
- Modified `read_hf_token()` function to prioritize reading from `.env` file
- Supports format: `hf_token = your_token_here` in `.env`
- Falls back to token file or environment variables if not found in `.env`

### 2. **Updated `processors/speaker_identification.py`**
- Added token authentication for PyAnnote embedding model loading
- Uses `use_auth_token=hf_token` parameter in `Model.from_pretrained()`

### 3. **Updated `utils/snr_measurement.py`**
- Added token authentication for PyAnnote segmentation model
- Uses token for `VoiceActivityDetection` pipeline

## Token File Location

The token is stored in `.env` file at project root:
```
HF_HUB_ENABLE_HF_TRANSFER=1
hf_token = <YOUR_HUGGINGFACE_TOKEN_HERE>
```

## Verification

The system now successfully:
- ✅ Loads token from `.env` file
- ✅ Authenticates with HuggingFace for gated models
- ✅ Loads PyAnnote embedding model (`pyannote/embedding`)
- ✅ Loads PyAnnote segmentation model (`pyannote/segmentation-3.0`)
- ✅ Processes audio with secondary speaker removal

## Main.sh Configuration

Current configuration in `main.sh`:
- Enhancement level: `selective_secondary_removal`
- All features enabled (Speaker ID, STT, Audio Enhancement, 35dB SNR)
- Processing 100 samples per dataset
- Target repository: `Thanarit/Thai-Voice-10000000`

## Usage

Simply run:
```bash
./main.sh
```

The system will automatically:
1. Read token from `.env` file
2. Authenticate with HuggingFace
3. Load required models
4. Process audio with robust secondary speaker removal
5. Upload to HuggingFace dataset repository