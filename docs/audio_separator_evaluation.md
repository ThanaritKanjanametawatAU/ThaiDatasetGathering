# Audio Separator Evaluation for Secondary Speaker Removal

## Summary

**python-audio-separator** is a promising solution for secondary speaker removal, but it currently has a **compatibility issue with NumPy 2.x**.

## Technical Findings

### Pros:
1. **State-of-the-art Models**: Uses pre-trained models from Ultimate Vocal Remover (UVR)
   - Best models achieve SDR (Signal-to-Distortion Ratio) > 12.9
   - Multiple architectures available: MDX-Net, VR Arch, Demucs, MDXC

2. **Vocal Isolation Approach**: By isolating vocals, it inherently reduces secondary speakers

3. **Full Audio Processing**: Unlike PyAnnote which only detected 36% of audio, audio-separator processes 100%

4. **Proven Technology**: These models are widely used for music source separation

### Cons:
1. **NumPy Compatibility**: Requires NumPy < 2.0, but the project uses NumPy 2.0.2
2. **External Dependency**: Adds another dependency to the project
3. **Model Size**: Models are large (50-600MB each)

## Implementation Status

Created integration code:
- `processors/audio_enhancement/audio_separator_secondary_removal.py` - Wrapper class
- `test_audio_separator_approach.py` - Test script
- Updated `core.py` to support audio-separator method
- Added configuration in `main.sh`

## Compatibility Issue

The main blocker is the NumPy version conflict:
```
ImportError: 
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash.
```

This happens because audio-separator depends on SciPy, which was compiled with NumPy 1.x.

## Solutions

### Option 1: Create Separate Environment (Recommended for Testing)
```bash
# Create new conda environment with NumPy 1.x
conda create -n audio-separator-env python=3.10 numpy<2
conda activate audio-separator-env
pip install audio-separator[cpu]
```

### Option 2: Downgrade NumPy (Not Recommended)
This could break other parts of the project that depend on NumPy 2.x features.

### Option 3: Wait for Updates
audio-separator needs to update its dependencies for NumPy 2.x compatibility.

### Option 4: Use Docker
Run audio-separator in a Docker container with the correct dependencies:
```bash
docker run -it -v `pwd`:/workdir beveradb/audio-separator input.wav
```

## Alternative Approaches

Since audio-separator isn't immediately usable due to compatibility issues, consider:

1. **Demucs Direct Integration**: Use the Demucs library directly (which audio-separator wraps)
2. **SpeechBrain Models**: Try models from SpeechBrain for source separation
3. **Asteroid Toolkit**: Another source separation toolkit
4. **Spleeter by Deezer**: Simpler but effective source separation

## Conclusion

Audio-separator shows promise for secondary speaker removal, but the NumPy compatibility issue prevents immediate integration. The existing PyAnnote-based approach with full audio processing (FullAudioSecondaryRemoval) remains the best current option until either:
1. The project can be refactored to use NumPy 1.x
2. Audio-separator updates for NumPy 2.x compatibility
3. An alternative source separation library is integrated