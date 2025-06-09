"""
Example usage of the enhanced audio loader.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.audio_loader import AudioLoader, AudioCache
from processors.audio_enhancement.audio_loader_integration import (
    EnhancedAudioProcessor, 
    load_audio, 
    standardize_audio
)
import numpy as np


def main():
    """Demonstrate audio loader usage."""
    
    # Example 1: Basic audio loading
    print("Example 1: Basic Audio Loading")
    print("-" * 50)
    
    loader = AudioLoader()
    
    # Create a test audio file
    import tempfile
    import soundfile as sf
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        # Generate test signal
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz tone
        
        # Add some noise
        audio += np.random.normal(0, 0.01, len(audio))
        
        # Save
        sf.write(tmp.name, audio, sample_rate)
        test_file = tmp.name
    
    try:
        # Load audio
        audio, sr = loader.load_audio(test_file)
        print(f"Loaded audio: shape={audio.shape}, sample_rate={sr}")
        
        # Get metadata
        metadata = loader.get_metadata(test_file)
        print(f"Metadata: {metadata}")
        
    finally:
        os.unlink(test_file)
    
    print()
    
    # Example 2: Enhanced audio processing
    print("Example 2: Enhanced Audio Processing")
    print("-" * 50)
    
    processor = EnhancedAudioProcessor()
    
    # Create another test file with different parameters
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        # Stereo audio at 48kHz
        duration = 1.5
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Different frequency for each channel
        left = np.sin(2 * np.pi * 440 * t) * 0.2
        right = np.sin(2 * np.pi * 880 * t) * 0.2
        
        stereo = np.stack([left, right], axis=1)
        
        # Add silence at beginning and end
        silence = np.zeros((int(sample_rate * 0.5), 2))
        stereo = np.vstack([silence, stereo, silence])
        
        sf.write(tmp.name, stereo, sample_rate)
        test_file = tmp.name
    
    try:
        # Load and preprocess
        processed, target_sr = processor.load_and_preprocess(
            test_file,
            target_sr=16000,
            target_channels=1,
            normalize=True,
            trim_silence=True
        )
        
        print(f"Original: 48kHz stereo with silence")
        print(f"Processed: shape={processed.shape}, sample_rate={target_sr}")
        print(f"Duration reduced from ~2.5s to ~{len(processed)/target_sr:.2f}s")
        
        # Validate
        is_valid = processor.validate_audio(processed)
        print(f"Validation passed: {is_valid}")
        
    finally:
        os.unlink(test_file)
    
    print()
    
    # Example 3: Batch processing with caching
    print("Example 3: Batch Processing with Cache")
    print("-" * 50)
    
    # Create multiple test files
    test_files = []
    for i in range(5):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Different frequencies for each file
            freq = 440 * (i + 1)
            duration = 0.5
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * freq * t) * 0.3
            
            sf.write(tmp.name, audio, sample_rate)
            test_files.append(tmp.name)
    
    try:
        # Load batch
        results = loader.load_batch(test_files, num_workers=4)
        print(f"Loaded {len(results)} files in parallel")
        
        # Check cache
        cached_result = loader.cache.get(test_files[0])
        print(f"First file cached: {cached_result is not None}")
        
        # Load again (should be from cache)
        audio2, sr2 = loader.load_audio(test_files[0])
        print(f"Second load of same file: from cache")
        
    finally:
        for f in test_files:
            os.unlink(f)
    
    print()
    
    # Example 4: Backward compatibility
    print("Example 4: Backward Compatibility")
    print("-" * 50)
    
    # Create test audio as bytes
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Convert to bytes
    import io
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    audio_bytes = buffer.getvalue()
    
    # Use backward-compatible function
    standardized = standardize_audio(
        audio_bytes,
        target_sample_rate=16000,
        target_channels=1,
        normalize_volume=True,
        target_db=-20.0
    )
    
    print(f"Original bytes: {len(audio_bytes)} bytes")
    print(f"Standardized bytes: {len(standardized)} bytes")
    
    # Load standardized audio to verify
    buffer = io.BytesIO(standardized)
    std_audio, std_sr = sf.read(buffer)
    print(f"Standardized audio: shape={std_audio.shape}, sample_rate={std_sr}")
    print(f"RMS level: {20 * np.log10(np.sqrt(np.mean(std_audio**2))):.1f} dB")


if __name__ == "__main__":
    main()