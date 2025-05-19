"""
Audio processing utilities for the Thai Audio Dataset Collection project.
"""

import io
import logging
from typing import Optional, Tuple, Union

# Set up logger
logger = logging.getLogger(__name__)

try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    logger.error("Required audio libraries not installed. Please install librosa, soundfile, and numpy.")
    raise

def is_valid_audio(audio_data: bytes) -> bool:
    """
    Check if the provided bytes contain valid audio data.
    
    Args:
        audio_data: Audio data as bytes
        
    Returns:
        bool: True if valid audio, False otherwise
    """
    if not audio_data or len(audio_data) < 10:  # Arbitrary minimum size
        return False
    
    try:
        # Try to load with librosa
        audio_file = io.BytesIO(audio_data)
        _, sr = librosa.load(audio_file, sr=None, duration=0.1)
        return sr > 0
    except Exception as e:
        logger.debug(f"Audio validation error: {str(e)}")
        return False

def convert_audio_format(
    audio_data: bytes, 
    source_format: Optional[str] = None, 
    target_format: str = "wav", 
    sample_rate: Optional[int] = None,
    channels: int = 1
) -> Optional[bytes]:
    """
    Convert audio to target format.
    
    Args:
        audio_data: Audio data as bytes
        source_format: Source audio format (optional, auto-detected if None)
        target_format: Target audio format (default: wav)
        sample_rate: Target sample rate (optional, preserved if None)
        channels: Target number of channels (default: 1 for mono)
        
    Returns:
        bytes: Converted audio data or None if conversion failed
    """
    try:
        # Load audio
        audio_file = io.BytesIO(audio_data)
        audio, sr = librosa.load(audio_file, sr=sample_rate, mono=(channels == 1))
        
        # Create output buffer
        output_buffer = io.BytesIO()
        
        # Write to target format
        sf.write(output_buffer, audio, sr if sample_rate is None else sample_rate, format=target_format)
        
        # Get bytes
        output_buffer.seek(0)
        return output_buffer.read()
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        return None

def get_audio_length(audio_data: bytes) -> Optional[float]:
    """
    Calculate the length of an audio file in seconds.
    
    Args:
        audio_data: Audio data as bytes
        
    Returns:
        float: Length in seconds or None if calculation failed
    """
    try:
        audio_file = io.BytesIO(audio_data)
        audio, sr = librosa.load(audio_file, sr=None)
        return len(audio) / sr
    except Exception as e:
        logger.error(f"Audio length calculation error: {str(e)}")
        return None

def get_audio_info(audio_data: bytes) -> Optional[dict]:
    """
    Get information about an audio file.
    
    Args:
        audio_data: Audio data as bytes
        
    Returns:
        dict: Audio information or None if extraction failed
    """
    try:
        audio_file = io.BytesIO(audio_data)
        audio, sr = librosa.load(audio_file, sr=None)
        
        return {
            "sample_rate": sr,
            "channels": 1 if audio.ndim == 1 else audio.shape[1],
            "length": len(audio) / sr,
            "format": "wav"  # Default format after librosa loading
        }
    except Exception as e:
        logger.error(f"Audio info extraction error: {str(e)}")
        return None
