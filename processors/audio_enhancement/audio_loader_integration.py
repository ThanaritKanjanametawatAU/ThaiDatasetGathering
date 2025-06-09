"""
Integration module to bridge the enhanced audio loader with existing codebase.
"""

import logging
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
import io

from processors.audio_enhancement.audio_loader import (
    AudioLoader,
    AudioPreprocessor,
    AudioValidator,
    AudioCache
)
from utils.audio import (
    is_valid_audio,
    get_audio_length,
    get_audio_info
)
from config import AUDIO_CONFIG

logger = logging.getLogger(__name__)


class EnhancedAudioProcessor:
    """
    Enhanced audio processor that integrates the new audio loader
    with the existing audio utilities.
    """
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize enhanced audio processor.
        
        Args:
            cache_size: Maximum number of cached audio files
        """
        self.cache = AudioCache(max_size=cache_size)
        self.loader = AudioLoader(cache=self.cache)
        self.preprocessor = AudioPreprocessor()
        self.validator = AudioValidator()
        
        # Load configuration
        self.config = AUDIO_CONFIG
        
    def load_and_preprocess(
        self,
        filepath: str,
        target_sr: Optional[int] = None,
        target_channels: Optional[int] = None,
        normalize: bool = True,
        trim_silence: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.
        
        Args:
            filepath: Path to audio file
            target_sr: Target sample rate (default from config)
            target_channels: Target channels (default from config)
            normalize: Whether to normalize amplitude
            trim_silence: Whether to trim silence
            
        Returns:
            Tuple of (preprocessed_audio, sample_rate)
        """
        # Use config defaults if not specified
        target_sr = target_sr or self.config.get('target_sample_rate', 16000)
        target_channels = target_channels or self.config.get('target_channels', 1)
        
        # Load audio
        audio, sr = self.loader.load_audio(filepath)
        
        # Preprocess
        processed = self.preprocessor.process(
            audio=audio,
            source_sr=sr,
            target_sr=target_sr,
            target_channels=target_channels,
            normalize=normalize,
            trim_silence=trim_silence,
            normalize_method=self.config.get('normalize_method', 'db'),
            target_db=self.config.get('target_db', -20.0)
        )
        
        return processed, target_sr
        
    def load_from_bytes(
        self,
        audio_bytes: bytes,
        source_format: Optional[str] = None,
        preprocess: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes (compatible with existing code).
        
        Args:
            audio_bytes: Audio data as bytes
            source_format: Source format hint
            preprocess: Whether to apply preprocessing
            
        Returns:
            Tuple of (audio, sample_rate)
        """
        # Save to temporary file for processing
        import tempfile
        
        ext = f".{source_format}" if source_format else ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            if preprocess:
                return self.load_and_preprocess(tmp_path)
            else:
                return self.loader.load_audio(tmp_path)
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    def validate_audio(
        self,
        audio: Union[np.ndarray, bytes, dict],
        check_duration: bool = True,
        check_amplitude: bool = True
    ) -> bool:
        """
        Validate audio data (compatible with existing is_valid_audio).
        
        Args:
            audio: Audio data in various formats
            check_duration: Whether to check duration limits
            check_amplitude: Whether to check amplitude
            
        Returns:
            bool: True if valid
        """
        try:
            # Handle different input formats
            if isinstance(audio, dict):
                # HuggingFace format
                if "array" in audio and "sampling_rate" in audio:
                    audio_array = np.array(audio["array"])
                    sr = audio["sampling_rate"]
                else:
                    return False
            elif isinstance(audio, bytes):
                # Load from bytes
                audio_array, sr = self.load_from_bytes(audio, preprocess=False)
            elif isinstance(audio, np.ndarray):
                # Assume standard sample rate if not provided
                audio_array = audio
                sr = self.config.get('target_sample_rate', 16000)
            else:
                return False
                
            # Check corruption
            if self.validator.check_corruption(audio_array):
                return False
                
            # Check duration
            if check_duration:
                if not self.validator.validate_duration(
                    audio_array, 
                    sr,
                    min_duration=self.config.get('min_duration', 0.1),
                    max_duration=self.config.get('max_duration', 3600)
                ):
                    return False
                    
            # Check amplitude
            if check_amplitude:
                if not self.validator.validate_amplitude(audio_array):
                    return False
                    
            return True
            
        except Exception as e:
            logger.debug(f"Audio validation failed: {str(e)}")
            return False
            
    def get_audio_metadata(
        self,
        filepath: str
    ) -> Dict[str, Any]:
        """
        Get enhanced metadata for audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            dict: Audio metadata
        """
        # Get basic metadata
        metadata = self.loader.get_metadata(filepath)
        
        # Load audio for additional analysis
        try:
            audio, sr = self.loader.load_audio(filepath)
            
            # Add quality metrics
            metadata.update({
                'rms_amplitude': float(np.sqrt(np.mean(audio**2))),
                'max_amplitude': float(np.max(np.abs(audio))),
                'db_level': float(20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-8)),
                'has_clipping': np.any(np.abs(audio) > 0.95),
                'is_silent': np.sqrt(np.mean(audio**2)) < 0.001,
                'estimated_snr': self._estimate_snr(audio, sr)
            })
            
        except Exception as e:
            logger.warning(f"Could not analyze audio quality: {str(e)}")
            
        return metadata
        
    def _estimate_snr(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """
        Estimate SNR of audio signal.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            float: Estimated SNR in dB
        """
        try:
            # Simple SNR estimation using frame-based approach
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Calculate frame energies
            energies = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy = np.mean(frame**2)
                if energy > 0:
                    energies.append(energy)
                    
            if len(energies) < 10:
                return None
                
            energies = np.array(energies)
            
            # Estimate noise as lower percentile
            noise_energy = np.percentile(energies, 10)
            
            # Estimate signal as upper percentile
            signal_energy = np.percentile(energies, 90)
            
            # Calculate SNR
            if noise_energy > 0:
                snr = 10 * np.log10(signal_energy / noise_energy)
                return float(snr)
            else:
                return float('inf')
                
        except Exception:
            return None


# Create a singleton instance for backward compatibility
_enhanced_processor = None


def get_enhanced_audio_processor() -> EnhancedAudioProcessor:
    """Get or create the enhanced audio processor instance."""
    global _enhanced_processor
    if _enhanced_processor is None:
        _enhanced_processor = EnhancedAudioProcessor()
    return _enhanced_processor


# Backward-compatible functions that use the enhanced loader
def load_audio(
    filepath: str,
    sr: Optional[int] = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio file (backward compatible with librosa.load interface).
    
    Args:
        filepath: Path to audio file
        sr: Target sample rate (None to preserve original)
        mono: Convert to mono
        offset: Start reading after this time (seconds)
        duration: Only load this much audio (seconds)
        
    Returns:
        Tuple of (audio, sample_rate)
    """
    processor = get_enhanced_audio_processor()
    
    # Load full audio first
    audio, orig_sr = processor.loader.load_audio(filepath)
    
    # Apply offset and duration
    if offset > 0 or duration is not None:
        start_sample = int(offset * orig_sr)
        if duration is not None:
            end_sample = start_sample + int(duration * orig_sr)
            audio = audio[start_sample:end_sample]
        else:
            audio = audio[start_sample:]
            
    # Convert to mono if needed
    if mono and audio.ndim > 1:
        audio = processor.preprocessor.normalize_channels(audio, target_channels=1)
        
    # Resample if needed
    if sr is not None and sr != orig_sr:
        audio = processor.preprocessor.convert_sample_rate(audio, orig_sr, sr)
        return audio, sr
    else:
        return audio, orig_sr


def standardize_audio(
    audio_data: bytes,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    normalize_volume: bool = True,
    target_db: float = -20.0
) -> Optional[bytes]:
    """
    Standardize audio to common format (backward compatible).
    
    Args:
        audio_data: Input audio data as bytes
        target_sample_rate: Target sample rate
        target_channels: Target number of channels
        normalize_volume: Whether to normalize volume
        target_db: Target dB level
        
    Returns:
        bytes: Standardized audio data or None if failed
    """
    try:
        processor = get_enhanced_audio_processor()
        
        # Load and preprocess
        audio, sr = processor.load_from_bytes(
            audio_data,
            preprocess=True
        )
        
        # Convert back to bytes
        import soundfile as sf
        output_buffer = io.BytesIO()
        sf.write(output_buffer, audio, target_sample_rate, format='WAV', subtype='PCM_16')
        output_buffer.seek(0)
        
        return output_buffer.read()
        
    except Exception as e:
        logger.error(f"Audio standardization failed: {str(e)}")
        return None