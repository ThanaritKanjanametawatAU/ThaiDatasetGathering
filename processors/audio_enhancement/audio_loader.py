"""
Enhanced audio loader and preprocessor with format detection, validation, and caching.
"""

import os
import logging
from typing import Optional, Tuple, Union, Dict, Any, List, Generator
from pathlib import Path
import io
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from functools import lru_cache

import numpy as np
import librosa
import soundfile as sf
import audioread
from scipy import signal

logger = logging.getLogger(__name__)


# Custom exceptions
class AudioLoadError(Exception):
    """Base exception for audio loading errors."""
    pass


class UnsupportedFormatError(AudioLoadError):
    """File format not supported."""
    def __init__(self, filepath, detected_format=None):
        self.filepath = filepath
        self.format = detected_format
        super().__init__(f"Unsupported format: {detected_format} for file: {filepath}")


class CorruptedFileError(AudioLoadError):
    """Audio file is corrupted."""
    pass


class PreprocessingError(AudioLoadError):
    """Error during preprocessing."""
    pass


class AudioCache:
    """LRU cache for audio files with size limits."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize audio cache.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Tuple[np.ndarray, int]]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
        return None
        
    def set(self, key: str, value: Tuple[np.ndarray, int]):
        """Set item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove oldest
                    self.cache.popitem(last=False)
            self.cache[key] = value
            
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()


class AudioValidator:
    """Validates audio files and data."""
    
    def __init__(self):
        """Initialize validator."""
        self.supported_formats = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac'}
        
    def validate_duration(
        self, 
        audio: np.ndarray, 
        sr: int, 
        min_duration: float = 0.01,
        max_duration: float = 3600.0
    ) -> bool:
        """
        Validate audio duration.
        
        Args:
            audio: Audio array
            sr: Sample rate
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            bool: True if valid duration
        """
        duration = len(audio) / sr
        return min_duration <= duration <= max_duration
        
    def validate_amplitude(
        self, 
        audio: np.ndarray,
        min_amplitude: float = 0.001,
        check_clipping: bool = True
    ) -> bool:
        """
        Validate audio amplitude.
        
        Args:
            audio: Audio array
            min_amplitude: Minimum RMS amplitude
            check_clipping: Check for clipping
            
        Returns:
            bool: True if valid amplitude
        """
        # Check if not silent
        rms = np.sqrt(np.mean(audio**2))
        if rms < min_amplitude:
            return False
            
        # Check for clipping
        if check_clipping:
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
            clipping_ratio = clipped_samples / len(audio)
            if clipping_ratio > 0.01:  # More than 1% clipped
                return False
                
        return True
        
    def validate_format(self, filepath: str) -> bool:
        """
        Validate file format.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            bool: True if supported format
        """
        ext = Path(filepath).suffix.lower()
        return ext in self.supported_formats
        
    def check_corruption(self, audio: np.ndarray) -> bool:
        """
        Check if audio data is corrupted.
        
        Args:
            audio: Audio array
            
        Returns:
            bool: True if corrupted
        """
        # Check for NaN or Inf values
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return True
            
        # Check for extreme values
        if np.any(np.abs(audio) > 10.0):  # Unreasonably high values
            return True
            
        return False


class AudioPreprocessor:
    """Handles audio preprocessing operations."""
    
    def __init__(self):
        """Initialize preprocessor."""
        pass
        
    def convert_sample_rate(
        self, 
        audio: np.ndarray, 
        source_sr: int, 
        target_sr: int,
        res_type: str = 'kaiser_best'
    ) -> np.ndarray:
        """
        Convert sample rate using high-quality resampling.
        
        Args:
            audio: Input audio
            source_sr: Source sample rate
            target_sr: Target sample rate
            res_type: Resampling type
            
        Returns:
            np.ndarray: Resampled audio
        """
        if source_sr == target_sr:
            return audio
            
        # Use librosa for high-quality resampling
        resampled = librosa.resample(
            audio, 
            orig_sr=source_sr, 
            target_sr=target_sr,
            res_type=res_type
        )
        
        return resampled
        
    def normalize_channels(
        self, 
        audio: np.ndarray, 
        target_channels: int = 1,
        method: str = 'average'
    ) -> np.ndarray:
        """
        Normalize number of channels.
        
        Args:
            audio: Input audio
            target_channels: Target number of channels
            method: Method for stereo to mono conversion
            
        Returns:
            np.ndarray: Channel-normalized audio
        """
        if audio.ndim == 1 and target_channels == 1:
            return audio
            
        if audio.ndim == 2 and target_channels == 1:
            # Stereo to mono
            if method == 'average':
                return np.mean(audio, axis=1)
            elif method == 'left':
                return audio[:, 0]
            elif method == 'right':
                return audio[:, 1]
            elif method == 'max':
                return np.max(audio, axis=1)
            else:
                return np.mean(audio, axis=1)
                
        if audio.ndim == 1 and target_channels == 2:
            # Mono to stereo (duplicate)
            return np.stack([audio, audio], axis=1)
            
        return audio
        
    def normalize_amplitude(
        self, 
        audio: np.ndarray,
        method: str = 'peak',
        target_rms: float = 0.1,
        target_db: float = -20.0
    ) -> np.ndarray:
        """
        Normalize audio amplitude.
        
        Args:
            audio: Input audio
            method: Normalization method ('peak', 'rms', 'lufs')
            target_rms: Target RMS value
            target_db: Target dB level
            
        Returns:
            np.ndarray: Amplitude-normalized audio
        """
        if method == 'peak':
            # Peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio / peak
            return audio
            
        elif method == 'rms':
            # RMS normalization
            current_rms = np.sqrt(np.mean(audio**2))
            if current_rms > 0:
                return audio * (target_rms / current_rms)
            return audio
            
        elif method == 'db':
            # dB normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_amplitude = 10**(target_db / 20.0)
                scaling_factor = target_amplitude / rms
                return np.clip(audio * scaling_factor, -1.0, 1.0)
            return audio
            
        return audio
        
    def trim_silence(
        self, 
        audio: np.ndarray, 
        sr: int,
        top_db: int = 30,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Trim silence from beginning and end.
        
        Args:
            audio: Input audio
            sr: Sample rate
            top_db: Threshold in dB below peak
            frame_length: Frame length for energy calculation
            hop_length: Hop length for energy calculation
            
        Returns:
            np.ndarray: Trimmed audio
        """
        # Return original if too short
        if len(audio) < frame_length:
            return audio
            
        trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Ensure we don't return empty audio
        if len(trimmed) == 0:
            return audio
            
        return trimmed
        
    def process(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int = 16000,
        target_channels: int = 1,
        normalize: bool = True,
        trim_silence: bool = True,
        normalize_method: str = 'db',
        target_db: float = -20.0
    ) -> np.ndarray:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            audio: Input audio
            source_sr: Source sample rate
            target_sr: Target sample rate
            target_channels: Target number of channels
            normalize: Whether to normalize amplitude
            trim_silence: Whether to trim silence
            normalize_method: Normalization method
            target_db: Target dB level
            
        Returns:
            np.ndarray: Preprocessed audio
        """
        # Convert sample rate
        if source_sr != target_sr:
            audio = self.convert_sample_rate(audio, source_sr, target_sr)
            
        # Normalize channels
        audio = self.normalize_channels(audio, target_channels)
        
        # Trim silence
        if trim_silence:
            audio = self.trim_silence(audio, target_sr)
            
        # Normalize amplitude
        if normalize:
            audio = self.normalize_amplitude(
                audio, 
                method=normalize_method,
                target_db=target_db
            )
            
        return audio


class AudioLoader:
    """Enhanced audio loader with format detection and fallback mechanisms."""
    
    def __init__(self, cache: Optional[AudioCache] = None):
        """
        Initialize audio loader.
        
        Args:
            cache: Optional audio cache instance
        """
        self.cache = cache or AudioCache()
        self.validator = AudioValidator()
        self.preprocessor = AudioPreprocessor()
        
    def detect_format(self, filepath: str) -> Optional[str]:
        """
        Detect audio format from file header.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            str: Detected format or None
        """
        try:
            with open(filepath, 'rb') as f:
                header = f.read(512)
                
            # Check magic numbers
            if header.startswith(b'RIFF') and b'WAVE' in header[:12]:
                return 'wav'
            elif header.startswith(b'fLaC'):
                return 'flac'
            elif header.startswith(b'OggS'):
                return 'ogg'
            elif header.startswith(b'ID3') or header[0:2] == b'\xff\xfb':
                return 'mp3'
            elif b'ftyp' in header[:12]:
                return 'm4a'
                
        except Exception:
            pass
            
        # Fallback to extension
        ext = Path(filepath).suffix.lower()[1:]
        return ext if ext else None
        
    def _load_with_librosa(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio using librosa."""
        audio, sr = librosa.load(filepath, sr=None, mono=False)
        if audio.ndim > 1:
            audio = audio.T  # Transpose to (samples, channels)
        return audio, sr
        
    def _load_with_soundfile(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile."""
        audio, sr = sf.read(filepath, always_2d=False)
        return audio, sr
        
    def _load_with_audioread(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio using audioread."""
        with audioread.audio_open(filepath) as f:
            sr = f.samplerate
            channels = f.channels
            duration = f.duration
            
            # Read all data
            data = []
            for buf in f:
                data.append(buf)
                
            # Convert to numpy array
            audio = np.frombuffer(b''.join(data), dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            
            if channels > 1:
                audio = audio.reshape(-1, channels)
                
        return audio, sr
        
    def _load_with_ffmpeg(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio using ffmpeg (via librosa with audioread backend)."""
        # This uses ffmpeg through audioread
        audio, sr = librosa.load(filepath, sr=None, mono=False)
        if audio.ndim > 1:
            audio = audio.T
        return audio, sr
        
    def _load_with_fallback(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio with fallback chain.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio, sample_rate)
        """
        loaders = [
            ('librosa', self._load_with_librosa),
            ('soundfile', self._load_with_soundfile),
            ('audioread', self._load_with_audioread),
            ('ffmpeg', self._load_with_ffmpeg)
        ]
        
        last_error = None
        for loader_name, loader_func in loaders:
            try:
                logger.debug(f"Trying to load with {loader_name}")
                audio, sr = loader_func(filepath)
                
                # Validate that we got valid data
                if audio is None or sr is None or sr <= 0:
                    raise ValueError(f"Invalid audio data returned by {loader_name}")
                    
                return audio, sr
            except Exception as e:
                logger.debug(f"{loader_name} failed: {str(e)}")
                last_error = e
                continue
                
        # Check if file is corrupted
        try:
            with open(filepath, 'rb') as f:
                header = f.read(512)
                if header.startswith(b'RIFF') and len(header) < 44:
                    raise CorruptedFileError(f"Corrupted WAV file: {filepath}")
        except:
            pass
            
        raise AudioLoadError(f"All loaders failed. Last error: {last_error}")
        
    def load_audio(
        self, 
        filepath: str,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with format detection and caching.
        
        Args:
            filepath: Path to audio file
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (audio, sample_rate)
        """
        # Check cache
        if use_cache:
            cached = self.cache.get(filepath)
            if cached is not None:
                logger.debug(f"Loaded from cache: {filepath}")
                return cached
                
        # Validate format
        if not self.validator.validate_format(filepath):
            detected_format = self.detect_format(filepath)
            raise UnsupportedFormatError(filepath, detected_format)
            
        # Check file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
            
        # Load with fallback chain
        try:
            audio, sr = self._load_with_fallback(filepath)
        except CorruptedFileError:
            # Re-raise corruption errors directly
            raise
        except Exception as e:
            # Check if it's a corrupted file by examining the error
            error_msg = str(e).lower()
            if 'corrupt' in error_msg or 'invalid' in error_msg or 'bad' in error_msg:
                raise CorruptedFileError(f"Audio file is corrupted: {filepath}")
            raise AudioLoadError(f"Failed to load audio: {str(e)}")
            
        # Validate audio data
        if self.validator.check_corruption(audio):
            raise CorruptedFileError(f"Audio file is corrupted: {filepath}")
            
        # Cache result
        if use_cache:
            self.cache.set(filepath, (audio, sr))
            
        return audio, sr
        
    def load_audio_streaming(
        self, 
        filepath: str,
        chunk_size: int = 1024*1024,
        overlap: int = 0
    ) -> Generator[np.ndarray, None, None]:
        """
        Load audio in chunks for memory efficiency.
        
        Args:
            filepath: Path to audio file
            chunk_size: Size of each chunk in samples
            overlap: Number of overlapping samples
            
        Yields:
            Audio chunks
        """
        with sf.SoundFile(filepath) as f:
            sr = f.samplerate
            channels = f.channels
            
            while True:
                chunk = f.read(chunk_size, dtype='float32')
                
                if len(chunk) == 0:
                    break
                    
                yield chunk
                
                # Seek back for overlap
                if overlap > 0 and len(chunk) == chunk_size:
                    f.seek(f.tell() - overlap)
                    
    def get_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Extract metadata from audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            dict: Audio metadata
        """
        try:
            info = sf.info(filepath)
            
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'frames': info.frames
            }
        except Exception as e:
            # Fallback to loading audio
            audio, sr = self.load_audio(filepath, use_cache=False)
            
            return {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'channels': 1 if audio.ndim == 1 else audio.shape[1],
                'format': self.detect_format(filepath),
                'frames': len(audio)
            }
            
    def load_batch(
        self, 
        filepaths: List[str],
        num_workers: int = 8
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Load multiple audio files in parallel.
        
        Args:
            filepaths: List of file paths
            num_workers: Number of parallel workers
            
        Returns:
            List of (audio, sample_rate) tuples
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.load_audio, path): path 
                for path in filepaths
            }
            
            # Collect results in order
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {str(e)}")
                    results.append((None, None))
                    
        return results
        
    def warm_cache(self, filepaths: List[str], num_workers: int = 8):
        """
        Pre-load files into cache.
        
        Args:
            filepaths: List of file paths to cache
            num_workers: Number of parallel workers
        """
        logger.info(f"Warming cache with {len(filepaths)} files")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.load_audio, path, use_cache=True)
                for path in filepaths
            ]
            
            # Wait for all to complete
            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                    if (i + 1) % 100 == 0:
                        logger.info(f"Cached {i + 1}/{len(filepaths)} files")
                except Exception as e:
                    logger.warning(f"Failed to cache file: {str(e)}")
                    
        logger.info("Cache warming complete")