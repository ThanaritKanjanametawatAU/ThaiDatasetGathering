"""
Advanced Edge Case Handling Module (S05_T19)
Comprehensive edge case handling for 10M+ diverse audio samples with intelligent error recovery
"""

import os
import logging
import subprocess
import tempfile
import shutil
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import librosa
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors encountered during processing"""
    CORRUPTED_AUDIO = "corrupted_audio"
    UNSUPPORTED_FORMAT = "unsupported_format"
    ENCODING_ERROR = "encoding_error"
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    LANGUAGE_MISMATCH = "language_mismatch"
    SILENCE_DETECTED = "silence_detected"
    LOW_QUALITY = "low_quality"
    METADATA_ERROR = "metadata_error"
    UNKNOWN_ERROR = "unknown_error"


class RetryStrategy(Enum):
    """Different retry strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"
    ESCALATION = "escalation"


class RecoveryMethod(Enum):
    """Audio recovery methods"""
    FFMPEG_REPAIR = "ffmpeg_repair"
    LIBROSA_FALLBACK = "librosa_fallback"
    SEGMENT_EXTRACTION = "segment_extraction"
    FORMAT_CONVERSION = "format_conversion"
    TRIM_SILENCE = "trim_silence"
    NOISE_REDUCTION = "noise_reduction"


@dataclass
class ErrorInfo:
    """Information about an encountered error"""
    error_type: str
    error_message: str
    category: ErrorCategory
    timestamp: datetime = field(default_factory=datetime.now)
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_method: Optional[RecoveryMethod] = None
    recovery_successful: bool = False
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryResult:
    """Result of audio recovery attempt"""
    successful: bool
    method_used: RecoveryMethod
    output_path: Optional[str] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    detected_languages: Dict[str, float]  # language -> confidence
    thai_confidence: float
    thai_segments: List[Tuple[float, float]] = field(default_factory=list)  # (start, end) times
    total_duration: float = 0.0
    thai_duration: float = 0.0
    is_primarily_thai: bool = False


class AudioRecoveryEngine(ABC):
    """Abstract base class for audio recovery engines"""
    
    @abstractmethod
    def can_handle(self, error_info: ErrorInfo) -> bool:
        """Check if this engine can handle the given error"""
        pass
    
    @abstractmethod
    def recover(self, file_path: str, error_info: ErrorInfo) -> RecoveryResult:
        """Attempt to recover the audio file"""
        pass


class FFmpegRecoveryEngine(AudioRecoveryEngine):
    """FFmpeg-based audio recovery engine"""
    
    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()
        self.supported_formats = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma']
    
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable"""
        try:
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Common locations
        common_paths = ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', 'ffmpeg']
        for path in common_paths:
            try:
                subprocess.run([path, '-version'], capture_output=True, check=True)
                return path
            except:
                continue
        
        logger.warning("FFmpeg not found. Audio recovery capabilities limited.")
        return None
    
    def can_handle(self, error_info: ErrorInfo) -> bool:
        """Check if FFmpeg can handle this error"""
        if not self.ffmpeg_path:
            return False
        
        return error_info.category in [
            ErrorCategory.CORRUPTED_AUDIO,
            ErrorCategory.UNSUPPORTED_FORMAT,
            ErrorCategory.ENCODING_ERROR
        ]
    
    def recover(self, file_path: str, error_info: ErrorInfo) -> RecoveryResult:
        """Attempt audio recovery using FFmpeg"""
        start_time = time.time()
        
        if not self.ffmpeg_path:
            return RecoveryResult(
                successful=False,
                method_used=RecoveryMethod.FFMPEG_REPAIR,
                error_message="FFmpeg not available",
                processing_time=time.time() - start_time
            )
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Try different recovery strategies
            strategies = [
                self._repair_corrupted,
                self._convert_format,
                self._extract_valid_segments
            ]
            
            for strategy in strategies:
                result = strategy(file_path, temp_dir, error_info)
                if result.successful:
                    result.processing_time = time.time() - start_time
                    return result
            
            return RecoveryResult(
                successful=False,
                method_used=RecoveryMethod.FFMPEG_REPAIR,
                error_message="All FFmpeg recovery strategies failed",
                processing_time=time.time() - start_time
            )
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _repair_corrupted(self, file_path: str, temp_dir: str, error_info: ErrorInfo) -> RecoveryResult:
        """Attempt to repair corrupted audio"""
        output_path = os.path.join(temp_dir, f"repaired_{os.path.basename(file_path)}")
        
        try:
            # Use FFmpeg with error correction flags
            cmd = [
                self.ffmpeg_path,
                '-i', file_path,
                '-c:a', 'pcm_s16le',  # Convert to standard PCM
                '-ar', '16000',       # Standard sample rate
                '-ac', '1',           # Mono
                '-f', 'wav',          # WAV format
                '-y',                 # Overwrite output
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path):
                # Verify the output
                quality_score = self._assess_quality(output_path)
                
                return RecoveryResult(
                    successful=True,
                    method_used=RecoveryMethod.FFMPEG_REPAIR,
                    output_path=output_path,
                    quality_score=quality_score,
                    metadata={"ffmpeg_cmd": cmd, "ffmpeg_output": result.stderr}
                )
        
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg repair timeout for {file_path}")
        except Exception as e:
            logger.warning(f"FFmpeg repair failed for {file_path}: {e}")
        
        return RecoveryResult(
            successful=False,
            method_used=RecoveryMethod.FFMPEG_REPAIR,
            error_message="Repair attempt failed"
        )
    
    def _convert_format(self, file_path: str, temp_dir: str, error_info: ErrorInfo) -> RecoveryResult:
        """Convert to a more compatible format"""
        output_path = os.path.join(temp_dir, f"converted_{Path(file_path).stem}.wav")
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', file_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path):
                quality_score = self._assess_quality(output_path)
                
                return RecoveryResult(
                    successful=True,
                    method_used=RecoveryMethod.FORMAT_CONVERSION,
                    output_path=output_path,
                    quality_score=quality_score,
                    metadata={"original_format": Path(file_path).suffix}
                )
        
        except Exception as e:
            logger.warning(f"Format conversion failed for {file_path}: {e}")
        
        return RecoveryResult(
            successful=False,
            method_used=RecoveryMethod.FORMAT_CONVERSION,
            error_message="Format conversion failed"
        )
    
    def _extract_valid_segments(self, file_path: str, temp_dir: str, error_info: ErrorInfo) -> RecoveryResult:
        """Extract valid segments from partially corrupted file"""
        try:
            # First, get file info
            cmd = [self.ffmpeg_path, '-i', file_path, '-f', 'null', '-']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Look for duration info in stderr
            duration = self._extract_duration(result.stderr)
            if not duration or duration < 1.0:
                return RecoveryResult(
                    successful=False,
                    method_used=RecoveryMethod.SEGMENT_EXTRACTION,
                    error_message="No valid duration found"
                )
            
            # Extract first valid portion (up to 50% of file)
            extract_duration = min(duration * 0.5, 60.0)  # Max 60 seconds
            output_path = os.path.join(temp_dir, f"segment_{Path(file_path).stem}.wav")
            
            cmd = [
                self.ffmpeg_path,
                '-i', file_path,
                '-t', str(extract_duration),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path):
                quality_score = self._assess_quality(output_path)
                
                return RecoveryResult(
                    successful=True,
                    method_used=RecoveryMethod.SEGMENT_EXTRACTION,
                    output_path=output_path,
                    quality_score=quality_score,
                    metadata={"extracted_duration": extract_duration, "original_duration": duration}
                )
        
        except Exception as e:
            logger.warning(f"Segment extraction failed for {file_path}: {e}")
        
        return RecoveryResult(
            successful=False,
            method_used=RecoveryMethod.SEGMENT_EXTRACTION,
            error_message="Segment extraction failed"
        )
    
    def _extract_duration(self, ffmpeg_output: str) -> Optional[float]:
        """Extract duration from FFmpeg output"""
        try:
            lines = ffmpeg_output.split('\\n')
            for line in lines:
                if 'Duration:' in line:
                    # Extract time format HH:MM:SS.ms
                    time_str = line.split('Duration:')[1].split(',')[0].strip()
                    time_parts = time_str.split(':')
                    if len(time_parts) == 3:
                        hours = float(time_parts[0])
                        minutes = float(time_parts[1])
                        seconds = float(time_parts[2])
                        return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return None
    
    def _assess_quality(self, file_path: str) -> float:
        """Assess quality of recovered audio"""
        try:
            # Use FFmpeg to get audio statistics
            cmd = [
                self.ffmpeg_path,
                '-i', file_path,
                '-af', 'astats=metadata=1:reset=1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Parse audio statistics for quality indicators
            quality_score = 0.5  # Default score
            
            if 'RMS level' in result.stderr:
                quality_score += 0.2
            if 'Peak level' in result.stderr:
                quality_score += 0.2
            if 'Duration' in result.stderr:
                quality_score += 0.1
            
            return min(quality_score, 1.0)
        
        except:
            return 0.3  # Conservative score if assessment fails


class LibrosaRecoveryEngine(AudioRecoveryEngine):
    """Librosa-based audio recovery engine"""
    
    def can_handle(self, error_info: ErrorInfo) -> bool:
        """Check if librosa can handle this error"""
        return error_info.category in [
            ErrorCategory.CORRUPTED_AUDIO,
            ErrorCategory.LOW_QUALITY,
            ErrorCategory.SILENCE_DETECTED
        ]
    
    def recover(self, file_path: str, error_info: ErrorInfo) -> RecoveryResult:
        """Attempt audio recovery using librosa"""
        start_time = time.time()
        
        try:
            # Try to load with different parameters
            strategies = [
                lambda: librosa.load(file_path, sr=16000, mono=True),
                lambda: librosa.load(file_path, sr=22050, mono=True),
                lambda: librosa.load(file_path, sr=None, mono=True),
                lambda: librosa.load(file_path, sr=16000, mono=True, offset=1.0),  # Skip first second
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    audio, sr = strategy()
                    
                    if len(audio) > 0:
                        # Process the audio
                        processed_audio = self._process_audio(audio, sr)
                        
                        # Save to temporary file
                        temp_dir = tempfile.mkdtemp()
                        output_path = os.path.join(temp_dir, f"recovered_{Path(file_path).stem}.wav")
                        
                        import soundfile as sf
                        sf.write(output_path, processed_audio, 16000)
                        
                        quality_score = self._assess_audio_quality(processed_audio, 16000)
                        
                        return RecoveryResult(
                            successful=True,
                            method_used=RecoveryMethod.LIBROSA_FALLBACK,
                            output_path=output_path,
                            quality_score=quality_score,
                            metadata={"strategy_used": i, "sample_rate": sr},
                            processing_time=time.time() - start_time
                        )
                
                except Exception as e:
                    logger.debug(f"Librosa strategy {i} failed: {e}")
                    continue
            
            return RecoveryResult(
                successful=False,
                method_used=RecoveryMethod.LIBROSA_FALLBACK,
                error_message="All librosa strategies failed",
                processing_time=time.time() - start_time
            )
        
        except Exception as e:
            return RecoveryResult(
                successful=False,
                method_used=RecoveryMethod.LIBROSA_FALLBACK,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _process_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio to improve quality"""
        # Trim silence
        audio = librosa.effects.trim(audio, top_db=20)[0]
        
        # Normalize
        if len(audio) > 0:
            audio = librosa.util.normalize(audio)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        return audio
    
    def _assess_audio_quality(self, audio: np.ndarray, sr: int) -> float:
        """Assess quality of recovered audio"""
        if len(audio) == 0:
            return 0.0
        
        quality_score = 0.0
        
        # Check RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        avg_rms = np.mean(rms)
        if avg_rms > 0.001:  # Has some energy
            quality_score += 0.3
        
        # Check spectral content
        stft = librosa.stft(audio)
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
        if np.mean(spectral_centroid) > 500:  # Has higher frequency content
            quality_score += 0.3
        
        # Check duration
        duration = len(audio) / sr
        if duration > 1.0:  # At least 1 second
            quality_score += 0.2
        
        # Check for silence
        silence_ratio = np.sum(np.abs(audio) < 0.01) / len(audio)
        if silence_ratio < 0.8:  # Less than 80% silence
            quality_score += 0.2
        
        return min(quality_score, 1.0)


class LanguageDetector:
    """Multi-language detection with Thai extraction capabilities"""
    
    def __init__(self):
        self.whisper_model = None
        self._initialize_whisper()
    
    def _initialize_whisper(self):
        """Initialize Whisper model for language detection"""
        try:
            import whisper
            # Use smallest model for language detection
            self.whisper_model = whisper.load_model("tiny")
            logger.info("Whisper model loaded for language detection")
        except ImportError:
            logger.warning("Whisper not available. Language detection capabilities limited.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
    
    def detect_language(self, audio_path: str) -> LanguageDetectionResult:
        """Detect languages in audio file with focus on Thai content"""
        if not self.whisper_model:
            return self._fallback_detection(audio_path)
        
        try:
            # Load and transcribe with language detection
            result = self.whisper_model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                task="transcribe",
                fp16=False
            )
            
            detected_language = result.get("language", "unknown")
            
            # Analyze segments for Thai content
            thai_segments = []
            total_duration = 0.0
            thai_duration = 0.0
            
            segments = result.get("segments", [])
            for segment in segments:
                start = segment.get("start", 0.0)
                end = segment.get("end", 0.0)
                text = segment.get("text", "")
                seg_language = segment.get("language", detected_language)
                
                duration = end - start
                total_duration = max(total_duration, end)
                
                # Check if segment contains Thai content
                if self._is_thai_content(text, seg_language):
                    thai_segments.append((start, end))
                    thai_duration += duration
            
            # Calculate confidence scores
            thai_confidence = self._calculate_thai_confidence(result, thai_duration, total_duration)
            
            # Determine if primarily Thai
            is_primarily_thai = (
                detected_language == "th" or 
                thai_confidence > 0.7 or 
                (thai_duration / total_duration > 0.5 if total_duration > 0 else False)
            )
            
            return LanguageDetectionResult(
                detected_languages={detected_language: 1.0},  # Whisper gives primary language
                thai_confidence=thai_confidence,
                thai_segments=thai_segments,
                total_duration=total_duration,
                thai_duration=thai_duration,
                is_primarily_thai=is_primarily_thai
            )
        
        except Exception as e:
            logger.error(f"Whisper language detection failed: {e}")
            return self._fallback_detection(audio_path)
    
    def _is_thai_content(self, text: str, language: str) -> bool:
        """Check if text contains Thai content"""
        if language == "th":
            return True
        
        # Check for Thai unicode characters
        thai_chars = sum(1 for char in text if '\\u0e00' <= char <= '\\u0e7f')
        return thai_chars > len(text) * 0.3  # At least 30% Thai characters
    
    def _calculate_thai_confidence(self, whisper_result: dict, thai_duration: float, total_duration: float) -> float:
        """Calculate confidence that audio is Thai"""
        confidence = 0.0
        
        # Language detection confidence
        if whisper_result.get("language") == "th":
            confidence += 0.5
        
        # Duration ratio
        if total_duration > 0:
            duration_ratio = thai_duration / total_duration
            confidence += duration_ratio * 0.3
        
        # Text analysis
        segments = whisper_result.get("segments", [])
        if segments:
            thai_segments = sum(1 for seg in segments 
                              if self._is_thai_content(seg.get("text", ""), seg.get("language", "")))
            confidence += (thai_segments / len(segments)) * 0.2
        
        return min(confidence, 1.0)
    
    def _fallback_detection(self, audio_path: str) -> LanguageDetectionResult:
        """Fallback language detection without Whisper"""
        # Simple heuristic based on file metadata or name
        filename = Path(audio_path).name.lower()
        
        thai_confidence = 0.3  # Default low confidence
        
        # Check filename for Thai indicators
        thai_indicators = ["thai", "th", "ไทย", "thailand"]
        if any(indicator in filename for indicator in thai_indicators):
            thai_confidence = 0.8
        
        return LanguageDetectionResult(
            detected_languages={"unknown": 1.0},
            thai_confidence=thai_confidence,
            thai_segments=[],
            total_duration=0.0,
            thai_duration=0.0,
            is_primarily_thai=thai_confidence > 0.5
        )
    
    def extract_thai_segments(self, audio_path: str, language_result: LanguageDetectionResult) -> List[str]:
        """Extract Thai segments from audio file"""
        if not language_result.thai_segments:
            return []
        
        extracted_files = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i, (start, end) in enumerate(language_result.thai_segments):
                output_path = os.path.join(temp_dir, f"thai_segment_{i}.wav")
                
                # Use FFmpeg to extract segment
                if self._extract_segment(audio_path, output_path, start, end):
                    extracted_files.append(output_path)
        
        except Exception as e:
            logger.error(f"Thai segment extraction failed: {e}")
        
        return extracted_files
    
    def _extract_segment(self, input_path: str, output_path: str, start: float, end: float) -> bool:
        """Extract audio segment using FFmpeg"""
        try:
            # Find FFmpeg
            ffmpeg_path = "ffmpeg"  # Assume it's in PATH
            
            cmd = [
                ffmpeg_path,
                '-i', input_path,
                '-ss', str(start),
                '-t', str(end - start),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0 and os.path.exists(output_path)
        
        except Exception:
            return False


class IntelligentRetryManager:
    """Intelligent retry system with error categorization and escalation"""
    
    def __init__(self):
        self.error_history: Dict[str, List[ErrorInfo]] = {}
        self.retry_strategies: Dict[ErrorCategory, RetryStrategy] = {
            ErrorCategory.CORRUPTED_AUDIO: RetryStrategy.ESCALATION,
            ErrorCategory.UNSUPPORTED_FORMAT: RetryStrategy.ESCALATION,
            ErrorCategory.ENCODING_ERROR: RetryStrategy.EXPONENTIAL_BACKOFF,
            ErrorCategory.NETWORK_ERROR: RetryStrategy.EXPONENTIAL_BACKOFF,
            ErrorCategory.MEMORY_ERROR: RetryStrategy.LINEAR_BACKOFF,
            ErrorCategory.TIMEOUT_ERROR: RetryStrategy.LINEAR_BACKOFF,
            ErrorCategory.LANGUAGE_MISMATCH: RetryStrategy.NO_RETRY,
            ErrorCategory.SILENCE_DETECTED: RetryStrategy.ESCALATION,
            ErrorCategory.LOW_QUALITY: RetryStrategy.ESCALATION,
            ErrorCategory.METADATA_ERROR: RetryStrategy.IMMEDIATE,
            ErrorCategory.UNKNOWN_ERROR: RetryStrategy.LINEAR_BACKOFF,
        }
        
        self.max_retries_per_category = {
            ErrorCategory.CORRUPTED_AUDIO: 5,
            ErrorCategory.UNSUPPORTED_FORMAT: 3,
            ErrorCategory.ENCODING_ERROR: 3,
            ErrorCategory.NETWORK_ERROR: 5,
            ErrorCategory.MEMORY_ERROR: 2,
            ErrorCategory.TIMEOUT_ERROR: 3,
            ErrorCategory.LANGUAGE_MISMATCH: 0,
            ErrorCategory.SILENCE_DETECTED: 2,
            ErrorCategory.LOW_QUALITY: 2,
            ErrorCategory.METADATA_ERROR: 1,
            ErrorCategory.UNKNOWN_ERROR: 2,
        }
    
    def categorize_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorCategory:
        """Categorize error based on type and context"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ["connection", "network", "timeout", "unreachable"]):
            return ErrorCategory.NETWORK_ERROR
        
        # Memory-related errors
        if any(keyword in error_str for keyword in ["memory", "allocation", "out of memory"]):
            return ErrorCategory.MEMORY_ERROR
        
        # Check for unsupported format first (more specific)
        if any(keyword in error_str for keyword in ["unsupported", "unknown", "invalid"]) and \
           any(keyword in error_str for keyword in ["format", "codec"]):
            return ErrorCategory.UNSUPPORTED_FORMAT
        
        # Audio format/encoding errors
        if any(keyword in error_str for keyword in ["format", "codec", "encoding", "decode", "decoding", "parsing"]):
            return ErrorCategory.ENCODING_ERROR
        
        # Corrupted file indicators
        if any(keyword in error_str for keyword in ["corrupt", "invalid", "damaged", "truncated"]):
            return ErrorCategory.CORRUPTED_AUDIO
        
        # Timeout errors
        if "timeout" in error_type or "timeout" in error_str:
            return ErrorCategory.TIMEOUT_ERROR
        
        # Silence/quality issues
        if context:
            if context.get("silence_detected", False):
                return ErrorCategory.SILENCE_DETECTED
            if context.get("low_quality", False):
                return ErrorCategory.LOW_QUALITY
            if context.get("language_mismatch", False):
                return ErrorCategory.LANGUAGE_MISMATCH
        
        # Metadata errors
        if any(keyword in error_str for keyword in ["metadata", "tag", "header"]):
            return ErrorCategory.METADATA_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """Determine if error should be retried"""
        strategy = self.retry_strategies.get(error_info.category, RetryStrategy.NO_RETRY)
        max_retries = self.max_retries_per_category.get(error_info.category, 0)
        
        if strategy == RetryStrategy.NO_RETRY or error_info.retry_count >= max_retries:
            return False
        
        # Check error history for patterns
        file_path = error_info.file_path
        if file_path and file_path in self.error_history:
            recent_errors = [e for e in self.error_history[file_path] 
                           if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
            
            # Don't retry if we've seen the same error too many times recently
            same_category_count = sum(1 for e in recent_errors if e.category == error_info.category)
            if same_category_count >= max_retries:
                return False
        
        return True
    
    def calculate_retry_delay(self, error_info: ErrorInfo) -> float:
        """Calculate delay before retry"""
        strategy = self.retry_strategies.get(error_info.category, RetryStrategy.LINEAR_BACKOFF)
        base_delay = 1.0  # Base delay in seconds
        
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            return base_delay * (error_info.retry_count + 1)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay * (2 ** error_info.retry_count)
        elif strategy == RetryStrategy.ESCALATION:
            # Escalate to different recovery methods
            return base_delay * 0.5  # Quick escalation
        else:
            return base_delay
    
    def record_error(self, error_info: ErrorInfo):
        """Record error in history for pattern analysis"""
        file_path = error_info.file_path
        if file_path:
            if file_path not in self.error_history:
                self.error_history[file_path] = []
            
            self.error_history[file_path].append(error_info)
            
            # Keep only recent history (last 100 errors per file)
            self.error_history[file_path] = self.error_history[file_path][-100:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        stats = {
            "total_files_with_errors": len(self.error_history),
            "errors_by_category": {},
            "recovery_success_rate": {},
            "most_problematic_files": []
        }
        
        all_errors = []
        for file_errors in self.error_history.values():
            all_errors.extend(file_errors)
        
        # Count by category
        for error in all_errors:
            category = error.category.value
            if category not in stats["errors_by_category"]:
                stats["errors_by_category"][category] = 0
            stats["errors_by_category"][category] += 1
        
        # Calculate recovery success rates
        for category in ErrorCategory:
            category_errors = [e for e in all_errors if e.category == category]
            if category_errors:
                successful_recoveries = sum(1 for e in category_errors if e.recovery_successful)
                stats["recovery_success_rate"][category.value] = successful_recoveries / len(category_errors)
        
        # Find most problematic files
        file_error_counts = {path: len(errors) for path, errors in self.error_history.items()}
        sorted_files = sorted(file_error_counts.items(), key=lambda x: x[1], reverse=True)
        stats["most_problematic_files"] = sorted_files[:10]
        
        return stats


class AdvancedEdgeCaseHandler:
    """Main edge case handler coordinating all recovery mechanisms"""
    
    def __init__(self):
        self.recovery_engines = [
            FFmpegRecoveryEngine(),
            LibrosaRecoveryEngine()
        ]
        self.language_detector = LanguageDetector()
        self.retry_manager = IntelligentRetryManager()
        
        # Performance tracking
        self.processing_stats = {
            "files_processed": 0,
            "files_recovered": 0,
            "recovery_methods_used": {},
            "processing_time_total": 0.0
        }
        
        logger.info("Advanced Edge Case Handler initialized")
    
    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Process file with comprehensive edge case handling"""
        start_time = time.time()
        
        result = {
            "file_path": file_path,
            "success": False,
            "error_info": None,
            "recovery_result": None,
            "language_result": None,
            "thai_segments": [],
            "processing_time": 0.0,
            "metadata": {}
        }
        
        try:
            # First, try normal processing
            result.update(self._try_normal_processing(file_path, **kwargs))
            
            if result["success"]:
                # Check language if successful
                result["language_result"] = self.language_detector.detect_language(file_path)
                
                # Extract Thai segments if needed
                if result["language_result"].is_primarily_thai:
                    result["thai_segments"] = self.language_detector.extract_thai_segments(
                        file_path, result["language_result"]
                    )
                
            else:
                # Attempt recovery
                error_info = result["error_info"]
                if error_info and self.retry_manager.should_retry(error_info):
                    result["recovery_result"] = self._attempt_recovery(file_path, error_info)
                    
                    if result["recovery_result"].successful:
                        result["success"] = True
                        # Check language on recovered file
                        recovered_file = result["recovery_result"].output_path
                        if recovered_file:
                            result["language_result"] = self.language_detector.detect_language(recovered_file)
        
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            error_info = ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                category=self.retry_manager.categorize_error(e),
                file_path=file_path
            )
            result["error_info"] = error_info
            self.retry_manager.record_error(error_info)
        
        finally:
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            # Update statistics
            self.processing_stats["files_processed"] += 1
            self.processing_stats["processing_time_total"] += processing_time
            
            if result["success"]:
                self.processing_stats["files_recovered"] += 1
            
            if result.get("recovery_result"):
                method = result["recovery_result"].method_used.value
                if method not in self.processing_stats["recovery_methods_used"]:
                    self.processing_stats["recovery_methods_used"][method] = 0
                self.processing_stats["recovery_methods_used"][method] += 1
        
        return result
    
    def _try_normal_processing(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Attempt normal file processing"""
        try:
            # Simulate normal audio processing
            # In practice, this would call the actual audio processing pipeline
            
            # Basic file validation
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if os.path.getsize(file_path) == 0:
                raise ValueError("File is empty")
            
            # Try to load with librosa for validation
            try:
                audio, sr = librosa.load(file_path, sr=None, duration=1.0)  # Load first second
                if len(audio) == 0:
                    raise ValueError("No audio data found")
            except Exception as e:
                # Create error info for recovery
                error_info = ErrorInfo(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    category=self.retry_manager.categorize_error(e),
                    file_path=file_path
                )
                return {"success": False, "error_info": error_info}
            
            return {"success": True, "metadata": {"sample_rate": sr, "duration": len(audio)/sr}}
        
        except Exception as e:
            error_info = ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                category=self.retry_manager.categorize_error(e),
                file_path=file_path
            )
            self.retry_manager.record_error(error_info)
            return {"success": False, "error_info": error_info}
    
    def _attempt_recovery(self, file_path: str, error_info: ErrorInfo) -> RecoveryResult:
        """Attempt to recover file using available engines"""
        for engine in self.recovery_engines:
            if engine.can_handle(error_info):
                logger.info(f"Attempting recovery with {engine.__class__.__name__} for {file_path}")
                
                recovery_result = engine.recover(file_path, error_info)
                
                if recovery_result.successful:
                    error_info.recovery_attempted = True
                    error_info.recovery_method = recovery_result.method_used
                    error_info.recovery_successful = True
                    
                    logger.info(f"Recovery successful for {file_path} using {recovery_result.method_used.value}")
                    return recovery_result
        
        # No engine could recover the file
        error_info.recovery_attempted = True
        error_info.recovery_successful = False
        
        return RecoveryResult(
            successful=False,
            method_used=RecoveryMethod.FFMPEG_REPAIR,  # Default
            error_message="No recovery engine could handle this error"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {
            "processing_stats": self.processing_stats.copy(),
            "error_stats": self.retry_manager.get_error_statistics(),
            "recovery_success_rate": 0.0,
            "average_processing_time": 0.0
        }
        
        if self.processing_stats["files_processed"] > 0:
            stats["recovery_success_rate"] = (
                self.processing_stats["files_recovered"] / self.processing_stats["files_processed"]
            )
            stats["average_processing_time"] = (
                self.processing_stats["processing_time_total"] / self.processing_stats["files_processed"]
            )
        
        return stats
    
    def batch_process(self, file_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple files with edge case handling"""
        results = []
        
        for file_path in file_paths:
            result = self.process_file(file_path, **kwargs)
            results.append(result)
            
            # Log progress periodically
            if len(results) % 100 == 0:
                logger.info(f"Processed {len(results)}/{len(file_paths)} files")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    handler = AdvancedEdgeCaseHandler()
    
    # Process single file
    test_files = [
        "/path/to/test_audio.wav",
        "/path/to/corrupted_audio.mp3",
        "/path/to/thai_audio.wav"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            result = handler.process_file(test_file)
            print(f"File: {test_file}")
            print(f"Success: {result['success']}")
            if result.get('language_result'):
                print(f"Thai confidence: {result['language_result'].thai_confidence}")
            print("---")
    
    # Get statistics
    stats = handler.get_statistics()
    print("Processing Statistics:")
    print(json.dumps(stats, indent=2, default=str))