"""
SpeechBrain-based Speaker Separation Module

Replaces the existing speaker_separation.py with a more powerful SpeechBrain-based
implementation using SepFormer for complete secondary speaker removal.
"""

import numpy as np
import torch
import torchaudio
import logging
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import os
from concurrent.futures import ThreadPoolExecutor
import gc

from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import SpeakerRecognition

logger = logging.getLogger(__name__)


@dataclass
class SeparationConfig:
    """Configuration for speaker separation"""
    model_name: str = "speechbrain/sepformer-whamr16k"
    device: str = "cuda"
    batch_size: int = 16  # Optimized for RTX 5090
    confidence_threshold: float = 0.7
    quality_thresholds: Dict = field(default_factory=lambda: {
        "min_pesq": 3.5,
        "min_stoi": 0.85,
        "max_spectral_distortion": 0.15
    })
    speaker_selection: str = "energy"  # energy|embedding|manual
    chunk_duration: float = 10.0  # Max chunk size for memory efficiency
    use_mixed_precision: bool = True  # For faster processing
    cache_dir: str = "/media/ssd1/SparkVoiceProject/models/speechbrain"


@dataclass
class SeparationInput:
    """Input data for separation"""
    audio: np.ndarray  # Audio signal (mono, 16kHz)
    sample_rate: int = 16000
    config: SeparationConfig = field(default_factory=SeparationConfig)


@dataclass
class SeparationOutput:
    """Output from separation process"""
    audio: np.ndarray  # Cleaned single-speaker audio
    confidence: float  # Confidence score (0-1)
    metrics: Dict[str, float]  # Quality metrics
    rejected: bool  # Whether audio was rejected
    rejection_reason: Optional[str] = None
    processing_time_ms: float = 0.0
    num_speakers_detected: int = 0


@dataclass
class ProcessingMetrics:
    """Detailed processing metrics"""
    snr_improvement: float = 0.0
    pesq_score: float = 0.0
    stoi_score: float = 0.0
    spectral_distortion: float = 0.0
    num_speakers_detected: int = 0
    primary_speaker_duration: float = 0.0
    secondary_speaker_duration: float = 0.0
    overlap_duration: float = 0.0


class GPUMemoryManager:
    """Manages GPU memory for optimal performance"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.max_memory_gb = 32  # RTX 5090
        
    def get_optimal_batch_size(self, audio_duration: float) -> int:
        """Calculate optimal batch size based on audio duration and available memory"""
        if self.device == "cpu" or not torch.cuda.is_available():
            return 1
            
        # Get available GPU memory
        torch.cuda.empty_cache()
        free_memory = torch.cuda.mem_get_info()[0] / 1e9  # Convert to GB
        
        # Estimate memory usage per sample (empirical values for SepFormer)
        memory_per_second = 0.5  # GB per second of audio
        memory_per_sample = audio_duration * memory_per_second
        
        # Calculate safe batch size (use 80% of available memory)
        safe_memory = free_memory * 0.8
        optimal_batch_size = int(safe_memory / memory_per_sample)
        
        # Clamp to reasonable range
        return max(1, min(optimal_batch_size, 32))
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class SpeechBrainSeparator:
    """
    Advanced speaker separation using SpeechBrain's SepFormer model.
    Completely replaces the existing speaker_separation.py implementation.
    """
    
    def __init__(self, config: Optional[SeparationConfig] = None):
        """
        Initialize SpeechBrain separator.
        
        Args:
            config: Separation configuration
        """
        self.config = config or SeparationConfig()
        self.memory_manager = GPUMemoryManager(self.config.device)
        
        # Set up cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.environ['SPEECHBRAIN_CACHE'] = self.config.cache_dir
        
        # Initialize models
        self._init_models()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful_separations': 0,
            'rejections': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
    def _init_models(self):
        """Initialize SpeechBrain models"""
        try:
            logger.info(f"Loading SepFormer model: {self.config.model_name}")
            
            # Initialize separation model
            self.separator = SepformerSeparation.from_hparams(
                source=self.config.model_name,
                savedir=os.path.join(self.config.cache_dir, "sepformer"),
                run_opts={"device": self.config.device}
            )
            
            # Initialize speaker embedding model for verification
            self.speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(self.config.cache_dir, "ecapa"),
                run_opts={"device": self.config.device}
            )
            
            logger.info("Models loaded successfully")
            
            # Warm up models
            self._warmup_models()
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def _warmup_models(self):
        """Warm up models with dummy data for faster first inference"""
        try:
            dummy_audio = torch.randn(1, 16000).to(self.config.device)
            with torch.no_grad():
                _ = self.separator.separate_batch(dummy_audio)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def separate_speakers(self, 
                         audio_array: np.ndarray, 
                         sample_rate: int = 16000) -> SeparationOutput:
        """
        Perform speaker separation to extract only the primary speaker.
        
        Args:
            audio_array: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            SeparationOutput with cleaned audio or rejection
        """
        start_time = time.time()
        
        try:
            # Validate input
            if len(audio_array) == 0:
                return SeparationOutput(
                    audio=audio_array,
                    confidence=0.0,
                    metrics={},
                    rejected=True,
                    rejection_reason="Empty audio"
                )
            
            # Check for NaN/Inf values
            if not np.isfinite(audio_array).all():
                # Clean the audio
                audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
                logger.warning("Input audio contained NaN/Inf values, replaced with zeros")
            
            # Convert to tensor - ensure float32 to avoid float16 issues
            if audio_array.dtype == np.float16:
                audio_array = audio_array.astype(np.float32)
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Move to device
            audio_tensor = audio_tensor.to(self.config.device)
            
            # Perform separation
            separated_sources = self._perform_separation(audio_tensor)
            
            # Select primary speaker
            primary_audio, confidence = self._select_primary_speaker(
                separated_sources, audio_tensor
            )
            
            # Calculate quality metrics
            metrics = self._calculate_metrics(
                audio_tensor.cpu().numpy()[0],
                primary_audio
            )
            
            # Check quality thresholds
            rejection_reason = self._check_quality_thresholds(metrics, confidence)
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(confidence, rejection_reason is None, processing_time_ms)
            
            return SeparationOutput(
                audio=primary_audio,
                confidence=confidence,
                metrics=metrics,
                rejected=rejection_reason is not None,
                rejection_reason=rejection_reason,
                processing_time_ms=processing_time_ms,
                num_speakers_detected=separated_sources.shape[2]
            )
            
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            return SeparationOutput(
                audio=audio_array,
                confidence=0.0,
                metrics={},
                rejected=True,
                rejection_reason=f"Processing error: {e}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        finally:
            # Clear GPU cache after processing
            self.memory_manager.clear_cache()
    
    def _perform_separation(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the actual separation using SepFormer.
        
        Args:
            audio_tensor: Input audio tensor
            
        Returns:
            Separated sources tensor
        """
        with torch.no_grad():
            if self.config.use_mixed_precision and self.config.device == "cuda":
                with torch.amp.autocast("cuda"):
                    separated = self.separator.separate_batch(audio_tensor)
            else:
                separated = self.separator.separate_batch(audio_tensor)
        
        return separated
    
    def _select_primary_speaker(self, 
                               separated_sources: torch.Tensor,
                               original_audio: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Select the primary speaker from separated sources.
        
        Args:
            separated_sources: Separated audio sources
            original_audio: Original mixed audio
            
        Returns:
            Primary speaker audio and confidence score
        """
        if self.config.speaker_selection == "energy":
            return self._select_by_energy(separated_sources)
        elif self.config.speaker_selection == "embedding":
            return self._select_by_embedding(separated_sources, original_audio)
        else:
            # Default to first source
            return separated_sources[0, :, 0].cpu().numpy(), 0.8
    
    def _select_by_energy(self, separated_sources: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Select primary speaker based on energy (RMS).
        
        Args:
            separated_sources: Separated audio sources
            
        Returns:
            Primary speaker audio and confidence
        """
        # Calculate RMS energy for each source
        energies = []
        for i in range(separated_sources.shape[2]):
            source = separated_sources[0, :, i]
            rms = torch.sqrt(torch.mean(source ** 2))
            energies.append(rms.item())
        
        # Select source with highest energy
        primary_idx = np.argmax(energies)
        primary_audio = separated_sources[0, :, primary_idx].cpu().numpy()
        
        # Calculate confidence based on energy ratio
        total_energy = sum(energies)
        if total_energy > 0:
            confidence = energies[primary_idx] / total_energy
        else:
            confidence = 0.0
        
        return primary_audio, confidence
    
    def _select_by_embedding(self, 
                            separated_sources: torch.Tensor,
                            original_audio: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Select primary speaker based on speaker embeddings.
        
        Args:
            separated_sources: Separated audio sources
            original_audio: Original mixed audio
            
        Returns:
            Primary speaker audio and confidence
        """
        # Get embedding for original audio
        original_embedding = self.speaker_model.encode_batch(original_audio)
        
        # Get embeddings for each separated source
        similarities = []
        for i in range(separated_sources.shape[2]):
            source = separated_sources[0:1, :, i]
            source_embedding = self.speaker_model.encode_batch(source)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                original_embedding, source_embedding
            ).item()
            similarities.append(similarity)
        
        # Select source with highest similarity to original
        primary_idx = np.argmax(similarities)
        primary_audio = separated_sources[0, :, primary_idx].cpu().numpy()
        confidence = similarities[primary_idx]
        
        return primary_audio, confidence
    
    def separate_all_speakers(self, audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        Separate all speakers in the audio (public interface for complete_separation).
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            List of separated audio sources as numpy arrays
        """
        try:
            # Ensure float32
            if audio.dtype == np.float16:
                audio = audio.astype(np.float32)
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Move to device
            audio_tensor = audio_tensor.to(self.config.device)
            
            # Perform separation
            separated_sources = self._perform_separation(audio_tensor)
            
            # Convert each source to numpy array
            sources_list = []
            for i in range(separated_sources.shape[2]):
                source = separated_sources[0, :, i].cpu().numpy()
                sources_list.append(source)
            
            return sources_list
            
        except Exception as e:
            logger.error(f"Failed to separate speakers: {e}")
            return [audio]  # Return original on failure
    
    def _calculate_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        Calculate quality metrics for the processed audio.
        
        Args:
            original: Original audio
            processed: Processed audio
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        try:
            # Import metrics functions
            from utils.audio_metrics import calculate_snr, calculate_stoi
            
            # SNR improvement
            original_snr = calculate_snr(original, original)  # Self-SNR as baseline
            processed_snr = calculate_snr(processed, original)
            metrics['snr_improvement'] = processed_snr - original_snr
            
            # STOI (Short-Time Objective Intelligibility)
            metrics['stoi'] = calculate_stoi(original, processed, 16000)
            
            # Spectral distortion
            metrics['spectral_distortion'] = self._calculate_spectral_distortion(
                original, processed
            )
            
            # Energy ratio
            metrics['energy_ratio'] = np.sqrt(np.mean(processed ** 2)) / \
                                     (np.sqrt(np.mean(original ** 2)) + 1e-10)
            
        except Exception as e:
            logger.warning(f"Failed to calculate some metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_spectral_distortion(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Calculate spectral distortion between original and processed audio.
        
        Args:
            original: Original audio
            processed: Processed audio
            
        Returns:
            Spectral distortion value
        """
        try:
            # Compute spectrograms
            orig_spec = np.abs(np.fft.rfft(original))
            proc_spec = np.abs(np.fft.rfft(processed))
            
            # Normalize
            orig_spec = orig_spec / (np.max(orig_spec) + 1e-10)
            proc_spec = proc_spec / (np.max(proc_spec) + 1e-10)
            
            # Calculate distortion
            distortion = np.mean(np.abs(orig_spec - proc_spec))
            
            return float(distortion)
        except:
            return 0.0
    
    def _check_quality_thresholds(self, metrics: Dict[str, float], confidence: float) -> Optional[str]:
        """
        Check if the separation meets quality thresholds.
        
        Args:
            metrics: Calculated metrics
            confidence: Confidence score
            
        Returns:
            Rejection reason or None if passes
        """
        # Check confidence threshold
        if confidence < self.config.confidence_threshold:
            return f"Low confidence: {confidence:.2f} < {self.config.confidence_threshold}"
        
        # Check STOI
        if 'stoi' in metrics and metrics['stoi'] < self.config.quality_thresholds['min_stoi']:
            return f"Low STOI: {metrics['stoi']:.2f} < {self.config.quality_thresholds['min_stoi']}"
        
        # Check spectral distortion
        if 'spectral_distortion' in metrics and \
           metrics['spectral_distortion'] > self.config.quality_thresholds['max_spectral_distortion']:
            return f"High spectral distortion: {metrics['spectral_distortion']:.2f}"
        
        return None
    
    def _update_stats(self, confidence: float, success: bool, processing_time: float):
        """Update processing statistics"""
        self.stats['total_processed'] += 1
        if success:
            self.stats['successful_separations'] += 1
        else:
            self.stats['rejections'] += 1
        self.stats['total_processing_time'] += processing_time
        
        # Update running average confidence
        n = self.stats['total_processed']
        self.stats['average_confidence'] = \
            ((n - 1) * self.stats['average_confidence'] + confidence) / n
    
    def process_batch(self, audio_batch: List[np.ndarray], 
                     sample_rate: int = 16000) -> List[SeparationOutput]:
        """
        Process multiple audio files in batch for efficiency.
        
        Args:
            audio_batch: List of audio arrays
            sample_rate: Sample rate
            
        Returns:
            List of SeparationOutput objects
        """
        results = []
        
        # Process in optimal batch sizes
        batch_size = self.memory_manager.get_optimal_batch_size(
            max([len(a) / sample_rate for a in audio_batch])
        )
        
        for i in range(0, len(audio_batch), batch_size):
            batch = audio_batch[i:i + batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
                batch_results = list(executor.map(
                    lambda a: self.separate_speakers(a, sample_rate),
                    batch
                ))
            
            results.extend(batch_results)
        
        return results
    
    def get_stats(self) -> Dict[str, float]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_separations'] / stats['total_processed']
            stats['average_processing_time_ms'] = \
                stats['total_processing_time'] / stats['total_processed']
        return stats


# Backward compatibility alias
SpeakerSeparator = SpeechBrainSeparator