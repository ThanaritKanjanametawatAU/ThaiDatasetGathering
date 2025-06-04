"""
Advanced speaker separation module with SepFormer and Conv-TasNet
"""

import numpy as np
import torch
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
import time

# Try to import speechbrain for advanced models
try:
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.inference.separation import ConvTasNet
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old import path
        from speechbrain.pretrained import SepformerSeparation, ConvTasNet
        SPEECHBRAIN_AVAILABLE = True
    except ImportError:
        SPEECHBRAIN_AVAILABLE = False
        logging.warning("SpeechBrain not installed. Advanced separation models unavailable. "
                       "Install with: pip install speechbrain")

logger = logging.getLogger(__name__)


@dataclass
class SeparationResult:
    """Result from speaker separation"""
    success: bool
    primary_audio: Optional[np.ndarray]
    metrics: Dict[str, float]
    separation_method: str
    processing_time: float
    excluded_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExclusionCriteria:
    """Criteria for excluding poor quality separations"""
    min_si_sdr: float = 10.0
    min_pesq: float = 3.0
    min_stoi: float = 0.85
    max_attempts: int = 2
    
    def should_exclude(self, result: SeparationResult) -> bool:
        """Check if separation result should be excluded"""
        if not result.success:
            return True
            
        metrics = result.metrics
        
        # Check SI-SDR
        if 'si_sdr' in metrics and metrics['si_sdr'] < self.min_si_sdr:
            result.excluded_reason = f"SI-SDR below threshold ({metrics['si_sdr']:.1f} < {self.min_si_sdr})"
            return True
            
        # Check PESQ
        if 'pesq' in metrics and metrics['pesq'] < self.min_pesq:
            result.excluded_reason = f"PESQ below threshold ({metrics['pesq']:.2f} < {self.min_pesq})"
            return True
            
        # Check STOI
        if 'stoi' in metrics and metrics['stoi'] < self.min_stoi:
            result.excluded_reason = f"STOI below threshold ({metrics['stoi']:.3f} < {self.min_stoi})"
            return True
            
        return False


class SepFormerEngine:
    """SepFormer-based speaker separation engine"""
    
    def __init__(self, use_gpu: bool = True, max_duration: float = 60.0):
        """
        Initialize SepFormer engine.
        
        Args:
            use_gpu: Whether to use GPU if available
            max_duration: Maximum audio duration in seconds (for memory management)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_duration = max_duration
        self.model = None
        self.model_name = "speechbrain/sepformer-wham"
        
        if SPEECHBRAIN_AVAILABLE:
            self._initialize_model()
            
    def _initialize_model(self):
        """Initialize the SepFormer model"""
        try:
            device = "cuda" if self.use_gpu else "cpu"
            self.model = SepformerSeparation.from_hparams(
                source=self.model_name,
                savedir="pretrained_models/sepformer-wham",
                run_opts={"device": device}
            )
            logger.info(f"SepFormer model initialized on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize SepFormer: {e}")
            self.model = None
            
    def is_available(self) -> bool:
        """Check if engine is available"""
        return self.model is not None
        
    def separate(self, audio: np.ndarray, sample_rate: int) -> SeparationResult:
        """
        Separate speakers using SepFormer.
        
        Args:
            audio: Mixed audio signal
            sample_rate: Sample rate
            
        Returns:
            SeparationResult with separated audio
        """
        start_time = time.time()
        
        if not self.is_available():
            return SeparationResult(
                success=False,
                primary_audio=None,
                metrics={},
                separation_method="sepformer",
                processing_time=0.0,
                excluded_reason="Model not available"
            )
            
        try:
            # Check duration constraint
            duration = len(audio) / sample_rate
            if duration > self.max_duration:
                # Fallback to Conv-TasNet
                logger.info(f"Audio too long for SepFormer ({duration:.1f}s > {self.max_duration}s), "
                           "falling back to Conv-TasNet")
                engine = ConvTasNetEngine(use_gpu=self.use_gpu)
                result = engine.separate(audio, sample_rate)
                result.metadata['fallback_used'] = True
                result.metadata['original_method'] = 'sepformer'
                result.metadata['fallback_reason'] = 'Memory constraint'
                return result
                
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio.astype(np.float32))
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
                
            # Separate sources
            separated_sources = self.model.separate_batch(audio_tensor)
            
            # Extract primary source (assuming first source is primary)
            primary = separated_sources[:, 0, :].squeeze().cpu().numpy()
            
            # Ensure same length as input
            if len(primary) != len(audio):
                if len(primary) > len(audio):
                    primary = primary[:len(audio)]
                else:
                    primary = np.pad(primary, (0, len(audio) - len(primary)))
                    
            # Calculate basic metrics
            from utils.audio_metrics import calculate_si_sdr
            si_sdr = calculate_si_sdr(audio, primary)
            
            processing_time = time.time() - start_time
            
            return SeparationResult(
                success=True,
                primary_audio=primary,
                metrics={'si_sdr': si_sdr},
                separation_method="sepformer",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"SepFormer separation failed: {e}")
            return SeparationResult(
                success=False,
                primary_audio=None,
                metrics={},
                separation_method="sepformer",
                processing_time=time.time() - start_time,
                excluded_reason=f"Separation failed: {str(e)}"
            )


class ConvTasNetEngine:
    """Conv-TasNet-based speaker separation engine"""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize Conv-TasNet engine.
        
        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
        self.model_name = "speechbrain/conv-tasnet-wham"
        
        if SPEECHBRAIN_AVAILABLE:
            self._initialize_model()
            
    def _initialize_model(self):
        """Initialize the Conv-TasNet model"""
        try:
            device = "cuda" if self.use_gpu else "cpu"
            self.model = ConvTasNet.from_hparams(
                source=self.model_name,
                savedir="pretrained_models/conv-tasnet-wham",
                run_opts={"device": device}
            )
            logger.info(f"Conv-TasNet model initialized on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize Conv-TasNet: {e}")
            self.model = None
            
    def is_available(self) -> bool:
        """Check if engine is available"""
        return self.model is not None
        
    def separate(self, audio: np.ndarray, sample_rate: int) -> SeparationResult:
        """
        Separate speakers using Conv-TasNet.
        
        Args:
            audio: Mixed audio signal
            sample_rate: Sample rate
            
        Returns:
            SeparationResult with separated audio
        """
        start_time = time.time()
        
        if not self.is_available():
            return SeparationResult(
                success=False,
                primary_audio=None,
                metrics={},
                separation_method="conv-tasnet",
                processing_time=0.0,
                excluded_reason="Model not available"
            )
            
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio.astype(np.float32))
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
                
            # Separate sources
            separated_sources = self.model.separate_batch(audio_tensor)
            
            # Extract primary source
            primary = separated_sources[:, 0, :].squeeze().cpu().numpy()
            
            # Ensure same length
            if len(primary) != len(audio):
                if len(primary) > len(audio):
                    primary = primary[:len(audio)]
                else:
                    primary = np.pad(primary, (0, len(audio) - len(primary)))
                    
            # Calculate basic metrics
            from utils.audio_metrics import calculate_si_sdr
            si_sdr = calculate_si_sdr(audio, primary)
            
            processing_time = time.time() - start_time
            
            return SeparationResult(
                success=True,
                primary_audio=primary,
                metrics={'si_sdr': si_sdr},
                separation_method="conv-tasnet",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Conv-TasNet separation failed: {e}")
            return SeparationResult(
                success=False,
                primary_audio=None,
                metrics={},
                separation_method="conv-tasnet",
                processing_time=time.time() - start_time,
                excluded_reason=f"Separation failed: {str(e)}"
            )