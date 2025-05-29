"""
Facebook Denoiser Integration
Primary enhancement engine with GPU acceleration.
"""

import numpy as np
import torch
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DenoiserEngine:
    """
    Facebook Denoiser implementation for high-quality noise reduction.
    """
    
    def __init__(
        self, 
        device: str = 'cuda',
        model_path: Optional[str] = None
    ):
        """
        Initialize Denoiser engine.
        
        Args:
            device: Processing device ('cuda' or 'cpu')
            model_path: Path to pretrained model
        """
        self.device = device
        self.model = None
        self.model_loaded = False
        
        # Try to import denoiser
        try:
            from denoiser import pretrained
            from denoiser.dsp import convert_audio
            self.denoiser_available = True
            self.pretrained = pretrained
            self.convert_audio = convert_audio
        except ImportError:
            logger.warning("Facebook Denoiser not installed. Install with: pip install denoiser")
            self.denoiser_available = False
            return
            
        # Load model
        self._load_model(model_path)
        
    def _load_model(self, model_path: Optional[str] = None):
        """
        Load the denoiser model.
        
        Args:
            model_path: Custom model path or None for default
        """
        if not self.denoiser_available:
            return
            
        try:
            if model_path:
                # Load custom model
                self.model = torch.load(model_path, map_location=self.device)
                logger.info(f"Loaded custom denoiser model from {model_path}")
            else:
                # Load pretrained model
                self.model = self.pretrained.dns64().to(self.device)
                logger.info("Loaded pretrained DNS64 denoiser model")
                
            self.model.eval()
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load denoiser model: {e}")
            self.model_loaded = False
            
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        denoiser_dry: float = 0.02,
        **kwargs
    ) -> np.ndarray:
        """
        Process audio with Facebook Denoiser.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            denoiser_dry: Dry/wet ratio (0=fully wet, 1=fully dry)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Enhanced audio array
        """
        if not self.model_loaded:
            logger.warning("Denoiser model not loaded, returning original audio")
            return audio
            
        try:
            # Store original shape
            original_shape = audio.shape
            original_dtype = audio.dtype
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
                
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).to(self.device)
            
            # Add batch and channel dimensions if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
                
            # Apply denoiser
            with torch.no_grad():
                # Convert sample rate if needed
                if sample_rate != self.model.sample_rate:
                    audio_tensor = self.convert_audio(
                        audio_tensor,
                        sample_rate,
                        self.model.sample_rate,
                        self.model.chin
                    )
                    
                # Process
                enhanced_tensor = self.model(audio_tensor)
                
                # Convert back to original sample rate
                if sample_rate != self.model.sample_rate:
                    enhanced_tensor = self.convert_audio(
                        enhanced_tensor,
                        self.model.sample_rate,
                        sample_rate,
                        1
                    )
                    
            # Convert back to numpy
            enhanced = enhanced_tensor.squeeze().cpu().numpy()
            
            # Apply dry/wet mix
            if denoiser_dry > 0:
                # Ensure shapes match for mixing
                if enhanced.shape != audio.shape:
                    if len(enhanced) < len(audio):
                        # Pad enhanced
                        enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
                    else:
                        # Trim enhanced
                        enhanced = enhanced[:len(audio)]
                        
                # Mix with original
                enhanced = denoiser_dry * audio + (1 - denoiser_dry) * enhanced
                
            # Restore original scale
            if max_val > 0:
                enhanced = enhanced * max_val
                
            # Restore original dtype
            if original_dtype != np.float32:
                enhanced = enhanced.astype(original_dtype)
                
            # Ensure output shape matches input
            if enhanced.shape != original_shape:
                if len(enhanced) < len(audio):
                    enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
                else:
                    enhanced = enhanced[:len(audio)]
                    
            return enhanced
            
        except Exception as e:
            logger.error(f"Denoiser processing failed: {e}")
            return audio
            
    def batch_process(
        self,
        audio_batch: list,
        sample_rate: int,
        **kwargs
    ) -> list:
        """
        Process batch of audio files.
        
        Args:
            audio_batch: List of audio arrays
            sample_rate: Sample rate
            **kwargs: Processing parameters
            
        Returns:
            List of enhanced audio arrays
        """
        if not self.model_loaded:
            return audio_batch
            
        try:
            # Convert to batch tensor
            max_len = max(len(a) for a in audio_batch)
            batch_size = len(audio_batch)
            
            # Pad all to same length
            padded_batch = []
            lengths = []
            for audio in audio_batch:
                lengths.append(len(audio))
                if len(audio) < max_len:
                    padded = np.pad(audio, (0, max_len - len(audio)))
                else:
                    padded = audio
                padded_batch.append(padded)
                
            # Stack into tensor
            batch_tensor = torch.stack([
                torch.from_numpy(a.astype(np.float32)) for a in padded_batch
            ]).to(self.device)
            
            # Add channel dimension
            batch_tensor = batch_tensor.unsqueeze(1)
            
            # Process batch
            with torch.no_grad():
                enhanced_batch = self.model(batch_tensor)
                
            # Convert back
            enhanced_list = []
            for i, length in enumerate(lengths):
                enhanced = enhanced_batch[i, 0, :length].cpu().numpy()
                enhanced_list.append(enhanced)
                
            return enhanced_list
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return audio_batch