"""
Ensemble Speech-to-Text implementation using Wav2Vec2 Thai and Whisper Large V3.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
import warnings

logger = logging.getLogger(__name__)


class EnsembleSTT:
    """
    Ensemble STT using two models:
    1. Wav2Vec2 Thai (airesearch/wav2vec2-large-xlsr-53-th) - Primary
    2. Whisper Large V3 (openai/whisper-large-v3) - Secondary
    
    Returns the transcript with the highest confidence score.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the ensemble STT models.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Ensemble STT on device: {self.device}")
        
        # Initialize models lazily to save memory
        self._wav2vec2_model = None
        self._wav2vec2_processor = None
        self._whisper_model = None
        self._whisper_processor = None
        
        # Model configurations
        self.wav2vec2_model_id = "airesearch/wav2vec2-large-xlsr-53-th"
        self.whisper_model_id = "openai/whisper-large-v3"
        
    def _load_wav2vec2(self):
        """Lazy load Wav2Vec2 model and processor."""
        if self._wav2vec2_model is None:
            logger.info(f"Loading Wav2Vec2 model: {self.wav2vec2_model_id}")
            try:
                self._wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                    self.wav2vec2_model_id
                )
                self._wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                    self.wav2vec2_model_id
                ).to(self.device)
                self._wav2vec2_model.eval()
                logger.info("Wav2Vec2 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Wav2Vec2 model: {e}")
                raise
                
    def _load_whisper(self):
        """Lazy load Whisper model and processor."""
        if self._whisper_model is None:
            logger.info(f"Loading Whisper model: {self.whisper_model_id}")
            try:
                self._whisper_processor = WhisperProcessor.from_pretrained(
                    self.whisper_model_id
                )
                self._whisper_model = WhisperForConditionalGeneration.from_pretrained(
                    self.whisper_model_id
                ).to(self.device)
                self._whisper_model.eval()
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
                
    def _transcribe_wav2vec2(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe using Wav2Vec2 Thai model.
        
        Args:
            audio_data: Audio array at 16kHz
            
        Returns:
            Tuple of (transcript, confidence_score)
        """
        try:
            self._load_wav2vec2()
            
            # Process audio
            inputs = self._wav2vec2_processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get logits
            with torch.no_grad():
                logits = self._wav2vec2_model(**inputs).logits
                
            # Get predicted IDs
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode to text
            transcript = self._wav2vec2_processor.batch_decode(
                predicted_ids
            )[0].strip()
            
            # Calculate confidence from logits
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values.mean().item()
            
            return transcript, confidence
            
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {e}")
            return "", 0.0
            
    def _transcribe_whisper(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe using Whisper Large V3 model.
        
        Args:
            audio_data: Audio array at 16kHz
            
        Returns:
            Tuple of (transcript, confidence_score)
        """
        try:
            self._load_whisper()
            
            # Process audio
            inputs = self._whisper_processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            # Force Thai language
            forced_decoder_ids = self._whisper_processor.get_decoder_prompt_ids(
                language="th", 
                task="transcribe"
            )
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self._whisper_model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
            # Decode to text
            transcript = self._whisper_processor.batch_decode(
                predicted_ids.sequences, 
                skip_special_tokens=True
            )[0].strip()
            
            # Calculate confidence from scores
            if hasattr(predicted_ids, 'scores') and predicted_ids.scores:
                # Average confidence from generation scores
                all_scores = []
                for score in predicted_ids.scores:
                    probs = torch.softmax(score, dim=-1)
                    max_probs = probs.max(dim=-1).values
                    all_scores.extend(max_probs.cpu().numpy())
                confidence = float(np.mean(all_scores)) if all_scores else 0.8
            else:
                # Default confidence for Whisper
                confidence = 0.8
                
            return transcript, confidence
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return "", 0.0
            
    def transcribe(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe audio using ensemble approach (highest confidence).
        
        Args:
            audio_data: Audio array at 16kHz
            
        Returns:
            Tuple of (transcript, confidence_score)
        """
        # Ensure audio is numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
            
        # Get predictions from both models
        wav2vec2_text, wav2vec2_conf = self._transcribe_wav2vec2(audio_data)
        whisper_text, whisper_conf = self._transcribe_whisper(audio_data)
        
        logger.debug(f"Wav2Vec2: '{wav2vec2_text}' (conf: {wav2vec2_conf:.3f})")
        logger.debug(f"Whisper: '{whisper_text}' (conf: {whisper_conf:.3f})")
        
        # Return highest confidence result with spaces stripped
        if wav2vec2_conf >= whisper_conf:
            # Remove all spaces from the transcript
            cleaned_text = wav2vec2_text.replace(" ", "")
            return cleaned_text, wav2vec2_conf
        else:
            # Remove all spaces from the transcript
            cleaned_text = whisper_text.replace(" ", "")
            return cleaned_text, whisper_conf
            
    def transcribe_with_fallback(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Transcribe with fallback strategy for empty results.
        
        Args:
            audio_data: Audio array at 16kHz
            
        Returns:
            Tuple of (transcript, confidence_score)
        """
        # Try primary transcription
        transcript, confidence = self.transcribe(audio_data)
        
        if transcript.strip():
            return transcript, confidence
            
        # If empty, try individual models with different parameters
        logger.warning("Primary transcription returned empty, trying fallback...")
        
        # Try Whisper with beam search
        try:
            self._load_whisper()
            inputs = self._whisper_processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                predicted_ids = self._whisper_model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=self._whisper_processor.get_decoder_prompt_ids(
                        language="th", 
                        task="transcribe"
                    ),
                    num_beams=5,
                    length_penalty=1.0,
                    max_length=225,
                )
                
            transcript = self._whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            if transcript:
                # Remove spaces from fallback transcript too
                cleaned_transcript = transcript.replace(" ", "")
                return cleaned_transcript, 0.5  # Lower confidence for fallback
                
        except Exception as e:
            logger.error(f"Fallback transcription failed: {e}")
            
        # Last resort
        return "[INAUDIBLE]", 0.1
        
    def transcribe_batch(self, audio_batch: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Batch transcription for efficiency.
        
        Args:
            audio_batch: List of audio arrays at 16kHz
            
        Returns:
            List of (transcript, confidence_score) tuples
        """
        results = []
        
        # Process in smaller sub-batches to avoid OOM
        batch_size = 4  # Adjust based on GPU memory
        
        for i in range(0, len(audio_batch), batch_size):
            sub_batch = audio_batch[i:i + batch_size]
            
            # Process each audio in the sub-batch
            for audio in sub_batch:
                result = self.transcribe(audio)
                results.append(result)
                
            # Clear cache after each sub-batch
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
        return results
        
    def __del__(self):
        """Clean up GPU memory on deletion."""
        if hasattr(self, 'device') and self.device == 'cuda':
            torch.cuda.empty_cache()