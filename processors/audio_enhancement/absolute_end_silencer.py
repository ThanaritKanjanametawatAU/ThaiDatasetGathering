"""
Absolute End Silencer Module

An extremely aggressive module that unconditionally silences the end portion of audio.
This is used as the ultimate fallback when all other methods fail to remove secondary speakers.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class AbsoluteEndSilencer:
    """
    Unconditionally silence the end portion of audio.
    This guarantees no secondary speaker can be present at the end.
    """
    
    def __init__(self, silence_duration: float = 2.5):
        """
        Initialize absolute end silencer.
        
        Args:
            silence_duration: Duration in seconds to silence at end
        """
        self.silence_duration = silence_duration
        
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Unconditionally silence the end portion of audio.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            Audio with end portion silenced
        """
        silence_samples = int(self.silence_duration * sample_rate)
        
        # Don't silence more than half the audio
        silence_samples = min(silence_samples, len(audio) // 2)
        
        if silence_samples <= 0:
            return audio
        
        # Copy audio and silence the end
        processed = audio.copy()
        silence_start = len(audio) - silence_samples
        
        # Apply a short fade to avoid clicks
        fade_samples = min(int(0.05 * sample_rate), silence_samples // 4)
        if fade_samples > 0:
            fade = np.linspace(1.0, 0.0, fade_samples)
            processed[silence_start:silence_start + fade_samples] *= fade
        
        # Complete silence for the rest
        processed[silence_start + fade_samples:] = 0
        
        logger.info(f"Applied absolute silence to last {silence_samples/sample_rate:.2f} seconds")
        
        return processed