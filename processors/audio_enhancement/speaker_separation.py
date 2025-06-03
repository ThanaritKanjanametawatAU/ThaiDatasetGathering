"""
Speaker Separation Module

Integrates Asteroid/SepFormer for advanced speech separation with
confidence-based suppression and flexible secondary speaker handling.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import torch
import logging
from dataclasses import dataclass

# Import detection modules
from .detection.overlap_detector import OverlapDetector
from .detection.secondary_speaker import AdaptiveSecondaryDetection, DetectionResult

# Try to import asteroid
try:
    from asteroid.models import SepFormer
    from asteroid import separate
    HAS_ASTEROID = True
except ImportError:
    HAS_ASTEROID = False
    logging.warning("Asteroid not installed. Install with: pip install asteroid")

logger = logging.getLogger(__name__)


@dataclass 
class SeparationConfig:
    """Configuration for speaker separation"""
    min_duration: float = 0.1
    max_duration: float = 5.0
    speaker_similarity_threshold: float = 0.7
    suppression_strength: float = 0.6
    confidence_threshold: float = 0.5
    detection_methods: List[str] = None
    use_sepformer: bool = True
    preserve_main_speaker: bool = True
    artifact_removal: bool = True
    
    def __post_init__(self):
        if self.detection_methods is None:
            self.detection_methods = ["embedding", "vad", "energy", "spectral"]


class SpeakerSeparator:
    """
    Advanced speaker separation with flexible secondary speaker handling
    """
    
    def __init__(self, config: Optional[SeparationConfig] = None):
        """
        Initialize speaker separator
        
        Args:
            config: Separation configuration
        """
        self.config = config or SeparationConfig()
        
        # Initialize detectors
        self.overlap_detector = OverlapDetector()
        self.secondary_detector = AdaptiveSecondaryDetection(
            min_duration=self.config.min_duration,
            max_duration=self.config.max_duration,
            speaker_similarity_threshold=self.config.speaker_similarity_threshold,
            confidence_threshold=self.config.confidence_threshold,
            detection_methods=self.config.detection_methods
        )
        
        # Initialize SepFormer if available
        self.sepformer = None
        if HAS_ASTEROID and self.config.use_sepformer:
            try:
                self.sepformer = SepFormer.from_pretrained('mpariente/sepformer-wsj02mix')
                self.sepformer.eval()
                logger.info("SepFormer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load SepFormer: {e}")
                
    def separate_speakers(self, 
                         audio_array: np.ndarray, 
                         sample_rate: int = 16000) -> Dict[str, Union[np.ndarray, List[DetectionResult]]]:
        """
        Perform speaker separation with secondary speaker suppression
        
        Args:
            audio_array: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Dictionary containing:
                - 'audio': Processed audio with secondary speakers suppressed
                - 'detections': List of secondary speaker detections
                - 'metrics': Processing metrics
        """
        # Detect secondary speakers
        detections = self.secondary_detector.detect(audio_array, sample_rate)
        
        # Detect overlaps
        overlaps = self.overlap_detector.detect_overlaps(audio_array, sample_rate)
        
        # Process audio
        if self.sepformer is not None and (detections or overlaps):
            processed_audio = self._separate_with_sepformer(
                audio_array, sample_rate, detections, overlaps
            )
        elif detections:
            processed_audio = self._suppress_secondary_speakers(
                audio_array, sample_rate, detections
            )
        else:
            # No secondary speakers detected, return copy to ensure modification
            processed_audio = audio_array.copy()
            
        # Remove artifacts if enabled
        if self.config.artifact_removal:
            processed_audio = self._remove_artifacts(processed_audio, sample_rate)
            
        # Calculate metrics
        metrics = self._calculate_metrics(audio_array, processed_audio, detections)
        
        return {
            'audio': processed_audio,
            'detections': detections,
            'overlaps': overlaps,
            'metrics': metrics
        }
    
    def _separate_with_sepformer(self,
                               audio_array: np.ndarray,
                               sample_rate: int,
                               detections: List[DetectionResult],
                               overlaps: List[Tuple[float, float]]) -> np.ndarray:
        """
        Use SepFormer for advanced speaker separation
        
        Args:
            audio_array: Input audio
            sample_rate: Sample rate
            detections: Secondary speaker detections
            overlaps: Overlap segments
            
        Returns:
            Separated audio
        """
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
            
            # Normalize
            audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            # Perform separation
            with torch.no_grad():
                separated = self.sepformer.separate(audio_tensor)
                
            # Convert back to numpy
            separated_np = separated.squeeze().numpy()
            
            # If multiple sources, select main speaker
            if len(separated_np.shape) > 1:
                # Use energy-based selection for main speaker
                energies = [np.sqrt(np.mean(s**2)) for s in separated_np]
                main_idx = np.argmax(energies)
                main_audio = separated_np[main_idx]
            else:
                main_audio = separated_np
                
            # Apply additional suppression based on detections
            for detection in detections:
                if detection.confidence > 0.8:
                    main_audio = self._apply_suppression(
                        main_audio, sample_rate, 
                        detection.start_time, detection.end_time,
                        strength=self.config.suppression_strength * detection.confidence
                    )
                    
            return main_audio
            
        except Exception as e:
            logger.error(f"SepFormer separation failed: {e}")
            # Fallback to basic suppression
            return self._suppress_secondary_speakers(audio_array, sample_rate, detections)
    
    def _suppress_secondary_speakers(self,
                                   audio_array: np.ndarray,
                                   sample_rate: int,
                                   detections: List[DetectionResult]) -> np.ndarray:
        """
        Suppress secondary speakers using confidence-based attenuation
        
        Args:
            audio_array: Input audio
            sample_rate: Sample rate  
            detections: Secondary speaker detections
            
        Returns:
            Audio with secondary speakers suppressed
        """
        processed = audio_array.copy()
        
        for detection in detections:
            # Calculate suppression strength based on confidence
            if detection.confidence >= 0.8:
                strength = self.config.suppression_strength
            elif detection.confidence >= 0.5:
                strength = self.config.suppression_strength * 0.7
            else:
                strength = self.config.suppression_strength * 0.4
                
            # Apply graduated suppression
            processed = self._apply_suppression(
                processed, sample_rate,
                detection.start_time, detection.end_time,
                strength=strength
            )
            
        return processed
    
    def _apply_suppression(self,
                         audio_array: np.ndarray,
                         sample_rate: int,
                         start_time: float,
                         end_time: float,
                         strength: float = 0.6) -> np.ndarray:
        """
        Apply suppression to a specific time range
        
        Args:
            audio_array: Input audio
            sample_rate: Sample rate
            start_time: Start time in seconds
            end_time: End time in seconds
            strength: Suppression strength (0-1)
            
        Returns:
            Audio with suppression applied
        """
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure valid indices
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_array), end_sample)
        
        if start_sample >= end_sample:
            return audio_array
            
        # Expand suppression window for better removal
        expansion = int(0.05 * sample_rate)  # 50ms expansion
        start_sample = max(0, start_sample - expansion)
        end_sample = min(len(audio_array), end_sample + expansion)
        
        # Create smooth envelope for suppression
        fade_samples = int(0.02 * sample_rate)  # 20ms fade (longer for smoother transition)
        
        # Extract segment
        segment = audio_array[start_sample:end_sample].copy()
        
        # For strong suppression (> 0.8), use multiple techniques
        if strength > 0.8:
            # 1. Aggressive amplitude reduction
            segment *= (1 - strength) * 0.1  # Extra reduction factor
            
            # 2. Apply spectral suppression
            if len(segment) > 256:  # Minimum length for FFT
                # FFT
                fft = np.fft.rfft(segment)
                magnitude = np.abs(fft)
                phase = np.angle(fft)
                
                # Aggressive frequency suppression across all bands
                freq_bins = len(magnitude)
                # Create aggressive suppression curve
                suppression_curve = np.ones(freq_bins) * (1 - strength) * 0.05
                
                # Apply suppression
                magnitude *= suppression_curve
                
                # Add noise floor to mask remaining signal
                noise_floor = np.random.randn(len(magnitude)) * 0.001
                magnitude += np.abs(noise_floor)
                
                # Reconstruct
                fft_suppressed = magnitude * np.exp(1j * phase)
                segment = np.fft.irfft(fft_suppressed, n=len(segment))
            
            # 3. Apply additional time-domain suppression
            # Gate-like effect for very low amplitude
            gate_threshold = np.max(np.abs(segment)) * 0.1
            segment[np.abs(segment) < gate_threshold] *= 0.01
            
        else:
            # Standard suppression for lower strength values
            # Apply spectral suppression
            if len(segment) > 256:  # Minimum length for FFT
                # FFT
                fft = np.fft.rfft(segment)
                magnitude = np.abs(fft)
                phase = np.angle(fft)
                
                # Suppress high-frequency components more (voice characteristics)
                freq_bins = len(magnitude)
                suppression_curve = np.linspace(1 - strength * 0.5, 1 - strength, freq_bins)
                
                # Apply suppression
                magnitude *= suppression_curve
                
                # Reconstruct
                fft_suppressed = magnitude * np.exp(1j * phase)
                segment = np.fft.irfft(fft_suppressed, n=len(segment))
            else:
                # Simple amplitude reduction for very short segments
                segment *= (1 - strength)
            
        # Apply fade in/out to avoid clicks
        if fade_samples > 0 and len(segment) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            segment[:fade_samples] *= fade_in
            segment[-fade_samples:] *= fade_out
        
        # For very strong suppression, don't blend with original
        if strength > 0.9:
            # Just use the heavily suppressed segment
            pass
        else:
            # Blend with original for moderate suppression
            original = audio_array[start_sample:end_sample].copy()
            segment = segment * strength + original * (1 - strength)
        
        # Apply to audio
        result = audio_array.copy()
        result[start_sample:end_sample] = segment
        
        return result
    
    def _remove_artifacts(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Remove artifacts from processed audio
        
        Args:
            audio_array: Processed audio
            sample_rate: Sample rate
            
        Returns:
            Audio with artifacts removed
        """
        # Apply gentle low-pass filter to remove high-frequency artifacts
        from scipy.signal import butter, filtfilt
        
        # Design filter
        nyquist = sample_rate / 2
        cutoff = 7000  # 7kHz cutoff
        b, a = butter(5, cutoff / nyquist, btype='low')
        
        # Apply filter
        filtered = filtfilt(b, a, audio_array)
        
        # Blend with original to preserve some high frequencies
        result = filtered * 0.8 + audio_array * 0.2
        
        # Remove clicks and pops
        result = self._remove_clicks(result, sample_rate)
        
        return result
    
    def _remove_clicks(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Remove clicks and pops from audio
        
        Args:
            audio_array: Input audio
            sample_rate: Sample rate
            
        Returns:
            Audio with clicks removed
        """
        # Detect clicks using difference threshold
        diff = np.abs(np.diff(audio_array))
        threshold = np.mean(diff) + 3 * np.std(diff)
        
        click_indices = np.where(diff > threshold)[0]
        
        # Interpolate over clicks
        result = audio_array.copy()
        for idx in click_indices:
            if 1 < idx < len(audio_array) - 2:
                # Linear interpolation
                result[idx] = (result[idx-1] + result[idx+1]) / 2
                
        return result
    
    def _calculate_metrics(self,
                         original: np.ndarray,
                         processed: np.ndarray,
                         detections: List[DetectionResult]) -> Dict[str, float]:
        """
        Calculate processing metrics
        
        Args:
            original: Original audio
            processed: Processed audio
            detections: Secondary speaker detections
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate SNR improvement
        signal_power = np.mean(processed**2)
        noise_power = np.mean((original - processed)**2)
        
        if noise_power > 0:
            snr_improvement = 10 * np.log10(signal_power / noise_power)
        else:
            snr_improvement = float('inf')
            
        metrics['snr_improvement_db'] = snr_improvement
        
        # Calculate suppression statistics
        if detections:
            total_duration = sum(d.duration for d in detections)
            avg_confidence = np.mean([d.confidence for d in detections])
            
            metrics['secondary_speaker_duration'] = total_duration
            metrics['secondary_speaker_count'] = len(detections)
            metrics['average_confidence'] = avg_confidence
        else:
            metrics['secondary_speaker_duration'] = 0.0
            metrics['secondary_speaker_count'] = 0
            metrics['average_confidence'] = 0.0
            
        # Calculate similarity preservation
        from scipy.stats import pearsonr
        
        if len(original) == len(processed):
            correlation, _ = pearsonr(original, processed)
            metrics['similarity_preservation'] = correlation
        else:
            metrics['similarity_preservation'] = 0.0
            
        return metrics
    
    def adaptive_suppression(self,
                           audio_array: np.ndarray,
                           sample_rate: int = 16000,
                           target_similarity: float = 0.95) -> np.ndarray:
        """
        Apply adaptive suppression to maintain target similarity
        
        Args:
            audio_array: Input audio
            sample_rate: Sample rate
            target_similarity: Target similarity to preserve
            
        Returns:
            Adaptively suppressed audio
        """
        # Start with full processing
        result = self.separate_speakers(audio_array, sample_rate)
        processed = result['audio']
        detections = result['detections']
        
        # Calculate current similarity
        from scipy.stats import pearsonr
        current_similarity, _ = pearsonr(audio_array, processed)
        
        # If similarity is too low, reduce suppression
        if current_similarity < target_similarity and detections:
            # Calculate adjustment factor
            adjustment = target_similarity / current_similarity
            
            # Reprocess with reduced suppression
            original_strength = self.config.suppression_strength
            self.config.suppression_strength *= adjustment
            
            processed = self._suppress_secondary_speakers(
                audio_array, sample_rate, detections
            )
            
            # Restore original config
            self.config.suppression_strength = original_strength
            
        return processed