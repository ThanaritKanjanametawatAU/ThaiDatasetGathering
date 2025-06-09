"""
Time-based Secondary Speaker Removal
More aggressive approach that silences regions with any secondary speaker activity
"""
import numpy as np
import torch
from scipy import signal
from typing import List, Tuple, Optional, Dict
import logging

from .speechbrain_separator import SpeechBrainSeparator, SeparationConfig

logger = logging.getLogger(__name__)


class TimeBasedSecondaryRemoval:
    """Remove secondary speakers using time-based segmentation"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        config = SeparationConfig(device=device)
        self.separator = SpeechBrainSeparator(config=config)
        
        # Very strict thresholds
        self.energy_threshold = 0.02
        self.min_silence_duration = 0.2  # 200ms
        self.min_speech_duration = 0.3   # 300ms
        
    def remove_all_secondary_speakers(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Aggressively remove all secondary speakers using time-based approach.
        
        Strategy:
        1. Detect all speech segments in the audio
        2. Identify which segments belong to dominant speaker
        3. Completely silence all other segments
        """
        # First, detect all speech segments
        speech_segments = self._detect_speech_segments(audio, sample_rate)
        
        if len(speech_segments) == 0:
            return np.zeros_like(audio)
        
        # Separate speakers to identify sources
        try:
            separated_sources = self.separator.separate_all_speakers(audio, sample_rate)
            
            if not separated_sources or len(separated_sources) <= 1:
                # Fallback: keep only the longest continuous segment
                return self._keep_longest_segment(audio, speech_segments, sample_rate)
            
            # Identify dominant speaker based on total speaking time
            dominant_idx = self._identify_dominant_by_time(separated_sources, sample_rate)
            
            # Create time-based mask
            mask = self._create_time_based_mask(
                audio, separated_sources[dominant_idx], speech_segments, sample_rate
            )
            
            # Apply mask aggressively
            result = audio * mask
            
            # Final cleanup - ensure complete silence in masked regions
            result = self._enforce_silence(result, mask)
            
            return result
            
        except Exception as e:
            logger.error(f"Time-based removal failed: {e}")
            return self._keep_longest_segment(audio, speech_segments, sample_rate)
    
    def _detect_speech_segments(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
        """Detect all speech segments in audio"""
        # Calculate short-time energy
        window_size = int(0.02 * sample_rate)  # 20ms
        
        energy = np.zeros(len(audio))
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i+window_size]
            energy[i:i+window_size] = np.sqrt(np.mean(window**2))
        
        # Find speech regions
        is_speech = energy > self.energy_threshold
        
        # Find segment boundaries
        diff = np.diff(np.concatenate(([0], is_speech.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Filter out short segments
        segments = []
        min_samples = int(self.min_speech_duration * sample_rate)
        
        for start, end in zip(starts, ends):
            if end - start >= min_samples:
                segments.append((start, end))
        
        return segments
    
    def _identify_dominant_by_time(self, sources: List[np.ndarray], sample_rate: int) -> int:
        """Identify dominant speaker by total speaking time"""
        speaking_times = []
        
        for i, source in enumerate(sources):
            # Calculate total time where source has significant energy
            energy = np.abs(source)
            speaking_samples = np.sum(energy > self.energy_threshold)
            speaking_time = speaking_samples / sample_rate
            speaking_times.append(speaking_time)
            logger.info(f"Source {i}: {speaking_time:.2f}s speaking time")
        
        # Return index of source with most speaking time
        dominant_idx = np.argmax(speaking_times)
        logger.info(f"Dominant speaker is source {dominant_idx}")
        return dominant_idx
    
    def _create_time_based_mask(self, original: np.ndarray, dominant_source: np.ndarray,
                                speech_segments: List[Tuple[int, int]], sample_rate: int) -> np.ndarray:
        """Create mask based on time segments where dominant speaker is active"""
        mask = np.zeros(len(original))
        
        # For each speech segment, check if dominant speaker is active
        for start, end in speech_segments:
            segment_dominant = dominant_source[start:end]
            segment_original = original[start:end]
            
            # Calculate energy ratio
            dominant_energy = np.sqrt(np.mean(segment_dominant**2))
            original_energy = np.sqrt(np.mean(segment_original**2))
            
            # If dominant speaker has significant energy in this segment, keep it
            if dominant_energy > self.energy_threshold:
                # Check if dominant speaker is primary contributor
                if original_energy > 0:
                    energy_ratio = dominant_energy / original_energy
                    # Only keep if dominant speaker contributes significantly
                    if energy_ratio > 0.5:
                        mask[start:end] = 1.0
        
        # Smooth mask transitions
        smooth_window = int(0.01 * sample_rate)  # 10ms
        if smooth_window > 1:
            mask = signal.savgol_filter(mask, smooth_window, 3)
        
        return mask
    
    def _enforce_silence(self, audio: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enforce complete silence in masked regions"""
        # Create hard binary mask
        binary_mask = (mask > 0.5).astype(float)
        
        # Apply binary mask
        result = audio * binary_mask
        
        # Double-check: force silence where mask is 0
        silence_indices = np.where(binary_mask == 0)[0]
        result[silence_indices] = 0.0
        
        return result
    
    def _keep_longest_segment(self, audio: np.ndarray, segments: List[Tuple[int, int]], 
                             sample_rate: int) -> np.ndarray:
        """Fallback: keep only the longest continuous speech segment"""
        if not segments:
            return np.zeros_like(audio)
        
        # Find longest segment
        longest_duration = 0
        longest_segment = segments[0]
        
        for start, end in segments:
            duration = (end - start) / sample_rate
            if duration > longest_duration:
                longest_duration = duration
                longest_segment = (start, end)
        
        # Create mask for longest segment only
        mask = np.zeros(len(audio))
        mask[longest_segment[0]:longest_segment[1]] = 1.0
        
        # Apply mask
        return audio * mask
    
    def process_with_metrics(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
        """Process and return metrics"""
        result = self.remove_all_secondary_speakers(audio, sample_rate)
        
        # Calculate metrics
        segments = self._detect_speech_segments(result, sample_rate)
        
        # Energy at beginning and end
        begin_samples = int(0.5 * sample_rate)
        end_samples = int(0.5 * sample_rate)
        
        begin_energy = -100
        if len(result) > begin_samples and np.any(result[:begin_samples] != 0):
            begin_energy = 20 * np.log10(np.sqrt(np.mean(result[:begin_samples]**2)) + 1e-10)
        
        end_energy = -100
        if len(result) > end_samples and np.any(result[-end_samples:] != 0):
            end_energy = 20 * np.log10(np.sqrt(np.mean(result[-end_samples:]**2)) + 1e-10)
        
        metrics = {
            'num_segments': len(segments),
            'begin_energy_db': begin_energy,
            'end_energy_db': end_energy,
            'total_silence_ratio': np.sum(result == 0) / len(result)
        }
        
        return result, metrics