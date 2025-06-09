"""
Complete Secondary Speaker Removal Implementation
Ensures only the dominant speaker remains in the audio
"""
import numpy as np
import torch
from scipy import signal
from typing import List, Tuple, Optional
import logging

from .dominant_speaker_separation import DominantSpeakerSeparator, SpeakerActivity
from .speechbrain_separator import SpeechBrainSeparator, SeparationConfig

logger = logging.getLogger(__name__)


class CompleteSecondaryRemoval:
    """Completely removes all secondary speakers from audio"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        # Create config for SpeechBrainSeparator
        config = SeparationConfig(device=device)
        self.separator = SpeechBrainSeparator(config=config)
        self.dominant_analyzer = DominantSpeakerSeparator(device=device)
        
        # Aggressive silence thresholds
        self.silence_threshold = 0.001  # Very low threshold for silence
        self.activity_threshold = 0.01   # Threshold for voice activity
        self.min_segment_duration = 0.1  # Minimum 100ms segments
        
    def remove_all_secondary_speakers(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Completely remove all secondary speakers from audio.
        
        This method:
        1. Separates all speakers
        2. Identifies the dominant speaker
        3. Creates a mask for dominant speaker activity
        4. Applies aggressive silencing to all other regions
        
        Args:
            audio: Input audio with multiple speakers
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio with only the dominant speaker
        """
        # First, separate all speakers
        try:
            separated_sources = self.separator.separate_all_speakers(audio, sample_rate)
            
            if not separated_sources or len(separated_sources) <= 1:
                logger.warning("No separation performed, applying fallback")
                return self._fallback_removal(audio, sample_rate)
            
            # Identify dominant speaker
            dominant_idx = self.dominant_analyzer.identify_dominant_speaker(separated_sources, sample_rate)
            dominant_source = separated_sources[dominant_idx]
            
            # Analyze when dominant speaker is active
            dominant_mask = self._create_speaker_activity_mask(dominant_source, sample_rate)
            
            # IMPORTANT: Use the separated dominant source, not the original mixed audio!
            # This ensures we only get the dominant speaker's voice
            result = self._apply_strict_mask(dominant_source, dominant_mask, dominant_source)
            
            # Final cleanup - ensure no secondary speakers remain
            result = self._final_cleanup(result, sample_rate)
            
            return result
            
        except Exception as e:
            logger.error(f"Complete removal failed: {e}")
            return self._fallback_removal(audio, sample_rate)
    
    def _create_speaker_activity_mask(self, speaker_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create a binary mask for when the speaker is active"""
        # Calculate short-time energy
        window_size = int(0.02 * sample_rate)  # 20ms windows
        hop_size = int(0.01 * sample_rate)     # 10ms hop
        
        # Pad audio for windowing
        padded = np.pad(speaker_audio, (window_size//2, window_size//2), mode='constant')
        
        # Calculate energy for each window
        energy = np.zeros(len(speaker_audio))
        
        for i in range(0, len(speaker_audio) - window_size, hop_size):
            window = padded[i:i+window_size]
            window_energy = np.sqrt(np.mean(window**2))
            
            # Fill the energy array
            start_idx = i
            end_idx = min(i + window_size, len(energy))
            energy[start_idx:end_idx] = np.maximum(energy[start_idx:end_idx], window_energy)
        
        # NEW: More sophisticated activity detection
        # Find the median energy level (more robust than percentile)
        energy_nonzero = energy[energy > 0]
        if len(energy_nonzero) > 0:
            # Use lower percentile as baseline to capture more of the primary speaker
            baseline_energy = np.percentile(energy_nonzero, 25)  # 25th percentile
            # Set threshold to 1.5x baseline (was 2x median)
            threshold = max(self.activity_threshold, baseline_energy * 1.5)
        else:
            threshold = self.activity_threshold
        
        # Create binary mask
        mask = (energy > threshold).astype(float)
        
        # Less aggressive filtering to preserve primary speaker
        # Primary speaker typically has longer segments but we shouldn't be too strict
        min_segment_samples = int(0.15 * sample_rate)  # 150ms minimum (was 300ms)
        
        # Find continuous segments
        diff = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Collect segment info for debugging
        segments_info = []
        for start, end in zip(starts, ends):
            duration = (end - start) / sample_rate
            avg_energy = np.mean(energy[start:end])
            segments_info.append((start/sample_rate, end/sample_rate, duration, avg_energy))
        
        if segments_info:
            logger.debug(f"Found {len(segments_info)} segments: {[(s[0], s[1], s[2]) for s in segments_info[:3]]}")
        
        # Filter: keep longer segments with sufficient energy
        filtered_mask = np.zeros_like(mask)
        for start, end in zip(starts, ends):
            segment_duration = (end - start) / sample_rate
            segment_energy = np.mean(energy[start:end])
            
            # Keep segment if it's long enough OR has high energy
            # This helps preserve primary speaker segments
            if (end - start >= min_segment_samples) or (segment_energy > threshold * 1.2):
                filtered_mask[start:end] = 1.0
        
        # Slight expansion to avoid cutting speech
        expansion_samples = int(0.02 * sample_rate)  # 20ms expansion
        if expansion_samples > 1:
            expanded_mask = signal.convolve(filtered_mask, np.ones(expansion_samples)/expansion_samples, mode='same')
            expanded_mask = (expanded_mask > 0.1).astype(float)
        else:
            expanded_mask = filtered_mask
        
        return expanded_mask
    
    def _apply_strict_mask(self, original: np.ndarray, mask: np.ndarray, 
                          dominant_source: np.ndarray) -> np.ndarray:
        """Apply mask strictly to remove secondary speakers"""
        # Ensure same length
        min_len = min(len(original), len(mask), len(dominant_source))
        original = original[:min_len]
        mask = mask[:min_len]
        dominant_source = dominant_source[:min_len]
        
        # Create binary mask - be very strict
        binary_mask = (mask > 0.5).astype(float)
        
        # Apply mask to dominant source for cleaner result
        result = dominant_source * binary_mask
        
        # Additional aggressive silence enforcement
        # Find ALL regions where mask is not clearly active
        silence_regions = np.where(mask < 0.9)[0]
        
        # Force these regions to absolute silence
        if len(silence_regions) > 0:
            result[silence_regions] = 0.0
        
        # Double-check: ensure no audio outside dominant speaker regions
        # This is critical for complete removal
        for i in range(len(result)):
            if binary_mask[i] == 0:
                result[i] = 0.0
        
        return result
    
    def _final_cleanup(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Final cleanup to ensure no secondary speakers remain"""
        # Identify potential secondary speaker regions (end of audio is common)
        # Check last 500ms
        last_500ms_samples = int(0.5 * sample_rate)
        
        if len(audio) > last_500ms_samples:
            # Analyze energy in the last portion
            last_segment = audio[-last_500ms_samples:]
            
            # If there's significant energy but it's likely a secondary speaker
            # (can be detected by sudden energy after silence)
            mid_point = len(audio) - last_500ms_samples
            
            # Check if there was silence before this segment
            check_window = int(0.2 * sample_rate)
            if mid_point > check_window:
                before_segment = audio[mid_point-check_window:mid_point]
                before_energy = np.sqrt(np.mean(before_segment**2))
                last_energy = np.sqrt(np.mean(last_segment**2))
                
                # If energy suddenly appears after silence, it's likely secondary
                if before_energy < 0.01 and last_energy > 0.02:
                    logger.info("Detected likely secondary speaker at end, removing")
                    audio[-last_500ms_samples:] *= 0.0
        
        # Apply final gate to remove any low-level artifacts
        envelope = np.abs(signal.hilbert(audio))
        gate = (envelope > self.silence_threshold).astype(float)
        
        # Smooth the gate
        smooth_samples = int(0.005 * sample_rate)  # 5ms
        if smooth_samples > 1:
            gate = signal.savgol_filter(gate, smooth_samples, 3)
        
        audio = audio * gate
        
        return audio
    
    def _fallback_removal(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Fallback method when separation fails - use energy-based segmentation"""
        logger.info("Using fallback secondary removal method")
        
        # More sophisticated energy-based approach
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = int(0.05 * sample_rate)    # 50ms hop
        
        # Calculate energy profile
        energy_profile = []
        timestamps = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            energy = np.sqrt(np.mean(window**2))
            energy_profile.append(energy)
            timestamps.append(i / sample_rate)
        
        energy_profile = np.array(energy_profile)
        
        # Find the longest continuous high-energy segment
        # This is likely the primary speaker
        threshold = np.percentile(energy_profile[energy_profile > 0], 30)
        is_speech = energy_profile > threshold
        
        # Find continuous segments
        segments = []
        start_idx = None
        
        for i, active in enumerate(is_speech):
            if active and start_idx is None:
                start_idx = i
            elif not active and start_idx is not None:
                segments.append((start_idx, i))
                start_idx = None
        
        if start_idx is not None:
            segments.append((start_idx, len(is_speech)))
        
        # Find the longest segment (likely primary speaker)
        if not segments:
            return np.zeros_like(audio)
        
        longest_segment = max(segments, key=lambda x: x[1] - x[0])
        
        # Create mask with smooth transitions
        mask = np.zeros(len(audio))
        
        start_sample = longest_segment[0] * hop_size
        end_sample = min(longest_segment[1] * hop_size + window_size, len(audio))
        
        # Apply mask with smooth fade in/out
        fade_samples = int(0.02 * sample_rate)  # 20ms fade
        
        # Main segment
        mask[start_sample:end_sample] = 1.0
        
        # Fade in
        if start_sample > fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            mask[start_sample-fade_samples:start_sample] = fade_in
        
        # Fade out
        if end_sample + fade_samples < len(mask):
            fade_out = np.linspace(1, 0, fade_samples)
            mask[end_sample:end_sample+fade_samples] = fade_out
        
        # Apply mask
        result = audio * mask
        
        logger.info(f"Fallback: kept segment from {start_sample/sample_rate:.1f}s to {end_sample/sample_rate:.1f}s")
        
        return result
    
    def process_with_verification(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, dict]:
        """Process audio and return verification metrics"""
        # Process
        result = self.remove_all_secondary_speakers(audio, sample_rate)
        
        # Verify removal effectiveness
        metrics = self._calculate_removal_metrics(audio, result, sample_rate)
        
        return result, metrics
    
    def _calculate_removal_metrics(self, original: np.ndarray, processed: np.ndarray, 
                                  sample_rate: int) -> dict:
        """Calculate metrics to verify secondary speaker removal"""
        # Check silence at typical secondary speaker locations
        # Beginning (first 500ms)
        begin_samples = int(0.5 * sample_rate)
        begin_energy = -100
        if len(processed) > begin_samples:
            segment = processed[:begin_samples]
            if np.any(segment != 0):
                begin_energy = 20 * np.log10(np.sqrt(np.mean(segment**2)) + 1e-10)
        
        # End (last 500ms)
        end_samples = int(0.5 * sample_rate)
        end_energy = -100
        if len(processed) > end_samples:
            segment = processed[-end_samples:]
            if np.any(segment != 0):
                end_energy = 20 * np.log10(np.sqrt(np.mean(segment**2)) + 1e-10)
        
        # Overall statistics
        total_samples = len(processed)
        silent_samples = np.sum(np.abs(processed) < self.silence_threshold)
        silence_ratio = silent_samples / total_samples
        
        # Activity detection
        activity_mask = np.abs(processed) > self.activity_threshold
        num_active_regions = self._count_active_regions(activity_mask)
        
        return {
            'begin_energy_db': begin_energy,
            'end_energy_db': end_energy,
            'silence_ratio': silence_ratio,
            'num_active_regions': num_active_regions,
            'removal_effective': begin_energy < -50 and end_energy < -50
        }
    
    def _count_active_regions(self, mask: np.ndarray) -> int:
        """Count number of separate active regions"""
        # Find transitions
        diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        return len(starts)