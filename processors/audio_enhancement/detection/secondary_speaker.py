"""
Secondary Speaker Detection Module

Multi-modal detection system for identifying secondary speakers in audio recordings.
Supports flexible duration detection from 0.1s to 5s and uses multiple detection
methods including speaker embeddings, VAD, energy analysis, and spectral features.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple
import torch
from scipy import signal
from scipy.stats import skew, kurtosis
import logging
from dataclasses import dataclass

# Use existing speaker identification system
from processors.speaker_identification import SpeakerIdentification

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of secondary speaker detection"""
    start_time: float
    end_time: float
    confidence: float
    detection_methods: List[str]
    speaker_similarity: Optional[float] = None
    energy_ratio: Optional[float] = None
    spectral_distance: Optional[float] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class SecondarySpeakerDetector:
    """Base class for secondary speaker detection"""
    
    def __init__(self,
                 min_duration: float = 0.1,
                 max_duration: float = 5.0,
                 confidence_threshold: float = 0.5):
        """
        Initialize detector
        
        Args:
            min_duration: Minimum duration to detect (seconds)
            max_duration: Maximum duration to detect (seconds)
            confidence_threshold: Minimum confidence for detection
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.confidence_threshold = confidence_threshold
        
    def detect(self, audio_array: np.ndarray, sample_rate: int = 16000) -> List[DetectionResult]:
        """
        Detect secondary speakers in audio
        
        Args:
            audio_array: Audio signal
            sample_rate: Sample rate
            
        Returns:
            List of detection results
        """
        raise NotImplementedError


class AdaptiveSecondaryDetection(SecondarySpeakerDetector):
    """
    Adaptive multi-modal secondary speaker detection
    
    Uses multiple detection methods and adapts to different speech patterns
    """
    
    def __init__(self,
                 min_duration: float = 0.1,
                 max_duration: float = 5.0,
                 speaker_similarity_threshold: float = 0.7,
                 energy_threshold: float = 0.3,
                 spectral_threshold: float = 0.5,
                 confidence_threshold: float = 0.5,
                 detection_methods: Optional[List[str]] = None):
        """
        Initialize adaptive detector
        
        Args:
            min_duration: Minimum duration to detect (seconds)
            max_duration: Maximum duration to detect (seconds)  
            speaker_similarity_threshold: Threshold for speaker embedding similarity
            energy_threshold: Threshold for energy-based detection
            spectral_threshold: Threshold for spectral-based detection
            confidence_threshold: Minimum confidence for detection
            detection_methods: List of methods to use (default: all)
        """
        super().__init__(min_duration, max_duration, confidence_threshold)
        
        self.speaker_similarity_threshold = speaker_similarity_threshold
        self.energy_threshold = energy_threshold
        self.spectral_threshold = spectral_threshold
        
        # Available detection methods
        self.available_methods = ["embedding", "vad", "energy", "spectral"]
        self.detection_methods = detection_methods or self.available_methods
        
        # Initialize speaker identification system
        try:
            # Create a minimal config for speaker identification
            speaker_config = {
                'model': 'pyannote/embedding',
                'store_embeddings': False,
                'fresh': True,
                'clustering': {
                    'algorithm': 'hdbscan',
                    'min_cluster_size': 2,
                    'min_samples': 1,
                    'metric': 'cosine'
                }
            }
            self.speaker_id = SpeakerIdentification(speaker_config)
            self.has_speaker_id = True
        except Exception as e:
            logger.warning(f"Could not initialize speaker identification: {e}")
            self.has_speaker_id = False
            
    def detect(self, audio_array: np.ndarray, sample_rate: int = 16000) -> List[DetectionResult]:
        """
        Detect secondary speakers using multiple methods
        
        Args:
            audio_array: Audio signal
            sample_rate: Sample rate
            
        Returns:
            List of detection results
        """
        all_detections = []
        
        # Method 1: Speaker embedding detection
        if "embedding" in self.detection_methods and self.has_speaker_id:
            embedding_detections = self._detect_by_embedding(audio_array, sample_rate)
            all_detections.extend(embedding_detections)
            
        # Method 2: VAD-based detection
        if "vad" in self.detection_methods:
            vad_detections = self._detect_by_vad(audio_array, sample_rate)
            all_detections.extend(vad_detections)
            
        # Method 3: Energy-based detection
        if "energy" in self.detection_methods:
            energy_detections = self._detect_by_energy(audio_array, sample_rate)
            all_detections.extend(energy_detections)
            
        # Method 4: Spectral feature detection
        if "spectral" in self.detection_methods:
            spectral_detections = self._detect_by_spectral(audio_array, sample_rate)
            all_detections.extend(spectral_detections)
            
        # Merge and filter detections
        merged_detections = self._merge_detections(all_detections)
        filtered_detections = [
            d for d in merged_detections 
            if d.confidence >= self.confidence_threshold and
               self.min_duration <= d.duration <= self.max_duration
        ]
        
        return filtered_detections
    
    def _detect_by_embedding(self, audio_array: np.ndarray, sample_rate: int) -> List[DetectionResult]:
        """
        Detect secondary speakers using speaker embeddings
        
        Args:
            audio_array: Audio signal
            sample_rate: Sample rate
            
        Returns:
            List of detection results
        """
        detections = []
        
        # Extract main speaker profile from multiple segments for better representation
        # Use first 2 seconds and last 2 seconds to get better main speaker profile
        profile_segments = []
        
        # First 2 seconds
        if len(audio_array) > 2 * sample_rate:
            profile_segments.append(audio_array[:2 * sample_rate])
        else:
            profile_segments.append(audio_array[:sample_rate])
            
        # Last 2 seconds
        if len(audio_array) > 4 * sample_rate:
            profile_segments.append(audio_array[-2 * sample_rate:])
            
        # Extract embeddings for profile segments
        main_embeddings = []
        for segment in profile_segments:
            try:
                embedding = self.speaker_id.extract_embedding(segment, sample_rate)
                main_embeddings.append(embedding)
            except Exception as e:
                logger.debug(f"Failed to extract embedding: {e}")
                
        if not main_embeddings:
            logger.warning("Failed to extract any main speaker embeddings")
            return detections
            
        # Average embeddings for better representation
        main_embedding = np.mean(main_embeddings, axis=0)
            
        # Sliding window analysis with smaller windows for better granularity
        window_size = int(0.3 * sample_rate)  # 300ms windows (was 500ms)
        hop_size = int(0.05 * sample_rate)    # 50ms hop (was 100ms)
        
        for i in range(0, len(audio_array) - window_size, hop_size):
            segment = audio_array[i:i+window_size]
            
            try:
                segment_embedding = self.speaker_id.extract_embedding(segment, sample_rate)
                
                # Calculate similarity with main speaker
                similarity = self._cosine_similarity(main_embedding, segment_embedding)
                
                # If similarity is low, possible secondary speaker
                if similarity < self.speaker_similarity_threshold:
                    # Expand window to find full segment
                    start, end = self._expand_detection_window(
                        audio_array, i, i + window_size, sample_rate, similarity
                    )
                    
                    detection = DetectionResult(
                        start_time=start / sample_rate,
                        end_time=end / sample_rate,
                        confidence=1.0 - similarity,
                        detection_methods=["embedding"],
                        speaker_similarity=similarity
                    )
                    
                    detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Failed to process segment: {e}")
                continue
                
        return self._merge_adjacent_detections(detections)
    
    def _detect_by_vad(self, audio_array: np.ndarray, sample_rate: int) -> List[DetectionResult]:
        """
        Detect using Voice Activity Detection patterns
        
        Args:
            audio_array: Audio signal
            sample_rate: Sample rate
            
        Returns:
            List of detection results
        """
        detections = []
        
        # Simple energy-based VAD
        frame_size = int(0.02 * sample_rate)  # 20ms frames
        hop_size = int(0.01 * sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio_array) - frame_size, hop_size):
            frame = audio_array[i:i+frame_size]
            energy.append(np.sqrt(np.mean(frame**2)))
            
        energy = np.array(energy)
        
        # Find speech segments
        threshold = np.mean(energy) * 0.5
        speech_mask = energy > threshold
        
        # Find transitions
        transitions = np.diff(speech_mask.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        # Ensure proper pairing
        if len(starts) > 0 and len(ends) > 0:
            if starts[0] > ends[0]:
                ends = ends[1:]
            if len(starts) > len(ends):
                starts = starts[:len(ends)]
                
        # Analyze segments for anomalies
        for start_idx, end_idx in zip(starts, ends):
            segment_energy = energy[start_idx:end_idx]
            
            # Check for energy dips or spikes that might indicate speaker change
            if len(segment_energy) > 5:
                energy_std = np.std(segment_energy)
                energy_mean = np.mean(segment_energy)
                
                # Look for significant variations
                variations = np.abs(segment_energy - energy_mean)
                if np.any(variations > 2 * energy_std):
                    start_time = start_idx * hop_size / sample_rate
                    end_time = end_idx * hop_size / sample_rate
                    
                    detection = DetectionResult(
                        start_time=start_time,
                        end_time=end_time,
                        confidence=0.6,
                        detection_methods=["vad"]
                    )
                    
                    detections.append(detection)
                    
        return detections
    
    def _detect_by_energy(self, audio_array: np.ndarray, sample_rate: int) -> List[DetectionResult]:
        """
        Detect using energy-based analysis
        
        Args:
            audio_array: Audio signal
            sample_rate: Sample rate
            
        Returns:
            List of detection results
        """
        detections = []
        
        # Calculate short-term energy
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_size = int(0.01 * sample_rate)     # 10ms hop
        
        energy_profile = []
        for i in range(0, len(audio_array) - frame_size, hop_size):
            frame = audio_array[i:i+frame_size]
            energy_profile.append(np.sqrt(np.mean(frame**2)))
            
        energy_profile = np.array(energy_profile)
        
        # Smooth energy profile
        from scipy.ndimage import gaussian_filter1d
        smoothed_energy = gaussian_filter1d(energy_profile, sigma=5)
        
        # Find local minima and maxima
        from scipy.signal import find_peaks
        
        # Find peaks (potential new speaker onsets)
        peaks, peak_props = find_peaks(smoothed_energy, 
                                     prominence=np.std(smoothed_energy) * 0.5,
                                     distance=20)  # Minimum 200ms between peaks
        
        # Find valleys (potential speaker transitions)
        valleys, valley_props = find_peaks(-smoothed_energy,
                                         prominence=np.std(smoothed_energy) * 0.3,
                                         distance=10)
        
        # Analyze transitions
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            
            # Find valley between peaks
            valleys_between = valleys[(valleys > peak1) & (valleys < peak2)]
            
            if len(valleys_between) > 0:
                valley = valleys_between[0]
                
                # Check energy ratio
                peak_energy = smoothed_energy[peak2]
                valley_energy = smoothed_energy[valley]
                
                if valley_energy < peak_energy * self.energy_threshold:
                    start_time = valley * hop_size / sample_rate
                    end_time = peak2 * hop_size / sample_rate
                    
                    detection = DetectionResult(
                        start_time=start_time,
                        end_time=end_time,
                        confidence=0.7,
                        detection_methods=["energy"],
                        energy_ratio=valley_energy / peak_energy
                    )
                    
                    detections.append(detection)
                    
        return detections
    
    def _detect_by_spectral(self, audio_array: np.ndarray, sample_rate: int) -> List[DetectionResult]:
        """
        Detect using spectral features
        
        Args:
            audio_array: Audio signal
            sample_rate: Sample rate
            
        Returns:
            List of detection results
        """
        detections = []
        
        # Extract spectral features
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = int(0.05 * sample_rate)   # 50ms hop
        
        spectral_features = []
        
        for i in range(0, len(audio_array) - window_size, hop_size):
            segment = audio_array[i:i+window_size]
            
            # Calculate spectral features
            features = self._extract_spectral_features(segment, sample_rate)
            spectral_features.append(features)
            
        if not spectral_features:
            return detections
            
        spectral_features = np.array(spectral_features)
        
        # Check if we have enough data
        if spectral_features.ndim < 2 or spectral_features.shape[0] < 2:
            return detections
        
        # Find anomalies in spectral trajectory
        for feat_idx in range(spectral_features.shape[1]):
            feature = spectral_features[:, feat_idx]
            
            # Calculate local statistics
            window = min(20, len(feature) // 3)  # Adaptive window size
            
            if window < 2:
                continue
                
            for i in range(window, len(feature) - window):
                local_mean = np.mean(feature[i-window:i+window])
                local_std = np.std(feature[i-window:i+window])
                
                # Check for significant deviations
                if abs(feature[i] - local_mean) > 2.5 * local_std:
                    start_time = (i - 2) * hop_size / sample_rate
                    end_time = (i + 2) * hop_size / sample_rate
                    
                    detection = DetectionResult(
                        start_time=max(0, start_time),
                        end_time=min(len(audio_array) / sample_rate, end_time),
                        confidence=0.6,
                        detection_methods=["spectral"],
                        spectral_distance=abs(feature[i] - local_mean) / local_std
                    )
                    
                    detections.append(detection)
                    
        return self._merge_adjacent_detections(detections)
    
    def _extract_spectral_features(self, segment: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract spectral features from audio segment
        
        Args:
            segment: Audio segment
            sample_rate: Sample rate
            
        Returns:
            Feature vector
        """
        # Compute spectrum
        fft = np.fft.rfft(segment)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(segment), 1/sample_rate)
        
        # Spectral centroid
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0
            
        # Spectral spread
        if np.sum(magnitude) > 0:
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
        else:
            spectral_spread = 0
            
        # Spectral flux
        spectral_flux = np.sum(np.diff(magnitude) ** 2)
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0][0]
            spectral_rolloff = freqs[rolloff_idx]
        else:
            spectral_rolloff = 0
            
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(segment)))) / (2 * len(segment))
        
        return np.array([spectral_centroid, spectral_spread, spectral_flux, spectral_rolloff, zcr])
    
    def _expand_detection_window(self, 
                               audio_array: np.ndarray,
                               start_idx: int,
                               end_idx: int,
                               sample_rate: int,
                               similarity: float) -> Tuple[int, int]:
        """
        Expand detection window to capture full secondary speaker segment
        
        Args:
            audio_array: Audio signal
            start_idx: Initial start index
            end_idx: Initial end index
            sample_rate: Sample rate
            similarity: Speaker similarity score
            
        Returns:
            Expanded (start, end) indices
        """
        # Expansion parameters based on confidence
        expansion_factor = max(0.5, 1.0 - similarity)
        max_expansion = int(self.max_duration * sample_rate)
        
        # Backward expansion
        energy_threshold = np.std(audio_array) * 0.1
        new_start = start_idx
        
        while new_start > 0 and start_idx - new_start < max_expansion:
            if abs(audio_array[new_start]) < energy_threshold:
                break
            new_start -= 1
            
        # Forward expansion
        new_end = end_idx
        
        while new_end < len(audio_array) and new_end - end_idx < max_expansion:
            if abs(audio_array[new_end]) < energy_threshold:
                break
            new_end += 1
            
        return new_start, new_end
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score [0, 1]
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return (dot_product / (norm1 * norm2) + 1) / 2  # Normalize to [0, 1]
    
    def _merge_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Merge overlapping detections from different methods
        
        Args:
            detections: List of detections
            
        Returns:
            Merged detections
        """
        if not detections:
            return []
            
        # Sort by start time
        sorted_detections = sorted(detections, key=lambda d: d.start_time)
        
        merged = []
        current = sorted_detections[0]
        
        for detection in sorted_detections[1:]:
            # Check for overlap
            if detection.start_time <= current.end_time:
                # Merge detections
                methods = list(set(current.detection_methods + detection.detection_methods))
                
                # Combine confidence scores
                confidence = max(current.confidence, detection.confidence)
                
                # Update other metrics
                speaker_similarity = min(
                    current.speaker_similarity or 1.0,
                    detection.speaker_similarity or 1.0
                )
                
                current = DetectionResult(
                    start_time=current.start_time,
                    end_time=max(current.end_time, detection.end_time),
                    confidence=confidence,
                    detection_methods=methods,
                    speaker_similarity=speaker_similarity,
                    energy_ratio=current.energy_ratio or detection.energy_ratio,
                    spectral_distance=max(
                        current.spectral_distance or 0,
                        detection.spectral_distance or 0
                    )
                )
            else:
                merged.append(current)
                current = detection
                
        merged.append(current)
        
        return merged
    
    def _merge_adjacent_detections(self, 
                                 detections: List[DetectionResult],
                                 gap_threshold: float = 0.2) -> List[DetectionResult]:
        """
        Merge detections that are close together
        
        Args:
            detections: List of detections
            gap_threshold: Maximum gap to merge (seconds)
            
        Returns:
            Merged detections
        """
        if not detections:
            return []
            
        sorted_detections = sorted(detections, key=lambda d: d.start_time)
        merged = [sorted_detections[0]]
        
        for detection in sorted_detections[1:]:
            last = merged[-1]
            
            # Check if gap is small enough to merge
            if detection.start_time - last.end_time <= gap_threshold:
                # Merge detections
                merged[-1] = DetectionResult(
                    start_time=last.start_time,
                    end_time=detection.end_time,
                    confidence=max(last.confidence, detection.confidence),
                    detection_methods=list(set(last.detection_methods + detection.detection_methods)),
                    speaker_similarity=min(
                        last.speaker_similarity or 1.0,
                        detection.speaker_similarity or 1.0
                    ),
                    energy_ratio=last.energy_ratio or detection.energy_ratio,
                    spectral_distance=max(
                        last.spectral_distance or 0,
                        detection.spectral_distance or 0
                    )
                )
            else:
                merged.append(detection)
                
        return merged