"""
Overlap Detection Module

Detects overlapping speech, simultaneous speakers, and prosody discontinuities.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging
from scipy import signal
import librosa

# Optional imports
try:
    import torch
    from pyannote.audio import Inference
    from pyannote.audio.pipelines import SpeakerDiarization
    from pyannote.audio import Model as PyannoteModel
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("Pyannote not available for advanced overlap detection")

logger = logging.getLogger(__name__)


@dataclass
class OverlapConfig:
    """Configuration for overlap detection."""
    vad_onset: float = 0.5  # VAD onset threshold
    vad_offset: float = 0.4  # VAD offset threshold
    energy_window: float = 0.025  # Energy analysis window (25ms)
    energy_hop: float = 0.010  # Energy analysis hop (10ms)
    prosody_window: float = 0.1  # Prosody analysis window (100ms)
    min_overlap_duration: float = 0.05  # Minimum overlap (50ms)
    spectral_bands: int = 8  # Number of spectral bands for analysis
    anomaly_threshold: float = 2.0  # Standard deviations for anomaly


class OverlapDetector:
    """
    Detects overlapping speech and simultaneous speakers.
    Uses multiple techniques for robust detection.
    """
    
    def __init__(self, config: Optional[OverlapConfig] = None):
        """Initialize overlap detector."""
        self.config = config or OverlapConfig()
        self.diarization_pipeline = None
        
        if PYANNOTE_AVAILABLE:
            self._initialize_diarization()
    
    def _initialize_diarization(self):
        """Initialize speaker diarization pipeline."""
        try:
            # Initialize with pre-trained model
            self.diarization_pipeline = SpeakerDiarization(
                segmentation="pyannote/segmentation",
                embedding="pyannote/embedding",
                clustering="AgglomerativeClustering"
            )
            
            # Set parameters
            self.diarization_pipeline.instantiate({
                "segmentation": {
                    "min_duration_off": 0.0,
                    "threshold": self.config.vad_onset
                },
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 1,
                    "threshold": 0.7
                }
            })
            
            logger.info("Diarization pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize diarization: {e}")
            self.diarization_pipeline = None
    
    def detect_overlaps(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Detect overlapping speech in audio.
        
        Args:
            audio: Input audio array
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary containing:
            - overlaps: List of (start, end) overlap regions
            - anomalies: Energy/spectral anomaly regions
            - prosody_breaks: Prosody discontinuity points
            - confidence_scores: Confidence for each detection
        """
        results = {
            "overlaps": [],
            "anomalies": [],
            "prosody_breaks": [],
            "confidence_scores": {},
            "details": {}
        }
        
        # Run diarization-based overlap detection
        if self.diarization_pipeline and PYANNOTE_AVAILABLE:
            diarization_overlaps = self._diarization_overlap_detection(audio, sample_rate)
            results["overlaps"].extend(diarization_overlaps)
        
        # Energy-based anomaly detection
        energy_anomalies = self._energy_anomaly_detection(audio, sample_rate)
        results["anomalies"].extend(energy_anomalies)
        
        # Spectral overlap detection
        spectral_overlaps = self._spectral_overlap_detection(audio, sample_rate)
        results["overlaps"].extend(spectral_overlaps)
        
        # Prosody discontinuity detection
        prosody_breaks = self._prosody_discontinuity_detection(audio, sample_rate)
        results["prosody_breaks"].extend(prosody_breaks)
        
        # Merge and deduplicate overlaps
        results["overlaps"] = self._merge_overlaps(results["overlaps"])
        
        # Calculate confidence scores
        results["confidence_scores"] = self._calculate_confidence_scores(results)
        
        return results
    
    def _diarization_overlap_detection(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Detect overlaps using speaker diarization."""
        if not self.diarization_pipeline:
            return []
        
        try:
            # Check if torch is available
            if not PYANNOTE_AVAILABLE:
                logger.debug("Pyannote/torch not available for diarization")
                return []
                
            # Run diarization
            diarization = self.diarization_pipeline({
                "waveform": torch.from_numpy(audio).unsqueeze(0),
                "sample_rate": sample_rate
            })
            
            # Find overlapping segments
            overlaps = []
            timeline = diarization.get_timeline()
            segments = list(timeline)
            
            for i in range(len(segments) - 1):
                for j in range(i + 1, len(segments)):
                    seg1 = segments[i]
                    seg2 = segments[j]
                    
                    # Check for overlap
                    overlap_start = max(seg1.start, seg2.start)
                    overlap_end = min(seg1.end, seg2.end)
                    
                    if overlap_end > overlap_start:
                        duration = overlap_end - overlap_start
                        if duration >= self.config.min_overlap_duration:
                            overlaps.append((overlap_start, overlap_end))
            
            return overlaps
            
        except Exception as e:
            logger.error(f"Diarization overlap detection failed: {e}")
            return []
    
    def _energy_anomaly_detection(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Detect anomalies in energy patterns."""
        window_samples = int(self.config.energy_window * sample_rate)
        hop_samples = int(self.config.energy_hop * sample_rate)
        
        # Calculate energy envelope
        energies = []
        for i in range(0, len(audio) - window_samples, hop_samples):
            window = audio[i:i + window_samples]
            energy = np.sqrt(np.mean(window ** 2))  # RMS energy
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Detect anomalies
        anomalies = []
        
        if len(energies) > 10:
            # Calculate local statistics
            window_size = 20  # 200ms at 10ms hop
            
            for i in range(window_size, len(energies) - window_size):
                # Local mean and std
                local_window = energies[i-window_size:i+window_size]
                local_mean = np.mean(local_window)
                local_std = np.std(local_window)
                
                # Check for anomaly
                if abs(energies[i] - local_mean) > self.config.anomaly_threshold * local_std:
                    start_time = i * self.config.energy_hop
                    end_time = (i + 1) * self.config.energy_hop
                    
                    # Extend anomaly region
                    j = i + 1
                    while j < len(energies) - window_size:
                        if abs(energies[j] - local_mean) > self.config.anomaly_threshold * local_std:
                            end_time = (j + 1) * self.config.energy_hop
                            j += 1
                        else:
                            break
                    
                    if end_time - start_time >= self.config.min_overlap_duration:
                        anomalies.append((start_time, end_time))
        
        return self._merge_adjacent_regions(anomalies)
    
    def _spectral_overlap_detection(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Detect overlaps using spectral analysis."""
        # STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Divide into frequency bands
        freq_bins = magnitude.shape[0]
        band_size = freq_bins // self.config.spectral_bands
        
        overlaps = []
        hop_time = 512 / sample_rate
        
        # Analyze each time frame
        for t in range(magnitude.shape[1]):
            # Count active bands
            active_bands = 0
            band_energies = []
            
            for b in range(self.config.spectral_bands):
                start_bin = b * band_size
                end_bin = min((b + 1) * band_size, freq_bins)
                
                band_energy = np.sum(magnitude[start_bin:end_bin, t])
                band_energies.append(band_energy)
                
                # Check if band is active
                if band_energy > np.mean(magnitude[start_bin:end_bin, :]) * 1.5:
                    active_bands += 1
            
            # Multiple active bands might indicate overlap
            if active_bands >= self.config.spectral_bands * 0.6:
                # Check spectral spread
                band_energies = np.array(band_energies)
                spread = np.std(band_energies) / (np.mean(band_energies) + 1e-8)
                
                if spread < 0.5:  # Energy is spread across bands
                    start_time = t * hop_time
                    end_time = (t + 1) * hop_time
                    overlaps.append((start_time, end_time))
        
        return self._merge_adjacent_regions(overlaps)
    
    def _prosody_discontinuity_detection(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[float]:
        """Detect prosody discontinuities."""
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=sample_rate,
            hop_length=512,
            fmin=80,
            fmax=400
        )
        
        # Get pitch contour
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch if pitch > 0 else np.nan)
        
        pitch_contour = np.array(pitch_contour)
        
        # Find discontinuities
        discontinuities = []
        hop_time = 512 / sample_rate
        
        # Look for sudden pitch jumps
        valid_indices = ~np.isnan(pitch_contour)
        if np.sum(valid_indices) > 10:
            valid_pitches = pitch_contour[valid_indices]
            valid_times = np.where(valid_indices)[0]
            
            # Calculate pitch differences
            pitch_diff = np.abs(np.diff(valid_pitches))
            median_diff = np.median(pitch_diff)
            
            # Find large jumps
            jump_indices = np.where(pitch_diff > 3 * median_diff)[0]
            
            for idx in jump_indices:
                if idx < len(valid_times) - 1:
                    time_point = valid_times[idx] * hop_time
                    discontinuities.append(time_point)
        
        # Also check energy discontinuities
        energy_discontinuities = self._detect_energy_discontinuities(audio, sample_rate)
        discontinuities.extend(energy_discontinuities)
        
        # Remove duplicates and sort
        discontinuities = sorted(list(set(discontinuities)))
        
        return discontinuities
    
    def _detect_energy_discontinuities(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[float]:
        """Detect sudden energy changes."""
        window_samples = int(self.config.prosody_window * sample_rate)
        hop_samples = window_samples // 2
        
        discontinuities = []
        
        for i in range(0, len(audio) - 2 * window_samples, hop_samples):
            # Energy before and after
            energy_before = np.sqrt(np.mean(audio[i:i+window_samples] ** 2))
            energy_after = np.sqrt(np.mean(audio[i+window_samples:i+2*window_samples] ** 2))
            
            # Check for sudden drop or rise
            if energy_before > 0:
                ratio = energy_after / energy_before
                if ratio < 0.3 or ratio > 3.0:
                    time_point = (i + window_samples) / sample_rate
                    discontinuities.append(time_point)
        
        return discontinuities
    
    def _merge_overlaps(
        self,
        overlaps: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Merge overlapping regions."""
        if not overlaps:
            return []
        
        # Sort by start time
        overlaps.sort(key=lambda x: x[0])
        
        merged = []
        current_start, current_end = overlaps[0]
        
        for start, end in overlaps[1:]:
            if start <= current_end + 0.05:  # 50ms gap tolerance
                current_end = max(current_end, end)
            else:
                if current_end - current_start >= self.config.min_overlap_duration:
                    merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add last region
        if current_end - current_start >= self.config.min_overlap_duration:
            merged.append((current_start, current_end))
        
        return merged
    
    def _merge_adjacent_regions(
        self,
        regions: List[Tuple[float, float]],
        gap_threshold: float = 0.05
    ) -> List[Tuple[float, float]]:
        """Merge adjacent regions with small gaps."""
        if not regions:
            return []
        
        regions.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end = regions[0]
        
        for start, end in regions[1:]:
            if start - current_end <= gap_threshold:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        return merged
    
    def _calculate_confidence_scores(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Calculate confidence scores for detections."""
        confidence_scores = {}
        
        # Overlap confidence based on detection method count
        overlap_confidences = []
        for overlap in results["overlaps"]:
            # Check how many methods detected this overlap
            method_count = 0
            
            # Check against anomalies
            for anomaly in results["anomalies"]:
                if self._regions_overlap(overlap, anomaly):
                    method_count += 1
            
            # Base confidence
            confidence = 0.5 + (method_count * 0.25)
            confidence = min(1.0, confidence)
            overlap_confidences.append(confidence)
        
        confidence_scores["overlaps"] = overlap_confidences
        
        # Anomaly confidence (based on severity)
        anomaly_confidences = [0.7] * len(results["anomalies"])  # Default
        confidence_scores["anomalies"] = anomaly_confidences
        
        # Prosody break confidence
        prosody_confidences = [0.6] * len(results["prosody_breaks"])  # Default
        confidence_scores["prosody_breaks"] = prosody_confidences
        
        return confidence_scores
    
    def _regions_overlap(
        self,
        region1: Tuple[float, float],
        region2: Tuple[float, float]
    ) -> bool:
        """Check if two regions overlap."""
        start1, end1 = region1
        start2, end2 = region2
        return start1 < end2 and start2 < end1


class SimultaneousSpeechDetector:
    """
    Specialized detector for simultaneous speech patterns.
    """
    
    def __init__(self):
        self.min_simultaneous_duration = 0.1  # 100ms
        self.harmonic_threshold = 0.3
    
    def detect_simultaneous_speech(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[float, float, float]]:
        """
        Detect regions with simultaneous speech.
        
        Returns:
            List of (start, end, confidence) tuples
        """
        # Analyze harmonic structure
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        simultaneous_regions = []
        hop_time = 512 / sample_rate
        
        for t in range(magnitude.shape[1]):
            # Look for multiple harmonic series
            spectrum = magnitude[:, t]
            
            # Find peaks
            peaks = signal.find_peaks(spectrum, height=np.max(spectrum) * 0.1)[0]
            
            if len(peaks) > 10:  # Many peaks might indicate multiple speakers
                # Check for multiple fundamental frequencies
                fundamentals = self._find_fundamentals(spectrum, sample_rate)
                
                if len(fundamentals) > 1:
                    start_time = t * hop_time
                    end_time = (t + 1) * hop_time
                    confidence = min(1.0, len(fundamentals) / 3.0)
                    simultaneous_regions.append((start_time, end_time, confidence))
        
        # Merge adjacent regions
        return self._merge_simultaneous_regions(simultaneous_regions)
    
    def _find_fundamentals(
        self,
        spectrum: np.ndarray,
        sample_rate: int
    ) -> List[float]:
        """Find fundamental frequencies in spectrum."""
        # Simple autocorrelation-based method
        autocorr = np.correlate(spectrum, spectrum, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks = signal.find_peaks(autocorr, height=np.max(autocorr) * self.harmonic_threshold)[0]
        
        # Convert to frequencies
        fundamentals = []
        for peak in peaks[:3]:  # Max 3 fundamentals
            if peak > 0:
                freq = sample_rate / (2 * peak)
                if 80 <= freq <= 400:  # Human voice range
                    fundamentals.append(freq)
        
        return fundamentals
    
    def _merge_simultaneous_regions(
        self,
        regions: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Merge adjacent simultaneous speech regions."""
        if not regions:
            return []
        
        regions.sort(key=lambda x: x[0])
        merged = []
        
        current_start, current_end, current_conf = regions[0]
        
        for start, end, conf in regions[1:]:
            if start - current_end <= 0.05:  # 50ms gap
                current_end = end
                current_conf = max(current_conf, conf)
            else:
                if current_end - current_start >= self.min_simultaneous_duration:
                    merged.append((current_start, current_end, current_conf))
                current_start, current_end, current_conf = start, end, conf
        
        if current_end - current_start >= self.min_simultaneous_duration:
            merged.append((current_start, current_end, current_conf))
        
        return merged