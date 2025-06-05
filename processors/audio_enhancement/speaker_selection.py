"""
Speaker Selection Module

Provides intelligent speaker selection strategies for identifying the primary speaker
from multiple separated sources.
"""

import numpy as np
import torch
import logging
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import librosa
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class SpeakerCharacteristics:
    """Characteristics of a speaker"""
    energy: float
    pitch_mean: float
    pitch_std: float
    speaking_rate: float
    duration: float
    silence_ratio: float
    spectral_centroid: float


class SpeakerSelector:
    """
    Advanced speaker selection using multiple heuristics.
    """
    
    def __init__(self, method: str = "hybrid"):
        """
        Initialize speaker selector.
        
        Args:
            method: Selection method (energy|pitch|duration|hybrid)
        """
        self.method = method
        self.sample_rate = 16000
        
    def select_primary_speaker(self, 
                              sources: List[np.ndarray],
                              original_audio: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """
        Select the primary speaker from multiple sources.
        
        Args:
            sources: List of separated audio sources
            original_audio: Original mixed audio (optional)
            
        Returns:
            Primary speaker index and confidence score
        """
        if len(sources) == 1:
            return 0, 1.0
            
        # Extract characteristics for each source
        characteristics = [self._extract_characteristics(src) for src in sources]
        
        # Apply selection method
        if self.method == "energy":
            return self._select_by_energy(characteristics)
        elif self.method == "pitch":
            return self._select_by_pitch(characteristics)
        elif self.method == "duration":
            return self._select_by_duration(characteristics)
        elif self.method == "hybrid":
            return self._select_hybrid(characteristics, sources, original_audio)
        else:
            # Default to energy-based
            return self._select_by_energy(characteristics)
    
    def _extract_characteristics(self, audio: np.ndarray) -> SpeakerCharacteristics:
        """
        Extract speaker characteristics from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            SpeakerCharacteristics object
        """
        # Energy
        energy = np.sqrt(np.mean(audio ** 2))
        
        # Pitch analysis
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            pitch_mean = np.nanmean(f0) if f0 is not None else 0
            pitch_std = np.nanstd(f0) if f0 is not None else 0
        except:
            pitch_mean = pitch_std = 0
        
        # Speaking rate (zero-crossing rate as proxy)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        speaking_rate = np.mean(zcr)
        
        # Duration (non-silence duration)
        silence_threshold = 0.01 * np.max(np.abs(audio))
        non_silence = np.abs(audio) > silence_threshold
        duration = np.sum(non_silence) / self.sample_rate
        silence_ratio = 1 - (np.sum(non_silence) / len(audio))
        
        # Spectral centroid
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        )
        
        return SpeakerCharacteristics(
            energy=float(energy),
            pitch_mean=float(pitch_mean),
            pitch_std=float(pitch_std),
            speaking_rate=float(speaking_rate),
            duration=float(duration),
            silence_ratio=float(silence_ratio),
            spectral_centroid=float(spectral_centroid)
        )
    
    def _select_by_energy(self, characteristics: List[SpeakerCharacteristics]) -> Tuple[int, float]:
        """
        Select speaker with highest energy.
        """
        energies = [c.energy for c in characteristics]
        primary_idx = np.argmax(energies)
        
        # Confidence based on energy ratio
        total_energy = sum(energies)
        confidence = energies[primary_idx] / total_energy if total_energy > 0 else 0
        
        return primary_idx, confidence
    
    def _select_by_pitch(self, characteristics: List[SpeakerCharacteristics]) -> Tuple[int, float]:
        """
        Select speaker with most stable pitch (lower std deviation).
        """
        # Score based on pitch stability and presence
        scores = []
        for c in characteristics:
            if c.pitch_mean > 0:
                # Lower std is better, higher mean pitch indicates clear speech
                score = c.pitch_mean / (c.pitch_std + 1)
            else:
                score = 0
            scores.append(score)
        
        primary_idx = np.argmax(scores)
        confidence = scores[primary_idx] / (sum(scores) + 1e-10)
        
        return primary_idx, confidence
    
    def _select_by_duration(self, characteristics: List[SpeakerCharacteristics]) -> Tuple[int, float]:
        """
        Select speaker with longest speaking duration.
        """
        durations = [c.duration for c in characteristics]
        primary_idx = np.argmax(durations)
        
        # Confidence based on duration ratio
        total_duration = sum(durations)
        confidence = durations[primary_idx] / total_duration if total_duration > 0 else 0
        
        return primary_idx, confidence
    
    def _select_hybrid(self, 
                      characteristics: List[SpeakerCharacteristics],
                      sources: List[np.ndarray],
                      original_audio: Optional[np.ndarray]) -> Tuple[int, float]:
        """
        Hybrid selection using multiple criteria.
        """
        scores = []
        
        for i, c in enumerate(characteristics):
            # Weighted scoring
            score = 0
            
            # Energy contribution (40%)
            energy_score = c.energy / (sum(ch.energy for ch in characteristics) + 1e-10)
            score += 0.4 * energy_score
            
            # Duration contribution (30%)
            duration_score = c.duration / (sum(ch.duration for ch in characteristics) + 1e-10)
            score += 0.3 * duration_score
            
            # Pitch stability contribution (20%)
            if c.pitch_mean > 0:
                pitch_score = 1 / (1 + c.pitch_std / c.pitch_mean)
            else:
                pitch_score = 0
            score += 0.2 * pitch_score
            
            # Speech presence contribution (10%)
            speech_score = 1 - c.silence_ratio
            score += 0.1 * speech_score
            
            scores.append(score)
        
        # Select highest scoring speaker
        primary_idx = np.argmax(scores)
        confidence = scores[primary_idx]
        
        # Additional confidence adjustment based on score distribution
        score_std = np.std(scores)
        if score_std < 0.1:  # All speakers are similar
            confidence *= 0.7
        
        return primary_idx, confidence
    
    def validate_selection(self, 
                          selected_audio: np.ndarray,
                          all_sources: List[np.ndarray]) -> Dict[str, float]:
        """
        Validate the speaker selection decision.
        
        Args:
            selected_audio: The selected primary speaker audio
            all_sources: All separated sources
            
        Returns:
            Validation metrics
        """
        metrics = {}
        
        # Check if selected speaker has reasonable characteristics
        char = self._extract_characteristics(selected_audio)
        
        # Energy should be reasonable
        metrics['energy_valid'] = 0.001 < char.energy < 1.0
        
        # Should have some speech content
        metrics['has_speech'] = char.duration > 0.5 and char.silence_ratio < 0.8
        
        # Pitch should be in human range if detected
        if char.pitch_mean > 0:
            metrics['pitch_valid'] = 80 < char.pitch_mean < 400
        else:
            metrics['pitch_valid'] = True  # No pitch is okay
        
        # Overall validation score
        metrics['validation_score'] = sum([
            metrics['energy_valid'],
            metrics['has_speech'],
            metrics['pitch_valid']
        ]) / 3.0
        
        return metrics