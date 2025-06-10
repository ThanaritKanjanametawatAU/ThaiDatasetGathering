"""
Dominant Speaker Identifier Module (S03_T05)
Identifies the primary/dominant speaker in multi-speaker audio segments
"""

import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import torch
from scipy.spatial.distance import cosine
import warnings

logger = logging.getLogger(__name__)


class DominanceMethod(Enum):
    """Methods for determining speaker dominance"""
    DURATION = "duration"  # Based on speaking time
    ENERGY = "energy"      # Based on speech energy/volume
    HYBRID = "hybrid"      # Combination of duration and energy


@dataclass
class DominanceConfig:
    """Configuration for dominance calculation"""
    min_duration_ratio: float = 0.1      # Minimum ratio to be considered
    energy_weight: float = 0.5           # Weight for energy in hybrid mode
    duration_weight: float = 0.5         # Weight for duration in hybrid mode
    overlap_handling: str = "split"      # How to handle overlaps: 'split', 'ignore', 'assign'
    similarity_threshold: float = 0.85   # Threshold for speaker similarity


@dataclass
class SpeakerDominance:
    """Result of dominant speaker identification"""
    dominant_speaker: Optional[str]
    confidence: float
    total_speakers: int
    speaker_durations: Dict[str, float] = field(default_factory=dict)
    speaker_energies: Dict[str, float] = field(default_factory=dict)
    speaker_ratios: Dict[str, float] = field(default_factory=dict)
    is_balanced: bool = False
    overlap_regions: List[Tuple[float, float]] = field(default_factory=list)
    qualified_speakers: List[str] = field(default_factory=list)
    speaker_similarities: Optional[Dict[Tuple[str, str], float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DominantSpeakerIdentifier:
    """Identifies the dominant speaker in multi-speaker audio"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 dominance_method: DominanceMethod = DominanceMethod.DURATION,
                 config: Optional[DominanceConfig] = None,
                 streaming_mode: bool = False,
                 window_duration: float = 5.0,
                 update_interval: float = 1.0):
        """Initialize the Dominant Speaker Identifier.
        
        Args:
            sample_rate: Audio sample rate
            dominance_method: Method for determining dominance
            config: Configuration parameters
            streaming_mode: Enable streaming mode
            window_duration: Window size for streaming mode
            update_interval: Update interval for streaming mode
        """
        self.sample_rate = sample_rate
        self.dominance_method = dominance_method
        self.config = config or DominanceConfig()
        self.streaming_mode = streaming_mode
        self.window_duration = window_duration
        self.update_interval = update_interval
        
        # Streaming state
        if streaming_mode:
            self._init_streaming_state()
    
    def _init_streaming_state(self):
        """Initialize streaming state variables"""
        self.streaming_buffer = []
        self.streaming_diarization = []
        self.streaming_embeddings = {}
        self.last_update_time = 0.0
        self.current_dominant = None
    
    def identify_dominant(self,
                         audio: np.ndarray,
                         diarization: List[Dict],
                         embeddings: Dict[str, np.ndarray],
                         analyze_similarity: bool = False) -> SpeakerDominance:
        """Identify the dominant speaker in audio.
        
        Args:
            audio: Audio signal
            diarization: List of diarization segments with speaker labels
            embeddings: Speaker embeddings dictionary
            analyze_similarity: Whether to analyze speaker similarities
            
        Returns:
            SpeakerDominance result
        """
        # Handle empty cases
        if not diarization:
            return SpeakerDominance(
                dominant_speaker=None,
                confidence=0.0,
                total_speakers=0
            )
        
        # Validate embeddings match diarization
        speakers_in_diarization = set(seg["speaker"] for seg in diarization)
        missing_embeddings = speakers_in_diarization - set(embeddings.keys())
        if missing_embeddings:
            raise ValueError(f"Missing embeddings for speakers: {missing_embeddings}")
        
        # Calculate durations and energies
        speaker_stats = self._calculate_speaker_stats(audio, diarization)
        
        # Find overlapping regions
        overlap_regions = self._find_overlap_regions(diarization)
        
        # Calculate speaker similarities if requested
        speaker_similarities = None
        if analyze_similarity and len(embeddings) > 1:
            speaker_similarities = self._calculate_speaker_similarities(embeddings)
        
        # Determine dominant speaker based on method
        if self.dominance_method == DominanceMethod.DURATION:
            dominant, confidence = self._duration_based_dominance(speaker_stats)
        elif self.dominance_method == DominanceMethod.ENERGY:
            dominant, confidence = self._energy_based_dominance(speaker_stats)
        else:  # HYBRID
            dominant, confidence = self._hybrid_dominance(speaker_stats)
        
        # Filter qualified speakers
        total_duration = sum(stats["duration"] for stats in speaker_stats.values())
        qualified_speakers = [
            speaker for speaker, stats in speaker_stats.items()
            if stats["duration"] / total_duration >= self.config.min_duration_ratio
        ]
        
        # Check if distribution is balanced
        is_balanced = self._check_balanced_distribution(speaker_stats)
        
        # Create result
        return SpeakerDominance(
            dominant_speaker=dominant,
            confidence=confidence,
            total_speakers=len(speaker_stats),
            speaker_durations={s: stats["duration"] for s, stats in speaker_stats.items()},
            speaker_energies={s: stats["energy"] for s, stats in speaker_stats.items()},
            speaker_ratios={s: stats["duration"] / total_duration 
                          for s, stats in speaker_stats.items()},
            is_balanced=is_balanced,
            overlap_regions=overlap_regions,
            qualified_speakers=qualified_speakers,
            speaker_similarities=speaker_similarities,
            metadata={
                "method": self.dominance_method.value,
                "total_duration": total_duration,
                "overlap_duration": sum(end - start for start, end in overlap_regions)
            }
        )
    
    def _calculate_speaker_stats(self, 
                                audio: np.ndarray,
                                diarization: List[Dict]) -> Dict[str, Dict]:
        """Calculate duration and energy statistics for each speaker"""
        speaker_stats = defaultdict(lambda: {"duration": 0.0, "energy": 0.0, "segments": []})
        
        audio_duration = len(audio) / self.sample_rate
        
        for segment in diarization:
            speaker = segment["speaker"]
            start = max(0, segment["start"])
            end = min(audio_duration, segment["end"])
            
            if end <= start:
                continue
            
            # Update duration
            duration = end - start
            speaker_stats[speaker]["duration"] += duration
            speaker_stats[speaker]["segments"].append((start, end))
            
            # Calculate energy for this segment
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) > 0:
                # RMS energy
                energy = np.sqrt(np.mean(segment_audio ** 2))
                speaker_stats[speaker]["energy"] += energy * duration
        
        # Normalize energy by duration
        for stats in speaker_stats.values():
            if stats["duration"] > 0:
                stats["energy"] /= stats["duration"]
        
        return dict(speaker_stats)
    
    def _find_overlap_regions(self, diarization: List[Dict]) -> List[Tuple[float, float]]:
        """Find regions where multiple speakers overlap"""
        if len(diarization) < 2:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(diarization, key=lambda x: x["start"])
        
        overlaps = []
        for i in range(len(sorted_segments) - 1):
            for j in range(i + 1, len(sorted_segments)):
                seg1 = sorted_segments[i]
                seg2 = sorted_segments[j]
                
                # Check for overlap
                overlap_start = max(seg1["start"], seg2["start"])
                overlap_end = min(seg1["end"], seg2["end"])
                
                if overlap_start < overlap_end:
                    overlaps.append((overlap_start, overlap_end))
        
        # Merge overlapping regions
        if overlaps:
            overlaps = self._merge_overlaps(overlaps)
        
        return overlaps
    
    def _merge_overlaps(self, overlaps: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping regions"""
        if not overlaps:
            return []
        
        # Sort by start time
        overlaps.sort(key=lambda x: x[0])
        
        merged = [overlaps[0]]
        for start, end in overlaps[1:]:
            if start <= merged[-1][1]:
                # Merge with previous
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
    
    def _calculate_speaker_similarities(self, 
                                      embeddings: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise speaker similarities"""
        similarities = {}
        speakers = list(embeddings.keys())
        
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                speaker1, speaker2 = speakers[i], speakers[j]
                
                # Cosine similarity
                emb1 = embeddings[speaker1]
                emb2 = embeddings[speaker2]
                
                # Normalize embeddings
                emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
                emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
                
                similarity = 1 - cosine(emb1_norm, emb2_norm)
                similarities[(speaker1, speaker2)] = similarity
                similarities[(speaker2, speaker1)] = similarity  # Symmetric
        
        return similarities
    
    def _duration_based_dominance(self, 
                                 speaker_stats: Dict[str, Dict]) -> Tuple[Optional[str], float]:
        """Determine dominance based on speaking duration"""
        if not speaker_stats:
            return None, 0.0
        
        # Sort by duration
        sorted_speakers = sorted(
            speaker_stats.items(),
            key=lambda x: x[1]["duration"],
            reverse=True
        )
        
        dominant_speaker = sorted_speakers[0][0]
        dominant_duration = sorted_speakers[0][1]["duration"]
        
        # Calculate confidence based on duration ratio
        total_duration = sum(stats["duration"] for stats in speaker_stats.values())
        if total_duration > 0:
            confidence = dominant_duration / total_duration
        else:
            confidence = 0.0
        
        return dominant_speaker, confidence
    
    def _energy_based_dominance(self,
                               speaker_stats: Dict[str, Dict]) -> Tuple[Optional[str], float]:
        """Determine dominance based on speech energy"""
        if not speaker_stats:
            return None, 0.0
        
        # Sort by energy
        sorted_speakers = sorted(
            speaker_stats.items(),
            key=lambda x: x[1]["energy"],
            reverse=True
        )
        
        dominant_speaker = sorted_speakers[0][0]
        dominant_energy = sorted_speakers[0][1]["energy"]
        
        # Calculate confidence based on energy ratio
        total_energy = sum(stats["energy"] for stats in speaker_stats.values())
        if total_energy > 0:
            confidence = dominant_energy / total_energy
        else:
            confidence = 0.0
        
        # Adjust confidence based on duration ratio as well
        duration_ratio = speaker_stats[dominant_speaker]["duration"] / sum(
            stats["duration"] for stats in speaker_stats.values()
        )
        confidence = 0.7 * confidence + 0.3 * duration_ratio
        
        return dominant_speaker, confidence
    
    def _hybrid_dominance(self,
                         speaker_stats: Dict[str, Dict]) -> Tuple[Optional[str], float]:
        """Determine dominance using hybrid approach"""
        if not speaker_stats:
            return None, 0.0
        
        # Calculate weighted scores
        total_duration = sum(stats["duration"] for stats in speaker_stats.values())
        total_energy = sum(stats["energy"] for stats in speaker_stats.values())
        
        speaker_scores = {}
        for speaker, stats in speaker_stats.items():
            duration_score = stats["duration"] / total_duration if total_duration > 0 else 0
            energy_score = stats["energy"] / total_energy if total_energy > 0 else 0
            
            # Combined score
            score = (self.config.duration_weight * duration_score + 
                    self.config.energy_weight * energy_score)
            speaker_scores[speaker] = score
        
        # Find dominant speaker
        dominant_speaker = max(speaker_scores, key=speaker_scores.get)
        confidence = speaker_scores[dominant_speaker]
        
        return dominant_speaker, confidence
    
    def _check_balanced_distribution(self, speaker_stats: Dict[str, Dict]) -> bool:
        """Check if speaker distribution is balanced"""
        if len(speaker_stats) < 2:
            return False
        
        durations = [stats["duration"] for stats in speaker_stats.values()]
        total_duration = sum(durations)
        
        if total_duration == 0:
            return True
        
        # Calculate ratios
        ratios = [d / total_duration for d in durations]
        
        # Check if max ratio is less than threshold (e.g., 60%)
        max_ratio = max(ratios)
        return max_ratio < 0.6
    
    def process_batch(self,
                     audio_segments: List[np.ndarray],
                     diarizations: List[List[Dict]],
                     embeddings_list: List[Dict[str, np.ndarray]]) -> List[SpeakerDominance]:
        """Process multiple audio segments in batch.
        
        Args:
            audio_segments: List of audio arrays
            diarizations: List of diarization results
            embeddings_list: List of embedding dictionaries
            
        Returns:
            List of SpeakerDominance results
        """
        results = []
        
        for audio, diarization, embeddings in zip(audio_segments, diarizations, embeddings_list):
            result = self.identify_dominant(audio, diarization, embeddings)
            results.append(result)
        
        return results
    
    def update_streaming(self,
                        audio_chunk: np.ndarray,
                        chunk_diarization: List[Dict],
                        embeddings: Dict[str, np.ndarray],
                        timestamp: float) -> SpeakerDominance:
        """Update streaming dominance calculation.
        
        Args:
            audio_chunk: New audio chunk
            chunk_diarization: Diarization for the chunk
            embeddings: Speaker embeddings
            timestamp: Current timestamp
            
        Returns:
            Updated SpeakerDominance result
        """
        if not self.streaming_mode:
            raise RuntimeError("Streaming mode not enabled")
        
        # Update buffer
        self.streaming_buffer.append({
            "audio": audio_chunk,
            "diarization": chunk_diarization,
            "timestamp": timestamp
        })
        
        # Remove old chunks outside window
        window_start = timestamp - self.window_duration
        self.streaming_buffer = [
            chunk for chunk in self.streaming_buffer
            if chunk["timestamp"] >= window_start
        ]
        
        # Check if update is needed
        if timestamp - self.last_update_time < self.update_interval:
            # Return previous result
            if self.current_dominant:
                return self.current_dominant
            else:
                return SpeakerDominance(
                    dominant_speaker=None,
                    confidence=0.0,
                    total_speakers=0
                )
        
        # Concatenate audio in window
        if self.streaming_buffer:
            window_audio = np.concatenate([chunk["audio"] for chunk in self.streaming_buffer])
            
            # Merge diarization
            window_diarization = []
            for chunk in self.streaming_buffer:
                # Adjust timestamps relative to window start
                chunk_offset = chunk["timestamp"] - window_start
                for seg in chunk["diarization"]:
                    adjusted_seg = {
                        "speaker": seg["speaker"],
                        "start": seg["start"] + chunk_offset,
                        "end": seg["end"] + chunk_offset
                    }
                    window_diarization.append(adjusted_seg)
            
            # Update embeddings
            self.streaming_embeddings.update(embeddings)
            
            # Calculate dominance
            result = self.identify_dominant(
                window_audio,
                window_diarization,
                self.streaming_embeddings
            )
            
            self.current_dominant = result
            self.last_update_time = timestamp
            
            return result
        else:
            return SpeakerDominance(
                dominant_speaker=None,
                confidence=0.0,
                total_speakers=0
            )