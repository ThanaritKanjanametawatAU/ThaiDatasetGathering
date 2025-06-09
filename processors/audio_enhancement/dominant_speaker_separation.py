#!/usr/bin/env python3
"""
Dominant Speaker Separation Module
==================================

Enhanced implementation that identifies the dominant speaker throughout the audio,
not just the first or loudest speaker. Uses advanced speaker diarization and
activity analysis to determine which speaker is the main/dominant one.

Key Improvements:
1. Identifies dominant speaker based on total speaking time
2. Excludes initial speakers if they only appear briefly
3. Uses speaker embeddings to track consistency
4. Handles overlapping speech regions correctly
"""

import numpy as np
import torch
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import signal
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SpeakerActivity:
    """Track activity for each speaker"""
    speaker_id: int
    total_duration: float
    active_segments: List[Tuple[float, float]]
    average_energy: float
    embedding: Optional[np.ndarray] = None
    
    @property
    def speaking_ratio(self) -> float:
        """Ratio of time this speaker is active"""
        if not self.active_segments:
            return 0.0
        total_active = sum(end - start for start, end in self.active_segments)
        total_duration = max(end for _, end in self.active_segments)
        return total_active / total_duration if total_duration > 0 else 0.0


class DominantSpeakerSeparator:
    """Enhanced separator that correctly identifies the dominant speaker"""
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 min_speaker_duration: float = 2.0,  # Minimum duration to be considered dominant
                 activity_threshold: float = 0.01,
                 embedding_similarity_threshold: float = 0.85):
        """
        Initialize dominant speaker separator.
        
        Args:
            device: Device to run models on
            min_speaker_duration: Minimum speaking duration to be considered dominant
            activity_threshold: Energy threshold for voice activity
            embedding_similarity_threshold: Threshold for speaker similarity
        """
        self.device = device
        self.min_speaker_duration = min_speaker_duration
        self.activity_threshold = activity_threshold
        self.embedding_similarity_threshold = embedding_similarity_threshold
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load separation and embedding models"""
        try:
            from processors.audio_enhancement.speechbrain_separator import SpeechBrainSeparator
            self.separator = SpeechBrainSeparator()
            logger.info("Loaded SpeechBrain separator")
        except Exception as e:
            logger.error(f"Failed to load separator: {e}")
            self.separator = None
        
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            self.embedder = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/media/ssd1/SparkVoiceProject/models/speechbrain/ecapa",
                run_opts={"device": self.device}
            )
            logger.info("Loaded speaker embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            self.embedder = None
    
    def analyze_speaker_activity(self, audio: np.ndarray, sample_rate: int, 
                                window_size: float = 0.5) -> List[SpeakerActivity]:
        """
        Analyze speaker activity throughout the audio.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate in Hz
            window_size: Window size in seconds for analysis
            
        Returns:
            List of SpeakerActivity objects
        """
        # Convert to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Calculate voice activity in windows
        window_samples = int(window_size * sample_rate)
        hop_samples = int(0.1 * sample_rate)  # 100ms hop
        
        activities = []
        current_speaker = None
        speaker_segments = {}
        
        for i in range(0, len(audio) - window_samples, hop_samples):
            window = audio[i:i+window_samples]
            start_time = i / sample_rate
            end_time = (i + window_samples) / sample_rate
            
            # Check if window has speech
            energy = np.sqrt(np.mean(window**2))
            if energy > self.activity_threshold:
                # Get speaker embedding for this window if embedder available
                if self.embedder is not None:
                    try:
                        # Ensure proper shape for embedder
                        window_tensor = torch.from_numpy(window).unsqueeze(0)
                        embedding = self.embedder.encode_batch(window_tensor)
                        embedding = embedding.squeeze().cpu().numpy()
                        
                        # Find matching speaker or create new one
                        speaker_id = self._find_matching_speaker(embedding, speaker_segments)
                        
                        if speaker_id not in speaker_segments:
                            speaker_segments[speaker_id] = {
                                'segments': [],
                                'energies': [],
                                'embedding': embedding
                            }
                        
                        speaker_segments[speaker_id]['segments'].append((start_time, end_time))
                        speaker_segments[speaker_id]['energies'].append(energy)
                        
                    except Exception as e:
                        logger.debug(f"Embedding extraction failed: {e}")
                        # Fallback to energy-based detection
                        if 0 not in speaker_segments:
                            speaker_segments[0] = {'segments': [], 'energies': [], 'embedding': None}
                        speaker_segments[0]['segments'].append((start_time, end_time))
                        speaker_segments[0]['energies'].append(energy)
                else:
                    # No embedder, use single speaker
                    if 0 not in speaker_segments:
                        speaker_segments[0] = {'segments': [], 'energies': [], 'embedding': None}
                    speaker_segments[0]['segments'].append((start_time, end_time))
                    speaker_segments[0]['energies'].append(energy)
        
        # Convert to SpeakerActivity objects
        for speaker_id, data in speaker_segments.items():
            # Merge adjacent segments
            merged_segments = self._merge_segments(data['segments'])
            total_duration = sum(end - start for start, end in merged_segments)
            
            activity = SpeakerActivity(
                speaker_id=speaker_id,
                total_duration=total_duration,
                active_segments=merged_segments,
                average_energy=np.mean(data['energies']) if data['energies'] else 0,
                embedding=data.get('embedding')
            )
            activities.append(activity)
        
        # Sort by total duration (dominant speaker first)
        activities.sort(key=lambda x: x.total_duration, reverse=True)
        
        return activities
    
    def _find_matching_speaker(self, embedding: np.ndarray, speaker_segments: Dict) -> int:
        """Find matching speaker based on embedding similarity"""
        if not speaker_segments:
            return 0
        
        best_match = None
        best_similarity = -1
        
        for speaker_id, data in speaker_segments.items():
            if data.get('embedding') is not None:
                # Compute cosine similarity
                similarity = np.dot(embedding, data['embedding']) / (
                    np.linalg.norm(embedding) * np.linalg.norm(data['embedding']) + 1e-10
                )
                
                if similarity > best_similarity and similarity > self.embedding_similarity_threshold:
                    best_similarity = similarity
                    best_match = speaker_id
        
        # If no good match, create new speaker
        if best_match is None:
            best_match = len(speaker_segments)
        
        return best_match
    
    def _merge_segments(self, segments: List[Tuple[float, float]], 
                       gap_threshold: float = 0.5) -> List[Tuple[float, float]]:
        """Merge adjacent segments with small gaps"""
        if not segments:
            return []
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x[0])
        
        merged = [segments[0]]
        for start, end in segments[1:]:
            last_start, last_end = merged[-1]
            
            # If gap is small enough, merge
            if start - last_end <= gap_threshold:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged
    
    def identify_dominant_speaker(self, separated_sources: List[np.ndarray], 
                                sample_rate: int) -> int:
        """
        Identify the dominant speaker from separated sources.
        
        The dominant speaker is the one who speaks the most throughout the audio,
        not necessarily the first or loudest speaker.
        
        Args:
            separated_sources: List of separated audio sources
            sample_rate: Sample rate in Hz
            
        Returns:
            Index of dominant speaker
        """
        if len(separated_sources) == 1:
            return 0
        
        speaker_activities = []
        
        for i, source in enumerate(separated_sources):
            # Analyze activity for this source
            activities = self.analyze_speaker_activity(source, sample_rate)
            
            # Get primary activity (should be mostly one speaker per source)
            if activities:
                primary_activity = activities[0]
                primary_activity.speaker_id = i  # Override with source index
                speaker_activities.append(primary_activity)
            else:
                # Create empty activity
                speaker_activities.append(SpeakerActivity(
                    speaker_id=i,
                    total_duration=0.0,
                    active_segments=[],
                    average_energy=0.0
                ))
        
        # Log activity analysis
        for activity in speaker_activities:
            logger.info(f"Source {activity.speaker_id}: duration={activity.total_duration:.1f}s, "
                       f"ratio={activity.speaking_ratio:.2f}, energy={activity.average_energy:.3f}")
        
        # Find dominant speaker (most total speaking time)
        # Exclude speakers with very short duration
        valid_speakers = [
            activity for activity in speaker_activities 
            if activity.total_duration >= self.min_speaker_duration
        ]
        
        if not valid_speakers:
            # If no speaker meets minimum duration, use the one with most time
            dominant = max(speaker_activities, key=lambda x: x.total_duration)
        else:
            dominant = max(valid_speakers, key=lambda x: x.total_duration)
        
        logger.info(f"Identified source {dominant.speaker_id} as dominant speaker "
                   f"(duration: {dominant.total_duration:.1f}s)")
        
        return dominant.speaker_id
    
    def extract_dominant_speaker(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract the dominant speaker from mixed audio.
        
        Args:
            audio: Mixed audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio containing only the dominant speaker
        """
        if self.separator is None:
            logger.warning("No separator available, returning original audio")
            return audio
        
        # First, check if separation is needed by analyzing the original audio
        activities = self.analyze_speaker_activity(audio, sample_rate)
        
        if len(activities) <= 1:
            logger.info("Single speaker detected, no separation needed")
            return audio
        
        # Always perform separation when multiple speakers detected
        # Remove the check that was preventing separation
        logger.info(f"Multiple speakers detected: {len(activities)} speakers")
        logger.info(f"Dominant speaker duration: {activities[0].total_duration:.1f}s, ratio: {activities[0].speaking_ratio:.2f}")
        
        logger.info(f"Multiple speakers detected, performing separation")
        
        # Separate speakers
        try:
            separated_sources = self.separator.separate_all_speakers(audio, sample_rate)
            
            if not separated_sources or len(separated_sources) == 1:
                logger.warning("Separation failed or returned single source")
                return audio
            
            # Identify dominant speaker
            dominant_idx = self.identify_dominant_speaker(separated_sources, sample_rate)
            
            # Get dominant speaker audio
            dominant_audio = separated_sources[dominant_idx]
            
            # Post-process
            dominant_audio = self._post_process(dominant_audio, audio)
            
            return dominant_audio
            
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            return audio
    
    def _post_process(self, separated: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Post-process separated audio"""
        # Ensure same length
        if len(separated) != len(original):
            if len(separated) > len(original):
                separated = separated[:len(original)]
            else:
                separated = np.pad(separated, (0, len(original) - len(separated)))
        
        # Apply gentle noise gate to remove artifacts
        gate_threshold = 0.005
        envelope = np.abs(signal.hilbert(separated))
        smooth_envelope = signal.savgol_filter(envelope, 1001, 3)
        gate = (smooth_envelope > gate_threshold).astype(float)
        
        # Smooth gate transitions
        gate = signal.savgol_filter(gate, 101, 3)
        separated = separated * gate
        
        # Normalize to similar level as original
        original_rms = np.sqrt(np.mean(original**2))
        separated_rms = np.sqrt(np.mean(separated**2) + 1e-10)
        
        if separated_rms > 0:
            separated = separated * (original_rms / separated_rms) * 0.8  # Slightly quieter
        
        # Ensure no clipping
        max_val = np.max(np.abs(separated))
        if max_val > 0.95:
            separated = separated * (0.95 / max_val)
        
        return separated