#!/usr/bin/env python3
"""
Complete Speaker Separation Module
=================================

Professional implementation for separating overlapping speakers throughout
entire audio files, not just at the end. Uses state-of-the-art source
separation models and speaker identification.

Key Features:
1. Full audio source separation using SpeechBrain SepFormer
2. Speaker identification using embeddings
3. Overlapping speech handling throughout the file
4. Batch processing capability
"""

import numpy as np
import torch
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class OverlapRegion:
    """Represents a region with overlapping speech"""
    start_time: float
    end_time: float
    duration: float
    num_speakers: int


@dataclass 
class SpeakerAnalysis:
    """Analysis of speakers in audio"""
    num_speakers: int
    has_overlapping_speech: bool
    overlap_regions: List[OverlapRegion]
    primary_speaker_ratio: float


class CompleteSeparator:
    """Complete speaker separation using source separation models"""
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 energy_threshold: float = 0.01,
                 overlap_threshold: float = 0.8):
        """
        Initialize complete separator.
        
        Args:
            device: Device to run models on
            energy_threshold: Minimum energy to consider as speech
            overlap_threshold: Threshold for detecting overlap
        """
        self.device = device
        self.energy_threshold = energy_threshold
        self.overlap_threshold = overlap_threshold
        
        # Load separation model
        self._load_separation_model()
        
        # Load speaker embedding model  
        self._load_embedding_model()
    
    def _load_separation_model(self):
        """Load the SpeechBrain SepFormer model"""
        try:
            from processors.audio_enhancement.speechbrain_separator import SpeechBrainSeparator
            self.separator = SpeechBrainSeparator()
            logger.info("Loaded SpeechBrain separator for complete separation")
        except Exception as e:
            logger.error(f"Failed to load separator: {e}")
            self.separator = None
    
    def _load_embedding_model(self):
        """Load speaker embedding model for identification"""
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
    
    def analyze_overlapping_speakers(self, audio: np.ndarray, sample_rate: int) -> SpeakerAnalysis:
        """
        Analyze audio for overlapping speakers throughout the file.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            SpeakerAnalysis with overlap information
        """
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Detect speech activity in windows
        window_size = int(0.5 * sample_rate)  # 500ms windows
        hop_size = int(0.1 * sample_rate)     # 100ms hop
        
        overlap_regions = []
        
        # Analyze each window
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            
            # Check if this window likely has multiple speakers
            if self._has_multiple_speakers(window, sample_rate):
                start_time = i / sample_rate
                end_time = (i + window_size) / sample_rate
                
                # Merge with previous region if overlapping
                if overlap_regions and overlap_regions[-1].end_time >= start_time:
                    overlap_regions[-1].end_time = end_time
                    overlap_regions[-1].duration = end_time - overlap_regions[-1].start_time
                else:
                    overlap_regions.append(OverlapRegion(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        num_speakers=2  # Conservative estimate
                    ))
        
        # Calculate statistics
        total_duration = len(audio) / sample_rate
        overlap_duration = sum(r.duration for r in overlap_regions)
        
        return SpeakerAnalysis(
            num_speakers=2 if overlap_regions else 1,
            has_overlapping_speech=len(overlap_regions) > 0,
            overlap_regions=overlap_regions,
            primary_speaker_ratio=1 - (overlap_duration / total_duration)
        )
    
    def _has_multiple_speakers(self, window: np.ndarray, sample_rate: int) -> bool:
        """Check if a window likely contains multiple speakers"""
        # Method 1: Spectral complexity
        freqs, psd = signal.periodogram(window, sample_rate)
        
        # Find peaks in spectrum
        peaks, properties = signal.find_peaks(psd, height=np.max(psd) * 0.1, distance=20)
        
        # Method 2: Harmonic analysis
        # Multiple speakers often have different fundamental frequencies
        fund_freqs = []
        if len(peaks) > 0:
            # Look for harmonic patterns
            for peak_idx in peaks[:5]:  # Check first 5 peaks
                freq = freqs[peak_idx]
                if 80 < freq < 800:  # Human speech range
                    fund_freqs.append(freq)
        
        # Method 3: Energy distribution
        # Calculate spectral entropy - higher entropy suggests multiple sources
        psd_norm = psd / (np.sum(psd) + 1e-10)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
        
        # Method 4: Time-domain analysis
        # Check for amplitude modulation patterns
        envelope = np.abs(signal.hilbert(window))
        envelope_std = np.std(envelope)
        envelope_mean = np.mean(envelope)
        modulation_index = envelope_std / (envelope_mean + 1e-10)
        
        # Combine indicators (ULTRA sensitive for aggressive removal)
        has_multiple = (
            len(peaks) > 3 or  # Lowered: Many spectral peaks
            len(fund_freqs) >= 1 or  # Lowered: Any multiple fundamentals
            spectral_entropy > 3.0 or  # Lowered: High spectral complexity
            modulation_index > 0.3  # Lowered: High amplitude variation
        )
        
        return has_multiple
    
    def separate_speakers(self, audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        Separate all speakers in the audio using source separation.
        
        Args:
            audio: Mixed audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            List of separated audio sources
        """
        if self.separator is None:
            logger.warning("No separator available, returning original audio")
            return [audio]
        
        # Convert to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure audio is properly normalized
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        try:
            # Use SpeechBrain separator
            separated = self.separator.separate_all_speakers(audio, sample_rate)
            
            if separated is not None and len(separated) > 1:
                logger.info(f"Successfully separated {len(separated)} sources")
                return separated
            else:
                logger.warning("Separation failed or returned single source")
                return [audio]
                
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            return [audio]
    
    def identify_primary_speaker(self, separated_sources: List[np.ndarray], 
                               sample_rate: int) -> Optional[int]:
        """
        Identify which separated source is the primary/dominant speaker.
        
        Enhanced strategy:
        1. Analyze speaking time throughout the audio (not just energy)
        2. Identify the speaker who speaks the most overall
        3. Exclude brief initial speakers
        
        Args:
            separated_sources: List of separated audio sources
            sample_rate: Sample rate in Hz
            
        Returns:
            Index of primary speaker source
        """
        if len(separated_sources) == 1:
            return 0
        
        # Use the enhanced dominant speaker separator for better identification
        try:
            from processors.audio_enhancement.dominant_speaker_separation import DominantSpeakerSeparator
            dominant_separator = DominantSpeakerSeparator(device=self.device)
            dominant_idx = dominant_separator.identify_dominant_speaker(separated_sources, sample_rate)
            return dominant_idx
        except Exception as e:
            logger.warning(f"Failed to use dominant speaker separator: {e}")
            # Fallback to original method
            return self._identify_primary_speaker_fallback(separated_sources, sample_rate)
    
    def _identify_primary_speaker_fallback(self, separated_sources: List[np.ndarray], 
                                          sample_rate: int) -> int:
        """Fallback method for identifying primary speaker"""
        scores = []
        
        for i, source in enumerate(separated_sources):
            # Calculate energy
            energy = np.sqrt(np.mean(source**2))
            
            # Calculate active duration (non-silence)
            active_samples = np.sum(np.abs(source) > self.energy_threshold)
            active_duration = active_samples / sample_rate
            
            # Calculate consistency (less variance = more consistent)
            window_energies = []
            window_size = int(0.5 * sample_rate)
            for j in range(0, len(source) - window_size, window_size):
                window_energy = np.sqrt(np.mean(source[j:j+window_size]**2))
                window_energies.append(window_energy)
            
            consistency = 1.0 / (np.std(window_energies) + 0.01) if window_energies else 1.0
            
            # Combined score
            score = energy * active_duration * consistency
            scores.append(score)
            
            logger.debug(f"Source {i}: energy={energy:.3f}, duration={active_duration:.1f}s, "
                        f"consistency={consistency:.3f}, score={score:.3f}")
        
        # Return source with highest score
        primary_idx = np.argmax(scores)
        logger.info(f"Identified source {primary_idx} as primary speaker (fallback)")
        
        return primary_idx
    
    def extract_primary_speaker(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract only the primary speaker from mixed audio.
        
        This is the main entry point that combines all steps:
        1. Separate all speakers
        2. Identify primary speaker
        3. Return only primary speaker audio
        
        Args:
            audio: Mixed audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio containing only the primary speaker
        """
        # First check if separation is needed
        analysis = self.analyze_overlapping_speakers(audio, sample_rate)
        
        if not analysis.has_overlapping_speech:
            logger.info("No overlapping speech detected, returning original audio")
            return audio
        
        logger.info(f"Detected overlapping speech in {len(analysis.overlap_regions)} regions")
        
        # Separate speakers
        separated_sources = self.separate_speakers(audio, sample_rate)
        
        if len(separated_sources) == 1:
            logger.warning("Separation returned only one source")
            return separated_sources[0]
        
        # Identify primary speaker
        primary_idx = self.identify_primary_speaker(separated_sources, sample_rate)
        
        if primary_idx is None:
            logger.warning("Could not identify primary speaker, using first source")
            primary_idx = 0
        
        # Return primary speaker audio
        primary_audio = separated_sources[primary_idx]
        
        # Post-process to ensure quality
        primary_audio = self._post_process(primary_audio, audio)
        
        return primary_audio
    
    def _post_process(self, separated: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Post-process separated audio to ensure quality"""
        # Ensure same length
        if len(separated) != len(original):
            if len(separated) > len(original):
                separated = separated[:len(original)]
            else:
                separated = np.pad(separated, (0, len(original) - len(separated)))
        
        # Normalize to similar level as original
        original_rms = np.sqrt(np.mean(original**2))
        separated_rms = np.sqrt(np.mean(separated**2))
        
        if separated_rms > 0:
            separated = separated * (original_rms / separated_rms)
        
        # Ensure no clipping
        max_val = np.max(np.abs(separated))
        if max_val > 0.95:
            separated = separated * (0.95 / max_val)
        
        return separated
    
    def process_batch(self, audio_list: List[np.ndarray], sample_rate: int) -> List[np.ndarray]:
        """
        Process multiple audio files in batch.
        
        Args:
            audio_list: List of audio signals
            sample_rate: Sample rate in Hz
            
        Returns:
            List of processed audio with only primary speakers
        """
        results = []
        
        for i, audio in enumerate(audio_list):
            logger.info(f"Processing audio {i+1}/{len(audio_list)}")
            try:
                cleaned = self.extract_primary_speaker(audio, sample_rate)
                results.append(cleaned)
            except Exception as e:
                logger.error(f"Failed to process audio {i}: {e}")
                results.append(audio)  # Return original on failure
        
        return results