"""
Secondary Speaker Removal Module
================================

Implements advanced techniques for detecting and removing secondary speakers
from audio, especially at the end of utterances.

Based on best practices:
- Voice Activity Detection (VAD) for speech/silence segmentation
- Energy-based speaker change detection
- Spectral analysis for speaker differentiation
- Smart end-of-utterance detection
"""

import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of secondary speaker detection"""
    has_secondary_at_end: bool
    secondary_start_time: float
    num_speakers: int
    confidence: float = 0.0


@dataclass
class SpeechSegment:
    """Represents a speech segment"""
    start_time: float
    end_time: float
    energy: float
    speaker_id: Optional[int] = None


@dataclass
class EnergyChange:
    """Represents an energy change point"""
    time: float
    energy_ratio: float
    before_energy: float
    after_energy: float


@dataclass
class EndAnalysis:
    """Analysis of audio end characteristics"""
    has_secondary_speaker: bool
    primary_end_time: float
    secondary_start_time: float
    confidence: float


@dataclass
class SpeakerInfo:
    """Information about a speaker"""
    dominant_frequency: float
    energy_level: float
    spectral_centroid: float


@dataclass
class SpeakerAnalysis:
    """Result of speaker analysis"""
    speakers: List[SpeakerInfo]
    num_speakers: int


@dataclass
class DetectionResult:
    """Result of secondary speaker detection"""
    has_secondary_at_end: bool
    secondary_start_time: float
    num_speakers: int
    confidence: float = 0.0


class VoiceActivityDetector:
    """Voice Activity Detection for speech/silence segmentation"""
    
    def __init__(self, 
                 frame_duration: float = 0.025,
                 energy_threshold_percentile: int = 30,
                 min_speech_duration: float = 0.1):
        self.frame_duration = frame_duration
        self.energy_threshold_percentile = energy_threshold_percentile
        self.min_speech_duration = min_speech_duration
    
    def detect_speech_segments(self, audio: np.ndarray, sample_rate: int) -> List[SpeechSegment]:
        """Detect speech segments in audio"""
        frame_size = int(self.frame_duration * sample_rate)
        hop_size = frame_size // 2
        
        # Calculate frame energies
        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energy = np.sqrt(np.mean(frame**2))
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Dynamic threshold based on percentile
        threshold = np.percentile(energies, self.energy_threshold_percentile)
        
        # Find speech regions
        speech_mask = energies > threshold
        
        # Find segment boundaries
        segments = []
        in_speech = False
        start_idx = 0
        
        for i, is_speech in enumerate(speech_mask):
            if is_speech and not in_speech:
                # Speech starts
                start_idx = i
                in_speech = True
            elif not is_speech and in_speech:
                # Speech ends
                start_time = start_idx * hop_size / sample_rate
                end_time = i * hop_size / sample_rate
                
                # Check minimum duration
                if end_time - start_time >= self.min_speech_duration:
                    segment_energy = np.mean(energies[start_idx:i])
                    segments.append(SpeechSegment(start_time, end_time, segment_energy))
                
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            start_time = start_idx * hop_size / sample_rate
            end_time = len(audio) / sample_rate
            if end_time - start_time >= self.min_speech_duration:
                segment_energy = np.mean(energies[start_idx:])
                segments.append(SpeechSegment(start_time, end_time, segment_energy))
        
        return segments


class EnergyAnalyzer:
    """Analyze energy changes to detect speaker transitions"""
    
    def __init__(self, 
                 window_duration: float = 0.1,
                 change_threshold: float = 1.3):
        self.window_duration = window_duration
        self.change_threshold = change_threshold
    
    def detect_energy_changes(self, audio: np.ndarray, sample_rate: int) -> List[EnergyChange]:
        """Detect significant energy changes"""
        window_size = int(self.window_duration * sample_rate)
        hop_size = window_size // 2
        
        # Calculate windowed energies
        energies = []
        times = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            energy = np.sqrt(np.mean(window**2))
            energies.append(energy)
            times.append((i + window_size // 2) / sample_rate)
        
        energies = np.array(energies)
        times = np.array(times)
        
        # Smooth energies (less aggressive)
        smoothed = gaussian_filter1d(energies, sigma=1)
        
        # Find significant changes
        changes = []
        for i in range(1, len(smoothed) - 1):
            # Compare before and after
            before_avg = np.mean(smoothed[max(0, i-3):i])
            after_avg = np.mean(smoothed[i:min(len(smoothed), i+3)])
            
            if before_avg > 0:
                ratio = after_avg / before_avg
                
                # Check for significant change
                if ratio > self.change_threshold or ratio < 1/self.change_threshold:
                    changes.append(EnergyChange(
                        time=times[i],
                        energy_ratio=ratio,
                        before_energy=before_avg,
                        after_energy=after_avg
                    ))
        
        return changes


class SmartEndDetector:
    """Smart detection of speech end vs secondary speaker"""
    
    def __init__(self,
                 analysis_window: float = 1.0,
                 silence_threshold: float = 0.02,
                 fade_threshold: float = 0.5):
        self.analysis_window = analysis_window
        self.silence_threshold = silence_threshold
        self.fade_threshold = fade_threshold
    
    def analyze_end(self, audio: np.ndarray, sample_rate: int) -> EndAnalysis:
        """Analyze the end of audio for speaker changes"""
        analysis_samples = int(self.analysis_window * sample_rate)
        
        if len(audio) < analysis_samples:
            analysis_samples = len(audio)
        
        end_audio = audio[-analysis_samples:]
        
        # Detect energy profile
        window_size = int(0.05 * sample_rate)  # 50ms windows
        hop_size = window_size // 2
        
        energies = []
        for i in range(0, len(end_audio) - window_size, hop_size):
            window = end_audio[i:i+window_size]
            energy = np.sqrt(np.mean(window**2))
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Find silence gaps
        silence_mask = energies < self.silence_threshold
        
        # Look for pattern: speech -> silence -> speech
        has_secondary = False
        primary_end = 0
        secondary_start = 0
        
        # Find last significant silence gap
        silence_regions = []
        in_silence = False
        start_idx = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                start_idx = i
                in_silence = True
            elif not is_silent and in_silence:
                if i - start_idx > 2:  # At least 3 frames of silence
                    silence_regions.append((start_idx, i))
                in_silence = False
        
        if silence_regions and len(energies) > silence_regions[-1][1] + 2:
            # Check if there's speech after the last silence
            last_silence_end = silence_regions[-1][1]
            after_silence_energy = np.mean(energies[last_silence_end:])
            
            if after_silence_energy > self.silence_threshold * 2:
                has_secondary = True
                primary_end = (len(audio) - analysis_samples + silence_regions[-1][0] * hop_size) / sample_rate
                secondary_start = (len(audio) - analysis_samples + last_silence_end * hop_size) / sample_rate
        
        # Calculate confidence based on energy difference
        confidence = 0.8 if has_secondary else 0.2
        
        return EndAnalysis(
            has_secondary_speaker=has_secondary,
            primary_end_time=primary_end,
            secondary_start_time=secondary_start,
            confidence=confidence
        )


class SpectralAnalyzer:
    """Analyze spectral characteristics to identify speakers"""
    
    def analyze_speakers(self, audio: np.ndarray, sample_rate: int) -> SpeakerAnalysis:
        """Analyze spectral characteristics of speakers"""
        speakers = []
        
        # Analyze beginning, middle, and end segments
        segment_duration = 0.5  # 500ms segments
        segment_samples = int(segment_duration * sample_rate)
        
        # Beginning segment
        if len(audio) > segment_samples:
            begin_segment = audio[:segment_samples]
        else:
            begin_segment = audio
            
        # End segment
        if len(audio) > segment_samples:
            end_segment = audio[-segment_samples:]
        else:
            end_segment = audio
            
        # Analyze both segments
        for i, segment in enumerate([begin_segment, end_segment]):
            if len(segment) > 1024:
                # Compute spectrum
                freqs, power = signal.periodogram(segment, sample_rate)
                
                # Find dominant frequency
                dominant_idx = np.argmax(power[1:1000]) + 1  # Skip DC, limit to 1kHz
                dominant_freq = freqs[dominant_idx]
                
                # Calculate spectral centroid
                centroid = np.sum(freqs[:1000] * power[:1000]) / np.sum(power[:1000])
                
                # Energy level
                energy = np.sqrt(np.mean(segment**2))
                
                speakers.append(SpeakerInfo(
                    dominant_frequency=dominant_freq,
                    energy_level=energy,
                    spectral_centroid=centroid
                ))
        
        # Determine number of speakers based on spectral differences
        num_speakers = 1
        if len(speakers) > 1:
            freq_diff = abs(speakers[0].dominant_frequency - speakers[1].dominant_frequency)
            centroid_diff = abs(speakers[0].spectral_centroid - speakers[1].spectral_centroid)
            
            # More sophisticated detection: frequency OR centroid difference
            if freq_diff > 50 or centroid_diff > 100:
                num_speakers = 2
        
        return SpeakerAnalysis(
            speakers=speakers,
            num_speakers=num_speakers
        )


class SecondaryRemover:
    """Main class for secondary speaker detection and removal"""
    
    def __init__(self,
                 vad_threshold: float = 0.02,
                 energy_change_threshold: float = 1.5,
                 min_secondary_duration: float = 0.05,
                 max_secondary_duration: float = 1.0,
                 fade_duration: float = 0.05):
        self.vad = VoiceActivityDetector()
        self.energy_analyzer = EnergyAnalyzer(change_threshold=energy_change_threshold)
        self.end_detector = SmartEndDetector()
        self.spectral_analyzer = SpectralAnalyzer()
        self.min_secondary_duration = min_secondary_duration
        self.max_secondary_duration = max_secondary_duration
        self.fade_duration = fade_duration
    
    def detect_secondary_speakers(self, audio: np.ndarray, sample_rate: int) -> DetectionResult:
        """Detect secondary speakers in audio"""
        # Analyze end of audio
        end_analysis = self.end_detector.analyze_end(audio, sample_rate)
        
        # Get speaker count from spectral analysis
        speaker_analysis = self.spectral_analyzer.analyze_speakers(audio, sample_rate)
        
        return DetectionResult(
            has_secondary_at_end=end_analysis.has_secondary_speaker,
            secondary_start_time=end_analysis.secondary_start_time,
            num_speakers=speaker_analysis.num_speakers,
            confidence=end_analysis.confidence
        )
    
    def remove_secondary_speakers(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Remove secondary speakers from audio"""
        # Detect speech segments
        segments = self.vad.detect_speech_segments(audio, sample_rate)
        
        if not segments:
            return audio
        
        # Detect energy changes
        energy_changes = self.energy_analyzer.detect_energy_changes(audio, sample_rate)
        
        # Analyze end
        end_analysis = self.end_detector.analyze_end(audio, sample_rate)
        
        # Process audio
        result = audio.copy()
        
        # Handle secondary speaker at end
        detection_result = self.detect_secondary_speakers(audio, sample_rate)
        
        if detection_result.has_secondary_at_end:
            fade_samples = int(self.fade_duration * sample_rate)
            secondary_start_sample = int(detection_result.secondary_start_time * sample_rate)
            
            # Apply fade before secondary speaker
            if secondary_start_sample > fade_samples:
                fade = np.linspace(1.0, 0.0, fade_samples)
                result[secondary_start_sample-fade_samples:secondary_start_sample] *= fade
            
            # Silence secondary speaker
            result[secondary_start_sample:] = 0.0
        
        # Handle other secondary speakers based on energy changes
        # Group energy changes to find speaker transitions
        if energy_changes and len(segments) > 0:
            for i, change in enumerate(energy_changes):
                change_sample = int(change.time * sample_rate)
                
                # Check if this is likely a secondary speaker
                if change.energy_ratio > 1.3:  # Energy increases
                    # Look for next change back to primary
                    next_change_time = None
                    for j in range(i + 1, len(energy_changes)):
                        if energy_changes[j].energy_ratio < 0.8:  # Energy decreases
                            next_change_time = energy_changes[j].time
                            break
                    
                    if next_change_time:
                        # This is a secondary speaker segment
                        duration = next_change_time - change.time
                        if self.min_secondary_duration <= duration <= self.max_secondary_duration:
                            end_sample = int(next_change_time * sample_rate)
                            
                            # Apply fade and silence
                            fade_samples = int(self.fade_duration * sample_rate)
                            if change_sample > fade_samples:
                                fade = np.linspace(1.0, 0.0, fade_samples)
                                result[change_sample-fade_samples:change_sample] *= fade
                            
                            result[change_sample:end_sample] = 0.0
                            
                            # Apply fade in after secondary speaker
                            if end_sample + fade_samples < len(result):
                                fade_in = np.linspace(0.0, 1.0, fade_samples)
                                result[end_sample:end_sample+fade_samples] *= fade_in
        
        return result
    
    def detect_secondary_speakers(self, audio: np.ndarray, sample_rate: int) -> DetectionResult:
        """Detect secondary speakers in audio"""
        # Analyze energy changes
        energy_changes = self.energy_analyzer.detect_energy_changes(audio, sample_rate)
        
        # Analyze spectral characteristics
        spectral_analysis = self.spectral_analyzer.analyze_speakers(audio, sample_rate)
        
        # Analyze end for secondary speaker
        end_analysis = self.end_detector.analyze_end(audio, sample_rate)
        
        # Check for secondary speaker at end based on multiple criteria
        has_secondary_at_end = False
        secondary_start_time = 0.0
        confidence = 0.0
        
        # Method 1: End analysis detection
        if end_analysis.has_secondary_speaker:
            has_secondary_at_end = True
            secondary_start_time = end_analysis.secondary_start_time
            confidence = end_analysis.confidence
        
        # Method 2: Energy change detection at end
        if not has_secondary_at_end and energy_changes:
            # Check if there's a significant energy change near the end
            for change in reversed(energy_changes):
                duration_to_end = (len(audio) / sample_rate) - change.time
                
                if (self.min_secondary_duration <= duration_to_end <= self.max_secondary_duration
                    and change.energy_ratio > 1.5):
                    has_secondary_at_end = True
                    secondary_start_time = change.time
                    confidence = min(1.0, change.energy_ratio / 2.0)
                    break
        
        # Method 3: Spectral analysis for different speakers
        if not has_secondary_at_end and spectral_analysis.num_speakers > 1:
            # Look for spectral changes in the last portion
            last_window = int(0.5 * sample_rate)
            if len(audio) > last_window:
                last_segment = audio[-last_window:]
                main_segment = audio[:-last_window]
                
                # Compare spectral characteristics
                last_fft = np.fft.rfft(last_segment * np.hanning(len(last_segment)))
                main_fft = np.fft.rfft(main_segment[:last_window] * np.hanning(last_window))
                
                last_spectrum = np.abs(last_fft)
                main_spectrum = np.abs(main_fft)
                
                # Calculate spectral distance
                spectral_distance = np.sqrt(np.mean((last_spectrum - main_spectrum)**2))
                
                if spectral_distance > 0.3:  # Threshold for different speakers
                    has_secondary_at_end = True
                    secondary_start_time = (len(audio) - last_window) / sample_rate
                    confidence = min(1.0, spectral_distance)
        
        return DetectionResult(
            has_secondary_at_end=has_secondary_at_end,
            secondary_start_time=secondary_start_time,
            num_speakers=spectral_analysis.num_speakers,
            confidence=confidence
        )