"""
Scenario Classifier Module (S03_T08)
Classifies audio scenarios and acoustic environments for optimal processing
"""

import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
import warnings

logger = logging.getLogger(__name__)


class AudioScenario(Enum):
    """Audio scenario types"""
    SINGLE_SPEAKER = "single_speaker"
    MULTI_SPEAKER = "multi_speaker"
    OVERLAPPING_SPEECH = "overlapping_speech"
    MUSIC = "music"
    NOISE = "noise"
    SILENCE = "silence"
    MIXED_CONTENT = "mixed_content"
    UNKNOWN = "unknown"


@dataclass
class ClassifierConfig:
    """Configuration for scenario classifier"""
    sensitivity: str = "medium"  # high, medium, low
    min_confidence: float = 0.5
    frame_duration: float = 0.025  # 25ms frames
    hop_duration: float = 0.010    # 10ms hop
    min_speech_duration: float = 0.1
    min_music_duration: float = 0.5
    enable_environment_analysis: bool = True


@dataclass
class ScenarioFeatures:
    """Extracted features for scenario classification"""
    # Energy features
    energy_level: float
    energy_variation: float
    silence_ratio: float
    
    # Spectral features
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    spectral_flatness: float
    spectral_complexity: float
    spectral_stability: float
    harmonic_ratio: float
    
    # Temporal features
    zero_crossing_rate: float
    temporal_variation: float
    onset_rate: float
    
    # Speech-specific features
    speech_ratio: float
    formant_clarity: float
    pitch_variation: float
    
    # Multi-speaker features
    speaker_change_rate: float
    spectral_diversity: float
    
    # Noise features
    noise_ratio: float
    
    # Music features
    rhythm_strength: float
    tonal_stability: float
    
    # Fields with default values (must come last)
    noise_type: str = "unknown"
    reverb_estimate: float = 0.0
    snr_estimate: float = 0.0


@dataclass
class ClassificationResult:
    """Result of scenario classification"""
    scenario: AudioScenario
    confidence: float
    estimated_speakers: int
    features: ScenarioFeatures
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            "scenario": self.scenario.value,
            "confidence": self.confidence,
            "estimated_speakers": self.estimated_speakers,
            "features": {
                "energy_level": self.features.energy_level,
                "spectral_centroid": self.features.spectral_centroid,
                "speech_ratio": self.features.speech_ratio,
                "harmonic_ratio": self.features.harmonic_ratio,
                "speaker_change_rate": self.features.speaker_change_rate
            },
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class ScenarioClassifier:
    """Audio scenario classifier for multi-modal audio analysis"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 config: Optional[ClassifierConfig] = None,
                 streaming_mode: bool = False,
                 window_duration: float = 5.0,
                 update_interval: float = 1.0):
        """Initialize the scenario classifier.
        
        Args:
            sample_rate: Audio sample rate
            config: Classification configuration
            streaming_mode: Enable streaming classification
            window_duration: Window size for streaming mode
            update_interval: Update interval for streaming mode
        """
        self.sample_rate = sample_rate
        self.config = config or ClassifierConfig()
        self.streaming_mode = streaming_mode
        self.window_duration = window_duration
        self.update_interval = update_interval
        
        # Frame parameters
        self.frame_length = int(self.config.frame_duration * sample_rate)
        self.hop_length = int(self.config.hop_duration * sample_rate)
        
        # Streaming state
        if streaming_mode:
            self._init_streaming_state()
        
        # Classification thresholds based on sensitivity
        self._set_sensitivity_thresholds()
    
    def _init_streaming_state(self):
        """Initialize streaming state variables"""
        self.streaming_buffer = []
        self.last_update_time = 0.0
        self.current_classification = None
        self.classification_history = []
    
    def _set_sensitivity_thresholds(self):
        """Set classification thresholds based on sensitivity setting"""
        if self.config.sensitivity == "high":
            self.speech_threshold = 0.3
            self.music_threshold = 0.4
            self.noise_threshold = 0.6
            self.silence_threshold = 0.01
        elif self.config.sensitivity == "low":
            self.speech_threshold = 0.7
            self.music_threshold = 0.8
            self.noise_threshold = 0.9
            self.silence_threshold = 0.005
        else:  # medium
            self.speech_threshold = 0.5
            self.music_threshold = 0.6
            self.noise_threshold = 0.8
            self.silence_threshold = 0.01
    
    def classify_scenario(self, audio: np.ndarray,
                         analyze_environment: bool = False) -> ClassificationResult:
        """Classify the audio scenario.
        
        Args:
            audio: Audio signal
            analyze_environment: Include acoustic environment analysis
            
        Returns:
            Classification result
        """
        # Validate input
        if len(audio) == 0:
            raise ValueError("Audio array is empty")
        
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise ValueError("Audio contains NaN or infinite values")
        
        # Normalize audio
        audio = self._normalize_audio(audio)
        
        # Extract features
        features = self.extract_features(audio)
        
        # Classify scenario
        scenario, confidence, num_speakers = self._classify_from_features(features)
        
        # Environment analysis
        metadata = {}
        if analyze_environment and self.config.enable_environment_analysis:
            env_info = self._analyze_acoustic_environment(audio, features)
            metadata["acoustic_environment"] = env_info
        
        return ClassificationResult(
            scenario=scenario,
            confidence=confidence,
            estimated_speakers=num_speakers,
            features=features,
            metadata=metadata
        )
    
    def extract_features(self, audio: np.ndarray) -> ScenarioFeatures:
        """Extract comprehensive features for classification.
        
        Args:
            audio: Audio signal
            
        Returns:
            Extracted features
        """
        # Energy features
        energy_features = self._extract_energy_features(audio)
        
        # Spectral features
        spectral_features = self._extract_spectral_features(audio)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(audio)
        
        # Speech features
        speech_features = self._extract_speech_features(audio)
        
        # Multi-speaker features
        speaker_features = self._extract_speaker_features(audio)
        
        # Noise features
        noise_features = self._extract_noise_features(audio)
        
        # Music features
        music_features = self._extract_music_features(audio)
        
        # Combine all features
        return ScenarioFeatures(
            # Energy
            energy_level=energy_features["energy_level"],
            energy_variation=energy_features["energy_variation"],
            silence_ratio=energy_features["silence_ratio"],
            
            # Spectral
            spectral_centroid=spectral_features["centroid"],
            spectral_bandwidth=spectral_features["bandwidth"],
            spectral_rolloff=spectral_features["rolloff"],
            spectral_flatness=spectral_features["flatness"],
            spectral_complexity=spectral_features["complexity"],
            spectral_stability=spectral_features["stability"],
            harmonic_ratio=spectral_features["harmonic_ratio"],
            
            # Temporal
            zero_crossing_rate=temporal_features["zcr"],
            temporal_variation=temporal_features["variation"],
            onset_rate=temporal_features["onset_rate"],
            
            # Speech
            speech_ratio=speech_features["speech_ratio"],
            formant_clarity=speech_features["formant_clarity"],
            pitch_variation=speech_features["pitch_variation"],
            
            # Speaker
            speaker_change_rate=speaker_features["change_rate"],
            spectral_diversity=speaker_features["diversity"],
            
            # Noise
            noise_ratio=noise_features["noise_ratio"],
            
            # Music
            rhythm_strength=music_features["rhythm_strength"],
            tonal_stability=music_features["tonal_stability"],
            
            # Fields with defaults (optional)
            noise_type=noise_features["noise_type"],
            reverb_estimate=0.0,
            snr_estimate=0.0
        )
    
    def _extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract energy-based features"""
        # Frame the signal
        frames = self._frame_signal(audio)
        frame_energies = np.array([np.sqrt(np.mean(frame ** 2)) for frame in frames])
        
        # Overall energy level
        energy_level = np.mean(frame_energies)
        
        # Energy variation
        energy_variation = np.std(frame_energies) / (energy_level + 1e-8)
        
        # Silence ratio
        silence_frames = frame_energies < self.silence_threshold
        silence_ratio = np.mean(silence_frames)
        
        return {
            "energy_level": float(energy_level),
            "energy_variation": float(energy_variation),
            "silence_ratio": float(silence_ratio)
        }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features"""
        # Compute spectrogram
        f, t, Sxx = scipy_signal.spectrogram(
            audio, self.sample_rate,
            nperseg=self.frame_length,
            noverlap=self.frame_length - self.hop_length
        )
        
        # Power spectrum
        power_spectrum = np.abs(Sxx)
        
        # Spectral centroid
        if np.sum(power_spectrum) > 0:
            centroid = np.sum(f[:, np.newaxis] * power_spectrum) / np.sum(power_spectrum, axis=0)
            spectral_centroid = np.mean(centroid)
        else:
            spectral_centroid = 0.0
        
        # Spectral bandwidth
        if spectral_centroid > 0:
            bandwidth = np.sqrt(np.sum(((f[:, np.newaxis] - spectral_centroid) ** 2) * power_spectrum) / 
                               np.sum(power_spectrum, axis=0))
            spectral_bandwidth = np.mean(bandwidth)
        else:
            spectral_bandwidth = 0.0
        
        # Spectral rolloff (95% of energy)
        cumsum_spectrum = np.cumsum(power_spectrum, axis=0)
        total_energy = cumsum_spectrum[-1, :]
        rolloff_idx = np.argmax(cumsum_spectrum >= 0.95 * total_energy, axis=0)
        spectral_rolloff = np.mean(f[rolloff_idx])
        
        # Spectral flatness
        spectral_flatness = self._compute_spectral_flatness(power_spectrum)
        
        # Spectral complexity (entropy)
        spectral_complexity = self._compute_spectral_complexity(power_spectrum)
        
        # Spectral stability (frame-to-frame correlation)
        spectral_stability = self._compute_spectral_stability(power_spectrum)
        
        # Harmonic ratio
        harmonic_ratio = self._compute_harmonic_ratio(audio)
        
        return {
            "centroid": float(spectral_centroid),
            "bandwidth": float(spectral_bandwidth),
            "rolloff": float(spectral_rolloff),
            "flatness": float(spectral_flatness),
            "complexity": float(spectral_complexity),
            "stability": float(spectral_stability),
            "harmonic_ratio": float(harmonic_ratio)
        }
    
    def _extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal features"""
        # Zero crossing rate
        frames = self._frame_signal(audio)
        zcr_values = []
        for frame in frames:
            if len(frame) > 1:
                zcr = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2.0 * len(frame))
                zcr_values.append(zcr)
        
        zero_crossing_rate = np.mean(zcr_values) if zcr_values else 0.0
        
        # Temporal variation
        frame_energies = np.array([np.sqrt(np.mean(frame ** 2)) for frame in frames])
        temporal_variation = np.std(np.diff(frame_energies)) if len(frame_energies) > 1 else 0.0
        
        # Onset detection rate
        onset_rate = self._compute_onset_rate(audio)
        
        return {
            "zcr": float(zero_crossing_rate),
            "variation": float(temporal_variation),
            "onset_rate": float(onset_rate)
        }
    
    def _extract_speech_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract speech-specific features"""
        # Speech probability based on spectral characteristics
        speech_ratio = self._estimate_speech_ratio(audio)
        
        # Formant clarity (simplified)
        formant_clarity = self._estimate_formant_clarity(audio)
        
        # Pitch variation
        pitch_variation = self._estimate_pitch_variation(audio)
        
        return {
            "speech_ratio": float(speech_ratio),
            "formant_clarity": float(formant_clarity),
            "pitch_variation": float(pitch_variation)
        }
    
    def _extract_speaker_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract multi-speaker features"""
        # Speaker change detection
        change_rate = self._estimate_speaker_change_rate(audio)
        
        # Spectral diversity (measure of different spectral patterns)
        diversity = self._compute_spectral_diversity(audio)
        
        return {
            "change_rate": float(change_rate),
            "diversity": float(diversity)
        }
    
    def _extract_noise_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract noise-specific features"""
        # Noise ratio estimation
        noise_ratio = self._estimate_noise_ratio(audio)
        
        # Noise type classification
        noise_type = self._classify_noise_type(audio)
        
        return {
            "noise_ratio": float(noise_ratio),
            "noise_type": noise_type
        }
    
    def _extract_music_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract music-specific features"""
        # Rhythm strength
        rhythm_strength = self._compute_rhythm_strength(audio)
        
        # Tonal stability
        tonal_stability = self._compute_tonal_stability(audio)
        
        return {
            "rhythm_strength": float(rhythm_strength),
            "tonal_stability": float(tonal_stability)
        }
    
    def _classify_from_features(self, features: ScenarioFeatures) -> Tuple[AudioScenario, float, int]:
        """Classify scenario from extracted features"""
        # Priority-based classification for better accuracy
        
        # 1. Silence detection (highest priority)
        if features.silence_ratio > 0.9 and features.energy_level < self.silence_threshold:
            return AudioScenario.SILENCE, 0.95, 0
        
        # 2. Check for very low energy (silence)
        if features.energy_level < self.silence_threshold * 2:
            return AudioScenario.SILENCE, 0.9, 0
        
        # 3. Noise detection (high noise ratio, low speech ratio)
        if features.noise_ratio > 0.65 and features.speech_ratio < 0.3:
            return AudioScenario.NOISE, features.noise_ratio, 0
        
        # 4. Music detection (harmonic content OR strong rhythm)
        music_score = features.harmonic_ratio * features.tonal_stability
        # Music can be detected by strong harmonics OR strong rhythm
        if (music_score > 0.1 and features.rhythm_strength > 0.7) or \
           (features.harmonic_ratio > 0.2 and features.tonal_stability > 0.3):
            combined_music_score = max(music_score, features.rhythm_strength * 0.8)
            return AudioScenario.MUSIC, combined_music_score, 0
        
        # 5. Speech-based scenarios
        speech_score = self._compute_speech_score(features)
        
        if speech_score > self.speech_threshold:
            # Check for multiple speakers
            if features.speaker_change_rate > 0.6 and features.spectral_diversity > 0.5:
                # Multi-speaker scenario
                multi_score = speech_score * (features.speaker_change_rate + features.spectral_diversity) / 2
                return AudioScenario.MULTI_SPEAKER, multi_score, 2
            
            # Check for overlapping speech (high complexity, low stability)
            elif features.spectral_complexity > 0.7 and features.spectral_stability < 0.5:
                overlap_score = speech_score * features.spectral_complexity
                return AudioScenario.OVERLAPPING_SPEECH, overlap_score, 2
            
            # Single speaker
            else:
                return AudioScenario.SINGLE_SPEAKER, speech_score, 1
        
        # 6. Mixed content (moderate speech + music)
        if speech_score > 0.3 and music_score > 0.3:
            mixed_score = (speech_score + music_score) / 2
            return AudioScenario.MIXED_CONTENT, mixed_score, 1
        
        # 7. Default to unknown if nothing matches
        return AudioScenario.UNKNOWN, 0.3, 0
    
    def _compute_speech_score(self, features: ScenarioFeatures) -> float:
        """Compute overall speech likelihood score"""
        # Combine multiple speech indicators
        speech_indicators = [
            features.speech_ratio,
            features.formant_clarity,
            min(features.spectral_centroid / 4000, 1.0),  # Normalize centroid
            1 - features.spectral_flatness,  # Speech is not flat
            min(features.zero_crossing_rate * 5, 1.0)  # Moderate ZCR
        ]
        
        # Weighted combination
        weights = [0.4, 0.2, 0.2, 0.1, 0.1]
        speech_score = sum(w * s for w, s in zip(weights, speech_indicators))
        
        return np.clip(speech_score, 0, 1)
    
    def _estimate_speaker_count(self, features: ScenarioFeatures, scenario: AudioScenario) -> int:
        """Estimate number of speakers based on features and scenario"""
        if scenario in [AudioScenario.SILENCE, AudioScenario.NOISE]:
            return 0
        elif scenario == AudioScenario.SINGLE_SPEAKER:
            return 1
        elif scenario == AudioScenario.MULTI_SPEAKER:
            # Estimate based on speaker change rate and spectral diversity
            if features.speaker_change_rate > 0.8 and features.spectral_diversity > 0.7:
                return 3  # Likely 3+ speakers
            else:
                return 2  # Likely 2 speakers
        elif scenario == AudioScenario.OVERLAPPING_SPEECH:
            return max(2, int(features.spectral_complexity * 3))
        elif scenario in [AudioScenario.MUSIC, AudioScenario.MIXED_CONTENT]:
            # Could have singers/speakers
            if features.speech_ratio > 0.3:
                return 1
            else:
                return 0
        else:
            return 0
    
    def classify_batch(self, audio_list: List[np.ndarray]) -> List[ClassificationResult]:
        """Classify multiple audio segments in batch"""
        results = []
        for audio in audio_list:
            result = self.classify_scenario(audio)
            results.append(result)
        return results
    
    def update_streaming(self, audio_chunk: np.ndarray, timestamp: float) -> Optional[ClassificationResult]:
        """Update streaming classification with new audio chunk"""
        if not self.streaming_mode:
            raise RuntimeError("Streaming mode not enabled")
        
        # Add chunk to buffer
        self.streaming_buffer.append({
            "audio": audio_chunk,
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
            return self.current_classification
        
        # Concatenate audio in window
        if self.streaming_buffer:
            window_audio = np.concatenate([chunk["audio"] for chunk in self.streaming_buffer])
            
            # Classify window
            result = self.classify_scenario(window_audio)
            
            # Update state
            self.current_classification = result
            self.last_update_time = timestamp
            self.classification_history.append({
                "timestamp": timestamp,
                "result": result
            })
            
            # Limit history size
            if len(self.classification_history) > 100:
                self.classification_history.pop(0)
            
            return result
        
        return None
    
    def detect_transitions(self, audio: np.ndarray, segment_duration: float = 2.0) -> List[Dict]:
        """Detect scenario transitions in audio"""
        segment_samples = int(segment_duration * self.sample_rate)
        transitions = []
        
        prev_scenario = None
        prev_timestamp = 0
        
        # Process segments
        for i in range(0, len(audio), segment_samples):
            segment = audio[i:i + segment_samples]
            if len(segment) < segment_samples // 2:  # Skip very short segments
                break
            
            # Classify segment
            result = self.classify_scenario(segment)
            current_scenario = result.scenario
            current_timestamp = i / self.sample_rate
            
            # Detect transition
            if prev_scenario is not None and current_scenario != prev_scenario:
                transition = {
                    "timestamp": current_timestamp,
                    "from_scenario": prev_scenario.value,
                    "to_scenario": current_scenario.value,
                    "confidence": result.confidence
                }
                transitions.append(transition)
            
            prev_scenario = current_scenario
            prev_timestamp = current_timestamp
        
        return transitions
    
    def _analyze_acoustic_environment(self, audio: np.ndarray, features: ScenarioFeatures) -> Dict:
        """Analyze acoustic environment characteristics"""
        env_info = {}
        
        # Reverberation estimation
        reverb_level = self._estimate_reverberation(audio)
        env_info["reverb_level"] = float(reverb_level)
        
        # Noise level estimation
        noise_level = features.noise_ratio
        env_info["noise_level"] = float(noise_level)
        
        # SNR estimation
        snr_estimate = self._estimate_snr(audio, features)
        env_info["snr_estimate"] = float(snr_estimate)
        
        # Environment classification
        if reverb_level > 0.7:
            env_type = "reverberant"
        elif noise_level > 0.8:
            env_type = "noisy"
        elif features.energy_level < 0.1:
            env_type = "quiet"
        else:
            env_type = "normal"
        
        env_info["environment_type"] = env_type
        
        return env_info
    
    # Helper methods for feature computation
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio signal while preserving energy levels for classification"""
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # For classification, preserve energy levels by only normalizing if signal is very loud
        max_val = np.max(np.abs(audio))
        
        # Only normalize if signal is clipping or very loud (> 1.0)
        # This preserves energy information for silence detection
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio
    
    def _frame_signal(self, audio: np.ndarray) -> List[np.ndarray]:
        """Frame the signal for analysis"""
        frames = []
        for i in range(0, len(audio) - self.frame_length, self.hop_length):
            frame = audio[i:i + self.frame_length]
            frames.append(frame)
        return frames
    
    def _compute_spectral_flatness(self, power_spectrum: np.ndarray) -> float:
        """Compute spectral flatness (measure of tonality vs noise)"""
        # Avoid log of zero
        power_spectrum = power_spectrum + 1e-10
        
        # Geometric mean / arithmetic mean
        geo_mean = np.exp(np.mean(np.log(power_spectrum), axis=0))
        arith_mean = np.mean(power_spectrum, axis=0)
        
        flatness = geo_mean / (arith_mean + 1e-10)
        return float(np.mean(flatness))
    
    def _compute_spectral_complexity(self, power_spectrum: np.ndarray) -> float:
        """Compute spectral complexity using entropy"""
        # Normalize to probabilities
        prob_spectrum = power_spectrum / (np.sum(power_spectrum, axis=0, keepdims=True) + 1e-10)
        
        # Compute entropy
        entropy = -np.sum(prob_spectrum * np.log(prob_spectrum + 1e-10), axis=0)
        
        # Normalize by max possible entropy
        max_entropy = np.log(prob_spectrum.shape[0])
        normalized_entropy = entropy / max_entropy
        
        return float(np.mean(normalized_entropy))
    
    def _compute_spectral_stability(self, power_spectrum: np.ndarray) -> float:
        """Compute spectral stability (frame-to-frame correlation)"""
        if power_spectrum.shape[1] < 2:
            return 1.0
        
        correlations = []
        for i in range(power_spectrum.shape[1] - 1):
            frame1 = power_spectrum[:, i]
            frame2 = power_spectrum[:, i + 1]
            
            # Normalize frames
            frame1 = frame1 / (np.linalg.norm(frame1) + 1e-10)
            frame2 = frame2 / (np.linalg.norm(frame2) + 1e-10)
            
            # Compute correlation
            correlation = np.dot(frame1, frame2)
            correlations.append(correlation)
        
        return float(np.mean(correlations))
    
    def _compute_harmonic_ratio(self, audio: np.ndarray) -> float:
        """Estimate harmonic content ratio"""
        # Use frequency domain approach for better harmonic detection
        
        # Compute power spectrum
        spectrum = np.abs(np.fft.rfft(audio)) ** 2
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        if len(spectrum) < 10:
            return 0.0
        
        # Find peaks in spectrum
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(spectrum, height=np.max(spectrum) * 0.1)
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate harmonic relationships
        peak_freqs = freqs[peaks]
        peak_amplitudes = spectrum[peaks]
        
        # Sort by amplitude
        sorted_indices = np.argsort(peak_amplitudes)[::-1]
        peak_freqs = peak_freqs[sorted_indices]
        peak_amplitudes = peak_amplitudes[sorted_indices]
        
        # Check for harmonic relationships
        harmonic_score = 0.0
        total_energy = np.sum(peak_amplitudes)
        
        if len(peak_freqs) >= 2:
            fundamental_candidates = peak_freqs[:3]  # Check top 3 peaks as potential fundamentals
            
            for fundamental in fundamental_candidates:
                if fundamental < 50 or fundamental > 2000:  # Skip very low/high frequencies
                    continue
                
                harmonic_energy = 0.0
                
                # Look for harmonics (2f, 3f, 4f, 5f)
                for harmonic_num in range(2, 6):
                    target_freq = fundamental * harmonic_num
                    
                    # Find closest peak
                    freq_diffs = np.abs(peak_freqs - target_freq)
                    closest_idx = np.argmin(freq_diffs)
                    
                    # If close enough (within 5% tolerance)
                    if freq_diffs[closest_idx] < target_freq * 0.05:
                        harmonic_energy += peak_amplitudes[closest_idx]
                
                # Calculate harmonic ratio for this fundamental
                if total_energy > 0:
                    candidate_ratio = harmonic_energy / total_energy
                    harmonic_score = max(harmonic_score, candidate_ratio)
        
        return float(np.clip(harmonic_score, 0, 1))
    
    def _compute_onset_rate(self, audio: np.ndarray) -> float:
        """Compute onset detection rate"""
        # Simple onset detection using spectral flux
        frames = self._frame_signal(audio)
        spectral_flux = []
        
        prev_spectrum = None
        for frame in frames:
            spectrum = np.abs(np.fft.rfft(frame))
            
            if prev_spectrum is not None:
                flux = np.sum(np.maximum(0, spectrum - prev_spectrum))
                spectral_flux.append(flux)
            
            prev_spectrum = spectrum
        
        if not spectral_flux:
            return 0.0
        
        # Count onsets (peaks in spectral flux)
        threshold = np.mean(spectral_flux) + np.std(spectral_flux)
        onsets = np.sum(np.array(spectral_flux) > threshold)
        
        # Rate per second
        duration = len(audio) / self.sample_rate
        onset_rate = onsets / duration if duration > 0 else 0.0
        
        return float(onset_rate)
    
    def _estimate_speech_ratio(self, audio: np.ndarray) -> float:
        """Estimate ratio of speech content"""
        frames = self._frame_signal(audio)
        speech_frames = 0
        total_valid_frames = 0
        
        for frame in frames:
            if len(frame) < 10:
                continue
            
            # Energy check - skip very quiet frames
            energy = np.sqrt(np.mean(frame ** 2))
            if energy < self.silence_threshold * 0.5:
                continue
            
            total_valid_frames += 1
            
            # Spectral characteristics
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)
            
            # Check multiple speech indicators
            speech_indicators = 0
            
            # 1. Speech frequency band (300-3400 Hz)
            speech_band = (freqs >= 300) & (freqs <= 3400)
            if np.sum(speech_band) > 0:
                speech_energy = np.sum(spectrum[speech_band])
                total_energy = np.sum(spectrum)
                if speech_energy / (total_energy + 1e-10) > 0.4:
                    speech_indicators += 1
            
            # 2. Formant-like peaks in speech range
            f1_range = (freqs >= 200) & (freqs <= 1000)
            f2_range = (freqs >= 800) & (freqs <= 2500)
            if np.sum(f1_range) > 0 and np.sum(f2_range) > 0:
                f1_energy = np.max(spectrum[f1_range])
                f2_energy = np.max(spectrum[f2_range])
                avg_energy = np.mean(spectrum)
                if f1_energy > avg_energy * 2 or f2_energy > avg_energy * 2:
                    speech_indicators += 1
            
            # 3. Moderate zero crossing rate (typical for speech)
            if len(frame) > 1:
                zcr = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2.0 * len(frame))
                if 0.02 < zcr < 0.3:  # Typical speech ZCR range
                    speech_indicators += 1
            
            # 4. Not too flat spectrum (speech has structure)
            if np.sum(spectrum) > 0:
                norm_spectrum = spectrum / np.sum(spectrum)
                entropy = -np.sum(norm_spectrum * np.log(norm_spectrum + 1e-10))
                max_entropy = np.log(len(norm_spectrum))
                normalized_entropy = entropy / max_entropy
                if 0.3 < normalized_entropy < 0.85:  # Speech has moderate entropy
                    speech_indicators += 1
            
            # 5. Check for noise-like characteristics (exclude if too noise-like)
            # Spectral flatness check - noise has high flatness
            if np.sum(spectrum) > 0:
                # Simple flatness for this frame
                geo_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
                arith_mean = np.mean(spectrum)
                frame_flatness = geo_mean / (arith_mean + 1e-10)
                if frame_flatness > 0.8:  # Very flat spectrum = noise
                    speech_indicators = max(0, speech_indicators - 2)  # Strong penalty
            
            # Consider it speech if it meets at least 2 criteria and doesn't look like noise
            if speech_indicators >= 2:
                speech_frames += 1
        
        if total_valid_frames == 0:
            return 0.0
        
        speech_ratio = speech_frames / total_valid_frames
        return float(speech_ratio)
    
    def _estimate_formant_clarity(self, audio: np.ndarray) -> float:
        """Estimate formant clarity (simplified)"""
        # Compute power spectrum
        spectrum = np.abs(np.fft.rfft(audio)) ** 2
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Look for formant-like peaks in typical ranges
        # F1: 200-1000 Hz, F2: 800-2500 Hz
        f1_range = (freqs >= 200) & (freqs <= 1000)
        f2_range = (freqs >= 800) & (freqs <= 2500)
        
        if np.sum(f1_range) > 0 and np.sum(f2_range) > 0:
            f1_peak = np.max(spectrum[f1_range])
            f2_peak = np.max(spectrum[f2_range])
            total_energy = np.sum(spectrum)
            
            formant_clarity = (f1_peak + f2_peak) / (total_energy + 1e-10)
            return float(np.clip(formant_clarity, 0, 1))
        
        return 0.0
    
    def _estimate_pitch_variation(self, audio: np.ndarray) -> float:
        """Estimate pitch variation over time"""
        # Simple pitch tracking using autocorrelation
        frames = self._frame_signal(audio)
        pitches = []
        
        for frame in frames:
            if len(frame) < 100:
                continue
            
            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find pitch period
            min_period = int(self.sample_rate / 500)  # Max 500 Hz
            max_period = int(self.sample_rate / 50)   # Min 50 Hz
            
            if max_period < len(autocorr):
                search_range = autocorr[min_period:max_period]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    pitch = self.sample_rate / peak_idx
                    pitches.append(pitch)
        
        if len(pitches) > 1:
            pitch_variation = np.std(pitches) / (np.mean(pitches) + 1e-10)
            return float(np.clip(pitch_variation, 0, 2))
        
        return 0.0
    
    def _estimate_speaker_change_rate(self, audio: np.ndarray) -> float:
        """Estimate rate of speaker changes"""
        # Segment audio and compare spectral characteristics
        segment_duration = 1.0  # 1 second segments
        segment_samples = int(segment_duration * self.sample_rate)
        
        if len(audio) < 2 * segment_samples:
            return 0.0
        
        segment_features = []
        for i in range(0, len(audio) - segment_samples, segment_samples):
            segment = audio[i:i + segment_samples]
            
            # Extract spectral centroid as speaker characteristic
            spectrum = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
            
            if np.sum(spectrum) > 0:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                segment_features.append(centroid)
        
        # Count significant changes
        if len(segment_features) < 2:
            return 0.0
        
        changes = 0
        threshold = np.std(segment_features) * 0.5
        
        for i in range(1, len(segment_features)):
            if abs(segment_features[i] - segment_features[i-1]) > threshold:
                changes += 1
        
        change_rate = changes / (len(segment_features) - 1)
        return float(change_rate)
    
    def _compute_spectral_diversity(self, audio: np.ndarray) -> float:
        """Compute spectral diversity across time"""
        # Frame audio and compute spectral features
        frames = self._frame_signal(audio)
        centroids = []
        
        for frame in frames:
            if len(frame) < 10:
                continue
            
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)
            
            if np.sum(spectrum) > 0:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                centroids.append(centroid)
        
        if len(centroids) > 1:
            diversity = np.std(centroids) / (np.mean(centroids) + 1e-10)
            return float(np.clip(diversity, 0, 2))
        
        return 0.0
    
    def _estimate_noise_ratio(self, audio: np.ndarray) -> float:
        """Estimate ratio of noise content"""
        speech_ratio = self._estimate_speech_ratio(audio)
        
        # Analyze spectral characteristics for noise
        spectrum = np.abs(np.fft.rfft(audio))
        flatness = self._compute_spectral_flatness(spectrum.reshape(-1, 1))
        
        # Noise indicators
        noise_indicators = []
        
        # 1. High spectral flatness (white/pink noise characteristic)
        noise_indicators.append(flatness)
        
        # 2. Low speech ratio
        noise_indicators.append(1 - speech_ratio)
        
        # 3. High zero crossing rate variability (noise characteristic)
        frames = self._frame_signal(audio)
        zcr_values = []
        for frame in frames:
            if len(frame) > 1:
                zcr = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2.0 * len(frame))
                zcr_values.append(zcr)
        
        if zcr_values:
            zcr_var = np.std(zcr_values) / (np.mean(zcr_values) + 1e-10)
            noise_indicators.append(min(zcr_var, 1.0))
        else:
            noise_indicators.append(0.5)
        
        # 4. Lack of harmonic structure
        harmonic_ratio = self._compute_harmonic_ratio(audio)
        noise_indicators.append(1 - harmonic_ratio)
        
        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]
        noise_ratio = sum(w * indicator for w, indicator in zip(weights, noise_indicators))
        
        return float(np.clip(noise_ratio, 0, 1))
    
    def _classify_noise_type(self, audio: np.ndarray) -> str:
        """Classify type of noise"""
        spectrum = np.abs(np.fft.rfft(audio)) ** 2
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Analyze spectral slope
        if len(spectrum) > 1:
            # Fit line to log spectrum vs log frequency
            log_freqs = np.log(freqs[1:] + 1e-10)
            log_spectrum = np.log(spectrum[1:] + 1e-10)
            
            slope = np.polyfit(log_freqs, log_spectrum, 1)[0]
            
            if slope < -1.5:
                return "pink"  # 1/f noise
            elif slope < -0.5:
                return "brown"  # 1/f^2 noise
            else:
                return "white"  # Flat spectrum
        
        return "unknown"
    
    def _compute_rhythm_strength(self, audio: np.ndarray) -> float:
        """Compute rhythm strength for music detection"""
        # Analyze temporal envelope for rhythm
        
        # Calculate temporal envelope
        from scipy.signal import hilbert
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        # Smooth envelope to extract tempo
        from scipy.signal import savgol_filter
        if len(envelope) > 21:
            smoothed_envelope = savgol_filter(envelope, 21, 3)
        else:
            smoothed_envelope = envelope
        
        # Downsample envelope for tempo analysis
        downsample_factor = max(1, len(smoothed_envelope) // 1000)
        if downsample_factor > 1:
            envelope_ds = smoothed_envelope[::downsample_factor]
            sample_rate_ds = self.sample_rate / downsample_factor
        else:
            envelope_ds = smoothed_envelope
            sample_rate_ds = self.sample_rate
        
        if len(envelope_ds) < 10:
            return 0.0
        
        # Compute autocorrelation of envelope to find periodic patterns
        autocorr = np.correlate(envelope_ds, envelope_ds, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for peaks in autocorrelation corresponding to musical tempos
        # 60-200 BPM -> 1-3.33 Hz -> periods of 0.3-1.0 seconds
        min_period_samples = int(0.3 * sample_rate_ds)  # 200 BPM
        max_period_samples = int(1.0 * sample_rate_ds)  # 60 BPM
        
        if max_period_samples < len(autocorr) and min_period_samples < max_period_samples:
            search_range = autocorr[min_period_samples:max_period_samples]
            
            if len(search_range) > 0:
                max_autocorr = np.max(search_range)
                # Normalize by zero-lag autocorrelation
                rhythm_strength = max_autocorr / (autocorr[0] + 1e-10)
                
                # Additional check: look for regularity in envelope variations
                envelope_var = np.std(envelope_ds) / (np.mean(envelope_ds) + 1e-10)
                if envelope_var < 0.1:  # Too flat - no rhythm
                    rhythm_strength *= 0.1
                
                return float(np.clip(rhythm_strength, 0, 1))
        
        return 0.0
    
    def _compute_tonal_stability(self, audio: np.ndarray) -> float:
        """Compute tonal stability for music detection"""
        # Harmonic ratio is a good indicator of tonal stability
        harmonic_ratio = self._compute_harmonic_ratio(audio)
        
        # Also consider spectral stability
        f, t, Sxx = scipy_signal.spectrogram(
            audio, self.sample_rate,
            nperseg=self.frame_length,
            noverlap=self.frame_length - self.hop_length
        )
        
        spectral_stability = self._compute_spectral_stability(np.abs(Sxx))
        
        # Combine harmonic and spectral stability
        tonal_stability = 0.7 * harmonic_ratio + 0.3 * spectral_stability
        return float(np.clip(tonal_stability, 0, 1))
    
    def _estimate_reverberation(self, audio: np.ndarray) -> float:
        """Estimate reverberation level"""
        # Simple method: analyze decay characteristics
        # Compute envelope using Hilbert transform
        analytic_signal = scipy_signal.hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        # Look for exponential decay patterns
        # Smooth envelope
        smoothed_envelope = scipy_signal.savgol_filter(
            envelope, 
            window_length=min(101, len(envelope)//10*2+1), 
            polyorder=3
        )
        
        # Compute decay rate
        if len(smoothed_envelope) > 100:
            # Find decay regions
            decay_rates = []
            for i in range(len(smoothed_envelope) - 50):
                segment = smoothed_envelope[i:i+50]
                if np.max(segment) > 0.1:  # Significant signal
                    # Linear fit to log envelope
                    log_segment = np.log(segment + 1e-10)
                    slope = np.polyfit(range(len(log_segment)), log_segment, 1)[0]
                    decay_rates.append(-slope)  # Positive for decay
            
            if decay_rates:
                avg_decay = np.mean(decay_rates)
                # Map decay rate to reverb level
                reverb_level = np.clip(avg_decay / 0.1, 0, 1)
                return float(reverb_level)
        
        return 0.0
    
    def _estimate_snr(self, audio: np.ndarray, features: ScenarioFeatures) -> float:
        """Estimate signal-to-noise ratio"""
        # Use energy-based estimation
        frames = self._frame_signal(audio)
        frame_energies = np.array([np.sqrt(np.mean(frame ** 2)) for frame in frames])
        
        # Estimate noise floor from quietest frames
        noise_floor = np.percentile(frame_energies, 10)
        
        # Estimate signal level from loudest frames
        signal_level = np.percentile(frame_energies, 90)
        
        if noise_floor > 0:
            snr_linear = signal_level / noise_floor
            snr_db = 20 * np.log10(snr_linear)
            return float(np.clip(snr_db, 0, 60))
        
        return 30.0  # Default moderate SNR