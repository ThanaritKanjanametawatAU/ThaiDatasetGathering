"""
Voice Activity Detection Module
Implements robust VAD with multiple detection methods
"""

import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
import librosa
from scipy import signal as scipy_signal
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import optional dependencies
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    webrtcvad = None

logger = logging.getLogger(__name__)


class VADMethod(Enum):
    """VAD method enumeration"""
    ENERGY = "energy"
    SPECTRAL = "spectral"
    WEBRTC = "webrtc"
    NEURAL = "neural"
    HYBRID = "hybrid"


@dataclass
class SpeechSegment:
    """Represents a speech segment"""
    start_time: float
    end_time: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        """Get segment duration"""
        return self.end_time - self.start_time


@dataclass
class VADResult:
    """VAD detection result"""
    speech_ratio: float
    frame_decisions: List[bool]
    speech_frames: np.ndarray
    num_frames: int
    confidence: float = 1.0
    frame_duration: float = 0.03  # seconds
    contains_music: Optional[bool] = None


class FeatureExtractor:
    """Extract features for VAD"""
    
    def __init__(self, sample_rate: int = 16000, frame_size: int = 480, hop_size: int = 240):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract multiple features for VAD
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Energy
        features['energy'] = self._compute_energy(audio)
        
        # Zero crossing rate
        features['zcr'] = self._compute_zcr(audio)
        
        # Get reference length from energy feature
        ref_length = len(features['energy'])
        
        # Spectral features
        stft = librosa.stft(audio, n_fft=self.frame_size, hop_length=self.hop_size)
        magnitude = np.abs(stft)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sample_rate
        )[0]
        # Ensure same length as energy
        if len(spectral_centroid) != ref_length:
            spectral_centroid = np.interp(
                np.linspace(0, 1, ref_length),
                np.linspace(0, 1, len(spectral_centroid)),
                spectral_centroid
            )
        features['spectral_centroid'] = spectral_centroid
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sample_rate
        )[0]
        if len(spectral_rolloff) != ref_length:
            spectral_rolloff = np.interp(
                np.linspace(0, 1, ref_length),
                np.linspace(0, 1, len(spectral_rolloff)),
                spectral_rolloff
            )
        features['spectral_rolloff'] = spectral_rolloff
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(
            S=magnitude
        )[0]
        if len(spectral_flatness) != ref_length:
            spectral_flatness = np.interp(
                np.linspace(0, 1, ref_length),
                np.linspace(0, 1, len(spectral_flatness)),
                spectral_flatness
            )
        features['spectral_flatness'] = spectral_flatness
        
        return features
    
    def _compute_energy(self, audio: np.ndarray) -> np.ndarray:
        """Compute frame energy"""
        # Frame the signal
        frames = librosa.util.frame(
            audio, 
            frame_length=self.frame_size, 
            hop_length=self.hop_size
        )
        
        # Compute RMS energy
        energy = np.sqrt(np.mean(frames ** 2, axis=0))
        
        return energy
    
    def _compute_zcr(self, audio: np.ndarray) -> np.ndarray:
        """Compute zero crossing rate"""
        # Use same framing as energy to ensure consistent length
        frames = librosa.util.frame(
            audio, 
            frame_length=self.frame_size, 
            hop_length=self.hop_size
        )
        
        # Compute ZCR for each frame
        zcr = np.array([
            np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2.0 * len(frame))
            for frame in frames.T
        ])
        
        return zcr


class EnergyVAD:
    """Energy-based VAD"""
    
    def __init__(self, sample_rate: int, frame_size: int, hop_size: int,
                 aggressiveness: int = 1, adaptive_threshold: bool = False):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.aggressiveness = aggressiveness
        self.adaptive_threshold = adaptive_threshold
        self.energy_threshold = None
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(sample_rate, frame_size, hop_size)
    
    def detect(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect speech using energy
        
        Returns:
            frame_decisions: Boolean array of speech/non-speech
            probabilities: Speech probability per frame
        """
        # Extract energy
        features = self.feature_extractor.extract_features(audio)
        energy = features['energy']
        
        # Handle empty or very short audio
        if len(energy) == 0:
            return np.array([], dtype=bool), np.array([])
        
        # Compute threshold
        if self.adaptive_threshold or self.energy_threshold is None:
            # Adaptive threshold based on energy distribution
            sorted_energy = np.sort(energy)
            noise_floor_idx = max(1, int(0.1 * len(sorted_energy)))  # At least 1 sample
            noise_floor = np.mean(sorted_energy[:noise_floor_idx])
            
            # Handle case where noise floor is NaN or 0
            if np.isnan(noise_floor) or noise_floor == 0:
                noise_floor = np.min(energy[energy > 0]) if np.any(energy > 0) else 1e-8
            
            # Set threshold based on aggressiveness
            threshold_multipliers = [1.5, 2.0, 2.5, 3.0]
            multiplier = threshold_multipliers[min(self.aggressiveness, 3)]
            self.energy_threshold = noise_floor * multiplier
        
        # Make decisions
        frame_decisions = energy > self.energy_threshold
        
        # Convert to probabilities (sigmoid-like)
        energy_normalized = energy / (self.energy_threshold + 1e-8)
        probabilities = 1 / (1 + np.exp(-5 * (energy_normalized - 1)))
        
        return frame_decisions, probabilities


class SpectralVAD:
    """Spectral feature-based VAD"""
    
    def __init__(self, sample_rate: int, frame_size: int, hop_size: int,
                 aggressiveness: int = 1):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.aggressiveness = aggressiveness
        self.feature_extractor = FeatureExtractor(sample_rate, frame_size, hop_size)
    
    def detect(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect speech using spectral features"""
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Combine features for decision
        # Speech typically has:
        # - Moderate spectral centroid (not too low, not too high)
        # - Lower spectral flatness (more tonal)
        # - Moderate energy
        
        # Normalize features
        energy = features['energy']
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        
        centroid = features['spectral_centroid']
        centroid_norm = centroid / (self.sample_rate / 2)
        
        flatness = features['spectral_flatness']
        
        # Speech likelihood based on features
        # Energy contribution
        energy_score = energy_norm
        
        # Centroid contribution (prefer mid-range frequencies)
        centroid_score = 1 - 2 * np.abs(centroid_norm - 0.3)
        centroid_score = np.clip(centroid_score, 0, 1)
        
        # Flatness contribution (lower is more speech-like)
        flatness_score = 1 - flatness
        
        # Combine scores
        weights = [0.4, 0.3, 0.3]  # energy, centroid, flatness
        speech_score = (weights[0] * energy_score + 
                       weights[1] * centroid_score + 
                       weights[2] * flatness_score)
        
        # Apply aggressiveness
        thresholds = [0.3, 0.4, 0.5, 0.6]
        threshold = thresholds[min(self.aggressiveness, 3)]
        
        frame_decisions = speech_score > threshold
        probabilities = speech_score
        
        return frame_decisions, probabilities


class NeuralVAD:
    """Neural network-based VAD"""
    
    def __init__(self, sample_rate: int, frame_size: int, hop_size: int,
                 model_path: Optional[str] = None):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.feature_extractor = FeatureExtractor(sample_rate, frame_size, hop_size)
        
        # Load or create model
        if model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_simple_model()
    
    def _create_simple_model(self) -> nn.Module:
        """Create a simple neural VAD model"""
        class SimpleVAD(nn.Module):
            def __init__(self, input_dim=5):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 2)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return SimpleVAD()
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load pre-trained model"""
        # Placeholder - would load actual model
        return self._create_simple_model()
    
    def detect(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect speech using neural network"""
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Prepare input features
        feature_matrix = np.stack([
            features['energy'],
            features['zcr'],
            features['spectral_centroid'] / (self.sample_rate / 2),
            features['spectral_rolloff'] / (self.sample_rate / 2),
            features['spectral_flatness']
        ], axis=1)
        
        # Normalize
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_matrix)
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            speech_probs = probs[:, 1].numpy()
        
        # Make decisions
        frame_decisions = speech_probs > 0.5
        
        return frame_decisions, speech_probs


class WebRTCVAD:
    """WebRTC-based VAD wrapper"""
    
    def __init__(self, sample_rate: int, frame_size: int, hop_size: int,
                 aggressiveness: int = 1):
        if not WEBRTC_AVAILABLE:
            raise ImportError("webrtcvad not available. Install with: pip install webrtcvad")
        
        # WebRTC VAD only supports specific sample rates
        supported_rates = [8000, 16000, 32000, 48000]
        if sample_rate not in supported_rates:
            raise ValueError(f"WebRTC VAD only supports sample rates: {supported_rates}")
        
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # WebRTC expects specific frame durations
        self.frame_duration_ms = 30  # 10, 20, or 30 ms
    
    def detect(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect speech using WebRTC VAD"""
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Process frames
        frame_size_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        num_frames = len(audio_int16) // frame_size_samples
        
        frame_decisions = []
        for i in range(num_frames):
            start = i * frame_size_samples
            end = start + frame_size_samples
            frame = audio_int16[start:end].tobytes()
            
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            frame_decisions.append(is_speech)
        
        frame_decisions = np.array(frame_decisions)
        
        # Convert to probabilities (binary for WebRTC)
        probabilities = frame_decisions.astype(float)
        
        return frame_decisions, probabilities


class VoiceActivityDetector:
    """Main Voice Activity Detector class"""
    
    def __init__(self, method: Union[str, VADMethod] = 'hybrid',
                 sample_rate: int = 16000,
                 frame_duration: float = 0.03,
                 aggressiveness: int = 1,
                 adaptive_threshold: bool = False):
        """Initialize VAD
        
        Args:
            method: Detection method ('energy', 'spectral', 'webrtc', 'neural', 'hybrid')
            sample_rate: Audio sample rate
            frame_duration: Frame duration in seconds
            aggressiveness: Detection aggressiveness (0-3)
            adaptive_threshold: Use adaptive threshold
        """
        # Convert string to enum
        if isinstance(method, str):
            method = VADMethod(method)
        
        self.method = method
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(frame_duration * sample_rate)
        self.hop_size = self.frame_size // 2
        self.aggressiveness = aggressiveness
        self.adaptive_threshold = adaptive_threshold
        
        # Initialize detectors based on method
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize detection methods"""
        self.detectors = {}
        
        if self.method in [VADMethod.ENERGY, VADMethod.HYBRID]:
            self.detectors['energy'] = EnergyVAD(
                self.sample_rate, self.frame_size, self.hop_size,
                self.aggressiveness, self.adaptive_threshold
            )
        
        if self.method in [VADMethod.SPECTRAL, VADMethod.HYBRID]:
            self.detectors['spectral'] = SpectralVAD(
                self.sample_rate, self.frame_size, self.hop_size,
                self.aggressiveness
            )
        
        if self.method == VADMethod.WEBRTC:
            try:
                self.detectors['webrtc'] = WebRTCVAD(
                    self.sample_rate, self.frame_size, self.hop_size,
                    self.aggressiveness
                )
            except (ImportError, ValueError) as e:
                logger.warning(f"WebRTC VAD not available: {e}. Falling back to energy VAD.")
                self.method = VADMethod.ENERGY
                self.detectors['energy'] = EnergyVAD(
                    self.sample_rate, self.frame_size, self.hop_size,
                    self.aggressiveness, self.adaptive_threshold
                )
        
        if self.method == VADMethod.NEURAL:
            self.detectors['neural'] = NeuralVAD(
                self.sample_rate, self.frame_size, self.hop_size
            )
    
    def detect(self, audio: np.ndarray) -> VADResult:
        """Detect voice activity in audio
        
        Args:
            audio: Audio signal
            
        Returns:
            VADResult with detection results
        """
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio[0]  # Take first channel
        
        # Get decisions based on method
        if self.method == VADMethod.HYBRID:
            # Combine multiple methods
            all_decisions = []
            all_probs = []
            
            for name, detector in self.detectors.items():
                decisions, probs = detector.detect(audio)
                all_decisions.append(decisions)
                all_probs.append(probs)
            
            # Combine by averaging probabilities
            combined_probs = np.mean(all_probs, axis=0)
            frame_decisions = combined_probs > 0.5
            speech_probs = combined_probs
        else:
            # Single method
            detector_name = self.method.value
            if detector_name == 'webrtc' and 'webrtc' not in self.detectors:
                detector_name = 'energy'  # Fallback
            
            detector = self.detectors[detector_name]
            frame_decisions, speech_probs = detector.detect(audio)
        
        # Calculate metrics
        speech_frames = np.where(frame_decisions)[0]
        speech_ratio = np.mean(frame_decisions) if len(frame_decisions) > 0 else 0.0
        
        # Calculate confidence
        if len(speech_probs) > 0:
            confidence = np.mean(np.maximum(speech_probs, 1 - speech_probs))
        else:
            confidence = 0.0
        
        # Create result
        result = VADResult(
            speech_ratio=float(speech_ratio),
            frame_decisions=frame_decisions.tolist(),
            speech_frames=speech_frames,
            num_frames=len(frame_decisions),
            confidence=float(confidence),
            frame_duration=self.frame_duration
        )
        
        # Check for music (simplified - based on spectral features)
        if 'spectral' in self.detectors:
            # Music tends to have stable spectral characteristics
            # This is a simplified check
            result.contains_music = False  # Placeholder
        
        return result
    
    def get_speech_segments(self, audio: np.ndarray,
                           min_speech_duration: float = 0.1,
                           min_silence_duration: float = 0.1,
                           speech_pad_duration: float = 0.05) -> List[SpeechSegment]:
        """Get speech segments from audio
        
        Args:
            audio: Audio signal
            min_speech_duration: Minimum speech segment duration
            min_silence_duration: Minimum silence duration to split segments
            speech_pad_duration: Padding to add around speech segments
            
        Returns:
            List of speech segments
        """
        # Get frame decisions
        result = self.detect(audio)
        frame_decisions = np.array(result.frame_decisions)
        
        # Convert frames to time
        frame_times = np.arange(len(frame_decisions)) * self.hop_size / self.sample_rate
        
        # Find speech regions
        speech_regions = []
        in_speech = False
        start_time = 0
        
        for i, (time, is_speech) in enumerate(zip(frame_times, frame_decisions)):
            if is_speech and not in_speech:
                # Speech onset
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                # Speech offset
                end_time = time
                speech_regions.append((start_time, end_time))
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            speech_regions.append((start_time, frame_times[-1]))
        
        # Merge close segments
        merged_regions = []
        for start, end in speech_regions:
            if merged_regions and start - merged_regions[-1][1] < min_silence_duration:
                # Merge with previous
                merged_regions[-1] = (merged_regions[-1][0], end)
            else:
                merged_regions.append((start, end))
        
        # Filter by minimum duration and add padding
        segments = []
        for start, end in merged_regions:
            duration = end - start
            if duration >= min_speech_duration:
                # Add padding
                padded_start = max(0, start - speech_pad_duration)
                padded_end = min(len(audio) / self.sample_rate, end + speech_pad_duration)
                
                # Calculate confidence for this segment
                start_frame = int(start * self.sample_rate / self.hop_size)
                end_frame = int(end * self.sample_rate / self.hop_size)
                segment_confidence = result.confidence  # Simplified
                
                segment = SpeechSegment(
                    start_time=padded_start,
                    end_time=padded_end,
                    confidence=segment_confidence
                )
                segments.append(segment)
        
        return segments
    
    def get_speech_probability(self, audio: np.ndarray) -> np.ndarray:
        """Get frame-wise speech probabilities
        
        Args:
            audio: Audio signal
            
        Returns:
            Array of speech probabilities per frame
        """
        # Get decisions based on method
        if self.method == VADMethod.HYBRID:
            # Combine multiple methods
            all_probs = []
            
            for name, detector in self.detectors.items():
                _, probs = detector.detect(audio)
                all_probs.append(probs)
            
            # Average probabilities
            probabilities = np.mean(all_probs, axis=0)
        else:
            # Single method
            detector_name = self.method.value
            if detector_name == 'webrtc' and 'webrtc' not in self.detectors:
                detector_name = 'energy'  # Fallback
            
            detector = self.detectors[detector_name]
            _, probabilities = detector.detect(audio)
        
        return probabilities


class StreamingVAD:
    """Streaming VAD for real-time processing"""
    
    def __init__(self, method: str = 'energy', sample_rate: int = 16000,
                 chunk_duration: float = 0.032, context_duration: float = 0.5):
        """Initialize streaming VAD
        
        Args:
            method: VAD method
            sample_rate: Audio sample rate
            chunk_duration: Duration of each chunk in seconds
            context_duration: Context window duration
        """
        self.method = method
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(chunk_duration * sample_rate)
        self.context_duration = context_duration
        self.context_size = int(context_duration * sample_rate)
        
        # Initialize detector
        self.detector = VoiceActivityDetector(
            method=method,
            sample_rate=sample_rate,
            frame_duration=chunk_duration
        )
        
        # Ring buffer for context
        self.buffer = np.zeros(self.context_size)
        self.buffer_idx = 0
        self.total_chunks = 0
        
        # Smoothing
        self.decision_history = []
        self.hangover_frames = 5
        self.onset_frames = 2
    
    def process_chunk(self, chunk: np.ndarray) -> Dict:
        """Process single audio chunk
        
        Args:
            chunk: Audio chunk
            
        Returns:
            Dictionary with detection results
        """
        # Add to buffer
        chunk_len = len(chunk)
        if self.buffer_idx + chunk_len <= self.context_size:
            self.buffer[self.buffer_idx:self.buffer_idx + chunk_len] = chunk
            self.buffer_idx += chunk_len
        else:
            # Wrap around
            remaining = self.context_size - self.buffer_idx
            self.buffer[self.buffer_idx:] = chunk[:remaining]
            self.buffer[:chunk_len - remaining] = chunk[remaining:]
            self.buffer_idx = chunk_len - remaining
        
        # Get current context
        if self.total_chunks * self.chunk_size < self.context_size:
            # Not enough context yet
            context = self.buffer[:self.buffer_idx]
        else:
            # Full context available
            context = np.roll(self.buffer, -self.buffer_idx)
        
        # Ensure we have enough samples for detection
        if len(context) < self.detector.frame_size:
            # Not enough samples yet
            self.total_chunks += 1
            return {
                'is_speech': False,
                'probability': 0.0,
                'timestamp': self.total_chunks * self.chunk_duration
            }
        
        # Detect on context
        result = self.detector.detect(context)
        
        # Get latest frame decision
        if len(result.frame_decisions) > 0:
            latest_decision = result.frame_decisions[-1]
            latest_prob = result.confidence
        else:
            latest_decision = False
            latest_prob = 0.0
        
        # Apply smoothing
        self.decision_history.append(latest_decision)
        if len(self.decision_history) > self.hangover_frames:
            self.decision_history.pop(0)
        
        # Make smoothed decision
        if sum(self.decision_history[-self.onset_frames:]) >= self.onset_frames:
            # Onset detected
            smoothed_decision = True
        elif sum(self.decision_history) == 0:
            # All recent frames are non-speech
            smoothed_decision = False
        else:
            # In hangover period
            smoothed_decision = True
        
        self.total_chunks += 1
        
        return {
            'is_speech': smoothed_decision,
            'probability': latest_prob,
            'timestamp': self.total_chunks * self.chunk_duration
        }


# Placeholder for model loading function
def load_neural_vad_model(model_path: str) -> nn.Module:
    """Load pre-trained neural VAD model"""
    # This would load an actual pre-trained model
    # For now, return a simple model
    model = NeuralVAD(16000, 480, 240)
    return model.model