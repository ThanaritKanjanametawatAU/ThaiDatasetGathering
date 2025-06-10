"""
Speaker Diarization Module

This module provides speaker diarization functionality to segment audio
into speaker turns and identify different speakers in a conversation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import json
import pandas as pd
import warnings
from pathlib import Path

# Try to import optional dependencies
try:
    import speechbrain as sb
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    warnings.warn("SpeechBrain not available. Using simple embeddings.")

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    warnings.warn("pyannote.audio not available. Using custom implementation.")


class ClusteringMethod(Enum):
    """Available clustering methods"""
    SPECTRAL = "spectral"
    AGGLOMERATIVE = "agglomerative"
    KMEANS = "kmeans"


class SegmentationMethod(Enum):
    """Available segmentation methods"""
    UNIFORM = "uniform"
    ENERGY_BASED = "energy_based"
    CHANGE_DETECTION = "change_detection"


@dataclass
class SpeakerSegment:
    """Represents a speaker segment"""
    start_time: float
    end_time: float
    speaker_id: int
    confidence: float
    channel: Optional[int] = None
    
    @property
    def duration(self) -> float:
        """Get segment duration"""
        return self.end_time - self.start_time


@dataclass
class SpeakerEmbedding:
    """Speaker embedding representation"""
    vector: np.ndarray
    confidence: float
    segment_id: Optional[int] = None


@dataclass
class DiarizationResult:
    """Result of speaker diarization"""
    segments: List[SpeakerSegment]
    num_speakers: int
    overlap_ratio: float
    embeddings: Optional[Dict[int, SpeakerEmbedding]] = None
    
    def to_rttm(self, filename: str = "audio") -> str:
        """Convert to RTTM format"""
        lines = []
        for segment in self.segments:
            duration = segment.end_time - segment.start_time
            line = f"SPEAKER {filename} 1 {segment.start_time:.3f} {duration:.3f} <NA> <NA> SPK{segment.speaker_id:02d} <NA> <NA>"
            lines.append(line)
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Convert to JSON format"""
        data = {
            "num_speakers": int(self.num_speakers),
            "overlap_ratio": float(self.overlap_ratio),
            "segments": [
                {
                    "start_time": float(seg.start_time),
                    "end_time": float(seg.end_time),
                    "speaker_id": int(seg.speaker_id),
                    "confidence": float(seg.confidence),
                    "channel": int(seg.channel) if seg.channel is not None else None
                }
                for seg in self.segments
            ]
        }
        return json.dumps(data, indent=2)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        data = []
        for seg in self.segments:
            data.append({
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "speaker_id": seg.speaker_id,
                "confidence": seg.confidence,
                "channel": seg.channel
            })
        return pd.DataFrame(data)


class SimpleEmbeddingExtractor:
    """Simple embedding extractor using spectral features"""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.model = self._create_model()
    
    def _create_model(self) -> nn.Module:
        """Create a simple neural network for embeddings"""
        class EmbeddingNet(nn.Module):
            def __init__(self, input_dim=40, embedding_dim=256):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, embedding_dim)
                )
            
            def forward(self, x):
                return F.normalize(self.encoder(x), p=2, dim=-1)
        
        return EmbeddingNet(embedding_dim=self.embedding_dim)
    
    def extract(self, features: np.ndarray) -> np.ndarray:
        """Extract embedding from features"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            if features_tensor.dim() == 1:
                features_tensor = features_tensor.unsqueeze(0)
            embedding = self.model(features_tensor)
            return embedding.numpy().squeeze()


class SpeakerDiarization:
    """Main speaker diarization class"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        embedding_model: Optional[str] = None,
        clustering_method: Union[str, ClusteringMethod] = ClusteringMethod.SPECTRAL,
        min_speakers: int = 2,
        max_speakers: int = 10,
        language: str = 'en',
        use_channel_info: bool = False
    ):
        self.sample_rate = sample_rate
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.language = language
        self.use_channel_info = use_channel_info
        
        # Set clustering method
        if isinstance(clustering_method, str):
            self.clustering_method = ClusteringMethod(clustering_method.lower())
        else:
            self.clustering_method = clustering_method
        
        # Initialize embedding model
        self.embedding_model = self._init_embedding_model(embedding_model)
        
        # Initialize VAD
        from .voice_activity_detector import VoiceActivityDetector
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        
        # Segmentation parameters
        self.window_size = 1.5  # seconds
        self.step_size = 0.75   # seconds
    
    def _init_embedding_model(self, model_name: Optional[str]) -> Any:
        """Initialize embedding extraction model"""
        if SPEECHBRAIN_AVAILABLE and model_name:
            try:
                return EncoderClassifier.from_hparams(
                    source=model_name,
                    savedir=f"pretrained_models/{model_name}"
                )
            except Exception as e:
                warnings.warn(f"Failed to load SpeechBrain model: {e}")
        
        # Fallback to simple extractor
        return SimpleEmbeddingExtractor()
    
    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: Optional[int] = None
    ) -> DiarizationResult:
        """Perform speaker diarization on audio
        
        Args:
            audio: Audio signal (mono or multi-channel)
            num_speakers: Number of speakers (if known)
            
        Returns:
            DiarizationResult object
        """
        # Validate input
        if audio.size == 0:
            raise ValueError("Empty audio provided")
        
        if audio.ndim > 2:
            raise ValueError(f"Invalid audio shape: {audio.shape}. Expected 1D or 2D array.")
        
        # Check minimum length
        min_samples = int(0.5 * self.sample_rate)  # 0.5 seconds minimum
        if audio.shape[-1] < min_samples:
            raise ValueError(f"Audio too short. Minimum {min_samples} samples required.")
        
        # Handle multi-channel audio
        if audio.ndim == 2 and self.use_channel_info:
            return self._diarize_multichannel(audio, num_speakers)
        
        # Convert to mono if needed
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        
        # Apply VAD
        vad_result = self.vad.detect(audio)
        speech_segments = self.vad.get_speech_segments(audio)
        
        if len(speech_segments) == 0:
            # No speech detected
            return DiarizationResult(segments=[], num_speakers=0, overlap_ratio=0.0)
        
        # Extract embeddings from speech segments
        embeddings = []
        segment_info = []
        
        for segment in speech_segments:
            start_sample = int(segment.start_time * self.sample_rate)
            end_sample = int(segment.end_time * self.sample_rate)
            segment_audio = audio[start_sample:end_sample]
            
            # Skip very short segments
            if len(segment_audio) < self.sample_rate * 0.3:  # 300ms minimum
                continue
            
            # Extract windows from segment
            windows = self._extract_windows(segment_audio)
            
            for window, (win_start, win_end) in windows:
                embedding = self.extract_embedding(window)
                embeddings.append(embedding.vector)
                
                # Convert to absolute time
                start_time = segment.start_time + win_start
                end_time = segment.start_time + win_end
                segment_info.append((start_time, end_time, embedding.confidence))
        
        if len(embeddings) == 0:
            return DiarizationResult(segments=[], num_speakers=0, overlap_ratio=0.0)
        
        # Stack embeddings
        embeddings_array = np.array(embeddings)
        
        # Determine number of speakers
        if num_speakers is None:
            num_speakers = self._estimate_num_speakers(embeddings_array)
        else:
            num_speakers = np.clip(num_speakers, self.min_speakers, self.max_speakers)
        
        # Cluster embeddings
        labels = self._cluster_embeddings(embeddings_array, num_speakers)
        
        # Create segments
        segments = []
        for i, (start_time, end_time, confidence) in enumerate(segment_info):
            segment = SpeakerSegment(
                start_time=start_time,
                end_time=end_time,
                speaker_id=int(labels[i]),
                confidence=confidence
            )
            segments.append(segment)
        
        # Merge adjacent segments from same speaker
        segments = self._merge_segments(segments)
        
        # Calculate overlap ratio
        overlap_ratio = self._calculate_overlap_ratio(segments)
        
        # Create speaker embeddings
        speaker_embeddings = self._create_speaker_embeddings(embeddings_array, labels)
        
        return DiarizationResult(
            segments=segments,
            num_speakers=num_speakers,
            overlap_ratio=overlap_ratio,
            embeddings=speaker_embeddings
        )
    
    def _extract_windows(self, audio: np.ndarray) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
        """Extract overlapping windows from audio"""
        windows = []
        window_samples = int(self.window_size * self.sample_rate)
        step_samples = int(self.step_size * self.sample_rate)
        
        for start in range(0, len(audio) - window_samples + 1, step_samples):
            end = start + window_samples
            window = audio[start:end]
            
            # Window times relative to segment start
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            
            windows.append((window, (start_time, end_time)))
        
        # Handle last window if needed
        if len(audio) > window_samples:
            remaining = len(audio) - (len(windows) * step_samples)
            if remaining > self.sample_rate * 0.3:  # At least 300ms
                window = audio[-window_samples:]
                start_time = (len(audio) - window_samples) / self.sample_rate
                end_time = len(audio) / self.sample_rate
                windows.append((window, (start_time, end_time)))
        
        return windows
    
    def extract_embedding(self, audio: np.ndarray) -> SpeakerEmbedding:
        """Extract speaker embedding from audio segment"""
        if SPEECHBRAIN_AVAILABLE and hasattr(self.embedding_model, 'encode_batch'):
            # Use SpeechBrain model
            with torch.no_grad():
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
                embeddings = self.embedding_model.encode_batch(audio_tensor)
                embedding_vector = embeddings.squeeze().numpy()
                confidence = 0.9  # High confidence for pretrained model
        else:
            # Use simple feature extraction
            features = self._extract_features(audio)
            embedding_vector = self.embedding_model.extract(features)
            confidence = 0.7  # Lower confidence for simple model
        
        return SpeakerEmbedding(vector=embedding_vector, confidence=confidence)
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for embedding"""
        # Simple MFCC-based features
        from scipy import signal
        from scipy.fftpack import dct
        
        # Pre-emphasis
        pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # Frame the signal
        frame_size = int(0.025 * self.sample_rate)  # 25ms
        frame_step = int(0.01 * self.sample_rate)   # 10ms
        
        # Apply window
        frames = []
        for start in range(0, len(pre_emphasized) - frame_size, frame_step):
            frame = pre_emphasized[start:start + frame_size]
            frame = frame * np.hamming(frame_size)
            frames.append(frame)
        
        if not frames:
            # Too short, return zeros
            return np.zeros(40)
        
        # Compute power spectrum
        NFFT = 512
        mag_frames = np.abs(np.fft.rfft(frames, NFFT))
        pow_frames = (mag_frames ** 2) / NFFT
        
        # Mel filterbank
        num_filters = 40
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        fft_bins = np.floor((NFFT + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filterbank
        filterbank = np.zeros((num_filters, int(NFFT / 2 + 1)))
        for i in range(1, num_filters + 1):
            filterbank[i-1, fft_bins[i-1]:fft_bins[i]] = \
                (np.arange(fft_bins[i-1], fft_bins[i]) - fft_bins[i-1]) / (fft_bins[i] - fft_bins[i-1])
            filterbank[i-1, fft_bins[i]:fft_bins[i+1]] = \
                (fft_bins[i+1] - np.arange(fft_bins[i], fft_bins[i+1])) / (fft_bins[i+1] - fft_bins[i])
        
        # Apply filterbank
        filter_banks = np.dot(pow_frames, filterbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        
        # Mean over time
        features = np.mean(filter_banks, axis=0)
        
        return features
    
    def _estimate_num_speakers(self, embeddings: np.ndarray) -> int:
        """Estimate number of speakers using eigenvalue analysis"""
        # Compute affinity matrix
        affinity = self._compute_affinity_matrix(embeddings)
        
        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(affinity)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Find elbow point
        if len(eigenvalues) > 2:
            diffs = np.diff(eigenvalues)
            elbow = np.argmax(diffs) + 1
            num_speakers = min(max(elbow, self.min_speakers), self.max_speakers)
        else:
            num_speakers = self.min_speakers
        
        return num_speakers
    
    def _compute_affinity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute affinity matrix from embeddings"""
        n_samples = len(embeddings)
        affinity = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    affinity[i, j] = 1.0
                else:
                    # Cosine similarity with safety checks
                    try:
                        # Normalize vectors
                        vec1 = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-8)
                        vec2 = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
                        
                        # Compute dot product
                        similarity = np.dot(vec1, vec2)
                        
                        # Ensure valid range
                        similarity = np.clip(similarity, -1.0, 1.0)
                        
                        # Convert to affinity (0 to 1)
                        affinity_value = (similarity + 1) / 2
                        
                        affinity[i, j] = affinity_value
                        affinity[j, i] = affinity_value
                    except Exception:
                        # Fallback to 0 similarity
                        affinity[i, j] = 0.0
                        affinity[j, i] = 0.0
        
        return affinity
    
    def _cluster_embeddings(self, embeddings: np.ndarray, num_speakers: int) -> np.ndarray:
        """Cluster embeddings into speakers"""
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)
        
        if self.clustering_method == ClusteringMethod.SPECTRAL:
            # Compute affinity
            affinity = self._compute_affinity_matrix(embeddings_norm)
            clustering = SpectralClustering(
                n_clusters=num_speakers,
                affinity='precomputed',
                random_state=42
            )
            labels = clustering.fit_predict(affinity)
            
        elif self.clustering_method == ClusteringMethod.AGGLOMERATIVE:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings_norm)
            
        elif self.clustering_method == ClusteringMethod.KMEANS:
            clustering = KMeans(
                n_clusters=num_speakers,
                n_init=10,
                random_state=42
            )
            labels = clustering.fit_predict(embeddings_norm)
            
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        return labels
    
    def _merge_segments(self, segments: List[SpeakerSegment], 
                       gap_threshold: float = 0.3) -> List[SpeakerSegment]:
        """Merge adjacent segments from the same speaker"""
        if not segments:
            return segments
        
        # Sort by start time
        segments = sorted(segments, key=lambda s: s.start_time)
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # Check if same speaker and close enough
            gap = next_seg.start_time - current.end_time
            if (current.speaker_id == next_seg.speaker_id and 
                gap <= gap_threshold):
                # Merge segments
                current = SpeakerSegment(
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                    speaker_id=current.speaker_id,
                    confidence=min(current.confidence, next_seg.confidence)
                )
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        return merged
    
    def _calculate_overlap_ratio(self, segments: List[SpeakerSegment]) -> float:
        """Calculate ratio of overlapping speech"""
        if not segments:
            return 0.0
        
        # Find total duration
        total_duration = max(seg.end_time for seg in segments)
        
        # Create timeline
        resolution = 0.01  # 10ms resolution
        timeline = np.zeros(int(total_duration / resolution))
        
        # Mark speaker activity
        for segment in segments:
            start_idx = int(segment.start_time / resolution)
            end_idx = int(segment.end_time / resolution)
            timeline[start_idx:end_idx] += 1
        
        # Calculate overlap
        overlap_frames = np.sum(timeline > 1)
        speech_frames = np.sum(timeline > 0)
        
        if speech_frames > 0:
            return overlap_frames / speech_frames
        return 0.0
    
    def _create_speaker_embeddings(self, embeddings: np.ndarray, 
                                  labels: np.ndarray) -> Dict[int, SpeakerEmbedding]:
        """Create average embedding for each speaker"""
        speaker_embeddings = {}
        
        for speaker_id in np.unique(labels):
            speaker_mask = labels == speaker_id
            speaker_embs = embeddings[speaker_mask]
            
            # Average embedding
            avg_embedding = np.mean(speaker_embs, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            speaker_embeddings[speaker_id] = SpeakerEmbedding(
                vector=avg_embedding,
                confidence=0.8
            )
        
        return speaker_embeddings
    
    def _diarize_multichannel(self, audio: np.ndarray, 
                             num_speakers: Optional[int]) -> DiarizationResult:
        """Diarize multi-channel audio"""
        n_channels = audio.shape[0]
        all_segments = []
        
        # Process each channel
        for channel_idx in range(n_channels):
            channel_audio = audio[channel_idx]
            result = self.diarize(channel_audio, num_speakers=1)
            
            # Add channel information
            for segment in result.segments:
                segment.channel = channel_idx
                segment.speaker_id = channel_idx  # Initial assignment
            
            all_segments.extend(result.segments)
        
        # If we have expected number of speakers less than channels,
        # cluster the channels
        if num_speakers and num_speakers < n_channels:
            # Extract channel embeddings
            channel_embeddings = []
            for channel_idx in range(n_channels):
                channel_audio = audio[channel_idx]
                # Get average embedding for channel
                emb = self.extract_embedding(channel_audio)
                channel_embeddings.append(emb.vector)
            
            # Cluster channels
            channel_embeddings = np.array(channel_embeddings)
            labels = self._cluster_embeddings(channel_embeddings, num_speakers)
            
            # Update speaker IDs
            for segment in all_segments:
                if segment.channel is not None:
                    segment.speaker_id = int(labels[segment.channel])
        
        # Calculate overlap
        overlap_ratio = self._calculate_overlap_ratio(all_segments)
        
        return DiarizationResult(
            segments=all_segments,
            num_speakers=num_speakers or n_channels,
            overlap_ratio=overlap_ratio
        )
    
    def compute_similarity(self, emb1: SpeakerEmbedding, 
                          emb2: SpeakerEmbedding) -> float:
        """Compute similarity between two embeddings"""
        # Normalize vectors
        vec1 = emb1.vector / (np.linalg.norm(emb1.vector) + 1e-8)
        vec2 = emb2.vector / (np.linalg.norm(emb2.vector) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2)
        
        # Convert to 0-1 range
        similarity = (similarity + 1) / 2
        
        return float(np.clip(similarity, 0, 1))
    
    def resegment(self, result: DiarizationResult,
                 min_segment_duration: float = 1.0,
                 merge_threshold: float = 0.5) -> DiarizationResult:
        """Resegment diarization result with new parameters"""
        segments = result.segments.copy()
        
        # Remove short segments
        segments = [s for s in segments if s.duration >= min_segment_duration]
        
        # Merge similar adjacent segments
        segments = self._merge_segments(segments, gap_threshold=merge_threshold)
        
        # Recalculate metrics
        num_speakers = len(set(s.speaker_id for s in segments))
        overlap_ratio = self._calculate_overlap_ratio(segments)
        
        return DiarizationResult(
            segments=segments,
            num_speakers=num_speakers,
            overlap_ratio=overlap_ratio,
            embeddings=result.embeddings
        )
    
    def evaluate(self, hypothesis: DiarizationResult, 
                reference: DiarizationResult) -> Dict[str, float]:
        """Evaluate diarization performance"""
        # Simplified DER calculation
        # In practice, use pyannote.metrics for accurate evaluation
        
        # Handle empty results
        if not hypothesis.segments and not reference.segments:
            return {
                'der': 0.0,
                'confusion': 0.0,
                'missed_speech': 0.0,
                'false_alarm': 0.0
            }
        
        if not hypothesis.segments:
            # All speech missed
            return {
                'der': 1.0,
                'confusion': 0.0,
                'missed_speech': 1.0,
                'false_alarm': 0.0
            }
        
        if not reference.segments:
            # All detected speech is false alarm
            return {
                'der': 1.0,
                'confusion': 0.0,
                'missed_speech': 0.0,
                'false_alarm': 1.0
            }
        
        total_duration = max(
            max(s.end_time for s in hypothesis.segments),
            max(s.end_time for s in reference.segments)
        )
        
        # Create timelines
        resolution = 0.01
        n_frames = int(total_duration / resolution)
        
        hyp_timeline = np.zeros(n_frames, dtype=int)
        ref_timeline = np.zeros(n_frames, dtype=int)
        
        # Fill timelines
        for seg in hypothesis.segments:
            start_idx = int(seg.start_time / resolution)
            end_idx = int(seg.end_time / resolution)
            hyp_timeline[start_idx:end_idx] = seg.speaker_id + 1
        
        for seg in reference.segments:
            start_idx = int(seg.start_time / resolution)
            end_idx = int(seg.end_time / resolution)
            ref_timeline[start_idx:end_idx] = seg.speaker_id + 1
        
        # Calculate metrics
        speech_frames = np.sum(ref_timeline > 0)
        
        # Missed speech
        missed = np.sum((ref_timeline > 0) & (hyp_timeline == 0))
        
        # False alarm
        false_alarm = np.sum((ref_timeline == 0) & (hyp_timeline > 0))
        
        # Speaker confusion (simplified - doesn't account for optimal mapping)
        confusion = np.sum((ref_timeline > 0) & (hyp_timeline > 0) & 
                          (ref_timeline != hyp_timeline))
        
        # DER
        der = (missed + false_alarm + confusion) / speech_frames if speech_frames > 0 else 1.0
        
        return {
            'der': float(der),
            'confusion': float(confusion / speech_frames) if speech_frames > 0 else 0.0,
            'missed_speech': float(missed / speech_frames) if speech_frames > 0 else 0.0,
            'false_alarm': float(false_alarm / speech_frames) if speech_frames > 0 else 0.0
        }


class StreamingDiarizer:
    """Streaming/online speaker diarizer"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        window_duration: float = 2.0,
        step_duration: float = 0.5,
        max_speakers: int = 10
    ):
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.max_speakers = max_speakers
        
        # Initialize diarizer
        self.diarizer = SpeakerDiarization(
            sample_rate=sample_rate,
            max_speakers=max_speakers
        )
        
        # Buffer for audio
        self.buffer_size = int(window_duration * sample_rate)
        self.buffer = np.zeros(self.buffer_size)
        self.buffer_pos = 0
        
        # Speaker tracking
        self.speaker_history = {}
        self.next_speaker_id = 0
        self.embeddings_db = []
        
        # Results
        self.segments = []
        self.current_time = 0.0
    
    def process_chunk(self, chunk: np.ndarray) -> List[SpeakerSegment]:
        """Process audio chunk and return new segments"""
        # Add to buffer
        chunk_size = len(chunk)
        
        if self.buffer_pos + chunk_size <= self.buffer_size:
            self.buffer[self.buffer_pos:self.buffer_pos + chunk_size] = chunk
            self.buffer_pos += chunk_size
        else:
            # Shift buffer
            shift = int(self.step_duration * self.sample_rate)
            self.buffer[:-shift] = self.buffer[shift:]
            self.buffer[-chunk_size:] = chunk
            self.buffer_pos = self.buffer_size
        
        # Check if we have enough data
        if self.buffer_pos < self.buffer_size * 0.75:
            self.current_time += chunk_size / self.sample_rate
            return []
        
        # Process current window
        window = self.buffer[:self.buffer_pos]
        
        # Extract embedding
        embedding = self.diarizer.extract_embedding(window)
        
        # Find best matching speaker
        speaker_id = self._find_or_create_speaker(embedding)
        
        # Create segment
        segment = SpeakerSegment(
            start_time=self.current_time,
            end_time=self.current_time + len(window) / self.sample_rate,
            speaker_id=speaker_id,
            confidence=embedding.confidence
        )
        
        self.segments.append(segment)
        self.current_time += chunk_size / self.sample_rate
        
        return [segment]
    
    def _find_or_create_speaker(self, embedding: SpeakerEmbedding) -> int:
        """Find matching speaker or create new one"""
        if not self.embeddings_db:
            # First speaker
            self.embeddings_db.append((self.next_speaker_id, embedding))
            self.next_speaker_id += 1
            return 0
        
        # Find best match
        best_similarity = -1
        best_speaker_id = -1
        
        for speaker_id, speaker_emb in self.embeddings_db:
            similarity = self.diarizer.compute_similarity(embedding, speaker_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker_id = speaker_id
        
        # Threshold for new speaker
        if best_similarity < 0.7 and len(self.embeddings_db) < self.max_speakers:
            # New speaker
            new_id = self.next_speaker_id
            self.embeddings_db.append((new_id, embedding))
            self.next_speaker_id += 1
            return new_id
        
        return best_speaker_id