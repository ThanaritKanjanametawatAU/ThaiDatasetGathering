"""
Speaker Utilities Module

Provides speaker profiling, embedding extraction, and comparison utilities.
Integrates with existing SpeakerIdentification system.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import logging
from pathlib import Path
import librosa
import torch

# Optional imports
try:
    from pyannote.audio import Model, Inference
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("Pyannote not available for speaker embeddings")

try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

# Import from existing system
try:
    from processors.speaker_identification import SpeakerIdentification
    EXISTING_SPEAKER_ID = True
except ImportError:
    EXISTING_SPEAKER_ID = False
    logging.warning("Existing SpeakerIdentification not available")

logger = logging.getLogger(__name__)


class SpeakerProfiler:
    """
    Creates and manages speaker profiles from audio segments.
    """
    
    def __init__(self, model_name: str = "pyannote/embedding"):
        """Initialize speaker profiler with specified model."""
        self.model_name = model_name
        self.model = None
        self.inference = None
        
        if PYANNOTE_AVAILABLE:
            self._load_model()
        elif SPEECHBRAIN_AVAILABLE:
            self._load_speechbrain_model()
        else:
            logger.warning("No speaker embedding model available")
    
    def _load_model(self):
        """Load pyannote embedding model."""
        try:
            self.model = Model.from_pretrained(self.model_name)
            self.inference = Inference(self.model, window="whole")
            logger.info(f"Loaded speaker embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def _load_speechbrain_model(self):
        """Load SpeechBrain model as fallback."""
        try:
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="tmp_speaker_model"
            )
            logger.info("Loaded SpeechBrain speaker model")
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}")
    
    def create_profile(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: Optional[List[Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Create speaker profile from audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            segments: Optional list of (start, end) segments to use
            
        Returns:
            Speaker embedding vector
        """
        if segments:
            # Extract embeddings from specific segments
            embeddings = []
            for start, end in segments:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment = audio[start_sample:end_sample]
                
                if len(segment) > sample_rate * 0.1:  # Min 100ms
                    embedding = self.extract_embedding(segment, sample_rate)
                    embeddings.append(embedding)
            
            if embeddings:
                # Average embeddings
                return np.mean(embeddings, axis=0)
        
        # Use entire audio
        return self.extract_embedding(audio, sample_rate)
    
    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract speaker embedding from audio."""
        if self.inference and PYANNOTE_AVAILABLE:
            # Pyannote expects specific format
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            embedding = self.inference({"waveform": waveform, "sample_rate": sample_rate})
            return embedding.numpy().flatten()
        
        elif self.model and SPEECHBRAIN_AVAILABLE:
            # SpeechBrain embedding
            embedding = self.model.encode_batch(torch.tensor(audio).unsqueeze(0))
            return embedding.squeeze().numpy()
        
        else:
            # Fallback to simple features
            return self._extract_fallback_features(audio, sample_rate)
    
    def _extract_fallback_features(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract basic audio features as fallback."""
        features = []
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # Energy
        energy = np.sqrt(np.mean(audio ** 2))
        features.append(energy)
        
        return np.array(features)


def extract_speaker_embedding(
    audio: np.ndarray,
    sample_rate: int,
    model_name: str = "pyannote/embedding"
) -> np.ndarray:
    """
    Extract speaker embedding from audio.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        model_name: Model to use for extraction
        
    Returns:
        Speaker embedding vector
    """
    profiler = SpeakerProfiler(model_name)
    return profiler.extract_embedding(audio, sample_rate)


def compare_embeddings(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Compare two speaker embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        metric: Similarity metric (cosine, euclidean)
        
    Returns:
        Similarity score (0-1 for cosine, distance for euclidean)
    """
    if metric == "cosine":
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Normalize to 0-1 range
        return (similarity + 1) / 2
    
    elif metric == "euclidean":
        # Euclidean distance
        return np.linalg.norm(embedding1 - embedding2)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def create_speaker_profile(
    audio: np.ndarray,
    sample_rate: int,
    clean_segments: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Create a robust speaker profile from audio.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        clean_segments: Optional clean speech segments
        
    Returns:
        Speaker profile embedding
    """
    profiler = SpeakerProfiler()
    return profiler.create_profile(audio, sample_rate, clean_segments)


class SpeakerComparator:
    """
    Advanced speaker comparison with multiple metrics.
    """
    
    def __init__(self):
        self.thresholds = {
            "same_speaker": 0.8,
            "similar_speaker": 0.6,
            "different_speaker": 0.4
        }
    
    def compare(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Compare two audio segments for speaker similarity.
        
        Returns:
            Dictionary with similarity metrics and classification
        """
        # Extract embeddings
        embedding1 = extract_speaker_embedding(audio1, sample_rate)
        embedding2 = extract_speaker_embedding(audio2, sample_rate)
        
        # Calculate similarities
        cosine_sim = compare_embeddings(embedding1, embedding2, "cosine")
        euclidean_dist = compare_embeddings(embedding1, embedding2, "euclidean")
        
        # Classification
        if cosine_sim >= self.thresholds["same_speaker"]:
            classification = "same_speaker"
        elif cosine_sim >= self.thresholds["similar_speaker"]:
            classification = "similar_speaker"
        elif cosine_sim >= self.thresholds["different_speaker"]:
            classification = "possibly_different"
        else:
            classification = "different_speaker"
        
        return {
            "cosine_similarity": cosine_sim,
            "euclidean_distance": euclidean_dist,
            "classification": classification,
            "confidence": self._calculate_confidence(cosine_sim, classification)
        }
    
    def _calculate_confidence(
        self,
        similarity: float,
        classification: str
    ) -> float:
        """Calculate confidence in classification."""
        if classification == "same_speaker":
            # Higher similarity = higher confidence
            return min(1.0, (similarity - 0.7) * 3.33)
        elif classification == "different_speaker":
            # Lower similarity = higher confidence
            return min(1.0, (0.5 - similarity) * 2.5)
        else:
            # Medium confidence for uncertain cases
            return 0.5


class ExistingSystemIntegration:
    """
    Integration with existing SpeakerIdentification system.
    """
    
    def __init__(self):
        self.speaker_id_system = None
        
        if EXISTING_SPEAKER_ID:
            try:
                self.speaker_id_system = SpeakerIdentification()
                logger.info("Integrated with existing SpeakerIdentification")
            except Exception as e:
                logger.error(f"Failed to integrate: {e}")
    
    def get_speaker_embedding_from_id(
        self,
        speaker_id: str
    ) -> Optional[np.ndarray]:
        """Get speaker embedding for a known speaker ID."""
        if not self.speaker_id_system:
            return None
        
        # This would require modification to existing system
        # to expose speaker embeddings
        logger.info(f"Getting embedding for speaker {speaker_id}")
        return None
    
    def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Optional[str]:
        """Identify speaker using existing system."""
        if not self.speaker_id_system:
            return None
        
        try:
            # Process audio through existing system
            result = self.speaker_id_system.process_audio(audio, sample_rate)
            return result.get("speaker_id")
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return None


def calculate_speaker_similarity_matrix(
    audio_segments: List[np.ndarray],
    sample_rate: int
) -> np.ndarray:
    """
    Calculate pairwise speaker similarity matrix.
    
    Args:
        audio_segments: List of audio arrays
        sample_rate: Sample rate for all segments
        
    Returns:
        Similarity matrix (N x N)
    """
    n_segments = len(audio_segments)
    similarity_matrix = np.zeros((n_segments, n_segments))
    
    # Extract all embeddings
    embeddings = []
    for segment in audio_segments:
        embedding = extract_speaker_embedding(segment, sample_rate)
        embeddings.append(embedding)
    
    # Calculate pairwise similarities
    for i in range(n_segments):
        for j in range(i, n_segments):
            similarity = compare_embeddings(embeddings[i], embeddings[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    return similarity_matrix


def cluster_speakers(
    audio_segments: List[np.ndarray],
    sample_rate: int,
    threshold: float = 0.7
) -> List[int]:
    """
    Cluster audio segments by speaker.
    
    Args:
        audio_segments: List of audio arrays
        sample_rate: Sample rate
        threshold: Similarity threshold for clustering
        
    Returns:
        List of cluster labels
    """
    # Calculate similarity matrix
    similarity_matrix = calculate_speaker_similarity_matrix(audio_segments, sample_rate)
    
    # Simple agglomerative clustering
    n_segments = len(audio_segments)
    labels = list(range(n_segments))
    
    # Find pairs above threshold
    for i in range(n_segments):
        for j in range(i + 1, n_segments):
            if similarity_matrix[i, j] >= threshold:
                # Merge clusters
                old_label = labels[j]
                new_label = labels[i]
                for k in range(n_segments):
                    if labels[k] == old_label:
                        labels[k] = new_label
    
    # Renumber clusters
    unique_labels = list(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = [label_map[label] for label in labels]
    
    return labels