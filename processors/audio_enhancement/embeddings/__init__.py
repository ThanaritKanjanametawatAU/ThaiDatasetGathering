"""Speaker embedding extraction module."""

from .speaker_embedding_extractor import (
    SpeakerEmbeddingExtractor,
    EmbeddingResult,
    SpeakerEmbeddingError,
    ModelNotAvailableError,
    AudioTooShortError
)

__all__ = [
    'SpeakerEmbeddingExtractor',
    'EmbeddingResult',
    'SpeakerEmbeddingError',
    'ModelNotAvailableError',
    'AudioTooShortError'
]