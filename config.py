"""
Configuration settings for the Thai Audio Dataset Collection project.
"""

import os
import logging
from enum import Enum, auto

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure directories exist
for directory in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "GigaSpeech2": {
        "name": "GigaSpeech2",
        "source": "speechcolab/gigaspeech2",
        "language_filter": "th",
        "description": "GigaSpeech2 Thai language dataset",
        "processor_class": "GigaSpeech2Processor"
    },
    "ProcessedVoiceTH": {
        "name": "ProcessedVoiceTH",
        "source": "Porameht/processed-voice-th-169k",
        "description": "Processed Voice TH dataset",
        "processor_class": "ProcessedVoiceTHProcessor"
    },
    "MozillaCommonVoice": {
        "name": "MozillaCommonVoice",
        "source": "mozilla-foundation/common_voice_11_0",
        "language_filter": "th",
        "description": "Mozilla Common Voice dataset (Thai only)",
        "processor_class": "MozillaCommonVoiceProcessor"
    }
}

# Target dataset configuration
TARGET_DATASET = {
    "name": "Thanarit/Thai-Voice",
    "description": "Combined Thai audio dataset from multiple sources",
    "language": "th"
}

# Schema configuration
SCHEMA = {
    "ID": str,          # Sequential ID (S1, S2, ...)
    "speaker_id": str,  # Speaker identifier (SPK_00001, SPK_00002, ...)
    "Language": str,    # Always "th" for Thai
    "audio": dict,      # Audio data in HuggingFace format (array, sampling_rate, path)
    "transcript": str,  # Transcript text
    "length": float,    # Audio length in seconds
    "dataset_name": str,        # Source dataset name (NEW)
    "confidence_score": float,  # STT confidence score 0.0-1.0 (NEW)
    "enhancement_metadata": dict,  # Audio enhancement metadata (optional)
}

# Export TARGET_SCHEMA for compatibility
TARGET_SCHEMA = SCHEMA

# Validation rules
VALIDATION_RULES = {
    "ID": {
        "pattern": r"^(S\d+|temp_\d+)$",  # Allow both S{n} and temp_{n} for streaming
        "required": True,
        "unique": True,
        "error_message": "ID must be in format 'S{n}' or 'temp_{n}' where n is a number"
    },
    "Language": {
        "allowed_values": ["th"],
        "required": True,
        "error_message": "Language must be 'th'"
    },
    "audio": {
        "required": True,
        # Audio is now a dict with array, sampling_rate, and path keys
        "error_message": "Audio must be present and in correct format"
    },
    "transcript": {
        "required": True,  # CHANGED: Now required (100% coverage)
        "min_length": 1,   # Cannot be empty
        "max_length": 10000,  # Reasonable maximum length
        "error_message": "Transcript must be non-empty and under 10000 characters"
    },
    "length": {
        "required": True,
        "min_value": 0.01,  # At least 0.01 seconds (very permissive)
        "max_value": 3600,  # Maximum 1 hour
        "error_message": "Length must be between 0.01 and 3600 seconds"
    },
    "dataset_name": {
        "required": True,
        "min_length": 1,
        "error_message": "Dataset name must be provided"
    },
    "confidence_score": {
        "required": True,
        "min_value": 0.0,
        "max_value": 1.0,
        "error_message": "Confidence score must be between 0.0 and 1.0"
    },
    "speaker_id": {
        "required": True,
        "pattern": r"^SPK_\d{5}$",
        "error_message": "Speaker ID must be in format 'SPK_XXXXX' where X is a digit"
    },
    "enhancement_metadata": {
        "required": False,  # Optional field
        "type": dict,
        "error_message": "Enhancement metadata must be a dictionary"
    }
}

# Audio processing configuration
AUDIO_CONFIG = {
    "target_format": "wav",
    "target_sample_rate": 16000,
    "target_channels": 1,
    "normalize_volume": True,
    "target_db": -20.0,  # Target volume level in dB
    "trim_silence": True,
    "enable_standardization": True  # Master switch for audio standardization
}

# Logging configuration
LOG_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(LOG_DIR, "dataset_processing.log")
}

# Streaming configuration
STREAMING_CONFIG = {
    "batch_size": 1000,           # Number of samples to process before yielding
    "upload_batch_size": 10000,   # Number of samples before uploading a shard
    "shard_size": "5GB",          # Maximum size of each shard file
    "max_retries": 3,             # Maximum retries for network operations
    "retry_delay": 5,             # Delay between retries in seconds
}


# Error categories
class ErrorCategory(Enum):
    NETWORK = auto()
    FILE_ACCESS = auto()
    VALIDATION = auto()
    PROCESSING = auto()
    AUTHENTICATION = auto()
    UPLOAD = auto()
    UNKNOWN = auto()


# Exit codes
EXIT_CODES = {
    "SUCCESS": 0,
    "GENERAL_ERROR": 1,
    "ARGUMENT_ERROR": 2,
    "AUTHENTICATION_ERROR": 3,
    "DATASET_ACCESS_ERROR": 4,
    "PROCESSING_ERROR": 5,
    "UPLOAD_ERROR": 6
}

# Huggingface configuration
HF_TOKEN_FILE = os.path.join(PROJECT_ROOT, "hf_token.txt")

# STT Configuration
STT_CONFIG = {
    "enable_stt": True,  # Enable STT for missing transcripts
    "stt_batch_size": 16,  # Batch size for STT processing
    "primary_model": "airesearch/wav2vec2-large-xlsr-53-th",
    "secondary_model": "openai/whisper-large-v3",
    "fallback_transcript": "[INAUDIBLE]",  # Used when STT fails
    "error_transcript": "[STT_ERROR]",     # Used when STT crashes
    "no_audio_transcript": "[NO_AUDIO]",   # Used when audio is missing
}

# Speaker Identification Configuration
SPEAKER_ID_CONFIG = {
    "enabled": False,  # Default disabled, enable with --enable-speaker-id
    "model": "pyannote/wespeaker-voxceleb-resnet34-LM",
    "embedding_dim": 256,
    "batch_size": 10000,
    "clustering": {
        "algorithm": "adaptive",  # Uses different algorithms based on batch size
        "min_cluster_size": 2,    # Minimum cluster size for HDBSCAN
        "min_samples": 1,         # Minimum samples for core points
        "metric": "cosine",
        "cluster_selection_epsilon": 0.3,  # Distance threshold for clustering
        "similarity_threshold": 0.7        # Similarity threshold for merging
    },
    "storage": {
        "store_embeddings": False,
        "embedding_format": "hdf5",
        "compression": "gzip"
    }
}

# Noise Reduction Configuration
NOISE_REDUCTION_CONFIG = {
    "enabled": False,  # Default disabled, enable with --enable-noise-reduction
    "device": "cuda",  # Use GPU if available
    "adaptive_mode": True,  # Automatically select enhancement level
    "default_level": "moderate",  # Default level if adaptive mode is off
    "levels": {
        "mild": {
            "dry_wet_ratio": 0.1,
            "prop_decrease": 0.6,
            "target_snr": 20
        },
        "moderate": {
            "dry_wet_ratio": 0.05,
            "prop_decrease": 0.8,
            "target_snr": 25
        },
        "aggressive": {
            "dry_wet_ratio": 0.02,
            "prop_decrease": 1.0,
            "target_snr": 30
        }
    },
    "quality_targets": {
        "min_snr_improvement": 5,     # Minimum dB improvement
        "max_snr_improvement": 10,    # Maximum dB improvement
        "min_pesq": 3.0,              # Minimum PESQ score
        "min_stoi": 0.85,             # Minimum STOI score
        "min_speaker_similarity": 0.95  # Minimum speaker preservation
    },
    "processing": {
        "batch_size": 32,             # GPU batch size
        "max_duration": 300,          # Max audio duration in seconds
        "clean_threshold_snr": 30,    # Skip if SNR > this value
        "fallback_to_cpu": True,      # Use CPU if GPU fails
        "warm_up_model": True,        # Warm up model on init
        "target_pesq": 3.0,           # Target PESQ score
        "target_stoi": 0.85          # Target STOI score
    },
    "engines": {
        "primary": "denoiser",        # Facebook Denoiser
        "fallback": "spectral",       # Spectral gating
        "denoiser_model": "dns64"     # Denoiser model name
    },
    "metadata_fields": [
        "original_snr",
        "enhanced_snr",
        "snr_improvement",
        "enhancement_level",
        "processing_time_ms",
        "engine_used",
        "noise_types"
    ]
}

# Audio Quality Enhancement to 35dB SNR Configuration
ENHANCEMENT_35DB_CONFIG = {
    "enabled": False,  # Default disabled, enable with --enable-35db-enhancement
    "target_snr_db": 35.0,
    "min_acceptable_snr_db": 30.0,
    "target_success_rate": 0.90,  # 90% of samples should achieve target
    "max_enhancement_passes": 3,
    "naturalness_weights": {
        "preserve_harmonics": 0.8,
        "suppress_noise": 0.2
    },
    "perceptual_limits": {
        "min_pesq": 3.5,
        "min_stoi": 0.85,
        "max_spectral_distortion": 0.15
    },
    "quality_thresholds": {
        "min_naturalness": 0.85,
        "max_spectral_distortion": 0.15,
        "min_harmonic_preservation": 0.85
    },
    "processing": {
        "max_iterations": 3,
        "batch_size": 32,
        "use_gpu": True,
        "max_processing_time": 1.8,  # seconds per sample
        "gpu_batch_size": 32
    },
    "fallback": {
        "include_failed": True,  # Include samples that don't reach 35dB
        "min_acceptable_snr": 25.0
    }
}
