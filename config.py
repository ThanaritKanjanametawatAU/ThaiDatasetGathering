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
    "VistecCommonVoiceTH": {
        "name": "VistecCommonVoiceTH",
        "source": "vistec-AI/commonvoice-th",
        "description": "VISTEC Common Voice TH dataset",
        "processor_class": "VistecCommonVoiceTHProcessor"
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
    "Language": str,    # Always "th" for Thai
    "audio": dict,      # Audio data in HuggingFace format (array, sampling_rate, path)
    "transcript": str,  # Transcript text
    "length": float,    # Audio length in seconds
}

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
        "required": False,  # Can be empty
        "max_length": 10000,  # Reasonable maximum length
        "error_message": "Transcript exceeds maximum length"
    },
    "length": {
        "required": True,
        "min_value": 0.01,  # At least 0.01 seconds (very permissive)
        "max_value": 3600,  # Maximum 1 hour
        "error_message": "Length must be between 0.01 and 3600 seconds"
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
