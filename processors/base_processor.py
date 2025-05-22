"""
Base class for dataset processors.
"""

import os
import logging
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Iterator, Union, Tuple
import time

from utils.audio import is_valid_audio, get_audio_length, standardize_audio, get_audio_info
from utils.logging import ProgressTracker, ProcessingLogger
from config import SCHEMA, VALIDATION_RULES, ErrorCategory, AUDIO_CONFIG

# Set up logger
logger = logging.getLogger(__name__)

class DatasetProcessingError(Exception):
    """Base exception for all dataset processing errors."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, retriable: bool = False):
        self.message = message
        self.category = category
        self.retriable = retriable
        super().__init__(self.message)

class NetworkError(DatasetProcessingError):
    """Exception for network-related errors."""

    def __init__(self, message: str, retriable: bool = True):
        super().__init__(message, ErrorCategory.NETWORK, retriable)

class FileAccessError(DatasetProcessingError):
    """Exception for file access errors."""

    def __init__(self, message: str, retriable: bool = False):
        super().__init__(message, ErrorCategory.FILE_ACCESS, retriable)

class ValidationError(DatasetProcessingError):
    """Exception for validation errors."""

    def __init__(self, message: str, retriable: bool = False):
        super().__init__(message, ErrorCategory.VALIDATION, retriable)

class BaseProcessor(ABC):
    """
    Base class for all dataset processors.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.log_dir = config.get("log_dir", "logs")

        # Audio configuration - merge with global config, prioritizing processor-specific config
        self.audio_config = {**AUDIO_CONFIG}
        if "audio_config" in config:
            self.audio_config.update(config["audio_config"])

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set up logger
        self.logger = logging.getLogger(f"processor.{self.name}")

        # Set up processing logger
        self.processing_logger = ProcessingLogger(self.log_dir, self.name)

        # Initialize progress tracker (will be updated in process method)
        self.progress_tracker = None

    @abstractmethod
    def process(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the dataset and yield samples in the standard schema.

        Args:
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample of the dataset
            sample_size: Number of samples to process in sample mode

        Yields:
            dict: Processed sample in standard schema
        """
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return information about the dataset.

        Returns:
            dict: Dataset information
        """
        pass

    @abstractmethod
    def estimate_size(self) -> int:
        """
        Estimate the size of the dataset.

        Returns:
            int: Estimated number of samples
        """
        pass

    def validate_sample(self, sample: Dict[str, Any]) -> List[str]:
        """
        Validate a sample against the standard schema.

        Args:
            sample: Sample to validate

        Returns:
            list: List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields and types
        for field, field_type in SCHEMA.items():
            rules = VALIDATION_RULES.get(field, {})

            # Check if field exists and is required
            if field not in sample:
                if rules.get("required", False):
                    errors.append(f"Missing required field: {field}")
                continue

            # Check field type
            if not isinstance(sample[field], field_type):
                errors.append(f"Field {field} has wrong type: expected {field_type.__name__}, got {type(sample[field]).__name__}")
                continue

            # Apply specific validation rules
            if "allowed_values" in rules and sample[field] not in rules["allowed_values"]:
                errors.append(f"{rules.get('error_message', f'Invalid value for {field}')}")

            if "pattern" in rules and isinstance(sample[field], str):
                pattern = rules["pattern"]
                if not re.match(pattern, sample[field]):
                    errors.append(f"{rules.get('error_message', f'Invalid pattern for {field}')}")

            if "min_size" in rules and hasattr(sample[field], "__len__"):
                min_size = rules["min_size"]
                if len(sample[field]) < min_size:
                    errors.append(f"{rules.get('error_message', f'Field {field} is too small')}")

            if "max_length" in rules and isinstance(sample[field], str):
                max_length = rules["max_length"]
                if len(sample[field]) > max_length:
                    errors.append(f"{rules.get('error_message', f'Field {field} is too long')}")

            if "min_value" in rules and isinstance(sample[field], (int, float)):
                min_value = rules["min_value"]
                if sample[field] < min_value:
                    errors.append(f"{rules.get('error_message', f'Field {field} is too small')}")

            if "max_value" in rules and isinstance(sample[field], (int, float)):
                max_value = rules["max_value"]
                if sample[field] > max_value:
                    errors.append(f"{rules.get('error_message', f'Field {field} is too large')}")

            if "validate_func" in rules and rules["validate_func"] == "is_valid_audio":
                if not is_valid_audio(sample[field]):
                    errors.append(f"{rules.get('error_message', f'Invalid audio data')}")

        return errors

    def preprocess_audio(self, audio_data: bytes, sample_id: str = "unknown") -> bytes:
        """
        Apply audio preprocessing and standardization.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_id: Sample identifier for logging
            
        Returns:
            bytes: Processed audio data
        """
        # Skip preprocessing if disabled
        if not self.audio_config.get("enable_standardization", True):
            self.logger.debug(f"Audio standardization disabled for {sample_id}")
            return audio_data
            
        try:
            self.logger.debug(f"Applying audio standardization for {sample_id}")
            
            # Log original audio info
            original_info = get_audio_info(audio_data)
            if original_info:
                self.logger.debug(f"Original audio - SR: {original_info['sample_rate']}Hz, "
                                f"Length: {original_info['length']:.2f}s, "
                                f"dB: {original_info['db_level']:.1f}dB")
            
            # Apply standardization
            standardized_audio = standardize_audio(
                audio_data,
                target_sample_rate=self.audio_config.get("target_sample_rate", 16000),
                target_channels=self.audio_config.get("target_channels", 1),
                normalize_volume=self.audio_config.get("normalize_volume", True),
                target_db=self.audio_config.get("target_db", -20.0)
            )
            
            if standardized_audio is not None:
                self.logger.debug("Audio standardization successful")
                
                # Log standardized audio info
                standardized_info = get_audio_info(standardized_audio)
                if standardized_info:
                    self.logger.debug(f"Standardized audio - SR: {standardized_info['sample_rate']}Hz, "
                                    f"Length: {standardized_info['length']:.2f}s, "
                                    f"dB: {standardized_info['db_level']:.1f}dB")
                return standardized_audio
            else:
                self.logger.warning(f"Audio standardization failed for {sample_id}, using original audio")
                return audio_data
                
        except Exception as e:
            self.logger.error(f"Error during audio preprocessing for {sample_id}: {str(e)}")
            return audio_data

    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_file: Optional[str] = None) -> str:
        """
        Save checkpoint data to file.

        Args:
            checkpoint_data: Checkpoint data
            checkpoint_file: Path to checkpoint file (optional)

        Returns:
            str: Path to checkpoint file
        """
        if checkpoint_file is None:
            timestamp = int(time.time())
            checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.name}_{timestamp}.json")

        # Add metadata
        checkpoint_data["metadata"] = {
            "processor": self.name,
            "timestamp": timestamp,
            "version": "1.0"
        }

        # Save to file
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        self.logger.info(f"Saved checkpoint to {checkpoint_file}")
        return checkpoint_file

    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data from file.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            dict: Checkpoint data or None if loading failed
        """
        try:
            if not os.path.exists(checkpoint_file):
                self.logger.error(f"Checkpoint file not found: {checkpoint_file}")
                return None

            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Validate checkpoint
            if "metadata" not in checkpoint_data or checkpoint_data["metadata"].get("processor") != self.name:
                self.logger.error(f"Invalid checkpoint file: {checkpoint_file}")
                return None

            self.logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return None

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the latest checkpoint file for this processor.

        Returns:
            str: Path to latest checkpoint file or None if not found
        """
        try:
            # Get all checkpoint files for this processor
            checkpoint_files = [
                os.path.join(self.checkpoint_dir, f)
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith(f"{self.name}_") and f.endswith(".json")
            ]

            if not checkpoint_files:
                return None

            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

            return checkpoint_files[0]
        except Exception as e:
            self.logger.error(f"Error getting latest checkpoint: {str(e)}")
            return None

    def generate_id(self, current_index: int) -> str:
        """
        Generate sequential ID in format S{n}.

        Args:
            current_index: Current index

        Returns:
            str: Generated ID
        """
        return f"S{current_index}"

    @staticmethod
    def extract_id_number(id_str: str) -> Optional[int]:
        """
        Extract numeric part from ID string.

        Args:
            id_str: ID string

        Returns:
            int: Numeric part or None if invalid
        """
        if id_str and id_str.startswith('S') and id_str[1:].isdigit():
            return int(id_str[1:])
        return None

    @staticmethod
    def get_next_id(existing_ids: List[str]) -> str:
        """
        Get next available ID based on existing IDs.

        Args:
            existing_ids: List of existing IDs

        Returns:
            str: Next available ID
        """
        if not existing_ids:
            return "S1"

        # Extract numeric parts
        numeric_ids = [BaseProcessor.extract_id_number(id_str) for id_str in existing_ids]
        numeric_ids = [n for n in numeric_ids if n is not None]

        if not numeric_ids:
            return "S1"

        # Get max and add 1
        next_num = max(numeric_ids) + 1
        return f"S{next_num}"
