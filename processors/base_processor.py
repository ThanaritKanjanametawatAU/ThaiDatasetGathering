"""
Base class for dataset processors with support for both cached and streaming modes.
"""

import os
import logging
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Iterator, Union, Tuple
import time
from pathlib import Path

from utils.audio import is_valid_audio, get_audio_length, standardize_audio, get_audio_info
from utils.logging import ProgressTracker, ProcessingLogger
from config import SCHEMA, VALIDATION_RULES, ErrorCategory, AUDIO_CONFIG

# Set up logger
logger = logging.getLogger(__name__)

# Import STT if available
try:
    from .stt.ensemble_stt import EnsembleSTT
    STT_AVAILABLE = True
except ImportError:
    logger.warning("STT module not available. STT features will be disabled.")
    STT_AVAILABLE = False

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
        
        # Get dataset name from config
        self.dataset_name = config.get("dataset_name", self.name)
        
        # Streaming configuration
        self.streaming_mode = config.get("streaming", False)
        self.batch_size = config.get("batch_size", 1000)
        self.shard_size = config.get("shard_size", "5GB")
        self.upload_batch_size = config.get("upload_batch_size", 10000)  # Upload after this many samples

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
        
        # Streaming checkpoint tracking
        self.streaming_checkpoint_file = None
        self.streaming_checkpoint_data = {}
        
        # STT configuration
        self.enable_stt = config.get("enable_stt", False)
        self.stt_batch_size = config.get("stt_batch_size", 16)
        self.stt_pipeline = None
        
        # Initialize STT if enabled
        if self.enable_stt and STT_AVAILABLE:
            try:
                self.logger.info("Initializing STT pipeline...")
                self.stt_pipeline = EnsembleSTT()
                self.logger.info("STT pipeline initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize STT pipeline: {e}")
                self.enable_stt = False

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
    
    @abstractmethod
    def process_streaming(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the dataset in streaming mode without downloading everything.

        Args:
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample of the dataset
            sample_size: Number of samples to process in sample mode

        Yields:
            dict: Processed sample in standard schema
        """
        pass
    
    def get_available_splits(self) -> List[str]:
        """
        Get available splits for the dataset.
        Override this method in subclasses to return actual splits.
        
        Returns:
            list: List of available split names
        """
        # Default implementation - subclasses should override
        return ["train"]
    
    def process_all_splits(self, checkpoint: Optional[str] = None, sample_mode: bool = False, 
                          sample_size: int = 5, checkpoint_interval: int = 1000) -> Iterator[Dict[str, Any]]:
        """
        Process all available splits and combine them into one stream.
        
        Args:
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample of the dataset
            sample_size: Number of samples to process in sample mode (distributed across splits)
            checkpoint_interval: Save checkpoint every N samples (default: 1000)
            
        Yields:
            dict: Processed sample in standard schema
        """
        available_splits = self.get_available_splits()
        self.logger.info(f"Processing all splits: {available_splits}")
        
        # Load checkpoint data if resuming
        checkpoint_data = None
        skip_until_split = None
        skip_count = 0
        processed_ids = []
        
        if checkpoint:
            checkpoint_data = self.load_unified_checkpoint(checkpoint)
            if checkpoint_data:
                skip_until_split = checkpoint_data.get('current_split')
                skip_count = checkpoint_data.get('samples_processed', 0)
                processed_ids = checkpoint_data.get('processed_ids', [])
                self.logger.info(f"Resuming from split '{skip_until_split}', sample {skip_count}")
        
        # In sample mode, distribute sample size across splits
        if sample_mode and len(available_splits) > 1:
            samples_per_split = max(1, sample_size // len(available_splits))
            remaining_samples = sample_size - (samples_per_split * len(available_splits))
        else:
            samples_per_split = sample_size
            remaining_samples = 0
            
        total_samples_yielded = 0
        total_samples_processed = skip_count
        should_skip = skip_until_split is not None
        
        for i, split in enumerate(available_splits):
            # Skip splits before the checkpoint split
            if should_skip and split != skip_until_split:
                self.logger.info(f"Skipping split: {split} (before checkpoint)")
                continue
            elif should_skip and split == skip_until_split:
                should_skip = False
                self.logger.info(f"Resuming at split: {split}")
            else:
                self.logger.info(f"Processing split: {split}")
            
            # Add extra samples to first split if needed
            split_sample_size = samples_per_split + (remaining_samples if i == 0 else 0)
            
            # Override the process_streaming method to handle specific split
            # This should be implemented in subclasses
            split_samples_yielded = 0
            
            try:
                # Process this split
                # Calculate how many samples to skip in this split
                split_skip_count = 0
                if checkpoint_data and split == skip_until_split:
                    split_skip_count = checkpoint_data.get('split_index', 0)
                    
                for split_sample_index, sample in enumerate(self._process_single_split(split, checkpoint if split == skip_until_split else None, 
                                                       sample_mode, split_sample_size)):
                    # Skip samples in the resumed split
                    if split == skip_until_split and split_sample_index < split_skip_count:
                        continue
                        
                    # Add metadata about original split (for internal tracking)
                    original_split = split
                    
                    # Process with STT if enabled
                    if self.enable_stt:
                        sample = self.process_sample_with_stt(sample, total_samples_yielded)
                    else:
                        # Just add the required fields
                        sample["dataset_name"] = self.dataset_name
                        sample["confidence_score"] = 1.0 if sample.get("transcript", "").strip() else 0.0
                        
                    # Note: We don't add _original_split to the output as it's not in the schema
                    yield sample
                    
                    split_samples_yielded += 1
                    total_samples_yielded += 1
                    total_samples_processed += 1
                    
                    # Track processed IDs for checkpoint
                    if 'ID' in sample:
                        processed_ids.append(sample['ID'])
                    
                    # Periodic checkpoint saving
                    if total_samples_processed > 0 and total_samples_processed % checkpoint_interval == 0:
                        self.save_unified_checkpoint(
                            samples_processed=total_samples_processed,
                            current_split=split,
                            split_index=split_samples_yielded,
                            processed_ids=processed_ids[-checkpoint_interval:],  # Keep last N IDs
                            dataset_specific_data={"total_samples_yielded": total_samples_yielded}
                        )
                        self.logger.info(f"Periodic checkpoint saved at {total_samples_processed} samples")
                    
                    # Check if we've reached sample limit
                    if sample_mode and total_samples_yielded >= sample_size:
                        self.logger.info(f"Reached sample limit of {sample_size}")
                        # Save final checkpoint
                        self.save_unified_checkpoint(
                            samples_processed=total_samples_processed,
                            current_split=split,
                            split_index=split_samples_yielded,
                            processed_ids=processed_ids[-checkpoint_interval:],
                            dataset_specific_data={"total_samples_yielded": total_samples_yielded}
                        )
                        return
                        
            except Exception as e:
                self.logger.error(f"Error processing split {split}: {e}")
                # Save checkpoint on error
                self.save_unified_checkpoint(
                    samples_processed=total_samples_processed,
                    current_split=split,
                    split_index=split_samples_yielded,
                    processed_ids=processed_ids[-checkpoint_interval:],
                    dataset_specific_data={"total_samples_yielded": total_samples_yielded, "error": str(e)}
                )
                if not sample_mode:
                    raise
                    
        self.logger.info(f"Processed {total_samples_yielded} samples from all splits")
        
        # Save final checkpoint
        if total_samples_yielded > 0:
            self.save_unified_checkpoint(
                samples_processed=total_samples_processed,
                current_split=available_splits[-1] if available_splits else None,
                split_index=split_samples_yielded,
                processed_ids=processed_ids[-checkpoint_interval:],
                dataset_specific_data={"total_samples_yielded": total_samples_yielded, "completed": True}
            )
    
    def _process_single_split(self, split: str, checkpoint: Optional[str] = None, 
                             sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process a single split. Override in subclasses to implement split-specific logic.
        
        Args:
            split: Split name to process
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample
            sample_size: Number of samples to process
            
        Yields:
            dict: Processed sample in standard schema
        """
        # Default implementation - use existing process_streaming
        # Subclasses should override to handle split parameter
        if self.streaming_mode:
            yield from self.process_streaming(checkpoint, sample_mode, sample_size)
        else:
            yield from self.process(checkpoint, sample_mode, sample_size)

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

    def create_hf_audio_format(self, audio_data: bytes, sample_id: str = "unknown") -> Dict[str, Any]:
        """
        Convert audio bytes to HuggingFace Audio-compatible format.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_id: Sample identifier for path generation
            
        Returns:
            dict: Audio data in HuggingFace format with array, sampling_rate, and path
        """
        try:
            import soundfile as sf
            import io
            import numpy as np
            
            # Read audio data to get array and sampling rate
            buffer = io.BytesIO(audio_data)
            array, sampling_rate = sf.read(buffer)
            
            # Ensure array is float32 for consistency
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            
            # Create HuggingFace Audio-compatible format
            return {
                "array": array,
                "sampling_rate": int(sampling_rate),
                "path": f"{sample_id}.wav"  # Virtual path for identification
            }
            
        except Exception as e:
            self.logger.error(f"Failed to convert audio to HuggingFace format for {sample_id}: {str(e)}")
            # Fallback: return a minimal format that won't break the dataset
            return {
                "array": [],
                "sampling_rate": 16000,
                "path": f"{sample_id}.wav"
            }

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

        # Ensure timestamp is set
        if "timestamp" not in checkpoint_data:
            checkpoint_data["timestamp"] = int(time.time())

        # Add metadata
        checkpoint_data["metadata"] = {
            "processor": self.name,
            "timestamp": checkpoint_data["timestamp"],
            "version": "2.0"  # Updated version for unified format
        }

        # Save to file
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        self.logger.info(f"Saved checkpoint to {checkpoint_file}")
        return checkpoint_file
    
    def save_streaming_checkpoint(self, shard_num: int, samples_processed: int, 
                                 last_sample_id: str, dataset_specific_data: Optional[Dict] = None) -> str:
        """
        Save streaming checkpoint data.

        Args:
            shard_num: Current shard number
            samples_processed: Total number of samples processed
            last_sample_id: ID of the last processed sample
            dataset_specific_data: Any dataset-specific checkpoint data

        Returns:
            str: Path to checkpoint file
        """
        checkpoint_data = {
            "mode": "streaming",
            "shard_num": shard_num,
            "samples_processed": samples_processed,
            "last_sample_id": last_sample_id,
            "dataset_specific": dataset_specific_data or {},
            "timestamp": int(time.time()),
            "processor": self.name
        }
        
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.name}_streaming_checkpoint.json")
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Saved streaming checkpoint: shard {shard_num}, {samples_processed} samples")
        return checkpoint_file
    
    def load_streaming_checkpoint(self, checkpoint_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load streaming checkpoint data.

        Args:
            checkpoint_file: Path to checkpoint file (optional)

        Returns:
            dict: Checkpoint data or None if not found
        """
        if checkpoint_file is None:
            checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.name}_streaming_checkpoint.json")
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Handle both old and new checkpoint formats
            if 'shard_num' in data:
                self.logger.info(f"Loaded streaming checkpoint: shard {data['shard_num']}, {data['samples_processed']} samples")
            else:
                # Convert old format to new format
                samples_processed = data.get('samples_processed', data.get('processed_count', 0))
                self.logger.info(f"Loaded checkpoint: {samples_processed} samples processed")
                # Add missing fields for compatibility
                data['shard_num'] = 0  # Default shard number
                data['samples_processed'] = samples_processed
            
            return data
        
        return None
    
    def save_unified_checkpoint(self, samples_processed: int, current_split: Optional[str] = None,
                              split_index: Optional[int] = None, shard_num: Optional[int] = None,
                              last_sample_id: Optional[str] = None, processed_ids: Optional[List[str]] = None,
                              dataset_specific_data: Optional[Dict] = None) -> str:
        """
        Save a unified checkpoint that works for both regular and streaming modes.
        
        Args:
            samples_processed: Total number of samples processed
            current_split: Current split being processed (e.g., "train", "test")
            split_index: Index within the current split
            shard_num: Current shard number (for streaming mode)
            last_sample_id: ID of the last processed sample
            processed_ids: List of processed sample IDs (for regular mode)
            dataset_specific_data: Any dataset-specific checkpoint data
            
        Returns:
            str: Path to checkpoint file
        """
        checkpoint_data = {
            "mode": "unified",
            "samples_processed": samples_processed,
            "current_split": current_split,
            "split_index": split_index,
            "shard_num": shard_num or 0,
            "last_sample_id": last_sample_id,
            "processed_ids": processed_ids or [],
            "dataset_specific": dataset_specific_data or {},
            "timestamp": int(time.time()),
            "processor": self.name
        }
        
        # Use consistent filename for easy resume
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.name}_unified_checkpoint.json")
        
        # Also save with timestamp for history
        timestamp_file = os.path.join(self.checkpoint_dir, f"{self.name}_{int(time.time())}.json")
        
        # Save both files
        for file_path in [checkpoint_file, timestamp_file]:
            with open(file_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Saved unified checkpoint: {samples_processed} samples, split: {current_split}")
        return checkpoint_file
    
    def load_unified_checkpoint(self, checkpoint_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load unified checkpoint data, compatible with both old and new formats.
        
        Args:
            checkpoint_file: Path to checkpoint file (optional)
            
        Returns:
            dict: Checkpoint data or None if not found
        """
        # Try multiple checkpoint locations
        if checkpoint_file is None:
            possible_files = [
                os.path.join(self.checkpoint_dir, f"{self.name}_unified_checkpoint.json"),
                os.path.join(self.checkpoint_dir, f"{self.name}_streaming_checkpoint.json"),
            ]
            
            # Find the most recent checkpoint
            for file_path in possible_files:
                if os.path.exists(file_path):
                    checkpoint_file = file_path
                    break
                    
            if checkpoint_file is None:
                # Look for timestamped checkpoints
                checkpoint_file = self.get_latest_checkpoint()
                
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Convert old formats to unified format
            if data.get('mode') != 'unified':
                unified_data = {
                    "mode": "unified",
                    "samples_processed": data.get('samples_processed', data.get('processed_count', 0)),
                    "current_split": data.get('current_split', 'train'),
                    "split_index": data.get('split_index', data.get('current_index', 0)),
                    "shard_num": data.get('shard_num', 0),
                    "last_sample_id": data.get('last_sample_id', None),
                    "processed_ids": data.get('processed_ids', []),
                    "dataset_specific": data.get('dataset_specific', {}),
                    "timestamp": data.get('timestamp', int(time.time())),
                    "processor": data.get('processor', self.name)
                }
                data = unified_data
            
            self.logger.info(f"Loaded unified checkpoint: {data['samples_processed']} samples, split: {data.get('current_split')}")
            return data
        
        return None
    
    # Helper methods for streaming mode to reduce code duplication
    
    def _initialize_streaming_state(self, checkpoint: Optional[str] = None) -> tuple:
        """
        Initialize streaming state from checkpoint if available.
        
        Args:
            checkpoint: Path to checkpoint file (optional)
            
        Returns:
            tuple: (checkpoint_data, skip_count)
        """
        checkpoint_data = None
        skip_count = 0
        
        if checkpoint:
            # Use unified checkpoint loading which handles all formats
            checkpoint_data = self.load_unified_checkpoint(checkpoint)
            if checkpoint_data:
                skip_count = checkpoint_data.get('samples_processed', 0)
                self.logger.info(f"Resuming streaming from sample {skip_count}")
                
        return checkpoint_data, skip_count
    
    def _should_skip_sample(self, samples_processed: int, skip_count: int, 
                           samples_yielded: int, sample_mode: bool, sample_size: int) -> tuple:
        """
        Check if current sample should be skipped.
        
        Args:
            samples_processed: Total samples processed so far
            skip_count: Number of samples to skip for checkpoint resume
            samples_yielded: Number of samples yielded so far
            sample_mode: Whether in sample mode
            sample_size: Target sample size
            
        Returns:
            tuple: (should_skip, should_break)
        """
        # Skip samples before checkpoint
        if samples_processed < skip_count:
            return True, False
            
        # Check sample mode limit
        if sample_mode and samples_yielded >= sample_size:
            return False, True
            
        return False, False
    
    def _extract_audio_bytes(self, audio_data: Any) -> Optional[bytes]:
        """
        Extract audio bytes from various audio data formats.
        
        Args:
            audio_data: Audio data in various formats
            
        Returns:
            bytes or None: Extracted audio bytes
        """
        if not audio_data:
            self.logger.warning("No audio data provided")
            return None
            
        # Handle direct bytes
        if isinstance(audio_data, (bytes, bytearray)):
            return bytes(audio_data)
            
        # Handle HuggingFace audio format
        if isinstance(audio_data, dict):
            audio_bytes = audio_data.get('bytes')
            if audio_bytes:
                return audio_bytes
                
            # Convert array to bytes if needed
            if 'array' in audio_data:
                try:
                    import soundfile as sf
                    import io
                    import numpy as np
                    
                    array = audio_data['array']
                    if isinstance(array, list):
                        array = np.array(array, dtype=np.float32)
                        
                    buffer = io.BytesIO()
                    sampling_rate = audio_data.get('sampling_rate', 16000)
                    sf.write(buffer, array, sampling_rate, format='wav')
                    buffer.seek(0)
                    return buffer.read()
                except Exception as e:
                    self.logger.error(f"Failed to convert audio array to bytes: {e}")
                    return None
                    
        self.logger.warning(f"Unknown audio data format: {type(audio_data)}")
        return None
    
    def _process_audio_for_streaming(self, audio_bytes: bytes, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Process audio bytes for streaming: validate, preprocess, and convert to HF format.
        
        Args:
            audio_bytes: Raw audio bytes
            sample_id: Sample identifier for logging
            
        Returns:
            dict or None: HuggingFace audio format dict or None if invalid
        """
        # Validate audio
        if not is_valid_audio(audio_bytes):
            self.logger.warning(f"Invalid audio format for {sample_id}")
            return None
            
        # Apply audio preprocessing if enabled
        if self.audio_config.get("enable_standardization", True):
            try:
                audio_bytes = self.preprocess_audio(audio_bytes, sample_id)
            except Exception as e:
                self.logger.error(f"Audio preprocessing failed for {sample_id}: {e}")
                return None
                
        # Convert to HuggingFace format
        try:
            return self.create_hf_audio_format(audio_bytes, sample_id)
        except Exception as e:
            self.logger.error(f"Failed to create HF audio format for {sample_id}: {e}")
            return None
    
    def _create_streaming_sample(self, audio_hf: Dict[str, Any], transcript: str, 
                                samples_processed: int, language: str = "th",
                                speaker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a sample in standard schema for streaming.
        
        Args:
            audio_hf: Audio in HuggingFace format
            transcript: Text transcript
            samples_processed: Sample count for ID generation
            language: Language code
            speaker_id: Optional speaker ID
            
        Returns:
            dict: Sample in standard schema
        """
        # Calculate audio length
        length = get_audio_length(audio_hf)
        if length is None:
            self.logger.warning("Could not calculate audio length, using default")
            length = 1.0
            
        # Generate speaker_id if not provided
        if speaker_id is None:
            # Generate a default speaker_id for this processor/dataset
            # Use a hash of dataset name and sample index to create deterministic IDs
            import hashlib
            speaker_hash = hashlib.md5(f"{self.dataset_name}_{samples_processed}".encode()).hexdigest()
            speaker_num = int(speaker_hash[:8], 16) % 100000  # Convert to number 0-99999
            speaker_id = f"SPK_{speaker_num:05d}"
            
        return {
            "ID": f"temp_{samples_processed}",  # Will be assigned sequential ID later
            "speaker_id": speaker_id,
            "Language": language,
            "audio": audio_hf,
            "transcript": transcript,
            "length": length,
            "dataset_name": self.dataset_name,  # NEW FIELD
            "confidence_score": 1.0  # Will be updated by process_sample_with_stt
        }
    
    def process_sample_with_stt(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """
        Process a sample with STT if needed to ensure 100% transcript coverage.
        
        Args:
            sample: Sample to process
            sample_idx: Sample index for logging
            
        Returns:
            dict: Processed sample with guaranteed transcript
        """
        # Add dataset name
        sample["dataset_name"] = self.dataset_name
        
        # Check if transcript exists and is non-empty
        transcript = sample.get("transcript", "").strip()
        
        if transcript:
            # Existing transcript - set confidence to 1.0
            sample["confidence_score"] = 1.0
            return sample
            
        # No transcript - need STT
        if not self.enable_stt or not self.stt_pipeline:
            # STT not available - use fallback
            self.logger.warning(f"No transcript for sample {sample_idx} and STT not available")
            sample["transcript"] = "[NO_TRANSCRIPT]"
            sample["confidence_score"] = 0.0
            return sample
            
        # Extract audio data for STT
        try:
            audio_bytes = self._extract_audio_bytes(sample.get("audio"))
            if not audio_bytes:
                self.logger.error(f"No audio data for sample {sample_idx}")
                sample["transcript"] = "[NO_AUDIO]"
                sample["confidence_score"] = 0.0
                return sample
                
            # Convert to numpy array for STT
            import soundfile as sf
            import io
            import numpy as np
            
            buffer = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(buffer)
            
            # Ensure 16kHz
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                
            # Run STT
            transcript, confidence = self.stt_pipeline.transcribe(audio_array)
            
            # Ensure non-empty transcript
            if not transcript.strip():
                # Try fallback
                transcript, confidence = self.stt_pipeline.transcribe_with_fallback(audio_array)
                
            sample["transcript"] = transcript
            sample["confidence_score"] = confidence
            
            self.logger.info(f"STT generated transcript for sample {sample_idx}: '{transcript[:50]}...' (conf: {confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"STT failed for sample {sample_idx}: {e}")
            sample["transcript"] = "[STT_ERROR]"
            sample["confidence_score"] = 0.0
            
        return sample
    
    def process_batch_with_stt(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of samples with STT for efficiency.
        
        Args:
            batch: List of samples to process
            
        Returns:
            list: Processed samples with guaranteed transcripts
        """
        # Separate samples needing STT
        need_stt = []
        have_transcript = []
        
        for i, sample in enumerate(batch):
            # Add dataset name
            sample["dataset_name"] = self.dataset_name
            
            if not sample.get("transcript") or not sample["transcript"].strip():
                need_stt.append((i, sample))
            else:
                sample["confidence_score"] = 1.0
                have_transcript.append((i, sample))
                
        # Process STT batch if needed
        if need_stt and self.enable_stt and self.stt_pipeline:
            self.logger.info(f"Running STT on {len(need_stt)} samples")
            
            # Extract audio for batch
            audio_batch = []
            for idx, sample in need_stt:
                try:
                    audio_bytes = self._extract_audio_bytes(sample.get("audio"))
                    if audio_bytes:
                        import soundfile as sf
                        import io
                        buffer = io.BytesIO(audio_bytes)
                        audio_array, sr = sf.read(buffer)
                        
                        # Ensure 16kHz
                        if sr != 16000:
                            import librosa
                            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                            
                        audio_batch.append(audio_array)
                    else:
                        audio_batch.append(None)
                except Exception as e:
                    self.logger.error(f"Failed to extract audio for sample {idx}: {e}")
                    audio_batch.append(None)
                    
            # Run batch STT
            try:
                results = self.stt_pipeline.transcribe_batch([a for a in audio_batch if a is not None])
                result_idx = 0
                
                for i, (idx, sample) in enumerate(need_stt):
                    if audio_batch[i] is not None:
                        transcript, confidence = results[result_idx]
                        result_idx += 1
                        
                        # Ensure non-empty transcript
                        if transcript.strip():
                            sample["transcript"] = transcript
                            sample["confidence_score"] = confidence
                        else:
                            sample["transcript"] = "[INAUDIBLE]"
                            sample["confidence_score"] = 0.1
                    else:
                        sample["transcript"] = "[NO_AUDIO]"
                        sample["confidence_score"] = 0.0
                        
                    have_transcript.append((idx, sample))
                    
            except Exception as e:
                self.logger.error(f"Batch STT failed: {e}")
                # Fallback for failed batch
                for idx, sample in need_stt:
                    sample["transcript"] = "[STT_ERROR]"
                    sample["confidence_score"] = 0.0
                    have_transcript.append((idx, sample))
        else:
            # No STT available - use fallbacks
            for idx, sample in need_stt:
                sample["transcript"] = "[NO_TRANSCRIPT]"
                sample["confidence_score"] = 0.0
                have_transcript.append((idx, sample))
                
        # Sort by original index and return
        have_transcript.sort(key=lambda x: x[0])
        return [sample for _, sample in have_transcript]
    
    def _log_streaming_progress(self, samples_processed: int, interval: int = 100) -> None:
        """
        Log streaming progress at regular intervals.
        
        Args:
            samples_processed: Number of samples processed
            interval: Logging interval
        """
        if samples_processed % interval == 0 and samples_processed > 0:
            self.logger.info(f"Processed {samples_processed} samples in streaming mode")

    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data from file.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            dict: Checkpoint data or None if loading failed
        """
        try:
            # Use unified checkpoint loading which handles all formats
            checkpoint_data = self.load_unified_checkpoint(checkpoint_file)
            
            if checkpoint_data:
                # Validate processor if metadata exists
                if "metadata" in checkpoint_data:
                    processor_name = checkpoint_data["metadata"].get("processor")
                    if processor_name and processor_name != self.name:
                        self.logger.error(f"Checkpoint is for different processor: {processor_name}")
                        return None
                        
                self.logger.info(f"Loaded checkpoint from {checkpoint_file}")
                return checkpoint_data
            else:
                self.logger.error(f"Failed to load checkpoint from {checkpoint_file}")
                return None
                
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
