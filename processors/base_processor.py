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

# Import Audio Enhancement if available
try:
    from .audio_enhancement import AudioEnhancer
    ENHANCEMENT_AVAILABLE = True
except ImportError:
    logger.warning("Audio enhancement module not available. Enhancement features will be disabled.")
    ENHANCEMENT_AVAILABLE = False

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
        
        # Noise reduction configuration (backward compatibility)
        self.noise_reduction_enabled = config.get("enable_noise_reduction", False)
        self.noise_reduction_config = config.get("noise_reduction_config", {})
        
        # Audio enhancement configuration (new approach)
        self.audio_enhancement = config.get("audio_enhancement", {})
        if self.audio_enhancement.get("enabled", False):
            # Override old noise reduction config
            self.noise_reduction_enabled = True
            self.audio_enhancer = self.audio_enhancement.get("enhancer")
            self.enhancement_level = self.audio_enhancement.get("level")
            self.enhancement_batch_size = self.audio_enhancement.get("batch_size", 10)
            self.enhancement_metrics_collector = self.audio_enhancement.get("metrics_collector")
            self.enhancement_dashboard = self.audio_enhancement.get("dashboard")
        else:
            self.audio_enhancer = None
            self.enhancement_level = None
            
        # NEW: Pattern→MetricGAN+ configuration
        self.pattern_metricgan_config = config.get("pattern_metricgan_config", {})
        self.enhancement_level = config.get("enhancement_level", "moderate")
        
        # Initialize Pattern→MetricGAN+ if enabled
        if self.enhancement_level == "pattern_metricgan_plus":
            self._initialize_pattern_metricgan_enhancement()
            
        self.enhancement_stats = {
            "total_enhanced": 0,
            "enhancement_failures": 0,
            "average_snr_improvement": 0.0,
            "average_processing_time": 0.0,
            "noise_types_found": {}
        }
        
        # Initialize audio enhancer if enabled (old way)
        if self.noise_reduction_enabled and not self.audio_enhancer:
            self._initialize_audio_enhancer()
        self.streaming_checkpoint_data = {}
        
        # STT configuration
        self.enable_stt = config.get("enable_stt", False)
        self.stt_batch_size = config.get("stt_batch_size", 16)
        self.stt_pipeline = None
        
        # Note: Audio enhancement is initialized via _initialize_audio_enhancer() if noise_reduction_enabled is True
        
        # Initialize STT if enabled
        if self.enable_stt and STT_AVAILABLE:
            try:
                self.logger.info("Initializing STT pipeline...")
                self.stt_pipeline = EnsembleSTT()
                self.logger.info("STT pipeline initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize STT pipeline: {e}")
                self.enable_stt = False

                
    def _apply_noise_reduction_with_metadata(self, audio_data: bytes, sample_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """
        Apply noise reduction to audio data and return metadata.
        
        Args:
            audio_data: Audio data as bytes
            sample_id: Sample identifier for logging
            
        Returns:
            Tuple of (enhanced_audio_bytes, metadata) or None if enhancement failed
        """
        if not self.audio_enhancer:
            return None
            
        try:
            import soundfile as sf
            import io
            import numpy as np
            
            # Convert bytes to numpy array
            buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(buffer)
            
            # Ensure audio is float32 (AudioEnhancer doesn't support float16)
            if audio_array.dtype == np.float16:
                audio_array = audio_array.astype(np.float32)
            
            # Apply enhancement with Pattern→MetricGAN+ configuration
            enhance_kwargs = {}
            if self.enhancement_level == "pattern_metricgan_plus":
                enhance_kwargs.update({
                    'pattern_metricgan_config': self.pattern_metricgan_config,
                    'enhancement_level': 'pattern_metricgan_plus',
                    'return_detailed_metadata': True
                })
            
            # Apply enhancement
            enhanced_array, metadata = self.audio_enhancer.enhance(
                audio_array, sample_rate, return_metadata=True, **enhance_kwargs
            )
            
            # Add Pattern→MetricGAN+ specific metadata
            if self.enhancement_level == "pattern_metricgan_plus":
                metadata.update({
                    'enhancement_method': 'pattern_metricgan_plus',
                    'pattern_detection_used': True,
                    'metricgan_applied': metadata.get('metricgan_applied', False),
                    'loudness_enhanced': metadata.get('loudness_enhanced', False),
                    'configuration_source': 'cli' if hasattr(self, '_from_cli') else 'config'
                })
            
            # Update statistics
            # TODO: Implement _update_enhancement_stats method
            # self._update_enhancement_stats(metadata)
            
            # Convert back to bytes
            enhanced_buffer = io.BytesIO()
            # Ensure float32 for soundfile compatibility
            if enhanced_array.dtype == np.float16:
                enhanced_array = enhanced_array.astype(np.float32)
            sf.write(enhanced_buffer, enhanced_array, sample_rate, format='WAV')
            enhanced_buffer.seek(0)
            
            # Log enhancement results
            if metadata.get('snr_improvement', 0) > 0:
                self.logger.debug(
                    f"Enhanced {sample_id}: SNR {metadata.get('snr_before', 0):.1f} → "
                    f"{metadata.get('snr_after', 0):.1f} dB (+{metadata['snr_improvement']:.1f}dB) "
                    f"in {metadata.get('processing_time', 0)*1000:.1f}ms"
                )
                
            return enhanced_buffer.read(), metadata
            
        except Exception as e:
            self.logger.error(f"Audio enhancement failed for {sample_id}: {e}")
            self.enhancement_stats['enhancement_failures'] += 1
            return None

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

    def preprocess_audio(self, audio_data: bytes, sample_id: str = "unknown") -> Tuple[bytes, Optional[Dict[str, Any]]]:
        """
        Apply audio preprocessing, enhancement, and standardization.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_id: Sample identifier for logging
            
        Returns:
            Tuple of (processed_audio_bytes, enhancement_metadata)
        """
        enhancement_metadata = None
        
        # Skip preprocessing if disabled
        if not self.audio_config.get("enable_standardization", True):
            self.logger.debug(f"Audio standardization disabled for {sample_id}")
            return audio_data, enhancement_metadata
            
        try:
            self.logger.debug(f"Applying audio preprocessing for {sample_id}")
            
            # Log original audio info
            original_info = get_audio_info(audio_data)
            if original_info:
                self.logger.debug(f"Original audio - SR: {original_info['sample_rate']}Hz, "
                                f"Length: {original_info['length']:.2f}s, "
                                f"dB: {original_info['db_level']:.1f}dB")
            
            # Apply standardization first
            standardized_audio = standardize_audio(
                audio_data,
                target_sample_rate=self.audio_config.get("target_sample_rate", 16000),
                target_channels=self.audio_config.get("target_channels", 1),
                normalize_volume=self.audio_config.get("normalize_volume", True),
                target_db=self.audio_config.get("target_db", -20.0)
            )
            
            if standardized_audio is None:
                self.logger.warning(f"Audio standardization failed for {sample_id}, using original audio")
                standardized_audio = audio_data
            
            # Apply noise reduction if enabled
            if self.noise_reduction_enabled:
                enhanced_result = self._apply_noise_reduction_with_metadata(standardized_audio, sample_id)
                if enhanced_result is not None:
                    enhanced_audio, enhancement_metadata = enhanced_result
                    standardized_audio = enhanced_audio
                    
            # Log final audio info
            final_info = get_audio_info(standardized_audio)
            if final_info:
                self.logger.debug(f"Final audio - SR: {final_info['sample_rate']}Hz, "
                                f"Length: {final_info['length']:.2f}s, "
                                f"dB: {final_info['db_level']:.1f}dB")
                
            return standardized_audio, enhancement_metadata
                
        except Exception as e:
            self.logger.error(f"Error during audio preprocessing for {sample_id}: {str(e)}")
            return audio_data, enhancement_metadata

    def _initialize_audio_enhancer(self):
        """Initialize the audio enhancement module."""
        try:
            from processors.audio_enhancement import AudioEnhancer
            from config import NOISE_REDUCTION_CONFIG
            
            device = NOISE_REDUCTION_CONFIG.get("device", "cuda")
            level = NOISE_REDUCTION_CONFIG.get("default_level", "moderate")
            adaptive_mode = NOISE_REDUCTION_CONFIG.get("adaptive_mode", True)
            
            self.audio_enhancer = AudioEnhancer(
                device=device,
                level=level,
                adaptive_mode=adaptive_mode
            )
            
            self.logger.info(f"Initialized audio enhancer on {device} with level {level}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio enhancer: {e}")
            self.noise_reduction_enabled = False
            self.audio_enhancer = None

    def _initialize_pattern_metricgan_enhancement(self):
        """Initialize Pattern→MetricGAN+ enhancement pipeline."""
        try:
            # Validate configuration
            self._validate_pattern_metricgan_config()
            
            # Set enhancement flags
            self.noise_reduction_enabled = True
            
            # Load Pattern→MetricGAN+ configuration into audio enhancement
            if not self.audio_enhancer and ENHANCEMENT_AVAILABLE:
                self._initialize_audio_enhancer()
                
            # Override enhancement level in enhancer
            if self.audio_enhancer:
                self.audio_enhancer.enhancement_level = "pattern_metricgan_plus"
                
            self.logger.info("Pattern→MetricGAN+ enhancement initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pattern→MetricGAN+ enhancement: {e}")
            # Fall back to standard enhancement
            self.enhancement_level = "moderate"
            self.pattern_metricgan_config = {}

    def _validate_pattern_metricgan_config(self):
        """Validate Pattern→MetricGAN+ configuration parameters."""
        config = self.pattern_metricgan_config
        
        # Validate pattern detection config
        pattern_detection = config.get("pattern_detection", {})
        confidence_threshold = pattern_detection.get("confidence_threshold", 0.8)
        if not 0.5 <= confidence_threshold <= 1.0:
            raise ValueError(f"Pattern confidence threshold {confidence_threshold} must be between 0.5 and 1.0")
        
        # Validate pattern suppression config
        pattern_suppression = config.get("pattern_suppression", {})
        suppression_factor = pattern_suppression.get("suppression_factor", 0.15)
        if not 0.0 <= suppression_factor <= 1.0:
            raise ValueError(f"Pattern suppression factor {suppression_factor} must be between 0.0 and 1.0")
        
        # Validate loudness enhancement config
        loudness_enhancement = config.get("loudness_enhancement", {})
        target_multiplier = loudness_enhancement.get("target_multiplier", 1.6)
        if not 1.0 <= target_multiplier <= 3.0:
            raise ValueError(f"Loudness multiplier {target_multiplier} must be between 1.0 and 3.0")
        
        self.logger.info("Pattern→MetricGAN+ configuration validated successfully")
            
    def _apply_noise_reduction(self, audio_data: bytes, sample_id: str) -> Optional[bytes]:
        """
        Apply noise reduction to audio data.
        
        Args:
            audio_data: Audio data as bytes
            sample_id: Sample identifier
            
        Returns:
            Enhanced audio data or None if failed
        """
        if not self.audio_enhancer:
            return None
            
        try:
            import soundfile as sf
            import io
            
            # Convert bytes to numpy array
            buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(buffer)
            
            # Apply enhancement
            enhanced_array, metrics = self.audio_enhancer.enhance(
                audio_array,
                sample_rate,
                auto_level=True
            )
            
            # Update stats
            self.enhancement_stats["total_enhanced"] += 1
            
            if metrics.get("enhancement_applied", False):
                # Update improvement stats
                snr_improvement = metrics.get("snr_improvement", 0)
                n = self.enhancement_stats["total_enhanced"]
                self.enhancement_stats["average_snr_improvement"] = (
                    (self.enhancement_stats["average_snr_improvement"] * (n - 1) + snr_improvement) / n
                )
                
                # Update processing time
                proc_time = metrics.get("processing_time_ms", 0)
                self.enhancement_stats["average_processing_time"] = (
                    (self.enhancement_stats["average_processing_time"] * (n - 1) + proc_time) / n
                )
                
                # Track noise types
                for noise_type in metrics.get("noise_types_detected", []):
                    self.enhancement_stats["noise_types_found"][noise_type] = (
                        self.enhancement_stats["noise_types_found"].get(noise_type, 0) + 1
                    )
                    
                # Log enhancement info
                self.logger.debug(
                    f"Enhanced {sample_id}: SNR {metrics.get('original_snr', 0):.1f} → "
                    f"{metrics.get('enhanced_snr', 0):.1f} dB "
                    f"(+{snr_improvement:.1f} dB) in {proc_time:.0f}ms"
                )
            
            # Convert back to bytes
            buffer_out = io.BytesIO()
            # Ensure float32 for soundfile compatibility
            if enhanced_array.dtype == np.float16:
                enhanced_array = enhanced_array.astype(np.float32)
            sf.write(buffer_out, enhanced_array, sample_rate, format='WAV')
            buffer_out.seek(0)
            enhanced_bytes = buffer_out.read()
            
            # Store enhancement metadata
            if hasattr(self, "_current_sample_metadata"):
                self._current_sample_metadata["enhancement_metrics"] = metrics
            
            return enhanced_bytes
            
        except Exception as e:
            self.logger.error(f"Noise reduction failed for {sample_id}: {e}")
            self.enhancement_stats["enhancement_failures"] += 1
            return None

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
                    # Ensure float32 for soundfile compatibility
                    if array.dtype == np.float16:
                        array = array.astype(np.float32)
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
                audio_bytes, enhancement_metadata = self.preprocess_audio(audio_bytes, sample_id)
                # TODO: Store enhancement_metadata if needed
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
    
    def get_available_splits(self, dataset_source: str = None, default_splits: List[str] = None) -> List[str]:
        """
        Get available splits for a dataset.
        
        Args:
            dataset_source: Dataset source identifier (optional, uses self.source if not provided)
            default_splits: Default splits to check (default: ["train", "validation", "test"])
            
        Returns:
            List[str]: Available splits
        """
        from datasets import load_dataset
        
        # Use instance source if not provided
        if dataset_source is None:
            dataset_source = getattr(self, 'source', None)
            if dataset_source is None:
                raise ValueError("No dataset source provided and self.source not set")
        
        if default_splits is None:
            default_splits = ["train", "validation", "test"]
            
        available_splits = []
        for split in default_splits:
            try:
                # Try loading with streaming to check if split exists
                dataset = load_dataset(
                    dataset_source,
                    split=split,
                    streaming=True
                )
                # Try to get first item to verify split has data
                next(iter(dataset))
                available_splits.append(split)
                self.logger.info(f"Found split: {split}")
            except (ValueError, StopIteration) as e:
                self.logger.debug(f"Split {split} not available: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error checking split {split}: {str(e)}")
        
        if not available_splits:
            self.logger.warning("No standard splits found, defaulting to 'train'")
            available_splits = ["train"]
        
        return available_splits
    
    def _process_split_streaming_generic(self, dataset_source: str, split: str, 
                                       transcript_field: str = "transcript",
                                       skip_count: int = 0, sample_limit: Optional[int] = None,
                                       prefix: str = None) -> Iterator[Dict[str, Any]]:
        """
        Generic streaming split processor to reduce code duplication.
        
        Args:
            dataset_source: Dataset source identifier
            split: Split name
            transcript_field: Field name for transcript (e.g., "sentence" or "transcript")
            skip_count: Number of samples to skip
            sample_limit: Maximum samples to process
            prefix: Prefix for sample IDs
            
        Yields:
            Dict[str, Any]: Processed samples
        """
        from datasets import load_dataset
        
        self.logger.info(f"Processing {split} split in streaming mode")
        
        if prefix is None:
            prefix = f"{self.name}_{split}"
        
        # Load dataset in streaming mode
        dataset = load_dataset(
            dataset_source,
            split=split,
            streaming=True
        )
        
        samples_processed = 0
        samples_yielded = 0
        
        for sample in dataset:
            # Skip samples if resuming
            if samples_processed < skip_count:
                samples_processed += 1
                continue
            
            # Check sample limit
            if sample_limit and samples_yielded >= sample_limit:
                self.logger.info(f"Reached sample limit of {sample_limit}")
                break
            
            try:
                # Extract audio and transcript
                audio_data = sample.get('audio', {})
                transcript = sample.get(transcript_field, '')
                
                # Skip if no audio
                if not audio_data:
                    self.logger.warning(f"Skipping sample {samples_processed} without audio")
                    samples_processed += 1
                    continue
                
                # Extract audio bytes
                audio_bytes = self._extract_audio_bytes(audio_data)
                if not audio_bytes:
                    samples_processed += 1
                    continue
                
                # Process audio
                audio_hf = self._process_audio_for_streaming(audio_bytes, f"{prefix}_{samples_processed}")
                if not audio_hf:
                    samples_processed += 1
                    continue
                
                # Get additional fields (can be overridden in subclasses)
                additional_fields = self._extract_additional_fields(sample)
                
                # Create sample
                processed_sample = self._create_streaming_sample(
                    audio_hf, transcript, 
                    samples_processed, samples_yielded,
                    additional_fields
                )
                
                # Apply STT if enabled and transcript is empty
                if self.config.get("enable_stt", False) and not processed_sample.get("transcript", "").strip():
                    processed_sample = self.process_sample_with_stt(processed_sample, samples_processed)
                
                # Validate sample
                errors = self.validate_sample(processed_sample)
                if errors:
                    self.logger.warning(f"Sample {samples_processed} validation errors: {errors}")
                    samples_processed += 1
                    continue
                
                samples_processed += 1
                samples_yielded += 1
                
                yield processed_sample
                
                # Log progress
                self._log_streaming_progress(samples_processed)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {samples_processed}: {str(e)}")
                samples_processed += 1
                continue
        
        self.logger.info(f"Completed {split} split: {samples_yielded} samples yielded out of {samples_processed} processed")
    
    def _extract_additional_fields(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional fields from sample. Override in subclasses for dataset-specific fields.
        
        Args:
            sample: Raw sample data
            
        Returns:
            Dict[str, Any]: Additional fields to include in processed sample
        """
        return {}
    
    def _save_processing_checkpoint(self, processed_count: int, current_index: int, processed_ids: set) -> None:
        """
        Save processing checkpoint.

        Args:
            processed_count: Number of processed samples
            current_index: Current index for ID generation
            processed_ids: Set of processed sample IDs
        """
        checkpoint_data = {
            "processed_count": processed_count,
            "current_index": current_index,
            "processed_ids": list(processed_ids)
        }

        self.save_checkpoint(checkpoint_data)
