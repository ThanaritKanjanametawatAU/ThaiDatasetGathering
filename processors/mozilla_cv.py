"""
Processor for the Mozilla Common Voice dataset.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Iterator
import json
import time

from processors.base_processor import BaseProcessor, NetworkError, ValidationError
from utils.audio import get_audio_length, is_valid_audio
from utils.logging import ProgressTracker
from config import ErrorCategory

# Set up logger
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    logger.error("Required datasets library not installed. Please install datasets.")
    raise

class MozillaCommonVoiceProcessor(BaseProcessor):
    """
    Processor for the Mozilla Common Voice dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Mozilla Common Voice processor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.source = config.get("source", "mozilla-foundation/common_voice_11_0")
        self.language_filter = config.get("language_filter", "th")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)
        self.split = config.get("split", "train")

    def get_available_splits(self) -> List[str]:
        """
        Get available splits for Mozilla Common Voice dataset.
        
        Returns:
            list: List of available split names
        """
        available_splits = []
        
        for split in ["train", "validation", "test"]:
            try:
                # Try to load the split info
                self.logger.debug(f"Checking if split '{split}' exists...")
                dataset_info = load_dataset(
                    self.source,
                    self.language_filter,
                    split=split,
                    streaming=True,
                    trust_remote_code=True
                )
                # If we can iterate at least once, the split exists
                for _ in dataset_info:
                    available_splits.append(split)
                    self.logger.info(f"Found split: {split}")
                    break
            except Exception as e:
                self.logger.debug(f"Split '{split}' not available: {e}")
                
        if not available_splits:
            # Default to train if no splits found
            self.logger.warning("No splits found, defaulting to 'train'")
            available_splits = ["train"]
            
        return available_splits
    
    def _process_single_split(self, split: str, checkpoint: Optional[str] = None, 
                             sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process a single split of Mozilla Common Voice dataset.
        
        Args:
            split: Split name to process
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample
            sample_size: Number of samples to process
            
        Yields:
            dict: Processed sample in standard schema
        """
        self.logger.info(f"Processing Mozilla Common Voice split: {split}")
        
        # Use the existing process method but with specific split
        if self.streaming_mode:
            yield from self._process_split_streaming(split, checkpoint, sample_mode, sample_size)
        else:
            yield from self._process_split_cached(split, checkpoint, sample_mode, sample_size)
    
    def _process_split_streaming(self, split: str, checkpoint: Optional[str] = None,
                                  sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process a specific split in streaming mode.
        """
        # Initialize streaming state
        checkpoint_data, skip_count = self._initialize_streaming_state(checkpoint)
        
        try:
            # Load dataset in streaming mode for specific split
            self.logger.info(f"Loading Mozilla Common Voice {split} split in streaming mode for language: {self.language_filter}")
            
            dataset = load_dataset(
                self.source,
                self.language_filter,
                split=split,
                streaming=True,
                trust_remote_code=True
            )
            
            # Skip to checkpoint if resuming
            samples_processed = 0
            samples_yielded = 0
            
            for sample in dataset:
                # Check if we should skip or stop
                should_skip, should_break = self._should_skip_sample(
                    samples_processed, skip_count, samples_yielded, sample_mode, sample_size
                )
                if should_skip:
                    samples_processed += 1
                    continue
                if should_break:
                    break
                
                try:
                    # Extract audio and transcript
                    audio_data = sample.get('audio', {})
                    transcript = sample.get('sentence', '')
                    
                    # Skip if no audio
                    if not audio_data:
                        self.logger.warning(f"Skipping sample without audio")
                        continue
                    
                    # Extract audio bytes
                    audio_bytes = self._extract_audio_bytes(audio_data)
                    if not audio_bytes:
                        continue
                    
                    # Process audio (validate, preprocess, convert to HF format)
                    audio_hf = self._process_audio_for_streaming(audio_bytes, f"mozilla_cv_{samples_processed}")
                    if not audio_hf:
                        continue
                    
                    # Create sample in standard schema
                    processed_sample = self._create_streaming_sample(
                        audio_hf, transcript, samples_processed
                    )
                    
                    # Validate sample
                    errors = self.validate_sample(processed_sample)
                    if errors:
                        self.logger.warning(f"Sample validation errors: {errors}")
                        continue
                    
                    samples_processed += 1
                    samples_yielded += 1
                    
                    yield processed_sample
                    
                    # Log progress
                    self._log_streaming_progress(samples_processed)
                    
                except Exception as e:
                    self.logger.error(f"Error processing sample {samples_processed}: {str(e)}")
                    continue
            
            self.logger.info(f"Completed streaming processing of {split} split: {samples_yielded} samples yielded")
            
        except Exception as e:
            self.logger.error(f"Error in streaming processing {split} split: {str(e)}")
            raise
    
    def _process_split_cached(self, split: str, checkpoint: Optional[str] = None, 
                              sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process a specific split in cached mode (original process method adapted for splits).
        """
        # Load checkpoint if provided
        checkpoint_data = None
        if checkpoint:
            checkpoint_data = self.load_checkpoint(checkpoint)

        # Initialize processing state
        processed_count = 0
        current_index = 1
        processed_ids = set()

        if checkpoint_data:
            processed_count = checkpoint_data.get("processed_count", 0)
            current_index = checkpoint_data.get("current_index", 1)
            processed_ids = set(checkpoint_data.get("processed_ids", []))

            self.logger.info(f"Resuming from checkpoint: processed_count={processed_count}, current_index={current_index}")

        # Load dataset for specific split
        try:
            # Load dataset with language filter and split
            dataset = load_dataset(self.source, self.language_filter, split=split, trust_remote_code=True)

            # Initialize progress tracker
            total_items = len(dataset)

            # If in sample mode, only process a small sample
            if sample_mode:
                self.logger.info(f"Sample mode: Processing only {sample_size} samples from Mozilla Common Voice {split} split")
                # Take a small sample from the dataset
                dataset = dataset.select(range(min(sample_size, total_items)))
                total_items = len(dataset)

            self.progress_tracker = ProgressTracker(total_items=total_items)

            self.logger.info(f"Processing Mozilla Common Voice {split} split: {total_items} samples")

            # Process samples
            for i, sample in enumerate(dataset):
                # Skip already processed samples
                sample_id = sample.get("client_id", "") + "_" + sample.get("path", str(i))
                if sample_id in processed_ids:
                    self.progress_tracker.update(items_processed=0, skipped=1)
                    continue

                try:
                    # Convert to standard schema
                    processed_sample = self._convert_sample(sample, current_index)

                    # Validate sample
                    errors = self.validate_sample(processed_sample)
                    if errors:
                        error_msg = "; ".join(errors)
                        self.logger.warning(f"Sample validation failed: {error_msg}")
                        self.processing_logger.log_sample(
                            sample_id=sample_id,
                            status="skipped",
                            details={"reason": f"Validation failed: {error_msg}"}
                        )
                        self.progress_tracker.update(items_processed=0, skipped=1)
                        continue

                    # Log processing
                    self.processing_logger.log_sample(
                        sample_id=sample_id,
                        status="processed",
                        details={"output_id": processed_sample["ID"]}
                    )

                    # Update state
                    processed_count += 1
                    current_index += 1
                    processed_ids.add(sample_id)

                    # Update progress
                    self.progress_tracker.update()

                    # Save checkpoint periodically
                    if processed_count % 1000 == 0:
                        self._save_processing_checkpoint(processed_count, current_index, processed_ids)

                    yield processed_sample

                except Exception as e:
                    self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
                    self.processing_logger.log_sample(
                        sample_id=sample_id,
                        status="error",
                        details={"error": str(e)}
                    )
                    self.progress_tracker.update(items_processed=0, errors=1)

            # Save final checkpoint
            self._save_processing_checkpoint(processed_count, current_index, processed_ids)

            # Log summary
            self.progress_tracker.log_summary()
            self.processing_logger.save_json_log()

        except Exception as e:
            self.logger.error(f"Error processing Mozilla Common Voice {split} split: {str(e)}")
            raise NetworkError(f"Failed to load Mozilla Common Voice dataset: {str(e)}")

    def process(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the Mozilla Common Voice dataset and yield samples in the standard schema.

        Args:
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample of the dataset
            sample_size: Number of samples to process in sample mode

        Yields:
            dict: Processed sample in standard schema
        """
        # Load checkpoint if provided
        checkpoint_data = None
        if checkpoint:
            checkpoint_data = self.load_checkpoint(checkpoint)

        # Initialize processing state
        processed_count = 0
        current_index = 1
        processed_ids = set()

        if checkpoint_data:
            processed_count = checkpoint_data.get("processed_count", 0)
            current_index = checkpoint_data.get("current_index", 1)
            processed_ids = set(checkpoint_data.get("processed_ids", []))

            self.logger.info(f"Resuming from checkpoint: processed_count={processed_count}, current_index={current_index}")

        # Load dataset
        try:
            # Load dataset with language filter
            dataset = load_dataset(self.source, self.language_filter, split=self.split, trust_remote_code=True)

            # Initialize progress tracker
            total_items = len(dataset)

            # If in sample mode, only process a small sample
            if sample_mode:
                self.logger.info(f"Sample mode: Processing only {sample_size} samples from Mozilla Common Voice dataset")
                # Take a small sample from the dataset
                dataset = dataset.select(range(min(sample_size, total_items)))
                total_items = len(dataset)

            self.progress_tracker = ProgressTracker(total_items=total_items)

            self.logger.info(f"Processing Mozilla Common Voice dataset: {total_items} samples")

            # Process samples
            for i, sample in enumerate(dataset):
                # Skip already processed samples
                sample_id = sample.get("client_id", "") + "_" + sample.get("path", str(i))
                if sample_id in processed_ids:
                    self.progress_tracker.update(items_processed=0, skipped=1)
                    continue

                try:
                    # Convert to standard schema
                    processed_sample = self._convert_sample(sample, current_index)

                    # Validate sample
                    errors = self.validate_sample(processed_sample)
                    if errors:
                        error_msg = "; ".join(errors)
                        self.logger.warning(f"Sample validation failed: {error_msg}")
                        self.processing_logger.log_sample(
                            sample_id=sample_id,
                            status="skipped",
                            details={"reason": f"Validation failed: {error_msg}"}
                        )
                        self.progress_tracker.update(items_processed=0, skipped=1)
                        continue

                    # Log processing
                    self.processing_logger.log_sample(
                        sample_id=sample_id,
                        status="processed",
                        details={"output_id": processed_sample["ID"]}
                    )

                    # Update state
                    processed_count += 1
                    current_index += 1
                    processed_ids.add(sample_id)

                    # Update progress
                    self.progress_tracker.update()

                    # Save checkpoint periodically
                    if processed_count % 1000 == 0:
                        self._save_processing_checkpoint(processed_count, current_index, processed_ids)

                    yield processed_sample

                except Exception as e:
                    self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
                    self.processing_logger.log_sample(
                        sample_id=sample_id,
                        status="error",
                        details={"error": str(e)}
                    )
                    self.progress_tracker.update(items_processed=0, errors=1)

            # Save final checkpoint
            self._save_processing_checkpoint(processed_count, current_index, processed_ids)

            # Log summary
            self.progress_tracker.log_summary()
            self.processing_logger.save_json_log()

        except Exception as e:
            self.logger.error(f"Error processing Mozilla Common Voice dataset: {str(e)}")
            raise NetworkError(f"Failed to load Mozilla Common Voice dataset: {str(e)}")

    def _convert_sample(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Convert a Mozilla Common Voice sample to the standard schema.

        Args:
            sample: Mozilla Common Voice sample
            index: Current index for ID generation

        Returns:
            dict: Sample in standard schema
        """
        # Generate ID
        id_str = self.generate_id(index)

        # Get audio data
        audio_data = sample.get("audio", {}).get("bytes")
        if not audio_data:
            raise ValidationError("Missing audio data")

        # Apply audio preprocessing through base processor
        audio_data = self.preprocess_audio(audio_data, id_str)

        # Convert audio bytes to HuggingFace Audio format for proper preview functionality
        audio_dict = self.create_hf_audio_format(audio_data, id_str)

        # Calculate length from the HuggingFace format (more consistent)
        length = get_audio_length(audio_dict)
        if length is None:
            raise ValidationError("Failed to calculate audio length")

        # Get transcript
        transcript = sample.get("sentence", "")

        # Create standard sample
        return {
            "ID": id_str,
            "Language": "th",
            "audio": audio_dict,
            "transcript": transcript,
            "length": length
        }

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

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return information about the dataset.

        Returns:
            dict: Dataset information
        """
        try:
            dataset = load_dataset(self.source, self.language_filter, split=self.split)

            return {
                "name": self.name,
                "source": self.source,
                "language": self.language_filter,
                "split": self.split,
                "total_samples": len(dataset),
                "features": list(dataset.features.keys())
            }
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {str(e)}")
            return {
                "name": self.name,
                "source": self.source,
                "language": self.language_filter,
                "split": self.split,
                "error": str(e)
            }

    def estimate_size(self) -> int:
        """
        Estimate the size of the dataset.

        Returns:
            int: Estimated number of samples
        """
        try:
            dataset = load_dataset(self.source, self.language_filter, split=self.split)
            return len(dataset)
        except Exception as e:
            self.logger.error(f"Error estimating dataset size: {str(e)}")
            return 0
    
    def process_streaming(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the Mozilla Common Voice dataset in streaming mode.

        Args:
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample of the dataset
            sample_size: Number of samples to process in sample mode

        Yields:
            dict: Processed sample in standard schema
        """
        # Initialize streaming state
        checkpoint_data, skip_count = self._initialize_streaming_state(checkpoint)
        
        try:
            # Load dataset in streaming mode
            self.logger.info(f"Loading Mozilla Common Voice in streaming mode for language: {self.language_filter}")
            
            dataset = load_dataset(
                self.source,
                self.language_filter,
                split=self.split,
                streaming=True,
                trust_remote_code=True
            )
            
            # Skip to checkpoint if resuming
            samples_processed = 0
            samples_yielded = 0
            
            for sample in dataset:
                # Check if we should skip or stop
                should_skip, should_break = self._should_skip_sample(
                    samples_processed, skip_count, samples_yielded, sample_mode, sample_size
                )
                if should_skip:
                    samples_processed += 1
                    continue
                if should_break:
                    break
                
                try:
                    # Extract audio and transcript
                    audio_data = sample.get('audio', {})
                    transcript = sample.get('sentence', '')
                    
                    # Skip if no audio
                    if not audio_data:
                        self.logger.warning(f"Skipping sample without audio")
                        continue
                    
                    # Extract audio bytes
                    audio_bytes = self._extract_audio_bytes(audio_data)
                    if not audio_bytes:
                        continue
                    
                    # Process audio (validate, preprocess, convert to HF format)
                    audio_hf = self._process_audio_for_streaming(audio_bytes, f"mozilla_cv_{samples_processed}")
                    if not audio_hf:
                        continue
                    
                    # Create sample in standard schema
                    processed_sample = self._create_streaming_sample(
                        audio_hf, transcript, samples_processed
                    )
                    
                    # Validate sample
                    errors = self.validate_sample(processed_sample)
                    if errors:
                        self.logger.warning(f"Sample validation errors: {errors}")
                        continue
                    
                    samples_processed += 1
                    samples_yielded += 1
                    
                    yield processed_sample
                    
                    # Log progress
                    self._log_streaming_progress(samples_processed)
                    
                except Exception as e:
                    self.logger.error(f"Error processing sample {samples_processed}: {str(e)}")
                    continue
            
            self.logger.info(f"Completed streaming processing: {samples_yielded} samples yielded")
            
        except Exception as e:
            self.logger.error(f"Error in streaming processing: {str(e)}")
            raise
