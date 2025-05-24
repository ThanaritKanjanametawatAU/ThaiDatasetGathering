"""
Processor for the Processed Voice TH dataset.
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

class ProcessedVoiceTHProcessor(BaseProcessor):
    """
    Processor for the Processed Voice TH dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Processed Voice TH processor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.source = config.get("source", "Porameht/processed-voice-th-169k")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)

    def process(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the Processed Voice TH dataset and yield samples in the standard schema.

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
            dataset = load_dataset(self.source, split="train")

            # Initialize progress tracker
            total_items = len(dataset)

            # If in sample mode, only process a small sample
            if sample_mode:
                self.logger.info(f"Sample mode: Processing only {sample_size} samples from Processed Voice TH dataset")
                # Take a small sample from the dataset
                dataset = dataset.select(range(min(sample_size, total_items)))
                total_items = len(dataset)

            self.progress_tracker = ProgressTracker(total_items=total_items)

            self.logger.info(f"Processing Processed Voice TH dataset: {total_items} samples")

            # Process samples
            for i, sample in enumerate(dataset):
                # Skip already processed samples
                sample_id = sample.get("id", str(i))
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
            self.logger.error(f"Error processing Processed Voice TH dataset: {str(e)}")
            raise NetworkError(f"Failed to load Processed Voice TH dataset: {str(e)}")

    def _convert_sample(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Convert a Processed Voice TH sample to the standard schema.

        Args:
            sample: Processed Voice TH sample
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
        transcript = sample.get("text", "")

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
            dataset = load_dataset(self.source, split="train")

            return {
                "name": self.name,
                "source": self.source,
                "total_samples": len(dataset),
                "features": list(dataset.features.keys())
            }
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {str(e)}")
            return {
                "name": self.name,
                "source": self.source,
                "error": str(e)
            }

    def estimate_size(self) -> int:
        """
        Estimate the size of the dataset.

        Returns:
            int: Estimated number of samples
        """
        try:
            dataset = load_dataset(self.source, split="train")
            return len(dataset)
        except Exception as e:
            self.logger.error(f"Error estimating dataset size: {str(e)}")
            return 0
    
    def process_streaming(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the Processed Voice TH dataset in streaming mode.

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
            self.logger.info(f"Loading Processed Voice TH in streaming mode from {self.source}")
            
            dataset = load_dataset(
                self.source,
                split='train',
                streaming=True
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
                    transcript = sample.get('text', '')
                    
                    # Skip if no audio
                    if not audio_data:
                        self.logger.warning(f"Skipping sample without audio")
                        continue
                    
                    # Extract audio bytes
                    audio_bytes = self._extract_audio_bytes(audio_data)
                    if not audio_bytes:
                        continue
                    
                    # Process audio (validate, preprocess, convert to HF format)
                    audio_hf = self._process_audio_for_streaming(audio_bytes, f"processed_voice_{samples_processed}")
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
