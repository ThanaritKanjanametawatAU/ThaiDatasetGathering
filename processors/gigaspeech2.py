"""
Processor for the GigaSpeech2 dataset.
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

class GigaSpeech2Processor(BaseProcessor):
    """
    Processor for the GigaSpeech2 dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GigaSpeech2 processor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.source = config.get("source", "speechcolab/gigaspeech2")
        self.language_filter = config.get("language_filter", "th")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)

    def process(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the GigaSpeech2 dataset and yield samples in the standard schema.

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

            # Filter for Thai language
            if self.language_filter:
                dataset = dataset.filter(lambda x: x.get("language") == self.language_filter)

            # Initialize progress tracker
            total_items = len(dataset)

            # If in sample mode, only process a small sample
            if sample_mode:
                self.logger.info(f"Sample mode: Processing only {sample_size} samples from GigaSpeech2 dataset")
                # Take a small sample from the dataset
                dataset = dataset.select(range(min(sample_size, total_items)))
                total_items = len(dataset)

            self.progress_tracker = ProgressTracker(total_items=total_items)

            self.logger.info(f"Processing GigaSpeech2 dataset: {total_items} samples (after filtering)")

            # Process samples
            for i, sample in enumerate(dataset):
                # Skip already processed samples
                if sample.get("id") in processed_ids:
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
                            sample_id=sample.get("id", f"unknown_{i}"),
                            status="skipped",
                            details={"reason": f"Validation failed: {error_msg}"}
                        )
                        self.progress_tracker.update(items_processed=0, skipped=1)
                        continue

                    # Log processing
                    self.processing_logger.log_sample(
                        sample_id=sample.get("id", f"unknown_{i}"),
                        status="processed",
                        details={"output_id": processed_sample["ID"]}
                    )

                    # Update state
                    processed_count += 1
                    current_index += 1
                    processed_ids.add(sample.get("id"))

                    # Update progress
                    self.progress_tracker.update()

                    # Save checkpoint periodically
                    if processed_count % 1000 == 0:
                        self._save_processing_checkpoint(processed_count, current_index, processed_ids)

                    yield processed_sample

                except Exception as e:
                    self.logger.error(f"Error processing sample {sample.get('id', f'unknown_{i}')}: {str(e)}")
                    self.processing_logger.log_sample(
                        sample_id=sample.get("id", f"unknown_{i}"),
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
            self.logger.error(f"Error processing GigaSpeech2 dataset: {str(e)}")
            raise NetworkError(f"Failed to load GigaSpeech2 dataset: {str(e)}")

    def _convert_sample(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Convert a GigaSpeech2 sample to the standard schema.

        Args:
            sample: GigaSpeech2 sample
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

        # Calculate length if not provided
        length = sample.get("audio", {}).get("duration")
        if length is None:
            length = get_audio_length(audio_data)
            if length is None:
                raise ValidationError("Failed to calculate audio length")

        # Get transcript
        transcript = sample.get("text", "")

        # Create standard sample
        return {
            "ID": id_str,
            "Language": "th",
            "audio": audio_data,
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

            # Filter for Thai language
            if self.language_filter:
                dataset = dataset.filter(lambda x: x.get("language") == self.language_filter)

            return {
                "name": self.name,
                "source": self.source,
                "language_filter": self.language_filter,
                "total_samples": len(dataset),
                "features": list(dataset.features.keys())
            }
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {str(e)}")
            return {
                "name": self.name,
                "source": self.source,
                "language_filter": self.language_filter,
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

            # Filter for Thai language
            if self.language_filter:
                dataset = dataset.filter(lambda x: x.get("language") == self.language_filter)

            return len(dataset)
        except Exception as e:
            self.logger.error(f"Error estimating dataset size: {str(e)}")
            return 0
