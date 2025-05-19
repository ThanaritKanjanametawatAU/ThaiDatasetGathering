"""
Processor for the VISTEC Common Voice TH dataset.
"""

import os
import logging
import tempfile
import shutil
import csv
from typing import Optional, Dict, Any, List, Iterator
import json
import time

from processors.base_processor import BaseProcessor, NetworkError, FileAccessError, ValidationError
from utils.audio import get_audio_length, is_valid_audio
from utils.logging import ProgressTracker
from config import ErrorCategory

# Set up logger
logger = logging.getLogger(__name__)

try:
    import requests
    import soundfile as sf
    import numpy as np
except ImportError:
    logger.error("Required libraries not installed. Please install requests, soundfile, and numpy.")
    raise

class VistecCommonVoiceTHProcessor(BaseProcessor):
    """
    Processor for the VISTEC Common Voice TH dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VISTEC Common Voice TH processor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.source = config.get("source", "vistec-AI/commonvoice-th")
        self.repo_url = config.get("repo_url", "https://github.com/vistec-AI/commonvoice-th")
        self.data_dir = config.get("data_dir", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vistec_cv_th"))
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)

        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)

    def process(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the VISTEC Common Voice TH dataset and yield samples in the standard schema.

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
        processed_files = set()

        if checkpoint_data:
            processed_count = checkpoint_data.get("processed_count", 0)
            current_index = checkpoint_data.get("current_index", 1)
            processed_files = set(checkpoint_data.get("processed_files", []))

            self.logger.info(f"Resuming from checkpoint: processed_count={processed_count}, current_index={current_index}")

        # Download or clone repository if needed
        self._ensure_dataset_available()

        # Get dataset files
        try:
            # Find CSV files
            csv_files = self._find_csv_files()
            if not csv_files:
                raise FileAccessError("No CSV files found in the dataset")

            # Get audio files
            audio_files = self._find_audio_files()
            if not audio_files:
                raise FileAccessError("No audio files found in the dataset")

            # Create mapping of clip_name to audio_path
            audio_map = {os.path.splitext(os.path.basename(path))[0]: path for path in audio_files}

            # Count total samples
            total_items = 0
            for csv_file in csv_files:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    total_items += sum(1 for _ in reader)

            # Initialize progress tracker
            self.progress_tracker = ProgressTracker(total_items=total_items)

            # If in sample mode, only process a small sample
            sample_count = 0
            sample_limit = sample_size if sample_mode else float('inf')

            if sample_mode:
                self.logger.info(f"Sample mode: Processing only {sample_size} samples from VISTEC Common Voice TH dataset")
            else:
                self.logger.info(f"Processing VISTEC Common Voice TH dataset: {total_items} samples")

            # Process samples from each CSV file
            for csv_file in csv_files:
                try:
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)

                        for row in reader:
                            clip_name = row.get("path")
                            if not clip_name:
                                self.logger.warning("Missing path in CSV row")
                                self.progress_tracker.update(items_processed=0, skipped=1)
                                continue

                            # Skip already processed files
                            if clip_name in processed_files:
                                self.progress_tracker.update(items_processed=0, skipped=1)
                                continue

                            # Get audio file path
                            audio_path = audio_map.get(os.path.splitext(clip_name)[0])
                            if not audio_path:
                                self.logger.warning(f"Audio file not found for clip: {clip_name}")
                                self.processing_logger.log_sample(
                                    sample_id=clip_name,
                                    status="skipped",
                                    details={"reason": "Audio file not found"}
                                )
                                self.progress_tracker.update(items_processed=0, skipped=1)
                                continue

                            try:
                                # Convert to standard schema
                                processed_sample = self._convert_sample(row, audio_path, current_index)

                                # Validate sample
                                errors = self.validate_sample(processed_sample)
                                if errors:
                                    error_msg = "; ".join(errors)
                                    self.logger.warning(f"Sample validation failed: {error_msg}")
                                    self.processing_logger.log_sample(
                                        sample_id=clip_name,
                                        status="skipped",
                                        details={"reason": f"Validation failed: {error_msg}"}
                                    )
                                    self.progress_tracker.update(items_processed=0, skipped=1)
                                    continue

                                # Log processing
                                self.processing_logger.log_sample(
                                    sample_id=clip_name,
                                    status="processed",
                                    details={"output_id": processed_sample["ID"]}
                                )

                                # Update state
                                processed_count += 1
                                current_index += 1
                                processed_files.add(clip_name)

                                # Update progress
                                self.progress_tracker.update()

                                # Save checkpoint periodically
                                if processed_count % 1000 == 0:
                                    self._save_processing_checkpoint(processed_count, current_index, processed_files)

                                yield processed_sample

                                # Check if we've reached the sample limit
                                sample_count += 1
                                if sample_count >= sample_limit:
                                    self.logger.info(f"Sample mode: Reached limit of {sample_size} samples")
                                    break

                            except Exception as e:
                                self.logger.error(f"Error processing sample {clip_name}: {str(e)}")
                                self.processing_logger.log_sample(
                                    sample_id=clip_name,
                                    status="error",
                                    details={"error": str(e)}
                                )
                                self.progress_tracker.update(items_processed=0, errors=1)

                        # Break out of outer loop if we've reached the sample limit
                        if sample_count >= sample_limit:
                            break

                except Exception as e:
                    self.logger.error(f"Error processing CSV file {csv_file}: {str(e)}")

            # Save final checkpoint
            self._save_processing_checkpoint(processed_count, current_index, processed_files)

            # Log summary
            self.progress_tracker.log_summary()
            self.processing_logger.save_json_log()

        except Exception as e:
            self.logger.error(f"Error processing VISTEC Common Voice TH dataset: {str(e)}")
            raise

    def _ensure_dataset_available(self) -> None:
        """
        Ensure the dataset is available locally.
        """
        # Check if data directory already contains dataset
        if self._is_dataset_available():
            self.logger.info("VISTEC Common Voice TH dataset already available")
            return

        # Download dataset
        self.logger.info("Downloading VISTEC Common Voice TH dataset...")

        # For now, we'll just create a placeholder
        # In a real implementation, this would download or clone the repository
        os.makedirs(os.path.join(self.data_dir, "clips"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "validated"), exist_ok=True)

        # Create a placeholder README
        with open(os.path.join(self.data_dir, "README.md"), 'w') as f:
            f.write("# VISTEC Common Voice TH Dataset\n\nPlaceholder for dataset download.\n")

        self.logger.info("Dataset download placeholder created. In a real implementation, this would download the actual dataset.")

    def _is_dataset_available(self) -> bool:
        """
        Check if the dataset is available locally.

        Returns:
            bool: True if dataset is available, False otherwise
        """
        # Check for expected directories and files
        return (
            os.path.exists(self.data_dir) and
            os.path.exists(os.path.join(self.data_dir, "clips"))
        )

    def _find_csv_files(self) -> List[str]:
        """
        Find CSV files in the dataset.

        Returns:
            list: List of CSV file paths
        """
        csv_files = []

        # Look for CSV files in validated directory
        validated_dir = os.path.join(self.data_dir, "validated")
        if os.path.exists(validated_dir):
            for file in os.listdir(validated_dir):
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(validated_dir, file))

        return csv_files

    def _find_audio_files(self) -> List[str]:
        """
        Find audio files in the dataset.

        Returns:
            list: List of audio file paths
        """
        audio_files = []

        # Look for audio files in clips directory
        clips_dir = os.path.join(self.data_dir, "clips")
        if os.path.exists(clips_dir):
            for file in os.listdir(clips_dir):
                if file.endswith((".mp3", ".wav")):
                    audio_files.append(os.path.join(clips_dir, file))

        return audio_files

    def _convert_sample(self, row: Dict[str, str], audio_path: str, index: int) -> Dict[str, Any]:
        """
        Convert a VISTEC Common Voice TH sample to the standard schema.

        Args:
            row: CSV row data
            audio_path: Path to audio file
            index: Current index for ID generation

        Returns:
            dict: Sample in standard schema
        """
        # Generate ID
        id_str = self.generate_id(index)

        # Read audio file
        with open(audio_path, 'rb') as f:
            audio_data = f.read()

        if not audio_data:
            raise ValidationError("Empty audio file")

        # Calculate length
        length = get_audio_length(audio_data)
        if length is None:
            # Try using soundfile directly
            try:
                info = sf.info(audio_path)
                length = info.duration
            except Exception as e:
                raise ValidationError(f"Failed to calculate audio length: {str(e)}")

        # Get transcript
        transcript = row.get("sentence", "")

        # Create standard sample
        return {
            "ID": id_str,
            "Language": "th",
            "audio": audio_data,
            "transcript": transcript,
            "length": length
        }

    def _save_processing_checkpoint(self, processed_count: int, current_index: int, processed_files: set) -> None:
        """
        Save processing checkpoint.

        Args:
            processed_count: Number of processed samples
            current_index: Current index for ID generation
            processed_files: Set of processed file paths
        """
        checkpoint_data = {
            "processed_count": processed_count,
            "current_index": current_index,
            "processed_files": list(processed_files)
        }

        self.save_checkpoint(checkpoint_data)

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return information about the dataset.

        Returns:
            dict: Dataset information
        """
        # Ensure dataset is available
        self._ensure_dataset_available()

        # Get CSV files
        csv_files = self._find_csv_files()

        # Get audio files
        audio_files = self._find_audio_files()

        # Count total samples
        total_items = 0
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    total_items += sum(1 for _ in reader)
            except Exception:
                pass

        return {
            "name": self.name,
            "source": self.source,
            "total_samples": total_items,
            "csv_files": len(csv_files),
            "audio_files": len(audio_files)
        }

    def estimate_size(self) -> int:
        """
        Estimate the size of the dataset.

        Returns:
            int: Estimated number of samples
        """
        # Ensure dataset is available
        self._ensure_dataset_available()

        # Get CSV files
        csv_files = self._find_csv_files()

        # Count total samples
        total_items = 0
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    total_items += sum(1 for _ in reader)
            except Exception:
                pass

        return total_items
