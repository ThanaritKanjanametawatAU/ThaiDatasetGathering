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
from utils.cache import CacheManager
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
        self.chunk_size = config.get("chunk_size", 10000)
        self.max_cache_gb = config.get("max_cache_gb", 100.0)
        
        # Initialize cache manager
        cache_dir = os.path.join(config.get("cache_dir", "./cache"), "GigaSpeech2")
        self.cache_manager = CacheManager(cache_dir, self.max_cache_gb)

    def process(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5, sample_archives: int = 1) -> Iterator[Dict[str, Any]]:
        """
        Process the GigaSpeech2 dataset and yield samples in the standard schema.

        Args:
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample of the dataset
            sample_size: Number of samples to process in sample mode
            sample_archives: Number of archive files to download in sample mode

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

        # Load dataset based on mode
        try:
            if sample_mode:
                # Use targeted Thai archive downloads for sample processing
                self.logger.info(f"Sample mode: Downloading {sample_archives} Thai archive(s) to extract {sample_size} samples")
                dataset = self._load_thai_archives_sample(sample_size, sample_archives)
                total_items = len(dataset)
            else:
                # Use chunked processing for full dataset
                self.logger.info(f"Full mode: Using chunked processing with {self.chunk_size} samples per chunk")
                dataset, total_items = self._load_chunked_dataset(checkpoint_data)

            self.progress_tracker = ProgressTracker(total_items=total_items)

            self.logger.info(f"Processing GigaSpeech2 dataset: {total_items} samples (after filtering)")

            # Process samples based on mode
            if sample_mode:
                yield from self._process_samples(dataset, processed_ids, current_index, processed_count)
            else:
                yield from self._process_chunks(dataset, processed_ids, current_index, processed_count)

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

        # Get audio data - HuggingFace datasets typically store audio in 'audio' key
        audio_info = sample.get("audio", {})
        audio_data = None
        
        if isinstance(audio_info, dict):
            # Debug: Log the audio info structure
            self.logger.debug(f"Audio info keys: {list(audio_info.keys())}")
            
            # HuggingFace audio datasets typically have 'bytes' key for raw audio data
            if "bytes" in audio_info:
                audio_data = audio_info["bytes"]
            elif "array" in audio_info and "sampling_rate" in audio_info:
                # Convert numpy array to bytes using soundfile
                import soundfile as sf
                import io
                import numpy as np
                
                array = audio_info["array"]
                sampling_rate = audio_info["sampling_rate"]
                
                # Convert to bytes
                buffer = io.BytesIO()
                if isinstance(array, list):
                    array = np.array(array, dtype=np.float32)
                sf.write(buffer, array, sampling_rate, format='WAV')
                buffer.seek(0)
                audio_data = buffer.read()
            elif "path" in audio_info:
                # Read file from path
                import os
                path = audio_info["path"]
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        audio_data = f.read()
        elif isinstance(audio_info, (bytes, bytearray)):
            # Direct bytes data
            audio_data = audio_info
        
        # Fallback to 'wav' key (GigaSpeech2 specific)
        if not audio_data:
            wav_info = sample.get("wav")
            if wav_info:
                if isinstance(wav_info, dict):
                    if "bytes" in wav_info:
                        audio_data = wav_info["bytes"]
                    elif "array" in wav_info:
                        # Handle array data
                        import soundfile as sf
                        import io
                        import numpy as np
                        
                        array = wav_info["array"]
                        sampling_rate = wav_info.get("sampling_rate", 16000)  # Default sample rate
                        
                        buffer = io.BytesIO()
                        if isinstance(array, list):
                            array = np.array(array, dtype=np.float32)
                        sf.write(buffer, array, sampling_rate, format='WAV')
                        buffer.seek(0)
                        audio_data = buffer.read()
                elif isinstance(wav_info, (bytes, bytearray)):
                    audio_data = wav_info
            
        if not audio_data:
            raise ValidationError("Missing audio data")

        # Ensure audio_data is bytes
        if not isinstance(audio_data, (bytes, bytearray)):
            raise ValidationError(f"Audio data must be bytes, got {type(audio_data)}")

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
    
    def process_streaming(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process the GigaSpeech2 dataset in streaming mode without downloading everything.

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
            self.logger.info(f"Loading GigaSpeech2 in streaming mode from {self.source}")
            
            # Load language-specific data files to avoid streaming through all languages
            using_language_specific = False
            if self.language_filter:
                self.logger.info(f"Loading {self.language_filter} data files directly")
                try:
                    # Use data_files to load only the specific language
                    data_pattern = f'data/{self.language_filter}/train/*.tar.gz'
                    dataset = load_dataset(
                        self.source,
                        data_files={'train': data_pattern},
                        split='train',
                        streaming=True
                    )
                    self.logger.info(f"Successfully loaded {self.language_filter} data files")
                    using_language_specific = True
                except Exception as e:
                    self.logger.warning(f"Failed to load {self.language_filter} data directly: {e}")
                    self.logger.info("Falling back to default dataset with filtering")
                    dataset = load_dataset(
                        self.source,
                        'default',
                        split='train',
                        streaming=True
                    )
            else:
                # Load default dataset
                dataset = load_dataset(
                    self.source,
                    'default',  # Use default config
                    split='train',
                    streaming=True
                )
            
            # Log info about streaming through dataset
            self.logger.info(f"Streaming through dataset to find {self.language_filter} samples")
            
            # Skip to checkpoint if resuming
            samples_processed = 0
            samples_yielded = 0
            samples_examined = 0
            
            for sample in dataset:
                samples_examined += 1
                
                # Log first few Thai samples to debug fields
                if self.language_filter and f'/data/{self.language_filter}/' in sample.get('__url__', '') and samples_processed <= 3:
                    self.logger.info(f"Thai sample {samples_processed+1}: keys={list(sample.keys())}, has_wav={'wav' in sample}, has_text={'text' in sample}")
                
                # Log progress every 1000 samples examined (more efficient)
                if samples_examined % 1000 == 0:
                    self.logger.info(f"Examined {samples_examined} samples, found {samples_processed} {self.language_filter} samples, yielded {samples_yielded}")
                    
                # Early termination warning for sample mode (only log once)
                if sample_mode and samples_examined == sample_size * 100 and samples_yielded < sample_size:
                    self.logger.warning(f"Examined {samples_examined} samples but only found {samples_yielded} {self.language_filter} samples. Thai content might be sparse.")
                
                # Filter for language - only needed if we're using the fallback dataset
                # If we successfully loaded language-specific files, no filtering needed
                if self.language_filter and not using_language_specific:
                    # Extract language from URL: data/{lang}/train/*.tar.gz
                    url = sample.get('__url__', '')
                    if f'/data/{self.language_filter}/' not in url:
                        continue
                    
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
                    # GigaSpeech2 uses 'wav' key for audio data
                    audio_data = sample.get('wav', sample.get('audio', {}))
                    
                    # GigaSpeech2 transcripts are in separate TSV files, not in the webdataset
                    # For now, we'll use empty transcript for streaming mode
                    # TODO: Load transcripts from TSV files if needed
                    transcript = sample.get('text', sample.get('transcript', ''))
                    if not transcript:
                        # Log once that transcripts are not available in streaming mode
                        if samples_processed == 0:
                            self.logger.info("Note: GigaSpeech2 transcripts are in separate TSV files, not available in streaming mode")
                    
                    # Skip if no audio
                    if not audio_data:
                        self.logger.warning(f"Skipping sample without audio")
                        continue
                    
                    # Extract audio bytes
                    audio_bytes = self._extract_audio_bytes(audio_data)
                    if not audio_bytes:
                        continue
                    
                    # Process audio (validate, preprocess, convert to HF format)
                    audio_hf = self._process_audio_for_streaming(audio_bytes, f"gigaspeech2_{samples_processed}")
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
    
    def _load_thai_archives_sample(self, sample_size: int, sample_archives: int) -> Any:
        """
        Load a small sample by downloading specific Thai archive files.
        
        Args:
            sample_size: Number of samples to extract
            sample_archives: Number of Thai archive files to download
            
        Returns:
            Dataset: Small dataset with collected samples
        """
        from datasets import Dataset
        
        # Load specific Thai archive files using data_files parameter
        thai_archives = []
        for i in range(sample_archives):
            thai_archives.append(f"data/th/train/{i}.tar.gz")
        
        self.logger.info(f"Loading Thai archives: {thai_archives}")
        
        try:
            # Load dataset from specific Thai archive files with streaming first
            # to ensure audio data is properly loaded
            streaming_dataset = load_dataset(
                self.source, 
                split="train",
                data_files=thai_archives,
                streaming=True  # Use streaming to properly load audio
            )
            
            # Collect samples from streaming dataset
            samples = []
            count = 0
            for sample in streaming_dataset:
                samples.append(sample)
                count += 1
                if count >= sample_size:
                    break
            
            self.logger.info(f"Loaded {len(samples)} samples from {len(thai_archives)} Thai archive(s)")
            
            # Convert to non-streaming dataset
            return Dataset.from_list(samples)
            
        except Exception as e:
            self.logger.error(f"Failed to load Thai archives: {e}")
            # Fallback to streaming mode if archive loading fails
            self.logger.info("Falling back to streaming mode...")
            return self._load_streaming_fallback(sample_size)
    
    def _load_chunked_dataset(self, checkpoint_data: Optional[Dict[str, Any]]) -> tuple:
        """
        Load dataset in chunks to manage memory and cache usage.
        
        Args:
            checkpoint_data: Checkpoint data for resuming
            
        Returns:
            tuple: (dataset, total_items) where dataset is an iterator over chunks
        """
        # Get total dataset size first
        total_size = self._get_total_dataset_size()
        
        # Determine starting chunk based on checkpoint
        start_chunk = 0
        if checkpoint_data:
            start_chunk = checkpoint_data.get("current_chunk", 0)
            self.logger.info(f"Resuming from chunk {start_chunk}")
        
        # Return a generator that yields chunks
        return self._chunk_generator(start_chunk, total_size), total_size
    
    def _get_total_dataset_size(self) -> int:
        """
        Get the total size of the Thai-filtered dataset.
        
        Returns:
            int: Total number of Thai samples
        """
        try:
            # Load dataset info without downloading
            from datasets import get_dataset_infos
            dataset_infos = get_dataset_infos(self.source)
            
            # Estimate Thai samples (this is approximate)
            # We'll use streaming to get a better estimate
            self.logger.info("Estimating total Thai samples in dataset...")
            
            dataset = load_dataset(self.source, split="train", streaming=True)
            
            # Sample a portion to estimate the percentage of Thai content
            sample_count = 0
            thai_count = 0
            estimation_samples = 10000  # Sample 10k to estimate
            
            for sample in dataset:
                sample_count += 1
                if sample.get("language") == self.language_filter:
                    thai_count += 1
                
                if sample_count >= estimation_samples:
                    break
            
            # Estimate total based on the full dataset size
            thai_ratio = thai_count / sample_count if sample_count > 0 else 0
            
            # Get approximate total dataset size (this might not be exact)
            # For now, we'll use a conservative estimate
            estimated_total = int(estimation_samples * (thai_count / max(sample_count, 1)) * 100)
            
            self.logger.info(f"Estimated {estimated_total} Thai samples in dataset (based on {thai_count}/{sample_count} ratio)")
            
            return estimated_total
            
        except Exception as e:
            self.logger.warning(f"Could not estimate dataset size: {e}")
            return 1000000  # Default large estimate
    
    def _chunk_generator(self, start_chunk: int, total_size: int):
        """
        Generator that yields dataset chunks.
        
        Args:
            start_chunk: Starting chunk index
            total_size: Total estimated dataset size
            
        Yields:
            Dataset: Chunk of the dataset
        """
        chunk_idx = start_chunk
        
        while True:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_size)
            
            if start_idx >= total_size:
                break
            
            # Enforce cache limit before loading new chunk
            if not self.cache_manager.enforce_cache_limit():
                self.logger.error("Failed to enforce cache limit")
                raise RuntimeError("Cache limit enforcement failed")
            
            self.logger.info(f"Loading chunk {chunk_idx}: samples {start_idx}-{end_idx}")
            
            try:
                # Load chunk using split slicing
                chunk_dataset = load_dataset(
                    self.source, 
                    split=f"train[{start_idx}:{end_idx}]"
                )
                
                # Filter for Thai language
                if self.language_filter:
                    chunk_dataset = chunk_dataset.filter(
                        lambda x: x.get("language") == self.language_filter
                    )
                
                self.logger.info(f"Chunk {chunk_idx} loaded: {len(chunk_dataset)} Thai samples")
                
                # Log cache status
                cache_info = self.cache_manager.get_cache_info()
                self.logger.info(f"Cache usage: {cache_info['size_gb']:.1f}GB / {cache_info['max_size_gb']:.1f}GB ({cache_info['usage_percent']:.1f}%)")
                
                yield chunk_dataset, chunk_idx
                
                chunk_idx += 1
                
            except Exception as e:
                self.logger.error(f"Failed to load chunk {chunk_idx}: {e}")
                # Try to continue with next chunk
                chunk_idx += 1
                if chunk_idx * self.chunk_size >= total_size:
                    break
                continue
    
    def _load_streaming_fallback(self, sample_size: int) -> Any:
        """
        Fallback streaming method if direct archive loading fails.
        
        Args:
            sample_size: Number of samples to collect
            
        Returns:
            Dataset: Small dataset with collected samples
        """
        from datasets import Dataset
        
        self.logger.info(f"Using streaming fallback to collect {sample_size} Thai samples...")
        
        # Use streaming mode to avoid downloading entire dataset
        dataset = load_dataset(self.source, split="train", streaming=True)
        
        # Filter for Thai language and collect samples up to sample_size
        filtered_samples = []
        samples_seen = 0
        
        for sample in dataset:
            samples_seen += 1
            
            # Log progress every 1000 samples
            if samples_seen % 1000 == 0:
                self.logger.info(f"Streamed {samples_seen} samples, collected {len(filtered_samples)} Thai samples")
            
            if sample.get("language") == self.language_filter:
                filtered_samples.append(sample)
                if len(filtered_samples) >= sample_size:
                    break
        
        self.logger.info(f"Collected {len(filtered_samples)} Thai samples from {samples_seen} total samples")
        
        # Convert to non-streaming format for easier processing
        return Dataset.from_list(filtered_samples)
    
    def _process_samples(self, dataset, processed_ids: set, current_index: int, processed_count: int) -> Iterator[Dict[str, Any]]:
        """
        Process samples from a regular dataset (sample mode).
        
        Args:
            dataset: Dataset to process
            processed_ids: Set of already processed sample IDs
            current_index: Current index for ID generation
            processed_count: Current count of processed samples
            
        Yields:
            dict: Processed samples
        """
        self.logger.info(f"Processing {len(dataset)} samples in dataset")
        for i, sample in enumerate(dataset):
            self.logger.info(f"Processing sample {i+1}/{len(dataset)}: {sample.get('id', f'unknown_{i}')}")
            
            # Skip already processed samples (only if they have a valid ID)
            sample_id = sample.get("id")
            if sample_id is not None and sample_id in processed_ids:
                self.logger.info(f"Skipping already processed sample: {sample_id}")
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
                if sample_id is not None:
                    processed_ids.add(sample_id)

                # Update progress
                self.progress_tracker.update()

                # Save checkpoint periodically
                if processed_count % 100 == 0:  # More frequent saves for sample mode
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
    
    def _process_chunks(self, chunk_generator, processed_ids: set, current_index: int, processed_count: int) -> Iterator[Dict[str, Any]]:
        """
        Process samples from chunked dataset (full mode).
        
        Args:
            chunk_generator: Generator yielding dataset chunks
            processed_ids: Set of already processed sample IDs
            current_index: Current index for ID generation
            processed_count: Current count of processed samples
            
        Yields:
            dict: Processed samples
        """
        for chunk_dataset, chunk_idx in chunk_generator:
            self.logger.info(f"Processing chunk {chunk_idx} with {len(chunk_dataset)} samples")
            
            for i, sample in enumerate(chunk_dataset):
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
                            sample_id=sample.get("id", f"unknown_{chunk_idx}_{i}"),
                            status="skipped",
                            details={"reason": f"Validation failed: {error_msg}"}
                        )
                        self.progress_tracker.update(items_processed=0, skipped=1)
                        continue

                    # Log processing
                    self.processing_logger.log_sample(
                        sample_id=sample.get("id", f"unknown_{chunk_idx}_{i}"),
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
                        checkpoint_data = {
                            "processed_count": processed_count,
                            "current_index": current_index,
                            "processed_ids": list(processed_ids),
                            "current_chunk": chunk_idx
                        }
                        self.save_checkpoint(checkpoint_data)

                    yield processed_sample

                except Exception as e:
                    self.logger.error(f"Error processing sample {sample.get('id', f'unknown_{chunk_idx}_{i}')}: {str(e)}")
                    self.processing_logger.log_sample(
                        sample_id=sample.get("id", f"unknown_{chunk_idx}_{i}"),
                        status="error",
                        details={"error": str(e)}
                    )
                    self.progress_tracker.update(items_processed=0, errors=1)
            
            # Log chunk completion
            self.logger.info(f"Completed chunk {chunk_idx}")
