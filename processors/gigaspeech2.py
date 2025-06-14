"""
Processor for the GigaSpeech2 dataset.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Iterator, Tuple
import json
import csv

from processors.base_processor import BaseProcessor, NetworkError, ValidationError
from utils.audio import get_audio_length, is_valid_audio
from utils.logging import ProgressTracker
from utils.cache import CacheManager
from config import ErrorCategory

# Set up logger
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, load_dataset_builder
    from huggingface_hub import hf_hub_download
except ImportError:
    logger.error("Required datasets/huggingface_hub library not installed. Please install datasets and huggingface_hub.")
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
        
        # Transcript mappings cache
        self.transcript_mappings = {}  # {split: {segment_id: transcript}}
        self.transcript_cache_dir = os.path.join(cache_dir, "transcripts")
        os.makedirs(self.transcript_cache_dir, exist_ok=True)

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
    
    def _load_transcript_mapping(self, split: str) -> Dict[str, str]:
        """
        Load transcript mapping from TSV file for a specific split.
        
        Args:
            split: Split name (train, dev, test)
            
        Returns:
            dict: Mapping of segment_id to transcript
        """
        # Check if we already have this mapping cached
        if split in self.transcript_mappings:
            return self.transcript_mappings[split]
        
        # Check local cache first
        cache_file = os.path.join(self.transcript_cache_dir, f"{split}_transcripts.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                self.transcript_mappings[split] = mapping
                self.logger.info(f"Loaded {len(mapping)} transcripts from cache for split '{split}'")
                return mapping
            except Exception as e:
                self.logger.warning(f"Failed to load transcript cache: {e}")
        
        # Determine the TSV filename based on split
        if split == "train":
            # For train split, we have train_raw and train_refined
            # Default to train_refined for better quality
            tsv_filename = f"{self.language_filter}/train_refined.tsv"
            # Also try train_raw as fallback
            fallback_tsv = f"{self.language_filter}/train_raw.tsv"
        else:
            tsv_filename = f"{self.language_filter}/{split}.tsv"
            fallback_tsv = None
        
        mapping = {}
        
        try:
            # Download TSV file from HuggingFace
            self.logger.info(f"Downloading transcript file: {tsv_filename}")
            tsv_path = hf_hub_download(
                repo_id=self.source,
                filename=f"data/{tsv_filename}",
                repo_type="dataset",
                cache_dir=self.transcript_cache_dir
            )
            
            # Parse TSV file
            self.logger.info(f"Parsing transcript file: {tsv_path}")
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 2:
                        segment_id = row[0].strip()
                        transcript = row[1].strip()
                        mapping[segment_id] = transcript
            
            self.logger.info(f"Loaded {len(mapping)} transcripts for split '{split}'")
            
        except Exception as e:
            self.logger.warning(f"Failed to load {tsv_filename}: {e}")
            
            # Try fallback if available
            if fallback_tsv:
                try:
                    self.logger.info(f"Trying fallback transcript file: {fallback_tsv}")
                    tsv_path = hf_hub_download(
                        repo_id=self.source,
                        filename=f"data/{fallback_tsv}",
                        repo_type="dataset",
                        cache_dir=self.transcript_cache_dir
                    )
                    
                    with open(tsv_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='\t')
                        for row in reader:
                            if len(row) >= 2:
                                segment_id = row[0].strip()
                                transcript = row[1].strip()
                                mapping[segment_id] = transcript
                    
                    self.logger.info(f"Loaded {len(mapping)} transcripts from fallback")
                    
                except Exception as e2:
                    self.logger.error(f"Failed to load fallback TSV: {e2}")
        
        # Cache the mapping if we got any transcripts
        if mapping:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Cached {len(mapping)} transcripts for split '{split}'")
            except Exception as e:
                self.logger.warning(f"Failed to cache transcripts: {e}")
        
        self.transcript_mappings[split] = mapping
        return mapping
    
    def _extract_segment_id(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Extract segment ID from a sample.
        
        Args:
            sample: Dataset sample
            
        Returns:
            str: Segment ID if found, None otherwise
        """
        # First try the __key__ field which contains the actual segment ID
        key = sample.get('__key__', '')
        if key:
            # __key__ format is like "0/685/0-685-9"
            # The segment ID in TSV is the last part: "0-685-9"
            parts = key.split('/')
            if parts:
                # Return the last part which should match TSV segment IDs
                return parts[-1]
        
        # Try different fields where segment ID might be stored
        segment_id = sample.get('id')
        if segment_id:
            return segment_id
        
        # Try to extract from audio path or URL
        audio_info = sample.get('audio', {})
        if isinstance(audio_info, dict):
            # Check if there's a path field
            path = audio_info.get('path', '')
            if path:
                # Extract filename without extension as segment ID
                filename = os.path.basename(path)
                segment_id = os.path.splitext(filename)[0]
                if segment_id:
                    return segment_id
        
        # Check URL field (common in webdataset format)
        url = sample.get('__url__', '')
        if url:
            # Extract filename from URL
            filename = os.path.basename(url)
            # Remove .tar.gz or other extensions
            segment_id = filename.split('.')[0]
            if segment_id:
                return segment_id
        
        return None

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
                # Ensure float32 for soundfile compatibility
                if array.dtype == np.float16:
                    array = array.astype(np.float32)
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
                        # Ensure float32 for soundfile compatibility
                        if array.dtype == np.float16:
                            array = array.astype(np.float32)
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
        audio_data, enhancement_metadata = self.preprocess_audio(audio_data, id_str)

        # Convert audio bytes to HuggingFace Audio format for proper preview functionality
        audio_dict = self.create_hf_audio_format(audio_data, id_str)

        # Calculate length from the HuggingFace format (more consistent)
        length = get_audio_length(audio_dict)
        if length is None:
            raise ValidationError("Failed to calculate audio length")

        # Get transcript - first try from TSV mapping
        transcript = ""
        confidence_score = 1.0
        
        # Try to get transcript from TSV file first
        segment_id = self._extract_segment_id(sample)
        
        # Extract speaker ID from segment ID if possible
        speaker_id = None
        if segment_id:
            # Segment ID format: "0-685-9" where 685 might be speaker ID
            parts = segment_id.split('-')
            if len(parts) >= 2:
                try:
                    speaker_num = int(parts[1])
                    speaker_id = f"SPK_{speaker_num:05d}"
                except ValueError:
                    pass
            
            # Determine split (default to train if not specified)
            split = sample.get('split', 'train')
            
            # Load transcript mapping for this split if not already loaded
            transcript_mapping = self._load_transcript_mapping(split)
            
            # Look up transcript
            if segment_id in transcript_mapping:
                transcript = transcript_mapping[segment_id]
                self.logger.debug(f"Found transcript for segment {segment_id} from TSV")
            else:
                self.logger.debug(f"No transcript found for segment {segment_id} in TSV")
        
        # Fallback to transcript from sample if not found in TSV
        if not transcript:
            transcript = sample.get("text", sample.get("transcript", ""))
            if transcript:
                self.logger.debug("Using transcript from sample data")
            else:
                # If still no transcript and STT is enabled, it will be handled by the base processor
                self.logger.debug("No transcript available from TSV or sample")
                confidence_score = 0.0  # Will be updated by STT if enabled
        
        # Generate default speaker_id if not extracted
        if speaker_id is None:
            # Use a hash of dataset name and sample index to create deterministic IDs
            import hashlib
            speaker_hash = hashlib.md5(f"GigaSpeech2_{index}".encode()).hexdigest()
            speaker_num = int(speaker_hash[:8], 16) % 100000  # Convert to number 0-99999
            speaker_id = f"SPK_{speaker_num:05d}"

        # Create standard sample
        sample = {
            "ID": id_str,
            "speaker_id": speaker_id,
            "Language": "th",
            "audio": audio_dict,
            "transcript": transcript,
            "length": length,
            "dataset_name": "GigaSpeech2",
            "confidence_score": confidence_score
        }
        
        # Apply STT if enabled and transcript is empty
        if self.config.get("enable_stt", False) and not sample.get("transcript", "").strip():
            sample = self.process_sample_with_stt(sample, index)
        
        return sample


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
    
    def get_available_splits(self) -> List[str]:
        """
        Get available splits for GigaSpeech2 dataset.
        
        Returns:
            list: List of available split names
        """
        # GigaSpeech2 typically has train, validation, and test splits
        # We'll check which ones are actually available
        available_splits = []
        
        for split in ["train", "validation", "test"]:
            try:
                # Try to load the split info
                self.logger.debug(f"Checking if split '{split}' exists...")
                dataset_info = load_dataset(
                    self.source,
                    split=split,
                    streaming=True
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
        Process a single split of GigaSpeech2 dataset.
        
        Args:
            split: Split name to process
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample
            sample_size: Number of samples to process
            
        Yields:
            dict: Processed sample in standard schema
        """
        self.logger.info(f"Processing GigaSpeech2 split: {split}")
        
        # Temporarily override the split for streaming
        original_split = getattr(self, '_current_split', 'train')
        self._current_split = split
        
        try:
            # Use the existing streaming implementation but with the specific split
            yield from self._process_split_streaming(split, checkpoint, sample_mode, sample_size)
        finally:
            self._current_split = original_split
    
    def _process_split_streaming(self, split: str, checkpoint: Optional[str] = None, 
                                 sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process a specific split in streaming mode.
        
        Args:
            split: Split name to process
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample
            sample_size: Number of samples to process
            
        Yields:
            dict: Processed sample in standard schema
        """
        # Initialize streaming state
        checkpoint_data, skip_count = self._initialize_streaming_state(checkpoint)
        
        try:
            # Load dataset in streaming mode for the specific split
            self.logger.info(f"Loading GigaSpeech2 split '{split}' in streaming mode")
            
            # Load language-specific data files to avoid streaming through all languages
            using_language_specific = False
            if self.language_filter:
                self.logger.info(f"Loading {self.language_filter} data files directly for split '{split}'")
                try:
                    # Use data_files to load only the specific language and split
                    data_pattern = f'data/{self.language_filter}/{split}/*.tar.gz'
                    dataset = load_dataset(
                        self.source,
                        data_files={split: data_pattern},
                        split=split,
                        streaming=True
                    )
                    self.logger.info(f"Successfully loaded {self.language_filter} data files for split '{split}'")
                    using_language_specific = True
                except Exception as e:
                    self.logger.warning(f"Failed to load {self.language_filter} data directly: {e}")
                    self.logger.info(f"Falling back to default dataset with filtering for split '{split}'")
                    dataset = load_dataset(
                        self.source,
                        'default',
                        split=split,
                        streaming=True
                    )
            else:
                # Load default dataset for the split
                dataset = load_dataset(
                    self.source,
                    'default',
                    split=split,
                    streaming=True
                )
            
            # Process samples (similar to process_streaming but with split tracking)
            samples_processed = 0
            samples_yielded = 0
            samples_examined = 0
            
            for sample in dataset:
                samples_examined += 1
                
                # Filter for language if needed
                if self.language_filter and not using_language_specific:
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
                    # Extract audio and process
                    audio_data = sample.get('wav', sample.get('audio', {}))
                    
                    # Get transcript from TSV mapping
                    transcript = ""
                    segment_id = self._extract_segment_id(sample)
                    if segment_id:
                        # Load transcript mapping for this split
                        transcript_mapping = self._load_transcript_mapping(split)
                        if segment_id in transcript_mapping:
                            transcript = transcript_mapping[segment_id]
                            if samples_processed < 5:  # Log first few successes
                                self.logger.info(f"Found transcript for segment {segment_id} from TSV in split '{split}'")
                    
                    # Fallback to transcript from sample
                    if not transcript:
                        transcript = sample.get('text', sample.get('transcript', ''))
                    
                    if not audio_data:
                        continue
                    
                    audio_bytes = self._extract_audio_bytes(audio_data)
                    if not audio_bytes:
                        continue
                    
                    audio_hf = self._process_audio_for_streaming(audio_bytes, f"gigaspeech2_{split}_{samples_processed}")
                    if not audio_hf:
                        continue
                    
                    # Extract speaker ID from segment ID if possible
                    speaker_id = None
                    if segment_id:
                        # Segment ID format: "0-685-9" where 685 might be speaker ID
                        parts = segment_id.split('-')
                        if len(parts) >= 2:
                            try:
                                speaker_num = int(parts[1])
                                speaker_id = f"SPK_{speaker_num:05d}"
                            except ValueError:
                                pass
                    
                    # Create sample with split info
                    processed_sample = self._create_streaming_sample(
                        audio_hf, transcript, samples_processed, speaker_id=speaker_id
                    )
                    
                    # Apply STT if enabled and transcript is empty
                    if self.config.get("enable_stt", False) and not processed_sample.get("transcript", "").strip():
                        processed_sample = self.process_sample_with_stt(processed_sample, samples_processed)
                    
                    # Validate sample
                    errors = self.validate_sample(processed_sample)
                    if errors:
                        self.logger.warning(f"Sample validation errors: {errors}")
                        continue
                    
                    samples_processed += 1
                    samples_yielded += 1
                    
                    yield processed_sample
                    
                except Exception as e:
                    self.logger.error(f"Error processing sample {samples_processed} in split '{split}': {str(e)}")
                    continue
            
            self.logger.info(f"Completed processing split '{split}': {samples_yielded} samples yielded")
            
        except Exception as e:
            self.logger.error(f"Error processing split '{split}': {str(e)}")
            raise
    
    def process_all_splits(self, checkpoint: Optional[str] = None, sample_mode: bool = False, sample_size: int = 5) -> Iterator[Dict[str, Any]]:
        """
        Process all available splits of the GigaSpeech2 dataset.
        
        Args:
            checkpoint: Path to checkpoint file (optional)
            sample_mode: If True, only process a small sample
            sample_size: Number of samples to process in sample mode
            
        Yields:
            dict: Processed sample in standard schema
        """
        # Get available splits
        available_splits = self.get_available_splits()
        
        self.logger.info(f"Processing all available splits: {available_splits}")
        
        # Process each split
        for split in available_splits:
            yield from self._process_single_split(split, checkpoint, sample_mode, sample_size)
    
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
                    
                    # Get transcript from TSV mapping
                    transcript = ""
                    segment_id = self._extract_segment_id(sample)
                    
                    # Debug logging for first few samples
                    if samples_processed < 5:
                        self.logger.info(f"Sample {samples_processed + 1} - Extracted segment_id: {segment_id}, __key__: {sample.get('__key__')}, keys: {list(sample.keys())}")
                    
                    if segment_id:
                        # Load transcript mapping for train split
                        transcript_mapping = self._load_transcript_mapping('train')
                        if segment_id in transcript_mapping:
                            transcript = transcript_mapping[segment_id]
                            if samples_processed < 5:  # Log first few successes
                                self.logger.info(f"Found transcript for segment {segment_id} from TSV")
                        else:
                            if samples_processed < 5:
                                # Show a few keys from the mapping to understand the format
                                sample_keys = list(transcript_mapping.keys())[:5]
                                self.logger.info(f"Segment ID '{segment_id}' not found in TSV. Sample TSV keys: {sample_keys}")
                    
                    # Fallback to transcript from sample
                    if not transcript:
                        transcript = sample.get('text', sample.get('transcript', ''))
                    
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
                    
                    # Extract speaker ID from segment ID if possible
                    speaker_id = None
                    if segment_id:
                        # Segment ID format: "0-685-9" where 685 might be speaker ID
                        parts = segment_id.split('-')
                        if len(parts) >= 2:
                            try:
                                speaker_num = int(parts[1])
                                speaker_id = f"SPK_{speaker_num:05d}"
                            except ValueError:
                                pass
                    
                    # Create sample in standard schema
                    processed_sample = self._create_streaming_sample(
                        audio_hf, transcript, samples_processed, speaker_id=speaker_id
                    )
                    
                    # Apply STT if enabled and transcript is empty
                    if self.config.get("enable_stt", False) and not processed_sample.get("transcript", "").strip():
                        processed_sample = self.process_sample_with_stt(processed_sample, samples_processed)
                    
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
