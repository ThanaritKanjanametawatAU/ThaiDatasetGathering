"""
Streaming utilities for processing large datasets without full downloads.
"""

import os
import logging
from typing import Dict, Any, Iterator, Optional, List, Tuple
from datasets import load_dataset, Dataset, IterableDataset, Features, Value, Audio
from huggingface_hub import HfApi, upload_file
import pyarrow as pa
import pyarrow.parquet as pq
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class StreamingUploader:
    """Handles streaming upload of dataset shards to Hugging Face."""
    
    def __init__(self, repo_id: str, token: Optional[str] = None, private: bool = False, append_mode: bool = False):
        """
        Initialize streaming uploader.
        
        Args:
            repo_id: Hugging Face repository ID
            token: Hugging Face token
            private: Whether to make the dataset private
            append_mode: Whether to append to existing dataset (preserves existing shards)
        """
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self.append_mode = append_mode
        self.api = HfApi(token=token)
        self.shard_num = 0
        self.total_samples = 0
        
        # Create repo if it doesn't exist
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            logger.info(f"Repository {repo_id} ready for streaming upload")
        except Exception as e:
            logger.error(f"Failed to create/access repository: {str(e)}")
            raise
        
        # If append mode, find the highest existing shard number
        if self.append_mode:
            existing_shards = self._get_existing_shard_numbers()
            if existing_shards:
                self.shard_num = max(existing_shards) + 1
                logger.info(f"Append mode: Starting from shard {self.shard_num:05d}")
            else:
                logger.info("Append mode: No existing shards found, starting from shard 0")
    
    def _get_existing_shard_numbers(self) -> List[int]:
        """
        Get list of existing shard numbers in the repository.
        
        Returns:
            List of shard numbers found in the repository
        """
        try:
            # List all files in the repository
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            
            shard_numbers = []
            for file_path in files:
                # Look for shard files in data/train/ directory
                if file_path.startswith("data/train/shard_") and file_path.endswith(".parquet"):
                    # Extract shard number from filename
                    # Format: data/train/shard_00000.parquet
                    try:
                        shard_part = file_path.replace("data/train/shard_", "").replace(".parquet", "")
                        shard_num = int(shard_part)
                        shard_numbers.append(shard_num)
                    except ValueError:
                        # Skip files that don't match the expected pattern
                        continue
            
            return shard_numbers
        except Exception as e:
            logger.warning(f"Error listing existing shards: {str(e)}. Starting from shard 0.")
            return []
    
    def upload_batch(self, samples: List[Dict[str, Any]], shard_num: Optional[int] = None) -> Tuple[bool, str]:
        """
        Upload a batch of samples as a parquet shard.
        
        Args:
            samples: List of samples to upload
            shard_num: Optional shard number (auto-increments if not provided)
            
        Returns:
            Tuple of (success, shard_filename)
        """
        if not samples:
            return True, ""
        
        if shard_num is None:
            shard_num = self.shard_num
            self.shard_num += 1
        
        shard_filename = f"shard_{shard_num:05d}.parquet"
        temp_path = f"/tmp/{shard_filename}"
        
        try:
            # Define features for the dataset
            features = Features({
                "ID": Value("string"),
                "speaker_id": Value("string"),
                "Language": Value("string"),
                "audio": Audio(sampling_rate=16000),
                "transcript": Value("string"),
                "length": Value("float32"),
                "dataset_name": Value("string"),
                "confidence_score": Value("float64")
            })
            
            # Convert samples to Dataset with explicit features
            # The Audio feature will handle the conversion automatically
            dataset = Dataset.from_list(samples, features=features)
            
            # Save as parquet
            dataset.to_parquet(temp_path)
            
            # Upload to Hub
            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=f"data/train/{shard_filename}",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            
            self.total_samples += len(samples)
            logger.info(f"Uploaded {shard_filename} with {len(samples)} samples. Total: {self.total_samples}")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return True, shard_filename
            
        except Exception as e:
            logger.error(f"Failed to upload shard {shard_filename}: {str(e)}")
            # Cleanup on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, shard_filename
    
    def upload_dataset_card(self, dataset_info: Dict[str, Any]):
        """Upload dataset card with metadata."""
        card_content = f"""---
dataset_info:
  features:
  - name: ID
    dtype: string
  - name: speaker_id
    dtype: string
  - name: Language
    dtype: string
  - name: audio
    dtype: 
      audio:
        sampling_rate: 16000
  - name: transcript
    dtype: string
  - name: length
    dtype: float32
  - name: dataset_name
    dtype: string
  - name: confidence_score
    dtype: float64
  splits:
  - name: train
    num_examples: {dataset_info.get('total_samples', 0)}
  download_size: {dataset_info.get('download_size', 0)}
  dataset_size: {dataset_info.get('dataset_size', 0)}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.parquet
---

# {dataset_info.get('name', 'Thai Voice Dataset')}

{dataset_info.get('description', 'Combined Thai audio dataset from multiple sources')}

## Dataset Details

- **Total samples**: {dataset_info.get('total_samples', 0):,}
- **Total duration**: {dataset_info.get('total_duration_hours', 0):.2f} hours
- **Language**: Thai (th)
- **Audio format**: 16kHz mono WAV
- **Volume normalization**: -20dB

## Sources

{dataset_info.get('sources_description', '')}

## Usage

```python
from datasets import load_dataset

# Load with streaming to avoid downloading everything
dataset = load_dataset("{self.repo_id}", streaming=True)

# Iterate through samples
for sample in dataset['train']:
    print(sample['ID'], sample['transcript'][:50])
    # Process audio: sample['audio']
    break
```

## Schema

- `ID`: Unique identifier (S1, S2, S3, ...)
- `speaker_id`: Speaker identifier (SPK_00001, SPK_00002, ...)
- `Language`: Language code (always "th" for Thai)
- `audio`: Audio data with 16kHz sampling rate
- `transcript`: Text transcript of the audio
- `length`: Duration in seconds
- `dataset_name`: Source dataset name (e.g., "GigaSpeech2", "ProcessedVoiceTH", "MozillaCommonVoice")
- `confidence_score`: Confidence score of the transcript (0.0-1.0)
  - 1.0: Original transcript from source dataset
  - <1.0: STT-generated transcript
  - 0.0: Fallback transcript (e.g., [NO_TRANSCRIPT])

## Processing Details

This dataset was created using streaming processing to handle large-scale data without requiring full downloads.
Audio has been standardized to 16kHz mono with -20dB volume normalization.
"""
        
        try:
            upload_file(
                path_or_fileobj=card_content.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            logger.info("Dataset card uploaded successfully")
        except Exception as e:
            logger.error(f"Failed to upload dataset card: {str(e)}")


def create_streaming_dataset(datasets_configs: List[Dict[str, Any]], 
                           interleave: bool = True,
                           probabilities: Optional[List[float]] = None) -> IterableDataset:
    """
    Create a combined streaming dataset from multiple sources.
    
    Args:
        datasets_configs: List of dataset configurations
        interleave: Whether to interleave datasets (True) or concatenate (False)
        probabilities: Sampling probabilities for interleaving
        
    Returns:
        IterableDataset: Combined streaming dataset
    """
    streaming_datasets = []
    
    for config in datasets_configs:
        logger.info(f"Loading {config['name']} in streaming mode...")
        
        dataset_args = {
            "path": config['source'],
            "split": config.get('split', 'train'),
            "streaming": True
        }
        
        # Add language filter if specified
        if 'language_filter' in config:
            dataset_args['language'] = config['language_filter']
        
        # Add any additional kwargs
        if 'load_kwargs' in config:
            dataset_args.update(config['load_kwargs'])
        
        dataset = load_dataset(**dataset_args)
        streaming_datasets.append(dataset)
    
    # Combine datasets
    if len(streaming_datasets) == 1:
        return streaming_datasets[0]
    elif interleave:
        from datasets import interleave_datasets
        return interleave_datasets(streaming_datasets, probabilities=probabilities)
    else:
        from datasets import concatenate_datasets
        return concatenate_datasets(streaming_datasets)


def estimate_dataset_size(dataset_config: Dict[str, Any], sample_size: int = 100) -> Dict[str, Any]:
    """
    Estimate dataset size by sampling.
    
    Args:
        dataset_config: Dataset configuration
        sample_size: Number of samples to use for estimation
        
    Returns:
        Dict with size estimates
    """
    try:
        # Load a small sample
        dataset = load_dataset(
            dataset_config['source'],
            split=dataset_config.get('split', 'train'),
            streaming=True
        )
        
        total_bytes = 0
        total_duration = 0
        samples_checked = 0
        
        for sample in dataset:
            if samples_checked >= sample_size:
                break
            
            # Estimate audio size
            if 'audio' in sample:
                if isinstance(sample['audio'], dict) and 'array' in sample['audio']:
                    # Estimate bytes from array
                    audio_bytes = len(sample['audio']['array']) * 2  # 16-bit audio
                    total_bytes += audio_bytes
                
                if 'length' in sample:
                    total_duration += sample['length']
            
            samples_checked += 1
        
        if samples_checked > 0:
            avg_sample_size = total_bytes / samples_checked
            avg_duration = total_duration / samples_checked
            
            # Try to get total count if available
            estimated_total = dataset_config.get('estimated_size', 100000)
            
            return {
                'avg_sample_size_bytes': avg_sample_size,
                'avg_duration_seconds': avg_duration,
                'estimated_total_samples': estimated_total,
                'estimated_total_size_gb': (avg_sample_size * estimated_total) / (1024**3),
                'estimated_total_duration_hours': (avg_duration * estimated_total) / 3600
            }
    except Exception as e:
        logger.error(f"Failed to estimate dataset size: {str(e)}")
    
    return {
        'avg_sample_size_bytes': 0,
        'avg_duration_seconds': 0,
        'estimated_total_samples': 0,
        'estimated_total_size_gb': 0,
        'estimated_total_duration_hours': 0
    }


class StreamingBatchProcessor:
    """Process streaming datasets in batches with checkpoint support."""
    
    def __init__(self, batch_size: int = 1000, checkpoint_dir: str = "checkpoints"):
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.current_batch = []
        self.samples_processed = 0
        
    def process_with_checkpoints(self, 
                               dataset_iterator: Iterator[Dict[str, Any]],
                               process_fn,
                               checkpoint_file: Optional[str] = None) -> Iterator[List[Dict[str, Any]]]:
        """
        Process dataset with checkpoint support.
        
        Args:
            dataset_iterator: Iterator over dataset samples
            process_fn: Function to process each sample
            checkpoint_file: Path to checkpoint file
            
        Yields:
            Batches of processed samples
        """
        # Load checkpoint if exists
        start_from = 0
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_from = checkpoint.get('samples_processed', 0)
                logger.info(f"Resuming from sample {start_from}")
        
        # Skip to checkpoint
        for _ in range(start_from):
            next(dataset_iterator, None)
        
        self.samples_processed = start_from
        
        for sample in dataset_iterator:
            try:
                # Process sample
                processed = process_fn(sample)
                if processed:
                    self.current_batch.append(processed)
                    self.samples_processed += 1
                
                # Yield batch when full
                if len(self.current_batch) >= self.batch_size:
                    yield self.current_batch
                    self.current_batch = []
                    
                    # Save checkpoint
                    if checkpoint_file:
                        self._save_checkpoint(checkpoint_file)
                        
            except Exception as e:
                logger.error(f"Error processing sample: {str(e)}")
                continue
        
        # Yield remaining samples
        if self.current_batch:
            yield self.current_batch
    
    def _save_checkpoint(self, checkpoint_file: str):
        """Save processing checkpoint."""
        checkpoint_data = {
            'samples_processed': self.samples_processed,
            'timestamp': time.time()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)