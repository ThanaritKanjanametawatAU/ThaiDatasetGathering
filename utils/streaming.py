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
    
    def __init__(self, repo_id: str, token: Optional[str] = None, private: bool = False, 
                 append_mode: bool = False, split: str = "train", checkpoint_path: Optional[str] = None):
        """
        Initialize streaming uploader.
        
        Args:
            repo_id: Hugging Face repository ID
            token: Hugging Face token
            private: Whether to make the dataset private
            append_mode: Whether to append to existing dataset (preserves existing shards)
            split: Dataset split name (default: "train")
            checkpoint_path: Path to checkpoint file for tracking uploads
        """
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self.append_mode = append_mode
        self.split = split
        self.checkpoint_path = checkpoint_path
        self.api = HfApi(token=token)
        self.shard_num = 0
        self.total_samples = 0
        self.uploaded_ids = set()  # Track uploaded sample IDs to prevent duplicates
        self.current_batch = []  # Current batch of samples
        self.batch_size = 1000  # Default batch size
        
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
        else:
            # Fresh mode - delete existing data files
            logger.info("Fresh mode: Cleaning existing data files from repository")
            self._clean_existing_data()
        
        # Load checkpoint if available
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self._load_checkpoint()
    
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
    
    def _load_checkpoint(self):
        """Load checkpoint from disk."""
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.shard_num = checkpoint.get('shard_index', 0)
            self.total_samples = checkpoint.get('samples_uploaded', 0)
            self.uploaded_ids = set(checkpoint.get('uploaded_ids', []))
            self.split = checkpoint.get('split', self.split)
            
            logger.info(f"Loaded checkpoint: {len(self.uploaded_ids)} samples already uploaded")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
    
    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        if not self.checkpoint_path:
            return
        
        checkpoint = {
            'shard_index': self.shard_num,
            'samples_uploaded': self.total_samples,
            'uploaded_ids': list(self.uploaded_ids),
            'split': self.split,
            'dataset_name': self.repo_id
        }
        
        # Atomic write with temporary file
        temp_path = self.checkpoint_path + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            # Atomic rename
            os.replace(temp_path, self.checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def add_sample(self, sample: Dict[str, Any]):
        """Add a sample to the upload queue, preventing duplicates."""
        sample_id = sample.get('ID', str(self.total_samples))
        
        # Skip if already uploaded
        if sample_id in self.uploaded_ids:
            logger.debug(f"Skipping duplicate sample: {sample_id}")
            return
        
        # Add to current batch
        self.current_batch.append(sample)
        
        # If batch is full, upload it
        if len(self.current_batch) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Upload current batch if not empty."""
        if not self.current_batch:
            return
        
        # Upload the batch
        success, _ = self.upload_shard(self.current_batch, self.shard_num)
        
        if success:
            # Track uploaded IDs
            for sample in self.current_batch:
                sample_id = sample.get('ID', str(self.total_samples))
                self.uploaded_ids.add(sample_id)
                self.total_samples += 1
            
            # Save checkpoint
            self._save_checkpoint()
            
            # Prepare for next shard
            self.shard_num += 1
            self.current_batch = []
        else:
            logger.error("Failed to upload batch, keeping samples in queue")
    
    def _clean_existing_data(self):
        """Delete existing data files from the repository for fresh mode."""
        try:
            # List all files in the repository
            files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            
            # Find data files to delete
            files_to_delete = []
            for file_path in files:
                # Delete parquet files in data directory
                if file_path.startswith("data/") and file_path.endswith(".parquet"):
                    files_to_delete.append(file_path)
            
            # Delete files
            if files_to_delete:
                logger.info(f"Deleting {len(files_to_delete)} existing data files")
                for file_path in files_to_delete:
                    try:
                        self.api.delete_file(
                            path_in_repo=file_path,
                            repo_id=self.repo_id,
                            repo_type="dataset",
                            commit_message=f"Clean data for fresh mode: {file_path}"
                        )
                        logger.debug(f"Deleted: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {str(e)}")
                logger.info("Finished cleaning existing data files")
            else:
                logger.info("No existing data files to clean")
                
        except Exception as e:
            logger.error(f"Error cleaning existing data: {str(e)}")
            # Don't raise - allow process to continue even if cleanup fails
    
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
        
        # Filter out already uploaded samples
        filtered_samples = []
        for sample in samples:
            sample_id = sample.get('ID', str(self.total_samples))
            if sample_id not in self.uploaded_ids:
                filtered_samples.append(sample)
                self.uploaded_ids.add(sample_id)
            else:
                logger.debug(f"Skipping duplicate sample in upload_batch: {sample_id}")
        
        # If all samples were duplicates, return success
        if not filtered_samples:
            logger.info(f"All {len(samples)} samples were duplicates, skipping upload")
            return True, ""
        
        samples = filtered_samples
        
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
                "confidence_score": Value("float32")  # Changed from float64 to float32
            })
            
            # Clean samples - remove any fields not in the schema
            allowed_fields = set(features.keys())
            cleaned_samples = []
            for sample in samples:
                cleaned_sample = {k: v for k, v in sample.items() if k in allowed_fields}
                cleaned_samples.append(cleaned_sample)
            
            # Convert samples to Dataset with explicit features
            # The Audio feature will handle the conversion automatically
            dataset = Dataset.from_list(cleaned_samples, features=features)
            
            # Save as parquet with compression for viewer compatibility
            # Use pyarrow directly for better control
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Convert to arrow table
            table = dataset.data.table if hasattr(dataset, 'data') else pa.Table.from_pandas(dataset.to_pandas())
            
            # Write with proper settings for HuggingFace viewer
            pq.write_table(
                table,
                temp_path,
                row_group_size=100,  # 100 rows per group for better streaming
                compression='snappy'  # Good balance of speed and size
            )
            
            # Upload to Hub
            # Use split name in path
            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=f"data/{self.split}/{shard_filename}",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            
            self.total_samples += len(samples)
            logger.info(f"Uploaded {shard_filename} with {len(samples)} samples. Total: {self.total_samples}")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Save checkpoint after successful upload
            self._save_checkpoint()
            
            return True, shard_filename
            
        except Exception as e:
            logger.error(f"Failed to upload shard {shard_filename}: {str(e)}")
            # Cleanup on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, shard_filename
    
    def _read_existing_dataset_card(self) -> Optional[str]:
        """
        Read the existing dataset card from the repository.
        
        Returns:
            The existing README content, or None if not found
        """
        try:
            # Download the existing README.md file
            from huggingface_hub import hf_hub_download
            readme_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="README.md",
                repo_type="dataset",
                token=self.token
            )
            
            with open(readme_path, 'r') as f:
                content = f.read()
            
            return content
        except Exception as e:
            logger.info(f"No existing dataset card found: {str(e)}")
            return None
    
    def _parse_existing_stats(self, readme_content: str) -> Dict[str, Any]:
        """
        Parse existing statistics from a dataset card.
        
        Args:
            readme_content: The README.md content
            
        Returns:
            Dict with existing statistics
        """
        stats = {
            'total_samples': 0,
            'total_duration_hours': 0.0,
            'existing_datasets': []
        }
        
        if not readme_content:
            return stats
        
        lines = readme_content.split('\n')
        
        for line in lines:
            # Parse num_examples from YAML header
            if "num_examples:" in line:
                try:
                    stats['total_samples'] = int(line.split(':')[1].strip())
                except ValueError:
                    pass
            
            # Parse total samples from markdown
            elif "Total samples**:" in line:
                try:
                    sample_str = line.split('**:')[1].strip().replace(',', '')
                    stats['total_samples'] = int(sample_str)
                except ValueError:
                    pass
            
            # Parse total duration from markdown
            elif "Total duration**:" in line:
                try:
                    duration_str = line.split('**:')[1].strip().split()[0]
                    stats['total_duration_hours'] = float(duration_str)
                except ValueError:
                    pass
            
            # Parse existing dataset names
            elif line.strip().startswith("1. **") or line.strip().startswith("2. **") or line.strip().startswith("3. **"):
                if "**:" in line:
                    try:
                        dataset_name = line.split("**")[1]
                        if dataset_name and dataset_name not in stats['existing_datasets']:
                            stats['existing_datasets'].append(dataset_name)
                    except IndexError:
                        pass
        
        return stats
    
    def upload_dataset_card(self, dataset_info: Dict[str, Any], append_mode: Optional[bool] = None):
        """Upload dataset card with metadata."""
        if append_mode is None:
            append_mode = self.append_mode
        
        # Initialize final stats
        final_stats = {
            'total_samples': dataset_info.get('total_samples', 0),
            'total_duration_hours': dataset_info.get('total_duration_hours', 0),
            'dataset_names': dataset_info.get('dataset_names', [])
        }
        
        # If in append mode, read and merge existing stats
        if append_mode:
            existing_readme = self._read_existing_dataset_card()
            if existing_readme:
                existing_stats = self._parse_existing_stats(existing_readme)
                
                # Accumulate statistics
                final_stats['total_samples'] += existing_stats['total_samples']
                final_stats['total_duration_hours'] += existing_stats['total_duration_hours']
                
                # Merge dataset names
                all_datasets = existing_stats['existing_datasets'] + final_stats['dataset_names']
                # Remove duplicates while preserving order
                seen = set()
                final_stats['dataset_names'] = []
                for dataset in all_datasets:
                    if dataset not in seen:
                        seen.add(dataset)
                        final_stats['dataset_names'].append(dataset)
        
        # Generate source datasets description
        sources_list = []
        for i, dataset_name in enumerate(final_stats['dataset_names'], 1):
            if dataset_name == "GigaSpeech2":
                desc = "Large-scale multilingual speech corpus"
            elif dataset_name == "ProcessedVoiceTH":
                desc = "Thai voice dataset with processed audio"
            elif dataset_name == "MozillaCommonVoice":
                desc = "Mozilla Common Voice Thai dataset"
            else:
                desc = "Thai audio dataset"
            sources_list.append(f"{i}. **{dataset_name}**: {desc}")
        
        sources_description = f"Processed {len(final_stats['dataset_names'])} datasets in streaming mode"
        if sources_list:
            sources_description += "\n\n## Source Datasets\n\n" + "\n".join(sources_list)
        
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
    num_examples: {final_stats['total_samples']}
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

- **Total samples**: {final_stats['total_samples']:,}
- **Total duration**: {final_stats['total_duration_hours']:.2f} hours
- **Language**: Thai (th)
- **Audio format**: 16kHz mono WAV
- **Volume normalization**: -20dB

## Sources

{sources_description}

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
    
    def __init__(self, batch_size: int = 1000, checkpoint_dir: str = "checkpoints", 
                 process_fn=None, checkpoint_path: Optional[str] = None):
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = checkpoint_path
        self.process_fn = process_fn
        self.current_batch = []
        self.samples_processed = 0
        self.processed_ids = set()  # Track processed IDs to prevent duplicates
    
    def add_sample(self, sample: Dict[str, Any]):
        """Add a sample to the processing queue, preventing duplicates."""
        sample_id = sample.get('ID', str(self.samples_processed))
        
        # Skip if already processed
        if sample_id in self.processed_ids:
            logger.debug(f"Skipping duplicate sample in processor: {sample_id}")
            return
        
        # Add to current batch
        self.current_batch.append(sample)
        self.processed_ids.add(sample_id)
        
        # Process batch when full
        if len(self.current_batch) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Process current batch if not empty."""
        if not self.current_batch:
            return
        
        # Process the batch if process_fn is provided
        if self.process_fn:
            processed_batch = self.process_fn(self.current_batch)
            if processed_batch:
                self.samples_processed += len(processed_batch)
        else:
            self.samples_processed += len(self.current_batch)
        
        # Clear the batch
        self.current_batch = []
        
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