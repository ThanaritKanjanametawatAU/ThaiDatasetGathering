#!/usr/bin/env python3
"""
Main entry point for the Thai Audio Dataset Collection project.
"""

import os
import sys
import argparse
import logging
import importlib
from typing import List, Dict, Any, Optional

from config import (
    DATASET_CONFIG, TARGET_DATASET, LOG_CONFIG, EXIT_CODES,
    HF_TOKEN_FILE, CHECKPOINT_DIR, LOG_DIR
)
from utils.logging import setup_logging
from utils.huggingface import read_hf_token, authenticate_hf, create_hf_dataset, upload_dataset, get_last_id
from utils.streaming import StreamingUploader
from processors.base_processor import BaseProcessor

# Set up logger
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Thai Audio Dataset Collection")

    # Primary mode options
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--fresh", action="store_true", help="Create a new dataset from scratch")
    mode_group.add_argument("--append", action="store_true", help="Append to an existing dataset")

    # Dataset selection
    parser.add_argument("--all", action="store_true", help="Process all available datasets")
    parser.add_argument("datasets", nargs="*", help="Dataset names to process")

    # Additional options
    parser.add_argument("--resume", action="store_true", help="Resume processing from latest checkpoint")
    parser.add_argument("--checkpoint", help="Resume processing from specific checkpoint file")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to Huggingface")
    parser.add_argument("--private", action="store_true", help="Make the Huggingface dataset private")
    parser.add_argument("--output", help="Output directory for local dataset")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--sample", action="store_true", help="Process only a small sample from each dataset")
    parser.add_argument("--sample-size", type=int, default=5,
                        help="Number of samples to process in sample mode (default: 5)")
    parser.add_argument("--sample-archives", type=int, default=1,
                        help="Number of archive files to download in sample mode (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Number of samples per chunk in full processing mode (default: 10000)")
    parser.add_argument("--max-cache-gb", type=float, default=100.0,
                        help="Maximum cache size in GB (default: 100)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before processing")
    # Streaming options
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode to process datasets without full download")
    parser.add_argument("--streaming-batch-size", type=int, default=1000,
                        help="Batch size for streaming mode (default: 1000)")
    parser.add_argument("--upload-batch-size", type=int, default=10000,
                        help="Number of samples before uploading a shard (default: 10000)")
    # Audio processing options
    parser.add_argument("--no-standardization", action="store_true",
                        help="Disable audio standardization (keep original format)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate in Hz (default: 16000)")
    parser.add_argument("--target-db", type=float, default=-20.0, help="Target volume level in dB (default: -20.0)")
    parser.add_argument("--no-volume-norm", action="store_true", help="Disable volume normalization")
    # STT options
    parser.add_argument("--enable-stt", action="store_true", help="Enable STT for missing transcripts")
    parser.add_argument("--stt-batch-size", type=int, default=16, help="Batch size for STT processing (default: 16)")
    parser.add_argument("--no-stt", action="store_true", help="Explicitly disable STT (overrides config)")
    # HuggingFace repository options
    parser.add_argument("--hf-repo", type=str,
                        help="HuggingFace repository to push dataset to (e.g., 'username/dataset-name')")
    # Speaker identification arguments
    speaker_group = parser.add_argument_group('Speaker Identification')
    speaker_group.add_argument(
        '--enable-speaker-id',
        action='store_true',
        help='Enable speaker identification and clustering'
    )
    speaker_group.add_argument(
        '--speaker-model',
        type=str,
        default='pyannote/embedding',
        help='Speaker embedding model to use (default: pyannote/embedding)'
    )
    speaker_group.add_argument(
        '--speaker-batch-size',
        type=int,
        default=10000,
        help='Batch size for speaker clustering (default: 10000)'
    )
    speaker_group.add_argument(
        '--speaker-threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for existing cluster assignment (default: 0.7)'
    )
    speaker_group.add_argument(
        '--store-embeddings',
        action='store_true',
        help='Store speaker embeddings for future use'
    )
    speaker_group.add_argument(
        '--speaker-min-cluster-size',
        type=int,
        default=15,
        help='Minimum cluster size for HDBSCAN (default: 15)'
    )
    speaker_group.add_argument(
        '--speaker-min-samples',
        type=int,
        default=10,
        help='Minimum samples for core points in HDBSCAN (default: 10)'
    )
    speaker_group.add_argument(
        '--speaker-epsilon',
        type=float,
        default=0.3,
        help='Cluster selection epsilon for HDBSCAN (default: 0.3)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.datasets:
        parser.error("Either --all or at least one dataset name must be specified")

    if args.all and args.datasets:
        parser.error("Cannot use --all with specific dataset names")

    if args.append and args.all:
        parser.error("Cannot use --append with --all")

    if args.checkpoint and args.resume:
        parser.error("Cannot use both --checkpoint and --resume")

    return args

def get_processor_class(processor_class_name: str) -> type:
    """
    Get processor class by name.

    Args:
        processor_class_name: Name of processor class

    Returns:
        type: Processor class
    """
    try:
        # Map processor class names to module names
        module_mapping = {
            'GigaSpeech2Processor': 'gigaspeech2',
            'ProcessedVoiceTHProcessor': 'processed_voice_th',
            'MozillaCommonVoiceProcessor': 'mozilla_cv'
        }
        
        if processor_class_name in module_mapping:
            module_name = f"processors.{module_mapping[processor_class_name]}"
        else:
            # Fallback to extracting base name
            if processor_class_name.endswith('Processor'):
                base_name = processor_class_name[:-9].lower()
            else:
                base_name = processor_class_name.lower()
            module_name = f"processors.{base_name}"
        
        module = importlib.import_module(module_name)
        return getattr(module, processor_class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load processor class {processor_class_name}: {str(e)}")
        sys.exit(EXIT_CODES["GENERAL_ERROR"])

def create_processor(dataset_name: str, config: Dict[str, Any]) -> BaseProcessor:
    """
    Create a processor instance for the specified dataset.

    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary

    Returns:
        BaseProcessor: Processor instance
    """
    dataset_config = DATASET_CONFIG.get(dataset_name)
    if not dataset_config:
        logger.error(f"Unknown dataset: {dataset_name}")
        sys.exit(EXIT_CODES["ARGUMENT_ERROR"])

    processor_class_name = dataset_config.get("processor_class")
    if not processor_class_name:
        logger.error(f"No processor class specified for dataset: {dataset_name}")
        sys.exit(EXIT_CODES["GENERAL_ERROR"])

    processor_class = get_processor_class(processor_class_name)

    # Merge dataset config with global config
    merged_config = {**config, **dataset_config}

    # Handle cache clearing if requested
    if config.get("clear_cache", False):
        from utils.cache import CacheManager
        cache_dir = os.path.join(config.get("cache_dir", "./cache"), dataset_name)
        cache_manager = CacheManager(cache_dir, config.get("max_cache_gb", 100.0))
        if cache_manager.clear_cache():
            logger.info(f"Cache cleared for {dataset_name}")
        else:
            logger.warning(f"Failed to clear cache for {dataset_name}")

    return processor_class(merged_config)

def get_datasets_to_process(args: argparse.Namespace) -> List[str]:
    """
    Get list of datasets to process based on command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        list: List of dataset names to process
    """
    if args.all:
        return list(DATASET_CONFIG.keys())
    return args.datasets

def get_checkpoint_file(args: argparse.Namespace, dataset_name: str) -> Optional[str]:
    """
    Get checkpoint file path based on command-line arguments.

    Args:
        args: Parsed command-line arguments
        dataset_name: Name of the dataset

    Returns:
        str: Path to checkpoint file or None if not using checkpoint
    """
    if args.checkpoint:
        return args.checkpoint

    if args.resume:
        # Create processor to get latest checkpoint
        processor = create_processor(dataset_name, {
            "checkpoint_dir": CHECKPOINT_DIR,
            "log_dir": LOG_DIR
        })
        return processor.get_latest_checkpoint()

    return None

def process_dataset(
    processor: BaseProcessor,
    checkpoint_file: Optional[str] = None,
    sample_mode: bool = False,
    sample_size: int = 5,
    sample_archives: int = 1
) -> List[Dict[str, Any]]:
    """
    Process a dataset using the specified processor.

    Args:
        processor: Dataset processor
        checkpoint_file: Path to checkpoint file (optional)
        sample_mode: If True, only process a small sample of the dataset
        sample_size: Number of samples to process in sample mode

    Returns:
        list: List of processed samples
    """
    logger.info(f"Processing dataset: {processor.name}")

    # Get dataset info only if not in sample mode (to avoid downloading full dataset)
    if not sample_mode:
        dataset_info = processor.get_dataset_info()
        logger.info(f"Dataset info: {dataset_info}")
    else:
        logger.info(f"Skipping dataset info in sample mode to avoid full download")

    # Process dataset - use process_all_splits to combine all splits
    samples = list(processor.process_all_splits(
        checkpoint=checkpoint_file,
        sample_mode=sample_mode,
        sample_size=sample_size
    ))

    logger.info(f"Processed {len(samples)} samples from {processor.name}")

    return samples

def combine_datasets(
    processed_datasets: Dict[str, List[Dict[str, Any]]],
    start_id: int = 1
) -> List[Dict[str, Any]]:
    """
    Combine processed datasets.

    Args:
        processed_datasets: Dictionary of dataset name to processed samples
        start_id: Starting ID number

    Returns:
        list: Combined dataset
    """
    combined_samples = []
    current_id = start_id

    for dataset_name, samples in processed_datasets.items():
        logger.info(f"Adding {len(samples)} samples from {dataset_name}")

        for sample in samples:
            # Update ID
            sample["ID"] = f"S{current_id}"
            current_id += 1

            combined_samples.append(sample)

    logger.info(f"Combined dataset has {len(combined_samples)} samples")

    return combined_samples

def process_streaming_mode(args, dataset_names: List[str]) -> int:
    """
    Process datasets in streaming mode without full download.
    
    Args:
        args: Command line arguments
        dataset_names: List of dataset names to process
        
    Returns:
        int: Exit code
    """
    # Read HF token first
    token = read_hf_token(HF_TOKEN_FILE)
    if not token and not args.no_upload:
        logger.error("Huggingface token not found or invalid")
        return EXIT_CODES["AUTHENTICATION_ERROR"]
    
    # Initialize streaming uploader
    uploader = None
    if not args.no_upload:
        # Determine target repository - use command-line arg if provided, otherwise use default
        target_repo = args.hf_repo if args.hf_repo else TARGET_DATASET["name"]
        
        uploader = StreamingUploader(
            repo_id=target_repo,
            token=token,
            private=args.private,
            append_mode=args.append
        )
    
    # Track global sample ID
    start_id = 1
    if args.append:
        # Use the same target repository
        last_id = get_last_id(target_repo)
        if last_id is not None:
            start_id = last_id + 1
            logger.info(f"Appending to existing dataset, starting from ID: S{start_id}")
    
    current_id = start_id
    
    # Initialize speaker identification if enabled
    speaker_identifier = None
    embedding_buffer = []
    
    if args.enable_speaker_id:
        from processors.speaker_identification import SpeakerIdentification
        
        speaker_config = {
            'model': args.speaker_model,
            'clustering': {
                'algorithm': 'hdbscan',
                'min_cluster_size': args.speaker_min_cluster_size,
                'min_samples': args.speaker_min_samples,
                'metric': 'cosine',
                'cluster_selection_epsilon': args.speaker_epsilon,
                'similarity_threshold': args.speaker_threshold
            },
            'batch_size': args.speaker_batch_size,
            'store_embeddings': args.store_embeddings,
            'embedding_path': os.path.join(args.output or '.', 'speaker_embeddings.h5'),
            'model_path': os.path.join(CHECKPOINT_DIR, 'speaker_model.json'),
            'fresh': args.fresh  # Pass fresh flag to speaker identification
        }
        
        speaker_identifier = SpeakerIdentification(speaker_config)
        logger.info("Initialized speaker identification system")
    
    # Process each dataset
    total_samples = 0
    batch_buffer = []
    
    for dataset_name in dataset_names:
        try:
            logger.info(f"Processing {dataset_name} in streaming mode")
            
            # Reset clustering state for new dataset to prevent cross-dataset speaker merging
            if speaker_identifier:
                # Clear existing clusters and centroids but keep the counter for unique IDs
                speaker_identifier.existing_clusters = {}
                speaker_identifier.cluster_centroids = None
                logger.info(f"Reset speaker clustering state for {dataset_name}, counter at: {speaker_identifier.speaker_counter}")
            
            # Create processor config
            processor_config = {
                "checkpoint_dir": CHECKPOINT_DIR,
                "log_dir": LOG_DIR,
                "streaming": True,
                "batch_size": args.streaming_batch_size,
                "upload_batch_size": args.upload_batch_size,
                "audio_config": {
                    "enable_standardization": not args.no_standardization,
                    "target_sample_rate": args.sample_rate,
                    "target_channels": 1,
                    "normalize_volume": not args.no_volume_norm,
                    "target_db": args.target_db,
                },
                "dataset_name": dataset_name,
                "enable_stt": args.enable_stt if not args.no_stt else False,
                "stt_batch_size": args.stt_batch_size
            }
            
            # Create processor
            processor = create_processor(dataset_name, processor_config)
            
            # Get checkpoint if resuming
            checkpoint_file = get_checkpoint_file(args, dataset_name) if args.resume else None
            
            # Process all splits in streaming mode
            sample_count = 0
            for sample in processor.process_all_splits(
                checkpoint=checkpoint_file,
                sample_mode=args.sample,
                sample_size=args.sample_size
            ):
                # Assign sequential ID
                sample["ID"] = f"S{current_id}"
                current_id += 1
                
                # Handle speaker identification
                if speaker_identifier:
                    # Add to embedding buffer for batch processing
                    embedding_buffer.append(sample)
                    
                    # Process speaker IDs when buffer is full
                    if len(embedding_buffer) >= args.speaker_batch_size:
                        logger.info(f"Processing speaker identification batch of {len(embedding_buffer)} samples")
                        
                        # Get speaker IDs for the batch
                        speaker_ids = speaker_identifier.process_batch(embedding_buffer)
                        
                        # Assign speaker IDs to samples
                        for sample, speaker_id in zip(embedding_buffer, speaker_ids):
                            sample['speaker_id'] = speaker_id
                            batch_buffer.append(sample)
                        
                        embedding_buffer = []
                    # else: sample stays in embedding_buffer until batch is full or end of processing
                else:
                    # No speaker identification - assign unique ID based on sample counter
                    sample['speaker_id'] = f"SPK_{total_samples + 1:05d}"
                    batch_buffer.append(sample)
                
                sample_count += 1
                total_samples += 1
                
                # Upload batch when buffer is full
                if len(batch_buffer) >= args.upload_batch_size:
                    if uploader:
                        success, shard_name = uploader.upload_batch(batch_buffer)
                        if not success:
                            logger.error(f"Failed to upload batch {shard_name}")
                            return EXIT_CODES["UPLOAD_ERROR"]
                    
                    # Save checkpoint
                    processor.save_streaming_checkpoint(
                        shard_num=uploader.shard_num if uploader else 0,
                        samples_processed=total_samples,
                        last_sample_id=sample["ID"]
                    )
                    
                    batch_buffer = []
                
                # Log progress
                if sample_count % 1000 == 0:
                    logger.info(f"Processed {sample_count} samples from {dataset_name}, total: {total_samples}")
            
            # Process any remaining embeddings for this dataset before moving to next
            if speaker_identifier and embedding_buffer:
                logger.info(f"Processing final speaker batch for {dataset_name}: {len(embedding_buffer)} samples")
                speaker_ids = speaker_identifier.process_batch(embedding_buffer)
                
                for sample, speaker_id in zip(embedding_buffer, speaker_ids):
                    sample['speaker_id'] = speaker_id
                    batch_buffer.append(sample)
                
                embedding_buffer = []  # Clear buffer for next dataset
            
            logger.info(f"Completed {dataset_name}: {sample_count} samples")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name} in streaming mode: {str(e)}")
            if args.verbose:
                logger.exception(e)
            # Continue with next dataset instead of failing completely
    
    # Process remaining embeddings (should be empty now since we process at end of each dataset)
    if speaker_identifier and embedding_buffer:
        logger.warning(f"Unexpected: {len(embedding_buffer)} samples remain in embedding buffer")
        speaker_ids = speaker_identifier.process_batch(embedding_buffer)
        
        for sample, speaker_id in zip(embedding_buffer, speaker_ids):
            sample['speaker_id'] = speaker_id
            batch_buffer.append(sample)
    
    # Upload remaining samples
    if batch_buffer and uploader:
        success, shard_name = uploader.upload_batch(batch_buffer)
        if not success:
            logger.error(f"Failed to upload final batch {shard_name}")
            return EXIT_CODES["UPLOAD_ERROR"]
    
    # Clean up speaker identifier
    if speaker_identifier:
        speaker_identifier.cleanup()
    
    # Upload dataset card
    if uploader:
        dataset_info = {
            "name": TARGET_DATASET["name"],
            "description": TARGET_DATASET["description"],
            "total_samples": total_samples,
            "total_duration_hours": 0,  # Would need to track this
            "sources_description": f"Processed {len(dataset_names)} datasets in streaming mode"
        }
        uploader.upload_dataset_card(dataset_info)
    
    logger.info(f"Streaming processing completed: {total_samples} total samples")
    return EXIT_CODES["SUCCESS"]


def main() -> int:
    """
    Main entry point.

    Returns:
        int: Exit code
    """
    # Parse arguments
    args = parse_arguments()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else LOG_CONFIG["level"]
    setup_logging(log_file=LOG_CONFIG["file"], level=log_level)

    logger.info("Starting Thai Audio Dataset Collection")
    logger.info(f"Arguments: {args}")

    # Get datasets to process
    dataset_names = get_datasets_to_process(args)
    logger.info(f"Datasets to process: {dataset_names}")

    # Check if streaming mode
    if args.streaming:
        logger.info("Using streaming mode - processing without full download")
        return process_streaming_mode(args, dataset_names)
    
    # Original processing mode (with caching)
    processed_datasets = {}

    for dataset_name in dataset_names:
        try:
            # Get checkpoint file
            checkpoint_file = get_checkpoint_file(args, dataset_name)

            # Create processor with enhanced config
            processor_config = {
                "checkpoint_dir": CHECKPOINT_DIR,
                "log_dir": LOG_DIR,
                "cache_dir": "./cache",
                "chunk_size": args.chunk_size,
                "max_cache_gb": args.max_cache_gb,
                "clear_cache": args.clear_cache,
                "streaming": False,  # Explicitly set to false for cache mode
                # Audio processing configuration
                "audio_config": {
                    "enable_standardization": not args.no_standardization,
                    "target_sample_rate": args.sample_rate,
                    "target_channels": 1,
                    "normalize_volume": not args.no_volume_norm,
                    "target_db": args.target_db,
                    "target_format": "wav",
                    "trim_silence": True
                },
                # STT configuration
                "dataset_name": dataset_name,
                "enable_stt": args.enable_stt if not args.no_stt else False,
                "stt_batch_size": args.stt_batch_size
            }
            processor = create_processor(dataset_name, processor_config)

            # Process dataset
            samples = process_dataset(
                processor,
                checkpoint_file,
                sample_mode=args.sample,
                sample_size=args.sample_size,
                sample_archives=getattr(args, 'sample_archives', 1)
            )

            # Store processed samples
            processed_datasets[dataset_name] = samples

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            if args.verbose:
                logger.exception(e)
            return EXIT_CODES["PROCESSING_ERROR"]

    # Combine datasets
    start_id = 1

    if args.append:
        # Determine target repository - use command-line arg if provided, otherwise use default
        target_repo = args.hf_repo if args.hf_repo else TARGET_DATASET["name"]
        
        # Get last ID from existing dataset
        last_id = get_last_id(target_repo)
        if last_id is not None:
            start_id = last_id + 1
            logger.info(f"Appending to existing dataset, starting from ID: S{start_id}")
        else:
            logger.warning(f"Could not determine last ID from existing dataset, starting from S1")

    combined_samples = combine_datasets(processed_datasets, start_id)
    
    # Apply speaker identification if enabled (non-streaming mode)
    if args.enable_speaker_id:
        from processors.speaker_identification import SpeakerIdentification
        
        logger.info("Applying speaker identification to combined dataset")
        
        speaker_config = {
            'model': args.speaker_model,
            'clustering': {
                'algorithm': 'hdbscan',
                'min_cluster_size': args.speaker_min_cluster_size,
                'min_samples': args.speaker_min_samples,
                'metric': 'cosine',
                'cluster_selection_epsilon': args.speaker_epsilon,
                'similarity_threshold': args.speaker_threshold
            },
            'batch_size': args.speaker_batch_size,
            'store_embeddings': args.store_embeddings,
            'embedding_path': os.path.join(args.output or '.', 'speaker_embeddings.h5'),
            'model_path': os.path.join(CHECKPOINT_DIR, 'speaker_model.json'),
            'fresh': args.fresh  # Pass fresh flag to speaker identification
        }
        
        speaker_identifier = SpeakerIdentification(speaker_config)
        
        # Process in batches
        for i in range(0, len(combined_samples), args.speaker_batch_size):
            batch = combined_samples[i:i + args.speaker_batch_size]
            logger.info(f"Processing speaker identification batch {i//args.speaker_batch_size + 1}")
            
            speaker_ids = speaker_identifier.process_batch(batch)
            
            # Assign speaker IDs
            for sample, speaker_id in zip(batch, speaker_ids):
                sample['speaker_id'] = speaker_id
        
        speaker_identifier.cleanup()
    else:
        # Assign unique speaker IDs if not using identification
        for sample in combined_samples:
            sample['speaker_id'] = f"SPK_{int(sample['ID'][1:]):05d}"

    # Save locally if output directory specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, "combined_dataset.json")

        with open(output_file, 'w') as f:
            import json
            json.dump([{k: str(v) if k == "audio" else v for k, v in sample.items()} for sample in combined_samples], f, indent=2)

        logger.info(f"Saved combined dataset to {output_file}")

    # Upload to Huggingface if not disabled
    if not args.no_upload:
        # Read Huggingface token
        token = read_hf_token(HF_TOKEN_FILE)
        if not token:
            logger.error("Huggingface token not found or invalid")
            return EXIT_CODES["AUTHENTICATION_ERROR"]

        # Authenticate with Huggingface
        if not authenticate_hf(token):
            logger.error("Huggingface authentication failed")
            return EXIT_CODES["AUTHENTICATION_ERROR"]

        # Create dataset
        dataset = create_hf_dataset(combined_samples)
        if not dataset:
            logger.error("Failed to create Huggingface dataset")
            return EXIT_CODES["UPLOAD_ERROR"]

        # Determine target repository - use command-line arg if provided, otherwise use default
        target_repo = args.hf_repo if args.hf_repo else TARGET_DATASET["name"]
        
        # Upload dataset
        if not upload_dataset(dataset, target_repo, private=args.private, token=token):
            logger.error("Failed to upload dataset to Huggingface")
            return EXIT_CODES["UPLOAD_ERROR"]

        logger.info(f"Successfully uploaded dataset to {target_repo}")

    logger.info("Thai Audio Dataset Collection completed successfully")
    return EXIT_CODES["SUCCESS"]

if __name__ == "__main__":
    sys.exit(main())
