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
    HF_TOKEN_FILE, CHECKPOINT_DIR, LOG_DIR, validate_enhancement_level_compatibility
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
    
    # Audio enhancement arguments
    enhancement_group = parser.add_argument_group('Audio Enhancement')
    enhancement_group.add_argument(
        '--enable-audio-enhancement',
        action='store_true',
        help='Enable audio quality enhancement (noise reduction, clarity improvement)'
    )
    enhancement_group.add_argument(
        '--enhancement-batch-size',
        type=int,
        default=32,
        help='Batch size for audio enhancement processing (default: 32)'
    )
    enhancement_group.add_argument(
        '--enhancement-dashboard',
        action='store_true',
        help='Enable real-time dashboard for monitoring enhancement progress'
    )
    enhancement_group.add_argument(
        '--enhancement-level',
        type=str,
        choices=['mild', 'moderate', 'aggressive', 'ultra_aggressive', 
                 'selective_secondary_removal', 'pattern_metricgan_plus'],  # ADD NEW CHOICE
        default='moderate',
        help='Enhancement level: mild, moderate, aggressive, ultra_aggressive, selective_secondary_removal, or pattern_metricgan_plus (default: moderate)'
    )
    enhancement_group.add_argument(
        '--enhancement-gpu',
        action='store_true',
        help='Use GPU for audio enhancement (if available)'
    )
    enhancement_group.add_argument(
        '--enable-secondary-speaker-removal',
        action='store_true',
        help='Enable secondary speaker removal using SpeechBrain SepFormer'
    )
    enhancement_group.add_argument(
        '--use-audio-separator',
        action='store_true',
        help='Use audio-separator method instead of SpeechBrain (experimental)'
    )
    
    # 35dB SNR Enhancement arguments
    enhancement_35db_group = parser.add_argument_group('35dB SNR Enhancement')
    enhancement_35db_group.add_argument(
        '--enable-35db-enhancement',
        action='store_true',
        help='Enable 35dB SNR enhancement mode for voice cloning TTS training'
    )
    enhancement_35db_group.add_argument(
        '--target-snr',
        type=float,
        default=35.0,
        help='Target SNR in dB (default: 35.0)'
    )
    enhancement_35db_group.add_argument(
        '--min-acceptable-snr',
        type=float,
        default=30.0,
        help='Minimum acceptable SNR in dB for inclusion (default: 30.0)'
    )
    enhancement_35db_group.add_argument(
        '--snr-success-rate',
        type=float,
        default=0.90,
        help='Target success rate for achieving SNR (default: 0.90)'
    )
    enhancement_35db_group.add_argument(
        '--max-enhancement-iterations',
        type=int,
        default=3,
        help='Maximum enhancement iterations per sample (default: 3)'
    )
    enhancement_35db_group.add_argument(
        '--include-failed-samples',
        action='store_true',
        help='Include samples that fail to reach target SNR with metadata'
    )

    # NEW: Add Pattern→MetricGAN+ specific arguments
    pattern_metricgan_group = parser.add_argument_group('Pattern→MetricGAN+ Enhancement')
    pattern_metricgan_group.add_argument(
        '--pattern-confidence-threshold',
        type=float,
        default=0.8,
        help='Confidence threshold for pattern detection (default: 0.8)'
    )
    pattern_metricgan_group.add_argument(
        '--pattern-suppression-factor',
        type=float,
        default=0.15,
        help='Factor for pattern suppression - lower values suppress more (default: 0.15)'
    )
    pattern_metricgan_group.add_argument(
        '--pattern-padding-ms',
        type=int,
        default=50,
        help='Padding around detected patterns in milliseconds (default: 50)'
    )
    pattern_metricgan_group.add_argument(
        '--loudness-multiplier',
        type=float,
        default=1.6,
        help='Target loudness multiplier for enhancement (default: 1.6 = 160%%)'
    )
    pattern_metricgan_group.add_argument(
        '--disable-metricgan',
        action='store_true',
        help='Disable MetricGAN+ processing (use only pattern suppression and loudness)'
    )
    pattern_metricgan_group.add_argument(
        '--metricgan-device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device for MetricGAN+ processing (default: auto)'
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

    # NEW: Handle Pattern→MetricGAN+ compatibility and validation
    if hasattr(args, 'enhancement_level'):
        args.enhancement_level = validate_enhancement_level_compatibility(args.enhancement_level)
        
        # Auto-enable audio enhancement for Pattern→MetricGAN+
        if args.enhancement_level == 'pattern_metricgan_plus':
            args.enable_audio_enhancement = True
            logger.info("Auto-enabled audio enhancement for Pattern→MetricGAN+ level")
    
    # Validate Pattern→MetricGAN+ specific arguments
    if getattr(args, 'pattern_confidence_threshold', None) is not None:
        if not 0.5 <= args.pattern_confidence_threshold <= 1.0:
            parser.error("Pattern confidence threshold must be between 0.5 and 1.0")
    
    if getattr(args, 'pattern_suppression_factor', None) is not None:
        if not 0.0 <= args.pattern_suppression_factor <= 1.0:
            parser.error("Pattern suppression factor must be between 0.0 and 1.0")
    
    if getattr(args, 'loudness_multiplier', None) is not None:
        if not 1.0 <= args.loudness_multiplier <= 3.0:
            parser.error("Loudness multiplier must be between 1.0 and 3.0")

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
    
    # NEW: Add Pattern→MetricGAN+ configuration if enabled
    if merged_config.get("enhancement_level") == "pattern_metricgan_plus":
        from config import PATTERN_METRICGAN_CONFIG
        
        # Override with command-line arguments if provided
        pattern_config = dict(PATTERN_METRICGAN_CONFIG)
        
        # Update with CLI arguments
        if "pattern_confidence_threshold" in merged_config:
            pattern_config["pattern_detection"]["confidence_threshold"] = merged_config["pattern_confidence_threshold"]
        if "pattern_suppression_factor" in merged_config:
            pattern_config["pattern_suppression"]["suppression_factor"] = merged_config["pattern_suppression_factor"]
        if "pattern_padding_ms" in merged_config:
            pattern_config["pattern_suppression"]["padding_ms"] = merged_config["pattern_padding_ms"]
        if "loudness_multiplier" in merged_config:
            pattern_config["loudness_enhancement"]["target_multiplier"] = merged_config["loudness_multiplier"]
        if "disable_metricgan" in merged_config:
            pattern_config["metricgan"]["enabled"] = not merged_config["disable_metricgan"]
        if "metricgan_device" in merged_config:
            pattern_config["metricgan"]["device"] = merged_config["metricgan_device"]
        
        merged_config["pattern_metricgan_config"] = pattern_config
        logger.info(f"Pattern→MetricGAN+ configuration applied for {dataset_name}")

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
        
        # When resuming, we should append to avoid overwriting existing data
        append_mode = args.append or args.resume
        
        # Create checkpoint path for uploader
        uploader_checkpoint_path = os.path.join("checkpoints", f"{target_repo.replace('/', '_')}_upload_checkpoint.json")
        
        uploader = StreamingUploader(
            repo_id=target_repo,
            token=token,
            private=args.private,
            append_mode=append_mode,
            checkpoint_path=uploader_checkpoint_path
        )
    
    # Track global sample ID
    start_id = 1
    if args.append or args.resume:
        # Use the same target repository
        last_id = get_last_id(target_repo)
        if last_id is not None:
            start_id = last_id + 1
            if args.append:
                logger.info(f"Appending to existing dataset, starting from ID: S{start_id}")
            else:
                logger.info(f"Resuming from checkpoint, continuing from ID: S{start_id}")
    
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
            'fresh': args.fresh and not args.resume  # Don't reset speaker model when resuming
        }
        
        speaker_identifier = SpeakerIdentification(speaker_config)
        logger.info("Initialized speaker identification system")
    
    # Initialize audio enhancement if enabled
    audio_enhancer = None
    enhancement_metrics_collector = None
    enhancement_dashboard = None
    enhancement_buffer = []
    
    if args.enable_audio_enhancement or args.enable_35db_enhancement:
        from processors.audio_enhancement.core import AudioEnhancer
        from monitoring.metrics_collector import MetricsCollector
        from monitoring.dashboard import EnhancementDashboard
        
        # Configure workers based on CPU count for better performance
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        enhancement_config = {
            'use_gpu': args.enhancement_gpu,
            'fallback_to_cpu': True,
            'workers': max(4, cpu_count // 2),  # Use half of CPU cores, minimum 4
            'enhancement_level': args.enhancement_level,  # Pass the configured enhancement level
            'enable_35db_enhancement': args.enable_35db_enhancement  # Enable 35dB mode
        }
        
        # Update config if 35dB enhancement is enabled
        if args.enable_35db_enhancement:
            from config import ENHANCEMENT_35DB_CONFIG
            # Override with command-line arguments
            ENHANCEMENT_35DB_CONFIG['enabled'] = True
            ENHANCEMENT_35DB_CONFIG['target_snr_db'] = args.target_snr
            ENHANCEMENT_35DB_CONFIG['min_acceptable_snr_db'] = args.min_acceptable_snr
            ENHANCEMENT_35DB_CONFIG['target_success_rate'] = args.snr_success_rate
            ENHANCEMENT_35DB_CONFIG['processing']['max_iterations'] = args.max_enhancement_iterations
            ENHANCEMENT_35DB_CONFIG['fallback']['include_failed'] = args.include_failed_samples
            
            logger.info(f"35dB SNR enhancement enabled - Target: {args.target_snr}dB, Min: {args.min_acceptable_snr}dB")
        
        audio_enhancer = AudioEnhancer(**enhancement_config)
        logger.info(f"Initialized audio enhancement system (level: {args.enhancement_level})")
        
        # Initialize metrics collector
        metrics_dir = os.path.join(args.output or '.', 'enhancement_metrics')
        enhancement_metrics_collector = MetricsCollector(metrics_dir=metrics_dir)
        
        # Initialize dashboard if requested
        if args.enhancement_dashboard:
            enhancement_dashboard = EnhancementDashboard(
                metrics_collector=enhancement_metrics_collector,
                update_interval=10
            )
            logger.info("Started enhancement monitoring dashboard")
    
    # Helper function to check and upload batch
    def check_and_upload_batch(processor=None, last_sample_id=None, force_process_embeddings=False):
        nonlocal batch_buffer, embedding_buffer
        
        # CRITICAL FIX: Always process embeddings if we're about to upload
        # This ensures speaker clustering happens on full batches, not individual samples
        upload_will_trigger = len(batch_buffer) + len(embedding_buffer) >= args.upload_batch_size
        
        # Process any remaining embeddings before upload if forced or buffer is getting full
        if speaker_identifier and embedding_buffer and \
           (force_process_embeddings or upload_will_trigger):
            logger.info(f"Processing speaker identification batch of {len(embedding_buffer)} samples before upload")
            speaker_ids = speaker_identifier.process_batch(embedding_buffer)
            for sample, speaker_id in zip(embedding_buffer, speaker_ids):
                sample['speaker_id'] = speaker_id
                batch_buffer.append(sample)
            embedding_buffer = []
        
        if len(batch_buffer) >= args.upload_batch_size and uploader:
            # Filter out excluded samples if using 35dB enhancement
            if args.enable_35db_enhancement and not args.include_failed_samples:
                batch_buffer = [s for s in batch_buffer if not s.get('_exclude', False)]
            
            # Upload only the required batch size
            upload_batch = batch_buffer[:args.upload_batch_size]
            success, shard_name = uploader.upload_batch(upload_batch)
            if not success:
                logger.error(f"Failed to upload batch {shard_name}")
                return False
            
            # Save checkpoint if processor provided
            if processor and last_sample_id:
                processor.save_streaming_checkpoint(
                    shard_num=uploader.shard_num if uploader else 0,
                    samples_processed=total_samples,
                    last_sample_id=last_sample_id or upload_batch[-1]["ID"]
                )
            
            # Keep any remaining samples
            batch_buffer = batch_buffer[args.upload_batch_size:]
            return True
        return True
    
    # Process each dataset
    total_samples = 0
    total_duration_seconds = 0  # Track total duration across all datasets
    batch_buffer = []
    
    for dataset_name in dataset_names:
        try:
            logger.info(f"Processing {dataset_name} in streaming mode")
            
            # Reset clustering state for new dataset to prevent cross-dataset speaker merging
            if speaker_identifier:
                # Reset clustering state (embeddings) but preserve speaker counter
                # This prevents cross-dataset speaker matching while maintaining unique IDs
                speaker_identifier.reset_for_new_dataset(reset_counter=False)
                logger.info(f"Processing {dataset_name}, continuing from speaker ID: SPK_{speaker_identifier.speaker_counter:05d}")
            
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
                "stt_batch_size": args.stt_batch_size,
                "audio_enhancement": {
                    "enabled": args.enable_audio_enhancement,
                    "enhancer": audio_enhancer,
                    "metrics_collector": enhancement_metrics_collector,
                    "dashboard": enhancement_dashboard,
                    "batch_size": args.enhancement_batch_size if args.enable_audio_enhancement else None,
                    "level": args.enhancement_level if args.enable_audio_enhancement else None
                },
                # NEW: Add Pattern→MetricGAN+ specific configuration
                "enhancement_level": args.enhancement_level,
                "pattern_confidence_threshold": getattr(args, 'pattern_confidence_threshold', 0.8),
                "pattern_suppression_factor": getattr(args, 'pattern_suppression_factor', 0.15),
                "pattern_padding_ms": getattr(args, 'pattern_padding_ms', 50),
                "loudness_multiplier": getattr(args, 'loudness_multiplier', 1.6),
                "disable_metricgan": getattr(args, 'disable_metricgan', False),
                "metricgan_device": getattr(args, 'metricgan_device', 'auto')
            }
            
            # Create processor
            processor = create_processor(dataset_name, processor_config)
            
            # Get checkpoint if resuming
            checkpoint_file = get_checkpoint_file(args, dataset_name) if args.resume else None
            
            # Process all splits in streaming mode
            sample_count = 0
            dataset_duration_seconds = 0  # Track duration for this dataset
            for sample in processor.process_all_splits(
                checkpoint=checkpoint_file,
                sample_mode=args.sample,
                sample_size=args.sample_size
            ):
                # Assign sequential ID
                sample["ID"] = f"S{current_id}"
                current_id += 1
                
                # Handle audio enhancement with batch processing
                if audio_enhancer:
                    # Add to enhancement buffer for batch processing
                    enhancement_buffer.append(sample)
                    
                    # Process enhancement when buffer is full
                    if len(enhancement_buffer) >= args.enhancement_batch_size:
                        logger.info(f"Processing audio enhancement batch of {len(enhancement_buffer)} samples")
                        
                        try:
                            # Prepare batch
                            audio_batch = [
                                (s['audio']['array'], s['audio']['sampling_rate'], s['ID'])
                                for s in enhancement_buffer
                            ]
                            
                            # Process batch
                            if args.enable_35db_enhancement:
                                # Use 35dB enhancement mode
                                enhanced_results = []
                                for audio, sr, identifier in audio_batch:
                                    enhanced, metadata = audio_enhancer.enhance_to_target_snr(audio, sr, args.target_snr)
                                    enhanced_results.append((enhanced, metadata))
                            else:
                                # Use standard enhancement
                                enhanced_results = audio_enhancer.process_batch(
                                    audio_batch
                                    # Uses configured workers from AudioEnhancer
                                )
                            
                            # Update samples with enhanced audio
                            for sample_to_enhance, (enhanced_audio, metadata) in zip(enhancement_buffer, enhanced_results):
                                sample_to_enhance['audio']['array'] = enhanced_audio
                                sample_to_enhance['enhancement_metadata'] = metadata
                                
                                # Add SNR field if using 35dB enhancement
                                if args.enable_35db_enhancement and 'snr_db' in metadata:
                                    sample_to_enhance['snr_db'] = metadata['snr_db']
                                    
                                    # Optionally exclude samples below minimum SNR
                                    if not args.include_failed_samples and metadata['snr_db'] < args.min_acceptable_snr:
                                        # Mark for exclusion
                                        sample_to_enhance['_exclude'] = True
                                
                                # Add metrics to collector
                                if enhancement_metrics_collector:
                                    metrics_dict = {
                                        'snr_improvement': metadata.get('snr_improvement', 0),
                                        'pesq_score': metadata.get('pesq', 0),
                                        'stoi_score': metadata.get('stoi', 0),
                                        'processing_time': metadata.get('processing_time', 0),
                                        'noise_level': metadata.get('noise_level', 'unknown'),
                                        'audio_file': sample_to_enhance['ID']
                                    }
                                    enhancement_metrics_collector.add_sample(sample_to_enhance['ID'], metrics_dict)
                                
                                # Update dashboard
                                if enhancement_dashboard:
                                    enhancement_dashboard.update_progress(sample_to_enhance['ID'], success=True)
                            
                            # Move enhanced samples to speaker identification
                            if speaker_identifier:
                                embedding_buffer.extend(enhancement_buffer)
                                
                                # Check if we should process speaker IDs after adding enhanced samples
                                # This ensures we process speaker IDs in batches, not individually
                                if len(embedding_buffer) >= args.speaker_batch_size:
                                    logger.info(f"Processing speaker identification batch of {len(embedding_buffer)} samples after enhancement")
                                    speaker_ids = speaker_identifier.process_batch(embedding_buffer)
                                    for sample, speaker_id in zip(embedding_buffer, speaker_ids):
                                        sample['speaker_id'] = speaker_id
                                        batch_buffer.append(sample)
                                    embedding_buffer = []
                            else:
                                # No speaker identification - assign IDs and add to batch buffer
                                for enhanced_sample in enhancement_buffer:
                                    enhanced_sample['speaker_id'] = f"SPK_{total_samples + 1:05d}"
                                    batch_buffer.append(enhanced_sample)
                            
                            enhancement_buffer = []
                            
                        except Exception as e:
                            logger.error(f"Batch audio enhancement failed: {str(e)}")
                            # Fall back to original audio for the batch
                            for sample_to_enhance in enhancement_buffer:
                                if enhancement_dashboard:
                                    enhancement_dashboard.update_progress(sample_to_enhance['ID'], success=False)
                                # Move to next stage with original audio
                                if speaker_identifier:
                                    embedding_buffer.append(sample_to_enhance)
                                else:
                                    sample_to_enhance['speaker_id'] = f"SPK_{total_samples + 1:05d}"
                                    batch_buffer.append(sample_to_enhance)
                            enhancement_buffer = []
                            
                            # Check if we should process speaker IDs after fallback
                            if speaker_identifier and len(embedding_buffer) >= args.speaker_batch_size:
                                logger.info(f"Processing speaker identification batch of {len(embedding_buffer)} samples after enhancement fallback")
                                speaker_ids = speaker_identifier.process_batch(embedding_buffer)
                                for sample, speaker_id in zip(embedding_buffer, speaker_ids):
                                    sample['speaker_id'] = speaker_id
                                    batch_buffer.append(sample)
                                embedding_buffer = []
                            
                            # Check if we should upload after processing enhancement batch
                            if not check_and_upload_batch():
                                return EXIT_CODES["UPLOAD_ERROR"]
                
                elif audio_enhancer:
                    # Sample is still in enhancement buffer, skip rest of processing
                    continue
                
                # Handle speaker identification (only if not in enhancement buffer)
                if speaker_identifier and not (audio_enhancer and len(enhancement_buffer) > 0):
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
                        
                        # Check if we should upload after processing speaker batch
                        if not check_and_upload_batch():
                            return EXIT_CODES["UPLOAD_ERROR"]
                    # else: sample stays in embedding_buffer until batch is full or end of processing
                elif not speaker_identifier:
                    # No speaker identification - assign unique ID based on sample counter
                    sample['speaker_id'] = f"SPK_{total_samples + 1:05d}"
                    batch_buffer.append(sample)
                # else: sample is being processed by speaker identification or audio enhancement
                
                sample_count += 1
                total_samples += 1
                
                # Accumulate duration
                sample_duration = sample.get('length', 0) or 0
                dataset_duration_seconds += sample_duration
                total_duration_seconds += sample_duration
                
                # Only check upload after meaningful progress to ensure proper batching
                # This prevents premature uploads that break speaker clustering
                should_check_upload = (
                    # Don't check too early to allow S1-S10 clustering
                    (sample_count >= 50 and sample_count % 25 == 0) or
                    # Always check if batch buffer is getting too full
                    len(batch_buffer) >= args.upload_batch_size * 0.9
                )
                
                if should_check_upload:
                    # Upload batch when buffer is full
                    if not check_and_upload_batch(processor, sample["ID"]):
                        return EXIT_CODES["UPLOAD_ERROR"]
                
                # Log progress
                if sample_count % 1000 == 0:
                    logger.info(f"Processed {sample_count} samples from {dataset_name}, total: {total_samples}")
            
            # Process any remaining enhancement buffer for this dataset
            if audio_enhancer and enhancement_buffer:
                logger.info(f"Processing final enhancement batch for {dataset_name}: {len(enhancement_buffer)} samples")
                try:
                    # Prepare batch
                    audio_batch = [
                        (s['audio']['array'], s['audio']['sampling_rate'], s['ID'])
                        for s in enhancement_buffer
                    ]
                    
                    # Process batch
                    if args.enable_35db_enhancement:
                        # Use 35dB enhancement mode
                        enhanced_results = []
                        for audio, sr, identifier in audio_batch:
                            enhanced, metadata = audio_enhancer.enhance_to_target_snr(audio, sr, args.target_snr)
                            enhanced_results.append((enhanced, metadata))
                    else:
                        # Use standard enhancement
                        enhanced_results = audio_enhancer.process_batch(
                            audio_batch
                            # Uses configured workers from AudioEnhancer
                        )
                    
                    # Update samples with enhanced audio
                    for sample_to_enhance, (enhanced_audio, metadata) in zip(enhancement_buffer, enhanced_results):
                        sample_to_enhance['audio']['array'] = enhanced_audio
                        sample_to_enhance['enhancement_metadata'] = metadata
                        
                        # Add SNR field if using 35dB enhancement
                        if args.enable_35db_enhancement and 'snr_db' in metadata:
                            sample_to_enhance['snr_db'] = metadata['snr_db']
                            
                            # Optionally exclude samples below minimum SNR
                            if not args.include_failed_samples and metadata['snr_db'] < args.min_acceptable_snr:
                                # Mark for exclusion
                                sample_to_enhance['_exclude'] = True
                        
                        # Add metrics to collector
                        if enhancement_metrics_collector:
                            metrics_dict = {
                                'snr_improvement': metadata.get('snr_improvement', 0),
                                'pesq_score': metadata.get('pesq', 0),
                                'stoi_score': metadata.get('stoi', 0),
                                'processing_time': metadata.get('processing_time', 0),
                                'noise_level': metadata.get('noise_level', 'unknown'),
                                'audio_file': sample_to_enhance['ID']
                            }
                            enhancement_metrics_collector.add_sample(sample_to_enhance['ID'], metrics_dict)
                        
                        # Update dashboard
                        if enhancement_dashboard:
                            enhancement_dashboard.update_progress(sample_to_enhance['ID'], success=True)
                    
                    # Move enhanced samples to speaker identification
                    if speaker_identifier:
                        embedding_buffer.extend(enhancement_buffer)
                        
                        # Check if we should process speaker IDs after adding enhanced samples
                        # This ensures we process speaker IDs in batches, not individually
                        if len(embedding_buffer) >= args.speaker_batch_size:
                            logger.info(f"Processing speaker identification batch of {len(embedding_buffer)} samples after final enhancement")
                            speaker_ids = speaker_identifier.process_batch(embedding_buffer)
                            for sample, speaker_id in zip(embedding_buffer, speaker_ids):
                                sample['speaker_id'] = speaker_id
                                batch_buffer.append(sample)
                            embedding_buffer = []
                    else:
                        # No speaker identification - assign IDs and add to batch buffer
                        for enhanced_sample in enhancement_buffer:
                            enhanced_sample['speaker_id'] = f"SPK_{total_samples + 1:05d}"
                            batch_buffer.append(enhanced_sample)
                    
                except Exception as e:
                    logger.error(f"Final batch audio enhancement failed: {str(e)}")
                    # Fall back to original audio for the batch
                    for sample_to_enhance in enhancement_buffer:
                        if enhancement_dashboard:
                            enhancement_dashboard.update_progress(sample_to_enhance['ID'], success=False)
                        # Move to next stage with original audio
                        if speaker_identifier:
                            embedding_buffer.append(sample_to_enhance)
                        else:
                            sample_to_enhance['speaker_id'] = f"SPK_{total_samples + 1:05d}"
                            batch_buffer.append(sample_to_enhance)
                
                enhancement_buffer = []  # Clear buffer for next dataset
            
            # Process any remaining embeddings for this dataset before moving to next
            if speaker_identifier and embedding_buffer:
                logger.info(f"Processing final speaker batch for {dataset_name}: {len(embedding_buffer)} samples")
                speaker_ids = speaker_identifier.process_batch(embedding_buffer)
                
                for sample, speaker_id in zip(embedding_buffer, speaker_ids):
                    sample['speaker_id'] = speaker_id
                    batch_buffer.append(sample)
                
                embedding_buffer = []  # Clear buffer for next dataset
            
            # Force upload check at end of dataset to clear buffers
            if not check_and_upload_batch(processor, f"S{current_id-1}", force_process_embeddings=True):
                return EXIT_CODES["UPLOAD_ERROR"]
            
            dataset_duration_hours = dataset_duration_seconds / 3600.0
            logger.info(f"Completed {dataset_name}: {sample_count} samples, {dataset_duration_hours:.2f} hours")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name} in streaming mode: {str(e)}")
            if args.verbose:
                logger.exception(e)
            # Continue with next dataset instead of failing completely
    
    # Process any remaining enhancement buffer (should be empty but check to be safe)
    if audio_enhancer and enhancement_buffer:
        logger.warning(f"Unexpected: {len(enhancement_buffer)} samples remain in enhancement buffer")
        try:
            # Process remaining samples
            audio_batch = [
                (s['audio']['array'], s['audio']['sampling_rate'], s['ID'])
                for s in enhancement_buffer
            ]
            
            if args.enable_35db_enhancement:
                # Use 35dB enhancement mode
                enhanced_results = []
                for audio, sr, identifier in audio_batch:
                    enhanced, metadata = audio_enhancer.enhance_to_target_snr(audio, sr, args.target_snr)
                    enhanced_results.append((enhanced, metadata))
            else:
                # Use standard enhancement
                enhanced_results = audio_enhancer.process_batch(
                    audio_batch
                    # Uses configured workers from AudioEnhancer
                )
            
            for sample_to_enhance, (enhanced_audio, metadata) in zip(enhancement_buffer, enhanced_results):
                sample_to_enhance['audio']['array'] = enhanced_audio
                sample_to_enhance['enhancement_metadata'] = metadata
                
                # Add SNR field if using 35dB enhancement
                if args.enable_35db_enhancement and 'snr_db' in metadata:
                    sample_to_enhance['snr_db'] = metadata['snr_db']
                    
                    # Optionally exclude samples below minimum SNR
                    if not args.include_failed_samples and metadata['snr_db'] < args.min_acceptable_snr:
                        # Mark for exclusion
                        sample_to_enhance['_exclude'] = True
                
                if enhancement_metrics_collector:
                    metrics_dict = {
                        'snr_improvement': metadata.get('snr_improvement', 0),
                        'pesq_score': metadata.get('pesq', 0),
                        'stoi_score': metadata.get('stoi', 0),
                        'processing_time': metadata.get('processing_time', 0),
                        'noise_level': metadata.get('noise_level', 'unknown'),
                        'audio_file': sample_to_enhance['ID']
                    }
                    enhancement_metrics_collector.add_sample(sample_to_enhance['ID'], metrics_dict)
                
                if speaker_identifier:
                    embedding_buffer.append(sample_to_enhance)
                else:
                    sample_to_enhance['speaker_id'] = f"SPK_{total_samples + 1:05d}"
                    batch_buffer.append(sample_to_enhance)
                    
        except Exception as e:
            logger.error(f"Final enhancement processing failed: {str(e)}")
            # Fall back to original audio
            for sample_to_enhance in enhancement_buffer:
                if speaker_identifier:
                    embedding_buffer.append(sample_to_enhance)
                else:
                    sample_to_enhance['speaker_id'] = f"SPK_{total_samples + 1:05d}"
                    batch_buffer.append(sample_to_enhance)
    
    # Process remaining embeddings (should be empty now since we process at end of each dataset)
    if speaker_identifier and embedding_buffer:
        logger.warning(f"Unexpected: {len(embedding_buffer)} samples remain in embedding buffer")
        speaker_ids = speaker_identifier.process_batch(embedding_buffer)
        
        for sample, speaker_id in zip(embedding_buffer, speaker_ids):
            sample['speaker_id'] = speaker_id
            batch_buffer.append(sample)
        embedding_buffer = []
    
    # Upload remaining samples with forced embedding processing
    if batch_buffer and uploader:
        # Make sure any lingering embeddings are processed first
        if speaker_identifier and embedding_buffer:
            check_and_upload_batch(force_process_embeddings=True)
        
        # Filter out excluded samples if using 35dB enhancement
        if args.enable_35db_enhancement and not args.include_failed_samples:
            batch_buffer = [s for s in batch_buffer if not s.get('_exclude', False)]
            logger.info(f"Final batch after filtering: {len(batch_buffer)} samples")
        
        success, shard_name = uploader.upload_batch(batch_buffer)
        if not success:
            logger.error(f"Failed to upload final batch {shard_name}")
            return EXIT_CODES["UPLOAD_ERROR"]
    
    # Clean up speaker identifier
    if speaker_identifier:
        speaker_identifier.cleanup()
    
    # Clean up audio enhancement components
    # AudioEnhancer doesn't have a cleanup method
    
    if enhancement_dashboard:
        enhancement_dashboard.stop()
    
    if enhancement_metrics_collector:
        summary_path = os.path.join(enhancement_metrics_collector.metrics_dir, 'summary.json')
        enhancement_metrics_collector.export_to_json(summary_path)
        logger.info(f"Enhancement metrics saved to {summary_path}")
    
    # Upload dataset card
    total_duration_hours = total_duration_seconds / 3600.0
    if uploader:
        dataset_info = {
            "name": TARGET_DATASET["name"],
            "description": TARGET_DATASET["description"],
            "total_samples": total_samples,
            "total_duration_hours": total_duration_hours,
            "sources_description": f"Processed {len(dataset_names)} datasets in streaming mode",
            "dataset_names": dataset_names
        }
        uploader.upload_dataset_card(dataset_info)
    
    logger.info(f"Streaming processing completed: {total_samples} total samples, {total_duration_hours:.2f} hours")
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
                "stt_batch_size": args.stt_batch_size,
                # Audio enhancement configuration
                "enable_audio_enhancement": args.enable_audio_enhancement,
                "enhancement_level": args.enhancement_level,
                "enhancement_gpu": args.enhancement_gpu,
                "enable_secondary_speaker_removal": args.enable_secondary_speaker_removal,
                "use_audio_separator": args.use_audio_separator,
                "enable_35db_enhancement": args.enable_35db_enhancement,
                "target_snr": args.target_snr,
                # NEW: Add Pattern→MetricGAN+ specific configuration
                "pattern_confidence_threshold": getattr(args, 'pattern_confidence_threshold', 0.8),
                "pattern_suppression_factor": getattr(args, 'pattern_suppression_factor', 0.15),
                "pattern_padding_ms": getattr(args, 'pattern_padding_ms', 50),
                "loudness_multiplier": getattr(args, 'loudness_multiplier', 1.6),
                "disable_metricgan": getattr(args, 'disable_metricgan', False),
                "metricgan_device": getattr(args, 'metricgan_device', 'auto')
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
        
        logger.info("Applying speaker identification per dataset to ensure separation")
        
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
            'fresh': args.fresh and not args.resume  # Don't reset speaker model when resuming
        }
        
        speaker_identifier = SpeakerIdentification(speaker_config)
        
        # Process speaker identification PER DATASET to ensure no cross-dataset clustering
        dataset_start_idx = 0
        for dataset_name, samples in processed_datasets.items():
            logger.info(f"Processing speaker identification for {dataset_name}")
            
            # Reset clustering state but preserve counter to prevent cross-dataset matching
            speaker_identifier.reset_for_new_dataset(reset_counter=False)
            
            # Find samples from this dataset in combined_samples
            dataset_samples = []
            dataset_indices = []
            for idx, sample in enumerate(combined_samples[dataset_start_idx:dataset_start_idx + len(samples)]):
                dataset_samples.append(sample)
                dataset_indices.append(dataset_start_idx + idx)
            
            # Process in batches for this dataset
            for i in range(0, len(dataset_samples), args.speaker_batch_size):
                batch = dataset_samples[i:i + args.speaker_batch_size]
                batch_indices = dataset_indices[i:i + args.speaker_batch_size]
                logger.info(f"Processing {dataset_name} speaker batch {i//args.speaker_batch_size + 1}")
                
                speaker_ids = speaker_identifier.process_batch(batch)
                
                # Assign speaker IDs
                for idx, speaker_id in zip(batch_indices, speaker_ids):
                    combined_samples[idx]['speaker_id'] = speaker_id
            
            dataset_start_idx += len(samples)
        
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
