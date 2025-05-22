#!/usr/bin/env python3
"""
Main entry point for the Thai Audio Dataset Collection project.
"""

import os
import sys
import argparse
import logging
import importlib
from typing import List, Dict, Any, Optional, Tuple

from config import (
    DATASET_CONFIG, TARGET_DATASET, LOG_CONFIG, EXIT_CODES,
    HF_TOKEN_FILE, CHECKPOINT_DIR, LOG_DIR
)
from utils.logging import setup_logging
from utils.huggingface import read_hf_token, authenticate_hf, create_hf_dataset, upload_dataset, get_last_id
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
    parser.add_argument("--sample-size", type=int, default=5, help="Number of samples to process in sample mode (default: 5)")

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
        # Extract the base name without 'Processor' suffix
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
    sample_size: int = 5
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

    # Get dataset info
    dataset_info = processor.get_dataset_info()
    logger.info(f"Dataset info: {dataset_info}")

    # Process dataset
    samples = list(processor.process(
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

    # Process datasets
    processed_datasets = {}

    for dataset_name in dataset_names:
        try:
            # Get checkpoint file
            checkpoint_file = get_checkpoint_file(args, dataset_name)

            # Create processor
            processor = create_processor(dataset_name, {
                "checkpoint_dir": CHECKPOINT_DIR,
                "log_dir": LOG_DIR
            })

            # Process dataset
            samples = process_dataset(
                processor,
                checkpoint_file,
                sample_mode=args.sample,
                sample_size=args.sample_size
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
        # Get last ID from existing dataset
        last_id = get_last_id(TARGET_DATASET["name"])
        if last_id is not None:
            start_id = last_id + 1
            logger.info(f"Appending to existing dataset, starting from ID: S{start_id}")
        else:
            logger.warning(f"Could not determine last ID from existing dataset, starting from S1")

    combined_samples = combine_datasets(processed_datasets, start_id)

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

        # Upload dataset
        if not upload_dataset(dataset, TARGET_DATASET["name"], private=args.private, token=token):
            logger.error("Failed to upload dataset to Huggingface")
            return EXIT_CODES["UPLOAD_ERROR"]

        logger.info(f"Successfully uploaded dataset to {TARGET_DATASET['name']}")

    logger.info("Thai Audio Dataset Collection completed successfully")
    return EXIT_CODES["SUCCESS"]

if __name__ == "__main__":
    sys.exit(main())
