"""
Huggingface interaction utilities for the Thai Audio Dataset Collection
project.
"""

import os
import logging
from typing import Optional, Dict, Any, Iterable
import tempfile
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)

try:
    from datasets import Dataset, Audio, load_dataset, Features, Value
    from huggingface_hub import login, HfApi
except ImportError:
    logger.error(
        "Required Huggingface libraries not installed. "
        "Please install datasets and huggingface_hub."
    )
    raise


def read_hf_token(token_file: str) -> Optional[str]:
    """
    Read Huggingface token from file.

    Args:
        token_file: Path to token file

    Returns:
        str: Huggingface token or None if file not found
    """
    try:
        if not os.path.exists(token_file):
            logger.error(f"Huggingface token file not found: {token_file}")
            return None

        with open(token_file, 'r') as f:
            token = f.read().strip()

        if not token:
            logger.error(f"Huggingface token file is empty: {token_file}")
            return None

        return token
    except Exception as e:
        logger.error(f"Error reading Huggingface token: {str(e)}")
        return None


def authenticate_hf(token: str) -> bool:
    """
    Authenticate with Huggingface.

    Args:
        token: Huggingface token

    Returns:
        bool: True if authentication successful, False otherwise
    """
    try:
        login(token=token)
        logger.info("Successfully authenticated with Huggingface")
        return True
    except Exception as e:
        logger.error(f"Huggingface authentication error: {str(e)}")
        return False


def create_hf_dataset(
    samples: Iterable[Dict[str, Any]],
    features: Optional[Dict[str, Any]] = None
) -> Optional[Dataset]:
    """
    Create a Huggingface dataset from samples.

    Args:
        samples: Iterable of samples
        features: Dataset features (optional)

    Returns:
        Dataset: Huggingface dataset or None if creation failed
    """
    try:
        # Convert samples to list if not already
        samples_list = list(samples)

        if not samples_list:
            logger.error("No samples provided for dataset creation")
            return None

        # Create default features if not provided
        if features is None:
            # Define features explicitly to ensure proper Audio type
            features = Features({
                "ID": Value("string"),
                "speaker_id": Value("string"),
                "Language": Value("string"),
                "audio": Audio(sampling_rate=16000),  # Explicit audio feature
                "transcript": Value("string"),
                "length": Value("float32"),
                "dataset_name": Value("string"),
                "confidence_score": Value("float32")  # Changed from float64 to float32
            })

        # Create dataset with explicit features
        # Get all keys from features
        if features:
            feature_keys = list(features.keys())
        else:
            feature_keys = [
                "ID", "speaker_id", "Language", "audio",
                "transcript", "length", "dataset_name", "confidence_score"
            ]
        data_dict = {
            key: [sample.get(key) for sample in samples_list]
            for key in feature_keys
        }
        dataset = Dataset.from_dict(data_dict, features=features)

        logger.info(f"Created dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Dataset creation error: {str(e)}")
        return None


def upload_dataset(
    dataset: Dataset,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None
) -> bool:
    """
    Upload dataset to Huggingface.

    Args:
        dataset: Huggingface dataset
        repo_id: Repository ID (e.g., "username/dataset-name")
        private: Whether the repository should be private
        token: Huggingface token (optional if already authenticated)

    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        # Push to hub
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token
        )

        logger.info(f"Successfully uploaded dataset to {repo_id}")
        return True
    except Exception as e:
        logger.error(f"Dataset upload error: {str(e)}")
        return False


def get_dataset_info(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a Huggingface dataset.

    Args:
        dataset_name: Dataset name (e.g., "username/dataset-name")

    Returns:
        dict: Dataset information or None if retrieval failed
    """
    try:
        # Load dataset info
        dataset = load_dataset(dataset_name, split="train")

        # Get dataset info
        info = {
            "name": dataset_name,
            "num_samples": len(dataset),
            "features": list(dataset.features.keys()),
            "size_in_bytes": (
                dataset.info.size_in_bytes
                if hasattr(dataset.info, "size_in_bytes")
                else None
            )
        }

        return info
    except Exception as e:
        logger.error(
            f"Error getting dataset info for {dataset_name}: {str(e)}"
        )
        return None


def get_last_id(dataset_name: str) -> Optional[int]:
    """
    Get the last ID from an existing dataset.

    Args:
        dataset_name: Dataset name (e.g., "username/dataset-name")

    Returns:
        int: Numeric part of the last ID or None if retrieval failed
    """
    # Try new approach first: read parquet files directly
    try:
        logger.info(
            f"Attempting to get last ID from {dataset_name} using parquet file approach"
        )
        
        # Initialize HfApi
        api = HfApi()
        
        # List all parquet files in the dataset
        try:
            files = api.list_repo_files(
                repo_id=dataset_name,
                repo_type="dataset"
            )
            
            # Filter for parquet files in the train split
            parquet_files = [
                f for f in files 
                if f.endswith('.parquet') and 'train' in f
            ]
            
            if not parquet_files:
                logger.warning(
                    f"No parquet files found in {dataset_name}, "
                    "falling back to load_dataset approach"
                )
                raise ValueError("No parquet files found")
            
            logger.info(f"Found {len(parquet_files)} parquet files to process")
            
            # Track max ID across all parquet files
            max_id = 0
            
            # Process each parquet file
            for parquet_file in parquet_files:
                try:
                    # Download parquet file to temporary location
                    with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp_file:
                        # Download the file
                        file_path = api.hf_hub_download(
                            repo_id=dataset_name,
                            filename=parquet_file,
                            repo_type="dataset",
                            local_dir=tempfile.gettempdir(),
                            force_download=True
                        )
                        
                        # Read parquet file
                        df = pd.read_parquet(file_path)
                        
                        # Check if ID column exists
                        if 'ID' not in df.columns:
                            logger.warning(
                                f"ID column not found in {parquet_file}, skipping"
                            )
                            continue
                        
                        # Extract numeric IDs
                        for id_str in df['ID']:
                            if (
                                id_str 
                                and isinstance(id_str, str) 
                                and id_str.startswith('S') 
                                and id_str[1:].isdigit()
                            ):
                                numeric_id = int(id_str[1:])
                                max_id = max(max_id, numeric_id)
                        
                        # Clean up downloaded file
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            
                except Exception as e:
                    logger.warning(
                        f"Error processing parquet file {parquet_file}: {str(e)}"
                    )
                    continue
            
            if max_id > 0:
                logger.info(
                    f"Successfully found max ID {max_id} from parquet files"
                )
                return max_id
            else:
                logger.warning(
                    "No valid IDs found in parquet files, "
                    "falling back to load_dataset approach"
                )
                raise ValueError("No valid IDs found in parquet files")
                
        except Exception as e:
            logger.warning(
                f"Error with parquet approach: {str(e)}, "
                "falling back to load_dataset approach"
            )
            
    except Exception as e:
        logger.warning(
            f"Parquet approach failed: {str(e)}, "
            "using fallback load_dataset approach"
        )
    
    # Fallback to original approach using load_dataset
    try:
        logger.info("Using fallback load_dataset approach")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")

        # Check if dataset has ID field
        if "ID" not in dataset.features:
            logger.error(f"Dataset {dataset_name} does not have an ID field")
            return None

        # Get all IDs
        ids = dataset["ID"]

        # Extract numeric parts
        numeric_ids = []
        for id_str in ids:
            if id_str and id_str.startswith('S') and id_str[1:].isdigit():
                numeric_ids.append(int(id_str[1:]))

        if not numeric_ids:
            logger.warning(f"No valid IDs found in dataset {dataset_name}")
            return 0

        # Return max ID
        return max(numeric_ids)
    except Exception as e:
        error_str = str(e)
        # Check if this is a SplitInfo mismatch error
        if (
            "SplitInfo" in error_str
            and "expected" in error_str
            and "recorded" in error_str
        ):
            logger.info(
                f"SplitInfo mismatch detected for {dataset_name}, "
                "retrying with force_redownload"
            )
            try:
                # Retry with force_redownload to bypass cached split info
                dataset = load_dataset(
                    dataset_name,
                    split="train",
                    download_mode="force_redownload"
                )

                # Check if dataset has ID field
                if "ID" not in dataset.features:
                    logger.error(
                        f"Dataset {dataset_name} does not have an ID field"
                    )
                    return None

                # Get all IDs
                ids = dataset["ID"]

                # Extract numeric parts
                numeric_ids = []
                for id_str in ids:
                    if (
                        id_str
                        and id_str.startswith('S')
                        and id_str[1:].isdigit()
                    ):
                        numeric_ids.append(int(id_str[1:]))

                if not numeric_ids:
                    logger.warning(
                        f"No valid IDs found in dataset {dataset_name}"
                    )
                    return 0

                # Return max ID
                return max(numeric_ids)
            except Exception as retry_e:
                logger.error(
                    f"Error getting last ID from dataset {dataset_name} "
                    f"after retry: {str(retry_e)}"
                )
                return None
        else:
            # For other errors, log and return None
            logger.error(
                f"Error getting last ID from dataset {dataset_name}: {str(e)}"
            )
            return None
