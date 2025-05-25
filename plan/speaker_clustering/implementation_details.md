# Speaker Clustering Implementation Details

## Overview

This document provides detailed implementation guidance for adding speaker identification and clustering to the Thai Audio Dataset Collection project. The system will process 10M+ audio files to identify and group speakers while maintaining high accuracy and scalability.

## Requirements Summary

### Scale & Performance
- **Dataset Size**: 10M+ audio files (growing)
- **Audio Length**: 2-5 seconds typically
- **Hardware**: RTX 5090 (32GB VRAM), 128GB RAM
- **Processing**: Overnight batch processing acceptable
- **Storage**: ~100GB available for embeddings

### Quality Constraints
- **Preference**: Over-segmentation (avoid merging different speakers)
- **Uncertainty Handling**: Assign unique speaker IDs to uncertain clusters
- **Global Uniqueness**: Speaker IDs must be unique across all datasets
- **Append Compatibility**: Must work with `--append` flag for incremental updates

## Schema Changes

### Updated Dataset Schema

```python
# config.py
TARGET_SCHEMA = {
    "ID": str,           # Sequential identifier: S1, S2, S3...
    "speaker_id": str,   # Speaker group identifier: SPK_00001, SPK_00002...
    "Language": str,     # Language code: "th"
    "audio": dict,       # HuggingFace audio format
    "transcript": str,   # Transcript text
    "length": float,     # Audio duration in seconds
    "dataset_name": str, # Source dataset name
    "confidence_score": float  # Transcript confidence
}
```

## Architecture Design

### Component Overview

1. **Speaker Identification Module** (`processors/speaker_identification.py`)
   - Embedding extraction using pyannote/embedding
   - Incremental HDBSCAN clustering
   - Persistent model management for append mode

2. **Streaming Integration**
   - Extract embeddings during audio processing
   - Buffer embeddings until batch size reached
   - Cluster and assign speaker IDs before upload

3. **Storage System**
   - Optional embedding storage (HDF5/Parquet format)
   - Speaker model persistence for append operations
   - Efficient memory management

### Processing Pipeline

```
Audio Stream → Embedding Extraction → Buffer Management → HDBSCAN Clustering → Speaker ID Assignment → Upload
```

## Implementation Steps

### 1. Create Speaker Identification Module

```python
# processors/speaker_identification.py
import torch
import numpy as np
import h5py
from pyannote.audio import Model
import hdbscan
from typing import List, Dict, Optional, Tuple
import logging
import json
import os

logger = logging.getLogger(__name__)

class SpeakerIdentification:
    """Speaker identification and clustering for audio samples."""
    
    def __init__(self, config: Dict):
        """Initialize speaker identification system.
        
        Args:
            config: Configuration dictionary with the following keys:
                - model: Embedding model name (default: 'pyannote/embedding')
                - batch_size: Clustering batch size (default: 10000)
                - min_cluster_size: Minimum cluster size (default: 15)
                - min_samples: Minimum samples for core points (default: 10)
                - epsilon: Cluster selection epsilon (default: 0.3)
                - threshold: Similarity threshold for existing clusters (default: 0.7)
                - store_embeddings: Whether to store embeddings (default: False)
                - model_path: Path to save/load speaker model
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load embedding model
        logger.info(f"Loading embedding model: {config.get('model', 'pyannote/embedding')}")
        self.model = Model.from_pretrained(
            config.get('model', 'pyannote/embedding')
        ).to(self.device)
        
        # Clustering parameters (conservative for over-segmentation)
        self.clustering_params = {
            'min_cluster_size': config.get('min_cluster_size', 15),
            'min_samples': config.get('min_samples', 10),
            'metric': 'cosine',
            'cluster_selection_epsilon': config.get('epsilon', 0.3),
            'cluster_selection_method': 'eom',
            'prediction_data': True
        }
        
        # Storage settings
        self.store_embeddings = config.get('store_embeddings', False)
        self.embedding_file = None
        if self.store_embeddings:
            embedding_path = config.get('embedding_path', 'speaker_embeddings.h5')
            self.embedding_file = h5py.File(embedding_path, 'a')
        
        # Speaker tracking
        self.speaker_counter = 1
        self.existing_clusters = {}
        self.model_path = config.get('model_path', 'speaker_model.json')
        
        # Load existing model if available
        self.load_model()
    
    def extract_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract speaker embedding from audio.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Speaker embedding as numpy array
        """
        with torch.no_grad():
            # Prepare audio for model
            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Extract embedding
            embedding = self.model({
                'waveform': audio_tensor,
                'sample_rate': sample_rate
            })
            
            return embedding.cpu().numpy().squeeze()
    
    def process_batch(self, audio_samples: List[Dict]) -> List[str]:
        """Process a batch of audio samples and assign speaker IDs.
        
        Args:
            audio_samples: List of audio samples with 'audio' and 'ID' keys
            
        Returns:
            List of speaker IDs corresponding to input samples
        """
        # Extract embeddings
        embeddings = []
        sample_ids = []
        
        for sample in audio_samples:
            try:
                embedding = self.extract_embedding(
                    sample['audio']['array'],
                    sample['audio']['sampling_rate']
                )
                embeddings.append(embedding)
                sample_ids.append(sample['ID'])
            except Exception as e:
                logger.error(f"Failed to extract embedding for {sample['ID']}: {str(e)}")
                embeddings.append(np.zeros(512))  # Dummy embedding
                sample_ids.append(sample['ID'])
        
        embeddings = np.array(embeddings)
        
        # Cluster embeddings
        speaker_ids = self.cluster_embeddings(embeddings, sample_ids)
        
        return speaker_ids
    
    def cluster_embeddings(self, embeddings: np.ndarray, sample_ids: List[str]) -> List[str]:
        """Cluster embeddings and assign speaker IDs.
        
        Args:
            embeddings: Array of speaker embeddings
            sample_ids: List of sample IDs
            
        Returns:
            List of speaker IDs
        """
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(**self.clustering_params)
        labels = clusterer.fit_predict(embeddings)
        
        # Handle existing clusters (for append mode)
        if self.existing_clusters:
            labels = self._merge_with_existing(embeddings, labels, clusterer)
        
        # Convert labels to speaker IDs
        speaker_ids = []
        for i, label in enumerate(labels):
            if label == -1:  # Noise/uncertain cluster
                # Assign unique speaker ID
                speaker_id = f"SPK_{self.speaker_counter:05d}"
                self.speaker_counter += 1
            else:
                # Check if this label already has a speaker ID
                if label not in self.existing_clusters:
                    self.existing_clusters[label] = f"SPK_{self.speaker_counter:05d}"
                    self.speaker_counter += 1
                speaker_id = self.existing_clusters[label]
            
            speaker_ids.append(speaker_id)
        
        # Store embeddings if requested
        if self.store_embeddings and self.embedding_file:
            self._store_embeddings(embeddings, sample_ids, speaker_ids)
        
        # Update model with new clusters
        self._update_model(embeddings, labels, clusterer)
        
        return speaker_ids
    
    def _merge_with_existing(self, embeddings: np.ndarray, labels: np.ndarray, 
                            clusterer: hdbscan.HDBSCAN) -> np.ndarray:
        """Merge new clusters with existing ones based on similarity.
        
        Args:
            embeddings: New embeddings
            labels: Initial cluster labels from HDBSCAN
            clusterer: HDBSCAN clusterer instance
            
        Returns:
            Updated labels
        """
        if not hasattr(self, 'cluster_centroids') or self.cluster_centroids is None:
            return labels
        
        threshold = self.config.get('threshold', 0.7)
        
        # Compute similarities between new embeddings and existing centroids
        similarities = self._compute_cosine_similarity(embeddings, self.cluster_centroids)
        
        # For each embedding, check if it's similar enough to existing clusters
        for i in range(len(embeddings)):
            if labels[i] != -1:  # Only process non-noise points
                max_sim_idx = np.argmax(similarities[i])
                max_sim = similarities[i][max_sim_idx]
                
                if max_sim > threshold:
                    # Assign to existing cluster
                    labels[i] = max_sim_idx + 1000  # Offset to distinguish from new clusters
        
        return labels
    
    def _compute_cosine_similarity(self, embeddings1: np.ndarray, 
                                  embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(norm1, norm2.T)
        
        return similarities
    
    def _store_embeddings(self, embeddings: np.ndarray, sample_ids: List[str], 
                         speaker_ids: List[str]):
        """Store embeddings for future use.
        
        Args:
            embeddings: Speaker embeddings
            sample_ids: Sample IDs
            speaker_ids: Assigned speaker IDs
        """
        try:
            # Create datasets if they don't exist
            if 'embeddings' not in self.embedding_file:
                self.embedding_file.create_dataset(
                    'embeddings', shape=(0, embeddings.shape[1]),
                    maxshape=(None, embeddings.shape[1]),
                    dtype='float32', compression='gzip'
                )
                self.embedding_file.create_dataset(
                    'sample_ids', shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
                self.embedding_file.create_dataset(
                    'speaker_ids', shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
            
            # Append new data
            n_existing = self.embedding_file['embeddings'].shape[0]
            n_new = embeddings.shape[0]
            
            self.embedding_file['embeddings'].resize((n_existing + n_new, embeddings.shape[1]))
            self.embedding_file['embeddings'][n_existing:] = embeddings
            
            self.embedding_file['sample_ids'].resize((n_existing + n_new,))
            self.embedding_file['sample_ids'][n_existing:] = sample_ids
            
            self.embedding_file['speaker_ids'].resize((n_existing + n_new,))
            self.embedding_file['speaker_ids'][n_existing:] = speaker_ids
            
            self.embedding_file.flush()
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
    
    def _update_model(self, embeddings: np.ndarray, labels: np.ndarray, 
                     clusterer: hdbscan.HDBSCAN):
        """Update speaker model with new clustering results.
        
        Args:
            embeddings: Embeddings used for clustering
            labels: Cluster labels
            clusterer: HDBSCAN clusterer instance
        """
        # Compute cluster centroids for valid clusters
        unique_labels = set(labels) - {-1}
        
        if not hasattr(self, 'cluster_centroids'):
            self.cluster_centroids = []
        
        for label in unique_labels:
            if label < 1000:  # New cluster (not merged with existing)
                mask = labels == label
                centroid = embeddings[mask].mean(axis=0)
                self.cluster_centroids.append(centroid)
        
        self.cluster_centroids = np.array(self.cluster_centroids) if self.cluster_centroids else None
        
        # Save model
        self.save_model()
    
    def save_model(self):
        """Save speaker model for future use."""
        model_data = {
            'speaker_counter': self.speaker_counter,
            'existing_clusters': self.existing_clusters,
            'cluster_centroids': self.cluster_centroids.tolist() if self.cluster_centroids is not None else None,
            'clustering_params': self.clustering_params
        }
        
        try:
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            logger.info(f"Saved speaker model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save speaker model: {str(e)}")
    
    def load_model(self):
        """Load existing speaker model if available."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    model_data = json.load(f)
                
                self.speaker_counter = model_data.get('speaker_counter', 1)
                self.existing_clusters = model_data.get('existing_clusters', {})
                
                centroids = model_data.get('cluster_centroids')
                if centroids:
                    self.cluster_centroids = np.array(centroids)
                else:
                    self.cluster_centroids = None
                
                logger.info(f"Loaded speaker model from {self.model_path}")
                logger.info(f"Resuming from speaker counter: {self.speaker_counter}")
                
            except Exception as e:
                logger.error(f"Failed to load speaker model: {str(e)}")
    
    def close(self):
        """Clean up resources."""
        if self.embedding_file:
            self.embedding_file.close()
        self.save_model()
```

### 2. Integrate into Streaming Pipeline

Modify `main.py` to integrate speaker identification:

```python
# main.py - Add speaker identification to streaming mode
def process_streaming_mode(args, dataset_names: List[str]) -> int:
    """Process datasets in streaming mode with speaker identification."""
    
    # ... existing code ...
    
    # Initialize speaker identification if enabled
    speaker_identifier = None
    embedding_buffer = []
    
    if args.enable_speaker_id:
        from processors.speaker_identification import SpeakerIdentification
        
        speaker_config = {
            'model': args.speaker_model,
            'batch_size': args.speaker_batch_size,
            'min_cluster_size': args.speaker_min_cluster_size,
            'min_samples': args.speaker_min_samples,
            'epsilon': args.speaker_epsilon,
            'threshold': args.speaker_threshold,
            'store_embeddings': args.store_embeddings,
            'embedding_path': os.path.join(args.output or '.', 'speaker_embeddings.h5'),
            'model_path': os.path.join(CHECKPOINT_DIR, 'speaker_model.json')
        }
        
        speaker_identifier = SpeakerIdentification(speaker_config)
        logger.info("Initialized speaker identification system")
    
    # Process each dataset
    for dataset_name in dataset_names:
        try:
            # ... existing processing code ...
            
            for sample in processor.process_all_splits(...):
                # Assign sequential ID
                sample["ID"] = f"S{current_id}"
                current_id += 1
                
                # Add to embedding buffer if speaker ID enabled
                if speaker_identifier:
                    embedding_buffer.append(sample)
                
                # Process speaker IDs when buffer is full
                if speaker_identifier and len(embedding_buffer) >= args.speaker_batch_size:
                    logger.info(f"Processing speaker identification batch of {len(embedding_buffer)} samples")
                    
                    # Get speaker IDs for the batch
                    speaker_ids = speaker_identifier.process_batch(embedding_buffer)
                    
                    # Assign speaker IDs to samples
                    for sample, speaker_id in zip(embedding_buffer, speaker_ids):
                        sample['speaker_id'] = speaker_id
                        batch_buffer.append(sample)
                    
                    embedding_buffer = []
                else:
                    # No speaker identification - assign unique ID
                    if not speaker_identifier:
                        sample['speaker_id'] = f"SPK_{current_id:05d}"
                    batch_buffer.append(sample)
                
                # Upload batch when buffer is full
                if len(batch_buffer) >= args.upload_batch_size:
                    # ... existing upload code ...
    
    # Process remaining embeddings
    if speaker_identifier and embedding_buffer:
        logger.info(f"Processing final speaker identification batch of {len(embedding_buffer)} samples")
        speaker_ids = speaker_identifier.process_batch(embedding_buffer)
        
        for sample, speaker_id in zip(embedding_buffer, speaker_ids):
            sample['speaker_id'] = speaker_id
            batch_buffer.append(sample)
    
    # ... rest of the function ...
    
    # Clean up speaker identifier
    if speaker_identifier:
        speaker_identifier.close()
```

### 3. Add Command Line Arguments

Add new arguments to `parse_arguments()` in `main.py`:

```python
def parse_arguments():
    """Parse command-line arguments with speaker identification options."""
    parser = argparse.ArgumentParser(
        description="Thai Audio Dataset Collection with Speaker Identification"
    )
    
    # ... existing arguments ...
    
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
    
    return parser.parse_args()
```

### 4. Update Configuration

Add speaker identification configuration to `config.py`:

```python
# config.py
SPEAKER_ID_CONFIG = {
    "enabled": False,  # Default disabled, enable with --enable-speaker-id
    "model": "pyannote/embedding",
    "embedding_dim": 512,
    "batch_size": 10000,
    "clustering": {
        "algorithm": "hdbscan",
        "min_cluster_size": 15,
        "min_samples": 10,
        "metric": "cosine",
        "cluster_selection_epsilon": 0.3,
        "similarity_threshold": 0.7
    },
    "storage": {
        "store_embeddings": False,
        "embedding_format": "hdf5",
        "compression": "gzip"
    }
}

# Update TARGET_SCHEMA to include speaker_id
TARGET_SCHEMA = {
    "ID": str,
    "speaker_id": str,  # New field
    "Language": str,
    "audio": dict,
    "transcript": str,
    "length": float,
    "dataset_name": str,
    "confidence_score": float
}
```

### 5. Create Tests

Create comprehensive tests for speaker identification:

```python
# tests/test_speaker_identification.py
import unittest
import numpy as np
import tempfile
import os
from processors.speaker_identification import SpeakerIdentification

class TestSpeakerIdentification(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model': 'pyannote/embedding',
            'batch_size': 100,
            'store_embeddings': True,
            'embedding_path': os.path.join(self.temp_dir, 'test_embeddings.h5'),
            'model_path': os.path.join(self.temp_dir, 'test_model.json')
        }
        self.speaker_id = SpeakerIdentification(self.config)
    
    def test_embedding_extraction(self):
        """Test embedding extraction from audio."""
        # Create dummy audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        sample_rate = 16000
        
        embedding = self.speaker_id.extract_embedding(audio, sample_rate)
        
        # Check embedding shape
        self.assertEqual(len(embedding.shape), 1)
        self.assertEqual(embedding.shape[0], 512)
    
    def test_batch_processing(self):
        """Test batch processing of audio samples."""
        # Create dummy samples
        samples = []
        for i in range(20):
            samples.append({
                'ID': f'S{i+1}',
                'audio': {
                    'array': np.random.randn(16000 * 2),  # 2 seconds
                    'sampling_rate': 16000
                }
            })
        
        # Process batch
        speaker_ids = self.speaker_id.process_batch(samples)
        
        # Check results
        self.assertEqual(len(speaker_ids), len(samples))
        self.assertTrue(all(sid.startswith('SPK_') for sid in speaker_ids))
    
    def test_clustering_over_segmentation(self):
        """Test that clustering prefers over-segmentation."""
        # Create similar embeddings (should be different speakers with conservative settings)
        embeddings = np.random.randn(30, 512)
        embeddings[10:20] = embeddings[0] + np.random.randn(10, 512) * 0.1  # Similar to first
        
        sample_ids = [f'S{i+1}' for i in range(30)]
        speaker_ids = self.speaker_id.cluster_embeddings(embeddings, sample_ids)
        
        # With conservative settings, similar embeddings should still be separate
        unique_speakers = len(set(speaker_ids))
        self.assertGreater(unique_speakers, 2)  # Should have multiple clusters
    
    def test_model_persistence(self):
        """Test saving and loading speaker model."""
        # Process some samples
        samples = [{
            'ID': f'S{i+1}',
            'audio': {
                'array': np.random.randn(16000),
                'sampling_rate': 16000
            }
        } for i in range(10)]
        
        self.speaker_id.process_batch(samples)
        
        # Save model
        self.speaker_id.save_model()
        
        # Create new instance and load model
        new_speaker_id = SpeakerIdentification(self.config)
        
        # Check that state is preserved
        self.assertEqual(new_speaker_id.speaker_counter, self.speaker_id.speaker_counter)
        self.assertEqual(new_speaker_id.existing_clusters, self.speaker_id.existing_clusters)
    
    def tearDown(self):
        """Clean up test environment."""
        self.speaker_id.close()
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
```

## Usage Examples

### Basic Usage with Speaker Identification

```bash
# Process all datasets with speaker identification
python main.py --fresh --all --streaming --enable-speaker-id

# With custom parameters for conservative clustering
python main.py --fresh --all --streaming \
    --enable-speaker-id \
    --speaker-batch-size 5000 \
    --speaker-min-cluster-size 20 \
    --speaker-threshold 0.8

# Store embeddings for future analysis
python main.py --fresh --all --streaming \
    --enable-speaker-id \
    --store-embeddings

# Append mode with speaker identification
python main.py --append --all --streaming \
    --enable-speaker-id \
    --speaker-model pyannote/embedding
```

### Sample Mode Testing

```bash
# Test speaker identification with small sample
python main.py --fresh --all --sample --sample-size 100 \
    --enable-speaker-id \
    --speaker-batch-size 50
```

## Performance Considerations

### Memory Management

1. **VRAM Usage**:
   - Embedding extraction: ~2-4GB VRAM
   - Can process in parallel with STT
   - Batch size adjustable based on available memory

2. **RAM Usage**:
   - Embedding buffer: ~40MB per 10K samples
   - Clustering: ~1GB for 100K samples
   - HDF5 storage: Compressed, efficient access

3. **Processing Speed**:
   - Embedding extraction: ~1000 samples/second on RTX 5090
   - Clustering: ~1 second per 10K samples
   - Negligible impact on overall pipeline speed

### Optimization Strategies

1. **Batch Processing**:
   - Optimal batch size: 5K-10K samples
   - Balance between clustering quality and memory usage
   - Process during I/O operations to hide latency

2. **Incremental Updates**:
   - Store cluster centroids for fast comparison
   - Update model periodically, not after every batch
   - Use approximate nearest neighbor search for large datasets

3. **Storage Optimization**:
   - Compress embeddings with gzip in HDF5
   - Store only necessary metadata
   - Implement cleanup for old embeddings

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `--speaker-batch-size`
   - Disable `--store-embeddings` if not needed
   - Process datasets sequentially

2. **Poor Clustering**:
   - Increase `--speaker-min-cluster-size` for more conservative clustering
   - Adjust `--speaker-epsilon` for tighter clusters
   - Check audio quality and preprocessing

3. **Slow Processing**:
   - Ensure GPU is being used for embeddings
   - Reduce batch size if GPU memory is limiting
   - Use faster embedding model if available

## Future Enhancements

1. **Advanced Clustering**:
   - Online clustering algorithms for true streaming
   - Hierarchical clustering for better organization
   - Active learning for uncertain samples

2. **Quality Metrics**:
   - Clustering quality scores
   - Speaker purity metrics
   - Automated parameter tuning

3. **Integration Features**:
   - Web UI for speaker visualization
   - Export speaker segments for training
   - Cross-dataset speaker linking