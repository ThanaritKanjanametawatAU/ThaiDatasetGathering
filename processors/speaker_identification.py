"""
Speaker identification and clustering module for audio samples.

This module provides functionality to extract speaker embeddings from audio
and cluster them to identify unique speakers across the dataset.
"""

import torch
import numpy as np
from pyannote.audio import Model, Inference
import hdbscan
from typing import List, Dict, Optional, Tuple
import h5py
import logging
import json
import os
import librosa
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


class SpeakerIdentification:
    """Speaker identification and clustering for audio samples."""
    
    def __init__(self, config: Dict):
        """Initialize speaker identification system.
        
        Args:
            config: Configuration dictionary with model and clustering parameters
                - model: Model name or path (default: pyannote/wespeaker-voxceleb-resnet34-LM)
                - clustering: Dict with clustering parameters
                    - min_cluster_size: Minimum cluster size for HDBSCAN
                    - min_samples: Minimum samples for core points
                    - metric: Distance metric for clustering
                    - cluster_selection_epsilon: Epsilon for cluster selection
                    - similarity_threshold: Threshold for matching to existing clusters
                - store_embeddings: Whether to store embeddings (default: False)
                - embedding_path: Path to store embeddings (default: speaker_embeddings.h5)
                - model_path: Path to save/load speaker model (default: speaker_model.json)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load embedding model
        logger.info(f"Loading embedding model: {config.get('model', 'pyannote/wespeaker-voxceleb-resnet34-LM')}")
        self.model = Model.from_pretrained(
            config.get('model', 'pyannote/wespeaker-voxceleb-resnet34-LM')
        ).to(self.device)
        
        # Clustering parameters from config - use tuned values for Thai Voice dataset
        clustering_config = config.get('clustering', {})
        self.clustering_params = {
            'min_cluster_size': clustering_config.get('min_cluster_size', 5),
            'min_samples': clustering_config.get('min_samples', 3),
            'metric': 'euclidean',  # Will use normalized embeddings for cosine similarity
            'cluster_selection_epsilon': clustering_config.get('cluster_selection_epsilon', 0.5),
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
        self.cluster_centroids = None  # Initialize to avoid AttributeError
        
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
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Use Inference wrapper for proper embedding extraction
        # This handles windowing and model-specific preprocessing
        with torch.no_grad():
            waveform = torch.from_numpy(audio).float()
            
            # Ensure 2D tensor for Inference (channel, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Create input dict for Inference
            waveform_dict = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            # Use inference wrapper to get embedding
            # This will handle the 5-second duration requirement internally
            inference = Inference(self.model, window="whole")
            embedding = inference(waveform_dict)
            
            return embedding
    
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
        failed_samples = []
        
        for i, sample in enumerate(audio_samples):
            try:
                embedding = self.extract_embedding(
                    sample['audio']['array'],
                    sample['audio']['sampling_rate']
                )
                embeddings.append(embedding)
                sample_ids.append(sample['ID'])
            except Exception as e:
                logger.error(f"Failed to extract embedding for {sample['ID']}: {str(e)}")
                # Track failed samples instead of using dummy embeddings
                failed_samples.append((i, sample['ID']))
        
        if not embeddings:
            logger.warning("No embeddings extracted from batch")
            return ['UNKNOWN'] * len(audio_samples)
        
        # Convert to numpy array
        embeddings = np.vstack(embeddings)
        
        # Cluster embeddings
        speaker_ids = self.cluster_embeddings(embeddings, sample_ids)
        
        # Handle failed samples by inserting UNKNOWN at correct positions
        final_speaker_ids = []
        embedding_idx = 0
        for i in range(len(audio_samples)):
            if any(failed_idx == i for failed_idx, _ in failed_samples):
                final_speaker_ids.append('UNKNOWN')
            else:
                final_speaker_ids.append(speaker_ids[embedding_idx])
                embedding_idx += 1
        
        return final_speaker_ids
    
    def cluster_embeddings(self, embeddings: np.ndarray, sample_ids: List[str]) -> List[str]:
        """Cluster embeddings and assign speaker IDs using adaptive clustering.
        
        Args:
            embeddings: Array of speaker embeddings
            sample_ids: List of sample IDs
            
        Returns:
            List of speaker IDs
        """
        # Handle case with too few samples for clustering
        if len(embeddings) < 2:
            # Too few samples - each gets unique ID
            speaker_ids = []
            for _ in range(len(embeddings)):
                speaker_ids.append(f"SPK_{self.speaker_counter:05d}")
                self.speaker_counter += 1
            return speaker_ids
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Use adaptive clustering based on batch size and similarity distribution
        # For Thai Voice dataset, we need more lenient parameters
        if len(embeddings) < 50:
            # Small batch - use Agglomerative Clustering
            # This works better for small datasets with varying similarities
            
            # First compute similarity matrix to determine appropriate threshold
            cosine_distances = pairwise_distances(normalized_embeddings, metric='cosine')
            similarities = 1 - cosine_distances
            
            # Check if there are distinct groups
            # Look at the distribution of similarities
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            mean_sim = np.mean(upper_triangle)
            std_sim = np.std(upper_triangle)
            
            # Adaptive distance threshold based on similarity distribution
            # If similarities are very high, use stricter threshold
            if mean_sim > 0.9:
                distance_threshold = 0.2  # Corresponds to 0.8 minimum similarity
            elif mean_sim > 0.7:
                distance_threshold = 0.4  # Corresponds to 0.6 minimum similarity
            else:
                distance_threshold = 0.6  # Corresponds to 0.4 minimum similarity
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='precomputed',
                linkage='average'
            )
            
            labels = clustering.fit_predict(cosine_distances)
            
        else:
            # Larger batch - use HDBSCAN with adjusted parameters
            # Adjust parameters based on similarity distribution
            similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
            mean_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            
            if mean_similarity < 0.6:
                # Low overall similarity - use very lenient parameters
                epsilon = 0.7
                min_cluster_size = max(2, len(embeddings) // 20)
                min_samples = 1
            else:
                # Higher similarity - can use stricter parameters
                epsilon = self.clustering_params.get('cluster_selection_epsilon', 0.5)
                min_cluster_size = self.clustering_params.get('min_cluster_size', 5)
                min_samples = self.clustering_params.get('min_samples', 3)
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_epsilon=epsilon,
                cluster_selection_method='eom',
                prediction_data=True
            )
            labels = clusterer.fit_predict(normalized_embeddings)
        
        # Log clustering results
        unique_labels = set(labels) - {-1}
        noise_count = np.sum(labels == -1)
        logger.info(f"Clustering found {len(unique_labels)} clusters and {noise_count} noise points "
                   f"from {len(embeddings)} samples")
        
        # Handle existing clusters (for append mode)
        if self.existing_clusters:
            labels = self._merge_with_existing(normalized_embeddings, labels, None)
        
        # Convert labels to speaker IDs
        speaker_ids = []
        for i, label in enumerate(labels):
            if label == -1:  # Noise/uncertain cluster
                # For noise points, try to find the most similar existing cluster
                if self.cluster_centroids is not None and len(self.cluster_centroids) > 0:
                    similarities = self._compute_cosine_similarity(
                        normalized_embeddings[i:i+1], 
                        self.cluster_centroids
                    )
                    max_sim_idx = np.argmax(similarities[0])
                    max_sim = similarities[0][max_sim_idx]
                    
                    if max_sim > 0.4:  # Lower threshold for noise points
                        # Assign to existing cluster
                        speaker_id = self.existing_clusters.get(max_sim_idx + 1000, 
                                                               f"SPK_{self.speaker_counter:05d}")
                        if max_sim_idx + 1000 not in self.existing_clusters:
                            self.existing_clusters[max_sim_idx + 1000] = speaker_id
                            self.speaker_counter += 1
                    else:
                        # Create new speaker ID
                        speaker_id = f"SPK_{self.speaker_counter:05d}"
                        self.speaker_counter += 1
                else:
                    # No existing clusters - create new speaker ID
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
        self._update_model(normalized_embeddings, labels, None)
        
        return speaker_ids
    
    def _merge_with_existing(self, embeddings: np.ndarray, labels: np.ndarray, 
                            clusterer) -> np.ndarray:
        """Merge new clusters with existing ones based on similarity.
        
        Args:
            embeddings: New embeddings (normalized)
            labels: Initial cluster labels
            clusterer: Clustering instance (not used)
            
        Returns:
            Updated labels
        """
        if not hasattr(self, 'cluster_centroids') or self.cluster_centroids is None:
            return labels
        
        threshold = self.config.get('clustering', {}).get('similarity_threshold', 0.6)
        
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
            embeddings1: First set of embeddings (assumed normalized)
            embeddings2: Second set of embeddings (assumed normalized)
            
        Returns:
            Similarity matrix
        """
        # Both should already be normalized
        similarities = np.dot(embeddings1, embeddings2.T)
        return similarities
    
    def _store_embeddings(self, embeddings: np.ndarray, sample_ids: List[str], 
                         speaker_ids: List[str]):
        """Store embeddings in HDF5 file.
        
        Args:
            embeddings: Speaker embeddings
            sample_ids: Sample IDs
            speaker_ids: Assigned speaker IDs
        """
        if not self.embedding_file:
            return
        
        try:
            # Create group for this batch
            batch_name = f"batch_{len(self.embedding_file)}"
            batch_group = self.embedding_file.create_group(batch_name)
            
            # Store data
            batch_group.create_dataset('embeddings', data=embeddings)
            batch_group.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
            batch_group.create_dataset('speaker_ids', data=np.array(speaker_ids, dtype='S'))
            
            # Flush to disk
            self.embedding_file.flush()
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
    
    def _update_model(self, embeddings: np.ndarray, labels: np.ndarray, 
                     clusterer):
        """Update speaker model with new clustering results.
        
        Args:
            embeddings: Embeddings used for clustering (normalized)
            labels: Cluster labels
            clusterer: Clustering instance (not used)
        """
        # Compute cluster centroids for valid clusters
        unique_labels = set(labels) - {-1}
        
        # Initialize or get existing centroids
        if self.cluster_centroids is not None:
            cluster_centroids_list = self.cluster_centroids.tolist()
        else:
            cluster_centroids_list = []
        
        for label in unique_labels:
            if label < 1000:  # New cluster (not merged with existing)
                mask = labels == label
                centroid = embeddings[mask].mean(axis=0)
                cluster_centroids_list.append(centroid)
        
        self.cluster_centroids = np.array(cluster_centroids_list) if cluster_centroids_list else None
    
    def save_model(self):
        """Save speaker model to disk."""
        # Convert existing_clusters keys to strings for JSON compatibility
        existing_clusters_str = {str(k): v for k, v in self.existing_clusters.items()}
        
        model_data = {
            'speaker_counter': self.speaker_counter,
            'existing_clusters': existing_clusters_str,
            'cluster_centroids': self.cluster_centroids.tolist() if self.cluster_centroids is not None else None
        }
        
        try:
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            logger.info(f"Saved speaker model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save speaker model: {str(e)}")
    
    def load_model(self):
        """Load speaker model from disk."""
        # Check if fresh flag is set - handle different types
        fresh_value = self.config.get('fresh', False)
        # Convert string values to boolean
        if isinstance(fresh_value, str):
            fresh_value = fresh_value.lower() in ['true', 'yes', '1']
        elif isinstance(fresh_value, int):
            fresh_value = bool(fresh_value)
        
        if fresh_value:
            logger.info("Fresh mode enabled - resetting speaker counter to 1")
            # Delete existing model file if it exists
            if os.path.exists(self.model_path):
                try:
                    os.remove(self.model_path)
                    logger.info(f"Deleted existing speaker model: {self.model_path}")
                except Exception as e:
                    logger.error(f"Failed to delete speaker model: {str(e)}")
            
            # Reset to initial state
            self.speaker_counter = 1
            self.existing_clusters = {}
            self.cluster_centroids = None
            return
        
        # Normal load behavior when not fresh
        if not os.path.exists(self.model_path):
            logger.info("No existing speaker model found")
            return
        
        try:
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
            
            self.speaker_counter = model_data.get('speaker_counter', 1)
            self.existing_clusters = model_data.get('existing_clusters', {})
            
            # Convert string keys to integers for cluster labels
            self.existing_clusters = {int(k): v for k, v in self.existing_clusters.items()}
            
            centroids = model_data.get('cluster_centroids')
            self.cluster_centroids = np.array(centroids) if centroids else None
            
            logger.info(f"Loaded speaker model from {self.model_path}")
            logger.info(f"Resuming from speaker {self.speaker_counter}, {len(self.existing_clusters)} existing clusters")
            
        except Exception as e:
            logger.error(f"Failed to load speaker model: {str(e)}")
    
    def reset_state(self):
        """Reset clustering state but keep speaker counter.
        
        This is used when switching between datasets to ensure
        speaker IDs are not reused across datasets.
        """
        # Clear existing clusters and centroids
        self.existing_clusters = {}
        self.cluster_centroids = None
        
        logger.info(f"Reset clustering state, speaker counter at {self.speaker_counter}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.embedding_file:
            self.embedding_file.close()
        
        # Save model before cleanup
        self.save_model()