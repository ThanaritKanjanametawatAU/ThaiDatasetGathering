"""
Speaker identification and clustering module for audio samples.

This module provides functionality to extract speaker embeddings from audio
and cluster them to identify unique speakers across the dataset.
"""

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
            'metric': 'euclidean',  # Will use normalized embeddings for cosine similarity
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
        # Normalize embeddings for cosine similarity (using euclidean on normalized = cosine)
        normalized_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(**self.clustering_params)
        labels = clusterer.fit_predict(normalized_embeddings)
        
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
        # Convert integer keys to strings for JSON serialization
        existing_clusters_str = {str(k): v for k, v in self.existing_clusters.items()}
        
        model_data = {
            'speaker_counter': self.speaker_counter,
            'existing_clusters': existing_clusters_str,
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
                
                # Convert string keys to int for existing_clusters
                self.existing_clusters = {int(k) if k.isdigit() else k: v 
                                        for k, v in self.existing_clusters.items()}
                
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