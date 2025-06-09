# Task S03_T02: Build Speaker Diarization System

## Task Overview
Build a comprehensive speaker diarization system that segments audio by speaker turns, identifying "who spoke when" in multi-speaker recordings.

## Technical Requirements

### Core Implementation
- **Diarization System** (`processors/audio_enhancement/diarization/speaker_diarizer.py`)
  - Speaker segmentation
  - Clustering algorithms
  - Turn detection
  - Overlap handling

### Key Features
1. **Segmentation Methods**
   - Change point detection
   - Sliding window approach
   - Neural segmentation
   - Hybrid methods

2. **Clustering Approaches**
   - Spectral clustering
   - Agglomerative clustering
   - Online clustering
   - Neural clustering

3. **Advanced Features**
   - Overlapping speech detection
   - Number of speakers estimation
   - Speaker turn refinement
   - Confidence scoring

## TDD Requirements

### Test Structure
```
tests/test_speaker_diarizer.py
- test_two_speaker_diarization()
- test_multi_speaker_diarization()
- test_overlapping_speech()
- test_speaker_count_estimation()
- test_turn_boundary_accuracy()
- test_long_audio_processing()
```

### Test Data Requirements
- Two-speaker conversations
- Multi-speaker meetings
- Overlapping speech samples
- Various speaking styles

## Implementation Approach

### Phase 1: Core Diarization
```python
class SpeakerDiarizer:
    def __init__(self, clustering_method='spectral'):
        self.clustering_method = clustering_method
        self.embedder = SpeakerEmbedder()
        
    def diarize(self, audio, num_speakers=None):
        # Perform speaker diarization
        pass
    
    def get_speaker_segments(self):
        # Return speaker-labeled segments
        pass
    
    def estimate_num_speakers(self, audio):
        # Estimate number of speakers
        pass
```

### Phase 2: Advanced Features
- PyAnnote integration
- Real-time diarization
- Overlap resolution
- Quality metrics

### Phase 3: Optimization
- GPU acceleration
- Streaming support
- Memory efficiency
- Parallel processing

## Acceptance Criteria
1. ✅ Diarization Error Rate (DER) < 15%
2. ✅ Correct speaker count estimation > 90%
3. ✅ Handle up to 10 speakers
4. ✅ Process 1 hour audio in < 5 minutes
5. ✅ Overlap detection accuracy > 80%

## Example Usage
```python
from processors.audio_enhancement.diarization import SpeakerDiarizer

# Initialize diarizer
diarizer = SpeakerDiarizer(clustering_method='spectral')

# Perform diarization
results = diarizer.diarize(audio_data, num_speakers=None)

# Get speaker segments
segments = results.get_timeline()
for segment, track, label in segments.itertracks(yield_label=True):
    print(f"Speaker {label}: {segment.start:.2f}s - {segment.end:.2f}s")

# Visualize diarization
results.visualize(output_path='diarization.png')

# Get speaker statistics
stats = results.get_statistics()
print(f"Number of speakers: {stats['num_speakers']}")
print(f"Total speaking time: {stats['total_speech']:.1f}s")
```

## Dependencies
- PyAnnote.audio for diarization
- SpeechBrain for embeddings
- Scikit-learn for clustering
- NumPy/SciPy for processing
- Matplotlib for visualization

## Performance Targets
- DER: < 15% on benchmark datasets
- Processing speed: > 20x real-time
- Memory usage: < 2GB for 1-hour audio
- Latency: < 1s for initialization

## Notes
- Consider online diarization for streaming
- Implement speaker tracking across segments
- Support for domain adaptation
- Enable custom clustering parameters

## Neural Network Architectures

### End-to-End Neural Diarization

1. **EEND (End-to-End Neural Diarization)**
   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   
   class EEND(nn.Module):
       """End-to-End Neural Diarization with Self-Attention"""
       def __init__(self, n_speakers=4, input_dim=345, hidden_dim=256, 
                    n_layers=4, n_heads=8):
           super(EEND, self).__init__()
           
           self.n_speakers = n_speakers
           
           # Input projection
           self.input_projection = nn.Linear(input_dim, hidden_dim)
           
           # Transformer encoder
           encoder_layer = nn.TransformerEncoderLayer(
               d_model=hidden_dim,
               nhead=n_heads,
               dim_feedforward=1024,
               dropout=0.1,
               activation='relu'
           )
           self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
           
           # Speaker prediction head
           self.speaker_classifier = nn.Sequential(
               nn.Linear(hidden_dim, hidden_dim // 2),
               nn.ReLU(),
               nn.Dropout(0.1),
               nn.Linear(hidden_dim // 2, n_speakers)
           )
           
       def forward(self, x, mask=None):
           # x: (batch, time, features)
           x = self.input_projection(x)
           
           # Transformer encoding
           x = x.transpose(0, 1)  # (time, batch, features)
           encoded = self.transformer(x, src_key_padding_mask=mask)
           encoded = encoded.transpose(0, 1)  # (batch, time, features)
           
           # Speaker predictions
           logits = self.speaker_classifier(encoded)
           
           return torch.sigmoid(logits)  # Multi-label classification
   ```

2. **Self-Attentive EEND (SA-EEND)**
   ```python
   class SelfAttentiveEEND(nn.Module):
       """SA-EEND with encoder-decoder architecture"""
       def __init__(self, n_speakers=4, input_dim=345, hidden_dim=256):
           super(SelfAttentiveEEND, self).__init__()
           
           # Encoder
           self.encoder = nn.LSTM(
               input_dim, hidden_dim, 
               num_layers=4, 
               batch_first=True, 
               bidirectional=True,
               dropout=0.3
           )
           
           # Self-attention blocks
           self.self_attention_blocks = nn.ModuleList([
               SelfAttentionBlock(hidden_dim * 2) for _ in range(4)
           ])
           
           # Decoder
           self.decoder = nn.LSTM(
               hidden_dim * 2, hidden_dim,
               num_layers=2,
               batch_first=True,
               dropout=0.3
           )
           
           # Output projection
           self.output_projection = nn.Linear(hidden_dim, n_speakers)
           
       def forward(self, x):
           # Encode
           encoded, _ = self.encoder(x)
           
           # Self-attention
           for attention_block in self.self_attention_blocks:
               encoded = attention_block(encoded)
           
           # Decode
           decoded, _ = self.decoder(encoded)
           
           # Project to speakers
           output = self.output_projection(decoded)
           
           return torch.sigmoid(output)
   
   
   class SelfAttentionBlock(nn.Module):
       """Multi-head self-attention block"""
       def __init__(self, hidden_dim, n_heads=8):
           super(SelfAttentionBlock, self).__init__()
           
           self.attention = nn.MultiheadAttention(
               hidden_dim, n_heads, dropout=0.1
           )
           self.norm1 = nn.LayerNorm(hidden_dim)
           self.norm2 = nn.LayerNorm(hidden_dim)
           
           self.feed_forward = nn.Sequential(
               nn.Linear(hidden_dim, hidden_dim * 4),
               nn.ReLU(),
               nn.Dropout(0.1),
               nn.Linear(hidden_dim * 4, hidden_dim)
           )
           
       def forward(self, x):
           # Self-attention
           x_t = x.transpose(0, 1)
           attn_out, _ = self.attention(x_t, x_t, x_t)
           x = x + attn_out.transpose(0, 1)
           x = self.norm1(x)
           
           # Feed-forward
           x = x + self.feed_forward(x)
           x = self.norm2(x)
           
           return x
   ```

### Feature Extraction Pipelines

1. **Advanced Acoustic Features**
   ```python
   class DiarizationFeatureExtractor:
       """Extract features optimized for speaker diarization"""
       
       def __init__(self, sample_rate=16000):
           self.sample_rate = sample_rate
           
           # Feature configuration
           self.frame_length = 0.025  # 25ms
           self.frame_shift = 0.010   # 10ms
           self.n_mfcc = 19
           self.n_mels = 80
           
       def extract_features(self, audio):
           """Extract comprehensive features for diarization"""
           # MFCC features
           mfcc = librosa.feature.mfcc(
               y=audio, 
               sr=self.sample_rate,
               n_mfcc=self.n_mfcc,
               n_fft=int(self.frame_length * self.sample_rate),
               hop_length=int(self.frame_shift * self.sample_rate)
           )
           
           # Delta and delta-delta
           mfcc_delta = librosa.feature.delta(mfcc)
           mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
           
           # Mel-spectrogram
           mel_spec = librosa.feature.melspectrogram(
               y=audio,
               sr=self.sample_rate,
               n_mels=self.n_mels,
               n_fft=int(self.frame_length * self.sample_rate),
               hop_length=int(self.frame_shift * self.sample_rate)
           )
           mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
           
           # Pitch features
           f0, voiced_flag, voiced_probs = librosa.pyin(
               audio, 
               fmin=librosa.note_to_hz('C2'),
               fmax=librosa.note_to_hz('C7'),
               sr=self.sample_rate
           )
           
           # Energy features
           energy = librosa.feature.rms(
               y=audio,
               frame_length=int(self.frame_length * self.sample_rate),
               hop_length=int(self.frame_shift * self.sample_rate)
           )
           
           # Stack all features
           features = np.vstack([
               mfcc,
               mfcc_delta,
               mfcc_delta2,
               mel_spec_db,
               f0[np.newaxis, :],
               voiced_probs[np.newaxis, :],
               energy
           ])
           
           return features.T  # (time, features)
   ```

2. **Speaker Embedding Extraction**
   ```python
   class SpeakerEmbeddingExtractor:
       """Extract speaker embeddings using pre-trained models"""
       
       def __init__(self, model_name='ecapa_tdnn'):
           self.model_name = model_name
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           
           # Load pre-trained model
           if model_name == 'ecapa_tdnn':
               self.model = self._load_ecapa_tdnn()
           elif model_name == 'x_vector':
               self.model = self._load_x_vector()
           else:
               raise ValueError(f"Unknown model: {model_name}")
           
           self.model.eval()
           self.model.to(self.device)
           
       def _load_ecapa_tdnn(self):
           """Load ECAPA-TDNN model"""
           from speechbrain.pretrained import EncoderClassifier
           return EncoderClassifier.from_hparams(
               source="speechbrain/spkrec-ecapa-voxceleb",
               savedir="pretrained_models/ecapa_tdnn"
           )
           
       def extract_embeddings(self, audio, window_size=3.0, hop_size=0.5):
           """Extract windowed embeddings"""
           embeddings = []
           timestamps = []
           
           # Convert to samples
           window_samples = int(window_size * 16000)
           hop_samples = int(hop_size * 16000)
           
           # Extract embeddings for each window
           for start in range(0, len(audio) - window_samples, hop_samples):
               window = audio[start:start + window_samples]
               
               # Extract embedding
               with torch.no_grad():
                   embedding = self.model.encode_batch(
                       torch.tensor(window).unsqueeze(0).to(self.device)
                   )
               
               embeddings.append(embedding.cpu().numpy().squeeze())
               timestamps.append(start / 16000)
           
           return np.array(embeddings), np.array(timestamps)
   ```

### Clustering Algorithms

1. **Spectral Clustering with Affinity Refinement**
   ```python
   class RefinedSpectralClustering:
       """Spectral clustering with affinity matrix refinement"""
       
       def __init__(self, n_clusters=None, affinity_type='cosine'):
           self.n_clusters = n_clusters
           self.affinity_type = affinity_type
           
       def fit_predict(self, embeddings):
           """Perform clustering on embeddings"""
           # Compute affinity matrix
           if self.affinity_type == 'cosine':
               affinity = self._cosine_affinity(embeddings)
           elif self.affinity_type == 'gaussian':
               affinity = self._gaussian_affinity(embeddings)
           else:
               raise ValueError(f"Unknown affinity type: {self.affinity_type}")
           
           # Refine affinity matrix
           affinity_refined = self._refine_affinity(affinity)
           
           # Estimate number of clusters if not provided
           if self.n_clusters is None:
               self.n_clusters = self._estimate_n_clusters(affinity_refined)
           
           # Perform spectral clustering
           clustering = SpectralClustering(
               n_clusters=self.n_clusters,
               affinity='precomputed',
               assign_labels='kmeans',
               random_state=42
           )
           
           labels = clustering.fit_predict(affinity_refined)
           
           return labels
       
       def _cosine_affinity(self, embeddings):
           """Compute cosine similarity matrix"""
           # Normalize embeddings
           embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
           
           # Compute cosine similarity
           affinity = np.dot(embeddings_norm, embeddings_norm.T)
           
           # Convert to distance
           affinity = (affinity + 1) / 2  # Scale to [0, 1]
           
           return affinity
       
       def _refine_affinity(self, affinity, p_neighbors=0.3):
           """Refine affinity matrix using nearest neighbors"""
           n_samples = affinity.shape[0]
           k = max(1, int(p_neighbors * n_samples))
           
           # Keep only k nearest neighbors
           refined = np.zeros_like(affinity)
           
           for i in range(n_samples):
               # Get k nearest neighbors
               nearest = np.argpartition(affinity[i], -k)[-k:]
               
               # Symmetrize
               for j in nearest:
                   refined[i, j] = affinity[i, j]
                   refined[j, i] = affinity[j, i]
           
           return refined
       
       def _estimate_n_clusters(self, affinity):
           """Estimate optimal number of clusters using eigenvalues"""
           # Compute normalized Laplacian
           degree = np.sum(affinity, axis=1)
           D_sqrt_inv = np.diag(1.0 / np.sqrt(degree + 1e-10))
           L_norm = np.eye(len(affinity)) - D_sqrt_inv @ affinity @ D_sqrt_inv
           
           # Compute eigenvalues
           eigenvalues = np.linalg.eigvalsh(L_norm)
           
           # Find eigengap
           eigengaps = np.diff(eigenvalues)
           n_clusters = np.argmax(eigengaps[:10]) + 1  # Look at first 10
           
           return max(2, n_clusters)  # At least 2 speakers
   ```

2. **Online Clustering for Streaming**
   ```python
   class OnlineSpeakerClustering:
       """Online clustering for streaming diarization"""
       
       def __init__(self, threshold=0.5, max_speakers=10):
           self.threshold = threshold
           self.max_speakers = max_speakers
           self.centroids = []
           self.cluster_sizes = []
           
       def update(self, embedding):
           """Update clustering with new embedding"""
           if not self.centroids:
               # First embedding - create new cluster
               self.centroids.append(embedding)
               self.cluster_sizes.append(1)
               return 0
           
           # Compute distances to existing centroids
           distances = [
               self._distance(embedding, centroid) 
               for centroid in self.centroids
           ]
           
           min_distance = min(distances)
           closest_cluster = np.argmin(distances)
           
           if min_distance < self.threshold:
               # Assign to existing cluster and update centroid
               self._update_centroid(closest_cluster, embedding)
               return closest_cluster
           elif len(self.centroids) < self.max_speakers:
               # Create new cluster
               self.centroids.append(embedding)
               self.cluster_sizes.append(1)
               return len(self.centroids) - 1
           else:
               # Force assignment to closest cluster
               self._update_centroid(closest_cluster, embedding)
               return closest_cluster
       
       def _distance(self, emb1, emb2):
           """Compute distance between embeddings"""
           # Cosine distance
           cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
           return 1 - cos_sim
       
       def _update_centroid(self, cluster_idx, embedding):
           """Update cluster centroid incrementally"""
           n = self.cluster_sizes[cluster_idx]
           self.centroids[cluster_idx] = (
               (self.centroids[cluster_idx] * n + embedding) / (n + 1)
           )
           self.cluster_sizes[cluster_idx] += 1
   ```

### Source Separation Techniques

1. **Neural Source Separation**
   ```python
   class NeuralSpeakerSeparation(nn.Module):
       """Separate overlapping speakers using neural networks"""
       
       def __init__(self, n_speakers=2, n_filters=512):
           super(NeuralSpeakerSeparation, self).__init__()
           
           self.n_speakers = n_speakers
           
           # Encoder
           self.encoder = nn.Conv1d(1, n_filters, kernel_size=16, stride=8)
           
           # Separator
           self.separator = nn.Sequential(
               nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv1d(n_filters, n_filters * n_speakers, kernel_size=3, padding=1),
           )
           
           # Decoder
           self.decoder = nn.ConvTranspose1d(
               n_filters, 1, kernel_size=16, stride=8
           )
           
       def forward(self, mixture):
           # mixture: (batch, 1, time)
           
           # Encode
           encoded = self.encoder(mixture)
           
           # Separate
           masks = self.separator(encoded)
           masks = masks.view(-1, self.n_speakers, encoded.shape[1], encoded.shape[2])
           masks = F.softmax(masks, dim=1)
           
           # Apply masks and decode
           separated = []
           for i in range(self.n_speakers):
               masked = encoded * masks[:, i]
               decoded = self.decoder(masked)
               separated.append(decoded)
           
           return torch.stack(separated, dim=1)
   ```

### Quality Preservation Methods

1. **Overlap-Aware Diarization**
   ```python
   class OverlapAwareDiarization:
       """Handle overlapping speech in diarization"""
       
       def __init__(self, overlap_threshold=0.3):
           self.overlap_threshold = overlap_threshold
           self.overlap_detector = OverlapDetector()
           
       def process(self, audio, initial_diarization):
           """Refine diarization considering overlaps"""
           # Detect overlap regions
           overlap_probs = self.overlap_detector.detect(audio)
           
           # Find high-overlap regions
           overlap_regions = self._find_overlap_regions(
               overlap_probs, self.overlap_threshold
           )
           
           # Refine diarization in overlap regions
           refined_diarization = initial_diarization.copy()
           
           for start, end in overlap_regions:
               # Extract segment
               segment = audio[int(start * 16000):int(end * 16000)]
               
               # Separate speakers
               separated = self._separate_speakers(segment)
               
               # Update diarization
               refined_diarization = self._update_diarization(
                   refined_diarization, separated, start, end
               )
           
           return refined_diarization
       
       def _find_overlap_regions(self, overlap_probs, threshold):
           """Find regions with overlapping speech"""
           regions = []
           in_overlap = False
           start = None
           
           for i, prob in enumerate(overlap_probs):
               if prob > threshold and not in_overlap:
                   start = i * 0.01  # Frame to seconds
                   in_overlap = True
               elif prob <= threshold and in_overlap:
                   end = i * 0.01
                   regions.append((start, end))
                   in_overlap = False
           
           return regions
   ```

2. **Quality-Preserving Post-Processing**
   ```python
   class DiarizationPostProcessor:
       """Post-process diarization results for quality"""
       
       def __init__(self, min_segment_duration=0.3, min_gap_duration=0.3):
           self.min_segment_duration = min_segment_duration
           self.min_gap_duration = min_gap_duration
           
       def process(self, diarization):
           """Apply post-processing to improve quality"""
           # Merge short segments
           diarization = self._merge_short_segments(diarization)
           
           # Fill short gaps
           diarization = self._fill_short_gaps(diarization)
           
           # Smooth boundaries
           diarization = self._smooth_boundaries(diarization)
           
           # Resolve conflicts
           diarization = self._resolve_conflicts(diarization)
           
           return diarization
       
       def _merge_short_segments(self, diarization):
           """Merge segments shorter than threshold"""
           merged = []
           
           for segment in diarization:
               if segment['duration'] < self.min_segment_duration:
                   # Try to merge with adjacent segment
                   if merged and merged[-1]['speaker'] == segment['speaker']:
                       gap = segment['start'] - merged[-1]['end']
                       if gap < self.min_gap_duration:
                           merged[-1]['end'] = segment['end']
                           continue
               
               merged.append(segment)
           
           return merged
       
       def _smooth_boundaries(self, diarization):
           """Smooth segment boundaries using voice activity"""
           smoothed = []
           
           for i, segment in enumerate(diarization):
               # Adjust start time
               if i > 0:
                   prev_end = diarization[i-1]['end']
                   gap = segment['start'] - prev_end
                   if gap < 0.1:  # Small gap
                       segment['start'] = prev_end
               
               # Adjust end time
               if i < len(diarization) - 1:
                   next_start = diarization[i+1]['start']
                   gap = next_start - segment['end']
                   if gap < 0.1:  # Small gap
                       segment['end'] = next_start
               
               smoothed.append(segment)
           
           return smoothed
   ```

### Production-Ready Implementation

```python
class ProductionDiarizationSystem:
    """Production-ready speaker diarization system"""
    
    def __init__(self, config_path='diarization_config.yaml'):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.vad = VoiceActivityDetector()
        self.feature_extractor = DiarizationFeatureExtractor()
        self.embedding_extractor = SpeakerEmbeddingExtractor()
        self.diarizer = EEND(n_speakers=self.config['max_speakers'])
        self.post_processor = DiarizationPostProcessor()
        
        # Load models
        self._load_models()
        
    def diarize(self, audio_path, num_speakers=None):
        """Perform complete diarization pipeline"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Voice activity detection
        speech_segments = self.vad.get_speech_segments(audio)
        
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Neural diarization
        with torch.no_grad():
            diarization_raw = self.diarizer(
                torch.tensor(features).unsqueeze(0)
            )
        
        # Convert to segments
        segments = self._neural_output_to_segments(
            diarization_raw.squeeze().numpy(),
            speech_segments
        )
        
        # Post-process
        segments_refined = self.post_processor.process(segments)
        
        # Handle overlaps if needed
        if self.config['handle_overlaps']:
            overlap_handler = OverlapAwareDiarization()
            segments_refined = overlap_handler.process(audio, segments_refined)
        
        return {
            'segments': segments_refined,
            'audio_duration': len(audio) / sr,
            'num_speakers': len(set(s['speaker'] for s in segments_refined))
        }
    
    def diarize_streaming(self, audio_stream):
        """Perform online diarization on audio stream"""
        online_clusterer = OnlineSpeakerClustering()
        
        for chunk in audio_stream:
            # Extract embedding for chunk
            embedding = self.embedding_extractor.extract_single(chunk)
            
            # Update clustering
            speaker_id = online_clusterer.update(embedding)
            
            yield {
                'timestamp': chunk['timestamp'],
                'speaker_id': speaker_id,
                'confidence': chunk.get('confidence', 1.0)
            }
```