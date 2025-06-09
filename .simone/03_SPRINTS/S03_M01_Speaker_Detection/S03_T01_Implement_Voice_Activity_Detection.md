# Task S03_T01: Implement Voice Activity Detection

## Task Overview
Implement a robust Voice Activity Detection (VAD) system that accurately identifies speech segments in audio, distinguishing them from silence, noise, and non-speech sounds.

## Technical Requirements

### Core Implementation
- **VAD Module** (`processors/audio_enhancement/detection/voice_activity_detector.py`)
  - Energy-based detection
  - Spectral feature analysis
  - Deep learning VAD models
  - Adaptive threshold mechanisms

### Key Features
1. **Multi-Method Detection**
   - Energy and zero-crossing rate
   - Spectral entropy analysis
   - DNN-based detection (WebRTC VAD)
   - Hybrid approach combining methods

2. **Robustness Features**
   - Noise-robust detection
   - Music/speech discrimination
   - Adaptive sensitivity
   - Hangover and hold-off times

3. **Output Formats**
   - Frame-level decisions
   - Segment timestamps
   - Confidence scores
   - Speech probability curves

## TDD Requirements

### Test Structure
```
tests/test_voice_activity_detector.py
- test_clean_speech_detection()
- test_noisy_speech_detection()
- test_music_rejection()
- test_silence_detection()
- test_segment_boundaries()
- test_real_time_processing()
```

### Test Data Requirements
- Clean speech samples
- Noisy speech (various SNRs)
- Music and non-speech audio
- Mixed content (speech + music)

## Implementation Approach

### Phase 1: Core VAD
```python
class VoiceActivityDetector:
    def __init__(self, method='hybrid', sample_rate=16000):
        self.method = method
        self.sample_rate = sample_rate
        self.frame_size = int(0.03 * sample_rate)  # 30ms frames
        
    def detect(self, audio):
        # Perform VAD on audio
        pass
    
    def get_speech_segments(self, audio):
        # Return list of (start, end) timestamps
        pass
    
    def get_speech_probability(self, audio):
        # Return frame-wise speech probabilities
        pass
```

### Phase 2: Advanced Detection
- Deep learning models (Silero VAD)
- Multi-channel VAD
- Speaker-aware VAD
- Context-aware detection

### Phase 3: Integration
- Real-time streaming support
- Pipeline integration
- Performance optimization
- Visualization tools

## Acceptance Criteria
1. ✅ > 95% accuracy on clean speech
2. ✅ > 85% accuracy at 0dB SNR
3. ✅ Real-time processing capability
4. ✅ Music/speech discrimination > 90%
5. ✅ Smooth segment boundaries

## Example Usage
```python
from processors.audio_enhancement.detection import VoiceActivityDetector

# Initialize VAD
vad = VoiceActivityDetector(method='hybrid', sample_rate=16000)

# Detect voice activity
results = vad.detect(audio_data)
print(f"Speech detected: {results.speech_ratio:.1%}")

# Get speech segments
segments = vad.get_speech_segments(audio_data)
for start, end in segments:
    print(f"Speech segment: {start:.2f}s - {end:.2f}s")

# Get frame-wise probabilities
probs = vad.get_speech_probability(audio_data)
plt.plot(probs)
plt.ylabel('Speech Probability')
plt.show()
```

## Dependencies
- NumPy for signal processing
- SciPy for filtering
- WebRTC VAD or py-webrtcvad
- Torch/TensorFlow for DNN models
- Librosa for audio features

## Performance Targets
- Processing speed: > 100x real-time
- Latency: < 50ms
- Memory usage: < 100MB
- Accuracy: > 95% on clean speech

## Notes
- Consider different VAD aggressiveness levels
- Implement smoothing for segment boundaries
- Support for multi-channel audio
- Enable fine-tuning for specific domains

## Neural Network Architectures

### Deep Learning VAD Models

1. **CNN-LSTM Architecture**
   ```python
   import torch
   import torch.nn as nn
   
   class CNNLSTMVAD(nn.Module):
       """CNN-LSTM architecture for robust VAD"""
       def __init__(self, input_dim=40, hidden_dim=128, num_layers=2):
           super(CNNLSTMVAD, self).__init__()
           
           # CNN feature extractor
           self.cnn = nn.Sequential(
               nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.BatchNorm1d(64),
               nn.Conv1d(64, 128, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.BatchNorm1d(128),
               nn.MaxPool1d(2)
           )
           
           # Bidirectional LSTM
           self.lstm = nn.LSTM(
               input_size=128,
               hidden_size=hidden_dim,
               num_layers=num_layers,
               batch_first=True,
               bidirectional=True,
               dropout=0.3
           )
           
           # Output layer
           self.fc = nn.Sequential(
               nn.Linear(hidden_dim * 2, 64),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(64, 2)  # Binary classification
           )
           
       def forward(self, x):
           # x shape: (batch, time, features)
           x = x.transpose(1, 2)  # (batch, features, time)
           
           # CNN processing
           cnn_out = self.cnn(x)
           cnn_out = cnn_out.transpose(1, 2)  # (batch, time, features)
           
           # LSTM processing
           lstm_out, _ = self.lstm(cnn_out)
           
           # Classification
           output = self.fc(lstm_out)
           
           return output
   ```

2. **Transformer-based VAD**
   ```python
   class TransformerVAD(nn.Module):
       """Transformer architecture for VAD with attention mechanisms"""
       def __init__(self, input_dim=40, d_model=256, nhead=8, num_layers=4):
           super(TransformerVAD, self).__init__()
           
           # Input projection
           self.input_projection = nn.Linear(input_dim, d_model)
           
           # Positional encoding
           self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
           
           # Transformer encoder
           encoder_layer = nn.TransformerEncoderLayer(
               d_model=d_model,
               nhead=nhead,
               dim_feedforward=1024,
               dropout=0.1,
               activation='gelu'
           )
           self.transformer = nn.TransformerEncoder(
               encoder_layer,
               num_layers=num_layers
           )
           
           # Output layers
           self.output_projection = nn.Sequential(
               nn.Linear(d_model, 128),
               nn.ReLU(),
               nn.Dropout(0.1),
               nn.Linear(128, 2)
           )
           
       def forward(self, x, mask=None):
           # Project input
           x = self.input_projection(x)
           x = self.pos_encoder(x)
           
           # Transformer encoding
           x = x.transpose(0, 1)  # (time, batch, features)
           encoded = self.transformer(x, src_key_padding_mask=mask)
           encoded = encoded.transpose(0, 1)  # (batch, time, features)
           
           # Classification
           output = self.output_projection(encoded)
           
           return output
   ```

3. **Lightweight MobileNet VAD**
   ```python
   class MobileNetVAD(nn.Module):
       """Lightweight VAD for edge deployment"""
       def __init__(self, input_dim=40):
           super(MobileNetVAD, self).__init__()
           
           # Depthwise separable convolutions
           self.features = nn.Sequential(
               # First block
               self._depthwise_separable_conv(input_dim, 32, stride=1),
               self._depthwise_separable_conv(32, 64, stride=2),
               self._depthwise_separable_conv(64, 128, stride=1),
               self._depthwise_separable_conv(128, 128, stride=2),
               self._depthwise_separable_conv(128, 256, stride=1),
               self._depthwise_separable_conv(256, 256, stride=2),
           )
           
           # Global average pooling
           self.gap = nn.AdaptiveAvgPool1d(1)
           
           # Classifier
           self.classifier = nn.Sequential(
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(128, 2)
           )
           
       def _depthwise_separable_conv(self, in_channels, out_channels, stride):
           return nn.Sequential(
               # Depthwise
               nn.Conv1d(in_channels, in_channels, kernel_size=3,
                        stride=stride, padding=1, groups=in_channels),
               nn.BatchNorm1d(in_channels),
               nn.ReLU6(),
               # Pointwise
               nn.Conv1d(in_channels, out_channels, kernel_size=1),
               nn.BatchNorm1d(out_channels),
               nn.ReLU6()
           )
           
       def forward(self, x):
           x = x.transpose(1, 2)  # (batch, features, time)
           features = self.features(x)
           pooled = self.gap(features).squeeze(-1)
           output = self.classifier(pooled)
           return output
   ```

### Feature Extraction Pipelines

1. **Multi-Scale Feature Extraction**
   ```python
   class MultiScaleFeatureExtractor:
       """Extract features at multiple time scales"""
       
       def __init__(self, sample_rate=16000):
           self.sample_rate = sample_rate
           self.window_sizes = [10, 25, 50]  # ms
           self.hop_sizes = [5, 10, 25]      # ms
           
       def extract_features(self, audio):
           """Extract multi-scale features"""
           features = []
           
           for window_ms, hop_ms in zip(self.window_sizes, self.hop_sizes):
               # Convert to samples
               window_samples = int(window_ms * self.sample_rate / 1000)
               hop_samples = int(hop_ms * self.sample_rate / 1000)
               
               # Extract features at this scale
               scale_features = self._extract_scale_features(
                   audio, window_samples, hop_samples
               )
               features.append(scale_features)
           
           # Combine multi-scale features
           combined = self._combine_features(features)
           
           return combined
       
       def _extract_scale_features(self, audio, window, hop):
           """Extract features at specific scale"""
           features = {
               'energy': self._compute_energy(audio, window, hop),
               'zcr': self._compute_zcr(audio, window, hop),
               'spectral_centroid': self._compute_spectral_centroid(audio, window, hop),
               'spectral_rolloff': self._compute_spectral_rolloff(audio, window, hop),
               'mfcc': self._compute_mfcc(audio, window, hop)
           }
           
           return features
   ```

2. **Advanced Feature Engineering**
   ```python
   def extract_advanced_vad_features(audio, sr=16000):
       """Extract comprehensive features for VAD"""
       
       # Time-domain features
       energy = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
       zcr = librosa.feature.zero_crossing_rate(audio, frame_length=512, hop_length=256)[0]
       
       # Frequency-domain features
       stft = librosa.stft(audio, n_fft=512, hop_length=256)
       magnitude = np.abs(stft)
       
       # Spectral features
       spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
       spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
       spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
       spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
       
       # Cepstral features
       mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
       delta_mfcc = librosa.feature.delta(mfcc)
       delta2_mfcc = librosa.feature.delta(mfcc, order=2)
       
       # Pitch features
       f0, voiced_flag, voiced_probs = librosa.pyin(
           audio, fmin=50, fmax=400, sr=sr
       )
       
       # Combine all features
       features = np.vstack([
           energy,
           zcr,
           spectral_centroid,
           spectral_bandwidth,
           spectral_rolloff,
           spectral_flatness,
           mfcc,
           delta_mfcc,
           delta2_mfcc,
           voiced_probs
       ]).T
       
       return features
   ```

### Model Optimization Techniques

1. **Quantization for Edge Deployment**
   ```python
   def quantize_vad_model(model, calibration_data):
       """Quantize VAD model for edge deployment"""
       import torch.quantization as quant
       
       # Prepare model for quantization
       model.eval()
       model.qconfig = quant.get_default_qconfig('fbgemm')
       
       # Fuse modules
       model_fused = quant.fuse_modules(
           model,
           [['conv', 'bn', 'relu']] * len(model.features)
       )
       
       # Prepare for quantization
       model_prepared = quant.prepare(model_fused)
       
       # Calibrate with representative data
       with torch.no_grad():
           for batch in calibration_data:
               model_prepared(batch)
       
       # Convert to quantized model
       model_quantized = quant.convert(model_prepared)
       
       # Verify size reduction
       original_size = get_model_size(model)
       quantized_size = get_model_size(model_quantized)
       print(f"Size reduction: {original_size/quantized_size:.2f}x")
       
       return model_quantized
   ```

2. **Knowledge Distillation**
   ```python
   class VADDistillationTrainer:
       """Train lightweight VAD using knowledge distillation"""
       
       def __init__(self, teacher_model, student_model, temperature=3.0):
           self.teacher = teacher_model
           self.student = student_model
           self.temperature = temperature
           self.criterion = nn.KLDivLoss(reduction='batchmean')
           
       def distillation_loss(self, student_logits, teacher_logits, labels, alpha=0.7):
           """Compute distillation loss"""
           # Soft targets from teacher
           teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
           student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
           
           # Distillation loss
           distill_loss = self.criterion(student_log_probs, teacher_probs) * (self.temperature ** 2)
           
           # Student loss
           student_loss = F.cross_entropy(student_logits, labels)
           
           # Combined loss
           total_loss = alpha * distill_loss + (1 - alpha) * student_loss
           
           return total_loss
   ```

### Real-Time Processing Strategies

1. **Streaming VAD with Ring Buffer**
   ```python
   class StreamingVAD:
       """Real-time streaming VAD implementation"""
       
       def __init__(self, model, sample_rate=16000, chunk_duration=0.032):
           self.model = model
           self.sample_rate = sample_rate
           self.chunk_size = int(chunk_duration * sample_rate)
           
           # Ring buffer for context
           self.context_duration = 0.5  # 500ms context
           self.buffer_size = int(self.context_duration * sample_rate)
           self.ring_buffer = RingBuffer(self.buffer_size)
           
           # Feature extractor
           self.feature_extractor = StreamingFeatureExtractor(sample_rate)
           
           # Smoothing
           self.smoother = HangoverSmoother(
               hangover_frames=10,
               onset_frames=3
           )
           
       def process_chunk(self, audio_chunk):
           """Process single chunk in real-time"""
           # Add to ring buffer
           self.ring_buffer.append(audio_chunk)
           
           # Get context window
           context = self.ring_buffer.get_data()
           
           # Extract features
           features = self.feature_extractor.extract(context)
           
           # Run model inference
           with torch.no_grad():
               logits = self.model(features.unsqueeze(0))
               probs = F.softmax(logits, dim=-1)
           
           # Apply smoothing
           smoothed_decision = self.smoother.smooth(probs[0, -1, 1].item())
           
           return {
               'is_speech': smoothed_decision > 0.5,
               'probability': smoothed_decision,
               'timestamp': self.get_current_timestamp()
           }
   ```

2. **GPU-Accelerated Batch Processing**
   ```python
   class BatchVADProcessor:
       """Process multiple audio streams in parallel on GPU"""
       
       def __init__(self, model, batch_size=32, device='cuda'):
           self.model = model.to(device)
           self.batch_size = batch_size
           self.device = device
           
           # Pre-allocate GPU tensors
           self.feature_buffer = torch.zeros(
               (batch_size, 1000, 40),  # max 1000 frames
               device=device
           )
           
       def process_batch(self, audio_batch):
           """Process batch of audio files"""
           results = []
           
           # Extract features in parallel
           features_list = []
           with ThreadPoolExecutor(max_workers=8) as executor:
               features_list = list(executor.map(
                   self.extract_features,
                   audio_batch
               ))
           
           # Pad and batch features
           features_tensor = self.pad_and_batch(features_list)
           
           # Run batch inference
           with torch.no_grad():
               logits = self.model(features_tensor)
               probs = F.softmax(logits, dim=-1)
           
           # Post-process results
           for i, (audio, feat_len) in enumerate(zip(audio_batch, features_list)):
               speech_probs = probs[i, :feat_len.shape[0], 1].cpu().numpy()
               segments = self.probs_to_segments(speech_probs)
               
               results.append({
                   'segments': segments,
                   'speech_ratio': np.mean(speech_probs > 0.5),
                   'confidence': np.mean(np.maximum(speech_probs, 1 - speech_probs))
               })
           
           return results
   ```

### Performance Optimization

1. **ONNX Export for Deployment**
   ```python
   def export_vad_to_onnx(model, output_path, optimize=True):
       """Export VAD model to ONNX format"""
       model.eval()
       
       # Dummy input
       dummy_input = torch.randn(1, 100, 40)  # (batch, time, features)
       
       # Export
       torch.onnx.export(
           model,
           dummy_input,
           output_path,
           export_params=True,
           opset_version=11,
           do_constant_folding=True,
           input_names=['input'],
           output_names=['output'],
           dynamic_axes={
               'input': {0: 'batch_size', 1: 'sequence_length'},
               'output': {0: 'batch_size', 1: 'sequence_length'}
           }
       )
       
       if optimize:
           # Optimize ONNX model
           import onnx
           from onnxruntime.transformers import optimizer
           
           model_onnx = onnx.load(output_path)
           optimized_model = optimizer.optimize_model(
               model_onnx,
               model_type='bert',  # Use BERT optimizations
               num_heads=8,
               hidden_size=256
           )
           
           onnx.save(optimized_model, output_path.replace('.onnx', '_optimized.onnx'))
   ```

2. **TensorRT Optimization**
   ```python
   def optimize_with_tensorrt(onnx_path, trt_path, fp16=True):
       """Optimize VAD model with TensorRT"""
       import tensorrt as trt
       
       TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
       
       with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
           
           # Configure builder
           config = builder.create_builder_config()
           config.max_workspace_size = 1 << 30  # 1GB
           
           if fp16:
               config.set_flag(trt.BuilderFlag.FP16)
           
           # Parse ONNX
           with open(onnx_path, 'rb') as model:
               parser.parse(model.read())
           
           # Build engine
           engine = builder.build_engine(network, config)
           
           # Save engine
           with open(trt_path, 'wb') as f:
               f.write(engine.serialize())
           
           return engine
   ```

### Integration Example

```python
class ProductionVADSystem:
    """Production-ready VAD system with all optimizations"""
    
    def __init__(self, model_path, device='cuda', use_tensorrt=True):
        self.device = device
        
        # Load optimized model
        if use_tensorrt and device == 'cuda':
            self.engine = self._load_tensorrt_engine(model_path)
            self.inference_fn = self._tensorrt_inference
        else:
            self.model = self._load_torch_model(model_path)
            self.inference_fn = self._torch_inference
        
        # Feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor()
        
        # Post-processor
        self.post_processor = VADPostProcessor(
            min_speech_duration=0.3,
            min_silence_duration=0.3,
            speech_pad_duration=0.1
        )
        
    def detect_speech(self, audio, return_probability=False):
        """Detect speech in audio"""
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Run inference
        probabilities = self.inference_fn(features)
        
        # Post-process
        segments = self.post_processor.process(probabilities)
        
        if return_probability:
            return segments, probabilities
        
        return segments
        
    def process_stream(self, audio_stream, chunk_duration=0.032):
        """Process audio stream in real-time"""
        streaming_vad = StreamingVAD(
            self.inference_fn,
            chunk_duration=chunk_duration
        )
        
        for chunk in audio_stream:
            result = streaming_vad.process_chunk(chunk)
            yield result
```