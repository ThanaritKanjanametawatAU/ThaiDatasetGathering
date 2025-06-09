# Requirement R03: Advanced Secondary Speaker Detection

## Overview
Implement state-of-the-art secondary speaker detection and removal using ML approaches, achieving production-grade accuracy for handling 10M+ diverse audio samples.

## Acceptance Criteria

### 1. PyAnnote.audio Integration
- [ ] Diarization Error Rate (DER) < 6.3%
- [ ] Real-time factor < 2.5% on standard hardware
- [ ] Support for unknown number of speakers
- [ ] Automatic speaker counting

### 2. Voice Embedding System
- [ ] ECAPA-TDNN implementation
- [ ] 192-dimensional embeddings with attention
- [ ] Equal Error Rate (EER) < 0.96% on VoxCeleb
- [ ] Embedding extraction < 50ms per utterance

### 3. Speaker Verification
- [ ] Cosine similarity computation
- [ ] Configurable thresholds (0.6-0.8)
- [ ] PLDA scoring implementation
- [ ] 10-20% improvement over cosine similarity

### 4. Overlapping Speech Detection
- [ ] Temporal convolutional networks
- [ ] 29.1% improvement over traditional methods
- [ ] CountNet for concurrent speaker estimation
- [ ] Human-level performance for 3+ speakers

### 5. Residual Voice Detection
- [ ] Spectral analysis for voice remnants
- [ ] Temporal consistency checking
- [ ] Multi-window embedding analysis
- [ ] Confidence scoring for detections

## Technical Implementation

```python
class SecondarySpeakerDetector:
    def __init__(self, config: Dict[str, Any]):
        self.diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        self.embedding_model = Model.from_pretrained("pyannote/embedding")
        self.overlap_detector = OverlapNet()
        self.plda_scorer = PLDAScorer.load(config['plda_model_path'])
        
    def detect_speakers(self, audio: np.ndarray, sr: int) -> SpeakerDetectionResult:
        # Perform diarization
        diarization = self.diarization({"waveform": audio, "sample_rate": sr})
        
        # Extract embeddings for each speaker
        embeddings = {}
        for speaker, segments in diarization.itertracks(yield_label=True):
            speaker_audio = self.extract_segments(audio, segments, sr)
            embeddings[speaker] = self.embedding_model(speaker_audio)
        
        # Detect overlapping regions
        overlap_regions = self.overlap_detector.detect(audio, sr)
        
        # Identify primary speaker
        primary_speaker = self.identify_primary_speaker(embeddings, diarization)
        
        # Find secondary speakers
        secondary_speakers = []
        for speaker, embedding in embeddings.items():
            if speaker != primary_speaker:
                confidence = self.verify_secondary_speaker(
                    embedding, 
                    embeddings[primary_speaker],
                    method='plda'
                )
                if confidence > self.config['secondary_threshold']:
                    secondary_speakers.append({
                        'speaker': speaker,
                        'confidence': confidence,
                        'segments': diarization[speaker]
                    })
        
        return SpeakerDetectionResult(
            primary=primary_speaker,
            secondary=secondary_speakers,
            overlaps=overlap_regions,
            embeddings=embeddings
        )
    
    def verify_secondary_speaker(self, emb1: np.ndarray, emb2: np.ndarray, 
                                method: str = 'cosine') -> float:
        if method == 'cosine':
            return 1 - cosine(emb1, emb2)
        elif method == 'plda':
            return self.plda_scorer.score(emb1, emb2)
        else:
            raise ValueError(f"Unknown method: {method}")
```

## Performance Requirements
- Diarization: < 1s per minute of audio
- Embedding extraction: < 100ms per speaker
- Overall detection: < 2s for 30s audio
- Memory usage: < 2GB for model loading

## Advanced Features
- Multi-scale temporal analysis
- Cross-channel speaker tracking
- Adaptive threshold adjustment
- Online diarization capability

## Testing Requirements
- Benchmark on standard datasets (VoxCeleb, AMI)
- Multi-speaker scenario testing
- Overlapping speech validation
- Edge case handling (whispers, shouting)

## Dependencies
- pyannote.audio >= 3.1
- speechbrain >= 0.5
- torch >= 2.0
- scikit-learn (for PLDA)

## Definition of Done
- [ ] All detection components implemented
- [ ] Performance benchmarks met
- [ ] Accuracy targets achieved
- [ ] Robust error handling
- [ ] Comprehensive documentation
- [ ] Integration with removal pipeline