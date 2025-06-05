# SpeechBrain Secondary Speaker Removal Implementation Plan

## 1. Executive Summary

This plan outlines the integration of SpeechBrain's SepFormer model to completely remove secondary speakers from audio files for voice cloning training. The solution will replace the current ineffective speaker separation system with a robust, GPU-accelerated approach that guarantees single-speaker output.

**Key Objectives:**
- Complete removal of all secondary speakers from audio files
- Process 100 samples in under 3 minutes using RTX 5090 (32GB VRAM)
- Maintain high audio quality for voice cloning training
- Automatic rejection of files that cannot be cleaned to single-speaker quality

**Success Criteria:**
- Zero audible secondary speaker content in processed audio
- 0.7+ confidence threshold for main speaker selection
- Processing speed: <1.8 seconds per 8-second audio file
- 90%+ success rate on typical mixed-speaker audio

**Estimated Timeline:** 2-3 weeks for full implementation and testing

## 2. Business Requirements

### Problem Statement
The current audio enhancement pipeline detects secondary speakers but fails to remove them effectively. Samples S3 and S5 demonstrate this issue where secondary speakers remain audible despite using `--enhancement-level ultra_aggressive`. This contamination makes the audio unsuitable for voice cloning training, which requires clean single-speaker audio.

### User Personas and Use Cases
- **Primary User:** AI/ML Engineer preparing datasets for voice cloning
- **Use Case 1:** Batch process Thai audio datasets to extract only primary speaker
- **Use Case 2:** Automatically reject audio files with inseparable speakers
- **Use Case 3:** Generate quality metrics for processed audio

### Functional Requirements
1. **Complete Secondary Speaker Removal**
   - FR1.1: Remove all secondary speaker audio completely
   - FR1.2: Preserve primary speaker quality (PESQ > 3.5, STOI > 0.85)
   - FR1.3: Handle overlapping speech, turn-taking, and background conversations
   - FR1.4: Process 4-8 second audio clips efficiently

2. **Speaker Selection Logic**
   - FR2.1: Automatically identify primary speaker using energy-based heuristics
   - FR2.2: Support manual speaker selection override
   - FR2.3: Provide confidence scores for speaker selection

3. **Quality Assurance**
   - FR3.1: Validate single-speaker output
   - FR3.2: Reject files below 0.7 confidence threshold
   - FR3.3: Generate detailed processing metrics

### Non-Functional Requirements
- **Performance:** Process 100 samples in under 180 seconds
- **Scalability:** Handle batch sizes up to 32 samples (GPU memory permitting)
- **Reliability:** 99%+ uptime with graceful error handling
- **Compatibility:** Integrate seamlessly with existing pipeline

### Acceptance Criteria
1. No audible secondary speaker in processed audio (verified by listening tests)
2. Processing time meets performance requirements
3. Quality metrics meet thresholds (PESQ > 3.5, STOI > 0.85)
4. Successfully processes test samples S3 and S5

## 3. Technical Architecture

### System Architecture Overview
```
Input Audio (4-8s) → Pre-processing → Speaker Diarization → SepFormer Separation 
                                           ↓
Output Audio ← Quality Validation ← Main Speaker Selection ← Post-processing
```

### Technology Stack
- **Core Framework:** SpeechBrain 1.0.0+
- **Separation Model:** SepFormer (speechbrain/sepformer-whamr16k)
- **Speaker Embeddings:** ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
- **GPU Framework:** PyTorch 2.0+ with CUDA 12.x
- **Audio Processing:** torchaudio, librosa
- **Quality Metrics:** pypesq, pystoi

### Component Architecture

1. **SpeechBrainSeparator Class** (replaces current speaker_separation.py)
   - Manages SepFormer model loading and inference
   - Handles GPU memory optimization
   - Provides batch processing capabilities

2. **SpeakerSelector Module**
   - Implements energy-based primary speaker detection
   - Calculates speaker embeddings for verification
   - Provides confidence scoring

3. **QualityValidator Module**
   - Verifies single-speaker output
   - Calculates audio quality metrics
   - Implements rejection logic

### Integration Points
- Replaces `processors/audio_enhancement/speaker_separation.py`
- Integrates with `AudioEnhancer` in `core.py`
- Uses existing audio metrics from `utils/audio_metrics.py`

## 4. Data Models & Schemas

### Input Schema
```python
class SeparationInput:
    audio: np.ndarray  # Audio signal (mono, 16kHz)
    sample_rate: int = 16000
    config: SeparationConfig
```

### Separation Configuration
```python
@dataclass
class SeparationConfig:
    model_name: str = "speechbrain/sepformer-whamr16k"
    device: str = "cuda"
    batch_size: int = 16  # Optimized for RTX 5090
    confidence_threshold: float = 0.7
    quality_thresholds: Dict = field(default_factory=lambda: {
        "min_pesq": 3.5,
        "min_stoi": 0.85,
        "max_spectral_distortion": 0.15
    })
    speaker_selection: str = "energy"  # energy|embedding|manual
    chunk_duration: float = 10.0  # Max chunk size for memory efficiency
```

### Output Schema
```python
class SeparationOutput:
    audio: np.ndarray  # Cleaned single-speaker audio
    confidence: float  # Confidence score (0-1)
    metrics: Dict[str, float]  # Quality metrics
    rejected: bool  # Whether audio was rejected
    rejection_reason: Optional[str]
    processing_time_ms: float
```

### Processing Metrics
```python
class ProcessingMetrics:
    snr_improvement: float
    pesq_score: float
    stoi_score: float
    spectral_distortion: float
    num_speakers_detected: int
    primary_speaker_duration: float
    secondary_speaker_duration: float
    overlap_duration: float
```

## 5. Implementation Plan

### Development Phases

**Phase 1: Core SpeechBrain Integration (Week 1)**
- Task 1.1: Set up SpeechBrain environment and dependencies
- Task 1.2: Implement SpeechBrainSeparator class
- Task 1.3: Create GPU memory management utilities
- Task 1.4: Implement basic SepFormer inference
- Task 1.5: Write unit tests for core functionality

**Phase 2: Speaker Selection & Processing (Week 2)**
- Task 2.1: Implement energy-based speaker selection
- Task 2.2: Add speaker embedding comparison
- Task 2.3: Create batch processing pipeline
- Task 2.4: Implement chunking for long audio
- Task 2.5: Add confidence scoring

**Phase 3: Quality Validation & Integration (Week 3)**
- Task 3.1: Implement quality validation module
- Task 3.2: Add rejection logic with thresholds
- Task 3.3: Integrate with AudioEnhancer
- Task 3.4: Update command-line interface
- Task 3.5: Performance optimization

### Task Dependencies
```
1.1 → 1.2 → 1.3 → 1.4 → 1.5
         ↓
      2.1 → 2.2 → 2.3 → 2.4 → 2.5
                        ↓
                  3.1 → 3.2 → 3.3 → 3.4 → 3.5
```

### Risk Assessment
1. **GPU Memory Limitations**
   - Risk: OOM errors with large batches
   - Mitigation: Dynamic batch sizing, chunking strategy

2. **Model Performance**
   - Risk: Slower than 3-minute target
   - Mitigation: Batch optimization, mixed precision

3. **Quality Degradation**
   - Risk: Over-aggressive separation damages audio
   - Mitigation: Post-processing, quality thresholds

### Resource Requirements
- 1 Senior ML Engineer (full-time, 3 weeks)
- RTX 5090 GPU (32GB VRAM) for development/testing
- Access to mixed-speaker test dataset
- Voice cloning model for validation

## 6. Testing Strategy

### Unit Testing
```python
# Test speaker separation accuracy
def test_complete_secondary_removal():
    """Verify no secondary speaker remains"""
    
# Test performance requirements
def test_processing_speed():
    """Verify 100 samples < 3 minutes"""
    
# Test quality preservation
def test_audio_quality_metrics():
    """Verify PESQ > 3.5, STOI > 0.85"""
```

### Integration Testing
1. Test with existing enhancement pipeline
2. Verify command-line interface functionality
3. Test batch processing with various sizes
4. Validate checkpoint/resume functionality

### End-to-End Testing Scenarios
1. **Overlapping Speech:** Two speakers talking simultaneously
2. **Turn-Taking:** Speakers alternating clearly
3. **Background Chatter:** Main speaker with crowd noise
4. **Mixed Scenarios:** Combination of above

### Performance Testing
- Benchmark: 100 8-second samples
- Target: < 180 seconds total processing
- Measure: GPU utilization, memory usage, RTF

### Test Data Requirements
- 50 samples with known secondary speakers
- 20 samples with overlapping speech
- 20 samples with background conversations
- 10 edge cases (whispers, similar voices)

## 7. Deployment & Operations

### Deployment Strategy
1. Replace existing `speaker_separation.py`
2. Update AudioEnhancer to use new module
3. Add SpeechBrain models to model cache
4. Update documentation and examples

### Environment Configuration
```yaml
# Environment variables
SPEECHBRAIN_CACHE: /path/to/models
CUDA_VISIBLE_DEVICES: 0
PYTORCH_CUDA_ALLOC_CONF: max_split_size_mb:512

# Model downloads
models:
  - speechbrain/sepformer-whamr16k
  - speechbrain/spkrec-ecapa-voxceleb
```

### Monitoring Requirements
- Processing speed (samples/second)
- Rejection rate
- Quality metrics distribution
- GPU memory usage
- Error rates by type

### Maintenance Considerations
- Model updates from SpeechBrain
- Performance tuning based on usage
- Quality threshold adjustments
- Cache management

## 8. Future Considerations

### Potential Enhancements
1. **Multi-GPU Support:** Distribute processing across multiple GPUs
2. **Advanced Models:** Explore newer separation models (MossFormer, etc.)
3. **Real-time Processing:** Streaming separation for live audio
4. **Language-Specific Models:** Thai-optimized separation

### Scalability Planning
- Cloud GPU deployment options
- Containerization with Docker
- Kubernetes orchestration
- Auto-scaling based on queue size

### Technical Debt Considerations
- Regular model performance benchmarking
- Code optimization opportunities
- Documentation updates
- Test coverage expansion

### Knowledge Transfer
- Developer documentation
- Usage examples
- Performance tuning guide
- Troubleshooting playbook