# Secondary Speaker Removal Implementation Plan

## 1. Executive Summary

This plan addresses the issue of secondary speakers (overlapping speech) remaining in the Thai audio dataset despite existing preprocessing. The solution involves fixing the current implementation, adding robust speaker separation models, and implementing automated evaluation metrics.

**Key Objectives:**
- Remove overlapping speech from all audio samples
- Preserve primary speaker with minimal quality loss
- Implement automated evaluation for quality assurance
- Create manual test cases for validation

**Success Criteria:**
- 95%+ of overlapping speech segments detected and removed
- Primary speaker similarity score > 0.95
- Automated metrics showing improvement (SI-SDR, PESQ, STOI)
- Manual evaluation confirming single-speaker audio

**Estimated Timeline:** 2-3 weeks

## 2. Business Requirements

### Problem Statement
The current audio enhancement pipeline with `ultra_aggressive` mode is not effectively removing secondary speakers from audio samples. Users report hearing overlapping speech in the processed dataset, which degrades the quality for speech recognition and synthesis tasks.

### User Personas
1. **ML Engineers**: Need clean single-speaker audio for training ASR/TTS models
2. **Dataset Curators**: Require high-quality datasets meeting specific standards
3. **Researchers**: Need reliable data for speech processing experiments

### Functional Requirements

#### FR1: Secondary Speaker Detection
- Detect overlapping speech segments with >90% accuracy
- Support detection of multiple simultaneous speakers
- Identify segments as short as 0.05 seconds
- Work across all dataset sources (GigaSpeech2, ProcessedVoiceTH, Mozilla CV)

#### FR2: Speaker Separation
- Isolate primary speaker with minimal artifacts
- Maintain audio quality (PESQ > 3.0, STOI > 0.85)
- Preserve speaker characteristics (similarity > 0.95)
- Handle various overlap scenarios (partial, full, multiple speakers)

#### FR3: Quality Assurance
- Automated evaluation metrics for all processed samples
- Manual test cases for validation
- Exclusion of samples where separation fails
- Transcription of audio without existing transcripts after processing

### Non-Functional Requirements

#### Performance
- Process 100 samples in < 3 minutes on GPU
- Support batch processing for efficiency
- Fallback to CPU if GPU unavailable

#### Quality
- No audible artifacts in separated audio
- Consistent volume levels
- Natural-sounding speech

#### Scalability
- Handle datasets with millions of samples
- Support streaming and cached modes
- Resume from checkpoints

## 3. Technical Architecture

### Current State Analysis

The existing implementation has several issues:

1. **Configuration Mismatch**: The `ultra_aggressive` enhancement level checks for `check_secondary` or `use_speaker_separation` flags, but these are not set
2. **Weak Detection**: The similarity threshold (0.7) is too high for effective detection
3. **Limited Separation**: The suppression strength (0.6) is insufficient
4. **No Verification**: No automated metrics to verify removal success

### Proposed Architecture

```
┌─────────────────────┐
│   Audio Input       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Overlap Detection   │ ← PyAnnote OSD
│ (Multi-method)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Decision Logic      │
│ ├─ No overlap → Pass│
│ ├─ Overlap → Sep.   │
│ └─ Failed → Exclude │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Speaker Separation  │ ← SepFormer/Conv-TasNet
│ (Primary isolation) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Quality Validation  │ ← SI-SDR, PESQ, STOI
│ & Metrics          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Clean Output      │
└─────────────────────┘
```

### Technology Stack

1. **Detection Models**:
   - PyAnnote Overlapped Speech Detection
   - Energy-based detection (existing)
   - VAD-based detection (existing)
   - Spectral analysis (existing)

2. **Separation Models**:
   - SepFormer (state-of-the-art, best quality)
   - Conv-TasNet (faster, good quality)
   - Spectral masking (fallback)

3. **Evaluation Metrics**:
   - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
   - PESQ (Perceptual Evaluation of Speech Quality)
   - STOI (Short-Time Objective Intelligibility)
   - Speaker similarity (cosine similarity of embeddings)

## 4. Data Models & Schemas

### Detection Result Schema
```python
@dataclass
class OverlapDetection:
    start_time: float
    end_time: float
    confidence: float
    overlap_ratio: float  # % of segment with overlap
    num_speakers: int
    detection_method: str
```

### Separation Result Schema
```python
@dataclass
class SeparationResult:
    success: bool
    primary_audio: np.ndarray
    metrics: Dict[str, float]  # SI-SDR, PESQ, STOI
    separation_method: str
    processing_time: float
    excluded_reason: Optional[str]  # If failed
```

### Enhanced Metadata Schema
```python
enhancement_metadata = {
    "enhanced": bool,
    "noise_level": str,
    "enhancement_level": str,
    "secondary_speaker_detected": bool,
    "overlap_segments": List[OverlapDetection],
    "separation_applied": bool,
    "separation_method": str,
    "si_sdr_improvement": float,
    "pesq_score": float,
    "stoi_score": float,
    "speaker_similarity": float,
    "excluded": bool,
    "exclusion_reason": Optional[str]
}
```

## 5. Implementation Plan

### Phase 1: Fix Current Implementation (Week 1)

#### Task 1.1: Configuration Fix
- Modify `AudioEnhancer` to properly enable secondary speaker removal for `ultra_aggressive` mode
- Set `check_secondary=True` and `use_speaker_separation=True` flags
- Lower detection thresholds for better sensitivity

#### Task 1.2: Enhance Detection
- Integrate PyAnnote overlapped speech detection model
- Combine with existing detection methods (voting ensemble)
- Implement confidence scoring for detections

#### Task 1.3: Strengthen Separation
- Increase suppression strength to 0.95
- Lower similarity threshold to 0.5
- Implement aggressive spectral masking

### Phase 2: Advanced Separation Models (Week 2)

#### Task 2.1: Integrate SepFormer
- Add SepFormer model for high-quality separation
- Implement batch processing for efficiency
- Add fallback to Conv-TasNet if memory constrained

#### Task 2.2: Implement Exclusion Logic
- Define failure criteria (SI-SDR < threshold)
- Implement automatic exclusion of failed samples
- Log exclusion reasons for analysis

#### Task 2.3: Post-Processing
- Apply spectral smoothing to reduce artifacts
- Normalize audio levels
- Verify single-speaker output

### Phase 3: Evaluation & Testing (Week 3)

#### Task 3.1: Automated Metrics
- Implement SI-SDR calculation
- Add PESQ and STOI evaluation
- Create evaluation dashboard

#### Task 3.2: Manual Test Cases
- Create 100-sample test set with known overlaps
- Define evaluation criteria
- Implement A/B comparison interface

#### Task 3.3: Integration Testing
- Test with all dataset sources
- Verify streaming mode compatibility
- Ensure checkpoint resumability

## 6. Testing Strategy

### Unit Tests
```python
# Test overlap detection accuracy
def test_overlap_detection():
    # Create synthetic overlap
    # Verify detection
    # Check confidence scores

# Test separation quality
def test_speaker_separation():
    # Mix two speakers
    # Separate
    # Verify SI-SDR improvement

# Test exclusion logic
def test_exclusion_criteria():
    # Create poor quality mix
    # Verify exclusion
    # Check reason logging
```

### Integration Tests
- End-to-end processing with real samples
- Batch processing verification
- Checkpoint/resume functionality
- GPU/CPU fallback testing

### Manual Evaluation Test Cases

#### Test Set Creation
1. **Clean Single Speaker** (20 samples) - Baseline
2. **Partial Overlap** (20 samples) - 10-30% overlap
3. **Heavy Overlap** (20 samples) - 50%+ overlap
4. **Multiple Speakers** (20 samples) - 3+ speakers
5. **Challenging Cases** (20 samples) - Similar voices, background noise

#### Evaluation Criteria
- **Primary**: No audible secondary speaker
- **Quality**: Natural sounding, no artifacts
- **Intelligibility**: Clear speech, correct transcription
- **Consistency**: Similar quality across samples

### Performance Testing
- Benchmark processing speed (samples/second)
- Memory usage monitoring
- GPU utilization tracking
- Scalability testing (1K, 10K, 100K samples)

## 7. Deployment & Operations

### Deployment Strategy
1. **Testing Environment**: Deploy to test branch first
2. **Gradual Rollout**: Process subset of data initially
3. **Monitoring**: Track metrics and error rates
4. **Full Deployment**: Apply to entire dataset

### Configuration Updates
```bash
# Updated main.sh configuration
ENHANCEMENT_LEVEL="secondary_speaker_removal"  # New dedicated mode
SPEAKER_SEPARATION_MODEL="sepformer"  # Or "conv-tasnet"
OVERLAP_DETECTION_THRESHOLD="0.3"  # Sensitive detection
SEPARATION_CONFIDENCE_THRESHOLD="0.7"  # Quality threshold
EXCLUDE_FAILED_SEPARATIONS="true"  # Remove poor quality
```

### Monitoring Requirements
- Track detection rate (% samples with overlap detected)
- Monitor separation success rate
- Log exclusion statistics
- Performance metrics (processing time, resource usage)

### Operational Procedures
1. **Daily Monitoring**: Check processing statistics
2. **Quality Audits**: Random sample verification
3. **Error Investigation**: Analyze failed separations
4. **Model Updates**: Periodic retraining if needed

## 8. Future Considerations

### Potential Enhancements
1. **Multi-Speaker Separation**: Handle 3+ simultaneous speakers
2. **Real-Time Processing**: Streaming separation
3. **Language-Specific Models**: Thai-optimized separation
4. **Active Learning**: Improve models with user feedback

### Scalability Planning
- Distributed processing for large datasets
- Model quantization for faster inference
- Caching of separation results
- Progressive enhancement strategies

### Technical Debt
- Refactor detection pipeline for modularity
- Optimize memory usage in batch processing
- Improve error handling and recovery
- Document all configuration options

### Research Opportunities
- Explore unsupervised separation methods
- Investigate Thai-specific speech characteristics
- Develop custom evaluation metrics
- Study impact on downstream tasks (ASR/TTS)

## Appendix: Implementation Code Structure

### Key Files to Modify
1. `processors/audio_enhancement/core.py` - Main enhancement logic
2. `processors/audio_enhancement/detection/overlap_detector.py` - New overlap detection
3. `processors/audio_enhancement/separation/` - New separation models
4. `processors/audio_enhancement/evaluation/` - Metrics implementation
5. `config.py` - New configuration options
6. `main.sh` - Updated parameters

### Configuration Example
```python
SECONDARY_SPEAKER_CONFIG = {
    "detection": {
        "models": ["pyannote", "energy", "vad", "spectral"],
        "voting_threshold": 0.5,  # Majority vote
        "confidence_threshold": 0.3
    },
    "separation": {
        "primary_model": "sepformer",
        "fallback_model": "conv-tasnet",
        "quality_threshold": {
            "si_sdr": 10.0,  # dB
            "pesq": 3.0,
            "stoi": 0.85
        }
    },
    "exclusion": {
        "enabled": True,
        "max_attempts": 2,
        "log_excluded": True
    }
}
```