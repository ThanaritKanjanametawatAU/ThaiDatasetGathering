# Task S03_T05: Build Dominant Speaker Identifier

## Task Overview
Build a system that identifies the dominant speaker in audio segments, determining which speaker contributes most to the conversation and should be preserved in enhancement.

## Technical Requirements

### Core Implementation
- **Dominant Speaker Identifier** (`processors/audio_enhancement/analysis/dominant_speaker_identifier.py`)
  - Speaking time analysis
  - Energy contribution measurement
  - Linguistic importance scoring
  - Confidence estimation

### Key Features
1. **Analysis Metrics**
   - Total speaking duration
   - Average energy levels
   - Turn-taking patterns
   - Interruption analysis

2. **Scoring Methods**
   - Time-based dominance
   - Energy-based dominance
   - Content-based dominance
   - Weighted composite scoring

3. **Context Awareness**
   - Conversation type detection
   - Role identification
   - Cultural pattern recognition
   - Domain-specific rules

## TDD Requirements

### Test Structure
```
tests/test_dominant_speaker_identifier.py
- test_time_based_dominance()
- test_energy_based_dominance()
- test_multi_speaker_scenarios()
- test_edge_cases()
- test_scoring_accuracy()
- test_real_conversations()
```

### Test Data Requirements
- Balanced conversations
- Dominant speaker scenarios
- Interview recordings
- Group discussions

## Implementation Approach

### Phase 1: Core Identifier
```python
class DominantSpeakerIdentifier:
    def __init__(self, method='composite'):
        self.method = method
        self.weights = self._init_weights()
        
    def identify_dominant(self, audio, diarization):
        # Identify dominant speaker
        pass
    
    def compute_dominance_scores(self, speaker_segments):
        # Calculate dominance for each speaker
        pass
    
    def get_speaker_statistics(self, audio, speaker_id):
        # Detailed statistics for a speaker
        pass
```

### Phase 2: Advanced Analysis
- Machine learning models
- Contextual understanding
- Multi-modal analysis
- Adaptive weighting

### Phase 3: Integration
- Real-time identification
- Pipeline integration
- Visualization tools
- API endpoints

## Acceptance Criteria
1. ✅ Identification accuracy > 95%
2. ✅ Support for up to 10 speakers
3. ✅ Processing speed > 100x real-time
4. ✅ Confidence scores provided
5. ✅ Handle edge cases gracefully

## Example Usage
```python
from processors.audio_enhancement.analysis import DominantSpeakerIdentifier

# Initialize identifier
identifier = DominantSpeakerIdentifier(method='composite')

# Identify dominant speaker
result = identifier.identify_dominant(audio_data, diarization_result)
print(f"Dominant speaker: {result.speaker_id}")
print(f"Dominance score: {result.score:.2f}")
print(f"Confidence: {result.confidence:.2f}")

# Get all speaker scores
scores = identifier.compute_dominance_scores(diarization_result)
for speaker_id, score in scores.items():
    print(f"Speaker {speaker_id}: {score:.2f}")

# Get detailed statistics
stats = identifier.get_speaker_statistics(audio_data, 'SPEAKER_01')
print(f"Speaking time: {stats.duration:.1f}s ({stats.percentage:.1f}%)")
print(f"Average energy: {stats.avg_energy:.2f} dB")
print(f"Number of turns: {stats.num_turns}")
```

## Dependencies
- NumPy for calculations
- Pandas for data analysis
- SciPy for statistics
- Librosa for audio analysis
- Matplotlib for visualization

## Performance Targets
- Identification time: < 100ms
- Memory usage: < 100MB
- Accuracy: > 95%
- Support for long audio: > 1 hour

## Notes
- Consider conversation context
- Handle speaker role detection
- Support for weighted importance
- Enable custom dominance criteria