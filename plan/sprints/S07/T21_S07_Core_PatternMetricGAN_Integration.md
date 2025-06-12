# Task 21 - S07: Core Pattern→MetricGAN+ Integration

## Overview
Integrate the proven Pattern→MetricGAN+ → 160% loudness enhancement pipeline into the main audio processing framework as a native enhancement level, following the established BaseProcessor and Factory patterns used throughout the codebase.

## Background
The Pattern→MetricGAN+ approach has been extensively tested and proven effective:
- **Ultra-conservative pattern detection** (>0.8 confidence threshold)
- **Gentle suppression** (50ms padding, 85% suppression, keep 15% of original)
- **MetricGAN+ enhancement** for overall quality improvement
- **160% loudness normalization** to match original levels

This approach provides excellent primary speaker preservation while effectively removing interruptions with natural sound quality and no audio clipping.

## Architectural Guidance
This task follows established architectural decisions documented in the project:

### Referenced Architecture Documents:
- **`docs/architecture.md`**: BaseProcessor Pattern - All audio processing must inherit from BaseProcessor
- **`docs/audio_analysis_architecture.md`**: 6-Stage Pipeline integration via DecisionEngine and EnhancementOrchestrator
- **`processors/audio_enhancement/decision_framework.py`**: Strategy Pattern - Must integrate as new enhancement strategy
- **`processors/audio_enhancement/loudness_normalizer.py`**: Extend existing LoudnessNormalizer (don't replace)
- **`docs/secondary_speaker_implementation_summary.md`**: TDD Requirements - 37 tests passing standard with comprehensive edge case coverage

### Key Architectural Constraints:
- **Modular Design**: Must not break existing processor architecture
- **Factory Pattern**: Use established factory pattern for processor creation  
- **Strategy Pattern**: Integrate with existing strategy selection system
- **Performance**: Must meet <100ms latency and >10 files/second targets
- **Backward Compatibility**: All existing enhancement levels must continue working

## Technical Requirements

### 1. Core Integration Architecture

#### 1.1 Enhancement Level Registration
- Add `pattern_metricgan_plus` enhancement level to `AudioEnhancer.ENHANCEMENT_LEVELS`
- Configure with appropriate parameters:
  ```python
  'pattern_metricgan_plus': {
      'skip': False,
      'use_pattern_detection': True,
      'pattern_confidence_threshold': 0.8,
      'pattern_suppression_factor': 0.15,  # Keep 15%, suppress 85%
      'pattern_padding_ms': 50,
      'use_metricgan': True,
      'apply_loudness_normalization': True,
      'target_loudness_multiplier': 1.6,  # 160%
      'passes': 1
  }
  ```

#### 1.2 Factory Pattern Integration
- Create `PatternMetricGANProcessor` class extending base enhancement pattern
- Register in enhancement factory for dynamic instantiation
- Follow Strategy pattern used by existing processors

#### 1.3 BaseProcessor Integration
- Extend `BaseProcessor._apply_noise_reduction_with_metadata()` to support new enhancement level
- Ensure compatibility with existing audio preprocessing pipeline
- Maintain backward compatibility with current enhancement levels

### 2. Implementation Components

#### 2.1 Pattern Detection Engine (`PatternDetectionEngine`)
```python
class PatternDetectionEngine:
    """Ultra-conservative interruption pattern detection"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
    
    def detect_interruption_patterns(
        self, 
        audio: np.ndarray, 
        sr: int
    ) -> List[InterruptionPattern]:
        """Detect interruption patterns with ultra-conservative approach"""
        # Implementation based on existing pattern detection logic
        pass
    
    def _merge_interruptions(
        self, 
        interruptions: List[dict], 
        sr: int, 
        min_gap: float = 0.2
    ) -> List[InterruptionPattern]:
        """Merge nearby interruptions"""
        pass
```

#### 2.2 Pattern Suppression Engine (`PatternSuppressionEngine`)
```python
class PatternSuppressionEngine:
    """Apply gentle pattern suppression with padding"""
    
    def __init__(self, padding_ms: int = 50, suppression_factor: float = 0.15):
        self.padding_ms = padding_ms
        self.suppression_factor = suppression_factor
    
    def apply_pattern_removal(
        self, 
        audio: np.ndarray, 
        patterns: List[InterruptionPattern], 
        sr: int
    ) -> np.ndarray:
        """Apply ultra-conservative pattern removal"""
        # Implementation based on existing suppression logic
        pass
```

#### 2.3 MetricGAN+ Integration (`MetricGANProcessor`)
```python
class MetricGANProcessor:
    """MetricGAN+ enhancement processor"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._load_metricgan()
    
    def enhance_with_metricgan(
        self, 
        audio: np.ndarray, 
        sr: int = 16000
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhance audio using MetricGAN+ with proper normalization"""
        # Implementation based on existing MetricGAN integration
        pass
    
    def _load_metricgan(self):
        """Load MetricGAN+ model from SpeechBrain"""
        pass
```

#### 2.4 Loudness Enhancement Integration
- Extend existing `LoudnessNormalizer` to support 160% enhancement mode
- Add `enhance_loudness_160_percent()` method
- Integrate with Pattern→MetricGAN+ pipeline

### 3. Core Enhancement Pipeline Integration

#### 3.1 AudioEnhancer Core Changes
Modify `processors/audio_enhancement/core.py`:

```python
# Add to ENHANCEMENT_LEVELS
'pattern_metricgan_plus': {
    'skip': False,
    'use_pattern_detection': True,
    'pattern_confidence_threshold': 0.8,
    'pattern_suppression_factor': 0.15,
    'pattern_padding_ms': 50,
    'use_metricgan': True,
    'apply_loudness_normalization': True,
    'target_loudness_multiplier': 1.6,
    'passes': 1
}

# Add to __init__ method
self.pattern_detector = None
self.pattern_suppressor = None  
self.metricgan_processor = None

# Add initialization in _init_engines()
def _init_pattern_metricgan_pipeline(self):
    """Initialize Pattern→MetricGAN+ pipeline components"""
    try:
        from .pattern_detection import PatternDetectionEngine
        from .pattern_suppression import PatternSuppressionEngine
        from .metricgan_processor import MetricGANProcessor
        
        self.pattern_detector = PatternDetectionEngine()
        self.pattern_suppressor = PatternSuppressionEngine()
        self.metricgan_processor = MetricGANProcessor(
            device='cuda' if self.use_gpu else 'cpu'
        )
        
        logger.info("Pattern→MetricGAN+ pipeline initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Pattern→MetricGAN+ pipeline: {e}")
```

#### 3.2 Enhancement Processing Logic
Add to `enhance()` method in core.py:

```python
# Handle pattern_metricgan_plus enhancement level
if noise_level == 'pattern_metricgan_plus':
    return self._process_pattern_metricgan_plus(
        audio, sample_rate, return_metadata
    )

def _process_pattern_metricgan_plus(
    self, 
    audio: np.ndarray, 
    sample_rate: int, 
    return_metadata: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """Process audio with Pattern→MetricGAN+ pipeline"""
    
    start_time = time.time()
    metadata = {
        'enhancement_level': 'pattern_metricgan_plus',
        'patterns_detected': 0,
        'metricgan_applied': False,
        'loudness_enhanced': False
    }
    
    try:
        # Step 1: Pattern Detection
        if self.pattern_detector:
            patterns = self.pattern_detector.detect_interruption_patterns(
                audio, sample_rate
            )
            metadata['patterns_detected'] = len(patterns)
            
            # Step 2: Pattern Suppression
            if patterns and self.pattern_suppressor:
                audio = self.pattern_suppressor.apply_pattern_removal(
                    audio, patterns, sample_rate
                )
                metadata['patterns_suppressed'] = len(patterns)
        
        # Step 3: MetricGAN+ Enhancement
        if self.metricgan_processor:
            enhanced_audio, metricgan_metadata = (
                self.metricgan_processor.enhance_with_metricgan(
                    audio, sample_rate
                )
            )
            audio = enhanced_audio
            metadata['metricgan_applied'] = True
            metadata.update(metricgan_metadata)
        
        # Step 4: 160% Loudness Enhancement
        original_audio = kwargs.get('original_audio', audio)
        if hasattr(self, 'loudness_normalizer'):
            audio = self.loudness_normalizer.enhance_loudness_160_percent(
                audio, original_audio, sample_rate
            )
            metadata['loudness_enhanced'] = True
        
        # Final metrics
        processing_time = time.time() - start_time
        metadata['processing_time'] = processing_time
        
        if return_metadata:
            return audio, metadata
        return audio
        
    except Exception as e:
        logger.error(f"Pattern→MetricGAN+ processing failed: {e}")
        metadata['error'] = str(e)
        if return_metadata:
            return audio, metadata
        return audio
```

### 4. Quality Metrics Integration

#### 4.1 Enhanced Metrics Collection
- Integrate with existing `utils.audio_metrics` module
- Add Pattern→MetricGAN+ specific metrics:
  - Pattern detection accuracy
  - Suppression effectiveness
  - MetricGAN+ quality improvement
  - Loudness enhancement metrics

#### 4.2 Quality Validation
```python
class PatternMetricGANQualityValidator:
    """Quality validation for Pattern→MetricGAN+ processing"""
    
    def validate_enhancement_quality(
        self, 
        original: np.ndarray,
        enhanced: np.ndarray, 
        sr: int
    ) -> Dict[str, Any]:
        """Validate enhancement quality"""
        
        metrics = {
            'pesq_score': self._calculate_pesq(original, enhanced, sr),
            'stoi_score': self._calculate_stoi(original, enhanced, sr),
            'si_sdr_improvement': self._calculate_si_sdr(original, enhanced),
            'loudness_preservation': self._check_loudness_preservation(
                original, enhanced, sr
            ),
            'pattern_suppression_effectiveness': (
                self._measure_pattern_suppression(original, enhanced, sr)
            )
        }
        
        return metrics
```

### 5. Testing Strategy (TDD Approach)

#### 5.1 Unit Tests
Create comprehensive test suite following project TDD patterns:

```python
# tests/test_pattern_metricgan_integration.py
class TestPatternMetricGANIntegration(unittest.TestCase):
    """Test Pattern→MetricGAN+ integration following TDD approach"""
    
    def test_pattern_detection_accuracy(self):
        """Test pattern detection with known interruption samples"""
        pass
    
    def test_pattern_suppression_effectiveness(self):
        """Test pattern suppression preserves primary speaker"""
        pass
    
    def test_metricgan_quality_improvement(self):
        """Test MetricGAN+ improves audio quality metrics"""
        pass
    
    def test_loudness_enhancement_160_percent(self):
        """Test 160% loudness enhancement maintains quality"""
        pass
    
    def test_end_to_end_pipeline(self):
        """Test complete Pattern→MetricGAN+ pipeline"""
        pass
    
    def test_integration_with_base_processor(self):
        """Test integration with BaseProcessor audio preprocessing"""
        pass
```

#### 5.2 Integration Tests
```python
# tests/test_pattern_metricgan_processor_integration.py
class TestPatternMetricGANProcessorIntegration(unittest.TestCase):
    """Test integration with main processor pipeline"""
    
    def test_processor_initialization(self):
        """Test processor initializes with Pattern→MetricGAN+ support"""
        pass
    
    def test_audio_enhancement_with_pattern_metricgan(self):
        """Test audio enhancement using pattern_metricgan_plus level"""
        pass
    
    def test_streaming_mode_compatibility(self):
        """Test compatibility with streaming processing mode"""
        pass
```

#### 5.3 Performance Tests
```python
# tests/test_pattern_metricgan_performance.py
class TestPatternMetricGANPerformance(unittest.TestCase):
    """Test performance characteristics"""
    
    def test_processing_speed_benchmarks(self):
        """Test processing speed meets requirements"""
        # Target: Process 50 samples in <10 minutes
        pass
    
    def test_memory_usage_efficiency(self):
        """Test memory usage stays within bounds"""
        pass
    
    def test_gpu_utilization_optimization(self):
        """Test GPU utilization efficiency"""
        pass
```

### 6. Configuration Integration

#### 6.1 Config Integration
Update `config.py` to support Pattern→MetricGAN+ configuration:

```python
# Pattern→MetricGAN+ Configuration
PATTERN_METRICGAN_CONFIG = {
    "enabled": True,
    "pattern_detection": {
        "confidence_threshold": 0.8,
        "energy_threshold_percentile": 75,
        "zcr_threshold_percentile": 80,
        "spectral_threshold_percentile": 70,
        "context_energy_multiplier": 2.0
    },
    "pattern_suppression": {
        "padding_ms": 50,
        "suppression_factor": 0.15,
        "min_gap_seconds": 0.2
    },
    "metricgan": {
        "model_source": "speechbrain/metricgan-plus-voicebank",
        "device": "auto",  # auto, cuda, cpu
        "batch_size": 1
    },
    "loudness_enhancement": {
        "target_multiplier": 1.6,
        "method": "rms",  # rms, peak, lufs
        "headroom_db": -1.0,
        "soft_limit": True
    }
}
```

#### 6.2 BaseProcessor Configuration Support
Update BaseProcessor to support Pattern→MetricGAN+ configuration:

```python
# In BaseProcessor.__init__()
self.pattern_metricgan_config = config.get("pattern_metricgan_config", {})
if self.pattern_metricgan_config.get("enabled", False):
    self.enhancement_level = "pattern_metricgan_plus"
```

### 7. Documentation Integration

#### 7.1 Code Documentation
- Add comprehensive docstrings following project patterns
- Include usage examples and parameter descriptions
- Document integration points and dependencies

#### 7.2 Technical Documentation
- Update main README with Pattern→MetricGAN+ enhancement level
- Add technical specification document
- Include performance benchmarks and quality metrics

### 8. Success Criteria

#### 8.1 Functional Requirements
- ✅ Pattern→MetricGAN+ enhancement level available in AudioEnhancer
- ✅ Successfully processes audio with >0.8 confidence pattern detection
- ✅ Preserves primary speaker quality (PESQ > 3.0, STOI > 0.85)
- ✅ Achieves 160% loudness enhancement without clipping
- ✅ Integration with BaseProcessor audio preprocessing
- ✅ Compatible with streaming and batch processing modes

#### 8.2 Performance Requirements
- ✅ Process 50 audio samples in <10 minutes on GPU
- ✅ Memory usage <8GB for typical processing batch
- ✅ CPU fallback available when GPU not accessible
- ✅ Processing speed >5 samples/minute on average hardware

#### 8.3 Quality Requirements
- ✅ Pattern detection accuracy >90% for known interruption types
- ✅ Primary speaker preservation score >95%
- ✅ No audio clipping or artifacts in enhanced output
- ✅ Consistent loudness levels across processed samples
- ✅ MetricGAN+ quality improvement measurable via PESQ/STOI

#### 8.4 Integration Requirements
- ✅ All existing tests continue to pass
- ✅ Backward compatibility with existing enhancement levels
- ✅ Configuration through standard config files
- ✅ Error handling and graceful degradation
- ✅ Comprehensive test coverage >90%

### 9. Implementation Plan

#### Phase 1: Core Components (Week 1)
1. Create `PatternDetectionEngine` with ultra-conservative detection
2. Implement `PatternSuppressionEngine` with gentle suppression
3. Develop `MetricGANProcessor` with SpeechBrain integration
4. Extend `LoudnessNormalizer` for 160% enhancement

#### Phase 2: Core Integration (Week 1)
1. Add `pattern_metricgan_plus` enhancement level to AudioEnhancer
2. Integrate processing pipeline in `core.py`
3. Update `BaseProcessor` for Pattern→MetricGAN+ support
4. Implement configuration integration

#### Phase 3: Testing & Validation (Week 2)
1. Develop comprehensive unit test suite
2. Create integration tests with existing processors
3. Performance benchmarking and optimization
4. Quality validation and metrics collection

#### Phase 4: Documentation & Polish (Week 2)
1. Complete code documentation and examples
2. Update technical documentation
3. Performance tuning and optimization
4. Final integration testing and validation

### 10. Dependencies

#### 10.1 Internal Dependencies
- `processors/audio_enhancement/core.py` - Core enhancement framework
- `processors/audio_enhancement/loudness_normalizer.py` - Loudness processing
- `processors/base_processor.py` - Base processing framework
- `utils/audio_metrics.py` - Quality metrics calculation

#### 10.2 External Dependencies
- `speechbrain` - MetricGAN+ model access
- `torch` - Deep learning framework
- `librosa` - Audio analysis and processing
- `scipy` - Signal processing utilities
- `numpy` - Numerical operations

#### 10.3 Hardware Requirements
- GPU recommended for MetricGAN+ processing
- Minimum 8GB RAM for batch processing
- CPU fallback support required

### 11. Risk Mitigation

#### 11.1 Technical Risks
- **MetricGAN+ model loading failures**: Implement robust error handling and CPU fallback
- **Memory limitations**: Add batch size optimization and memory monitoring
- **Pattern detection accuracy**: Use proven conservative thresholds with validation

#### 11.2 Integration Risks
- **Backward compatibility**: Comprehensive testing with existing enhancement levels
- **Performance regression**: Benchmark against current processing speeds
- **Configuration conflicts**: Clear separation of Pattern→MetricGAN+ specific config

### 12. Complexity Assessment
**Medium Complexity** - Involves integration of multiple proven components into existing framework. The individual components (pattern detection, MetricGAN+, loudness enhancement) are already validated. Main complexity is in proper integration following established patterns and ensuring performance optimization.

### 13. Estimated Duration
**2 weeks** - Given the medium complexity and established component base

### 14. Status
**📋 PLANNED** - Ready for implementation following TDD approach