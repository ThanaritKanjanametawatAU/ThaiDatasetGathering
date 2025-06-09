# Task: Build Issue Categorization Engine

## Task ID
S01_T06

## Description
Develop an intelligent issue categorization engine that automatically classifies detected audio problems into actionable categories. This engine will analyze patterns, metrics, and anomalies to provide structured categorization with confidence scores and severity ratings.

## Status
**Status**: 🔴 Not Started  
**Assigned To**: Unassigned  
**Created**: 2025-06-09  
**Updated**: 2025-06-09

## Technical Requirements

### Categorization System
1. **Issue Categories**
   - Noise Issues (background, hum, hiss)
   - Technical Artifacts (clipping, distortion, compression)
   - Recording Problems (echo, reverb, room acoustics)
   - Format Issues (codec artifacts, sample rate problems)
   - Speech Quality (mumbling, secondary speakers, interruptions)

2. **Categorization Engine**
   ```python
   class IssueCategorizer:
       def categorize_issue(self, patterns: List[Pattern], metrics: Dict) -> IssueReport
       def calculate_severity(self, issue: Issue) -> SeverityScore
       def prioritize_issues(self, issues: List[Issue]) -> List[Issue]
       def generate_remediation_suggestions(self, issue: Issue) -> List[Suggestion]
   ```

3. **Multi-label Classification**
   - Hierarchical categorization
   - Confidence scoring per category
   - Issue co-occurrence analysis
   - Temporal issue tracking

4. **Severity Assessment**
   - Impact on intelligibility
   - Frequency of occurrence
   - Remediation difficulty
   - Cumulative effect analysis

### Implementation Steps
1. Define comprehensive issue taxonomy
2. Create rule-based categorization baseline
3. Implement ML-based classifier
4. Develop severity scoring algorithm
5. Build confidence calculation system
6. Create issue correlation analyzer
7. Implement suggestion generator
8. Add explanatory reasoning system

## Test Requirements (TDD)

### Test First Approach
1. **Categorization Accuracy Tests**
   ```python
   def test_noise_categorization():
       # Test with known noise types
       # Verify correct category assignment
       # Test confidence thresholds
   
   def test_artifact_categorization():
       # Test technical artifact detection
       # Verify sub-category accuracy
       # Test edge cases
   ```

2. **Severity Assessment Tests**
   ```python
   def test_severity_scoring():
       # Test with issues of known severity
       # Verify score consistency
       # Test cumulative effects
   
   def test_severity_thresholds():
       # Test critical vs minor classification
       # Verify threshold calibration
       # Test borderline cases
   ```

3. **Multi-label Tests**
   ```python
   def test_multiple_issues():
       # Test with audio containing multiple problems
       # Verify all issues detected
       # Test interaction effects
   
   def test_issue_prioritization():
       # Test priority ordering logic
       # Verify critical issues first
       # Test tie-breaking rules
   ```

4. **Suggestion Tests**
   ```python
   def test_remediation_suggestions():
       # Test suggestion relevance
       # Verify actionability
       # Test suggestion ranking
   ```

## Acceptance Criteria
- [ ] Categorizes issues into 15+ distinct categories
- [ ] Achieves >85% categorization accuracy
- [ ] Provides confidence scores for all categorizations
- [ ] Generates severity scores with clear rationale
- [ ] Produces actionable remediation suggestions
- [ ] Handles multi-label scenarios correctly
- [ ] Processes categorization in <50ms per sample
- [ ] Includes comprehensive issue catalog documentation

## Dependencies
- S01_T03 (SNR metrics)
- S01_T04 (Spectral analysis)
- S01_T05 (Pattern detection)
- scikit-learn for classification
- pandas for data structuring

## Estimated Effort
**Duration**: 1-2 days  
**Complexity**: Medium-High

## Detailed Algorithm Specifications

### Issue Categorization Pipeline
```
1. Feature Aggregation:
   a. Collect metrics from SNR calculator
   b. Gather spectral anomalies
   c. Compile detected patterns
   d. Merge into feature vector

2. Multi-label Classification:
   a. Apply sigmoid activation for each category
   b. Threshold probabilities (default: 0.5)
   c. Handle label dependencies
   d. Resolve conflicts

3. Severity Calculation:
   severity = Σ(w_i * impact_i * confidence_i)
   where:
   - w_i = category weight
   - impact_i = issue impact score
   - confidence_i = detection confidence

4. Hierarchical Categorization:
   Level 1: Noise | Technical | Recording | Format | Speech
   Level 2: Specific sub-categories
   Level 3: Detailed issue types
```

### Issue Taxonomy Algorithm
```
1. Noise Issues:
   a. Background Noise:
      - Stationary: SNR < threshold
      - Non-stationary: time-varying SNR
      - Colored noise: spectral shape analysis
   b. Electrical Interference:
      - 50/60Hz hum: spectral peaks
      - GSM buzz: periodic bursts
      - Ground loops: harmonic series

2. Technical Artifacts:
   a. Clipping Detection:
      - Hard clipping: |x| == 1.0
      - Soft clipping: high-order harmonics
      - Frequency: clipping_rate = clips/second
   b. Compression Artifacts:
      - Pumping: envelope modulation
      - Breathing: noise floor variation
      - Musical noise: spectral artifacts

3. Recording Problems:
   a. Room Acoustics:
      - RT60 estimation
      - Early reflections
      - Modal resonances
   b. Microphone Issues:
      - Proximity effect
      - Wind noise
      - Handling noise
```

### Confidence Score Calculation
```
1. Base Confidence:
   C_base = detector_confidence * feature_quality

2. Contextual Adjustment:
   C_context = C_base * context_weight
   where context_weight depends on:
   - Temporal consistency
   - Spectral coherence
   - Cross-validation with other detectors

3. Final Confidence:
   C_final = min(C_context * calibration_factor, 1.0)
```

### Mathematical Formulations
- **Issue Impact**: I = severity * prevalence * remediation_difficulty
- **Category Probability**: P(c|x) = σ(W^T x + b) for multi-label
- **Severity Score**: S = log(1 + Σ(issue_weights * issue_counts))
- **Confidence Calibration**: C_cal = isotonic_regression(C_raw)
- **Priority Score**: P = severity * (1 - remediation_ease) * confidence

## Integration with Existing Codebase

### Files to Interface With
1. **processors/audio_enhancement/quality_validator.py**
   - Use quality validation results
   - Share threshold definitions
   - Coordinate validation logic

2. **processors/audio_enhancement/quality_monitor.py**
   - Report categorized issues
   - Update monitoring dashboard
   - Track issue trends

3. **utils/audio_metrics.py**
   - Access computed metrics
   - Use metric history
   - Share metric definitions

4. **processors/audio_enhancement/enhancement_orchestrator.py**
   - Provide categorization for decisions
   - Suggest enhancement strategies
   - Priority-based processing

### Integration Architecture
```python
# Integration with existing quality system
from processors.audio_enhancement.quality_monitor import QualityMonitor
from utils.audio_metrics import AudioMetrics

class IntegratedIssueCategorizer:
    def __init__(self, config=None):
        self.quality_monitor = QualityMonitor()
        self.metrics_analyzer = AudioMetrics()
        self.ml_categorizer = MLCategorizer()
        self.rule_engine = RuleBasedCategorizer()
        
    def categorize_comprehensive(self, audio, sr, analysis_results):
        # Gather all inputs
        quality_report = self.quality_monitor.analyze(audio, sr)
        metrics = self.metrics_analyzer.compute_all(audio, sr)
        patterns = analysis_results.get('patterns', [])
        anomalies = analysis_results.get('anomalies', [])
        
        # Multi-method categorization
        ml_categories = self.ml_categorizer.predict(
            metrics, patterns, anomalies
        )
        rule_categories = self.rule_engine.apply_rules(
            quality_report, metrics
        )
        
        # Merge and resolve conflicts
        final_categories = self._merge_categorizations(
            ml_categories, rule_categories
        )
        
        # Calculate severity and priority
        for category in final_categories:
            category.severity = self._calculate_severity(category, metrics)
            category.priority = self._calculate_priority(category)
            
        return IssueReport(
            categories=final_categories,
            summary=self._generate_summary(final_categories),
            recommendations=self._generate_recommendations(final_categories)
        )
```

## Configuration Examples

### Issue Categorization Configuration (categorization_config.yaml)
```yaml
issue_categorizer:
  taxonomy:
    noise_issues:
      subcategories:
        - name: "background_noise"
          indicators:
            - low_snr: true
            - spectral_floor_high: true
          severity_weight: 0.7
          thresholds:
            snr_threshold: 15
            noise_floor_db: -40
            
        - name: "electrical_interference"
          indicators:
            - harmonic_peaks: true
            - periodic_pattern: true
          severity_weight: 0.8
          detection:
            fundamental_freqs: [50, 60]
            harmonic_tolerance_hz: 2
            
    technical_artifacts:
      subcategories:
        - name: "clipping"
          indicators:
            - amplitude_saturation: true
            - harmonic_distortion: true
          severity_weight: 0.9
          thresholds:
            clip_threshold: 0.99
            clip_duration_ms: 1
            
        - name: "codec_artifacts"
          indicators:
            - frequency_cutoff: true
            - quantization_noise: true
          severity_weight: 0.6
          
  ml_classifier:
    model_type: "multilabel_ensemble"
    models:
      - type: "random_forest"
        n_estimators: 200
        max_features: "sqrt"
      - type: "gradient_boost"
        n_estimators: 100
        learning_rate: 0.05
      - type: "neural_network"
        layers: [128, 64, 32]
        activation: "relu"
        
    threshold_optimization:
      method: "f1_macro"
      cv_folds: 5
      
    calibration:
      method: "isotonic"
      cv_folds: 3
      
  severity_calculation:
    weights:
      intelligibility_impact: 0.4
      occurrence_frequency: 0.3
      remediation_difficulty: 0.3
      
    scaling:
      method: "logarithmic"
      base: 2
      
  remediation:
    suggestion_templates:
      noise_reduction:
        - "Apply spectral subtraction with α={alpha}"
        - "Use Wiener filtering with noise profile"
        - "Implement adaptive filtering"
        
      clipping_repair:
        - "Apply cubic spline interpolation"
        - "Use AR model prediction"
        - "Implement soft limiting at -3dB"
        
  performance:
    batch_processing: true
    cache_enabled: true
    parallel_categories: true
    gpu_inference: false
```

### Remediation Strategy Configuration (remediation_config.json)
```json
{
  "remediation_strategies": {
    "background_noise": {
      "methods": [
        {
          "name": "spectral_subtraction",
          "effectiveness": 0.8,
          "complexity": "low",
          "parameters": {
            "alpha": 2.0,
            "beta": 0.1,
            "noise_estimation": "minimum_statistics"
          }
        },
        {
          "name": "wiener_filter",
          "effectiveness": 0.9,
          "complexity": "medium",
          "parameters": {
            "frame_size_ms": 25,
            "overlap": 0.5
          }
        }
      ],
      "selection_criteria": {
        "snr_range": [-10, 15],
        "noise_type": ["stationary", "slowly_varying"]
      }
    },
    "clipping": {
      "methods": [
        {
          "name": "cubic_interpolation",
          "effectiveness": 0.7,
          "complexity": "low",
          "suitable_for": ["short_clips", "isolated_samples"]
        },
        {
          "name": "ar_extrapolation",
          "effectiveness": 0.85,
          "complexity": "high",
          "suitable_for": ["continuous_clipping", "speech"]
        }
      ]
    }
  },
  "strategy_selection": {
    "mode": "adaptive",
    "factors": ["severity", "audio_type", "processing_time"]
  }
}
```

## Error Handling Strategy

### Exception Hierarchy
```python
class CategorizationError(Exception):
    """Base exception for categorization errors"""
    pass

class InsufficientDataError(CategorizationError):
    """Not enough data for reliable categorization"""
    pass

class ModelInferenceError(CategorizationError):
    """ML model inference failed"""
    pass

class ConflictResolutionError(CategorizationError):
    """Failed to resolve category conflicts"""
    pass
```

### Robust Categorization Pipeline
```python
class RobustCategorizer:
    def categorize_with_fallback(self, audio_data, analysis):
        try:
            # Primary ML-based categorization
            return self._ml_categorization(audio_data, analysis)
        except ModelInferenceError:
            logger.warning("ML categorization failed, using rules")
            return self._rule_based_categorization(audio_data, analysis)
        except InsufficientDataError:
            logger.warning("Insufficient data, using minimal categorization")
            return self._minimal_categorization(audio_data)
        except Exception as e:
            logger.error(f"Categorization failed: {e}")
            return self._safe_default_categories()
```

### Conflict Resolution
```python
class CategoryConflictResolver:
    def resolve_conflicts(self, categories):
        """Resolve conflicting category assignments"""
        
        # Build conflict graph
        conflicts = self._find_conflicts(categories)
        
        # Apply resolution strategies
        for conflict in conflicts:
            if conflict.type == 'mutual_exclusion':
                # Keep higher confidence category
                categories = self._resolve_by_confidence(conflict, categories)
            elif conflict.type == 'subsumption':
                # Keep more specific category
                categories = self._resolve_by_specificity(conflict, categories)
            elif conflict.type == 'correlation':
                # Merge correlated categories
                categories = self._merge_correlated(conflict, categories)
                
        return categories
```

## Performance Optimization

### Batch Categorization
```python
class BatchCategorizer:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.feature_extractor = BatchFeatureExtractor()
        self.model = load_model('categorizer_model.pkl')
        
    def categorize_batch(self, audio_files):
        """Efficient batch processing"""
        results = []
        
        for i in range(0, len(audio_files), self.batch_size):
            batch = audio_files[i:i + self.batch_size]
            
            # Parallel feature extraction
            features = self.feature_extractor.extract_parallel(batch)
            
            # Batch inference
            predictions = self.model.predict_batch(features)
            
            # Post-process results
            for j, pred in enumerate(predictions):
                results.append(self._process_prediction(pred, batch[j]))
                
        return results
```

### Caching Strategy
```python
class CategoryCache:
    def __init__(self, ttl=3600):
        self.cache = TTLCache(maxsize=1000, ttl=ttl)
        self.feature_hasher = FeatureHasher()
        
    def get_or_compute(self, audio_features, compute_func):
        # Create cache key from features
        cache_key = self.feature_hasher.hash(audio_features)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Compute and cache
        result = compute_func(audio_features)
        self.cache[cache_key] = result
        return result
```

### Real-time Categorization
```python
class RealTimeCategorizer:
    def __init__(self, window_size=1.0, update_interval=0.1):
        self.window_size = window_size
        self.update_interval = update_interval
        self.category_tracker = CategoryTracker()
        
    def process_stream(self, audio_stream):
        """Real-time streaming categorization"""
        buffer = AudioBuffer(self.window_size)
        
        for chunk in audio_stream:
            buffer.add(chunk)
            
            if buffer.ready():
                # Quick categorization
                categories = self._quick_categorize(buffer.get())
                
                # Update tracking
                self.category_tracker.update(categories)
                
                # Emit results
                yield self.category_tracker.get_current_state()
```

## Production Considerations

### Monitoring Integration
```python
# Prometheus metrics
categorization_latency = Histogram(
    'categorization_latency_seconds',
    'Time to categorize issues',
    ['method', 'category_count']
)

category_distribution = Counter(
    'issue_categories_total',
    'Distribution of issue categories',
    ['category', 'severity']
)

remediation_suggestions = Counter(
    'remediation_suggestions_total',
    'Suggested remediation methods',
    ['issue_type', 'method']
)
```

### A/B Testing Framework
```python
class CategorizationABTest:
    def __init__(self, control_model, test_model):
        self.control = control_model
        self.test = test_model
        self.results_tracker = ABTestResults()
        
    def categorize_with_ab_test(self, audio_data, user_id):
        # Determine test group
        in_test_group = hash(user_id) % 100 < 10  # 10% in test
        
        if in_test_group:
            result = self.test.categorize(audio_data)
            self.results_tracker.record('test', result)
        else:
            result = self.control.categorize(audio_data)
            self.results_tracker.record('control', result)
            
        return result
```

### Explanatory AI
```python
class ExplainableCategorizer:
    def categorize_with_explanation(self, audio_data, analysis):
        # Get predictions
        categories = self.model.predict(audio_data, analysis)
        
        # Generate explanations
        explanations = {}
        for category in categories:
            # SHAP values for feature importance
            shap_values = self.explainer.explain(
                audio_data, category
            )
            
            # Natural language explanation
            explanation = self._generate_explanation(
                category, shap_values, analysis
            )
            
            explanations[category.name] = {
                'confidence': category.confidence,
                'key_factors': shap_values.top_features(5),
                'explanation': explanation
            }
            
        return categories, explanations
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Poor Categorization Accuracy**
   - **Symptom**: Many miscategorized issues
   - **Cause**: Model not trained on similar data
   - **Solution**:
     ```python
     # Retrain with domain-specific data
     categorizer.retrain(
         new_training_data,
         fine_tune=True,
         epochs=10
     )
     ```

2. **Conflicting Categories**
   - **Symptom**: Multiple contradictory labels
   - **Cause**: Threshold too low
   - **Solution**:
     ```python
     # Adjust thresholds
     categorizer.set_thresholds({
         'noise': 0.7,      # Increase from 0.5
         'technical': 0.6,
         'recording': 0.6
     })
     ```

3. **Missing Obvious Issues**
   - **Symptom**: Known issues not categorized
   - **Cause**: Features not capturing issue
   - **Solution**:
     ```python
     # Add custom rules
     categorizer.add_rule(
         name="custom_hum_detection",
         condition=lambda m: m['peak_50hz'] > -20,
         category="electrical_hum",
         confidence=0.9
     )
     ```

4. **Slow Categorization**
   - **Symptom**: >100ms per sample
   - **Cause**: Complex model, no optimization
   - **Solution**:
     ```python
     # Enable optimizations
     categorizer = IssueCategorizer(
         model='lightweight',
         batch_mode=True,
         cache_enabled=True,
         gpu=True
     )
     ```

### Debugging Tools
```python
class CategorizationDebugger:
    def debug_categorization(self, audio_data, expected_categories):
        """Comprehensive categorization debugging"""
        
        # Run with detailed logging
        with self.detailed_logging():
            predicted = self.categorizer.categorize(audio_data)
            
        # Analyze discrepancies
        analysis = {
            'predicted': predicted,
            'expected': expected_categories,
            'feature_analysis': self._analyze_features(audio_data),
            'rule_trace': self._trace_rules(audio_data),
            'model_internals': self._inspect_model(audio_data),
            'confidence_distribution': self._analyze_confidence(predicted)
        }
        
        # Generate report
        self.generate_debug_report(analysis)
        return analysis
```

### Visualization Tools
```python
def visualize_categorization_results(audio_data, categorization_report):
    """Create comprehensive categorization visualization"""
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Category distribution
    ax1 = fig.add_subplot(gs[0, :2])
    plot_category_distribution(ax1, categorization_report.categories)
    
    # Severity heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    plot_severity_heatmap(ax2, categorization_report.severity_matrix)
    
    # Confidence scores
    ax3 = fig.add_subplot(gs[1, 0])
    plot_confidence_scores(ax3, categorization_report.confidences)
    
    # Feature importance
    ax4 = fig.add_subplot(gs[1, 1])
    plot_feature_importance(ax4, categorization_report.feature_importance)
    
    # Remediation suggestions
    ax5 = fig.add_subplot(gs[1, 2])
    plot_remediation_effectiveness(ax5, categorization_report.suggestions)
    
    # Timeline view
    ax6 = fig.add_subplot(gs[2, :])
    plot_issue_timeline(ax6, categorization_report.temporal_analysis)
    
    plt.tight_layout()
    return fig
```

## Notes
- Design for easy addition of new categories
- Consider integration with existing quality validators
- Implement explanatory AI for transparency
- Plan for multilingual issue descriptions
- Consider real-time categorization needs

## References
- [Audio Quality Assessment](https://www.aes.org/e-lib/browse.cfm?elib=18974)
- [Multi-label Classification](https://scikit-learn.org/stable/modules/multiclass.html)
- Existing quality monitoring in quality_monitor.py