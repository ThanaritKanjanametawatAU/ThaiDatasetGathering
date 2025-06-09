"""
Issue Categorization Engine for Audio Quality Assessment.

Automatically classifies detected audio problems into actionable categories
with confidence scores, severity ratings, and remediation suggestions.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


@dataclass
class IssueCategory:
    """Issue category representation."""
    name: str
    confidence: float
    parent: Optional[str] = None
    details: Dict = field(default_factory=dict)
    severity: Optional['SeverityScore'] = None
    context: Optional[Dict] = None
    priority_score: float = 0.0


@dataclass
class SeverityScore:
    """Severity score representation."""
    score: float
    level: str  # 'low', 'medium', 'high', 'critical'
    factors: Dict = field(default_factory=dict)


@dataclass
class Issue:
    """Issue representation."""
    name: str
    impact: float
    frequency: float


@dataclass
class RemediationSuggestion:
    """Remediation suggestion representation."""
    method: str
    effectiveness: float
    complexity: str  # 'low', 'medium', 'high'
    parameters: Dict = field(default_factory=dict)


@dataclass
class IssueReport:
    """Complete issue report."""
    categories: List[IssueCategory]
    summary: str
    recommendations: List[RemediationSuggestion]
    temporal_analysis: Optional[Dict] = None
    explanations: Optional[Dict] = None
    addressed_issues: Optional[List] = None
    remaining_issues: Optional[List] = None


class IssueCategorizer:
    """
    Intelligent issue categorization engine for audio quality problems.
    """
    
    def __init__(self, config=None, batch_mode=False, real_time_mode=False,
                 cache_enabled=True, custom_taxonomy=None):
        """
        Initialize the issue categorizer.
        
        Args:
            config: Configuration dict
            batch_mode: Enable batch processing optimizations
            real_time_mode: Enable real-time streaming mode
            cache_enabled: Enable result caching
            custom_taxonomy: Custom issue taxonomy to use
        """
        self.config = config or {}
        self.batch_mode = batch_mode
        self.real_time_mode = real_time_mode
        self.cache_enabled = cache_enabled
        
        # Initialize taxonomy
        self.taxonomy = self._initialize_taxonomy()
        if custom_taxonomy:
            self.taxonomy.update(custom_taxonomy)
        
        # Initialize categorization components
        self.ml_categorizer = MLCategorizer(self.config.get('ml_config', {}))
        self.rule_engine = RuleBasedCategorizer(self.taxonomy)
        
        # Cache for results
        self._cache = {} if cache_enabled else None
        
        # Real-time tracking
        if real_time_mode:
            self._audio_buffer = []
            self._category_history = []
    
    def _initialize_taxonomy(self) -> Dict:
        """Initialize the issue taxonomy."""
        return {
            'noise_issues': {
                'subcategories': {
                    'background_noise': {
                        'indicators': ['low_snr', 'high_noise_floor'],
                        'severity_weight': 0.7,
                        'thresholds': {'snr': 15, 'noise_floor_db': -40}
                    },
                    'electrical_interference': {
                        'indicators': ['harmonic_peaks', 'periodic_pattern'],
                        'severity_weight': 0.8,
                        'detection': {'fundamental_freqs': [50, 60], 'harmonic_tolerance_hz': 2}
                    }
                }
            },
            'technical_artifacts': {
                'subcategories': {
                    'clipping': {
                        'indicators': ['amplitude_saturation', 'harmonic_distortion'],
                        'severity_weight': 0.9,
                        'thresholds': {'clip_threshold': 0.99, 'clip_duration_ms': 1}
                    },
                    'codec_artifacts': {
                        'indicators': ['frequency_cutoff', 'quantization_noise'],
                        'severity_weight': 0.6
                    }
                }
            },
            'recording_problems': {
                'subcategories': {
                    'echo': {
                        'indicators': ['reflection_pattern', 'rt60_high'],
                        'severity_weight': 0.7
                    },
                    'room_acoustics': {
                        'indicators': ['modal_resonances', 'excessive_reverb'],
                        'severity_weight': 0.6
                    }
                }
            },
            'format_issues': {
                'subcategories': {
                    'sample_rate_problems': {
                        'indicators': ['aliasing', 'frequency_folding'],
                        'severity_weight': 0.8
                    },
                    'bit_depth_issues': {
                        'indicators': ['quantization_noise', 'dynamic_range_loss'],
                        'severity_weight': 0.5
                    }
                }
            },
            'speech_quality': {
                'subcategories': {
                    'insufficient_speech': {
                        'indicators': ['low_speech_duration', 'high_silence_ratio'],
                        'severity_weight': 0.8,
                        'thresholds': {'min_speech_duration': 2.0, 'max_silence_ratio': 0.7}
                    },
                    'multiple_speakers': {
                        'indicators': ['pitch_variance', 'energy_discontinuities'],
                        'severity_weight': 0.9
                    },
                    'excessive_silence': {
                        'indicators': ['high_silence_ratio'],
                        'severity_weight': 0.6,
                        'thresholds': {'max_silence_ratio': 0.7}
                    }
                }
            }
        }
    
    def categorize_issue(self, patterns: List[Dict], metrics: Dict) -> IssueReport:
        """
        Main categorization method.
        
        Args:
            patterns: Detected patterns
            metrics: Calculated metrics
            
        Returns:
            IssueReport with categorized issues
        """
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = self._generate_cache_key(patterns, metrics)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Perform categorization
        start_time = time.time()
        
        # Get categories from both ML and rules
        ml_categories = self.ml_categorizer.predict(metrics, patterns)
        rule_categories = self.rule_engine.apply_rules(metrics, patterns)
        
        # Merge and resolve conflicts
        final_categories = self._merge_categorizations(ml_categories, rule_categories)
        
        # Calculate severity and priority
        for category in final_categories:
            category.severity = self._calculate_severity(category, metrics)
            category.priority_score = self._calculate_priority(category)
        
        # Generate report
        report = IssueReport(
            categories=final_categories,
            summary=self._generate_summary(final_categories),
            recommendations=self._generate_recommendations(final_categories)
        )
        
        # Track processing time
        processing_time = time.time() - start_time
        logger.debug(f"Categorization completed in {processing_time*1000:.1f}ms")
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = report
        
        return report
    
    def categorize_comprehensive(self, patterns: List[Dict], metrics: Dict) -> IssueReport:
        """
        Comprehensive categorization using all available information.
        
        Args:
            patterns: Detected patterns
            metrics: Calculated metrics
            
        Returns:
            Comprehensive IssueReport
        """
        # Get base categorization
        report = self.categorize_issue(patterns, metrics)
        
        # Add temporal analysis if in real-time mode
        if self.real_time_mode and self._category_history:
            report.temporal_analysis = self._analyze_temporal_trends()
        
        # Add explanations
        report.explanations = self._generate_explanations(report.categories, metrics, patterns)
        
        return report
    
    def categorize_with_explanation(self, patterns: List[Dict], metrics: Dict) -> IssueReport:
        """
        Categorize with detailed explanations.
        
        Args:
            patterns: Detected patterns
            metrics: Calculated metrics
            
        Returns:
            IssueReport with explanations
        """
        report = self.categorize_comprehensive(patterns, metrics)
        return report
    
    def _merge_categorizations(self, ml_categories: List[IssueCategory], 
                             rule_categories: List[IssueCategory]) -> List[IssueCategory]:
        """Merge ML and rule-based categorizations."""
        merged = {}
        
        # Add all categories to merged dict
        for cat in ml_categories + rule_categories:
            if cat.name in merged:
                # Keep higher confidence
                if cat.confidence > merged[cat.name].confidence:
                    merged[cat.name] = cat
            else:
                merged[cat.name] = cat
        
        return list(merged.values())
    
    def _calculate_severity(self, category: IssueCategory, metrics: Dict) -> SeverityScore:
        """Calculate severity score for an issue."""
        # Support both IssueCategory and Issue objects
        if hasattr(category, 'impact') and hasattr(category, 'frequency'):
            # This is an Issue object from test
            impact = category.impact
            frequency = category.frequency
            severity_score = impact * frequency
        else:
            # This is an IssueCategory object
            # Get base severity from taxonomy
            base_severity = 0.5
            for parent, info in self.taxonomy.items():
                for subcat, subinfo in info.get('subcategories', {}).items():
                    if subcat == category.name:
                        base_severity = subinfo.get('severity_weight', 0.5)
                        break
            
            # Adjust based on metrics
            impact_factor = 1.0
            if category.name == 'background_noise' and 'snr' in metrics:
                # Lower SNR = higher impact
                impact_factor = max(0.5, min(1.5, 20.0 / (metrics['snr'] + 1)))
            elif category.name == 'clipping' and 'locations' in category.details:
                # More clipping locations = higher impact
                impact_factor = min(1.5, 1 + len(category.details['locations']) * 0.1)
            
            # Calculate final score
            severity_score = base_severity * impact_factor * category.confidence
        
        # Determine level
        if severity_score >= 0.8:
            level = 'critical'
        elif severity_score >= 0.6:
            level = 'high'
        elif severity_score >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return SeverityScore(
            score=severity_score,
            level=level,
            factors={
                'base_severity': base_severity if not hasattr(category, 'impact') else category.impact,
                'impact_factor': impact_factor if not hasattr(category, 'impact') else category.frequency,
                'confidence': category.confidence if hasattr(category, 'confidence') else 1.0
            }
        )
    
    def _calculate_priority(self, category: IssueCategory) -> float:
        """Calculate priority score for an issue."""
        if not category.severity:
            return 0.0
        
        # Priority = severity * (1 - remediation_ease) * confidence
        remediation_ease = self._estimate_remediation_ease(category.name)
        priority = category.severity.score * (1 - remediation_ease) * category.confidence
        
        return priority
    
    def _estimate_remediation_ease(self, category_name: str) -> float:
        """Estimate how easy it is to fix an issue."""
        ease_scores = {
            'background_noise': 0.7,  # Relatively easy with filters
            'electrical_interference': 0.8,  # Easy with notch filter
            'clipping': 0.3,  # Hard to fix properly
            'codec_artifacts': 0.2,  # Very hard to fix
            'insufficient_speech': 0.1,  # Cannot add speech
            'multiple_speakers': 0.2,  # Hard to separate
            'echo': 0.5,  # Moderate difficulty
            'excessive_silence': 0.9,  # Easy to trim
        }
        return ease_scores.get(category_name, 0.5)
    
    def prioritize_issues(self, categories: List[IssueCategory]) -> List[IssueCategory]:
        """Prioritize issues by severity and impact."""
        # Sort by priority score (descending)
        return sorted(categories, key=lambda c: c.priority_score, reverse=True)
    
    def generate_remediation_suggestions(self, category: IssueCategory) -> List[RemediationSuggestion]:
        """Generate remediation suggestions for an issue."""
        suggester = RemediationSuggestionGenerator()
        return suggester.generate_suggestions(category)
    
    def _generate_summary(self, categories: List[IssueCategory]) -> str:
        """Generate summary of detected issues."""
        if not categories:
            return "No significant audio quality issues detected."
        
        # Count issues by parent category
        parent_counts = defaultdict(int)
        for cat in categories:
            parent = cat.parent or 'other'
            parent_counts[parent] += 1
        
        # Build summary
        summary_parts = []
        total_issues = len(categories)
        
        if total_issues == 1:
            summary_parts.append(f"Detected 1 audio quality issue")
        else:
            summary_parts.append(f"Detected {total_issues} audio quality issues")
        
        # Add breakdown by category
        category_summaries = []
        for parent, count in sorted(parent_counts.items(), key=lambda x: x[1], reverse=True):
            category_summaries.append(f"{count} {parent.replace('_', ' ')}")
        
        if category_summaries:
            summary_parts.append(f"({', '.join(category_summaries)})")
        
        # Add severity information
        critical_count = sum(1 for cat in categories if cat.severity and cat.severity.level == 'critical')
        if critical_count > 0:
            summary_parts.append(f"with {critical_count} critical issue(s)")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_recommendations(self, categories: List[IssueCategory]) -> List[RemediationSuggestion]:
        """Generate overall recommendations."""
        all_suggestions = []
        
        # Get suggestions for each category
        for category in categories:
            suggestions = self.generate_remediation_suggestions(category)
            all_suggestions.extend(suggestions)
        
        # Remove duplicates and sort by effectiveness
        unique_suggestions = {}
        for sugg in all_suggestions:
            if sugg.method not in unique_suggestions or sugg.effectiveness > unique_suggestions[sugg.method].effectiveness:
                unique_suggestions[sugg.method] = sugg
        
        # Return top suggestions
        sorted_suggestions = sorted(unique_suggestions.values(), 
                                  key=lambda s: s.effectiveness, reverse=True)
        return sorted_suggestions[:5]  # Top 5 suggestions
    
    def _generate_explanations(self, categories: List[IssueCategory], 
                             metrics: Dict, patterns: List[Dict]) -> Dict:
        """Generate explanations for categorizations."""
        explanations = {}
        
        for category in categories:
            # Identify key factors
            key_factors = []
            
            # Check metrics
            if category.name == 'background_noise' and 'snr' in metrics:
                key_factors.append(f"SNR: {metrics['snr']:.1f} dB")
            elif category.name == 'clipping' and patterns:
                clip_count = sum(1 for p in patterns if p.get('type') == 'clipping')
                key_factors.append(f"Clipping events: {clip_count}")
            elif category.name == 'insufficient_speech' and 'speech_duration' in metrics:
                key_factors.append(f"Speech duration: {metrics['speech_duration']:.1f}s")
            
            # Generate explanation text
            if category.name == 'background_noise':
                explanation = f"Detected background noise due to low signal-to-noise ratio. {' '.join(key_factors)}."
            elif category.name == 'clipping':
                explanation = f"Audio clipping detected, indicating amplitude saturation. {' '.join(key_factors)}."
            elif category.name == 'electrical_interference':
                explanation = "Electrical interference detected, likely from power line hum or electronic devices."
            else:
                explanation = f"{category.name.replace('_', ' ').title()} detected with {category.confidence:.0%} confidence."
            
            explanations[category.name] = {
                'confidence': category.confidence,
                'key_factors': key_factors,
                'explanation': explanation
            }
        
        return explanations
    
    def _generate_cache_key(self, patterns: List[Dict], metrics: Dict) -> str:
        """Generate cache key from inputs."""
        # Create a stable hash of inputs
        pattern_str = json.dumps(sorted([str(p) for p in patterns]))
        metric_str = json.dumps(sorted(metrics.items()))
        return f"{hash(pattern_str)}_{hash(metric_str)}"
    
    def _resolve_conflicts(self, categories: List[IssueCategory]) -> List[IssueCategory]:
        """Resolve conflicting categories."""
        # Simple conflict resolution: keep higher confidence
        conflicts = {
            ('background_noise', 'silence'): 'background_noise',  # Noise takes precedence
            ('clipping', 'loudness'): 'clipping',  # Clipping is more specific
        }
        
        resolved = []
        skip = set()
        
        for i, cat1 in enumerate(categories):
            if i in skip:
                continue
                
            for j, cat2 in enumerate(categories[i+1:], i+1):
                if j in skip:
                    continue
                    
                conflict_key = (cat1.name, cat2.name)
                if conflict_key in conflicts:
                    # Keep the one specified in conflicts
                    if conflicts[conflict_key] == cat1.name:
                        skip.add(j)
                    else:
                        skip.add(i)
                        break
                elif conflict_key[::-1] in conflicts:
                    # Check reverse order
                    if conflicts[conflict_key[::-1]] == cat1.name:
                        skip.add(j)
                    else:
                        skip.add(i)
                        break
            
            if i not in skip:
                resolved.append(cat1)
        
        return resolved
    
    def process_chunk(self, audio_chunk: np.ndarray) -> IssueReport:
        """Process audio chunk for real-time categorization."""
        if not self.real_time_mode:
            raise ValueError("Real-time mode not enabled")
        
        # Add to buffer
        self._audio_buffer.append(audio_chunk)
        
        # Keep only recent chunks (e.g., last 5 seconds)
        if len(self._audio_buffer) > 5:
            self._audio_buffer.pop(0)
        
        # Analyze combined buffer
        combined_audio = np.concatenate(self._audio_buffer)
        
        # Quick analysis (simplified for real-time)
        quick_metrics = {
            'snr': self._quick_snr_estimate(combined_audio),
            'energy': np.sqrt(np.mean(combined_audio ** 2))
        }
        
        # Categorize
        report = self.categorize_issue([], quick_metrics)
        
        # Update history
        self._category_history.append(report.categories)
        if len(self._category_history) > 10:
            self._category_history.pop(0)
        
        return report
    
    def _quick_snr_estimate(self, audio: np.ndarray) -> float:
        """Quick SNR estimation for real-time processing."""
        # Simple energy-based estimation
        signal_power = np.percentile(audio ** 2, 90)
        noise_power = np.percentile(audio ** 2, 10)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 40.0
        
        return snr
    
    def _analyze_temporal_trends(self) -> Dict:
        """Analyze temporal trends in categories."""
        if not self._category_history:
            return {}
        
        # Count category occurrences
        category_counts = defaultdict(int)
        for categories in self._category_history:
            for cat in categories:
                category_counts[cat.name] += 1
        
        # Identify persistent issues
        total_frames = len(self._category_history)
        persistent_issues = [
            cat for cat, count in category_counts.items()
            if count / total_frames > 0.5  # Present in >50% of frames
        ]
        
        return {
            'persistent_issues': persistent_issues,
            'category_counts': dict(category_counts),
            'total_frames': total_frames
        }
    
    def categorize_batch(self, batch_data: List[Tuple[Dict, List]]) -> List[IssueReport]:
        """Batch categorization for multiple samples."""
        results = []
        
        for metrics, patterns in batch_data:
            report = self.categorize_issue(patterns, metrics)
            results.append(report)
        
        return results
    
    def categorize_from_quality_report(self, quality_report: float) -> IssueReport:
        """Categorize based on quality monitor report."""
        # Convert quality score to potential issues
        metrics = {
            'quality_score': quality_report,
            'naturalness': quality_report
        }
        
        patterns = []
        if quality_report < 0.5:
            patterns.append({'type': 'quality_degradation', 'severity': 1 - quality_report})
        
        return self.categorize_issue(patterns, metrics)
    
    def categorize_enhancement_results(self, enhancement_metrics: Dict) -> IssueReport:
        """Categorize enhancement results to identify addressed/remaining issues."""
        # Analyze what was improved
        snr_improvement = enhancement_metrics.get('final_snr', 0) - enhancement_metrics.get('initial_snr', 0)
        
        addressed = []
        remaining = []
        
        # Check if noise was addressed
        if snr_improvement > 5:
            addressed.append('background_noise')
        elif enhancement_metrics.get('final_snr', 0) < 20:
            remaining.append('background_noise')
        
        # Create report
        report = self.categorize_issue([], enhancement_metrics)
        report.addressed_issues = addressed
        report.remaining_issues = remaining
        
        return report
    
    def _compute_categorization(self, patterns: List[Dict], metrics: Dict) -> IssueReport:
        """Compute categorization (for testing cache)."""
        return self.categorize_issue(patterns, metrics)
    
    def model(self, model_type: str = 'default'):
        """Initialize with specific model type (for A/B testing)."""
        return IssueCategorizer(config={'model_type': model_type})


class MLCategorizer:
    """Machine learning based categorizer."""
    
    def __init__(self, config: Dict = None):
        """Initialize ML categorizer."""
        self.config = config or {}
        self.model = self._load_model()
        self.threshold = 0.5
        self.category_names = [
            'background_noise', 'electrical_interference', 'clipping',
            'codec_artifacts', 'insufficient_speech', 'multiple_speakers',
            'echo', 'room_acoustics'
        ]
    
    def _load_model(self):
        """Load or initialize ML model."""
        # For now, return a mock model
        # In production, this would load a trained model
        return MockMLModel()
    
    def predict(self, metrics: Dict, patterns: List[Dict]) -> List[IssueCategory]:
        """Predict issue categories using ML model."""
        # Extract features
        features = self._extract_features(metrics, patterns)
        
        # Get predictions
        try:
            probabilities = self.model.predict_proba(features)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return []
        
        # Convert to categories
        categories = []
        for i, prob in enumerate(probabilities[0]):
            if prob > self.threshold:
                category = IssueCategory(
                    name=self.category_names[i % len(self.category_names)],
                    confidence=float(prob),
                    parent=self._get_parent_category(self.category_names[i % len(self.category_names)])
                )
                categories.append(category)
        
        return categories
    
    def _extract_features(self, metrics: Dict, patterns: List[Dict]) -> np.ndarray:
        """Extract features for ML model."""
        features = []
        
        # Metric features
        features.append(metrics.get('snr', 0))
        features.append(metrics.get('stoi', 0))
        features.append(metrics.get('spectral_distortion', 0))
        features.append(metrics.get('energy_ratio', 0))
        features.append(metrics.get('silence_ratio', 0))
        features.append(metrics.get('speech_duration', 0))
        
        # Pattern features
        clip_count = sum(1 for p in patterns if p.get('type') == 'clipping')
        hum_detected = any(p.get('type') == 'hum' for p in patterns)
        
        features.append(clip_count)
        features.append(float(hum_detected))
        
        # Pad to expected size
        while len(features) < 10:
            features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def _get_parent_category(self, category_name: str) -> str:
        """Get parent category for a subcategory."""
        parent_map = {
            'background_noise': 'noise_issues',
            'electrical_interference': 'noise_issues',
            'clipping': 'technical_artifacts',
            'codec_artifacts': 'technical_artifacts',
            'insufficient_speech': 'speech_quality',
            'multiple_speakers': 'speech_quality',
            'echo': 'recording_problems',
            'room_acoustics': 'recording_problems'
        }
        return parent_map.get(category_name, 'other')
    
    def optimize_thresholds(self, X_val: np.ndarray, y_val: np.ndarray) -> List[float]:
        """Optimize thresholds for each category."""
        # Simple threshold optimization
        # In production, this would use proper optimization
        optimal_thresholds = []
        
        for i in range(y_val.shape[1]):
            # Try different thresholds
            best_threshold = 0.5
            best_score = 0
            
            for threshold in np.arange(0.1, 0.9, 0.1):
                # Calculate F1 score for this threshold
                predictions = (self.model.predict_proba(X_val)[:, i] > threshold).astype(int)
                
                # Simple F1 calculation
                tp = np.sum((predictions == 1) & (y_val[:, i] == 1))
                fp = np.sum((predictions == 1) & (y_val[:, i] == 0))
                fn = np.sum((predictions == 0) & (y_val[:, i] == 1))
                
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                if f1 > best_score:
                    best_score = f1
                    best_threshold = threshold
            
            optimal_thresholds.append(best_threshold)
        
        return optimal_thresholds
    
    def calibrate_probabilities(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate prediction probabilities."""
        # Simple isotonic calibration simulation
        # In production, use sklearn.calibration
        calibrated = np.copy(raw_probs)
        
        # Ensure monotonicity
        for i in range(1, len(calibrated)):
            if calibrated[i] < calibrated[i-1]:
                calibrated[i] = calibrated[i-1]
        
        return calibrated


class RuleBasedCategorizer:
    """Rule-based categorizer using expert knowledge."""
    
    def __init__(self, taxonomy: Dict):
        """Initialize rule-based categorizer."""
        self.taxonomy = taxonomy
        self.rules = self._initialize_rules()
        self.custom_rules = []
    
    def _initialize_rules(self) -> List[Dict]:
        """Initialize categorization rules."""
        rules = [
            # Noise rules
            {
                'name': 'low_snr_noise',
                'condition': lambda m, p: m.get('snr', 100) < 15,
                'category': 'background_noise',
                'confidence': lambda m: max(0.7, 1 - m.get('snr', 0) / 30),
                'priority': 5
            },
            # Clipping rules
            {
                'name': 'clipping_detection',
                'condition': lambda m, p: any(pat.get('type') == 'clipping' for pat in p),
                'category': 'clipping',
                'confidence': lambda m: 0.9,
                'priority': 8
            },
            # Electrical interference
            {
                'name': 'hum_detection',
                'condition': lambda m, p: any(pat.get('type') == 'hum' for pat in p),
                'category': 'electrical_interference',
                'confidence': lambda m: 0.85,
                'priority': 7
            },
            # Speech quality rules
            {
                'name': 'insufficient_speech',
                'condition': lambda m, p: m.get('speech_duration', 100) < 2.0,
                'category': 'insufficient_speech',
                'confidence': lambda m: 0.8,
                'priority': 6
            },
            {
                'name': 'excessive_silence',
                'condition': lambda m, p: m.get('silence_ratio', 0) > 0.7,
                'category': 'excessive_silence',
                'confidence': lambda m: 0.75,
                'priority': 4
            },
            # Multiple speakers
            {
                'name': 'multiple_speakers',
                'condition': lambda m, p: (m.get('pitch_cv', 0) > 0.5 or 
                                         m.get('energy_discontinuities', 0) > 10),
                'category': 'multiple_speakers',
                'confidence': lambda m: min(0.9, m.get('pitch_cv', 0) + 0.4),
                'priority': 9
            }
        ]
        
        return sorted(rules, key=lambda r: r['priority'], reverse=True)
    
    def apply_rules(self, metrics: Dict, patterns: List[Dict]) -> List[IssueCategory]:
        """Apply rules to detect issues."""
        categories = []
        
        # Apply built-in rules
        for rule in self.rules:
            try:
                if rule['condition'](metrics, patterns):
                    category = IssueCategory(
                        name=rule['category'],
                        confidence=rule['confidence'](metrics),
                        parent=self._get_parent_category(rule['category']),
                        details=self._extract_details(rule['category'], metrics, patterns)
                    )
                    categories.append(category)
            except Exception as e:
                logger.warning(f"Rule {rule['name']} failed: {e}")
        
        # Apply custom rules
        for rule in self.custom_rules:
            try:
                if rule['condition'](metrics, patterns):
                    category = IssueCategory(
                        name=rule['category'],
                        confidence=rule['confidence'],
                        parent=self._get_parent_category(rule['category'])
                    )
                    categories.append(category)
            except Exception as e:
                logger.warning(f"Custom rule {rule['name']} failed: {e}")
        
        return categories
    
    def add_rule(self, name: str, condition: callable, category: str, 
                 confidence: float, priority: int = 5):
        """Add a custom rule."""
        self.custom_rules.append({
            'name': name,
            'condition': condition,
            'category': category,
            'confidence': confidence,
            'priority': priority
        })
        
        # Re-sort custom rules by priority
        self.custom_rules.sort(key=lambda r: r['priority'], reverse=True)
    
    def _get_parent_category(self, category_name: str) -> str:
        """Get parent category from taxonomy."""
        for parent, info in self.taxonomy.items():
            subcategories = info.get('subcategories', {})
            if category_name in subcategories:
                return parent
        
        # Fallback mapping for categories not in taxonomy
        fallback_map = {
            'background_noise': 'noise_issues',
            'electrical_interference': 'noise_issues',
            'clipping': 'technical_artifacts',
            'codec_artifacts': 'technical_artifacts',
            'insufficient_speech': 'speech_quality',
            'multiple_speakers': 'speech_quality',
            'excessive_silence': 'speech_quality',
            'echo': 'recording_problems',
            'room_acoustics': 'recording_problems',
            'silence': 'recording_problems'
        }
        return fallback_map.get(category_name, 'other')
    
    def _extract_details(self, category: str, metrics: Dict, patterns: List[Dict]) -> Dict:
        """Extract relevant details for the category."""
        details = {}
        
        if category == 'clipping':
            # Extract clipping locations
            clip_patterns = [p for p in patterns if p.get('type') == 'clipping']
            if clip_patterns:
                details['locations'] = []
                for pattern in clip_patterns:
                    if 'locations' in pattern:
                        details['locations'].extend(pattern['locations'])
                    elif 'location' in pattern:
                        details['locations'].append(pattern['location'])
        
        elif category == 'electrical_interference':
            # Extract frequency information
            hum_patterns = [p for p in patterns if p.get('type') == 'hum']
            if hum_patterns:
                details['frequency'] = hum_patterns[0].get('frequency', 50)
                details['harmonics'] = hum_patterns[0].get('harmonics', [])
        
        return details


class RemediationSuggestionGenerator:
    """Generate remediation suggestions for audio issues."""
    
    def __init__(self):
        """Initialize suggestion generator."""
        self.suggestion_database = self._initialize_suggestions()
    
    def _initialize_suggestions(self) -> Dict:
        """Initialize suggestion database."""
        return {
            'background_noise': [
                RemediationSuggestion(
                    method="Spectral Subtraction",
                    effectiveness=0.8,
                    complexity="low",
                    parameters={"alpha": 2.0, "beta": 0.1}
                ),
                RemediationSuggestion(
                    method="Wiener Filtering",
                    effectiveness=0.9,
                    complexity="medium",
                    parameters={"frame_size_ms": 25, "overlap": 0.5}
                ),
                RemediationSuggestion(
                    method="Adaptive Filtering",
                    effectiveness=0.85,
                    complexity="high",
                    parameters={"filter_order": 32, "adaptation_rate": 0.01}
                )
            ],
            'electrical_interference': [
                RemediationSuggestion(
                    method="Notch Filter",
                    effectiveness=0.95,
                    complexity="low",
                    parameters={"frequency": 50, "q_factor": 30}
                ),
                RemediationSuggestion(
                    method="Comb Filter",
                    effectiveness=0.9,
                    complexity="medium",
                    parameters={"fundamental": 50, "harmonics": 5}
                )
            ],
            'clipping': [
                RemediationSuggestion(
                    method="Cubic Spline Interpolation",
                    effectiveness=0.7,
                    complexity="low",
                    parameters={"window_size": 5}
                ),
                RemediationSuggestion(
                    method="AR Model Extrapolation",
                    effectiveness=0.85,
                    complexity="high",
                    parameters={"model_order": 20, "context_size": 100}
                ),
                RemediationSuggestion(
                    method="Soft Limiting",
                    effectiveness=0.6,
                    complexity="low",
                    parameters={"threshold_db": -3, "ratio": 4}
                )
            ],
            'insufficient_speech': [
                RemediationSuggestion(
                    method="Cannot Add Missing Speech",
                    effectiveness=0.0,
                    complexity="high",
                    parameters={}
                ),
                RemediationSuggestion(
                    method="Silence Trimming",
                    effectiveness=0.3,
                    complexity="low",
                    parameters={"threshold_db": -40, "min_silence_ms": 100}
                )
            ],
            'excessive_silence': [
                RemediationSuggestion(
                    method="Automatic Silence Removal",
                    effectiveness=0.95,
                    complexity="low",
                    parameters={"threshold_db": -40, "min_speech_ms": 250}
                ),
                RemediationSuggestion(
                    method="Dynamic Range Compression",
                    effectiveness=0.7,
                    complexity="medium",
                    parameters={"threshold_db": -20, "ratio": 3, "attack_ms": 5}
                )
            ]
        }
    
    def generate_suggestions(self, issue: IssueCategory) -> List[RemediationSuggestion]:
        """Generate suggestions for an issue."""
        suggestions = self.suggestion_database.get(issue.name, [])
        
        # Adjust for context if available
        if issue.context:
            suggestions = self._adjust_for_context(suggestions, issue.context)
        
        # Sort by effectiveness
        suggestions = sorted(suggestions, key=lambda s: s.effectiveness, reverse=True)
        
        return suggestions
    
    def _adjust_for_context(self, suggestions: List[RemediationSuggestion], 
                          context: Dict) -> List[RemediationSuggestion]:
        """Adjust suggestions based on context."""
        adjusted = []
        
        for suggestion in suggestions:
            # Copy suggestion
            adj_suggestion = RemediationSuggestion(
                method=suggestion.method,
                effectiveness=suggestion.effectiveness,
                complexity=suggestion.complexity,
                parameters=suggestion.parameters.copy()
            )
            
            # Adjust based on context
            if 'noise_type' in context:
                if context['noise_type'] == 'stationary' and 'wiener' in suggestion.method.lower():
                    adj_suggestion.effectiveness *= 1.1  # Wiener works well for stationary
                elif context['noise_type'] == 'non-stationary' and 'adaptive' in suggestion.method.lower():
                    adj_suggestion.effectiveness *= 1.1  # Adaptive works well for non-stationary
            
            # Adjust parameters
            if 'frequency' in context and 'frequency' in adj_suggestion.parameters:
                adj_suggestion.parameters['frequency'] = context['frequency']
            
            adjusted.append(adj_suggestion)
        
        return adjusted


class MockMLModel:
    """Mock ML model for testing."""
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Mock prediction."""
        # Generate pseudo-random probabilities based on features
        np.random.seed(int(np.sum(features) * 1000) % 2**32)
        
        # 8 categories
        probs = np.random.rand(features.shape[0], 8)
        
        # Make some correlations with features
        if features[0, 0] < 15:  # Low SNR
            probs[:, 0] += 0.3  # Background noise more likely
        
        if features[0, 6] > 0:  # Clipping count
            probs[:, 2] += 0.4  # Clipping more likely
        
        # Normalize
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs


class CategorizationABTest:
    """A/B testing framework for categorization."""
    
    def __init__(self, control_model, test_model):
        """Initialize A/B test."""
        self.control = control_model
        self.test = test_model
        self.results_tracker = ABTestResults()
    
    def categorize_with_ab_test(self, data: Dict, user_id: str) -> IssueReport:
        """Categorize with A/B test assignment."""
        # Determine test group based on user ID
        in_test_group = hash(user_id) % 100 < 10  # 10% in test
        
        if in_test_group:
            result = self.test.categorize_issue([], data)
            self.results_tracker.record('test', result)
        else:
            result = self.control.categorize_issue([], data)
            self.results_tracker.record('control', result)
        
        return result


class ABTestResults:
    """Track A/B test results."""
    
    def __init__(self):
        self.results = {'control': [], 'test': []}
    
    def record(self, group: str, result: IssueReport):
        """Record a result."""
        self.results[group].append(result)
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            'control_count': len(self.results['control']),
            'test_count': len(self.results['test'])
        }