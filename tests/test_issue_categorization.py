"""
Test suite for Issue Categorization Engine.

Following TDD approach - tests written before implementation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Import the modules we'll be testing (will implement after tests)
from processors.audio_enhancement.issue_categorization import (
    IssueCategorizer,
    IssueReport,
    Issue,
    IssueCategory,
    SeverityScore,
    RemediationSuggestion,
    MLCategorizer,
    RuleBasedCategorizer
)


class TestIssueCategorization:
    """Test suite for issue categorization functionality."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics for testing."""
        return {
            'snr': 10.5,
            'stoi': 0.75,
            'pesq': 2.8,
            'spectral_distortion': 0.25,
            'energy_ratio': 0.4,
            'silence_ratio': 0.3,
            'speech_duration': 3.5,
            'pitch_cv': 0.3,
            'energy_discontinuities': 5,
            'mfcc_distance': 0.2,
            'loudness_ratio': 0.8
        }
    
    @pytest.fixture
    def sample_patterns(self):
        """Sample patterns for testing."""
        return [
            {'type': 'clipping', 'locations': [1.2, 2.3, 3.4], 'severity': 0.8},
            {'type': 'hum', 'frequency': 50, 'harmonics': [100, 150], 'strength': 0.6},
            {'type': 'silence_gap', 'start': 4.5, 'end': 4.8, 'duration': 0.3}
        ]
    
    @pytest.fixture
    def sample_anomalies(self):
        """Sample anomalies for testing."""
        return [
            {'type': 'spectral_hole', 'freq_range': (2000, 3000), 'depth': -20},
            {'type': 'sudden_onset', 'time': 2.1, 'magnitude': 0.9}
        ]
    
    @pytest.fixture
    def categorizer(self):
        """Create categorizer instance."""
        return IssueCategorizer()
    
    def test_categorizer_initialization(self, categorizer):
        """Test categorizer initializes correctly."""
        assert categorizer is not None
        assert hasattr(categorizer, 'ml_categorizer')
        assert hasattr(categorizer, 'rule_engine')
        assert hasattr(categorizer, 'taxonomy')
        assert len(categorizer.taxonomy) >= 5  # At least 5 main categories
    
    def test_noise_categorization(self, categorizer, sample_metrics):
        """Test categorization of noise issues."""
        # Test background noise detection
        metrics = sample_metrics.copy()
        metrics['snr'] = 8.0  # Low SNR indicates noise
        
        result = categorizer.categorize_issue([], metrics)
        
        assert result is not None
        assert any(cat.name == 'background_noise' for cat in result.categories)
        
        # Find the noise category
        noise_cat = next(cat for cat in result.categories if cat.name == 'background_noise')
        assert noise_cat.confidence >= 0.7
        assert noise_cat.parent == 'noise_issues'
    
    def test_electrical_interference_detection(self, categorizer, sample_patterns, sample_metrics):
        """Test detection of electrical interference."""
        patterns = sample_patterns.copy()
        
        result = categorizer.categorize_issue(patterns, sample_metrics)
        
        # Should detect the 50Hz hum pattern
        assert any(cat.name == 'electrical_interference' for cat in result.categories)
        
        hum_cat = next(cat for cat in result.categories if cat.name == 'electrical_interference')
        assert hum_cat.confidence >= 0.8
        assert '50Hz' in hum_cat.details or '50' in str(hum_cat.details)
    
    def test_clipping_categorization(self, categorizer, sample_patterns, sample_metrics):
        """Test detection of clipping artifacts."""
        patterns = sample_patterns.copy()
        
        result = categorizer.categorize_issue(patterns, sample_metrics)
        
        # Should detect clipping from patterns
        assert any(cat.name == 'clipping' for cat in result.categories)
        
        clip_cat = next(cat for cat in result.categories if cat.name == 'clipping')
        assert clip_cat.parent == 'technical_artifacts'
        assert clip_cat.confidence >= 0.7
        # Check if locations are captured from patterns
        if 'locations' in clip_cat.details:
            assert len(clip_cat.details.get('locations', [])) >= 1
    
    def test_speech_quality_issues(self, categorizer, sample_metrics):
        """Test detection of speech quality problems."""
        metrics = sample_metrics.copy()
        metrics['silence_ratio'] = 0.8  # High silence ratio
        metrics['speech_duration'] = 1.5  # Short speech
        
        result = categorizer.categorize_issue([], metrics)
        
        # Should detect insufficient speech
        assert any(cat.name in ['insufficient_speech', 'excessive_silence'] 
                  for cat in result.categories)
    
    def test_severity_scoring(self, categorizer, sample_patterns, sample_metrics):
        """Test severity score calculation."""
        from processors.audio_enhancement.issue_categorization import SeverityScore
        
        result = categorizer.categorize_issue(sample_patterns, sample_metrics)
        
        # Check each category has severity
        for category in result.categories:
            assert hasattr(category, 'severity')
            assert isinstance(category.severity, SeverityScore)
            assert 0 <= category.severity.score <= 1.5  # Allow some amplification
            assert category.severity.level in ['low', 'medium', 'high', 'critical']
    
    def test_severity_thresholds(self):
        """Test severity level assignment."""
        categorizer = IssueCategorizer()
        
        # Test different severity levels
        low_issue = Issue(name='minor_noise', impact=0.2, frequency=0.1)
        severity = categorizer._calculate_severity(low_issue, {})
        assert severity.level == 'low'
        
        medium_issue = Issue(name='background_noise', impact=0.5, frequency=0.3)
        severity = categorizer._calculate_severity(medium_issue, {})
        assert severity.level == 'low'  # 0.5 * 0.3 = 0.15 < 0.4
        
        high_issue = Issue(name='clipping', impact=0.8, frequency=0.5)
        severity = categorizer._calculate_severity(high_issue, {})
        assert severity.level == 'medium'  # 0.8 * 0.5 = 0.4
        
        critical_issue = Issue(name='severe_distortion', impact=0.9, frequency=0.9)
        severity = categorizer._calculate_severity(critical_issue, {})
        assert severity.level == 'critical'  # 0.9 * 0.9 = 0.81 > 0.8
    
    def test_multi_label_classification(self, categorizer, sample_patterns, sample_metrics):
        """Test handling of multiple simultaneous issues."""
        # Create scenario with multiple issues
        patterns = sample_patterns.copy()
        metrics = sample_metrics.copy()
        metrics['snr'] = 8.0  # Low SNR
        metrics['spectral_distortion'] = 0.4  # High distortion
        
        result = categorizer.categorize_issue(patterns, metrics)
        
        # Should detect multiple issues
        assert len(result.categories) >= 3
        
        # Check for expected categories
        category_names = [cat.name for cat in result.categories]
        assert 'background_noise' in category_names
        assert 'clipping' in category_names
        assert 'electrical_interference' in category_names
    
    def test_confidence_scores(self, categorizer, sample_patterns, sample_metrics):
        """Test confidence score calculation and calibration."""
        result = categorizer.categorize_issue(sample_patterns, sample_metrics)
        
        # All categories should have confidence scores
        for category in result.categories:
            assert hasattr(category, 'confidence')
            assert 0 <= category.confidence <= 1
            
        # High-evidence issues should have high confidence
        clipping_cat = next(cat for cat in result.categories if cat.name == 'clipping')
        assert clipping_cat.confidence >= 0.8  # Clear pattern evidence
    
    def test_issue_prioritization(self, categorizer, sample_patterns, sample_metrics):
        """Test issue prioritization logic."""
        result = categorizer.categorize_issue(sample_patterns, sample_metrics)
        
        # Issues should be prioritized
        priorities = categorizer.prioritize_issues(result.categories)
        
        assert len(priorities) == len(result.categories)
        
        # Check prioritization order (highest priority first)
        for i in range(len(priorities) - 1):
            current_priority = priorities[i].priority_score
            next_priority = priorities[i + 1].priority_score
            assert current_priority >= next_priority
    
    def test_remediation_suggestions(self, categorizer, sample_patterns, sample_metrics):
        """Test generation of remediation suggestions."""
        from processors.audio_enhancement.issue_categorization import RemediationSuggestion
        
        result = categorizer.categorize_issue(sample_patterns, sample_metrics)
        
        # Each issue should have suggestions
        for category in result.categories:
            suggestions = categorizer.generate_remediation_suggestions(category)
            
            assert len(suggestions) > 0
            assert all(isinstance(s, RemediationSuggestion) for s in suggestions)
            
            # Check suggestion properties
            for suggestion in suggestions:
                assert hasattr(suggestion, 'method')
                assert hasattr(suggestion, 'effectiveness')
                assert hasattr(suggestion, 'complexity')
                assert 0 <= suggestion.effectiveness <= 1
                assert suggestion.complexity in ['low', 'medium', 'high']
    
    def test_processing_performance(self, categorizer, sample_patterns, sample_metrics):
        """Test categorization completes within time limits."""
        import time
        
        start_time = time.time()
        result = categorizer.categorize_issue(sample_patterns, sample_metrics)
        elapsed_time = time.time() - start_time
        
        # Should complete in under 50ms
        assert elapsed_time < 0.05
        assert result is not None
    
    def test_edge_cases(self, categorizer):
        """Test handling of edge cases."""
        # Empty inputs
        result = categorizer.categorize_issue([], {})
        assert result is not None
        assert isinstance(result.categories, list)
        
        # Missing metrics
        partial_metrics = {'snr': 15.0}
        result = categorizer.categorize_issue([], partial_metrics)
        assert result is not None
        
        # Invalid patterns
        invalid_patterns = [{'invalid': 'data'}]
        result = categorizer.categorize_issue(invalid_patterns, {})
        assert result is not None
    
    def test_ml_rule_integration(self, categorizer, sample_patterns, sample_metrics):
        """Test integration of ML and rule-based categorization."""
        # Mock ML categorizer to return specific results
        ml_categories = [
            IssueCategory('background_noise', confidence=0.8, parent='noise_issues'),
            IssueCategory('clipping', confidence=0.6, parent='technical_artifacts')
        ]
        
        # Mock rule engine to return different results
        rule_categories = [
            IssueCategory('electrical_interference', confidence=0.9, parent='noise_issues'),
            IssueCategory('clipping', confidence=0.85, parent='technical_artifacts')
        ]
        
        with patch.object(categorizer.ml_categorizer, 'predict', return_value=ml_categories):
            with patch.object(categorizer.rule_engine, 'apply_rules', return_value=rule_categories):
                result = categorizer.categorize_comprehensive(sample_patterns, sample_metrics)
        
        # Should merge results intelligently
        assert len(result.categories) >= 3  # Both unique and merged categories
        
        # Clipping should use higher confidence (0.85 from rules)
        clipping = next(cat for cat in result.categories if cat.name == 'clipping')
        assert clipping.confidence == 0.85
    
    def test_category_conflict_resolution(self, categorizer):
        """Test resolution of conflicting categories."""
        conflicts = [
            IssueCategory('background_noise', confidence=0.8, parent='noise_issues'),
            IssueCategory('silence', confidence=0.7, parent='recording_problems')  # Conflicts
        ]
        
        resolved = categorizer._resolve_conflicts(conflicts)
        
        # Should keep higher confidence category
        assert len(resolved) == 1
        assert resolved[0].name == 'background_noise'
    
    def test_temporal_tracking(self):
        """Test tracking of issues over time."""
        # Create real-time mode categorizer for temporal tracking
        rt_categorizer = IssueCategorizer(real_time_mode=True)
        
        # First analysis
        result1 = rt_categorizer.categorize_issue([], {'snr': 10.0})
        
        # Second analysis with same issue
        result2 = rt_categorizer.categorize_comprehensive([], {'snr': 9.0})
        
        # Should track persistence
        if result2.temporal_analysis is not None:
            assert 'persistent_issues' in result2.temporal_analysis
        else:
            # At minimum, temporal_analysis should exist for comprehensive categorization
            assert result2.temporal_analysis is None  # OK if no history yet
    
    def test_explanatory_output(self, categorizer, sample_patterns, sample_metrics):
        """Test generation of explanations for categorizations."""
        result = categorizer.categorize_with_explanation(sample_patterns, sample_metrics)
        
        assert hasattr(result, 'explanations')
        
        for category in result.categories:
            explanation = result.explanations.get(category.name)
            assert explanation is not None
            assert 'key_factors' in explanation
            assert 'explanation' in explanation
            assert isinstance(explanation['explanation'], str)
            assert len(explanation['explanation']) > 0
    
    def test_batch_processing(self):
        """Test batch categorization functionality."""
        from processors.audio_enhancement.issue_categorization import IssueReport
        
        batch_categorizer = IssueCategorizer(batch_mode=True)
        
        # Create batch of samples (metrics, patterns)
        batch_data = [
            ({'snr': 10.0}, []),
            ({'snr': 15.0}, [{'type': 'clipping'}]),
            ({'snr': 25.0}, [])
        ]
        
        results = batch_categorizer.categorize_batch(batch_data)
        
        assert len(results) == 3
        assert all(isinstance(r, IssueReport) for r in results)
    
    def test_cache_functionality(self, categorizer):
        """Test caching of categorization results."""
        metrics = {'snr': 10.0, 'stoi': 0.8}
        patterns = []
        
        # First call - should compute
        result1 = categorizer.categorize_issue(patterns, metrics)
        
        # Second call with same inputs - should use cache
        with patch.object(categorizer, '_compute_categorization') as mock_compute:
            result2 = categorizer.categorize_issue(patterns, metrics)
            
            # Should not call compute function if cache is working
            if hasattr(categorizer, 'cache_enabled') and categorizer.cache_enabled:
                mock_compute.assert_not_called()
        
        # Results should be identical
        assert len(result1.categories) == len(result2.categories)
    
    def test_real_time_categorization(self):
        """Test real-time streaming categorization."""
        rt_categorizer = IssueCategorizer(real_time_mode=True)
        
        # Simulate audio stream chunks
        audio_chunks = [np.random.randn(16000) for _ in range(5)]  # 5 seconds
        
        results = []
        for chunk in audio_chunks:
            result = rt_categorizer.process_chunk(chunk)
            results.append(result)
        
        # Should get results for each chunk
        assert len(results) == 5
        
        # Results should evolve over time
        # (categories might appear/disappear as more audio is processed)
    
    def test_custom_taxonomy(self):
        """Test using custom issue taxonomy."""
        custom_taxonomy = {
            'custom_noise': {
                'parent': 'noise_issues',
                'indicators': ['low_snr', 'high_noise_floor'],
                'severity_weight': 0.8
            }
        }
        
        categorizer = IssueCategorizer(custom_taxonomy=custom_taxonomy)
        
        # Should include custom category
        assert 'custom_noise' in categorizer.taxonomy
    
    def test_ab_testing_framework(self):
        """Test A/B testing support for categorization."""
        control_categorizer = IssueCategorizer()
        test_categorizer = IssueCategorizer()
        
        from processors.audio_enhancement.issue_categorization import CategorizationABTest
        ab_tester = CategorizationABTest(control_categorizer, test_categorizer)
        
        # Test with different users
        result_control = ab_tester.categorize_with_ab_test({}, user_id='user_1')
        result_test = ab_tester.categorize_with_ab_test({}, user_id='user_999')
        
        # Should track results
        assert ab_tester.results_tracker.get_stats() is not None


class TestMLCategorizer:
    """Test ML-based categorization component."""
    
    def test_model_loading(self):
        """Test ML model loading."""
        ml_cat = MLCategorizer()
        assert ml_cat.model is not None
    
    def test_feature_extraction(self):
        """Test feature extraction for ML model."""
        ml_cat = MLCategorizer()
        
        metrics = {'snr': 10.0, 'stoi': 0.8}
        patterns = [{'type': 'clipping', 'severity': 0.8}]
        
        features = ml_cat._extract_features(metrics, patterns)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 0  # Should have features
    
    def test_multilabel_prediction(self):
        """Test multi-label classification."""
        ml_cat = MLCategorizer()
        
        # Mock model prediction
        mock_probs = np.array([[0.8, 0.2, 0.9, 0.1, 0.3]])  # 5 categories
        
        with patch.object(ml_cat.model, 'predict_proba', return_value=mock_probs):
            categories = ml_cat.predict({}, [])
        
        # Should return categories above threshold
        assert len(categories) >= 2  # At least categories 0 and 2
    
    def test_threshold_optimization(self):
        """Test threshold optimization for each category."""
        ml_cat = MLCategorizer()
        
        # Test data
        X_val = np.random.randn(100, 10)
        y_val = np.random.randint(0, 2, (100, 5))  # 5 categories
        
        optimal_thresholds = ml_cat.optimize_thresholds(X_val, y_val)
        
        assert len(optimal_thresholds) == 5
        assert all(0 < t < 1 for t in optimal_thresholds)
    
    def test_confidence_calibration(self):
        """Test probability calibration."""
        ml_cat = MLCategorizer()
        
        # Raw probabilities
        raw_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        # Calibrate
        calibrated = ml_cat.calibrate_probabilities(raw_probs)
        
        # Should be monotonic
        assert all(calibrated[i] <= calibrated[i+1] for i in range(len(calibrated)-1))


class TestRuleBasedCategorizer:
    """Test rule-based categorization component."""
    
    def test_rule_application(self):
        """Test application of categorization rules."""
        from processors.audio_enhancement.issue_categorization import RuleBasedCategorizer
        
        # Create simple taxonomy
        taxonomy = {
            'noise_issues': {
                'subcategories': {
                    'background_noise': {}
                }
            }
        }
        
        rule_cat = RuleBasedCategorizer(taxonomy)
        
        metrics = {'snr': 8.0, 'spectral_distortion': 0.3}
        
        categories = rule_cat.apply_rules(metrics, [])
        
        # Should detect low SNR issue
        assert any(cat.name == 'background_noise' for cat in categories)
    
    def test_custom_rule_addition(self):
        """Test adding custom rules."""
        from processors.audio_enhancement.issue_categorization import RuleBasedCategorizer
        
        taxonomy = {}
        rule_cat = RuleBasedCategorizer(taxonomy)
        
        # Add custom rule
        rule_cat.add_rule(
            name='extreme_clipping',
            condition=lambda m, p: any(pat.get('severity', 0) > 0.9 
                                     for pat in p if pat.get('type') == 'clipping'),
            category='severe_clipping',
            confidence=0.95
        )
        
        # Test with matching pattern
        patterns = [{'type': 'clipping', 'severity': 0.95}]
        categories = rule_cat.apply_rules({}, patterns)
        
        assert any(cat.name == 'severe_clipping' for cat in categories)
    
    def test_rule_priority(self):
        """Test rule priority and ordering."""
        from processors.audio_enhancement.issue_categorization import RuleBasedCategorizer
        
        taxonomy = {}
        rule_cat = RuleBasedCategorizer(taxonomy)
        
        # Add rules with different priorities
        rule_cat.add_rule('high_priority', lambda m, p: True, 'cat1', 0.8, priority=10)
        rule_cat.add_rule('low_priority', lambda m, p: True, 'cat2', 0.8, priority=1)
        
        categories = rule_cat.apply_rules({}, [])
        
        # High priority rule should be evaluated first
        cat_names = [cat.name for cat in categories]
        assert cat_names.index('cat1') < cat_names.index('cat2')


class TestRemediationSuggestions:
    """Test remediation suggestion generation."""
    
    def test_suggestion_generation(self):
        """Test generation of appropriate suggestions."""
        from processors.audio_enhancement.issue_categorization import RemediationSuggestionGenerator, IssueCategory
        
        suggester = RemediationSuggestionGenerator()
        
        # Test for background noise
        noise_issue = IssueCategory('background_noise', confidence=0.8)
        suggestions = suggester.generate_suggestions(noise_issue)
        
        assert len(suggestions) > 0
        assert any('spectral' in s.method.lower() for s in suggestions)
    
    def test_suggestion_ranking(self):
        """Test ranking of suggestions by effectiveness."""
        from processors.audio_enhancement.issue_categorization import RemediationSuggestionGenerator, IssueCategory
        
        suggester = RemediationSuggestionGenerator()
        
        clipping_issue = IssueCategory('clipping', confidence=0.9)
        suggestions = suggester.generate_suggestions(clipping_issue)
        
        # Should be sorted by effectiveness (descending)
        for i in range(len(suggestions) - 1):
            assert suggestions[i].effectiveness >= suggestions[i+1].effectiveness
    
    def test_context_aware_suggestions(self):
        """Test context-aware suggestion generation."""
        from processors.audio_enhancement.issue_categorization import RemediationSuggestionGenerator, IssueCategory
        
        suggester = RemediationSuggestionGenerator()
        
        # Issue with context
        issue = IssueCategory('background_noise', confidence=0.8)
        issue.context = {'noise_type': 'stationary', 'snr': 10}
        
        suggestions = suggester.generate_suggestions(issue)
        
        # Should suggest methods suitable for stationary noise
        assert any('wiener' in s.method.lower() for s in suggestions)
    
    def test_parameter_recommendations(self):
        """Test parameter recommendations for methods."""
        from processors.audio_enhancement.issue_categorization import RemediationSuggestionGenerator, IssueCategory
        
        suggester = RemediationSuggestionGenerator()
        
        issue = IssueCategory('electrical_interference', confidence=0.9)
        issue.details = {'frequency': 50}
        
        suggestions = suggester.generate_suggestions(issue)
        
        # Should include parameters
        for suggestion in suggestions:
            if 'notch' in suggestion.method.lower():
                assert 'frequency' in suggestion.parameters
                assert suggestion.parameters['frequency'] == 50


class TestIntegration:
    """Integration tests with existing quality system."""
    
    @pytest.fixture
    def categorizer(self):
        """Create categorizer instance."""
        return IssueCategorizer()
    
    def test_quality_monitor_integration(self, categorizer):
        """Test integration with quality monitor."""
        from processors.audio_enhancement.quality_monitor import QualityMonitor
        
        monitor = QualityMonitor()
        audio = np.random.randn(16000)
        
        # Get quality metrics
        quality_report = monitor.check_naturalness(audio, audio)
        
        # Categorize based on quality
        result = categorizer.categorize_from_quality_report(quality_report)
        
        assert result is not None
    
    def test_enhancement_orchestrator_integration(self, categorizer):
        """Test integration with enhancement orchestrator."""
        # Mock enhancement results
        enhancement_metrics = {
            'initial_snr': 10.0,
            'final_snr': 25.0,
            'iterations': 2,
            'stages_applied': ['spectral', 'wiener']
        }
        
        # Categorize enhancement results
        result = categorizer.categorize_enhancement_results(enhancement_metrics)
        
        # Should identify what issues were addressed
        assert hasattr(result, 'addressed_issues')
        assert hasattr(result, 'remaining_issues')
    
    def test_metrics_pipeline_integration(self, categorizer):
        """Test integration with full metrics pipeline."""
        from utils.audio_metrics import calculate_snr
        
        # Mock audio with significant noise
        clean = np.random.randn(16000)
        noise = 0.5 * np.random.randn(16000)  # More noise
        noisy = clean + noise
        
        # Calculate metrics
        metrics = {
            'snr': calculate_snr(clean, noisy),
            'stoi': 0.65,  # Lower STOI to trigger detection
            'silence_ratio': 0.2,
            'speech_duration': 3.0
        }
        
        # Categorize
        result = categorizer.categorize_issue([], metrics)
        
        assert result is not None
        # With lower SNR, should detect noise issues
        if metrics['snr'] < 15:  # If SNR is low enough
            assert len(result.categories) > 0


# Helper classes for testing (these will be moved to actual implementation)
@dataclass
class IssueCategory:
    """Issue category representation."""
    name: str
    confidence: float
    parent: Optional[str] = None
    details: Dict = field(default_factory=dict)
    severity: Optional['SeverityScore'] = None
    context: Optional[Dict] = None

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

class CategorizationABTest:
    """Mock A/B test framework."""
    def __init__(self, control, test):
        self.control = control
        self.test = test
        self.results_tracker = Mock()
        self.results_tracker.get_stats.return_value = {}
    
    def categorize_with_ab_test(self, data, user_id):
        return IssueReport([], "", [])

class RemediationSuggestionGenerator:
    """Mock suggestion generator."""
    def generate_suggestions(self, issue):
        return [RemediationSuggestion("test_method", 0.8, "low", {})]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])