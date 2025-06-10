"""
Test module for Enhancement Scoring System
Tests unified scoring, ranking, and assessment functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from processors.audio_enhancement.scoring.enhancement_scorer import (
    EnhancementScorer,
    ScoringProfile,
    EnhancementScore,
    ScoreCategory,
    ScoreExplainer
)


class TestEnhancementScorer:
    """Test Enhancement Scorer functionality"""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing"""
        return {
            'pesq': 3.5,
            'stoi': 0.85,
            'si_sdr': 15.2,
            'snr': 25.0,
            'naturalness': 0.8
        }
    
    @pytest.fixture
    def batch_metrics(self):
        """Create batch of metrics for ranking tests"""
        np.random.seed(42)
        batch = []
        for i in range(100):
            batch.append({
                'id': f'sample_{i}',
                'metrics': {
                    'pesq': np.random.uniform(1.0, 4.5),
                    'stoi': np.random.uniform(0.3, 1.0),
                    'si_sdr': np.random.uniform(-5.0, 25.0),
                    'snr': np.random.uniform(0.0, 40.0),
                    'naturalness': np.random.uniform(0.2, 1.0)
                }
            })
        return batch
    
    @pytest.fixture
    def scorer(self):
        """Create an EnhancementScorer instance"""
        return EnhancementScorer(profile='balanced')
    
    def test_initialization(self):
        """Test scorer initialization with different profiles"""
        # Test default initialization
        scorer = EnhancementScorer()
        assert scorer.profile == 'balanced'
        assert scorer.weights is not None
        assert len(scorer.score_history) == 0
        
        # Test with different profiles
        profiles = ['intelligibility', 'quality', 'balanced', 'naturalness']
        for profile in profiles:
            scorer = EnhancementScorer(profile=profile)
            assert scorer.profile == profile
            assert scorer.weights is not None
            
        # Test custom profile
        custom_weights = {
            'pesq': 0.3,
            'stoi': 0.3,
            'si_sdr': 0.2,
            'snr': 0.1,
            'naturalness': 0.1
        }
        scorer = EnhancementScorer(profile='custom', custom_weights=custom_weights)
        assert scorer.profile == 'custom'
        assert scorer.weights == custom_weights
    
    def test_basic_score_calculation(self, scorer, sample_metrics):
        """Test basic score calculation"""
        score = scorer.calculate_score(sample_metrics)
        
        # Verify score structure
        assert isinstance(score, EnhancementScore)
        assert 0 <= score.value <= 100
        assert score.grade in ['A', 'B', 'C', 'D', 'F']
        assert score.metrics == sample_metrics
        assert score.timestamp is not None
        
        # Test with perfect metrics
        perfect_metrics = {
            'pesq': 4.5,
            'stoi': 1.0,
            'si_sdr': 30.0,
            'snr': 40.0,
            'naturalness': 1.0
        }
        perfect_score = scorer.calculate_score(perfect_metrics)
        assert perfect_score.value >= 95
        assert perfect_score.grade == 'A'
        
        # Test with poor metrics
        poor_metrics = {
            'pesq': 1.0,
            'stoi': 0.3,
            'si_sdr': -5.0,
            'snr': 0.0,
            'naturalness': 0.2
        }
        poor_score = scorer.calculate_score(poor_metrics)
        assert poor_score.value <= 30
        assert poor_score.grade in ['D', 'F']
    
    def test_weighted_scoring(self):
        """Test weighted scoring with different profiles"""
        metrics = {
            'pesq': 4.0,
            'stoi': 0.9,
            'si_sdr': 20.0,
            'snr': 30.0,
            'naturalness': 0.85
        }
        
        # Test intelligibility profile (should favor STOI)
        intel_scorer = EnhancementScorer(profile='intelligibility')
        intel_score = intel_scorer.calculate_score(metrics)
        
        # Test quality profile (should favor PESQ)
        quality_scorer = EnhancementScorer(profile='quality')
        quality_score = quality_scorer.calculate_score(metrics)
        
        # Test naturalness profile
        natural_scorer = EnhancementScorer(profile='naturalness')
        natural_score = natural_scorer.calculate_score(metrics)
        
        # Scores should be different based on profile
        assert intel_score.value != quality_score.value
        assert quality_score.value != natural_score.value
        
        # Test with metrics favoring intelligibility
        high_intel_metrics = metrics.copy()
        high_intel_metrics['stoi'] = 0.95
        high_intel_metrics['pesq'] = 2.5
        
        intel_score_high = intel_scorer.calculate_score(high_intel_metrics)
        quality_score_high = quality_scorer.calculate_score(high_intel_metrics)
        
        # Intelligibility scorer should rate this higher
        assert intel_score_high.value > quality_score_high.value
    
    def test_score_normalization(self, scorer):
        """Test score normalization across different metric ranges"""
        # Test with out-of-range metrics
        extreme_metrics = {
            'pesq': 5.0,  # Above normal range
            'stoi': 1.5,  # Above normal range
            'si_sdr': 50.0,  # Very high
            'snr': 60.0,  # Very high
            'naturalness': 1.2  # Above normal range
        }
        
        score = scorer.calculate_score(extreme_metrics)
        assert 0 <= score.value <= 100  # Should still be normalized
        
        # Test with negative values
        negative_metrics = {
            'pesq': 0.5,  # Below normal range
            'stoi': -0.1,  # Negative
            'si_sdr': -20.0,  # Very negative
            'snr': -10.0,  # Negative
            'naturalness': -0.2  # Negative
        }
        
        score = scorer.calculate_score(negative_metrics)
        assert 0 <= score.value <= 100  # Should still be normalized
        assert score.value < 20  # Should be very low
    
    def test_ranking_system(self, scorer, batch_metrics):
        """Test sample ranking functionality"""
        # Rank samples
        ranked_results = scorer.rank_samples(batch_metrics)
        
        # Verify ranking properties
        assert len(ranked_results) == len(batch_metrics)
        
        # Check that samples are properly ranked
        for i in range(len(ranked_results) - 1):
            assert ranked_results[i].score >= ranked_results[i + 1].score
        
        # Verify rank information
        for i, result in enumerate(ranked_results):
            assert result.rank == i + 1
            assert result.id in [sample['id'] for sample in batch_metrics]
            assert 0 <= result.score <= 100
    
    def test_percentile_calculation(self, scorer, batch_metrics):
        """Test percentile calculation"""
        # Calculate scores for batch
        for sample in batch_metrics:
            score = scorer.calculate_score(sample['metrics'])
            scorer.score_history.append(score.value)
        
        # Test percentile calculation
        test_score = 75.0
        percentile = scorer.get_percentile(test_score)
        assert 0 <= percentile <= 100
        
        # Test edge cases
        min_percentile = scorer.get_percentile(0.0)
        assert min_percentile == 0.0
        
        max_percentile = scorer.get_percentile(100.0)
        assert max_percentile == 100.0
        
        # Test median
        median_score = np.median(scorer.score_history)
        median_percentile = scorer.get_percentile(median_score)
        assert 45 <= median_percentile <= 55  # Should be around 50th percentile
    
    def test_custom_profiles(self):
        """Test custom scoring profiles"""
        # Create custom profile emphasizing PESQ
        custom_weights = {
            'pesq': 0.5,
            'stoi': 0.2,
            'si_sdr': 0.15,
            'snr': 0.1,
            'naturalness': 0.05
        }
        
        scorer = EnhancementScorer(profile='custom', custom_weights=custom_weights)
        
        # Test that weights sum to 1
        assert abs(sum(scorer.weights.values()) - 1.0) < 0.001
        
        # Test scoring with custom weights
        metrics = {
            'pesq': 4.0,
            'stoi': 0.7,
            'si_sdr': 10.0,
            'snr': 20.0,
            'naturalness': 0.6
        }
        
        score = scorer.calculate_score(metrics)
        assert isinstance(score, EnhancementScore)
        
        # Verify PESQ has most influence
        high_pesq_metrics = metrics.copy()
        high_pesq_metrics['pesq'] = 4.5
        high_pesq_score = scorer.calculate_score(high_pesq_metrics)
        
        low_pesq_metrics = metrics.copy()
        low_pesq_metrics['pesq'] = 2.0
        low_pesq_score = scorer.calculate_score(low_pesq_metrics)
        
        # PESQ change should have large impact
        assert (high_pesq_score.value - low_pesq_score.value) > 20
    
    def test_score_categories(self, scorer):
        """Test score categorization"""
        # Test category assignment with more accurate metric values
        categories = {
            'excellent': {'min': 90, 'metrics': {'pesq': 4.2, 'stoi': 0.95, 'si_sdr': 25.0, 'snr': 35.0, 'naturalness': 0.95}},
            'good': {'min': 70, 'metrics': {'pesq': 3.8, 'stoi': 0.88, 'si_sdr': 18.0, 'snr': 28.0, 'naturalness': 0.85}},
            'fair': {'min': 50, 'metrics': {'pesq': 2.8, 'stoi': 0.75, 'si_sdr': 10.0, 'snr': 20.0, 'naturalness': 0.7}},
            'poor': {'min': 30, 'metrics': {'pesq': 2.0, 'stoi': 0.65, 'si_sdr': 5.0, 'snr': 15.0, 'naturalness': 0.6}},
            'very_poor': {'min': 0, 'metrics': {'pesq': 1.5, 'stoi': 0.5, 'si_sdr': 0.0, 'snr': 10.0, 'naturalness': 0.4}}
        }
        
        for category, data in categories.items():
            score = scorer.calculate_score(data['metrics'])
            assert score.category == category
            
            # Check grade assignment based on score value
            if score.value >= 90:
                assert score.grade == 'A'
            elif score.value >= 80:
                assert score.grade == 'B'
            elif score.value >= 70:
                assert score.grade == 'C'
            elif score.value >= 60:
                assert score.grade == 'D'
            else:
                assert score.grade == 'F'
    
    def test_score_explanation(self, scorer, sample_metrics):
        """Test score explanation generation"""
        score = scorer.calculate_score(sample_metrics)
        
        # Get explanation
        explanation = scorer.explain_score(score)
        
        assert isinstance(explanation, dict)
        assert 'overall_assessment' in explanation
        assert 'metric_contributions' in explanation
        assert 'strengths' in explanation
        assert 'weaknesses' in explanation
        assert 'recommendations' in explanation
        
        # Check metric contributions
        contributions = explanation['metric_contributions']
        assert len(contributions) == len(sample_metrics)
        
        for metric, contrib in contributions.items():
            assert 'value' in contrib
            assert 'normalized' in contrib
            assert 'weight' in contrib
            assert 'contribution' in contrib
            assert contrib['contribution'] >= 0
        
        # Sum of contributions should equal score
        total_contrib = sum(c['contribution'] for c in contributions.values())
        assert abs(total_contrib - score.value) < 1.0
    
    def test_batch_processing_performance(self, scorer):
        """Test performance with large batch"""
        # Create large batch
        large_batch = []
        for i in range(1000):
            large_batch.append({
                'id': f'sample_{i}',
                'metrics': {
                    'pesq': np.random.uniform(1.0, 4.5),
                    'stoi': np.random.uniform(0.3, 1.0),
                    'si_sdr': np.random.uniform(-5.0, 25.0),
                    'snr': np.random.uniform(0.0, 40.0),
                    'naturalness': np.random.uniform(0.2, 1.0)
                }
            })
        
        # Time batch ranking
        import time
        start_time = time.time()
        ranked_results = scorer.rank_samples(large_batch)
        elapsed_time = time.time() - start_time
        
        # Should complete within 100ms
        assert elapsed_time < 0.1
        assert len(ranked_results) == 1000
    
    def test_score_stability(self, scorer):
        """Test score stability and consistency"""
        metrics = {
            'pesq': 3.2,
            'stoi': 0.82,
            'si_sdr': 14.5,
            'snr': 23.0,
            'naturalness': 0.78
        }
        
        # Calculate score multiple times
        scores = []
        for _ in range(10):
            score = scorer.calculate_score(metrics)
            scores.append(score.value)
        
        # All scores should be identical
        assert all(s == scores[0] for s in scores)
        
        # Test with slight variations
        varied_scores = []
        for i in range(10):
            varied_metrics = metrics.copy()
            # Add tiny variation
            varied_metrics['pesq'] += 0.001 * i
            score = scorer.calculate_score(varied_metrics)
            varied_scores.append(score.value)
        
        # Scores should change smoothly
        for i in range(len(varied_scores) - 1):
            diff = abs(varied_scores[i + 1] - varied_scores[i])
            assert diff < 0.5  # Small change in input = small change in output
    
    def test_missing_metrics_handling(self, scorer):
        """Test handling of missing or partial metrics"""
        # Test with missing metric
        partial_metrics = {
            'pesq': 3.5,
            'stoi': 0.85,
            'si_sdr': 15.0
            # Missing snr and naturalness
        }
        
        score = scorer.calculate_score(partial_metrics)
        assert isinstance(score, EnhancementScore)
        assert score.warnings is not None
        # Check for missing metric warnings
        assert any('Missing metric: snr' in w for w in score.warnings)
        assert any('Missing metric: naturalness' in w for w in score.warnings)
        
        # Test with empty metrics
        with pytest.raises(ValueError):
            scorer.calculate_score({})
        
        # Test with invalid metric values
        invalid_metrics = {
            'pesq': 'invalid',
            'stoi': None,
            'si_sdr': float('inf'),
            'snr': float('nan'),
            'naturalness': -999
        }
        
        with pytest.raises(ValueError):
            scorer.calculate_score(invalid_metrics)
    
    def test_historical_tracking(self, scorer, sample_metrics):
        """Test historical score tracking"""
        # Calculate multiple scores
        for i in range(50):
            varied_metrics = sample_metrics.copy()
            varied_metrics['pesq'] = np.random.uniform(2.0, 4.0)
            score = scorer.calculate_score(varied_metrics)
        
        # Check history
        assert len(scorer.score_history) == 50
        
        # Get statistics
        stats = scorer.get_historical_stats()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'percentiles' in stats
        
        # Test time-based filtering
        recent_stats = scorer.get_historical_stats(days=1)
        assert recent_stats['count'] == 50  # All are recent
        
        # Test clearing history
        scorer.clear_history()
        assert len(scorer.score_history) == 0