"""
Enhancement Scoring System
Combines multiple quality metrics into unified enhancement scores
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ScoreCategory(Enum):
    """Score quality categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


@dataclass
class EnhancementScore:
    """Represents an enhancement score with metadata"""
    value: float
    grade: str
    category: str
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    profile: str = "balanced"
    warnings: Optional[List[str]] = None
    percentile: Optional[float] = None
    
    def __post_init__(self):
        """Validate and normalize score"""
        if not 0 <= self.value <= 100:
            logger.warning(f"Score {self.value} outside normal range, clamping to [0, 100]")
            self.value = max(0, min(100, self.value))


@dataclass
class RankedSample:
    """Represents a ranked sample"""
    id: str
    score: float
    rank: int
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class ScoringProfile:
    """Defines scoring weight profiles"""
    
    PROFILES = {
        'balanced': {
            'pesq': 0.25,
            'stoi': 0.25,
            'si_sdr': 0.20,
            'snr': 0.15,
            'naturalness': 0.15
        },
        'intelligibility': {
            'pesq': 0.15,
            'stoi': 0.40,  # Emphasize intelligibility
            'si_sdr': 0.20,
            'snr': 0.15,
            'naturalness': 0.10
        },
        'quality': {
            'pesq': 0.40,  # Emphasize perceptual quality
            'stoi': 0.20,
            'si_sdr': 0.15,
            'snr': 0.15,
            'naturalness': 0.10
        },
        'naturalness': {
            'pesq': 0.20,
            'stoi': 0.20,
            'si_sdr': 0.15,
            'snr': 0.10,
            'naturalness': 0.35  # Emphasize naturalness
        }
    }
    
    @classmethod
    def get_profile(cls, name: str) -> Dict[str, float]:
        """Get scoring profile by name"""
        return cls.PROFILES.get(name, cls.PROFILES['balanced'])


class ScoreExplainer:
    """Explains score calculations and provides insights"""
    
    @staticmethod
    def explain(score: EnhancementScore, weights: Dict[str, float],
                normalized_values: Dict[str, float]) -> Dict:
        """Generate detailed score explanation"""
        explanation = {
            'overall_assessment': ScoreExplainer._get_overall_assessment(score),
            'metric_contributions': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Calculate metric contributions
        for metric, value in score.metrics.items():
            if metric in weights and metric in normalized_values:
                contribution = normalized_values[metric] * weights[metric] * 100
                explanation['metric_contributions'][metric] = {
                    'value': value,
                    'normalized': normalized_values[metric],
                    'weight': weights[metric],
                    'contribution': contribution
                }
                
                # Identify strengths and weaknesses
                if normalized_values[metric] > 0.8:
                    explanation['strengths'].append(f"Excellent {metric}: {value:.2f}")
                elif normalized_values[metric] < 0.5:
                    explanation['weaknesses'].append(f"Poor {metric}: {value:.2f}")
        
        # Generate recommendations
        explanation['recommendations'] = ScoreExplainer._generate_recommendations(
            score, normalized_values
        )
        
        return explanation
    
    @staticmethod
    def _get_overall_assessment(score: EnhancementScore) -> str:
        """Get overall quality assessment"""
        if score.value >= 90:
            return "Excellent enhancement quality with minimal artifacts"
        elif score.value >= 70:
            return "Good enhancement quality suitable for most applications"
        elif score.value >= 50:
            return "Fair enhancement quality with noticeable limitations"
        elif score.value >= 30:
            return "Poor enhancement quality requiring improvement"
        else:
            return "Very poor enhancement quality, consider alternative methods"
    
    @staticmethod
    def _generate_recommendations(score: EnhancementScore,
                                normalized_values: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Check each metric
        if normalized_values.get('pesq', 1.0) < 0.6:
            recommendations.append("Consider improving perceptual quality through better noise reduction")
        
        if normalized_values.get('stoi', 1.0) < 0.6:
            recommendations.append("Focus on improving speech intelligibility")
        
        if normalized_values.get('si_sdr', 1.0) < 0.5:
            recommendations.append("Enhance source separation to reduce interference")
        
        if normalized_values.get('naturalness', 1.0) < 0.6:
            recommendations.append("Reduce processing artifacts to improve naturalness")
        
        if not recommendations:
            recommendations.append("Maintain current enhancement settings")
        
        return recommendations


class EnhancementScorer:
    """Main enhancement scoring system"""
    
    # Metric normalization ranges
    METRIC_RANGES = {
        'pesq': {'min': 1.0, 'max': 4.5, 'optimal': 4.0},
        'stoi': {'min': 0.0, 'max': 1.0, 'optimal': 0.95},
        'si_sdr': {'min': -10.0, 'max': 30.0, 'optimal': 20.0},
        'snr': {'min': 0.0, 'max': 40.0, 'optimal': 30.0},
        'naturalness': {'min': 0.0, 'max': 1.0, 'optimal': 0.9}
    }
    
    def __init__(self, profile: str = 'balanced',
                 custom_weights: Optional[Dict[str, float]] = None):
        """Initialize enhancement scorer
        
        Args:
            profile: Scoring profile name
            custom_weights: Custom metric weights (must sum to 1)
        """
        self.profile = profile
        
        if profile == 'custom' and custom_weights:
            # Normalize custom weights
            total = sum(custom_weights.values())
            self.weights = {k: v/total for k, v in custom_weights.items()}
        else:
            self.weights = ScoringProfile.get_profile(profile)
        
        self.score_history = []
        self.explainer = ScoreExplainer()
    
    def calculate_score(self, metrics: Dict[str, float]) -> EnhancementScore:
        """Calculate unified enhancement score
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            EnhancementScore object
        """
        # Validate metrics
        if not metrics:
            raise ValueError("Metrics cannot be empty")
        
        # Check for invalid values
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                raise ValueError(f"Invalid value for {metric}: {value}")
        
        # Normalize metrics
        normalized = {}
        warnings = []
        
        for metric, weight in self.weights.items():
            if metric in metrics:
                normalized[metric] = self._normalize_metric(metric, metrics[metric])
            else:
                # Handle missing metrics
                warnings.append(f"Missing metric: {metric}")
                normalized[metric] = 0.5  # Default to middle value
        
        # Calculate weighted score
        score_value = sum(normalized[m] * self.weights[m] for m in self.weights) * 100
        
        # Determine grade and category
        grade = self._calculate_grade(score_value)
        category = self._calculate_category(score_value)
        
        # Create score object
        score = EnhancementScore(
            value=score_value,
            grade=grade,
            category=category,
            metrics=metrics,
            profile=self.profile,
            warnings=warnings if warnings else None
        )
        
        # Add to history
        self.score_history.append(score_value)
        
        return score
    
    def _normalize_metric(self, metric: str, value: float) -> float:
        """Normalize metric value to [0, 1] range"""
        if metric not in self.METRIC_RANGES:
            return max(0, min(1, value))  # Default normalization
        
        ranges = self.METRIC_RANGES[metric]
        
        # Clamp to valid range
        value = max(ranges['min'], min(ranges['max'], value))
        
        # Normalize with non-linear scaling
        if metric == 'pesq':
            # PESQ: 1-1.5 (very poor), 1.5-2.5 (poor), 2.5-3.5 (fair), 3.5-4.5 (good)
            if value < 1.5:
                return 0.1 * (value - 1.0) / 0.5
            elif value < 2.5:
                return 0.1 + 0.3 * (value - 1.5) / 1.0
            elif value < 3.5:
                return 0.4 + 0.3 * (value - 2.5) / 1.0
            else:
                return 0.7 + 0.3 * (value - 3.5) / 1.0
        
        elif metric == 'stoi':
            # STOI: Non-linear mapping emphasizing high values
            if value < 0.5:
                return 0.3 * value / 0.5
            elif value < 0.8:
                return 0.3 + 0.4 * (value - 0.5) / 0.3
            else:
                return 0.7 + 0.3 * (value - 0.8) / 0.2
        
        elif metric == 'si_sdr':
            # SI-SDR: Map to [0, 1] with emphasis on positive values
            if value < 0:
                return 0.2 * (value - ranges['min']) / (0 - ranges['min'])
            elif value < 10:
                return 0.2 + 0.3 * value / 10
            elif value < 20:
                return 0.5 + 0.3 * (value - 10) / 10
            else:
                return 0.8 + 0.2 * (value - 20) / 10
        
        else:
            # Linear normalization for other metrics
            return (value - ranges['min']) / (ranges['max'] - ranges['min'])
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _calculate_category(self, score: float) -> str:
        """Calculate quality category from score"""
        if score >= 90:
            return ScoreCategory.EXCELLENT.value
        elif score >= 70:
            return ScoreCategory.GOOD.value
        elif score >= 50:
            return ScoreCategory.FAIR.value
        elif score >= 30:
            return ScoreCategory.POOR.value
        else:
            return ScoreCategory.VERY_POOR.value
    
    def rank_samples(self, sample_metrics: List[Dict[str, any]]) -> List[RankedSample]:
        """Rank multiple samples by enhancement score
        
        Args:
            sample_metrics: List of dicts with 'id' and 'metrics' keys
            
        Returns:
            List of RankedSample objects sorted by score
        """
        # Calculate scores for all samples
        scored_samples = []
        for sample in sample_metrics:
            score = self.calculate_score(sample['metrics'])
            scored_samples.append({
                'id': sample['id'],
                'score': score.value,
                'metrics': sample['metrics']
            })
        
        # Sort by score (descending)
        scored_samples.sort(key=lambda x: x['score'], reverse=True)
        
        # Create ranked samples
        ranked = []
        for rank, sample in enumerate(scored_samples, 1):
            ranked.append(RankedSample(
                id=sample['id'],
                score=sample['score'],
                rank=rank,
                metrics=sample['metrics']
            ))
        
        return ranked
    
    def get_percentile(self, score: float) -> float:
        """Calculate percentile ranking for a score
        
        Args:
            score: Score value to calculate percentile for
            
        Returns:
            Percentile value (0-100)
        """
        if not self.score_history:
            return 50.0  # Default to median if no history
        
        # Count scores below the given score
        below_count = sum(1 for s in self.score_history if s < score)
        percentile = (below_count / len(self.score_history)) * 100
        
        return percentile
    
    def explain_score(self, score: EnhancementScore) -> Dict:
        """Generate detailed explanation for a score
        
        Args:
            score: EnhancementScore to explain
            
        Returns:
            Dictionary with explanation details
        """
        # Normalize metrics for explanation
        normalized = {}
        for metric in score.metrics:
            if metric in self.weights:
                normalized[metric] = self._normalize_metric(metric, score.metrics[metric])
        
        return self.explainer.explain(score, self.weights, normalized)
    
    def get_historical_stats(self, days: Optional[int] = None) -> Dict:
        """Get statistical summary of historical scores
        
        Args:
            days: Number of days to include (None for all)
            
        Returns:
            Dictionary with statistical measures
        """
        if not self.score_history:
            return {
                'count': 0,
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'percentiles': {}
            }
        
        # Filter by time if requested
        scores = self.score_history
        if days is not None:
            # For simplicity, assume all scores are recent in this implementation
            pass
        
        # Calculate statistics
        scores_array = np.array(scores)
        percentiles = [10, 25, 50, 75, 90]
        
        return {
            'count': len(scores),
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'percentiles': {
                p: float(np.percentile(scores_array, p)) 
                for p in percentiles
            }
        }
    
    def clear_history(self):
        """Clear score history"""
        self.score_history = []
    
    def save_profile(self, filename: str):
        """Save current scoring profile to file
        
        Args:
            filename: Path to save profile
        """
        profile_data = {
            'profile': self.profile,
            'weights': self.weights,
            'history_stats': self.get_historical_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2)
    
    def load_profile(self, filename: str):
        """Load scoring profile from file
        
        Args:
            filename: Path to load profile from
        """
        with open(filename, 'r') as f:
            profile_data = json.load(f)
        
        self.profile = profile_data.get('profile', 'balanced')
        self.weights = profile_data.get('weights', ScoringProfile.get_profile('balanced'))
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}