"""
Quality Threshold Manager for audio enhancement pipeline.
Provides dynamic threshold management, multi-level quality gates, and adaptive adjustment.
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a quality check."""
    passed: bool
    score: float
    failures: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    severity: str = 'none'  # none, minor, major, critical
    metric_scores: Dict[str, float] = field(default_factory=dict)
    recovery_suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityGate:
    """A quality gate for a specific pipeline stage."""
    name: str
    thresholds: Dict[str, Dict[str, float]]
    manager: Optional['QualityThresholdManager'] = None
    
    def check(self, metrics: Dict[str, float]) -> QualityCheckResult:
        """Check metrics against this gate's thresholds."""
        if self.manager:
            # Use manager's check_quality with stage-specific thresholds
            original_thresholds = self.manager.thresholds
            self.manager.thresholds = self.thresholds
            result = self.manager.check_quality(metrics)
            self.manager.thresholds = original_thresholds
            return result
        else:
            # Simple check without manager features
            failures = {}
            for metric, value in metrics.items():
                if metric in self.thresholds:
                    threshold = self.thresholds[metric]
                    if value < threshold.get('min', -np.inf):
                        failures[metric] = {
                            'value': value,
                            'threshold': threshold['min'],
                            'type': 'below_minimum'
                        }
            
            passed = len(failures) == 0
            score = 1.0 if passed else 0.5
            return QualityCheckResult(passed=passed, score=score, failures=failures)


class QualityThresholdManager:
    """
    Dynamic quality threshold management system for audio enhancement.
    
    Features:
    - Multiple profile support (production, development, dataset-specific)
    - Dynamic threshold adjustment based on historical performance
    - Multi-stage quality gates
    - Adaptive learning for threshold optimization
    - Outlier detection
    """
    
    # Absolute minimum thresholds (safety limits)
    ABSOLUTE_MINIMUMS = {
        'pesq': 2.0,
        'stoi': 0.5,
        'si_sdr': 5.0,
        'snr': 10.0,
        'mos': 2.0,
        'clarity': 0.4
    }
    
    # Default thresholds if none provided
    DEFAULT_THRESHOLDS = {
        'production': {
            'pesq': {'min': 3.0, 'target': 3.5, 'max': 5.0},
            'stoi': {'min': 0.8, 'target': 0.9, 'max': 1.0},
            'si_sdr': {'min': 10.0, 'target': 15.0, 'max': 30.0},
            'snr': {'min': 20.0, 'target': 25.0, 'max': 40.0}
        }
    }
    
    def __init__(
        self,
        profile: str = 'production',
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        metric_weights: Optional[Dict[str, float]] = None,
        enable_adaptive_learning: bool = False
    ):
        """
        Initialize Quality Threshold Manager.
        
        Args:
            profile: Profile name to use
            config: Dictionary with threshold configurations
            config_path: Path to YAML/JSON configuration file
            metric_weights: Weights for different metrics (sum to 1.0)
            enable_adaptive_learning: Enable ML-based threshold optimization
        """
        self.profile = profile
        self.history: List[Dict[str, Any]] = []
        self.metric_weights = metric_weights or {}
        self.enable_adaptive_learning = enable_adaptive_learning
        self.violation_callback: Optional[Callable] = None
        
        # Load thresholds
        self.config = self._load_config(config, config_path)
        self.thresholds = self._load_profile(profile)
        
        # Initialize adaptive learning components
        if enable_adaptive_learning:
            self.scaler = StandardScaler()
            self.success_predictor = LogisticRegression()
            self.optimized_thresholds = None
            self._samples_for_learning = []
        
        # Cache for stage-specific gates
        self._quality_gates: Dict[str, QualityGate] = {}
    
    def _load_config(self, config: Optional[Dict], config_path: Optional[str]) -> Dict:
        """Load configuration from various sources."""
        if config is not None:
            return config
        elif config_path is not None:
            path = Path(config_path)
            if path.suffix == '.yaml' or path.suffix == '.yml':
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        else:
            return self.DEFAULT_THRESHOLDS
    
    def _load_profile(self, profile: str) -> Dict[str, Dict[str, float]]:
        """Load a specific profile from configuration."""
        if profile not in self.config:
            if profile == 'production' and self.config == self.DEFAULT_THRESHOLDS:
                return self.DEFAULT_THRESHOLDS['production']
            raise ValueError(f"Profile '{profile}' not found in configuration")
        
        profile_config = self.config[profile]
        
        # Handle inheritance
        if '_inherit' in profile_config:
            base_profile = profile_config['_inherit']
            base_config = self._load_profile(base_profile).copy()
            # Override with current profile values
            for metric, thresholds in profile_config.items():
                if metric != '_inherit':
                    base_config[metric] = thresholds
            return base_config
        
        # Handle stage-specific configurations
        if any(key in ['pre_enhancement', 'post_enhancement', 'final_validation'] 
               for key in profile_config):
            # This is a stage-based configuration
            # Return the full stage config for now, will be handled by get_quality_gate
            return profile_config
        
        return profile_config
    
    def check_quality(self, metrics: Dict[str, float]) -> QualityCheckResult:
        """
        Check if metrics meet quality thresholds.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            QualityCheckResult with pass/fail status and details
        """
        failures = {}
        metric_scores = {}
        total_score = 0.0
        total_weight = 0.0
        
        # Check stage-specific thresholds if available
        actual_thresholds = self.thresholds
        if 'pre_enhancement' in actual_thresholds:
            # This is a stage-based config, use post_enhancement as default
            actual_thresholds = actual_thresholds.get('post_enhancement', {})
        
        for metric, value in metrics.items():
            if metric not in actual_thresholds:
                continue
                
            threshold = actual_thresholds[metric]
            min_val = threshold.get('min', -np.inf)
            target_val = threshold.get('target', min_val)
            max_val = threshold.get('max', np.inf)
            
            # Check violations
            if value < min_val:
                failures[metric] = {
                    'value': value,
                    'threshold': min_val,
                    'type': 'below_minimum'
                }
                metric_score = 0.0
            elif value > max_val:
                failures[metric] = {
                    'value': value,
                    'threshold': max_val,
                    'type': 'above_maximum'
                }
                metric_score = 0.8  # Still decent if above max
            else:
                # Calculate score based on position relative to target
                if value >= target_val:
                    metric_score = 1.0
                else:
                    # Linear interpolation between min and target
                    if target_val > min_val:
                        metric_score = (value - min_val) / (target_val - min_val)
                    else:
                        metric_score = 1.0  # If target == min, any value >= min is good
            
            metric_scores[metric] = metric_score
            
            # Apply weights if available
            weight = self.metric_weights.get(metric, 1.0)
            total_score += metric_score * weight
            total_weight += weight
        
        # Calculate overall score
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 1.0 if len(failures) == 0 else 0.0
        
        # Determine severity
        severity = self._determine_severity(failures, metric_scores)
        
        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(failures, metrics)
        
        result = QualityCheckResult(
            passed=len(failures) == 0,
            score=overall_score,
            failures=failures,
            severity=severity,
            metric_scores=metric_scores,
            recovery_suggestions=recovery_suggestions
        )
        
        # Call violation callback if set
        if not result.passed and self.violation_callback:
            self.violation_callback(result)
        
        return result
    
    def _determine_severity(self, failures: Dict, metric_scores: Dict) -> str:
        """Determine severity level of quality issues."""
        if not failures:
            return 'none'
        
        # Calculate average deviation from thresholds
        deviations = []
        for metric, failure in failures.items():
            value = failure['value']
            threshold = failure['threshold']
            if failure['type'] == 'below_minimum':
                deviation = (threshold - value) / threshold
            else:
                deviation = (value - threshold) / threshold
            deviations.append(deviation)
        
        avg_deviation = np.mean(deviations)
        
        if avg_deviation < 0.05:  # Less than 5% deviation
            return 'minor'
        elif avg_deviation < 0.25:  # Less than 25% deviation
            return 'major'
        else:
            return 'critical'
    
    def _generate_recovery_suggestions(
        self, 
        failures: Dict, 
        metrics: Dict[str, float]
    ) -> List[str]:
        """Generate suggestions for recovering from quality failures."""
        suggestions = []
        
        if 'pesq' in failures and failures['pesq']['type'] == 'below_minimum':
            suggestions.append("Apply enhanced noise reduction and spectral enhancement")
        
        if 'stoi' in failures and failures['stoi']['type'] == 'below_minimum':
            suggestions.append("Improve speech intelligibility with clarity enhancement")
        
        if 'si_sdr' in failures:
            suggestions.append("Apply source separation to isolate speech signal")
        
        if 'snr' in failures and failures['snr']['type'] == 'below_minimum':
            suggestions.append("Increase noise suppression strength")
        
        # Check for multiple failures
        if len(failures) >= 3:
            suggestions.insert(0, "Consider more aggressive enhancement settings")
        
        return suggestions
    
    def update_thresholds(self, performance_data: List[Dict[str, float]]):
        """
        Update thresholds based on historical performance data.
        Uses percentile-based adjustment while respecting absolute minimums.
        
        Args:
            performance_data: List of metric dictionaries from past processing
        """
        if not performance_data:
            return
        
        # Add to history
        self.history.extend(performance_data)
        
        # Calculate percentiles for each metric
        for metric in self.thresholds:
            if metric.startswith('_'):  # Skip special keys like _inherit
                continue
                
            values = [d[metric] for d in performance_data if metric in d]
            if not values:
                continue
            
            # Calculate percentiles
            p25 = np.percentile(values, 25)
            p50 = np.percentile(values, 50)
            p75 = np.percentile(values, 75)
            p95 = np.percentile(values, 95)
            
            # Update thresholds with safety limits
            old_threshold = self.thresholds[metric].copy()
            
            # Adjust minimum to 25th percentile, but not below absolute minimum
            new_min = max(p25, self.ABSOLUTE_MINIMUMS.get(metric, -np.inf))
            if metric in self.thresholds:
                self.thresholds[metric]['min'] = new_min
            
            # Adjust target to 75th percentile
            if metric in self.thresholds:
                self.thresholds[metric]['target'] = p75
            
            # Adjust max to 95th percentile
            if metric in self.thresholds and 'max' in self.thresholds[metric]:
                self.thresholds[metric]['max'] = max(p95, self.thresholds[metric]['max'])
            
            logger.info(f"Updated {metric} thresholds: {old_threshold} -> {self.thresholds[metric]}")
    
    def get_quality_gate(self, stage: str) -> QualityGate:
        """
        Get quality gate for a specific pipeline stage.
        
        Args:
            stage: Stage name (e.g., 'pre_enhancement', 'post_enhancement')
            
        Returns:
            QualityGate for the specified stage
        """
        if stage not in self._quality_gates:
            # Check if stage-specific thresholds exist
            if stage in self.thresholds:
                gate_thresholds = self.thresholds[stage]
            else:
                # Use default thresholds
                gate_thresholds = self.thresholds
            
            self._quality_gates[stage] = QualityGate(
                name=stage,
                thresholds=gate_thresholds,
                manager=self
            )
        
        return self._quality_gates[stage]
    
    def set_violation_callback(self, callback: Callable[[QualityCheckResult], None]):
        """Set callback function to be called on threshold violations."""
        self.violation_callback = callback
    
    def add_sample(self, sample: Dict[str, Any]):
        """
        Add a sample to history for adaptive learning.
        
        Args:
            sample: Dictionary with metrics and optional success indicator
        """
        self.history.append(sample)
        
        if self.enable_adaptive_learning:
            self._samples_for_learning.append(sample)
    
    def optimize_thresholds(self):
        """
        Optimize thresholds using machine learning based on historical data.
        Requires enable_adaptive_learning=True and samples with 'enhancement_success' field.
        """
        if not self.enable_adaptive_learning:
            raise RuntimeError("Adaptive learning not enabled")
        
        if not self._samples_for_learning:
            raise ValueError("No samples available for learning")
        
        # Prepare data for learning
        X = []
        y = []
        
        for sample in self._samples_for_learning:
            if 'enhancement_success' not in sample:
                continue
            
            features = []
            for metric in ['pesq', 'stoi', 'si_sdr', 'snr']:
                if metric in sample:
                    features.append(sample[metric])
                else:
                    features.append(0.0)  # Use 0 if metric missing
            
            if len(features) == 4:  # All metrics present
                X.append(features)
                y.append(1 if sample['enhancement_success'] else 0)
        
        if len(X) < 10:
            logger.warning("Not enough samples for optimization")
            return
        
        # Train success predictor
        X = np.array(X)
        y = np.array(y)
        
        # Check if we have both success and failure samples
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning("Need both success and failure samples for learning")
            return
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Use class weights to handle imbalanced data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.success_predictor.fit(X_scaled, y)
        
        # Find optimal thresholds
        # Use grid search to find thresholds that maximize success rate
        success_samples = X[y == 1]
        if len(success_samples) > 0:
            self.optimized_thresholds = {}
            metrics = ['pesq', 'stoi', 'si_sdr', 'snr']
            
            for i, metric in enumerate(metrics):
                values = success_samples[:, i]
                self.optimized_thresholds[metric] = {
                    'min': np.percentile(values, 10),
                    'target': np.percentile(values, 50),
                    'max': np.percentile(values, 90)
                }
            
            logger.info("Optimized thresholds based on success patterns")
    
    def predict_enhancement_success(self, metrics: Dict[str, float]) -> float:
        """
        Predict probability of enhancement success for given metrics.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Probability of success (0.0 to 1.0)
        """
        if not self.enable_adaptive_learning:
            raise RuntimeError("Adaptive learning not enabled")
        
        if not hasattr(self, 'success_predictor') or not hasattr(self.success_predictor, 'coef_'):
            return 0.5  # No model trained yet
        
        # Prepare features
        features = []
        for metric in ['pesq', 'stoi', 'si_sdr', 'snr']:
            if metric in metrics:
                features.append(metrics[metric])
            else:
                # Use median from history if available
                hist_values = [s[metric] for s in self.history if metric in s]
                if hist_values:
                    features.append(np.median(hist_values))
                else:
                    features.append(0.0)
        
        if len(features) != 4:
            return 0.5
        
        # Predict
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        prob = self.success_predictor.predict_proba(X_scaled)[0, 1]
        
        return float(prob)
    
    def switch_profile(self, new_profile: str):
        """
        Switch to a different threshold profile.
        
        Args:
            new_profile: Name of the profile to switch to
        """
        self.profile = new_profile
        self.thresholds = self._load_profile(new_profile)
        self._quality_gates.clear()  # Clear cached gates
        logger.info(f"Switched to profile: {new_profile}")
    
    def detect_outliers(
        self, 
        metrics: Dict[str, float], 
        z_threshold: float = 2.0
    ) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        """
        Detect if metrics contain outliers based on historical data.
        
        Args:
            metrics: Dictionary of metric values
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            Tuple of (is_outlier, outlier_info)
        """
        if len(self.history) < 20:
            # Not enough history for reliable outlier detection
            return False, {}
        
        outlier_info = {}
        has_outlier = False
        
        for metric, value in metrics.items():
            # Get historical values
            hist_values = [s[metric] for s in self.history if metric in s]
            if len(hist_values) < 20:
                continue
            
            # Calculate z-score
            mean = np.mean(hist_values)
            std = np.std(hist_values)
            
            if std > 0:
                z_score = (value - mean) / std
            else:
                # If no variation in history, check if value differs from mean
                if value != mean:
                    z_score = np.inf if value > mean else -np.inf
                else:
                    z_score = 0.0
                
            if abs(z_score) > z_threshold:
                has_outlier = True
                outlier_info[metric] = {
                    'value': value,
                    'mean': mean,
                    'std': std,
                    'z_score': z_score
                }
        
        return has_outlier, outlier_info