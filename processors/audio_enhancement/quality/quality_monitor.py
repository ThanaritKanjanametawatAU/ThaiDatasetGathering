"""
Quality Monitoring Pipeline for Pattern→MetricGAN+

Provides continuous quality monitoring, trend detection, and regression testing
for Pattern→MetricGAN+ enhancement to ensure production readiness.
"""

import numpy as np
import logging
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .pattern_metricgan_validator import (
    PatternMetricGANQualityValidator,
    QualityReport,
    QualityThresholds
)

logger = logging.getLogger(__name__)


@dataclass
class QualityAlert:
    """Quality degradation alert"""
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_name: str
    current_value: float
    threshold_value: float
    deviation: float
    timestamp: str
    affected_samples: List[str]
    description: str


@dataclass
class BatchQualityReport:
    """Quality report for a batch of processed samples"""
    batch_id: str
    timestamp: str
    total_samples: int
    passed_samples: int
    failed_samples: int
    pass_rate: float
    average_processing_time: float
    quality_metrics: Dict[str, float]
    alerts: List[QualityAlert]
    trend_analysis: Dict[str, any]


@dataclass
class QualityTrend:
    """Quality trend data"""
    metric_name: str
    time_window: str
    values: List[float]
    timestamps: List[str]
    trend_direction: str  # 'improving', 'stable', 'degrading'
    slope: float
    confidence: float


class PatternMetricGANQualityMonitor:
    """Continuous quality monitoring for Pattern→MetricGAN+ enhancement"""
    
    def __init__(self, metrics_collector=None, alert_thresholds: Optional[Dict] = None):
        self.validator = PatternMetricGANQualityValidator()
        self.metrics_collector = metrics_collector
        
        # Default alert thresholds (percentage degradation from baseline)
        self.alert_thresholds = alert_thresholds or {
            'pass_rate': {'critical': 0.7, 'high': 0.8, 'medium': 0.85, 'low': 0.9},
            'pesq_score': {'critical': 2.5, 'high': 2.8, 'medium': 3.0, 'low': 3.1},
            'stoi_score': {'critical': 0.75, 'high': 0.80, 'medium': 0.83, 'low': 0.85},
            'pattern_suppression_effectiveness': {'critical': 0.80, 'high': 0.85, 'medium': 0.88, 'low': 0.90},
            'processing_time': {'critical': 3.0, 'high': 2.5, 'medium': 2.0, 'low': 1.8}  # seconds
        }
        
        # Historical data for trend analysis
        self.quality_history = []
        self.trend_window_size = 50  # Number of recent samples for trend analysis
        
        logger.info("Initialized Pattern→MetricGAN+ quality monitor")
    
    def validate_batch_quality(
        self,
        batch_samples: List[Tuple[np.ndarray, np.ndarray, str]]
    ) -> BatchQualityReport:
        """
        Validate quality across a batch of processed samples.
        
        Args:
            batch_samples: List of (original, enhanced, sample_id) tuples
            
        Returns:
            BatchQualityReport: Comprehensive batch quality report
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Validating batch quality for {len(batch_samples)} samples (ID: {batch_id})")
        
        start_time = time.time()
        
        # Validate each sample
        reports = []
        for original, enhanced, sample_id in batch_samples:
            try:
                report = self.validator.validate_enhancement(
                    original, enhanced, sample_id
                )
                reports.append(report)
                
                # Add to quality history
                self.quality_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'sample_id': sample_id,
                    'report': report
                })
                
            except Exception as e:
                logger.error(f"Failed to validate sample {sample_id}: {e}")
                continue
        
        # Trim history if too large
        if len(self.quality_history) > self.trend_window_size * 2:
            self.quality_history = self.quality_history[-self.trend_window_size:]
        
        # Calculate batch statistics
        total_samples = len(reports)
        passed_samples = sum(1 for r in reports if r.overall_pass)
        failed_samples = total_samples - passed_samples
        pass_rate = passed_samples / total_samples if total_samples > 0 else 0.0
        
        processing_time = time.time() - start_time
        avg_processing_time = processing_time / total_samples if total_samples > 0 else 0.0
        
        # Calculate average quality metrics
        quality_metrics = self._calculate_batch_quality_metrics(reports)
        
        # Perform trend analysis
        trend_analysis = self._analyze_quality_trends()
        
        # Generate alerts
        alerts = self._check_quality_alerts(quality_metrics, pass_rate, avg_processing_time, reports)
        
        batch_report = BatchQualityReport(
            batch_id=batch_id,
            timestamp=datetime.now().isoformat(),
            total_samples=total_samples,
            passed_samples=passed_samples,
            failed_samples=failed_samples,
            pass_rate=pass_rate,
            average_processing_time=avg_processing_time,
            quality_metrics=quality_metrics,
            alerts=alerts,
            trend_analysis=trend_analysis
        )
        
        logger.info(f"Batch quality validation complete: {pass_rate:.1%} pass rate, {len(alerts)} alerts")
        
        return batch_report
    
    def _calculate_batch_quality_metrics(self, reports: List[QualityReport]) -> Dict[str, float]:
        """Calculate average quality metrics for a batch"""
        if not reports:
            return {}
        
        metrics = {}
        
        # Enhanced metrics
        for metric_name in ['pesq', 'stoi', 'si_sdr', 'snr']:
            values = [
                r.enhanced_metrics.get(metric_name, 0.0) 
                for r in reports 
                if r.enhanced_metrics
            ]
            if values:
                metrics[f'avg_{metric_name}'] = np.mean(values)
                metrics[f'std_{metric_name}'] = np.std(values)
        
        # Pattern-specific metrics
        for metric_name in ['pattern_suppression_effectiveness', 'primary_speaker_preservation', 
                           'loudness_consistency', 'naturalness_score']:
            values = [
                r.pattern_metrics.get(metric_name, 0.0) 
                for r in reports 
                if r.pattern_metrics
            ]
            if values:
                metrics[f'avg_{metric_name}'] = np.mean(values)
                metrics[f'std_{metric_name}'] = np.std(values)
        
        # Processing performance
        processing_times = [r.processing_time for r in reports]
        if processing_times:
            metrics['avg_processing_time'] = np.mean(processing_times)
            metrics['max_processing_time'] = np.max(processing_times)
            metrics['std_processing_time'] = np.std(processing_times)
        
        return metrics
    
    def _analyze_quality_trends(self) -> Dict[str, any]:
        """Analyze quality trends from recent history"""
        if len(self.quality_history) < 10:
            return {'status': 'insufficient_data', 'sample_count': len(self.quality_history)}
        
        trends = {}
        
        # Analyze key metrics trends
        key_metrics = [
            ('enhanced_metrics', 'pesq'),
            ('enhanced_metrics', 'stoi'),
            ('pattern_metrics', 'pattern_suppression_effectiveness'),
            ('pattern_metrics', 'naturalness_score')
        ]
        
        for metric_category, metric_name in key_metrics:
            values = []
            timestamps = []
            
            for entry in self.quality_history[-self.trend_window_size:]:
                report = entry['report']
                metric_dict = getattr(report, metric_category, {})
                
                if metric_name in metric_dict:
                    values.append(metric_dict[metric_name])
                    timestamps.append(entry['timestamp'])
            
            if len(values) >= 5:
                trend = self._calculate_trend(values, timestamps, metric_name)
                trends[metric_name] = trend
        
        # Analyze pass rate trend
        pass_rates = []
        pass_timestamps = []
        
        # Calculate pass rate for windows of samples
        window_size = 10
        for i in range(len(self.quality_history) - window_size + 1):
            window = self.quality_history[i:i + window_size]
            window_pass_rate = sum(1 for entry in window if entry['report'].overall_pass) / window_size
            pass_rates.append(window_pass_rate)
            pass_timestamps.append(window[-1]['timestamp'])
        
        if len(pass_rates) >= 3:
            trends['pass_rate'] = self._calculate_trend(pass_rates, pass_timestamps, 'pass_rate')
        
        trends['analysis_timestamp'] = datetime.now().isoformat()
        trends['samples_analyzed'] = len(self.quality_history)
        
        return trends
    
    def _calculate_trend(self, values: List[float], timestamps: List[str], metric_name: str) -> QualityTrend:
        """Calculate trend for a specific metric"""
        if len(values) < 3:
            return QualityTrend(
                metric_name=metric_name,
                time_window=f"last_{len(values)}_samples",
                values=values,
                timestamps=timestamps,
                trend_direction='unknown',
                slope=0.0,
                confidence=0.0
            )
        
        # Calculate linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate correlation coefficient for confidence
        correlation = np.corrcoef(x, values)[0, 1]
        confidence = abs(correlation)
        
        # Determine trend direction
        if abs(slope) < 0.001:  # Essentially flat
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'improving'
        else:
            trend_direction = 'degrading'
        
        return QualityTrend(
            metric_name=metric_name,
            time_window=f"last_{len(values)}_samples",
            values=values,
            timestamps=timestamps,
            trend_direction=trend_direction,
            slope=slope,
            confidence=confidence
        )
    
    def _check_quality_alerts(
        self,
        quality_metrics: Dict[str, float],
        pass_rate: float,
        avg_processing_time: float,
        reports: List[QualityReport]
    ) -> List[QualityAlert]:
        """Check for quality degradation alerts"""
        alerts = []
        
        # Check pass rate
        alerts.extend(self._check_metric_threshold('pass_rate', pass_rate, reports))
        
        # Check processing time
        alerts.extend(self._check_metric_threshold('processing_time', avg_processing_time, reports))
        
        # Check quality metrics
        for metric_name, value in quality_metrics.items():
            if metric_name.startswith('avg_'):
                base_metric = metric_name[4:]  # Remove 'avg_' prefix
                if base_metric in self.alert_thresholds:
                    alerts.extend(self._check_metric_threshold(base_metric, value, reports))
        
        return alerts
    
    def _check_metric_threshold(
        self,
        metric_name: str,
        current_value: float,
        reports: List[QualityReport]
    ) -> List[QualityAlert]:
        """Check if a metric violates alert thresholds"""
        if metric_name not in self.alert_thresholds:
            return []
        
        thresholds = self.alert_thresholds[metric_name]
        alerts = []
        
        # Determine severity based on threshold violation
        severity = None
        threshold_value = None
        
        for level in ['critical', 'high', 'medium', 'low']:
            threshold = thresholds[level]
            
            # Different comparison logic for different metrics
            if metric_name == 'processing_time':
                # Higher is worse for processing time
                if current_value > threshold:
                    severity = level
                    threshold_value = threshold
                    break
            else:
                # Lower is worse for quality metrics
                if current_value < threshold:
                    severity = level
                    threshold_value = threshold
                    break
        
        if severity:
            # Get affected samples
            affected_samples = []
            if metric_name == 'pass_rate':
                affected_samples = [r.sample_id for r in reports if not r.overall_pass]
            else:
                # For other metrics, find samples below threshold
                for report in reports:
                    sample_metric_value = self._get_sample_metric_value(report, metric_name)
                    if sample_metric_value is not None and sample_metric_value < threshold_value:
                        affected_samples.append(report.sample_id)
            
            deviation = abs(current_value - threshold_value)
            
            alert = QualityAlert(
                alert_id=f"alert_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type='threshold_violation',
                severity=severity,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                deviation=deviation,
                timestamp=datetime.now().isoformat(),
                affected_samples=affected_samples[:10],  # Limit to first 10 for brevity
                description=f"{metric_name} ({current_value:.3f}) below {severity} threshold ({threshold_value:.3f})"
            )
            
            alerts.append(alert)
        
        return alerts
    
    def _get_sample_metric_value(self, report: QualityReport, metric_name: str) -> Optional[float]:
        """Get metric value from a specific sample report"""
        if metric_name == 'pass_rate':
            return 1.0 if report.overall_pass else 0.0
        elif metric_name == 'processing_time':
            return report.processing_time
        elif metric_name in ['pesq_score', 'stoi_score']:
            base_metric = metric_name.replace('_score', '')
            return report.enhanced_metrics.get(base_metric)
        elif metric_name in report.pattern_metrics:
            return report.pattern_metrics.get(metric_name)
        
        return None
    
    def detect_quality_degradation(
        self,
        recent_results: List[QualityReport]
    ) -> List[QualityAlert]:
        """Detect trends indicating quality degradation"""
        if len(recent_results) < 10:
            return []
        
        alerts = []
        
        # Check for declining trends in key metrics
        key_metrics = ['pesq', 'stoi', 'pattern_suppression_effectiveness']
        
        for metric_name in key_metrics:
            values = []
            for report in recent_results:
                if metric_name in ['pesq', 'stoi']:
                    value = report.enhanced_metrics.get(metric_name)
                else:
                    value = report.pattern_metrics.get(metric_name)
                
                if value is not None:
                    values.append(value)
            
            if len(values) >= 5:
                # Calculate trend
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                
                # Check for significant decline
                if slope < -0.01:  # Declining trend threshold
                    correlation = abs(np.corrcoef(x, values)[0, 1])
                    
                    if correlation > 0.7:  # Strong correlation indicates real trend
                        alert = QualityAlert(
                            alert_id=f"trend_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            alert_type='trend_degradation',
                            severity='medium' if slope > -0.05 else 'high',
                            metric_name=metric_name,
                            current_value=values[-1],
                            threshold_value=values[0],
                            deviation=abs(slope),
                            timestamp=datetime.now().isoformat(),
                            affected_samples=[r.sample_id for r in recent_results[-5:]],
                            description=f"{metric_name} showing declining trend (slope: {slope:.4f})"
                        )
                        alerts.append(alert)
        
        return alerts
    
    def generate_quality_dashboard_metrics(self) -> Dict[str, any]:
        """Generate metrics for quality monitoring dashboard"""
        if not self.quality_history:
            return {'status': 'no_data', 'message': 'No quality data available'}
        
        recent_reports = [entry['report'] for entry in self.quality_history[-20:]]
        
        # Calculate current metrics
        current_pass_rate = sum(1 for r in recent_reports if r.overall_pass) / len(recent_reports)
        
        current_metrics = self._calculate_batch_quality_metrics(recent_reports)
        
        # Get trends
        trends = self._analyze_quality_trends()
        
        # Count active alerts
        dummy_batch_report = self.validate_batch_quality([])  # Get current alert thresholds
        active_alerts_count = len(dummy_batch_report.alerts) if hasattr(dummy_batch_report, 'alerts') else 0
        
        dashboard_metrics = {
            'timestamp': datetime.now().isoformat(),
            'current_status': {
                'pass_rate': current_pass_rate,
                'total_samples_analyzed': len(self.quality_history),
                'recent_samples': len(recent_reports)
            },
            'current_metrics': current_metrics,
            'trends': trends,
            'alerts': {
                'active_count': active_alerts_count,
                'severity_breakdown': {}  # Would be populated with actual alert analysis
            },
            'performance': {
                'avg_processing_time': current_metrics.get('avg_processing_time', 0.0),
                'max_processing_time': current_metrics.get('max_processing_time', 0.0)
            }
        }
        
        return dashboard_metrics


class QualityRegressionTester:
    """Regression testing for Pattern→MetricGAN+ quality"""
    
    def __init__(self, baseline_data_path: Optional[str] = None):
        self.validator = PatternMetricGANQualityValidator()
        self.baseline_data_path = baseline_data_path
        self.baseline_results = {}
        
        # Load baseline data if available
        if baseline_data_path and os.path.exists(baseline_data_path):
            self._load_baseline_data()
        
        logger.info("Initialized quality regression tester")
    
    def _load_baseline_data(self):
        """Load baseline quality results"""
        try:
            with open(self.baseline_data_path, 'r') as f:
                data = json.load(f)
                self.baseline_results = data.get('baseline_results', {})
            logger.info(f"Loaded {len(self.baseline_results)} baseline quality results")
        except Exception as e:
            logger.error(f"Failed to load baseline data: {e}")
    
    def establish_baseline(
        self,
        test_samples: List[Tuple[np.ndarray, np.ndarray, str]],
        version: str = "v1.0"
    ) -> Dict[str, QualityReport]:
        """
        Establish baseline quality results for regression testing.
        
        Args:
            test_samples: List of (original, enhanced, sample_id) tuples
            version: Version identifier for the baseline
            
        Returns:
            Dict[str, QualityReport]: Baseline quality reports
        """
        logger.info(f"Establishing quality baseline for version {version} with {len(test_samples)} samples")
        
        baseline_reports = {}
        
        for original, enhanced, sample_id in test_samples:
            try:
                report = self.validator.validate_enhancement(
                    original, enhanced, sample_id
                )
                baseline_reports[sample_id] = report
                
            except Exception as e:
                logger.error(f"Failed to establish baseline for {sample_id}: {e}")
                continue
        
        # Store baseline results
        self.baseline_results[version] = {
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(baseline_reports),
            'reports': {
                sample_id: {
                    'overall_pass': report.overall_pass,
                    'enhanced_metrics': report.enhanced_metrics,
                    'pattern_metrics': report.pattern_metrics,
                    'processing_time': report.processing_time
                }
                for sample_id, report in baseline_reports.items()
            }
        }
        
        # Save to file if path provided
        if self.baseline_data_path:
            self._save_baseline_data()
        
        logger.info(f"Established baseline with {len(baseline_reports)} samples for version {version}")
        
        return baseline_reports
    
    def _save_baseline_data(self):
        """Save baseline data to file"""
        try:
            os.makedirs(os.path.dirname(self.baseline_data_path), exist_ok=True)
            
            data = {
                'last_updated': datetime.now().isoformat(),
                'baseline_results': self.baseline_results
            }
            
            with open(self.baseline_data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved baseline data to {self.baseline_data_path}")
            
        except Exception as e:
            logger.error(f"Failed to save baseline data: {e}")
    
    def test_against_baseline_quality(
        self,
        current_samples: List[Tuple[np.ndarray, np.ndarray, str]],
        baseline_version: str = "v1.0",
        tolerance: float = 0.05
    ) -> Dict[str, any]:
        """
        Test current quality against established baselines.
        
        Args:
            current_samples: Current test samples
            baseline_version: Version to compare against
            tolerance: Acceptable degradation threshold (5% by default)
            
        Returns:
            Dict: Regression test report
        """
        logger.info(f"Running regression test against baseline {baseline_version}")
        
        if baseline_version not in self.baseline_results:
            return {
                'status': 'error',
                'message': f"Baseline version {baseline_version} not found"
            }
        
        baseline_data = self.baseline_results[baseline_version]
        baseline_reports = baseline_data['reports']
        
        # Test current samples
        current_reports = {}
        for original, enhanced, sample_id in current_samples:
            if sample_id in baseline_reports:
                try:
                    report = self.validator.validate_enhancement(
                        original, enhanced, sample_id
                    )
                    current_reports[sample_id] = report
                except Exception as e:
                    logger.error(f"Failed to test sample {sample_id}: {e}")
                    continue
        
        # Compare results
        regression_results = self._compare_quality_results(
            baseline_reports, current_reports, tolerance
        )
        
        regression_report = {
            'test_timestamp': datetime.now().isoformat(),
            'baseline_version': baseline_version,
            'baseline_timestamp': baseline_data['timestamp'],
            'tolerance': tolerance,
            'samples_tested': len(current_reports),
            'samples_in_baseline': len(baseline_reports),
            'regression_results': regression_results,
            'overall_status': 'pass' if regression_results['overall_regression_detected'] == False else 'fail'
        }
        
        logger.info(f"Regression test complete: {regression_report['overall_status']}")
        
        return regression_report
    
    def _compare_quality_results(
        self,
        baseline_reports: Dict[str, any],
        current_reports: Dict[str, QualityReport],
        tolerance: float
    ) -> Dict[str, any]:
        """Compare current results with baseline"""
        
        regressions = []
        improvements = []
        stable_metrics = []
        
        # Key metrics to compare
        comparison_metrics = [
            ('enhanced_metrics', 'pesq'),
            ('enhanced_metrics', 'stoi'),
            ('pattern_metrics', 'pattern_suppression_effectiveness'),
            ('pattern_metrics', 'naturalness_score')
        ]
        
        for sample_id in current_reports:
            if sample_id not in baseline_reports:
                continue
                
            baseline_data = baseline_reports[sample_id]
            current_report = current_reports[sample_id]
            
            for metric_category, metric_name in comparison_metrics:
                baseline_value = baseline_data.get(metric_category, {}).get(metric_name)
                
                if metric_category == 'enhanced_metrics':
                    current_value = current_report.enhanced_metrics.get(metric_name)
                else:
                    current_value = current_report.pattern_metrics.get(metric_name)
                
                if baseline_value is not None and current_value is not None:
                    # Calculate relative change
                    relative_change = (current_value - baseline_value) / baseline_value
                    
                    if relative_change < -tolerance:  # Significant degradation
                        regressions.append({
                            'sample_id': sample_id,
                            'metric': metric_name,
                            'baseline_value': baseline_value,
                            'current_value': current_value,
                            'relative_change': relative_change,
                            'absolute_change': current_value - baseline_value
                        })
                    elif relative_change > tolerance:  # Significant improvement
                        improvements.append({
                            'sample_id': sample_id,
                            'metric': metric_name,
                            'baseline_value': baseline_value,
                            'current_value': current_value,
                            'relative_change': relative_change,
                            'absolute_change': current_value - baseline_value
                        })
                    else:  # Stable
                        stable_metrics.append({
                            'sample_id': sample_id,
                            'metric': metric_name,
                            'baseline_value': baseline_value,
                            'current_value': current_value,
                            'relative_change': relative_change
                        })
        
        # Calculate overall pass rate comparison
        baseline_pass_rate = sum(
            1 for report_data in baseline_reports.values() 
            if report_data.get('overall_pass', False)
        ) / len(baseline_reports)
        
        current_pass_rate = sum(
            1 for report in current_reports.values() 
            if report.overall_pass
        ) / len(current_reports)
        
        pass_rate_change = current_pass_rate - baseline_pass_rate
        
        return {
            'overall_regression_detected': len(regressions) > 0 or pass_rate_change < -tolerance,
            'regressions': regressions,
            'improvements': improvements,
            'stable_metrics': stable_metrics,
            'pass_rate_comparison': {
                'baseline_pass_rate': baseline_pass_rate,
                'current_pass_rate': current_pass_rate,
                'change': pass_rate_change,
                'significant': abs(pass_rate_change) > tolerance
            },
            'summary': {
                'total_comparisons': len(regressions) + len(improvements) + len(stable_metrics),
                'regression_count': len(regressions),
                'improvement_count': len(improvements),
                'stable_count': len(stable_metrics)
            }
        }
    
    def validate_version_compatibility(
        self,
        version_results: Dict[str, List[QualityReport]]
    ) -> Dict[str, any]:
        """Validate quality consistency across versions"""
        
        if len(version_results) < 2:
            return {
                'status': 'error',
                'message': 'Need at least 2 versions for compatibility testing'
            }
        
        versions = list(version_results.keys())
        compatibility_matrix = {}
        
        # Compare each version pair
        for i, version_a in enumerate(versions):
            for version_b in versions[i+1:]:
                # Calculate compatibility score
                compatibility_score = self._calculate_version_compatibility(
                    version_results[version_a],
                    version_results[version_b]
                )
                
                pair_key = f"{version_a}_vs_{version_b}"
                compatibility_matrix[pair_key] = compatibility_score
        
        # Overall compatibility assessment
        avg_compatibility = np.mean([
            score['compatibility_score'] 
            for score in compatibility_matrix.values()
        ])
        
        compatibility_report = {
            'test_timestamp': datetime.now().isoformat(),
            'versions_tested': versions,
            'compatibility_matrix': compatibility_matrix,
            'overall_compatibility': avg_compatibility,
            'compatibility_status': 'compatible' if avg_compatibility > 0.9 else 'issues_detected'
        }
        
        return compatibility_report
    
    def _calculate_version_compatibility(
        self,
        reports_a: List[QualityReport],
        reports_b: List[QualityReport]
    ) -> Dict[str, any]:
        """Calculate compatibility score between two versions"""
        
        # Calculate average metrics for each version
        metrics_a = self._calculate_average_metrics(reports_a)
        metrics_b = self._calculate_average_metrics(reports_b)
        
        # Calculate differences
        metric_differences = {}
        compatibility_scores = []
        
        for metric_name in metrics_a:
            if metric_name in metrics_b:
                diff = abs(metrics_a[metric_name] - metrics_b[metric_name])
                relative_diff = diff / max(metrics_a[metric_name], 0.001)
                
                metric_differences[metric_name] = {
                    'value_a': metrics_a[metric_name],
                    'value_b': metrics_b[metric_name],
                    'absolute_difference': diff,
                    'relative_difference': relative_diff
                }
                
                # Score based on how close the values are
                compatibility_score = max(0, 1 - relative_diff * 5)  # Penalize >20% differences
                compatibility_scores.append(compatibility_score)
        
        overall_compatibility = np.mean(compatibility_scores) if compatibility_scores else 0.0
        
        return {
            'compatibility_score': overall_compatibility,
            'metric_differences': metric_differences,
            'total_metrics_compared': len(compatibility_scores)
        }
    
    def _calculate_average_metrics(self, reports: List[QualityReport]) -> Dict[str, float]:
        """Calculate average metrics from a list of reports"""
        if not reports:
            return {}
        
        metrics = {}
        
        # Enhanced metrics
        for metric_name in ['pesq', 'stoi', 'si_sdr']:
            values = [
                r.enhanced_metrics.get(metric_name, 0.0) 
                for r in reports 
                if r.enhanced_metrics and metric_name in r.enhanced_metrics
            ]
            if values:
                metrics[f'avg_{metric_name}'] = np.mean(values)
        
        # Pattern metrics
        for metric_name in ['pattern_suppression_effectiveness', 'naturalness_score']:
            values = [
                r.pattern_metrics.get(metric_name, 0.0) 
                for r in reports 
                if r.pattern_metrics and metric_name in r.pattern_metrics
            ]
            if values:
                metrics[f'avg_{metric_name}'] = np.mean(values)
        
        # Pass rate
        pass_rate = sum(1 for r in reports if r.overall_pass) / len(reports)
        metrics['pass_rate'] = pass_rate
        
        return metrics