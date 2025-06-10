"""
Trend Analysis Module for Audio Enhancement Quality Metrics
Provides time series analysis, prediction, and anomaly detection
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from scipy import stats, signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction enumeration"""
    UPWARD = "upward"
    DOWNWARD = "downward"
    STABLE = "stable"
    UNKNOWN = "unknown"


class TrendPattern(Enum):
    """Common trend patterns"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    IRREGULAR = "irregular"


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    timestamp: datetime
    metric: str
    actual_value: float
    expected_value: float
    deviation: float
    severity: str  # 'low', 'medium', 'high'
    confidence: float
    
    @property
    def deviation_percentage(self) -> float:
        """Calculate deviation as percentage"""
        if self.expected_value == 0:
            return 100.0
        return abs((self.actual_value - self.expected_value) / self.expected_value) * 100


@dataclass
class Prediction:
    """Represents a quality prediction"""
    metric: str
    mean: np.ndarray
    std: np.ndarray
    confidence_interval: Dict[str, np.ndarray]
    timestamps: pd.DatetimeIndex
    model_info: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manages trend-based alerts"""
    
    def __init__(self):
        self.alerts = {}
        self.alert_history = []
    
    def add_alert(self, alert_id: str, metric: str, condition: str, 
                  threshold: Optional[float] = None, **kwargs):
        """Add an alert configuration"""
        self.alerts[alert_id] = {
            'metric': metric,
            'condition': condition,
            'threshold': threshold,
            'enabled': True,
            **kwargs
        }
    
    def check_alerts(self, data: pd.DataFrame, trends: Dict, 
                    anomalies: List[Anomaly]) -> List[Dict]:
        """Check all configured alerts against current data"""
        triggered_alerts = []
        
        for alert_id, config in self.alerts.items():
            if not config['enabled']:
                continue
            
            alert = self._check_single_alert(alert_id, config, data, trends, anomalies)
            if alert:
                triggered_alerts.append(alert)
                self.alert_history.append(alert)
        
        return triggered_alerts
    
    def _check_single_alert(self, alert_id: str, config: Dict, 
                           data: pd.DataFrame, trends: Dict, 
                           anomalies: List[Anomaly]) -> Optional[Dict]:
        """Check a single alert condition"""
        metric = config['metric']
        condition = config['condition']
        threshold = config['threshold']
        
        if condition == 'declining':
            if metric in trends and trends[metric]['direction'] == TrendDirection.DOWNWARD:
                if abs(trends[metric]['slope']) > threshold:
                    return {
                        'id': alert_id,
                        'timestamp': datetime.now(),
                        'severity': 'high',
                        'message': f"{metric} is declining at rate {trends[metric]['slope']:.3f}/day",
                        'metric': metric,
                        'value': trends[metric]['slope']
                    }
        
        elif condition == 'below_threshold':
            if metric in data.columns:
                # Use last 5 values instead of 10 for more recent data
                recent_values = data[metric].tail(5)
                below_count = sum(recent_values < threshold)
                if below_count > 2:  # More than half below threshold
                    return {
                        'id': alert_id,
                        'timestamp': datetime.now(),
                        'severity': 'medium',
                        'message': f"{metric} consistently below threshold {threshold}",
                        'metric': metric,
                        'value': recent_values.mean()
                    }
        
        elif condition == 'anomaly':
            metric_anomalies = [a for a in anomalies if a.metric == metric or metric == 'all']
            if metric_anomalies:
                worst_anomaly = max(metric_anomalies, key=lambda x: x.deviation)
                return {
                    'id': alert_id,
                    'timestamp': worst_anomaly.timestamp,
                    'severity': worst_anomaly.severity,
                    'message': f"Anomaly detected in {worst_anomaly.metric}: {worst_anomaly.deviation_percentage:.1f}% deviation",
                    'metric': worst_anomaly.metric,
                    'value': worst_anomaly.actual_value
                }
        
        return None


class TrendAnalyzer:
    """Main trend analysis system"""
    
    def __init__(self, window_size: int = 100, models: Optional[Dict] = None):
        """Initialize trend analyzer
        
        Args:
            window_size: Default window size for moving averages
            models: Pre-configured prediction models
        """
        self.window_size = window_size
        self.models = models or {}
        self.alert_manager = AlertManager()
        self.scaler = StandardScaler()
        self._trend_cache = {}
    
    def calculate_moving_averages(self, data: np.ndarray, 
                                 windows: List[int]) -> Dict[str, np.ndarray]:
        """Calculate multiple moving averages
        
        Args:
            data: Time series data
            windows: List of window sizes
            
        Returns:
            Dictionary of moving averages
        """
        result = {}
        
        for window in windows:
            # Use pandas for efficient rolling window calculation
            series = pd.Series(data)
            ma = series.rolling(window=window, min_periods=1).mean()
            result[f'ma_{window}'] = ma.values
        
        return result
    
    def analyze_trends(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze trends for all metrics in the dataframe
        
        Args:
            data: DataFrame with timestamp and metric columns
            
        Returns:
            Dictionary of trend analysis results per metric
        """
        if len(data) < 5:
            raise ValueError("Insufficient data for trend analysis (need at least 5 points)")
        
        trends = {}
        
        # Identify metric columns (exclude timestamp)
        metric_columns = [col for col in data.columns if col != 'timestamp']
        
        for metric in metric_columns:
            if metric in data.columns:
                trends[metric] = self._analyze_single_trend(data, metric)
        
        return trends
    
    def _analyze_single_trend(self, data: pd.DataFrame, metric: str) -> Dict:
        """Analyze trend for a single metric"""
        values = data[metric].values
        timestamps = data['timestamp'] if 'timestamp' in data else data.index
        
        # Remove NaN values
        mask = ~np.isnan(values)
        values = values[mask]
        
        if len(values) < 3:
            return {
                'direction': TrendDirection.UNKNOWN,
                'slope': 0.0,
                'r_squared': 0.0,
                'confidence_interval': {'lower': 0.0, 'upper': 0.0}
            }
        
        # Fit linear regression
        X = np.arange(len(values)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, values)
        
        # Calculate slope per day (assuming daily data)
        slope = model.coef_[0]
        predictions = model.predict(X)
        
        # Calculate R-squared
        ss_res = np.sum((values - predictions) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine trend direction
        if abs(slope) < 0.001:  # Threshold for stability
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.UPWARD
        else:
            direction = TrendDirection.DOWNWARD
        
        # Calculate confidence interval for slope
        residuals = values - predictions
        std_error = np.sqrt(np.sum(residuals**2) / (len(values) - 2))
        t_stat = stats.t.ppf(0.975, len(values) - 2)  # 95% CI
        margin = t_stat * std_error / np.sqrt(np.sum((X - X.mean())**2))
        
        # Check for seasonal patterns
        seasonal_info = self._detect_seasonality(values)
        
        result = {
            'direction': direction,
            'slope': float(slope),
            'r_squared': float(r_squared),
            'confidence_interval': {
                'lower': float(slope - margin),
                'upper': float(slope + margin)
            }
        }
        
        if seasonal_info['detected']:
            result['seasonal_pattern'] = seasonal_info
        
        return result
    
    def _detect_seasonality(self, values: np.ndarray) -> Dict:
        """Detect seasonal patterns in data"""
        if len(values) < 14:  # Need at least 2 weeks for weekly pattern
            return {'detected': False}
        
        # Try common periods (daily, weekly)
        periods_to_test = [7, 30]  # Weekly, monthly
        
        for period in periods_to_test:
            if len(values) >= 2 * period:
                # Use autocorrelation to detect seasonality
                autocorr = pd.Series(values).autocorr(lag=period)
                
                if autocorr > 0.7:  # Strong correlation at this lag
                    return {
                        'detected': True,
                        'period': period,
                        'strength': float(autocorr)
                    }
        
        return {'detected': False}
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        sensitivity: str = 'medium') -> List[Anomaly]:
        """Detect anomalies in time series data
        
        Args:
            data: DataFrame with metrics
            sensitivity: 'low', 'medium', or 'high'
            
        Returns:
            List of detected anomalies
        """
        # Set threshold based on sensitivity
        thresholds = {
            'low': 3.0,     # 3 standard deviations
            'medium': 2.5,  # 2.5 standard deviations
            'high': 2.0     # 2 standard deviations
        }
        threshold = thresholds.get(sensitivity, 2.5)
        
        anomalies = []
        metric_columns = [col for col in data.columns if col != 'timestamp']
        
        for metric in metric_columns:
            if metric not in data.columns:
                continue
            
            values = data[metric].values
            
            # Use rolling statistics for anomaly detection
            window = min(20, len(values) // 3)
            if window < 3:
                continue
            
            series = pd.Series(values)
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()
            
            # Detect anomalies
            for i in range(len(values)):
                if pd.isna(rolling_mean.iloc[i]) or pd.isna(rolling_std.iloc[i]):
                    continue
                
                z_score = abs((values[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i])
                
                if z_score > threshold:
                    # Determine severity
                    if z_score > 4:
                        severity = 'high'
                    elif z_score > 3:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    timestamp = data['timestamp'].iloc[i] if 'timestamp' in data else datetime.now()
                    
                    anomaly = Anomaly(
                        timestamp=timestamp,
                        metric=metric,
                        actual_value=float(values[i]),
                        expected_value=float(rolling_mean.iloc[i]),
                        deviation=float(values[i] - rolling_mean.iloc[i]),
                        severity=severity,
                        confidence=min(0.99, z_score / 5.0)  # Cap at 99%
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def fit(self, data: pd.DataFrame):
        """Fit prediction models on historical data
        
        Args:
            data: Historical data for training
        """
        metric_columns = [col for col in data.columns if col != 'timestamp']
        
        for metric in metric_columns:
            if metric in data.columns:
                # For simplicity, use linear regression
                # In production, could use ARIMA, Prophet, etc.
                values = data[metric].dropna().values
                
                if len(values) > 10:
                    X = np.arange(len(values)).reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, values)
                    
                    self.models[metric] = {
                        'model': model,
                        'last_value': values[-1],
                        'last_index': len(values) - 1,
                        'std': np.std(values),
                        'mean': np.mean(values)
                    }
    
    def predict_quality(self, horizon: int = 10) -> Dict[str, Prediction]:
        """Predict future quality metrics
        
        Args:
            horizon: Number of periods to predict
            
        Returns:
            Dictionary of predictions per metric
        """
        predictions = {}
        
        for metric, model_info in self.models.items():
            model = model_info['model']
            last_index = model_info['last_index']
            std = model_info['std']
            
            # Generate future indices
            future_indices = np.arange(last_index + 1, last_index + horizon + 1).reshape(-1, 1)
            
            # Make predictions
            mean_predictions = model.predict(future_indices)
            
            # Add some uncertainty that grows with horizon
            uncertainty = std * np.sqrt(1 + np.arange(1, horizon + 1) / 10)
            
            # Calculate confidence intervals
            z_score = 1.96  # 95% confidence
            lower_bound = mean_predictions - z_score * uncertainty
            upper_bound = mean_predictions + z_score * uncertainty
            
            predictions[metric] = {
                'mean': mean_predictions,
                'std': uncertainty,
                'confidence_interval': {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
            }
        
        return predictions
    
    def analyze_seasonality(self, data: pd.DataFrame, metric: str) -> Dict:
        """Analyze seasonal patterns in detail
        
        Args:
            data: Time series data
            metric: Metric to analyze
            
        Returns:
            Detailed seasonality analysis
        """
        if metric not in data.columns:
            raise ValueError(f"Metric {metric} not found in data")
        
        values = data[metric].dropna().values
        
        if len(values) < 14:
            return {'has_seasonality': False}
        
        # Try to detect periodicity using FFT
        fft = np.fft.fft(values)
        power = np.abs(fft)**2
        freqs = np.fft.fftfreq(len(values))
        
        # Find dominant frequencies (excluding DC component)
        idx = np.argsort(power[1:len(values)//2])[::-1] + 1
        dominant_freq = freqs[idx[0]]
        
        if dominant_freq > 0:
            period = int(1 / dominant_freq)
            
            # Validate the period using autocorrelation
            if period > 0 and period < len(values) // 2:
                autocorr = pd.Series(values).autocorr(lag=period)
                
                if autocorr > 0.5:  # Moderate to strong correlation
                    # Extract seasonal component
                    seasonal_component = self._extract_seasonal_component(values, period)
                    
                    # Calculate strength of seasonality
                    # Use a simple measure: correlation at the detected period
                    seasonal_strength = autocorr
                    
                    return {
                        'has_seasonality': True,
                        'period': period,
                        'strength': float(max(0, min(1, seasonal_strength))),
                        'seasonal_component': seasonal_component
                    }
        
        return {'has_seasonality': False}
    
    def _extract_seasonal_component(self, values: np.ndarray, period: int) -> np.ndarray:
        """Extract seasonal component from time series"""
        # Simple seasonal decomposition
        n_periods = len(values) // period
        seasonal = np.zeros(len(values))
        
        for i in range(period):
            indices = np.arange(i, len(values), period)
            if len(indices) > 0:
                seasonal_mean = np.mean(values[indices])
                seasonal[indices] = seasonal_mean
        
        # Center the seasonal component
        seasonal = seasonal - np.mean(seasonal)
        
        return seasonal
    
    def set_alert(self, alert_id: str, metric: str, condition: str,
                  threshold: Optional[float] = None, **kwargs):
        """Set up an alert
        
        Args:
            alert_id: Unique identifier for the alert
            metric: Metric to monitor
            condition: Alert condition ('declining', 'below_threshold', 'anomaly')
            threshold: Threshold value for the condition
        """
        self.alert_manager.add_alert(alert_id, metric, condition, threshold, **kwargs)
    
    def process_alerts(self, data: pd.DataFrame) -> List[Dict]:
        """Process data and check for alerts
        
        Args:
            data: Current data to check
            
        Returns:
            List of triggered alerts
        """
        # Analyze trends
        trends = self.analyze_trends(data)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(data, sensitivity='medium')
        
        # Check alerts
        alerts = self.alert_manager.check_alerts(data, trends, anomalies)
        
        return alerts
    
    def analyze_multivariate(self, data: pd.DataFrame) -> Dict:
        """Analyze relationships between multiple metrics
        
        Args:
            data: DataFrame with multiple metrics
            
        Returns:
            Multivariate analysis results
        """
        metric_columns = [col for col in data.columns if col != 'timestamp']
        
        # Calculate correlation matrix
        correlation_matrix = {}
        for metric1 in metric_columns:
            correlation_matrix[metric1] = {}
            for metric2 in metric_columns:
                if metric1 in data and metric2 in data:
                    corr = data[metric1].corr(data[metric2])
                    correlation_matrix[metric1][metric2] = float(corr)
        
        # Find leading indicators
        leading_indicators = self._find_leading_indicators(data, metric_columns)
        
        # Identify metric relationships
        relationships = self._identify_relationships(correlation_matrix)
        
        return {
            'correlation_matrix': correlation_matrix,
            'leading_indicators': leading_indicators,
            'relationships': relationships
        }
    
    def _find_leading_indicators(self, data: pd.DataFrame, 
                                metrics: List[str]) -> Dict[str, List[str]]:
        """Find metrics that lead others in time"""
        leading_indicators = {}
        
        for target_metric in metrics:
            leaders = []
            
            for lead_metric in metrics:
                if lead_metric != target_metric:
                    # Check correlation with lag
                    max_corr = 0
                    best_lag = 0
                    
                    for lag in range(1, min(10, len(data) // 4)):
                        if lag < len(data):
                            corr = data[lead_metric].shift(lag).corr(data[target_metric])
                            if abs(corr) > abs(max_corr):
                                max_corr = corr
                                best_lag = lag
                    
                    if abs(max_corr) > 0.7:  # Strong correlation
                        leaders.append({
                            'metric': lead_metric,
                            'lag': best_lag,
                            'correlation': float(max_corr)
                        })
            
            if leaders:
                leading_indicators[target_metric] = leaders
        
        return leading_indicators
    
    def _identify_relationships(self, correlation_matrix: Dict) -> List[Dict]:
        """Identify significant relationships between metrics"""
        relationships = []
        processed_pairs = set()
        
        for metric1, correlations in correlation_matrix.items():
            for metric2, corr in correlations.items():
                if metric1 != metric2:
                    pair = tuple(sorted([metric1, metric2]))
                    if pair not in processed_pairs and abs(corr) > 0.7:
                        processed_pairs.add(pair)
                        
                        relationship_type = 'positive' if corr > 0 else 'negative'
                        strength = 'strong' if abs(corr) > 0.9 else 'moderate'
                        
                        relationships.append({
                            'metrics': list(pair),
                            'correlation': float(corr),
                            'type': relationship_type,
                            'strength': strength
                        })
        
        return relationships
    
    def detect_change_points(self, data: pd.DataFrame, metric: str) -> List[Dict]:
        """Detect points where trend changes significantly
        
        Args:
            data: Time series data
            metric: Metric to analyze
            
        Returns:
            List of detected change points
        """
        if metric not in data.columns:
            raise ValueError(f"Metric {metric} not found in data")
        
        values = data[metric].values
        if len(values) < 20:
            return []
        
        change_points = []
        
        # Use a sliding window approach to detect trend changes
        window = 10
        threshold = 0.5  # Minimum change in slope
        
        for i in range(window, len(values) - window):
            # Calculate slopes before and after
            before_values = values[i-window:i]
            after_values = values[i:i+window]
            
            # Fit linear trends
            x_before = np.arange(window)
            x_after = np.arange(window)
            
            slope_before = np.polyfit(x_before, before_values, 1)[0]
            slope_after = np.polyfit(x_after, after_values, 1)[0]
            
            # Check if trend direction changes significantly
            slope_diff = slope_after - slope_before
            
            # Normalize by data scale
            data_range = np.max(values) - np.min(values)
            normalized_diff = abs(slope_diff) * window / data_range
            
            if normalized_diff > threshold:
                # Determine trend directions
                old_trend = TrendDirection.UPWARD if slope_before > 0 else TrendDirection.DOWNWARD
                new_trend = TrendDirection.UPWARD if slope_after > 0 else TrendDirection.DOWNWARD
                
                # Only add if trend actually reverses
                if old_trend != new_trend:
                    change_points.append({
                        'timestamp': data['timestamp'].iloc[i] if 'timestamp' in data else i,
                        'confidence': min(0.99, normalized_diff),
                        'old_trend': old_trend,
                        'new_trend': new_trend
                    })
        
        # Filter to keep only most significant change points
        if change_points:
            change_points.sort(key=lambda x: x['confidence'], reverse=True)
            # Keep top change points with minimum separation
            filtered = [change_points[0]]
            for cp in change_points[1:]:
                if all(abs(data[data['timestamp'] == cp['timestamp']].index[0] - 
                          data[data['timestamp'] == f['timestamp']].index[0]) > 10 
                       for f in filtered):
                    filtered.append(cp)
            change_points = filtered
        
        return change_points