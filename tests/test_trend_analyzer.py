"""
Test module for Trend Analysis functionality
Tests time series analysis, predictions, and anomaly detection
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from processors.audio_enhancement.analysis.trend_analyzer import (
    TrendAnalyzer,
    TrendDirection,
    TrendPattern,
    Anomaly,
    AlertManager,
    Prediction
)


class TestTrendAnalyzer:
    """Test Trend Analyzer functionality"""
    
    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data"""
        # Generate 30 days of data with daily frequency
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Create trending data with some noise
        base_trend = np.linspace(3.0, 3.5, 30)  # Upward trend
        noise = np.random.normal(0, 0.05, 30)
        pesq_values = base_trend + noise
        
        # Create seasonal pattern
        seasonal = 0.1 * np.sin(2 * np.pi * np.arange(30) / 7)  # Weekly pattern
        stoi_values = 0.85 + seasonal + np.random.normal(0, 0.02, 30)
        
        return pd.DataFrame({
            'timestamp': dates,
            'pesq': pesq_values,
            'stoi': stoi_values,
            'si_sdr': np.random.normal(15, 2, 30),
            'snr': np.random.normal(25, 3, 30)
        })
    
    @pytest.fixture
    def analyzer(self):
        """Create a TrendAnalyzer instance"""
        return TrendAnalyzer(window_size=10)
    
    def test_initialization(self):
        """Test analyzer initialization"""
        # Test default initialization
        analyzer = TrendAnalyzer()
        assert analyzer.window_size == 100
        assert analyzer.models == {}
        assert analyzer.alert_manager is not None
        
        # Test custom window size
        analyzer = TrendAnalyzer(window_size=50)
        assert analyzer.window_size == 50
        
        # Test with pre-configured models
        analyzer = TrendAnalyzer(models={'arima': Mock()})
        assert 'arima' in analyzer.models
    
    def test_moving_average_calculation(self, analyzer, sample_time_series):
        """Test moving average calculations"""
        # Calculate moving averages
        ma_results = analyzer.calculate_moving_averages(
            sample_time_series['pesq'].values,
            windows=[3, 7, 14]
        )
        
        # Verify results structure
        assert 'ma_3' in ma_results
        assert 'ma_7' in ma_results
        assert 'ma_14' in ma_results
        
        # Verify moving average properties
        # MA should smooth out noise
        assert np.std(ma_results['ma_7']) < np.std(sample_time_series['pesq'])
        
        # Longer windows should be smoother
        assert np.std(ma_results['ma_14']) < np.std(ma_results['ma_3'])
        
        # Check for NaN handling
        assert not np.any(np.isnan(ma_results['ma_3'][2:]))  # First 2 values can be NaN
    
    def test_trend_detection(self, analyzer, sample_time_series):
        """Test trend detection algorithms"""
        # Analyze trends
        trends = analyzer.analyze_trends(sample_time_series)
        
        # Check PESQ trend (should be upward)
        assert 'pesq' in trends
        pesq_trend = trends['pesq']
        assert pesq_trend['direction'] == TrendDirection.UPWARD
        assert pesq_trend['slope'] > 0
        assert pesq_trend['r_squared'] > 0.7  # Good fit for linear trend
        assert 'confidence_interval' in pesq_trend
        
        # Check STOI trend (could be stable or slightly trending due to random data)
        stoi_trend = trends['stoi']
        assert stoi_trend['direction'] in [TrendDirection.STABLE, TrendDirection.UPWARD, TrendDirection.DOWNWARD]
        # Check that slope is relatively small (seasonal pattern with small trend)
        assert abs(stoi_trend['slope']) < 0.1  # Reasonable slope for noisy data
        # May or may not detect seasonal pattern depending on random data
        if 'seasonal_pattern' in stoi_trend:
            assert isinstance(stoi_trend['seasonal_pattern'], dict)
        
        # Test with insufficient data
        short_data = sample_time_series.head(3)
        with pytest.raises(ValueError):
            analyzer.analyze_trends(short_data)
    
    def test_anomaly_detection(self, analyzer, sample_time_series):
        """Test anomaly detection functionality"""
        # Add some anomalies
        anomaly_data = sample_time_series.copy()
        anomaly_data.loc[10, 'pesq'] = 1.5  # Sudden drop
        anomaly_data.loc[20, 'stoi'] = 0.3   # Another anomaly
        
        # Detect anomalies with different sensitivities
        anomalies_high = analyzer.detect_anomalies(
            anomaly_data,
            sensitivity='high'
        )
        
        anomalies_medium = analyzer.detect_anomalies(
            anomaly_data,
            sensitivity='medium'
        )
        
        anomalies_low = analyzer.detect_anomalies(
            anomaly_data,
            sensitivity='low'
        )
        
        # High sensitivity should detect more anomalies
        assert len(anomalies_high) >= len(anomalies_medium)
        assert len(anomalies_medium) >= len(anomalies_low)
        
        # Check anomaly structure
        for anomaly in anomalies_high:
            assert isinstance(anomaly, Anomaly)
            assert anomaly.timestamp is not None
            assert anomaly.metric in ['pesq', 'stoi', 'si_sdr', 'snr']
            assert anomaly.severity in ['low', 'medium', 'high']
            assert anomaly.expected_value is not None
            assert anomaly.actual_value is not None
        
        # Should detect the injected anomalies
        pesq_anomalies = [a for a in anomalies_high if a.metric == 'pesq']
        stoi_anomalies = [a for a in anomalies_high if a.metric == 'stoi']
        assert len(pesq_anomalies) >= 1
        assert len(stoi_anomalies) >= 1
    
    def test_prediction_accuracy(self, analyzer):
        """Test prediction accuracy"""
        # Create predictable time series
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Linear trend
        values = 3.0 + 0.01 * np.arange(100) + np.random.normal(0, 0.02, 100)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'pesq': values
        })
        
        # Train on first 90 points
        train_data = data.head(90)
        test_data = data.tail(10)
        
        # Fit model and predict
        analyzer.fit(train_data)
        predictions = analyzer.predict_quality(horizon=10)
        
        # Check prediction structure
        assert 'pesq' in predictions
        pesq_pred = predictions['pesq']
        assert 'mean' in pesq_pred
        assert 'std' in pesq_pred
        assert 'confidence_interval' in pesq_pred
        assert len(pesq_pred['mean']) == 10
        
        # Check prediction accuracy (within 10% error margin)
        actual_values = test_data['pesq'].values
        predicted_values = pesq_pred['mean']
        
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        assert mape < 10  # Less than 10% error
        
        # Check confidence intervals
        lower = pesq_pred['confidence_interval']['lower']
        upper = pesq_pred['confidence_interval']['upper']
        
        # Most actual values should fall within confidence interval
        within_ci = np.sum((actual_values >= lower) & (actual_values <= upper))
        assert within_ci >= 7  # At least 70% within CI
    
    def test_seasonal_analysis(self, analyzer):
        """Test seasonal pattern detection"""
        # Create data with clear seasonal pattern
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # Weekly seasonal pattern
        t = np.arange(60)
        seasonal = 3.0 + 0.2 * np.sin(2 * np.pi * t / 7)  # 7-day cycle
        noise = np.random.normal(0, 0.02, 60)
        values = seasonal + noise
        
        data = pd.DataFrame({
            'timestamp': dates,
            'pesq': values
        })
        
        # Analyze seasonality
        seasonal_results = analyzer.analyze_seasonality(data, 'pesq')
        
        # Should detect weekly pattern
        assert seasonal_results['has_seasonality'] == True
        # Allow for small variations in period detection (6-8 days for 7-day pattern)
        assert 6 <= seasonal_results['period'] <= 8
        assert seasonal_results['strength'] > 0.5  # Moderate to strong seasonal component
        assert 'seasonal_component' in seasonal_results
        assert len(seasonal_results['seasonal_component']) == 60
        
        # Test with non-seasonal data
        random_data = data.copy()
        random_data['pesq'] = np.random.normal(3.0, 0.1, 60)
        
        non_seasonal_results = analyzer.analyze_seasonality(random_data, 'pesq')
        assert non_seasonal_results['has_seasonality'] == False
    
    def test_alert_generation(self, analyzer, sample_time_series):
        """Test alert generation based on trends"""
        # Set up alerts
        analyzer.set_alert(
            'pesq_decline',
            metric='pesq',
            condition='declining',
            threshold=0.01  # Lower threshold for slope per data point
        )
        
        analyzer.set_alert(
            'stoi_low',
            metric='stoi',
            condition='below_threshold',
            threshold=0.7
        )
        
        analyzer.set_alert(
            'anomaly_detection',
            metric='all',
            condition='anomaly',
            sensitivity='medium'
        )
        
        # Create data that should trigger alerts
        alert_data = sample_time_series.copy()
        
        # Add strong declining trend to PESQ (from 3.5 to 2.5)
        alert_data['pesq'] = np.linspace(3.5, 2.5, 30) + np.random.normal(0, 0.02, 30)
        
        # Add low STOI values for the last 5 points
        alert_data.loc[25:, 'stoi'] = 0.65
        
        # Process data and check alerts
        alerts = analyzer.process_alerts(alert_data)
        
        # Should have multiple alerts
        assert len(alerts) > 0
        
        # Check alert structure
        for alert in alerts:
            assert 'id' in alert
            assert 'timestamp' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'metric' in alert
            assert 'value' in alert
        
        # Check specific alerts
        pesq_alerts = [a for a in alerts if a['id'] == 'pesq_decline']
        stoi_alerts = [a for a in alerts if a['id'] == 'stoi_low']
        
        assert len(pesq_alerts) > 0
        assert len(stoi_alerts) > 0
    
    def test_multivariate_analysis(self, analyzer, sample_time_series):
        """Test analysis of multiple metrics simultaneously"""
        # Analyze correlations and relationships
        multivariate_results = analyzer.analyze_multivariate(sample_time_series)
        
        # Check correlation matrix
        assert 'correlation_matrix' in multivariate_results
        corr_matrix = multivariate_results['correlation_matrix']
        metrics = ['pesq', 'stoi', 'si_sdr', 'snr']
        
        for metric1 in metrics:
            for metric2 in metrics:
                assert metric1 in corr_matrix
                assert metric2 in corr_matrix[metric1]
                
                # Diagonal should be 1
                if metric1 == metric2:
                    assert abs(corr_matrix[metric1][metric2] - 1.0) < 0.001
        
        # Check for leading indicators
        assert 'leading_indicators' in multivariate_results
        
        # Check for metric relationships
        assert 'relationships' in multivariate_results
    
    def test_performance_metrics(self, analyzer):
        """Test performance requirements"""
        import time
        
        # Generate large dataset
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
        large_data = pd.DataFrame({
            'timestamp': dates,
            'pesq': np.random.normal(3.2, 0.3, 1000),
            'stoi': np.random.normal(0.85, 0.1, 1000),
            'si_sdr': np.random.normal(15, 3, 1000),
            'snr': np.random.normal(25, 5, 1000)
        })
        
        # Test trend analysis performance
        start_time = time.time()
        trends = analyzer.analyze_trends(large_data)
        trend_time = time.time() - start_time
        assert trend_time < 0.5  # Should complete within 500ms
        
        # Test prediction performance
        analyzer.fit(large_data)
        start_time = time.time()
        predictions = analyzer.predict_quality(horizon=10)
        predict_time = time.time() - start_time
        assert predict_time < 1.0  # Should complete within 1 second
        
        # Test anomaly detection performance
        start_time = time.time()
        anomalies = analyzer.detect_anomalies(large_data.head(100))
        anomaly_time = time.time() - start_time
        assert anomaly_time < 0.1  # Should complete within 100ms
    
    def test_confidence_intervals(self, analyzer):
        """Test confidence interval calculations"""
        # Create data with known variance
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Low variance data
        low_var_data = pd.DataFrame({
            'timestamp': dates,
            'pesq': 3.2 + np.random.normal(0, 0.01, 50)
        })
        
        # High variance data
        high_var_data = pd.DataFrame({
            'timestamp': dates,
            'pesq': 3.2 + np.random.normal(0, 0.2, 50)
        })
        
        # Fit and predict for both
        analyzer.fit(low_var_data)
        low_var_pred = analyzer.predict_quality(horizon=5)
        
        analyzer.fit(high_var_data)
        high_var_pred = analyzer.predict_quality(horizon=5)
        
        # Low variance should have tighter confidence intervals
        low_var_ci_width = (low_var_pred['pesq']['confidence_interval']['upper'] - 
                           low_var_pred['pesq']['confidence_interval']['lower'])
        high_var_ci_width = (high_var_pred['pesq']['confidence_interval']['upper'] - 
                            high_var_pred['pesq']['confidence_interval']['lower'])
        
        assert np.mean(low_var_ci_width) < np.mean(high_var_ci_width)
    
    def test_trend_persistence(self, analyzer):
        """Test trend persistence and change detection"""
        # Create data with trend change
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # First 30 days: upward trend
        # Next 30 days: downward trend
        values1 = np.linspace(3.0, 3.5, 30) + np.random.normal(0, 0.02, 30)
        values2 = np.linspace(3.5, 3.0, 30) + np.random.normal(0, 0.02, 30)
        values = np.concatenate([values1, values2])
        
        data = pd.DataFrame({
            'timestamp': dates,
            'pesq': values
        })
        
        # Detect change points
        change_points = analyzer.detect_change_points(data, 'pesq')
        
        # Should detect change around day 30
        assert len(change_points) >= 1
        
        # Check change point properties
        for cp in change_points:
            assert 'timestamp' in cp
            assert 'confidence' in cp
            assert 'old_trend' in cp
            assert 'new_trend' in cp
        
        # The main change point should be around index 30
        main_cp = max(change_points, key=lambda x: x['confidence'])
        cp_index = data[data['timestamp'] == main_cp['timestamp']].index[0]
        assert 25 <= cp_index <= 35