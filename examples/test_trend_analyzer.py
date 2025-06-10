#!/usr/bin/env python3
"""
Example script demonstrating Trend Analysis Module usage
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.audio_enhancement.analysis import TrendAnalyzer


def generate_sample_data(days=60):
    """Generate sample quality metrics data"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Create realistic patterns
    # PESQ: Gradual improvement with weekly variations
    trend = np.linspace(3.0, 3.4, days)
    weekly_pattern = 0.1 * np.sin(2 * np.pi * np.arange(days) / 7)
    noise = np.random.normal(0, 0.05, days)
    pesq = trend + weekly_pattern + noise
    
    # STOI: Stable with occasional dips
    stoi = 0.85 + np.random.normal(0, 0.02, days)
    stoi[30:33] = 0.75  # Quality dip
    stoi[50:52] = 0.78  # Another dip
    
    # SI-SDR: Random variations
    si_sdr = 15 + np.random.normal(0, 2, days)
    
    # SNR: Gradual improvement
    snr = 22 + np.linspace(0, 3, days) + np.random.normal(0, 1, days)
    
    return pd.DataFrame({
        'timestamp': dates,
        'pesq': pesq,
        'stoi': stoi,
        'si_sdr': si_sdr,
        'snr': snr
    })


def demonstrate_trend_analysis():
    """Demonstrate trend analysis capabilities"""
    print("=== Trend Analysis Demo ===\n")
    
    # Generate sample data
    data = generate_sample_data(60)
    print(f"Generated {len(data)} days of quality metrics data")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Initialize analyzer
    analyzer = TrendAnalyzer(window_size=7)
    
    # Analyze trends
    print("\n--- Analyzing Trends ---")
    trends = analyzer.analyze_trends(data)
    
    for metric, trend_info in trends.items():
        print(f"\n{metric.upper()}:")
        print(f"  Direction: {trend_info['direction'].value}")
        print(f"  Slope: {trend_info['slope']:.4f} per day")
        print(f"  R-squared: {trend_info['r_squared']:.3f}")
        print(f"  95% CI: [{trend_info['confidence_interval']['lower']:.4f}, "
              f"{trend_info['confidence_interval']['upper']:.4f}]")
        
        if 'seasonal_pattern' in trend_info:
            print(f"  Seasonal: Period={trend_info['seasonal_pattern']['period']} days, "
                  f"Strength={trend_info['seasonal_pattern']['strength']:.2f}")


def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection"""
    print("\n\n=== Anomaly Detection Demo ===\n")
    
    # Generate data with anomalies
    data = generate_sample_data(30)
    
    # Inject anomalies
    data.loc[10, 'pesq'] = 2.0  # Sudden drop in PESQ
    data.loc[20, 'stoi'] = 0.5   # Major STOI issue
    data.loc[25, 'snr'] = 10.0   # Low SNR
    
    # Initialize analyzer
    analyzer = TrendAnalyzer()
    
    # Detect anomalies with different sensitivities
    for sensitivity in ['low', 'medium', 'high']:
        anomalies = analyzer.detect_anomalies(data, sensitivity=sensitivity)
        print(f"\n{sensitivity.upper()} sensitivity: {len(anomalies)} anomalies detected")
        
        if sensitivity == 'medium' and anomalies:
            # Show details for medium sensitivity
            for i, anomaly in enumerate(anomalies[:3]):
                print(f"\nAnomaly {i+1}:")
                print(f"  Metric: {anomaly.metric}")
                print(f"  Timestamp: {anomaly.timestamp}")
                print(f"  Expected: {anomaly.expected_value:.3f}")
                print(f"  Actual: {anomaly.actual_value:.3f}")
                print(f"  Deviation: {anomaly.deviation_percentage:.1f}%")
                print(f"  Severity: {anomaly.severity}")


def demonstrate_predictions():
    """Demonstrate quality predictions"""
    print("\n\n=== Quality Prediction Demo ===\n")
    
    # Generate historical data
    historical_data = generate_sample_data(90)
    
    # Initialize and fit analyzer
    analyzer = TrendAnalyzer()
    analyzer.fit(historical_data)
    
    # Make predictions
    predictions = analyzer.predict_quality(horizon=14)  # 2 weeks ahead
    
    print("14-day quality predictions:")
    for metric in ['pesq', 'stoi', 'si_sdr', 'snr']:
        if metric in predictions:
            pred = predictions[metric]
            mean_pred = np.mean(pred['mean'])
            std_pred = np.mean(pred['std'])
            
            print(f"\n{metric.upper()}:")
            print(f"  Current: {historical_data[metric].iloc[-1]:.3f}")
            print(f"  Predicted (mean): {mean_pred:.3f} ± {std_pred:.3f}")
            print(f"  7-day forecast: {pred['mean'][6]:.3f}")
            print(f"  14-day forecast: {pred['mean'][-1]:.3f}")


def demonstrate_alerts():
    """Demonstrate alert system"""
    print("\n\n=== Alert System Demo ===\n")
    
    # Generate data with issues
    data = generate_sample_data(30)
    
    # Create declining PESQ trend
    data['pesq'] = np.linspace(3.2, 2.8, 30) + np.random.normal(0, 0.02, 30)
    
    # Add consistent low STOI
    data.loc[20:, 'stoi'] = 0.68
    
    # Initialize analyzer and set alerts
    analyzer = TrendAnalyzer()
    
    # Configure alerts
    analyzer.set_alert('quality_decline', metric='pesq', 
                      condition='declining', threshold=0.01)
    analyzer.set_alert('low_intelligibility', metric='stoi', 
                      condition='below_threshold', threshold=0.75)
    analyzer.set_alert('detect_anomalies', metric='all', 
                      condition='anomaly', sensitivity='medium')
    
    # Process alerts
    alerts = analyzer.process_alerts(data)
    
    print(f"Total alerts triggered: {len(alerts)}")
    for alert in alerts:
        print(f"\n[{alert['severity'].upper()}] {alert['id']}:")
        print(f"  {alert['message']}")
        print(f"  Metric: {alert['metric']}, Value: {alert['value']:.3f}")


def demonstrate_seasonality():
    """Demonstrate seasonal analysis"""
    print("\n\n=== Seasonality Analysis Demo ===\n")
    
    # Generate data with clear weekly pattern
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    t = np.arange(90)
    
    # Weekly pattern in PESQ
    pesq = 3.2 + 0.2 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 0.02, 90)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'pesq': pesq
    })
    
    # Analyze seasonality
    analyzer = TrendAnalyzer()
    seasonal_analysis = analyzer.analyze_seasonality(data, 'pesq')
    
    print("Seasonal Analysis Results:")
    print(f"  Has seasonality: {seasonal_analysis['has_seasonality']}")
    if seasonal_analysis['has_seasonality']:
        print(f"  Period: {seasonal_analysis['period']} days")
        print(f"  Strength: {seasonal_analysis['strength']:.2f}")
        print(f"  Pattern: Weekly variations detected")


def demonstrate_change_detection():
    """Demonstrate change point detection"""
    print("\n\n=== Change Point Detection Demo ===\n")
    
    # Generate data with trend change
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    
    # First 30 days: improving trend
    # Next 30 days: declining trend
    values1 = np.linspace(3.0, 3.5, 30) + np.random.normal(0, 0.02, 30)
    values2 = np.linspace(3.5, 3.0, 30) + np.random.normal(0, 0.02, 30)
    pesq = np.concatenate([values1, values2])
    
    data = pd.DataFrame({
        'timestamp': dates,
        'pesq': pesq
    })
    
    # Detect change points
    analyzer = TrendAnalyzer()
    change_points = analyzer.detect_change_points(data, 'pesq')
    
    print(f"Detected {len(change_points)} change points")
    for i, cp in enumerate(change_points):
        print(f"\nChange Point {i+1}:")
        print(f"  Date: {cp['timestamp']}")
        print(f"  Old trend: {cp['old_trend'].value}")
        print(f"  New trend: {cp['new_trend'].value}")
        print(f"  Confidence: {cp['confidence']:.2f}")


def main():
    """Run all demonstrations"""
    print("Audio Enhancement Trend Analysis Demo")
    print("=" * 50)
    
    demonstrate_trend_analysis()
    demonstrate_anomaly_detection()
    demonstrate_predictions()
    demonstrate_alerts()
    demonstrate_seasonality()
    demonstrate_change_detection()
    
    print("\n\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()