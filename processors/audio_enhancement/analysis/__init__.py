"""
Audio Enhancement Analysis Module
Time series analysis, trend detection, and predictive analytics
"""

from .trend_analyzer import (
    TrendAnalyzer,
    TrendDirection,
    TrendPattern,
    Anomaly,
    AlertManager,
    Prediction
)

__all__ = [
    'TrendAnalyzer',
    'TrendDirection',
    'TrendPattern',
    'Anomaly',
    'AlertManager',
    'Prediction'
]