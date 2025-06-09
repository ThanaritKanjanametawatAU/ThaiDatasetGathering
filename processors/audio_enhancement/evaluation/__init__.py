"""
Audio enhancement evaluation module.

This module provides comprehensive tools for evaluating and comparing
different audio enhancement strategies, algorithms, and parameters.
"""

from .comparison_framework import (
    ComparisonFramework,
    StatisticalAnalyzer,
    VisualizationEngine,
    ComparisonMethod,
    ComparisonReport,
    ComparisonError,
    MetricConfig,
    ComparisonConfig
)

__all__ = [
    'ComparisonFramework',
    'StatisticalAnalyzer',
    'VisualizationEngine',
    'ComparisonMethod',
    'ComparisonReport',
    'ComparisonError',
    'MetricConfig',
    'ComparisonConfig'
]