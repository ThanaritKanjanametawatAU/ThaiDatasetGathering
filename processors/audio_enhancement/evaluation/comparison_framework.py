"""
Comprehensive Comparison Framework for Audio Enhancement Evaluation.

This module provides a complete framework for comparing different audio enhancement
methods, including statistical analysis, visualization, and report generation.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
import matplotlib

# Optional import for statsmodels
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    # Simple Bonferroni correction fallback
    def multipletests(pvals, alpha=0.05, method='bonferroni'):
        n_tests = len(pvals)
        if method == 'bonferroni':
            adjusted_pvals = np.minimum(np.array(pvals) * n_tests, 1.0)
            rejected = adjusted_pvals < alpha
        else:
            # For other methods, just use original p-values
            adjusted_pvals = np.array(pvals)
            rejected = adjusted_pvals < alpha
        return rejected, adjusted_pvals, None, None
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import psutil
import os

# Optional imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Import metric calculators
from ..metrics import (
    PESQCalculator, STOICalculator, SISDRCalculator,
    PESQError, STOIError, SISDRError
)
from utils.audio_metrics import calculate_snr

logger = logging.getLogger(__name__)


class ComparisonError(Exception):
    """Base exception for comparison framework errors."""
    pass


@dataclass
class MetricConfig:
    """Configuration for a specific metric."""
    name: str
    calculator_class: type
    higher_is_better: bool = True
    display_name: Optional[str] = None
    unit: Optional[str] = None
    range: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name.upper()


@dataclass
class ComparisonConfig:
    """Configuration for the comparison framework."""
    metrics: List[str] = field(default_factory=lambda: ['pesq', 'stoi', 'si_sdr'])
    statistical_tests: bool = True
    confidence_level: float = 0.95
    multiple_comparison_correction: str = 'bonferroni'
    bootstrap_iterations: int = 1000
    visualization_options: Dict[str, Any] = field(default_factory=dict)
    report_options: Dict[str, Any] = field(default_factory=dict)
    parallel_processing: bool = True
    n_workers: Optional[int] = None


@dataclass
class ComparisonMethod:
    """Container for a method to be compared."""
    name: str
    audio_data: np.ndarray
    reference: Optional[np.ndarray] = None
    sample_rate: int = 16000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Container for comparison results."""
    methods: List[str]
    metrics: List[str]
    results: Dict[str, Any]
    statistical_analysis: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, str]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[ComparisonConfig] = None


class StatisticalAnalyzer:
    """Statistical analysis engine for comparison framework."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
    
    def paired_t_test(self, scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, Any]:
        """
        Perform paired statistical test on two sets of scores.
        
        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            
        Returns:
            Dictionary containing test results
        """
        # Check for equal length
        if len(scores_a) != len(scores_b):
            raise ValueError("Score arrays must have equal length")
        
        # Calculate differences
        differences = scores_a - scores_b
        
        # Test for normality
        try:
            _, normality_p = stats.shapiro(differences)
            is_normal = normality_p >= 0.05
        except:
            is_normal = False
        
        if is_normal and len(differences) >= 30:
            # Use paired t-test for normal data
            statistic, p_value = stats.ttest_rel(scores_a, scores_b)
            test_name = 'Paired t-test'
        else:
            # Use Wilcoxon signed-rank test for non-normal data
            try:
                statistic, p_value = stats.wilcoxon(scores_a, scores_b)
                test_name = 'Wilcoxon signed-rank'
            except:
                # Fallback to t-test if Wilcoxon fails
                statistic, p_value = stats.ttest_rel(scores_a, scores_b)
                test_name = 'Paired t-test (fallback)'
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(differences)
        # Use pooled standard deviation
        std_a = np.std(scores_a, ddof=1)
        std_b = np.std(scores_b, ddof=1)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        
        if pooled_std > 0:
            cohens_d = mean_diff / pooled_std
        else:
            cohens_d = 0.0
        
        return {
            'test_name': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < self.alpha),
            'cohens_d': float(cohens_d),
            'effect_size': self._interpret_cohens_d(cohens_d),
            'mean_difference': float(mean_diff),
            'ci_95': self._calculate_confidence_interval(differences)
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data."""
        mean = np.mean(data)
        se = stats.sem(data)
        margin = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return (float(mean - margin), float(mean + margin))
    
    def apply_multiple_comparison_correction(self, p_values: List[float], 
                                           method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply correction for multiple comparisons.
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'holm', 'fdr_bh', 'fdr_by')
            
        Returns:
            Dictionary with correction results
        """
        valid_methods = ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']
        if method not in valid_methods:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Apply correction
        rejected, p_adjusted, _, _ = multipletests(
            p_values,
            alpha=self.alpha,
            method=method
        )
        
        return {
            'rejected': rejected.tolist(),
            'p_adjusted': p_adjusted.tolist(),
            'method': method,
            'n_significant': int(np.sum(rejected))
        }
    
    def bootstrap_confidence_interval(self, scores: np.ndarray, metric_fn: Callable,
                                    n_bootstrap: int = 1000, ci: int = 95) -> Dict[str, Any]:
        """
        Compute bootstrap confidence intervals.
        
        Args:
            scores: Array of scores
            metric_fn: Function to compute metric
            n_bootstrap: Number of bootstrap iterations
            ci: Confidence level (as percentage)
            
        Returns:
            Dictionary with bootstrap results
        """
        n_samples = len(scores)
        bootstrap_results = []
        
        # Perform bootstrap
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            resampled = scores[indices]
            
            # Compute metric on resampled data
            result = metric_fn(resampled)
            bootstrap_results.append(result)
        
        # Compute confidence interval
        alpha = (100 - ci) / 2
        lower = np.percentile(bootstrap_results, alpha)
        upper = np.percentile(bootstrap_results, 100 - alpha)
        
        return {
            'mean': float(np.mean(bootstrap_results)),
            'std': float(np.std(bootstrap_results)),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'ci_level': ci
        }
    
    def test_normality(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        Test if data follows normal distribution.
        
        Args:
            data: Array of values to test
            
        Returns:
            Tuple of (is_normal, p_value)
        """
        try:
            # Shapiro test requires at least 3 samples
            if len(data) < 3:
                return False, 0.0
            
            # Check for zero variance
            if np.var(data) < 1e-10:
                return False, 0.0
            
            _, p_value = stats.shapiro(data)
            return bool(p_value >= 0.05), float(p_value)
        except Exception as e:
            logger.debug(f"Normality test failed: {e}")
            return False, 0.0
    
    def calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        
        if pooled_std > 0:
            return (mean1 - mean2) / pooled_std
        return 0.0
    
    def calculate_glass_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Glass's delta effect size."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1 = np.std(group1, ddof=1)
        
        if std1 > 0:
            return (mean1 - mean2) / std1
        return 0.0
    
    def calculate_hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size."""
        n1, n2 = len(group1), len(group2)
        cohens_d = self.calculate_cohens_d(group1, group2)
        
        # Correction factor for small samples
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        return cohens_d * correction
    
    def calculate_required_sample_size(self, effect_size: float, alpha: float = 0.05,
                                     power: float = 0.8) -> int:
        """
        Calculate required sample size for given effect size and power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Desired statistical power
            
        Returns:
            Required sample size per group
        """
        # Using approximation formula for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
    
    def calculate_achieved_power(self, n: int, effect_size: float, alpha: float = 0.05) -> float:
        """
        Calculate achieved statistical power.
        
        Args:
            n: Sample size per group
            effect_size: Effect size (Cohen's d)
            alpha: Significance level
            
        Returns:
            Achieved power (0-1)
        """
        # Using approximation for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z = effect_size * np.sqrt(n/2) - z_alpha
        power = stats.norm.cdf(z)
        
        return float(power)


class VisualizationEngine:
    """Visualization engine for comparison framework."""
    
    def __init__(self):
        """Initialize visualization engine."""
        self.color_schemes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'colorblind': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
                          '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#208F8F']
        }
        self.current_scheme = 'default'
    
    def get_color_scheme(self, scheme_name: str = 'default') -> List[str]:
        """Get color scheme by name."""
        return self.color_schemes.get(scheme_name, self.color_schemes['default'])
    
    def create_radar_chart(self, methods: List[str], metrics: List[str],
                          scores: Dict[str, List[float]], output_path: str) -> None:
        """
        Create radar chart for multi-metric comparison.
        
        Args:
            methods: List of method names
            metrics: List of metric names
            scores: Dictionary mapping method names to score lists
            output_path: Path to save the chart
        """
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        colors = self.get_color_scheme(self.current_scheme)
        
        # Normalize scores to 0-1 range for visualization
        all_scores = []
        for method_scores in scores.values():
            all_scores.extend(method_scores)
        
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        for i, (method, method_scores) in enumerate(scores.items()):
            # Normalize scores
            normalized = [(s - min_score) / score_range for s in method_scores]
            normalized += normalized[:1]  # Complete the circle
            angles_plot = angles + angles[:1]
            
            ax.plot(angles_plot, normalized, 'o-', linewidth=2, 
                   label=method, color=colors[i % len(colors)])
            ax.fill(angles_plot, normalized, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        plt.title('Multi-Metric Performance Comparison', size=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_box_plot(self, data: Dict[str, np.ndarray], title: str, output_path: str) -> None:
        """
        Create box plot for score distributions.
        
        Args:
            data: Dictionary mapping method names to score arrays
            title: Plot title
            output_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        plot_data = []
        labels = []
        for method, scores in data.items():
            plot_data.append(scores)
            labels.append(method)
        
        # Create box plot
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Customize colors
        colors = self.get_color_scheme(self.current_scheme)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_significance_heatmap(self, methods: List[str], p_values_matrix: np.ndarray,
                                   output_path: str) -> None:
        """
        Create heatmap showing statistical significance between methods.
        
        Args:
            methods: List of method names
            p_values_matrix: Matrix of p-values
            output_path: Path to save the heatmap
        """
        n_methods = len(methods)
        
        # Create significance matrix and annotations
        sig_matrix = np.zeros((n_methods, n_methods))
        annotations = []
        
        for i in range(n_methods):
            row_annotations = []
            for j in range(n_methods):
                if i == j:
                    row_annotations.append('-')
                    sig_matrix[i, j] = -1  # Special value for diagonal
                else:
                    p_val = p_values_matrix[i, j]
                    if p_val < 0.001:
                        sig_matrix[i, j] = 3
                        row_annotations.append('***')
                    elif p_val < 0.01:
                        sig_matrix[i, j] = 2
                        row_annotations.append('**')
                    elif p_val < 0.05:
                        sig_matrix[i, j] = 1
                        row_annotations.append('*')
                    else:
                        sig_matrix[i, j] = 0
                        row_annotations.append('ns')
            annotations.append(row_annotations)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Custom colormap
        colors_list = ['#f0f0f0', '#ffd9b3', '#ffb366', '#ff8c1a', '#ffffff']
        n_colors = len(colors_list)
        cmap = matplotlib.colors.ListedColormap(colors_list)
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        sns.heatmap(
            sig_matrix,
            annot=annotations,
            fmt='',
            cmap=cmap,
            norm=norm,
            xticklabels=methods,
            yticklabels=methods,
            cbar_kws={'label': 'Significance Level', 'ticks': [-1, 0, 1, 2, 3]},
            square=True,
            linewidths=0.5
        )
        
        # Customize colorbar labels
        cbar = plt.gca().collections[0].colorbar
        cbar.set_ticklabels(['N/A', 'ns', 'p<0.05', 'p<0.01', 'p<0.001'])
        
        plt.title('Statistical Significance Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metric_evolution_plot(self, data: Dict[str, np.ndarray], time_points: List[int],
                                   metric_name: str, output_path: str) -> None:
        """
        Create plot showing metric evolution over time.
        
        Args:
            data: Dictionary mapping method names to metric values over time
            time_points: List of time points
            metric_name: Name of the metric being plotted
            output_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = self.get_color_scheme(self.current_scheme)
        
        for i, (method, values) in enumerate(data.items()):
            ax.plot(time_points, values, 'o-', label=method, 
                   color=colors[i % len(colors)], linewidth=2, markersize=6)
        
        ax.set_xlabel('Time Point')
        ax.set_ylabel(f'{metric_name} Score')
        ax.set_title(f'{metric_name} Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_comparison(self, data: Dict[str, np.ndarray], output_path: str,
                                    plot_type: str = 'violin') -> None:
        """
        Create interactive comparison plot using Plotly.
        
        Args:
            data: Dictionary mapping method names to score arrays
            output_path: Path to save the HTML file
            plot_type: Type of plot ('violin', 'box', 'strip')
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, skipping interactive plot")
            return
        
        # Prepare data for Plotly
        df_data = []
        for method, scores in data.items():
            for score in scores:
                df_data.append({'Method': method, 'Score': score})
        
        df = pd.DataFrame(df_data)
        
        # Create plot based on type
        if plot_type == 'violin':
            fig = px.violin(df, x='Method', y='Score', box=True, points="all")
        elif plot_type == 'box':
            fig = px.box(df, x='Method', y='Score', points="all")
        else:  # strip
            fig = px.strip(df, x='Method', y='Score')
        
        fig.update_layout(
            title='Interactive Score Comparison',
            xaxis_title='Method',
            yaxis_title='Score',
            template='plotly_white'
        )
        
        fig.write_html(output_path)
    
    def create_multi_panel_comparison(self, data: Dict[str, Dict[str, np.ndarray]],
                                    output_path: str, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create multi-panel figure for comparing multiple metrics.
        
        Args:
            data: Nested dict {metric: {method: scores}}
            output_path: Path to save the figure
            figsize: Figure size
        """
        n_metrics = len(data)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = self.get_color_scheme(self.current_scheme)
        
        for idx, (metric, method_data) in enumerate(data.items()):
            ax = axes[idx]
            
            # Prepare data for plotting
            plot_data = []
            labels = []
            for method, scores in method_data.items():
                plot_data.append(scores)
                labels.append(method)
            
            # Create violin plot
            parts = ax.violinplot(plot_data, positions=range(len(labels)), 
                                 showmeans=True, showextrema=True)
            
            # Customize colors
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Metric Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class ComparisonFramework:
    """Main comparison framework for audio enhancement evaluation."""
    
    def __init__(self, metrics: Optional[List[str]] = None, config: Optional[ComparisonConfig] = None,
                 streaming_mode: bool = False):
        """
        Initialize comparison framework.
        
        Args:
            metrics: List of metrics to use
            config: Configuration object
            streaming_mode: Enable streaming comparison mode
        """
        if config is not None:
            self.config = config
            self.metrics = config.metrics
        else:
            self.metrics = metrics or ['pesq', 'stoi', 'si_sdr']
            self.config = ComparisonConfig(metrics=self.metrics)
        
        # Validate metrics
        self._validate_metrics()
        
        # Initialize components
        self.statistical_analyzer = StatisticalAnalyzer(alpha=1 - self.config.confidence_level)
        self.visualization_engine = VisualizationEngine()
        
        # Storage
        self.methods: Dict[str, ComparisonMethod] = {}
        self.results: Dict[str, Any] = {}
        self.streaming_mode = streaming_mode
        
        if streaming_mode:
            self.streaming_buffer: Dict[str, List[float]] = {}
            self.streaming_window_size = 100
        
        # Metric calculators
        self._metric_calculators: Dict[str, Any] = {}
        self._initialize_metric_calculators()
        
        # Performance tracking
        self._computation_times: Dict[str, float] = {}
    
    def _validate_metrics(self) -> None:
        """Validate that all metrics are supported."""
        supported_metrics = ['pesq', 'stoi', 'si_sdr', 'snr']
        
        for metric in self.metrics:
            if metric not in supported_metrics and not metric.startswith('custom_'):
                raise ComparisonError(f"Unknown metric: {metric}. Supported metrics: {supported_metrics}")
    
    def _initialize_metric_calculators(self) -> None:
        """Initialize metric calculator instances."""
        if 'pesq' in self.metrics:
            self._metric_calculators['pesq'] = PESQCalculator()
        
        if 'stoi' in self.metrics:
            self._metric_calculators['stoi'] = STOICalculator()
        
        if 'si_sdr' in self.metrics:
            self._metric_calculators['si_sdr'] = SISDRCalculator()
    
    def _get_metric_calculator(self, metric: str) -> Any:
        """Get metric calculator for given metric."""
        if metric == 'snr':
            # SNR uses a function, not a class
            return None
        
        return self._metric_calculators.get(metric)
    
    def add_method(self, name: str, audio_data: np.ndarray, reference: Optional[np.ndarray] = None,
                   sample_rate: int = 16000, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a method for comparison.
        
        Args:
            name: Method name
            audio_data: Audio data array
            reference: Reference audio (optional)
            sample_rate: Sample rate
            metadata: Additional metadata
        """
        if name in self.methods:
            raise ComparisonError(f"Method '{name}' already exists")
        
        # Validate audio lengths if reference provided
        if reference is not None and len(audio_data) != len(reference):
            raise ComparisonError("Audio length mismatch between method and reference")
        
        method = ComparisonMethod(
            name=name,
            audio_data=audio_data,
            reference=reference,
            sample_rate=sample_rate,
            metadata=metadata or {}
        )
        
        self.methods[name] = method
    
    def compare(self, methods: Optional[List[str]] = None, statistical_tests: bool = None) -> Dict[str, Any]:
        """
        Perform comparison of methods.
        
        Args:
            methods: List of method names to compare (None for all)
            statistical_tests: Override config setting for statistical tests
            
        Returns:
            Dictionary containing comparison results
        """
        if not self.methods:
            raise ComparisonError("No methods added for comparison")
        
        if methods is None:
            methods = list(self.methods.keys())
        
        if statistical_tests is None:
            statistical_tests = self.config.statistical_tests
        
        # Compute metrics for all methods
        start_time = time.time()
        results = self._compute_metrics(methods)
        self._computation_times['metrics'] = time.time() - start_time
        
        # Rank methods
        results = self._rank_methods(results)
        
        # Perform statistical analysis if requested
        if statistical_tests and len(methods) > 1:
            start_time = time.time()
            statistical_results = self._perform_statistical_analysis(methods, results)
            results['statistical_analysis'] = statistical_results
            self._computation_times['statistics'] = time.time() - start_time
        
        self.results = results
        return results
    
    def _compute_metrics(self, methods: List[str]) -> Dict[str, Any]:
        """Compute all metrics for specified methods."""
        results = {}
        
        # Use parallel processing if enabled
        if self.config.parallel_processing and len(methods) > 1:
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = {}
                for method_name in methods:
                    future = executor.submit(self._compute_method_metrics, method_name)
                    futures[method_name] = future
                
                for method_name, future in futures.items():
                    results[method_name] = future.result()
        else:
            for method_name in methods:
                results[method_name] = self._compute_method_metrics(method_name)
        
        return results
    
    def _compute_method_metrics(self, method_name: str) -> Dict[str, Any]:
        """Compute metrics for a single method."""
        method = self.methods[method_name]
        method_results = {
            'metrics': {},
            'metadata': method.metadata
        }
        
        for metric in self.metrics:
            try:
                score = self._compute_single_metric(
                    metric, method.audio_data, method.reference, method.sample_rate
                )
                method_results['metrics'][metric] = score
            except Exception as e:
                logger.warning(f"Failed to compute {metric} for {method_name}: {str(e)}")
                method_results['metrics'][metric] = None
        
        return method_results
    
    def _compute_single_metric(self, metric: str, audio: np.ndarray, 
                              reference: Optional[np.ndarray], sample_rate: int) -> float:
        """Compute a single metric value."""
        if metric == 'snr':
            if reference is None:
                raise ValueError("SNR requires reference audio")
            return calculate_snr(reference, audio)
        
        calculator = self._get_metric_calculator(metric)
        if calculator is None:
            raise ValueError(f"No calculator for metric: {metric}")
        
        if metric == 'pesq':
            if reference is None:
                raise ValueError("PESQ requires reference audio")
            result = calculator.calculate(reference, audio, sample_rate)
            # Handle both object with .overall_score attribute and direct float return
            if hasattr(result, 'overall_score'):
                return result.overall_score
            else:
                return float(result)
        
        elif metric == 'stoi':
            if reference is None:
                raise ValueError("STOI requires reference audio")
            result = calculator.calculate(reference, audio, sample_rate)
            # Handle both object with .overall_score attribute and direct float return  
            if hasattr(result, 'overall_score'):
                return result.overall_score
            else:
                return float(result)
        
        elif metric == 'si_sdr':
            if reference is None:
                raise ValueError("SI-SDR requires reference audio")
            result = calculator.calculate(reference, audio)
            # Handle both object with .sdr attribute and direct float return
            if hasattr(result, 'sdr'):
                return result.sdr
            else:
                return float(result)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _rank_methods(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank methods based on metric scores."""
        # Calculate average normalized scores
        method_scores = {}
        
        for method, method_results in results.items():
            if method == 'statistical_analysis':
                continue
            
            scores = []
            for metric, value in method_results['metrics'].items():
                if value is not None:
                    scores.append(value)
            
            if scores:
                method_scores[method] = np.mean(scores)
            else:
                method_scores[method] = 0.0
        
        # Sort methods by score
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Assign ranks
        for rank, (method, _) in enumerate(sorted_methods, 1):
            results[method]['rank'] = rank
            results[method]['overall_score'] = method_scores[method]
        
        return results
    
    def _perform_statistical_analysis(self, methods: List[str], results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        n_methods = len(methods)
        statistical_results = {
            'pairwise_comparisons': {},
            'effect_sizes': {},
            'confidence_intervals': {}
        }
        
        # Collect all scores for each method
        method_scores = {}
        for method in methods:
            scores = []
            for metric in self.metrics:
                value = results[method]['metrics'].get(metric)
                if value is not None:
                    scores.append(value)
            method_scores[method] = np.array(scores)
        
        # Pairwise comparisons
        p_values = []
        for i, method_a in enumerate(methods):
            for j, method_b in enumerate(methods):
                if i < j:  # Only compare each pair once
                    comparison_key = f"{method_a}_vs_{method_b}"
                    
                    if len(method_scores[method_a]) > 0 and len(method_scores[method_b]) > 0:
                        comparison_result = self.statistical_analyzer.paired_t_test(
                            method_scores[method_a],
                            method_scores[method_b]
                        )
                        statistical_results['pairwise_comparisons'][comparison_key] = comparison_result
                        p_values.append(comparison_result['p_value'])
        
        # Apply multiple comparison correction if needed
        if len(p_values) > 1 and self.config.multiple_comparison_correction:
            correction_result = self.statistical_analyzer.apply_multiple_comparison_correction(
                p_values,
                method=self.config.multiple_comparison_correction
            )
            statistical_results['multiple_comparison_correction'] = correction_result
        
        # Bootstrap confidence intervals
        for method in methods:
            if len(method_scores[method]) > 0:
                ci_result = self.statistical_analyzer.bootstrap_confidence_interval(
                    method_scores[method],
                    metric_fn=np.mean,
                    n_bootstrap=self.config.bootstrap_iterations
                )
                statistical_results['confidence_intervals'][method] = ci_result
        
        return statistical_results
    
    def get_best_method(self, metric: str = None) -> Dict[str, Any]:
        """
        Get the best performing method.
        
        Args:
            metric: Specific metric to use (None for overall)
            
        Returns:
            Dictionary with method name and score
        """
        if not self.results:
            raise ComparisonError("No comparison results available. Run compare() first.")
        
        best_method = None
        best_score = -float('inf')
        
        for method, method_results in self.results.items():
            if method == 'statistical_analysis':
                continue
            
            if metric:
                score = method_results['metrics'].get(metric)
            else:
                score = method_results.get('overall_score', 0)
            
            if score is not None and score > best_score:
                best_score = score
                best_method = method
        
        if best_method:
            return {
                'name': best_method,
                'score': best_score,
                'metric': metric or 'overall'
            }
        
        return None
    
    def generate_report(self, output_path: str, format: str = 'pdf') -> None:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Path to save the report
            format: Report format ('pdf', 'html', 'json')
        """
        if not self.results:
            raise ComparisonError("No comparison results available. Run compare() first.")
        
        output_path = Path(output_path)
        
        if format == 'json':
            self._generate_json_report(output_path)
        elif format == 'html':
            self._generate_html_report(output_path)
        elif format == 'pdf':
            self._generate_pdf_report(output_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_json_report(self, output_path: Path) -> None:
        """Generate JSON report."""
        report_data = {
            'methods': list(self.methods.keys()),
            'metrics': self.metrics,
            'results': self.results,
            'statistical_analysis': self.results.get('statistical_analysis', {}),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'confidence_level': self.config.confidence_level,
                'multiple_comparison_correction': self.config.multiple_comparison_correction,
                'bootstrap_iterations': self.config.bootstrap_iterations
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _generate_html_report(self, output_path: Path) -> None:
        """Generate HTML report."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Audio Enhancement Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .significant {{ background-color: #ffffcc; }}
                .best {{ font-weight: bold; color: #4CAF50; }}
                h1, h2, h3 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Audio Enhancement Comparison Report</h1>
            <p>Generated: {timestamp}</p>
            
            <h2>Method Scores</h2>
            <table>
                <tr>
                    <th>Method</th>
                    {metric_headers}
                    <th>Overall Score</th>
                    <th>Rank</th>
                </tr>
                {method_rows}
            </table>
            
            {statistical_section}
            
            <h2>Configuration</h2>
            <ul>
                <li>Confidence Level: {confidence_level}</li>
                <li>Multiple Comparison Correction: {correction_method}</li>
                <li>Bootstrap Iterations: {bootstrap_iterations}</li>
            </ul>
        </body>
        </html>
        """
        
        # Build metric headers
        metric_headers = ''.join(f'<th>{metric.upper()}</th>' for metric in self.metrics)
        
        # Build method rows
        method_rows = []
        for method, results in self.results.items():
            if method == 'statistical_analysis':
                continue
            
            row = f'<tr><td>{method}</td>'
            for metric in self.metrics:
                score = results['metrics'].get(metric, 'N/A')
                if isinstance(score, float):
                    score = f'{score:.3f}'
                row += f'<td>{score}</td>'
            
            overall = results.get('overall_score', 0)
            rank = results.get('rank', 'N/A')
            row += f'<td>{overall:.3f}</td><td>{rank}</td></tr>'
            method_rows.append(row)
        
        # Build statistical section
        statistical_section = ''
        if 'statistical_analysis' in self.results:
            statistical_section = '<h2>Statistical Analysis</h2>'
            stats = self.results['statistical_analysis']
            
            if 'pairwise_comparisons' in stats:
                statistical_section += '<h3>Pairwise Comparisons</h3><ul>'
                for comparison, result in stats['pairwise_comparisons'].items():
                    sig_class = 'significant' if result['significant'] else ''
                    statistical_section += f'''
                    <li class="{sig_class}">
                        {comparison}: p={result["p_value"]:.4f}, 
                        effect size={result["effect_size"]} (d={result["cohens_d"]:.3f})
                    </li>
                    '''
                statistical_section += '</ul>'
        
        # Format the HTML
        html_final = html_content.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            metric_headers=metric_headers,
            method_rows='\n'.join(method_rows),
            statistical_section=statistical_section,
            confidence_level=self.config.confidence_level,
            correction_method=self.config.multiple_comparison_correction,
            bootstrap_iterations=self.config.bootstrap_iterations
        )
        
        with open(output_path, 'w') as f:
            f.write(html_final)
    
    def _generate_pdf_report(self, output_path: Path) -> None:
        """Generate PDF report."""
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available, falling back to HTML report")
            self._generate_html_report(output_path.with_suffix('.html'))
            return
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#333333'),
            spaceAfter=30
        )
        story.append(Paragraph('Audio Enhancement Comparison Report', title_style))
        story.append(Spacer(1, 12))
        
        # Timestamp
        story.append(Paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Method scores table
        story.append(Paragraph('Method Scores', styles['Heading2']))
        
        # Prepare table data
        table_data = [['Method'] + [m.upper() for m in self.metrics] + ['Overall', 'Rank']]
        
        for method, results in self.results.items():
            if method == 'statistical_analysis':
                continue
            
            row = [method]
            for metric in self.metrics:
                score = results['metrics'].get(metric, 'N/A')
                if isinstance(score, float):
                    score = f'{score:.3f}'
                row.append(score)
            
            overall = results.get('overall_score', 0)
            rank = results.get('rank', 'N/A')
            row.extend([f'{overall:.3f}', str(rank)])
            table_data.append(row)
        
        # Create table
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Statistical analysis section
        if 'statistical_analysis' in self.results:
            story.append(Paragraph('Statistical Analysis', styles['Heading2']))
            stats = self.results['statistical_analysis']
            
            if 'pairwise_comparisons' in stats:
                story.append(Paragraph('Pairwise Comparisons:', styles['Heading3']))
                for comparison, result in stats['pairwise_comparisons'].items():
                    text = f"{comparison}: p={result['p_value']:.4f}, "
                    text += f"effect size={result['effect_size']} (d={result['cohens_d']:.3f})"
                    if result['significant']:
                        text = f"<b>{text}</b>"
                    story.append(Paragraph(text, styles['Normal']))
                story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
    
    def compare_batch(self, batch_data: List[Dict[str, Any]], sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Perform batch comparison on multiple samples.
        
        Args:
            batch_data: List of sample dictionaries
            sample_rate: Sample rate for all audio
            
        Returns:
            Batch comparison results
        """
        sample_results = []
        aggregate_scores = {method: {metric: [] for metric in self.metrics} 
                          for method in batch_data[0]['methods'].keys()}
        
        # Process each sample
        for sample in batch_data:
            # Clear previous methods
            self.methods.clear()
            
            # Add methods for this sample
            for method_name, audio in sample['methods'].items():
                self.add_method(
                    method_name,
                    audio,
                    reference=sample.get('reference'),
                    sample_rate=sample_rate
                )
            
            # Compare
            results = self.compare(statistical_tests=False)
            sample_results.append({
                'id': sample['id'],
                'results': results
            })
            
            # Collect scores for aggregation
            for method_name, method_results in results.items():
                if method_name != 'statistical_analysis':
                    for metric, score in method_results['metrics'].items():
                        if score is not None:
                            aggregate_scores[method_name][metric].append(score)
        
        # Calculate aggregate statistics
        aggregate_results = {
            'mean_scores': {},
            'std_scores': {},
            'confidence_intervals': {}
        }
        
        for method, metric_scores in aggregate_scores.items():
            aggregate_results['mean_scores'][method] = {}
            aggregate_results['std_scores'][method] = {}
            aggregate_results['confidence_intervals'][method] = {}
            
            for metric, scores in metric_scores.items():
                if scores:
                    aggregate_results['mean_scores'][method][metric] = np.mean(scores)
                    aggregate_results['std_scores'][method][metric] = np.std(scores)
                    
                    # Bootstrap CI
                    ci_result = self.statistical_analyzer.bootstrap_confidence_interval(
                        np.array(scores),
                        metric_fn=np.mean
                    )
                    aggregate_results['confidence_intervals'][method][metric] = ci_result
        
        batch_results = {
            'sample_results': sample_results,
            'aggregate_results': aggregate_results,
            'n_samples': len(batch_data)
        }
        
        return batch_results
    
    def add_streaming_sample(self, sample: Dict[str, float]) -> None:
        """
        Add a sample for streaming comparison.
        
        Args:
            sample: Dictionary with method scores
        """
        if not self.streaming_mode:
            raise ComparisonError("Streaming mode not enabled")
        
        for method, score in sample.items():
            if method == 'timestamp':
                continue
            
            if method not in self.streaming_buffer:
                self.streaming_buffer[method] = []
            
            self.streaming_buffer[method].append(score)
            
            # Maintain window size
            if len(self.streaming_buffer[method]) > self.streaming_window_size:
                self.streaming_buffer[method].pop(0)
    
    def get_rolling_statistics(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get rolling statistics for streaming comparison.
        
        Args:
            window_size: Window size (None for current buffer size)
            
        Returns:
            Rolling statistics for each method
        """
        if not self.streaming_mode:
            raise ComparisonError("Streaming mode not enabled")
        
        stats = {}
        
        for method, buffer in self.streaming_buffer.items():
            if not buffer:
                continue
            
            # Use specified window size or current buffer
            if window_size:
                data = buffer[-window_size:]
            else:
                data = buffer
            
            stats[method] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'min': np.min(data),
                'max': np.max(data),
                'trend': self._calculate_trend(data)
            }
        
        return stats
    
    def _calculate_trend(self, data: List[float]) -> str:
        """Calculate trend direction from data."""
        if len(data) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        if abs(slope) < 0.001:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def parameter_sweep(self, base_audio: np.ndarray, reference: np.ndarray,
                       param_grid: Dict[str, List[Any]], enhancement_fn: Callable,
                       sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Evaluate different parameter combinations.
        
        Args:
            base_audio: Base audio to enhance
            reference: Reference audio
            param_grid: Parameter grid to sweep
            enhancement_fn: Enhancement function
            sample_rate: Sample rate
            
        Returns:
            Parameter sweep results
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        all_results = []
        
        for combo in combinations:
            # Create parameter dictionary
            params = dict(zip(param_names, combo))
            
            # Apply enhancement
            enhanced = enhancement_fn(base_audio, **params)
            
            # Compute metrics
            metric_scores = {}
            for metric in self.metrics:
                try:
                    score = self._compute_single_metric(metric, enhanced, reference, sample_rate)
                    metric_scores[metric] = score
                except:
                    metric_scores[metric] = None
            
            all_results.append({
                'params': params,
                'metrics': metric_scores,
                'overall_score': np.mean([s for s in metric_scores.values() if s is not None])
            })
        
        # Find best parameters
        best_result = max(all_results, key=lambda x: x['overall_score'])
        
        # Calculate parameter importance (simplified)
        param_importance = {}
        for param in param_names:
            scores_by_value = {}
            for result in all_results:
                value = result['params'][param]
                if value not in scores_by_value:
                    scores_by_value[value] = []
                scores_by_value[value].append(result['overall_score'])
            
            # Calculate variance across parameter values
            mean_scores = [np.mean(scores) for scores in scores_by_value.values()]
            param_importance[param] = np.var(mean_scores)
        
        return {
            'best_params': best_result['params'],
            'best_score': best_result['overall_score'],
            'all_results': all_results,
            'param_importance': param_importance
        }
    
    def cross_dataset_validation(self, datasets: Dict[str, Dict[str, List[np.ndarray]]],
                                methods: Dict[str, Callable], sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Perform cross-dataset validation.
        
        Args:
            datasets: Dictionary of datasets
            methods: Dictionary of enhancement methods
            sample_rate: Sample rate
            
        Returns:
            Cross-dataset validation results
        """
        results = {
            'per_dataset': {},
            'overall': {},
            'consistency_scores': {}
        }
        
        method_scores_by_dataset = {method: {} for method in methods}
        
        # Evaluate each method on each dataset
        for dataset_name, dataset in datasets.items():
            results['per_dataset'][dataset_name] = {}
            
            for method_name, method_fn in methods.items():
                scores = []
                
                for sample, reference in zip(dataset['samples'], dataset['references']):
                    # Apply method
                    enhanced = method_fn(sample)
                    
                    # Compute metrics
                    metric_scores = []
                    for metric in self.metrics:
                        try:
                            score = self._compute_single_metric(metric, enhanced, reference, sample_rate)
                            metric_scores.append(score)
                        except:
                            pass
                    
                    if metric_scores:
                        scores.append(np.mean(metric_scores))
                
                avg_score = np.mean(scores) if scores else 0
                results['per_dataset'][dataset_name][method_name] = {
                    'mean_score': avg_score,
                    'std_score': np.std(scores) if scores else 0,
                    'n_samples': len(scores)
                }
                
                method_scores_by_dataset[method_name][dataset_name] = avg_score
        
        # Calculate overall scores and consistency
        for method_name in methods:
            dataset_scores = list(method_scores_by_dataset[method_name].values())
            results['overall'][method_name] = {
                'mean_score': np.mean(dataset_scores),
                'std_across_datasets': np.std(dataset_scores)
            }
            
            # Consistency score (inverse of coefficient of variation)
            mean_score = np.mean(dataset_scores)
            if mean_score > 0:
                cv = np.std(dataset_scores) / mean_score
                consistency = 1 / (1 + cv)
            else:
                consistency = 0
            
            results['consistency_scores'][method_name] = consistency
        
        return results
    
    def export_results(self, output_path: str) -> None:
        """Export comparison results to file."""
        output_path = Path(output_path)
        
        export_data = {
            'results': self.results,
            'methods': {name: {
                'sample_rate': method.sample_rate,
                'metadata': method.metadata
            } for name, method in self.methods.items()},
            'metrics': self.metrics,
            'config': {
                'confidence_level': self.config.confidence_level,
                'multiple_comparison_correction': self.config.multiple_comparison_correction
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def import_results(self, input_path: str) -> None:
        """Import comparison results from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.results = data['results']
        self.metrics = data['metrics']
        
        # Update config if available
        if 'config' in data:
            self.config.confidence_level = data['config'].get('confidence_level', 0.95)
            self.config.multiple_comparison_correction = data['config'].get(
                'multiple_comparison_correction', 'bonferroni'
            )
    
    def analyze_metric_correlations(self, metric_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics.
        
        Args:
            metric_data: Dictionary mapping metric names to score arrays
            
        Returns:
            Correlation analysis results
        """
        metric_names = list(metric_data.keys())
        n_metrics = len(metric_names)
        
        # Calculate correlation matrix
        correlation_matrix = np.zeros((n_metrics, n_metrics))
        p_values = np.zeros((n_metrics, n_metrics))
        
        for i, metric_i in enumerate(metric_names):
            for j, metric_j in enumerate(metric_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    p_values[i, j] = 0.0
                else:
                    # Calculate Pearson correlation
                    corr, p_val = stats.pearsonr(metric_data[metric_i], metric_data[metric_j])
                    correlation_matrix[i, j] = corr
                    p_values[i, j] = p_val
        
        # Interpret correlations
        interpretations = []
        for i, metric_i in enumerate(metric_names):
            for j, metric_j in enumerate(metric_names):
                if i < j:  # Only look at upper triangle
                    corr = correlation_matrix[i, j]
                    p_val = p_values[i, j]
                    
                    strength = 'weak'
                    if abs(corr) > 0.7:
                        strength = 'strong'
                    elif abs(corr) > 0.4:
                        strength = 'moderate'
                    
                    if p_val < 0.05:
                        interpretations.append({
                            'metrics': (metric_i, metric_j),
                            'correlation': corr,
                            'strength': strength,
                            'significant': True
                        })
        
        return {
            'correlation_matrix': correlation_matrix,
            'p_values': p_values,
            'metric_names': metric_names,
            'interpretation': interpretations
        }
    
    def learn_metric_weights(self, comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Learn optimal metric weights from user preferences.
        
        Args:
            comparison_data: List of comparisons with user preferences
            
        Returns:
            Optimal weights and fit quality
        """
        # Extract metrics and preferences
        X = []
        y = []
        
        metric_names = list(comparison_data[0]['metrics'].keys())
        
        for item in comparison_data:
            metric_values = [item['metrics'][m] for m in metric_names]
            X.append(metric_values)
            y.append(item['user_preference'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Define objective function
        def objective(weights):
            # Compute weighted scores
            scores = X @ weights
            
            # Convert to rankings
            predicted_rankings = np.argsort(-scores)
            true_rankings = np.argsort(-y)
            
            # Calculate ranking error
            error = np.sum((predicted_rankings - true_rankings) ** 2)
            return error
        
        # Constraints: weights sum to 1, all non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]
        
        # Initial guess: equal weights
        n_metrics = len(metric_names)
        initial_weights = np.ones(n_metrics) / n_metrics
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints
        )
        
        # Calculate fit quality
        if result.success:
            final_scores = X @ result.x
            correlation = np.corrcoef(final_scores, y)[0, 1]
            fit_quality = correlation ** 2  # R-squared
        else:
            fit_quality = 0.0
        
        return {
            'weights': dict(zip(metric_names, result.x)),
            'success': result.success,
            'fit_quality': fit_quality,
            'n_samples': len(comparison_data)
        }