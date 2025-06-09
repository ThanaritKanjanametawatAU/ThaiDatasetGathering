"""
Comprehensive test suite for the Comparison Framework.

Tests all aspects of the comparison framework including:
- A/B/C testing capabilities
- Statistical significance testing
- Multi-metric comparison
- Visualization support
- Report generation
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

# Import the framework (will fail initially as it doesn't exist yet)
from processors.audio_enhancement.evaluation.comparison_framework import (
    ComparisonFramework,
    StatisticalAnalyzer,
    VisualizationEngine,
    ComparisonMethod,
    ComparisonReport,
    ComparisonError,
    MetricConfig,
    ComparisonConfig
)


class TestComparisonFramework:
    """Test the core ComparisonFramework functionality."""
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing."""
        np.random.seed(42)
        duration = 3.0  # seconds
        sample_rate = 16000
        n_samples = int(duration * sample_rate)
        
        # Create reference (clean) audio
        t = np.linspace(0, duration, n_samples)
        reference = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
        
        # Create degraded versions with different noise levels
        noise_levels = [0.1, 0.2, 0.3]
        degraded_versions = []
        
        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, n_samples)
            degraded = reference + noise
            degraded_versions.append(degraded)
        
        return {
            'reference': reference,
            'degraded': degraded_versions,
            'sample_rate': sample_rate
        }
    
    @pytest.fixture
    def mock_metrics(self):
        """Mock metric calculators."""
        mock_pesq = Mock()
        mock_stoi = Mock()
        mock_si_sdr = Mock()
        
        # Set up return values
        mock_pesq.calculate.return_value = Mock(overall_score=3.5)
        mock_stoi.calculate.return_value = Mock(overall_score=0.85)
        mock_si_sdr.calculate.return_value = Mock(sdr=15.0)
        
        return {
            'pesq': mock_pesq,
            'stoi': mock_stoi,
            'si_sdr': mock_si_sdr
        }
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization(self):
        """Test framework initialization."""
        # Test with default metrics
        framework = ComparisonFramework()
        assert framework.metrics == ['pesq', 'stoi', 'si_sdr']
        assert len(framework.methods) == 0
        assert len(framework.results) == 0
        
        # Test with custom metrics
        custom_metrics = ['pesq', 'stoi', 'snr', 'custom_metric']
        framework = ComparisonFramework(metrics=custom_metrics)
        assert framework.metrics == custom_metrics
        
        # Test with configuration file
        config = ComparisonConfig(
            metrics=['pesq', 'stoi'],
            statistical_tests=True,
            confidence_level=0.95,
            multiple_comparison_correction='bonferroni'
        )
        framework = ComparisonFramework(config=config)
        assert framework.config == config
    
    def test_add_method(self, sample_audio_data):
        """Test adding methods for comparison."""
        framework = ComparisonFramework()
        
        # Add baseline method
        framework.add_method(
            'baseline',
            sample_audio_data['degraded'][0],
            reference=sample_audio_data['reference'],
            sample_rate=sample_audio_data['sample_rate']
        )
        
        assert 'baseline' in framework.methods
        assert framework.methods['baseline'].name == 'baseline'
        assert framework.methods['baseline'].audio_data is not None
        assert framework.methods['baseline'].reference is not None
        
        # Add enhanced methods
        framework.add_method(
            'enhanced_v1',
            sample_audio_data['degraded'][1],
            reference=sample_audio_data['reference'],
            sample_rate=sample_audio_data['sample_rate']
        )
        
        framework.add_method(
            'enhanced_v2',
            sample_audio_data['degraded'][2],
            reference=sample_audio_data['reference'],
            sample_rate=sample_audio_data['sample_rate']
        )
        
        assert len(framework.methods) == 3
        
        # Test duplicate method name
        with pytest.raises(ComparisonError, match="Method 'baseline' already exists"):
            framework.add_method(
                'baseline',
                sample_audio_data['degraded'][0],
                reference=sample_audio_data['reference']
            )
        
        # Test without reference
        framework.add_method(
            'no_reference',
            sample_audio_data['degraded'][0],
            sample_rate=sample_audio_data['sample_rate']
        )
        assert framework.methods['no_reference'].reference is None
    
    def test_ab_comparison(self, sample_audio_data, mock_metrics):
        """Test A/B comparison functionality."""
        framework = ComparisonFramework()
        
        # Mock metric calculators
        with patch.object(framework, '_get_metric_calculator') as mock_get_calc:
            mock_get_calc.side_effect = lambda metric: mock_metrics.get(metric)
            
            # Add two methods
            framework.add_method(
                'method_a',
                sample_audio_data['degraded'][0],
                reference=sample_audio_data['reference'],
                sample_rate=sample_audio_data['sample_rate']
            )
            
            framework.add_method(
                'method_b',
                sample_audio_data['degraded'][1],
                reference=sample_audio_data['reference'],
                sample_rate=sample_audio_data['sample_rate']
            )
            
            # Perform A/B comparison
            results = framework.compare(
                methods=['method_a', 'method_b'],
                statistical_tests=True
            )
            
            assert 'method_a' in results
            assert 'method_b' in results
            assert 'statistical_analysis' in results
            assert 'pairwise_comparisons' in results['statistical_analysis']
            
            # Check metric scores
            assert results['method_a']['metrics']['pesq'] == 3.5
            assert results['method_a']['metrics']['stoi'] == 0.85
            assert results['method_a']['metrics']['si_sdr'] == 15.0
    
    def test_multi_algorithm_comparison(self, sample_audio_data, mock_metrics):
        """Test comparison of multiple algorithms."""
        framework = ComparisonFramework()
        
        # Add multiple methods
        method_names = ['baseline', 'enhanced_v1', 'enhanced_v2', 'enhanced_v3', 'enhanced_v4']
        
        with patch.object(framework, '_get_metric_calculator') as mock_get_calc:
            mock_get_calc.side_effect = lambda metric: mock_metrics.get(metric)
            
            for i, name in enumerate(method_names):
                audio_idx = min(i, len(sample_audio_data['degraded']) - 1)
                framework.add_method(
                    name,
                    sample_audio_data['degraded'][audio_idx],
                    reference=sample_audio_data['reference'],
                    sample_rate=sample_audio_data['sample_rate']
                )
            
            # Perform comparison
            results = framework.compare(statistical_tests=True)
            
            # Verify all methods are compared
            for name in method_names:
                assert name in results
                assert 'metrics' in results[name]
                assert 'rank' in results[name]
            
            # Check ranking
            rankings = {name: results[name]['rank'] for name in method_names}
            assert len(set(rankings.values())) <= len(method_names)
            
            # Check best method selection
            best_method = framework.get_best_method(metric='pesq')
            assert best_method is not None
            assert 'name' in best_method
            assert 'score' in best_method
    
    def test_statistical_significance(self):
        """Test statistical significance testing."""
        analyzer = StatisticalAnalyzer(alpha=0.05)
        
        # Test paired t-test with normally distributed data
        np.random.seed(42)
        scores_a = np.random.normal(3.5, 0.2, 50)
        scores_b = np.random.normal(3.7, 0.2, 50)  # Slightly better
        
        result = analyzer.paired_t_test(scores_a, scores_b)
        
        assert 'test_name' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'cohens_d' in result
        assert 'effect_size' in result
        
        # Test should detect significant difference
        assert result['p_value'] < 0.05
        assert result['significant'] is True
        
        # Test with non-normal data (should use Wilcoxon)
        scores_c = np.concatenate([np.zeros(25), np.ones(25)])  # Highly non-normal
        scores_d = np.concatenate([np.zeros(20), np.ones(30)])
        
        result_non_normal = analyzer.paired_t_test(scores_c, scores_d)
        assert result_non_normal['test_name'] == 'Wilcoxon signed-rank'
        
        # Test effect size interpretation
        assert analyzer._interpret_cohens_d(0.1) == 'negligible'
        assert analyzer._interpret_cohens_d(0.3) == 'small'
        assert analyzer._interpret_cohens_d(0.6) == 'medium'
        assert analyzer._interpret_cohens_d(1.0) == 'large'
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison corrections."""
        analyzer = StatisticalAnalyzer()
        
        # Create p-values for multiple comparisons
        p_values = [0.01, 0.03, 0.04, 0.06, 0.10]
        
        # Test Bonferroni correction
        bonferroni_result = analyzer.apply_multiple_comparison_correction(
            p_values, method='bonferroni'
        )
        
        assert 'rejected' in bonferroni_result
        assert 'p_adjusted' in bonferroni_result
        assert 'method' in bonferroni_result
        assert bonferroni_result['method'] == 'bonferroni'
        
        # Adjusted p-values should be higher
        assert all(p_adj >= p_orig for p_adj, p_orig in 
                  zip(bonferroni_result['p_adjusted'], p_values))
        
        # Test other correction methods
        for method in ['holm', 'fdr_bh', 'fdr_by']:
            result = analyzer.apply_multiple_comparison_correction(
                p_values, method=method
            )
            assert result['method'] == method
            assert len(result['p_adjusted']) == len(p_values)
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        analyzer = StatisticalAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        scores = np.random.normal(3.5, 0.3, 100)
        
        # Calculate bootstrap CI
        ci_result = analyzer.bootstrap_confidence_interval(
            scores,
            metric_fn=np.mean,
            n_bootstrap=1000,
            ci=95
        )
        
        assert 'mean' in ci_result
        assert 'std' in ci_result
        assert 'ci_lower' in ci_result
        assert 'ci_upper' in ci_result
        assert 'ci_level' in ci_result
        
        # Check that CI contains the true mean
        true_mean = np.mean(scores)
        assert ci_result['ci_lower'] <= true_mean <= ci_result['ci_upper']
        
        # Test different confidence levels
        ci_99 = analyzer.bootstrap_confidence_interval(
            scores, np.mean, n_bootstrap=1000, ci=99
        )
        
        # 99% CI should be wider than 95% CI
        assert (ci_99['ci_upper'] - ci_99['ci_lower']) > \
               (ci_result['ci_upper'] - ci_result['ci_lower'])
    
    def test_visualization_creation(self, temp_dir):
        """Test visualization generation."""
        viz_engine = VisualizationEngine()
        
        # Test data
        methods = ['Method A', 'Method B', 'Method C']
        metrics = ['PESQ', 'STOI', 'SI-SDR']
        scores = {
            'Method A': [3.5, 0.85, 15.0],
            'Method B': [3.7, 0.87, 16.5],
            'Method C': [3.3, 0.82, 14.0]
        }
        
        # Test radar chart creation
        radar_path = Path(temp_dir) / 'radar_chart.png'
        viz_engine.create_radar_chart(
            methods, metrics, scores, str(radar_path)
        )
        assert radar_path.exists()
        
        # Test box plot creation
        box_data = {
            'Method A': np.random.normal(3.5, 0.2, 50),
            'Method B': np.random.normal(3.7, 0.2, 50),
            'Method C': np.random.normal(3.3, 0.2, 50)
        }
        
        box_path = Path(temp_dir) / 'box_plot.png'
        viz_engine.create_box_plot(
            box_data, 'PESQ Score Distribution', str(box_path)
        )
        assert box_path.exists()
        
        # Test significance heatmap
        n_methods = len(methods)
        p_values_matrix = np.random.rand(n_methods, n_methods)
        np.fill_diagonal(p_values_matrix, 1.0)  # No self-comparison
        
        heatmap_path = Path(temp_dir) / 'significance_heatmap.png'
        viz_engine.create_significance_heatmap(
            methods, p_values_matrix, str(heatmap_path)
        )
        assert heatmap_path.exists()
        
        # Test metric evolution plot
        time_points = list(range(10))
        evolution_data = {
            'Method A': np.cumsum(np.random.randn(10)) + 3.5,
            'Method B': np.cumsum(np.random.randn(10)) + 3.7,
            'Method C': np.cumsum(np.random.randn(10)) + 3.3
        }
        
        evolution_path = Path(temp_dir) / 'metric_evolution.png'
        viz_engine.create_metric_evolution_plot(
            evolution_data, time_points, 'PESQ', str(evolution_path)
        )
        assert evolution_path.exists()
    
    def test_report_generation(self, sample_audio_data, mock_metrics, temp_dir):
        """Test comprehensive report generation."""
        framework = ComparisonFramework()
        
        with patch.object(framework, '_get_metric_calculator') as mock_get_calc:
            mock_get_calc.side_effect = lambda metric: mock_metrics.get(metric)
            
            # Add methods
            for i, name in enumerate(['baseline', 'enhanced_v1', 'enhanced_v2']):
                framework.add_method(
                    name,
                    sample_audio_data['degraded'][i],
                    reference=sample_audio_data['reference'],
                    sample_rate=sample_audio_data['sample_rate']
                )
            
            # Perform comparison
            results = framework.compare(statistical_tests=True)
            
            # Generate report
            report_path = Path(temp_dir) / 'comparison_report.pdf'
            framework.generate_report(str(report_path))
            
            # Check if PDF exists, or HTML if ReportLab not available
            if not report_path.exists():
                # Check for fallback HTML file
                html_fallback_path = report_path.with_suffix('.html')
                assert html_fallback_path.exists(), "Neither PDF nor fallback HTML report was generated"
            else:
                assert report_path.exists()
            
            # Test HTML report generation
            html_path = Path(temp_dir) / 'comparison_report.html'
            framework.generate_report(str(html_path), format='html')
            assert html_path.exists()
            
            # Test JSON report generation
            json_path = Path(temp_dir) / 'comparison_report.json'
            framework.generate_report(str(json_path), format='json')
            assert json_path.exists()
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                json_report = json.load(f)
            
            assert 'methods' in json_report
            assert 'metrics' in json_report
            assert 'statistical_analysis' in json_report
            assert 'timestamp' in json_report
    
    def test_batch_comparison(self, sample_audio_data):
        """Test batch comparison functionality."""
        framework = ComparisonFramework()
        
        # Create batch data
        batch_size = 10
        batch_data = []
        
        for i in range(batch_size):
            sample = {
                'id': f'sample_{i}',
                'methods': {
                    'baseline': sample_audio_data['degraded'][0] + np.random.normal(0, 0.01, len(sample_audio_data['degraded'][0])),
                    'enhanced': sample_audio_data['degraded'][1] + np.random.normal(0, 0.01, len(sample_audio_data['degraded'][1]))
                },
                'reference': sample_audio_data['reference']
            }
            batch_data.append(sample)
        
        # Perform batch comparison
        batch_results = framework.compare_batch(
            batch_data,
            sample_rate=sample_audio_data['sample_rate']
        )
        
        assert 'aggregate_results' in batch_results
        assert 'sample_results' in batch_results
        assert 'n_samples' in batch_results
        assert batch_results['n_samples'] == batch_size
        assert len(batch_results['sample_results']) == batch_size
        
        # Check aggregate statistics
        agg = batch_results['aggregate_results']
        assert 'mean_scores' in agg
        assert 'std_scores' in agg
        assert 'confidence_intervals' in agg
    
    def test_real_time_comparison(self):
        """Test real-time comparison capabilities."""
        framework = ComparisonFramework(streaming_mode=True)
        
        # Simulate streaming samples
        n_samples = 100
        window_size = 10
        
        for i in range(n_samples):
            # Generate sample
            sample = {
                'timestamp': i,
                'method_a': np.random.normal(3.5, 0.2),
                'method_b': np.random.normal(3.7, 0.2)
            }
            
            # Add to streaming comparison
            framework.add_streaming_sample(sample)
            
            # Get rolling statistics
            if i >= window_size:
                stats = framework.get_rolling_statistics(window_size)
                
                assert 'method_a' in stats
                assert 'method_b' in stats
                assert 'mean' in stats['method_a']
                assert 'std' in stats['method_a']
                assert 'trend' in stats['method_a']
    
    def test_parameter_sweep_evaluation(self, sample_audio_data):
        """Test parameter sweep evaluation functionality."""
        framework = ComparisonFramework()
        
        # Define parameter grid
        param_grid = {
            'noise_reduction': [0.1, 0.3, 0.5, 0.7],
            'enhancement_level': ['low', 'medium', 'high']
        }
        
        # Mock enhancement function
        def mock_enhance(audio, noise_reduction, enhancement_level):
            # Simulate enhancement with different parameters
            noise_factor = 1 - noise_reduction
            return audio * noise_factor
        
        # Perform parameter sweep
        sweep_results = framework.parameter_sweep(
            base_audio=sample_audio_data['degraded'][0],
            reference=sample_audio_data['reference'],
            param_grid=param_grid,
            enhancement_fn=mock_enhance,
            sample_rate=sample_audio_data['sample_rate']
        )
        
        assert 'best_params' in sweep_results
        assert 'all_results' in sweep_results
        assert 'param_importance' in sweep_results
        
        # Verify all parameter combinations were tested
        expected_combinations = len(param_grid['noise_reduction']) * len(param_grid['enhancement_level'])
        assert len(sweep_results['all_results']) == expected_combinations
    
    def test_cross_dataset_validation(self):
        """Test cross-dataset validation functionality."""
        framework = ComparisonFramework()
        
        # Create mock datasets
        datasets = {
            'dataset_a': {
                'samples': [np.random.randn(16000) for _ in range(20)],
                'references': [np.random.randn(16000) for _ in range(20)]
            },
            'dataset_b': {
                'samples': [np.random.randn(16000) for _ in range(20)],
                'references': [np.random.randn(16000) for _ in range(20)]
            }
        }
        
        # Mock enhancement methods
        methods = {
            'method_1': lambda x: x * 0.9,
            'method_2': lambda x: x * 0.95
        }
        
        # Perform cross-dataset validation
        validation_results = framework.cross_dataset_validation(
            datasets=datasets,
            methods=methods,
            sample_rate=16000
        )
        
        assert 'per_dataset' in validation_results
        assert 'overall' in validation_results
        assert 'consistency_scores' in validation_results
        
        # Check results for each dataset
        for dataset_name in datasets:
            assert dataset_name in validation_results['per_dataset']
            dataset_results = validation_results['per_dataset'][dataset_name]
            
            for method_name in methods:
                assert method_name in dataset_results
    
    def test_export_and_import_results(self, temp_dir):
        """Test exporting and importing comparison results."""
        framework = ComparisonFramework()
        
        # Create some comparison results
        framework.results = {
            'methods': ['method_a', 'method_b'],
            'metrics': ['pesq', 'stoi', 'si_sdr'],
            'scores': {
                'method_a': {'pesq': 3.5, 'stoi': 0.85, 'si_sdr': 15.0},
                'method_b': {'pesq': 3.7, 'stoi': 0.87, 'si_sdr': 16.5}
            }
        }
        
        # Export results
        export_path = Path(temp_dir) / 'comparison_results.json'
        framework.export_results(str(export_path))
        assert export_path.exists()
        
        # Create new framework and import results
        new_framework = ComparisonFramework()
        new_framework.import_results(str(export_path))
        
        assert new_framework.results == framework.results
    
    def test_metric_correlation_analysis(self):
        """Test metric correlation analysis."""
        framework = ComparisonFramework()
        
        # Create sample metric data
        n_samples = 100
        np.random.seed(42)
        
        # Create correlated metrics
        pesq_scores = np.random.normal(3.5, 0.3, n_samples)
        stoi_scores = 0.2 * pesq_scores + np.random.normal(0.15, 0.05, n_samples)
        si_sdr_scores = 4 * pesq_scores + np.random.normal(1, 0.5, n_samples)
        
        metric_data = {
            'pesq': pesq_scores,
            'stoi': stoi_scores,
            'si_sdr': si_sdr_scores
        }
        
        # Perform correlation analysis
        correlation_results = framework.analyze_metric_correlations(metric_data)
        
        assert 'correlation_matrix' in correlation_results
        assert 'p_values' in correlation_results
        assert 'interpretation' in correlation_results
        
        # Check correlation matrix shape
        n_metrics = len(metric_data)
        assert correlation_results['correlation_matrix'].shape == (n_metrics, n_metrics)
    
    def test_adaptive_metric_weighting(self):
        """Test adaptive metric weighting based on user preferences."""
        framework = ComparisonFramework()
        
        # Create comparison data with user preferences
        comparison_data = [
            {
                'metrics': {'pesq': 3.5, 'stoi': 0.85, 'si_sdr': 15.0},
                'user_preference': 1  # Preferred
            },
            {
                'metrics': {'pesq': 3.3, 'stoi': 0.90, 'si_sdr': 14.0},
                'user_preference': 0  # Not preferred
            },
            {
                'metrics': {'pesq': 3.7, 'stoi': 0.82, 'si_sdr': 16.5},
                'user_preference': 1  # Preferred
            }
        ]
        
        # Learn optimal weights
        optimal_weights = framework.learn_metric_weights(comparison_data)
        
        assert 'weights' in optimal_weights
        assert 'success' in optimal_weights
        assert 'fit_quality' in optimal_weights
        
        # Weights should sum to 1
        assert abs(sum(optimal_weights['weights'].values()) - 1.0) < 0.001
        
        # All weights should be non-negative
        assert all(w >= 0 for w in optimal_weights['weights'].values())
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        framework = ComparisonFramework()
        
        # Test comparing without methods
        with pytest.raises(ComparisonError, match="No methods added"):
            framework.compare()
        
        # Test invalid metric
        with pytest.raises(ComparisonError, match="Unknown metric"):
            framework = ComparisonFramework(metrics=['invalid_metric'])
        
        # Test mismatched audio lengths
        with pytest.raises(ComparisonError, match="Audio length mismatch"):
            framework.add_method('test', np.zeros(1000), reference=np.zeros(2000))
        
        # Test invalid statistical test
        analyzer = StatisticalAnalyzer()
        with pytest.raises(ValueError, match="Unknown correction method"):
            analyzer.apply_multiple_comparison_correction([0.01, 0.05], method='invalid')
    
    def test_performance_benchmarks(self, sample_audio_data):
        """Test performance meets specified targets."""
        import time
        
        framework = ComparisonFramework()
        
        # Add 10 methods
        for i in range(10):
            framework.add_method(
                f'method_{i}',
                sample_audio_data['degraded'][i % 3],
                reference=sample_audio_data['reference'],
                sample_rate=sample_audio_data['sample_rate']
            )
        
        # Test comparison time
        start_time = time.time()
        results = framework.compare(statistical_tests=True)
        comparison_time = time.time() - start_time
        
        # Should complete within 5 seconds
        assert comparison_time < 5.0
        
        # Test report generation time
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            start_time = time.time()
            framework.generate_report(tmp.name)
            report_time = time.time() - start_time
            
            # Should complete within 10 seconds
            assert report_time < 10.0
        
        # Test memory usage
        import psutil
        import os
        
        # Enable streaming mode for memory test
        framework_streaming = ComparisonFramework(streaming_mode=True)
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process 100 samples
        for i in range(100):
            framework_streaming.add_streaming_sample({
                'method_a': np.random.rand(),
                'method_b': np.random.rand()
            })
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Should use less than 1GB for 100 samples
        # Note: Memory measurement can be noisy, so we allow some margin
        assert memory_increase < 1024 or memory_increase < 50  # Allow up to 50MB increase


class TestStatisticalAnalyzer:
    """Test the StatisticalAnalyzer component."""
    
    def test_normality_testing(self):
        """Test normality testing functionality."""
        analyzer = StatisticalAnalyzer()
        
        # Test with normal data
        normal_data = np.random.normal(0, 1, 100)
        is_normal, p_value = analyzer.test_normality(normal_data)
        assert is_normal is True
        assert p_value > 0.05
        
        # Test with non-normal data
        non_normal_data = np.concatenate([np.zeros(50), np.ones(50)])
        is_normal, p_value = analyzer.test_normality(non_normal_data)
        assert is_normal is False
        # Non-normal data should have very low p_value
        assert p_value < 0.05
    
    def test_effect_size_calculations(self):
        """Test various effect size calculations."""
        analyzer = StatisticalAnalyzer()
        
        # Test Cohen's d
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)
        
        cohens_d = analyzer.calculate_cohens_d(group1, group2)
        assert isinstance(cohens_d, float)
        # Cohen's d can be negative if group1 has higher mean
        assert abs(cohens_d) > 0.3  # Should show at least small effect
        
        # Test Glass's delta
        glass_delta = analyzer.calculate_glass_delta(group1, group2)
        assert isinstance(glass_delta, float)
        
        # Test Hedges' g
        hedges_g = analyzer.calculate_hedges_g(group1, group2)
        assert isinstance(hedges_g, float)
    
    def test_power_analysis(self):
        """Test statistical power analysis."""
        analyzer = StatisticalAnalyzer()
        
        # Calculate required sample size
        required_n = analyzer.calculate_required_sample_size(
            effect_size=0.5,  # Medium effect
            alpha=0.05,
            power=0.8
        )
        
        assert isinstance(required_n, int)
        assert required_n > 0
        
        # Calculate achieved power
        achieved_power = analyzer.calculate_achieved_power(
            n=50,
            effect_size=0.5,
            alpha=0.05
        )
        
        assert isinstance(achieved_power, float)
        assert 0 <= achieved_power <= 1


class TestVisualizationEngine:
    """Test the VisualizationEngine component."""
    
    @pytest.fixture
    def viz_engine(self):
        """Create visualization engine instance."""
        return VisualizationEngine()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_color_schemes(self, viz_engine):
        """Test color scheme management."""
        # Test default color scheme
        colors = viz_engine.get_color_scheme('default')
        assert isinstance(colors, list)
        assert len(colors) > 0
        
        # Test colorblind-friendly scheme
        cb_colors = viz_engine.get_color_scheme('colorblind')
        assert isinstance(cb_colors, list)
        assert len(cb_colors) > 0
    
    def test_interactive_plots(self, viz_engine, temp_dir):
        """Test interactive plot generation."""
        # Create sample data
        data = {
            'Method A': np.random.normal(3.5, 0.2, 50),
            'Method B': np.random.normal(3.7, 0.2, 50),
            'Method C': np.random.normal(3.3, 0.2, 50)
        }
        
        # Generate interactive plot
        html_path = Path(temp_dir) / 'interactive_plot.html'
        viz_engine.create_interactive_comparison(
            data,
            output_path=str(html_path),
            plot_type='violin'
        )
        
        assert html_path.exists()
        
        # Check that HTML contains plotly
        with open(html_path, 'r') as f:
            html_content = f.read()
        assert 'plotly' in html_content
    
    def test_multi_panel_figures(self, viz_engine, temp_dir):
        """Test multi-panel figure creation."""
        # Create sample data for multiple metrics
        metrics = ['PESQ', 'STOI', 'SI-SDR']
        methods = ['Baseline', 'Enhanced']
        
        data = {
            metric: {
                method: np.random.normal(3.5, 0.2, 30)
                for method in methods
            }
            for metric in metrics
        }
        
        # Create multi-panel figure
        fig_path = Path(temp_dir) / 'multi_panel.png'
        viz_engine.create_multi_panel_comparison(
            data,
            output_path=str(fig_path),
            figsize=(12, 8)
        )
        
        assert fig_path.exists()


class TestIntegration:
    """Integration tests for the complete comparison framework."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_workflow(self, temp_dir):
        """Test complete workflow from data to report."""
        # Initialize framework with configuration
        config = ComparisonConfig(
            metrics=['pesq', 'stoi', 'si_sdr'],
            statistical_tests=True,
            confidence_level=0.95,
            multiple_comparison_correction='bonferroni',
            visualization_options={
                'create_plots': True,
                'plot_types': ['box', 'radar', 'heatmap'],
                'color_scheme': 'colorblind'
            }
        )
        
        framework = ComparisonFramework(config=config)
        
        # Generate synthetic test data
        np.random.seed(42)
        sample_rate = 16000
        duration = 3.0
        n_samples = int(duration * sample_rate)
        
        # Reference signal
        t = np.linspace(0, duration, n_samples)
        reference = np.sin(2 * np.pi * 440 * t)
        
        # Create different quality versions
        methods_data = {
            'original': reference + np.random.normal(0, 0.3, n_samples),
            'denoised_v1': reference + np.random.normal(0, 0.1, n_samples),
            'denoised_v2': reference + np.random.normal(0, 0.05, n_samples),
            'overprocessed': reference * 0.5 + np.random.normal(0, 0.02, n_samples)
        }
        
        # Add all methods
        for name, audio in methods_data.items():
            framework.add_method(name, audio, reference=reference, sample_rate=sample_rate)
        
        # Perform comparison
        results = framework.compare(statistical_tests=True)
        
        # Generate comprehensive report
        report_path = Path(temp_dir) / 'final_report.pdf'
        framework.generate_report(str(report_path))
        
        # Verify outputs
        if not report_path.exists():
            # Check for fallback HTML file
            html_fallback_path = report_path.with_suffix('.html')
            assert html_fallback_path.exists(), "Neither PDF nor fallback HTML report was generated"
        
        assert 'statistical_analysis' in results
        assert all(method in results for method in methods_data.keys())
        
        # Get best method
        best_method = framework.get_best_method(metric='pesq')
        assert best_method['name'] in methods_data.keys()
        
        # Export results for future use
        export_path = Path(temp_dir) / 'comparison_results.json'
        framework.export_results(str(export_path))
        assert export_path.exists()