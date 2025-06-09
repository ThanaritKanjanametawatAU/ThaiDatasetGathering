# Task S02_T04: Build Comparison Framework

## Task Overview
Build a comprehensive comparison framework that enables systematic evaluation of different enhancement strategies, parameter configurations, and processing pipelines.

## Technical Requirements

### Core Implementation
- **Comparison Framework** (`processors/audio_enhancement/evaluation/comparison_framework.py`)
  - A/B/C testing capabilities
  - Statistical significance testing
  - Multi-metric comparison
  - Automated report generation

### Key Features
1. **Flexible Comparison Types**
   - Before/after enhancement
   - Multiple algorithm comparison
   - Parameter sweep evaluation
   - Cross-dataset validation

2. **Statistical Analysis**
   - Paired t-tests for significance
   - Effect size calculation (Cohen's d)
   - Confidence intervals
   - Distribution analysis

3. **Visualization Suite**
   - Metric comparison charts
   - Box plots for distributions
   - Radar charts for multi-metric view
   - Time-series quality tracking

## TDD Requirements

### Test Structure
```
tests/test_comparison_framework.py
- test_ab_comparison()
- test_multi_algorithm_comparison()
- test_statistical_significance()
- test_report_generation()
- test_visualization_creation()
- test_batch_comparison()
```

### Test Data Requirements
- Multiple enhancement outputs
- Ground truth references
- Various quality levels
- Edge case scenarios

## Implementation Approach

### Phase 1: Core Framework
```python
class ComparisonFramework:
    def __init__(self, metrics=['pesq', 'stoi', 'si_sdr']):
        self.metrics = metrics
        self.results = {}
    
    def add_method(self, name, audio_data, reference=None):
        # Add method for comparison
        pass
    
    def compare(self, statistical_tests=True):
        # Perform comprehensive comparison
        pass
    
    def generate_report(self, output_path):
        # Create detailed comparison report
        pass
```

### Phase 2: Statistical Engine
- Hypothesis testing implementation
- Multiple comparison corrections
- Effect size calculations
- Bootstrap confidence intervals

### Phase 3: Visualization & Reporting
- Interactive dashboards
- PDF report generation
- Real-time comparison UI
- Export capabilities

## Acceptance Criteria
1. ✅ Support for 5+ simultaneous method comparisons
2. ✅ Statistical significance testing with p-values
3. ✅ Automated report generation in multiple formats
4. ✅ Real-time comparison capabilities
5. ✅ Integration with existing metrics suite

## Example Usage
```python
from processors.audio_enhancement.evaluation import ComparisonFramework

# Initialize framework
framework = ComparisonFramework(
    metrics=['pesq', 'stoi', 'si_sdr', 'snr']
)

# Add methods to compare
framework.add_method('baseline', baseline_audio, reference_audio)
framework.add_method('enhanced_v1', enhanced_v1_audio, reference_audio)
framework.add_method('enhanced_v2', enhanced_v2_audio, reference_audio)

# Perform comparison
results = framework.compare(statistical_tests=True)

# Generate report
framework.generate_report('comparison_report.pdf')

# Get best method
best_method = framework.get_best_method(metric='pesq')
print(f"Best method: {best_method['name']} (PESQ: {best_method['score']:.2f})")
```

## Dependencies
- NumPy for numerical operations
- SciPy for statistical tests
- Matplotlib/Plotly for visualizations
- Pandas for data management
- ReportLab for PDF generation

## Performance Targets
- Comparison of 10 methods: < 5 seconds
- Report generation: < 10 seconds
- Real-time metric updates: < 100ms
- Memory usage: < 1GB for 100 samples

## Notes
- Consider multiple testing corrections (Bonferroni)
- Include both objective and subjective metrics
- Support for custom metric plugins
- Enable reproducible comparisons with saved configs

## Statistical Analysis Methods

### Hypothesis Testing Framework

1. **Paired Statistical Tests**
   ```python
   from scipy import stats
   import numpy as np
   
   class StatisticalAnalyzer:
       def __init__(self, alpha=0.05):
           self.alpha = alpha
           
       def paired_t_test(self, scores_a, scores_b):
           """Paired t-test for comparing two methods"""
           # Check assumptions
           differences = scores_a - scores_b
           _, normality_p = stats.shapiro(differences)
           
           if normality_p < 0.05:
               # Use Wilcoxon signed-rank test for non-normal data
               statistic, p_value = stats.wilcoxon(scores_a, scores_b)
               test_name = 'Wilcoxon signed-rank'
           else:
               # Use paired t-test for normal data
               statistic, p_value = stats.ttest_rel(scores_a, scores_b)
               test_name = 'Paired t-test'
           
           # Effect size (Cohen's d)
           mean_diff = np.mean(differences)
           pooled_std = np.sqrt((np.std(scores_a)**2 + np.std(scores_b)**2) / 2)
           cohens_d = mean_diff / pooled_std
           
           return {
               'test_name': test_name,
               'statistic': statistic,
               'p_value': p_value,
               'significant': p_value < self.alpha,
               'cohens_d': cohens_d,
               'effect_size': self._interpret_cohens_d(cohens_d)
           }
       
       def _interpret_cohens_d(self, d):
           """Interpret Cohen's d effect size"""
           abs_d = abs(d)
           if abs_d < 0.2:
               return 'negligible'
           elif abs_d < 0.5:
               return 'small'
           elif abs_d < 0.8:
               return 'medium'
           else:
               return 'large'
   ```

2. **Multiple Comparison Corrections**
   ```python
   from statsmodels.stats.multitest import multipletests
   
   def apply_multiple_comparison_correction(p_values, method='bonferroni'):
       """Apply correction for multiple comparisons"""
       methods = {
           'bonferroni': 'bonferroni',
           'holm': 'holm',
           'fdr_bh': 'fdr_bh',  # Benjamini-Hochberg
           'fdr_by': 'fdr_by'   # Benjamini-Yekutieli
       }
       
       rejected, p_adjusted, _, _ = multipletests(
           p_values, 
           alpha=0.05, 
           method=methods[method]
       )
       
       return {
           'rejected': rejected,
           'p_adjusted': p_adjusted,
           'method': method
       }
   ```

3. **Bootstrap Confidence Intervals**
   ```python
   def bootstrap_confidence_interval(scores, metric_fn, n_bootstrap=1000, ci=95):
       """Compute bootstrap confidence intervals"""
       n_samples = len(scores)
       bootstrap_results = []
       
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
           'mean': np.mean(bootstrap_results),
           'std': np.std(bootstrap_results),
           'ci_lower': lower,
           'ci_upper': upper,
           'ci_level': ci
       }
   ```

### Visualization Techniques

1. **Multi-Metric Radar Charts**
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   
   def create_radar_chart(methods, metrics, scores, output_path):
       """Create radar chart for multi-metric comparison"""
       n_metrics = len(metrics)
       angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
       
       fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
       
       for method, method_scores in scores.items():
           # Normalize scores to 0-1 range
           normalized = normalize_scores(method_scores, metrics)
           normalized += normalized[:1]  # Complete the circle
           angles_plot = angles + angles[:1]
           
           ax.plot(angles_plot, normalized, 'o-', linewidth=2, label=method)
           ax.fill(angles_plot, normalized, alpha=0.25)
       
       ax.set_xticks(angles)
       ax.set_xticklabels(metrics)
       ax.set_ylim(0, 1)
       ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
       
       plt.title('Multi-Metric Performance Comparison')
       plt.tight_layout()
       plt.savefig(output_path, dpi=300, bbox_inches='tight')
       plt.close()
   ```

2. **Statistical Significance Heatmap**
   ```python
   import seaborn as sns
   
   def create_significance_heatmap(methods, p_values_matrix, output_path):
       """Create heatmap showing statistical significance between methods"""
       n_methods = len(methods)
       
       # Create significance matrix
       sig_matrix = np.zeros((n_methods, n_methods))
       annotations = []
       
       for i in range(n_methods):
           row_annotations = []
           for j in range(n_methods):
               if i == j:
                   row_annotations.append('-')
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
       sns.heatmap(
           sig_matrix,
           annot=annotations,
           fmt='',
           cmap='RdYlGn_r',
           xticklabels=methods,
           yticklabels=methods,
           cbar_kws={'label': 'Significance Level'}
       )
       
       plt.title('Statistical Significance Matrix')
       plt.tight_layout()
       plt.savefig(output_path, dpi=300, bbox_inches='tight')
       plt.close()
   ```

### Real-Time Monitoring Integration

1. **Live Comparison Dashboard**
   ```python
   from flask import Flask, jsonify, render_template
   import threading
   import time
   
   class RealTimeComparisonDashboard:
       def __init__(self, framework):
           self.framework = framework
           self.app = Flask(__name__)
           self.setup_routes()
           self.update_thread = None
           self.is_running = False
           
       def setup_routes(self):
           @self.app.route('/')
           def index():
               return render_template('comparison_dashboard.html')
           
           @self.app.route('/api/metrics')
           def get_metrics():
               return jsonify(self.framework.get_current_metrics())
           
           @self.app.route('/api/comparison')
           def get_comparison():
               return jsonify(self.framework.get_comparison_results())
           
       def start_monitoring(self, update_interval=1.0):
           """Start real-time monitoring"""
           self.is_running = True
           self.update_thread = threading.Thread(
               target=self._update_loop,
               args=(update_interval,)
           )
           self.update_thread.start()
           
       def _update_loop(self, interval):
           """Background update loop"""
           while self.is_running:
               self.framework.update_metrics()
               time.sleep(interval)
   ```

2. **Streaming Metric Updates**
   ```python
   class StreamingComparisonFramework(ComparisonFramework):
       def __init__(self, metrics, window_size=100):
           super().__init__(metrics)
           self.window_size = window_size
           self.metric_buffers = {
               method: {metric: [] for metric in metrics}
               for method in self.methods
           }
           
       def add_streaming_sample(self, method, sample_data):
           """Add new sample for streaming comparison"""
           # Compute metrics for new sample
           new_metrics = self.compute_metrics(sample_data)
           
           # Update buffers
           for metric, value in new_metrics.items():
               buffer = self.metric_buffers[method][metric]
               buffer.append(value)
               
               # Maintain window size
               if len(buffer) > self.window_size:
                   buffer.pop(0)
           
           # Trigger comparison update
           self._update_comparison()
           
       def get_rolling_statistics(self, method, metric):
           """Get rolling statistics for a method/metric"""
           buffer = self.metric_buffers[method][metric]
           
           if not buffer:
               return None
               
           return {
               'mean': np.mean(buffer),
               'std': np.std(buffer),
               'median': np.median(buffer),
               'q25': np.percentile(buffer, 25),
               'q75': np.percentile(buffer, 75),
               'trend': self._compute_trend(buffer)
           }
   ```

### Adaptive Learning Algorithms

1. **Dynamic Threshold Learning**
   ```python
   from sklearn.gaussian_process import GaussianProcessRegressor
   from sklearn.gaussian_process.kernels import RBF, WhiteKernel
   
   class AdaptiveComparisonLearner:
       def __init__(self):
           # Gaussian Process for learning metric relationships
           kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
           self.gp = GaussianProcessRegressor(kernel=kernel)
           self.history = []
           
       def learn_from_comparison(self, metrics, user_preference):
           """Learn from user preferences in comparisons"""
           # Store comparison result
           self.history.append({
               'metrics': metrics,
               'preference': user_preference
           })
           
           # Update model when enough data
           if len(self.history) > 10:
               X = np.array([h['metrics'] for h in self.history])
               y = np.array([h['preference'] for h in self.history])
               
               self.gp.fit(X, y)
               
       def predict_preference(self, metrics):
           """Predict user preference based on metrics"""
           if len(self.history) < 10:
               return None
               
           pred_mean, pred_std = self.gp.predict([metrics], return_std=True)
           
           return {
               'predicted_preference': pred_mean[0],
               'uncertainty': pred_std[0]
           }
   ```

2. **Automatic Weight Optimization**
   ```python
   from scipy.optimize import minimize
   
   def optimize_metric_weights(comparison_data, target_rankings):
       """Optimize metric weights to match target rankings"""
       n_metrics = len(comparison_data[0]['metrics'])
       
       def objective(weights):
           # Compute weighted scores
           weighted_scores = []
           for data in comparison_data:
               score = sum(w * m for w, m in zip(weights, data['metrics']))
               weighted_scores.append(score)
           
           # Compare with target rankings
           predicted_rankings = np.argsort(weighted_scores)[::-1]
           ranking_error = np.sum((predicted_rankings - target_rankings)**2)
           
           return ranking_error
       
       # Constraints: weights sum to 1, all non-negative
       constraints = [
           {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
           {'type': 'ineq', 'fun': lambda w: w}
       ]
       
       # Initial guess: equal weights
       initial_weights = np.ones(n_metrics) / n_metrics
       
       # Optimize
       result = minimize(
           objective,
           initial_weights,
           method='SLSQP',
           constraints=constraints
       )
       
       return {
           'optimal_weights': result.x,
           'success': result.success,
           'ranking_error': result.fun
       }
   ```

### Production-Ready Implementation

```python
class ProductionComparisonFramework:
    """Production-ready comparison framework with all features"""
    
    def __init__(self, config_path='comparison_config.yaml'):
        self.config = self._load_config(config_path)
        self.metrics = self.config['metrics']
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualization_engine = VisualizationEngine()
        self.adaptive_learner = AdaptiveComparisonLearner()
        self.results_cache = {}
        
    def comprehensive_compare(self, methods_data, output_dir='results'):
        """Perform comprehensive comparison with all analyses"""
        # Compute all metrics
        metric_results = self._compute_all_metrics(methods_data)
        
        # Statistical analysis
        statistical_results = self._perform_statistical_analysis(metric_results)
        
        # Generate visualizations
        viz_paths = self._generate_all_visualizations(
            metric_results, statistical_results, output_dir
        )
        
        # Create comprehensive report
        report = self._generate_comprehensive_report(
            metric_results, statistical_results, viz_paths
        )
        
        # Learn from results
        self._update_adaptive_model(metric_results)
        
        # Cache results
        self._cache_results(report)
        
        return report
        
    def _perform_statistical_analysis(self, metric_results):
        """Comprehensive statistical analysis"""
        methods = list(metric_results.keys())
        n_methods = len(methods)
        
        # Pairwise comparisons
        p_values_matrix = np.ones((n_methods, n_methods))
        effect_sizes = {}
        
        for i, method_a in enumerate(methods):
            for j, method_b in enumerate(methods):
                if i != j:
                    # Perform comparison
                    comparison = self.statistical_analyzer.paired_t_test(
                        metric_results[method_a]['all_scores'],
                        metric_results[method_b]['all_scores']
                    )
                    
                    p_values_matrix[i, j] = comparison['p_value']
                    effect_sizes[f"{method_a}_vs_{method_b}"] = comparison['cohens_d']
        
        # Apply multiple comparison correction
        p_values_flat = p_values_matrix[np.triu_indices(n_methods, k=1)]
        correction_results = apply_multiple_comparison_correction(p_values_flat)
        
        return {
            'p_values_matrix': p_values_matrix,
            'effect_sizes': effect_sizes,
            'correction_results': correction_results
        }
```