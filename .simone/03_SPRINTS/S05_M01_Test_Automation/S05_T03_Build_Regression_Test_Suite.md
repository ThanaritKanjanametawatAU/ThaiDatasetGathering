# Task S05_T03: Build Regression Test Suite

## Task Overview
Build a comprehensive regression test suite that automatically detects quality degradations, performance regressions, and functionality breaks across code changes.

## Technical Requirements

### Core Implementation
- **Regression Test Suite** (`tests/regression/regression_test_suite.py`)
  - Baseline management
  - Change detection
  - Automated execution
  - Result comparison

### Key Features
1. **Test Categories**
   - Quality regression tests
   - Performance regression tests
   - Functionality tests
   - API compatibility tests

2. **Detection Mechanisms**
   - Statistical significance testing
   - Threshold-based detection
   - Trend analysis
   - Anomaly detection

3. **Automation Features**
   - Git hook integration
   - CI/CD pipeline
   - Automated reporting
   - Baseline updates

## TDD Requirements

### Test Structure
```
tests/test_regression_suite.py
- test_quality_regression_detection()
- test_performance_regression()
- test_functionality_preservation()
- test_baseline_management()
- test_statistical_significance()
- test_report_generation()
```

### Test Data Requirements
- Historical baselines
- Reference outputs
- Performance benchmarks
- Known regression cases

## Implementation Approach

### Phase 1: Core Suite
```python
class RegressionTestSuite:
    def __init__(self, baseline_dir='baselines/'):
        self.baseline_dir = baseline_dir
        self.test_registry = TestRegistry()
        self.comparator = ResultComparator()
        
    def run_regression_tests(self, commit_hash=None):
        # Run complete regression suite
        pass
    
    def detect_regressions(self, current_results, baseline):
        # Detect any regressions
        pass
    
    def update_baseline(self, results, version):
        # Update baseline after verification
        pass
```

### Phase 2: Advanced Detection
- Machine learning regression detection
- Multi-dimensional analysis
- Root cause identification
- Impact assessment

#### Statistical Regression Detection
```python
class StatisticalRegressionDetector:
    """Advanced statistical methods for regression detection"""
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.historical_data = []
        self.change_point_detector = ChangePointDetector()
        
    def detect_regression(self, current_metrics, baseline_metrics):
        """Detect regression using multiple statistical tests"""
        detections = {}
        
        # Welch's t-test for unequal variances
        for metric_name in current_metrics:
            current = np.array(current_metrics[metric_name])
            baseline = np.array(baseline_metrics[metric_name])
            
            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(current, baseline, equal_var=False)
            
            # Effect size (Cohen's d)
            effect_size = self._compute_cohens_d(current, baseline)
            
            # Bayesian estimation
            posterior = self._bayesian_comparison(current, baseline)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = stats.mannwhitneyu(current, baseline, alternative='less')
            
            detections[metric_name] = {
                'is_regression': p_value < (1 - self.confidence_level) and np.mean(current) < np.mean(baseline),
                'p_value': p_value,
                'effect_size': effect_size,
                'confidence_interval': self._compute_confidence_interval(current, baseline),
                'bayesian_probability': posterior['prob_regression'],
                'mann_whitney_p': u_pvalue,
                'practical_significance': abs(effect_size) > 0.2  # Small effect threshold
            }
        
        return detections
    
    def _compute_cohens_d(self, group1, group2):
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return d
    
    def _bayesian_comparison(self, current, baseline):
        """Bayesian comparison of two groups"""
        import pymc3 as pm
        
        with pm.Model() as model:
            # Priors
            mu_current = pm.Normal('mu_current', mu=np.mean(current), sd=np.std(current))
            mu_baseline = pm.Normal('mu_baseline', mu=np.mean(baseline), sd=np.std(baseline))
            
            sigma_current = pm.HalfNormal('sigma_current', sd=np.std(current))
            sigma_baseline = pm.HalfNormal('sigma_baseline', sd=np.std(baseline))
            
            # Likelihoods
            current_obs = pm.Normal('current_obs', mu=mu_current, sd=sigma_current, observed=current)
            baseline_obs = pm.Normal('baseline_obs', mu=mu_baseline, sd=sigma_baseline, observed=baseline)
            
            # Difference
            diff = pm.Deterministic('diff', mu_current - mu_baseline)
            
            # Sample
            trace = pm.sample(2000, tune=1000, return_inferencedata=False)
        
        # Probability of regression
        prob_regression = np.mean(trace['diff'] < 0)
        
        return {
            'prob_regression': prob_regression,
            'diff_mean': np.mean(trace['diff']),
            'diff_hdi': pm.stats.hdi(trace['diff'], hdi_prob=0.95)
        }
    
    def detect_change_points(self, time_series):
        """Detect change points in performance metrics"""
        # PELT algorithm for change point detection
        from ruptures import Pelt
        
        model = "rbf"  # Radial basis function
        algo = Pelt(model=model, min_size=3, jump=1)
        
        # Fit and predict
        result = algo.fit_predict(time_series.reshape(-1, 1), pen=3)
        
        # Analyze segments
        segments = []
        prev_idx = 0
        
        for idx in result[:-1]:  # Last element is always n_samples
            segment = time_series[prev_idx:idx]
            segments.append({
                'start': prev_idx,
                'end': idx,
                'mean': np.mean(segment),
                'std': np.std(segment),
                'trend': self._compute_trend(segment)
            })
            prev_idx = idx
        
        return segments
```

#### Machine Learning Regression Detection
```python
class MLRegressionDetector:
    """Machine learning approach to regression detection"""
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(contamination=0.1),
            'one_class_svm': OneClassSVM(gamma='auto'),
            'lstm_autoencoder': self._build_lstm_autoencoder()
        }
        self.scaler = StandardScaler()
        
    def _build_lstm_autoencoder(self):
        """Build LSTM autoencoder for anomaly detection"""
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, input_shape=(None, 10)),
            LSTM(64, activation='relu', return_sequences=True),
            LSTM(32, activation='relu', return_sequences=False),
            RepeatVector(10),
            LSTM(32, activation='relu', return_sequences=True),
            LSTM(64, activation='relu', return_sequences=True),
            LSTM(128, activation='relu', return_sequences=True),
            TimeDistributed(Dense(10))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_on_baseline(self, baseline_metrics):
        """Train anomaly detection models on baseline data"""
        # Prepare features
        features = self._extract_features(baseline_metrics)
        scaled_features = self.scaler.fit_transform(features)
        
        # Train each model
        for name, model in self.models.items():
            if name == 'lstm_autoencoder':
                # Reshape for LSTM
                lstm_data = scaled_features.reshape(-1, 10, scaled_features.shape[1] // 10)
                model.fit(lstm_data, lstm_data, epochs=50, batch_size=32, verbose=0)
            else:
                model.fit(scaled_features)
    
    def detect_anomalies(self, current_metrics):
        """Detect anomalies in current metrics"""
        features = self._extract_features(current_metrics)
        scaled_features = self.scaler.transform(features)
        
        anomaly_scores = {}
        
        for name, model in self.models.items():
            if name == 'isolation_forest':
                scores = model.decision_function(scaled_features)
                predictions = model.predict(scaled_features)
                anomaly_scores[name] = {
                    'scores': scores,
                    'anomalies': predictions == -1,
                    'anomaly_rate': np.mean(predictions == -1)
                }
            
            elif name == 'lstm_autoencoder':
                lstm_data = scaled_features.reshape(-1, 10, scaled_features.shape[1] // 10)
                reconstructed = model.predict(lstm_data)
                mse = np.mean((lstm_data - reconstructed) ** 2, axis=(1, 2))
                threshold = np.percentile(mse, 95)
                anomaly_scores[name] = {
                    'scores': mse,
                    'anomalies': mse > threshold,
                    'anomaly_rate': np.mean(mse > threshold)
                }
        
        # Ensemble decision
        ensemble_anomalies = self._ensemble_decision(anomaly_scores)
        
        return {
            'individual_models': anomaly_scores,
            'ensemble': ensemble_anomalies
        }
    
    def _ensemble_decision(self, anomaly_scores):
        """Combine multiple models for robust detection"""
        # Voting mechanism
        all_anomalies = [scores['anomalies'] for scores in anomaly_scores.values()]
        
        # Majority voting
        votes = np.sum(all_anomalies, axis=0)
        threshold = len(all_anomalies) / 2
        
        ensemble_anomalies = votes > threshold
        
        # Weighted scoring based on model confidence
        weights = {'isolation_forest': 0.4, 'one_class_svm': 0.3, 'lstm_autoencoder': 0.3}
        
        weighted_scores = np.zeros(len(ensemble_anomalies))
        for name, scores in anomaly_scores.items():
            weighted_scores += weights.get(name, 0.33) * scores['scores']
        
        return {
            'anomalies': ensemble_anomalies,
            'confidence_scores': weighted_scores,
            'anomaly_rate': np.mean(ensemble_anomalies)
        }
```

#### Root Cause Analysis
```python
class RegressionRootCauseAnalyzer:
    """Analyze root causes of detected regressions"""
    def __init__(self):
        self.causal_model = None
        self.feature_importance = {}
        
    def analyze_regression(self, regression_data, system_state):
        """Perform root cause analysis on detected regression"""
        causes = []
        
        # 1. Timeline analysis
        timeline_causes = self._analyze_timeline(regression_data, system_state)
        causes.extend(timeline_causes)
        
        # 2. Component analysis
        component_causes = self._analyze_components(regression_data, system_state)
        causes.extend(component_causes)
        
        # 3. Dependency analysis
        dependency_causes = self._analyze_dependencies(regression_data, system_state)
        causes.extend(dependency_causes)
        
        # 4. Change correlation
        change_causes = self._correlate_with_changes(regression_data, system_state)
        causes.extend(change_causes)
        
        # Rank causes by likelihood
        ranked_causes = self._rank_causes(causes)
        
        return {
            'primary_cause': ranked_causes[0] if ranked_causes else None,
            'all_causes': ranked_causes,
            'causal_graph': self._build_causal_graph(causes),
            'recommendations': self._generate_recommendations(ranked_causes)
        }
    
    def _analyze_timeline(self, regression_data, system_state):
        """Analyze timeline for potential causes"""
        causes = []
        
        # Check for correlations with system events
        regression_time = regression_data['detection_time']
        
        # Recent deployments
        recent_deployments = system_state.get_deployments(
            start_time=regression_time - timedelta(hours=24),
            end_time=regression_time
        )
        
        for deployment in recent_deployments:
            correlation = self._compute_temporal_correlation(
                deployment['time'],
                regression_time
            )
            
            if correlation > 0.7:
                causes.append({
                    'type': 'deployment',
                    'description': f"Deployment {deployment['version']} at {deployment['time']}",
                    'confidence': correlation,
                    'evidence': {
                        'timing': 'Close temporal proximity',
                        'changes': deployment.get('changes', [])
                    }
                })
        
        return causes
    
    def _analyze_components(self, regression_data, system_state):
        """Analyze which components contributed to regression"""
        # Use SHAP values for component importance
        import shap
        
        # Build model of system performance
        X = system_state.get_component_metrics()
        y = regression_data['performance_delta']
        
        model = XGBRegressor()
        model.fit(X, y)
        
        # SHAP analysis
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Identify high-impact components
        component_impacts = np.abs(shap_values.values).mean(axis=0)
        component_names = system_state.get_component_names()
        
        causes = []
        for idx, impact in enumerate(component_impacts):
            if impact > np.percentile(component_impacts, 80):
                causes.append({
                    'type': 'component',
                    'description': f"Component {component_names[idx]} shows high impact",
                    'confidence': impact / np.max(component_impacts),
                    'evidence': {
                        'shap_value': impact,
                        'recent_changes': system_state.get_component_changes(component_names[idx])
                    }
                })
        
        return causes
```

### Phase 3: Integration
- Automated PR checks
- Continuous monitoring
- Trend dashboards
- Alert systems

## Acceptance Criteria
1. ✅ Detect regressions with 95% accuracy
2. ✅ False positive rate < 5%
3. ✅ Complete suite run < 10 minutes
4. ✅ Support for 50+ test cases
5. ✅ Automated baseline management

## Example Usage
```python
from tests.regression import RegressionTestSuite

# Initialize suite
regression_suite = RegressionTestSuite(baseline_dir='baselines/')

# Run regression tests
results = regression_suite.run_regression_tests(
    commit_hash='abc123',
    categories=['quality', 'performance', 'functionality']
)

print(f"Total tests: {results.total}")
print(f"Passed: {results.passed}")
print(f"Failed: {results.failed}")
print(f"Regressions detected: {len(results.regressions)}")

# Analyze regressions
for regression in results.regressions:
    print(f"\nRegression in: {regression.test_name}")
    print(f"Type: {regression.type}")
    print(f"Severity: {regression.severity}")
    print(f"Delta: {regression.delta}")
    print(f"Statistical significance: p={regression.p_value:.4f}")

# Get detailed report
report = regression_suite.generate_report(results)
print(f"\nSummary:")
print(f"Quality score: {report.quality_score:.2f} (baseline: {report.baseline_quality:.2f})")
print(f"Performance: {report.performance_ratio:.2f}x baseline")
print(f"API compatibility: {report.api_compatibility}%")

# Update baseline if approved
if results.approved:
    regression_suite.update_baseline(results, version='2.0')
    print("Baseline updated successfully")
```

## Dependencies
- pytest for test framework
- NumPy for statistics
- SciPy for significance testing
- GitPython for version control
- Jinja2 for reporting

## Performance Targets
- Test execution: < 10 minutes
- Regression detection: < 1 second
- Report generation: < 5 seconds
- Memory usage: < 500MB

## Notes
- Implement smart test selection
- Consider flaky test handling
- Support for partial runs
- Enable bisection for regression finding

## Advanced Regression Testing Theory

### Test Selection Optimization
```python
class SmartTestSelector:
    """Optimize test selection for faster regression detection"""
    def __init__(self, test_history):
        self.test_history = test_history
        self.failure_predictor = self._train_failure_predictor()
        
    def select_tests(self, code_changes, time_budget):
        """Select optimal test subset given constraints"""
        # Analyze code changes
        affected_components = self._analyze_code_changes(code_changes)
        
        # Compute test priorities
        test_priorities = []
        
        for test in self.test_history.get_all_tests():
            priority = self._compute_test_priority(
                test,
                affected_components,
                code_changes
            )
            
            test_priorities.append((test, priority))
        
        # Sort by priority
        test_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Select tests within time budget
        selected_tests = []
        total_time = 0
        
        for test, priority in test_priorities:
            estimated_time = self._estimate_test_time(test)
            
            if total_time + estimated_time <= time_budget:
                selected_tests.append(test)
                total_time += estimated_time
            else:
                # Try to fit smaller tests
                continue
        
        return selected_tests
    
    def _compute_test_priority(self, test, affected_components, code_changes):
        """Compute test priority using multiple factors"""
        # Historical failure rate
        failure_rate = self.test_history.get_failure_rate(test)
        
        # Code coverage of changed components
        coverage_score = self._compute_coverage_score(test, affected_components)
        
        # Predicted failure probability
        failure_prob = self.failure_predictor.predict_proba(
            self._extract_features(test, code_changes)
        )[0, 1]
        
        # Test importance (critical path, user-facing, etc.)
        importance = self.test_history.get_test_importance(test)
        
        # Combined priority score
        priority = (
            0.3 * failure_rate +
            0.3 * coverage_score +
            0.2 * failure_prob +
            0.2 * importance
        )
        
        return priority
```

### Flaky Test Detection and Handling
```python
class FlakyTestDetector:
    """Detect and handle flaky tests in regression suite"""
    def __init__(self, confidence_threshold=0.95):
        self.confidence_threshold = confidence_threshold
        self.test_results = defaultdict(list)
        
    def analyze_flakiness(self, test_name, window_size=100):
        """Analyze test flakiness using statistical methods"""
        results = self.test_results[test_name][-window_size:]
        
        if len(results) < 10:
            return {'is_flaky': False, 'confidence': 0, 'pattern': None}
        
        # Convert to binary (1 = pass, 0 = fail)
        binary_results = [1 if r['status'] == 'pass' else 0 for r in results]
        
        # Statistical tests for randomness
        # 1. Runs test
        runs_p_value = self._runs_test(binary_results)
        
        # 2. Autocorrelation
        autocorr = self._compute_autocorrelation(binary_results)
        
        # 3. Entropy
        entropy = self._compute_entropy(binary_results)
        
        # 4. Pattern detection
        patterns = self._detect_patterns(results)
        
        # Determine flakiness
        is_flaky = (
            runs_p_value < 0.05 or  # Fails runs test
            abs(autocorr) < 0.1 or  # Low autocorrelation
            entropy > 0.8  # High entropy
        )
        
        confidence = self._compute_flakiness_confidence(
            runs_p_value, autocorr, entropy
        )
        
        return {
            'is_flaky': is_flaky,
            'confidence': confidence,
            'pattern': patterns,
            'metrics': {
                'runs_p_value': runs_p_value,
                'autocorrelation': autocorr,
                'entropy': entropy,
                'failure_rate': 1 - np.mean(binary_results)
            }
        }
    
    def _detect_patterns(self, results):
        """Detect patterns in flaky test failures"""
        patterns = []
        
        # Time-based patterns
        failure_times = [r['timestamp'] for r in results if r['status'] == 'fail']
        
        if failure_times:
            # Check for time-of-day patterns
            hours = [t.hour for t in failure_times]
            hour_counts = Counter(hours)
            
            if max(hour_counts.values()) > len(failure_times) * 0.3:
                patterns.append({
                    'type': 'time_of_day',
                    'peak_hours': [h for h, c in hour_counts.most_common(3)]
                })
            
            # Check for day-of-week patterns
            weekdays = [t.weekday() for t in failure_times]
            weekday_counts = Counter(weekdays)
            
            if max(weekday_counts.values()) > len(failure_times) * 0.3:
                patterns.append({
                    'type': 'day_of_week',
                    'peak_days': [d for d, c in weekday_counts.most_common(2)]
                })
        
        # Environmental patterns
        envs = [r.get('environment', {}) for r in results]
        self._detect_environmental_patterns(envs, patterns)
        
        return patterns
```