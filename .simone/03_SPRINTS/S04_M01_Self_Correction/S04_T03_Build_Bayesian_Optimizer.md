# Task S04_T03: Build Bayesian Optimizer

## Task Overview
Build an advanced Bayesian optimization system that intelligently explores parameter spaces using probabilistic models to find optimal configurations efficiently.

## Technical Requirements

### Core Implementation
- **Bayesian Optimizer** (`processors/audio_enhancement/optimization/bayesian_optimizer.py`)
  - Gaussian Process modeling
  - Acquisition functions
  - Uncertainty quantification
  - Sequential optimization

### Key Features
1. **Surrogate Models**
   - Gaussian Processes (GP)
   - Random Forests
   - Neural network surrogates
   - Ensemble models

2. **Acquisition Functions**
   - Expected Improvement (EI)
   - Upper Confidence Bound (UCB)
   - Probability of Improvement (PI)
   - Knowledge Gradient

3. **Advanced Features**
   - Multi-objective optimization
   - Constraint handling
   - Batch suggestions
   - Transfer learning

## TDD Requirements

### Test Structure
```
tests/test_bayesian_optimizer.py
- test_gaussian_process_fitting()
- test_acquisition_functions()
- test_optimization_convergence()
- test_uncertainty_estimates()
- test_batch_optimization()
- test_constraint_satisfaction()
```

### Test Data Requirements
- Known optimization landscapes
- Multi-modal functions
- Constrained problems
- Noisy objectives

## Implementation Approach

### Phase 1: Core Optimizer
```python
class BayesianOptimizer:
    def __init__(self, bounds, acquisition='ei'):
        self.bounds = bounds
        self.acquisition = acquisition
        self.gp = GaussianProcessRegressor()
        self.observations = []
        
    def suggest_next(self, n_suggestions=1):
        # Suggest next parameters to evaluate
        pass
    
    def update(self, params, value):
        # Update model with new observation
        pass
    
    def get_uncertainty_map(self):
        # Return uncertainty estimates
        pass
```

### Phase 2: Advanced Modeling
- Deep kernel learning
- Multi-task GP
- Bayesian neural networks
- Active learning

#### Gaussian Process with Advanced Kernels
```python
class AdvancedGaussianProcess:
    """GP with sophisticated kernel designs for audio optimization"""
    def __init__(self, kernel_type='matern'):
        self.kernel = self._build_kernel(kernel_type)
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky decomposition
        
    def _build_kernel(self, kernel_type):
        """Build advanced kernel functions"""
        if kernel_type == 'deep':
            # Deep kernel learning
            return DeepKernel(
                base_kernel=gpytorch.kernels.RBFKernel(),
                num_dims=10,
                num_layers=3
            )
        elif kernel_type == 'spectral_mixture':
            # Spectral mixture kernel for multi-scale patterns
            return SpectralMixtureKernel(
                num_mixtures=4,
                ard_num_dims=True
            )
        elif kernel_type == 'matern':
            # Matern kernel with automatic relevance determination
            return gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=True,
                lengthscale_constraint=gpytorch.constraints.Interval(0.01, 10.0)
            )
    
    def fit(self, X, y, optimize_hyperparams=True):
        """Fit GP with automatic hyperparameter optimization"""
        self.X_train = X
        self.y_train = y
        
        if optimize_hyperparams:
            # Optimize kernel hyperparameters using marginal likelihood
            self._optimize_hyperparameters()
        
        # Compute Cholesky decomposition for efficient inference
        K = self.kernel(self.X_train, self.X_train)
        K += 1e-6 * torch.eye(len(self.X_train))  # Numerical stability
        self.L = torch.cholesky(K)
        
    def _optimize_hyperparameters(self):
        """Optimize hyperparameters using gradient-based methods"""
        optimizer = torch.optim.Adam(self.kernel.parameters(), lr=0.1)
        
        for _ in range(100):
            optimizer.zero_grad()
            
            # Negative log marginal likelihood
            K = self.kernel(self.X_train, self.X_train)
            K += 1e-6 * torch.eye(len(self.X_train))
            
            L = torch.cholesky(K)
            alpha = torch.cholesky_solve(self.y_train.unsqueeze(-1), L)
            
            # Log marginal likelihood
            log_marginal = -0.5 * self.y_train @ alpha.squeeze()
            log_marginal -= torch.sum(torch.log(torch.diag(L)))
            log_marginal -= 0.5 * len(self.X_train) * np.log(2 * np.pi)
            
            (-log_marginal).backward()  # Minimize negative log likelihood
            optimizer.step()
    
    def predict(self, X_test, return_std=True):
        """Predict with uncertainty quantification"""
        K_star = self.kernel(X_test, self.X_train)
        alpha = torch.cholesky_solve(self.y_train.unsqueeze(-1), self.L)
        
        # Mean prediction
        mean = K_star @ alpha
        
        if return_std:
            # Variance prediction
            v = torch.cholesky_solve(K_star.t(), self.L)
            K_star_star = self.kernel(X_test, X_test)
            var = torch.diag(K_star_star) - torch.sum(K_star * v.t(), dim=1)
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            
            return mean.squeeze(), std
        
        return mean.squeeze()
```

#### Advanced Acquisition Functions
```python
class AdvancedAcquisitionFunctions:
    """Sophisticated acquisition functions for Bayesian optimization"""
    
    @staticmethod
    def expected_improvement(mean, std, best_f, xi=0.01):
        """Expected Improvement with exploration parameter"""
        z = (mean - best_f - xi) / std
        ei = std * (z * norm.cdf(z) + norm.pdf(z))
        return ei
    
    @staticmethod
    def upper_confidence_bound(mean, std, beta=2.0, t=None):
        """UCB with theoretical regret bounds"""
        if t is not None:
            # Time-dependent beta for theoretical guarantees
            d = mean.shape[-1]  # dimension
            delta = 0.1
            beta = 2 * np.log(d * t**2 * np.pi**2 / (6 * delta))
        
        return mean + np.sqrt(beta) * std
    
    @staticmethod
    def knowledge_gradient(gp_model, X_candidate, mc_samples=100):
        """Knowledge Gradient acquisition function"""
        # Current best expected value
        current_mean, _ = gp_model.predict(gp_model.X_train)
        current_best = torch.max(current_mean)
        
        # Monte Carlo approximation of knowledge gradient
        kg_values = torch.zeros(len(X_candidate))
        
        for i, x in enumerate(X_candidate):
            # Predict at candidate point
            mean_x, std_x = gp_model.predict(x.unsqueeze(0))
            
            # Sample possible observations
            y_samples = torch.normal(mean_x, std_x, size=(mc_samples,))
            
            future_values = []
            for y_sample in y_samples:
                # Create hypothetical updated dataset
                X_new = torch.cat([gp_model.X_train, x.unsqueeze(0)])
                y_new = torch.cat([gp_model.y_train, y_sample.unsqueeze(0)])
                
                # Compute new posterior mean at training points
                gp_temp = copy.deepcopy(gp_model)
                gp_temp.fit(X_new, y_new, optimize_hyperparams=False)
                
                new_mean, _ = gp_temp.predict(gp_model.X_train)
                future_values.append(torch.max(new_mean))
            
            # Knowledge gradient is expected improvement in best value
            kg_values[i] = torch.mean(torch.stack(future_values)) - current_best
        
        return kg_values
    
    @staticmethod
    def max_value_entropy_search(gp_model, X_candidate, mc_samples=100):
        """Maximum Value Entropy Search (MES)"""
        # Approximate the distribution of the maximum value
        y_max_samples = []
        
        for _ in range(mc_samples):
            # Sample from GP posterior
            mean, cov = gp_model.predict_full_covariance(gp_model.X_train)
            y_sample = torch.multivariate_normal(mean, cov)
            y_max_samples.append(torch.max(y_sample))
        
        y_max_samples = torch.stack(y_max_samples)
        
        # Compute entropy reduction for each candidate
        mes_values = torch.zeros(len(X_candidate))
        
        for i, x in enumerate(X_candidate):
            # Predict at candidate
            mean_x, std_x = gp_model.predict(x.unsqueeze(0))
            
            # Compute mutual information
            # I(y_max; y_x) = H(y_x) - H(y_x | y_max)
            
            # H(y_x) - marginal entropy
            h_yx = 0.5 * torch.log(2 * np.pi * np.e * std_x**2)
            
            # H(y_x | y_max) - conditional entropy (approximated)
            h_yx_given_ymax = 0
            for y_max in y_max_samples:
                # Approximate conditional distribution
                conditional_mean = mean_x  # Simplified
                conditional_std = std_x * (1 - 0.1 * torch.sigmoid(y_max - mean_x))
                h_yx_given_ymax += 0.5 * torch.log(2 * np.pi * np.e * conditional_std**2)
            
            h_yx_given_ymax /= len(y_max_samples)
            
            mes_values[i] = h_yx - h_yx_given_ymax
        
        return mes_values
```

#### Multi-Task Bayesian Optimization
```python
class MultiTaskBayesianOptimizer:
    """Bayesian optimization with transfer learning across tasks"""
    def __init__(self, task_kernels, task_similarity_kernel):
        self.task_kernels = task_kernels
        self.similarity_kernel = task_similarity_kernel
        self.task_data = {}
        
    def add_observation(self, task_id, x, y):
        """Add observation for specific task"""
        if task_id not in self.task_data:
            self.task_data[task_id] = {'X': [], 'y': []}
        
        self.task_data[task_id]['X'].append(x)
        self.task_data[task_id]['y'].append(y)
    
    def predict(self, task_id, X_test):
        """Predict using multi-task GP"""
        # Build multi-task kernel
        mt_kernel = self._build_multitask_kernel()
        
        # Aggregate all task data
        X_all = []
        y_all = []
        task_indices = []
        
        for tid, data in self.task_data.items():
            X_all.extend(data['X'])
            y_all.extend(data['y'])
            task_indices.extend([tid] * len(data['X']))
        
        # Predict for target task
        X_all = torch.stack(X_all)
        y_all = torch.tensor(y_all)
        
        # Multi-task GP prediction
        K = mt_kernel(X_all, X_all, task_indices, task_indices)
        K_star = mt_kernel(X_test, X_all, [task_id] * len(X_test), task_indices)
        
        # Compute predictions
        L = torch.cholesky(K + 1e-6 * torch.eye(len(K)))
        alpha = torch.cholesky_solve(y_all.unsqueeze(-1), L)
        
        mean = K_star @ alpha
        
        # Compute variance
        v = torch.cholesky_solve(K_star.t(), L)
        K_star_star = mt_kernel(X_test, X_test, [task_id] * len(X_test), [task_id] * len(X_test))
        var = torch.diag(K_star_star) - torch.sum(K_star * v.t(), dim=1)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        
        return mean.squeeze(), std
    
    def transfer_knowledge(self, source_task, target_task, n_transfer=5):
        """Transfer promising points from source to target task"""
        if source_task not in self.task_data:
            return []
        
        # Find best points in source task
        source_y = torch.tensor(self.task_data[source_task]['y'])
        top_indices = torch.topk(source_y, min(n_transfer, len(source_y))).indices
        
        # Transfer points adjusted by task similarity
        similarity = self.similarity_kernel(source_task, target_task)
        transferred_points = [
            self.task_data[source_task]['X'][i] * similarity
            for i in top_indices
        ]
        
        return transferred_points
```

### Phase 3: Integration
- Hyperparameter tuning
- AutoML integration
- Distributed optimization
- Real-time adaptation

## Acceptance Criteria
1. ✅ 5x faster than grid search
2. ✅ Convergence within 50 iterations
3. ✅ Uncertainty estimates provided
4. ✅ Support for 20+ dimensions
5. ✅ Handle noisy objectives

## Example Usage
```python
from processors.audio_enhancement.optimization import BayesianOptimizer

# Define parameter bounds
bounds = {
    'noise_reduction': (0.0, 1.0),
    'enhancement_strength': (0.1, 2.0),
    'vad_sensitivity': (0.2, 0.8),
    'separation_threshold': (0.1, 0.9)
}

# Initialize optimizer
optimizer = BayesianOptimizer(bounds, acquisition='ei')

# Optimization loop
for i in range(50):
    # Get next suggestion
    params = optimizer.suggest_next()
    
    # Evaluate objective
    enhanced = enhance_audio(test_audio, **params)
    quality = evaluate_quality(enhanced)
    
    # Update model
    optimizer.update(params, quality)
    
    print(f"Iteration {i}: Quality = {quality:.3f}")

# Get best parameters
best_params = optimizer.get_best()
print(f"Best parameters: {best_params}")

# Visualize optimization
optimizer.plot_convergence()
optimizer.plot_acquisition_function()
optimizer.plot_uncertainty_map()
```

## Dependencies
- Scikit-optimize for Bayesian opt
- GPyTorch for advanced GP
- NumPy for numerics
- Matplotlib for visualization
- Botorch for advanced features

## Performance Targets
- Suggestion time: < 100ms
- Model update: < 200ms
- Memory usage: < 500MB
- Convergence: < 50 iterations

## Notes
- Consider warm starting strategies
- Implement trust region methods
- Support for discrete parameters
- Enable multi-fidelity optimization

## Advanced Theoretical Extensions

### Trust Region Bayesian Optimization
```python
class TrustRegionBO:
    """Trust region methods for safe Bayesian optimization"""
    def __init__(self, initial_radius=0.1, success_threshold=0.2):
        self.radius = initial_radius
        self.success_threshold = success_threshold
        self.center = None
        self.best_value = -np.inf
        
    def update_trust_region(self, x_new, y_new, y_pred):
        """Update trust region based on prediction accuracy"""
        # Compute improvement ratio
        actual_improvement = y_new - self.best_value
        predicted_improvement = y_pred - self.best_value
        
        if predicted_improvement > 0:
            rho = actual_improvement / predicted_improvement
        else:
            rho = 0
        
        # Update radius based on ratio
        if rho < 0.25:
            # Poor prediction, shrink region
            self.radius *= 0.5
        elif rho > 0.75 and np.linalg.norm(x_new - self.center) > 0.8 * self.radius:
            # Good prediction at boundary, expand region
            self.radius = min(2 * self.radius, 1.0)
        
        # Update center if improvement
        if actual_improvement > 0:
            self.center = x_new
            self.best_value = y_new
    
    def constrain_acquisition(self, acquisition_func):
        """Constrain acquisition function to trust region"""
        def constrained_acq(x):
            if np.linalg.norm(x - self.center) <= self.radius:
                return acquisition_func(x)
            else:
                # Penalize points outside trust region
                distance = np.linalg.norm(x - self.center)
                penalty = -1000 * (distance - self.radius)**2
                return penalty
        
        return constrained_acq
```

### Multi-Fidelity Bayesian Optimization
```python
class MultiFidelityBO:
    """Multi-fidelity optimization with cost-aware acquisition"""
    def __init__(self, fidelities, costs):
        self.fidelities = fidelities  # List of fidelity levels
        self.costs = costs  # Computational cost for each fidelity
        self.data = {f: {'X': [], 'y': []} for f in fidelities}
        
    def cost_aware_acquisition(self, x, fidelity, gp_models):
        """Acquisition that considers both information gain and cost"""
        # Predict at highest fidelity using all lower fidelity data
        mean_high, std_high = self._predict_high_fidelity(x, gp_models)
        
        # Information gain per unit cost
        info_gain = self._compute_information_gain(x, fidelity, gp_models)
        cost = self.costs[fidelity]
        
        return info_gain / cost
    
    def _predict_high_fidelity(self, x, gp_models):
        """Predict high-fidelity output using multi-fidelity GP"""
        # Use linear model of coregionalization
        predictions = []
        weights = []
        
        for fidelity in self.fidelities:
            if self.data[fidelity]['X']:
                mean, std = gp_models[fidelity].predict(x)
                
                # Weight by inverse variance and fidelity correlation
                weight = 1 / (std**2) * self._get_fidelity_correlation(fidelity)
                predictions.append(mean * weight)
                weights.append(weight)
        
        if weights:
            # Weighted combination
            combined_mean = sum(predictions) / sum(weights)
            combined_std = 1 / np.sqrt(sum(weights))
            return combined_mean, combined_std
        else:
            # Prior prediction
            return 0, 1
    
    def _get_fidelity_correlation(self, fidelity):
        """Estimate correlation between fidelity and highest fidelity"""
        # In practice, learn this from data
        fidelity_index = self.fidelities.index(fidelity)
        return 0.5 + 0.5 * (fidelity_index / len(self.fidelities))
```