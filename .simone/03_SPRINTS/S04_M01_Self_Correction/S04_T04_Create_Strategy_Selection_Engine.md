# Task S04_T04: Create Strategy Selection Engine

## Task Overview
Create an intelligent strategy selection engine that automatically chooses the best audio enhancement approach based on input characteristics and quality requirements.

## Technical Requirements

### Core Implementation
- **Strategy Selection Engine** (`processors/audio_enhancement/strategy/strategy_selector.py`)
  - Multi-armed bandit algorithms
  - Contextual decision making
  - Performance tracking
  - Dynamic adaptation

### Key Features
1. **Strategy Options**
   - Enhancement pipelines
   - Processing orders
   - Algorithm selections
   - Parameter presets

2. **Selection Methods**
   - Rule-based selection
   - Machine learning models
   - Reinforcement learning
   - Ensemble decisions

3. **Adaptation Mechanisms**
   - Online learning
   - Context awareness
   - Performance feedback
   - Exploration vs exploitation

## TDD Requirements

### Test Structure
```
tests/test_strategy_selector.py
- test_strategy_selection()
- test_contextual_decisions()
- test_performance_tracking()
- test_adaptation_mechanism()
- test_exploration_exploitation()
- test_ensemble_selection()
```

### Test Data Requirements
- Various audio types
- Strategy performance data
- Context features
- Historical outcomes

## Implementation Approach

### Phase 1: Core Engine
```python
class StrategySelector:
    def __init__(self, strategies, method='contextual_bandit'):
        self.strategies = strategies
        self.method = method
        self.performance_tracker = PerformanceTracker()
        self.model = self._init_model()
        
    def select_strategy(self, audio_features, context=None):
        # Select optimal strategy
        pass
    
    def update_performance(self, strategy_id, outcome):
        # Update strategy performance
        pass
    
    def get_strategy_stats(self):
        # Return performance statistics
        pass
```

### Phase 2: Advanced Selection
- Neural bandits
- Meta-learning
- Transfer learning
- Multi-objective selection

#### Multi-Armed Bandit Algorithms
```python
class NeuralContextualBandit:
    """Neural network-based contextual bandit for strategy selection"""
    def __init__(self, context_dim, n_arms, hidden_dims=[128, 64]):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.networks = self._build_arm_networks(hidden_dims)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.thompson_sampling = True
        
    def _build_arm_networks(self, hidden_dims):
        """Build separate neural networks for each arm"""
        networks = []
        for _ in range(self.n_arms):
            layers = []
            prev_dim = self.context_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))  # Output reward prediction
            
            network = nn.Sequential(*layers)
            networks.append(network)
        
        return nn.ModuleList(networks)
    
    def select_arm(self, context, exploration_mode='thompson'):
        """Select strategy using neural contextual bandit"""
        context_tensor = torch.FloatTensor(context)
        
        if exploration_mode == 'thompson':
            # Thompson sampling with dropout uncertainty
            rewards = []
            uncertainties = []
            
            for network in self.networks:
                network.train()  # Enable dropout
                
                # Multiple forward passes for uncertainty estimation
                predictions = []
                for _ in range(10):
                    with torch.no_grad():
                        pred = network(context_tensor)
                        predictions.append(pred.item())
                
                mean_reward = np.mean(predictions)
                std_reward = np.std(predictions)
                
                rewards.append(mean_reward)
                uncertainties.append(std_reward)
            
            # Thompson sampling
            sampled_rewards = [
                np.random.normal(r, u) for r, u in zip(rewards, uncertainties)
            ]
            return np.argmax(sampled_rewards)
            
        elif exploration_mode == 'ucb':
            # Neural UCB
            network.eval()  # Disable dropout
            
            ucb_values = []
            for i, network in enumerate(self.networks):
                with torch.no_grad():
                    mean_reward = network(context_tensor).item()
                    
                # Compute confidence bound
                n_samples = len(self.replay_buffer.get_arm_data(i))
                confidence = np.sqrt(2 * np.log(len(self.replay_buffer)) / max(n_samples, 1))
                
                ucb = mean_reward + confidence
                ucb_values.append(ucb)
            
            return np.argmax(ucb_values)
    
    def update(self, context, arm, reward):
        """Update neural bandit with observed reward"""
        self.replay_buffer.add(context, arm, reward)
        
        # Train network for selected arm
        if len(self.replay_buffer) > 100:
            self._train_networks()
    
    def _train_networks(self, batch_size=32, n_epochs=5):
        """Train arm networks on replay buffer"""
        for arm in range(self.n_arms):
            arm_data = self.replay_buffer.get_arm_data(arm)
            if len(arm_data) < batch_size:
                continue
            
            optimizer = torch.optim.Adam(self.networks[arm].parameters(), lr=0.001)
            
            for _ in range(n_epochs):
                # Sample batch
                batch = random.sample(arm_data, batch_size)
                contexts = torch.FloatTensor([b[0] for b in batch])
                rewards = torch.FloatTensor([b[1] for b in batch])
                
                # Train step
                predictions = self.networks[arm](contexts).squeeze()
                loss = F.mse_loss(predictions, rewards)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

#### Contextual Decision Making with Guarantees
```python
class LinUCBSelector:
    """Linear UCB with theoretical guarantees for strategy selection"""
    def __init__(self, d, n_arms, alpha=1.0):
        self.d = d  # Feature dimension
        self.n_arms = n_arms
        self.alpha = alpha
        
        # Initialize parameters for each arm
        self.A = [np.eye(d) for _ in range(n_arms)]  # Design matrix
        self.b = [np.zeros(d) for _ in range(n_arms)]  # Reward vector
        self.theta = [np.zeros(d) for _ in range(n_arms)]  # Parameters
        
    def select_arm(self, x_t):
        """Select arm with theoretical regret bound"""
        p_t = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # Compute parameter estimate
            A_inv = np.linalg.inv(self.A[a])
            self.theta[a] = A_inv @ self.b[a]
            
            # Compute UCB
            p_t[a] = self.theta[a].T @ x_t + self.alpha * np.sqrt(x_t.T @ A_inv @ x_t)
        
        return np.argmax(p_t)
    
    def update(self, x_t, a_t, r_t):
        """Update with observed reward"""
        self.A[a_t] += np.outer(x_t, x_t)
        self.b[a_t] += r_t * x_t
    
    def compute_regret_bound(self, T, delta=0.1):
        """Compute theoretical regret bound"""
        # LinUCB regret bound: O(d√(T log(T)))
        d = self.d
        K = self.n_arms
        
        # Set alpha for theoretical guarantee
        self.alpha = 1 + np.sqrt(np.log(2 * T * K / delta) / 2)
        
        # Expected regret bound
        regret_bound = 2 * self.alpha * d * np.sqrt(T * np.log(T))
        
        return regret_bound
```

#### Meta-Learning for Strategy Selection
```python
class MAMLStrategySelector:
    """Model-Agnostic Meta-Learning for rapid strategy adaptation"""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        
    def inner_loop_update(self, support_data):
        """Fast adaptation on new task"""
        # Clone model for inner loop
        fast_model = copy.deepcopy(self.model)
        
        # Gradient steps on support data
        for x, y in support_data:
            pred = fast_model(x)
            loss = F.mse_loss(pred, y)
            
            # Manual gradient update
            grads = torch.autograd.grad(loss, fast_model.parameters())
            
            # Update parameters
            for param, grad in zip(fast_model.parameters(), grads):
                param.data -= self.inner_lr * grad
        
        return fast_model
    
    def meta_train_step(self, task_batch):
        """Meta-training step across multiple tasks"""
        meta_loss = 0
        
        for task in task_batch:
            # Split into support and query
            support_data = task['support']
            query_data = task['query']
            
            # Inner loop adaptation
            adapted_model = self.inner_loop_update(support_data)
            
            # Evaluate on query set
            for x, y in query_data:
                pred = adapted_model(x)
                meta_loss += F.mse_loss(pred, y)
        
        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_to_new_scenario(self, new_scenario_data, n_steps=5):
        """Quickly adapt to new audio enhancement scenario"""
        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(n_steps):
            for x, y in new_scenario_data:
                pred = adapted_model(x)
                loss = F.mse_loss(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return adapted_model
```

#### Exploration vs Exploitation Theory
```python
class OptimalExplorationStrategy:
    """Theoretically optimal exploration strategies"""
    
    @staticmethod
    def gittins_index(successes, failures, discount_factor=0.9):
        """Compute Gittins index for optimal exploration"""
        # Approximation of Gittins index for Beta(s+1, f+1) distribution
        n = successes + failures
        if n == 0:
            return float('inf')  # Explore unknown arms
        
        p_hat = successes / n
        
        # Whittle's approximation
        c = np.sqrt(2 * np.log(1 / (1 - discount_factor)) / n)
        gittins = p_hat + c
        
        return gittins
    
    @staticmethod
    def information_directed_sampling(posterior_distributions, action_costs):
        """IDS for optimal information-cost tradeoff"""
        n_actions = len(posterior_distributions)
        
        # Compute information gain for each action
        info_gains = []
        expected_costs = []
        
        for i in range(n_actions):
            # KL divergence between posterior and prior
            info_gain = posterior_distributions[i].kl_divergence_to_prior()
            info_gains.append(info_gain)
            
            # Expected cost
            expected_cost = action_costs[i]
            expected_costs.append(expected_cost)
        
        # Information ratio for each action
        info_ratios = [
            cost**2 / (info_gain + 1e-10) 
            for info_gain, cost in zip(info_gains, expected_costs)
        ]
        
        # Select action with minimum information ratio
        return np.argmin(info_ratios)
```

### Phase 3: Integration
- Real-time selection
- A/B testing framework
- Monitoring dashboard
- API deployment

## Acceptance Criteria
1. ✅ Strategy selection accuracy > 85%
2. ✅ Adaptation within 10 iterations
3. ✅ Support for 10+ strategies
4. ✅ Real-time decision making
5. ✅ Explainable selections

## Example Usage
```python
from processors.audio_enhancement.strategy import StrategySelector

# Define available strategies
strategies = {
    'aggressive': AggressiveEnhancement(),
    'conservative': ConservativeEnhancement(),
    'balanced': BalancedEnhancement(),
    'neural': NeuralEnhancement(),
    'hybrid': HybridEnhancement()
}

# Initialize selector
selector = StrategySelector(strategies, method='contextual_bandit')

# Extract audio features
features = extract_audio_features(input_audio)
context = {
    'scenario': 'interview',
    'quality_target': 'high',
    'processing_time': 'fast'
}

# Select strategy
selected = selector.select_strategy(features, context)
print(f"Selected strategy: {selected.name}")
print(f"Confidence: {selected.confidence:.2f}")
print(f"Reasoning: {selected.explanation}")

# Apply strategy
enhanced = selected.strategy.process(input_audio)

# Update performance
quality_score = evaluate_quality(enhanced)
selector.update_performance(selected.name, quality_score)

# View statistics
stats = selector.get_strategy_stats()
for strategy, stat in stats.items():
    print(f"{strategy}: Success rate = {stat.success_rate:.2f}, Avg quality = {stat.avg_quality:.2f}")
```

## Dependencies
- Scikit-learn for ML models
- Vowpal Wabbit for bandits
- NumPy for calculations
- Pandas for tracking
- TensorFlow for neural models

## Performance Targets
- Selection time: < 10ms
- Model update: < 50ms
- Memory usage: < 200MB
- Convergence: < 100 selections

## Notes
- Balance exploration and exploitation
- Consider cold start problem
- Support for new strategies
- Enable strategy composition

## Advanced Strategy Selection Theory

### Regret Minimization Framework
```python
class RegretMinimizationFramework:
    """Theoretical framework for regret minimization in strategy selection"""
    def __init__(self, n_strategies):
        self.n_strategies = n_strategies
        self.cumulative_regret = 0
        self.strategy_regrets = np.zeros(n_strategies)
        
    def compute_counterfactual_regret(self, chosen_strategy, rewards):
        """Compute counterfactual regret for each strategy"""
        chosen_reward = rewards[chosen_strategy]
        
        for i in range(self.n_strategies):
            regret = rewards[i] - chosen_reward
            self.strategy_regrets[i] += max(0, regret)
        
        # Update cumulative regret
        best_reward = np.max(rewards)
        self.cumulative_regret += best_reward - chosen_reward
    
    def get_strategy_probabilities(self):
        """Get strategy probabilities using regret matching"""
        positive_regrets = np.maximum(self.strategy_regrets, 0)
        sum_regrets = np.sum(positive_regrets)
        
        if sum_regrets > 0:
            return positive_regrets / sum_regrets
        else:
            # Uniform distribution if no regrets
            return np.ones(self.n_strategies) / self.n_strategies
    
    def theoretical_regret_bound(self, T):
        """Compute theoretical regret bound"""
        # For regret matching, bound is O(√T)
        return 2 * np.sqrt(self.n_strategies * T)
```

### Hierarchical Strategy Selection
```python
class HierarchicalStrategySelector:
    """Hierarchical selection for complex strategy spaces"""
    def __init__(self, strategy_tree):
        self.strategy_tree = strategy_tree
        self.node_selectors = {}
        self._initialize_selectors(strategy_tree)
        
    def _initialize_selectors(self, node, path=""):
        """Initialize selectors for each node in hierarchy"""
        if node.is_leaf():
            return
        
        # Create selector for this node
        self.node_selectors[path] = LinUCBSelector(
            d=node.feature_dim,
            n_arms=len(node.children)
        )
        
        # Recursively initialize children
        for i, child in enumerate(node.children):
            child_path = f"{path}/{i}" if path else str(i)
            self._initialize_selectors(child, child_path)
    
    def select_strategy(self, context):
        """Hierarchical strategy selection"""
        current_node = self.strategy_tree
        path = ""
        selections = []
        
        while not current_node.is_leaf():
            # Select child using bandit at this level
            selector = self.node_selectors[path]
            child_idx = selector.select_arm(context)
            
            selections.append((path, child_idx))
            
            # Move to selected child
            current_node = current_node.children[child_idx]
            path = f"{path}/{child_idx}" if path else str(child_idx)
        
        return current_node.strategy, selections
    
    def update_hierarchical(self, selections, reward):
        """Update all selectors in the selection path"""
        # Propagate reward up the hierarchy
        for path, child_idx in selections:
            selector = self.node_selectors[path]
            
            # Can use different credit assignment strategies
            # Here we use full reward propagation
            selector.update(context, child_idx, reward)
```