# Task S05_T01: Build Test Case Generator

## Task Overview
Build an intelligent test case generator that automatically creates comprehensive test scenarios for audio enhancement validation, including edge cases and regression tests.

## Technical Requirements

### Core Implementation
- **Test Case Generator** (`tests/automation/test_case_generator.py`)
  - Synthetic audio generation
  - Scenario-based test creation
  - Edge case identification
  - Test data augmentation

### Key Features
1. **Test Audio Generation**
   - Clean speech synthesis
   - Noise injection
   - Multi-speaker scenarios
   - Artifact simulation

2. **Scenario Creation**
   - Real-world scenarios
   - Edge cases
   - Stress tests
   - Regression scenarios

3. **Test Organization**
   - Categorized test suites
   - Difficulty levels
   - Coverage tracking
   - Metadata annotation

## TDD Requirements

### Test Structure
```
tests/test_test_case_generator.py
- test_audio_generation()
- test_scenario_creation()
- test_edge_case_coverage()
- test_metadata_generation()
- test_reproducibility()
- test_test_suite_balance()
```

### Test Data Requirements
- Seed audio samples
- Noise profiles
- Scenario templates
- Quality targets

## Implementation Approach

### Phase 1: Core Generator
```python
class TestCaseGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        self.audio_synthesizer = AudioSynthesizer()
        self.scenario_builder = ScenarioBuilder()
        
    def generate_test_case(self, scenario_type, difficulty='medium'):
        # Generate single test case
        pass
    
    def generate_test_suite(self, size=100, coverage_targets=None):
        # Generate comprehensive test suite
        pass
    
    def generate_edge_cases(self, categories=None):
        # Focus on edge cases
        pass
```

### Phase 2: Advanced Generation
- GAN-based audio synthesis
- Adversarial test cases
- Property-based testing
- Mutation testing

#### Fuzzing Algorithms for Audio
```python
class AudioFuzzer:
    """Advanced fuzzing for audio enhancement testing"""
    def __init__(self, seed_corpus, mutation_rate=0.1):
        self.seed_corpus = seed_corpus
        self.mutation_rate = mutation_rate
        self.interesting_inputs = PriorityQueue()
        self.coverage_map = {}
        
    def mutate_audio(self, audio, mutation_type='random'):
        """Apply intelligent mutations to audio"""
        if mutation_type == 'random':
            mutations = [
                self._add_noise,
                self._time_stretch,
                self._pitch_shift,
                self._add_clicks,
                self._add_distortion,
                self._phase_corruption,
                self._spectral_mutation
            ]
            mutation_func = random.choice(mutations)
            return mutation_func(audio)
            
        elif mutation_type == 'guided':
            # Use coverage-guided mutation
            return self._coverage_guided_mutation(audio)
            
        elif mutation_type == 'adversarial':
            # Generate adversarial examples
            return self._adversarial_mutation(audio)
    
    def _coverage_guided_mutation(self, audio):
        """Mutate based on code coverage feedback"""
        # Analyze current coverage
        uncovered_paths = self._get_uncovered_paths()
        
        # Select mutation targeting uncovered code
        if 'extreme_snr' in uncovered_paths:
            # Generate extreme SNR case
            noise_level = np.random.choice([0.001, 10.0])  # Very clean or very noisy
            return self._add_noise(audio, noise_level)
            
        elif 'long_silence' in uncovered_paths:
            # Insert long silence
            silence_duration = np.random.uniform(5, 30)  # seconds
            return self._insert_silence(audio, silence_duration)
            
        elif 'rapid_changes' in uncovered_paths:
            # Create rapid amplitude changes
            return self._create_rapid_changes(audio)
        
        # Default to random mutation
        return self.mutate_audio(audio, 'random')
    
    def _adversarial_mutation(self, audio):
        """Generate adversarial examples using gradient-based methods"""
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).requires_grad_(True)
        
        # Define adversarial objective (e.g., maximize enhancement error)
        def adversarial_loss(perturbed_audio):
            enhanced = enhance_audio(perturbed_audio)
            quality_score = compute_quality(enhanced)
            return -quality_score  # Minimize quality
        
        # Compute gradient
        loss = adversarial_loss(audio_tensor)
        loss.backward()
        
        # Apply perturbation in gradient direction
        epsilon = 0.01  # Perturbation magnitude
        perturbation = epsilon * audio_tensor.grad.sign()
        
        # Create adversarial example
        adversarial_audio = audio_tensor + perturbation
        adversarial_audio = torch.clamp(adversarial_audio, -1, 1)
        
        return adversarial_audio.detach().numpy()
    
    def differential_fuzzing(self, implementations):
        """Compare multiple implementations to find discrepancies"""
        discrepancies = []
        
        for audio in self.seed_corpus:
            # Mutate audio
            mutated = self.mutate_audio(audio)
            
            # Run through all implementations
            results = {}
            for name, impl in implementations.items():
                try:
                    results[name] = impl(mutated)
                except Exception as e:
                    results[name] = f"ERROR: {e}"
            
            # Check for discrepancies
            if self._has_discrepancy(results):
                discrepancies.append({
                    'input': mutated,
                    'results': results,
                    'discrepancy_score': self._compute_discrepancy_score(results)
                })
        
        return discrepancies
```

#### Property-Based Testing Strategies
```python
class PropertyBasedAudioTester:
    """Property-based testing for audio enhancement"""
    def __init__(self):
        self.properties = []
        self.register_default_properties()
        
    def register_default_properties(self):
        """Register fundamental properties that should always hold"""
        self.add_property(
            "energy_conservation",
            lambda inp, out: np.abs(np.sum(inp**2) - np.sum(out**2)) < 0.1 * np.sum(inp**2),
            "Output energy should be within 10% of input energy"
        )
        
        self.add_property(
            "no_new_frequencies",
            lambda inp, out: self._check_frequency_property(inp, out),
            "Enhancement should not introduce new frequency components"
        )
        
        self.add_property(
            "temporal_consistency",
            lambda inp, out: len(out) == len(inp),
            "Output length should match input length"
        )
        
        self.add_property(
            "bounded_output",
            lambda inp, out: np.all(np.abs(out) <= 1.0),
            "Output should be bounded between -1 and 1"
        )
        
        self.add_property(
            "deterministic",
            lambda inp: self._check_determinism(inp),
            "Same input should produce same output"
        )
    
    def add_property(self, name, predicate, description):
        """Add a new property to test"""
        self.properties.append({
            'name': name,
            'predicate': predicate,
            'description': description,
            'failures': []
        })
    
    def generate_test_inputs(self, strategy='hypothesis'):
        """Generate test inputs using various strategies"""
        if strategy == 'hypothesis':
            # Use hypothesis-style generators
            from hypothesis import strategies as st
            
            # Audio parameters
            duration = st.floats(min_value=0.1, max_value=60.0)
            sample_rate = st.sampled_from([8000, 16000, 22050, 44100, 48000])
            channels = st.integers(min_value=1, max_value=2)
            
            # Generate audio
            @st.composite
            def audio_strategy(draw):
                dur = draw(duration)
                sr = draw(sample_rate)
                ch = draw(channels)
                
                # Generate different types of audio
                audio_type = draw(st.sampled_from(['sine', 'noise', 'speech', 'mixed']))
                
                if audio_type == 'sine':
                    freq = draw(st.floats(min_value=20, max_value=20000))
                    return self._generate_sine(dur, sr, freq, ch)
                elif audio_type == 'noise':
                    return self._generate_noise(dur, sr, ch)
                elif audio_type == 'speech':
                    return self._generate_synthetic_speech(dur, sr, ch)
                else:
                    return self._generate_mixed_audio(dur, sr, ch)
            
            return audio_strategy()
            
        elif strategy == 'grammar':
            # Grammar-based generation
            return self._grammar_based_generation()
    
    def shrink_failing_input(self, audio, property_predicate):
        """Minimize failing test case"""
        # Delta debugging approach
        min_failing = audio.copy()
        
        # Try removing segments
        segment_size = len(audio) // 2
        
        while segment_size > 100:  # Minimum segment size
            improved = False
            
            for i in range(0, len(min_failing), segment_size):
                # Try removing this segment
                candidate = np.concatenate([
                    min_failing[:i],
                    min_failing[i + segment_size:]
                ])
                
                if len(candidate) > 0 and not property_predicate(candidate):
                    min_failing = candidate
                    improved = True
                    break
            
            if not improved:
                segment_size //= 2
        
        return min_failing
```

#### Mutation Testing for Audio Processing
```python
class AudioMutationTester:
    """Mutation testing to evaluate test quality"""
    def __init__(self, source_code, test_suite):
        self.source_code = source_code
        self.test_suite = test_suite
        self.mutation_operators = self._define_mutation_operators()
        
    def _define_mutation_operators(self):
        """Define mutation operators for audio processing code"""
        return [
            # Arithmetic mutations
            ('arithmetic', r'(\+)', '-'),  # Replace + with -
            ('arithmetic', r'(\*)', '/'),  # Replace * with /
            ('arithmetic', r'(>)', '>='),   # Boundary mutations
            
            # Audio-specific mutations
            ('audio_param', r'sample_rate\s*=\s*(\d+)', lambda m: f'sample_rate = {int(m.group(1)) * 2}'),
            ('audio_param', r'window_size\s*=\s*(\d+)', lambda m: f'window_size = {int(m.group(1)) // 2}'),
            ('audio_param', r'fft_size\s*=\s*(\d+)', lambda m: f'fft_size = {int(m.group(1)) * 2}'),
            
            # Algorithm mutations
            ('algorithm', r'np\.mean', 'np.median'),
            ('algorithm', r'np\.fft\.fft', 'np.fft.rfft'),
            ('algorithm', r'scipy\.signal\.butter', 'scipy.signal.cheby1'),
            
            # Constant mutations
            ('constant', r'0\.5', '0.7'),
            ('constant', r'1\.0', '0.9'),
            
            # Control flow mutations
            ('control', r'if\s+', 'if not '),
            ('control', r'while\s+', 'if '),
            
            # Remove statements
            ('deletion', r'^\s*.*normalize.*$', ''),  # Remove normalization
            ('deletion', r'^\s*.*validate.*$', ''),   # Remove validation
        ]
    
    def generate_mutants(self):
        """Generate mutated versions of the code"""
        mutants = []
        
        for op_type, pattern, replacement in self.mutation_operators:
            # Find all matches
            import re
            matches = list(re.finditer(pattern, self.source_code, re.MULTILINE))
            
            for match in matches:
                # Create mutant
                mutant_code = self.source_code[:match.start()]
                
                if callable(replacement):
                    mutant_code += replacement(match)
                else:
                    mutant_code += replacement
                
                mutant_code += self.source_code[match.end():]
                
                mutants.append({
                    'type': op_type,
                    'location': match.start(),
                    'original': match.group(0),
                    'mutation': replacement if not callable(replacement) else 'dynamic',
                    'code': mutant_code
                })
        
        return mutants
    
    def evaluate_test_suite(self):
        """Evaluate test suite effectiveness using mutation score"""
        mutants = self.generate_mutants()
        killed_mutants = 0
        survived_mutants = []
        
        for mutant in mutants:
            # Execute test suite on mutant
            if self._is_mutant_killed(mutant['code']):
                killed_mutants += 1
            else:
                survived_mutants.append(mutant)
        
        mutation_score = killed_mutants / len(mutants) if mutants else 0
        
        return {
            'mutation_score': mutation_score,
            'total_mutants': len(mutants),
            'killed': killed_mutants,
            'survived': len(survived_mutants),
            'survived_mutants': survived_mutants
        }
```

#### Test Oracle Generation
```python
class TestOracleGenerator:
    """Generate test oracles for audio enhancement validation"""
    def __init__(self):
        self.oracle_strategies = [
            self._reference_implementation_oracle,
            self._metamorphic_oracle,
            self._statistical_oracle,
            self._perceptual_oracle
        ]
        
    def _reference_implementation_oracle(self, test_input):
        """Use reference implementation as oracle"""
        # Multiple trusted implementations
        references = [
            lambda x: enhance_audio_ref1(x),
            lambda x: enhance_audio_ref2(x),
            lambda x: enhance_audio_ref3(x)
        ]
        
        results = [ref(test_input) for ref in references]
        
        # Voting mechanism
        if self._majority_agree(results):
            return self._get_majority_result(results)
        else:
            # Use statistical analysis for divergent results
            return self._analyze_divergent_results(results)
    
    def _metamorphic_oracle(self, test_input):
        """Metamorphic testing relations"""
        relations = []
        
        # Relation 1: Scaling invariance
        scale_factor = 0.5
        scaled_input = test_input * scale_factor
        scaled_output = enhance_audio(scaled_input)
        original_output = enhance_audio(test_input)
        
        relations.append({
            'name': 'scaling_invariance',
            'holds': np.allclose(scaled_output / scale_factor, original_output, rtol=0.1),
            'expected': original_output,
            'actual': scaled_output / scale_factor
        })
        
        # Relation 2: Additive property
        silence = np.zeros_like(test_input)
        enhanced_silence = enhance_audio(silence)
        mixed = enhance_audio(test_input + silence)
        
        relations.append({
            'name': 'additive_property',
            'holds': np.allclose(mixed, original_output + enhanced_silence, rtol=0.1),
            'expected': original_output + enhanced_silence,
            'actual': mixed
        })
        
        return relations
```

### Phase 3: Integration
- CI/CD integration
- Automated benchmarking
- Test database
- Coverage analytics

## Acceptance Criteria
1. ✅ Generate 1000+ unique test cases
2. ✅ Cover 20+ scenario types
3. ✅ Reproducible generation
4. ✅ Realistic audio quality
5. ✅ Comprehensive metadata

## Example Usage
```python
from tests.automation import TestCaseGenerator

# Initialize generator
generator = TestCaseGenerator(seed=42)

# Generate single test case
test_case = generator.generate_test_case(
    scenario_type='noisy_interview',
    difficulty='hard'
)

print(f"Test case: {test_case.id}")
print(f"Scenario: {test_case.scenario}")
print(f"Audio duration: {test_case.duration}s")
print(f"SNR: {test_case.snr} dB")
print(f"Expected quality: {test_case.expected_quality}")

# Generate test suite
test_suite = generator.generate_test_suite(
    size=100,
    coverage_targets={
        'scenarios': ['interview', 'meeting', 'phone_call'],
        'snr_range': (-5, 30),
        'speakers': [1, 2, 3, 4]
    }
)

print(f"\nGenerated {len(test_suite)} test cases")
print(f"Coverage report:")
for category, coverage in test_suite.get_coverage().items():
    print(f"  {category}: {coverage:.1f}%")

# Generate edge cases
edge_cases = generator.generate_edge_cases(
    categories=['extreme_noise', 'very_short', 'multiple_speakers']
)

# Save test suite
test_suite.save('test_suite_v1.json')
```

## Dependencies
- NumPy for generation
- SciPy for signal synthesis
- Librosa for audio
- Faker for metadata
- PyDub for audio manipulation

## Performance Targets
- Test case generation: < 100ms each
- Suite generation (1000): < 1 minute
- Memory usage: < 1GB
- Storage: < 10MB per 100 cases

## Notes
- Ensure diversity in test cases
- Balance difficulty levels
- Include failure scenarios
- Support custom scenarios

## Advanced Test Generation Theory

### Combinatorial Test Design
```python
class CombinatorialTestGenerator:
    """Generate minimal test sets with maximum coverage"""
    def __init__(self, parameters):
        self.parameters = parameters
        
    def generate_pairwise_tests(self):
        """Generate tests covering all parameter pairs"""
        from itertools import combinations
        
        # All parameter pairs
        param_names = list(self.parameters.keys())
        pairs = list(combinations(param_names, 2))
        
        # Build covering array
        test_cases = []
        uncovered_pairs = self._get_all_value_pairs(pairs)
        
        while uncovered_pairs:
            # Greedy: select test covering most uncovered pairs
            best_test = self._find_best_test(uncovered_pairs)
            test_cases.append(best_test)
            
            # Update uncovered pairs
            covered = self._get_covered_pairs(best_test, pairs)
            uncovered_pairs -= covered
        
        return test_cases
    
    def generate_n_way_tests(self, n):
        """Generate n-way combinatorial tests"""
        # Use IPOG algorithm for efficient generation
        return self._ipog_algorithm(n)
```