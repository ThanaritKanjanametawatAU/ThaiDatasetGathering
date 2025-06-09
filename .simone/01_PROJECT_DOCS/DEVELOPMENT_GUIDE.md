# Development Guide - Thai Audio Dataset Collection

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for audio enhancement)
- 100GB+ free disk space for caching
- HuggingFace account with accepted PyAnnote model access

### Setup
```bash
# Clone repository
git clone [repository_url]
cd ThaiDatasetGathering

# Create conda environment
conda create -n thaidataset python=3.10
conda activate thaidataset

# Install dependencies
pip install -r requirements.txt

# Configure HuggingFace token in .env
echo "HF_HUB_ENABLE_HF_TRANSFER=1" > .env
echo "hf_token = your_token_here" >> .env
```

## Development Workflow

### 1. Feature Development (TDD Approach)

```bash
# Create test first
touch tests/test_new_feature.py

# Write failing tests
# Implement feature
# Run tests until passing
pytest tests/test_new_feature.py -v

# Run full test suite
pytest tests/
```

### 2. Audio Enhancement Development

When working on audio enhancement:
- Test with provided fixtures in `tests/fixtures/`
- Use `analyze_*.py` scripts for debugging
- Monitor with enhancement dashboard when enabled

### 3. Speaker Identification

Critical verification pattern:
- Samples S1-S8 and S10 must cluster together
- Sample S9 must be in a different cluster
- This verifies clustering algorithm correctness

## Code Guidelines

### General Principles (from CLAUDE.md)
1. **Preserve Existing Features** - Never break existing functionality
2. **Descriptive Names** - Use clear, meaningful variable names
3. **Edit Over Create** - Modify existing files rather than creating new ones
4. **Test Before Push** - Always run tests before committing

### Code Style
```python
# Good: Descriptive variable names
secondary_speaker_threshold = 0.7
primary_speaker_segments = detect_primary_speaker(audio)

# Bad: Unclear names
thresh = 0.7
ps = detect_ps(a)
```

### Error Handling
```python
try:
    # Risky operation
    result = process_audio(audio, sample_rate)
except Exception as e:
    logger.error(f"Audio processing failed: {e}")
    # Fallback behavior
    result = audio  # Return original if processing fails
```

## Testing Guidelines

### Running Tests
```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_audio_enhancement_integration.py

# With coverage
pytest tests/ --cov=processors --cov-report=html

# Verbose output
pytest tests/ -v
```

### Test Categories
1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Multi-component workflows  
3. **E2E Tests** - Full pipeline validation
4. **Performance Tests** - Speed and resource usage

### Test Data
- Synthetic signals in `tests/fixtures/synthetic_signals/`
- Real samples in `tests/fixtures/real_samples/`
- Noise profiles in `tests/fixtures/noise_profiles/`

## Debugging Tools

### Analysis Scripts
```bash
# Analyze S5 secondary speaker issue
python analyze_s5_issue.py

# Check energy calculations
python analyze_energy_calculation.py

# Debug speaker separation
python debug_separation_details.py
```

### Monitoring
```bash
# Enable dashboard during processing
python main.py --enhancement-dashboard

# Check metrics
cat enhancement_metrics/summary.json | jq
```

## Common Tasks

### Processing New Dataset
```bash
# Add to DATASET_CONFIG in config.py
# Create processor in processors/
# Test with sample data
python main.py YourDataset --sample --sample-size 10
```

### Updating Enhancement Algorithm
1. Review current implementation in `processors/audio_enhancement/`
2. Write tests for new behavior
3. Implement changes
4. Verify with real samples
5. Update documentation

### Fixing Speaker Clustering
1. Check threshold in config (default: 0.7)
2. Verify embedding model is loaded
3. Test with known samples (S1-S10)
4. Adjust HDBSCAN parameters if needed

## Deployment

### Local Testing
```bash
# Quick test with 10 samples
./main.sh  # Uses 100 samples by default

# Custom configuration
python main.py GigaSpeech2 \
    --sample --sample-size 50 \
    --enhancement-level moderate \
    --no-upload
```

### Production Run
```bash
# Full dataset processing
python main.py --all \
    --hf-repo Thanarit/Thai-Voice \
    --streaming \
    --enable-speaker-id \
    --enable-stt \
    --enable-audio-enhancement \
    --enable-35db-enhancement
```

## Troubleshooting

### Common Issues

1. **PyAnnote Authentication**
   - Ensure HF token is in .env file
   - Accept model access on HuggingFace
   - Check token has read permissions

2. **Memory Issues**
   - Use streaming mode
   - Reduce batch sizes
   - Clear cache regularly

3. **GPU Errors**
   - Check CUDA compatibility
   - Use CPU fallback: `--no-enhancement-gpu`

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints in code
logger.debug(f"Audio shape: {audio.shape}, SR: {sample_rate}")
```

## Contributing

1. Fork repository
2. Create feature branch
3. Follow TDD approach
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

## Resources

- Project Documentation: `.simone/01_PROJECT_DOCS/`
- Technical Architecture: `docs/architecture.md`
- Recent Changes: `CHANGELOG.md`
- Task Tracking: `.simone/04_GENERAL_TASKS/`