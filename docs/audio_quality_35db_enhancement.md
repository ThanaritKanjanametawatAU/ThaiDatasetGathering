# Audio Quality Enhancement to 35dB SNR

## Overview

The 35dB SNR enhancement feature is designed to preprocess audio data for high-quality voice cloning and TTS training. It ensures that audio samples achieve a minimum Signal-to-Noise Ratio (SNR) of 35dB while preserving naturalness and avoiding over-denoising artifacts.

## Features

- **Target SNR Achievement**: Enhances audio to reach 35dB SNR
- **Multi-Stage Processing**: Uses adaptive spectral subtraction, Wiener filtering, harmonic enhancement, and perceptual post-processing
- **Quality Preservation**: Monitors naturalness to prevent over-processing
- **GPU Acceleration**: Optimized batch processing for efficient computation
- **Comprehensive Metrics**: Tracks SNR, PESQ, STOI, and naturalness scores

## Usage

### Command Line

```bash
# Enable 35dB enhancement with default settings
python main.py --fresh --all --enable-35db-enhancement

# Customize enhancement parameters
python main.py --fresh --all \
    --enable-35db-enhancement \
    --target-snr 35 \
    --min-acceptable-snr 30 \
    --snr-success-rate 0.90 \
    --max-enhancement-iterations 3

# Include samples that fail to reach target (with metadata)
python main.py --fresh --all \
    --enable-35db-enhancement \
    --include-failed-samples
```

### Parameters

- `--enable-35db-enhancement`: Enable the 35dB SNR enhancement mode
- `--target-snr`: Target SNR in dB (default: 35.0)
- `--min-acceptable-snr`: Minimum acceptable SNR for inclusion (default: 30.0)
- `--snr-success-rate`: Target success rate for achieving SNR (default: 0.90)
- `--max-enhancement-iterations`: Maximum enhancement iterations per sample (default: 3)
- `--include-failed-samples`: Include samples that fail to reach target SNR with metadata

## Implementation Details

### Enhancement Pipeline

1. **SNR Measurement**: Initial assessment using Voice Activity Detection (VAD)
2. **Adaptive Spectral Subtraction**: Gentle noise reduction preserving speech
3. **Wiener Filtering**: Adaptive filtering based on noise characteristics
4. **Harmonic Enhancement**: Boosts speech harmonics while suppressing inter-harmonic noise
5. **Perceptual Post-Processing**: Improves naturalness with comfort noise and smoothing

### Quality Monitoring

The system continuously monitors:
- **SNR**: Signal-to-Noise Ratio in dB
- **PESQ**: Perceptual Evaluation of Speech Quality (target: >3.5)
- **STOI**: Short-Time Objective Intelligibility (target: >0.85)
- **Naturalness Score**: Custom metric to prevent over-processing (target: >0.85)

### Dataset Schema Updates

When 35dB enhancement is enabled, the dataset includes additional fields:

```python
{
    "ID": str,
    "speaker_id": str,
    "Language": str,
    "audio": {...},
    "transcript": str,
    "length": float,
    "dataset_name": str,
    "confidence_score": float,
    "snr_db": float,  # NEW: Signal-to-Noise Ratio in dB
    "audio_quality_metrics": {  # NEW: Quality metrics
        "pesq": float,
        "stoi": float,
        "mos_estimate": float
    },
    "enhancement_metadata": {
        "enhancement_applied": bool,
        "naturalness_score": float,
        "snr_improvement": float,
        "target_achieved": bool,
        "iterations": int,
        "stages_applied": list
    }
}
```

## Performance

- **Processing Speed**: <1.8 seconds per sample
- **Batch Size**: 32 samples (GPU), dynamically adjusted based on memory
- **Memory Usage**: Optimized for 32GB VRAM
- **Success Rate**: Target 90% of samples achieving 35dB SNR

## Quality Analysis

### Running Quality Analysis

```bash
# Extract samples for analysis
python scripts/analysis/extract_quality_samples.py --samples 1000

# Analyze current audio quality
python scripts/analysis/analyze_current_quality.py

# Results will be in test_sets/quality_analysis/reports/
```

### Expected Results

Based on the analysis, the enhancement should achieve:
- **Mean SNR Improvement**: 10-15 dB
- **Success Rate**: >90% samples reaching 35dB
- **PESQ Score**: >3.5 (Good quality)
- **STOI Score**: >0.85 (High intelligibility)

## Best Practices

1. **Test on Small Sample**: Use `--sample` mode first to verify quality
2. **Monitor Metrics**: Check enhancement_metrics/ directory for detailed logs
3. **Adjust Thresholds**: Fine-tune based on your specific use case
4. **GPU Memory**: Monitor GPU usage and adjust batch size if needed

## Troubleshooting

### Common Issues

1. **Low Success Rate**
   - Check input audio quality
   - Increase `--max-enhancement-iterations`
   - Lower `--min-acceptable-snr` threshold

2. **Over-processing**
   - Monitor naturalness scores
   - Adjust enhancement configuration in config.py
   - Use `--enhancement-level moderate` instead of aggressive

3. **Memory Issues**
   - Reduce batch size
   - Use CPU fallback with `--no-gpu`
   - Process datasets separately

## Configuration

The feature can be configured in `config.py`:

```python
ENHANCEMENT_35DB_CONFIG = {
    "enabled": False,  # Set via command line
    "target_snr_db": 35.0,
    "min_acceptable_snr_db": 30.0,
    "target_success_rate": 0.90,
    "max_enhancement_passes": 3,
    "naturalness_weights": {
        "preserve_harmonics": 0.8,
        "suppress_noise": 0.2
    },
    "perceptual_limits": {
        "min_pesq": 3.5,
        "min_stoi": 0.85,
        "max_spectral_distortion": 0.15
    }
}
```

## Future Improvements

- Deep learning-based denoisers integration
- Real-time adaptation based on content type
- Multi-modal enhancement using visual cues
- Custom models for Thai speech characteristics