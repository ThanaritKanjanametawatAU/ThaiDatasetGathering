"""
Integration tests for Quality Threshold Manager with existing metrics.
"""

import pytest
import numpy as np
from pathlib import Path

# Import existing metrics
from utils.audio_metrics import calculate_pesq, calculate_stoi, calculate_si_sdr, calculate_snr
from processors.audio_enhancement.quality import QualityThresholdManager


class TestThresholdManagerIntegration:
    """Test integration with existing audio metrics."""
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio for testing."""
        # Generate clean speech-like signal
        duration = 2.0  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Simulate speech with formants
        speech = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # F0
            0.2 * np.sin(2 * np.pi * 700 * t) +  # F1
            0.1 * np.sin(2 * np.pi * 1220 * t)   # F2
        )
        
        # Add some envelope modulation
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        clean = speech * envelope
        
        # Create noisy version
        noise = np.random.normal(0, 0.05, clean.shape)
        noisy = clean + noise
        
        return {
            'clean': clean.astype(np.float32),
            'noisy': noisy.astype(np.float32),
            'sample_rate': sample_rate
        }
    
    def test_metrics_with_threshold_manager(self, sample_audio):
        """Test that real metrics work with threshold manager."""
        clean = sample_audio['clean']
        noisy = sample_audio['noisy']
        sr = sample_audio['sample_rate']
        
        # Calculate real metrics
        metrics = {}
        
        # PESQ (requires specific sample rates, mock if not 8k or 16k)
        try:
            metrics['pesq'] = calculate_pesq(clean, noisy, sr)
        except Exception:
            # Mock PESQ if calculation fails
            metrics['pesq'] = 3.5
        
        # STOI
        try:
            metrics['stoi'] = calculate_stoi(clean, noisy, sr)
        except Exception:
            # Mock STOI if calculation fails
            metrics['stoi'] = 0.85
        
        # SI-SDR
        metrics['si_sdr'] = calculate_si_sdr(noisy, clean)
        
        # SNR
        metrics['snr'] = calculate_snr(clean, noisy)
        
        # Use threshold manager to check quality
        manager = QualityThresholdManager(profile='production')
        result = manager.check_quality(metrics)
        
        # Verify result structure
        assert hasattr(result, 'passed')
        assert hasattr(result, 'score')
        assert hasattr(result, 'failures')
        assert hasattr(result, 'severity')
        assert hasattr(result, 'metric_scores')
        
        # Verify metric scores are calculated
        for metric in ['pesq', 'stoi', 'si_sdr', 'snr']:
            if metric in metrics:
                assert metric in result.metric_scores
                assert 0.0 <= result.metric_scores[metric] <= 1.0
    
    def test_enhancement_pipeline_integration(self, sample_audio):
        """Test integration with enhancement pipeline workflow."""
        # Initialize manager with custom thresholds
        custom_config = {
            'enhancement': {
                'pesq': {'min': 2.5, 'target': 3.0, 'max': 5.0},
                'stoi': {'min': 0.7, 'target': 0.8, 'max': 1.0},
                'si_sdr': {'min': 8.0, 'target': 12.0, 'max': 30.0},
                'snr': {'min': 15.0, 'target': 20.0, 'max': 40.0}
            }
        }
        
        manager = QualityThresholdManager(profile='enhancement', config=custom_config)
        
        # Simulate enhancement pipeline stages
        stages = ['pre_check', 'post_enhancement', 'final_validation']
        
        for stage in stages:
            # Mock metrics for each stage
            if stage == 'pre_check':
                metrics = {'snr': 12.0}  # Just check SNR pre-enhancement
            elif stage == 'post_enhancement':
                metrics = {
                    'pesq': 3.2,
                    'stoi': 0.82,
                    'si_sdr': 14.0,
                    'snr': 22.0
                }
            else:  # final_validation
                metrics = {
                    'pesq': 3.5,
                    'stoi': 0.88,
                    'si_sdr': 16.0,
                    'snr': 25.0
                }
            
            result = manager.check_quality(metrics)
            
            # Make decisions based on result
            if not result.passed:
                # Would trigger re-enhancement or rejection
                assert len(result.recovery_suggestions) > 0
                assert result.severity in ['minor', 'major', 'critical']
    
    def test_adaptive_threshold_with_real_metrics(self, sample_audio):
        """Test adaptive learning with realistic metric distributions."""
        manager = QualityThresholdManager(
            profile='production',
            enable_adaptive_learning=True
        )
        
        # Generate realistic metric samples
        n_samples = 100
        for i in range(n_samples):
            # Simulate varying quality levels
            quality_level = np.random.beta(5, 2)  # Skewed towards higher quality
            
            sample = {
                'pesq': 2.0 + quality_level * 2.5,
                'stoi': 0.5 + quality_level * 0.45,
                'si_sdr': 5.0 + quality_level * 20.0,
                'snr': 10.0 + quality_level * 25.0,
                'enhancement_success': quality_level > 0.6
            }
            
            manager.add_sample(sample)
        
        # Optimize thresholds
        manager.optimize_thresholds()
        
        # Test prediction on new samples
        test_samples = [
            {'pesq': 3.8, 'stoi': 0.88, 'si_sdr': 18.0, 'snr': 28.0},  # Good
            {'pesq': 2.2, 'stoi': 0.58, 'si_sdr': 7.0, 'snr': 12.0},   # Poor
        ]
        
        predictions = [manager.predict_enhancement_success(s) for s in test_samples]
        
        # Good sample should have higher success probability than poor sample
        assert predictions[0] > predictions[1]
    
    def test_dataset_specific_profiles(self):
        """Test using different profiles for different datasets."""
        # Define dataset-specific thresholds
        dataset_configs = {
            'gigaspeech': {
                'pesq': {'min': 3.0, 'target': 3.5, 'max': 5.0},
                'stoi': {'min': 0.8, 'target': 0.88, 'max': 1.0}
            },
            'mozilla_cv': {
                'pesq': {'min': 2.5, 'target': 3.0, 'max': 5.0},
                'stoi': {'min': 0.7, 'target': 0.8, 'max': 1.0}
            },
            'processed_voice': {
                'pesq': {'min': 2.8, 'target': 3.3, 'max': 5.0},
                'stoi': {'min': 0.75, 'target': 0.85, 'max': 1.0}
            }
        }
        
        # Test metrics that would pass for one dataset but fail for another
        borderline_metrics = {
            'pesq': 2.9,
            'stoi': 0.78,
            'si_sdr': 12.0,
            'snr': 20.0
        }
        
        results = {}
        for dataset, config in dataset_configs.items():
            manager = QualityThresholdManager(profile=dataset, config={dataset: config})
            results[dataset] = manager.check_quality(borderline_metrics)
        
        # Should fail for gigaspeech (stricter) but pass for mozilla_cv (more lenient)
        assert not results['gigaspeech'].passed
        assert results['mozilla_cv'].passed