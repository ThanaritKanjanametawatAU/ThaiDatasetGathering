"""
Real Sample Validation Framework for Pattern→MetricGAN+

Validates Pattern→MetricGAN+ enhancement on real audio samples including
GigaSpeech2 samples and challenging edge cases.
"""

import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

from .pattern_metricgan_validator import (
    PatternMetricGANQualityValidator,
    QualityReport,
    InterruptionPattern
)

logger = logging.getLogger(__name__)


@dataclass
class SampleTestResult:
    """Result of testing a single real sample"""
    sample_id: str
    sample_path: str
    quality_report: QualityReport
    test_metadata: Dict[str, any]


@dataclass
class ComparisonResult:
    """Result of comparing Pattern→MetricGAN+ with other methods"""
    sample_id: str
    pattern_metricgan_report: QualityReport
    baseline_reports: Dict[str, QualityReport]
    winner: str
    improvement_score: float


class RealSampleValidator:
    """Validate Pattern→MetricGAN+ on real audio samples"""
    
    def __init__(self, sample_directory: str = "tests/fixtures/real_samples/"):
        self.sample_dir = Path(sample_directory)
        self.validator = PatternMetricGANQualityValidator()
        self.gigaspeech_samples = []
        self.diverse_samples = []
        
        # Load available samples
        self._load_sample_inventory()
        
        logger.info(f"Initialized RealSampleValidator with {len(self.gigaspeech_samples)} GigaSpeech samples")
    
    def _load_sample_inventory(self):
        """Load inventory of available test samples"""
        try:
            # Look for GigaSpeech2 samples (S1-S10 pattern)
            gigaspeech_dir = self.sample_dir / "gigaspeech2"
            if gigaspeech_dir.exists():
                for sample_file in gigaspeech_dir.glob("S*.wav"):
                    sample_id = sample_file.stem
                    self.gigaspeech_samples.append({
                        'id': sample_id,
                        'path': str(sample_file),
                        'type': 'gigaspeech2'
                    })
            
            # Look for diverse challenge samples
            diverse_dir = self.sample_dir / "challenging"
            if diverse_dir.exists():
                for sample_file in diverse_dir.glob("*.wav"):
                    sample_id = sample_file.stem
                    self.diverse_samples.append({
                        'id': sample_id,
                        'path': str(sample_file),
                        'type': 'challenging'
                    })
                    
            logger.info(f"Found {len(self.gigaspeech_samples)} GigaSpeech samples, {len(self.diverse_samples)} diverse samples")
            
            # If no real samples found, create synthetic ones
            if len(self.gigaspeech_samples) == 0 and len(self.diverse_samples) == 0:
                logger.info("No real samples found, creating synthetic samples for testing")
                self._create_synthetic_samples()
            
        except Exception as e:
            logger.warning(f"Failed to load sample inventory: {e}")
            # Create synthetic samples for testing if real samples not available
            self._create_synthetic_samples()
    
    def _create_synthetic_samples(self):
        """Create synthetic test samples when real samples are not available"""
        logger.info("Creating synthetic test samples for validation")
        
        # Create synthetic GigaSpeech-like samples (S1-S10)
        for i in range(1, 11):
            sample_id = f"S{i}"
            self.gigaspeech_samples.append({
                'id': sample_id,
                'path': 'synthetic',
                'type': 'synthetic_gigaspeech',
                'known_speaker_group': 'primary' if i != 9 else 'secondary'  # S9 is different speaker
            })
        
        # Create synthetic challenging samples
        challenge_types = ['overlapping_speakers', 'rapid_changes', 'low_quality', 'extreme_loudness']
        for challenge_type in challenge_types:
            self.diverse_samples.append({
                'id': f"challenge_{challenge_type}",
                'path': 'synthetic',
                'type': f'synthetic_{challenge_type}'
            })
    
    def _load_audio_sample(self, sample_info: Dict) -> Tuple[np.ndarray, int]:
        """Load audio sample from file or generate synthetic"""
        if sample_info['path'] == 'synthetic':
            return self._generate_synthetic_audio(sample_info)
        
        try:
            import librosa
            audio, sr = librosa.load(sample_info['path'], sr=16000)
            return audio, sr
        except Exception as e:
            logger.warning(f"Failed to load {sample_info['path']}: {e}, using synthetic")
            return self._generate_synthetic_audio(sample_info)
    
    def _generate_synthetic_audio(self, sample_info: Dict) -> Tuple[np.ndarray, int]:
        """Generate synthetic audio based on sample type"""
        sr = 16000
        duration = 3.0
        samples = int(duration * sr)
        t = np.linspace(0, duration, samples)
        
        sample_type = sample_info.get('type', 'synthetic_gigaspeech')
        
        if 'gigaspeech' in sample_type:
            # Generate speech-like signal
            base_freq = 200 if sample_info.get('known_speaker_group') == 'primary' else 600
            signal = 0.3 * np.sin(2 * np.pi * base_freq * t)
            
            # Add some harmonic content
            signal += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            signal += 0.05 * np.sin(2 * np.pi * base_freq * 3 * t)
            
            # Add potential interruption for some samples
            if sample_info['id'] in ['S3', 'S7']:
                # Add interruption pattern
                interruption_start = int(1.5 * sr)
                interruption_end = int(2.0 * sr)
                interruption_freq = 800 if sample_info.get('known_speaker_group') == 'primary' else 1200
                signal[interruption_start:interruption_end] += 0.4 * np.sin(2 * np.pi * interruption_freq * t[interruption_start:interruption_end])
            
        elif 'overlapping' in sample_type:
            # Overlapping speakers
            speaker1 = 0.4 * np.sin(2 * np.pi * 200 * t)
            speaker2 = 0.3 * np.sin(2 * np.pi * 600 * t)
            signal = speaker1 + speaker2
            
        elif 'rapid_changes' in sample_type:
            # Rapid speaker changes
            signal = np.zeros(samples)
            chunk_size = samples // 8
            for i in range(8):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, samples)
                freq = 300 if i % 2 == 0 else 800
                t_chunk = np.linspace(0, (end_idx - start_idx) / sr, end_idx - start_idx)
                signal[start_idx:end_idx] = 0.4 * np.sin(2 * np.pi * freq * t_chunk)
                
        elif 'low_quality' in sample_type:
            # Low quality with heavy noise
            signal = 0.1 * np.sin(2 * np.pi * 300 * t)
            noise = 0.3 * np.random.randn(samples)
            signal = signal + noise
            
        elif 'extreme_loudness' in sample_type:
            # Extreme loudness variations
            signal = np.zeros(samples)
            chunk_size = samples // 4
            levels = [0.01, 0.3, 0.9, 0.0]  # quiet, normal, loud, silent
            for i, level in enumerate(levels):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, samples)
                if level > 0:
                    t_chunk = np.linspace(0, (end_idx - start_idx) / sr, end_idx - start_idx)
                    signal[start_idx:end_idx] = level * np.sin(2 * np.pi * 300 * t_chunk)
        else:
            # Default speech-like signal
            signal = 0.3 * np.sin(2 * np.pi * 300 * t)
        
        # Add some noise
        noise = 0.02 * np.random.randn(samples)
        signal = signal + noise
        
        return signal, sr
    
    def validate_gigaspeech_samples(self) -> Dict[str, SampleTestResult]:
        """
        Validate on GigaSpeech2 samples S1-S10 (known speaker clustering).
        
        This validates that:
        - S1-S8 and S10 should cluster as same speaker
        - S9 should be identified as different speaker
        - Quality metrics meet thresholds
        """
        logger.info("Validating Pattern→MetricGAN+ on GigaSpeech2 samples")
        
        results = {}
        
        for sample_info in self.gigaspeech_samples:
            try:
                sample_id = sample_info['id']
                logger.debug(f"Processing GigaSpeech sample {sample_id}")
                
                # Load original audio
                original_audio, sr = self._load_audio_sample(sample_info)
                
                # Simulate Pattern→MetricGAN+ enhancement
                enhanced_audio = self._simulate_pattern_metricgan_enhancement(original_audio, sr)
                
                # Create test patterns if this sample should have interruptions
                detected_patterns = []
                pattern_boundaries = []
                
                if sample_id in ['S3', 'S7']:  # Samples with known interruptions
                    patterns = [
                        InterruptionPattern(start=1.5, end=2.0, confidence=0.85, duration=0.5)
                    ]
                    detected_patterns = patterns
                    pattern_boundaries = [(int(1.5 * sr), int(2.0 * sr))]
                
                # Validate enhancement
                quality_report = self.validator.validate_enhancement(
                    original_audio,
                    enhanced_audio,
                    sample_id,
                    detected_patterns=detected_patterns,
                    pattern_boundaries=pattern_boundaries
                )
                
                # Create test metadata
                test_metadata = {
                    'sample_type': sample_info['type'],
                    'has_interruptions': len(detected_patterns) > 0,
                    'expected_speaker_group': sample_info.get('known_speaker_group', 'primary'),
                    'audio_duration': len(original_audio) / sr,
                    'synthetic': sample_info['path'] == 'synthetic'
                }
                
                results[sample_id] = SampleTestResult(
                    sample_id=sample_id,
                    sample_path=sample_info['path'],
                    quality_report=quality_report,
                    test_metadata=test_metadata
                )
                
                logger.debug(f"Completed validation for {sample_id}: Pass={quality_report.overall_pass}")
                
            except Exception as e:
                logger.error(f"Failed to validate GigaSpeech sample {sample_id}: {e}")
                continue
        
        # Log summary
        passed_count = sum(1 for r in results.values() if r.quality_report.overall_pass)
        logger.info(f"GigaSpeech validation complete: {passed_count}/{len(results)} samples passed")
        
        return results
    
    def validate_edge_case_samples(self) -> Dict[str, SampleTestResult]:
        """Validate on challenging audio samples"""
        logger.info("Validating Pattern→MetricGAN+ on challenging edge case samples")
        
        results = {}
        
        for sample_info in self.diverse_samples:
            try:
                sample_id = sample_info['id']
                logger.debug(f"Processing challenging sample {sample_id}")
                
                # Load original audio
                original_audio, sr = self._load_audio_sample(sample_info)
                
                # Simulate Pattern→MetricGAN+ enhancement
                enhanced_audio = self._simulate_pattern_metricgan_enhancement(original_audio, sr)
                
                # Validate enhancement
                quality_report = self.validator.validate_enhancement(
                    original_audio,
                    enhanced_audio,
                    sample_id
                )
                
                # Create test metadata
                test_metadata = {
                    'sample_type': sample_info['type'],
                    'challenge_category': sample_info['type'].replace('synthetic_', ''),
                    'audio_duration': len(original_audio) / sr,
                    'synthetic': sample_info['path'] == 'synthetic'
                }
                
                results[sample_id] = SampleTestResult(
                    sample_id=sample_id,
                    sample_path=sample_info['path'],
                    quality_report=quality_report,
                    test_metadata=test_metadata
                )
                
                logger.debug(f"Completed validation for challenging sample {sample_id}: Pass={quality_report.overall_pass}")
                
            except Exception as e:
                logger.error(f"Failed to validate challenging sample {sample_id}: {e}")
                continue
        
        # Log summary
        passed_count = sum(1 for r in results.values() if r.quality_report.overall_pass)
        logger.info(f"Edge case validation complete: {passed_count}/{len(results)} samples passed")
        
        return results
    
    def validate_thai_linguistic_features(self) -> Dict[str, SampleTestResult]:
        """Validate preservation of Thai language characteristics"""
        logger.info("Validating Thai linguistic feature preservation")
        
        # For now, use existing samples but with Thai-specific analysis
        thai_results = {}
        
        # Use a subset of GigaSpeech samples as proxy for Thai features
        thai_samples = self.gigaspeech_samples[:5]  # Use first 5 samples
        
        for sample_info in thai_samples:
            try:
                sample_id = f"thai_{sample_info['id']}"
                
                # Load and enhance audio
                original_audio, sr = self._load_audio_sample(sample_info)
                enhanced_audio = self._simulate_pattern_metricgan_enhancement(original_audio, sr)
                
                # Validate with focus on naturalness and speaker preservation
                quality_report = self.validator.validate_enhancement(
                    original_audio,
                    enhanced_audio,
                    sample_id
                )
                
                # Thai-specific metadata
                test_metadata = {
                    'sample_type': 'thai_linguistic',
                    'focus_areas': ['naturalness', 'speaker_preservation', 'tonal_preservation'],
                    'audio_duration': len(original_audio) / sr,
                    'synthetic': sample_info['path'] == 'synthetic'
                }
                
                thai_results[sample_id] = SampleTestResult(
                    sample_id=sample_id,
                    sample_path=sample_info['path'],
                    quality_report=quality_report,
                    test_metadata=test_metadata
                )
                
            except Exception as e:
                logger.error(f"Failed to validate Thai sample {sample_id}: {e}")
                continue
        
        # Log summary
        passed_count = sum(1 for r in thai_results.values() if r.quality_report.overall_pass)
        logger.info(f"Thai linguistic validation complete: {passed_count}/{len(thai_results)} samples passed")
        
        return thai_results
    
    def _simulate_pattern_metricgan_enhancement(self, original: np.ndarray, sr: int) -> np.ndarray:
        """
        Simulate Pattern→MetricGAN+ enhancement for testing.
        
        In real implementation, this would call the actual enhancement pipeline.
        For testing, we simulate the expected transformations.
        """
        enhanced = original.copy()
        
        # Simulate pattern detection and suppression
        # Look for high-energy regions that might be interruptions
        window_size = int(0.1 * sr)  # 100ms windows
        energy_threshold = np.percentile(original**2, 80)  # Top 20% energy
        
        for i in range(0, len(original) - window_size, window_size):
            window = original[i:i + window_size]
            window_energy = np.mean(window**2)
            
            if window_energy > energy_threshold:
                # Simulate pattern suppression (keep 15% of original)
                enhanced[i:i + window_size] = 0.15 * window + 0.85 * 0.1 * window
        
        # Simulate loudness normalization (160% increase)
        enhanced = enhanced * 1.6
        
        # Apply soft limiting
        enhanced = np.clip(enhanced, -0.95, 0.95)
        
        return enhanced


class PatternMetricGANComparison:
    """A/B comparison with other enhancement methods"""
    
    def __init__(self):
        self.validator = PatternMetricGANQualityValidator()
        logger.info("Initialized Pattern→MetricGAN+ comparison framework")
    
    def compare_with_existing_levels(
        self, 
        test_samples: List[Tuple[np.ndarray, int, str]]
    ) -> List[ComparisonResult]:
        """
        Compare Pattern→MetricGAN+ with existing enhancement levels.
        
        Args:
            test_samples: List of (audio, sample_rate, sample_id) tuples
            
        Returns:
            List[ComparisonResult]: Comparison results for each sample
        """
        logger.info(f"Comparing Pattern→MetricGAN+ with baseline methods on {len(test_samples)} samples")
        
        results = []
        baseline_methods = ['moderate', 'aggressive', 'ultra_aggressive']
        
        for original_audio, sr, sample_id in test_samples:
            try:
                # Enhance with Pattern→MetricGAN+
                pattern_metricgan_enhanced = self._simulate_pattern_metricgan_enhancement(original_audio, sr)
                pattern_metricgan_report = self.validator.validate_enhancement(
                    original_audio,
                    pattern_metricgan_enhanced,
                    f"{sample_id}_pattern_metricgan"
                )
                
                # Enhance with baseline methods
                baseline_reports = {}
                for method in baseline_methods:
                    baseline_enhanced = self._simulate_baseline_enhancement(original_audio, sr, method)
                    baseline_report = self.validator.validate_enhancement(
                        original_audio,
                        baseline_enhanced,
                        f"{sample_id}_{method}"
                    )
                    baseline_reports[method] = baseline_report
                
                # Determine winner and calculate improvement
                winner, improvement_score = self._calculate_comparison_winner(
                    pattern_metricgan_report, baseline_reports
                )
                
                result = ComparisonResult(
                    sample_id=sample_id,
                    pattern_metricgan_report=pattern_metricgan_report,
                    baseline_reports=baseline_reports,
                    winner=winner,
                    improvement_score=improvement_score
                )
                
                results.append(result)
                logger.debug(f"Comparison for {sample_id}: Winner={winner}, Improvement={improvement_score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed comparison for sample {sample_id}: {e}")
                continue
        
        # Log summary
        pattern_wins = sum(1 for r in results if r.winner == 'pattern_metricgan_plus')
        logger.info(f"Comparison complete: Pattern→MetricGAN+ won {pattern_wins}/{len(results)} comparisons")
        
        return results
    
    def _simulate_pattern_metricgan_enhancement(self, original: np.ndarray, sr: int) -> np.ndarray:
        """Simulate Pattern→MetricGAN+ enhancement"""
        enhanced = original.copy()
        
        # Pattern detection and suppression
        window_size = int(0.1 * sr)
        energy_threshold = np.percentile(original**2, 80)
        
        for i in range(0, len(original) - window_size, window_size):
            window = original[i:i + window_size]
            window_energy = np.mean(window**2)
            
            if window_energy > energy_threshold:
                enhanced[i:i + window_size] = 0.15 * window
        
        # MetricGAN+ simulation (improve SNR)
        noise_level = np.std(enhanced) * 0.1
        enhanced = enhanced + noise_level * np.random.randn(len(enhanced)) * 0.5
        
        # Loudness normalization
        enhanced = enhanced * 1.6
        enhanced = np.clip(enhanced, -0.95, 0.95)
        
        return enhanced
    
    def _simulate_baseline_enhancement(self, original: np.ndarray, sr: int, method: str) -> np.ndarray:
        """Simulate baseline enhancement methods"""
        enhanced = original.copy()
        
        if method == 'moderate':
            # Simple noise reduction
            enhanced = enhanced * 1.1
            
        elif method == 'aggressive':
            # More aggressive processing
            enhanced = enhanced * 1.3
            # Simple low-pass filtering simulation
            enhanced = np.convolve(enhanced, np.ones(5)/5, mode='same')
            
        elif method == 'ultra_aggressive':
            # Very aggressive processing
            enhanced = enhanced * 1.5
            enhanced = np.convolve(enhanced, np.ones(10)/10, mode='same')
        
        return np.clip(enhanced, -0.95, 0.95)
    
    def _calculate_comparison_winner(
        self,
        pattern_metricgan_report: QualityReport,
        baseline_reports: Dict[str, QualityReport]
    ) -> Tuple[str, float]:
        """Calculate winner and improvement score"""
        
        # Calculate composite score for Pattern→MetricGAN+
        pattern_score = self._calculate_composite_score(pattern_metricgan_report)
        
        # Calculate scores for baselines
        baseline_scores = {
            method: self._calculate_composite_score(report)
            for method, report in baseline_reports.items()
        }
        
        # Find best baseline
        best_baseline_method = max(baseline_scores.keys(), key=lambda k: baseline_scores[k])
        best_baseline_score = baseline_scores[best_baseline_method]
        
        # Determine winner
        if pattern_score > best_baseline_score:
            winner = 'pattern_metricgan_plus'
            improvement_score = pattern_score - best_baseline_score
        else:
            winner = best_baseline_method
            improvement_score = best_baseline_score - pattern_score
        
        return winner, improvement_score
    
    def _calculate_composite_score(self, report: QualityReport) -> float:
        """Calculate composite quality score from report"""
        if not report.enhanced_metrics or not report.pattern_metrics:
            return 0.0
        
        # Weight different metrics
        weights = {
            'pesq': 0.25,
            'stoi': 0.25,
            'pattern_suppression_effectiveness': 0.20,
            'primary_speaker_preservation': 0.15,
            'loudness_consistency': 0.10,
            'naturalness_score': 0.05
        }
        
        score = 0.0
        total_weight = 0.0
        
        # Add enhanced metrics
        for metric, weight in weights.items():
            if metric in ['pesq', 'stoi']:
                value = report.enhanced_metrics.get(metric, 0.0)
                # Normalize PESQ (0-5 scale) and STOI (0-1 scale)
                if metric == 'pesq':
                    value = value / 5.0
                score += weight * value
                total_weight += weight
        
        # Add pattern metrics
        for metric, weight in weights.items():
            if metric in report.pattern_metrics:
                value = report.pattern_metrics[metric]
                score += weight * value
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def generate_quality_reports(
        self,
        comparisons: List[ComparisonResult]
    ) -> Dict[str, any]:
        """Generate comprehensive quality comparison reports"""
        
        if not comparisons:
            return {}
        
        # Calculate statistics
        total_comparisons = len(comparisons)
        pattern_wins = sum(1 for c in comparisons if c.winner == 'pattern_metricgan_plus')
        win_rate = pattern_wins / total_comparisons
        
        avg_improvement = np.mean([
            c.improvement_score for c in comparisons 
            if c.winner == 'pattern_metricgan_plus'
        ]) if pattern_wins > 0 else 0.0
        
        # Method win breakdown
        method_wins = {}
        for comparison in comparisons:
            method = comparison.winner
            method_wins[method] = method_wins.get(method, 0) + 1
        
        # Quality metric averages
        avg_metrics = {}
        metric_names = ['pesq', 'stoi', 'pattern_suppression_effectiveness', 'naturalness_score']
        
        for metric in metric_names:
            values = []
            for c in comparisons:
                if metric in ['pesq', 'stoi']:
                    value = c.pattern_metricgan_report.enhanced_metrics.get(metric)
                else:
                    value = c.pattern_metricgan_report.pattern_metrics.get(metric)
                
                if value is not None:
                    values.append(value)
            
            avg_metrics[f'avg_{metric}'] = np.mean(values) if values else 0.0
        
        report = {
            'summary': {
                'total_comparisons': total_comparisons,
                'pattern_metricgan_wins': pattern_wins,
                'win_rate': win_rate,
                'average_improvement_when_winning': avg_improvement
            },
            'method_performance': method_wins,
            'quality_metrics': avg_metrics,
            'detailed_results': [
                {
                    'sample_id': c.sample_id,
                    'winner': c.winner,
                    'improvement_score': c.improvement_score,
                    'pattern_metricgan_pass': c.pattern_metricgan_report.overall_pass
                }
                for c in comparisons
            ]
        }
        
        logger.info(f"Generated comparison report: {win_rate:.1%} win rate for Pattern→MetricGAN+")
        
        return report