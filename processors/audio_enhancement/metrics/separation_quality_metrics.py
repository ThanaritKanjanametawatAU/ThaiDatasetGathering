"""
Separation Quality Metrics Module (S03_T07)
Comprehensive evaluation metrics for audio separation quality
"""

import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from datetime import datetime
import warnings
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from collections import defaultdict

# Try to import optional dependencies
try:
    import pesq as pesq_module
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    pesq_module = None

try:
    import pystoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    pystoi = None

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Available quality metrics"""
    SI_SDR = "si_sdr"
    SI_SIR = "si_sir"
    SI_SAR = "si_sar"
    PESQ = "pesq"
    STOI = "stoi"
    SPECTRAL_DIVERGENCE = "spectral_divergence"
    LOG_SPECTRAL_DISTANCE = "log_spectral_distance"
    SNR_ESTIMATE = "snr_estimate"
    CLARITY_SCORE = "clarity_score"
    ARTIFACT_SCORE = "artifact_score"


@dataclass
class MetricsConfig:
    """Configuration for quality metrics"""
    enable_perceptual: bool = True
    enable_spectral: bool = True
    enable_reference_free: bool = True
    fft_size: int = 2048
    hop_size: int = 512
    epsilon: float = 1e-8
    adaptive_selection: bool = False
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    metrics: Dict[str, float]
    overall_quality: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary"""
        return {
            "metrics": self.metrics,
            "overall_quality": self.overall_quality,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "warnings": self.warnings
        }


@dataclass
class MultiChannelReport:
    """Report for multi-channel audio evaluation"""
    channel_reports: List[QualityReport]
    aggregate_quality: float
    channel_correlation: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SeparationQualityMetrics:
    """Comprehensive quality metrics for audio separation evaluation"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 config: Optional[MetricsConfig] = None):
        """Initialize the quality metrics calculator.
        
        Args:
            sample_rate: Audio sample rate
            config: Configuration parameters
        """
        self.sample_rate = sample_rate
        self.config = config or MetricsConfig()
        self._custom_metrics = {}
        self._adaptive_enabled = False
        
        # Check available dependencies
        if not PESQ_AVAILABLE and self.config.enable_perceptual:
            logger.warning("PESQ not available. Install with: pip install pesq")
        if not STOI_AVAILABLE and self.config.enable_perceptual:
            logger.warning("STOI not available. Install with: pip install pystoi")
    
    def calculate_si_sdr(self, reference: np.ndarray, separated: np.ndarray) -> float:
        """Calculate Scale-Invariant Signal-to-Distortion Ratio.
        
        Args:
            reference: Reference signal
            separated: Separated signal
            
        Returns:
            SI-SDR in dB
        """
        # Ensure same length
        min_len = min(len(reference), len(separated))
        reference = reference[:min_len]
        separated = separated[:min_len]
        
        # Remove mean
        reference = reference - np.mean(reference)
        separated = separated - np.mean(separated)
        
        # Scale invariant projection
        alpha = np.dot(separated, reference) / (np.dot(reference, reference) + self.config.epsilon)
        projection = alpha * reference
        
        # Calculate SI-SDR
        signal_power = np.sum(projection ** 2)
        noise_power = np.sum((separated - projection) ** 2)
        
        if noise_power < self.config.epsilon:
            return 40.0  # Cap at 40 dB for near-perfect separation
        
        si_sdr = 10 * np.log10(signal_power / (noise_power + self.config.epsilon))
        
        return float(np.clip(si_sdr, -20, 40))
    
    def calculate_si_sir(self, reference: np.ndarray, separated: np.ndarray,
                        mixture: Optional[np.ndarray] = None) -> float:
        """Calculate Scale-Invariant Signal-to-Interference Ratio.
        
        Args:
            reference: Reference signal
            separated: Separated signal
            mixture: Original mixture (optional)
            
        Returns:
            SI-SIR in dB
        """
        if mixture is None:
            # Approximate SI-SIR without mixture
            return self.calculate_si_sdr(reference, separated)
        
        # Ensure same length
        min_len = min(len(reference), len(separated), len(mixture))
        reference = reference[:min_len]
        separated = separated[:min_len]
        mixture = mixture[:min_len]
        
        # Remove mean
        reference = reference - np.mean(reference)
        separated = separated - np.mean(separated)
        mixture = mixture - np.mean(mixture)
        
        # Calculate interference as mixture minus reference
        interference = mixture - reference
        
        # Project separated onto reference and interference
        alpha_ref = np.dot(separated, reference) / (np.dot(reference, reference) + self.config.epsilon)
        alpha_int = np.dot(separated, interference) / (np.dot(interference, interference) + self.config.epsilon)
        
        signal_component = alpha_ref * reference
        interference_component = alpha_int * interference
        
        # Calculate SI-SIR
        signal_power = np.sum(signal_component ** 2)
        interference_power = np.sum(interference_component ** 2)
        
        if interference_power < self.config.epsilon:
            return 40.0  # No interference
        
        si_sir = 10 * np.log10(signal_power / (interference_power + self.config.epsilon))
        
        return float(np.clip(si_sir, -20, 40))
    
    def calculate_si_sar(self, reference: np.ndarray, separated: np.ndarray) -> float:
        """Calculate Scale-Invariant Signal-to-Artifacts Ratio.
        
        Args:
            reference: Reference signal
            separated: Separated signal
            
        Returns:
            SI-SAR in dB
        """
        # Ensure same length
        min_len = min(len(reference), len(separated))
        reference = reference[:min_len]
        separated = separated[:min_len]
        
        # Remove mean
        reference = reference - np.mean(reference)
        separated = separated - np.mean(separated)
        
        # Project separated onto reference
        alpha = np.dot(separated, reference) / (np.dot(reference, reference) + self.config.epsilon)
        projection = alpha * reference
        
        # Artifacts are everything not explained by scaled reference
        artifacts = separated - projection
        
        # Detect specific artifacts
        # 1. Clipping artifacts
        clipping_score = np.sum(np.abs(separated) > 0.95) / len(separated)
        
        # 2. Harmonic distortion (check for unexpected frequencies)
        ref_spectrum = np.abs(np.fft.rfft(reference))
        sep_spectrum = np.abs(np.fft.rfft(separated))
        
        # Normalize spectra
        ref_spectrum = ref_spectrum / (np.max(ref_spectrum) + self.config.epsilon)
        sep_spectrum = sep_spectrum / (np.max(sep_spectrum) + self.config.epsilon)
        
        # Find new peaks in separated not in reference
        ref_peaks = self._find_spectral_peaks(ref_spectrum)
        sep_peaks = self._find_spectral_peaks(sep_spectrum)
        
        new_peaks = len(sep_peaks - ref_peaks)
        distortion_score = new_peaks / (len(ref_peaks) + 1)
        
        # Combined artifact score
        artifact_penalty = 5 * (clipping_score + distortion_score)
        
        # Calculate SI-SAR
        signal_power = np.sum(projection ** 2)
        artifact_power = np.sum(artifacts ** 2)
        
        if artifact_power < self.config.epsilon:
            return 40.0 - artifact_penalty
        
        si_sar = 10 * np.log10(signal_power / (artifact_power + self.config.epsilon)) - artifact_penalty
        
        return float(np.clip(si_sar, -20, 40))
    
    def calculate_pesq(self, reference: np.ndarray, separated: np.ndarray,
                      sample_rate: Optional[int] = None) -> float:
        """Calculate PESQ (Perceptual Evaluation of Speech Quality).
        
        Args:
            reference: Reference signal
            separated: Separated signal
            sample_rate: Sample rate (uses instance sample rate if None)
            
        Returns:
            PESQ score (-0.5 to 4.5)
        """
        if not PESQ_AVAILABLE:
            warnings.warn("PESQ not available. Returning default value.")
            return 2.5  # Neutral score
        
        sample_rate = sample_rate or self.sample_rate
        
        # PESQ requires specific sample rates
        if sample_rate not in [8000, 16000]:
            # Resample to 16000
            from scipy.signal import resample
            factor = 16000 / sample_rate
            reference = resample(reference, int(len(reference) * factor))
            separated = resample(separated, int(len(separated) * factor))
            sample_rate = 16000
        
        try:
            # Ensure same length
            min_len = min(len(reference), len(separated))
            reference = reference[:min_len]
            separated = separated[:min_len]
            
            # Calculate PESQ
            mode = 'wb' if sample_rate == 16000 else 'nb'
            pesq_score = pesq_module.pesq(sample_rate, reference, separated, mode)
            
            return float(pesq_score)
        except Exception as e:
            logger.warning(f"PESQ calculation failed: {e}")
            return 2.5
    
    def calculate_stoi(self, reference: np.ndarray, separated: np.ndarray,
                      sample_rate: Optional[int] = None) -> float:
        """Calculate STOI (Short-Time Objective Intelligibility).
        
        Args:
            reference: Reference signal
            separated: Separated signal
            sample_rate: Sample rate (uses instance sample rate if None)
            
        Returns:
            STOI score (0 to 1)
        """
        if not STOI_AVAILABLE:
            warnings.warn("STOI not available. Returning default value.")
            return 0.75  # Neutral score
        
        sample_rate = sample_rate or self.sample_rate
        
        try:
            # Ensure same length
            min_len = min(len(reference), len(separated))
            reference = reference[:min_len]
            separated = separated[:min_len]
            
            # Calculate STOI
            stoi_score = pystoi.stoi(reference, separated, sample_rate, extended=False)
            
            return float(stoi_score)
        except Exception as e:
            logger.warning(f"STOI calculation failed: {e}")
            return 0.75
    
    def calculate_spectral_divergence(self, reference: np.ndarray,
                                    separated: np.ndarray) -> float:
        """Calculate spectral divergence between signals.
        
        Args:
            reference: Reference signal
            separated: Separated signal
            
        Returns:
            Spectral divergence (lower is better)
        """
        # Compute spectrograms
        f, t, ref_spec = scipy_signal.spectrogram(
            reference, self.sample_rate,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size
        )
        
        _, _, sep_spec = scipy_signal.spectrogram(
            separated, self.sample_rate,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size
        )
        
        # Convert to magnitude and add small epsilon
        ref_mag = np.abs(ref_spec) + self.config.epsilon
        sep_mag = np.abs(sep_spec) + self.config.epsilon
        
        # Normalize
        ref_mag = ref_mag / np.sum(ref_mag, axis=0, keepdims=True)
        sep_mag = sep_mag / np.sum(sep_mag, axis=0, keepdims=True)
        
        # Calculate KL divergence
        kl_div = np.sum(ref_mag * np.log(ref_mag / sep_mag))
        
        # Average over time
        spectral_div = kl_div / ref_mag.shape[1]
        
        return float(spectral_div)
    
    def calculate_log_spectral_distance(self, reference: np.ndarray,
                                      separated: np.ndarray) -> float:
        """Calculate log-spectral distance.
        
        Args:
            reference: Reference signal
            separated: Separated signal
            
        Returns:
            Log-spectral distance in dB
        """
        # Compute power spectra
        ref_fft = np.fft.rfft(reference, n=self.config.fft_size)
        sep_fft = np.fft.rfft(separated, n=self.config.fft_size)
        
        ref_power = np.abs(ref_fft) ** 2 + self.config.epsilon
        sep_power = np.abs(sep_fft) ** 2 + self.config.epsilon
        
        # Log spectral distance
        log_distance = np.sqrt(np.mean((10 * np.log10(ref_power) - 10 * np.log10(sep_power)) ** 2))
        
        return float(log_distance)
    
    def calculate_reference_free_metrics(self, separated: np.ndarray) -> Dict[str, float]:
        """Calculate reference-free quality metrics.
        
        Args:
            separated: Separated signal
            
        Returns:
            Dictionary of reference-free metrics
        """
        metrics = {}
        
        # 1. SNR estimate using energy distribution and signal structure
        frame_length = min(self.config.fft_size, len(separated) // 10)
        hop_length = frame_length // 2
        
        # Frame the signal and calculate RMS energy
        energies = []
        for i in range(0, len(separated) - frame_length, hop_length):
            frame = separated[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        if energies:
            energies = np.array(energies)
            
            # Use median-based estimation for better robustness
            energy_median = np.median(energies)
            energy_std = np.std(energies)
            
            # Estimate noise floor as minimum stable energy
            noise_floor = np.percentile(energies, 5)
            
            # Signal level as energy above noise with structure
            signal_levels = energies[energies > noise_floor * 2]
            if len(signal_levels) > 0:
                signal_level = np.mean(signal_levels)
            else:
                signal_level = np.max(energies)
            
            # Penalize signals with too much variability (likely noise)
            variability_penalty = min(energy_std / (energy_median + self.config.epsilon), 2.0)
            
            if noise_floor > 0 and signal_level > noise_floor:
                raw_snr = 20 * np.log10(signal_level / noise_floor)
                # Apply penalty for high variability
                snr_estimate = raw_snr - 10 * variability_penalty
                metrics["snr_estimate"] = float(np.clip(snr_estimate, 0, 60))
            else:
                metrics["snr_estimate"] = 20.0
        else:
            metrics["snr_estimate"] = 20.0
        
        # 2. Clarity score (based on spectral features)
        spectrum = np.abs(np.fft.rfft(separated))
        freqs = np.fft.rfftfreq(len(separated), 1/self.sample_rate)
        
        # Calculate spectral centroid
        if np.sum(spectrum) > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            # Normalize to 0-1 (assuming speech has centroid around 1-4 kHz)
            clarity_score = np.clip(centroid / 4000, 0, 1)
        else:
            clarity_score = 0.0
        
        metrics["clarity_score"] = float(clarity_score)
        
        # 3. Artifact score (detect common artifacts)
        # Clipping detection
        clipping_ratio = np.sum(np.abs(separated) > 0.95) / len(separated)
        
        # Silence ratio
        silence_ratio = np.sum(np.abs(separated) < 0.01) / len(separated)
        
        # Zero crossing rate calculation (manual implementation)
        zcr_values = []
        for i in range(0, len(separated) - frame_length, hop_length):
            frame = separated[i:i + frame_length]
            zcr = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2.0 * len(frame))
            zcr_values.append(zcr)
        
        if zcr_values:
            zcr_var = np.std(zcr_values) / (np.mean(zcr_values) + self.config.epsilon)
        else:
            zcr_var = 0.5
        
        # Combined artifact score (lower is better)
        artifact_score = clipping_ratio * 0.5 + silence_ratio * 0.3 + (1 - zcr_var) * 0.2
        metrics["artifact_score"] = float(np.clip(artifact_score, 0, 1))
        
        return metrics
    
    def evaluate_separation(self, reference: np.ndarray, separated: np.ndarray,
                          mixture: Optional[np.ndarray] = None,
                          include_custom_metrics: bool = False) -> QualityReport:
        """Comprehensive evaluation of separation quality.
        
        Args:
            reference: Reference signal
            separated: Separated signal
            mixture: Original mixture (optional)
            include_custom_metrics: Include registered custom metrics
            
        Returns:
            Comprehensive quality report
        """
        metrics = {}
        warnings_list = []
        
        # Reference-based metrics
        try:
            metrics["si_sdr"] = self.calculate_si_sdr(reference, separated)
        except Exception as e:
            logger.warning(f"SI-SDR calculation failed: {e}")
            warnings_list.append(f"SI-SDR calculation failed: {str(e)}")
        
        if mixture is not None:
            try:
                metrics["si_sir"] = self.calculate_si_sir(reference, separated, mixture)
            except Exception as e:
                logger.warning(f"SI-SIR calculation failed: {e}")
        
        try:
            metrics["si_sar"] = self.calculate_si_sar(reference, separated)
        except Exception as e:
            logger.warning(f"SI-SAR calculation failed: {e}")
        
        # Perceptual metrics
        if self.config.enable_perceptual:
            if PESQ_AVAILABLE:
                try:
                    metrics["pesq"] = self.calculate_pesq(reference, separated)
                except Exception as e:
                    logger.warning(f"PESQ calculation failed: {e}")
            
            if STOI_AVAILABLE:
                try:
                    metrics["stoi"] = self.calculate_stoi(reference, separated)
                except Exception as e:
                    logger.warning(f"STOI calculation failed: {e}")
        
        # Spectral metrics
        if self.config.enable_spectral:
            try:
                metrics["spectral_divergence"] = self.calculate_spectral_divergence(reference, separated)
                metrics["log_spectral_distance"] = self.calculate_log_spectral_distance(reference, separated)
            except Exception as e:
                logger.warning(f"Spectral metrics calculation failed: {e}")
        
        # Reference-free metrics
        if self.config.enable_reference_free:
            ref_free_metrics = self.calculate_reference_free_metrics(separated)
            metrics.update(ref_free_metrics)
        
        # Custom metrics
        if include_custom_metrics:
            for name, metric_info in self._custom_metrics.items():
                try:
                    value = metric_info["func"](reference, separated, sample_rate=self.sample_rate)
                    metrics[name] = float(value)
                except Exception as e:
                    logger.warning(f"Custom metric '{name}' failed: {e}")
        
        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(metrics)
        
        # Create metadata
        metadata = {
            "sample_rate": self.sample_rate,
            "signal_length": len(reference),
            "metrics_computed": list(metrics.keys())
        }
        
        # Add recommended metrics if adaptive
        if self._adaptive_enabled:
            recommended = self._recommend_metrics(reference, separated)
            metadata["recommended_metrics"] = recommended
        
        return QualityReport(
            metrics=metrics,
            overall_quality=overall_quality,
            timestamp=datetime.now(),
            metadata=metadata,
            warnings=warnings_list
        )
    
    def evaluate_batch(self, references: List[np.ndarray],
                      separated_list: List[np.ndarray],
                      metrics: Optional[List[QualityMetric]] = None) -> List[QualityReport]:
        """Batch evaluation of multiple separations.
        
        Args:
            references: List of reference signals
            separated_list: List of separated signals
            metrics: Specific metrics to compute (all if None)
            
        Returns:
            List of quality reports
        """
        if len(references) != len(separated_list):
            raise ValueError("Number of references and separated signals must match")
        
        results = []
        
        for ref, sep in zip(references, separated_list):
            if metrics:
                # Compute only specified metrics
                metric_values = {}
                for metric in metrics:
                    if metric == QualityMetric.SI_SDR:
                        metric_values["si_sdr"] = self.calculate_si_sdr(ref, sep)
                    elif metric == QualityMetric.SPECTRAL_DIVERGENCE:
                        metric_values["spectral_divergence"] = self.calculate_spectral_divergence(ref, sep)
                    # Add other metrics as needed
                
                overall_quality = self._calculate_overall_quality(metric_values)
                
                report = QualityReport(
                    metrics=metric_values,
                    overall_quality=overall_quality,
                    timestamp=datetime.now()
                )
            else:
                # Full evaluation
                report = self.evaluate_separation(ref, sep)
            
            results.append(report)
        
        return results
    
    def evaluate_segments(self, reference: np.ndarray, separated: np.ndarray,
                         segment_duration: float) -> List[QualityReport]:
        """Evaluate quality on audio segments.
        
        Args:
            reference: Reference signal
            separated: Separated signal
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of quality reports for each segment
        """
        segment_samples = int(segment_duration * self.sample_rate)
        num_segments = int(np.ceil(len(reference) / segment_samples))
        
        results = []
        
        for i in range(num_segments):
            start = i * segment_samples
            end = min((i + 1) * segment_samples, len(reference))
            
            ref_segment = reference[start:end]
            sep_segment = separated[start:end]
            
            # Evaluate segment
            report = self.evaluate_separation(ref_segment, sep_segment)
            report.metadata["segment_index"] = i
            report.metadata["segment_start"] = start / self.sample_rate
            report.metadata["segment_end"] = end / self.sample_rate
            
            results.append(report)
        
        return results
    
    def evaluate_multichannel(self, reference: np.ndarray,
                            separated: np.ndarray) -> MultiChannelReport:
        """Evaluate multi-channel audio.
        
        Args:
            reference: Reference signal (channels x samples)
            separated: Separated signal (channels x samples)
            
        Returns:
            Multi-channel quality report
        """
        if reference.ndim != 2 or separated.ndim != 2:
            raise ValueError("Input must be 2D arrays (channels x samples)")
        
        if reference.shape[0] != separated.shape[0]:
            raise ValueError("Number of channels must match")
        
        num_channels = reference.shape[0]
        channel_reports = []
        
        # Evaluate each channel
        for ch in range(num_channels):
            report = self.evaluate_separation(reference[ch], separated[ch])
            report.metadata["channel"] = ch
            channel_reports.append(report)
        
        # Calculate aggregate quality
        aggregate_quality = np.mean([r.overall_quality for r in channel_reports])
        
        # Calculate channel correlation if stereo
        channel_correlation = None
        if num_channels == 2:
            corr, _ = pearsonr(separated[0], separated[1])
            channel_correlation = float(corr)
        
        return MultiChannelReport(
            channel_reports=channel_reports,
            aggregate_quality=float(aggregate_quality),
            channel_correlation=channel_correlation,
            metadata={"num_channels": num_channels}
        )
    
    def register_custom_metric(self, name: str, func: Callable,
                             range: Tuple[float, float] = (0, 1),
                             higher_is_better: bool = True,
                             optimal_value: Optional[float] = None):
        """Register a custom metric function.
        
        Args:
            name: Metric name
            func: Metric function (reference, separated, **kwargs) -> float
            range: Expected value range
            higher_is_better: Whether higher values are better
            optimal_value: Optimal value for the metric
        """
        self._custom_metrics[name] = {
            "func": func,
            "range": range,
            "higher_is_better": higher_is_better,
            "optimal_value": optimal_value
        }
    
    def enable_adaptive_selection(self, enabled: bool = True):
        """Enable/disable adaptive metric selection."""
        self._adaptive_enabled = enabled
    
    def analyze_results(self, results: List[QualityReport]) -> Dict[str, Any]:
        """Statistical analysis of multiple quality reports.
        
        Args:
            results: List of quality reports
            
        Returns:
            Statistical summary
        """
        if not results:
            return {}
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        for report in results:
            for metric, value in report.metrics.items():
                all_metrics[metric].append(value)
        
        # Calculate statistics
        stats = {
            "mean": {},
            "std": {},
            "min": {},
            "max": {},
            "percentiles": {25: {}, 50: {}, 75: {}}
        }
        
        for metric, values in all_metrics.items():
            values_array = np.array(values)
            stats["mean"][metric] = float(np.mean(values_array))
            stats["std"][metric] = float(np.std(values_array))
            stats["min"][metric] = float(np.min(values_array))
            stats["max"][metric] = float(np.max(values_array))
            
            for p in [25, 50, 75]:
                stats["percentiles"][p][metric] = float(np.percentile(values_array, p))
        
        # Overall quality statistics
        overall_qualities = [r.overall_quality for r in results]
        stats["overall_quality"] = {
            "mean": float(np.mean(overall_qualities)),
            "std": float(np.std(overall_qualities)),
            "min": float(np.min(overall_qualities)),
            "max": float(np.max(overall_qualities))
        }
        
        return stats
    
    def _find_spectral_peaks(self, spectrum: np.ndarray,
                           prominence_threshold: float = 0.1) -> set:
        """Find spectral peaks."""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(spectrum, prominence=prominence_threshold * np.max(spectrum))
        return set(peaks)
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        if not metrics:
            return 0.0
        
        # Define metric weights and normalization
        metric_info = {
            "si_sdr": {"weight": 0.3, "range": (-20, 40), "optimal": 40},
            "si_sir": {"weight": 0.15, "range": (-20, 40), "optimal": 40},
            "si_sar": {"weight": 0.15, "range": (-20, 40), "optimal": 40},
            "pesq": {"weight": 0.1, "range": (-0.5, 4.5), "optimal": 4.5},
            "stoi": {"weight": 0.1, "range": (0, 1), "optimal": 1},
            "spectral_divergence": {"weight": 0.08, "range": (0, 5), "optimal": 0},
            "snr_estimate": {"weight": 0.07, "range": (0, 60), "optimal": 40},
            "clarity_score": {"weight": 0.03, "range": (0, 1), "optimal": 1},
            "artifact_score": {"weight": 0.02, "range": (0, 1), "optimal": 0}
        }
        
        total_weight = 0
        weighted_score = 0
        
        for metric, value in metrics.items():
            if metric in metric_info:
                info = metric_info[metric]
                
                # Normalize to 0-1
                min_val, max_val = info["range"]
                optimal = info["optimal"]
                
                if optimal == max_val:  # Higher is better
                    normalized = (value - min_val) / (max_val - min_val)
                elif optimal == min_val:  # Lower is better
                    normalized = 1 - (value - min_val) / (max_val - min_val)
                else:  # Optimal is in middle
                    if value <= optimal:
                        normalized = (value - min_val) / (optimal - min_val)
                    else:
                        normalized = 1 - (value - optimal) / (max_val - optimal)
                
                normalized = np.clip(normalized, 0, 1)
                
                weighted_score += normalized * info["weight"]
                total_weight += info["weight"]
        
        if total_weight > 0:
            return float(weighted_score / total_weight)
        else:
            return 0.5  # Default neutral score
    
    def _recommend_metrics(self, reference: np.ndarray,
                          separated: np.ndarray) -> List[str]:
        """Recommend metrics based on signal characteristics."""
        recommended = []
        
        # Analyze signal characteristics
        # Check if speech-like (modulated)
        envelope = np.abs(scipy_signal.hilbert(reference))
        envelope_var = np.std(envelope) / (np.mean(envelope) + self.config.epsilon)
        
        if envelope_var > 0.3:  # Likely speech
            recommended.extend(["stoi", "pesq", "si_sdr"])
        else:  # Likely music or steady signal
            recommended.extend(["si_sdr", "spectral_divergence", "si_sar"])
        
        return recommended


# Import librosa if available (for reference-free metrics)
try:
    import librosa
except ImportError:
    logger.warning("librosa not available. Some reference-free metrics may not work.")
    librosa = None