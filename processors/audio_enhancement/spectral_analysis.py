"""
Comprehensive Spectral Analysis Module for Audio Quality Assessment.
Provides frequency-domain feature extraction and anomaly detection.
"""

import numpy as np
import torch
import librosa
from scipy import signal, ndimage
from scipy.linalg import toeplitz
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SpectralAnalysisError(Exception):
    """Base exception for spectral analysis errors."""
    pass


class FFTError(SpectralAnalysisError):
    """FFT computation failed."""
    def __init__(self, reason: str, audio_shape: Optional[tuple] = None):
        self.reason = reason
        self.audio_shape = audio_shape
        super().__init__(f"FFT failed: {reason}")


class FeatureExtractionError(SpectralAnalysisError):
    """Feature extraction failed."""
    pass


class AnomalyDetectionError(SpectralAnalysisError):
    """Anomaly detection failed."""
    pass


class SpectralAnalyzer:
    """
    Core spectral analysis class for audio feature extraction and anomaly detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize spectral analyzer with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # STFT parameters
        self.n_fft = self.config.get('n_fft', 2048)
        self.hop_length = self.config.get('hop_length', self.n_fft // 4)
        self.window = self.config.get('window', 'hann')
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = self.config.get('anomaly_thresholds', {
            'spectral_hole': 0.3,
            'harmonic_distortion': 5.0,  # THD percentage
            'aliasing': 0.1,
            'codec_artifact': 48  # dB/octave
        })
        
        # Feature extraction settings
        self.n_mfcc = self.config.get('n_mfcc', 13)
        self.n_mels = self.config.get('n_mels', 128)
        
        # Device for computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_stft(self, audio: np.ndarray, n_fft: Optional[int] = None) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Input audio signal
            n_fft: FFT size (uses default if None)
            
        Returns:
            Complex STFT matrix [frequency_bins x time_frames]
        """
        if len(audio) == 0:
            raise FFTError("Empty audio signal", audio_shape=(0,))
            
        n_fft = n_fft or self.n_fft
        hop_length = n_fft // 4
        
        try:
            # Handle NaN values
            if np.any(np.isnan(audio)):
                logger.warning("NaN values detected in audio, replacing with zeros")
                audio = np.nan_to_num(audio, nan=0.0)
            
            # Compute STFT
            stft_matrix = librosa.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                window=self.window,
                center=True,
                pad_mode='reflect'
            )
            
            return stft_matrix
            
        except Exception as e:
            raise FFTError(f"STFT computation failed: {str(e)}", audio_shape=audio.shape)
    
    def extract_spectral_features(self, stft: np.ndarray, sr: int = 16000) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from STFT.
        
        Args:
            stft: Complex STFT matrix
            sr: Sample rate
            
        Returns:
            Dictionary of spectral features
        """
        magnitude = np.abs(stft)
        power = magnitude ** 2
        
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = self._spectral_centroid(magnitude, sr)
        
        # Spectral rolloff
        features['spectral_rolloff'] = self._spectral_rolloff(magnitude, sr)
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = self._spectral_bandwidth(
            magnitude, sr, features['spectral_centroid']
        )
        
        # Spectral flux
        features['spectral_flux'] = self._spectral_flux(magnitude)
        
        # Spectral flatness
        features['spectral_flatness'] = self._spectral_flatness(magnitude)
        
        # MFCCs
        features['mfcc'] = self._extract_mfcc(stft, sr)
        
        # Zero crossing rate (from time domain - need to reconstruct)
        features['zero_crossing_rate'] = self._zero_crossing_rate(stft, sr)
        
        return features
    
    def detect_spectral_anomalies(self, stft: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect various spectral anomalies.
        
        Args:
            stft: Complex STFT matrix
            
        Returns:
            List of detected anomalies with details
        """
        anomalies = []
        magnitude = np.abs(stft)
        
        # Detect spectral holes
        holes = self._detect_spectral_holes(magnitude)
        anomalies.extend(holes)
        
        # Detect harmonic distortion
        distortion = self._detect_harmonic_distortion(stft)
        anomalies.extend(distortion)
        
        # Detect aliasing artifacts
        aliasing = self._detect_aliasing(magnitude)
        anomalies.extend(aliasing)
        
        # Detect codec artifacts
        codec = self._detect_codec_artifacts(magnitude)
        anomalies.extend(codec)
        
        return anomalies
    
    def compute_harmonic_features(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Compute harmonic-related features.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of harmonic features
        """
        features = {}
        
        # Fundamental frequency detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=50,
            fmax=2000,
            sr=sr,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Get most likely fundamental
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            features['fundamental_frequency'] = float(np.median(valid_f0))
        else:
            features['fundamental_frequency'] = 0.0
        
        # Harmonic-to-noise ratio
        if features['fundamental_frequency'] > 0:
            features['harmonic_to_noise_ratio'] = self._compute_hnr(audio, sr, f0)
        else:
            features['harmonic_to_noise_ratio'] = 0.0
        
        # Extract harmonics and compute THD
        if features['fundamental_frequency'] > 0:
            harmonics, thd = self._analyze_harmonics(audio, sr, features['fundamental_frequency'])
            features['harmonics'] = harmonics
            features['total_harmonic_distortion'] = thd
        else:
            features['harmonics'] = []
            features['total_harmonic_distortion'] = 0.0
        
        return features
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Perform complete spectral analysis.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Complete analysis results
        """
        try:
            # Handle edge cases
            if len(audio) == 0:
                return {'error': 'Empty audio signal', 'features': {}}
            
            if len(audio) < self.n_fft:
                # Pad short signals
                audio = np.pad(audio, (0, self.n_fft - len(audio)), mode='constant')
            
            # Compute STFT
            stft = self.compute_stft(audio)
            
            # Extract features
            features = self.extract_spectral_features(stft, sample_rate)
            
            # Detect anomalies
            anomalies = self.detect_spectral_anomalies(stft)
            
            # Compute quality score
            quality_score = self._compute_quality_score(features, anomalies)
            
            # Generate visualization data
            viz_data = self._generate_visualization_data(stft, sample_rate, anomalies)
            
            return {
                'stft': stft,
                'features': features,
                'anomalies': anomalies,
                'quality_score': quality_score,
                'visualization': viz_data
            }
            
        except Exception as e:
            logger.error(f"Spectral analysis failed: {str(e)}")
            return {
                'error': str(e),
                'features': {},
                'anomalies': [],
                'quality_score': 0.0
            }
    
    def compute_quality_metrics(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Compute audio quality metrics.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Spectral entropy
        stft = self.compute_stft(audio)
        magnitude = np.abs(stft)
        metrics['spectral_entropy'] = self._spectral_entropy(magnitude)
        
        # Other quality indicators
        features = self.extract_spectral_features(stft, sr)
        
        # Tonality measure (inverse of flatness)
        metrics['tonality'] = 1.0 - np.mean(features['spectral_flatness'])
        
        # Spectral complexity
        metrics['spectral_complexity'] = np.std(features['spectral_centroid'])
        
        return metrics
    
    def analyze_formants(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, float]]:
        """
        Detect formants in speech signal.
        
        Args:
            audio: Speech signal
            sample_rate: Sample rate
            
        Returns:
            List of formant information
        """
        # Pre-emphasis
        pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # LPC analysis for formant detection
        order = int(sample_rate / 1000) + 2
        a = librosa.lpc(pre_emphasized, order=order)
        
        # Find roots
        roots = np.roots(a)
        
        # Convert to frequencies
        formants = []
        for root in roots:
            if np.imag(root) >= 0:  # Only positive frequencies
                angle = np.angle(root)
                freq = angle * sample_rate / (2 * np.pi)
                
                if 90 < freq < 5000:  # Reasonable formant range
                    bandwidth = -0.5 * sample_rate * np.log(np.abs(root)) / np.pi
                    formants.append({
                        'frequency': freq,
                        'bandwidth': bandwidth,
                        'amplitude': 1.0 / np.abs(root)
                    })
        
        # Sort by frequency and return top formants
        formants.sort(key=lambda x: x['frequency'])
        return formants[:5]  # Return up to 5 formants
    
    # Private helper methods
    
    def _spectral_centroid(self, magnitude: np.ndarray, sr: int) -> np.ndarray:
        """Calculate spectral centroid."""
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Compute weighted mean
        centroid = np.sum(magnitude * freqs[:, np.newaxis], axis=0) / (np.sum(magnitude, axis=0) + 1e-10)
        
        # Normalize by Nyquist frequency
        centroid = centroid / (sr / 2)
        
        return centroid
    
    def _spectral_rolloff(self, magnitude: np.ndarray, sr: int, rolloff_percent: float = 0.95) -> np.ndarray:
        """Calculate spectral rolloff."""
        # Compute cumulative sum along frequency axis
        cumsum = np.cumsum(magnitude, axis=0)
        
        # Find rolloff frequency
        threshold = rolloff_percent * cumsum[-1, :]
        rolloff_bins = np.argmax(cumsum >= threshold[np.newaxis, :], axis=0)
        
        # Convert to normalized frequency
        rolloff = rolloff_bins / magnitude.shape[0]
        
        return rolloff
    
    def _spectral_bandwidth(self, magnitude: np.ndarray, sr: int, centroid: np.ndarray) -> np.ndarray:
        """Calculate spectral bandwidth."""
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft) / (sr / 2)  # Normalized
        
        # Compute weighted standard deviation
        deviation = (freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** 2
        bandwidth = np.sqrt(
            np.sum(magnitude * deviation, axis=0) / (np.sum(magnitude, axis=0) + 1e-10)
        )
        
        return bandwidth
    
    def _spectral_flux(self, magnitude: np.ndarray) -> np.ndarray:
        """Calculate spectral flux."""
        # Compute difference between consecutive frames
        diff = np.diff(magnitude, axis=1, prepend=magnitude[:, 0:1])
        
        # Only consider positive differences (onset detection)
        flux = np.sum(np.maximum(0, diff) ** 2, axis=0)
        
        # Normalize
        flux = flux / (magnitude.shape[0] + 1e-10)
        
        return flux
    
    def _spectral_flatness(self, magnitude: np.ndarray) -> np.ndarray:
        """Calculate spectral flatness (Wiener entropy)."""
        # Add small epsilon to avoid log(0)
        magnitude = magnitude + 1e-10
        
        # Geometric mean
        log_magnitude = np.log(magnitude)
        geometric_mean = np.exp(np.mean(log_magnitude, axis=0))
        
        # Arithmetic mean
        arithmetic_mean = np.mean(magnitude, axis=0)
        
        # Flatness
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        return flatness
    
    def _extract_mfcc(self, stft: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features."""
        # Convert to power spectrogram
        power = np.abs(stft) ** 2
        
        # Mel spectrogram
        mel_basis = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=self.n_mels)
        mel_power = np.dot(mel_basis, power)
        
        # Log mel spectrogram
        log_mel = 10 * np.log10(mel_power + 1e-10)
        
        # DCT to get MFCCs
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=self.n_mfcc)
        
        # Normalize MFCCs (except C0 which represents energy)
        # This helps keep values in a reasonable range
        mfcc[1:, :] = (mfcc[1:, :] - np.mean(mfcc[1:, :], axis=1, keepdims=True)) / (np.std(mfcc[1:, :], axis=1, keepdims=True) + 1e-8)
        
        return mfcc
    
    def _zero_crossing_rate(self, stft: np.ndarray, sr: int) -> np.ndarray:
        """Estimate zero crossing rate from STFT."""
        # Reconstruct time-domain signal for each frame
        hop_length = self.hop_length
        zcr = []
        
        for i in range(stft.shape[1]):
            # Get frame spectrum
            frame_spectrum = stft[:, i]
            
            # Estimate high frequency content (proxy for ZCR)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
            high_freq_energy = np.sum(np.abs(frame_spectrum[freqs > 4000]) ** 2)
            total_energy = np.sum(np.abs(frame_spectrum) ** 2) + 1e-10
            
            zcr_estimate = high_freq_energy / total_energy
            zcr.append(zcr_estimate)
        
        return np.array(zcr)
    
    def _detect_spectral_holes(self, magnitude: np.ndarray) -> List[Dict[str, Any]]:
        """Detect spectral holes (missing frequency bands)."""
        anomalies = []
        
        # Compute local statistics
        window_size = 11  # Frequency bins
        local_mean = ndimage.uniform_filter1d(magnitude, window_size, axis=0)
        global_mean = np.mean(magnitude, axis=0)
        
        # Detect holes
        threshold = self.anomaly_thresholds['spectral_hole']
        hole_mask = local_mean < (threshold * global_mean[np.newaxis, :])
        
        # Find connected regions
        for t in range(magnitude.shape[1]):
            frame_holes = hole_mask[:, t]
            
            # Find consecutive True values
            hole_regions = []
            start = None
            
            for i, is_hole in enumerate(frame_holes):
                if is_hole and start is None:
                    start = i
                elif not is_hole and start is not None:
                    hole_regions.append((start, i))
                    start = None
            
            if start is not None:
                hole_regions.append((start, len(frame_holes)))
            
            # Convert to frequency ranges
            for start_bin, end_bin in hole_regions:
                if end_bin - start_bin >= 5:  # Minimum width
                    freq_start = start_bin * 16000 / self.n_fft
                    freq_end = end_bin * 16000 / self.n_fft
                    
                    anomalies.append({
                        'type': 'spectral_hole',
                        'time_frame': t,
                        'frequency_range': (freq_start, freq_end),
                        'severity': 'medium',
                        'confidence': 0.8
                    })
        
        return anomalies
    
    def _detect_harmonic_distortion(self, stft: np.ndarray) -> List[Dict[str, Any]]:
        """Detect harmonic distortion."""
        anomalies = []
        
        # Analyze middle portion for stable harmonics
        mid_frame = stft.shape[1] // 2
        spectrum = np.abs(stft[:, mid_frame])
        
        # Find peaks (potential harmonics)
        peaks, properties = signal.find_peaks(spectrum, height=np.max(spectrum) * 0.1)
        
        if len(peaks) > 1:
            # Estimate fundamental
            freq_resolution = 16000 / self.n_fft
            peak_freqs = peaks * freq_resolution
            
            # Find fundamental (lowest significant peak)
            fundamental_idx = 0
            fundamental_freq = peak_freqs[fundamental_idx]
            
            # Calculate THD
            fundamental_power = spectrum[peaks[fundamental_idx]] ** 2
            harmonic_power = 0
            
            for i in range(1, min(len(peaks), 10)):
                if i < len(peaks):
                    harmonic_power += spectrum[peaks[i]] ** 2
            
            if fundamental_power > 0:
                thd = np.sqrt(harmonic_power / fundamental_power) * 100
                
                if thd > self.anomaly_thresholds['harmonic_distortion']:
                    anomalies.append({
                        'type': 'harmonic_distortion',
                        'time_frame': mid_frame,
                        'thd_percentage': float(thd),
                        'fundamental_frequency': float(fundamental_freq),
                        'severity': 'high' if thd > 10 else 'medium',
                        'confidence': 0.9
                    })
        
        return anomalies
    
    def _detect_aliasing(self, magnitude: np.ndarray) -> List[Dict[str, Any]]:
        """Detect aliasing artifacts."""
        anomalies = []
        
        # Look for mirror frequencies (aliasing pattern)
        nyquist_bin = magnitude.shape[0] - 1
        
        for t in range(magnitude.shape[1]):
            frame = magnitude[:, t]
            
            # Check for unusual high-frequency content that mirrors low frequencies
            low_freq = frame[:nyquist_bin//4]
            high_freq = frame[3*nyquist_bin//4:]
            
            # Compute correlation between low and flipped high frequencies
            if len(high_freq) == len(low_freq):
                correlation = np.corrcoef(low_freq, high_freq[::-1])[0, 1]
                
                if correlation > 0.7:  # High correlation suggests aliasing
                    anomalies.append({
                        'type': 'aliasing',
                        'time_frame': t,
                        'correlation': float(correlation),
                        'severity': 'high',
                        'confidence': 0.7
                    })
        
        return anomalies
    
    def _detect_codec_artifacts(self, magnitude: np.ndarray) -> List[Dict[str, Any]]:
        """Detect codec artifacts like frequency cutoffs."""
        anomalies = []
        
        # Average spectrum across time
        avg_spectrum = np.mean(magnitude, axis=1)
        
        # Find sharp cutoff
        # Smooth spectrum
        smoothed = ndimage.gaussian_filter1d(avg_spectrum, sigma=3)
        
        # Compute derivative
        derivative = np.gradient(smoothed)
        
        # Find steep negative slopes (cutoffs)
        cutoff_threshold = -np.std(derivative) * 3
        potential_cutoffs = np.where(derivative < cutoff_threshold)[0]
        
        if len(potential_cutoffs) > 0:
            # Get the highest frequency cutoff
            cutoff_bin = potential_cutoffs[-1]
            cutoff_freq = cutoff_bin * 16000 / self.n_fft
            
            # Measure steepness
            if cutoff_bin < len(avg_spectrum) - 10:
                before = avg_spectrum[cutoff_bin]
                after = avg_spectrum[min(cutoff_bin + 10, len(avg_spectrum) - 1)]
                
                if before > 0:
                    steepness_db = 20 * np.log10((after + 1e-10) / (before + 1e-10))
                    steepness_db_per_octave = steepness_db * (16000 / self.n_fft) / cutoff_freq * 12
                    
                    if abs(steepness_db_per_octave) > self.anomaly_thresholds['codec_artifact']:
                        anomalies.append({
                            'type': 'codec_artifact',
                            'cutoff_frequency': float(cutoff_freq),
                            'steepness_db_per_octave': float(steepness_db_per_octave),
                            'severity': 'medium',
                            'confidence': 0.8
                        })
        
        return anomalies
    
    def _compute_hnr(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> float:
        """Compute Harmonic-to-Noise Ratio."""
        # Simplified HNR calculation
        # Use autocorrelation method
        
        # Get middle portion where f0 is stable
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) == 0:
            return 0.0
        
        median_f0 = np.median(valid_f0)
        period_samples = int(sr / median_f0)
        
        # Compute autocorrelation
        if len(audio) > 2 * period_samples:
            autocorr = np.correlate(audio, audio, mode='same')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peak at fundamental period
            peak_value = autocorr[period_samples]
            noise_floor = np.mean(autocorr[period_samples*2:period_samples*3])
            
            if noise_floor > 0:
                hnr_linear = peak_value / noise_floor
                hnr_db = 10 * np.log10(max(hnr_linear, 1e-10))
                return float(max(hnr_db, 0))
        
        return 0.0
    
    def _analyze_harmonics(self, audio: np.ndarray, sr: int, f0: float) -> Tuple[List[float], float]:
        """Analyze harmonic structure and compute THD."""
        # FFT of entire signal
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        harmonics = []
        harmonic_powers = []
        
        # Extract harmonics
        for h in range(1, 11):  # Up to 10th harmonic
            target_freq = h * f0
            
            # Find closest bin
            closest_bin = np.argmin(np.abs(freqs - target_freq))
            
            # Get peak around expected frequency
            search_range = 5  # bins
            start = max(0, closest_bin - search_range)
            end = min(len(magnitude), closest_bin + search_range + 1)
            
            peak_bin = start + np.argmax(magnitude[start:end])
            peak_freq = freqs[peak_bin]
            peak_power = magnitude[peak_bin] ** 2
            
            harmonics.append(float(peak_freq))
            harmonic_powers.append(peak_power)
        
        # Calculate THD
        if harmonic_powers[0] > 0:  # Fundamental power
            harmonic_sum = sum(harmonic_powers[1:])  # Sum of harmonics
            thd = np.sqrt(harmonic_sum / harmonic_powers[0]) * 100
        else:
            thd = 0.0
        
        return harmonics, float(thd)
    
    def _spectral_entropy(self, magnitude: np.ndarray) -> float:
        """Calculate spectral entropy."""
        # Average over time
        avg_spectrum = np.mean(magnitude, axis=1)
        
        # Normalize to probability distribution
        prob = avg_spectrum / (np.sum(avg_spectrum) + 1e-10)
        
        # Compute entropy
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(prob))
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy)
    
    def _compute_quality_score(self, features: Dict[str, np.ndarray], anomalies: List[Dict]) -> float:
        """
        Compute overall quality score based on features and anomalies.
        
        Args:
            features: Extracted spectral features
            anomalies: Detected anomalies
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Feature-based scoring
        # Penalize high spectral flux (instability)
        if 'spectral_flux' in features:
            flux_penalty = np.mean(features['spectral_flux']) * 0.1
            score -= min(flux_penalty, 0.2)
        
        # Reward good spectral structure
        if 'spectral_flatness' in features:
            # For speech/music, lower flatness is better (more tonal structure)
            flatness = np.mean(features['spectral_flatness'])
            # Map flatness to quality: 0.0-0.3 is good, 0.3-0.7 is okay, 0.7-1.0 is poor
            if flatness < 0.3:
                flatness_score = 1.0
            elif flatness < 0.7:
                flatness_score = 0.8 - (flatness - 0.3) * 0.5
            else:
                flatness_score = 0.6 - (flatness - 0.7) * 2.0
            score = score * 0.8 + flatness_score * 0.2
        
        # Anomaly-based penalties (capped to prevent negative scores)
        anomaly_penalties = {
            'spectral_hole': 0.02,  # Reduced penalties
            'harmonic_distortion': 0.03,
            'aliasing': 0.04,
            'codec_artifact': 0.02
        }
        
        total_penalty = 0
        for anomaly in anomalies:
            penalty = anomaly_penalties.get(anomaly['type'], 0.01)
            if anomaly.get('severity') == 'high':
                penalty *= 1.5
            total_penalty += penalty
        
        # Cap total penalty at 0.5 to prevent score going to 0
        total_penalty = min(total_penalty, 0.5)
        score -= total_penalty
        
        # Ensure score is in valid range
        score = np.clip(score, 0.0, 1.0)
        
        return float(score)
    
    def _generate_visualization_data(self, stft: np.ndarray, sr: int, anomalies: List[Dict]) -> Dict[str, Any]:
        """Generate data for visualization."""
        magnitude = np.abs(stft)
        
        # Convert to dB scale
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Time and frequency axes
        time_axis = np.arange(stft.shape[1]) * self.hop_length / sr
        freq_axis = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Create anomaly mask
        anomaly_mask = np.zeros_like(magnitude, dtype=bool)
        
        for anomaly in anomalies:
            if 'time_frame' in anomaly:
                t = anomaly['time_frame']
                
                if anomaly['type'] == 'spectral_hole' and 'frequency_range' in anomaly:
                    f_start, f_end = anomaly['frequency_range']
                    
                    # Convert to bins
                    bin_start = int(f_start * self.n_fft / sr)
                    bin_end = int(f_end * self.n_fft / sr)
                    
                    if 0 <= t < anomaly_mask.shape[1]:
                        anomaly_mask[bin_start:bin_end, t] = True
        
        return {
            'spectrogram': magnitude_db,
            'time_axis': time_axis,
            'frequency_axis': freq_axis,
            'anomaly_mask': anomaly_mask
        }


class AdvancedSpectralAnalyzer(SpectralAnalyzer):
    """
    Advanced spectral analyzer with integration to existing spectral processing.
    Extends base SpectralAnalyzer with additional features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced analyzer."""
        super().__init__(config)
        
        # Try to import parent class
        try:
            from processors.audio_enhancement.engines.spectral_gating import SpectralGatingEngine
            self.spectral_gating = SpectralGatingEngine()
            self.has_spectral_gating = True
        except ImportError:
            self.has_spectral_gating = False
            logger.warning("SpectralGatingEngine not available")
    
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        """
        Process audio with spectral gating (compatibility method).
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            **kwargs: Additional parameters
            
        Returns:
            Processed audio
        """
        if self.has_spectral_gating:
            return self.spectral_gating.process(audio, sample_rate, **kwargs)
        else:
            # Fallback - return original
            return audio
    
    def analyze_with_enhancement(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Analyze audio and suggest enhancements based on findings.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Analysis results with enhancement suggestions
        """
        # Perform standard analysis
        results = self.analyze(audio, sample_rate)
        
        # Add enhancement suggestions based on anomalies
        suggestions = []
        
        for anomaly in results.get('anomalies', []):
            if anomaly['type'] == 'spectral_hole':
                suggestions.append({
                    'type': 'spectral_interpolation',
                    'target_range': anomaly['frequency_range'],
                    'method': 'harmonic_synthesis'
                })
            elif anomaly['type'] == 'harmonic_distortion':
                suggestions.append({
                    'type': 'harmonic_suppression',
                    'target_harmonics': list(range(2, 10)),
                    'suppression_factor': 0.5
                })
            elif anomaly['type'] == 'codec_artifact':
                suggestions.append({
                    'type': 'bandwidth_extension',
                    'target_frequency': anomaly['cutoff_frequency'],
                    'extension_method': 'spectral_replication'
                })
        
        results['enhancement_suggestions'] = suggestions
        
        return results


class SpectralQualityValidator:
    """Validate spectral analysis quality."""
    
    def __init__(self):
        """Initialize quality validator."""
        self.checks = {
            'frequency_resolution': self._check_frequency_resolution,
            'time_resolution': self._check_time_resolution,
            'dynamic_range': self._check_dynamic_range,
            'feature_validity': self._check_feature_ranges
        }
    
    def validate_analysis(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate analysis results meet quality standards.
        
        Args:
            results: Analysis results to validate
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        for check_name, check_func in self.checks.items():
            try:
                validation_results[check_name] = check_func(results)
            except Exception as e:
                logger.error(f"Validation check {check_name} failed: {str(e)}")
                validation_results[check_name] = False
        
        return validation_results
    
    def _check_frequency_resolution(self, results: Dict[str, Any]) -> bool:
        """Check if frequency resolution is adequate."""
        if 'stft' in results:
            n_freq_bins = results['stft'].shape[0]
            return n_freq_bins >= 513  # At least 1024-point FFT
        return False
    
    def _check_time_resolution(self, results: Dict[str, Any]) -> bool:
        """Check if time resolution is adequate."""
        if 'stft' in results:
            n_time_frames = results['stft'].shape[1]
            return n_time_frames >= 10  # At least 10 frames
        return False
    
    def _check_dynamic_range(self, results: Dict[str, Any]) -> bool:
        """Check if dynamic range is reasonable."""
        if 'visualization' in results and 'spectrogram' in results['visualization']:
            spec_db = results['visualization']['spectrogram']
            dynamic_range = np.max(spec_db) - np.min(spec_db)
            return 40 <= dynamic_range <= 120  # Reasonable dB range
        return False
    
    def _check_feature_ranges(self, results: Dict[str, Any]) -> bool:
        """Check if features are in valid ranges."""
        if 'features' not in results:
            return False
        
        features = results['features']
        
        # Check spectral centroid
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid']
            if np.any(centroid < 0) or np.any(centroid > 1):
                return False
        
        # Check spectral flatness
        if 'spectral_flatness' in features:
            flatness = features['spectral_flatness']
            if np.any(flatness < 0) or np.any(flatness > 1):
                return False
        
        return True