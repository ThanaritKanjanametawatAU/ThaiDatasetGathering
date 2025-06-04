"""Noise profiling and characterization utilities."""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import json
from typing import Dict, List, Tuple


class NoiseProfiler:
    """Analyze and characterize noise in audio signals."""
    
    def __init__(self):
        self.frame_size = 2048
        self.hop_size = 512
        self.noise_types = {
            'white': {'color': 0.0, 'periodicity': 0.0},
            'pink': {'color': -1.0, 'periodicity': 0.0},
            'brown': {'color': -2.0, 'periodicity': 0.0},
            'periodic': {'color': None, 'periodicity': 0.8},
            'non_stationary': {'color': None, 'periodicity': None}
        }
    
    def analyze_noise(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Comprehensive noise analysis.
        
        Args:
            audio: numpy array of audio samples (should be noise segment)
            sample_rate: sampling rate in Hz
            
        Returns:
            dict: Noise characteristics
        """
        if len(audio) < self.frame_size:
            return {'type': 'unknown', 'characteristics': {}}
        
        # Frequency analysis
        freq_profile = self._frequency_analysis(audio, sample_rate)
        
        # Temporal analysis
        temporal_profile = self._temporal_analysis(audio, sample_rate)
        
        # Classify noise type
        noise_type = self._classify_noise(freq_profile, temporal_profile)
        
        return {
            'type': noise_type,
            'frequency_profile': freq_profile,
            'temporal_profile': temporal_profile,
            'characteristics': self._get_characteristics(noise_type, freq_profile, temporal_profile)
        }
    
    def _frequency_analysis(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Analyze frequency characteristics of noise."""
        # Compute power spectral density
        freqs, psd = signal.welch(audio, sample_rate, nperseg=self.frame_size)
        
        # Spectral slope (for color classification)
        log_freqs = np.log10(freqs[1:])  # Skip DC
        log_psd = np.log10(psd[1:] + 1e-10)
        slope, _ = np.polyfit(log_freqs, log_psd, 1)
        
        # Find peak frequencies
        peaks, properties = signal.find_peaks(psd, height=np.mean(psd) * 3)
        peak_freqs = freqs[peaks]
        peak_powers = properties['peak_heights']
        
        # Spectral centroid
        centroid = np.sum(freqs * psd) / np.sum(psd)
        
        # Spectral spread
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))
        
        return {
            'spectral_slope': slope,
            'peak_frequencies': peak_freqs.tolist(),
            'peak_powers': peak_powers.tolist(),
            'spectral_centroid': centroid,
            'spectral_spread': spread,
            'psd': psd.tolist(),
            'frequencies': freqs.tolist()
        }
    
    def _temporal_analysis(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Analyze temporal characteristics of noise."""
        # Check stationarity
        frame_length = int(0.1 * sample_rate)  # 100ms frames
        hop_length = int(0.05 * sample_rate)   # 50ms hop
        
        frame_powers = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            power = np.sqrt(np.mean(frame ** 2))
            frame_powers.append(power)
        
        frame_powers = np.array(frame_powers)
        
        # Stationarity measure (coefficient of variation)
        if np.mean(frame_powers) > 0:
            stationarity = 1.0 - (np.std(frame_powers) / np.mean(frame_powers))
        else:
            stationarity = 1.0
        
        # Periodicity detection
        autocorr = np.correlate(audio, audio, mode='same')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find periodic peaks
        min_period = int(0.002 * sample_rate)  # 2ms minimum
        max_period = int(0.1 * sample_rate)    # 100ms maximum
        
        peaks, _ = signal.find_peaks(autocorr[min_period:max_period], height=0.3)
        
        if len(peaks) > 0:
            # Periodicity strength
            periodicity = np.max(autocorr[min_period:max_period])
            period_samples = peaks[0] + min_period
            period_freq = sample_rate / period_samples
        else:
            periodicity = 0.0
            period_freq = 0.0
        
        # Impulse detection
        impulses = np.sum(np.abs(audio) > 3 * np.std(audio))
        impulse_rate = impulses / len(audio)
        
        return {
            'stationarity': stationarity,
            'periodicity': periodicity,
            'period_frequency': period_freq,
            'impulse_rate': impulse_rate,
            'mean_power': np.mean(frame_powers),
            'power_variance': np.var(frame_powers)
        }
    
    def _classify_noise(self, freq_profile: Dict, temporal_profile: Dict) -> str:
        """Classify noise type based on profiles."""
        slope = freq_profile['spectral_slope']
        periodicity = temporal_profile['periodicity']
        stationarity = temporal_profile['stationarity']
        
        # Check for periodic noise (AC hum, etc.)
        if periodicity > 0.7:
            return 'periodic'
        
        # Check for non-stationary noise
        if stationarity < 0.5:
            return 'non_stationary'
        
        # Classify by spectral color
        if -0.5 < slope < 0.5:
            return 'white'
        elif -1.5 < slope <= -0.5:
            return 'pink'
        elif slope <= -1.5:
            return 'brown'
        else:
            return 'colored'
    
    def _get_characteristics(self, noise_type: str, freq_profile: Dict, temporal_profile: Dict) -> Dict:
        """Get detailed characteristics for noise type."""
        chars = {
            'noise_type': noise_type,
            'spectral_slope': freq_profile['spectral_slope'],
            'centroid_hz': freq_profile['spectral_centroid'],
            'spread_hz': freq_profile['spectral_spread'],
            'stationarity': temporal_profile['stationarity'],
            'periodicity': temporal_profile['periodicity']
        }
        
        if noise_type == 'periodic':
            chars['fundamental_frequency'] = temporal_profile['period_frequency']
            if freq_profile['peak_frequencies']:
                chars['harmonic_frequencies'] = freq_profile['peak_frequencies'][:5]
        
        if noise_type == 'non_stationary':
            chars['power_variance'] = temporal_profile['power_variance']
            chars['impulse_rate'] = temporal_profile['impulse_rate']
        
        return chars
    
    def create_noise_profile(self, noise_samples: List[np.ndarray], sample_rate: int, profile_name: str) -> Dict:
        """Create comprehensive noise profile from multiple samples."""
        all_profiles = []
        
        for sample in noise_samples:
            profile = self.analyze_noise(sample, sample_rate)
            all_profiles.append(profile)
        
        # Aggregate characteristics
        aggregated = {
            'profile_name': profile_name,
            'sample_count': len(noise_samples),
            'sample_rate': sample_rate,
            'noise_types': {},
            'average_characteristics': {}
        }
        
        # Count noise types
        for profile in all_profiles:
            noise_type = profile['type']
            if noise_type not in aggregated['noise_types']:
                aggregated['noise_types'][noise_type] = 0
            aggregated['noise_types'][noise_type] += 1
        
        # Average characteristics
        char_keys = ['spectral_slope', 'centroid_hz', 'spread_hz', 'stationarity', 'periodicity']
        for key in char_keys:
            values = [p['characteristics'].get(key, 0) for p in all_profiles]
            aggregated['average_characteristics'][key] = np.mean(values)
        
        return aggregated
    
    def save_noise_database(self, profiles: Dict, filename: str):
        """Save noise profiles to JSON database."""
        with open(filename, 'w') as f:
            json.dump(profiles, f, indent=2)
    
    def load_noise_database(self, filename: str) -> Dict:
        """Load noise profiles from JSON database."""
        with open(filename, 'r') as f:
            return json.load(f)