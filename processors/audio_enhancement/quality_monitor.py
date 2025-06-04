"""Quality monitoring for audio enhancement."""

import numpy as np
from scipy import signal
import librosa


class QualityMonitor:
    """Monitor audio quality during enhancement."""
    
    def __init__(self):
        self.thresholds = {
            "spectral_distortion": 0.15,
            "phase_coherence": 0.9,
            "harmonic_preservation": 0.85
        }
    
    def check_naturalness(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Check naturalness of enhanced audio compared to original.
        
        Args:
            original: Original audio signal
            enhanced: Enhanced audio signal
            
        Returns:
            Naturalness score between 0 and 1
        """
        if len(original) != len(enhanced):
            # Pad or truncate to match lengths
            min_len = min(len(original), len(enhanced))
            original = original[:min_len]
            enhanced = enhanced[:min_len]
        
        if len(original) == 0:
            return 0.0
        
        # Calculate individual metrics
        spectral_dist = self._measure_spectral_distortion(original, enhanced)
        phase_coh = self._measure_phase_coherence(original, enhanced)
        harmonic_pres = self._measure_harmonic_preservation(original, enhanced)
        
        # Weight the metrics
        weights = {
            "spectral": 0.4,
            "phase": 0.3,
            "harmonic": 0.3
        }
        
        # Calculate scores (1 - normalized_error)
        spectral_score = max(0, 1 - spectral_dist / self.thresholds["spectral_distortion"])
        phase_score = max(0, phase_coh)  # Already normalized
        harmonic_score = max(0, harmonic_pres)  # Already normalized
        
        # Weighted average
        naturalness = (
            weights["spectral"] * spectral_score +
            weights["phase"] * phase_score +
            weights["harmonic"] * harmonic_score
        )
        
        return float(np.clip(naturalness, 0, 1))
    
    def _measure_spectral_distortion(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Measure spectral distortion between original and enhanced."""
        # Compute mel-spectrograms
        n_mels = 128
        hop_length = 512
        
        # Ensure minimum length for mel spectrogram
        if len(original) < 2048:
            return 0.0
        
        try:
            mel_orig = librosa.feature.melspectrogram(
                y=original, sr=16000, n_mels=n_mels, hop_length=hop_length
            )
            mel_enh = librosa.feature.melspectrogram(
                y=enhanced, sr=16000, n_mels=n_mels, hop_length=hop_length
            )
            
            # Convert to log scale
            mel_orig_db = librosa.power_to_db(mel_orig + 1e-10)
            mel_enh_db = librosa.power_to_db(mel_enh + 1e-10)
            
            # Calculate MSE in log domain
            mse = np.mean((mel_orig_db - mel_enh_db) ** 2)
            
            # Normalize by typical range (80 dB)
            distortion = np.sqrt(mse) / 80
            
            return float(distortion)
        except Exception:
            return 0.1  # Default moderate distortion on error
    
    def _measure_phase_coherence(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Measure phase coherence between original and enhanced."""
        # STFT parameters
        n_fft = 2048
        hop_length = 512
        
        if len(original) < n_fft:
            return 1.0  # Perfect coherence for short signals
        
        try:
            # Compute STFT
            orig_stft = librosa.stft(original, n_fft=n_fft, hop_length=hop_length)
            enh_stft = librosa.stft(enhanced, n_fft=n_fft, hop_length=hop_length)
            
            # Extract phase
            orig_phase = np.angle(orig_stft)
            enh_phase = np.angle(enh_stft)
            
            # Calculate phase difference
            phase_diff = np.abs(orig_phase - enh_phase)
            
            # Wrap phase difference to [-pi, pi]
            phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
            
            # Calculate coherence (1 - normalized phase error)
            phase_error = np.mean(np.abs(phase_diff)) / np.pi
            coherence = 1.0 - phase_error
            
            return float(coherence)
        except Exception:
            return 0.9  # Default high coherence on error
    
    def _measure_harmonic_preservation(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Measure how well harmonics are preserved."""
        # Find harmonic peaks in original
        orig_peaks = self._find_harmonic_peaks(original)
        enh_peaks = self._find_harmonic_peaks(enhanced)
        
        if len(orig_peaks) == 0:
            return 1.0  # No harmonics to preserve
        
        # Match peaks and calculate preservation
        preservation_scores = []
        
        for orig_freq, orig_mag in orig_peaks[:10]:  # Check first 10 harmonics
            # Find closest peak in enhanced
            if len(enh_peaks) > 0:
                freq_diffs = np.abs([f for f, _ in enh_peaks] - orig_freq)
                closest_idx = np.argmin(freq_diffs)
                
                if freq_diffs[closest_idx] < 20:  # Within 20 Hz
                    enh_freq, enh_mag = enh_peaks[closest_idx]
                    # Calculate magnitude preservation
                    mag_ratio = min(enh_mag / orig_mag, 2.0)  # Cap at 2x amplification
                    preservation_scores.append(mag_ratio)
                else:
                    preservation_scores.append(0.0)  # Harmonic lost
            else:
                preservation_scores.append(0.0)
        
        if preservation_scores:
            # Average preservation, penalize both loss and excessive amplification
            avg_preservation = np.mean([min(s, 1.0) for s in preservation_scores])
            return float(avg_preservation)
        else:
            return 1.0
    
    def _find_harmonic_peaks(self, audio: np.ndarray, sample_rate: int = 16000) -> list:
        """Find harmonic peaks in audio spectrum."""
        if len(audio) < 2048:
            return []
        
        try:
            # Compute magnitude spectrum
            freqs, psd = signal.welch(audio, sample_rate, nperseg=2048)
            
            # Find peaks
            peaks, properties = signal.find_peaks(
                psd,
                height=np.max(psd) * 0.1,  # At least 10% of max
                distance=20  # Minimum 20 bins apart
            )
            
            if len(peaks) == 0:
                return []
            
            # Sort by magnitude
            peak_mags = psd[peaks]
            sorted_idx = np.argsort(peak_mags)[::-1]
            
            # Return (frequency, magnitude) pairs
            peak_list = [(freqs[peaks[i]], peak_mags[i]) for i in sorted_idx]
            
            return peak_list
        except Exception:
            return []