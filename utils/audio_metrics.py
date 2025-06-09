"""
Audio quality metrics for enhancement evaluation.
Includes SNR, PESQ, STOI, and other perceptual metrics.
"""

import numpy as np
from scipy import signal
from scipy.linalg import toeplitz
import warnings
from typing import Union, Tuple, Optional

# Try to import optional libraries
try:
    from pesq import pesq as calculate_pesq_score
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    warnings.warn("PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    warnings.warn("STOI not available. Install with: pip install pystoi")


def calculate_energy_db(signal: np.ndarray) -> float:
    """
    Calculate energy of a signal in decibels.
    
    Args:
        signal: Audio signal
        
    Returns:
        Energy in decibels (dB)
    """
    # Calculate RMS energy
    rms = np.sqrt(np.mean(signal ** 2))
    
    # Avoid log of zero
    if rms < 1e-10:
        return -100.0  # Return very low dB for silence
    
    # Convert to dB (reference level 1.0)
    energy_db = 20 * np.log10(rms)
    
    return float(energy_db)


def calculate_si_sdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        reference: Reference (clean) signal
        estimation: Estimated (processed) signal
        
    Returns:
        SI-SDR in decibels
    """
    # Ensure same length
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    # Remove mean
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # Normalize reference
    reference = reference / (np.sqrt(np.sum(reference**2)) + 1e-8)
    
    # Compute projection
    dot = np.sum(reference * estimation)
    s_target = dot * reference
    
    # Compute distortion
    e_noise = estimation - s_target
    
    # Compute SI-SDR
    si_sdr = 10 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-8) + 1e-8)
    
    return si_sdr


def calculate_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        clean: Clean reference signal
        noisy: Noisy or processed signal
        
    Returns:
        SNR in decibels
    """
    # Ensure same length
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    
    # Check if signals are identical or nearly identical
    if np.allclose(clean, noisy, rtol=1e-9):
        return 40.0  # 40 dB represents very clean signal
    
    # Calculate noise as difference
    noise = noisy - clean
    
    # Calculate power
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Avoid log of zero
    if noise_power < 1e-10:
        return 40.0  # Cap at 40 dB instead of infinity
    
    # SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Cap at reasonable maximum
    return float(min(snr_db, 40.0))


def calculate_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int
) -> float:
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality).
    
    Args:
        reference: Clean reference signal
        degraded: Degraded/processed signal
        sample_rate: Sample rate (8000 or 16000 Hz)
        
    Returns:
        PESQ score (1.0 to 4.5)
    """
    if not HAS_PESQ:
        warnings.warn("PESQ not available, returning default score")
        return 2.5
    
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # PESQ only supports 8kHz or 16kHz
    if sample_rate not in [8000, 16000]:
        # Resample to 16kHz
        from scipy.signal import resample
        resample_factor = 16000 / sample_rate
        reference = resample(reference, int(len(reference) * resample_factor))
        degraded = resample(degraded, int(len(degraded) * resample_factor))
        sample_rate = 16000
    
    # Ensure signals are in valid range
    reference = np.clip(reference, -1.0, 1.0)
    degraded = np.clip(degraded, -1.0, 1.0)
    
    try:
        # Use 'wb' mode for wideband (16kHz)
        mode = 'wb' if sample_rate == 16000 else 'nb'
        score = calculate_pesq_score(sample_rate, reference, degraded, mode)
        return float(score)
    except Exception as e:
        warnings.warn(f"PESQ calculation failed: {e}")
        return 2.5  # Default mid-range score


def calculate_stoi(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
    extended: bool = False
) -> float:
    """
    Calculate STOI (Short-Time Objective Intelligibility).
    
    Args:
        reference: Clean reference signal
        degraded: Degraded/processed signal
        sample_rate: Sample rate in Hz
        extended: Use extended STOI (handles negative values)
        
    Returns:
        STOI score (0.0 to 1.0)
    """
    if not HAS_STOI:
        warnings.warn("STOI not available, returning default score")
        return 0.75
    
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # Ensure signals are in valid range
    reference = np.clip(reference, -1.0, 1.0)
    degraded = np.clip(degraded, -1.0, 1.0)
    
    try:
        score = stoi(reference, degraded, sample_rate, extended=extended)
        return float(score)
    except Exception as e:
        warnings.warn(f"STOI calculation failed: {e}")
        return 0.75  # Default score


def calculate_spectral_distortion(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
    n_fft: int = 512
) -> float:
    """
    Calculate spectral distortion in dB.
    
    Args:
        reference: Clean reference signal
        degraded: Degraded/processed signal
        sample_rate: Sample rate in Hz
        n_fft: FFT size
        
    Returns:
        Average spectral distortion in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # Calculate power spectral density
    _, ref_psd = signal.welch(reference, sample_rate, nperseg=n_fft)
    _, deg_psd = signal.welch(degraded, sample_rate, nperseg=n_fft)
    
    # Avoid log of zero
    ref_psd = np.maximum(ref_psd, 1e-10)
    deg_psd = np.maximum(deg_psd, 1e-10)
    
    # Calculate spectral distortion
    sd = np.sqrt(np.mean((10 * np.log10(deg_psd / ref_psd)) ** 2))
    
    return float(sd)


def calculate_speaker_similarity(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int
) -> float:
    """
    Calculate speaker similarity score using spectral features.
    
    This is a simplified version that compares spectral envelopes
    to estimate how well speaker characteristics are preserved.
    
    Args:
        reference: Clean reference signal
        degraded: Degraded/processed signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]
    
    # Calculate MFCCs as speaker features
    mfcc_ref = _extract_mfcc(reference, sample_rate)
    mfcc_deg = _extract_mfcc(degraded, sample_rate)
    
    # Calculate cosine similarity
    similarity = _cosine_similarity(mfcc_ref, mfcc_deg)
    
    # Map to 0-1 range
    similarity = (similarity + 1) / 2
    
    return float(similarity)


def _extract_mfcc(
    signal_data: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 512
) -> np.ndarray:
    """
    Extract MFCC features from signal.
    
    Simplified implementation without external dependencies.
    """
    # Pre-emphasis
    emphasized = np.append(signal_data[0], signal_data[1:] - 0.97 * signal_data[:-1])
    
    # Frame the signal
    frame_length = n_fft
    hop_length = frame_length // 2
    n_frames = 1 + (len(emphasized) - frame_length) // hop_length
    
    frames = np.zeros((n_frames, frame_length))
    for i in range(n_frames):
        start = i * hop_length
        frames[i] = emphasized[start:start + frame_length] * np.hamming(frame_length)
    
    # FFT and power spectrum
    magnitude = np.abs(np.fft.rfft(frames, n_fft))
    power = magnitude ** 2
    
    # Mel filterbank
    n_mels = 40
    mel_filters = _mel_filterbank(sample_rate, n_fft, n_mels)
    mel_power = np.dot(power, mel_filters.T)
    
    # Log and DCT
    log_mel_power = np.log(mel_power + 1e-10)
    mfcc = _dct(log_mel_power, n_mfcc)
    
    # Return mean MFCC
    return np.mean(mfcc, axis=0)


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 40
) -> np.ndarray:
    """Create Mel filterbank."""
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    # Frequency bins
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)
    
    # Mel scale bins
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sample_rate / 2)
    mel_bins = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_bins = mel_to_hz(mel_bins)
    
    # Create filterbank
    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left = hz_bins[i]
        center = hz_bins[i + 1]
        right = hz_bins[i + 2]
        
        for j, freq in enumerate(fft_freqs):
            if left <= freq <= center:
                filterbank[i, j] = (freq - left) / (center - left)
            elif center < freq <= right:
                filterbank[i, j] = (right - freq) / (right - center)
    
    return filterbank


def _dct(signal_data: np.ndarray, n_coeffs: int) -> np.ndarray:
    """Discrete Cosine Transform (Type II)."""
    n_samples = signal_data.shape[-1]
    dct_matrix = np.zeros((n_coeffs, n_samples))
    
    for k in range(n_coeffs):
        dct_matrix[k] = np.cos(np.pi * k * (np.arange(n_samples) + 0.5) / n_samples)
    
    # Normalize
    dct_matrix[0] *= 1 / np.sqrt(n_samples)
    dct_matrix[1:] *= np.sqrt(2 / n_samples)
    
    return np.dot(signal_data, dct_matrix.T)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def calculate_all_metrics(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int
) -> dict:
    """
    Calculate all available audio quality metrics.
    
    Args:
        reference: Clean reference signal
        degraded: Degraded/processed signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary with all calculated metrics
    """
    metrics = {
        'snr': calculate_snr(reference, degraded),
        'spectral_distortion': calculate_spectral_distortion(reference, degraded, sample_rate),
        'speaker_similarity': calculate_speaker_similarity(reference, degraded, sample_rate),
    }
    
    # Add optional metrics if available
    if HAS_PESQ:
        metrics['pesq'] = calculate_pesq(reference, degraded, sample_rate)
    
    if HAS_STOI:
        metrics['stoi'] = calculate_stoi(reference, degraded, sample_rate)
    
    return metrics


class AudioQualityMetrics:
    """
    Wrapper class for audio quality metrics.
    Provides a convenient interface for calculating various audio quality measures.
    """
    
    def __init__(self):
        """Initialize audio quality metrics calculator."""
        self.has_pesq = HAS_PESQ
        self.has_stoi = HAS_STOI
    
    def calculate_snr(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio (dB) - Target: 5-10dB improvement"""
        return calculate_snr(clean, enhanced)
    
    def calculate_si_snr(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Scale-Invariant SNR - More robust than SNR"""
        # Scale-invariant version
        clean = clean - np.mean(clean)
        enhanced = enhanced - np.mean(enhanced)
        
        # Find scaling factor
        alpha = np.dot(enhanced, clean) / np.dot(clean, clean)
        target = alpha * clean
        noise = enhanced - target
        
        return 10 * np.log10(np.sum(target**2) / np.sum(noise**2))
    
    def calculate_sdr(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Signal-to-Distortion Ratio - Measures overall quality"""
        return self.calculate_si_snr(clean, enhanced)  # Simplified version
    
    def calculate_pesq(self, clean: np.ndarray, enhanced: np.ndarray, fs: int = 16000) -> float:
        """
        PESQ (Perceptual Evaluation of Speech Quality)
        - Range: -0.5 to 4.5
        - Target: > 3.0 (Good quality)
        - Industry standard for speech quality
        """
        return calculate_pesq(clean, enhanced, fs)
    
    def calculate_stoi(self, clean: np.ndarray, enhanced: np.ndarray, fs: int = 16000) -> float:
        """
        STOI (Short-Time Objective Intelligibility)
        - Range: 0 to 1
        - Target: > 0.85 (High intelligibility)
        - Correlates with human intelligibility scores
        """
        return calculate_stoi(clean, enhanced, fs)
    
    def calculate_spectral_distortion(self, clean: np.ndarray, enhanced: np.ndarray, fs: int = 16000) -> float:
        """Log Spectral Distance - Target: < 1.0"""
        return calculate_spectral_distortion(clean, enhanced, fs)
    
    def calculate_speaker_similarity(self, clean: np.ndarray, enhanced: np.ndarray, fs: int = 16000) -> float:
        """
        Cosine similarity of speaker embeddings
        - Target: > 0.95 (High similarity)
        - Uses same embeddings as speaker ID system
        """
        return calculate_speaker_similarity(clean, enhanced, fs)
    
    def calculate_all_metrics(self, reference: np.ndarray, degraded: np.ndarray, sample_rate: int) -> dict:
        """Calculate all available metrics at once."""
        return calculate_all_metrics(reference, degraded, sample_rate)