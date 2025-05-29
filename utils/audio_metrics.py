"""
Audio quality metrics for evaluating enhancement performance.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_snr(reference: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio between reference and processed audio.
    
    Args:
        reference: Reference (clean) audio signal
        processed: Processed audio signal
        
    Returns:
        SNR in dB
    """
    if len(reference) != len(processed):
        min_len = min(len(reference), len(processed))
        reference = reference[:min_len]
        processed = processed[:min_len]
        
    # Calculate signal and noise power
    signal_power = np.mean(reference ** 2)
    noise = reference - processed
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
        
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def calculate_stoi(reference: np.ndarray, processed: np.ndarray, 
                  sample_rate: int, extended: bool = False) -> float:
    """
    Calculate Short-Time Objective Intelligibility (STOI) measure.
    
    Args:
        reference: Reference audio signal
        processed: Processed audio signal
        sample_rate: Sample rate of audio
        extended: Whether to use extended STOI
        
    Returns:
        STOI score between 0 and 1
    """
    try:
        from pystoi import stoi
        
        if len(reference) != len(processed):
            min_len = min(len(reference), len(processed))
            reference = reference[:min_len]
            processed = processed[:min_len]
            
        score = stoi(reference, processed, sample_rate, extended=extended)
        return float(score)
        
    except ImportError:
        logger.warning("pystoi not installed, returning dummy STOI score")
        # Return correlation as rough approximation
        if len(reference) > 0 and len(processed) > 0:
            correlation = np.corrcoef(reference, processed)[0, 1]
            return float(max(0, correlation))
        return 0.5


def calculate_pesq(reference: np.ndarray, degraded: np.ndarray, 
                  sample_rate: int, mode: str = 'wb') -> float:
    """
    Calculate Perceptual Evaluation of Speech Quality (PESQ).
    
    Args:
        reference: Reference audio signal
        degraded: Degraded audio signal
        sample_rate: Sample rate (8000 or 16000)
        mode: 'wb' for wideband (16kHz) or 'nb' for narrowband (8kHz)
        
    Returns:
        PESQ score (-0.5 to 4.5)
    """
    try:
        from pesq import pesq
        
        if len(reference) != len(degraded):
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]
            
        score = pesq(sample_rate, reference, degraded, mode)
        return float(score)
        
    except ImportError:
        logger.warning("pesq not installed, returning dummy PESQ score")
        # Return based on correlation
        if len(reference) > 0 and len(degraded) > 0:
            correlation = np.corrcoef(reference, degraded)[0, 1]
            # Map correlation to PESQ-like range
            pesq_score = 1.0 + (correlation + 1) * 1.75  # Maps to 1.0-4.5
            return float(pesq_score)
        return 2.5


def calculate_si_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        reference: Reference signal
        estimated: Estimated signal
        
    Returns:
        SI-SDR in dB
    """
    if len(reference) != len(estimated):
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]
        
    # Remove mean
    reference = reference - np.mean(reference)
    estimated = estimated - np.mean(estimated)
    
    # Compute SI-SDR
    alpha = np.dot(estimated, reference) / np.dot(reference, reference)
    projection = alpha * reference
    noise = estimated - projection
    
    si_sdr = 10 * np.log10(np.dot(projection, projection) / np.dot(noise, noise))
    
    return float(si_sdr)


def calculate_lsd(reference_spec: np.ndarray, estimated_spec: np.ndarray) -> float:
    """
    Calculate Log Spectral Distance (LSD).
    
    Args:
        reference_spec: Reference spectrogram (magnitude)
        estimated_spec: Estimated spectrogram (magnitude)
        
    Returns:
        LSD value (lower is better)
    """
    # Ensure same shape
    if reference_spec.shape != estimated_spec.shape:
        min_frames = min(reference_spec.shape[0], estimated_spec.shape[0])
        min_bins = min(reference_spec.shape[1], estimated_spec.shape[1])
        reference_spec = reference_spec[:min_frames, :min_bins]
        estimated_spec = estimated_spec[:min_frames, :min_bins]
        
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    
    # Calculate LSD
    log_diff = np.log(reference_spec + eps) - np.log(estimated_spec + eps)
    lsd = np.sqrt(np.mean(log_diff ** 2))
    
    return float(lsd)


def calculate_spectral_convergence(reference_spec: np.ndarray, 
                                 estimated_spec: np.ndarray) -> float:
    """
    Calculate spectral convergence.
    
    Args:
        reference_spec: Reference spectrogram
        estimated_spec: Estimated spectrogram
        
    Returns:
        Spectral convergence value
    """
    # Ensure same shape
    if reference_spec.shape != estimated_spec.shape:
        min_frames = min(reference_spec.shape[0], estimated_spec.shape[0])
        min_bins = min(reference_spec.shape[1], estimated_spec.shape[1])
        reference_spec = reference_spec[:min_frames, :min_bins]
        estimated_spec = estimated_spec[:min_frames, :min_bins]
        
    # Calculate Frobenius norm
    diff_norm = np.linalg.norm(reference_spec - estimated_spec, 'fro')
    ref_norm = np.linalg.norm(reference_spec, 'fro')
    
    if ref_norm == 0:
        return 0.0
        
    convergence = diff_norm / ref_norm
    
    return float(convergence)


def calculate_mcd(reference_mfcc: np.ndarray, estimated_mfcc: np.ndarray) -> float:
    """
    Calculate Mel Cepstral Distortion (MCD).
    
    Args:
        reference_mfcc: Reference MFCC features
        estimated_mfcc: Estimated MFCC features
        
    Returns:
        MCD value in dB
    """
    # Ensure same shape
    if reference_mfcc.shape != estimated_mfcc.shape:
        min_frames = min(reference_mfcc.shape[0], estimated_mfcc.shape[0])
        min_coeffs = min(reference_mfcc.shape[1], estimated_mfcc.shape[1])
        reference_mfcc = reference_mfcc[:min_frames, :min_coeffs]
        estimated_mfcc = estimated_mfcc[:min_frames, :min_coeffs]
        
    # Calculate MCD
    diff = reference_mfcc - estimated_mfcc
    mcd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1))) * (10 / np.log(10)) * np.sqrt(2)
    
    return float(mcd)


def calculate_rmse(reference: np.ndarray, estimated: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        reference: Reference signal
        estimated: Estimated signal
        
    Returns:
        RMSE value
    """
    if len(reference) != len(estimated):
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]
        
    rmse = np.sqrt(np.mean((reference - estimated) ** 2))
    
    return float(rmse)


def calculate_correlation(reference: np.ndarray, estimated: np.ndarray) -> float:
    """
    Calculate correlation coefficient between signals.
    
    Args:
        reference: Reference signal
        estimated: Estimated signal
        
    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(reference) != len(estimated):
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]
        
    if len(reference) == 0:
        return 0.0
        
    correlation = np.corrcoef(reference, estimated)[0, 1]
    
    if np.isnan(correlation):
        return 0.0
        
    return float(correlation)


def evaluate_enhancement_quality(original: np.ndarray, enhanced: np.ndarray,
                               sample_rate: int) -> dict:
    """
    Comprehensive evaluation of audio enhancement quality.
    
    Args:
        original: Original audio signal
        enhanced: Enhanced audio signal
        sample_rate: Sample rate
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['snr'] = calculate_snr(original, enhanced)
    metrics['rmse'] = calculate_rmse(original, enhanced)
    metrics['correlation'] = calculate_correlation(original, enhanced)
    
    # Perceptual metrics
    metrics['stoi'] = calculate_stoi(original, enhanced, sample_rate)
    
    # Try to calculate PESQ if available
    try:
        metrics['pesq'] = calculate_pesq(original, enhanced, sample_rate)
    except Exception:
        metrics['pesq'] = None
        
    # Signal distortion metrics
    metrics['si_sdr'] = calculate_si_sdr(original, enhanced)
    
    return metrics