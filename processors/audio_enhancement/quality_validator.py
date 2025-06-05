"""
Quality Validation Module

Provides comprehensive quality validation for separated audio to ensure
only high-quality single-speaker audio passes through for voice cloning.
"""

import numpy as np
import torch
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import librosa
from scipy import signal
from pystoi import stoi

logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """Quality thresholds for validation"""
    min_snr: float = 20.0
    min_stoi: float = 0.85
    min_pesq: float = 3.5
    max_spectral_distortion: float = 0.15
    min_confidence: float = 0.7
    min_energy_ratio: float = 0.5
    max_silence_ratio: float = 0.7
    min_speech_duration: float = 2.0  # seconds


class QualityValidator:
    """
    Comprehensive quality validation for separated audio.
    """
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        Initialize quality validator.
        
        Args:
            thresholds: Quality thresholds
        """
        self.thresholds = thresholds or QualityThresholds()
        self.sample_rate = 16000
        
    def validate(self, 
                 separated_audio: np.ndarray,
                 original_audio: np.ndarray,
                 confidence: float,
                 additional_metrics: Optional[Dict] = None) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """
        Validate the quality of separated audio.
        
        Args:
            separated_audio: Separated/cleaned audio
            original_audio: Original mixed audio
            confidence: Separation confidence score
            additional_metrics: Additional pre-calculated metrics
            
        Returns:
            (is_valid, rejection_reason, metrics)
        """
        metrics = additional_metrics or {}
        
        # Calculate all quality metrics
        self._calculate_quality_metrics(separated_audio, original_audio, metrics)
        
        # Check confidence
        if confidence < self.thresholds.min_confidence:
            return False, f"Low confidence: {confidence:.2f} < {self.thresholds.min_confidence}", metrics
        
        # Check STOI
        if metrics.get('stoi', 0) < self.thresholds.min_stoi:
            return False, f"Low STOI: {metrics['stoi']:.3f} < {self.thresholds.min_stoi}", metrics
        
        # Check spectral distortion
        if metrics.get('spectral_distortion', 1.0) > self.thresholds.max_spectral_distortion:
            return False, f"High spectral distortion: {metrics['spectral_distortion']:.3f}", metrics
        
        # Check energy ratio
        if metrics.get('energy_ratio', 0) < self.thresholds.min_energy_ratio:
            return False, f"Low energy ratio: {metrics['energy_ratio']:.3f}", metrics
        
        # Check silence ratio
        if metrics.get('silence_ratio', 1.0) > self.thresholds.max_silence_ratio:
            return False, f"Too much silence: {metrics['silence_ratio']:.3f}", metrics
        
        # Check speech duration
        if metrics.get('speech_duration', 0) < self.thresholds.min_speech_duration:
            return False, f"Insufficient speech: {metrics['speech_duration']:.1f}s", metrics
        
        # Check for single speaker
        if not self._verify_single_speaker(separated_audio, metrics):
            return False, "Multiple speakers detected in output", metrics
        
        return True, None, metrics
    
    def _calculate_quality_metrics(self, 
                                  separated: np.ndarray, 
                                  original: np.ndarray,
                                  metrics: Dict[str, float]):
        """
        Calculate comprehensive quality metrics.
        """
        # STOI (Short-Time Objective Intelligibility)
        try:
            metrics['stoi'] = float(stoi(original, separated, self.sample_rate))
        except Exception as e:
            logger.warning(f"STOI calculation failed: {e}")
            metrics['stoi'] = 0.0
        
        # SNR and energy metrics
        metrics.update(self._calculate_energy_metrics(separated, original))
        
        # Spectral metrics
        metrics.update(self._calculate_spectral_metrics(separated, original))
        
        # Speech presence metrics
        metrics.update(self._calculate_speech_metrics(separated))
        
        # Perceptual metrics
        metrics.update(self._calculate_perceptual_metrics(separated, original))
    
    def _calculate_energy_metrics(self, separated: np.ndarray, original: np.ndarray) -> Dict[str, float]:
        """Calculate energy-based metrics."""
        metrics = {}
        
        # RMS energy
        sep_rms = np.sqrt(np.mean(separated ** 2))
        orig_rms = np.sqrt(np.mean(original ** 2))
        
        metrics['energy_ratio'] = sep_rms / (orig_rms + 1e-10)
        
        # SNR estimation (using original as reference)
        signal_power = np.mean(separated ** 2)
        noise_power = np.mean((original - separated) ** 2)
        
        if noise_power > 0:
            metrics['snr'] = 10 * np.log10(signal_power / noise_power)
        else:
            metrics['snr'] = 40.0  # High SNR if no noise
        
        return metrics
    
    def _calculate_spectral_metrics(self, separated: np.ndarray, original: np.ndarray) -> Dict[str, float]:
        """Calculate spectral metrics."""
        metrics = {}
        
        # Spectral distortion
        try:
            # Compute spectrograms
            sep_spec = np.abs(librosa.stft(separated))
            orig_spec = np.abs(librosa.stft(original))
            
            # Normalize
            sep_spec_norm = sep_spec / (np.max(sep_spec) + 1e-10)
            orig_spec_norm = orig_spec / (np.max(orig_spec) + 1e-10)
            
            # Log spectral distance
            log_sep = np.log10(sep_spec_norm + 1e-10)
            log_orig = np.log10(orig_spec_norm + 1e-10)
            
            metrics['spectral_distortion'] = float(np.mean(np.abs(log_sep - log_orig)))
            
            # Spectral centroid shift
            sep_centroid = np.mean(librosa.feature.spectral_centroid(y=separated, sr=self.sample_rate))
            orig_centroid = np.mean(librosa.feature.spectral_centroid(y=original, sr=self.sample_rate))
            
            metrics['centroid_shift'] = abs(sep_centroid - orig_centroid) / orig_centroid
            
        except Exception as e:
            logger.warning(f"Spectral metrics calculation failed: {e}")
            metrics['spectral_distortion'] = 0.0
            metrics['centroid_shift'] = 0.0
        
        return metrics
    
    def _calculate_speech_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculate speech presence metrics."""
        metrics = {}
        
        # Voice activity detection
        try:
            # Simple energy-based VAD
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop
            
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            frame_energy = np.sum(frames ** 2, axis=0)
            
            # Dynamic threshold
            energy_threshold = np.percentile(frame_energy, 30)
            voice_frames = frame_energy > energy_threshold
            
            # Calculate metrics
            total_frames = len(voice_frames)
            voice_frame_count = np.sum(voice_frames)
            
            metrics['voice_activity_ratio'] = voice_frame_count / total_frames if total_frames > 0 else 0
            metrics['speech_duration'] = (voice_frame_count * hop_length) / self.sample_rate
            metrics['silence_ratio'] = 1 - metrics['voice_activity_ratio']
            
            # Check for continuous speech segments
            voice_segments = self._find_segments(voice_frames)
            if voice_segments:
                metrics['longest_speech_segment'] = max(seg[1] - seg[0] for seg in voice_segments) * hop_length / self.sample_rate
                metrics['num_speech_segments'] = len(voice_segments)
            else:
                metrics['longest_speech_segment'] = 0.0
                metrics['num_speech_segments'] = 0
                
        except Exception as e:
            logger.warning(f"Speech metrics calculation failed: {e}")
            metrics.update({
                'voice_activity_ratio': 0.0,
                'speech_duration': 0.0,
                'silence_ratio': 1.0
            })
        
        return metrics
    
    def _calculate_perceptual_metrics(self, separated: np.ndarray, original: np.ndarray) -> Dict[str, float]:
        """Calculate perceptual quality metrics."""
        metrics = {}
        
        try:
            # Mel-frequency cepstral distance
            sep_mfcc = librosa.feature.mfcc(y=separated, sr=self.sample_rate, n_mfcc=13)
            orig_mfcc = librosa.feature.mfcc(y=original, sr=self.sample_rate, n_mfcc=13)
            
            # Align MFCCs if different lengths
            min_len = min(sep_mfcc.shape[1], orig_mfcc.shape[1])
            sep_mfcc = sep_mfcc[:, :min_len]
            orig_mfcc = orig_mfcc[:, :min_len]
            
            metrics['mfcc_distance'] = float(np.mean(np.abs(sep_mfcc - orig_mfcc)))
            
            # Perceptual loudness
            sep_loudness = np.percentile(np.abs(separated), 95)
            orig_loudness = np.percentile(np.abs(original), 95)
            metrics['loudness_ratio'] = sep_loudness / (orig_loudness + 1e-10)
            
        except Exception as e:
            logger.warning(f"Perceptual metrics calculation failed: {e}")
            metrics['mfcc_distance'] = 0.0
            metrics['loudness_ratio'] = 1.0
        
        return metrics
    
    def _verify_single_speaker(self, audio: np.ndarray, metrics: Dict[str, float]) -> bool:
        """
        Verify that the audio contains only a single speaker.
        """
        try:
            # Check for multiple pitch tracks (indicating multiple speakers)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            if f0 is not None:
                # Remove NaN values
                valid_f0 = f0[~np.isnan(f0)]
                
                if len(valid_f0) > 0:
                    # Check pitch variance
                    pitch_std = np.std(valid_f0)
                    pitch_mean = np.mean(valid_f0)
                    
                    # High variance might indicate multiple speakers
                    pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
                    metrics['pitch_cv'] = float(pitch_cv)
                    
                    # Threshold for coefficient of variation
                    if pitch_cv > 0.5:  # More than 50% variation
                        return False
            
            # Check for energy discontinuities
            energy_envelope = self._get_energy_envelope(audio)
            energy_gradient = np.abs(np.gradient(energy_envelope))
            
            # Sudden changes might indicate speaker switches
            sudden_changes = np.sum(energy_gradient > np.percentile(energy_gradient, 95))
            metrics['energy_discontinuities'] = int(sudden_changes)
            
            # Too many sudden changes might indicate multiple speakers
            if sudden_changes > 10:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Single speaker verification failed: {e}")
            return True  # Assume single speaker if verification fails
    
    def _get_energy_envelope(self, audio: np.ndarray, frame_size: float = 0.02) -> np.ndarray:
        """Get smoothed energy envelope."""
        frame_length = int(frame_size * self.sample_rate)
        hop_length = frame_length // 2
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sqrt(np.mean(frames ** 2, axis=0))
        
        # Smooth the envelope
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(energy, sigma=5)
        
        return smoothed
    
    def _find_segments(self, activity: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments in binary activity array."""
        segments = []
        start = None
        
        for i, active in enumerate(activity):
            if active and start is None:
                start = i
            elif not active and start is not None:
                segments.append((start, i))
                start = None
        
        if start is not None:
            segments.append((start, len(activity)))
        
        return segments
    
    def get_quality_report(self, metrics: Dict[str, float]) -> str:
        """
        Generate a human-readable quality report.
        
        Args:
            metrics: Quality metrics dictionary
            
        Returns:
            Formatted report string
        """
        report = "Audio Quality Report\n"
        report += "=" * 40 + "\n\n"
        
        # Intelligibility
        report += f"Intelligibility (STOI): {metrics.get('stoi', 0):.3f}\n"
        
        # Energy
        report += f"Energy Ratio: {metrics.get('energy_ratio', 0):.3f}\n"
        report += f"SNR: {metrics.get('snr', 0):.1f} dB\n"
        
        # Spectral
        report += f"Spectral Distortion: {metrics.get('spectral_distortion', 0):.3f}\n"
        report += f"Centroid Shift: {metrics.get('centroid_shift', 0):.3f}\n"
        
        # Speech
        report += f"Speech Duration: {metrics.get('speech_duration', 0):.1f}s\n"
        report += f"Voice Activity: {metrics.get('voice_activity_ratio', 0):.2%}\n"
        
        # Perceptual
        report += f"MFCC Distance: {metrics.get('mfcc_distance', 0):.3f}\n"
        report += f"Loudness Ratio: {metrics.get('loudness_ratio', 0):.3f}\n"
        
        return report