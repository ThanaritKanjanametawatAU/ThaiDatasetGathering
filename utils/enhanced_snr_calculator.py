"""
Enhanced SNR (Signal-to-Noise Ratio) Calculator Module.
Implements comprehensive SNR calculation with multiple methods, VAD integration,
and advanced features for autonomous audio quality assessment.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, welch
from typing import List, Tuple, Optional, Union, Dict
import warnings
from dataclasses import dataclass
import logging

# Try to import optional VAD libraries
try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    warnings.warn("webrtcvad not available. Install with: pip install webrtcvad")

try:
    from pyannote.audio import Model, Inference
    from pyannote.audio.pipelines import VoiceActivityDetection
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False

# Import existing SNR functionality for backward compatibility
from utils.snr_measurement import SNRMeasurement


# Custom exceptions
class SNRError(Exception):
    """Base exception for SNR calculation errors."""
    pass


class InsufficientDataError(SNRError):
    """Not enough data for reliable SNR estimation."""
    def __init__(self, required_duration: float, actual_duration: float):
        self.required = required_duration
        self.actual = actual_duration
        super().__init__(f"Need {required_duration}s, got {actual_duration}s")


class NoSpeechDetectedError(SNRError):
    """No speech segments found for SNR calculation."""
    pass


class InvalidSignalError(SNRError):
    """Signal contains invalid values (NaN, Inf)."""
    pass


@dataclass
class SNRConfig:
    """Configuration for SNR calculation."""
    # Frame parameters
    frame_size: float = 0.025  # seconds
    frame_shift: float = 0.010  # seconds
    
    # Spectral parameters
    n_fft: int = 2048
    hop_length: int = 512
    
    # Noise estimation
    noise_estimation_method: str = "minimum_statistics"
    noise_window_size: float = 1.5  # seconds
    noise_bias_compensation: float = 1.5
    noise_smoothing_factor: float = 0.98
    
    # VAD parameters
    vad_enabled: bool = True
    vad_backend: str = "energy"  # "webrtcvad", "pyannote", or "energy"
    vad_aggressiveness: int = 2  # 0-3 for webrtcvad
    vad_energy_threshold: float = -40  # dB
    vad_min_speech_duration: float = 0.1  # seconds
    
    # Quality thresholds
    min_duration: float = 0.5  # minimum signal duration
    
    # Performance
    use_cache: bool = True
    parallel_processing: bool = False


class NoiseEstimator:
    """Advanced noise estimation with multiple techniques."""
    
    def __init__(self, config: Optional[SNRConfig] = None):
        self.config = config or SNRConfig()
        self.logger = logging.getLogger(__name__)
    
    def estimate_noise_floor(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate noise floor using minimum statistics or other methods.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Estimated noise power
        """
        if self.config.noise_estimation_method == "minimum_statistics":
            return self._minimum_statistics_estimation(audio, sr)
        elif self.config.noise_estimation_method == "percentile":
            return self._percentile_estimation(audio)
        else:
            return self._percentile_estimation(audio)
    
    def detect_silence_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """
        Detect silence segments in audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of (start_sample, end_sample) tuples
        """
        # Frame-based energy calculation
        frame_length = int(self.config.frame_size * sr)
        hop_length = int(self.config.frame_shift * sr)
        
        # Calculate frame energies
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.mean(frame ** 2)
            energies.append(energy)
        
        if not energies:
            return []
        
        energies = np.array(energies)
        
        # Use absolute threshold based on signal characteristics
        # For pure zeros (silence), energy will be near 0
        max_energy = np.max(energies)
        
        if max_energy < 1e-10:
            # All silence
            return [(0, len(audio))]
        
        # Set threshold at 1% of max energy (40dB below peak)
        threshold = max_energy * 0.01
        
        # Find silence frames
        silence_frames = energies < threshold
        
        # Convert to sample indices
        segments = []
        in_silence = False
        start_frame = 0
        
        for i, is_silence in enumerate(silence_frames):
            if is_silence and not in_silence:
                start_frame = i
                in_silence = True
            elif not is_silence and in_silence:
                start_sample = start_frame * hop_length
                end_sample = i * hop_length
                segments.append((start_sample, end_sample))
                in_silence = False
        
        # Handle last segment
        if in_silence:
            start_sample = start_frame * hop_length
            end_sample = len(audio)
            segments.append((start_sample, end_sample))
        
        # Also check beginning if it starts with silence
        if len(segments) == 0 or segments[0][0] > 0:
            # Check if beginning is silence
            if silence_frames[0]:
                # Find where speech starts
                for i, is_silence in enumerate(silence_frames):
                    if not is_silence:
                        segments.insert(0, (0, i * hop_length))
                        break
        
        return segments
    
    def spectral_noise_estimation(self, stft: np.ndarray) -> np.ndarray:
        """
        Estimate noise spectrum from STFT.
        
        Args:
            stft: Short-time Fourier transform
            
        Returns:
            Estimated noise spectrum
        """
        # Use minimum statistics across time
        magnitude = np.abs(stft)
        
        # Track minimum over sliding window
        window_frames = int(self.config.noise_window_size * 1000 / self.config.frame_shift)
        window_frames = max(window_frames, 10)
        
        min_spectrum = np.min(magnitude[:, :window_frames], axis=1)
        
        for i in range(window_frames, magnitude.shape[1]):
            window_min = np.min(magnitude[:, i-window_frames:i], axis=1)
            min_spectrum = np.minimum(min_spectrum, window_min)
        
        # Apply bias compensation
        return min_spectrum * self.config.noise_bias_compensation
    
    def adaptive_noise_tracking(self, audio: np.ndarray) -> np.ndarray:
        """
        Track noise adaptively throughout the signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Time-varying noise estimate
        """
        # Simple implementation using smoothed minimum tracking
        frame_length = 256
        hop_length = 128
        
        noise_estimate = np.zeros(len(audio))
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frame_power = np.mean(frame ** 2)
            
            if i == 0:
                noise_estimate[i:i + frame_length] = frame_power
            else:
                # Smooth update
                alpha = self.config.noise_smoothing_factor
                prev_estimate = noise_estimate[i - hop_length]
                new_estimate = alpha * prev_estimate + (1 - alpha) * frame_power
                
                # Only update if likely noise (power decreased)
                if new_estimate < prev_estimate * 1.5:
                    noise_estimate[i:i + frame_length] = new_estimate
                else:
                    noise_estimate[i:i + frame_length] = prev_estimate
        
        return noise_estimate
    
    def _minimum_statistics_estimation(self, audio: np.ndarray, sr: int) -> float:
        """Minimum statistics noise estimation."""
        # For sine waves and very clean signals, we need a different approach
        # First, try to detect if this is a clean tonal signal
        
        # Compute spectrum to check for tonality
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
        
        # Find peaks in spectrum
        max_idx = np.argmax(spectrum)
        max_power = spectrum[max_idx]
        median_power = np.median(spectrum)
        
        # If we have a strong tonal component (sine wave)
        if max_power > median_power * 100:  # 40dB above median
            # This is likely a clean tonal signal with noise
            # Use statistical approach on residual
            
            # High-pass filter to remove DC and very low frequencies
            b, a = scipy_signal.butter(5, 20 / (sr/2), btype='high')
            filtered = scipy_signal.filtfilt(b, a, audio)
            
            # Compute local variance to find noise
            window_size = int(0.01 * sr)  # 10ms windows
            local_vars = []
            
            for i in range(0, len(filtered) - window_size, window_size // 2):
                window = filtered[i:i+window_size]
                local_vars.append(np.var(window))
            
            # Noise power is minimum variance
            if local_vars:
                noise_power = np.percentile(local_vars, 5)
            else:
                noise_power = self._percentile_estimation(audio)
                
            return noise_power * self.config.noise_bias_compensation
        
        # Otherwise, use standard minimum statistics
        frame_length = int(0.02 * sr)  # 20ms frames
        hop_length = frame_length // 2
        
        # Calculate frame powers
        frame_powers = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            power = np.mean(frame ** 2)
            frame_powers.append(power)
        
        if not frame_powers:
            return self._percentile_estimation(audio)
        
        frame_powers = np.array(frame_powers)
        
        # Use sliding window minimum tracking
        window_size = int(self.config.noise_window_size * sr / hop_length)
        window_size = max(window_size, 10)
        
        min_powers = []
        for i in range(len(frame_powers)):
            start = max(0, i - window_size // 2)
            end = min(len(frame_powers), i + window_size // 2)
            min_powers.append(np.min(frame_powers[start:end]))
        
        # Take the median of minimum powers
        noise_power = np.median(min_powers)
        
        # Apply bias compensation
        return noise_power * self.config.noise_bias_compensation
    
    def _percentile_estimation(self, audio: np.ndarray) -> float:
        """Percentile-based noise estimation."""
        # Sort absolute values to find noise floor
        sorted_abs = np.sort(np.abs(audio))
        
        # Use bottom 10% for noise estimation
        noise_samples = sorted_abs[:len(sorted_abs)//10]
        
        # RMS of noise floor
        noise_rms = np.sqrt(np.mean(noise_samples ** 2))
        
        return noise_rms ** 2


class VoiceActivityDetector:
    """Unified VAD interface supporting multiple backends."""
    
    def __init__(self, config: Optional[SNRConfig] = None):
        self.config = config or SNRConfig()
        self.backend = None
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the appropriate VAD backend."""
        if self.config.vad_backend == "webrtcvad" and HAS_WEBRTCVAD:
            self.backend = webrtcvad.Vad(self.config.vad_aggressiveness)
        elif self.config.vad_backend == "pyannote" and HAS_PYANNOTE:
            try:
                from utils.huggingface import read_hf_token
                hf_token = read_hf_token()
                self.backend = VoiceActivityDetection(
                    segmentation="pyannote/segmentation-3.0",
                    use_auth_token=hf_token
                )
                self.backend.instantiate({
                    "onset": 0.5,
                    "offset": 0.5,
                    "min_duration_on": self.config.vad_min_speech_duration,
                    "min_duration_off": 0.1
                })
            except Exception:
                self.backend = None
    
    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Boolean mask of voice activity
        """
        if self.config.vad_backend == "webrtcvad" and self.backend:
            return self._webrtc_vad(audio, sr)
        elif self.config.vad_backend == "pyannote" and self.backend:
            return self._pyannote_vad(audio, sr)
        else:
            return self._energy_vad(audio, sr)
    
    def _webrtc_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """WebRTC VAD implementation."""
        # WebRTC VAD requires 16-bit PCM at specific sample rates
        supported_rates = [8000, 16000, 32000, 48000]
        
        # Resample if necessary
        if sr not in supported_rates:
            # Resample to nearest supported rate
            target_rate = min(supported_rates, key=lambda x: abs(x - sr))
            audio = scipy_signal.resample(audio, int(len(audio) * target_rate / sr))
            sr = target_rate
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Process in frames (WebRTC VAD requires 10, 20, or 30 ms frames)
        frame_duration_ms = 20
        frame_length = int(sr * frame_duration_ms / 1000)
        
        vad_mask = np.zeros(len(audio), dtype=bool)
        
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length].tobytes()
            is_speech = self.backend.is_speech(frame, sr)
            vad_mask[i:i + frame_length] = is_speech
        
        return vad_mask
    
    def _pyannote_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """PyAnnote VAD implementation."""
        import torch
        
        # Convert to torch tensor
        waveform = torch.tensor(audio).unsqueeze(0)
        
        # Get VAD output
        vad_output = self.backend({"waveform": waveform, "sample_rate": sr})
        
        # Convert to boolean mask
        vad_mask = np.zeros(len(audio), dtype=bool)
        
        for segment in vad_output:
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            vad_mask[start_sample:end_sample] = True
        
        return vad_mask
    
    def _energy_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Energy-based VAD fallback."""
        # Frame-based energy calculation
        frame_length = int(self.config.frame_size * sr)
        hop_length = int(self.config.frame_shift * sr)
        
        vad_mask = np.zeros(len(audio), dtype=bool)
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            
            # Energy in dB
            energy = np.mean(frame ** 2)
            energy_db = 10 * np.log10(energy + 1e-10)
            
            # Zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length)
            
            # Combined decision
            is_speech = (energy_db > self.config.vad_energy_threshold) and (zcr < 0.5)
            vad_mask[i:i + hop_length] = is_speech
        
        # Post-process: remove short segments
        min_samples = int(self.config.vad_min_speech_duration * sr)
        vad_mask = self._remove_short_segments(vad_mask, min_samples)
        
        return vad_mask
    
    def _remove_short_segments(self, mask: np.ndarray, min_length: int) -> np.ndarray:
        """Remove segments shorter than minimum length."""
        # Find segment boundaries
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Filter short segments
        result = mask.copy()
        for start, end in zip(starts, ends):
            if end - start < min_length:
                result[start:end] = False
        
        return result


class EnhancedSNRCalculator:
    """
    Enhanced SNR Calculator with multiple methods and advanced features.
    """
    
    def __init__(self, config: Optional[SNRConfig] = None):
        self.config = config or SNRConfig()
        self.noise_estimator = NoiseEstimator(self.config)
        self.vad = VoiceActivityDetector(self.config)
        self.legacy_snr = SNRMeasurement()  # For backward compatibility
        self.logger = logging.getLogger(__name__)
        
        # Cache for repeated calculations
        self._cache = {} if self.config.use_cache else None
    
    def calculate_snr(
        self, 
        audio: np.ndarray, 
        sr: int, 
        method: str = 'auto',
        min_duration: Optional[float] = None,
        weighting: Optional[str] = None
    ) -> float:
        """
        Calculate SNR using specified method.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            method: SNR calculation method ('auto', 'waveform', 'spectral', 
                    'segmental', 'vad_enhanced', 'legacy')
            min_duration: Minimum required duration in seconds
            weighting: Frequency weighting ('A', 'C', or None)
            
        Returns:
            SNR in decibels
            
        Raises:
            InvalidSignalError: If signal contains NaN or Inf
            InsufficientDataError: If signal is too short
        """
        # Validate input
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise InvalidSignalError("Signal contains NaN or Inf")
        
        # Check duration
        duration = len(audio) / sr
        min_dur = min_duration or self.config.min_duration
        if duration < min_dur:
            raise InsufficientDataError(min_dur, duration)
        
        # Normalize audio
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Check cache
        cache_key = None
        if self._cache is not None:
            cache_key = self._get_cache_key(audio, sr, method, weighting)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Select method
        if method == 'auto':
            method = self._select_best_method(audio, sr)
        
        # Calculate SNR
        if method == 'legacy':
            snr = self.legacy_snr.measure_snr(audio, sr)
        elif method == 'waveform':
            snr = self._calculate_waveform_snr(audio, sr)
        elif method == 'spectral':
            snr = self._calculate_spectral_snr(audio, sr, weighting)
        elif method == 'segmental':
            snr = self._calculate_segmental_snr(audio, sr)
        elif method == 'vad_enhanced':
            snr = self._calculate_vad_enhanced_snr(audio, sr)
        else:
            # Default to waveform
            snr = self._calculate_waveform_snr(audio, sr)
        
        # Cache result
        if cache_key:
            self._cache[cache_key] = snr
        
        return snr
    
    def calculate_snr_with_confidence(
        self, 
        audio: np.ndarray, 
        sr: int,
        method: str = 'auto'
    ) -> Tuple[float, float]:
        """
        Calculate SNR with confidence score.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            method: SNR calculation method
            
        Returns:
            Tuple of (SNR in dB, confidence score 0-1)
        """
        snr = self.calculate_snr(audio, sr, method)
        
        # Calculate confidence based on various factors
        confidence_factors = []
        
        # Duration factor
        duration = len(audio) / sr
        duration_confidence = min(duration / 5.0, 1.0)  # Full confidence at 5s+
        confidence_factors.append(duration_confidence)
        
        # Stationarity factor
        stationarity = self._estimate_stationarity(audio)
        confidence_factors.append(stationarity)
        
        # VAD confidence
        vad_mask = self.vad.detect(audio, sr)
        speech_ratio = np.sum(vad_mask) / len(vad_mask)
        vad_confidence = 1.0 if 0.2 < speech_ratio < 0.8 else 0.5
        confidence_factors.append(vad_confidence)
        
        # Overall confidence
        confidence = np.mean(confidence_factors)
        
        return snr, confidence
    
    def calculate_multiband_snr(
        self,
        audio: np.ndarray,
        sr: int,
        bands: List[Tuple[float, float]]
    ) -> List[float]:
        """
        Calculate SNR for multiple frequency bands.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            bands: List of (low_freq, high_freq) tuples
            
        Returns:
            List of SNR values for each band
        """
        multiband_snr = []
        
        for low_freq, high_freq in bands:
            # Design bandpass filter
            nyquist = sr / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Handle edge cases
            if low <= 0:
                # Low-pass filter
                b, a = butter(4, high, btype='low')
            elif high >= 1:
                # High-pass filter
                b, a = butter(4, low, btype='high')
            else:
                # Band-pass filter
                b, a = butter(4, [low, high], btype='band')
            
            # Filter signal
            filtered = filtfilt(b, a, audio)
            
            # Calculate SNR for this band
            band_snr = self.calculate_snr(filtered, sr, method='waveform')
            multiband_snr.append(band_snr)
        
        return multiband_snr
    
    def _calculate_waveform_snr(self, audio: np.ndarray, sr: int) -> float:
        """Waveform-based SNR calculation."""
        # For synthetic/clean signals, we need a more sophisticated approach
        
        # First, check if this is a synthetic clean signal
        # Compute local variance to detect uniform noise
        window_size = int(0.01 * sr)  # 10ms windows
        local_vars = []
        
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i+window_size]
            local_vars.append(np.var(window))
        
        if local_vars:
            var_ratio = np.max(local_vars) / (np.min(local_vars) + 1e-10)
            
            # If variance is relatively uniform, likely synthetic signal + noise
            if var_ratio < 10:
                # Use residual-based approach
                # Estimate signal by smoothing
                from scipy.ndimage import gaussian_filter1d
                smoothed = gaussian_filter1d(audio, sigma=5)
                
                # Residual is noise
                residual = audio - smoothed
                
                # Check if residual looks like noise
                if np.abs(np.mean(residual)) < 0.01 * np.std(residual):
                    noise_power = np.mean(residual ** 2)
                    signal_power = np.mean(smoothed ** 2)
                    
                    if noise_power > 1e-10:
                        snr = 10 * np.log10(signal_power / noise_power)
                        return np.clip(snr, -20, 60)
        
        # Standard approach for real audio
        noise_power = self.noise_estimator.estimate_noise_floor(audio, sr)
        
        # For very clean signals, use more conservative noise estimation
        if noise_power > np.mean(audio ** 2) * 0.1:
            # Use statistical approach
            # Compute histogram of sample values
            hist, bin_edges = np.histogram(np.abs(audio), bins=1000)
            
            # Find noise floor from histogram
            cumsum = np.cumsum(hist)
            idx_5percent = np.where(cumsum > len(audio) * 0.05)[0][0]
            noise_level = bin_edges[idx_5percent]
            noise_power = noise_level ** 2
        
        # Total power
        total_power = np.mean(audio ** 2)
        
        # Signal power estimation
        if total_power > noise_power:
            signal_power = total_power - noise_power
        else:
            # Use robust percentile approach
            sorted_abs = np.sort(np.abs(audio))
            noise_level = np.mean(sorted_abs[:len(sorted_abs)//10])
            signal_level = np.mean(sorted_abs[len(sorted_abs)*9//10:])
            noise_power = noise_level ** 2
            signal_power = signal_level ** 2
        
        # Ensure minimum noise floor
        noise_power = max(noise_power, 1e-10)
        signal_power = max(signal_power, noise_power * 1.1)  # Ensure positive SNR for clean signals
        
        # Calculate SNR
        snr = 10 * np.log10(signal_power / noise_power)
        return np.clip(snr, -20, 60)
    
    def _calculate_spectral_snr(
        self, 
        audio: np.ndarray, 
        sr: int,
        weighting: Optional[str] = None
    ) -> float:
        """Spectral domain SNR calculation."""
        # STFT parameters
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        
        # Compute STFT
        f, t, Zxx = scipy_signal.stft(
            audio, 
            fs=sr, 
            nperseg=n_fft, 
            noverlap=n_fft-hop_length,
            window='hann'
        )
        
        # Get VAD for frames
        if self.config.vad_enabled:
            vad_mask = self.vad.detect(audio, sr)
            # Convert sample mask to frame mask
            frame_length = n_fft
            n_frames = Zxx.shape[1]
            frame_vad = np.zeros(n_frames, dtype=bool)
            
            for i in range(n_frames):
                start = i * hop_length
                end = start + frame_length
                if end <= len(vad_mask):
                    frame_vad[i] = np.mean(vad_mask[start:end]) > 0.5
        else:
            frame_vad = np.ones(Zxx.shape[1], dtype=bool)
        
        # Separate speech and noise frames
        speech_frames = Zxx[:, frame_vad]
        noise_frames = Zxx[:, ~frame_vad]
        
        if speech_frames.shape[1] == 0:
            return self._estimate_snr_percentile(audio)
        
        if noise_frames.shape[1] == 0:
            # Estimate noise from minimum statistics
            noise_spectrum = self.noise_estimator.spectral_noise_estimation(Zxx)
            noise_power = np.mean(noise_spectrum ** 2)
        else:
            noise_power = np.mean(np.abs(noise_frames) ** 2)
        
        signal_power = np.mean(np.abs(speech_frames) ** 2)
        
        # Apply frequency weighting if specified
        if weighting == 'A':
            weights = self._a_weighting(f)
            signal_power = np.sum(weights * np.mean(np.abs(speech_frames) ** 2, axis=1))
            if noise_frames.shape[1] > 0:
                noise_power = np.sum(weights * np.mean(np.abs(noise_frames) ** 2, axis=1))
            signal_power /= np.sum(weights)
            noise_power /= np.sum(weights)
        
        if noise_power < 1e-10:
            return 60.0
        
        snr = 10 * np.log10(signal_power / noise_power)
        return np.clip(snr, -20, 60)
    
    def _calculate_segmental_snr(self, audio: np.ndarray, sr: int) -> float:
        """Segmental SNR calculation."""
        # Segment parameters
        segment_length = int(0.2 * sr)  # 200ms segments
        overlap = 0.5
        hop = int(segment_length * (1 - overlap))
        
        segment_snrs = []
        
        for i in range(0, len(audio) - segment_length, hop):
            segment = audio[i:i + segment_length]
            
            try:
                # Calculate SNR for this segment
                seg_snr = self._calculate_waveform_snr(segment, sr)
                segment_snrs.append(seg_snr)
            except Exception:
                # Skip problematic segments
                continue
        
        if not segment_snrs:
            return self._calculate_waveform_snr(audio, sr)
        
        # Return mean of segment SNRs
        return np.mean(segment_snrs)
    
    def _calculate_vad_enhanced_snr(self, audio: np.ndarray, sr: int) -> float:
        """VAD-enhanced SNR calculation."""
        # Get VAD mask
        vad_mask = self.vad.detect(audio, sr)
        
        # Check if we have both speech and silence
        speech_ratio = np.sum(vad_mask) / len(vad_mask)
        
        if speech_ratio == 0:
            # No speech detected, fall back to energy-based
            self.logger.warning("No speech detected, using energy-based SNR")
            return self._estimate_snr_percentile(audio)
        
        if speech_ratio == 1:
            # All speech, estimate noise floor
            noise_power = self.noise_estimator.estimate_noise_floor(audio, sr)
        else:
            # Calculate from actual silence regions
            noise_samples = audio[~vad_mask]
            noise_power = np.mean(noise_samples ** 2)
        
        # Calculate signal power from speech regions
        speech_samples = audio[vad_mask]
        signal_power = np.mean(speech_samples ** 2)
        
        if noise_power < 1e-10:
            return 60.0
        
        snr = 10 * np.log10(signal_power / noise_power)
        return np.clip(snr, -20, 60)
    
    def _estimate_snr_percentile(self, audio: np.ndarray) -> float:
        """Estimate SNR using percentile method when VAD is not available."""
        # Sort squared samples
        squared = np.sort(audio ** 2)
        
        # Noise: bottom 10%
        noise_power = np.mean(squared[:len(squared)//10])
        
        # Signal: top 50%
        signal_power = np.mean(squared[len(squared)//2:])
        
        if noise_power < 1e-10:
            return 60.0
        
        snr = 10 * np.log10(signal_power / noise_power)
        return np.clip(snr, -20, 60)
    
    def _select_best_method(self, audio: np.ndarray, sr: int) -> str:
        """Automatically select the best SNR calculation method."""
        duration = len(audio) / sr
        
        # For short signals, use waveform
        if duration < 1.0:
            return 'waveform'
        
        # Check if VAD is available and working
        if self.config.vad_enabled:
            try:
                vad_mask = self.vad.detect(audio, sr)
                speech_ratio = np.sum(vad_mask) / len(vad_mask)
                
                if 0.1 < speech_ratio < 0.9:
                    # Good VAD results
                    return 'vad_enhanced'
            except Exception:
                pass
        
        # For longer signals, use spectral
        if duration > 3.0:
            return 'spectral'
        
        # Default to waveform
        return 'waveform'
    
    def _estimate_stationarity(self, audio: np.ndarray) -> float:
        """Estimate signal stationarity (0-1, higher is more stationary)."""
        # Divide into segments
        n_segments = 10
        segment_length = len(audio) // n_segments
        
        segment_powers = []
        for i in range(n_segments):
            segment = audio[i*segment_length:(i+1)*segment_length]
            segment_powers.append(np.mean(segment ** 2))
        
        # Calculate coefficient of variation
        cv = np.std(segment_powers) / (np.mean(segment_powers) + 1e-10)
        
        # Convert to 0-1 scale (lower CV = more stationary)
        stationarity = 1.0 / (1.0 + cv)
        
        return stationarity
    
    def _a_weighting(self, frequencies: np.ndarray) -> np.ndarray:
        """Calculate A-weighting coefficients for frequencies."""
        # A-weighting formula
        f2 = frequencies ** 2
        f4 = f2 ** 2
        
        num = 12194 ** 2 * f4
        den = ((f2 + 20.6 ** 2) * 
               np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * 
               (f2 + 12194 ** 2))
        
        # Avoid division by zero
        den = np.maximum(den, 1e-10)
        
        weights = num / den
        
        # Normalize
        weights = weights / np.max(weights)
        
        return weights
    
    def _get_cache_key(
        self, 
        audio: np.ndarray, 
        sr: int, 
        method: str,
        weighting: Optional[str]
    ) -> str:
        """Generate cache key for SNR calculation."""
        # Simple hash based on audio statistics
        audio_hash = hash((
            len(audio),
            sr,
            method,
            weighting,
            np.mean(audio),
            np.std(audio),
            np.min(audio),
            np.max(audio)
        ))
        return str(audio_hash)