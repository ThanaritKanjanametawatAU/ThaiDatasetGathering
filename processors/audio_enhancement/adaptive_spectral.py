"""Adaptive spectral subtraction for gentle noise reduction."""

import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F


class AdaptiveSpectralSubtraction:
    """Adaptive spectral subtraction with musical noise reduction."""
    
    def __init__(self):
        self.frame_size = 2048
        self.hop_size = 512
        self.oversubtraction_factor = 1.5  # Start gentle
        self.window = signal.windows.hann(self.frame_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def process(self, audio: np.ndarray, sample_rate: int, target_snr: float = 35) -> np.ndarray:
        """
        Apply adaptive spectral subtraction.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            target_snr: Target SNR in dB
            
        Returns:
            Enhanced audio signal
        """
        if len(audio) < self.frame_size:
            return audio
        
        # Convert to torch for GPU processing if available
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.device)
        
        # STFT
        stft = torch.stft(
            audio_tensor,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            window=torch.from_numpy(self.window).to(self.device),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Noise estimation from first 100ms
        noise_frames = int(0.1 * sample_rate / self.hop_size)
        noise_frames = min(noise_frames, magnitude.shape[1])
        noise_spectrum = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)
        
        # Adaptive noise estimation
        noise_spectrum = self._update_noise_adaptively(magnitude, noise_spectrum)
        
        # Spectral subtraction with oversubtraction
        subtracted = magnitude - self.oversubtraction_factor * noise_spectrum
        
        # Wiener gain function to reduce musical noise
        gain = subtracted / (magnitude + 1e-10)
        gain = torch.clamp(gain, min=0.1, max=1.0)  # Minimum gain to preserve naturalness
        
        # Smooth gain over time
        gain = self._smooth_gain(gain)
        
        # Apply gain
        enhanced_magnitude = magnitude * gain
        
        # Preserve harmonics
        enhanced_magnitude = self._preserve_harmonics(enhanced_magnitude, magnitude)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        
        # ISTFT
        enhanced = torch.istft(
            enhanced_stft,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            window=torch.from_numpy(self.window).to(self.device),
            length=len(audio)
        )
        
        return enhanced.cpu().numpy()
    
    def _update_noise_adaptively(self, magnitude: torch.Tensor, initial_noise: torch.Tensor) -> torch.Tensor:
        """Adaptively update noise estimate."""
        alpha = 0.95  # Smoothing factor
        noise_estimate = initial_noise.clone()
        
        for i in range(1, magnitude.shape[1]):
            # Update noise in low-energy frames
            frame_energy = torch.mean(magnitude[:, i])
            noise_energy = torch.mean(noise_estimate)
            
            if frame_energy < 1.5 * noise_energy:
                # This is likely a noise frame
                noise_estimate = alpha * noise_estimate + (1 - alpha) * magnitude[:, i:i+1]
        
        return noise_estimate
    
    def _smooth_gain(self, gain: torch.Tensor) -> torch.Tensor:
        """Smooth gain function over time and frequency."""
        # Time smoothing
        kernel_size = 5
        if gain.shape[1] > kernel_size:
            gain = F.avg_pool1d(
                gain.unsqueeze(0).transpose(1, 2),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ).transpose(1, 2).squeeze(0)
        
        # Frequency smoothing
        if gain.shape[0] > kernel_size:
            gain = F.avg_pool1d(
                gain.unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ).squeeze(0)
        
        return gain
    
    def _preserve_harmonics(self, enhanced: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Preserve harmonic structure in enhanced signal."""
        # Find spectral peaks (potential harmonics)
        for i in range(enhanced.shape[1]):
            spectrum = original[:, i].cpu().numpy()
            peaks, _ = signal.find_peaks(spectrum, height=np.mean(spectrum) * 2)
            
            if len(peaks) > 0:
                # Boost harmonic regions slightly
                peak_tensor = torch.from_numpy(peaks).to(self.device)
                for peak in peak_tensor:
                    if peak > 2 and peak < enhanced.shape[0] - 2:
                        # Preserve peak and immediate neighbors
                        enhanced[peak-2:peak+3, i] = 0.8 * enhanced[peak-2:peak+3, i] + 0.2 * original[peak-2:peak+3, i]
        
        return enhanced