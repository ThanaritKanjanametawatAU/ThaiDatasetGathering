"""Adaptive Wiener filtering for speech enhancement."""

import numpy as np
import torch
import torch.nn.functional as F


class AdaptiveWienerFilter:
    """Adaptive Wiener filter using decision-directed approach."""
    
    def __init__(self):
        self.alpha = 0.98  # Smoothing factor for a priori SNR
        self.beta = 0.8    # Overestimation factor
        self.frame_size = 2048
        self.hop_size = 512
        self.min_gain = 0.1  # Minimum gain to preserve naturalness
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply adaptive Wiener filtering.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Enhanced audio signal
        """
        if len(audio) < self.frame_size:
            return audio
        
        # Convert to torch
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.device)
        
        # STFT
        window = torch.hann_window(self.frame_size).to(self.device)
        stft = torch.stft(
            audio_tensor,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            window=window,
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        power = magnitude ** 2
        
        # Initial noise estimation (first 100ms)
        noise_frames = int(0.1 * sample_rate / self.hop_size)
        noise_frames = min(noise_frames, power.shape[1])
        noise_power = torch.mean(power[:, :noise_frames], dim=1, keepdim=True)
        
        # Initialize a priori SNR
        snr_priori = torch.ones_like(power)
        
        # Process each frame
        enhanced_magnitude = torch.zeros_like(magnitude)
        
        for i in range(power.shape[1]):
            # A posteriori SNR
            snr_post = power[:, i:i+1] / (noise_power + 1e-10)
            
            # Update a priori SNR using decision-directed approach
            if i > 0:
                gain_prev = enhanced_magnitude[:, i-1:i] / (magnitude[:, i-1:i] + 1e-10)
                snr_priori[:, i:i+1] = self.alpha * (gain_prev ** 2) * snr_post + \
                                       (1 - self.alpha) * torch.maximum(snr_post - 1, torch.zeros_like(snr_post))
            else:
                snr_priori[:, i:i+1] = torch.maximum(snr_post - 1, torch.zeros_like(snr_post))
            
            # Wiener gain
            gain = snr_priori[:, i:i+1] / (1 + snr_priori[:, i:i+1])
            
            # Apply minimum gain
            gain = torch.maximum(gain, torch.tensor(self.min_gain).to(self.device))
            
            # Apply gain
            enhanced_magnitude[:, i] = gain.squeeze() * magnitude[:, i]
            
            # Update noise estimate in low-energy frames
            if torch.mean(power[:, i]) < 1.5 * torch.mean(noise_power):
                noise_power = self.beta * noise_power + (1 - self.beta) * power[:, i:i+1]
        
        # Smooth enhanced magnitude
        enhanced_magnitude = self._smooth_magnitude(enhanced_magnitude)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        
        # ISTFT
        enhanced = torch.istft(
            enhanced_stft,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            window=window,
            length=len(audio)
        )
        
        return enhanced.cpu().numpy()
    
    def _smooth_magnitude(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Smooth magnitude spectrum to reduce artifacts."""
        # Time smoothing
        kernel_time = torch.tensor([0.25, 0.5, 0.25]).to(self.device)
        if magnitude.shape[1] > 3:
            # Apply 1D convolution along time axis
            # Reshape for proper convolution: [batch, channels, length]
            mag_time = magnitude.transpose(0, 1).unsqueeze(1)  # [time, 1, freq]
            smoothed_time = F.conv1d(
                mag_time,
                kernel_time.view(1, 1, -1),
                padding=1
            ).squeeze(1).transpose(0, 1)  # Back to [freq, time]
            magnitude = smoothed_time
        
        # Frequency smoothing
        kernel_freq = torch.tensor([0.25, 0.5, 0.25]).to(self.device)
        if magnitude.shape[0] > 3:
            # Apply 1D convolution along frequency axis
            mag_freq = magnitude.unsqueeze(0)  # [1, freq, time]
            smoothed_freq = F.conv1d(
                mag_freq,
                kernel_freq.view(1, 1, -1),
                padding=1,
                groups=1
            ).squeeze(0)
            magnitude = smoothed_freq
        
        return magnitude
    
    def estimate_clean_speech(self, noisy_spectrum: torch.Tensor, noise_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Estimate clean speech spectrum from noisy spectrum.
        
        Args:
            noisy_spectrum: Complex spectrogram of noisy speech
            noise_spectrum: Estimated noise power spectrum
            
        Returns:
            Estimated clean speech spectrum
        """
        noisy_power = torch.abs(noisy_spectrum) ** 2
        
        # A posteriori SNR
        snr_post = noisy_power / (noise_spectrum + 1e-10)
        
        # A priori SNR (simplified without history)
        snr_priori = torch.maximum(snr_post - 1, torch.zeros_like(snr_post))
        
        # Wiener gain
        gain = snr_priori / (1 + snr_priori)
        gain = torch.maximum(gain, torch.tensor(self.min_gain).to(self.device))
        
        # Apply gain
        clean_spectrum = gain * noisy_spectrum
        
        return clean_spectrum