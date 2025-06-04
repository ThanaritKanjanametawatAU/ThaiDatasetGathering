"""Harmonic enhancement for preserving speech quality."""

import numpy as np
import torch
import librosa
from scipy import signal


class HarmonicEnhancer:
    """Enhance harmonic components while suppressing inter-harmonic noise."""
    
    def __init__(self):
        self.frame_size = 2048
        self.hop_size = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhance harmonic structure of speech.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Harmonically enhanced audio
        """
        if len(audio) < self.frame_size:
            return audio
        
        # Pitch detection using librosa
        pitches, magnitudes = self._detect_pitch(audio, sample_rate)
        
        # STFT
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.device)
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
        
        # Process each frame
        enhanced_magnitude = torch.zeros_like(magnitude)
        
        for i in range(magnitude.shape[1]):
            if i < len(pitches) and magnitudes[i] > 0.1:  # Voiced frame
                pitch_hz = pitches[i]
                if pitch_hz > 0:
                    # Enhance harmonics
                    frame_enhanced = self._enhance_frame_harmonics(
                        magnitude[:, i],
                        pitch_hz,
                        sample_rate
                    )
                    enhanced_magnitude[:, i] = frame_enhanced
                else:
                    # Unvoiced - gentle processing
                    enhanced_magnitude[:, i] = magnitude[:, i]
            else:
                # No pitch detected - preserve original
                enhanced_magnitude[:, i] = magnitude[:, i]
        
        # Blend with original for naturalness
        blend_factor = 0.7
        enhanced_magnitude = blend_factor * enhanced_magnitude + (1 - blend_factor) * magnitude
        
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
    
    def _detect_pitch(self, audio: np.ndarray, sample_rate: int) -> tuple:
        """Detect pitch using librosa."""
        # Use librosa's pitch detection
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=sample_rate,
            fmin=50,
            fmax=400,
            threshold=0.1,
            hop_length=self.hop_size
        )
        
        # Extract the most likely pitch for each frame
        pitch_values = []
        magnitude_values = []
        
        for t in range(pitches.shape[1]):
            # Find the bin with maximum magnitude
            max_bin = np.argmax(magnitudes[:, t])
            pitch = pitches[max_bin, t]
            mag = magnitudes[max_bin, t]
            
            pitch_values.append(pitch)
            magnitude_values.append(mag)
        
        return np.array(pitch_values), np.array(magnitude_values)
    
    def _enhance_frame_harmonics(self, frame_spectrum: torch.Tensor, pitch_hz: float, sample_rate: int) -> torch.Tensor:
        """Enhance harmonics in a single frame."""
        enhanced = frame_spectrum.clone()
        
        # Frequency resolution
        freq_res = sample_rate / self.frame_size
        
        # Generate harmonic comb filter
        num_harmonics = 10
        harmonic_width = 2  # bins
        
        for h in range(1, num_harmonics + 1):
            harmonic_freq = h * pitch_hz
            harmonic_bin = int(harmonic_freq / freq_res)
            
            if 0 < harmonic_bin < len(frame_spectrum) - harmonic_width:
                # Find peak near expected harmonic
                search_start = max(0, harmonic_bin - 5)
                search_end = min(len(frame_spectrum), harmonic_bin + 5)
                
                local_peak = torch.argmax(frame_spectrum[search_start:search_end]) + search_start
                
                # Enhance harmonic region
                enhancement_factor = 1.5 / h  # Decreasing enhancement for higher harmonics
                
                for offset in range(-harmonic_width, harmonic_width + 1):
                    bin_idx = local_peak + offset
                    if 0 <= bin_idx < len(frame_spectrum):
                        weight = np.exp(-0.5 * (offset / harmonic_width) ** 2)  # Gaussian weight
                        enhanced[bin_idx] = frame_spectrum[bin_idx] * (1 + enhancement_factor * weight)
                
                # Suppress inter-harmonic regions
                if h < num_harmonics:
                    next_harmonic_bin = int((h + 1) * pitch_hz / freq_res)
                    inter_start = local_peak + harmonic_width + 1
                    inter_end = min(next_harmonic_bin - harmonic_width, len(frame_spectrum))
                    
                    if inter_start < inter_end:
                        suppression_factor = 0.7
                        enhanced[inter_start:inter_end] *= suppression_factor
        
        return enhanced