"""SNR (Signal-to-Noise Ratio) measurement utilities for audio quality assessment."""

import numpy as np
from scipy import signal
from pyannote.audio import Inference
from pyannote.audio.pipelines import VoiceActivityDetection
import torch


class SNRMeasurement:
    """Measure Signal-to-Noise Ratio with Voice Activity Detection."""
    
    def __init__(self):
        self.vad_model = None
        self._init_vad()
    
    def _init_vad(self):
        """Initialize Voice Activity Detection model."""
        try:
            # Use pyannote VAD
            self.vad_model = VoiceActivityDetection(
                segmentation="pyannote/segmentation-3.0",
                use_auth_token=False
            )
            # Set hyperparameters
            self.vad_model.instantiate({
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_on": 0.1,
                "min_duration_off": 0.1
            })
        except Exception:
            # Fallback to energy-based VAD if pyannote fails
            self.vad_model = None
    
    def measure_snr(self, audio, sample_rate):
        """
        Measure SNR of audio signal.
        
        Args:
            audio: numpy array of audio samples
            sample_rate: sampling rate in Hz
            
        Returns:
            float: SNR in dB
        """
        if len(audio) == 0:
            return float('-inf')
        
        # Normalize audio
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Get VAD segments
        vad_segments = self._get_vad_segments(audio, sample_rate)
        
        if not vad_segments:
            # All silence or all speech - use percentile method
            return self._estimate_snr_percentile(audio)
        
        # Calculate noise floor from silence segments
        noise_power = self._estimate_noise_floor(audio, vad_segments, sample_rate)
        
        # Calculate signal power from speech segments
        signal_power = self._estimate_signal_power(audio, vad_segments, sample_rate)
        
        # Calculate SNR
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 60.0  # Very clean signal
        
        return np.clip(snr_db, -20, 60)
    
    def _get_vad_segments(self, audio, sample_rate):
        """Get voice activity detection segments."""
        if self.vad_model is None:
            return self._energy_based_vad(audio, sample_rate)
        
        try:
            # Use pyannote VAD
            waveform = torch.tensor(audio).unsqueeze(0)
            vad_output = self.vad_model({"waveform": waveform, "sample_rate": sample_rate})
            
            segments = []
            for segment in vad_output:
                segments.append((segment.start, segment.end))
            return segments
        except Exception:
            # Fallback to energy-based VAD
            return self._energy_based_vad(audio, sample_rate)
    
    def _energy_based_vad(self, audio, sample_rate):
        """Simple energy-based VAD fallback."""
        # Frame-based energy calculation
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Calculate frame energies
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sum(frame ** 2) / frame_length
            energies.append(energy)
        
        if not energies:
            return []
        
        energies = np.array(energies)
        
        # Dynamic threshold
        energy_threshold = np.percentile(energies, 30)
        
        # Find speech segments
        speech_frames = energies > energy_threshold
        
        # Convert frame indices to time segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                start_time = start_frame * hop_length / sample_rate
                end_time = i * hop_length / sample_rate
                if end_time - start_time > 0.1:  # Min 100ms
                    segments.append((start_time, end_time))
                in_speech = False
        
        # Handle last segment
        if in_speech:
            start_time = start_frame * hop_length / sample_rate
            end_time = len(audio) / sample_rate
            segments.append((start_time, end_time))
        
        return segments
    
    def _estimate_noise_floor(self, audio, vad_segments, sample_rate):
        """Estimate noise floor from non-speech segments."""
        # Get silence segments
        silence_samples = []
        last_end = 0
        
        for start, end in vad_segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Add silence before this speech segment
            if start_sample > last_end:
                silence_samples.extend(audio[last_end:start_sample])
            
            last_end = end_sample
        
        # Add final silence
        if last_end < len(audio):
            silence_samples.extend(audio[last_end:])
        
        if not silence_samples:
            # No silence found - use quietest parts
            return self._estimate_noise_percentile(audio)
        
        silence_samples = np.array(silence_samples)
        
        # Use 5th percentile to avoid outliers
        noise_rms = np.sqrt(np.percentile(silence_samples ** 2, 5))
        return noise_rms ** 2
    
    def _estimate_signal_power(self, audio, vad_segments, sample_rate):
        """Estimate signal power from speech segments."""
        speech_samples = []
        
        for start, end in vad_segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            speech_samples.extend(audio[start_sample:end_sample])
        
        if not speech_samples:
            # All silence - use full signal
            speech_samples = audio
        
        speech_samples = np.array(speech_samples)
        
        # Use RMS for power estimation
        signal_rms = np.sqrt(np.mean(speech_samples ** 2))
        return signal_rms ** 2
    
    def _estimate_snr_percentile(self, audio):
        """Estimate SNR using percentile method when VAD is not available."""
        # Sort squared samples
        squared = np.sort(audio ** 2)
        
        # Noise: bottom 10%
        noise_power = np.mean(squared[:len(squared)//10])
        
        # Signal: top 50%
        signal_power = np.mean(squared[len(squared)//2:])
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 60.0
        
        return np.clip(snr_db, -20, 60)
    
    def _estimate_noise_percentile(self, audio):
        """Estimate noise using percentile when no silence detected."""
        # Use 5th percentile of squared samples
        return np.percentile(audio ** 2, 5)