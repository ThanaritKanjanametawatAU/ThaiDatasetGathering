"""GPU-optimized batch processing for audio enhancement."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
import logging


logger = logging.getLogger(__name__)


class GPUEnhancementBatch:
    """GPU-optimized batch enhancement processing."""
    
    def __init__(self, batch_size: int = 32, device: str = 'cuda'):
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.frame_size = 2048
        self.hop_size = 512
        
        if self.device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            logger.warning("CUDA not available, using CPU")
    
    def process_batch(self, audio_batch: List[np.ndarray], sample_rate: int = 16000) -> List[np.ndarray]:
        """
        Process a batch of audio samples on GPU.
        
        Args:
            audio_batch: List of audio numpy arrays
            sample_rate: Sample rate in Hz
            
        Returns:
            List of enhanced audio arrays
        """
        if not audio_batch:
            return []
        
        # Step 1: Prepare batch tensors
        padded_batch, original_lengths = self._prepare_batch(audio_batch)
        
        # Step 2: Batch STFT
        batch_stft = self._batch_stft(padded_batch)
        
        # Step 3: Batch enhancement
        enhanced_stft = self._batch_enhance(batch_stft)
        
        # Step 4: Batch synthesis
        enhanced_batch = self._batch_istft(enhanced_stft, original_lengths)
        
        # Convert back to numpy
        enhanced_list = []
        for i, length in enumerate(original_lengths):
            enhanced = enhanced_batch[i, :length].cpu().numpy()
            enhanced_list.append(enhanced)
        
        return enhanced_list
    
    def _prepare_batch(self, audio_batch: List[np.ndarray]) -> Tuple[torch.Tensor, List[int]]:
        """Prepare batch for GPU processing."""
        # Record original lengths
        original_lengths = [len(audio) for audio in audio_batch]
        
        # Find max length
        max_length = max(original_lengths)
        
        # Pad to same length
        padded_batch = []
        for audio in audio_batch:
            if len(audio) < max_length:
                padded = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            else:
                padded = audio
            padded_batch.append(padded)
        
        # Convert to torch tensor
        batch_tensor = torch.stack([
            torch.from_numpy(audio.astype(np.float32)) for audio in padded_batch
        ]).to(self.device)
        
        return batch_tensor, original_lengths
    
    def _batch_stft(self, batch: torch.Tensor) -> torch.Tensor:
        """Batch STFT processing."""
        window = torch.hann_window(self.frame_size).to(self.device)
        
        # Process each item in batch
        stft_list = []
        for i in range(batch.shape[0]):
            stft = torch.stft(
                batch[i],
                n_fft=self.frame_size,
                hop_length=self.hop_size,
                window=window,
                return_complex=True
            )
            stft_list.append(stft)
        
        # Stack into batch
        batch_stft = torch.stack(stft_list)
        return batch_stft
    
    def _batch_enhance(self, batch_stft: torch.Tensor) -> torch.Tensor:
        """Apply enhancement in batch."""
        # Extract magnitude and phase
        magnitude = torch.abs(batch_stft)
        phase = torch.angle(batch_stft)
        
        # Batch noise estimation (first 100ms)
        noise_frames = min(10, magnitude.shape[2])
        noise_spectrum = torch.mean(magnitude[:, :, :noise_frames], dim=2, keepdim=True)
        
        # Batch spectral subtraction
        enhanced_magnitude = self._batch_spectral_subtraction(magnitude, noise_spectrum)
        
        # Batch Wiener filtering
        enhanced_magnitude = self._batch_wiener_filter(enhanced_magnitude, noise_spectrum)
        
        # Reconstruct complex STFT
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        
        return enhanced_stft
    
    def _batch_spectral_subtraction(self, magnitude: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Vectorized spectral subtraction."""
        # Oversubtraction factor
        alpha = 1.5
        
        # Subtract noise
        subtracted = magnitude - alpha * noise
        
        # Ensure non-negative
        subtracted = torch.maximum(subtracted, 0.1 * magnitude)
        
        # Smooth over time
        kernel = torch.ones(1, 1, 5).to(self.device) / 5
        if subtracted.shape[2] > 5:
            subtracted = F.conv1d(
                subtracted.transpose(1, 2),
                kernel,
                padding=2
            ).transpose(1, 2)
        
        return subtracted
    
    def _batch_wiener_filter(self, magnitude: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Vectorized Wiener filtering."""
        # Estimate SNR
        signal_power = magnitude ** 2
        noise_power = noise ** 2
        
        # A priori SNR estimation
        snr_priori = torch.maximum(signal_power / (noise_power + 1e-10) - 1, torch.zeros_like(signal_power))
        
        # Wiener gain
        gain = snr_priori / (1 + snr_priori)
        gain = torch.maximum(gain, torch.tensor(0.1).to(self.device))
        
        # Apply gain
        enhanced = magnitude * gain
        
        return enhanced
    
    def _batch_istft(self, batch_stft: torch.Tensor, original_lengths: List[int]) -> torch.Tensor:
        """Batch ISTFT processing."""
        window = torch.hann_window(self.frame_size).to(self.device)
        max_length = max(original_lengths)
        
        # Process each item in batch
        audio_list = []
        for i in range(batch_stft.shape[0]):
            audio = torch.istft(
                batch_stft[i],
                n_fft=self.frame_size,
                hop_length=self.hop_size,
                window=window,
                length=max_length
            )
            audio_list.append(audio)
        
        # Stack into batch
        batch_audio = torch.stack(audio_list)
        return batch_audio


class MemoryEfficientProcessor:
    """Memory-efficient processing with dynamic batch sizing."""
    
    def __init__(self, max_memory_gb: float = 30):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Base memory requirements
        self.bytes_per_sample = 4  # float32
        self.stft_expansion_factor = 4  # Approximate STFT memory expansion
        self.overhead_factor = 1.5  # Safety margin
    
    def estimate_batch_size(self, audio_length: int, sample_rate: int) -> int:
        """
        Estimate optimal batch size for given audio length.
        
        Args:
            audio_length: Number of samples in audio
            sample_rate: Sample rate in Hz
            
        Returns:
            Recommended batch size
        """
        # Calculate memory per sample
        memory_per_audio = (
            audio_length * self.bytes_per_sample * 
            self.stft_expansion_factor * 
            self.overhead_factor
        )
        
        # Account for GPU memory already in use
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            available = self.max_memory_bytes - max(allocated, reserved)
        else:
            available = self.max_memory_bytes
        
        # Calculate batch size
        batch_size = int(available / memory_per_audio)
        
        # Apply constraints
        batch_size = max(1, batch_size)  # At least 1
        batch_size = min(128, batch_size)  # Max 128
        
        # Adjust based on audio duration
        duration_seconds = audio_length / sample_rate
        if duration_seconds < 1.0:
            batch_size = min(64, batch_size)
        elif duration_seconds > 10.0:
            batch_size = min(16, batch_size)
        elif duration_seconds > 30.0:
            batch_size = min(4, batch_size)
        
        logger.info(f"Estimated batch size: {batch_size} for {duration_seconds:.1f}s audio")
        
        return batch_size
    
    def process_with_memory_management(self, audio_list: List[np.ndarray], 
                                      sample_rate: int,
                                      processor: GPUEnhancementBatch) -> List[np.ndarray]:
        """
        Process audio list with dynamic batch sizing.
        
        Args:
            audio_list: List of audio arrays
            sample_rate: Sample rate
            processor: GPU batch processor
            
        Returns:
            List of processed audio
        """
        if not audio_list:
            return []
        
        # Group by similar lengths
        length_groups = self._group_by_length(audio_list)
        
        all_processed = []
        
        for length_range, group_audios in length_groups.items():
            # Estimate batch size for this group
            avg_length = int(np.mean([len(a) for a in group_audios]))
            batch_size = self.estimate_batch_size(avg_length, sample_rate)
            
            # Process in batches
            for i in range(0, len(group_audios), batch_size):
                batch = group_audios[i:i + batch_size]
                
                try:
                    processed = processor.process_batch(batch, sample_rate)
                    all_processed.extend(processed)
                    
                    # Clear cache periodically
                    if self.device.type == 'cuda' and i % (batch_size * 4) == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM with batch size {batch_size}, reducing...")
                        torch.cuda.empty_cache()
                        
                        # Process one by one
                        for audio in batch:
                            try:
                                processed = processor.process_batch([audio], sample_rate)
                                all_processed.extend(processed)
                            except Exception as e2:
                                logger.error(f"Failed to process audio: {e2}")
                                all_processed.append(audio)  # Return original
                    else:
                        raise
        
        return all_processed
    
    def _group_by_length(self, audio_list: List[np.ndarray]) -> dict:
        """Group audio by similar lengths for efficient batching."""
        groups = {
            'short': [],     # < 1 second
            'medium': [],    # 1-5 seconds
            'long': [],      # 5-10 seconds
            'very_long': []  # > 10 seconds
        }
        
        for audio in audio_list:
            duration = len(audio) / 16000  # Assume 16kHz
            
            if duration < 1.0:
                groups['short'].append(audio)
            elif duration < 5.0:
                groups['medium'].append(audio)
            elif duration < 10.0:
                groups['long'].append(audio)
            else:
                groups['very_long'].append(audio)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}