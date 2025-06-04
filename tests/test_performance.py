"""Test performance requirements for 35dB enhancement."""

import unittest
import numpy as np
import time
from unittest.mock import patch

from processors.audio_enhancement.core import AudioEnhancer
from processors.audio_enhancement.gpu_enhancement import GPUEnhancementBatch, MemoryEfficientProcessor


class TestPerformance(unittest.TestCase):
    """Test performance requirements."""
    
    def setUp(self):
        self.sample_rate = 16000
    
    def test_processing_speed_requirement(self):
        """Test 6.1: Process 100 samples in under 3 minutes."""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        # Create 100 test samples
        samples = []
        for i in range(100):
            # Varying durations 1-5 seconds
            duration = 1 + np.random.rand() * 4
            signal_clean = self._create_realistic_speech(duration=duration)
            noisy = self._add_noise_at_snr(signal_clean, 20 + np.random.rand() * 10)
            samples.append((noisy, self.sample_rate))
        
        # Time processing
        start_time = time.time()
        
        for audio, sr in samples:
            enhanced, metadata = enhancer.enhance_to_target_snr(audio, sr)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Must be under 180 seconds (3 minutes)
        self.assertLess(total_time, 180, 
                       f"Processing took {total_time:.1f}s, exceeds 3 minute limit")
        
        # Calculate per-sample time
        per_sample = total_time / 100
        self.assertLess(per_sample, 1.8, 
                       f"Per-sample time {per_sample:.2f}s exceeds 1.8s limit")
    
    def test_batch_processing_efficiency(self):
        """Test 6.2: Batch processing is more efficient than sequential."""
        gpu_processor = GPUEnhancementBatch(batch_size=8, device='cpu')  # Use CPU for testing
        
        # Create test batch
        batch = []
        for i in range(16):
            signal_clean = self._create_realistic_speech(duration=2.0)
            noisy = self._add_noise_at_snr(signal_clean, 25)
            batch.append(noisy)
        
        # Time batch processing
        start_batch = time.time()
        enhanced_batch = gpu_processor.process_batch(batch, self.sample_rate)
        batch_time = time.time() - start_batch
        
        # Time sequential processing (first 4 samples)
        start_seq = time.time()
        enhanced_seq = []
        for audio in batch[:4]:
            result = gpu_processor.process_batch([audio], self.sample_rate)
            enhanced_seq.extend(result)
        seq_time = time.time() - start_seq
        
        # Batch should be more efficient (normalize by number of samples)
        batch_per_sample = batch_time / len(batch)
        seq_per_sample = seq_time / 4
        
        # Batch should be at least 20% faster per sample
        self.assertLess(batch_per_sample, seq_per_sample * 0.8,
                       "Batch processing not efficient enough")
    
    def test_memory_usage_estimation(self):
        """Test 6.3: Memory usage estimation is accurate."""
        memory_manager = MemoryEfficientProcessor(max_memory_gb=30)
        
        # Test various audio lengths
        test_cases = [
            {"duration": 1.0, "expected_range": (32, 128)},
            {"duration": 5.0, "expected_range": (16, 64)},
            {"duration": 10.0, "expected_range": (8, 32)},
            {"duration": 30.0, "expected_range": (2, 8)},
        ]
        
        for case in test_cases:
            audio_length = int(case["duration"] * self.sample_rate)
            batch_size = memory_manager.estimate_batch_size(audio_length, self.sample_rate)
            
            min_expected, max_expected = case["expected_range"]
            self.assertGreaterEqual(batch_size, min_expected,
                                  f"Batch size too small for {case['duration']}s audio")
            self.assertLessEqual(batch_size, max_expected,
                               f"Batch size too large for {case['duration']}s audio")
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.max_memory_allocated')
    @patch('torch.cuda.reset_peak_memory_stats')
    def test_gpu_memory_management(self, mock_reset, mock_max_mem, mock_allocated, mock_cuda):
        """Test 6.4: GPU memory is properly managed."""
        # Mock CUDA availability
        mock_cuda.return_value = True
        mock_allocated.return_value = 0
        mock_max_mem.return_value = 10 * 1024**3  # 10GB used
        
        gpu_processor = GPUEnhancementBatch(batch_size=32, device='cuda')
        
        # Create large batch
        batch = []
        for i in range(32):
            # 5 second audio
            audio = np.random.randn(5 * self.sample_rate).astype(np.float32)
            batch.append(audio)
        
        # Process (mocked, so won't actually use GPU)
        with patch.object(gpu_processor, '_batch_enhance') as mock_enhance:
            mock_enhance.return_value = np.zeros((32, 1025, 100), dtype=np.complex64)
            
            # This should work without memory errors
            try:
                enhanced_batch = gpu_processor.process_batch(batch)
                success = True
            except Exception as e:
                success = False
                print(f"GPU processing failed: {e}")
        
        self.assertTrue(success, "GPU memory management failed")
    
    def test_processing_timeout(self):
        """Test 6.5: Processing respects timeout limits."""
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        # Create a very long audio (60 seconds)
        long_audio = self._create_realistic_speech(duration=60.0)
        noisy_long = self._add_noise_at_snr(long_audio, 20)
        
        # Processing should complete within reasonable time
        start_time = time.time()
        enhanced, metadata = enhancer.enhance_to_target_snr(noisy_long, self.sample_rate)
        process_time = time.time() - start_time
        
        # Should process long audio in reasonable time (< 100s for 60s audio)
        self.assertLess(process_time, 100,
                       f"Processing 60s audio took {process_time:.1f}s")
    
    def test_concurrent_processing(self):
        """Test 6.6: Concurrent processing works correctly."""
        from concurrent.futures import ThreadPoolExecutor
        
        enhancer = AudioEnhancer(enable_35db_enhancement=True)
        
        # Create test samples
        samples = []
        for i in range(10):
            signal_clean = self._create_realistic_speech(duration=2.0)
            noisy = self._add_noise_at_snr(signal_clean, 25)
            samples.append(noisy)
        
        # Process concurrently
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for audio in samples:
                future = executor.submit(enhancer.enhance_to_target_snr, audio, self.sample_rate)
                futures.append(future)
            
            for future in futures:
                enhanced, metadata = future.result()
                results.append((enhanced, metadata))
        
        # All should complete successfully
        self.assertEqual(len(results), 10)
        
        # Check all were enhanced
        for enhanced, metadata in results:
            self.assertIsNotNone(enhanced)
            self.assertIn("snr_db", metadata)
    
    def _create_realistic_speech(self, duration: float = 3.0) -> np.ndarray:
        """Create speech-like signal."""
        sr = self.sample_rate
        t = np.linspace(0, duration, int(duration * sr))
        
        # Base frequency
        f0 = 150 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
        
        # Harmonics
        signal = np.zeros_like(t)
        for i in range(1, 10):
            signal += np.sin(2 * np.pi * f0 * i * t) / i
        
        # Envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        signal *= envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return signal
    
    def _add_noise_at_snr(self, signal: np.ndarray, target_snr_db: float) -> np.ndarray:
        """Add noise to achieve specific SNR."""
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (target_snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise


if __name__ == "__main__":
    unittest.main()