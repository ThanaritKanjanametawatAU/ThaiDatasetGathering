"""
Core audio enhancement pipeline with smart adaptive processing.
Orchestrates multiple enhancement engines based on audio characteristics.
"""

import numpy as np
import time
import logging
from typing import Optional, Dict, Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .engines.denoiser import DenoiserEngine
from .engines.spectral_gating import SpectralGatingEngine
from .speaker_separation import SpeakerSeparator, SeparationConfig
from utils.audio_metrics import (
    calculate_snr, calculate_pesq, calculate_stoi,
    calculate_spectral_distortion, calculate_speaker_similarity,
    calculate_all_metrics
)

logger = logging.getLogger(__name__)


class AudioEnhancer:
    """
    Main audio enhancement class with smart adaptive processing.
    Supports progressive enhancement with quality targets.
    """
    
    # Enhancement level configurations
    ENHANCEMENT_LEVELS = {
        'clean': {
            'skip': True,
            'denoiser_ratio': 0.0,
            'spectral_ratio': 0.0
        },
        'mild': {
            'skip': False,
            'denoiser_ratio': 0.05,  # Lower = more denoising
            'spectral_ratio': 0.3,
            'passes': 1
        },
        'moderate': {
            'skip': False,
            'denoiser_ratio': 0.02,  # Lower = more denoising
            'spectral_ratio': 0.5,
            'passes': 2
        },
        'aggressive': {
            'skip': False,
            'denoiser_ratio': 0.005,  # Lower = more denoising (was 0.01)
            'spectral_ratio': 0.8,    # Higher = more spectral gating (was 0.7)
            'passes': 4               # More passes for better cleaning (was 3)
        },
        'ultra_aggressive': {
            'skip': False,
            'denoiser_ratio': 0.0,    # Full wet signal (maximum denoising)
            'spectral_ratio': 0.9,    # Very high spectral gating
            'passes': 5,              # Maximum passes
            'preserve_ratio': 0.5     # Mix 50% original to prevent over-processing
        },
        'secondary_speaker': {
            'skip': False,
            'use_speaker_separation': True,
            'suppression_strength': 0.6,
            'passes': 1
        }
    }
    
    def __init__(
        self,
        use_gpu: bool = True,
        fallback_to_cpu: bool = True,
        clean_threshold_snr: float = 30.0,
        target_pesq: float = 3.0,
        target_stoi: float = 0.85
    ):
        """
        Initialize audio enhancer.
        
        Args:
            use_gpu: Whether to try using GPU acceleration
            fallback_to_cpu: Whether to fallback to CPU if GPU fails
            clean_threshold_snr: SNR threshold above which audio is considered clean
            target_pesq: Target PESQ score for enhancement
            target_stoi: Target STOI score for enhancement
        """
        self.use_gpu = use_gpu
        self.fallback_to_cpu = fallback_to_cpu
        self.clean_threshold_snr = clean_threshold_snr
        self.target_pesq = target_pesq
        self.target_stoi = target_stoi
        
        # Initialize engines
        self._init_engines()
        
        # Initialize speaker separator
        separation_config = SeparationConfig(
            suppression_strength=0.6,
            confidence_threshold=0.5,
            preserve_main_speaker=True,
            use_sepformer=False  # Start with False to avoid dependency issues
        )
        self.speaker_separator = SpeakerSeparator(separation_config)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'skipped_clean': 0,
            'enhanced': 0,
            'total_snr_improvement': 0.0,
            'total_processing_time': 0.0,
            'secondary_speakers_detected': 0,
            'secondary_speakers_removed': 0
        }
    
    def _init_engines(self):
        """Initialize enhancement engines with fallback logic."""
        # Try to initialize Denoiser (GPU preference)
        try:
            device = 'cuda' if self.use_gpu else 'cpu'
            self.denoiser = DenoiserEngine(device=device)
            self.primary_engine = 'denoiser'
            logger.info(f"Initialized Denoiser engine on {device}")
        except Exception as e:
            logger.warning(f"Failed to initialize Denoiser: {e}")
            self.denoiser = None
            self.primary_engine = None
        
        # Always initialize spectral gating as fallback
        try:
            self.spectral_gating = SpectralGatingEngine()
            if self.primary_engine is None:
                self.primary_engine = 'spectral_gating'
            logger.info("Initialized Spectral Gating engine")
        except Exception as e:
            logger.warning(f"Failed to initialize Spectral Gating: {e}")
            self.spectral_gating = None
        
        if self.primary_engine is None:
            raise RuntimeError("No enhancement engines available!")
    
    def assess_noise_level(
        self,
        audio: np.ndarray,
        sample_rate: int,
        quick: bool = True
    ) -> str:
        """
        Quickly assess noise level in audio, including secondary speaker detection.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            quick: Whether to use quick assessment (< 0.1s)
            
        Returns:
            Noise level category: 'clean', 'mild', 'moderate', 'aggressive', 'secondary_speaker'
        """
        start_time = time.time()
        
        try:
            # Quick secondary speaker detection
            if quick and hasattr(self, 'speaker_separator'):
                # Do a quick detection check
                separation_result = self.speaker_separator.separate_speakers(
                    audio[:min(len(audio), sample_rate * 2)],  # First 2 seconds
                    sample_rate
                )
                if separation_result['detections']:
                    # If we detect secondary speakers with high confidence
                    max_confidence = max([d.confidence for d in separation_result['detections']])
                    if max_confidence > 0.6:
                        logger.debug(f"Secondary speaker detected with confidence {max_confidence:.2f}")
                        return 'secondary_speaker'
            
            # Original noise assessment
            # Quick SNR estimation using first few seconds
            if quick and len(audio) > sample_rate * 2:
                # Use only first 2 seconds for quick assessment
                audio_sample = audio[:sample_rate * 2]
            else:
                audio_sample = audio
            
            # Estimate noise using high-frequency content
            from scipy import signal
            
            # High-pass filter to isolate noise
            nyquist = sample_rate / 2
            high_cutoff = min(4000, nyquist * 0.8) / nyquist
            b, a = signal.butter(5, high_cutoff, 'high')
            noise_estimate = signal.filtfilt(b, a, audio_sample)
            
            # Calculate pseudo-SNR
            signal_power = np.mean(audio_sample ** 2)
            noise_power = np.mean(noise_estimate ** 2)
            
            # Also check overall variance as indicator of noise
            variance = np.var(audio_sample)
            
            if noise_power > 0:
                snr_estimate = 10 * np.log10(signal_power / noise_power)
            else:
                snr_estimate = float('inf')
                
            # If variance is high, assume some noise present
            if variance > 0.05 and snr_estimate > 30:  # More sensitive (was 0.1)
                snr_estimate = 20  # Lower cap to ensure more processing (was 25)
            
            # Categorize noise level (more sensitive)
            if snr_estimate > 40:  # Much higher threshold for "clean" (was 30)
                level = 'clean'
            elif snr_estimate > 20:  # Higher threshold (was 15)
                level = 'mild'
            elif snr_estimate > 10:  # Higher threshold (was 5)
                level = 'moderate'
            elif snr_estimate > 0:
                level = 'aggressive'
            else:
                level = 'ultra_aggressive'  # Use ultra for very noisy audio
            
            assessment_time = time.time() - start_time
            logger.debug(f"Noise assessment took {assessment_time:.3f}s, level: {level}, SNR: {snr_estimate:.1f}dB")
            
            return level
            
        except Exception as e:
            logger.error(f"Noise assessment failed: {e}")
            return 'moderate'  # Default to moderate if assessment fails
    
    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: int,
        noise_level: Optional[str] = None,
        target_pesq: Optional[float] = None,
        target_stoi: Optional[float] = None,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Enhance audio with smart adaptive processing.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            noise_level: Override noise level assessment
            target_pesq: Override target PESQ score
            target_stoi: Override target STOI score
            return_metadata: Whether to return enhancement metadata
            
        Returns:
            Enhanced audio signal, optionally with metadata
        """
        start_time = time.time()
        self.stats['total_processed'] += 1
        
        # Use provided targets or defaults
        target_pesq = target_pesq or self.target_pesq
        target_stoi = target_stoi or self.target_stoi
        
        # Assess noise level if not provided
        if noise_level is None:
            noise_level = self.assess_noise_level(audio, sample_rate)
        
        # Get enhancement configuration
        config = self.ENHANCEMENT_LEVELS.get(noise_level, self.ENHANCEMENT_LEVELS['moderate'])
        
        # Skip if clean
        if config.get('skip', False):
            self.stats['skipped_clean'] += 1
            if return_metadata:
                metadata = {
                    'enhanced': False,
                    'noise_level': noise_level,
                    'enhancement_level': 'none',
                    'snr_before': self.clean_threshold_snr,
                    'snr_after': self.clean_threshold_snr,
                    'processing_time': time.time() - start_time,
                    'engine_used': 'none',
                    'secondary_speaker_detected': False
                }
                return audio, metadata
            return audio
        
        # Check if we should use speaker separation
        if config.get('use_speaker_separation', False) or noise_level == 'secondary_speaker':
            # Use speaker separator
            separation_result = self.speaker_separator.separate_speakers(audio, sample_rate)
            enhanced = separation_result['audio']
            detections = separation_result['detections']
            
            # Update statistics
            if detections:
                self.stats['secondary_speakers_detected'] += 1
                self.stats['secondary_speakers_removed'] += len(detections)
            
            if return_metadata:
                processing_time = time.time() - start_time
                metadata = {
                    'enhanced': True,
                    'noise_level': noise_level,
                    'enhancement_level': 'secondary_speaker_removal',
                    'snr_before': initial_metrics.get('snr', 0) if 'initial_metrics' in locals() else 0,
                    'snr_after': self._measure_quality_quick(enhanced, audio, sample_rate).get('snr', 0),
                    'processing_time': processing_time,
                    'engine_used': 'speaker_separator',
                    'secondary_speaker_detected': len(detections) > 0,
                    'secondary_speaker_confidence': max([d.confidence for d in detections]) if detections else 0.0,
                    'num_secondary_speakers': len(detections),
                    'speaker_similarity': separation_result['metrics'].get('similarity_preservation', 1.0)
                }
                return enhanced, metadata
            return enhanced
        
        # Progressive enhancement (original code)
        enhanced = audio.copy()
        original_audio = audio.copy()
        passes_applied = 0
        engine_used = self.primary_engine
        
        # Calculate initial metrics
        initial_metrics = self._measure_quality_quick(enhanced, original_audio, sample_rate)
        
        for pass_num in range(config.get('passes', 1)):
            # Check if targets are already met
            if pass_num > 0:
                current_metrics = self._measure_quality_quick(enhanced, original_audio, sample_rate)
                if (current_metrics.get('pesq', 0) >= target_pesq and 
                    current_metrics.get('stoi', 0) >= target_stoi):
                    break
            
            # Apply enhancement
            if self.denoiser and self.denoiser.model is not None:
                # Use Denoiser (GPU)
                enhanced = self.denoiser.process(
                    enhanced, sample_rate,
                    denoiser_dry=config['denoiser_ratio']
                )
                engine_used = 'denoiser'
            elif self.spectral_gating and self.spectral_gating.available:
                # Fallback to spectral gating (CPU)
                enhanced = self.spectral_gating.process(
                    enhanced, sample_rate
                )
                engine_used = 'spectral_gating'
            else:
                logger.warning("No enhancement engines available")
                break
            
            passes_applied += 1
        
        # Calculate final metrics
        final_metrics = self._measure_quality_quick(enhanced, original_audio, sample_rate)
        
        # Update statistics
        self.stats['enhanced'] += 1
        snr_improvement = final_metrics.get('snr', 0) - initial_metrics.get('snr', 0)
        self.stats['total_snr_improvement'] += snr_improvement
        
        processing_time = time.time() - start_time
        self.stats['total_processing_time'] += processing_time
        
        if return_metadata:
            metadata = {
                'enhanced': True,
                'noise_level': noise_level,
                'enhancement_level': f"{noise_level}_{passes_applied}pass",
                'snr_before': initial_metrics.get('snr', 0),
                'snr_after': final_metrics.get('snr', 0),
                'snr_improvement': snr_improvement,
                'processing_time': processing_time,
                'engine_used': engine_used,
                'passes_applied': passes_applied,
                'pesq': final_metrics.get('pesq', 0),
                'stoi': final_metrics.get('stoi', 0)
            }
            return enhanced, metadata
        
        return enhanced
    
    def _measure_quality(
        self,
        audio: np.ndarray,
        reference: np.ndarray,
        sample_rate: int
    ) -> Dict:
        """Measure audio quality metrics."""
        return calculate_all_metrics(reference, audio, sample_rate)
    
    def _measure_quality_quick(
        self,
        audio: np.ndarray,
        reference: np.ndarray,
        sample_rate: int
    ) -> Dict:
        """Quick quality measurement for real-time decisions."""
        # Just calculate SNR for speed
        snr = calculate_snr(reference, audio)
        return {'snr': snr}
    
    def process_batch(
        self,
        audio_batch: List[Tuple[np.ndarray, int, str]],
        max_workers: int = 4
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Process a batch of audio files in parallel.
        
        Args:
            audio_batch: List of (audio, sample_rate, identifier) tuples
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of (enhanced_audio, metadata) tuples
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    self.enhance,
                    audio, sr, return_metadata=True
                ): idx
                for idx, (audio, sr, _) in enumerate(audio_batch)
            }
            
            # Collect results in order
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    enhanced, metadata = future.result()
                    results.append((idx, enhanced, metadata))
                except Exception as e:
                    logger.error(f"Batch processing error for item {idx}: {e}")
                    # Return original audio on error
                    audio, sr, identifier = audio_batch[idx]
                    results.append((idx, audio, {'error': str(e)}))
        
        # Sort by original index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Return without indices
        return [(enhanced, metadata) for _, enhanced, metadata in results]
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats['enhanced'] > 0:
            stats['avg_snr_improvement'] = stats['total_snr_improvement'] / stats['enhanced']
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_processed']
        else:
            stats['avg_snr_improvement'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        # Add engine info
        stats['engines'] = {
            'primary': self.primary_engine,
            'denoiser_available': self.denoiser is not None and self.denoiser.model is not None,
            'spectral_gating_available': self.spectral_gating is not None and self.spectral_gating.available
        }
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'skipped_clean': 0,
            'enhanced': 0,
            'total_snr_improvement': 0.0,
            'total_processing_time': 0.0
        }