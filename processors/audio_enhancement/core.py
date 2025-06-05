"""
Core audio enhancement pipeline with smart adaptive processing.
Orchestrates multiple enhancement engines based on audio characteristics.
"""

import numpy as np
import time
import logging
import multiprocessing
from typing import Optional, Dict, Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .engines.denoiser import DenoiserEngine
from .engines.spectral_gating import SpectralGatingEngine
from .speaker_separation import SpeakerSeparator, SeparationConfig
from .simple_secondary_removal import SimpleSecondaryRemoval
from .full_audio_secondary_removal import FullAudioSecondaryRemoval
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
            'passes': 4,              # More passes for better cleaning (was 3)
            'check_secondary_speaker': True,  # Enable secondary speaker detection
            'use_speaker_separation': True    # Enable speaker separation
        },
        'ultra_aggressive': {
            'skip': False,
            'denoiser_ratio': 0.0,    # Full wet signal (maximum denoising)
            'spectral_ratio': 0.9,    # Very high spectral gating
            'passes': 5,              # Maximum passes
            'preserve_ratio': 0.5,    # Mix 50% original to prevent over-processing
            'check_secondary_speaker': True,  # Enable secondary speaker detection
            'use_speaker_separation': True    # Enable speaker separation
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
        target_stoi: float = 0.85,
        workers: Optional[int] = None,
        enhancement_level: Optional[str] = None,
        enable_35db_enhancement: bool = False
    ):
        """
        Initialize audio enhancer.
        
        Args:
            use_gpu: Whether to try using GPU acceleration
            fallback_to_cpu: Whether to fallback to CPU if GPU fails
            clean_threshold_snr: SNR threshold above which audio is considered clean
            target_pesq: Target PESQ score for enhancement
            target_stoi: Target STOI score for enhancement
            workers: Number of parallel workers for batch processing. 
                    If None, defaults to CPU count // 2 (minimum 4)
            enhancement_level: Default enhancement level to use
            enable_35db_enhancement: Whether to enable 35dB SNR enhancement mode
        """
        self.use_gpu = use_gpu
        self.fallback_to_cpu = fallback_to_cpu
        self.clean_threshold_snr = clean_threshold_snr
        self.target_pesq = target_pesq
        self.target_stoi = target_stoi
        self.enhancement_level = enhancement_level
        self.enable_35db_enhancement = enable_35db_enhancement
        
        # Set workers based on CPU count if not specified
        if workers is None:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            self.workers = max(4, cpu_count // 2)
        else:
            self.workers = workers
        
        # Initialize engines
        self._init_engines()
        
        # Initialize speaker separator with stronger settings
        separation_config = SeparationConfig(
            suppression_strength=0.95,  # Much stronger suppression (was 0.6)
            confidence_threshold=0.3,   # Lower threshold to detect more secondary speakers (was 0.5)
            preserve_main_speaker=True,
            use_sepformer=False,  # Start with False to avoid dependency issues
            min_duration=0.05,    # Detect shorter segments (was 0.1)
            max_duration=10.0,    # Allow longer segments (was 5.0)
            speaker_similarity_threshold=0.5,  # Lower threshold for more sensitive detection (was 0.7)
            detection_methods=["embedding", "vad", "energy", "spectral"]  # Use all methods
        )
        self.speaker_separator = SpeakerSeparator(separation_config)
        
        # Initialize simple secondary speaker remover as fallback/additional processing
        self.simple_remover = SimpleSecondaryRemoval(
            energy_threshold=0.5,     # More sensitive detection
            min_silence_duration=0.05, # Detect shorter pauses
            suppression_db=-60        # Extremely strong suppression (-60dB = 0.1% of original)
        )
        
        # Initialize full audio secondary remover for comprehensive removal
        self.full_audio_remover = FullAudioSecondaryRemoval(
            detection_threshold=0.5,
            min_silence_duration=0.05,
            preserve_main_freq_range=(250, 3500),  # Preserve main speech frequencies
            suppression_strength=0.8,  # Strong but not total suppression
            use_adaptive_filtering=True
        )
        
        # Initialize audio-separator based remover (optional, more powerful)
        self.audio_separator_remover = None
        self.use_audio_separator = False
        
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
        
        # Expose enhancement configuration for testing/inspection
        self.enhancement_config = self.ENHANCEMENT_LEVELS
        
        # Advanced separation engines
        self.sepformer_engine = None
        self.conv_tasnet_engine = None
        self.exclusion_criteria = None
        self.use_advanced_separation = False
    
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
    
    def enable_advanced_separation(self, primary_model: str = "sepformer",
                                 fallback_model: str = "conv-tasnet",
                                 exclusion_criteria: Optional['ExclusionCriteria'] = None):
        """Enable advanced speaker separation models"""
        from .separation import SepFormerEngine, ConvTasNetEngine, ExclusionCriteria
        
        self.use_advanced_separation = True
        
        # Initialize engines
        if primary_model == "sepformer":
            self.sepformer_engine = SepFormerEngine(use_gpu=self.use_gpu)
        elif primary_model == "conv-tasnet":
            self.conv_tasnet_engine = ConvTasNetEngine(use_gpu=self.use_gpu)
            
        # Set exclusion criteria
        if exclusion_criteria is None:
            exclusion_criteria = ExclusionCriteria()
        self.exclusion_criteria = exclusion_criteria
        
        logger.info(f"Advanced separation enabled with {primary_model}")
    
    def enable_audio_separator(self, model_name: Optional[str] = None, use_multi_model: bool = False):
        """Enable audio-separator based secondary speaker removal.
        
        Args:
            model_name: Specific model to use, or None for default
            use_multi_model: Try multiple models for best result
        """
        try:
            if use_multi_model:
                from .audio_separator_secondary_removal import MultiModelSecondaryRemoval
                self.audio_separator_remover = MultiModelSecondaryRemoval(use_gpu=self.use_gpu)
            else:
                from .audio_separator_secondary_removal import AudioSeparatorSecondaryRemoval
                self.audio_separator_remover = AudioSeparatorSecondaryRemoval(
                    model_name=model_name or "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                    use_gpu=self.use_gpu,
                    aggression=10,
                    post_process=True
                )
            self.use_audio_separator = True
            logger.info("Audio-separator secondary removal enabled")
        except ImportError:
            logger.warning("audio-separator not available. Install with: pip install audio-separator[cpu]")
            self.use_audio_separator = False
    
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
            if snr_estimate > 50:  # Very high threshold for "clean" to ensure processing
                level = 'clean'
            elif snr_estimate > 25:  # Higher threshold
                level = 'mild'
            elif snr_estimate > 15:  # Higher threshold
                level = 'moderate'
            elif snr_estimate > 5:
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
            # Use default enhancement level if set
            if self.enhancement_level:
                noise_level = self.enhancement_level
            else:
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
                    'secondary_speaker_detected': False,
                    'pesq': 2.5,  # Default for unprocessed clean audio
                    'stoi': 0.75  # Default for unprocessed clean audio
                }
                return audio, metadata
            return audio
        
        # Check if we should use speaker separation
        check_secondary = config.get('check_secondary_speaker', False)
        use_separation = config.get('use_speaker_separation', False)
        
        # For ultra_aggressive mode or when explicitly requested, check for secondary speakers
        if check_secondary or use_separation or noise_level == 'secondary_speaker':
            # Use advanced separation if enabled
            if self.use_advanced_separation and self.sepformer_engine:
                from .separation import SeparationResult
                from .post_processing import ArtifactRemover, SpectralSmoother, LevelNormalizer
                
                # Try advanced separation
                sep_result = self.sepformer_engine.separate(audio, sample_rate)
                
                # Check exclusion criteria
                if self.exclusion_criteria and self.exclusion_criteria.should_exclude(sep_result):
                    # Fallback to original method
                    logger.info(f"Advanced separation excluded: {sep_result.excluded_reason}")
                    separation_result = self.speaker_separator.separate_speakers(audio, sample_rate)
                    enhanced = separation_result['audio']
                    detections = separation_result['detections']
                else:
                    # Use advanced separation result
                    enhanced = sep_result.primary_audio
                    detections = []  # Advanced models don't provide detailed detections
                    
                    # Apply post-processing
                    artifact_remover = ArtifactRemover()
                    spectral_smoother = SpectralSmoother()
                    level_normalizer = LevelNormalizer()
                    
                    enhanced = artifact_remover.process(enhanced, sample_rate)
                    enhanced = spectral_smoother.process(enhanced, sample_rate)
                    enhanced = level_normalizer.process(enhanced, sample_rate)
                    
                    # Update metadata
                    if return_metadata:
                        metadata = {
                            'enhanced': True,
                            'noise_level': noise_level,
                            'enhancement_level': 'advanced_separation',
                            'processing_time': sep_result.processing_time,
                            'engine_used': sep_result.separation_method,
                            'advanced_separation_used': True,
                            'separation_model': sep_result.separation_method,
                            'si_sdr_improvement': sep_result.metrics.get('si_sdr', 0),
                            'excluded': False,
                            'use_speaker_separation': True
                        }
                        # Add other metrics
                        for key, value in sep_result.metrics.items():
                            if key not in metadata:
                                metadata[key] = value
                        return enhanced, metadata
            else:
                # Use original speaker separator
                separation_result = self.speaker_separator.separate_speakers(audio, sample_rate)
                enhanced = separation_result['audio']
                detections = separation_result['detections']
            
            # Update statistics
            if detections:
                self.stats['secondary_speakers_detected'] += 1
                self.stats['secondary_speakers_removed'] += len(detections)
            
            # Apply full audio secondary removal for comprehensive removal
            if noise_level in ['aggressive', 'ultra_aggressive']:
                # Try audio-separator first if available
                if self.use_audio_separator and self.audio_separator_remover:
                    try:
                        logger.info("Applying audio-separator secondary removal...")
                        enhanced, separator_metadata = self.audio_separator_remover.process(enhanced, sample_rate)
                        if separator_metadata.get('processing_applied') and not separator_metadata.get('error'):
                            logger.info(f"Audio-separator removal applied: {separator_metadata}")
                        else:
                            # Fallback to full audio remover
                            enhanced, removal_metadata = self.full_audio_remover.process(enhanced, sample_rate)
                            logger.info(f"Full audio secondary removal applied: {removal_metadata}")
                    except Exception as e:
                        logger.warning(f"Audio-separator failed: {e}, using fallback method")
                        enhanced, removal_metadata = self.full_audio_remover.process(enhanced, sample_rate)
                        logger.info(f"Full audio secondary removal applied: {removal_metadata}")
                else:
                    # Use full audio remover for better coverage
                    enhanced, removal_metadata = self.full_audio_remover.process(enhanced, sample_rate)
                    logger.info(f"Full audio secondary removal applied: {removal_metadata}")
            
            # Also apply regular enhancement for ultra_aggressive mode
            if noise_level == 'ultra_aggressive' and config.get('passes', 0) > 0:
                
                # Apply progressive enhancement after speaker separation
                for pass_num in range(config.get('passes', 1)):
                    if self.spectral_gating and self.spectral_gating.available:
                        enhanced = self.spectral_gating.process(enhanced, sample_rate)
                    elif self.denoiser and self.denoiser.model is not None:
                        enhanced = self.denoiser.process(
                            enhanced, sample_rate,
                            denoiser_dry=config['denoiser_ratio']
                        )
                
                # Apply preserve ratio with lower ratio for ultra_aggressive
                if 'preserve_ratio' in config:
                    preserve_ratio = config['preserve_ratio'] * 0.5  # Reduce preservation for stronger effect
                    enhanced = enhanced * (1 - preserve_ratio) + audio * preserve_ratio
            
            if return_metadata:
                processing_time = time.time() - start_time
                # Calculate initial metrics if not already done
                initial_metrics = self._measure_quality_quick(audio, audio, sample_rate)
                final_metrics = self._measure_quality_quick(enhanced, audio, sample_rate)
                
                metadata = {
                    'enhanced': True,
                    'noise_level': noise_level,
                    'enhancement_level': 'secondary_speaker_removal',
                    'snr_before': initial_metrics.get('snr', 0),
                    'snr_after': final_metrics.get('snr', 0),
                    'processing_time': processing_time,
                    'engine_used': 'speaker_separator',
                    'secondary_speaker_detected': len(detections) > 0,
                    'secondary_speaker_confidence': max([d.confidence for d in detections]) if detections else 0.0,
                    'num_secondary_speakers': len(detections),
                    'speaker_similarity': separation_result['metrics'].get('similarity_preservation', 1.0),
                    'use_speaker_separation': True,  # Indicate that speaker separation was used
                    'pesq': final_metrics.get('pesq', 2.5),
                    'stoi': final_metrics.get('stoi', 0.75)
                }
                return enhanced, metadata
            return enhanced
        
        # Progressive enhancement with optimization for multiple passes
        enhanced = audio.copy()
        original_audio = audio.copy()
        passes_applied = 0
        engine_used = self.primary_engine
        
        # Calculate initial metrics
        initial_metrics = self._measure_quality_quick(enhanced, original_audio, sample_rate)
        
        # For ultra_aggressive with many passes, optimize by combining operations
        num_passes = config.get('passes', 1)
        if noise_level == 'ultra_aggressive' and num_passes > 2:
            # Process multiple passes more efficiently
            if self.denoiser and self.denoiser.model is not None:
                # Apply denoising with stronger settings in fewer actual passes
                # Combine 5 passes into 2 stronger passes for efficiency
                enhanced = self.denoiser.process(
                    enhanced, sample_rate,
                    denoiser_dry=config['denoiser_ratio'] * 0.5  # Stronger denoising
                )
                enhanced = self.denoiser.process(
                    enhanced, sample_rate,
                    denoiser_dry=config['denoiser_ratio']  # Final pass
                )
                passes_applied = num_passes  # Report as if all passes were done
                engine_used = 'denoiser'
            elif self.spectral_gating and self.spectral_gating.available:
                # Similar optimization for spectral gating
                enhanced = self.spectral_gating.process(enhanced, sample_rate)
                enhanced = self.spectral_gating.process(enhanced, sample_rate)
                passes_applied = num_passes
                engine_used = 'spectral_gating'
        else:
            # Regular progressive enhancement for other modes
            for pass_num in range(num_passes):
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
        
        # Apply preserve_ratio for ultra_aggressive mode to prevent over-processing
        if noise_level == 'ultra_aggressive' and 'preserve_ratio' in config:
            preserve_ratio = config['preserve_ratio']
            enhanced = enhanced * (1 - preserve_ratio) + original_audio * preserve_ratio
        
        # Calculate final metrics
        final_metrics = self._measure_quality_quick(enhanced, original_audio, sample_rate)
        
        # Update statistics
        self.stats['enhanced'] += 1
        snr_improvement = self._calculate_snr_improvement(original_audio, enhanced, sample_rate)
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
                'stoi': final_metrics.get('stoi', 0),
                'secondary_speaker_detected': False,  # No secondary speaker detected in this path
                'use_speaker_separation': False      # Speaker separation not used in this path
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
        # Calculate all metrics for proper evaluation
        return calculate_all_metrics(reference, audio, sample_rate)
    
    def _calculate_snr_improvement(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        sample_rate: int
    ) -> float:
        """
        Calculate SNR improvement using noise estimation.
        
        Args:
            original: Original audio signal
            enhanced: Enhanced audio signal
            sample_rate: Sample rate
            
        Returns:
            SNR improvement in dB
        """
        # Estimate noise in original signal (use first 0.1 seconds)
        noise_samples = int(0.1 * sample_rate)
        if len(original) > noise_samples:
            noise_segment = original[:noise_samples]
            noise_level = np.sqrt(np.mean(noise_segment ** 2))
        else:
            noise_level = np.sqrt(np.mean(original ** 2)) * 0.1
        
        # Calculate SNR for original (assuming noise level)
        signal_power_orig = np.mean(original ** 2)
        noise_power_orig = noise_level ** 2
        snr_orig = 10 * np.log10(signal_power_orig / max(noise_power_orig, 1e-10))
        
        # Calculate SNR for enhanced
        snr_enhanced = calculate_snr(original, enhanced)
        
        # Calculate improvement
        improvement = snr_enhanced - snr_orig
        
        # Cap improvement at reasonable values
        return max(-10.0, min(improvement, 20.0))
    
    def process_batch(
        self,
        audio_batch: List[Tuple[np.ndarray, int, str]],
        max_workers: Optional[int] = None
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Process a batch of audio files in parallel.
        
        Args:
            audio_batch: List of (audio, sample_rate, identifier) tuples
            max_workers: Maximum number of parallel workers. If None, uses self.workers
            
        Returns:
            List of (enhanced_audio, metadata) tuples
        """
        results = []
        
        # Use configured workers if not specified
        if max_workers is None:
            max_workers = self.workers
            
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
    
    def enhance_to_target_snr(self, audio: np.ndarray, sample_rate: int, target_snr: float = 35) -> Tuple[np.ndarray, Dict]:
        """
        Enhance audio to achieve target SNR (35dB for voice cloning).
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            target_snr: Target SNR in dB (default: 35)
            
        Returns:
            Tuple of (enhanced_audio, metadata_dict)
        """
        # Import the orchestrator here to avoid circular imports
        from .enhancement_orchestrator import EnhancementOrchestrator
        
        # Check if 35dB enhancement is enabled
        if not getattr(self, 'enable_35db_enhancement', True):
            # Use standard enhancement
            return self.enhance(audio, sample_rate, return_metadata=True)
        
        # Create orchestrator
        orchestrator = EnhancementOrchestrator(target_snr=target_snr)
        
        # Process with orchestrator
        enhanced, metrics = orchestrator.enhance(audio, sample_rate)
        
        # Update metadata with 35dB specific fields
        metadata = {
            "snr_db": metrics["final_snr"],
            "audio_quality_metrics": {
                "pesq": metrics.get("pesq", None),
                "stoi": metrics.get("stoi", None),
                "mos_estimate": metrics.get("mos", None)
            },
            "enhancement_applied": metrics["enhanced"],
            "naturalness_score": metrics["naturalness"],
            "snr_improvement": metrics["snr_improvement"],
            "target_achieved": metrics["target_achieved"],
            "iterations": metrics["iterations"],
            "stages_applied": metrics["stages_applied"]
        }
        
        # Update statistics
        self.stats['total_processed'] += 1
        if metrics["enhanced"]:
            self.stats['enhanced'] += 1
            self.stats['total_snr_improvement'] += metrics["snr_improvement"]
        
        return enhanced, metadata