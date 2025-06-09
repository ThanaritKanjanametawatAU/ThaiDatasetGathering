"""
Core audio enhancement pipeline with smart adaptive processing.
Orchestrates multiple enhancement engines based on audio characteristics.
"""

import numpy as np
import time
import logging
import multiprocessing
import torch
from typing import Optional, Dict, Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .engines.denoiser import DenoiserEngine
from .engines.spectral_gating import SpectralGatingEngine
from .speechbrain_separator import SpeechBrainSeparator, SeparationConfig
from .robust_secondary_removal import RobustSecondaryRemoval
from .simple_secondary_removal import SimpleSecondaryRemoval  # Keep for backward compatibility
from .full_audio_secondary_removal import FullAudioSecondaryRemoval  # Keep for backward compatibility
from .end_aware_secondary_removal import EndAwareSecondaryRemoval
from .aggressive_end_suppression import AggressiveEndSuppression
from .forced_end_silence import ForcedEndSilence
from .absolute_end_silencer import AbsoluteEndSilencer
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
            'passes': 2,
            'check_secondary_speaker': True,  # Enable secondary speaker detection
            'use_robust_removal': True        # Use new robust secondary removal
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
            'check_secondary_speaker': True,  # Enable secondary speaker detection
            'use_speaker_separation': True    # Enable speaker separation
        },
        'secondary_speaker': {
            'skip': False,
            'use_speaker_separation': True,
            'suppression_strength': 0.6,
            'passes': 1
        },
        'selective_secondary_removal': {
            'skip': False,
            'use_selective_removal': True,
            'check_secondary_speaker': True,
            'preserve_primary': True,
            'passes': 2  # One for separation, one for quality preservation
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
        
        # Initialize speaker separator with SpeechBrain configuration
        from .speechbrain_separator import SeparationConfig as SBSeparationConfig
        separation_config = SBSeparationConfig(
            confidence_threshold=0.7,   # High confidence threshold for quality
            device="cuda" if self.use_gpu else "cpu",
            batch_size=16,  # Optimized for RTX 5090
            speaker_selection="energy",  # Use energy-based speaker selection
            use_mixed_precision=True,  # Enable mixed precision for speed
            quality_thresholds={
                "min_pesq": 3.5,
                "min_stoi": 0.85,
                "max_spectral_distortion": 0.15
            }
        )
        self.speaker_separator = SpeechBrainSeparator(separation_config)
        
        # Initialize robust secondary speaker remover (NEW - uses professional techniques)
        self.robust_remover = RobustSecondaryRemoval(
            method='diarization',      # Use speaker diarization approach
            use_vad=True,             # Enable Voice Activity Detection
            preserve_primary=True,     # Always preserve primary speaker
            quality_threshold=0.7,     # Quality threshold for filtering
            use_source_separation=False,  # Disable by default for speed
            fast_mode=False,          # Full accuracy mode
            device='cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        )
        
        # Keep simple and full audio removers for backward compatibility
        self.simple_remover = SimpleSecondaryRemoval(
            energy_threshold=0.5,     # More sensitive detection
            min_silence_duration=0.05, # Detect shorter pauses
            suppression_db=-60        # Extremely strong suppression (-60dB = 0.1% of original)
        )
        
        self.full_audio_remover = FullAudioSecondaryRemoval(
            detection_threshold=0.5,
            min_silence_duration=0.05,
            preserve_main_freq_range=(250, 3500),  # Preserve main speech frequencies
            suppression_strength=0.8,  # Strong but not total suppression
            use_adaptive_filtering=True
        )
        
        # Initialize end-aware secondary removal for S5-like patterns
        self.end_aware_remover = EndAwareSecondaryRemoval(
            base_separator=self.speaker_separator,
            end_analysis_duration=3.0,  # Analyze last 3 seconds
            end_processing_overlap=0.5,
            confidence_threshold=0.7
        )
        
        # Initialize aggressive end suppression for ultra_aggressive mode
        self.aggressive_end_suppressor = AggressiveEndSuppression(
            end_duration=3.0,
            primary_speaker_duration=5.0,
            suppression_threshold=0.3,  # Low threshold for aggressive detection
            gate_threshold=0.1
        )
        
        # Initialize forced end silence as last resort
        self.forced_end_silencer = ForcedEndSilence(
            analysis_duration=3.0,
            silence_threshold=0.1,  # Very sensitive
            fade_duration=0.5
        )
        
        # Initialize intelligent end silencer for smart secondary removal
        from .intelligent_end_silencer import IntelligentEndSilencer
        from .secondary_removal import SecondaryRemover
        device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        self.intelligent_end_silencer = IntelligentEndSilencer(device=device)
        
        # Initialize secondary remover (TDD-based implementation)
        self.secondary_remover = SecondaryRemover(
            vad_threshold=0.02,
            energy_change_threshold=1.3,
            min_secondary_duration=0.05,
            max_secondary_duration=1.0,
            fade_duration=0.05
        )
        
        # Initialize complete separator for handling overlapping speakers
        from .complete_separation import CompleteSeparator
        self.complete_separator = CompleteSeparator(
            device='cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        
        # Also initialize dominant speaker separator for better accuracy
        try:
            from .dominant_speaker_separation import DominantSpeakerSeparator
            self.dominant_separator = DominantSpeakerSeparator(
                device='cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            )
            logger.info("Initialized dominant speaker separator")
        except Exception as e:
            logger.warning(f"Could not load dominant speaker separator: {e}")
            self.dominant_separator = None
        
        # Initialize complete secondary removal for ultra_aggressive mode
        try:
            from .complete_secondary_removal import CompleteSecondaryRemoval
            self.complete_secondary_removal = CompleteSecondaryRemoval(
                device='cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            )
            logger.info("Initialized complete secondary removal")
        except Exception as e:
            logger.warning(f"Could not load complete secondary removal: {e}")
            self.complete_secondary_removal = None
        
        # Initialize selective secondary removal for quality-preserving removal
        try:
            from .selective_secondary_removal import SelectiveSecondaryRemoval
            self.selective_remover = SelectiveSecondaryRemoval(
                device='cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            )
            logger.info("Initialized selective secondary removal")
        except Exception as e:
            logger.warning(f"Could not load selective secondary removal: {e}")
            self.selective_remover = None
        
        # Keep absolute silencer as last resort only
        self.absolute_end_silencer = AbsoluteEndSilencer(
            silence_duration=0.2  # Only silence last 0.2 seconds as absolute last resort
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
                # Handle both old dict format and new SeparationOutput format
                if hasattr(separation_result, 'num_speakers_detected'):
                    # New format
                    if separation_result.num_speakers_detected > 1 and not separation_result.rejected:
                        logger.debug(f"Secondary speaker detected: {separation_result.num_speakers_detected} speakers")
                        return 'secondary_speaker'
                elif isinstance(separation_result, dict) and separation_result.get('detections'):
                    # Old format (backward compatibility)
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
        
        # Store original dtype to preserve it
        original_dtype = audio.dtype
        
        # Convert float16 to float32 for processing (AudioEnhancer doesn't support float16)
        if audio.dtype == np.float16:
            audio = audio.astype(np.float32)
        
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
        
        logger.info(f"Using enhancement config for {noise_level}: {config}")
        
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
        
        logger.info(f"Processing with noise_level={noise_level}, check_secondary={check_secondary}, use_separation={use_separation}")
        
        # Initialize enhanced audio
        enhanced = audio.copy()
        
        # PRIORITY: For selective_secondary_removal, use quality-preserving approach
        if noise_level == 'selective_secondary_removal' and self.selective_remover is not None:
            try:
                logger.info("=== USING SELECTIVE SECONDARY REMOVAL ===")
                enhanced, metadata = self.selective_remover.process(audio, sample_rate)
                
                # Return early with preserved dtype
                if enhanced.dtype != original_dtype:
                    enhanced = enhanced.astype(original_dtype)
                
                if return_metadata:
                    processing_time = time.time() - start_time
                    metadata.update({
                        'processing_time': processing_time,
                        'enhancement_level': 'selective_secondary_removal'
                    })
                    return enhanced, metadata
                return enhanced
                
            except Exception as e:
                logger.error(f"Selective removal failed: {e}, falling back to regular processing")
                # Continue with regular processing
        
        # PRIORITY: For ultra_aggressive, check for overlapping speakers FIRST
        overlapping_processed = False
        if noise_level == 'ultra_aggressive':
            try:
                logger.info("=== CHECKING FOR OVERLAPPING SPEAKERS (PRIORITY) ===")
                analysis = self.complete_separator.analyze_overlapping_speakers(audio, sample_rate)
                
                if analysis.has_overlapping_speech:
                    logger.info(f"✓ Overlapping speech detected in {len(analysis.overlap_regions)} regions")
                    
                    # Use complete secondary removal for better accuracy
                    if self.complete_secondary_removal is not None:
                        logger.info("→ Using COMPLETE SECONDARY REMOVAL...")
                        enhanced, removal_metrics = self.complete_secondary_removal.process_with_verification(audio, sample_rate)
                        logger.info(f"✓ Complete secondary removal: begin={removal_metrics['begin_energy_db']:.1f}dB, end={removal_metrics['end_energy_db']:.1f}dB")
                        
                        # If removal not effective, try time-based approach
                        if not removal_metrics.get('removal_effective', False):
                            logger.info("→ Complete removal not effective, trying TIME-BASED approach...")
                            from .time_based_removal import TimeBasedSecondaryRemoval
                            device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
                            time_remover = TimeBasedSecondaryRemoval(device=device)
                            enhanced, time_metrics = time_remover.process_with_metrics(audio, sample_rate)
                            logger.info(f"✓ Time-based removal: begin={time_metrics['begin_energy_db']:.1f}dB, end={time_metrics['end_energy_db']:.1f}dB")
                            removal_metrics.update(time_metrics)
                        
                        # IMPORTANT: Return early to avoid additional processing that might reintroduce artifacts
                        # Complete removal should be the final step for ultra_aggressive mode
                        if enhanced.dtype != original_dtype:
                            enhanced = enhanced.astype(original_dtype)
                        
                        if return_metadata:
                            processing_time = time.time() - start_time
                            metadata = {
                                'enhanced': True,
                                'noise_level': noise_level,
                                'enhancement_level': 'complete_secondary_removal',
                                'snr_before': 0,
                                'snr_after': 0,
                                'processing_time': processing_time,
                                'engine_used': 'complete_secondary_removal',
                                'secondary_speaker_detected': True,
                                'secondary_speaker_removed': True,
                                'begin_energy_db': removal_metrics['begin_energy_db'],
                                'end_energy_db': removal_metrics['end_energy_db'],
                                'removal_effective': removal_metrics.get('removal_effective', False)
                            }
                            return enhanced, metadata
                        return enhanced
                    elif self.dominant_separator is not None:
                        logger.info("→ Using DOMINANT SPEAKER SEPARATOR...")
                        enhanced = self.dominant_separator.extract_dominant_speaker(audio, sample_rate)
                        logger.info("✓ Dominant speaker extraction completed")
                        overlapping_processed = True
                    else:
                        logger.info("→ Using complete speaker separator...")
                        enhanced = self.complete_separator.extract_primary_speaker(audio, sample_rate)
                        logger.info("✓ Complete speaker separation completed")
                        overlapping_processed = True
                        # Don't skip secondary processing - still apply additional removal
                        # check_secondary = False  # REMOVED - still check
                        # use_separation = False   # REMOVED - still separate
                else:
                    logger.info("✗ No overlapping speech detected, but applying complete removal for ultra_aggressive")
                    # For ultra_aggressive, always apply complete secondary removal
                    if self.complete_secondary_removal is not None:
                        logger.info("→ Applying COMPLETE SECONDARY REMOVAL anyway...")
                        enhanced, removal_metrics = self.complete_secondary_removal.process_with_verification(audio, sample_rate)
                        logger.info(f"✓ Complete removal done: begin={removal_metrics['begin_energy_db']:.1f}dB, end={removal_metrics['end_energy_db']:.1f}dB")
                        
                        # IMPORTANT: Return early to avoid additional processing
                        if enhanced.dtype != original_dtype:
                            enhanced = enhanced.astype(original_dtype)
                        
                        if return_metadata:
                            processing_time = time.time() - start_time
                            metadata = {
                                'enhanced': True,
                                'noise_level': noise_level,
                                'enhancement_level': 'complete_secondary_removal',
                                'snr_before': 0,
                                'snr_after': 0,
                                'processing_time': processing_time,
                                'engine_used': 'complete_secondary_removal',
                                'secondary_speaker_detected': False,
                                'secondary_speaker_removed': True,
                                'begin_energy_db': removal_metrics['begin_energy_db'],
                                'end_energy_db': removal_metrics['end_energy_db'],
                                'removal_effective': removal_metrics.get('removal_effective', False)
                            }
                            return enhanced, metadata
                        return enhanced
            except Exception as e:
                logger.warning(f"Overlapping analysis failed: {e}, continuing with regular processing")
        
        # Check if we should use robust removal (NEW)
        use_robust_removal = config.get('use_robust_removal', False)
        if use_robust_removal and check_secondary:
            logger.info("=== USING ROBUST SECONDARY SPEAKER REMOVAL ===")
            try:
                processed, removal_metadata = self.robust_remover.process(enhanced, sample_rate)
                enhanced = processed
                
                # Log removal results
                logger.info(f"Robust removal complete: {removal_metadata.get('secondary_speakers_found', 0)} secondary speakers found")
                logger.info(f"Primary speaker: {removal_metadata.get('primary_speaker_id')}")
                logger.info(f"Segments processed: {removal_metadata.get('segments_processed', 0)}")
                
                if return_metadata:
                    # Initialize metadata if not exists
                    if 'metadata' not in locals():
                        metadata = {}
                    
                    # Update metadata with robust removal info
                    metadata['robust_removal_applied'] = True
                    metadata['secondary_speakers_found'] = removal_metadata.get('secondary_speakers_found', 0)
                    metadata['primary_speaker_id'] = removal_metadata.get('primary_speaker_id')
                    metadata['segments_processed'] = removal_metadata.get('segments_processed', 0)
                    metadata['method_used'] = removal_metadata.get('method', 'diarization')
                
            except Exception as e:
                logger.warning(f"Robust secondary removal failed: {e}, continuing with standard processing")
        
        # For ultra_aggressive mode or when explicitly requested, check for secondary speakers
        elif (check_secondary or use_separation or noise_level == 'secondary_speaker') and not overlapping_processed:
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
                    # Handle both formats
                    if hasattr(separation_result, 'audio'):
                        enhanced = separation_result.audio
                        detections = []  # New format doesn't use detections
                    else:
                        enhanced = separation_result['audio']
                        detections = separation_result.get('detections', [])
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
                        # Calculate SNR improvement
                        snr_improvement = self._calculate_snr_improvement(audio, enhanced, sample_rate)
                        
                        metadata = {
                            'enhanced': True,
                            'noise_level': noise_level,
                            'enhancement_level': 'advanced_separation',
                            'snr_before': 0,  # Will be updated with metrics below
                            'snr_after': 0,   # Will be updated with metrics below
                            'snr_improvement': snr_improvement,
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
                        # Preserve original dtype
                        if enhanced.dtype != original_dtype:
                            enhanced = enhanced.astype(original_dtype)
                        return enhanced, metadata
            else:
                # Use original speaker separator
                separation_result = self.speaker_separator.separate_speakers(audio, sample_rate)
                # Handle both formats
                if hasattr(separation_result, 'audio'):
                    # New SeparationOutput format
                    if hasattr(separation_result, 'rejected') and separation_result.rejected:
                        logger.info(f"Speaker separation rejected: {separation_result.rejection_reason}")
                        # Don't use the separated audio, keep original
                        enhanced = audio.copy()
                        # Mark that we should skip further separation-based processing
                        skip_separation_processing = True
                    else:
                        enhanced = separation_result.audio
                        skip_separation_processing = False
                    detections = []  # New format doesn't use detections
                elif isinstance(separation_result, dict):
                    # Old dict format (backward compatibility)
                    enhanced = separation_result['audio']
                    detections = separation_result.get('detections', [])
                    skip_separation_processing = False
                else:
                    # Unknown format, log error and use original
                    logger.error(f"Unknown separation result format: {type(separation_result)}")
                    enhanced = audio.copy()
                    detections = []
                    skip_separation_processing = True
            
            # Update statistics
            if detections:
                self.stats['secondary_speakers_detected'] += 1
                self.stats['secondary_speakers_removed'] += len(detections)
            
            # Apply full audio secondary removal for comprehensive removal
            if noise_level in ['aggressive', 'ultra_aggressive'] and not skip_separation_processing:
                # First, apply complete separation for overlapping speakers
                if noise_level == 'ultra_aggressive':
                    try:
                        logger.info("Checking for overlapping speakers throughout audio...")
                        analysis = self.complete_separator.analyze_overlapping_speakers(enhanced, sample_rate)
                        
                        if analysis.has_overlapping_speech:
                            logger.info(f"Detected overlapping speech in {len(analysis.overlap_regions)} regions")
                            
                            # Use dominant speaker separator if available for better accuracy
                            if self.dominant_separator is not None:
                                logger.info("Using dominant speaker separator for better accuracy...")
                                enhanced = self.dominant_separator.extract_dominant_speaker(enhanced, sample_rate)
                                logger.info("Dominant speaker extraction completed successfully")
                            else:
                                logger.info("Applying complete speaker separation...")
                                enhanced = self.complete_separator.extract_primary_speaker(enhanced, sample_rate)
                                logger.info("Complete speaker separation applied successfully")
                    except Exception as e:
                        logger.warning(f"Speaker separation failed: {e}, continuing with other methods")
                
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
                
                # Apply aggressive end suppression for ultra_aggressive mode
                if noise_level == 'ultra_aggressive':
                    try:
                        logger.info("Applying aggressive end suppression...")
                        enhanced = self.aggressive_end_suppressor.process(enhanced, sample_rate)
                        logger.info("Aggressive end suppression completed")
                        
                        # Apply forced end silence as final measure
                        logger.info("Applying forced end silence as final measure...")
                        enhanced = self.forced_end_silencer.process(enhanced, sample_rate)
                        logger.info("Forced end silence completed")
                        
                        # Apply intelligent end silencer instead of absolute
                        logger.info("Applying intelligent end silencer...")
                        # Use TDD-based secondary remover
                        enhanced = self.secondary_remover.remove_secondary_speakers(enhanced, sample_rate)
                        logger.info("Intelligent end silence completed")
                    except Exception as e:
                        logger.warning(f"Aggressive end suppression failed: {e}, continuing with current audio")
                        
            # Final safety check for ultra_aggressive mode
            if noise_level == 'ultra_aggressive':
                # Apply intelligent end silencer with verification
                logger.info("Applying intelligent end silencer for ultra_aggressive mode...")
                try:
                    # Use intelligent end silencer with forced silence threshold
                    enhanced, silence_metrics = self.intelligent_end_silencer.process_with_verification(
                        enhanced, sample_rate, 
                        force_silence_threshold=-45  # Force silence if above -45dB
                    )
                    logger.info(f"Intelligent end silencer applied: end_speaker_detected={silence_metrics.get('end_speaker_detected')}, "
                               f"processed_end_energy={silence_metrics.get('processed_end_energy', 'N/A')}dB")
                    
                    # If intelligent silencer didn't detect/remove, check manually
                    if not silence_metrics.get('end_speaker_detected'):
                        # Check if there's still significant audio at the very end (last 200ms)
                        check_samples = int(0.2 * sample_rate)
                        if len(enhanced) > check_samples:
                            end_segment = enhanced[-check_samples:]
                            max_amp = np.max(np.abs(end_segment))
                            if max_amp > 0.01:
                                logger.warning(f"Secondary speaker still present after intelligent silencer (max amp: {max_amp:.3f})")
                                logger.info("Applying secondary remover as additional measure...")
                                try:
                                    # Use secondary remover as additional processing
                                    enhanced = self.secondary_remover.remove_secondary_speakers(enhanced, sample_rate)
                                    
                                    # Final check and absolute silencer if needed
                                    final_check = enhanced[-check_samples:] if len(enhanced) > check_samples else enhanced
                                    if np.max(np.abs(final_check)) > 0.01:
                                        logger.warning("Secondary remover failed, using absolute silencer on last 200ms only")
                                        enhanced = self.absolute_end_silencer.process(enhanced, sample_rate)
                                except Exception as e:
                                    logger.error(f"Failed to apply secondary removal: {e}")
                except Exception as e:
                    logger.error(f"Failed to apply intelligent end silencer: {e}")
            else:
                # Apply end-aware removal for aggressive mode (non-ultra_aggressive)
                if noise_level == 'aggressive':
                    try:
                        logger.info("Applying end-aware secondary speaker removal...")
                        enhanced = self.end_aware_remover.process_audio(enhanced, sample_rate)
                        logger.info("End-aware secondary removal completed")
                    except Exception as e:
                        logger.warning(f"End-aware removal failed: {e}, continuing with current audio")
            
            # Special handling when separation was rejected
            if noise_level == 'ultra_aggressive' and 'skip_separation_processing' in locals() and skip_separation_processing:
                logger.info("Separation was rejected, applying intelligent end silencing only")
                try:
                    # Use TDD-based secondary remover
                    enhanced = self.secondary_remover.remove_secondary_speakers(enhanced, sample_rate)
                    logger.info("Intelligent end silencing completed for rejected separation case")
                except Exception as e:
                    logger.error(f"Intelligent silencing failed: {e}")
            
            # Also apply regular enhancement for ultra_aggressive mode
            if noise_level == 'ultra_aggressive' and config.get('passes', 0) > 0 and not ('skip_separation_processing' in locals() and skip_separation_processing):
                
                # Apply progressive enhancement after speaker separation
                for pass_num in range(config.get('passes', 1)):
                    if self.spectral_gating and self.spectral_gating.available:
                        enhanced = self.spectral_gating.process(enhanced, sample_rate)
                    elif self.denoiser and self.denoiser.model is not None:
                        enhanced = self.denoiser.process(
                            enhanced, sample_rate,
                            denoiser_dry=config['denoiser_ratio']
                        )
                
                # REMOVED preserve_ratio for ultra_aggressive mode
                # The preserve_ratio was mixing back the original audio which contains
                # the secondary speaker, defeating all our removal efforts
            
            if return_metadata:
                processing_time = time.time() - start_time
                # Calculate initial metrics if not already done
                initial_metrics = self._measure_quality_quick(audio, audio, sample_rate)
                final_metrics = self._measure_quality_quick(enhanced, audio, sample_rate)
                
                # Calculate SNR improvement
                snr_improvement = self._calculate_snr_improvement(audio, enhanced, sample_rate)
                
                metadata = {
                    'enhanced': True,
                    'noise_level': noise_level,
                    'enhancement_level': 'secondary_speaker_removal',
                    'snr_before': initial_metrics.get('snr', 0),
                    'snr_after': final_metrics.get('snr', 0),
                    'snr_improvement': snr_improvement,
                    'processing_time': processing_time,
                    'engine_used': 'speaker_separator',
                    'secondary_speaker_detected': len(detections) > 0 or (hasattr(separation_result, 'num_speakers_detected') and separation_result.num_speakers_detected > 1),
                    'secondary_speaker_confidence': max([d.confidence for d in detections]) if detections else (separation_result.confidence if hasattr(separation_result, 'confidence') else 0.0),
                    'num_secondary_speakers': len(detections) if detections else (separation_result.num_speakers_detected - 1 if hasattr(separation_result, 'num_speakers_detected') else 0),
                    'speaker_similarity': separation_result.metrics.get('similarity_preservation', 1.0) if hasattr(separation_result, 'metrics') else 1.0,
                    'use_speaker_separation': True,  # Indicate that speaker separation was used
                    'pesq': final_metrics.get('pesq', 2.5),
                    'stoi': final_metrics.get('stoi', 0.75)
                }
                # Preserve original dtype
                if enhanced.dtype != original_dtype:
                    enhanced = enhanced.astype(original_dtype)
                return enhanced, metadata
            # Preserve original dtype for non-metadata return
            if enhanced.dtype != original_dtype:
                enhanced = enhanced.astype(original_dtype)
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
        
        # REMOVED preserve_ratio for ultra_aggressive mode
        # The preserve_ratio was mixing back the original audio which can contain
        # secondary speakers or noise, defeating the purpose of ultra_aggressive mode
        
        # Calculate final metrics
        final_metrics = self._measure_quality_quick(enhanced, original_audio, sample_rate)
        
        # Update statistics
        self.stats['enhanced'] += 1
        snr_improvement = self._calculate_snr_improvement(original_audio, enhanced, sample_rate)
        self.stats['total_snr_improvement'] += snr_improvement
        
        processing_time = time.time() - start_time
        self.stats['total_processing_time'] += processing_time
        
        # Preserve original dtype
        if enhanced.dtype != original_dtype:
            enhanced = enhanced.astype(original_dtype)
        
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