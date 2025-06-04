"""Orchestrator for multi-stage audio enhancement targeting 35dB SNR."""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

from utils.snr_measurement import SNRMeasurement
from utils.audio_metrics import calculate_pesq, calculate_stoi
from processors.audio_enhancement.adaptive_spectral import AdaptiveSpectralSubtraction
from processors.audio_enhancement.wiener_filter import AdaptiveWienerFilter
from processors.audio_enhancement.harmonic_enhancer import HarmonicEnhancer
from processors.audio_enhancement.perceptual_post import PerceptualPostProcessor
from .quality_monitor import QualityMonitor


logger = logging.getLogger(__name__)


class EnhancementOrchestrator:
    """Orchestrate multi-stage enhancement to achieve target SNR."""
    
    def __init__(self, target_snr: float = 35, max_iterations: int = 3):
        self.target_snr = target_snr
        self.max_iterations = max_iterations
        
        # Initialize stages
        self.stages = [
            AdaptiveSpectralSubtraction(),
            AdaptiveWienerFilter(),
            HarmonicEnhancer(),
            PerceptualPostProcessor()
        ]
        
        # Initialize measurement tools
        self.snr_calc = SNRMeasurement()
        self.quality_monitor = QualityMonitor()
        
        # Callback for metrics tracking (optional)
        self._metric_callback = None
    
    def enhance(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
        """
        Enhance audio to target SNR.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (enhanced_audio, metrics_dict)
        """
        # Step 1: Initial assessment
        current_snr = self.snr_calc.measure_snr(audio, sample_rate)
        
        metrics = {
            "initial_snr": current_snr,
            "final_snr": current_snr,
            "enhanced": False,
            "iterations": 0,
            "naturalness": 1.0,
            "stages_applied": []
        }
        
        # Check if already meets target
        if current_snr >= self.target_snr:
            logger.info(f"Audio already meets target SNR: {current_snr:.1f} dB")
            return audio, metrics
        
        # Keep original for quality comparison
        original_audio = audio.copy()
        enhanced_audio = audio.copy()
        
        # Step 2: Progressive enhancement
        for iteration in range(self.max_iterations):
            logger.info(f"Enhancement iteration {iteration + 1}/{self.max_iterations}")
            
            previous_audio = enhanced_audio.copy()
            iteration_improved = False
            
            # Apply each stage
            for stage_idx, stage in enumerate(self.stages):
                stage_name = stage.__class__.__name__
                logger.debug(f"Applying {stage_name}")
                
                # Apply enhancement
                stage_output = stage.process(enhanced_audio, sample_rate)
                
                # Check quality metrics
                quality_metrics = self.quality_monitor.check_naturalness(original_audio, stage_output)
                
                # Callback for tracking (if set)
                if self._metric_callback:
                    current_snr_check = self.snr_calc.measure_snr(stage_output, sample_rate)
                    self._metric_callback(stage_output, stage_name)
                
                # Decide whether to keep this enhancement
                if quality_metrics >= 0.85:  # Naturalness threshold
                    enhanced_audio = stage_output
                    metrics["stages_applied"].append(stage_name)
                    iteration_improved = True
                    logger.debug(f"{stage_name} applied successfully, naturalness: {quality_metrics:.2f}")
                else:
                    logger.warning(f"{stage_name} rejected due to low naturalness: {quality_metrics:.2f}")
                    # Skip remaining stages in this iteration if quality dropped
                    break
            
            # Check if target reached
            current_snr = self.snr_calc.measure_snr(enhanced_audio, sample_rate)
            metrics["iterations"] = iteration + 1
            
            if current_snr >= self.target_snr:
                logger.info(f"Target SNR reached: {current_snr:.1f} dB")
                break
            
            # Check if no improvement in this iteration
            if not iteration_improved:
                logger.warning("No improvement in this iteration, stopping")
                enhanced_audio = previous_audio  # Revert to previous best
                break
        
        # Step 3: Final quality assessment
        final_snr = self.snr_calc.measure_snr(enhanced_audio, sample_rate)
        naturalness_score = self.quality_monitor.check_naturalness(original_audio, enhanced_audio)
        
        # Calculate additional metrics
        try:
            pesq_score = calculate_pesq(original_audio, enhanced_audio, sample_rate)
            stoi_score = calculate_stoi(original_audio, enhanced_audio, sample_rate)
        except Exception as e:
            logger.warning(f"Failed to calculate PESQ/STOI: {e}")
            pesq_score = None
            stoi_score = None
        
        # Update final metrics
        metrics.update({
            "final_snr": final_snr,
            "enhanced": True,
            "naturalness": float(naturalness_score),
            "pesq": pesq_score,
            "stoi": stoi_score,
            "snr_improvement": final_snr - metrics["initial_snr"],
            "target_achieved": final_snr >= self.target_snr
        })
        
        logger.info(f"Enhancement complete: {metrics['initial_snr']:.1f} -> {final_snr:.1f} dB")
        
        return enhanced_audio, metrics
    
    def set_metric_callback(self, callback):
        """Set callback function for metric tracking during processing."""
        self._metric_callback = callback