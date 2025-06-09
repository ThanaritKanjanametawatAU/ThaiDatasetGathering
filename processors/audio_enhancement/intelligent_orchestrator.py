"""
Intelligent Enhancement Orchestrator that integrates the Decision Framework
with the existing enhancement system.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

from processors.audio_enhancement.enhancement_orchestrator import EnhancementOrchestrator
from processors.audio_enhancement.decision_framework import (
    DecisionFramework,
    AudioAnalysis,
    Issue,
    DecisionContext,
    QualityMetrics
)
from utils.snr_measurement import SNRMeasurement
from processors.audio_enhancement.quality_validator import QualityValidator
import librosa

logger = logging.getLogger(__name__)


class IntelligentOrchestrator(EnhancementOrchestrator):
    """
    Enhanced orchestrator that uses the Decision Framework for intelligent
    audio processing decisions.
    """
    
    def __init__(self, config: Optional[Dict] = None, target_snr: float = 35):
        super().__init__(target_snr=target_snr)
        
        # Initialize decision framework
        self.decision_framework = DecisionFramework(
            config=config,
            enable_cache=True,
            preset="quality_focused"  # Default for voice cloning
        )
        
        # Quality validator for analysis
        self.quality_validator = QualityValidator()
        
        # Track decisions for learning
        self.current_decision_id = None
    
    def enhance(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
        """
        Enhance audio using intelligent decision making.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (enhanced_audio, metrics_dict)
        """
        # Step 1: Comprehensive audio analysis
        analysis = self._analyze_audio(audio, sample_rate)
        
        # Step 2: Make intelligent decision
        decision = self.decision_framework.make_decision(analysis)
        self.current_decision_id = decision.tracking_id
        
        logger.info(f"Decision made: {decision.action} (confidence: {decision.confidence:.2f})")
        logger.debug(f"Decision explanation: {decision.explanation}")
        
        # Step 3: Execute decision
        if decision.action == "no_enhancement" or decision.action == "pass_through":
            # No processing needed
            metrics = {
                "initial_snr": analysis.snr,
                "final_snr": analysis.snr,
                "enhanced": False,
                "decision_action": decision.action,
                "decision_confidence": decision.confidence,
                "decision_explanation": decision.explanation,
                "strategy_used": "none"
            }
            return audio, metrics
        
        # Step 4: Configure enhancement based on decision
        self._configure_enhancement(decision)
        
        # Step 5: Run enhancement with base class
        enhanced_audio, metrics = super().enhance(audio, sample_rate)
        
        # Step 6: Add decision metadata to metrics
        metrics.update({
            "decision_action": decision.action,
            "decision_confidence": decision.confidence,
            "decision_explanation": decision.explanation,
            "strategy_used": self._get_strategy_from_decision(decision)
        })
        
        # Step 7: Record outcome for learning
        if self.current_decision_id:
            success = metrics.get("target_achieved", False)
            quality_gain = metrics.get("snr_improvement", 0.0)
            self.decision_framework.record_outcome(
                self.current_decision_id,
                success=success,
                quality_gain=quality_gain
            )
        
        return enhanced_audio, metrics
    
    def _analyze_audio(self, audio: np.ndarray, sample_rate: int) -> AudioAnalysis:
        """
        Perform comprehensive audio analysis for decision making.
        """
        # Basic SNR measurement
        snr = self.snr_calc.measure_snr(audio, sample_rate)
        
        # Detect issues
        issues = self._detect_audio_issues(audio, sample_rate)
        
        # Calculate quality metrics
        try:
            # Spectral quality
            stft = np.abs(librosa.stft(audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=stft, sr=sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sample_rate))
            
            # Normalize to 0-1 range
            spectral_quality = min(1.0, spectral_rolloff / (sample_rate / 2))
            
            # Zero crossing rate (indicates noisiness)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
        except Exception as e:
            logger.warning(f"Failed to calculate spectral features: {e}")
            spectral_quality = 0.5
            zcr = 0.1
        
        # Build analysis object
        analysis = AudioAnalysis(
            snr=snr,
            spectral_quality=spectral_quality,
            issues=issues,
            audio_length=len(audio) / sample_rate,
            pesq=0.0,  # Would need reference for actual PESQ
            stoi=0.0,  # Would need reference for actual STOI
            constraints=self._get_constraints(),
            context=self._get_context()
        )
        
        return analysis
    
    def _detect_audio_issues(self, audio: np.ndarray, sample_rate: int) -> List[Issue]:
        """
        Detect various audio issues.
        """
        issues = []
        
        # Detect clipping
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.99:
            clipping_severity = min(1.0, (max_amplitude - 0.99) * 10)
            issues.append(Issue("clipping", severity=clipping_severity))
        
        # Detect noise (simple heuristic based on high-frequency energy)
        try:
            stft = np.abs(librosa.stft(audio))
            high_freq_energy = np.mean(stft[-int(stft.shape[0] * 0.2):, :])
            total_energy = np.mean(stft)
            
            if total_energy > 0:
                noise_ratio = high_freq_energy / total_energy
                if noise_ratio > 0.3:
                    noise_severity = min(1.0, (noise_ratio - 0.3) * 2)
                    issues.append(Issue("noise", severity=noise_severity))
        except Exception as e:
            logger.warning(f"Failed to detect noise: {e}")
        
        # Detect reverb (using clarity measure)
        try:
            # Simple reverb detection using energy decay
            envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
            if len(envelope) > 10:
                decay_rate = np.mean(np.diff(envelope[envelope > np.max(envelope) * 0.1]))
                if abs(decay_rate) < 0.1:  # Slow decay indicates reverb
                    reverb_severity = min(1.0, 0.1 / (abs(decay_rate) + 0.01))
                    issues.append(Issue("reverb", severity=reverb_severity))
        except Exception as e:
            logger.warning(f"Failed to detect reverb: {e}")
        
        return issues
    
    def _get_constraints(self) -> Dict:
        """
        Get processing constraints.
        """
        return {
            "max_latency": 1000,  # ms
            "preserve_harmonics": True,
            "target_use_case": "voice_cloning"
        }
    
    def _get_context(self) -> DecisionContext:
        """
        Get decision context.
        """
        return DecisionContext(
            use_case="voice_cloning",
            time_constraint=None,  # No real-time constraint
            audio_value="high"  # High value for voice cloning
        )
    
    def _configure_enhancement(self, decision):
        """
        Configure enhancement parameters based on decision.
        """
        # Adjust stages based on decision
        if decision.action == "aggressive_enhancement":
            # Use all stages with aggressive settings
            self.max_iterations = 3
            for stage in self.stages:
                if hasattr(stage, 'set_aggressiveness'):
                    stage.set_aggressiveness(0.9)
        
        elif decision.action == "moderate_enhancement":
            # Use moderate settings
            self.max_iterations = 2
            for stage in self.stages:
                if hasattr(stage, 'set_aggressiveness'):
                    stage.set_aggressiveness(0.6)
        
        elif decision.action == "light_enhancement":
            # Use light settings
            self.max_iterations = 1
            # Only use subset of stages
            self.stages = self.stages[:2]  # Only spectral and wiener
            for stage in self.stages:
                if hasattr(stage, 'set_aggressiveness'):
                    stage.set_aggressiveness(0.3)
        
        # Configure specific repair methods if needed
        if "noise_reduction" in decision.action:
            # Ensure noise reduction is prioritized
            pass
        
        if "declipping" in decision.action:
            # Add declipping stage if not present
            pass
    
    def _get_strategy_from_decision(self, decision) -> str:
        """
        Extract strategy name from decision.
        """
        if "aggressive" in decision.action:
            return "aggressive"
        elif "moderate" in decision.action:
            return "balanced"
        elif "light" in decision.action:
            return "conservative"
        else:
            return "custom"
    
    def set_use_case(self, use_case: str):
        """
        Set the use case for decision making.
        
        Args:
            use_case: One of "voice_cloning", "general", "real_time"
        """
        # Update decision framework preset based on use case
        if use_case == "voice_cloning":
            self.decision_framework = DecisionFramework(preset="quality_focused")
        elif use_case == "real_time":
            self.decision_framework = DecisionFramework(preset="speed_focused")
        else:
            self.decision_framework = DecisionFramework()  # Balanced
    
    def get_decision_stats(self) -> Dict:
        """
        Get statistics about decisions made.
        """
        return self.decision_framework.outcome_tracker.get_success_rates()


class IntelligentDecisionFramework(IntelligentOrchestrator):
    """
    Alias for backward compatibility and clarity.
    This class name matches what's expected in the task specification.
    """
    pass