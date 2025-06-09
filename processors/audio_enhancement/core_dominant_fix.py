#!/usr/bin/env python3
"""
Fixed version of core.py enhance method that prioritizes overlapping speech detection
and dominant speaker separation for ultra_aggressive mode.
"""

def enhance_fixed(self, audio, sample_rate=16000, noise_level=None, return_metadata=False):
    """
    Fixed enhancement method that checks for overlapping speakers BEFORE regular separation.
    
    This ensures dominant speaker separation is used when needed.
    """
    import time
    import numpy as np
    from copy import deepcopy
    
    start_time = time.time()
    original_dtype = audio.dtype
    
    # Initialize metadata
    metadata = {
        'enhanced': True,
        'noise_level': noise_level or 'unknown',
        'processing_time': 0,
        'engine_used': 'none',
        'secondary_speaker_detected': False,
        'snr_before': 0,
        'snr_after': 0,
        'pesq': 0,
        'stoi': 0
    }
    
    # Convert to float32 for processing
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize audio to [-1, 1] range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Auto-detect noise level if not specified
    if noise_level is None:
        if self.enhancement_level:
            noise_level = self.enhancement_level
        else:
            noise_level = self.assess_noise_level(audio, sample_rate)
    
    # Get enhancement configuration
    config = self.ENHANCEMENT_LEVELS.get(noise_level, self.ENHANCEMENT_LEVELS['moderate'])
    
    self.logger.info(f"Using enhancement config for {noise_level}: {config}")
    
    # Skip if clean
    if config.get('skip', False):
        self.stats['skipped_clean'] += 1
        if return_metadata:
            metadata.update({
                'enhanced': False,
                'enhancement_level': 'none',
                'snr_before': self.clean_threshold_snr,
                'snr_after': self.clean_threshold_snr,
                'processing_time': time.time() - start_time
            })
            return audio, metadata
        return audio
    
    # Start with a copy
    enhanced = audio.copy()
    
    # PRIORITY 1: Check for overlapping speakers FIRST in ultra_aggressive mode
    overlapping_processed = False
    if noise_level == 'ultra_aggressive':
        try:
            self.logger.info("=== PRIORITY CHECK: Analyzing for overlapping speakers ===")
            analysis = self.complete_separator.analyze_overlapping_speakers(enhanced, sample_rate)
            
            if analysis.has_overlapping_speech:
                self.logger.info(f"✓ Detected overlapping speech in {len(analysis.overlap_regions)} regions")
                self.logger.info(f"Primary speaker ratio: {analysis.primary_speaker_ratio:.2f}")
                
                # Use dominant speaker separator if available
                if self.dominant_separator is not None:
                    self.logger.info("→ Using DOMINANT SPEAKER SEPARATOR for better accuracy...")
                    enhanced = self.dominant_separator.extract_dominant_speaker(enhanced, sample_rate)
                    self.logger.info("✓ Dominant speaker extraction completed successfully")
                    metadata['secondary_speaker_detected'] = True
                    metadata['removal_method'] = 'dominant_speaker_separation'
                    metadata['overlapping_regions'] = len(analysis.overlap_regions)
                    overlapping_processed = True
                else:
                    self.logger.info("→ Using complete speaker separator...")
                    enhanced = self.complete_separator.extract_primary_speaker(enhanced, sample_rate)
                    self.logger.info("✓ Complete speaker separation applied successfully")
                    metadata['secondary_speaker_detected'] = True
                    metadata['removal_method'] = 'complete_speaker_separation'
                    metadata['overlapping_regions'] = len(analysis.overlap_regions)
                    overlapping_processed = True
            else:
                self.logger.info("✗ No overlapping speech detected")
                
        except Exception as e:
            self.logger.warning(f"Overlapping speaker analysis failed: {e}")
    
    # PRIORITY 2: Apply regular separation ONLY if no overlapping was processed
    if not overlapping_processed and config.get('use_speaker_separation', False):
        if self.speaker_separator is not None:
            try:
                self.logger.info("Checking for end-of-audio secondary speakers...")
                separation_result = self.speaker_separator.separate_speakers(enhanced, sample_rate)
                
                # Handle result format
                if hasattr(separation_result, 'audio'):
                    if hasattr(separation_result, 'rejected') and separation_result.rejected:
                        self.logger.info(f"Speaker separation rejected: {separation_result.rejection_reason}")
                        # Keep checking for other secondary speakers
                    else:
                        enhanced = separation_result.audio
                        metadata['secondary_speaker_detected'] = True
                        metadata['removal_method'] = 'speaker_separation'
                elif isinstance(separation_result, dict):
                    enhanced = separation_result.get('audio', enhanced)
                    if separation_result.get('detections'):
                        metadata['secondary_speaker_detected'] = True
                        metadata['removal_method'] = 'speaker_separation'
                        
            except Exception as e:
                self.logger.warning(f"Speaker separation failed: {e}")
    
    # Apply noise reduction based on config
    if config.get('denoiser_ratio', 0) > 0 or config.get('spectral_ratio', 0) > 0:
        self.logger.info("Applying noise reduction...")
        for pass_num in range(config.get('passes', 1)):
            if config.get('denoiser_ratio', 0) > 0 and self.denoiser_engine:
                enhanced = self.denoiser_engine.denoise(enhanced, sample_rate)
                metadata['engine_used'] = 'denoiser'
            
            if config.get('spectral_ratio', 0) > 0 and self.spectral_engine:
                enhanced = self.spectral_engine.reduce_noise(enhanced, sample_rate)
                metadata['engine_used'] = 'spectral' if metadata['engine_used'] == 'none' else 'combined'
    
    # Final safety checks for secondary speakers
    if noise_level == 'ultra_aggressive' and metadata.get('secondary_speaker_detected', False):
        # Apply intelligent end silencer as final safety
        try:
            if self.intelligent_end_silencer:
                self.logger.info("Applying intelligent end silencer as final safety measure...")
                enhanced = self.intelligent_end_silencer.process(enhanced, sample_rate)
        except Exception as e:
            self.logger.warning(f"Intelligent silencer failed: {e}")
    
    # Restore original data type
    if enhanced.dtype != original_dtype:
        if original_dtype == np.int16:
            enhanced = (enhanced * 32767).astype(np.int16)
        else:
            enhanced = enhanced.astype(original_dtype)
    
    # Update final metadata
    metadata['processing_time'] = time.time() - start_time
    self.stats['processed'] += 1
    
    if return_metadata:
        return enhanced, metadata
    return enhanced