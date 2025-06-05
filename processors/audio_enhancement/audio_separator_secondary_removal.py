"""
Secondary speaker removal using python-audio-separator.
Uses vocal isolation models to extract primary speaker.
"""

import numpy as np
import logging
import tempfile
import os
from typing import Tuple, Dict, Any, Optional
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioSeparatorSecondaryRemoval:
    """
    Remove secondary speakers using audio-separator library.
    """
    
    def __init__(self, 
                 model_name: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                 use_gpu: bool = False,
                 aggression: int = 5,
                 post_process: bool = True):
        """
        Initialize audio separator for secondary speaker removal.
        
        Args:
            model_name: Model to use for separation
            use_gpu: Whether to use GPU acceleration
            aggression: Intensity of extraction (-100 to 100)
            post_process: Enable post-processing to remove artifacts
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.aggression = aggression
        self.post_process = post_process
        self.separator = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of separator."""
        if self._initialized:
            return
            
        try:
            from audio_separator.separator import Separator
            
            # Initialize separator with configuration
            self.separator = Separator(
                log_level=logging.WARNING,
                model_file_dir="/tmp/audio-separator-models/",
                output_format="WAV",
                normalization_threshold=0.9,
                output_single_stem="Vocals",  # Only output vocals
                vr_params={
                    "batch_size": 1,
                    "window_size": 512,
                    "aggression": self.aggression,
                    "enable_tta": False,  # Disable for speed
                    "enable_post_process": self.post_process,
                    "post_process_threshold": 0.2,
                    "high_end_process": False
                }
            )
            
            # Load the model
            logger.info(f"Loading audio-separator model: {self.model_name}")
            self.separator.load_model(model_filename=self.model_name)
            self._initialized = True
            
        except ImportError:
            logger.error("audio-separator not installed. Install with: pip install audio-separator[cpu]")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize audio-separator: {e}")
            raise
            
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Remove secondary speakers using vocal separation.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        metadata = {
            'method': 'audio_separator',
            'model': self.model_name,
            'processing_applied': False,
            'error': None
        }
        
        # Initialize separator if needed
        try:
            self._initialize()
        except Exception as e:
            metadata['error'] = str(e)
            return audio, metadata
            
        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            temp_input_path = temp_input.name
            
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            temp_output_path = temp_output.name
            
        try:
            # Write input audio
            sf.write(temp_input_path, audio, sample_rate)
            
            # Set output directory to temp location
            self.separator.output_dir = os.path.dirname(temp_output_path)
            
            # Perform separation
            logger.info("Performing vocal separation...")
            output_files = self.separator.separate(temp_input_path)
            
            if not output_files:
                raise ValueError("No output files produced")
                
            # Read the vocal output
            vocal_file = output_files[0]  # Should be the vocals
            processed_audio, _ = sf.read(vocal_file)
            
            # Ensure same length as input
            if len(processed_audio) != len(audio):
                if len(processed_audio) > len(audio):
                    processed_audio = processed_audio[:len(audio)]
                else:
                    processed_audio = np.pad(processed_audio, (0, len(audio) - len(processed_audio)))
                    
            metadata['processing_applied'] = True
            
            # Calculate metrics
            original_power = np.mean(audio**2)
            processed_power = np.mean(processed_audio**2)
            
            if original_power > 0:
                metadata['power_reduction_db'] = 10 * np.log10(processed_power / original_power)
            
            # Clean up output files
            for f in output_files:
                try:
                    os.remove(f)
                except:
                    pass
                    
            return processed_audio, metadata
            
        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            metadata['error'] = str(e)
            return audio, metadata
            
        finally:
            # Clean up temporary files
            for path in [temp_input_path, temp_output_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except:
                    pass


class MultiModelSecondaryRemoval:
    """
    Try multiple models and strategies for secondary speaker removal.
    """
    
    def __init__(self, use_gpu: bool = False):
        """Initialize multi-model remover."""
        self.use_gpu = use_gpu
        
        # Different models to try
        self.models = [
            {
                'name': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
                'aggression': 10,  # Higher aggression for secondary removal
                'description': 'Best vocal SDR model'
            },
            {
                'name': 'vocals_mel_band_roformer.ckpt',
                'aggression': 15,
                'description': 'MelBand Roformer for vocals'
            },
            {
                'name': 'htdemucs_ft.yaml',
                'aggression': 5,
                'description': 'Demucs model (different architecture)'
            }
        ]
        
    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Try multiple models to find best secondary speaker removal.
        
        Returns:
            Tuple of (best_processed_audio, metadata)
        """
        best_audio = audio
        best_reduction = 0
        best_metadata = {'method': 'multi_model', 'models_tried': []}
        
        for model_config in self.models:
            try:
                logger.info(f"Trying model: {model_config['description']}")
                
                remover = AudioSeparatorSecondaryRemoval(
                    model_name=model_config['name'],
                    use_gpu=self.use_gpu,
                    aggression=model_config['aggression'],
                    post_process=True
                )
                
                processed, metadata = remover.process(audio, sample_rate)
                
                if metadata.get('processing_applied') and not metadata.get('error'):
                    reduction = abs(metadata.get('power_reduction_db', 0))
                    
                    best_metadata['models_tried'].append({
                        'model': model_config['name'],
                        'reduction_db': metadata.get('power_reduction_db', 0),
                        'error': metadata.get('error')
                    })
                    
                    # Keep the best result (most reduction while preserving main speaker)
                    if reduction > best_reduction and reduction < 20:  # Cap at -20dB to avoid over-processing
                        best_audio = processed
                        best_reduction = reduction
                        best_metadata['best_model'] = model_config['name']
                        best_metadata['best_reduction_db'] = metadata.get('power_reduction_db', 0)
                        
            except Exception as e:
                logger.warning(f"Model {model_config['name']} failed: {e}")
                best_metadata['models_tried'].append({
                    'model': model_config['name'],
                    'error': str(e)
                })
                
        best_metadata['processing_applied'] = best_reduction > 0
        
        return best_audio, best_metadata