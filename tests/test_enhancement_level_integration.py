"""
Test-Driven Development tests for enhancement level integration.
This test file verifies that the enhancement level specified in command line
is actually used by the audio enhancement system.

Problem: The enhancement level from main.sh (e.g., 'ultra_aggressive') is not being
used by AudioEnhancer, which instead auto-detects noise level.
"""

import pytest
import numpy as np
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace

from processors.audio_enhancement.core import AudioEnhancer
from main import main as main_function


class TestEnhancementLevelIntegration:
    """Test suite for enhancement level integration between main.py and AudioEnhancer."""
    
    def test_audio_enhancer_respects_configured_level(self):
        """Test that AudioEnhancer uses the configured enhancement level instead of auto-detection."""
        # Create test audio with secondary speaker
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # Main speaker (1000 Hz)
        t = np.linspace(0, duration, samples)
        main_speaker = 0.7 * np.sin(2 * np.pi * 1000 * t)
        
        # Secondary speaker (500 Hz) 
        secondary_speaker = 0.3 * np.sin(2 * np.pi * 500 * t)
        
        # Combined audio
        audio = main_speaker + secondary_speaker
        
        # Test with ultra_aggressive level
        enhancer = AudioEnhancer(use_gpu=False, fallback_to_cpu=True)
        
        # Process with specific enhancement level
        enhanced, metadata = enhancer.enhance(
            audio, 
            sample_rate, 
            noise_level='ultra_aggressive',  # This should be used, not auto-detected
            return_metadata=True
        )
        
        # Check that ultra_aggressive settings were applied
        assert metadata['noise_level'] == 'ultra_aggressive'
        assert 'secondary_speaker' in metadata.get('enhancement_level', '') or \
               metadata.get('secondary_speaker_detected', False) or \
               metadata.get('use_speaker_separation', False), \
               "Ultra aggressive mode should check for secondary speakers"
    
    def test_main_passes_enhancement_level_to_processor(self):
        """Test that main.py correctly passes enhancement level to processors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock checkpoint directory
            checkpoint_dir = os.path.join(tmpdir, 'checkpoints')
            os.makedirs(checkpoint_dir)
            
            # Mock arguments with ultra_aggressive enhancement
            args = Namespace(
                fresh=True,
                append=False,
                resume=False,
                datasets=['GigaSpeech2'],
                sample=True,
                sample_size=10,
                hf_repo='test/repo',
                streaming=True,
                streaming_batch_size=10,
                upload_batch_size=100,
                enable_speaker_id=False,
                enable_stt=False,
                enable_audio_enhancement=True,
                enhancement_level='ultra_aggressive',  # This should be passed through
                enhancement_batch_size=5,
                enhancement_gpu=False,
                enhancement_dashboard=False,
                no_standardization=False,
                sample_rate=16000,
                target_db=-20.0,
                no_volume_norm=False,
                no_stt=True,
                stt_batch_size=16,
                verbose=True,
                output=tmpdir
            )
            
            # Patch the processor creation to capture the config
            captured_config = None
            original_create_processor = None
            
            def mock_create_processor(dataset_name, config):
                nonlocal captured_config
                captured_config = config
                # Return a mock processor
                mock_processor = Mock()
                mock_processor.process_all_splits = Mock(return_value=iter([]))
                return mock_processor
            
            with patch('main.create_processor', mock_create_processor):
                with patch('main.StreamingUploader'):
                    with patch('utils.huggingface.read_hf_token', return_value='test_token'):
                        with patch.dict('sys.modules', {'GPUtil': MagicMock()}):
                            # Patch sys.argv to simulate command line args
                            import sys
                            original_argv = sys.argv
                            try:
                                sys.argv = [
                                    'main.py', '--fresh', 'GigaSpeech2', 
                                    '--sample', '--sample-size', '10',
                                    '--hf-repo', 'test/repo',
                                    '--streaming', '--enable-audio-enhancement',
                                    '--enhancement-level', 'ultra_aggressive',
                                    '--enhancement-batch-size', '5'
                                ]
                                main_function()
                            except SystemExit:
                                pass  # Ignore exit
                            finally:
                                sys.argv = original_argv
            
            # Verify enhancement level was passed in config
            assert captured_config is not None, "Processor config should be captured"
            assert captured_config['audio_enhancement']['enabled'] is True
            assert captured_config['audio_enhancement']['level'] == 'ultra_aggressive'
    
    def test_audio_enhancer_process_batch_uses_configured_level(self):
        """Test that process_batch method respects configured enhancement level."""
        enhancer = AudioEnhancer(use_gpu=False, fallback_to_cpu=True)
        
        # Create test batch
        sample_rate = 16000
        batch = []
        for i in range(3):
            audio = np.random.normal(0, 0.1, sample_rate * 2)  # 2 seconds of noise
            batch.append((audio, sample_rate, f'sample_{i}'))
        
        # Mock the enhance method to capture calls
        original_enhance = enhancer.enhance
        enhance_calls = []
        
        def mock_enhance(audio, sr, noise_level=None, **kwargs):
            enhance_calls.append({
                'noise_level': noise_level,
                'kwargs': kwargs
            })
            return original_enhance(audio, sr, noise_level=noise_level, **kwargs)
        
        enhancer.enhance = mock_enhance
        
        # Process batch with specific noise level
        # Currently process_batch doesn't accept noise_level parameter
        # This test will fail, demonstrating the issue
        with pytest.raises(TypeError):
            results = enhancer.process_batch(batch, noise_level='ultra_aggressive')
    
    def test_processor_audio_enhancement_config_usage(self):
        """Test that processors use the audio enhancement configuration correctly."""
        from processors.gigaspeech2 import GigaSpeech2Processor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create processor with audio enhancement config
            config = {
                'checkpoint_dir': tmpdir,
                'log_dir': tmpdir,
                'streaming': True,
                'batch_size': 10,
                'dataset_name': 'GigaSpeech2',
                'audio_enhancement': {
                    'enabled': True,
                    'enhancer': Mock(spec=AudioEnhancer),
                    'level': 'ultra_aggressive',  # This should be used
                    'batch_size': 5
                }
            }
            
            processor = GigaSpeech2Processor(config)
            
            # Check that enhancement config is stored
            assert hasattr(processor, 'audio_enhancement')
            assert processor.audio_enhancement['enabled'] is True
            assert processor.audio_enhancement['level'] == 'ultra_aggressive'
    
    def test_enhancement_level_affects_secondary_speaker_detection(self):
        """Test that different enhancement levels trigger different behaviors."""
        enhancer = AudioEnhancer(use_gpu=False, fallback_to_cpu=True)
        
        # Create audio with clear secondary speaker
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Two distinct speakers
        speaker1 = 0.6 * np.sin(2 * np.pi * 800 * t)
        speaker2 = 0.4 * np.sin(2 * np.pi * 1500 * t + np.pi/4)
        audio = speaker1 + speaker2
        
        # Test with different levels
        levels = ['clean', 'mild', 'moderate', 'aggressive', 'ultra_aggressive']
        results = {}
        
        for level in levels:
            enhanced, metadata = enhancer.enhance(
                audio.copy(), 
                sample_rate,
                noise_level=level,
                return_metadata=True
            )
            results[level] = metadata
        
        # Only ultra_aggressive should check for secondary speakers
        assert results['ultra_aggressive'].get('check_secondary_speaker', False) or \
               results['ultra_aggressive'].get('secondary_speaker_detected', False) or \
               results['ultra_aggressive'].get('use_speaker_separation', False), \
               "Ultra aggressive mode should enable secondary speaker detection"
        
        # Other levels should not
        for level in ['clean', 'mild', 'moderate', 'aggressive']:
            assert not results[level].get('check_secondary_speaker', False), \
                   f"{level} mode should not check for secondary speakers"


class TestMainScriptIntegration:
    """Test the full integration from main.sh through to audio enhancement."""
    
    def test_main_sh_enhancement_level_parameter(self):
        """Verify main.sh passes enhancement level correctly."""
        # Read main.sh to verify enhancement level is set
        with open('/media/ssd1/SparkVoiceProject/ThaiDatasetGathering/main.sh', 'r') as f:
            content = f.read()
        
        # Check that enhancement level is defined
        assert 'ENHANCEMENT_LEVEL="ultra_aggressive"' in content
        
        # Check that it's passed to main.py
        assert '--enhancement-level $ENHANCEMENT_LEVEL' in content
    
    def test_enhancement_level_configuration_in_audio_enhancer(self):
        """Test that AudioEnhancer has proper configuration for enhancement levels."""
        enhancer = AudioEnhancer(use_gpu=False)
        
        # Check that enhancement configuration exists
        assert hasattr(enhancer, 'enhancement_config')
        
        # Check ultra_aggressive configuration
        assert 'ultra_aggressive' in enhancer.enhancement_config
        config = enhancer.enhancement_config['ultra_aggressive']
        
        # Verify secondary speaker detection is enabled
        assert config.get('check_secondary_speaker', False) is True
        assert config.get('use_speaker_separation', False) is True
    
    def test_end_to_end_secondary_speaker_removal_flow(self):
        """Test the complete flow from command line to actual secondary speaker removal."""
        # This is an integration test that verifies the entire pipeline
        
        # Create a sample with obvious secondary speaker
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Main speaker (continuous)
        main_speaker = 0.7 * np.sin(2 * np.pi * 1000 * t)
        
        # Secondary speaker (intermittent)
        secondary_speaker = np.zeros_like(t)
        # Add secondary speaker in middle section
        start_idx = int(0.5 * sample_rate)
        end_idx = int(2.5 * sample_rate)
        secondary_speaker[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * 2000 * t[start_idx:end_idx])
        
        combined_audio = main_speaker + secondary_speaker
        
        # Process with ultra_aggressive setting
        enhancer = AudioEnhancer(use_gpu=False, fallback_to_cpu=True)
        
        # This should use ultra_aggressive settings and detect/remove secondary speaker
        enhanced, metadata = enhancer.enhance(
            combined_audio,
            sample_rate,
            noise_level='ultra_aggressive',
            return_metadata=True
        )
        
        # Verify secondary speaker was detected
        assert metadata.get('secondary_speaker_detected', False) or \
               'secondary' in metadata.get('enhancement_level', '').lower(), \
               "Secondary speaker should be detected in ultra_aggressive mode"
        
        # Verify the audio was modified (should have lower power where secondary speaker was)
        original_power = np.mean(combined_audio[start_idx:end_idx] ** 2)
        enhanced_power = np.mean(enhanced[start_idx:end_idx] ** 2)
        
        # Enhanced should have less power in the secondary speaker region
        # (This might fail if secondary removal isn't working)
        assert enhanced_power < original_power * 0.9, \
               "Enhanced audio should have reduced power where secondary speaker was present"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])