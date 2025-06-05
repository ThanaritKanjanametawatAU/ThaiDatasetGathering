#!/usr/bin/env python3
"""
Test audio-separator approach for secondary speaker removal.
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.gigaspeech2 import GigaSpeech2Processor


class TestAudioSeparatorApproach(unittest.TestCase):
    """Test using audio-separator for secondary speaker removal."""
    
    def test_library_import(self):
        """Test if audio-separator can be imported."""
        try:
            from audio_separator.separator import Separator
            print("✓ audio-separator is installed and can be imported")
        except ImportError:
            print("✗ audio-separator not installed")
            print("Install with: pip install audio-separator[cpu]")
            self.skipTest("audio-separator not installed")
    
    def test_s5_with_audio_separator(self):
        """Test S5 sample with audio-separator."""
        print("\n=== Testing S5 with Audio Separator ===")
        
        # Check if library is available
        try:
            from processors.audio_enhancement.audio_separator_secondary_removal import (
                AudioSeparatorSecondaryRemoval
            )
        except ImportError as e:
            print(f"Import error: {e}")
            self.skipTest("audio-separator not available")
        
        # Get S5 sample
        config = {
            "name": "GigaSpeech2",
            "source": "speechcolab/gigaspeech2",
            "cache_dir": "./cache",
            "streaming": True,
        }
        
        processor = GigaSpeech2Processor(config)
        samples = list(processor.process_streaming(sample_mode=True, sample_size=5))
        
        self.assertGreaterEqual(len(samples), 5, "Need at least 5 samples")
        
        s5 = samples[4]
        audio_data = s5.get('audio', {})
        self.assertIsInstance(audio_data, dict)
        self.assertIn('array', audio_data)
        
        audio = audio_data['array']
        sr = audio_data.get('sampling_rate', 16000)
        
        print(f"S5 audio length: {len(audio)/sr:.2f}s")
        
        # Try audio separator
        remover = AudioSeparatorSecondaryRemoval(
            model_name="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            use_gpu=False,
            aggression=10,  # Higher aggression for secondary removal
            post_process=True
        )
        
        try:
            processed, metadata = remover.process(audio, sr)
            
            print(f"\nProcessing results:")
            print(f"  Method: {metadata.get('method')}")
            print(f"  Model: {metadata.get('model')}")
            print(f"  Applied: {metadata.get('processing_applied')}")
            print(f"  Error: {metadata.get('error')}")
            
            if metadata.get('processing_applied'):
                print(f"  Power reduction: {metadata.get('power_reduction_db', 0):.1f} dB")
                
                # Save output
                import soundfile as sf
                os.makedirs("test_audio_output", exist_ok=True)
                sf.write("test_audio_output/s5_audio_separator.wav", processed, sr)
                print("Saved: test_audio_output/s5_audio_separator.wav")
                
        except Exception as e:
            print(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_multi_model_approach(self):
        """Test multiple models for best result."""
        print("\n=== Testing Multi-Model Approach ===")
        
        try:
            from processors.audio_enhancement.audio_separator_secondary_removal import (
                MultiModelSecondaryRemoval
            )
        except ImportError:
            self.skipTest("audio-separator not available")
        
        # Get S5 sample
        config = {
            "name": "GigaSpeech2",
            "source": "speechcolab/gigaspeech2",
            "cache_dir": "./cache",
            "streaming": True,
        }
        
        processor = GigaSpeech2Processor(config)
        samples = list(processor.process_streaming(sample_mode=True, sample_size=5))
        
        s5 = samples[4]
        audio = s5['audio']['array']
        sr = s5['audio']['sampling_rate']
        
        # Try multi-model approach
        remover = MultiModelSecondaryRemoval(use_gpu=False)
        
        try:
            processed, metadata = remover.process(audio, sr)
            
            print("\nMulti-model results:")
            print(f"  Models tried: {len(metadata.get('models_tried', []))}")
            
            for model_result in metadata.get('models_tried', []):
                print(f"\n  Model: {model_result['model']}")
                if model_result.get('error'):
                    print(f"    Error: {model_result['error']}")
                else:
                    print(f"    Reduction: {model_result.get('reduction_db', 0):.1f} dB")
            
            if metadata.get('best_model'):
                print(f"\n  Best model: {metadata['best_model']}")
                print(f"  Best reduction: {metadata.get('best_reduction_db', 0):.1f} dB")
                
                # Save output
                import soundfile as sf
                os.makedirs("test_audio_output", exist_ok=True)
                sf.write("test_audio_output/s5_multi_model.wav", processed, sr)
                print("Saved: test_audio_output/s5_multi_model.wav")
                
        except Exception as e:
            print(f"Multi-model processing failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    unittest.main()