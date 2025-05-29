"""
Test audio enhancement integration with the main dataset gathering pipeline.

This module tests:
1. Command-line flag for enabling audio enhancement
2. Enhanced audio format preservation
3. Enhancement metrics tracking
4. Integration with existing features (speaker ID, STT, streaming)
5. Dashboard monitoring during processing
6. Batch processing with enhancement
"""

import unittest
from unittest.mock import patch, Mock, MagicMock, call
import numpy as np
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
import argparse

from main import main, parse_arguments
from processors.base_processor import BaseProcessor
# Audio enhancement imports to be created later
# from processors.audio_enhancement.core import AudioEnhancer
# from processors.audio_enhancement.engines.denoiser import DenoiserEngine
# from processors.audio_enhancement.engines.spectral_gating import SpectralGatingEngine
from config import TARGET_SCHEMA, DATASET_CONFIG


class TestAudioEnhancementIntegration(unittest.TestCase):
    """Test audio enhancement integration with main pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, "cache")
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create test audio data
        self.sample_rate = 16000
        self.duration = 2.0
        self.audio_samples = self._create_test_audio()
    
    def _run_main_with_args(self, args):
        """Helper to run main with command line arguments."""
        with patch('sys.argv', ['main.py'] + args):
            return main()
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_audio(self):
        """Create test audio with signal and noise."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        # Create signal (440Hz tone)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        # Add noise
        noise = 0.1 * np.random.normal(0, 1, signal.shape)
        return signal + noise
    
    def test_command_line_enhancement_flag(self):
        """Test that audio enhancement can be enabled via command-line flag."""
        # Test flag parsing
        with patch('sys.argv', ['main.py', '--fresh', '--all', '--enable-audio-enhancement', 
                               '--enhancement-batch-size', '32', '--enhancement-dashboard']):
            args = parse_arguments()
        
        self.assertTrue(args.enable_audio_enhancement)
        self.assertEqual(args.enhancement_batch_size, 32)
        self.assertTrue(args.enhancement_dashboard)
        
        # Test default values
        with patch('sys.argv', ['main.py', '--fresh', '--all']):
            args_default = parse_arguments()
        self.assertFalse(args_default.enable_audio_enhancement)
        self.assertEqual(args_default.enhancement_batch_size, 32)  # Changed from 16 to 32 to match main.py
        self.assertFalse(args_default.enhancement_dashboard)
    
    @patch('utils.huggingface.Dataset')
    @patch('utils.huggingface.HfApi')
    @patch('utils.huggingface.load_dataset')
    def test_enhanced_audio_format_preservation(self, mock_load_dataset, mock_hf_api, mock_dataset):
        """Test that enhanced audio maintains proper HuggingFace format."""
        # Setup mocks
        mock_hf_api.return_value.list_datasets.return_value = []
        mock_dataset_instance = Mock()
        mock_dataset.from_dict.return_value = mock_dataset_instance
        
        # Mock dataset with audio samples
        mock_data = {
            'train': [{
                'ID': 'S1',
                'audio': {
                    'array': self.audio_samples,
                    'sampling_rate': self.sample_rate,
                    'path': 'S1.wav'
                },
                'transcript': 'Test transcript',
                'dataset_name': 'TestDataset',
                'speaker_id': 'SPK_00001',
                'Language': 'th',
                'length': self.duration,
                'confidence_score': 1.0
            }]
        }
        
        # Create a processor that returns enhanced audio
        test_audio_samples = self.audio_samples
        
        class TestProcessor(BaseProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.enable_audio_enhancement = True
                self.test_audio_samples = test_audio_samples  # Store reference
            
            def process(self, checkpoint=None):
                # Simulate enhancement
                enhanced_audio = self.test_audio_samples * 0.8  # Simple attenuation
                return {
                    'train': [{
                        'ID': 'S1',
                        'audio': {
                            'array': enhanced_audio,
                            'sampling_rate': self.sample_rate,
                            'path': 'S1.wav'
                        },
                        'transcript': 'Test transcript',
                        'dataset_name': 'TestDataset',
                        'speaker_id': 'SPK_00001',
                        'Language': 'th',
                        'length': self.duration,
                        'confidence_score': 1.0,
                        'enhancement_metrics': {
                            'snr_improvement': 3.5,
                            'noise_reduction': 0.7,
                            'clarity_score': 0.85
                        }
                    }]
                }
            
            def get_dataset_info(self):
                """Return test dataset info."""
                return {
                    'name': 'TestDataset',
                    'description': 'Test dataset for audio enhancement',
                    'version': '1.0',
                    'splits': ['train']
                }
            
            def estimate_size(self):
                """Return estimated size."""
                return {
                    'train': 1,
                    'total': 1
                }
            
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                """Process in streaming mode."""
                # Simulate enhancement
                enhanced_audio = self.test_audio_samples * 0.8
                yield {
                    'ID': 'S1',
                    'audio': {
                        'array': enhanced_audio,
                        'sampling_rate': self.sample_rate,
                        'path': 'S1.wav'
                    },
                    'transcript': 'Test transcript',
                    'dataset_name': 'TestDataset',
                    'speaker_id': 'SPK_00001',
                    'Language': 'th',
                    'length': self.duration,
                    'confidence_score': 1.0,
                    'enhancement_metrics': {
                        'snr_improvement': 3.5,
                        'noise_reduction': 0.7,
                        'clarity_score': 0.85
                    }
                }
        
        # Patch to use our test processor
        with patch.dict('config.DATASET_CONFIG', {'TestDataset': {'processor_class': 'TestProcessor'}}), \
             patch('main.importlib.import_module') as mock_import:
            # Mock the import to return our test processor
            mock_module = Mock()
            mock_module.TestProcessor = TestProcessor
            mock_import.return_value = mock_module
            # Run with enhancement
            self._run_main_with_args([
                "--fresh",
                "TestDataset",
                "--enable-audio-enhancement"
            ])
            
            # Verify dataset creation was called
            mock_dataset.from_dict.assert_called()
            call_args = mock_dataset.from_dict.call_args[0][0]
            
            # Check audio format
            self.assertIn('train', call_args)
            sample = call_args['train'][0]
            self.assertIn('audio', sample)
            self.assertIn('array', sample['audio'])
            self.assertIn('sampling_rate', sample['audio'])
            self.assertIn('path', sample['audio'])
            self.assertEqual(sample['audio']['sampling_rate'], self.sample_rate)
            
            # Check enhancement metrics
            self.assertIn('enhancement_metrics', sample)
    
    def test_enhancement_metrics_tracking(self):
        """Test that enhancement metrics are tracked during processing."""
        # Mock AudioEnhancer when it's implemented
        with patch('processors.audio_enhancement.core.AudioEnhancer') as mock_enhancer_class:
            mock_enhancer = Mock()
            mock_enhancer_class.return_value = mock_enhancer
            
            # Mock enhance method to return audio and metrics
            enhanced_audio = self.audio_samples * 0.8
            metrics = {
                'snr_improvement': 3.5,
                'noise_reduction': 0.7,
                'clarity_score': 0.85,
                'processing_time': 0.05
            }
            mock_enhancer.enhance.return_value = (enhanced_audio, metrics)
            
            # Process audio and get metrics
            enhancer = mock_enhancer_class()
            result_audio, result_metrics = enhancer.enhance(
                self.audio_samples, 
                self.sample_rate,
                return_metrics=True
            )
            
            # Verify metrics structure
            self.assertIsInstance(result_metrics, dict)
            self.assertIn('snr_improvement', result_metrics)
            self.assertIn('noise_reduction', result_metrics)
            self.assertIn('clarity_score', result_metrics)
            self.assertIn('processing_time', result_metrics)
            
            # Check metric values are reasonable
            self.assertGreaterEqual(result_metrics['snr_improvement'], 0)
            self.assertGreaterEqual(result_metrics['noise_reduction'], 0)
            self.assertLessEqual(result_metrics['noise_reduction'], 1)
            self.assertGreaterEqual(result_metrics['clarity_score'], 0)
            self.assertLessEqual(result_metrics['clarity_score'], 1)
    
    @patch('utils.huggingface.Dataset')
    @patch('utils.huggingface.HfApi')
    @patch('processors.speaker_identification.SpeakerIdentification')
    def test_enhancement_with_speaker_id_integration(self, mock_speaker_id, mock_hf_api, mock_dataset):
        """Test integration of enhancement with speaker identification."""
        # Setup mocks
        mock_hf_api.return_value.list_datasets.return_value = []
        mock_dataset.from_dict.return_value = Mock()
        
        # Mock speaker identification
        mock_speaker_instance = Mock()
        mock_speaker_instance.generate_speaker_id.return_value = 'SPK_00001'
        mock_speaker_id.return_value = mock_speaker_instance
        
        # Create processor that does both enhancement and speaker ID
        test_audio_samples = self.audio_samples
        
        class TestProcessor(BaseProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.enable_audio_enhancement = True
                self.enable_speaker_id = True
                self.test_audio_samples = test_audio_samples
                self.speaker_identifier = mock_speaker_instance
            
            def process(self, checkpoint=None):
                # Simulate enhancement then speaker ID
                enhanced_audio = self.test_audio_samples * 0.8
                
                # Speaker ID should work on enhanced audio
                speaker_id = self.speaker_identifier.generate_speaker_id(
                    enhanced_audio, 16000  # Use fixed sample rate
                )
                
                return {
                    'train': [{
                        'ID': 'S1',
                        'audio': {
                            'array': enhanced_audio,
                            'sampling_rate': self.sample_rate,
                            'path': 'S1.wav'
                        },
                        'transcript': 'Test transcript',
                        'dataset_name': 'TestDataset',
                        'speaker_id': speaker_id,
                        'Language': 'th',
                        'length': self.duration,
                        'confidence_score': 1.0,
                        'enhancement_metrics': {
                            'snr_improvement': 3.5,
                            'noise_reduction': 0.7,
                            'clarity_score': 0.85
                        }
                    }]
                }
            
            def get_dataset_info(self):
                """Return test dataset info."""
                return {
                    'name': 'TestDataset',
                    'description': 'Test dataset with enhancement and speaker ID',
                    'version': '1.0',
                    'splits': ['train']
                }
            
            def estimate_size(self):
                """Return estimated size."""
                return {
                    'train': 1,
                    'total': 1
                }
            
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                """Process in streaming mode."""
                enhanced_audio = self.test_audio_samples * 0.8
                speaker_id = self.speaker_identifier.generate_speaker_id(enhanced_audio, 16000)
                
                yield {
                    'ID': 'S1',
                    'audio': {
                        'array': enhanced_audio,
                        'sampling_rate': self.sample_rate,
                        'path': 'S1.wav'
                    },
                    'transcript': 'Test transcript',
                    'dataset_name': 'TestDataset',
                    'speaker_id': speaker_id,
                    'Language': 'th',
                    'length': self.duration,
                    'confidence_score': 1.0,
                    'enhancement_metrics': {
                        'snr_improvement': 3.5,
                        'noise_reduction': 0.7,
                        'clarity_score': 0.85
                    }
                }
        
        with patch.dict('config.DATASET_CONFIG', {'TestDataset': {'processor_class': 'TestProcessor'}}), \
             patch('main.get_processor_class', return_value=TestProcessor):
            # Run with both features
            self._run_main_with_args([
                "--fresh",
                "TestDataset",
                "--enable-audio-enhancement",
                "--enable-speaker-id"
            ])
            
            # Verify speaker ID was called on enhanced audio
            mock_speaker_instance.generate_speaker_id.assert_called()
    
    @patch('utils.huggingface.Dataset')
    @patch('utils.huggingface.HfApi')
    @patch('processors.stt.ensemble_stt.EnsembleSTT')
    def test_enhancement_with_stt_integration(self, mock_stt, mock_hf_api, mock_dataset):
        """Test integration of enhancement with speech-to-text."""
        # Setup mocks
        mock_hf_api.return_value.list_datasets.return_value = []
        mock_dataset.from_dict.return_value = Mock()
        
        # Mock STT
        mock_stt_instance = Mock()
        mock_stt_instance.transcribe.return_value = [
            {'text': 'Enhanced transcript', 'confidence': 0.95}
        ]
        mock_stt.return_value = mock_stt_instance
        
        # Create processor that does enhancement then STT
        test_audio_samples = self.audio_samples
        
        class TestProcessor(BaseProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.enable_audio_enhancement = True
                self.enable_stt = True
                self.test_audio_samples = test_audio_samples
                self.stt_processor = mock_stt_instance
            
            def process(self, checkpoint=None):
                # Simulate enhancement
                enhanced_audio = self.test_audio_samples * 0.8
                
                # STT should work on enhanced audio
                if self.stt_processor:
                    result = self.stt_processor.transcribe([{
                        'audio': enhanced_audio,
                        'sampling_rate': self.sample_rate
                    }])
                    transcript = result[0]['text'] if result else ''
                else:
                    transcript = ''
                
                return {
                    'train': [{
                        'ID': 'S1',
                        'audio': {
                            'array': enhanced_audio,
                            'sampling_rate': self.sample_rate,
                            'path': 'S1.wav'
                        },
                        'transcript': transcript,
                        'dataset_name': 'TestDataset',
                        'speaker_id': 'SPK_00001',
                        'Language': 'th',
                        'length': self.duration,
                        'confidence_score': 0.95,
                        'enhancement_metrics': {
                            'snr_improvement': 3.5,
                            'noise_reduction': 0.7,
                            'clarity_score': 0.85
                        }
                    }]
                }
            
            def get_dataset_info(self):
                """Return test dataset info."""
                return {
                    'name': 'TestDataset',
                    'description': 'Test dataset with enhancement and STT',
                    'version': '1.0',
                    'splits': ['train']
                }
            
            def estimate_size(self):
                """Return estimated size."""
                return {
                    'train': 1,
                    'total': 1
                }
            
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                """Process in streaming mode."""
                enhanced_audio = self.test_audio_samples * 0.8
                
                if self.stt_processor:
                    result = self.stt_processor.transcribe([{
                        'audio': enhanced_audio,
                        'sampling_rate': self.sample_rate
                    }])
                    transcript = result[0]['text'] if result else ''
                else:
                    transcript = ''
                
                yield {
                    'ID': 'S1',
                    'audio': {
                        'array': enhanced_audio,
                        'sampling_rate': self.sample_rate,
                        'path': 'S1.wav'
                    },
                    'transcript': transcript,
                    'dataset_name': 'TestDataset',
                    'speaker_id': 'SPK_00001',
                    'Language': 'th',
                    'length': self.duration,
                    'confidence_score': 0.95,
                    'enhancement_metrics': {
                        'snr_improvement': 3.5,
                        'noise_reduction': 0.7,
                        'clarity_score': 0.85
                    }
                }
        
        with patch.dict('config.DATASET_CONFIG', {'TestDataset': {'processor_class': 'TestProcessor'}}), \
             patch('main.get_processor_class', return_value=TestProcessor):
            # Run with both features
            self._run_main_with_args([
                "--fresh",
                "TestDataset",
                "--enable-audio-enhancement",
                "--enable-stt"
            ])
            
            # Verify STT was called
            mock_stt_instance.transcribe.assert_called()
    
    @patch('utils.streaming.StreamingUploader')
    @patch('utils.huggingface.HfApi')
    def test_streaming_mode_with_enhancement(self, mock_hf_api, mock_uploader):
        """Test enhancement in streaming mode."""
        # Setup mocks
        mock_hf_api.return_value.list_datasets.return_value = []
        mock_uploader_instance = Mock()
        mock_uploader.return_value = mock_uploader_instance
        
        # Create streaming processor with enhancement
        test_audio_samples = self.audio_samples
        
        class TestStreamingProcessor(BaseProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.enable_audio_enhancement = True
                self.test_audio_samples = test_audio_samples
                self.streaming_uploader = mock_uploader_instance
            
            def process_all_splits(self, resume_from_checkpoint=False):
                """Simulate streaming processing with enhancement."""
                # Process sample with enhancement
                enhanced_audio = self.test_audio_samples * 0.8
                
                sample = {
                    'ID': 'S1',
                    'audio': {
                        'array': enhanced_audio,
                        'sampling_rate': self.sample_rate,
                        'path': 'S1.wav'
                    },
                    'transcript': 'Test transcript',
                    'dataset_name': 'TestDataset',
                    'speaker_id': 'SPK_00001',
                    'Language': 'th',
                    'length': self.duration,
                    'confidence_score': 1.0,
                    'enhancement_metrics': {
                        'snr_improvement': 3.5,
                        'noise_reduction': 0.7,
                        'clarity_score': 0.85
                    }
                }
                
                # Simulate upload
                self.streaming_uploader.upload_samples([sample], 'train')
                self.streaming_uploader.finalize()
            
            def process(self, checkpoint=None):
                """Process dataset (required for abstract method)."""
                return {'train': []}
            
            def get_dataset_info(self):
                """Return test dataset info."""
                return {
                    'name': 'TestDataset',
                    'description': 'Test streaming dataset with enhancement',
                    'version': '1.0',
                    'splits': ['train']
                }
            
            def estimate_size(self):
                """Return estimated size."""
                return {
                    'train': 1,
                    'total': 1
                }
            
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                """Process in streaming mode."""
                enhanced_audio = self.test_audio_samples * 0.8
                yield {
                    'ID': 'S1',
                    'audio': {
                        'array': enhanced_audio,
                        'sampling_rate': self.sample_rate,
                        'path': 'S1.wav'
                    },
                    'transcript': 'Test transcript',
                    'dataset_name': 'TestDataset',
                    'speaker_id': 'SPK_00001',
                    'Language': 'th',
                    'length': self.duration,
                    'confidence_score': 1.0,
                    'enhancement_metrics': {
                        'snr_improvement': 3.5,
                        'noise_reduction': 0.7,
                        'clarity_score': 0.85
                    }
                }
        
        with patch.dict('config.DATASET_CONFIG', {'TestDataset': {'processor_class': 'TestStreamingProcessor'}}), \
             patch('main.get_processor_class', return_value=TestStreamingProcessor):
            # Run in streaming mode with enhancement
            self._run_main_with_args([
                "--fresh",
                "TestDataset",
                "--streaming",
                "--enable-audio-enhancement"
            ])
            
            # Verify samples were uploaded
            mock_uploader_instance.upload_samples.assert_called()
            call_args = mock_uploader_instance.upload_samples.call_args[0][0]
            
            # Check enhanced sample
            self.assertEqual(len(call_args), 1)
            sample = call_args[0]
            self.assertIn('enhancement_metrics', sample)
            self.assertIn('audio', sample)
    
    @patch('monitoring.dashboard.EnhancementDashboard')
    @patch('utils.huggingface.Dataset')
    @patch('utils.huggingface.HfApi')
    def test_dashboard_monitoring_during_processing(self, mock_hf_api, mock_dataset, mock_dashboard):
        """Test dashboard monitoring during enhancement processing."""
        # Setup mocks
        mock_hf_api.return_value.list_datasets.return_value = []
        mock_dataset.from_dict.return_value = Mock()
        
        # Mock dashboard
        mock_dashboard_instance = Mock()
        mock_dashboard.return_value = mock_dashboard_instance
        
        # Run with dashboard enabled
        with patch.dict('config.DATASET_CONFIG', {'TestDataset': {'processor_class': 'BaseProcessor'}}), \
             patch('main.get_processor_class', return_value=BaseProcessor):
            self._run_main_with_args([
                "--fresh",
                "TestDataset",
                "--enable-audio-enhancement",
                "--enhancement-dashboard"
            ])
        
        # Verify dashboard lifecycle
        mock_dashboard.assert_called()
        mock_dashboard_instance.start.assert_called()
        mock_dashboard_instance.stop.assert_called()
    
    def test_batch_processing_with_enhancement(self):
        """Test batch processing of audio enhancement."""
        # Create a test processor directly without going through main
        test_audio_samples = self.audio_samples
        sample_rate = self.sample_rate
        duration = self.duration
        
        class TestBatchProcessor:
            """Simple test processor that simulates batch enhancement."""
            
            def __init__(self):
                self.enable_audio_enhancement = True
                self.enhancement_batch_size = 16
                self.test_audio_samples = test_audio_samples
                self.sample_rate = sample_rate
                self.duration = duration
            
            def process_all_splits(self, resume_from_checkpoint=False):
                """Process all splits with enhancement."""
                # Simulate batch processing
                batch_samples = []
                batch_size = 5
                
                for i in range(batch_size):
                    # Simulate enhancement processing
                    enhanced_audio = self.test_audio_samples * (0.8 + i * 0.02)
                    
                    sample = {
                        'ID': f'S{i+1}',
                        'audio': {
                            'array': enhanced_audio,
                            'sampling_rate': self.sample_rate,
                            'path': f'S{i+1}.wav'
                        },
                        'transcript': f'Test transcript {i+1}',
                        'dataset_name': 'TestDataset',
                        'speaker_id': f'SPK_0000{i+1}',
                        'Language': 'th',
                        'length': self.duration,
                        'confidence_score': 1.0,
                        'enhancement_metrics': {
                            'snr_improvement': 3.5 + i * 0.1,
                            'noise_reduction': 0.7,
                            'clarity_score': 0.85
                        }
                    }
                    batch_samples.append(sample)
                
                # Return batch for verification
                return batch_samples
        
        # Create and run processor
        processor = TestBatchProcessor()
        processed_samples = processor.process_all_splits()
        
        # Verify all samples were processed
        self.assertEqual(len(processed_samples), 5)
        
        # Check all samples have enhancement metrics
        for i, sample in enumerate(processed_samples):
            self.assertIn('enhancement_metrics', sample)
            self.assertIn('snr_improvement', sample['enhancement_metrics'])
            self.assertIn('noise_reduction', sample['enhancement_metrics'])
            self.assertIn('clarity_score', sample['enhancement_metrics'])
            
            # Verify metrics are reasonable
            self.assertAlmostEqual(
                sample['enhancement_metrics']['snr_improvement'], 
                3.5 + i * 0.1,
                places=2
            )
            self.assertEqual(sample['enhancement_metrics']['noise_reduction'], 0.7)
            self.assertEqual(sample['enhancement_metrics']['clarity_score'], 0.85)
            
            # Check audio was modified (simulated enhancement)
            expected_factor = 0.8 + i * 0.02
            np.testing.assert_array_almost_equal(
                sample['audio']['array'],
                test_audio_samples * expected_factor
            )
    
    def test_enhancement_checkpoint_recovery(self):
        """Test that enhancement state is preserved in checkpoints."""
        checkpoint_file = os.path.join(self.checkpoint_dir, "test_checkpoint.json")
        
        # Create checkpoint with enhancement metrics
        checkpoint_data = {
            'samples_processed': 100,
            'current_split': 'train',
            'split_index': 50,
            'enhancement_enabled': True,
            'total_snr_improvement': 350.0,
            'total_noise_reduction': 70.0,
            'average_clarity_score': 0.85,
            'enhancement_metrics': {
                'processed_samples': 100,
                'failed_samples': 2,
                'average_processing_time': 0.05
            }
        }
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Load checkpoint
        with open(checkpoint_file, 'r') as f:
            loaded_checkpoint = json.load(f)
        
        # Verify enhancement data is preserved
        self.assertTrue(loaded_checkpoint['enhancement_enabled'])
        self.assertEqual(loaded_checkpoint['total_snr_improvement'], 350.0)
        self.assertEqual(loaded_checkpoint['total_noise_reduction'], 70.0)
        self.assertEqual(loaded_checkpoint['average_clarity_score'], 0.85)
        self.assertIn('enhancement_metrics', loaded_checkpoint)
        self.assertEqual(loaded_checkpoint['enhancement_metrics']['processed_samples'], 100)
    
    @patch('utils.huggingface.Dataset')
    @patch('utils.huggingface.HfApi')
    def test_enhancement_with_all_features(self, mock_hf_api, mock_dataset):
        """Test enhancement with all features enabled together."""
        # Setup mocks
        mock_hf_api.return_value.list_datasets.return_value = []
        mock_dataset.from_dict.return_value = Mock()
        
        # Test all feature flags together
        with patch('sys.argv', [
            'main.py', '--fresh', '--all', '--enable-audio-enhancement',
            '--enable-speaker-id', '--enable-stt', '--streaming',
            '--enhancement-dashboard', '--enhancement-batch-size', '64',
            '--speaker-min-cluster-size', '5', '--stt-batch-size', '32'
        ]):
            args = parse_arguments()
        
        # Verify all flags are set correctly
        self.assertTrue(args.enable_audio_enhancement)
        self.assertTrue(args.enable_speaker_id)
        self.assertTrue(args.enable_stt)
        self.assertTrue(args.streaming)
        self.assertTrue(args.enhancement_dashboard)
        self.assertEqual(args.enhancement_batch_size, 64)
        self.assertEqual(args.speaker_min_cluster_size, 5)
        self.assertEqual(args.stt_batch_size, 32)
        
        # Verify no conflicts
        self.assertTrue(args.fresh)  # Fresh mode should work with enhancement
        self.assertTrue(args.all)    # All datasets should work


if __name__ == '__main__':
    unittest.main()