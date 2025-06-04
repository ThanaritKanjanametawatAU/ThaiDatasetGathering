#!/usr/bin/env python3
"""
Test suite for speaker ID continuity across resume operations.

This test ensures that speaker IDs continue from the last used ID when resuming,
preventing the critical issue of different speakers being assigned the same ID.
"""

import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from processors.speaker_identification import SpeakerIdentification
from processors.base_processor import BaseProcessor


class TestSpeakerIDResumeContinuity(unittest.TestCase):
    """Test speaker ID continuity when resuming from checkpoint."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Mock audio data
        self.sample_rate = 16000
        self.audio_duration = 2.0  # seconds
        self.num_samples = int(self.sample_rate * self.audio_duration)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def create_test_samples(self, num_samples, start_id=1):
        """Create test audio samples."""
        samples = []
        for i in range(num_samples):
            # Create slightly different audio for each sample
            audio = np.random.randn(self.num_samples).astype(np.float32) * 0.1
            audio += np.sin(2 * np.pi * (100 + i * 50) * np.arange(self.num_samples) / self.sample_rate) * 0.5
            
            samples.append({
                'ID': f'S{start_id + i}',
                'audio': {
                    'array': audio,
                    'sampling_rate': self.sample_rate,
                    'path': f'S{start_id + i}.wav'
                },
                'transcript': f'Test transcript {i}',
                'length': self.audio_duration
            })
        return samples
        
    def test_speaker_id_starts_from_1_in_fresh_mode(self):
        """Test that speaker IDs start from SPK_00001 in fresh mode."""
        config = {
            'model': 'pyannote/embedding',
            'model_path': os.path.join(self.checkpoint_dir, 'speaker_model.json'),
            'fresh': True  # Fresh mode should reset counter
        }
        
        speaker_id = SpeakerIdentification(config)
        samples = self.create_test_samples(5)
        
        # Process samples
        speaker_ids = speaker_id.process_batch(samples)
        
        # Verify first speaker ID is SPK_00001
        self.assertEqual(speaker_ids[0], 'SPK_00001')
        
        # Verify counter was saved
        self.assertTrue(os.path.exists(config['model_path']))
        with open(config['model_path'], 'r') as f:
            saved_state = json.load(f)
        self.assertGreaterEqual(saved_state['speaker_counter'], 1)
        
    def test_speaker_id_continues_from_saved_state_on_resume(self):
        """Test that speaker IDs continue from saved state when resuming."""
        model_path = os.path.join(self.checkpoint_dir, 'speaker_model.json')
        
        # First run: Process some samples
        config1 = {
            'model': 'pyannote/embedding',
            'model_path': model_path,
            'fresh': True
        }
        
        speaker_id1 = SpeakerIdentification(config1)
        samples1 = self.create_test_samples(10)
        speaker_ids1 = speaker_id1.process_batch(samples1)
        
        # Get the highest speaker counter from first run
        with open(model_path, 'r') as f:
            saved_state1 = json.load(f)
        last_counter = saved_state1.get('speaker_counter', 0)
        
        # Clean up first instance
        speaker_id1.cleanup()
        
        # Second run: Resume (fresh=False should load saved state)
        config2 = {
            'model': 'pyannote/embedding', 
            'model_path': model_path,
            'fresh': False  # Resume mode
        }
        
        speaker_id2 = SpeakerIdentification(config2)
        samples2 = self.create_test_samples(5, start_id=11)
        speaker_ids2 = speaker_id2.process_batch(samples2)
        
        # Verify new speaker IDs start after the last used ID
        new_speaker_numbers = []
        for spk_id in speaker_ids2:
            if spk_id.startswith('SPK_'):
                num = int(spk_id.split('_')[1])
                new_speaker_numbers.append(num)
        
        # All new speaker IDs should be >= last_counter (continuing from where we left off)
        self.assertTrue(all(num >= last_counter for num in new_speaker_numbers),
                       f"New speaker IDs {new_speaker_numbers} should all be >= {last_counter}")
        
    def test_speaker_id_does_not_reset_between_datasets_on_resume(self):
        """Test that speaker ID counter doesn't reset between datasets when resuming."""
        model_path = os.path.join(self.checkpoint_dir, 'speaker_model.json')
        
        # Initial run with dataset 1
        config = {
            'model': 'pyannote/embedding',
            'model_path': model_path,
            'fresh': True
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Process first dataset
        dataset1_samples = self.create_test_samples(5)
        dataset1_ids = speaker_id.process_batch(dataset1_samples)
        
        # Reset for new dataset (but keep counter)
        speaker_id.reset_for_new_dataset(reset_counter=False)
        
        # Process second dataset
        dataset2_samples = self.create_test_samples(5, start_id=6)
        dataset2_ids = speaker_id.process_batch(dataset2_samples)
        
        # Get final counter
        with open(model_path, 'r') as f:
            saved_state = json.load(f)
        final_counter = saved_state['speaker_counter']
        
        speaker_id.cleanup()
        
        # Now simulate resume - counter should continue from final_counter
        config_resume = {
            'model': 'pyannote/embedding',
            'model_path': model_path,
            'fresh': False  # Resume mode
        }
        
        speaker_id_resume = SpeakerIdentification(config_resume)
        
        # Process new samples
        resume_samples = self.create_test_samples(3, start_id=11)
        resume_ids = speaker_id_resume.process_batch(resume_samples)
        
        # Extract speaker numbers
        resume_numbers = [int(spk_id.split('_')[1]) for spk_id in resume_ids if spk_id.startswith('SPK_')]
        
        # All resumed IDs should be >= final_counter
        self.assertTrue(all(num >= final_counter for num in resume_numbers),
                       f"Resumed IDs {resume_numbers} should all be >= {final_counter}")
        
    def test_fresh_flag_false_loads_existing_speaker_model(self):
        """Test that fresh=False loads the existing speaker model and counter."""
        model_path = os.path.join(self.checkpoint_dir, 'speaker_model.json')
        
        # Create a saved state with specific counter
        saved_counter = 42
        saved_state = {
            'speaker_counter': saved_counter,
            'existing_clusters': {},
            'cluster_centroids': None
        }
        with open(model_path, 'w') as f:
            json.dump(saved_state, f)
        
        # Initialize with fresh=False
        config = {
            'model': 'pyannote/embedding',
            'model_path': model_path,
            'fresh': False
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Process samples
        samples = self.create_test_samples(3)
        speaker_ids = speaker_id.process_batch(samples)
        
        # New IDs should start from saved counter and increment
        speaker_numbers = [int(spk_id.split('_')[1]) for spk_id in speaker_ids if spk_id.startswith('SPK_')]
        # First ID should be saved_counter, then increment from there
        expected_numbers = [saved_counter + i for i in range(len(speaker_numbers))]
        self.assertEqual(speaker_numbers, expected_numbers,
                        f"Expected IDs {expected_numbers}, got {speaker_numbers}")
        
    def test_speaker_id_counter_increments_correctly_across_batches(self):
        """Test that speaker ID counter increments correctly across multiple batches."""
        config = {
            'model': 'pyannote/embedding',
            'model_path': os.path.join(self.checkpoint_dir, 'speaker_model.json'),
            'fresh': True
        }
        
        speaker_id = SpeakerIdentification(config)
        
        all_speaker_ids = []
        
        # Process multiple batches
        for batch_num in range(3):
            samples = self.create_test_samples(4, start_id=batch_num*4+1)
            batch_ids = speaker_id.process_batch(samples)
            all_speaker_ids.extend(batch_ids)
        
        # Extract all speaker numbers
        speaker_numbers = [int(spk_id.split('_')[1]) for spk_id in all_speaker_ids if spk_id.startswith('SPK_')]
        
        # Check for duplicates
        self.assertEqual(len(speaker_numbers), len(set(speaker_numbers)),
                        f"Found duplicate speaker IDs: {speaker_numbers}")
        
        # Verify counter is saved correctly
        with open(config['model_path'], 'r') as f:
            saved_state = json.load(f)
        self.assertGreaterEqual(saved_state['speaker_counter'], max(speaker_numbers))
        
    def test_integration_main_py_resume_preserves_speaker_counter(self):
        """Integration test: Verify main.py preserves speaker counter on resume."""
        # This test verifies the integration between main.py and SpeakerIdentification
        # We'll mock the necessary parts to test the flow
        
        from main import process_streaming_mode
        
        # Create mock args for initial run
        args_initial = Mock()
        args_initial.fresh = True
        args_initial.resume = False
        args_initial.append = False
        args_initial.no_upload = True
        args_initial.enable_speaker_id = True
        args_initial.speaker_model = 'pyannote/embedding'
        args_initial.speaker_batch_size = 5
        args_initial.speaker_threshold = 0.7
        args_initial.speaker_min_cluster_size = 15
        args_initial.speaker_min_samples = 10
        args_initial.speaker_epsilon = 0.3
        args_initial.store_embeddings = False
        args_initial.output = self.test_dir
        args_initial.sample = True
        args_initial.sample_size = 10
        args_initial.verbose = False
        args_initial.enable_audio_enhancement = False
        args_initial.enable_stt = False
        args_initial.no_stt = True
        args_initial.streaming_batch_size = 1000
        args_initial.upload_batch_size = 10
        args_initial.no_standardization = False
        args_initial.sample_rate = 16000
        args_initial.target_db = -20.0
        args_initial.no_volume_norm = False
        args_initial.hf_repo = None
        
        # Create a temporary checkpoint directory
        import config
        original_checkpoint_dir = config.CHECKPOINT_DIR
        config.CHECKPOINT_DIR = self.checkpoint_dir
        
        try:
            # Mock dataset processors to return test samples
            with patch('main.create_processor') as mock_create_processor:
                mock_processor = Mock()
                mock_processor.name = 'TestDataset'
                
                # Create samples that will be processed
                test_samples = self.create_test_samples(10)
                mock_processor.process_all_splits.return_value = iter(test_samples)
                mock_processor.save_streaming_checkpoint = Mock()
                
                mock_create_processor.return_value = mock_processor
                
                # Run initial processing
                with patch('main.read_hf_token', return_value='fake_token'):
                    process_streaming_mode(args_initial, ['TestDataset'])
                
                # Check speaker model was saved
                speaker_model_path = os.path.join(self.checkpoint_dir, 'speaker_model.json')
                self.assertTrue(os.path.exists(speaker_model_path))
                
                with open(speaker_model_path, 'r') as f:
                    initial_state = json.load(f)
                initial_counter = initial_state['speaker_counter']
                
                # Now test resume
                args_resume = Mock()
                args_resume.fresh = True  # Still fresh but with resume flag
                args_resume.resume = True  # This is the key flag
                args_resume.append = False
                args_resume.no_upload = True
                args_resume.enable_speaker_id = True
                args_resume.speaker_model = 'pyannote/embedding'
                args_resume.speaker_batch_size = 5
                args_resume.speaker_threshold = 0.7
                args_resume.speaker_min_cluster_size = 15
                args_resume.speaker_min_samples = 10
                args_resume.speaker_epsilon = 0.3
                args_resume.store_embeddings = False
                args_resume.output = self.test_dir
                args_resume.sample = True
                args_resume.sample_size = 5
                args_resume.verbose = False
                args_resume.enable_audio_enhancement = False
                args_resume.enable_stt = False
                args_resume.no_stt = True
                args_resume.streaming_batch_size = 1000
                args_resume.upload_batch_size = 10
                args_resume.no_standardization = False
                args_resume.sample_rate = 16000
                args_resume.target_db = -20.0
                args_resume.no_volume_norm = False
                args_resume.hf_repo = None
                
                # Create new samples for resume
                resume_samples = self.create_test_samples(5, start_id=11)
                mock_processor.process_all_splits.return_value = iter(resume_samples)
                
                # Run resume processing
                with patch('main.read_hf_token', return_value='fake_token'):
                    with patch('main.get_last_id', return_value=10):
                        process_streaming_mode(args_resume, ['TestDataset'])
                
                # Check speaker counter continued
                with open(speaker_model_path, 'r') as f:
                    resume_state = json.load(f)
                resume_counter = resume_state['speaker_counter']
                
                # Resume counter should be >= initial counter
                self.assertGreaterEqual(resume_counter, initial_counter,
                                      f"Resume counter {resume_counter} should be >= initial {initial_counter}")
                
        finally:
            # Restore original checkpoint dir
            config.CHECKPOINT_DIR = original_checkpoint_dir


if __name__ == '__main__':
    unittest.main()