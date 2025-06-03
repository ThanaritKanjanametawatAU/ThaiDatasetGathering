#!/usr/bin/env python3
"""
Comprehensive tests to verify speaker ID functionality after reverting secondary speaker removal changes.
Tests ensure S1-S8 and S10 cluster as same speaker, S9 is different.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from datasets import Dataset, Audio
import pandas as pd

# Import the processors and utilities
from processors.gigaspeech2 import GigaSpeech2Processor
from processors.speaker_identification import SpeakerIdentification
# Remove unused import

class TestSpeakerIDRevertVerification:
    """Test suite to verify speaker ID works correctly after reverting changes"""
    
    @pytest.fixture
    def speaker_config(self, tmp_path):
        """Speaker identification config"""
        return {
            'model': 'pyannote/embedding',
            'clustering': {
                'algorithm': 'hdbscan',
                'min_cluster_size': 5,
                'min_samples': 3,
                'metric': 'cosine',
                'similarity_threshold': 0.7  # Original working threshold
            },
            'batch_size': 10,
            'store_embeddings': False,
            'embedding_path': str(tmp_path / 'embeddings.h5'),
            'model_path': str(tmp_path / 'speaker_model.json'),
            'fresh': True
        }
    
    @pytest.fixture
    def test_audio_samples(self):
        """Create 10 test audio samples with known clustering pattern"""
        samples = []
        sample_rate = 16000
        duration = 2.0
        
        # S1-S8 and S10: Similar speakers (low pitch, similar timbre)
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 9]:
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Base frequency around 150Hz with small variations
            freq = 150 + np.random.uniform(-5, 5)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            # Add harmonics
            audio += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
            audio += 0.125 * np.sin(2 * np.pi * freq * 3 * t)
            # Add noise
            audio += np.random.normal(0, 0.01, len(audio))
            
            samples.append({
                'ID': f'S{i+1 if i < 8 else 10}',
                'audio': {
                    'array': audio.astype(np.float32),
                    'sampling_rate': sample_rate
                }
            })
        
        # S9: Different speaker (higher pitch, different timbre)
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 250  # Much higher pitch
        audio = 0.4 * np.sin(2 * np.pi * freq * t)
        audio += 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        audio += np.random.normal(0, 0.01, len(audio))
        
        samples.insert(8, {
            'ID': 'S9',
            'audio': {
                'array': audio.astype(np.float32),
                'sampling_rate': sample_rate
            }
        })
        
        return samples
    
    def test_speaker_identification_clustering(self, speaker_config, test_audio_samples):
        """Test that speaker identification clusters S1-S8,S10 together and S9 separately"""
        # Initialize speaker identification
        speaker_id = SpeakerIdentification(speaker_config)
        
        # Process samples
        speaker_ids = speaker_id.process_batch(test_audio_samples)
        
        # Verify clustering
        s1_to_s8_and_s10_ids = [speaker_ids[i] for i in [0,1,2,3,4,5,6,7,9]]
        s9_id = speaker_ids[8]
        
        # All S1-S8 and S10 should have the same speaker ID
        assert len(set(s1_to_s8_and_s10_ids)) == 1, \
            f"S1-S8 and S10 should have same speaker ID, got: {set(s1_to_s8_and_s10_ids)}"
        
        # S9 should have a different speaker ID
        assert s9_id not in s1_to_s8_and_s10_ids, \
            f"S9 should have different speaker ID, got S9={s9_id}, others={s1_to_s8_and_s10_ids[0]}"
        
        # Should have exactly 2 unique speakers
        all_speaker_ids = speaker_ids
        assert len(set(all_speaker_ids)) == 2, \
            f"Should have exactly 2 speakers, got {len(set(all_speaker_ids))}: {set(all_speaker_ids)}"
    
    def test_speaker_id_continuation_on_resume(self, speaker_config, test_audio_samples, tmp_path):
        """Test that speaker IDs continue correctly when resuming"""
        # First batch
        speaker_id = SpeakerIdentification(speaker_config)
        first_batch_ids = speaker_id.process_batch(test_audio_samples[:5])
        
        # Save state
        speaker_id.save_model()
        last_speaker_counter = speaker_id.speaker_counter
        
        # Create new instance (simulating resume)
        resume_config = speaker_config.copy()
        resume_config['fresh'] = False
        speaker_id_resumed = SpeakerIdentification(resume_config)
        
        # Verify speaker counter continues
        assert speaker_id_resumed.speaker_counter == last_speaker_counter, \
            f"Speaker counter should continue from {last_speaker_counter}, got {speaker_id_resumed.speaker_counter}"
        
        # Process more samples
        second_batch_ids = speaker_id_resumed.process_batch(test_audio_samples[5:])
        
        # Verify IDs are continuous
        all_ids = first_batch_ids + second_batch_ids
        unique_ids = sorted(set(all_ids))
        
        # Extract numeric parts and verify continuity
        numeric_ids = []
        for sid in unique_ids:
            if sid.startswith('SPK_'):
                numeric_ids.append(int(sid.split('_')[1]))
        
        numeric_ids.sort()
        for i in range(1, len(numeric_ids)):
            assert numeric_ids[i] - numeric_ids[i-1] == 1, \
                f"Speaker IDs should be continuous, gap between {numeric_ids[i-1]} and {numeric_ids[i]}"
    
    def test_main_sh_execution(self, tmp_path):
        """Test that main.sh runs without errors"""
        import subprocess
        
        # Run main.sh with test parameters
        env = os.environ.copy()
        env['HF_TOKEN'] = env.get('HF_TOKEN', 'test_token')
        
        result = subprocess.run(
            ['./main.sh', '--append'],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            env=env
        )
        
        # Should not have critical errors
        assert 'ModuleNotFoundError' not in result.stderr, \
            "main.sh should not have module import errors"
        assert 'omegaconf.base' not in result.stderr, \
            "Should not have omegaconf.base errors"
        assert result.returncode != 1, \
            f"main.sh should not exit with error code 1. stderr: {result.stderr}"
    
    def test_huggingface_dataset_verification(self):
        """Test loading and verifying the Huggingface dataset"""
        from datasets import load_dataset
        
        try:
            # Load a small sample from the dataset
            dataset = load_dataset(
                "Thanarit/Thai-Voice-10000000",
                split="train",
                streaming=True
            )
            
            # Get first 10 samples
            samples = []
            for i, sample in enumerate(dataset):
                if i >= 10:
                    break
                samples.append(sample)
            
            # Verify required fields exist
            required_fields = ['ID', 'speaker_id', 'audio', 'transcription', 'dataset']
            for sample in samples:
                for field in required_fields:
                    assert field in sample, f"Sample missing required field: {field}"
                
                # Verify audio is dict with array and sampling_rate
                assert isinstance(sample['audio'], dict), "Audio should be a dict"
                assert 'array' in sample['audio'], "Audio should have 'array' field"
                assert 'sampling_rate' in sample['audio'], "Audio should have 'sampling_rate' field"
                
                # Verify speaker_id format
                assert sample['speaker_id'].startswith('SPK_'), \
                    f"Speaker ID should start with SPK_, got: {sample['speaker_id']}"
                
                # Verify ID format
                assert sample['ID'].startswith('S'), \
                    f"ID should start with S, got: {sample['ID']}"
            
            print(f"✓ Successfully loaded and verified {len(samples)} samples from Huggingface")
            
        except Exception as e:
            # If dataset doesn't exist yet, that's OK for testing
            if "404" in str(e) or "Not Found" in str(e):
                pytest.skip("Dataset not yet available on Huggingface")
            else:
                raise
    
    def test_pyannote_model_loading(self, speaker_config):
        """Test that pyannote models can be loaded without omegaconf errors"""
        try:
            from pyannote.audio import Model, Inference
            
            # Should be able to load the model
            model = Model.from_pretrained('pyannote/embedding')
            assert model is not None, "Should be able to load pyannote embedding model"
            
            # Should be able to create inference
            inference = Inference(model)
            assert inference is not None, "Should be able to create pyannote inference"
            
            print("✓ Pyannote models loaded successfully without omegaconf errors")
            
        except ModuleNotFoundError as e:
            if 'omegaconf.base' in str(e):
                pytest.fail(f"Omegaconf error when loading pyannote: {e}")
            else:
                raise
    
    def test_no_secondary_speaker_removal_in_config(self):
        """Test that secondary speaker removal is not enabled by default"""
        from processors.audio_enhancement.core import AudioEnhancer
        
        # Check the class-level ENHANCEMENT_LEVELS
        level_config = AudioEnhancer.ENHANCEMENT_LEVELS.get('moderate', {})
        
        # These flags should not exist or be False
        assert not level_config.get('check_secondary_speaker', False), \
            "check_secondary_speaker should be False in moderate config"
        assert not level_config.get('use_speaker_separation', False), \
            "use_speaker_separation should be False in moderate config"
        
        # Also check ultra_aggressive doesn't have them
        ultra_config = AudioEnhancer.ENHANCEMENT_LEVELS.get('ultra_aggressive', {})
        assert not ultra_config.get('check_secondary_speaker', False), \
            "check_secondary_speaker should be False in ultra_aggressive config"
        assert not ultra_config.get('use_speaker_separation', False), \
            "use_speaker_separation should be False in ultra_aggressive config"
    
    def test_processor_imports_work(self):
        """Test that all processor imports work without errors"""
        try:
            from processors.gigaspeech2 import GigaSpeech2Processor
            from processors.mozilla_cv import MozillaCommonVoiceProcessor
            from processors.speaker_identification import SpeakerIdentification
            from processors.audio_enhancement import AudioEnhancer
            
            print("✓ All processor imports successful")
            
        except ImportError as e:
            pytest.fail(f"Import error: {e}")
        except ModuleNotFoundError as e:
            if 'omegaconf' in str(e):
                pytest.fail(f"Omegaconf-related import error: {e}")
            else:
                raise
    
    def test_resume_functionality_preserves_speaker_model(self, speaker_config, tmp_path):
        """Test that resume functionality preserves the speaker model state"""
        # Initialize and process first batch
        speaker_id = SpeakerIdentification(speaker_config)
        
        # Create some test embeddings
        test_embeddings = np.random.randn(5, 512).astype(np.float32)
        test_speaker_ids = ['SPK_00001', 'SPK_00001', 'SPK_00001', 'SPK_00002', 'SPK_00002']
        
        # Store embeddings
        if speaker_config.get('store_embeddings'):
            for emb, spk_id in zip(test_embeddings, test_speaker_ids):
                speaker_id.embeddings.append(emb)
                speaker_id.speaker_ids.append(spk_id)
        
        # Set speaker counter
        speaker_id.speaker_counter = 3  # Next should be SPK_00003
        
        # Save model
        speaker_id.save_model()
        
        # Create new instance with resume
        resume_config = speaker_config.copy()
        resume_config['fresh'] = False
        speaker_id_resumed = SpeakerIdentification(resume_config)
        
        # Verify state is preserved
        assert speaker_id_resumed.speaker_counter == 3, \
            f"Speaker counter should be 3, got {speaker_id_resumed.speaker_counter}"
        
        # If storing embeddings, verify they're loaded
        if speaker_config.get('store_embeddings') and hasattr(speaker_id_resumed, 'embeddings'):
            assert len(speaker_id_resumed.embeddings) == 5, \
                f"Should have 5 embeddings, got {len(speaker_id_resumed.embeddings)}"
            assert len(speaker_id_resumed.speaker_ids) == 5, \
                f"Should have 5 speaker IDs, got {len(speaker_id_resumed.speaker_ids)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])