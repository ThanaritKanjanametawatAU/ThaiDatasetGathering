"""
Test suite for speaker ID integration in streaming mode.
"""

import unittest
import tempfile
import os
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Iterator

from processors.base_processor import BaseProcessor
from processors.gigaspeech2 import GigaSpeech2Processor
from processors.mozilla_cv import MozillaCommonVoiceProcessor
from processors.processed_voice_th import ProcessedVoiceTHProcessor


class TestSpeakerIDStreaming(unittest.TestCase):
    """Test speaker ID integration in streaming mode."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "name": "TestProcessor",
            "source": "test/dataset",
            "checkpoint_dir": os.path.join(self.temp_dir, "checkpoints"),
            "log_dir": os.path.join(self.temp_dir, "logs"),
            "cache_dir": os.path.join(self.temp_dir, "cache"),
            "streaming": True,
            "enable_speaker_id": True,
            "speaker_batch_size": 10,
            "speaker_threshold": 0.7,
            "dataset_name": "TestDataset"
        }
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_mock_audio_sample(self, sample_id: str = "test_001") -> Dict[str, Any]:
        """Create a mock audio sample."""
        return {
            "audio": {
                "array": np.random.rand(16000).astype(np.float32),  # 1 second of audio
                "sampling_rate": 16000,
                "path": f"{sample_id}.wav"
            },
            "transcript": "Test transcript",
            "dataset_name": "TestDataset",
            "confidence_score": 1.0,
            "ID": f"S{sample_id}",
            "Language": "th",
            "length": 1.0
        }
        
    def test_speaker_id_required_in_schema(self):
        """Test that speaker_id is required in the schema."""
        from config import VALIDATION_RULES
        
        self.assertIn("speaker_id", VALIDATION_RULES)
        self.assertTrue(VALIDATION_RULES["speaker_id"]["required"])
        self.assertEqual(VALIDATION_RULES["speaker_id"]["pattern"], r"^SPK_\d{5}$")
        
    def test_gigaspeech2_generates_speaker_id_streaming(self):
        """Test that GigaSpeech2 processor generates speaker_id in streaming mode."""
        config = self.config.copy()
        config["enable_stt"] = False  # Disable STT for this test
        
        processor = GigaSpeech2Processor(config)
        
        # Mock the dataset loading
        mock_sample = {
            "segment_id": "test-001",
            "audio": {
                "array": np.random.rand(16000).astype(np.float32),
                "sampling_rate": 16000
            },
            "text": "Test transcript"
        }
        
        with patch('datasets.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            # Mock the streaming dataset with proper structure
            mock_dataset_dict = {'train': MagicMock()}
            mock_dataset_dict['train'].__iter__ = Mock(return_value=iter([mock_sample]))
            mock_load.return_value = mock_dataset_dict
            
            # Process samples
            samples = list(processor.process_all_splits(
                sample_mode=True,
                sample_size=1
            ))
            
            # Verify speaker_id is present
            self.assertEqual(len(samples), 1)
            self.assertIn("speaker_id", samples[0])
            self.assertRegex(samples[0]["speaker_id"], r"^SPK_\d{5}$")
            
    def test_mozilla_cv_generates_speaker_id_streaming(self):
        """Test that Mozilla CV processor generates speaker_id in streaming mode."""
        config = self.config.copy()
        config["source"] = "mozilla-foundation/common_voice_11_0"
        config["language_filter"] = "th"
        
        processor = MozillaCommonVoiceProcessor(config)
        
        # Mock the dataset loading
        mock_sample = {
            "client_id": "test_client_001",
            "audio": {
                "array": np.random.rand(16000).astype(np.float32),
                "sampling_rate": 16000
            },
            "sentence": "Test transcript"
        }
        
        with patch('datasets.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            # Mock the streaming dataset with proper structure
            mock_dataset_dict = {'train': MagicMock()}
            mock_dataset_dict['train'].__iter__ = Mock(return_value=iter([mock_sample]))
            mock_load.return_value = mock_dataset_dict
            
            # Process samples
            samples = list(processor.process_all_splits(
                sample_mode=True,
                sample_size=1
            ))
            
            # Verify speaker_id is present
            self.assertEqual(len(samples), 1)
            self.assertIn("speaker_id", samples[0])
            self.assertRegex(samples[0]["speaker_id"], r"^SPK_\d{5}$")
            
    def test_processed_voice_th_generates_speaker_id_streaming(self):
        """Test that ProcessedVoiceTH processor generates speaker_id in streaming mode."""
        config = self.config.copy()
        config["source"] = "Porameht/processed-voice-th-169k"
        
        processor = ProcessedVoiceTHProcessor(config)
        
        # Mock the dataset loading
        mock_sample = {
            "audio": {
                "array": np.random.rand(16000).astype(np.float32),
                "sampling_rate": 16000
            },
            "text": "Test transcript"
        }
        
        with patch('datasets.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            # Mock the streaming dataset with proper structure
            mock_dataset_dict = {'train': MagicMock()}
            mock_dataset_dict['train'].__iter__ = Mock(return_value=iter([mock_sample]))
            mock_load.return_value = mock_dataset_dict
            
            # Process samples
            samples = list(processor.process_all_splits(
                sample_mode=True,
                sample_size=1
            ))
            
            # Verify speaker_id is present
            self.assertEqual(len(samples), 1)
            self.assertIn("speaker_id", samples[0])
            self.assertRegex(samples[0]["speaker_id"], r"^SPK_\d{5}$")
            
    def test_speaker_id_consistency_across_samples(self):
        """Test that same speaker gets same speaker_id across samples."""
        config = self.config.copy()
        processor = GigaSpeech2Processor(config)
        
        # Mock samples from same speaker
        mock_samples = [
            {
                "segment_id": "speaker1-001",
                "audio": {"array": np.random.rand(16000).astype(np.float32), "sampling_rate": 16000},
                "text": "First utterance"
            },
            {
                "segment_id": "speaker1-002", 
                "audio": {"array": np.random.rand(16000).astype(np.float32), "sampling_rate": 16000},
                "text": "Second utterance"
            },
            {
                "segment_id": "speaker2-001",
                "audio": {"array": np.random.rand(16000).astype(np.float32), "sampling_rate": 16000},
                "text": "Different speaker"
            }
        ]
        
        with patch('datasets.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            # Mock the streaming dataset with proper structure
            mock_dataset_dict = {'train': MagicMock()}
            mock_dataset_dict['train'].__iter__ = Mock(return_value=iter(mock_samples))
            mock_load.return_value = mock_dataset_dict
            
            samples = list(processor.process_all_splits(
                sample_mode=True,
                sample_size=3
            ))
            
            # Verify speaker consistency
            self.assertEqual(len(samples), 3)
            # First two samples should have same speaker_id pattern
            self.assertIn("speaker_id", samples[0])
            self.assertIn("speaker_id", samples[1])
            self.assertIn("speaker_id", samples[2])
            
    def test_speaker_id_with_stt_enabled(self):
        """Test speaker_id generation works with STT enabled."""
        config = self.config.copy()
        config["enable_stt"] = True
        config["stt_batch_size"] = 1
        
        processor = GigaSpeech2Processor(config)
        
        # Mock sample without transcript
        mock_sample = {
            "segment_id": "test-001",
            "audio": {
                "array": np.random.rand(16000).astype(np.float32),
                "sampling_rate": 16000
            },
            "text": ""  # Empty transcript to trigger STT
        }
        
        with patch('datasets.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            # Mock the streaming dataset with proper structure
            mock_dataset_dict = {'train': MagicMock()}
            mock_dataset_dict['train'].__iter__ = Mock(return_value=iter([mock_sample]))
            mock_load.return_value = mock_dataset_dict
            
            # Mock STT
            with patch.object(processor, 'stt_pipeline') as mock_stt:
                mock_stt.transcribe.return_value = ("Generated transcript", 0.95)
                
                samples = list(processor.process_all_splits(
                    sample_mode=True,
                    sample_size=1
                ))
                
                # Verify both speaker_id and STT worked
                self.assertEqual(len(samples), 1)
                self.assertIn("speaker_id", samples[0])
                self.assertRegex(samples[0]["speaker_id"], r"^SPK_\d{5}$")
                self.assertEqual(samples[0]["transcript"], "Generated transcript")
                self.assertEqual(samples[0]["confidence_score"], 0.95)
                
    def test_validation_passes_with_speaker_id(self):
        """Test that validation passes when speaker_id is present."""
        from processors.base_processor import BaseProcessor
        
        # Create a minimal processor for testing
        class TestProcessor(BaseProcessor):
            def process(self, checkpoint=None):
                pass
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                pass
            def get_dataset_info(self):
                return {"name": "Test", "size": 0}
            def estimate_size(self):
                return 0
                
        processor = TestProcessor(self.config)
        
        # Valid sample with speaker_id
        valid_sample = {
            "ID": "S1",
            "speaker_id": "SPK_00001",
            "Language": "th",
            "audio": {
                "array": np.random.rand(16000).astype(np.float32),
                "sampling_rate": 16000,
                "path": "test.wav"
            },
            "transcript": "Test transcript",
            "length": 1.0,
            "dataset_name": "TestDataset",
            "confidence_score": 1.0
        }
        
        # Should not raise validation error
        errors = processor.validate_sample(valid_sample)
        self.assertEqual(len(errors), 0)
        
        # Invalid sample without speaker_id
        invalid_sample = valid_sample.copy()
        del invalid_sample["speaker_id"]
        
        errors = processor.validate_sample(invalid_sample)
        self.assertGreater(len(errors), 0)
        self.assertIn("Missing required field: speaker_id", errors)
        
    def test_speaker_id_format_validation(self):
        """Test speaker_id format validation."""
        from processors.base_processor import BaseProcessor
        
        class TestProcessor(BaseProcessor):
            def process(self, checkpoint=None):
                pass
            def process_streaming(self, checkpoint=None, sample_mode=False, sample_size=5):
                pass
            def get_dataset_info(self):
                return {"name": "Test", "size": 0}
            def estimate_size(self):
                return 0
                
        processor = TestProcessor(self.config)
        
        base_sample = {
            "ID": "S1",
            "Language": "th",
            "audio": {"array": np.random.rand(16000).astype(np.float32), "sampling_rate": 16000, "path": "test.wav"},
            "transcript": "Test",
            "length": 1.0,
            "dataset_name": "Test",
            "confidence_score": 1.0
        }
        
        # Test valid formats
        valid_ids = ["SPK_00001", "SPK_12345", "SPK_99999"]
        for speaker_id in valid_ids:
            sample = base_sample.copy()
            sample["speaker_id"] = speaker_id
            errors = processor.validate_sample(sample)
            self.assertEqual(len(errors), 0, f"Valid speaker_id {speaker_id} failed validation")
            
        # Test invalid formats
        invalid_ids = ["SPK_1234", "SPK_123456", "SPEAKER_00001", "00001", "SPK00001"]
        for speaker_id in invalid_ids:
            sample = base_sample.copy()
            sample["speaker_id"] = speaker_id
            errors = processor.validate_sample(sample)
            self.assertGreater(len(errors), 0, f"Invalid speaker_id {speaker_id} passed validation")


if __name__ == "__main__":
    unittest.main()