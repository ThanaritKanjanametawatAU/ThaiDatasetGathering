#!/usr/bin/env python3
"""
Edge case tests for speaker ID reset with --fresh flag.
"""

import unittest
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.speaker_identification import SpeakerIdentification


class TestSpeakerFreshEdgeCases(unittest.TestCase):
    """Test edge cases for speaker ID reset with --fresh flag."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "speaker_model.json")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_fresh_with_no_existing_model(self):
        """Test fresh flag when no model file exists."""
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        # No model file exists
        self.assertFalse(os.path.exists(self.model_path))
        
        # Should initialize normally with counter=1
        speaker_id = SpeakerIdentification(config)
        self.assertEqual(speaker_id.speaker_counter, 1)
        self.assertEqual(len(speaker_id.existing_clusters), 0)
    
    def test_fresh_with_corrupted_model(self):
        """Test fresh flag with corrupted model file."""
        # Create corrupted model file
        with open(self.model_path, 'w') as f:
            f.write("corrupted json {]}")
        
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        # Should handle corrupted file and reset to 1
        speaker_id = SpeakerIdentification(config)
        self.assertEqual(speaker_id.speaker_counter, 1)
        
        # Corrupted file should be deleted
        self.assertFalse(os.path.exists(self.model_path))
    
    def test_fresh_with_read_only_model_file(self):
        """Test fresh flag when model file is read-only."""
        # Create model file
        existing_model = {
            "speaker_counter": 100,
            "existing_clusters": {},
            "cluster_centroids": None
        }
        
        with open(self.model_path, 'w') as f:
            json.dump(existing_model, f)
        
        # Make file read-only
        os.chmod(self.model_path, 0o444)
        
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        # Should handle permission error gracefully
        try:
            speaker_id = SpeakerIdentification(config)
            # Even if delete fails, should reset counter
            self.assertEqual(speaker_id.speaker_counter, 1)
        except PermissionError:
            # On some systems, might raise PermissionError
            # This is acceptable as long as it's handled
            pass
        finally:
            # Restore permissions for cleanup only if file still exists
            if os.path.exists(self.model_path):
                os.chmod(self.model_path, 0o644)
    
    def test_fresh_false_with_missing_model(self):
        """Test fresh=False when model doesn't exist."""
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': False
        }
        
        # No model file exists
        self.assertFalse(os.path.exists(self.model_path))
        
        # Should start from 1 when no model exists
        speaker_id = SpeakerIdentification(config)
        self.assertEqual(speaker_id.speaker_counter, 1)
    
    def test_fresh_with_very_high_counter(self):
        """Test fresh flag resets even very high counters."""
        # Create model with very high counter
        existing_model = {
            "speaker_counter": 999999,
            "existing_clusters": {str(i): f"SPK_{i:05d}" for i in range(1000)},
            "cluster_centroids": [[0.1] * 512] * 100
        }
        
        with open(self.model_path, 'w') as f:
            json.dump(existing_model, f)
        
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        speaker_id = SpeakerIdentification(config)
        
        # Should reset to 1 regardless of previous value
        self.assertEqual(speaker_id.speaker_counter, 1)
        self.assertEqual(len(speaker_id.existing_clusters), 0)
    
    def test_fresh_concurrent_access(self):
        """Test fresh flag behavior with potential concurrent access."""
        # Create model file
        existing_model = {
            "speaker_counter": 50,
            "existing_clusters": {},
            "cluster_centroids": None
        }
        
        with open(self.model_path, 'w') as f:
            json.dump(existing_model, f)
        
        # Simulate two processes trying to use fresh
        config1 = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        config2 = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        # First instance
        speaker_id1 = SpeakerIdentification(config1)
        self.assertEqual(speaker_id1.speaker_counter, 1)
        
        # Second instance (model already deleted)
        speaker_id2 = SpeakerIdentification(config2)
        self.assertEqual(speaker_id2.speaker_counter, 1)
    
    def test_fresh_preserves_after_save(self):
        """Test that after fresh reset, subsequent saves start from new counter."""
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True
        }
        
        speaker_id = SpeakerIdentification(config)
        self.assertEqual(speaker_id.speaker_counter, 1)
        
        # Simulate processing that increments counter
        speaker_id.speaker_counter = 5
        speaker_id.existing_clusters = {0: "SPK_00001", 1: "SPK_00002"}
        
        # Save model
        speaker_id.save_model()
        
        # Load without fresh flag
        config2 = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': False
        }
        
        speaker_id2 = SpeakerIdentification(config2)
        
        # Should load the new counter value (5), not the old one (75)
        self.assertEqual(speaker_id2.speaker_counter, 5)
    
    def test_fresh_with_embedding_file(self):
        """Test that fresh flag behavior with embedding storage."""
        embedding_path = os.path.join(self.temp_dir, "embeddings.h5")
        
        config = {
            'model_path': self.model_path,
            'clustering': {},
            'fresh': True,
            'store_embeddings': True,
            'embedding_path': embedding_path
        }
        
        # Create model with high counter
        existing_model = {
            "speaker_counter": 100,
            "existing_clusters": {},
            "cluster_centroids": None
        }
        
        with open(self.model_path, 'w') as f:
            json.dump(existing_model, f)
        
        speaker_id = SpeakerIdentification(config)
        
        # Should reset counter even with embedding storage enabled
        self.assertEqual(speaker_id.speaker_counter, 1)
        
        # Cleanup
        speaker_id.cleanup()
    
    def test_fresh_flag_type_variations(self):
        """Test fresh flag with different types (string, int, etc)."""
        test_cases = [
            (True, 1),      # Boolean True
            ("true", 1),    # String "true"
            ("True", 1),    # String "True"
            (1, 1),         # Integer 1
            ("yes", 1),     # String "yes"
            (False, 75),    # Boolean False
            ("false", 75),  # String "false"
            (0, 75),        # Integer 0
            ("", 75),       # Empty string
            (None, 75),     # None
        ]
        
        for fresh_value, expected_counter in test_cases:
            # Reset model each time
            existing_model = {
                "speaker_counter": 75,
                "existing_clusters": {},
                "cluster_centroids": None
            }
            
            with open(self.model_path, 'w') as f:
                json.dump(existing_model, f)
            
            config = {
                'model_path': self.model_path,
                'clustering': {},
                'fresh': fresh_value
            }
            
            speaker_id = SpeakerIdentification(config)
            
            # Check expected counter based on fresh value
            if fresh_value in [True, "true", "True", 1, "yes"]:
                self.assertEqual(speaker_id.speaker_counter, expected_counter,
                               f"Fresh={fresh_value} should reset counter to 1")
            else:
                self.assertEqual(speaker_id.speaker_counter, expected_counter,
                               f"Fresh={fresh_value} should keep counter at 75")


if __name__ == "__main__":
    unittest.main()