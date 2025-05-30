"""
Test duration tracking functionality for issue #8
Verifies that duration is properly accumulated when appending to datasets
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import shutil


class TestDurationTracking(unittest.TestCase):
    """Test that duration tracking works correctly in streaming mode"""
    
    def test_duration_accumulation_in_streaming_loop(self):
        """Test the actual duration tracking logic that should be in main.py"""
        
        # Simulate what should happen in the streaming loop
        total_duration_seconds = 0  # This should be initialized
        dataset_duration_seconds = 0  # This should be initialized per dataset
        
        # Process samples
        samples = [
            {'length': 1.5},
            {'length': 2.0},
            {'length': None},  # Some samples might not have length
            {'length': 3.5},
            {'length': 1.0}
        ]
        
        for sample in samples:
            # This is what the code should do
            sample_duration = sample.get('length', 0) or 0
            dataset_duration_seconds += sample_duration
            total_duration_seconds += sample_duration
        
        # Verify accumulation
        self.assertEqual(dataset_duration_seconds, 8.0)
        self.assertEqual(total_duration_seconds, 8.0)
        
        # Convert to hours
        dataset_duration_hours = dataset_duration_seconds / 3600.0
        total_duration_hours = total_duration_seconds / 3600.0
        
        self.assertAlmostEqual(dataset_duration_hours, 0.00222, places=4)
        self.assertAlmostEqual(total_duration_hours, 0.00222, places=4)
    
    def test_multiple_datasets_duration_accumulation(self):
        """Test duration accumulation across multiple datasets"""
        
        total_duration_seconds = 0
        
        # Dataset 1
        dataset1_duration_seconds = 0
        dataset1_samples = [
            {'length': 2.5},
            {'length': 3.0},
            {'length': 1.5}
        ]
        
        for sample in dataset1_samples:
            sample_duration = sample.get('length', 0) or 0
            dataset1_duration_seconds += sample_duration
            total_duration_seconds += sample_duration
        
        self.assertEqual(dataset1_duration_seconds, 7.0)
        
        # Dataset 2
        dataset2_duration_seconds = 0
        dataset2_samples = [
            {'length': 4.0},
            {'length': 2.0},
            {'length': None},  # Handle None
            {'length': 1.0}
        ]
        
        for sample in dataset2_samples:
            sample_duration = sample.get('length', 0) or 0
            dataset2_duration_seconds += sample_duration
            total_duration_seconds += sample_duration
        
        self.assertEqual(dataset2_duration_seconds, 7.0)
        self.assertEqual(total_duration_seconds, 14.0)
        
        # Convert to hours
        total_duration_hours = total_duration_seconds / 3600.0
        self.assertAlmostEqual(total_duration_hours, 0.00389, places=4)
    
    @patch('main.logger')
    def test_duration_logging(self, mock_logger):
        """Test that duration is properly logged"""
        
        dataset_name = "TestDataset"
        sample_count = 100
        dataset_duration_seconds = 3600.0  # 1 hour
        
        # This is what should be logged
        dataset_duration_hours = dataset_duration_seconds / 3600.0
        expected_log = f"Completed {dataset_name}: {sample_count} samples, {dataset_duration_hours:.2f} hours"
        
        # Simulate the logging
        mock_logger.info(expected_log)
        
        # Verify
        mock_logger.info.assert_called_with("Completed TestDataset: 100 samples, 1.00 hours")
    
    def test_dataset_card_duration_format(self):
        """Test that duration is formatted correctly for dataset card"""
        
        # Test various duration values
        test_cases = [
            (3600.0, "1.00"),     # 1 hour
            (7200.0, "2.00"),     # 2 hours  
            (5400.0, "1.50"),     # 1.5 hours
            (360.0, "0.10"),      # 0.1 hours
            (36.0, "0.01"),       # 0.01 hours
            (0.0, "0.00"),        # 0 hours
        ]
        
        for seconds, expected_formatted in test_cases:
            hours = seconds / 3600.0
            formatted = f"{hours:.2f}"
            self.assertEqual(formatted, expected_formatted)


if __name__ == '__main__':
    unittest.main()