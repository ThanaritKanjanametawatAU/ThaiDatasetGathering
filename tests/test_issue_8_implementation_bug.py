"""
Test to verify the bug in main.py's duration tracking implementation
This test should FAIL with the current implementation, proving the bug exists
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import ast
import re


class TestIssue8ImplementationBug(unittest.TestCase):
    """Test that reveals the bugs in the current implementation"""
    
    def test_duration_variables_are_initialized_in_main_py(self):
        """Test that duration tracking variables are properly initialized"""
        
        # Read main.py and check for variable initialization
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for total_duration_seconds initialization before the dataset loop
        # This should be initialized before line 461 where the dataset loop starts
        pattern_total = r'total_duration_seconds\s*=\s*0'
        match_total = re.search(pattern_total, content[:20000])  # Check first part of file
        
        self.assertIsNotNone(match_total, 
            "total_duration_seconds must be initialized to 0 before the dataset processing loop")
        
        # Check that dataset_duration_seconds is initialized inside the dataset loop
        # Look for the pattern inside the for dataset_name loop
        pattern_dataset = r'for\s+dataset_name.*?:\s*\n(?:.*\n)*?\s*dataset_duration_seconds\s*=\s*0'
        match_dataset = re.search(pattern_dataset, content, re.MULTILINE | re.DOTALL)
        
        self.assertIsNotNone(match_dataset,
            "dataset_duration_seconds must be initialized to 0 at the start of each dataset processing")
    
    def test_duration_accumulation_in_sample_loop(self):
        """Test that duration is accumulated for each sample"""
        
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for duration accumulation pattern
        # Should have something like:
        # sample_duration = sample.get('length', 0) or 0
        # dataset_duration_seconds += sample_duration
        # total_duration_seconds += sample_duration
        
        pattern = r"sample\.get\(['\"]length['\"].*?\).*?\n.*?dataset_duration_seconds\s*\+=.*?\n.*?total_duration_seconds\s*\+="
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        
        self.assertIsNotNone(match,
            "Duration must be accumulated for each sample in the processing loop")
    
    def test_duration_used_in_logging(self):
        """Test that duration is properly used in logging"""
        
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check that dataset_duration_hours is calculated and logged
        pattern = r'dataset_duration_hours\s*=\s*dataset_duration_seconds\s*/\s*3600'
        match = re.search(pattern, content)
        
        self.assertIsNotNone(match,
            "dataset_duration_hours must be calculated from dataset_duration_seconds")
        
        # Check for logging with duration
        pattern_log = r'logger\.info.*?{dataset_name}.*?{dataset_duration_hours.*?}'
        match_log = re.search(pattern_log, content)
        
        self.assertIsNotNone(match_log,
            "Duration hours must be included in completion logging")
    
    def test_total_duration_passed_to_dataset_card(self):
        """Test that total duration is passed to dataset card"""
        
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check that total_duration_hours is calculated
        pattern_calc = r'total_duration_hours\s*=\s*total_duration_seconds\s*/\s*3600'
        match_calc = re.search(pattern_calc, content)
        
        self.assertIsNotNone(match_calc,
            "total_duration_hours must be calculated from total_duration_seconds")
        
        # Check that it's passed to dataset_info
        pattern_info = r'["\']total_duration_hours["\']\s*:\s*total_duration_hours'
        match_info = re.search(pattern_info, content)
        
        self.assertIsNotNone(match_info,
            "total_duration_hours must be passed to dataset_info for upload_dataset_card")


if __name__ == '__main__':
    unittest.main()