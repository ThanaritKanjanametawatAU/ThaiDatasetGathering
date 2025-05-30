"""
Test for fixing the SplitInfo error when loading datasets in append mode.

This test verifies the fix for issue #11: Dataset-Specific Processor Loading Error
"""

import unittest
from unittest.mock import Mock, patch
from utils.huggingface import get_last_id
import logging


class TestSplitInfoErrorFix(unittest.TestCase):
    """Test cases for fixing SplitInfo error in append mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.dataset_name = "Thanarit/Thai-Voice-Test2"

    @patch('utils.huggingface.load_dataset')
    def test_get_last_id_handles_split_info_mismatch_gracefully(self, mock_load_dataset):
        """Test that get_last_id handles SplitInfo mismatch without logging error."""
        # This test simulates the actual error we're seeing
        error_msg = (
            "[{'expected': SplitInfo(name='train', num_bytes=0, num_examples=0, "
            "shard_lengths=None, dataset_name=None), 'recorded': SplitInfo(name='train', "
            "num_bytes=23130973, num_examples=200, shard_lengths=None, "
            "dataset_name='thai-voice-test2')}]"
        )

        # First attempt raises the SplitInfo error
        mock_load_dataset.side_effect = [
            Exception(error_msg),
            # Second attempt with download_mode='force_redownload' should work
            self._create_mock_dataset(["S1", "S2", "S10", "S200"])
        ]

        # Capture logs
        with self.assertLogs('utils.huggingface', level='INFO') as log_context:
            result = get_last_id(self.dataset_name)

            # Should successfully return the last ID
            self.assertEqual(result, 200)

            # Should not have ERROR level logs, only INFO/WARNING
            error_logs = [log for log in log_context.output if 'ERROR' in log]
            self.assertEqual(
                len(error_logs), 0,
                "Should not log ERROR for SplitInfo mismatch"
            )

            # Should have INFO log about retrying
            info_logs = [
                log for log in log_context.output
                if 'INFO' in log and 'retry' in log.lower()
            ]
            self.assertGreater(
                len(info_logs), 0,
                "Should log INFO about retrying with force_redownload"
            )

    @patch('utils.huggingface.load_dataset')
    def test_get_last_id_returns_none_for_other_errors(self, mock_load_dataset):
        """Test that get_last_id still returns None for non-SplitInfo errors."""
        # Simulate a different error
        mock_load_dataset.side_effect = Exception("Network error")

        with self.assertLogs('utils.huggingface', level='ERROR') as log_context:
            result = get_last_id(self.dataset_name)

            # Should return None
            self.assertIsNone(result)

            # Should have ERROR log for non-SplitInfo errors
            error_logs = [log for log in log_context.output if 'ERROR' in log]
            self.assertGreater(
                len(error_logs), 0,
                "Should log ERROR for non-SplitInfo errors"
            )

    @patch('utils.huggingface.load_dataset')
    def test_append_mode_continues_correctly_after_split_info_fix(
        self, mock_load_dataset
    ):
        """Test that append mode works correctly after handling SplitInfo error."""
        # First call fails with SplitInfo error
        error_msg = (
            "[{'expected': SplitInfo(name='train', num_bytes=0, num_examples=0, "
            "shard_lengths=None, dataset_name=None), 'recorded': SplitInfo(name='train', "
            "num_bytes=23130973, num_examples=200, shard_lengths=None, "
            "dataset_name='thai-voice-test2')}]"
        )

        # Setup mock to fail first, then succeed
        mock_load_dataset.side_effect = [
            Exception(error_msg),
            self._create_mock_dataset(
                [f"S{i}" for i in range(1, 201)]  # S1 to S200
            )
        ]

        # Get last ID
        last_id = get_last_id(self.dataset_name)

        # Verify it returns the correct last ID
        self.assertEqual(last_id, 200)

        # Verify load_dataset was called twice
        self.assertEqual(mock_load_dataset.call_count, 2)

        # Verify second call used force_redownload
        second_call_kwargs = mock_load_dataset.call_args_list[1][1]
        self.assertEqual(
            second_call_kwargs.get('download_mode'),
            'force_redownload'
        )

    def _create_mock_dataset(self, ids):
        """Helper to create a mock dataset."""
        mock_dataset = Mock()
        mock_dataset.features = {
            "ID": "string",
            "audio": "audio",
            "transcript": "string"
        }
        mock_dataset.__getitem__ = Mock(
            side_effect=lambda key: ids if key == "ID" else None
        )
        return mock_dataset


if __name__ == '__main__':
    unittest.main()