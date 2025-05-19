"""
Logging utilities for the Thai Audio Dataset Collection project.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

class ProgressTracker:
    """
    Track progress of dataset processing.
    """
    
    def __init__(self, total_items: int = 0, log_interval: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            log_interval: How often to log progress (in number of items)
        """
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "total": total_items
        }
    
    def update(self, items_processed: int = 1, skipped: int = 0, errors: int = 0) -> None:
        """
        Update progress.
        
        Args:
            items_processed: Number of items processed in this update
            skipped: Number of items skipped in this update
            errors: Number of errors encountered in this update
        """
        self.processed_items += items_processed
        self.stats["processed"] += items_processed
        self.stats["skipped"] += skipped
        self.stats["errors"] += errors
        
        current_time = time.time()
        if (current_time - self.last_log_time) > 5 or self.processed_items % self.log_interval == 0:
            self._log_progress()
            self.last_log_time = current_time
    
    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed_time = time.time() - self.start_time
        items_per_second = self.processed_items / elapsed_time if elapsed_time > 0 else 0
        
        if self.total_items > 0:
            percent_complete = (self.processed_items / self.total_items) * 100
            eta_seconds = (self.total_items - self.processed_items) / items_per_second if items_per_second > 0 else 0
            
            self.logger.info(
                f"Progress: {self.processed_items}/{self.total_items} ({percent_complete:.2f}%) "
                f"- {items_per_second:.2f} items/s - ETA: {self._format_time(eta_seconds)}"
            )
        else:
            self.logger.info(
                f"Progress: {self.processed_items} items - {items_per_second:.2f} items/s "
                f"- Elapsed: {self._format_time(elapsed_time)}"
            )
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of processing."""
        elapsed_time = time.time() - self.start_time
        return {
            "processed": self.stats["processed"],
            "skipped": self.stats["skipped"],
            "errors": self.stats["errors"],
            "total": self.total_items,
            "elapsed_time": elapsed_time,
            "items_per_second": self.processed_items / elapsed_time if elapsed_time > 0 else 0
        }
    
    def log_summary(self) -> None:
        """Log summary of processing."""
        summary = self.get_summary()
        self.logger.info(
            f"Processing complete: {summary['processed']} processed, "
            f"{summary['skipped']} skipped, {summary['errors']} errors "
            f"in {self._format_time(summary['elapsed_time'])} "
            f"({summary['items_per_second']:.2f} items/s)"
        )


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


class ProcessingLogger:
    """
    Logger for dataset processing with JSON output capability.
    """
    
    def __init__(self, log_dir: str, dataset_name: str):
        """
        Initialize processing logger.
        
        Args:
            log_dir: Directory for log files
            dataset_name: Name of the dataset being processed
        """
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(f"processing.{dataset_name}")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create JSON log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_file = os.path.join(log_dir, f"{dataset_name}_{timestamp}.json")
        self.log_entries = []
    
    def log_sample(self, sample_id: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log information about a processed sample.
        
        Args:
            sample_id: ID of the sample
            status: Status of processing (e.g., "processed", "skipped", "error")
            details: Additional details about the sample
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_name,
            "sample_id": sample_id,
            "status": status
        }
        
        if details:
            entry.update(details)
        
        self.log_entries.append(entry)
        
        # Log to standard logger
        if status == "error":
            self.logger.error(f"Sample {sample_id}: {details.get('error', 'Unknown error')}")
        elif status == "skipped":
            self.logger.warning(f"Sample {sample_id}: Skipped - {details.get('reason', 'Unknown reason')}")
        else:
            self.logger.debug(f"Sample {sample_id}: Processed")
    
    def save_json_log(self) -> None:
        """Save log entries to JSON file."""
        with open(self.json_log_file, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
        
        self.logger.info(f"Saved processing log to {self.json_log_file}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about processed samples."""
        stats = {
            "processed": 0,
            "skipped": 0,
            "error": 0,
            "total": len(self.log_entries)
        }
        
        for entry in self.log_entries:
            status = entry.get("status")
            if status in stats:
                stats[status] += 1
        
        return stats
