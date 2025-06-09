"""
Data loading utilities for audio processing.
Provides lazy loading and efficient data handling.
"""

import os
import logging
from typing import Iterator, Tuple, List, Optional, Dict, Any
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

logger = logging.getLogger(__name__)


class LazyAudioLoader:
    """
    Lazy loader for audio files with prefetching and caching.
    """
    
    def __init__(self, files: List[str], 
                 prefetch_size: int = 5,
                 num_workers: int = 2):
        """
        Initialize lazy audio loader.
        
        Args:
            files: List of audio file paths
            prefetch_size: Number of files to prefetch
            num_workers: Number of worker threads for loading
        """
        self.files = files
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        
        self._current_index = 0
        self._prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self._stop_event = threading.Event()
        self._workers = []
        
        # Start prefetch workers
        self._start_workers()
    
    def _start_workers(self) -> None:
        """Start prefetch worker threads."""
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self) -> None:
        """Worker loop for prefetching audio files."""
        while not self._stop_event.is_set():
            try:
                # Get next file to load
                if self._current_index < len(self.files):
                    file_path = self.files[self._current_index]
                    self._current_index += 1
                    
                    # Load audio
                    try:
                        audio, sr = self._load_audio_file(file_path)
                        self._prefetch_queue.put((file_path, audio, sr))
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {e}")
                        # Put error marker
                        self._prefetch_queue.put((file_path, None, None))
                else:
                    # No more files, wait a bit
                    threading.Event().wait(0.1)
                    
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load a single audio file."""
        import soundfile as sf
        
        audio, sr = sf.read(file_path)
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        return audio, sr
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        """Iterate through audio files."""
        files_loaded = 0
        
        while files_loaded < len(self.files):
            try:
                # Get from prefetch queue
                file_path, audio, sr = self._prefetch_queue.get(timeout=30.0)
                
                if audio is not None:
                    yield audio, sr
                else:
                    # Error loading file, yield empty
                    yield np.array([]), 16000
                
                files_loaded += 1
                
            except queue.Empty:
                logger.error("Timeout waiting for audio file")
                break
    
    def __del__(self):
        """Clean up workers."""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
    
    def __len__(self) -> int:
        """Get number of files."""
        return len(self.files)