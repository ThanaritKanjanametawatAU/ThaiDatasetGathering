"""
Memory management utilities for efficient audio processing.
Includes memory pools, lazy loading, and resource management.
"""

import gc
import logging
import weakref

# Try to import psutil for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - resource monitoring will be limited")
from typing import Dict, List, Optional, Any, Tuple, Iterator
from dataclasses import dataclass
import numpy as np
from threading import Lock
import os

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""
    data: np.ndarray
    size: int
    dtype: np.dtype
    in_use: bool = False
    last_used: float = 0.0


class MemoryPool:
    """
    Memory pool for efficient array allocation and reuse.
    """
    
    def __init__(self, max_size: str = "1GB", enable_stats: bool = True):
        """
        Initialize memory pool.
        
        Args:
            max_size: Maximum pool size (e.g., "1GB", "500MB")
            enable_stats: Whether to track allocation statistics
        """
        self.max_size_bytes = self._parse_size(max_size)
        self.enable_stats = enable_stats
        
        self._blocks: List[MemoryBlock] = []
        self._lock = Lock()
        self._current_size = 0
        
        # Statistics
        self._stats = {
            'allocations': 0,
            'reuses': 0,
            'evictions': 0,
            'peak_usage': 0
        }
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 ** 3)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 ** 2)
        elif size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        else:
            return int(size_str)
    
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Allocate array from pool.
        
        Args:
            shape: Array shape
            dtype: Data type
            
        Returns:
            Numpy array
        """
        size = np.prod(shape) * np.dtype(dtype).itemsize
        
        with self._lock:
            # Try to find reusable block
            for block in self._blocks:
                if (not block.in_use and 
                    block.size >= size and 
                    block.dtype == dtype):
                    
                    # Reuse block
                    block.in_use = True
                    block.last_used = os.times().elapsed
                    
                    if self.enable_stats:
                        self._stats['reuses'] += 1
                    
                    # Reshape and return view
                    return block.data.ravel()[:np.prod(shape)].reshape(shape)
            
            # No suitable block found, allocate new
            if self._current_size + size > self.max_size_bytes:
                # Need to evict blocks
                self._evict_blocks(size)
            
            # Allocate new block
            data = np.empty(np.prod(shape), dtype=dtype)
            block = MemoryBlock(
                data=data,
                size=size,
                dtype=dtype,
                in_use=True,
                last_used=os.times().elapsed
            )
            
            self._blocks.append(block)
            self._current_size += size
            
            if self.enable_stats:
                self._stats['allocations'] += 1
                self._stats['peak_usage'] = max(self._stats['peak_usage'], self._current_size)
            
            return data.reshape(shape)
    
    def release(self, array: np.ndarray) -> None:
        """
        Release array back to pool.
        
        Args:
            array: Array to release
        """
        with self._lock:
            # Find corresponding block
            for block in self._blocks:
                if block.in_use and np.shares_memory(array, block.data):
                    block.in_use = False
                    break
    
    def _evict_blocks(self, needed_size: int) -> None:
        """Evict least recently used blocks to make space."""
        # Sort by last used time (oldest first)
        unused_blocks = [b for b in self._blocks if not b.in_use]
        unused_blocks.sort(key=lambda b: b.last_used)
        
        freed_size = 0
        blocks_to_remove = []
        
        for block in unused_blocks:
            if freed_size >= needed_size:
                break
            
            blocks_to_remove.append(block)
            freed_size += block.size
        
        # Remove blocks
        for block in blocks_to_remove:
            self._blocks.remove(block)
            self._current_size -= block.size
            if self.enable_stats:
                self._stats['evictions'] += 1
    
    def clear(self) -> None:
        """Clear all blocks from pool."""
        with self._lock:
            self._blocks.clear()
            self._current_size = 0
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                **self._stats,
                'current_size': self._current_size,
                'max_size': self.max_size_bytes,
                'utilization': self._current_size / self.max_size_bytes,
                'num_blocks': len(self._blocks),
                'blocks_in_use': sum(1 for b in self._blocks if b.in_use)
            }


class LazyAudioLoader:
    """
    Lazy loader for audio files with memory-efficient iteration.
    """
    
    def __init__(self, file_paths: List[str], 
                 batch_size: int = 1,
                 cache_size: int = 10):
        """
        Initialize lazy loader.
        
        Args:
            file_paths: List of audio file paths
            batch_size: Number of files to load at once
            cache_size: Number of files to keep in cache
        """
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.cache_size = cache_size
        
        self._cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._cache_order: List[str] = []
        self._memory_usage = 0
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, int]]:
        """Iterate through audio files."""
        for i in range(0, len(self.file_paths), self.batch_size):
            batch_paths = self.file_paths[i:i + self.batch_size]
            
            for path in batch_paths:
                yield self._load_file(path)
            
            # Force garbage collection after each batch
            gc.collect()
    
    def _load_file(self, path: str) -> Tuple[np.ndarray, int]:
        """Load single audio file with caching."""
        # Check cache first
        if path in self._cache:
            # Move to end (LRU)
            self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._cache[path]
        
        # Load file
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
            
            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Add to cache
            self._add_to_cache(path, (audio, sr))
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            # Return empty audio as fallback
            return np.array([]), 16000
    
    def _add_to_cache(self, path: str, data: Tuple[np.ndarray, int]) -> None:
        """Add file to cache with LRU eviction."""
        audio, sr = data
        size = audio.nbytes
        
        # Evict if needed
        while len(self._cache) >= self.cache_size:
            oldest_path = self._cache_order.pop(0)
            old_data = self._cache.pop(oldest_path)
            self._memory_usage -= old_data[0].nbytes
        
        # Add new entry
        self._cache[path] = data
        self._cache_order.append(path)
        self._memory_usage += size
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._memory_usage
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_order.clear()
        self._memory_usage = 0
        gc.collect()


class ResourceManager:
    """
    Manages system resources for audio processing.
    """
    
    def __init__(self, max_memory_percent: float = 80.0,
                 max_cpu_percent: float = 90.0):
        """
        Initialize resource manager.
        
        Args:
            max_memory_percent: Maximum memory usage percentage
            max_cpu_percent: Maximum CPU usage percentage
        """
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        
        # Track managed resources
        self._memory_pools: List[weakref.ref] = []
        self._active_processes = 0
        self._lock = Lock()
    
    def register_memory_pool(self, pool: MemoryPool) -> None:
        """Register a memory pool for management."""
        self._memory_pools.append(weakref.ref(pool))
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage."""
        if not PSUTIL_AVAILABLE:
            # Return dummy values if psutil not available
            return {
                'memory': {
                    'total': 0,
                    'available': 0,
                    'percent': 0,
                    'within_limit': True
                },
                'cpu': {
                    'percent': 0,
                    'count': 1,
                    'within_limit': True
                }
            }
        
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'within_limit': memory.percent <= self.max_memory_percent
            },
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count(),
                'within_limit': cpu_percent <= self.max_cpu_percent
            }
        }
    
    def wait_for_resources(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for resources to become available.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if resources available, False if timeout
        """
        import time
        start_time = time.time()
        
        while True:
            status = self.check_resources()
            
            if (status['memory']['within_limit'] and 
                status['cpu']['within_limit']):
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            # Wait and retry
            time.sleep(0.5)
            
            # Try to free memory
            self._cleanup_memory()
    
    def _cleanup_memory(self) -> None:
        """Attempt to free memory."""
        # Clean up dead references
        self._memory_pools = [ref for ref in self._memory_pools if ref() is not None]
        
        # Clear caches in memory pools
        for pool_ref in self._memory_pools:
            pool = pool_ref()
            if pool:
                # Evict unused blocks
                with pool._lock:
                    unused_blocks = [b for b in pool._blocks if not b.in_use]
                    for block in unused_blocks[:len(unused_blocks)//2]:  # Evict half
                        pool._blocks.remove(block)
                        pool._current_size -= block.size
        
        # Force garbage collection
        gc.collect()
    
    def acquire_processing_slot(self) -> bool:
        """
        Acquire a processing slot if resources available.
        
        Returns:
            True if slot acquired
        """
        with self._lock:
            if not self.wait_for_resources(timeout=5.0):
                return False
            
            self._active_processes += 1
            return True
    
    def release_processing_slot(self) -> None:
        """Release a processing slot."""
        with self._lock:
            self._active_processes = max(0, self._active_processes - 1)
    
    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on available memory."""
        if not PSUTIL_AVAILABLE:
            # Return conservative default if psutil not available
            return 4
            
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        
        # Assume ~100MB per audio file (conservative)
        batch_size = int(available_mb / 100)
        
        # Apply limits
        batch_size = max(1, min(batch_size, 32))
        
        return batch_size


# Global resource manager instance
_global_resource_manager = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        if not PSUTIL_AVAILABLE:
            # Just execute function without monitoring if psutil not available
            return func(*args, **kwargs)
            
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        if memory_increase > 100:  # Log if increase > 100MB
            logger.warning(f"{func.__name__} increased memory by {memory_increase:.1f}MB")
        
        return result
    
    return wrapper