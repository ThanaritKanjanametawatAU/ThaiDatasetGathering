"""
Cache management utilities for dataset processing.
"""

import os
import shutil
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages dataset cache with size limits and cleanup.
    """
    
    def __init__(self, cache_dir: str, max_size_gb: float = 100.0):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for caching datasets
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache manager initialized: {cache_dir} (max: {max_size_gb:.1f}GB)")
    
    def get_cache_size(self) -> int:
        """
        Get current cache size in bytes.
        
        Returns:
            int: Cache size in bytes
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        # File might have been deleted
                        continue
        except OSError:
            logger.warning(f"Could not calculate cache size for {self.cache_dir}")
        
        return total_size
    
    def get_cache_size_gb(self) -> float:
        """
        Get current cache size in GB.
        
        Returns:
            float: Cache size in GB
        """
        return self.get_cache_size() / 1024**3
    
    def is_cache_full(self) -> bool:
        """
        Check if cache exceeds size limit.
        
        Returns:
            bool: True if cache is full
        """
        return self.get_cache_size() > self.max_size_bytes
    
    def clear_cache(self) -> bool:
        """
        Clear the entire cache directory.
        
        Returns:
            bool: True if successful
        """
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cache cleared: {self.cache_dir}")
                return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
        
        return False
    
    def clear_old_cache(self, keep_latest_n: int = 1) -> bool:
        """
        Clear old cache files, keeping only the most recent ones.
        
        Args:
            keep_latest_n: Number of latest cache entries to keep
            
        Returns:
            bool: True if successful
        """
        try:
            # Get all cache subdirectories sorted by modification time
            cache_dirs = []
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    cache_dirs.append((item.stat().st_mtime, item))
            
            # Sort by modification time (newest first)
            cache_dirs.sort(reverse=True)
            
            # Remove old directories
            removed_count = 0
            for i, (mtime, cache_path) in enumerate(cache_dirs):
                if i >= keep_latest_n:
                    try:
                        shutil.rmtree(cache_path)
                        removed_count += 1
                        logger.info(f"Removed old cache: {cache_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache {cache_path}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleared {removed_count} old cache directories")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear old cache: {e}")
            return False
    
    def enforce_cache_limit(self) -> bool:
        """
        Enforce cache size limit by clearing old cache if needed.
        
        Returns:
            bool: True if cache is now within limits
        """
        if not self.is_cache_full():
            return True
        
        logger.warning(f"Cache size ({self.get_cache_size_gb():.1f}GB) exceeds limit ({self.max_size_bytes/1024**3:.1f}GB)")
        
        # Try clearing old cache first
        if self.clear_old_cache(keep_latest_n=1):
            if not self.is_cache_full():
                logger.info("Cache size reduced by clearing old entries")
                return True
        
        # If still over limit, clear everything
        logger.warning("Clearing entire cache to enforce size limit")
        return self.clear_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            dict: Cache information
        """
        size_bytes = self.get_cache_size()
        size_gb = size_bytes / 1024**3
        
        return {
            "cache_dir": str(self.cache_dir),
            "size_bytes": size_bytes,
            "size_gb": round(size_gb, 2),
            "max_size_gb": round(self.max_size_bytes / 1024**3, 2),
            "usage_percent": round((size_bytes / self.max_size_bytes) * 100, 1),
            "is_full": self.is_cache_full()
        }