"""
Plugin discovery and loading system for audio processors.
Supports dynamic loading, version checking, and dependency resolution.
"""

import os
import sys
import json
import importlib
import importlib.util
import inspect
import logging
from typing import Dict, List, Optional, Any, Type, Set, Tuple
from pathlib import Path
import pickle
import hashlib
from collections import defaultdict, deque

from .base import AudioProcessor, PluginInfo, PluginLoadError

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for all discovered plugins.
    Manages plugin metadata, dependencies, and version compatibility.
    """
    
    def __init__(self, plugin_paths: Optional[List[str]] = None, 
                 cache_dir: Optional[str] = None):
        """
        Initialize plugin registry.
        
        Args:
            plugin_paths: List of directories to search for plugins
            cache_dir: Directory for caching plugin information
        """
        self.plugin_paths = plugin_paths or []
        self.cache_dir = cache_dir
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_classes: Dict[str, Type[AudioProcessor]] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._cache_file = None
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_file = os.path.join(cache_dir, "plugin_registry.pkl")
    
    def register_plugin(self, plugin_class: Type[AudioProcessor]) -> None:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Plugin class to register
        """
        try:
            info = plugin_class.get_plugin_info()
            self.plugins[info.name] = info
            self.plugin_classes[info.name] = plugin_class
            
            # Build dependency graph
            instance = plugin_class()
            dependencies = instance.get_dependencies()
            self.dependency_graph[info.name] = set(dependencies)
            
            logger.info(f"Registered plugin: {info.name} v{info.version}")
            
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
            raise PluginLoadError(plugin_class.__name__, str(e))
    
    def get_plugin_class(self, name: str) -> Type[AudioProcessor]:
        """
        Get plugin class by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin class
        """
        if name not in self.plugin_classes:
            raise PluginLoadError(name, "Plugin not found in registry")
        return self.plugin_classes[name]
    
    def get_plugin_info(self, name: str) -> PluginInfo:
        """
        Get plugin information by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin information
        """
        if name not in self.plugins:
            raise PluginLoadError(name, "Plugin not found in registry")
        return self.plugins[name]
    
    def list_plugins(self, category: Optional[str] = None) -> List[PluginInfo]:
        """
        List all registered plugins.
        
        Args:
            category: Filter by category (optional)
            
        Returns:
            List of plugin information
        """
        plugins = list(self.plugins.values())
        if category:
            plugins = [p for p in plugins if p.category == category]
        return plugins
    
    def check_compatibility(self, plugin_name: str, version: str, 
                          allowed_versions: List[str]) -> bool:
        """
        Check if plugin version is compatible.
        
        Args:
            plugin_name: Plugin name
            version: Plugin version
            allowed_versions: List of allowed version patterns
            
        Returns:
            True if compatible
        """
        for allowed in allowed_versions:
            if self._version_matches(version, allowed):
                return True
        return False
    
    def _version_matches(self, version: str, pattern: str) -> bool:
        """
        Check if version matches pattern.
        
        Args:
            version: Version string (e.g., "1.2.3")
            pattern: Pattern string (e.g., "1.2", "1.2+", "1.x")
            
        Returns:
            True if matches
        """
        if pattern.endswith('+'):
            # Minimum version check
            pattern_parts = pattern[:-1].split('.')
            version_parts = version.split('.')
            
            for i, (v, p) in enumerate(zip(version_parts, pattern_parts)):
                if i < len(pattern_parts) - 1:
                    if v != p:
                        return False
                else:
                    return int(v) >= int(p)
            return True
            
        elif 'x' in pattern or '*' in pattern:
            # Wildcard pattern
            pattern_parts = pattern.replace('*', 'x').split('.')
            version_parts = version.split('.')
            
            for v, p in zip(version_parts, pattern_parts):
                if p != 'x' and v != p:
                    return False
            return True
            
        else:
            # Exact prefix match
            return version.startswith(pattern)
    
    def resolve_dependencies(self, plugins: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Resolve plugin load order based on dependencies.
        
        Args:
            plugins: Dictionary of plugin names to plugin info
            
        Returns:
            List of plugin names in load order
        """
        # Build dependency graph
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        for plugin, info in plugins.items():
            deps = info.get('dependencies', [])
            for dep in deps:
                graph[dep].add(plugin)
                in_degree[plugin] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([p for p in plugins if in_degree[p] == 0])
        result = []
        
        while queue:
            plugin = queue.popleft()
            result.append(plugin)
            
            for dependent in graph[plugin]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(result) != len(plugins):
            raise PluginLoadError("dependency_cycle", 
                                "Circular dependency detected in plugins")
        
        return result
    
    def save_cache(self) -> None:
        """Save registry to cache file."""
        if not self._cache_file:
            return
            
        cache_data = {
            'plugins': self.plugins,
            'dependency_graph': dict(self.dependency_graph),
            'plugin_paths': self.plugin_paths
        }
        
        try:
            with open(self._cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved plugin registry cache to {self._cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save plugin cache: {e}")
    
    def load_cache(self) -> bool:
        """
        Load registry from cache file.
        
        Returns:
            True if cache was loaded successfully
        """
        if not self._cache_file or not os.path.exists(self._cache_file):
            return False
            
        try:
            with open(self._cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.plugins = cache_data['plugins']
            self.dependency_graph = defaultdict(set, cache_data['dependency_graph'])
            
            # Note: plugin_classes need to be re-imported, not cached
            logger.info(f"Loaded plugin registry cache from {self._cache_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load plugin cache: {e}")
            return False


class PluginDiscovery:
    """
    Discovers plugins in specified directories.
    """
    
    def __init__(self, registry: PluginRegistry):
        """
        Initialize plugin discovery.
        
        Args:
            registry: Plugin registry to register discovered plugins
        """
        self.registry = registry
    
    def scan_directories(self, paths: Optional[List[str]] = None) -> List[PluginInfo]:
        """
        Scan directories for plugins.
        
        Args:
            paths: Directories to scan (uses registry paths if None)
            
        Returns:
            List of discovered plugin information
        """
        paths = paths or self.registry.plugin_paths
        discovered = []
        
        for path in paths:
            if not os.path.exists(path):
                logger.warning(f"Plugin path does not exist: {path}")
                continue
                
            logger.info(f"Scanning for plugins in: {path}")
            discovered.extend(self._scan_directory(path))
        
        return discovered
    
    def _scan_directory(self, directory: str) -> List[PluginInfo]:
        """
        Scan a single directory for plugins.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of discovered plugin information
        """
        discovered = []
        
        # Add directory to Python path temporarily
        if directory not in sys.path:
            sys.path.insert(0, directory)
        
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.py') and not filename.startswith('_'):
                    module_name = filename[:-3]
                    
                    try:
                        # Import module
                        spec = importlib.util.spec_from_file_location(
                            module_name, 
                            os.path.join(directory, filename)
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Find AudioProcessor subclasses
                            for name, obj in inspect.getmembers(module):
                                if (inspect.isclass(obj) and 
                                    issubclass(obj, AudioProcessor) and 
                                    obj != AudioProcessor):
                                    
                                    try:
                                        self.registry.register_plugin(obj)
                                        info = obj.get_plugin_info()
                                        discovered.append(info)
                                    except Exception as e:
                                        logger.error(f"Failed to register {name}: {e}")
                                        
                    except Exception as e:
                        logger.error(f"Failed to import {module_name}: {e}")
                        
        finally:
            # Remove directory from path
            if directory in sys.path:
                sys.path.remove(directory)
        
        return discovered


class PluginLoader:
    """
    Loads and instantiates plugins with dependency injection.
    """
    
    def __init__(self, registry: PluginRegistry):
        """
        Initialize plugin loader.
        
        Args:
            registry: Plugin registry
        """
        self.registry = registry
        self._instances: Dict[str, AudioProcessor] = {}
    
    def load_plugin(self, name: str, config: Optional[Dict[str, Any]] = None,
                   inject_dependencies: bool = True) -> AudioProcessor:
        """
        Load and instantiate a plugin.
        
        Args:
            name: Plugin name
            config: Plugin configuration
            inject_dependencies: Whether to inject dependencies
            
        Returns:
            Plugin instance
        """
        # Check if already instantiated
        if name in self._instances and not config:
            return self._instances[name]
        
        # Get plugin class
        plugin_class = self.registry.get_plugin_class(name)
        
        # Load dependencies first if needed
        if inject_dependencies:
            instance = plugin_class(config)
            deps = instance.get_dependencies()
            
            # Load each dependency
            dep_instances = {}
            for dep_name in deps:
                dep_instances[dep_name] = self.load_plugin(dep_name)
            
            # Inject dependencies if plugin supports it
            if hasattr(instance, 'inject_dependencies'):
                instance.inject_dependencies(dep_instances)
        else:
            instance = plugin_class(config)
        
        # Initialize plugin
        instance.initialize()
        
        # Cache instance if no custom config
        if not config:
            self._instances[name] = instance
        
        return instance
    
    def unload_plugin(self, name: str) -> None:
        """
        Unload a plugin and cleanup resources.
        
        Args:
            name: Plugin name
        """
        if name in self._instances:
            instance = self._instances[name]
            instance.cleanup()
            del self._instances[name]
    
    def unload_all(self) -> None:
        """Unload all cached plugins."""
        for name in list(self._instances.keys()):
            self.unload_plugin(name)


class PluginManager:
    """
    High-level manager for plugin lifecycle.
    Combines registry, discovery, and loading.
    """
    
    def __init__(self, plugin_paths: List[str], cache_dir: Optional[str] = None,
                 auto_discover: bool = True):
        """
        Initialize plugin manager.
        
        Args:
            plugin_paths: Directories to search for plugins
            cache_dir: Directory for caching
            auto_discover: Whether to discover plugins on initialization
        """
        self.registry = PluginRegistry(plugin_paths, cache_dir)
        self.discovery = PluginDiscovery(self.registry)
        self.loader = PluginLoader(self.registry)
        
        # Try to load from cache first
        if not self.registry.load_cache() and auto_discover:
            self.discover_plugins()
            self.registry.save_cache()
    
    def discover_plugins(self) -> List[PluginInfo]:
        """
        Discover all plugins in configured paths.
        
        Returns:
            List of discovered plugins
        """
        return self.discovery.scan_directories()
    
    def get_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> AudioProcessor:
        """
        Get a plugin instance.
        
        Args:
            name: Plugin name
            config: Plugin configuration
            
        Returns:
            Plugin instance
        """
        return self.loader.load_plugin(name, config)
    
    def list_plugins(self, category: Optional[str] = None) -> List[PluginInfo]:
        """
        List available plugins.
        
        Args:
            category: Filter by category
            
        Returns:
            List of plugin information
        """
        return self.registry.list_plugins(category)
    
    def reload_plugins(self) -> None:
        """Reload all plugins from disk."""
        self.loader.unload_all()
        self.registry = PluginRegistry(self.registry.plugin_paths, self.registry.cache_dir)
        self.discovery = PluginDiscovery(self.registry)
        self.loader = PluginLoader(self.registry)
        self.discover_plugins()
        self.registry.save_cache()