"""Configuration management UI for audio enhancement."""

import os
import json
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigurationUI:
    """Manages audio enhancement configuration through a UI interface."""
    
    # Valid configuration options
    VALID_NOISE_LEVELS = ['mild', 'moderate', 'aggressive']
    VALID_BATCH_SIZES = [16, 32, 64, 128, 256]
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 512
    
    DEFAULT_CONFIG = {
        'noise_reduction_level': 'moderate',
        'batch_size': 32,
        'gpu_device': 0,
        'adaptive_mode': True,
        'skip_clean_audio': True,
        'progressive_enhancement': True,
        'show_dashboard': True,
        'dashboard_update_interval': 100,
        'enable_comparison': True,
        'comparison_sample_rate': 0.01,
        'save_comparison_plots': True,
        'max_retries': 3,
        'fallback_enabled': True,
        'secondary_speaker_detection': {
            'enabled': True,
            'min_duration': 0.1,
            'max_duration': 5.0,
            'similarity_threshold': 0.7,
            'suppression_strength': 0.6,
            'confidence_threshold': 0.5
        }
    }
    
    def __init__(self, config_file: str = "enhancement_config.json"):
        """Initialize the configuration UI.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.current_config = self._load_or_create_config()
        self.config_history: List[Dict[str, Any]] = []
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load existing config or create default."""
        if os.path.exists(self.config_file):
            try:
                return self.load_config()
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config
            self.save_config(self.DEFAULT_CONFIG.copy())
            return self.DEFAULT_CONFIG.copy()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        # Merge with defaults for any missing keys
        merged_config = self.DEFAULT_CONFIG.copy()
        merged_config.update(config)
        
        self.current_config = merged_config
        return merged_config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """Save configuration to file.
        
        Args:
            config: Configuration to save (uses current if not provided)
        """
        if config is None:
            config = self.current_config
        
        # Add metadata
        config['_metadata'] = {
            'last_modified': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Save to file
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Add to history
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'config': config.copy()
        })
        
        logger.info(f"Configuration saved to {self.config_file}")
    
    def update_config(self, key: str, value: Any):
        """Update a configuration value.
        
        Args:
            key: Configuration key
            value: New value
        """
        # Handle nested keys
        if '.' in key:
            parts = key.split('.')
            config = self.current_config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self.current_config[key] = value
        
        logger.debug(f"Updated config: {key} = {value}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration values.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate noise reduction level
            if 'noise_reduction_level' in config:
                if config['noise_reduction_level'] not in self.VALID_NOISE_LEVELS:
                    logger.error(f"Invalid noise reduction level: {config['noise_reduction_level']}")
                    return False
            
            # Validate batch size
            if 'batch_size' in config:
                if not isinstance(config['batch_size'], int):
                    logger.error("Batch size must be an integer")
                    return False
                if config['batch_size'] < self.MIN_BATCH_SIZE or config['batch_size'] > self.MAX_BATCH_SIZE:
                    logger.error(f"Batch size must be between {self.MIN_BATCH_SIZE} and {self.MAX_BATCH_SIZE}")
                    return False
            
            # Validate GPU device
            if 'gpu_device' in config:
                if not isinstance(config['gpu_device'], int) or config['gpu_device'] < 0:
                    logger.error("GPU device must be a non-negative integer")
                    return False
            
            # Validate boolean fields
            bool_fields = ['adaptive_mode', 'skip_clean_audio', 'progressive_enhancement',
                          'show_dashboard', 'enable_comparison', 'save_comparison_plots',
                          'fallback_enabled']
            for field in bool_fields:
                if field in config and not isinstance(config[field], bool):
                    logger.error(f"{field} must be a boolean")
                    return False
            
            # Validate numeric fields
            numeric_fields = {
                'dashboard_update_interval': (1, 10000),
                'comparison_sample_rate': (0.0, 1.0),
                'max_retries': (0, 10)
            }
            for field, (min_val, max_val) in numeric_fields.items():
                if field in config:
                    val = config[field]
                    if not isinstance(val, (int, float)):
                        logger.error(f"{field} must be numeric")
                        return False
                    if val < min_val or val > max_val:
                        logger.error(f"{field} must be between {min_val} and {max_val}")
                        return False
            
            # Validate secondary speaker detection
            if 'secondary_speaker_detection' in config:
                ssd = config['secondary_speaker_detection']
                if not isinstance(ssd, dict):
                    logger.error("secondary_speaker_detection must be a dictionary")
                    return False
                
                # Validate sub-fields
                if 'min_duration' in ssd and (ssd['min_duration'] < 0 or ssd['min_duration'] > 10):
                    logger.error("min_duration must be between 0 and 10")
                    return False
                if 'max_duration' in ssd and (ssd['max_duration'] < 0 or ssd['max_duration'] > 60):
                    logger.error("max_duration must be between 0 and 60")
                    return False
                if 'similarity_threshold' in ssd and (ssd['similarity_threshold'] < 0 or ssd['similarity_threshold'] > 1):
                    logger.error("similarity_threshold must be between 0 and 1")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.current_config.copy()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.current_config = self.DEFAULT_CONFIG.copy()
        self.save_config()
        logger.info("Configuration reset to defaults")
    
    def export_config(self, export_path: str):
        """Export configuration to a file.
        
        Args:
            export_path: Path to export configuration
        """
        config_to_export = self.current_config.copy()
        config_to_export['_export_metadata'] = {
            'exported_at': datetime.now().isoformat(),
            'source_file': self.config_file
        }
        
        with open(export_path, 'w') as f:
            json.dump(config_to_export, f, indent=2)
        
        logger.info(f"Configuration exported to {export_path}")
    
    def import_config(self, import_path: str):
        """Import configuration from a file.
        
        Args:
            import_path: Path to import configuration from
        """
        with open(import_path, 'r') as f:
            imported_config = json.load(f)
        
        # Remove metadata fields
        imported_config.pop('_metadata', None)
        imported_config.pop('_export_metadata', None)
        
        # Validate before importing
        if self.validate_config(imported_config):
            self.current_config = imported_config
            self.save_config()
            logger.info(f"Configuration imported from {import_path}")
        else:
            raise ValueError("Invalid configuration in import file")
    
    def get_config_diff(self, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get differences between current config and another config.
        
        Args:
            other_config: Configuration to compare with
            
        Returns:
            Dictionary of differences
        """
        diff = {}
        
        def compare_dicts(d1, d2, path=""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    diff[current_path] = {'status': 'added', 'value': d2[key]}
                elif key not in d2:
                    diff[current_path] = {'status': 'removed', 'value': d1[key]}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    diff[current_path] = {
                        'status': 'changed',
                        'old_value': d1[key],
                        'new_value': d2[key]
                    }
        
        compare_dicts(self.current_config, other_config)
        return diff
    
    def generate_config_report(self) -> Dict[str, Any]:
        """Generate a configuration report.
        
        Returns:
            Configuration report dictionary
        """
        return {
            'current_config': self.current_config,
            'validation_status': self.validate_config(self.current_config),
            'history_count': len(self.config_history),
            'last_modified': self.current_config.get('_metadata', {}).get('last_modified', 'Unknown'),
            'differences_from_default': self.get_config_diff(self.DEFAULT_CONFIG),
            'recommended_settings': self._get_recommended_settings()
        }
    
    def _get_recommended_settings(self) -> Dict[str, str]:
        """Get recommended settings based on use case."""
        return {
            'high_quality': {
                'noise_reduction_level': 'moderate',
                'batch_size': 32,
                'adaptive_mode': True,
                'progressive_enhancement': True,
                'secondary_speaker_detection.enabled': True
            },
            'fast_processing': {
                'noise_reduction_level': 'mild',
                'batch_size': 128,
                'adaptive_mode': True,
                'skip_clean_audio': True,
                'progressive_enhancement': False
            },
            'aggressive_cleaning': {
                'noise_reduction_level': 'aggressive',
                'batch_size': 16,
                'adaptive_mode': False,
                'progressive_enhancement': True,
                'secondary_speaker_detection.suppression_strength': 0.9
            }
        }