"""
Configuration management UI for audio enhancement settings.
Provides interface for updating and validating enhancement configurations.
"""
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigurationUI:
    """Manages audio enhancement configuration settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration UI.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or "enhancement_config.json"
        self.current_config = self._load_default_config()
        
        # Load existing config if available
        if os.path.exists(self.config_file):
            self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            # Basic settings
            "enabled": True,
            "noise_reduction_level": "moderate",
            "gpu_device": 0,
            "batch_size": 32,
            "model_cache": "./models/denoiser",
            "fallback_enabled": True,
            "max_retries": 3,
            
            # Adaptive processing
            "adaptive_mode": True,
            "skip_clean_audio": True,
            "clean_audio_threshold": 30.0,  # SNR in dB
            "progressive_enhancement": True,
            
            # Secondary speaker settings
            "suppress_secondary_speakers": True,
            "secondary_speaker_config": {
                "min_duration": 0.1,
                "max_duration": 5.0,
                "speaker_similarity_threshold": 0.7,
                "suppression_strength": 0.6,
                "confidence_threshold": 0.5,
                "detection_methods": ["embedding", "vad", "energy"]
            },
            
            # Dashboard settings
            "show_dashboard": True,
            "dashboard_update_interval": 100,
            
            # Comparison settings
            "enable_comparison": True,
            "save_comparison_plots": True,
            "comparison_sample_rate": 0.01,
            
            # Quality targets
            "quality_targets": {
                "snr": 20.0,
                "pesq": 3.0,
                "stoi": 0.85
            },
            
            # Processing thresholds
            "noise_thresholds": {
                "mild": {"min_snr": 20, "max_snr": 30},
                "moderate": {"min_snr": 10, "max_snr": 20},
                "aggressive": {"min_snr": -10, "max_snr": 10}
            }
        }
    
    def load_config(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file."""
        filepath = filepath or self.config_file
        
        try:
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                self.current_config.update(loaded_config)
                return self.current_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.current_config
    
    def save_config(self, filepath: Optional[str] = None):
        """Save current configuration to file."""
        filepath = filepath or self.config_file
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_config, f, indent=2)
            print(f"Configuration saved to {filepath}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        # Deep update for nested dictionaries
        def deep_update(base: dict, update: dict):
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(self.current_config, updates)
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration values."""
        config = config or self.current_config
        
        try:
            # Validate noise reduction level
            valid_levels = ['mild', 'moderate', 'aggressive']
            if config.get('noise_reduction_level') not in valid_levels:
                print(f"Invalid noise_reduction_level. Must be one of {valid_levels}")
                return False
            
            # Validate batch size
            batch_size = config.get('batch_size', 32)
            if not isinstance(batch_size, int) or batch_size <= 0:
                print("Invalid batch_size. Must be a positive integer")
                return False
            
            # Validate GPU device
            gpu_device = config.get('gpu_device', 0)
            if not isinstance(gpu_device, int) or gpu_device < 0:
                print("Invalid gpu_device. Must be a non-negative integer")
                return False
            
            # Validate thresholds
            clean_threshold = config.get('clean_audio_threshold', 30.0)
            if not isinstance(clean_threshold, (int, float)) or clean_threshold < 0:
                print("Invalid clean_audio_threshold. Must be a non-negative number")
                return False
            
            # Validate secondary speaker config
            ss_config = config.get('secondary_speaker_config', {})
            if ss_config.get('min_duration', 0.1) >= ss_config.get('max_duration', 5.0):
                print("Invalid secondary speaker duration range")
                return False
            
            # Validate quality targets
            targets = config.get('quality_targets', {})
            for metric, value in targets.items():
                if not isinstance(value, (int, float)) or value < 0:
                    print(f"Invalid quality target for {metric}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        keys = key.split('.')
        value = self.current_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config_value(self, key: str, value: Any):
        """Set a configuration value by key."""
        keys = key.split('.')
        config = self.current_config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_noise_reduction_config(self, level: Optional[str] = None) -> Dict[str, Any]:
        """Get noise reduction configuration for a specific level."""
        level = level or self.current_config.get('noise_reduction_level', 'moderate')
        
        # Base configurations for each level
        configs = {
            "mild": {
                "denoiser_dry": 0.05,
                "spectral_gate_freq": 1000,
                "preserve_ratio": 0.9,
                "suppress_secondary_speakers": False,
                "vad_aggressiveness": 1
            },
            "moderate": {
                "denoiser_dry": 0.02,
                "spectral_gate_freq": 1500,
                "preserve_ratio": 0.7,
                "suppress_secondary_speakers": True,
                "vad_aggressiveness": 2,
                "secondary_detection": self.current_config.get('secondary_speaker_config', {})
            },
            "aggressive": {
                "denoiser_dry": 0.01,
                "spectral_gate_freq": 2000,
                "preserve_ratio": 0.5,
                "suppress_secondary_speakers": True,
                "vad_aggressiveness": 3,
                "secondary_detection": {
                    **self.current_config.get('secondary_speaker_config', {}),
                    "suppression_strength": 0.9,
                    "confidence_threshold": 0.3
                }
            }
        }
        
        return configs.get(level, configs['moderate'])
    
    def generate_cli_args(self) -> List[str]:
        """Generate command-line arguments from configuration."""
        args = []
        
        if self.current_config.get('enabled'):
            args.append('--enable-noise-reduction')
        
        args.extend([
            '--noise-reduction-level', self.current_config.get('noise_reduction_level', 'moderate'),
            '--batch-size', str(self.current_config.get('batch_size', 32)),
            '--gpu-device', str(self.current_config.get('gpu_device', 0))
        ])
        
        if self.current_config.get('adaptive_mode'):
            args.append('--adaptive-mode')
        
        if self.current_config.get('show_dashboard'):
            args.append('--show-dashboard')
        
        if self.current_config.get('enable_comparison'):
            args.append('--enable-comparison')
        
        # Secondary speaker settings
        ss_config = self.current_config.get('secondary_speaker_config', {})
        args.extend([
            '--secondary-min-duration', str(ss_config.get('min_duration', 0.1)),
            '--secondary-max-duration', str(ss_config.get('max_duration', 5.0)),
            '--speaker-similarity-threshold', str(ss_config.get('speaker_similarity_threshold', 0.7)),
            '--suppression-confidence', str(ss_config.get('confidence_threshold', 0.5))
        ])
        
        return args
    
    def generate_config_summary(self) -> str:
        """Generate a human-readable configuration summary."""
        summary = [
            "Audio Enhancement Configuration Summary",
            "=" * 40,
            f"Status: {'Enabled' if self.current_config.get('enabled') else 'Disabled'}",
            f"Noise Reduction Level: {self.current_config.get('noise_reduction_level')}",
            f"GPU Device: {self.current_config.get('gpu_device')}",
            f"Batch Size: {self.current_config.get('batch_size')}",
            "",
            "Features:",
            f"  Adaptive Mode: {'Yes' if self.current_config.get('adaptive_mode') else 'No'}",
            f"  Skip Clean Audio: {'Yes' if self.current_config.get('skip_clean_audio') else 'No'}",
            f"  Progressive Enhancement: {'Yes' if self.current_config.get('progressive_enhancement') else 'No'}",
            f"  Secondary Speaker Suppression: {'Yes' if self.current_config.get('suppress_secondary_speakers') else 'No'}",
            "",
            "Quality Targets:",
        ]
        
        targets = self.current_config.get('quality_targets', {})
        for metric, value in targets.items():
            summary.append(f"  {metric.upper()}: {value}")
        
        summary.extend([
            "",
            "Dashboard:",
            f"  Show Dashboard: {'Yes' if self.current_config.get('show_dashboard') else 'No'}",
            f"  Update Interval: {self.current_config.get('dashboard_update_interval')} files",
            "",
            "Comparison:",
            f"  Enable Comparison: {'Yes' if self.current_config.get('enable_comparison') else 'No'}",
            f"  Save Plots: {'Yes' if self.current_config.get('save_comparison_plots') else 'No'}",
            f"  Sample Rate: {self.current_config.get('comparison_sample_rate') * 100:.1f}%"
        ])
        
        return "\n".join(summary)
    
    def export_config(self, filepath: str, format: str = 'json'):
        """Export configuration in various formats."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.current_config, f, indent=2)
        elif format == 'yaml':
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(self.current_config, f, default_flow_style=False)
            except ImportError:
                print("PyYAML not installed. Using JSON format instead.")
                self.export_config(filepath, 'json')
        elif format == 'env':
            # Export as environment variables
            with open(filepath, 'w') as f:
                f.write("# Audio Enhancement Configuration\n")
                self._write_env_vars(f, self.current_config, prefix="AUDIO_ENHANCE")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _write_env_vars(self, f, config: dict, prefix: str):
        """Recursively write configuration as environment variables."""
        for key, value in config.items():
            env_key = f"{prefix}_{key.upper()}"
            if isinstance(value, dict):
                self._write_env_vars(f, value, env_key)
            elif isinstance(value, list):
                f.write(f'{env_key}="{",".join(map(str, value))}"\n')
            elif isinstance(value, bool):
                f.write(f"{env_key}={str(value).lower()}\n")
            else:
                f.write(f"{env_key}={value}\n")