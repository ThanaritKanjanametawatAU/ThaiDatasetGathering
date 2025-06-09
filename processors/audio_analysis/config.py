"""
Configuration loading and validation for the audio analysis system.
Supports YAML and JSON formats with schema validation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field

# Try to import pydantic for validation
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fall back to dataclasses if pydantic not available
    BaseModel = object
    Field = lambda **kwargs: None
    validator = lambda *args, **kwargs: lambda f: f

# Try to import optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not available - YAML configuration files will not be supported")

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logging.warning("jsonschema not available - configuration validation will be limited")

from .base import StageConfig, PipelineConfig, ValidationError

logger = logging.getLogger(__name__)


# JSON Schema for configuration validation
PIPELINE_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["pipeline"],
    "properties": {
        "pipeline": {
            "type": "object",
            "required": ["name", "version", "stages"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "author": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "stages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "processor"],
                        "properties": {
                            "name": {"type": "string"},
                            "processor": {"type": "string"},
                            "config": {"type": "object"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "error_handling": {
                                "type": "object",
                                "properties": {
                                    "strategy": {
                                        "type": "string",
                                        "enum": ["fail", "skip", "retry", "fallback"]
                                    },
                                    "max_attempts": {"type": "integer", "minimum": 1},
                                    "retry_delay": {"type": "number", "minimum": 0},
                                    "fallback_processor": {"type": "string"},
                                    "log_errors": {"type": "boolean"}
                                }
                            },
                            "parallel": {"type": "boolean"},
                            "required": {"type": "boolean"},
                            "timeout": {"type": "number", "minimum": 0}
                        }
                    }
                },
                "performance": {
                    "type": "object",
                    "properties": {
                        "max_memory_per_stage": {"type": "string"},
                        "timeout_per_stage": {"type": "number"},
                        "parallel_stages": {"type": "integer", "minimum": 1}
                    }
                }
            }
        },
        "plugins": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "auto_discover": {"type": "boolean"},
                "version_check": {
                    "type": "string",
                    "enum": ["strict", "compatible", "none"]
                },
                "compatibility_matrix": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }
}


# Use pydantic if available, otherwise fall back to dataclasses
if PYDANTIC_AVAILABLE:
    class PluginConfigPydantic(BaseModel):
        """Plugin system configuration with pydantic validation."""
        paths: List[str] = Field(default_factory=list)
        auto_discover: bool = Field(default=True)
        version_check: str = Field(default="compatible")
        compatibility_matrix: Dict[str, List[str]] = Field(default_factory=dict)
        
        @validator('version_check')
        def validate_version_check(cls, v):
            if v not in ["strict", "compatible", "none"]:
                raise ValueError(f"Invalid version_check: {v}")
            return v
    
    class PerformanceConfigPydantic(BaseModel):
        """Performance configuration with pydantic validation."""
        max_memory_per_stage: str = Field(default="2GB")
        timeout_per_stage: float = Field(default=300.0, gt=0)
        parallel_stages: int = Field(default=4, ge=1)
        
        @validator('max_memory_per_stage')
        def validate_memory_size(cls, v):
            # Validate memory size format
            import re
            if not re.match(r'^\d+(\.\d+)?(GB|MB|KB)$', v.upper()):
                raise ValueError(f"Invalid memory size format: {v}")
            return v
    
    # Import base types for pydantic model  
    from .base import PipelineConfig
    
    class ConfigPydantic(BaseModel):
        """Complete configuration with pydantic validation."""
        pipeline: PipelineConfig
        plugins: Optional[PluginConfigPydantic] = None
        performance: Optional[PerformanceConfigPydantic] = None
    
    # Use pydantic models
    PluginConfig = PluginConfigPydantic
    PerformanceConfig = PerformanceConfigPydantic
    Config = ConfigPydantic
else:
    # Fall back to dataclasses
    @dataclass
    class PluginConfig:
        """Plugin system configuration."""
        paths: List[str] = field(default_factory=list)
        auto_discover: bool = True
        version_check: str = "compatible"
        compatibility_matrix: Dict[str, List[str]] = field(default_factory=dict)


    @dataclass
    class PerformanceConfig:
        """Performance configuration."""
        max_memory_per_stage: str = "2GB"
        timeout_per_stage: float = 300.0
        parallel_stages: int = 4


    @dataclass
    class Config:
        """Complete configuration."""
        pipeline: PipelineConfig
        plugins: Optional[PluginConfig] = None
        performance: Optional[PerformanceConfig] = None


class ConfigLoader:
    """
    Loads and validates configuration files.
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize config loader.
        
        Args:
            schema: JSON schema for validation (uses default if None)
        """
        self.schema = schema or PIPELINE_CONFIG_SCHEMA
    
    def load(self, config_path: Union[str, Path]) -> Config:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load raw configuration
        raw_config = self._load_raw_config(config_path)
        
        # Validate against schema
        self._validate_config(raw_config)
        
        # Apply defaults
        config_with_defaults = self._apply_defaults(raw_config)
        
        # Convert to dataclasses
        return self._parse_config(config_with_defaults)
    
    def _load_raw_config(self, config_path: Path) -> Dict[str, Any]:
        """Load raw configuration from file."""
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to load YAML configuration files")
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                # Try to detect format
                content = f.read()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    if YAML_AVAILABLE:
                        return yaml.safe_load(content)
                    else:
                        raise ValueError("Unable to parse configuration file (PyYAML not available)")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(config, self.schema)
            except jsonschema.ValidationError as e:
                raise ValidationError(f"Configuration validation failed: {e.message}", 
                                    field='.'.join(str(p) for p in e.path))
        else:
            # Basic validation without jsonschema
            if 'pipeline' not in config:
                raise ValidationError("Missing required field: pipeline")
            pipeline = config['pipeline']
            if 'name' not in pipeline:
                raise ValidationError("Missing required field: pipeline.name")
            if 'version' not in pipeline:
                raise ValidationError("Missing required field: pipeline.version")
            if 'stages' not in pipeline:
                raise ValidationError("Missing required field: pipeline.stages")
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration."""
        # Apply pipeline defaults
        pipeline = config.get('pipeline', {})
        pipeline.setdefault('description', '')
        pipeline.setdefault('author', '')
        pipeline.setdefault('tags', [])
        pipeline.setdefault('performance', {})
        
        # Apply stage defaults
        for stage in pipeline.get('stages', []):
            stage.setdefault('config', {})
            stage.setdefault('dependencies', [])
            stage.setdefault('error_handling', {
                'strategy': 'fail',
                'log_errors': True
            })
            stage.setdefault('parallel', False)
            stage.setdefault('required', True)
            stage.setdefault('timeout', None)
        
        # Apply plugin defaults
        if 'plugins' in config:
            plugins = config['plugins']
            plugins.setdefault('paths', [])
            plugins.setdefault('auto_discover', True)
            plugins.setdefault('version_check', 'compatible')
            plugins.setdefault('compatibility_matrix', {})
        
        return config
    
    def _parse_config(self, config: Dict[str, Any]) -> Config:
        """Parse configuration into dataclasses."""
        # Parse pipeline config
        pipeline_data = config['pipeline']
        stages = [
            StageConfig(
                name=stage['name'],
                processor=stage['processor'],
                config=stage['config'],
                dependencies=stage['dependencies'],
                error_handling=stage['error_handling'],
                parallel=stage['parallel'],
                required=stage['required'],
                timeout=stage['timeout']
            )
            for stage in pipeline_data['stages']
        ]
        
        pipeline = PipelineConfig(
            name=pipeline_data['name'],
            version=pipeline_data['version'],
            stages=stages,
            description=pipeline_data['description'],
            author=pipeline_data['author'],
            tags=pipeline_data['tags'],
            performance=pipeline_data['performance']
        )
        
        # Parse plugin config
        plugins = None
        if 'plugins' in config:
            plugin_data = config['plugins']
            plugins = PluginConfig(
                paths=plugin_data['paths'],
                auto_discover=plugin_data['auto_discover'],
                version_check=plugin_data['version_check'],
                compatibility_matrix=plugin_data['compatibility_matrix']
            )
        
        # Parse performance config
        performance = None
        if 'performance' in pipeline_data:
            perf_data = pipeline_data['performance']
            performance = PerformanceConfig(
                max_memory_per_stage=perf_data.get('max_memory_per_stage', '2GB'),
                timeout_per_stage=perf_data.get('timeout_per_stage', 300.0),
                parallel_stages=perf_data.get('parallel_stages', 4)
            )
        
        return Config(
            pipeline=pipeline,
            plugins=plugins,
            performance=performance
        )
    
    def validate_file(self, config_path: Union[str, Path]) -> List[str]:
        """
        Validate a configuration file and return errors.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                errors.append(f"File not found: {config_path}")
                return errors
            
            raw_config = self._load_raw_config(config_path)
            self._validate_config(raw_config)
            
        except ValidationError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Failed to load configuration: {str(e)}")
        
        return errors


class ConfigBuilder:
    """
    Builder for creating configurations programmatically.
    """
    
    def __init__(self):
        """Initialize config builder."""
        self._pipeline_name = "unnamed_pipeline"
        self._pipeline_version = "1.0.0"
        self._description = ""
        self._author = ""
        self._tags = []
        self._stages = []
        self._plugin_paths = []
        self._performance = {}
    
    def with_pipeline(self, name: str, version: str) -> 'ConfigBuilder':
        """Set pipeline name and version."""
        self._pipeline_name = name
        self._pipeline_version = version
        return self
    
    def with_description(self, description: str) -> 'ConfigBuilder':
        """Set pipeline description."""
        self._description = description
        return self
    
    def with_author(self, author: str) -> 'ConfigBuilder':
        """Set pipeline author."""
        self._author = author
        return self
    
    def with_tags(self, tags: List[str]) -> 'ConfigBuilder':
        """Set pipeline tags."""
        self._tags = tags
        return self
    
    def add_stage(self, name: str, processor: str,
                  config: Optional[Dict[str, Any]] = None,
                  dependencies: Optional[List[str]] = None,
                  **kwargs) -> 'ConfigBuilder':
        """Add a pipeline stage."""
        stage = StageConfig(
            name=name,
            processor=processor,
            config=config or {},
            dependencies=dependencies or [],
            **kwargs
        )
        self._stages.append(stage)
        return self
    
    def with_plugin_paths(self, paths: List[str]) -> 'ConfigBuilder':
        """Set plugin search paths."""
        self._plugin_paths = paths
        return self
    
    def with_performance(self, **kwargs) -> 'ConfigBuilder':
        """Set performance configuration."""
        self._performance.update(kwargs)
        return self
    
    def build(self) -> Config:
        """Build the configuration."""
        pipeline = PipelineConfig(
            name=self._pipeline_name,
            version=self._pipeline_version,
            stages=self._stages,
            description=self._description,
            author=self._author,
            tags=self._tags,
            performance=self._performance
        )
        
        plugins = PluginConfig(paths=self._plugin_paths) if self._plugin_paths else None
        
        performance = PerformanceConfig(**self._performance) if self._performance else None
        
        return Config(
            pipeline=pipeline,
            plugins=plugins,
            performance=performance
        )
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to save YAML configuration files")
        
        config = self.build()
        config_dict = self._config_to_dict(config)
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config = self.build()
        config_dict = self._config_to_dict(config)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert config dataclasses to dictionary."""
        result = {
            'pipeline': {
                'name': config.pipeline.name,
                'version': config.pipeline.version,
                'description': config.pipeline.description,
                'author': config.pipeline.author,
                'tags': config.pipeline.tags,
                'stages': [
                    {
                        'name': stage.name,
                        'processor': stage.processor,
                        'config': stage.config,
                        'dependencies': stage.dependencies,
                        'error_handling': stage.error_handling,
                        'parallel': stage.parallel,
                        'required': stage.required,
                        'timeout': stage.timeout
                    }
                    for stage in config.pipeline.stages
                ],
                'performance': config.pipeline.performance
            }
        }
        
        if config.plugins:
            result['plugins'] = {
                'paths': config.plugins.paths,
                'auto_discover': config.plugins.auto_discover,
                'version_check': config.plugins.version_check,
                'compatibility_matrix': config.plugins.compatibility_matrix
            }
        
        return result