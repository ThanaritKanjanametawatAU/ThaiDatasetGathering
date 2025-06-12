# Task 22 - S07: Configuration CLI Integration for Pattern→MetricGAN+ Enhancement

## Overview
Integrate the Pattern→MetricGAN+ → 160% loudness enhancement method into the configuration management system and CLI interface, making it accessible through standard configuration files and command-line arguments while maintaining backward compatibility with existing enhancement levels.

## Background
The Pattern→MetricGAN+ approach (Task 21) provides proven audio enhancement capabilities:
- **Ultra-conservative pattern detection** (>0.8 confidence threshold)
- **Gentle suppression** (50ms padding, 85% suppression, keep 15% of original)
- **MetricGAN+ enhancement** for overall quality improvement
- **160% loudness normalization** to match original levels

This task ensures the new enhancement method is fully accessible through the existing CLI and configuration infrastructure used by dataset processors (gigaspeech2.py, processed_voice_th.py, mozilla_cv.py).

## Architectural Guidance
This task follows established architectural decisions documented in the project:

### Referenced Architecture Documents:
- **`config.py`**: Centralized configuration pattern - All enhancement parameters must be centrally managed
- **`audio_config.yaml`**: YAML-based configuration schema with strict validation
- **`main.py`**: CLI argument parsing patterns and enhancement level integration
- **`processors/base_processor.py`**: Configuration access patterns used by all processors
- **`docs/architecture.md`**: Configuration-driven architecture principle

### Key Architectural Constraints:
- **Centralized Config**: Must follow existing ENHANCEMENT_LEVELS configuration pattern
- **Schema Validation**: All new configurations must have proper validation
- **Environment Variables**: Support both CLI and environment variable configuration
- **Backward Compatibility**: Existing configurations must continue working unchanged
- **CLI Integration**: Must extend existing --enhancement-level argument choices

## Technical Requirements

### 1. Configuration System Integration

#### 1.1 Enhancement Level Registration
**File: `config.py`**

Add Pattern→MetricGAN+ configuration to existing NOISE_REDUCTION_CONFIG structure:

```python
# Update existing NOISE_REDUCTION_CONFIG
NOISE_REDUCTION_CONFIG = {
    "enabled": False,
    "device": "cuda",
    "adaptive_mode": True,
    "default_level": "moderate",
    "levels": {
        "mild": {
            "dry_wet_ratio": 0.1,
            "prop_decrease": 0.6,
            "target_snr": 20
        },
        "moderate": {
            "dry_wet_ratio": 0.05,
            "prop_decrease": 0.8,
            "target_snr": 25
        },
        "aggressive": {
            "dry_wet_ratio": 0.02,
            "prop_decrease": 1.0,
            "target_snr": 30
        },
        # NEW: Add Pattern→MetricGAN+ level
        "pattern_metricgan_plus": {
            "dry_wet_ratio": 0.0,    # Full processing
            "prop_decrease": 1.0,    # Maximum enhancement
            "target_snr": 35,        # High quality target
            "use_pattern_detection": True,
            "pattern_confidence_threshold": 0.8,
            "pattern_suppression_factor": 0.15,  # Keep 15%
            "pattern_padding_ms": 50,
            "use_metricgan": True,
            "apply_loudness_normalization": True,
            "target_loudness_multiplier": 1.6,  # 160%
            "passes": 1
        }
    },
    # ... existing configuration continues
}

# NEW: Add dedicated Pattern→MetricGAN+ configuration section
PATTERN_METRICGAN_CONFIG = {
    "enabled": False,  # Controlled by enhancement level selection
    "pattern_detection": {
        "confidence_threshold": 0.8,
        "energy_threshold_percentile": 75,
        "zcr_threshold_percentile": 80,
        "spectral_threshold_percentile": 70,
        "context_energy_multiplier": 2.0,
        "min_interruption_duration_ms": 100,
        "max_interruption_duration_ms": 3000
    },
    "pattern_suppression": {
        "padding_ms": 50,
        "suppression_factor": 0.15,  # Keep 15% of original
        "min_gap_seconds": 0.2,
        "fade_in_out_ms": 10,  # Smooth transitions
        "preserve_primary_speaker": True
    },
    "metricgan": {
        "model_source": "speechbrain/metricgan-plus-voicebank",
        "device": "auto",  # auto, cuda, cpu
        "batch_size": 1,
        "cache_model": True,
        "fallback_to_cpu": True
    },
    "loudness_enhancement": {
        "target_multiplier": 1.6,  # 160% of original
        "method": "rms",           # rms, peak, lufs
        "headroom_db": -1.0,       # Prevent clipping
        "soft_limit": True,
        "normalize_before_enhancement": True
    },
    "quality_validation": {
        "min_pesq_score": 2.5,
        "min_stoi_score": 0.80,
        "max_spectral_distortion": 0.2,
        "validate_pattern_suppression": True
    }
}
```

#### 1.2 Configuration Validation
**File: `config.py`**

Add validation rules for new configuration options:

```python
# Update VALIDATION_RULES with Pattern→MetricGAN+ validation
PATTERN_METRICGAN_VALIDATION = {
    "pattern_confidence_threshold": {
        "type": float,
        "min_value": 0.5,
        "max_value": 1.0,
        "error_message": "Pattern confidence threshold must be between 0.5 and 1.0"
    },
    "pattern_suppression_factor": {
        "type": float,
        "min_value": 0.0,
        "max_value": 1.0,
        "error_message": "Pattern suppression factor must be between 0.0 and 1.0"
    },
    "target_loudness_multiplier": {
        "type": float,
        "min_value": 1.0,
        "max_value": 3.0,
        "error_message": "Target loudness multiplier must be between 1.0 and 3.0"
    },
    "pattern_padding_ms": {
        "type": int,
        "min_value": 0,
        "max_value": 200,
        "error_message": "Pattern padding must be between 0 and 200 milliseconds"
    }
}
```

### 2. CLI Interface Integration

#### 2.1 Command-Line Arguments
**File: `main.py`**

Extend the existing enhancement argument group:

```python
# Update enhancement_group.add_argument for enhancement_level
enhancement_group.add_argument(
    '--enhancement-level',
    type=str,
    choices=['mild', 'moderate', 'aggressive', 'ultra_aggressive', 
             'selective_secondary_removal', 'pattern_metricgan_plus'],  # ADD NEW CHOICE
    default='moderate',
    help='Enhancement level: mild, moderate, aggressive, ultra_aggressive, selective_secondary_removal, or pattern_metricgan_plus (default: moderate)'
)

# NEW: Add Pattern→MetricGAN+ specific arguments
pattern_metricgan_group = parser.add_argument_group('Pattern→MetricGAN+ Enhancement')
pattern_metricgan_group.add_argument(
    '--pattern-confidence-threshold',
    type=float,
    default=0.8,
    help='Confidence threshold for pattern detection (default: 0.8)'
)
pattern_metricgan_group.add_argument(
    '--pattern-suppression-factor',
    type=float,
    default=0.15,
    help='Factor for pattern suppression - lower values suppress more (default: 0.15)'
)
pattern_metricgan_group.add_argument(
    '--pattern-padding-ms',
    type=int,
    default=50,
    help='Padding around detected patterns in milliseconds (default: 50)'
)
pattern_metricgan_group.add_argument(
    '--loudness-multiplier',
    type=float,
    default=1.6,
    help='Target loudness multiplier for enhancement (default: 1.6 = 160%%)'
)
pattern_metricgan_group.add_argument(
    '--disable-metricgan',
    action='store_true',
    help='Disable MetricGAN+ processing (use only pattern suppression and loudness)'
)
pattern_metricgan_group.add_argument(
    '--metricgan-device',
    type=str,
    choices=['auto', 'cuda', 'cpu'],
    default='auto',
    help='Device for MetricGAN+ processing (default: auto)'
)
```

#### 2.2 Configuration Processing
**File: `main.py`**

Update processor configuration building to include Pattern→MetricGAN+ settings:

```python
# In create_processor() function, update processor_config building
def create_processor(dataset_name: str, config: Dict[str, Any]) -> BaseProcessor:
    """Create a processor instance for the specified dataset."""
    # ... existing code ...
    
    # Merge dataset config with global config
    merged_config = {**config, **dataset_config}
    
    # NEW: Add Pattern→MetricGAN+ configuration if enabled
    if merged_config.get("enhancement_level") == "pattern_metricgan_plus":
        from config import PATTERN_METRICGAN_CONFIG
        
        # Override with command-line arguments if provided
        pattern_config = dict(PATTERN_METRICGAN_CONFIG)
        
        # Update with CLI arguments
        if "pattern_confidence_threshold" in merged_config:
            pattern_config["pattern_detection"]["confidence_threshold"] = merged_config["pattern_confidence_threshold"]
        if "pattern_suppression_factor" in merged_config:
            pattern_config["pattern_suppression"]["suppression_factor"] = merged_config["pattern_suppression_factor"]
        if "pattern_padding_ms" in merged_config:
            pattern_config["pattern_suppression"]["padding_ms"] = merged_config["pattern_padding_ms"]
        if "loudness_multiplier" in merged_config:
            pattern_config["loudness_enhancement"]["target_multiplier"] = merged_config["loudness_multiplier"]
        if "disable_metricgan" in merged_config:
            pattern_config["metricgan"]["enabled"] = not merged_config["disable_metricgan"]
        if "metricgan_device" in merged_config:
            pattern_config["metricgan"]["device"] = merged_config["metricgan_device"]
        
        merged_config["pattern_metricgan_config"] = pattern_config
        logger.info(f"Pattern→MetricGAN+ configuration applied for {dataset_name}")
    
    return processor_class(merged_config)

# In process_streaming_mode() function, update processor_config
def process_streaming_mode(args, dataset_names: List[str]) -> int:
    """Process datasets in streaming mode without full download."""
    # ... existing code ...
    
    # Create processor config
    processor_config = {
        "checkpoint_dir": CHECKPOINT_DIR,
        "log_dir": LOG_DIR,
        "streaming": True,
        "batch_size": args.streaming_batch_size,
        "upload_batch_size": args.upload_batch_size,
        "audio_config": {
            "enable_standardization": not args.no_standardization,
            "target_sample_rate": args.sample_rate,
            "target_channels": 1,
            "normalize_volume": not args.no_volume_norm,
            "target_db": args.target_db,
        },
        "dataset_name": dataset_name,
        "enable_stt": args.enable_stt if not args.no_stt else False,
        "stt_batch_size": args.stt_batch_size,
        "audio_enhancement": {
            "enabled": args.enable_audio_enhancement,
            "enhancer": audio_enhancer,
            "metrics_collector": enhancement_metrics_collector,
            "dashboard": enhancement_dashboard,
            "batch_size": args.enhancement_batch_size if args.enable_audio_enhancement else None,
            "level": args.enhancement_level if args.enable_audio_enhancement else None
        },
        # NEW: Add Pattern→MetricGAN+ specific configuration
        "enhancement_level": args.enhancement_level,
        "pattern_confidence_threshold": getattr(args, 'pattern_confidence_threshold', 0.8),
        "pattern_suppression_factor": getattr(args, 'pattern_suppression_factor', 0.15),
        "pattern_padding_ms": getattr(args, 'pattern_padding_ms', 50),
        "loudness_multiplier": getattr(args, 'loudness_multiplier', 1.6),
        "disable_metricgan": getattr(args, 'disable_metricgan', False),
        "metricgan_device": getattr(args, 'metricgan_device', 'auto')
    }
    
    # ... rest of function continues
```

### 3. Environment Variable Integration

#### 3.1 Environment Variable Support
**File: `config.py`**

Add environment variable support for Pattern→MetricGAN+ configuration:

```python
# NEW: Environment variable configuration loading
def load_pattern_metricgan_config_from_env():
    """Load Pattern→MetricGAN+ configuration from environment variables."""
    env_config = {}
    
    # Pattern detection environment variables
    if os.getenv('PATTERN_CONFIDENCE_THRESHOLD'):
        env_config.setdefault('pattern_detection', {})['confidence_threshold'] = float(os.getenv('PATTERN_CONFIDENCE_THRESHOLD'))
    
    if os.getenv('PATTERN_ENERGY_THRESHOLD'):
        env_config.setdefault('pattern_detection', {})['energy_threshold_percentile'] = int(os.getenv('PATTERN_ENERGY_THRESHOLD'))
    
    # Pattern suppression environment variables
    if os.getenv('PATTERN_SUPPRESSION_FACTOR'):
        env_config.setdefault('pattern_suppression', {})['suppression_factor'] = float(os.getenv('PATTERN_SUPPRESSION_FACTOR'))
        
    if os.getenv('PATTERN_PADDING_MS'):
        env_config.setdefault('pattern_suppression', {})['padding_ms'] = int(os.getenv('PATTERN_PADDING_MS'))
    
    # MetricGAN environment variables
    if os.getenv('METRICGAN_DEVICE'):
        env_config.setdefault('metricgan', {})['device'] = os.getenv('METRICGAN_DEVICE')
        
    if os.getenv('METRICGAN_BATCH_SIZE'):
        env_config.setdefault('metricgan', {})['batch_size'] = int(os.getenv('METRICGAN_BATCH_SIZE'))
    
    # Loudness enhancement environment variables
    if os.getenv('LOUDNESS_MULTIPLIER'):
        env_config.setdefault('loudness_enhancement', {})['target_multiplier'] = float(os.getenv('LOUDNESS_MULTIPLIER'))
        
    if os.getenv('LOUDNESS_METHOD'):
        env_config.setdefault('loudness_enhancement', {})['method'] = os.getenv('LOUDNESS_METHOD')
    
    return env_config

# Update PATTERN_METRICGAN_CONFIG with environment overrides
_env_pattern_config = load_pattern_metricgan_config_from_env()
if _env_pattern_config:
    def deep_update(base_dict, update_dict):
        """Deep update dictionary with nested updates."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(PATTERN_METRICGAN_CONFIG, _env_pattern_config)
```

#### 3.2 Environment Variable Documentation
**File: `config.py`**

Add documentation for supported environment variables:

```python
"""
Pattern→MetricGAN+ Environment Variables:

Pattern Detection:
- PATTERN_CONFIDENCE_THRESHOLD: Confidence threshold (0.5-1.0, default: 0.8)
- PATTERN_ENERGY_THRESHOLD: Energy threshold percentile (50-95, default: 75)

Pattern Suppression:
- PATTERN_SUPPRESSION_FACTOR: Suppression factor (0.0-1.0, default: 0.15)
- PATTERN_PADDING_MS: Padding in milliseconds (0-200, default: 50)

MetricGAN:
- METRICGAN_DEVICE: Processing device (auto/cuda/cpu, default: auto)
- METRICGAN_BATCH_SIZE: Batch size (1-8, default: 1)

Loudness Enhancement:
- LOUDNESS_MULTIPLIER: Target multiplier (1.0-3.0, default: 1.6)
- LOUDNESS_METHOD: Enhancement method (rms/peak/lufs, default: rms)

Example usage:
    export PATTERN_CONFIDENCE_THRESHOLD=0.9
    export LOUDNESS_MULTIPLIER=1.8
    python main.py --streaming --enhancement-level pattern_metricgan_plus gigaspeech2
"""
```

### 4. Dataset Processor Integration

#### 4.1 BaseProcessor Configuration Support
**File: `processors/base_processor.py`**

Extend BaseProcessor to support Pattern→MetricGAN+ configuration:

```python
# In BaseProcessor.__init__(), add Pattern→MetricGAN+ support
def __init__(self, config: Dict[str, Any]):
    """Initialize base processor."""
    # ... existing code ...
    
    # NEW: Pattern→MetricGAN+ configuration
    self.pattern_metricgan_config = config.get("pattern_metricgan_config", {})
    self.enhancement_level = config.get("enhancement_level", "moderate")
    
    # Initialize Pattern→MetricGAN+ if enabled
    if self.enhancement_level == "pattern_metricgan_plus":
        self._initialize_pattern_metricgan_enhancement()
        
    # ... rest of initialization

def _initialize_pattern_metricgan_enhancement(self):
    """Initialize Pattern→MetricGAN+ enhancement pipeline."""
    try:
        # Validate configuration
        self._validate_pattern_metricgan_config()
        
        # Set enhancement flags
        self.noise_reduction_enabled = True
        
        # Load Pattern→MetricGAN+ configuration into audio enhancement
        if not self.audio_enhancer and ENHANCEMENT_AVAILABLE:
            self._initialize_audio_enhancer()
            
        # Override enhancement level in enhancer
        if self.audio_enhancer:
            self.audio_enhancer.enhancement_level = "pattern_metricgan_plus"
            
        self.logger.info("Pattern→MetricGAN+ enhancement initialized successfully")
        
    except Exception as e:
        self.logger.error(f"Failed to initialize Pattern→MetricGAN+ enhancement: {e}")
        # Fall back to standard enhancement
        self.enhancement_level = "moderate"
        self.pattern_metricgan_config = {}

def _validate_pattern_metricgan_config(self):
    """Validate Pattern→MetricGAN+ configuration parameters."""
    config = self.pattern_metricgan_config
    
    # Validate pattern detection config
    pattern_detection = config.get("pattern_detection", {})
    confidence_threshold = pattern_detection.get("confidence_threshold", 0.8)
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError(f"Pattern confidence threshold {confidence_threshold} must be between 0.5 and 1.0")
    
    # Validate pattern suppression config
    pattern_suppression = config.get("pattern_suppression", {})
    suppression_factor = pattern_suppression.get("suppression_factor", 0.15)
    if not 0.0 <= suppression_factor <= 1.0:
        raise ValueError(f"Pattern suppression factor {suppression_factor} must be between 0.0 and 1.0")
    
    # Validate loudness enhancement config
    loudness_enhancement = config.get("loudness_enhancement", {})
    target_multiplier = loudness_enhancement.get("target_multiplier", 1.6)
    if not 1.0 <= target_multiplier <= 3.0:
        raise ValueError(f"Loudness multiplier {target_multiplier} must be between 1.0 and 3.0")
    
    self.logger.info("Pattern→MetricGAN+ configuration validated successfully")
```

#### 4.2 Enhanced Metadata Collection
**File: `processors/base_processor.py`**

Extend metadata collection for Pattern→MetricGAN+ processing:

```python
# Update _apply_noise_reduction_with_metadata() to include Pattern→MetricGAN+ metadata
def _apply_noise_reduction_with_metadata(self, audio_data: bytes, sample_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    """Apply noise reduction to audio data and return metadata."""
    if not self.audio_enhancer:
        return None
        
    try:
        # ... existing audio loading code ...
        
        # Apply enhancement with Pattern→MetricGAN+ configuration
        enhance_kwargs = {}
        if self.enhancement_level == "pattern_metricgan_plus":
            enhance_kwargs.update({
                'pattern_metricgan_config': self.pattern_metricgan_config,
                'enhancement_level': 'pattern_metricgan_plus',
                'return_detailed_metadata': True
            })
        
        # Apply enhancement
        enhanced_array, metadata = self.audio_enhancer.enhance(
            audio_array, sample_rate, return_metadata=True, **enhance_kwargs
        )
        
        # Add Pattern→MetricGAN+ specific metadata
        if self.enhancement_level == "pattern_metricgan_plus":
            metadata.update({
                'enhancement_method': 'pattern_metricgan_plus',
                'pattern_detection_used': True,
                'metricgan_applied': metadata.get('metricgan_applied', False),
                'loudness_enhanced': metadata.get('loudness_enhanced', False),
                'configuration_source': 'cli' if hasattr(self, '_from_cli') else 'config'
            })
        
        # ... rest of method continues ...
        
    except Exception as e:
        self.logger.error(f"Audio enhancement failed for {sample_id}: {e}")
        return None
```

### 5. Backward Compatibility Requirements

#### 5.1 Configuration Migration
**File: `config.py`**

Ensure backward compatibility with existing configurations:

```python
def validate_enhancement_level_compatibility(enhancement_level: str) -> str:
    """Validate and migrate enhancement level for backward compatibility."""
    
    # Map old enhancement levels to new ones if needed
    level_mapping = {
        'noise_reduction': 'moderate',          # Legacy mapping
        'advanced': 'aggressive',               # Legacy mapping
        'pattern_metricgan': 'pattern_metricgan_plus'  # Forward compatibility
    }
    
    if enhancement_level in level_mapping:
        logger.warning(f"Enhancement level '{enhancement_level}' deprecated, using '{level_mapping[enhancement_level]}'")
        return level_mapping[enhancement_level]
    
    # Validate against current levels
    valid_levels = list(NOISE_REDUCTION_CONFIG['levels'].keys())
    if enhancement_level not in valid_levels:
        logger.warning(f"Unknown enhancement level '{enhancement_level}', falling back to 'moderate'")
        return 'moderate'
    
    return enhancement_level
```

#### 5.2 CLI Argument Compatibility
**File: `main.py`**

Maintain compatibility with existing CLI usage patterns:

```python
# In parse_arguments(), add compatibility handling
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    # ... existing argument parsing ...
    
    args = parser.parse_args()
    
    # NEW: Handle Pattern→MetricGAN+ compatibility and validation
    if hasattr(args, 'enhancement_level'):
        from config import validate_enhancement_level_compatibility
        args.enhancement_level = validate_enhancement_level_compatibility(args.enhancement_level)
        
        # Auto-enable audio enhancement for Pattern→MetricGAN+
        if args.enhancement_level == 'pattern_metricgan_plus':
            args.enable_audio_enhancement = True
            logger.info("Auto-enabled audio enhancement for Pattern→MetricGAN+ level")
    
    # Validate Pattern→MetricGAN+ specific arguments
    if getattr(args, 'pattern_confidence_threshold', None) is not None:
        if not 0.5 <= args.pattern_confidence_threshold <= 1.0:
            parser.error("Pattern confidence threshold must be between 0.5 and 1.0")
    
    if getattr(args, 'pattern_suppression_factor', None) is not None:
        if not 0.0 <= args.pattern_suppression_factor <= 1.0:
            parser.error("Pattern suppression factor must be between 0.0 and 1.0")
    
    if getattr(args, 'loudness_multiplier', None) is not None:
        if not 1.0 <= args.loudness_multiplier <= 3.0:
            parser.error("Loudness multiplier must be between 1.0 and 3.0")
    
    return args
```

### 6. Testing Strategy (TDD Approach)

#### 6.1 Configuration Testing
**File: `tests/test_pattern_metricgan_configuration.py`**

```python
import unittest
import os
from unittest.mock import patch
from config import PATTERN_METRICGAN_CONFIG, load_pattern_metricgan_config_from_env, validate_enhancement_level_compatibility


class TestPatternMetricGANConfiguration(unittest.TestCase):
    """Test Pattern→MetricGAN+ configuration system."""
    
    def test_default_configuration_structure(self):
        """Test default configuration has required structure."""
        config = PATTERN_METRICGAN_CONFIG
        
        # Test required sections exist
        self.assertIn('pattern_detection', config)
        self.assertIn('pattern_suppression', config)
        self.assertIn('metricgan', config)
        self.assertIn('loudness_enhancement', config)
        
        # Test required parameters exist
        self.assertIn('confidence_threshold', config['pattern_detection'])
        self.assertIn('suppression_factor', config['pattern_suppression'])
        self.assertIn('target_multiplier', config['loudness_enhancement'])
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'PATTERN_CONFIDENCE_THRESHOLD': '0.9',
            'PATTERN_SUPPRESSION_FACTOR': '0.1',
            'LOUDNESS_MULTIPLIER': '1.8'
        }):
            env_config = load_pattern_metricgan_config_from_env()
            
            self.assertEqual(env_config['pattern_detection']['confidence_threshold'], 0.9)
            self.assertEqual(env_config['pattern_suppression']['suppression_factor'], 0.1)
            self.assertEqual(env_config['loudness_enhancement']['target_multiplier'], 1.8)
    
    def test_enhancement_level_compatibility(self):
        """Test enhancement level backward compatibility."""
        # Test valid current level
        self.assertEqual(validate_enhancement_level_compatibility('moderate'), 'moderate')
        
        # Test legacy mapping
        self.assertEqual(validate_enhancement_level_compatibility('pattern_metricgan'), 'pattern_metricgan_plus')
        
        # Test invalid level fallback
        self.assertEqual(validate_enhancement_level_compatibility('invalid_level'), 'moderate')
    
    def test_configuration_validation_bounds(self):
        """Test configuration parameter bounds validation."""
        from processors.base_processor import BaseProcessor
        
        # Test valid configuration
        valid_config = {
            "pattern_metricgan_config": {
                "pattern_detection": {"confidence_threshold": 0.8},
                "pattern_suppression": {"suppression_factor": 0.15},
                "loudness_enhancement": {"target_multiplier": 1.6}
            },
            "enhancement_level": "pattern_metricgan_plus"
        }
        
        processor = BaseProcessor(valid_config)
        # Should not raise exception
        processor._validate_pattern_metricgan_config()
        
        # Test invalid confidence threshold
        invalid_config = valid_config.copy()
        invalid_config["pattern_metricgan_config"]["pattern_detection"]["confidence_threshold"] = 1.5
        
        processor = BaseProcessor(invalid_config)
        with self.assertRaises(ValueError):
            processor._validate_pattern_metricgan_config()
```

#### 6.2 CLI Integration Testing
**File: `tests/test_pattern_metricgan_cli_integration.py`**

```python
import unittest
import argparse
from unittest.mock import patch, MagicMock
from main import parse_arguments, create_processor


class TestPatternMetricGANCLIIntegration(unittest.TestCase):
    """Test Pattern→MetricGAN+ CLI integration."""
    
    def test_enhancement_level_argument_parsing(self):
        """Test enhancement level argument parsing."""
        with patch('sys.argv', ['main.py', '--fresh', '--enhancement-level', 'pattern_metricgan_plus', 'gigaspeech2']):
            args = parse_arguments()
            self.assertEqual(args.enhancement_level, 'pattern_metricgan_plus')
            self.assertTrue(args.enable_audio_enhancement)
    
    def test_pattern_metricgan_specific_arguments(self):
        """Test Pattern→MetricGAN+ specific arguments."""
        with patch('sys.argv', [
            'main.py', '--fresh', '--enhancement-level', 'pattern_metricgan_plus',
            '--pattern-confidence-threshold', '0.9',
            '--pattern-suppression-factor', '0.1',
            '--loudness-multiplier', '1.8',
            'gigaspeech2'
        ]):
            args = parse_arguments()
            self.assertEqual(args.pattern_confidence_threshold, 0.9)
            self.assertEqual(args.pattern_suppression_factor, 0.1)
            self.assertEqual(args.loudness_multiplier, 1.8)
    
    def test_argument_validation(self):
        """Test CLI argument validation."""
        # Test invalid confidence threshold
        with patch('sys.argv', [
            'main.py', '--fresh', '--pattern-confidence-threshold', '1.5', 'gigaspeech2'
        ]):
            with self.assertRaises(SystemExit):
                parse_arguments()
    
    def test_processor_configuration_building(self):
        """Test processor configuration includes Pattern→MetricGAN+ settings."""
        mock_args = MagicMock()
        mock_args.enhancement_level = 'pattern_metricgan_plus'
        mock_args.pattern_confidence_threshold = 0.9
        mock_args.pattern_suppression_factor = 0.1
        mock_args.loudness_multiplier = 1.8
        
        config = {
            "enhancement_level": "pattern_metricgan_plus",
            "pattern_confidence_threshold": 0.9,
            "pattern_suppression_factor": 0.1,
            "loudness_multiplier": 1.8
        }
        
        with patch('main.get_processor_class') as mock_get_class:
            mock_processor_class = MagicMock()
            mock_get_class.return_value = mock_processor_class
            
            processor = create_processor("GigaSpeech2", config)
            
            # Verify processor was called with Pattern→MetricGAN+ config
            call_args = mock_processor_class.call_args[0][0]
            self.assertIn('pattern_metricgan_config', call_args)
            pattern_config = call_args['pattern_metricgan_config']
            self.assertEqual(pattern_config['pattern_detection']['confidence_threshold'], 0.9)
```

#### 6.3 Integration Testing
**File: `tests/test_pattern_metricgan_processor_integration.py`**

```python
import unittest
from unittest.mock import patch, MagicMock
from processors.base_processor import BaseProcessor


class TestPatternMetricGANProcessorIntegration(unittest.TestCase):
    """Test Pattern→MetricGAN+ integration with processors."""
    
    def test_base_processor_pattern_metricgan_initialization(self):
        """Test BaseProcessor initializes Pattern→MetricGAN+ correctly."""
        config = {
            "enhancement_level": "pattern_metricgan_plus",
            "pattern_metricgan_config": {
                "pattern_detection": {"confidence_threshold": 0.8},
                "pattern_suppression": {"suppression_factor": 0.15},
                "loudness_enhancement": {"target_multiplier": 1.6}
            }
        }
        
        with patch('processors.base_processor.ENHANCEMENT_AVAILABLE', True):
            processor = BaseProcessor(config)
            
            self.assertEqual(processor.enhancement_level, 'pattern_metricgan_plus')
            self.assertTrue(processor.noise_reduction_enabled)
            self.assertIsNotNone(processor.pattern_metricgan_config)
    
    def test_enhancement_metadata_collection(self):
        """Test enhanced metadata collection for Pattern→MetricGAN+."""
        config = {
            "enhancement_level": "pattern_metricgan_plus",
            "pattern_metricgan_config": {}
        }
        
        with patch('processors.base_processor.ENHANCEMENT_AVAILABLE', True):
            processor = BaseProcessor(config)
            
            # Mock audio enhancer
            mock_enhancer = MagicMock()
            mock_enhancer.enhance.return_value = (
                MagicMock(),  # enhanced audio
                {
                    'metricgan_applied': True,
                    'loudness_enhanced': True,
                    'patterns_detected': 2
                }
            )
            processor.audio_enhancer = mock_enhancer
            
            # Test enhancement with metadata
            result = processor._apply_noise_reduction_with_metadata(b'mock_audio', 'S1')
            
            self.assertIsNotNone(result)
            enhanced_audio, metadata = result
            self.assertEqual(metadata['enhancement_method'], 'pattern_metricgan_plus')
            self.assertTrue(metadata['pattern_detection_used'])
            self.assertTrue(metadata['metricgan_applied'])
    
    def test_streaming_mode_compatibility(self):
        """Test Pattern→MetricGAN+ works in streaming mode."""
        config = {
            "enhancement_level": "pattern_metricgan_plus",
            "streaming": True,
            "batch_size": 1000,
            "pattern_metricgan_config": {}
        }
        
        with patch('processors.base_processor.ENHANCEMENT_AVAILABLE', True):
            processor = BaseProcessor(config)
            
            self.assertTrue(processor.streaming_mode)
            self.assertEqual(processor.enhancement_level, 'pattern_metricgan_plus')
```

### 7. Documentation Requirements

#### 7.1 CLI Usage Documentation
**File: CLI usage examples in main.py docstring**

```python
"""
Main entry point for the Thai Audio Dataset Collection project.

Pattern→MetricGAN+ Enhancement Usage Examples:

1. Basic Pattern→MetricGAN+ processing:
   python main.py --streaming --enhancement-level pattern_metricgan_plus gigaspeech2

2. Custom Pattern→MetricGAN+ parameters:
   python main.py --streaming --enhancement-level pattern_metricgan_plus \
                  --pattern-confidence-threshold 0.9 \
                  --pattern-suppression-factor 0.1 \
                  --loudness-multiplier 1.8 \
                  gigaspeech2

3. Pattern→MetricGAN+ with GPU specification:
   python main.py --streaming --enhancement-level pattern_metricgan_plus \
                  --metricgan-device cuda \
                  gigaspeech2

4. Pattern→MetricGAN+ via environment variables:
   export PATTERN_CONFIDENCE_THRESHOLD=0.9
   export LOUDNESS_MULTIPLIER=1.8
   python main.py --streaming --enhancement-level pattern_metricgan_plus gigaspeech2

5. Multiple datasets with Pattern→MetricGAN+:
   python main.py --streaming --enhancement-level pattern_metricgan_plus \
                  gigaspeech2 processed_voice_th mozilla_cv
"""
```

#### 7.2 Configuration Documentation
**File: Update project README with Pattern→MetricGAN+ section**

```markdown
## Pattern→MetricGAN+ Enhancement

The Pattern→MetricGAN+ enhancement level provides advanced audio quality improvement through:

1. **Ultra-conservative pattern detection** - Identifies interruptions with >80% confidence
2. **Gentle pattern suppression** - Preserves 15% of original signal during suppression
3. **MetricGAN+ enhancement** - Improves overall audio quality using SpeechBrain
4. **160% loudness normalization** - Maintains natural loudness levels

### Usage

```bash
# Basic usage
python main.py --streaming --enhancement-level pattern_metricgan_plus gigaspeech2

# With custom parameters
python main.py --streaming --enhancement-level pattern_metricgan_plus \
               --pattern-confidence-threshold 0.9 \
               --loudness-multiplier 1.8 \
               gigaspeech2
```

### Configuration Options

| Parameter | CLI Argument | Environment Variable | Default | Description |
|-----------|--------------|---------------------|---------|-------------|
| Confidence Threshold | `--pattern-confidence-threshold` | `PATTERN_CONFIDENCE_THRESHOLD` | 0.8 | Pattern detection confidence (0.5-1.0) |
| Suppression Factor | `--pattern-suppression-factor` | `PATTERN_SUPPRESSION_FACTOR` | 0.15 | Signal preservation during suppression (0.0-1.0) |
| Padding | `--pattern-padding-ms` | `PATTERN_PADDING_MS` | 50 | Padding around patterns (ms) |
| Loudness Multiplier | `--loudness-multiplier` | `LOUDNESS_MULTIPLIER` | 1.6 | Target loudness enhancement (1.0-3.0) |
| MetricGAN Device | `--metricgan-device` | `METRICGAN_DEVICE` | auto | Processing device (auto/cuda/cpu) |
```

### 8. Success Criteria

#### 8.1 Functional Requirements
- ✅ Pattern→MetricGAN+ enhancement level available in CLI `--enhancement-level` choices
- ✅ All Pattern→MetricGAN+ parameters configurable via CLI arguments
- ✅ Environment variable support for all configuration options
- ✅ Configuration validation with meaningful error messages
- ✅ Integration with existing BaseProcessor audio enhancement pipeline
- ✅ Backward compatibility with existing enhancement levels
- ✅ Proper metadata collection for Pattern→MetricGAN+ processing

#### 8.2 Configuration Requirements
- ✅ Configuration schema properly integrated into `config.py`
- ✅ Validation rules prevent invalid parameter combinations
- ✅ Environment variables override default configuration
- ✅ CLI arguments override both defaults and environment variables
- ✅ Configuration migration handles legacy enhancement level names
- ✅ Error handling for missing dependencies (SpeechBrain, MetricGAN+)

#### 8.3 Integration Requirements
- ✅ All existing tests continue to pass
- ✅ BaseProcessor supports Pattern→MetricGAN+ without breaking changes
- ✅ Streaming mode compatibility maintained
- ✅ Dataset processors (gigaspeech2.py, processed_voice_th.py, mozilla_cv.py) work unchanged
- ✅ Enhancement metadata properly collected and included in output
- ✅ Quality metrics integration preserved

#### 8.4 Documentation Requirements
- ✅ CLI help shows Pattern→MetricGAN+ options with clear descriptions
- ✅ Configuration examples documented with realistic usage scenarios
- ✅ Environment variable documentation includes examples
- ✅ Error messages provide actionable guidance for parameter correction
- ✅ README updated with Pattern→MetricGAN+ usage section

### 9. Implementation Plan

#### Phase 1: Configuration Foundation (Days 1-2)
1. Add Pattern→MetricGAN+ configuration to `config.py`
2. Implement configuration validation and environment variable support
3. Add backward compatibility and migration logic
4. Create configuration loading and validation unit tests

#### Phase 2: CLI Integration (Days 3-4)
1. Extend CLI argument parser with Pattern→MetricGAN+ options
2. Update processor configuration building logic
3. Add CLI argument validation and error handling
4. Create CLI integration tests

#### Phase 3: Processor Integration (Days 5-6)
1. Extend BaseProcessor to support Pattern→MetricGAN+ configuration
2. Update metadata collection for enhanced tracking
3. Ensure streaming mode compatibility
4. Create processor integration tests

#### Phase 4: Testing & Documentation (Days 7)
1. Complete comprehensive test suite execution
2. Validate backward compatibility with existing workflows
3. Update documentation with usage examples
4. Perform end-to-end testing with real datasets

### 10. Dependencies

#### 10.1 Internal Dependencies
- `config.py` - Configuration management system
- `main.py` - CLI argument parsing and processor orchestration
- `processors/base_processor.py` - Base processor framework
- `processors/audio_enhancement/core.py` - Audio enhancement pipeline (Task 21)

#### 10.2 External Dependencies
- `argparse` - CLI argument parsing
- `os` - Environment variable access
- `logging` - Configuration validation logging

#### 10.3 Task Dependencies
- **Task 21**: Core Pattern→MetricGAN+ Integration must be completed first
- Pattern detection, suppression, and MetricGAN+ components must be available

### 11. Risk Mitigation

#### 11.1 Configuration Risks
- **Invalid parameter combinations**: Comprehensive validation with clear error messages
- **Environment variable conflicts**: Clear precedence order (CLI > ENV > defaults)
- **Configuration migration failures**: Robust fallback to safe defaults

#### 11.2 Integration Risks
- **Backward compatibility breaks**: Extensive testing with existing workflows
- **Performance regression**: Configuration loading optimization and lazy initialization
- **CLI complexity**: Clear grouping and help text organization

### 12. Testing Strategy

#### 12.1 Unit Testing
- Configuration loading and validation logic
- Environment variable processing
- CLI argument parsing and validation
- Parameter bound checking and error handling

#### 12.2 Integration Testing
- End-to-end CLI workflow with Pattern→MetricGAN+ level
- Configuration propagation through processor pipeline
- Metadata collection and enhancement tracking
- Streaming mode with Pattern→MetricGAN+ configuration

#### 12.3 Compatibility Testing
- All existing enhancement levels continue working
- Legacy CLI argument combinations remain functional
- Environment variable precedence operates correctly
- Error messages guide users to correct usage

### 13. Complexity Assessment
**Medium Complexity** - Involves integration with multiple existing systems (config, CLI, processors) but follows established patterns. The main complexity is ensuring proper configuration flow and maintaining backward compatibility across all integration points.

### 14. Estimated Duration
**1 week** - Given the medium complexity and need for thorough testing of integration points

### 15. Status
**📋 PLANNED** - Ready for implementation following TDD approach, dependent on Task 21 completion