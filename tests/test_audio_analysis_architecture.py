"""
Test suite for the core audio analysis architecture.
Following TDD approach - tests written before implementation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import os
import json
import tempfile

# Import modules to be tested (will be created)
from processors.audio_analysis.base import (
    AudioProcessor, QualityMetric, DecisionEngine, AudioAnalyzer,
    ProcessResult, PluginInfo, PipelineError, PluginLoadError,
    StageExecutionError, ValidationError
)
from processors.audio_analysis.plugin_system import (
    PluginRegistry, PluginLoader, PluginDiscovery
)
from processors.audio_analysis.pipeline import (
    Pipeline, PipelineStage, PipelineConfig, StageConfig
)
from processors.audio_analysis.factory import (
    ProcessorFactory, ComponentFactory
)


class TestAudioProcessorInterface:
    """Test the AudioProcessor abstract base class interface."""
    
    def test_audio_processor_interface_compliance(self):
        """Test that all processors implement required methods."""
        # Create a mock processor
        class MockProcessor(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                return ProcessResult(audio=audio, metadata={})
            
            def get_capabilities(self) -> Dict[str, Any]:
                return {"formats": ["wav", "mp3"], "sample_rates": [16000, 44100]}
            
            def validate_input(self, audio: np.ndarray, sr: int) -> bool:
                return len(audio) > 0 and sr > 0
            
            @classmethod
            def get_plugin_info(cls) -> PluginInfo:
                return PluginInfo(
                    name="MockProcessor",
                    version="1.0.0",
                    author="Test",
                    description="Mock processor for testing"
                )
            
            def get_dependencies(self) -> List[str]:
                return []
        
        # Instantiate and verify interface
        processor = MockProcessor()
        
        # Test required methods exist
        assert hasattr(processor, 'process')
        assert hasattr(processor, 'get_capabilities')
        assert hasattr(processor, 'validate_input')
        assert hasattr(processor, 'get_plugin_info')
        assert hasattr(processor, 'get_dependencies')
        
        # Test method signatures
        audio = np.zeros(1000)
        sr = 16000
        
        result = processor.process(audio, sr)
        assert isinstance(result, ProcessResult)
        
        capabilities = processor.get_capabilities()
        assert isinstance(capabilities, dict)
        
        is_valid = processor.validate_input(audio, sr)
        assert isinstance(is_valid, bool)
        
        plugin_info = processor.get_plugin_info()
        assert isinstance(plugin_info, PluginInfo)
        
        dependencies = processor.get_dependencies()
        assert isinstance(dependencies, list)
    
    def test_audio_processor_abstract_methods(self):
        """Test that AudioProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AudioProcessor()
    
    def test_process_result_dataclass(self):
        """Test ProcessResult dataclass functionality."""
        audio = np.zeros(1000)
        metadata = {"snr": 25.0, "processing_time": 0.5}
        
        result = ProcessResult(audio=audio, metadata=metadata)
        
        assert np.array_equal(result.audio, audio)
        assert result.metadata == metadata
        assert hasattr(result, 'errors')
        assert result.errors is None or result.errors == []


class TestPluginSystem:
    """Test the plugin discovery and loading system."""
    
    def test_plugin_discovery(self):
        """Test dynamic loading of plugins."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock plugin file
            plugin_code = '''
from processors.audio_analysis.base import AudioProcessor, ProcessResult, PluginInfo
import numpy as np

class TestPlugin(AudioProcessor):
    def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
        return ProcessResult(audio=audio, metadata={"plugin": "test"})
    
    def get_capabilities(self) -> dict:
        return {"test": True}
    
    def validate_input(self, audio: np.ndarray, sr: int) -> bool:
        return True
    
    @classmethod
    def get_plugin_info(cls) -> PluginInfo:
        return PluginInfo(
            name="TestPlugin",
            version="1.0.0",
            author="Test",
            description="Test plugin"
        )
    
    def get_dependencies(self) -> list:
        return []
'''
            
            plugin_path = os.path.join(temp_dir, "test_plugin.py")
            with open(plugin_path, 'w') as f:
                f.write(plugin_code)
            
            # Initialize plugin registry
            registry = PluginRegistry([temp_dir])
            
            # Discover plugins
            discovery = PluginDiscovery(registry)
            discovered = discovery.scan_directories()
            
            # Verify plugin was discovered
            assert len(discovered) > 0
            assert "TestPlugin" in [p.name for p in discovered]
    
    def test_plugin_version_compatibility(self):
        """Test version compatibility checks."""
        registry = PluginRegistry()
        
        # Test compatible versions
        assert registry.check_compatibility("AudioLoader", "1.0.0", ["1.0", "1.1"])
        assert registry.check_compatibility("AudioLoader", "1.1.5", ["1.0", "1.1"])
        
        # Test incompatible versions
        assert not registry.check_compatibility("AudioLoader", "2.0.0", ["1.0", "1.1"])
        assert not registry.check_compatibility("AudioLoader", "0.9.0", ["1.0", "1.1"])
    
    def test_plugin_loading_error_handling(self):
        """Test graceful handling of missing plugins."""
        loader = PluginLoader()
        
        # Try to load non-existent plugin
        with pytest.raises(PluginLoadError):
            loader.load_plugin("NonExistentPlugin")
    
    def test_plugin_dependency_resolution(self):
        """Test plugin dependency graph resolution."""
        registry = PluginRegistry()
        
        # Create mock plugins with dependencies
        plugins = {
            "A": {"dependencies": ["B", "C"]},
            "B": {"dependencies": ["D"]},
            "C": {"dependencies": ["D"]},
            "D": {"dependencies": []}
        }
        
        # Build dependency graph
        load_order = registry.resolve_dependencies(plugins)
        
        # Verify correct load order (D should come before B and C, which come before A)
        assert load_order.index("D") < load_order.index("B")
        assert load_order.index("D") < load_order.index("C")
        assert load_order.index("B") < load_order.index("A")
        assert load_order.index("C") < load_order.index("A")


class TestPipelineExecution:
    """Test the pipeline execution system."""
    
    def test_pipeline_stage_ordering(self):
        """Test stage ordering in pipeline."""
        # Create pipeline configuration
        config = PipelineConfig(
            name="test_pipeline",
            version="1.0.0",
            stages=[
                StageConfig(name="loader", processor="AudioLoader", dependencies=[]),
                StageConfig(name="preprocessor", processor="AudioPreprocessor", dependencies=["loader"]),
                StageConfig(name="analyzer", processor="AudioAnalyzer", dependencies=["preprocessor"])
            ]
        )
        
        # Create pipeline
        pipeline = Pipeline(config)
        
        # Get execution order
        order = pipeline.get_execution_order()
        
        # Verify correct ordering
        assert order == ["loader", "preprocessor", "analyzer"]
    
    def test_pipeline_error_propagation(self):
        """Test error propagation through pipeline stages."""
        # Create mock processor that fails
        class FailingProcessor(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                raise StageExecutionError("Stage failed")
            
            def get_capabilities(self) -> Dict[str, Any]:
                return {}
            
            def validate_input(self, audio: np.ndarray, sr: int) -> bool:
                return True
        
        # Create pipeline with failing stage
        pipeline = Pipeline()
        pipeline.add_stage("failing", FailingProcessor())
        
        # Test error propagation
        audio = np.zeros(1000)
        sr = 16000
        
        with pytest.raises(StageExecutionError):
            pipeline.execute(audio, sr)
    
    def test_pipeline_metadata_preservation(self):
        """Test metadata preservation through pipeline stages."""
        # Create processors that add metadata
        class MetadataProcessor1(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                return ProcessResult(audio=audio, metadata={"stage1": "processed"})
        
        class MetadataProcessor2(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                # Should receive metadata from previous stage
                return ProcessResult(
                    audio=audio, 
                    metadata={"stage1": "processed", "stage2": "processed"}
                )
        
        # Create pipeline
        pipeline = Pipeline()
        pipeline.add_stage("stage1", MetadataProcessor1())
        pipeline.add_stage("stage2", MetadataProcessor2())
        
        # Execute pipeline
        audio = np.zeros(1000)
        sr = 16000
        result = pipeline.execute(audio, sr)
        
        # Verify metadata preservation
        assert result.metadata.get("stage1") == "processed"
        assert result.metadata.get("stage2") == "processed"
    
    def test_pipeline_performance_monitoring(self):
        """Test performance monitoring hooks."""
        # Create pipeline with monitoring
        pipeline = Pipeline(enable_monitoring=True)
        
        # Add mock stages
        class SlowProcessor(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                import time
                time.sleep(0.1)  # Simulate processing time
                return ProcessResult(audio=audio, metadata={})
        
        pipeline.add_stage("slow", SlowProcessor())
        
        # Execute and check metrics
        audio = np.zeros(1000)
        sr = 16000
        result = pipeline.execute(audio, sr)
        
        # Get performance metrics
        metrics = pipeline.get_metrics()
        
        assert "slow" in metrics
        assert metrics["slow"]["execution_time"] >= 0.1
        assert metrics["slow"]["memory_usage"] >= 0


class TestConfigurationSystem:
    """Test configuration loading and validation."""
    
    def test_valid_configuration_loading(self):
        """Test loading of valid configurations."""
        config_yaml = """
pipeline:
  name: "test_pipeline"
  version: "1.0.0"
  stages:
    - name: "loader"
      processor: "AudioLoader"
      config:
        formats: ["wav", "mp3"]
        sample_rate: 16000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            f.flush()
            
            # Load configuration
            from processors.audio_analysis.config import ConfigLoader
            loader = ConfigLoader()
            config = loader.load(f.name)
            
            # Verify configuration
            assert config.pipeline.name == "test_pipeline"
            assert config.pipeline.version == "1.0.0"
            assert len(config.pipeline.stages) == 1
            assert config.pipeline.stages[0].name == "loader"
        
        os.unlink(f.name)
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        invalid_config = """
pipeline:
  name: "test_pipeline"
  # Missing required version field
  stages:
    - name: "loader"
      # Missing required processor field
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_config)
            f.flush()
            
            # Try to load invalid configuration
            from processors.audio_analysis.config import ConfigLoader
            loader = ConfigLoader()
            
            with pytest.raises(ValidationError):
                loader.load(f.name)
        
        os.unlink(f.name)
    
    def test_configuration_default_fallbacks(self):
        """Test default fallbacks for missing configuration values."""
        minimal_config = """
pipeline:
  name: "test_pipeline"
  version: "1.0.0"
  stages:
    - name: "loader"
      processor: "AudioLoader"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(minimal_config)
            f.flush()
            
            # Load configuration
            from processors.audio_analysis.config import ConfigLoader
            loader = ConfigLoader()
            config = loader.load(f.name)
            
            # Check defaults were applied
            stage = config.pipeline.stages[0]
            assert hasattr(stage, 'config')
            assert stage.config == {} or stage.config is not None
            assert hasattr(stage, 'error_handling')
        
        os.unlink(f.name)


class TestFactoryPatterns:
    """Test factory patterns for component instantiation."""
    
    def test_processor_factory(self):
        """Test processor factory creation."""
        factory = ProcessorFactory()
        
        # Register mock processor
        class MockProcessor(AudioProcessor):
            def __init__(self, config=None):
                self.config = config or {}
        
        factory.register("MockProcessor", MockProcessor)
        
        # Create processor instance
        config = {"test": True}
        processor = factory.create("MockProcessor", config)
        
        assert isinstance(processor, MockProcessor)
        assert processor.config == config
    
    def test_component_factory_with_dependencies(self):
        """Test component factory with dependency injection."""
        factory = ComponentFactory()
        
        # Register components
        class Logger:
            def log(self, msg):
                pass
        
        class Processor:
            def __init__(self, logger: Logger):
                self.logger = logger
        
        factory.register("Logger", Logger)
        factory.register("Processor", Processor, dependencies=["Logger"])
        
        # Create processor with injected dependencies
        processor = factory.create("Processor")
        
        assert isinstance(processor, Processor)
        assert isinstance(processor.logger, Logger)


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_end_to_end_pipeline_execution(self):
        """Test complete pipeline execution from config to result."""
        # Create temporary configuration
        config_yaml = """
pipeline:
  name: "integration_test"
  version: "1.0.0"
  stages:
    - name: "loader"
      processor: "MockLoader"
      config:
        validate: true
    - name: "processor"
      processor: "MockProcessor"
      dependencies: ["loader"]
      config:
        enhance: true
"""
        
        with tempfile.TemporaryFile(mode='w+', suffix='.yaml') as f:
            f.write(config_yaml)
            f.seek(0)
            
            # Mock the entire pipeline execution
            audio_input = np.random.randn(16000)  # 1 second at 16kHz
            sr = 16000
            
            # This would be the actual integration test
            # pipeline = Pipeline.from_config(f.name)
            # result = pipeline.execute(audio_input, sr)
            # assert result.audio.shape == audio_input.shape
            # assert "loader" in result.metadata
            # assert "processor" in result.metadata
    
    def test_parallel_stage_execution(self):
        """Test parallel execution of independent stages."""
        # Create pipeline with parallel stages
        pipeline = Pipeline(enable_parallel=True)
        
        # Add independent stages that can run in parallel
        class IndependentProcessor(AudioProcessor):
            def __init__(self, name):
                self.name = name
            
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                import time
                time.sleep(0.1)  # Simulate work
                return ProcessResult(audio=audio, metadata={self.name: "done"})
        
        # Add stages with no dependencies on each other
        pipeline.add_stage("analyzer1", IndependentProcessor("analyzer1"))
        pipeline.add_stage("analyzer2", IndependentProcessor("analyzer2"))
        pipeline.add_stage("combiner", IndependentProcessor("combiner"), 
                          dependencies=["analyzer1", "analyzer2"])
        
        # Execute pipeline
        audio = np.zeros(1000)
        sr = 16000
        
        start_time = time.time()
        result = pipeline.execute(audio, sr)
        execution_time = time.time() - start_time
        
        # Parallel execution should be faster than sequential
        # (0.2s for parallel vs 0.3s for sequential)
        assert execution_time < 0.25  # Allow some overhead
        
        # Verify all stages executed
        assert result.metadata.get("analyzer1") == "done"
        assert result.metadata.get("analyzer2") == "done"
        assert result.metadata.get("combiner") == "done"


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    def test_stage_level_retry(self):
        """Test retry mechanism at stage level."""
        # Create processor that fails then succeeds
        class FlakeyProcessor(AudioProcessor):
            attempt_count = 0
            
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise StageExecutionError("Temporary failure")
                return ProcessResult(audio=audio, metadata={"attempts": self.attempt_count})
        
        # Configure stage with retry
        stage = PipelineStage(
            name="flakey",
            processor=FlakeyProcessor(),
            error_handling={"strategy": "retry", "max_attempts": 3}
        )
        
        # Execute stage
        audio = np.zeros(1000)
        sr = 16000
        result = stage.execute(audio, sr)
        
        # Verify retry worked
        assert result.metadata["attempts"] == 3
    
    def test_fallback_processor(self):
        """Test fallback to alternative processor on failure."""
        # Create failing primary processor
        class PrimaryProcessor(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                raise StageExecutionError("Primary failed")
        
        # Create successful fallback processor
        class FallbackProcessor(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                return ProcessResult(audio=audio, metadata={"processor": "fallback"})
        
        # Configure stage with fallback
        stage = PipelineStage(
            name="processing",
            processor=PrimaryProcessor(),
            error_handling={
                "strategy": "fallback",
                "fallback_processor": FallbackProcessor()
            }
        )
        
        # Execute stage
        audio = np.zeros(1000)
        sr = 16000
        result = stage.execute(audio, sr)
        
        # Verify fallback was used
        assert result.metadata["processor"] == "fallback"
    
    def test_graceful_degradation(self):
        """Test graceful degradation on partial failures."""
        # Create pipeline with optional enhancement stages
        pipeline = Pipeline(enable_graceful_degradation=True)
        
        # Add required stage
        class RequiredProcessor(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                return ProcessResult(audio=audio, metadata={"required": True})
        
        # Add optional enhancement that fails
        class OptionalProcessor(AudioProcessor):
            def process(self, audio: np.ndarray, sr: int) -> ProcessResult:
                raise StageExecutionError("Enhancement failed")
        
        pipeline.add_stage("required", RequiredProcessor(), required=True)
        pipeline.add_stage("optional", OptionalProcessor(), required=False)
        
        # Execute pipeline
        audio = np.zeros(1000)
        sr = 16000
        result = pipeline.execute(audio, sr)
        
        # Verify required stage executed but optional was skipped
        assert result.metadata.get("required") is True
        assert "optional" not in result.metadata
        assert len(result.errors) > 0  # Error was logged but not fatal


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_plugin_registry_cache(self):
        """Test plugin registry caching for fast startup."""
        with tempfile.TemporaryDirectory() as cache_dir:
            # First load - should scan and cache
            registry1 = PluginRegistry(cache_dir=cache_dir)
            start_time = time.time()
            registry1.discover_plugins(["/some/plugin/path"])
            first_load_time = time.time() - start_time
            
            # Save cache
            registry1.save_cache()
            
            # Second load - should use cache
            registry2 = PluginRegistry(cache_dir=cache_dir)
            start_time = time.time()
            registry2.load_cache()
            second_load_time = time.time() - start_time
            
            # Cache load should be much faster
            assert second_load_time < first_load_time * 0.1
    
    def test_memory_pool_management(self):
        """Test memory pool for large array management."""
        from processors.audio_analysis.memory import MemoryPool
        
        # Create memory pool
        pool = MemoryPool(max_size="100MB")
        
        # Allocate arrays
        array1 = pool.allocate(shape=(1000, 1000), dtype=np.float32)
        array2 = pool.allocate(shape=(500, 500), dtype=np.float32)
        
        # Verify allocations
        assert array1.shape == (1000, 1000)
        assert array2.shape == (500, 500)
        
        # Release and reallocate
        pool.release(array1)
        array3 = pool.allocate(shape=(1000, 1000), dtype=np.float32)
        
        # Should reuse memory
        assert array3.data == array1.data  # Same memory location
    
    def test_lazy_loading_iterator(self):
        """Test lazy loading for large datasets."""
        from processors.audio_analysis.data import LazyAudioLoader
        
        # Create mock audio files
        files = [f"audio_{i}.wav" for i in range(1000)]
        
        # Create lazy loader
        loader = LazyAudioLoader(files)
        
        # Iterate through files
        loaded_count = 0
        for audio, sr in loader:
            loaded_count += 1
            if loaded_count >= 10:
                break
        
        # Only requested files should be loaded
        assert loaded_count == 10
        assert loader.get_memory_usage() < 100 * 1024 * 1024  # Less than 100MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])