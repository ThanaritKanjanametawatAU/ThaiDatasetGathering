"""
Demo script showing how to use the audio analysis architecture.
"""

import numpy as np
from pathlib import Path

# Import the architecture components
from .base import ProcessingStatus
from .plugin_system import PluginManager
from .pipeline import Pipeline
from .config import ConfigBuilder
from .factory import ProcessorFactory
from .example_processor import ExampleAudioLoader, ExampleNoiseReducer


def demo_basic_usage():
    """Demonstrate basic usage of the architecture."""
    print("=== Audio Analysis Architecture Demo ===\n")
    
    # 1. Create processors directly
    print("1. Direct processor usage:")
    loader = ExampleAudioLoader()
    
    # Create dummy audio
    sr = 16000
    duration = 1.0  # 1 second
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
    
    # Process audio
    result = loader.process(audio, sr)
    print(f"   - Loader result: {result.status}")
    print(f"   - Metadata: {result.metadata}")
    
    # 2. Use factory pattern
    print("\n2. Factory pattern usage:")
    factory = ProcessorFactory()
    factory.register("AudioLoader", ExampleAudioLoader)
    factory.register("NoiseReducer", ExampleNoiseReducer)
    
    # Create processor via factory
    reducer = factory.create_processor("NoiseReducer", {"reduction_strength": 0.7})
    result = reducer.process(audio, sr)
    print(f"   - Reducer result: {result.status}")
    print(f"   - Noise reduction applied: {result.metadata.get('noise_reduction_applied')}")
    
    # 3. Pipeline usage (without plugin manager)
    print("\n3. Pipeline usage:")
    pipeline = Pipeline()
    
    # Add stages manually
    pipeline.add_stage("loader", ExampleAudioLoader())
    pipeline.add_stage("reducer", ExampleNoiseReducer({"reduction_strength": 0.5}), 
                      dependencies=["loader"])
    
    # Execute pipeline
    result = pipeline.execute(audio, sr)
    print(f"   - Pipeline result: {result.status}")
    print(f"   - Stages executed: {result.metadata.get('stages_executed')}")
    print(f"   - Total time: {result.metadata.get('pipeline_time', 0):.3f}s")
    
    # 4. Configuration builder
    print("\n4. Configuration builder:")
    config = (ConfigBuilder()
              .with_pipeline("audio_enhancement", "1.0.0")
              .with_description("Example audio enhancement pipeline")
              .add_stage("load", "AudioLoader")
              .add_stage("reduce_noise", "NoiseReducer", 
                        config={"reduction_strength": 0.6},
                        dependencies=["load"])
              .build())
    
    print(f"   - Pipeline name: {config.pipeline.name}")
    print(f"   - Stages: {[s.name for s in config.pipeline.stages]}")
    
    # 5. Plugin capabilities
    print("\n5. Plugin capabilities:")
    capabilities = loader.get_capabilities()
    print(f"   - Supported formats: {capabilities['supported_formats']}")
    print(f"   - Real-time capable: {capabilities['real_time_capable']}")
    
    plugin_info = loader.get_plugin_info()
    print(f"   - Plugin: {plugin_info.name} v{plugin_info.version}")
    print(f"   - Category: {plugin_info.category}")
    
    print("\n=== Demo Complete ===")


def demo_error_handling():
    """Demonstrate error handling in the architecture."""
    print("\n=== Error Handling Demo ===\n")
    
    # Create pipeline with error handling
    pipeline = Pipeline(enable_graceful_degradation=True)
    
    # Add a processor that will fail
    class FailingProcessor(ExampleAudioLoader):
        def process(self, audio, sr, **kwargs):
            raise ValueError("Simulated processing error")
        
        @classmethod
        def get_plugin_info(cls):
            info = super().get_plugin_info()
            info.name = "FailingProcessor"
            return info
    
    # Build pipeline
    pipeline.add_stage("loader", ExampleAudioLoader())
    pipeline.add_stage("failing", FailingProcessor(), required=False)  # Optional stage
    pipeline.add_stage("reducer", ExampleNoiseReducer(), dependencies=["loader"])
    
    # Process with graceful degradation
    audio = np.random.randn(16000)  # 1 second of noise
    result = pipeline.execute(audio, 16000)
    
    print(f"Pipeline status: {result.status}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    
    # Check which stages ran
    for stage_name in ["loader", "failing", "reducer"]:
        stage_status = result.metadata.get(f"{stage_name}_status")
        if stage_status:
            print(f"Stage '{stage_name}': {stage_status}")
    
    print("\n=== Error Handling Demo Complete ===")


if __name__ == "__main__":
    # Run demos
    demo_basic_usage()
    demo_error_handling()