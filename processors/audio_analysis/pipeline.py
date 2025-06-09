"""
Pipeline execution system for audio processing.
Supports stage-based processing with error handling and monitoring.
"""

import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from dataclasses import dataclass, field

from .base import (
    AudioProcessor, ProcessResult, ProcessingStatus,
    StageConfig, PipelineConfig, StageExecutionError
)
from .plugin_system import PluginManager

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage."""
    execution_count: int = 0
    total_time: float = 0.0
    max_time: float = 0.0
    min_time: float = float('inf')
    avg_time: float = 0.0
    error_count: int = 0
    skip_count: int = 0
    memory_usage: float = 0.0
    
    def update(self, execution_time: float, memory: float = 0.0, 
               error: bool = False, skip: bool = False):
        """Update metrics with new execution data."""
        self.execution_count += 1
        self.total_time += execution_time
        self.max_time = max(self.max_time, execution_time)
        self.min_time = min(self.min_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.memory_usage = max(self.memory_usage, memory)
        
        if error:
            self.error_count += 1
        if skip:
            self.skip_count += 1


class PipelineStage:
    """
    A single stage in the processing pipeline.
    """
    
    def __init__(self, name: str, processor: AudioProcessor, 
                 config: Optional[StageConfig] = None):
        """
        Initialize pipeline stage.
        
        Args:
            name: Stage name
            processor: Audio processor instance
            config: Stage configuration
        """
        self.name = name
        self.processor = processor
        self.config = config or StageConfig(name=name, processor=processor.__class__.__name__)
        self.metrics = StageMetrics()
        self._last_result: Optional[ProcessResult] = None
    
    def execute(self, audio: np.ndarray, sr: int, 
                metadata: Optional[Dict[str, Any]] = None) -> ProcessResult:
        """
        Execute the stage processing.
        
        Args:
            audio: Input audio
            sr: Sample rate
            metadata: Metadata from previous stages
            
        Returns:
            Processing result
        """
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            # Validate input
            if not self.processor.validate_input(audio, sr):
                raise StageExecutionError(
                    self.name, 
                    "Input validation failed",
                    recoverable=False
                )
            
            # Process with retry logic if configured
            result = self._execute_with_retry(audio, sr, metadata)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics.update(execution_time)
            
            # Add stage metadata
            result.metadata[f"{self.name}_time"] = execution_time
            result.metadata[f"{self.name}_status"] = result.status.value
            
            self._last_result = result
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.update(execution_time, error=True)
            
            # Handle error based on configuration
            return self._handle_error(e, audio, sr, metadata)
    
    def _execute_with_retry(self, audio: np.ndarray, sr: int, 
                           metadata: Dict[str, Any]) -> ProcessResult:
        """Execute with retry logic."""
        error_config = self.config.error_handling
        max_attempts = error_config.get('max_attempts', 1)
        retry_delay = error_config.get('retry_delay', 0.1)
        
        last_error = None
        for attempt in range(max_attempts):
            try:
                return self.processor.process(audio, sr, metadata=metadata)
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    logger.warning(f"Stage {self.name} attempt {attempt + 1} failed, retrying...")
        
        raise last_error
    
    def _handle_error(self, error: Exception, audio: np.ndarray, 
                     sr: int, metadata: Dict[str, Any]) -> ProcessResult:
        """Handle stage execution error."""
        error_config = self.config.error_handling
        strategy = error_config.get('strategy', 'fail')
        
        if strategy == 'skip':
            # Skip stage and pass through input
            logger.warning(f"Stage {self.name} failed, skipping: {error}")
            self.metrics.update(0, skip=True)
            return ProcessResult(
                audio=audio,
                metadata=metadata,
                status=ProcessingStatus.SKIPPED,
                warnings=[f"Stage {self.name} skipped due to error: {str(error)}"]
            )
            
        elif strategy == 'fallback':
            # Use fallback processor if available
            fallback = error_config.get('fallback_processor')
            if fallback:
                logger.warning(f"Stage {self.name} failed, using fallback: {error}")
                return fallback.process(audio, sr, metadata=metadata)
        
        # Default: propagate error
        if isinstance(error, StageExecutionError):
            raise error
        else:
            raise StageExecutionError(self.name, str(error))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stage metrics."""
        return {
            'execution_count': self.metrics.execution_count,
            'avg_time': self.metrics.avg_time,
            'max_time': self.metrics.max_time,
            'min_time': self.metrics.min_time,
            'error_rate': (self.metrics.error_count / self.metrics.execution_count 
                          if self.metrics.execution_count > 0 else 0),
            'skip_rate': (self.metrics.skip_count / self.metrics.execution_count 
                         if self.metrics.execution_count > 0 else 0),
            'memory_usage': self.metrics.memory_usage
        }


class Pipeline:
    """
    Audio processing pipeline with stage management and orchestration.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None,
                 plugin_manager: Optional[PluginManager] = None,
                 enable_monitoring: bool = True,
                 enable_parallel: bool = False,
                 enable_graceful_degradation: bool = False):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            plugin_manager: Plugin manager for loading processors
            enable_monitoring: Enable performance monitoring
            enable_parallel: Enable parallel stage execution
            enable_graceful_degradation: Continue on optional stage failures
        """
        self.config = config
        self.plugin_manager = plugin_manager
        self.enable_monitoring = enable_monitoring
        self.enable_parallel = enable_parallel
        self.enable_graceful_degradation = enable_graceful_degradation
        
        self.stages: Dict[str, PipelineStage] = {}
        self.stage_order: List[str] = []
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.metrics: Dict[str, StageMetrics] = {}
        
        if config:
            self._build_from_config(config)
    
    def add_stage(self, name: str, processor: AudioProcessor,
                  dependencies: Optional[List[str]] = None,
                  required: bool = True) -> None:
        """
        Add a stage to the pipeline.
        
        Args:
            name: Stage name
            processor: Processor instance
            dependencies: List of stage names this depends on
            required: Whether stage is required for pipeline success
        """
        config = StageConfig(
            name=name,
            processor=processor.__class__.__name__,
            dependencies=dependencies or [],
            required=required
        )
        
        stage = PipelineStage(name, processor, config)
        self.stages[name] = stage
        
        # Update dependency graph
        if dependencies:
            for dep in dependencies:
                self.dependency_graph[dep].add(name)
        
        # Recalculate execution order
        self._update_execution_order()
    
    def _build_from_config(self, config: PipelineConfig) -> None:
        """Build pipeline from configuration."""
        if not self.plugin_manager:
            raise ValueError("Plugin manager required for config-based pipeline")
        
        for stage_config in config.stages:
            # Load processor from plugin manager
            processor = self.plugin_manager.get_plugin(
                stage_config.processor,
                stage_config.config
            )
            
            # Create stage
            stage = PipelineStage(stage_config.name, processor, stage_config)
            self.stages[stage_config.name] = stage
            
            # Build dependency graph
            for dep in stage_config.dependencies:
                self.dependency_graph[dep].add(stage_config.name)
        
        self._update_execution_order()
    
    def _update_execution_order(self) -> None:
        """Update stage execution order based on dependencies."""
        # Topological sort
        in_degree = defaultdict(int)
        for stage in self.stages:
            for dependent in self.dependency_graph[stage]:
                in_degree[dependent] += 1
        
        queue = deque([s for s in self.stages if in_degree[s] == 0])
        self.stage_order = []
        
        while queue:
            stage = queue.popleft()
            self.stage_order.append(stage)
            
            for dependent in self.dependency_graph[stage]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
    
    def get_execution_order(self) -> List[str]:
        """Get stage execution order."""
        return self.stage_order.copy()
    
    def execute(self, audio: np.ndarray, sr: int,
                initial_metadata: Optional[Dict[str, Any]] = None) -> ProcessResult:
        """
        Execute the pipeline.
        
        Args:
            audio: Input audio
            sr: Sample rate
            initial_metadata: Initial metadata
            
        Returns:
            Final processing result
        """
        start_time = time.time()
        metadata = initial_metadata or {}
        metadata['pipeline_start'] = start_time
        
        current_audio = audio
        errors = []
        warnings = []
        
        if self.enable_parallel:
            result = self._execute_parallel(current_audio, sr, metadata)
        else:
            result = self._execute_sequential(current_audio, sr, metadata)
        
        # Add pipeline metadata
        result.metadata['pipeline_time'] = time.time() - start_time
        result.metadata['stages_executed'] = len(self.stage_order)
        
        return result
    
    def _execute_sequential(self, audio: np.ndarray, sr: int,
                           metadata: Dict[str, Any]) -> ProcessResult:
        """Execute stages sequentially."""
        current_audio = audio
        current_metadata = metadata.copy()
        errors = []
        warnings = []
        
        for stage_name in self.stage_order:
            stage = self.stages[stage_name]
            
            try:
                logger.debug(f"Executing stage: {stage_name}")
                result = stage.execute(current_audio, sr, current_metadata)
                
                # Update for next stage
                current_audio = result.audio
                current_metadata.update(result.metadata)
                errors.extend(result.errors)
                warnings.extend(result.warnings)
                
            except Exception as e:
                if stage.config.required and not self.enable_graceful_degradation:
                    # Required stage failed - pipeline fails
                    raise
                else:
                    # Optional stage failed - log and continue
                    logger.error(f"Optional stage {stage_name} failed: {e}")
                    errors.append(f"Stage {stage_name} failed: {str(e)}")
        
        # Return final result
        status = ProcessingStatus.SUCCESS if not errors else ProcessingStatus.PARTIAL_SUCCESS
        return ProcessResult(
            audio=current_audio,
            metadata=current_metadata,
            status=status,
            errors=errors,
            warnings=warnings
        )
    
    def _execute_parallel(self, audio: np.ndarray, sr: int,
                         metadata: Dict[str, Any]) -> ProcessResult:
        """Execute independent stages in parallel."""
        # Group stages by dependency level
        levels = self._get_dependency_levels()
        
        current_audio = audio
        current_metadata = metadata.copy()
        errors = []
        warnings = []
        
        with ThreadPoolExecutor() as executor:
            for level in levels:
                # Execute all stages at this level in parallel
                futures: Dict[str, Future] = {}
                
                for stage_name in level:
                    stage = self.stages[stage_name]
                    # Each parallel stage gets a copy of the audio
                    future = executor.submit(
                        stage.execute, 
                        current_audio.copy(), 
                        sr, 
                        current_metadata.copy()
                    )
                    futures[stage_name] = future
                
                # Wait for all stages at this level to complete
                level_results = {}
                for stage_name, future in futures.items():
                    try:
                        result = future.result()
                        level_results[stage_name] = result
                    except Exception as e:
                        stage = self.stages[stage_name]
                        if stage.config.required:
                            raise
                        else:
                            logger.error(f"Parallel stage {stage_name} failed: {e}")
                            errors.append(f"Stage {stage_name} failed: {str(e)}")
                
                # Merge results (simplified - in practice might need custom merge logic)
                if level_results:
                    # Use the first result's audio as base
                    first_result = next(iter(level_results.values()))
                    current_audio = first_result.audio
                    
                    # Merge all metadata
                    for result in level_results.values():
                        current_metadata.update(result.metadata)
                        errors.extend(result.errors)
                        warnings.extend(result.warnings)
        
        status = ProcessingStatus.SUCCESS if not errors else ProcessingStatus.PARTIAL_SUCCESS
        return ProcessResult(
            audio=current_audio,
            metadata=current_metadata,
            status=status,
            errors=errors,
            warnings=warnings
        )
    
    def _get_dependency_levels(self) -> List[List[str]]:
        """Get stages grouped by dependency level for parallel execution."""
        levels = []
        remaining = set(self.stage_order)
        completed = set()
        
        while remaining:
            # Find stages that can be executed at this level
            current_level = []
            for stage in remaining:
                dependencies = self.stages[stage].config.dependencies
                if all(dep in completed for dep in dependencies):
                    current_level.append(stage)
            
            if not current_level:
                # Circular dependency or error
                raise ValueError("Unable to determine execution levels")
            
            levels.append(current_level)
            completed.update(current_level)
            remaining.difference_update(current_level)
        
        return levels
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all stages."""
        return {
            stage_name: stage.get_metrics()
            for stage_name, stage in self.stages.items()
        }
    
    def reset_metrics(self) -> None:
        """Reset all stage metrics."""
        for stage in self.stages.values():
            stage.metrics = StageMetrics()
    
    @classmethod
    def from_config(cls, config_path: str, 
                   plugin_manager: PluginManager) -> 'Pipeline':
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to configuration file
            plugin_manager: Plugin manager
            
        Returns:
            Pipeline instance
        """
        from .config import ConfigLoader
        loader = ConfigLoader()
        config = loader.load(config_path)
        return cls(config=config.pipeline, plugin_manager=plugin_manager)