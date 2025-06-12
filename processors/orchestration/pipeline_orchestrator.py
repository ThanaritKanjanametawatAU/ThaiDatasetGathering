"""
Data Pipeline Orchestration Module (S04_T01)
Enterprise-grade pipeline orchestration with monitoring, recovery, and REST API management
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Pipeline task status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    CREATED = "created"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TaskConfig:
    """Configuration for pipeline task"""
    task_id: str
    task_function: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    parallel: bool = True
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for entire pipeline"""
    pipeline_id: str
    name: str
    description: str = ""
    max_workers: int = 4
    timeout: Optional[float] = None
    retry_failed_tasks: bool = True
    continue_on_failure: bool = False
    save_intermediate_results: bool = True
    result_storage_path: Optional[str] = None
    monitoring_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    pipeline_id: str
    execution_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineOrchestrator:
    """Enterprise-grade pipeline orchestration system"""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.tasks: Dict[str, TaskConfig] = {}
        self.execution_history: Dict[str, PipelineResult] = {}
        self.current_execution: Optional[PipelineResult] = None
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.is_running = False
        self.is_paused = False
        self.cancel_requested = False
        
        # Task monitoring
        self.task_monitors: Dict[str, threading.Thread] = {}
        self.task_futures: Dict[str, Any] = {}
        
        # Recovery mechanisms
        self.checkpoint_data: Dict[str, Any] = {}
        self.recovery_enabled = True
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'pipeline_start': [],
            'pipeline_end': [],
            'task_start': [],
            'task_end': [],
            'task_retry': [],
            'pipeline_error': []
        }
        
        # Storage setup
        if config.result_storage_path:
            self.storage_path = Path(config.result_storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_path = None
            
        logger.info(f"Pipeline orchestrator initialized: {config.pipeline_id}")
    
    def add_task(self, task_config: TaskConfig):
        """Add task to pipeline.
        
        Args:
            task_config: Task configuration
        """
        if task_config.task_id in self.tasks:
            raise ValueError(f"Task {task_config.task_id} already exists")
        
        # Validate dependencies
        for dep in task_config.dependencies:
            if dep not in self.tasks and dep != task_config.task_id:
                logger.warning(f"Dependency {dep} not found for task {task_config.task_id}")
        
        self.tasks[task_config.task_id] = task_config
        logger.debug(f"Added task: {task_config.task_id}")
    
    def remove_task(self, task_id: str):
        """Remove task from pipeline.
        
        Args:
            task_id: Task identifier
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        # Check if other tasks depend on this one
        dependents = [t.task_id for t in self.tasks.values() if task_id in t.dependencies]
        if dependents:
            raise ValueError(f"Cannot remove task {task_id}: required by {dependents}")
        
        del self.tasks[task_id]
        logger.debug(f"Removed task: {task_id}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler.
        
        Args:
            event_type: Type of event
            handler: Handler function
        """
        if event_type not in self.event_handlers:
            raise ValueError(f"Unknown event type: {event_type}")
        
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event(self, event_type: str, **kwargs):
        """Trigger event handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(**kwargs)
            except Exception as e:
                logger.error(f"Event handler error ({event_type}): {e}")
    
    def execute(self, input_data: Any = None, execution_id: Optional[str] = None) -> PipelineResult:
        """Execute pipeline.
        
        Args:
            input_data: Input data for pipeline
            execution_id: Optional execution identifier
            
        Returns:
            Pipeline execution result
        """
        if self.is_running:
            raise RuntimeError("Pipeline is already running")
        
        execution_id = execution_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        # Initialize execution result
        self.current_execution = PipelineResult(
            pipeline_id=self.config.pipeline_id,
            execution_id=execution_id,
            status=PipelineStatus.RUNNING,
            start_time=start_time,
            total_tasks=len(self.tasks)
        )
        
        self.is_running = True
        self.cancel_requested = False
        
        try:
            logger.info(f"Starting pipeline execution: {execution_id}")
            self._trigger_event('pipeline_start', 
                             pipeline_id=self.config.pipeline_id,
                             execution_id=execution_id)
            
            # Create checkpoint
            self._create_checkpoint(input_data)
            
            # Execute tasks in dependency order
            execution_order = self._resolve_dependencies()
            task_results = self._execute_tasks(execution_order, input_data)
            
            # Update execution result
            self.current_execution.task_results = task_results
            self.current_execution.successful_tasks = sum(
                1 for r in task_results.values() if r.status == TaskStatus.SUCCESS
            )
            self.current_execution.failed_tasks = sum(
                1 for r in task_results.values() if r.status == TaskStatus.FAILED
            )
            
            # Determine final status
            if self.cancel_requested:
                self.current_execution.status = PipelineStatus.CANCELLED
            elif self.current_execution.failed_tasks > 0 and not self.config.continue_on_failure:
                self.current_execution.status = PipelineStatus.FAILED
            else:
                self.current_execution.status = PipelineStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.current_execution.status = PipelineStatus.FAILED
            self._trigger_event('pipeline_error', 
                             pipeline_id=self.config.pipeline_id,
                             execution_id=execution_id,
                             error=str(e))
        
        finally:
            self.current_execution.end_time = datetime.now()
            self.current_execution.execution_time = (
                self.current_execution.end_time - start_time
            ).total_seconds()
            
            self.is_running = False
            
            # Save results
            if self.storage_path:
                self._save_execution_result(self.current_execution)
            
            # Store in history
            self.execution_history[execution_id] = self.current_execution
            
            self._trigger_event('pipeline_end',
                             pipeline_id=self.config.pipeline_id,
                             execution_id=execution_id,
                             status=self.current_execution.status)
            
            logger.info(f"Pipeline execution completed: {execution_id} "
                       f"({self.current_execution.status.value})")
        
        return self.current_execution
    
    def _resolve_dependencies(self) -> List[List[str]]:
        """Resolve task dependencies to determine execution order.
        
        Returns:
            List of task batches in execution order
        """
        # Topological sort with parallel execution groups
        remaining_tasks = set(self.tasks.keys())
        execution_order = []
        
        while remaining_tasks:
            # Find tasks with no remaining dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                task = self.tasks[task_id]
                deps_satisfied = all(dep not in remaining_tasks for dep in task.dependencies)
                if deps_satisfied:
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Circular dependency detected
                raise RuntimeError(f"Circular dependency detected in tasks: {remaining_tasks}")
            
            # Group parallel and sequential tasks
            parallel_tasks = [t for t in ready_tasks if self.tasks[t].parallel]
            sequential_tasks = [t for t in ready_tasks if not self.tasks[t].parallel]
            
            # Add parallel tasks as one batch
            if parallel_tasks:
                execution_order.append(parallel_tasks)
            
            # Add sequential tasks as individual batches
            for task_id in sequential_tasks:
                execution_order.append([task_id])
            
            # Remove completed tasks
            remaining_tasks -= set(ready_tasks)
        
        logger.debug(f"Execution order: {execution_order}")
        return execution_order
    
    def _execute_tasks(self, execution_order: List[List[str]], input_data: Any) -> Dict[str, TaskResult]:
        """Execute tasks in resolved order.
        
        Args:
            execution_order: Task execution order
            input_data: Pipeline input data
            
        Returns:
            Dictionary of task results
        """
        task_results = {}
        task_outputs = {'_input': input_data}
        
        for batch in execution_order:
            if self.cancel_requested:
                # Cancel remaining tasks
                for task_id in batch:
                    task_results[task_id] = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.CANCELLED,
                        start_time=datetime.now()
                    )
                continue
            
            if len(batch) == 1:
                # Sequential execution
                task_id = batch[0]
                result = self._execute_single_task(task_id, task_outputs)
                task_results[task_id] = result
                
                if result.status == TaskStatus.SUCCESS:
                    task_outputs[task_id] = result.result
                elif not self.config.continue_on_failure:
                    break
            else:
                # Parallel execution
                batch_results = self._execute_parallel_tasks(batch, task_outputs)
                task_results.update(batch_results)
                
                # Update outputs
                for task_id, result in batch_results.items():
                    if result.status == TaskStatus.SUCCESS:
                        task_outputs[task_id] = result.result
                
                # Check for failures
                if not self.config.continue_on_failure:
                    failed_tasks = [r for r in batch_results.values() 
                                  if r.status == TaskStatus.FAILED]
                    if failed_tasks:
                        break
        
        return task_results
    
    def _execute_single_task(self, task_id: str, task_outputs: Dict[str, Any]) -> TaskResult:
        """Execute single task with retry logic.
        
        Args:
            task_id: Task identifier
            task_outputs: Available task outputs
            
        Returns:
            Task execution result
        """
        task_config = self.tasks[task_id]
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self._trigger_event('task_start', task_id=task_id)
        
        for attempt in range(task_config.retry_count + 1):
            try:
                result.attempts = attempt + 1
                
                if attempt > 0:
                    result.status = TaskStatus.RETRYING
                    time.sleep(task_config.retry_delay * attempt)
                    self._trigger_event('task_retry', task_id=task_id, attempt=attempt)
                
                # Prepare task inputs
                task_inputs = self._prepare_task_inputs(task_config, task_outputs)
                
                # Execute task with timeout
                if task_config.timeout:
                    future = self.executor.submit(task_config.task_function, **task_inputs)
                    task_result = future.result(timeout=task_config.timeout)
                else:
                    task_result = task_config.task_function(**task_inputs)
                
                result.status = TaskStatus.SUCCESS
                result.result = task_result
                break
                
            except Exception as e:
                error_msg = f"Task {task_id} failed (attempt {attempt + 1}): {str(e)}"
                logger.error(error_msg)
                result.error = error_msg
                
                if attempt >= task_config.retry_count:
                    result.status = TaskStatus.FAILED
                
        result.end_time = datetime.now()
        result.execution_time = (result.end_time - result.start_time).total_seconds()
        
        self._trigger_event('task_end', task_id=task_id, status=result.status)
        
        # Save intermediate results
        if self.config.save_intermediate_results and self.storage_path:
            self._save_task_result(result)
        
        return result
    
    def _execute_parallel_tasks(self, task_batch: List[str], 
                               task_outputs: Dict[str, Any]) -> Dict[str, TaskResult]:
        """Execute tasks in parallel.
        
        Args:
            task_batch: List of task IDs to execute in parallel
            task_outputs: Available task outputs
            
        Returns:
            Dictionary of task results
        """
        futures = {}
        
        # Submit all tasks
        for task_id in task_batch:
            future = self.executor.submit(self._execute_single_task, task_id, task_outputs)
            futures[future] = task_id
        
        # Collect results
        results = {}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                results[task_id] = result
            except Exception as e:
                logger.error(f"Parallel task {task_id} failed: {e}")
                results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error=str(e)
                )
        
        return results
    
    def _prepare_task_inputs(self, task_config: TaskConfig, 
                           task_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for task execution.
        
        Args:
            task_config: Task configuration
            task_outputs: Available task outputs
            
        Returns:
            Task input parameters
        """
        inputs = {}
        
        # Add dependency outputs
        for dep in task_config.dependencies:
            if dep in task_outputs:
                inputs[dep] = task_outputs[dep]
        
        # Add pipeline input if no dependencies and task function accepts it
        if not task_config.dependencies and '_input' in task_outputs:
            # Check if task function accepts input_data parameter
            import inspect
            sig = inspect.signature(task_config.task_function)
            if 'input_data' in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                inputs['input_data'] = task_outputs['_input']
        
        # Add task metadata only if function accepts it
        import inspect
        sig = inspect.signature(task_config.task_function)
        if 'task_metadata' in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            inputs['task_metadata'] = task_config.metadata
        
        return inputs
    
    def _create_checkpoint(self, input_data: Any):
        """Create execution checkpoint for recovery."""
        if not self.recovery_enabled:
            return
        
        checkpoint = {
            'pipeline_id': self.config.pipeline_id,
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data,
            'tasks': {tid: {'status': 'pending'} for tid in self.tasks.keys()}
        }
        
        self.checkpoint_data = checkpoint
        
        if self.storage_path:
            checkpoint_file = self.storage_path / f"checkpoint_{self.current_execution.execution_id}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
    
    def _save_execution_result(self, result: PipelineResult):
        """Save pipeline execution result."""
        if not self.storage_path:
            return
        
        result_file = self.storage_path / f"execution_{result.execution_id}.json"
        result_dict = {
            'pipeline_id': result.pipeline_id,
            'execution_id': result.execution_id,
            'status': result.status.value,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'execution_time': result.execution_time,
            'total_tasks': result.total_tasks,
            'successful_tasks': result.successful_tasks,
            'failed_tasks': result.failed_tasks,
            'task_results': {
                tid: {
                    'status': tr.status.value,
                    'start_time': tr.start_time.isoformat(),
                    'end_time': tr.end_time.isoformat() if tr.end_time else None,
                    'execution_time': tr.execution_time,
                    'attempts': tr.attempts,
                    'error': tr.error
                }
                for tid, tr in result.task_results.items()
            },
            'metadata': result.metadata
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def _save_task_result(self, result: TaskResult):
        """Save individual task result."""
        if not self.storage_path:
            return
        
        task_dir = self.storage_path / f"execution_{self.current_execution.execution_id}" / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = task_dir / f"{result.task_id}.json"
        result_dict = {
            'task_id': result.task_id,
            'status': result.status.value,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'execution_time': result.execution_time,
            'attempts': result.attempts,
            'error': result.error,
            'metadata': result.metadata
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def pause(self):
        """Pause pipeline execution."""
        if not self.is_running:
            raise RuntimeError("Pipeline is not running")
        
        self.is_paused = True
        logger.info(f"Pipeline paused: {self.config.pipeline_id}")
    
    def resume(self):
        """Resume paused pipeline execution."""
        if not self.is_paused:
            raise RuntimeError("Pipeline is not paused")
        
        self.is_paused = False
        logger.info(f"Pipeline resumed: {self.config.pipeline_id}")
    
    def cancel(self):
        """Cancel pipeline execution."""
        if not self.is_running:
            raise RuntimeError("Pipeline is not running")
        
        self.cancel_requested = True
        logger.info(f"Pipeline cancellation requested: {self.config.pipeline_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Pipeline status information
        """
        status = {
            'pipeline_id': self.config.pipeline_id,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'total_tasks': len(self.tasks),
            'execution_history_count': len(self.execution_history)
        }
        
        if self.current_execution:
            status.update({
                'current_execution_id': self.current_execution.execution_id,
                'current_status': self.current_execution.status.value,
                'successful_tasks': self.current_execution.successful_tasks,
                'failed_tasks': self.current_execution.failed_tasks,
                'execution_time': self.current_execution.execution_time
            })
        
        return status
    
    def get_execution_history(self) -> Dict[str, PipelineResult]:
        """Get pipeline execution history.
        
        Returns:
            Dictionary of execution results
        """
        return self.execution_history.copy()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.is_running:
            self.cancel()
        
        self.executor.shutdown(wait=True)
        logger.info(f"Pipeline orchestrator cleaned up: {self.config.pipeline_id}")


# Example usage and task functions
def example_data_extraction_task(input_data=None, **kwargs):
    """Example data extraction task."""
    time.sleep(1)  # Simulate work
    return {"extracted_data": "sample_data", "count": 100}


def example_processing_task(data_extraction=None, **kwargs):
    """Example data processing task."""
    time.sleep(2)  # Simulate work
    if data_extraction:
        return {"processed_data": f"processed_{data_extraction['extracted_data']}", 
                "count": data_extraction["count"] * 2}
    return {"processed_data": "default_processed", "count": 0}


def example_validation_task(data_processing=None, **kwargs):
    """Example validation task."""
    time.sleep(0.5)  # Simulate work
    if data_processing:
        return {"validation_result": True, "quality_score": 0.95}
    return {"validation_result": False, "quality_score": 0.0}


if __name__ == "__main__":
    # Example pipeline setup
    pipeline_config = PipelineConfig(
        pipeline_id="example_pipeline",
        name="Example Audio Processing Pipeline",
        max_workers=2,
        save_intermediate_results=True,
        result_storage_path="./pipeline_results"
    )
    
    orchestrator = PipelineOrchestrator(pipeline_config)
    
    # Add tasks
    orchestrator.add_task(TaskConfig(
        task_id="extract_data",
        task_function=example_data_extraction_task,
        dependencies=[],
        retry_count=2
    ))
    
    orchestrator.add_task(TaskConfig(
        task_id="process_data",
        task_function=example_processing_task,
        dependencies=["extract_data"],
        retry_count=3
    ))
    
    orchestrator.add_task(TaskConfig(
        task_id="validate_data",
        task_function=example_validation_task,
        dependencies=["process_data"],
        retry_count=1
    ))
    
    # Execute pipeline
    result = orchestrator.execute(input_data="sample_input")
    
    print(f"Pipeline Status: {result.status}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Successful Tasks: {result.successful_tasks}/{result.total_tasks}")
    
    # Cleanup
    orchestrator.cleanup()