"""Test suite for Pipeline Orchestration module (S04_T01).

This test suite validates the Pipeline Orchestration system with the following requirements:
1. Pipeline task execution with dependency resolution
2. Parallel and sequential task processing
3. Error handling and retry mechanisms
4. Pipeline monitoring and status reporting
5. REST API management and control
"""

import unittest
import time
import tempfile
import shutil
import json
import threading
from unittest.mock import Mock, patch, MagicMock
from processors.orchestration.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineConfig,
    TaskConfig,
    TaskStatus,
    PipelineStatus
)


class TestPipelineOrchestrator(unittest.TestCase):
    """Test suite for Pipeline Orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test results
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = PipelineConfig(
            pipeline_id="test_pipeline",
            name="Test Pipeline",
            max_workers=2,
            result_storage_path=self.temp_dir
        )
        
        self.orchestrator = PipelineOrchestrator(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.orchestrator.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_task_management(self):
        """Test 1.1: Task addition and removal."""
        # Test task addition
        task_config = TaskConfig(
            task_id="test_task",
            task_function=lambda: "test_result",
            dependencies=[],
            retry_count=2
        )
        
        self.orchestrator.add_task(task_config)
        self.assertIn("test_task", self.orchestrator.tasks)
        
        # Test duplicate task addition
        with self.assertRaises(ValueError):
            self.orchestrator.add_task(task_config)
        
        # Test task removal
        self.orchestrator.remove_task("test_task")
        self.assertNotIn("test_task", self.orchestrator.tasks)
        
        # Test removing non-existent task
        with self.assertRaises(ValueError):
            self.orchestrator.remove_task("non_existent")
    
    def test_simple_pipeline_execution(self):
        """Test 1.2: Simple pipeline execution."""
        # Create simple tasks
        def task_1():
            time.sleep(0.1)
            return "result_1"
        
        def task_2(task_1=None):
            time.sleep(0.1)
            return f"result_2_{task_1}"
        
        # Add tasks
        self.orchestrator.add_task(TaskConfig(
            task_id="task_1",
            task_function=task_1
        ))
        
        self.orchestrator.add_task(TaskConfig(
            task_id="task_2",
            task_function=task_2,
            dependencies=["task_1"]
        ))
        
        # Execute pipeline
        result = self.orchestrator.execute()
        
        # Verify execution
        self.assertEqual(result.status, PipelineStatus.SUCCESS)
        self.assertEqual(result.total_tasks, 2)
        self.assertEqual(result.successful_tasks, 2)
        self.assertEqual(result.failed_tasks, 0)
        self.assertGreater(result.execution_time, 0)
        
        # Verify task results
        self.assertIn("task_1", result.task_results)
        self.assertIn("task_2", result.task_results)
        
        task_1_result = result.task_results["task_1"]
        task_2_result = result.task_results["task_2"]
        
        self.assertEqual(task_1_result.status, TaskStatus.SUCCESS)
        self.assertEqual(task_2_result.status, TaskStatus.SUCCESS)
        self.assertEqual(task_1_result.result, "result_1")
        self.assertEqual(task_2_result.result, "result_2_result_1")
    
    def test_error_handling_and_retry(self):
        """Test 1.3: Error handling and retry mechanisms."""
        attempt_count = 0
        
        def failing_task():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError(f"Attempt {attempt_count} failed")
            return "success_after_retries"
        
        # Add task with retry
        self.orchestrator.add_task(TaskConfig(
            task_id="failing_task",
            task_function=failing_task,
            retry_count=3,
            retry_delay=0.1
        ))
        
        # Execute pipeline
        result = self.orchestrator.execute()
        
        # Verify retry behavior
        self.assertEqual(result.status, PipelineStatus.SUCCESS)
        self.assertEqual(result.successful_tasks, 1)
        
        task_result = result.task_results["failing_task"]
        self.assertEqual(task_result.status, TaskStatus.SUCCESS)
        self.assertEqual(task_result.attempts, 3)
        self.assertEqual(task_result.result, "success_after_retries")
    
    def test_task_failure_handling(self):
        """Test 1.4: Task failure handling."""
        def failing_task():
            raise RuntimeError("Task always fails")
        
        def success_task():
            return "success"
        
        # Add failing task
        self.orchestrator.add_task(TaskConfig(
            task_id="failing_task",
            task_function=failing_task,
            retry_count=1
        ))
        
        # Add dependent task
        self.orchestrator.add_task(TaskConfig(
            task_id="dependent_task",
            task_function=success_task,
            dependencies=["failing_task"]
        ))
        
        # Execute pipeline
        result = self.orchestrator.execute()
        
        # Verify failure handling
        self.assertEqual(result.status, PipelineStatus.FAILED)
        self.assertEqual(result.failed_tasks, 1)
        
        failing_result = result.task_results["failing_task"]
        self.assertEqual(failing_result.status, TaskStatus.FAILED)
        self.assertIsNotNone(failing_result.error)
    
    def test_status_monitoring(self):
        """Test 1.5: Pipeline status monitoring."""
        # Test initial status
        status = self.orchestrator.get_status()
        self.assertEqual(status['pipeline_id'], 'test_pipeline')
        self.assertFalse(status['is_running'])
        self.assertEqual(status['total_tasks'], 0)
        
        # Add a task
        self.orchestrator.add_task(TaskConfig(
            task_id="monitor_task",
            task_function=lambda: "result"
        ))
        
        # Check status after adding task
        status = self.orchestrator.get_status()
        self.assertEqual(status['total_tasks'], 1)
        
        # Execute and check status
        result = self.orchestrator.execute()
        
        # Check execution history
        history = self.orchestrator.get_execution_history()
        self.assertEqual(len(history), 1)
        self.assertIn(result.execution_id, history)
    
    def test_result_storage(self):
        """Test 1.6: Result storage and persistence."""
        # Add task
        self.orchestrator.add_task(TaskConfig(
            task_id="storage_task",
            task_function=lambda: {"data": "test_data", "count": 42}
        ))
        
        # Execute pipeline
        result = self.orchestrator.execute()
        
        # Verify result files are created
        execution_file = self.orchestrator.storage_path / f"execution_{result.execution_id}.json"
        self.assertTrue(execution_file.exists())
        
        # Load and verify result file
        with open(execution_file, 'r') as f:
            saved_result = json.load(f)
        
        self.assertEqual(saved_result['pipeline_id'], 'test_pipeline')
        self.assertEqual(saved_result['status'], 'success')
        self.assertEqual(saved_result['successful_tasks'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for pipeline orchestration."""
    
    def test_audio_processing_pipeline(self):
        """Test 2.1: Audio processing pipeline integration."""
        # Create pipeline for audio processing workflow
        config = PipelineConfig(
            pipeline_id="audio_pipeline",
            name="Audio Processing Pipeline",
            max_workers=2
        )
        
        orchestrator = PipelineOrchestrator(config)
        
        # Mock audio processing tasks
        def load_audio(input_data=None, **kwargs):
            return {"audio_data": "mock_audio", "sample_rate": 16000}
        
        def preprocess_audio(load=None, **kwargs):
            return {"preprocessed": f"prep_{load['audio_data']}", 
                   "sample_rate": load["sample_rate"]}
        
        def extract_features(preprocess=None, **kwargs):
            return {"features": f"feat_{preprocess['preprocessed']}"}
        
        def classify_audio(extract_features=None, **kwargs):
            return {"classification": "speech", "confidence": 0.95}
        
        # Add tasks
        orchestrator.add_task(TaskConfig(
            task_id="load",
            task_function=load_audio
        ))
        
        orchestrator.add_task(TaskConfig(
            task_id="preprocess",
            task_function=preprocess_audio,
            dependencies=["load"]
        ))
        
        orchestrator.add_task(TaskConfig(
            task_id="extract_features",
            task_function=extract_features,
            dependencies=["preprocess"]
        ))
        
        orchestrator.add_task(TaskConfig(
            task_id="classify",
            task_function=classify_audio,
            dependencies=["extract_features"]
        ))
        
        # Execute pipeline
        result = orchestrator.execute(input_data="test_audio.wav")
        
        # Verify execution
        self.assertEqual(result.status, PipelineStatus.SUCCESS)
        self.assertEqual(result.successful_tasks, 4)
        
        # Verify task chain
        classify_result = result.task_results["classify"]
        self.assertEqual(classify_result.status, TaskStatus.SUCCESS)
        self.assertEqual(classify_result.result["classification"], "speech")
        self.assertEqual(classify_result.result["confidence"], 0.95)
        
        orchestrator.cleanup()


if __name__ == "__main__":
    unittest.main()