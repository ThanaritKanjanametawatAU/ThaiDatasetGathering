"""Test suite for Model Registry and Versioning module (S04_T02).

This test suite validates the Model Registry system with the following requirements:
1. Model registration with semantic versioning
2. Model storage and retrieval with multiple formats
3. Performance tracking and metrics aggregation
4. A/B testing framework with statistical analysis
5. Deployment stage management
6. MLflow-style model lifecycle management
"""

import unittest
import tempfile
import shutil
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from processors.model_management.model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelStage,
    ModelFormat,
    PerformanceMetrics,
    ABTestConfig,
    ABTestResult,
    ExperimentStatus,
    LocalFileStorage,
    example_dummy_model
)


class TestModelRegistry(unittest.TestCase):
    """Test suite for Model Registry core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test registry
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(registry_path=self.temp_dir)
        
        # Create test models
        self.dummy_model_a = example_dummy_model()
        self.dummy_model_b = example_dummy_model()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.registry.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_registration(self):
        """Test 1.1: Model registration and versioning."""
        # Test model registration
        model_version = self.registry.register_model(
            model=self.dummy_model_a,
            model_id="test_model",
            version="1.0.0",
            name="Test Model",
            description="Test model for unit testing",
            stage=ModelStage.DEVELOPMENT,
            format=ModelFormat.PICKLE,
            tags={"framework": "test", "type": "classifier"},
            metrics={"accuracy": 0.85, "f1_score": 0.82},
            parameters={"learning_rate": 0.001, "epochs": 100}
        )
        
        # Verify registration
        self.assertEqual(model_version.model_id, "test_model")
        self.assertEqual(model_version.version, "1.0.0")
        self.assertEqual(model_version.name, "Test Model")
        self.assertEqual(model_version.stage, ModelStage.DEVELOPMENT)
        self.assertEqual(model_version.format, ModelFormat.PICKLE)
        self.assertEqual(model_version.tags["framework"], "test")
        self.assertEqual(model_version.metrics["accuracy"], 0.85)
        self.assertIsNotNone(model_version.file_path)
        self.assertGreater(model_version.file_size, 0)
        self.assertIsNotNone(model_version.checksum)
        
        # Test duplicate version registration
        with self.assertRaises(ValueError):
            self.registry.register_model(
                model=self.dummy_model_b,
                model_id="test_model",
                version="1.0.0",
                name="Duplicate Model"
            )
    
    def test_model_retrieval(self):
        """Test 1.2: Model storage and retrieval."""
        # Register model
        self.registry.register_model(
            model=self.dummy_model_a,
            model_id="retrieval_test",
            version="1.0.0",
            name="Retrieval Test Model",
            stage=ModelStage.STAGING
        )
        
        # Test model retrieval by version
        retrieved_model = self.registry.get_model("retrieval_test", version="1.0.0")
        self.assertIsNotNone(retrieved_model)
        self.assertEqual(retrieved_model.accuracy, self.dummy_model_a.accuracy)
        
        # Test model version metadata retrieval
        model_version = self.registry.get_model_version("retrieval_test", version="1.0.0")
        self.assertEqual(model_version.model_id, "retrieval_test")
        self.assertEqual(model_version.version, "1.0.0")
        self.assertEqual(model_version.stage, ModelStage.STAGING)
        
        # Test retrieval by stage
        retrieved_model_by_stage = self.registry.get_model("retrieval_test", stage=ModelStage.STAGING)
        self.assertIsNotNone(retrieved_model_by_stage)
        
        # Test non-existent model
        with self.assertRaises(ValueError):
            self.registry.get_model("non_existent")
        
        # Test non-existent version
        with self.assertRaises(ValueError):
            self.registry.get_model("retrieval_test", version="2.0.0")
    
    def test_model_stage_management(self):
        """Test 1.3: Model deployment stage management."""
        # Register model
        model_version = self.registry.register_model(
            model=self.dummy_model_a,
            model_id="stage_test",
            version="1.0.0",
            name="Stage Test Model",
            stage=ModelStage.DEVELOPMENT
        )
        
        # Test stage update
        updated_version = self.registry.update_model_stage(
            "stage_test", "1.0.0", ModelStage.STAGING
        )
        self.assertEqual(updated_version.stage, ModelStage.STAGING)
        
        # Verify stage was persisted
        retrieved_version = self.registry.get_model_version("stage_test", "1.0.0")
        self.assertEqual(retrieved_version.stage, ModelStage.STAGING)
        
        # Test promotion to production
        self.registry.update_model_stage("stage_test", "1.0.0", ModelStage.PRODUCTION)
        prod_model = self.registry.get_model("stage_test", stage=ModelStage.PRODUCTION)
        self.assertIsNotNone(prod_model)
    
    def test_model_listing(self):
        """Test 1.4: Model listing and filtering."""
        # Register multiple models and versions
        self.registry.register_model(
            model=self.dummy_model_a,
            model_id="list_test_1",
            version="1.0.0",
            name="List Test Model 1",
            stage=ModelStage.DEVELOPMENT
        )
        
        self.registry.register_model(
            model=self.dummy_model_b,
            model_id="list_test_1",
            version="2.0.0",
            name="List Test Model 1 v2",
            stage=ModelStage.STAGING
        )
        
        self.registry.register_model(
            model=self.dummy_model_a,
            model_id="list_test_2",
            version="1.0.0",
            name="List Test Model 2",
            stage=ModelStage.PRODUCTION
        )
        
        # Test listing all models
        all_models = self.registry.list_models()
        self.assertEqual(len(all_models), 3)
        
        # Test filtering by stage
        staging_models = self.registry.list_models(stage=ModelStage.STAGING)
        self.assertEqual(len(staging_models), 1)
        self.assertEqual(staging_models[0].model_id, "list_test_1")
        self.assertEqual(staging_models[0].version, "2.0.0")
        
        production_models = self.registry.list_models(stage=ModelStage.PRODUCTION)
        self.assertEqual(len(production_models), 1)
        self.assertEqual(production_models[0].model_id, "list_test_2")
    
    def test_model_deletion(self):
        """Test 1.5: Model version deletion."""
        # Register model
        self.registry.register_model(
            model=self.dummy_model_a,
            model_id="delete_test",
            version="1.0.0",
            name="Delete Test Model"
        )
        
        # Verify model exists
        self.assertIsNotNone(self.registry.get_model("delete_test", "1.0.0"))
        
        # Delete model version
        success = self.registry.delete_model_version("delete_test", "1.0.0")
        self.assertTrue(success)
        
        # Verify model is deleted
        with self.assertRaises(ValueError):
            self.registry.get_model("delete_test", "1.0.0")
        
        # Test deleting non-existent model
        success = self.registry.delete_model_version("non_existent", "1.0.0")
        self.assertFalse(success)
    
    def test_performance_tracking(self):
        """Test 1.6: Performance metrics tracking."""
        # Register model
        self.registry.register_model(
            model=self.dummy_model_a,
            model_id="perf_test",
            version="1.0.0",
            name="Performance Test Model"
        )
        
        # Record performance metrics
        performance1 = PerformanceMetrics(
            model_id="perf_test",
            version="1.0.0",
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            inference_time_ms=42.5,
            memory_usage_mb=256.0,
            throughput_requests_per_sec=23.5,
            environment="production",
            sample_count=1000
        )
        self.registry.record_performance(performance1)
        
        performance2 = PerformanceMetrics(
            model_id="perf_test",
            version="1.0.0",
            accuracy=0.87,
            precision=0.85,
            recall=0.89,
            f1_score=0.87,
            inference_time_ms=38.2,
            memory_usage_mb=248.0,
            throughput_requests_per_sec=26.1,
            environment="production",
            sample_count=1500
        )
        self.registry.record_performance(performance2)
        
        # Test performance history retrieval
        history = self.registry.get_performance_history("perf_test")
        self.assertEqual(len(history), 2)
        
        # Verify latest metrics come first
        self.assertEqual(history[0].accuracy, 0.87)
        self.assertEqual(history[1].accuracy, 0.85)
        
        # Test filtering by version
        version_history = self.registry.get_performance_history("perf_test", version="1.0.0")
        self.assertEqual(len(version_history), 2)
        
        # Test filtering by environment
        prod_history = self.registry.get_performance_history("perf_test", environment="production")
        self.assertEqual(len(prod_history), 2)
        
        # Test limit
        limited_history = self.registry.get_performance_history("perf_test", limit=1)
        self.assertEqual(len(limited_history), 1)


class TestABTesting(unittest.TestCase):
    """Test suite for A/B Testing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(registry_path=self.temp_dir)
        
        # Register test models
        self.registry.register_model(
            model=example_dummy_model(),
            model_id="ab_test_model",
            version="1.0.0",
            name="A/B Test Model A",
            stage=ModelStage.PRODUCTION
        )
        
        self.registry.register_model(
            model=example_dummy_model(),
            model_id="ab_test_model",
            version="1.1.0",
            name="A/B Test Model B",
            stage=ModelStage.STAGING
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.registry.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ab_test_creation(self):
        """Test 2.1: A/B test experiment creation."""
        config = ABTestConfig(
            experiment_id="test_experiment_1",
            name="Model A vs Model B",
            description="Testing version 1.0.0 vs 1.1.0",
            model_a_id="ab_test_model",
            model_a_version="1.0.0",
            model_b_id="ab_test_model",
            model_b_version="1.1.0",
            traffic_split=0.5,
            success_metric="accuracy",
            minimum_sample_size=1000,
            confidence_level=0.95,
            duration_hours=24
        )
        
        created_config = self.registry.create_ab_test(config)
        
        # Verify experiment creation
        self.assertEqual(created_config.experiment_id, "test_experiment_1")
        self.assertEqual(created_config.model_a_id, "ab_test_model")
        self.assertEqual(created_config.model_a_version, "1.0.0")
        self.assertEqual(created_config.model_b_id, "ab_test_model")
        self.assertEqual(created_config.model_b_version, "1.1.0")
        self.assertEqual(created_config.traffic_split, 0.5)
        self.assertIsNotNone(created_config.end_time)
        
        # Test duplicate experiment
        with self.assertRaises(ValueError):
            self.registry.create_ab_test(config)
        
        # Test invalid model references
        invalid_config = ABTestConfig(
            experiment_id="invalid_experiment",
            name="Invalid Test",
            model_a_id="non_existent",
            model_a_version="1.0.0",
            model_b_id="ab_test_model",
            model_b_version="1.1.0"
        )
        
        with self.assertRaises(ValueError):
            self.registry.create_ab_test(invalid_config)
    
    def test_ab_test_management(self):
        """Test 2.2: A/B test status management."""
        config = ABTestConfig(
            experiment_id="management_test",
            name="Management Test",
            model_a_id="ab_test_model",
            model_a_version="1.0.0",
            model_b_id="ab_test_model",
            model_b_version="1.1.0"
        )
        
        self.registry.create_ab_test(config)
        
        # Test status update
        self.registry.update_ab_test_status("management_test", ExperimentStatus.PAUSED)
        
        updated_config = self.registry.get_ab_test("management_test")
        self.assertEqual(updated_config.status, ExperimentStatus.PAUSED)
        
        # Test completion
        self.registry.update_ab_test_status("management_test", ExperimentStatus.COMPLETED)
        
        completed_config = self.registry.get_ab_test("management_test")
        self.assertEqual(completed_config.status, ExperimentStatus.COMPLETED)
        self.assertIsNotNone(completed_config.end_time)
    
    def test_ab_test_analysis(self):
        """Test 2.3: A/B test statistical analysis."""
        # Create experiment with start time in the past to include metrics
        start_time = datetime.now() - timedelta(hours=1)
        config = ABTestConfig(
            experiment_id="analysis_test",
            name="Analysis Test",
            model_a_id="ab_test_model",
            model_a_version="1.0.0",
            model_b_id="ab_test_model",
            model_b_version="1.1.0",
            success_metric="accuracy",
            start_time=start_time,
            end_time=datetime.now() + timedelta(hours=1)
        )
        self.registry.create_ab_test(config)
        
        # Generate performance data for model A (lower performance)
        for i in range(20):
            perf_a = PerformanceMetrics(
                model_id="ab_test_model",
                version="1.0.0",
                accuracy=0.80 + (i * 0.005),  # 0.80 to 0.895
                precision=0.78 + (i * 0.005),
                recall=0.82 + (i * 0.005),
                f1_score=0.80 + (i * 0.005),
                sample_count=100,
                timestamp=datetime.now() + timedelta(minutes=i)
            )
            self.registry.record_performance(perf_a)
        
        # Generate performance data for model B (higher performance)
        for i in range(20):
            perf_b = PerformanceMetrics(
                model_id="ab_test_model",
                version="1.1.0",
                accuracy=0.85 + (i * 0.005),  # 0.85 to 0.945
                precision=0.83 + (i * 0.005),
                recall=0.87 + (i * 0.005),
                f1_score=0.85 + (i * 0.005),
                sample_count=100,
                timestamp=datetime.now() + timedelta(minutes=i)
            )
            self.registry.record_performance(perf_b)
        
        # Analyze experiment
        result = self.registry.analyze_ab_test("analysis_test")
        
        # Verify analysis results
        self.assertEqual(result.experiment_id, "analysis_test")
        self.assertIsNotNone(result.model_a_metrics)
        self.assertIsNotNone(result.model_b_metrics)
        self.assertEqual(result.sample_sizes["model_a"], 20)
        self.assertEqual(result.sample_sizes["model_b"], 20)
        
        # Model B should perform better
        self.assertGreater(result.model_b_metrics.accuracy, result.model_a_metrics.accuracy)
        
        # Check if statistical significance is detected
        if result.statistical_significance:
            self.assertEqual(result.winner, "B")
            self.assertIsNotNone(result.p_value)
        
        self.assertIsNotNone(result.recommendation)
    
    def test_insufficient_data_analysis(self):
        """Test 2.4: A/B test analysis with insufficient data."""
        # Create experiment
        config = ABTestConfig(
            experiment_id="insufficient_data_test",
            name="Insufficient Data Test",
            model_a_id="ab_test_model",
            model_a_version="1.0.0",
            model_b_id="ab_test_model",
            model_b_version="1.1.0"
        )
        self.registry.create_ab_test(config)
        
        # Analyze without any performance data
        result = self.registry.analyze_ab_test("insufficient_data_test")
        
        # Should handle insufficient data gracefully
        self.assertEqual(result.experiment_id, "insufficient_data_test")
        self.assertFalse(result.statistical_significance)
        self.assertIn("Insufficient data", result.recommendation)


class TestModelStorage(unittest.TestCase):
    """Test suite for Model Storage functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = LocalFileStorage(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_local_file_storage(self):
        """Test 3.1: Local file storage operations."""
        # Create test model
        dummy_model = example_dummy_model()
        
        model_version = ModelVersion(
            model_id="storage_test",
            version="1.0.0",
            name="Storage Test Model",
            format=ModelFormat.PICKLE
        )
        
        # Test save
        file_path = self.storage.save_model(dummy_model, model_version)
        model_version.file_path = file_path  # Ensure file_path is set
        self.assertIsNotNone(file_path)
        self.assertTrue(Path(file_path).exists())
        self.assertGreater(model_version.file_size, 0)
        self.assertIsNotNone(model_version.checksum)
        
        # Test load
        loaded_model = self.storage.load_model(model_version)
        self.assertIsNotNone(loaded_model)
        self.assertEqual(loaded_model.accuracy, dummy_model.accuracy)
        
        # Test delete
        success = self.storage.delete_model(model_version)
        self.assertTrue(success)
        self.assertFalse(Path(file_path).exists())
    
    def test_checksum_verification(self):
        """Test 3.2: File integrity verification."""
        dummy_model = example_dummy_model()
        
        model_version = ModelVersion(
            model_id="checksum_test",
            version="1.0.0",
            name="Checksum Test Model",
            format=ModelFormat.PICKLE
        )
        
        # Save model
        file_path = self.storage.save_model(dummy_model, model_version)
        model_version.file_path = file_path  # Ensure file_path is set
        original_checksum = model_version.checksum
        
        # Corrupt the file
        with open(file_path, 'a') as f:
            f.write("corrupted data")
        
        # Loading should fail due to checksum mismatch
        with self.assertRaises(ValueError):
            self.storage.load_model(model_version)
    
    def test_model_formats(self):
        """Test 3.3: Different model format handling."""
        dummy_model = example_dummy_model()
        
        # Test different formats
        formats_to_test = [
            ModelFormat.PICKLE,
            ModelFormat.CUSTOM
        ]
        
        for model_format in formats_to_test:
            with self.subTest(format=model_format):
                model_version = ModelVersion(
                    model_id="format_test",
                    version=f"1.0.0_{model_format.value}",
                    name=f"Format Test Model {model_format.value}",
                    format=model_format
                )
                
                # Save and load
                file_path = self.storage.save_model(dummy_model, model_version)
                model_version.file_path = file_path  # Ensure file_path is set
                loaded_model = self.storage.load_model(model_version)
                
                # Verify
                self.assertIsNotNone(loaded_model)
                self.assertEqual(loaded_model.accuracy, dummy_model.accuracy)


class TestIntegration(unittest.TestCase):
    """Integration tests for model registry system."""
    
    def test_complete_model_lifecycle(self):
        """Test 4.1: Complete model lifecycle management."""
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_path=temp_dir)
        
        try:
            # 1. Register initial model
            model_v1 = registry.register_model(
                model=example_dummy_model(),
                model_id="lifecycle_model",
                version="1.0.0",
                name="Lifecycle Test Model",
                description="Initial version",
                stage=ModelStage.DEVELOPMENT,
                metrics={"accuracy": 0.80}
            )
            
            # 2. Promote to staging
            registry.update_model_stage("lifecycle_model", "1.0.0", ModelStage.STAGING)
            
            # 3. Record performance in staging
            staging_perf = PerformanceMetrics(
                model_id="lifecycle_model",
                version="1.0.0",
                accuracy=0.82,
                f1_score=0.79,
                environment="staging",
                sample_count=500
            )
            registry.record_performance(staging_perf)
            
            # 4. Deploy to production
            registry.update_model_stage("lifecycle_model", "1.0.0", ModelStage.PRODUCTION)
            
            # 5. Register improved version
            model_v2 = registry.register_model(
                model=example_dummy_model(),
                model_id="lifecycle_model",
                version="2.0.0",
                name="Lifecycle Test Model v2",
                description="Improved version",
                stage=ModelStage.STAGING,
                metrics={"accuracy": 0.85}
            )
            
            # 6. Set up A/B test
            ab_config = ABTestConfig(
                experiment_id="lifecycle_ab_test",
                name="v1.0.0 vs v2.0.0",
                model_a_id="lifecycle_model",
                model_a_version="1.0.0",
                model_b_id="lifecycle_model",
                model_b_version="2.0.0",
                success_metric="accuracy",
                duration_hours=1
            )
            registry.create_ab_test(ab_config)
            
            # 7. Generate performance data
            for i in range(15):
                # v1 performance
                perf_v1 = PerformanceMetrics(
                    model_id="lifecycle_model",
                    version="1.0.0",
                    accuracy=0.82 + (i * 0.002),
                    f1_score=0.79 + (i * 0.002),
                    environment="production",
                    sample_count=100
                )
                registry.record_performance(perf_v1)
                
                # v2 performance (better)
                perf_v2 = PerformanceMetrics(
                    model_id="lifecycle_model",
                    version="2.0.0",
                    accuracy=0.85 + (i * 0.002),
                    f1_score=0.82 + (i * 0.002),
                    environment="staging",
                    sample_count=100
                )
                registry.record_performance(perf_v2)
            
            # 8. Analyze A/B test
            ab_result = registry.analyze_ab_test("lifecycle_ab_test")
            
            # 9. Complete experiment
            registry.update_ab_test_status("lifecycle_ab_test", ExperimentStatus.COMPLETED)
            
            # 10. Promote winning model
            if ab_result.winner == "B":
                registry.update_model_stage("lifecycle_model", "2.0.0", ModelStage.PRODUCTION)
                registry.update_model_stage("lifecycle_model", "1.0.0", ModelStage.ARCHIVED)
            
            # Verify final state
            models = registry.list_models()
            self.assertEqual(len(models), 2)
            
            prod_model = registry.get_model("lifecycle_model", stage=ModelStage.PRODUCTION)
            self.assertIsNotNone(prod_model)
            
            history = registry.get_performance_history("lifecycle_model")
            self.assertGreater(len(history), 0)
            
            final_ab_config = registry.get_ab_test("lifecycle_ab_test")
            self.assertEqual(final_ab_config.status, ExperimentStatus.COMPLETED)
            
        finally:
            registry.shutdown()
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_registry_persistence(self):
        """Test 4.2: Registry persistence and recovery."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create registry and add models
            registry1 = ModelRegistry(registry_path=temp_dir)
            
            registry1.register_model(
                model=example_dummy_model(),
                model_id="persistence_test",
                version="1.0.0",
                name="Persistence Test Model",
                metrics={"accuracy": 0.88}
            )
            
            # Record performance
            perf = PerformanceMetrics(
                model_id="persistence_test",
                version="1.0.0",
                accuracy=0.89,
                sample_count=1000
            )
            registry1.record_performance(perf)
            
            # Create A/B test
            ab_config = ABTestConfig(
                experiment_id="persistence_ab_test",
                name="Persistence Test",
                model_a_id="persistence_test",
                model_a_version="1.0.0",
                model_b_id="persistence_test",
                model_b_version="1.0.0"
            )
            registry1.create_ab_test(ab_config)
            
            registry1.shutdown()
            
            # Create new registry instance (simulating restart)
            registry2 = ModelRegistry(registry_path=temp_dir)
            
            # Verify data was persisted
            models = registry2.list_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].model_id, "persistence_test")
            
            # Verify model can be loaded
            loaded_model = registry2.get_model("persistence_test", "1.0.0")
            self.assertIsNotNone(loaded_model)
            
            # Verify performance history
            history = registry2.get_performance_history("persistence_test")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].accuracy, 0.89)
            
            # Verify A/B test
            ab_test = registry2.get_ab_test("persistence_ab_test")
            self.assertEqual(ab_test.experiment_id, "persistence_ab_test")
            
            registry2.shutdown()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()