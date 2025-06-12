"""Test suite for Cost Optimization and Resource Management module (S06_T20).

This test suite validates the Cost Optimization system with the following requirements:
1. Spot instance management with interruption handling
2. Resource usage analytics with pattern recognition
3. Dynamic batch size auto-tuning
4. Cost-aware scheduling and resource pooling
5. Automated optimization actions with real-time monitoring
"""

import unittest
import time
import json
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from collections import deque

from processors.cost_optimization.cost_optimizer import (
    CostOptimizer,
    ResourceMonitor,
    CostTracker,
    DynamicBatchSizer,
    SpotInstanceManager,
    AutomatedOptimizer,
    ResourceMetrics,
    CostMetrics,
    BatchSizeConfig,
    SpotInstanceConfig,
    OptimizationRule,
    InstanceType,
    OptimizationAction,
    PriorityLevel
)


class TestResourceMonitor(unittest.TestCase):
    """Test suite for resource monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = ResourceMonitor(collection_interval=1)  # 1 second for testing
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.monitor.stop_monitoring()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    def test_metrics_collection(self, mock_pids, mock_net_io, mock_disk, mock_memory, mock_cpu):
        """Test 1.1: Resource metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 45.5
        
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 67.8
        mock_memory_obj.available = 8 * 1024**3  # 8GB
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = Mock()
        mock_disk_obj.percent = 78.2
        mock_disk_obj.free = 100 * 1024**3  # 100GB
        mock_disk.return_value = mock_disk_obj
        
        mock_net_obj = Mock()
        mock_net_obj.bytes_sent = 1000000
        mock_net_obj.bytes_recv = 2000000
        mock_net_io.return_value = mock_net_obj
        
        mock_pids.return_value = list(range(150))  # 150 processes
        
        # Collect metrics
        metrics = self.monitor._collect_metrics()
        
        # Verify metrics
        self.assertEqual(metrics.cpu_percent, 45.5)
        self.assertEqual(metrics.memory_percent, 67.8)
        self.assertAlmostEqual(metrics.memory_available_gb, 8.0, places=1)
        self.assertEqual(metrics.disk_usage_percent, 78.2)
        self.assertAlmostEqual(metrics.disk_available_gb, 100.0, places=1)
        self.assertEqual(metrics.active_processes, 150)
        self.assertIsInstance(metrics.timestamp, datetime)
    
    def test_monitoring_lifecycle(self):
        """Test 1.2: Monitoring start/stop lifecycle."""
        # Initially not monitoring
        self.assertFalse(self.monitor.is_monitoring)
        
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_monitoring)
        self.assertIsNotNone(self.monitor.monitor_thread)
        
        # Wait briefly for metrics collection
        time.sleep(1.5)
        
        # Should have collected some metrics
        self.assertGreater(len(self.monitor.metrics_history), 0)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_monitoring)
    
    def test_metrics_history_management(self):
        """Test 1.3: Metrics history management."""
        # Add test metrics (most recent first in deque)
        for i in range(5):
            metrics = ResourceMetrics(
                cpu_percent=i * 10.0,
                memory_percent=i * 15.0,
                timestamp=datetime.now() - timedelta(minutes=4-i)  # Reverse order for chronological
            )
            self.monitor.metrics_history.append(metrics)
        
        # Test current metrics (should be the last one added)
        current = self.monitor.get_current_metrics()
        self.assertIsNotNone(current)
        self.assertEqual(current.cpu_percent, 40.0)  # Last added (i=4, 4*10.0)
        
        # Test history retrieval
        history = self.monitor.get_metrics_history(hours=1)
        self.assertEqual(len(history), 5)
        
        # Test average calculation
        averages = self.monitor.get_average_metrics(hours=1)
        expected_cpu_avg = sum(i * 10.0 for i in range(5)) / 5
        self.assertAlmostEqual(averages["cpu_percent"], expected_cpu_avg, places=1)


class TestCostTracker(unittest.TestCase):
    """Test suite for cost tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        pricing_config = {
            "aws": {
                "t3.medium": {"on_demand": 0.0416, "spot": 0.0125},
                "m5.large": {"on_demand": 0.096, "spot": 0.029}
            }
        }
        self.tracker = CostTracker(pricing_config)
    
    def test_cost_metrics_recording(self):
        """Test 2.1: Cost metrics recording."""
        # Record cost metrics
        metrics = self.tracker.record_cost_metrics(
            instance_type=InstanceType.SPOT,
            instance_size="t3.medium",
            samples_processed=1000,
            runtime_hours=2.0,
            region="us-east-1"
        )
        
        # Verify metrics
        self.assertEqual(metrics.instance_type, InstanceType.SPOT)
        self.assertEqual(metrics.instance_size, "t3.medium")
        self.assertEqual(metrics.samples_processed, 1000)
        self.assertEqual(metrics.hourly_cost, 0.0125)  # Spot price
        self.assertEqual(metrics.cumulative_cost, 0.025)  # 2 hours * 0.0125
        self.assertAlmostEqual(metrics.spot_savings, 0.0291, places=4)  # 0.0416 - 0.0125
        self.assertGreater(metrics.efficiency_score, 0.0)
        self.assertEqual(metrics.cost_per_sample, 0.025 / 1000)
    
    def test_cost_analysis(self):
        """Test 2.2: Cost analysis and reporting."""
        # Record multiple cost sessions
        test_sessions = [
            (InstanceType.SPOT, "t3.medium", 500, 1.0),
            (InstanceType.ON_DEMAND, "t3.medium", 300, 0.5),
            (InstanceType.SPOT, "m5.large", 800, 1.5)
        ]
        
        for instance_type, instance_size, samples, hours in test_sessions:
            self.tracker.record_cost_metrics(
                instance_type=instance_type,
                instance_size=instance_size,
                samples_processed=samples,
                runtime_hours=hours
            )
        
        # Get cost analysis
        analysis = self.tracker.get_cost_analysis(days=7)
        
        # Verify analysis
        self.assertIn("total_cost", analysis)
        self.assertIn("total_spot_savings", analysis)
        self.assertIn("total_samples_processed", analysis)
        self.assertIn("spot_usage_percentage", analysis)
        
        self.assertEqual(analysis["total_samples_processed"], 1600)  # 500 + 300 + 800
        self.assertGreater(analysis["total_cost"], 0)
        self.assertGreater(analysis["spot_usage_percentage"], 0)
    
    def test_budget_management(self):
        """Test 2.3: Budget alerts and management."""
        # Set daily budget
        today = datetime.now().date().isoformat()
        self.tracker.set_daily_budget(today, 10.0)
        
        # Verify budget is set
        self.assertEqual(self.tracker.daily_budgets[today], 10.0)
        
        # Record high cost to trigger alert (would log warning)
        with self.assertLogs(level='WARNING') as log:
            # Record cost that exceeds budget
            for i in range(5):
                self.tracker.record_cost_metrics(
                    instance_type=InstanceType.ON_DEMAND,
                    instance_size="m5.large",
                    samples_processed=100,
                    runtime_hours=30.0  # High hours to exceed budget
                )
            
            # Check if warning was logged
            self.assertTrue(any("budget exceeded" in message.lower() for message in log.output))


class TestDynamicBatchSizer(unittest.TestCase):
    """Test suite for dynamic batch sizing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = BatchSizeConfig(
            current_batch_size=32,
            min_batch_size=8,
            max_batch_size=128,
            target_memory_utilization=0.8,
            target_processing_time=60.0
        )
        self.sizer = DynamicBatchSizer(config)
    
    def test_batch_size_adjustment(self):
        """Test 3.1: Batch size adjustment logic."""
        initial_batch_size = self.sizer.get_optimal_batch_size()
        self.assertEqual(initial_batch_size, 32)
        
        # Simulate high memory usage (should decrease batch size)
        self.sizer.update_performance(
            processing_time=45.0,
            memory_usage=0.95,  # High memory usage
            samples_processed=32,
            success_rate=1.0
        )
        
        new_batch_size = self.sizer.get_optimal_batch_size()
        self.assertLess(new_batch_size, initial_batch_size)
        
        # Simulate low memory usage (should increase batch size)
        for _ in range(3):  # Multiple updates to overcome adjustment dampening
            self.sizer.update_performance(
                processing_time=30.0,  # Fast processing
                memory_usage=0.5,     # Low memory usage
                samples_processed=new_batch_size,
                success_rate=1.0
            )
        
        final_batch_size = self.sizer.get_optimal_batch_size()
        self.assertGreaterEqual(final_batch_size, new_batch_size)
    
    def test_performance_tracking(self):
        """Test 3.2: Performance tracking and trend analysis."""
        # Add performance data with improving trend
        performance_values = [10.0, 12.0, 14.0, 16.0, 18.0]
        
        for perf in performance_values:
            self.sizer.update_performance(
                processing_time=60.0 / perf,  # Inverse relationship
                memory_usage=0.7,
                samples_processed=int(perf),
                success_rate=1.0
            )
        
        # Check performance summary
        summary = self.sizer.get_performance_summary()
        
        self.assertIn("recent_performance", summary)
        self.assertIn("average_performance", summary)
        self.assertIn("performance_trend", summary)
        
        # Trend should be positive (improving)
        self.assertGreater(summary["performance_trend"], 0)
    
    def test_batch_size_limits(self):
        """Test 3.3: Batch size limit enforcement."""
        # Try to force batch size below minimum
        for _ in range(10):
            self.sizer.update_performance(
                processing_time=120.0,  # Very slow
                memory_usage=0.99,     # Very high memory
                samples_processed=8,
                success_rate=0.5       # Low success rate
            )
        
        batch_size = self.sizer.get_optimal_batch_size()
        self.assertGreaterEqual(batch_size, self.sizer.config.min_batch_size)
        
        # Try to force batch size above maximum
        for _ in range(10):
            self.sizer.update_performance(
                processing_time=10.0,  # Very fast
                memory_usage=0.3,     # Very low memory
                samples_processed=128,
                success_rate=1.0      # Perfect success rate
            )
        
        batch_size = self.sizer.get_optimal_batch_size()
        self.assertLessEqual(batch_size, self.sizer.config.max_batch_size)


class TestSpotInstanceManager(unittest.TestCase):
    """Test suite for spot instance management."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = SpotInstanceConfig(
            max_spot_price=0.05,
            availability_zones=["us-east-1a", "us-east-1b"]
        )
        self.manager = SpotInstanceManager(config)
    
    def test_spot_availability_check(self):
        """Test 4.1: Spot instance availability checking."""
        instance_types = ["t3.medium", "m5.large"]
        availability = self.manager.check_spot_availability(instance_types)
        
        # Verify availability data structure
        self.assertEqual(len(availability), 2)
        
        for instance_type in instance_types:
            self.assertIn(instance_type, availability)
            instance_data = availability[instance_type]
            
            self.assertIn("current_price", instance_data)
            self.assertIn("availability_score", instance_data)
            self.assertIn("interruption_frequency", instance_data)
            self.assertIn("available_zones", instance_data)
            
            # Verify data types and ranges
            self.assertIsInstance(instance_data["current_price"], float)
            self.assertGreater(instance_data["current_price"], 0)
            self.assertGreaterEqual(instance_data["availability_score"], 0)
            self.assertLessEqual(instance_data["availability_score"], 1)
    
    def test_spot_instance_request(self):
        """Test 4.2: Spot instance request handling."""
        # Test successful request
        result = self.manager.request_spot_instances("t3.medium", 2, max_price=0.05)
        
        # Verify result structure
        self.assertIn("success", result)
        self.assertIn("instance_type", result)
        self.assertIn("count", result)
        
        if result["success"]:
            self.assertIn("instance_ids", result)
            self.assertIn("spot_price", result)
            self.assertEqual(len(result["instance_ids"]), 2)
            self.assertEqual(result["instance_type"], "t3.medium")
            self.assertLessEqual(result["spot_price"], 0.05)
        else:
            self.assertIn("error", result)
    
    def test_interruption_monitoring(self):
        """Test 4.3: Interruption warning monitoring."""
        # Monitor for interruptions (should be empty initially)
        warnings = self.manager.monitor_interruption_warnings()
        initial_count = len(warnings)
        
        # Run monitoring multiple times to potentially generate warnings
        for _ in range(20):  # Increase chances of generating a warning
            warnings = self.manager.monitor_interruption_warnings()
        
        # Verify warning structure if any are generated
        for warning in warnings:
            self.assertIn("instance_id", warning)
            self.assertIn("interruption_time", warning)
            self.assertIn("action", warning)
            self.assertIn("timestamp", warning)
            self.assertIsInstance(warning["interruption_time"], datetime)
    
    def test_interruption_handling(self):
        """Test 4.4: Spot instance interruption handling."""
        instance_id = "i-1234567890abcdef0"
        
        # Handle interruption
        result = self.manager.handle_interruption(instance_id)
        
        # Verify handling result
        self.assertIn("status", result)
        self.assertIn(result["status"], [
            "migration_completed", 
            "migration_failed", 
            "migration_already_in_progress"
        ])
        
        if result["status"] == "migration_completed":
            self.assertIn("original_instance", result)
            self.assertIn("new_instance", result)
            self.assertIn("migration_time", result)
            self.assertEqual(result["original_instance"], instance_id)
    
    def test_cost_savings_report(self):
        """Test 4.5: Cost savings reporting."""
        report = self.manager.get_cost_savings_report()
        
        # Verify report structure
        required_fields = [
            "total_on_demand_cost",
            "total_spot_cost", 
            "total_savings",
            "savings_percentage",
            "migration_success_rate"
        ]
        
        for field in required_fields:
            self.assertIn(field, report)
            self.assertIsInstance(report[field], (int, float))
        
        # Verify logical relationships
        self.assertGreaterEqual(report["total_on_demand_cost"], report["total_spot_cost"])
        self.assertGreaterEqual(report["savings_percentage"], 0)
        self.assertLessEqual(report["savings_percentage"], 100)
        self.assertGreaterEqual(report["migration_success_rate"], 0)
        self.assertLessEqual(report["migration_success_rate"], 1)


class TestAutomatedOptimizer(unittest.TestCase):
    """Test suite for automated optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.resource_monitor = Mock(spec=ResourceMonitor)
        self.cost_tracker = Mock(spec=CostTracker)
        self.batch_sizer = Mock(spec=DynamicBatchSizer)
        self.spot_manager = Mock(spec=SpotInstanceManager)
        
        self.optimizer = AutomatedOptimizer(
            self.resource_monitor,
            self.cost_tracker,
            self.batch_sizer,
            self.spot_manager
        )
    
    def test_optimization_rules_setup(self):
        """Test 5.1: Default optimization rules setup."""
        # Verify default rules are created
        self.assertGreater(len(self.optimizer.optimization_rules), 0)
        
        # Check rule structure
        for rule in self.optimizer.optimization_rules:
            self.assertIsInstance(rule, OptimizationRule)
            self.assertTrue(rule.rule_id)
            self.assertTrue(rule.name)
            self.assertTrue(rule.condition)
            self.assertIsInstance(rule.action, OptimizationAction)
            self.assertIsInstance(rule.priority, PriorityLevel)
    
    def test_condition_evaluation(self):
        """Test 5.2: Rule condition evaluation."""
        # Create test rule
        rule = OptimizationRule(
            rule_id="test_rule",
            name="Test Rule",
            condition="avg_cpu > 80 and avg_memory < 50",
            action=OptimizationAction.SCALE_UP
        )
        
        # Mock metrics
        current_metrics = ResourceMetrics(cpu_percent=85.0, memory_percent=40.0)
        avg_metrics = {"cpu_percent": 85.0, "memory_percent": 40.0}
        
        # Evaluate condition
        result = self.optimizer._evaluate_condition(rule, current_metrics, avg_metrics)
        self.assertTrue(result)
        
        # Test condition that should fail
        avg_metrics = {"cpu_percent": 70.0, "memory_percent": 40.0}
        result = self.optimizer._evaluate_condition(rule, current_metrics, avg_metrics)
        self.assertFalse(result)
    
    def test_optimization_action_execution(self):
        """Test 5.3: Optimization action execution."""
        # Mock dependencies
        self.batch_sizer.get_optimal_batch_size.return_value = 32
        self.batch_sizer.update_performance = Mock()
        self.spot_manager.request_spot_instances.return_value = {"success": True}
        
        # Test different actions
        current_metrics = ResourceMetrics(cpu_percent=85.0, memory_percent=90.0)
        avg_metrics = {"cpu_percent": 85.0, "memory_percent": 90.0}
        
        # Test scale up action
        scale_rule = OptimizationRule(
            rule_id="scale_up_test",
            name="Scale Up Test",
            condition="avg_cpu > 80",
            action=OptimizationAction.SCALE_UP
        )
        
        self.optimizer._execute_optimization_action(scale_rule, current_metrics, avg_metrics)
        
        # Verify action was recorded
        self.assertGreater(len(self.optimizer.optimization_history), 0)
        
        last_action = self.optimizer.optimization_history[-1]
        self.assertEqual(last_action["rule_id"], "scale_up_test")
        self.assertEqual(last_action["action"], "scale_up")
    
    def test_optimization_summary(self):
        """Test 5.4: Optimization summary generation."""
        # Add some test optimization history
        test_actions = [
            {
                "rule_id": "test_1",
                "action": "scale_up",
                "timestamp": datetime.now(),
                "success": True
            },
            {
                "rule_id": "test_2", 
                "action": "adjust_batch_size",
                "timestamp": datetime.now(),
                "success": True
            }
        ]
        
        self.optimizer.optimization_history.extend(test_actions)
        
        # Get summary
        summary = self.optimizer.get_optimization_summary()
        
        # Verify summary structure
        self.assertIn("total_optimizations_24h", summary)
        self.assertIn("action_breakdown", summary)
        self.assertIn("active_rules", summary)
        self.assertIn("optimization_success_rate", summary)
        self.assertIn("recent_actions", summary)
        
        # Verify counts
        self.assertEqual(summary["total_optimizations_24h"], 2)
        self.assertEqual(summary["optimization_success_rate"], 1.0)


class TestCostOptimizer(unittest.TestCase):
    """Test suite for main cost optimizer integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = CostOptimizer()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.optimizer.stop()
    
    def test_system_lifecycle(self):
        """Test 6.1: Cost optimizer system lifecycle."""
        # Initially not running
        self.assertFalse(self.optimizer.is_running)
        
        # Start system
        self.optimizer.start()
        self.assertTrue(self.optimizer.is_running)
        self.assertTrue(self.optimizer.resource_monitor.is_monitoring)
        
        # Stop system
        self.optimizer.stop()
        self.assertFalse(self.optimizer.is_running)
        self.assertFalse(self.optimizer.resource_monitor.is_monitoring)
    
    def test_batch_processing_with_optimization(self):
        """Test 6.2: Batch processing with cost optimization."""
        # Mock processing function
        def mock_processing_function(data, batch_size=32):
            time.sleep(0.01)  # Simulate processing time
            return {
                "success": True,
                "processed_items": batch_size,
                "data": f"processed_{data}"
            }
        
        # Process batch
        result = self.optimizer.process_batch("test_data", mock_processing_function)
        
        # Verify result structure
        self.assertIn("success", result)
        self.assertIn("processing_time", result)
        self.assertIn("batch_size", result)
        self.assertIn("samples_processed", result)
        
        if result["success"]:
            self.assertIn("result", result)
            self.assertGreater(result["processing_time"], 0)
            self.assertGreater(result["batch_size"], 0)
    
    def test_comprehensive_report_generation(self):
        """Test 6.3: Comprehensive report generation."""
        # Start system to collect some data
        self.optimizer.start()
        time.sleep(1)  # Let it collect some metrics
        
        # Generate report
        report = self.optimizer.get_comprehensive_report()
        
        # Verify report structure
        required_sections = [
            "system_status",
            "resource_metrics", 
            "cost_analysis",
            "batch_optimization",
            "spot_instance_savings",
            "optimization_summary",
            "recommendations"
        ]
        
        for section in required_sections:
            self.assertIn(section, report)
        
        # Verify system status
        self.assertIn("is_running", report["system_status"])
        self.assertIn("components_active", report["system_status"])
        
        # Verify recommendations structure
        recommendations = report["recommendations"]
        self.assertIsInstance(recommendations, list)
        
        for rec in recommendations:
            if rec:  # If recommendations exist
                self.assertIn("category", rec)
                self.assertIn("priority", rec)
                self.assertIn("recommendation", rec)
    
    def test_error_handling_in_batch_processing(self):
        """Test 6.4: Error handling during batch processing."""
        # Mock processing function that fails
        def failing_processing_function(data, batch_size=32):
            raise ValueError("Processing failed")
        
        # Process batch that will fail
        result = self.optimizer.process_batch("test_data", failing_processing_function)
        
        # Verify error handling
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("processing_time", result)
        self.assertEqual(result["error"], "Processing failed")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete cost optimization system."""
    
    def test_end_to_end_optimization_workflow(self):
        """Test 7.1: End-to-end optimization workflow."""
        # Initialize system
        optimizer = CostOptimizer()
        
        try:
            # Start optimization
            optimizer.start()
            
            # Simulate multiple batch processing sessions
            def test_processing_function(data, batch_size=32):
                # Simulate varying processing characteristics
                processing_time = 0.05 + (batch_size * 0.001)  # Scales with batch size
                time.sleep(processing_time)
                
                return {
                    "success": True,
                    "processed": batch_size,
                    "data_id": data
                }
            
            results = []
            for i in range(5):
                result = optimizer.process_batch(f"batch_{i}", test_processing_function)
                results.append(result)
                time.sleep(0.1)  # Brief pause between batches
            
            # Verify all batches processed
            self.assertEqual(len(results), 5)
            successful_batches = sum(1 for r in results if r["success"])
            self.assertEqual(successful_batches, 5)
            
            # Generate final report
            report = optimizer.get_comprehensive_report()
            
            # Verify system collected meaningful data
            self.assertIn("system_status", report)
            self.assertTrue(report["system_status"]["is_running"])
            
            # Verify cost tracking occurred
            cost_analysis = report["cost_analysis"]
            if "total_samples_processed" in cost_analysis:
                self.assertGreater(cost_analysis["total_samples_processed"], 0)
        
        finally:
            optimizer.stop()
    
    def test_optimization_rule_triggering(self):
        """Test 7.2: Optimization rule triggering under load."""
        optimizer = CostOptimizer()
        
        try:
            # Mock high resource usage to trigger optimization rules
            with patch.object(optimizer.resource_monitor, 'get_current_metrics') as mock_current:
                with patch.object(optimizer.resource_monitor, 'get_average_metrics') as mock_avg:
                    
                    # Mock high CPU and memory usage
                    mock_current.return_value = ResourceMetrics(
                        cpu_percent=90.0,
                        memory_percent=95.0
                    )
                    
                    mock_avg.return_value = {
                        "cpu_percent": 88.0,
                        "memory_percent": 92.0,
                        "disk_usage_percent": 70.0,
                        "network_io_mbps": 50.0,
                        "gpu_utilization": 0.0,
                        "active_processes": 200
                    }
                    
                    # Start system
                    optimizer.start()
                    
                    # Wait briefly for optimization evaluation
                    time.sleep(2)
                    
                    # Check if optimization actions were triggered
                    optimization_summary = optimizer.optimizer.get_optimization_summary()
                    
                    # System should have attempted some optimizations
                    self.assertIsInstance(optimization_summary, dict)
                    self.assertIn("active_rules", optimization_summary)
                    self.assertGreater(optimization_summary["active_rules"], 0)
        
        finally:
            optimizer.stop()
    
    def test_cost_savings_calculation(self):
        """Test 7.3: Cost savings calculation accuracy."""
        optimizer = CostOptimizer()
        
        # Record various cost scenarios
        cost_tracker = optimizer.cost_tracker
        
        # Scenario 1: High-cost on-demand processing
        cost_tracker.record_cost_metrics(
            instance_type=InstanceType.ON_DEMAND,
            instance_size="m5.large",
            samples_processed=1000,
            runtime_hours=2.0
        )
        
        # Scenario 2: Cost-effective spot processing  
        cost_tracker.record_cost_metrics(
            instance_type=InstanceType.SPOT,
            instance_size="m5.large", 
            samples_processed=1000,
            runtime_hours=2.0
        )
        
        # Analyze cost savings
        analysis = cost_tracker.get_cost_analysis(days=1)
        
        # Verify savings calculation
        self.assertGreater(analysis["total_spot_savings"], 0)
        self.assertGreater(analysis["spot_usage_percentage"], 0)
        self.assertLess(analysis["spot_usage_percentage"], 100)
        
        # Verify total samples processed
        self.assertEqual(analysis["total_samples_processed"], 2000)


if __name__ == "__main__":
    unittest.main()