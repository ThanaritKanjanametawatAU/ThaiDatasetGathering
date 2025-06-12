"""
Cost Optimization and Resource Management Module (S06_T20)
Comprehensive cost optimization strategies with spot instance management and analytics
"""

import os
import json
import logging
import time
import threading
import psutil
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class InstanceType(Enum):
    """Cloud instance types"""
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    PREEMPTIBLE = "preemptible"  # Google Cloud


class ResourceType(Enum):
    """Resource types to monitor"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class OptimizationAction(Enum):
    """Automated optimization actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MIGRATE_TO_SPOT = "migrate_to_spot"
    MIGRATE_TO_ON_DEMAND = "migrate_to_on_demand"
    ADJUST_BATCH_SIZE = "adjust_batch_size"
    PAUSE_PROCESSING = "pause_processing"
    RESUME_PROCESSING = "resume_processing"
    OPTIMIZE_SCHEDULE = "optimize_schedule"


class PriorityLevel(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_available_gb: float = 0.0
    network_io_mbps: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    active_processes: int = 0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CostMetrics:
    """Cost tracking metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    hourly_cost: float = 0.0
    cumulative_cost: float = 0.0
    instance_type: InstanceType = InstanceType.ON_DEMAND
    instance_size: str = ""
    region: str = ""
    spot_savings: float = 0.0
    efficiency_score: float = 0.0
    cost_per_sample: float = 0.0
    samples_processed: int = 0


@dataclass
class BatchSizeConfig:
    """Dynamic batch size configuration"""
    current_batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 256
    target_memory_utilization: float = 0.8
    target_processing_time: float = 60.0  # seconds
    adaptation_rate: float = 0.1
    performance_history: List[float] = field(default_factory=list)


@dataclass
class SpotInstanceConfig:
    """Spot instance configuration"""
    max_spot_price: float = 0.10  # Maximum price per hour
    interruption_buffer_minutes: int = 10
    backup_instance_type: str = "t3.medium"
    auto_migration_enabled: bool = True
    spot_fleet_target_capacity: int = 2
    diversification_enabled: bool = True
    availability_zones: List[str] = field(default_factory=list)


@dataclass
class OptimizationRule:
    """Automated optimization rule"""
    rule_id: str
    name: str
    condition: str  # Python expression to evaluate
    action: OptimizationAction
    threshold_values: Dict[str, float] = field(default_factory=dict)
    cooldown_minutes: int = 15
    enabled: bool = True
    priority: PriorityLevel = PriorityLevel.MEDIUM
    last_triggered: Optional[datetime] = None


class ResourceMonitor:
    """Real-time resource usage monitoring"""
    
    def __init__(self, collection_interval: int = 30):
        """Initialize resource monitor.
        
        Args:
            collection_interval: Seconds between metric collections
        """
        self.collection_interval = collection_interval
        self.metrics_history: deque = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Network IO tracking
        self._last_network_io = None
        self._last_network_time = None
        
        logger.info(f"Resource monitor initialized with {collection_interval}s interval")
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        disk_available_gb = disk.free / (1024**3)
        
        # Network metrics
        network_io_mbps = self._calculate_network_io()
        
        # GPU metrics (if available)
        gpu_utilization, gpu_memory_percent = self._get_gpu_metrics()
        
        # Process count
        active_processes = len(psutil.pids())
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            disk_available_gb=disk_available_gb,
            network_io_mbps=network_io_mbps,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
            active_processes=active_processes,
            load_average=load_avg
        )
    
    def _calculate_network_io(self) -> float:
        """Calculate network I/O in Mbps"""
        try:
            current_io = psutil.net_io_counters()
            current_time = time.time()
            
            if self._last_network_io and self._last_network_time:
                time_delta = current_time - self._last_network_time
                bytes_delta = (
                    (current_io.bytes_sent + current_io.bytes_recv) -
                    (self._last_network_io.bytes_sent + self._last_network_io.bytes_recv)
                )
                
                if time_delta > 0:
                    mbps = (bytes_delta * 8) / (time_delta * 1_000_000)
                    self._last_network_io = current_io
                    self._last_network_time = current_time
                    return mbps
            
            self._last_network_io = current_io
            self._last_network_time = current_time
            return 0.0
        
        except Exception:
            return 0.0
    
    def _get_gpu_metrics(self) -> Tuple[float, float]:
        """Get GPU utilization metrics"""
        try:
            # Try to use nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = (mem_info.used / mem_info.total) * 100
                
                return gpu_util, gpu_memory
        
        except (ImportError, Exception):
            pass
        
        return 0.0, 0.0
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return self._collect_metrics()
    
    def get_metrics_history(self, hours: int = 1) -> List[ResourceMetrics]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, hours: int = 1) -> Dict[str, float]:
        """Get average metrics over specified period"""
        history = self.get_metrics_history(hours)
        
        if not history:
            return {}
        
        return {
            "cpu_percent": statistics.mean(m.cpu_percent for m in history),
            "memory_percent": statistics.mean(m.memory_percent for m in history),
            "disk_usage_percent": statistics.mean(m.disk_usage_percent for m in history),
            "network_io_mbps": statistics.mean(m.network_io_mbps for m in history),
            "gpu_utilization": statistics.mean(m.gpu_utilization for m in history),
            "active_processes": statistics.mean(m.active_processes for m in history)
        }


class CostTracker:
    """Cost tracking and analysis"""
    
    def __init__(self, pricing_config: Dict[str, Dict[str, float]] = None):
        """Initialize cost tracker.
        
        Args:
            pricing_config: Cloud provider pricing configuration
        """
        self.pricing_config = pricing_config or self._get_default_pricing()
        self.cost_history: List[CostMetrics] = []
        self.daily_budgets: Dict[str, float] = {}
        self.alerts_enabled = True
        
        logger.info("Cost tracker initialized")
    
    def _get_default_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get default AWS pricing (approximate)"""
        return {
            "aws": {
                "t3.micro": {"on_demand": 0.0104, "spot": 0.0031},
                "t3.small": {"on_demand": 0.0208, "spot": 0.0062},
                "t3.medium": {"on_demand": 0.0416, "spot": 0.0125},
                "t3.large": {"on_demand": 0.0832, "spot": 0.0250},
                "m5.large": {"on_demand": 0.096, "spot": 0.029},
                "m5.xlarge": {"on_demand": 0.192, "spot": 0.058},
                "c5.large": {"on_demand": 0.085, "spot": 0.026},
                "c5.xlarge": {"on_demand": 0.17, "spot": 0.051}
            }
        }
    
    def record_cost_metrics(self, instance_type: InstanceType, instance_size: str,
                           samples_processed: int, runtime_hours: float,
                           region: str = "us-east-1") -> CostMetrics:
        """Record cost metrics for processing session"""
        provider = "aws"  # Default to AWS
        
        # Get pricing
        instance_pricing = self.pricing_config.get(provider, {}).get(instance_size, {})
        
        if instance_type == InstanceType.SPOT:
            hourly_cost = instance_pricing.get("spot", 0.0)
            spot_savings = instance_pricing.get("on_demand", 0.0) - hourly_cost
        else:
            hourly_cost = instance_pricing.get("on_demand", 0.0)
            spot_savings = 0.0
        
        total_cost = hourly_cost * runtime_hours
        cost_per_sample = total_cost / samples_processed if samples_processed > 0 else 0.0
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(
            samples_processed, runtime_hours, hourly_cost
        )
        
        metrics = CostMetrics(
            hourly_cost=hourly_cost,
            cumulative_cost=total_cost,
            instance_type=instance_type,
            instance_size=instance_size,
            region=region,
            spot_savings=spot_savings,
            efficiency_score=efficiency_score,
            cost_per_sample=cost_per_sample,
            samples_processed=samples_processed
        )
        
        self.cost_history.append(metrics)
        
        # Check budget alerts
        if self.alerts_enabled:
            self._check_budget_alerts(metrics)
        
        return metrics
    
    def _calculate_efficiency_score(self, samples: int, hours: float, cost: float) -> float:
        """Calculate processing efficiency score (0-1)"""
        if hours == 0 or cost == 0:
            return 0.0
        
        # Samples per hour per dollar
        efficiency = (samples / hours) / cost
        
        # Normalize to 0-1 scale (assuming 1000 samples/hour/dollar is excellent)
        return min(efficiency / 1000, 1.0)
    
    def _check_budget_alerts(self, metrics: CostMetrics):
        """Check if daily budget limits are exceeded"""
        today = datetime.now().date().isoformat()
        daily_budget = self.daily_budgets.get(today, float('inf'))
        
        # Calculate today's total cost
        today_costs = [
            m.cumulative_cost for m in self.cost_history
            if m.timestamp.date().isoformat() == today
        ]
        
        total_today = sum(today_costs)
        
        if total_today > daily_budget:
            logger.warning(
                f"Daily budget exceeded: ${total_today:.2f} > ${daily_budget:.2f}"
            )
    
    def get_cost_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive cost analysis"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.cost_history if m.timestamp >= cutoff_date
        ]
        
        if not recent_metrics:
            return {"error": "No cost data available"}
        
        total_cost = sum(m.cumulative_cost for m in recent_metrics)
        total_spot_savings = sum(m.spot_savings for m in recent_metrics)
        total_samples = sum(m.samples_processed for m in recent_metrics)
        
        avg_efficiency = statistics.mean(m.efficiency_score for m in recent_metrics)
        avg_cost_per_sample = statistics.mean(
            m.cost_per_sample for m in recent_metrics if m.cost_per_sample > 0
        )
        
        # Instance type breakdown
        instance_breakdown = defaultdict(float)
        for m in recent_metrics:
            instance_breakdown[f"{m.instance_type.value}_{m.instance_size}"] += m.cumulative_cost
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "total_spot_savings": total_spot_savings,
            "total_samples_processed": total_samples,
            "average_efficiency_score": avg_efficiency,
            "average_cost_per_sample": avg_cost_per_sample,
            "instance_breakdown": dict(instance_breakdown),
            "spot_usage_percentage": (
                sum(m.cumulative_cost for m in recent_metrics 
                    if m.instance_type == InstanceType.SPOT) / total_cost * 100
                if total_cost > 0 else 0
            )
        }
    
    def set_daily_budget(self, date: str, budget: float):
        """Set daily budget limit"""
        self.daily_budgets[date] = budget
        logger.info(f"Daily budget set for {date}: ${budget:.2f}")


class DynamicBatchSizer:
    """Dynamic batch size optimization"""
    
    def __init__(self, initial_config: BatchSizeConfig = None):
        """Initialize dynamic batch sizer"""
        self.config = initial_config or BatchSizeConfig()
        self.performance_buffer = deque(maxlen=20)  # Last 20 measurements
        self.memory_buffer = deque(maxlen=10)       # Last 10 memory measurements
        
        logger.info(f"Dynamic batch sizer initialized with batch size {self.config.current_batch_size}")
    
    def update_performance(self, processing_time: float, memory_usage: float, 
                          samples_processed: int, success_rate: float = 1.0):
        """Update performance metrics and adjust batch size"""
        # Calculate throughput (samples per second)
        throughput = samples_processed / processing_time if processing_time > 0 else 0
        
        performance_score = throughput * success_rate
        self.performance_buffer.append(performance_score)
        self.memory_buffer.append(memory_usage)
        
        # Decide on batch size adjustment
        adjustment = self._calculate_batch_adjustment(processing_time, memory_usage)
        
        if adjustment != 0:
            old_batch_size = self.config.current_batch_size
            self.config.current_batch_size = max(
                self.config.min_batch_size,
                min(self.config.max_batch_size, 
                    self.config.current_batch_size + adjustment)
            )
            
            if self.config.current_batch_size != old_batch_size:
                logger.info(
                    f"Batch size adjusted: {old_batch_size} -> {self.config.current_batch_size}"
                )
    
    def _calculate_batch_adjustment(self, processing_time: float, memory_usage: float) -> int:
        """Calculate batch size adjustment based on metrics"""
        adjustment = 0
        
        # Memory-based adjustment
        if memory_usage > self.config.target_memory_utilization:
            # Memory too high, decrease batch size
            adjustment -= max(1, int(self.config.current_batch_size * 0.1))
        elif memory_usage < self.config.target_memory_utilization * 0.7:
            # Memory utilization low, can increase batch size
            adjustment += max(1, int(self.config.current_batch_size * 0.1))
        
        # Processing time-based adjustment
        if processing_time > self.config.target_processing_time * 1.2:
            # Processing too slow, decrease batch size
            adjustment -= max(1, int(self.config.current_batch_size * 0.05))
        elif processing_time < self.config.target_processing_time * 0.8:
            # Processing fast, can increase batch size
            adjustment += max(1, int(self.config.current_batch_size * 0.05))
        
        # Performance trend-based adjustment
        if len(self.performance_buffer) >= 5:
            recent_trend = self._calculate_trend(list(self.performance_buffer)[-5:])
            if recent_trend < -0.1:  # Performance declining
                adjustment -= max(1, int(self.config.current_batch_size * 0.05))
        
        return adjustment
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in performance values"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size"""
        return self.config.current_batch_size
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_buffer:
            return {"error": "No performance data available"}
        
        return {
            "current_batch_size": self.config.current_batch_size,
            "min_batch_size": self.config.min_batch_size,
            "max_batch_size": self.config.max_batch_size,
            "recent_performance": list(self.performance_buffer),
            "average_performance": statistics.mean(self.performance_buffer),
            "performance_trend": self._calculate_trend(list(self.performance_buffer)),
            "memory_utilization": list(self.memory_buffer),
            "target_memory_utilization": self.config.target_memory_utilization
        }


class SpotInstanceManager:
    """Spot instance management with interruption handling"""
    
    def __init__(self, config: SpotInstanceConfig = None):
        """Initialize spot instance manager"""
        self.config = config or SpotInstanceConfig()
        self.current_spot_price = 0.0
        self.interruption_warnings = []
        self.migration_in_progress = False
        
        # Simulated cloud provider interface
        self.cloud_interface = None  # Would be actual cloud SDK
        
        logger.info("Spot instance manager initialized")
    
    def check_spot_availability(self, instance_types: List[str], 
                               availability_zones: List[str] = None) -> Dict[str, Dict]:
        """Check spot instance availability and pricing"""
        # Simulate spot instance availability check
        availability_zones = availability_zones or self.config.availability_zones
        
        availability = {}
        for instance_type in instance_types:
            availability[instance_type] = {
                "current_price": np.random.uniform(0.01, 0.08),  # Simulated price
                "availability_score": np.random.uniform(0.6, 1.0),  # Simulated availability
                "interruption_frequency": np.random.uniform(0.1, 0.3),  # Simulated frequency
                "available_zones": availability_zones
            }
        
        return availability
    
    def request_spot_instances(self, instance_type: str, count: int,
                             max_price: float = None) -> Dict[str, Any]:
        """Request spot instances"""
        max_price = max_price or self.config.max_spot_price
        
        # Simulate spot instance request
        success_probability = 0.8  # 80% success rate
        
        if np.random.random() < success_probability:
            # Successful request
            instance_ids = [f"i-{hashlib.md5(f'{instance_type}{i}{time.time()}'.encode()).hexdigest()[:10]}" 
                          for i in range(count)]
            
            result = {
                "success": True,
                "instance_ids": instance_ids,
                "instance_type": instance_type,
                "count": count,
                "spot_price": np.random.uniform(0.01, max_price),
                "availability_zone": np.random.choice(self.config.availability_zones) if self.config.availability_zones else "us-east-1a"
            }
        else:
            # Failed request
            result = {
                "success": False,
                "error": "Insufficient capacity",
                "instance_type": instance_type,
                "count": count
            }
        
        logger.info(f"Spot instance request: {result}")
        return result
    
    def monitor_interruption_warnings(self) -> List[Dict[str, Any]]:
        """Monitor for spot instance interruption warnings"""
        # Simulate interruption warning check
        # In real implementation, this would check cloud metadata service
        
        current_time = datetime.now()
        
        # Randomly generate interruption warning (low probability)
        if np.random.random() < 0.05:  # 5% chance per check
            warning = {
                "instance_id": f"i-{hashlib.md5(str(time.time()).encode()).hexdigest()[:10]}",
                "interruption_time": current_time + timedelta(minutes=2),
                "action": "terminate",
                "timestamp": current_time
            }
            self.interruption_warnings.append(warning)
            logger.warning(f"Spot interruption warning received: {warning}")
        
        # Clean up old warnings
        cutoff_time = current_time - timedelta(hours=1)
        self.interruption_warnings = [
            w for w in self.interruption_warnings if w["timestamp"] >= cutoff_time
        ]
        
        return self.interruption_warnings
    
    def handle_interruption(self, instance_id: str) -> Dict[str, Any]:
        """Handle spot instance interruption"""
        if self.migration_in_progress:
            return {"status": "migration_already_in_progress"}
        
        self.migration_in_progress = True
        
        try:
            # Simulate migration process
            logger.info(f"Handling interruption for instance {instance_id}")
            
            # 1. Save current state
            state_saved = self._save_instance_state(instance_id)
            
            # 2. Request backup instance
            backup_result = self.request_spot_instances(
                self.config.backup_instance_type, 1
            )
            
            if not backup_result["success"]:
                # Fall back to on-demand instance
                backup_result = self._request_on_demand_instance(
                    self.config.backup_instance_type
                )
            
            # 3. Migrate workload
            migration_result = self._migrate_workload(instance_id, backup_result)
            
            result = {
                "status": "migration_completed",
                "original_instance": instance_id,
                "new_instance": backup_result.get("instance_ids", ["unknown"])[0],
                "migration_time": datetime.now(),
                "state_preserved": state_saved,
                "migration_successful": migration_result
            }
            
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")
            result = {
                "status": "migration_failed",
                "error": str(e),
                "instance_id": instance_id
            }
        
        finally:
            self.migration_in_progress = False
        
        return result
    
    def _save_instance_state(self, instance_id: str) -> bool:
        """Save instance state for migration"""
        # Simulate state saving
        logger.info(f"Saving state for instance {instance_id}")
        time.sleep(0.1)  # Simulate save time
        return True
    
    def _request_on_demand_instance(self, instance_type: str) -> Dict[str, Any]:
        """Request on-demand instance as backup"""
        # Simulate on-demand instance request (high success rate)
        instance_id = f"i-{hashlib.md5(f'{instance_type}od{time.time()}'.encode()).hexdigest()[:10]}"
        
        return {
            "success": True,
            "instance_ids": [instance_id],
            "instance_type": instance_type,
            "pricing_type": "on_demand"
        }
    
    def _migrate_workload(self, source_instance: str, target_instance_info: Dict) -> bool:
        """Migrate workload to new instance"""
        # Simulate workload migration
        logger.info(f"Migrating workload from {source_instance} to {target_instance_info}")
        time.sleep(0.2)  # Simulate migration time
        return True
    
    def get_cost_savings_report(self) -> Dict[str, Any]:
        """Generate cost savings report for spot usage"""
        # This would calculate actual savings in a real implementation
        on_demand_cost = 100.0  # Simulated
        spot_cost = 35.0        # Simulated
        savings = on_demand_cost - spot_cost
        savings_percentage = (savings / on_demand_cost) * 100
        
        return {
            "total_on_demand_cost": on_demand_cost,
            "total_spot_cost": spot_cost,
            "total_savings": savings,
            "savings_percentage": savings_percentage,
            "interruption_count": len(self.interruption_warnings),
            "migration_success_rate": 0.95  # Simulated
        }


class AutomatedOptimizer:
    """Automated optimization engine"""
    
    def __init__(self, resource_monitor: ResourceMonitor, cost_tracker: CostTracker,
                 batch_sizer: DynamicBatchSizer, spot_manager: SpotInstanceManager):
        """Initialize automated optimizer"""
        self.resource_monitor = resource_monitor
        self.cost_tracker = cost_tracker
        self.batch_sizer = batch_sizer
        self.spot_manager = spot_manager
        
        self.optimization_rules: List[OptimizationRule] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.is_optimizing = False
        
        # Load default optimization rules
        self._setup_default_rules()
        
        logger.info("Automated optimizer initialized")
    
    def _setup_default_rules(self):
        """Setup default optimization rules"""
        self.optimization_rules = [
            OptimizationRule(
                rule_id="high_cpu_scale_up",
                name="Scale up on high CPU usage",
                condition="avg_cpu > 85",
                action=OptimizationAction.SCALE_UP,
                threshold_values={"avg_cpu": 85.0},
                cooldown_minutes=15,
                priority=PriorityLevel.HIGH
            ),
            OptimizationRule(
                rule_id="low_utilization_scale_down",
                name="Scale down on low utilization",
                condition="avg_cpu < 20 and avg_memory < 30",
                action=OptimizationAction.SCALE_DOWN,
                threshold_values={"avg_cpu": 20.0, "avg_memory": 30.0},
                cooldown_minutes=30,
                priority=PriorityLevel.MEDIUM
            ),
            OptimizationRule(
                rule_id="high_memory_reduce_batch",
                name="Reduce batch size on high memory",
                condition="avg_memory > 90",
                action=OptimizationAction.ADJUST_BATCH_SIZE,
                threshold_values={"avg_memory": 90.0},
                cooldown_minutes=5,
                priority=PriorityLevel.HIGH
            ),
            OptimizationRule(
                rule_id="cost_optimization_spot",
                name="Migrate to spot instances for cost savings",
                condition="hourly_cost > 0.1 and spot_available",
                action=OptimizationAction.MIGRATE_TO_SPOT,
                threshold_values={"hourly_cost": 0.1},
                cooldown_minutes=60,
                priority=PriorityLevel.MEDIUM
            )
        ]
    
    def start_optimization(self):
        """Start automated optimization monitoring"""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        optimization_thread.start()
        logger.info("Automated optimization started")
    
    def stop_optimization(self):
        """Stop automated optimization"""
        self.is_optimizing = False
        logger.info("Automated optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization monitoring loop"""
        while self.is_optimizing:
            try:
                self._evaluate_optimization_rules()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(60)
    
    def _evaluate_optimization_rules(self):
        """Evaluate all optimization rules"""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return
        
        avg_metrics = self.resource_monitor.get_average_metrics(hours=1)
        if not avg_metrics:
            return
        
        current_time = datetime.now()
        
        for rule in self.optimization_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if (rule.last_triggered and 
                (current_time - rule.last_triggered).total_seconds() < rule.cooldown_minutes * 60):
                continue
            
            # Evaluate condition
            if self._evaluate_condition(rule, current_metrics, avg_metrics):
                self._execute_optimization_action(rule, current_metrics, avg_metrics)
                rule.last_triggered = current_time
    
    def _evaluate_condition(self, rule: OptimizationRule, 
                           current_metrics: ResourceMetrics,
                           avg_metrics: Dict[str, float]) -> bool:
        """Evaluate rule condition"""
        try:
            # Create evaluation context
            context = {
                "avg_cpu": avg_metrics.get("cpu_percent", 0),
                "avg_memory": avg_metrics.get("memory_percent", 0),
                "avg_disk": avg_metrics.get("disk_usage_percent", 0),
                "current_cpu": current_metrics.cpu_percent,
                "current_memory": current_metrics.memory_percent,
                "hourly_cost": 0.1,  # Would get from cost tracker
                "spot_available": True  # Would check with spot manager
            }
            
            # Evaluate condition safely
            return eval(rule.condition, {"__builtins__": {}}, context)
        
        except Exception as e:
            logger.error(f"Error evaluating rule condition '{rule.condition}': {e}")
            return False
    
    def _execute_optimization_action(self, rule: OptimizationRule,
                                   current_metrics: ResourceMetrics,
                                   avg_metrics: Dict[str, float]):
        """Execute optimization action"""
        action_result = {"rule_id": rule.rule_id, "action": rule.action.value, "timestamp": datetime.now()}
        
        try:
            if rule.action == OptimizationAction.SCALE_UP:
                result = self._scale_up()
            elif rule.action == OptimizationAction.SCALE_DOWN:
                result = self._scale_down()
            elif rule.action == OptimizationAction.ADJUST_BATCH_SIZE:
                result = self._adjust_batch_size(current_metrics)
            elif rule.action == OptimizationAction.MIGRATE_TO_SPOT:
                result = self._migrate_to_spot()
            else:
                result = {"status": "action_not_implemented"}
            
            action_result.update(result)
            action_result["success"] = True
            
            logger.info(f"Executed optimization action: {rule.name}")
            
        except Exception as e:
            action_result.update({"success": False, "error": str(e)})
            logger.error(f"Failed to execute optimization action {rule.name}: {e}")
        
        self.optimization_history.append(action_result)
        
        # Keep only recent history
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
    
    def _scale_up(self) -> Dict[str, Any]:
        """Scale up resources"""
        # Simulate scaling up
        logger.info("Scaling up resources")
        return {"status": "scale_up_initiated", "new_capacity": "increased"}
    
    def _scale_down(self) -> Dict[str, Any]:
        """Scale down resources"""
        # Simulate scaling down
        logger.info("Scaling down resources")
        return {"status": "scale_down_initiated", "new_capacity": "decreased"}
    
    def _adjust_batch_size(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Adjust batch size based on current metrics"""
        old_batch_size = self.batch_sizer.get_optimal_batch_size()
        
        # Simulate performance update
        self.batch_sizer.update_performance(
            processing_time=60.0,
            memory_usage=metrics.memory_percent / 100.0,
            samples_processed=old_batch_size,
            success_rate=1.0
        )
        
        new_batch_size = self.batch_sizer.get_optimal_batch_size()
        
        return {
            "status": "batch_size_adjusted",
            "old_batch_size": old_batch_size,
            "new_batch_size": new_batch_size
        }
    
    def _migrate_to_spot(self) -> Dict[str, Any]:
        """Migrate to spot instances"""
        # Simulate spot migration
        result = self.spot_manager.request_spot_instances("t3.medium", 1)
        return {"status": "spot_migration_initiated", "spot_request": result}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        recent_actions = [
            action for action in self.optimization_history
            if (datetime.now() - action["timestamp"]).total_seconds() < 86400  # Last 24 hours
        ]
        
        action_counts = defaultdict(int)
        for action in recent_actions:
            action_counts[action["action"]] += 1
        
        return {
            "total_optimizations_24h": len(recent_actions),
            "action_breakdown": dict(action_counts),
            "active_rules": len([r for r in self.optimization_rules if r.enabled]),
            "optimization_success_rate": (
                sum(1 for a in recent_actions if a.get("success", False)) / len(recent_actions)
                if recent_actions else 0
            ),
            "recent_actions": recent_actions[-10:]  # Last 10 actions
        }


class CostOptimizer:
    """Main cost optimization and resource management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cost optimizer"""
        self.config = config or {}
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(
            collection_interval=self.config.get("monitoring_interval", 30)
        )
        self.cost_tracker = CostTracker(
            pricing_config=self.config.get("pricing_config")
        )
        self.batch_sizer = DynamicBatchSizer()
        self.spot_manager = SpotInstanceManager()
        self.optimizer = AutomatedOptimizer(
            self.resource_monitor, self.cost_tracker, 
            self.batch_sizer, self.spot_manager
        )
        
        # System state
        self.is_running = False
        
        logger.info("Cost optimizer system initialized")
    
    def start(self):
        """Start all optimization components"""
        if self.is_running:
            return
        
        self.resource_monitor.start_monitoring()
        self.optimizer.start_optimization()
        self.is_running = True
        
        logger.info("Cost optimization system started")
    
    def stop(self):
        """Stop all optimization components"""
        if not self.is_running:
            return
        
        self.resource_monitor.stop_monitoring()
        self.optimizer.stop_optimization()
        self.is_running = False
        
        logger.info("Cost optimization system stopped")
    
    def process_batch(self, batch_data: Any, processing_function: Callable) -> Dict[str, Any]:
        """Process batch with cost optimization"""
        batch_size = self.batch_sizer.get_optimal_batch_size()
        start_time = time.time()
        
        try:
            # Process the batch
            result = processing_function(batch_data, batch_size=batch_size)
            
            processing_time = time.time() - start_time
            current_metrics = self.resource_monitor.get_current_metrics()
            
            # Update batch sizer
            if current_metrics:
                self.batch_sizer.update_performance(
                    processing_time=processing_time,
                    memory_usage=current_metrics.memory_percent / 100.0,
                    samples_processed=batch_size,
                    success_rate=1.0 if result.get("success", True) else 0.0
                )
            
            # Record cost metrics
            self.cost_tracker.record_cost_metrics(
                instance_type=InstanceType.SPOT,  # Example
                instance_size="t3.medium",       # Example
                samples_processed=batch_size,
                runtime_hours=processing_time / 3600
            )
            
            return {
                "success": True,
                "processing_time": processing_time,
                "batch_size": batch_size,
                "samples_processed": batch_size,
                "result": result
            }
        
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update with failure
            if self.resource_monitor.get_current_metrics():
                self.batch_sizer.update_performance(
                    processing_time=processing_time,
                    memory_usage=0.5,  # Default
                    samples_processed=0,
                    success_rate=0.0
                )
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "batch_size": batch_size
            }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            "system_status": {
                "is_running": self.is_running,
                "components_active": {
                    "resource_monitoring": self.resource_monitor.is_monitoring,
                    "automated_optimization": self.optimizer.is_optimizing
                }
            },
            "resource_metrics": self.resource_monitor.get_average_metrics(hours=24),
            "cost_analysis": self.cost_tracker.get_cost_analysis(days=7),
            "batch_optimization": self.batch_sizer.get_performance_summary(),
            "spot_instance_savings": self.spot_manager.get_cost_savings_report(),
            "optimization_summary": self.optimizer.get_optimization_summary(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Get recent metrics
        avg_metrics = self.resource_monitor.get_average_metrics(hours=24)
        if not avg_metrics:
            return recommendations
        
        # CPU recommendations
        if avg_metrics.get("cpu_percent", 0) > 80:
            recommendations.append({
                "category": "scaling",
                "priority": "high",
                "recommendation": "Consider scaling up instances due to high CPU utilization",
                "current_value": f"{avg_metrics['cpu_percent']:.1f}%",
                "target_value": "< 70%"
            })
        
        elif avg_metrics.get("cpu_percent", 0) < 20:
            recommendations.append({
                "category": "cost_savings",
                "priority": "medium",
                "recommendation": "Consider scaling down instances due to low CPU utilization",
                "current_value": f"{avg_metrics['cpu_percent']:.1f}%",
                "target_value": "> 30%"
            })
        
        # Memory recommendations
        if avg_metrics.get("memory_percent", 0) > 85:
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "recommendation": "High memory usage detected, consider optimizing or scaling",
                "current_value": f"{avg_metrics['memory_percent']:.1f}%",
                "target_value": "< 80%"
            })
        
        # Cost optimization recommendations
        cost_analysis = self.cost_tracker.get_cost_analysis(days=7)
        if cost_analysis.get("spot_usage_percentage", 0) < 50:
            recommendations.append({
                "category": "cost_savings",
                "priority": "medium",
                "recommendation": "Increase spot instance usage to reduce costs",
                "current_value": f"{cost_analysis['spot_usage_percentage']:.1f}%",
                "target_value": "> 70%"
            })
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    optimizer = CostOptimizer()
    
    # Start optimization
    optimizer.start()
    
    # Simulate processing some batches
    def dummy_processing_function(data, batch_size=32):
        time.sleep(0.1)  # Simulate processing time
        return {"success": True, "processed": batch_size}
    
    for i in range(5):
        result = optimizer.process_batch(f"batch_{i}", dummy_processing_function)
        print(f"Batch {i}: {result}")
        time.sleep(1)
    
    # Get comprehensive report
    report = optimizer.get_comprehensive_report()
    print("\\nOptimization Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Stop optimization
    optimizer.stop()