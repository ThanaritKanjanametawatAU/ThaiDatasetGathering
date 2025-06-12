"""
Model Registry and Versioning Module (S04_T02)
MLflow-based model registry with semantic versioning, A/B testing, and deployment management
"""

import json
import logging
import pickle
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import threading
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model deployment stages"""
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    TESTING = "testing"
    DEVELOPMENT = "development"


class ModelFormat(Enum):
    """Supported model formats"""
    PICKLE = "pickle"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ExperimentStatus(Enum):
    """A/B testing experiment status"""
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    name: str
    description: str = ""
    stage: ModelStage = ModelStage.DEVELOPMENT
    format: ModelFormat = ModelFormat.PICKLE
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    file_size: int = 0
    checksum: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Model performance tracking"""
    model_id: str
    version: str
    timestamp: datetime = field(default_factory=datetime.now)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    throughput_requests_per_sec: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    environment: str = "production"
    sample_count: int = 0


@dataclass
class ABTestConfig:
    """A/B testing configuration"""
    experiment_id: str
    name: str
    description: str = ""
    model_a_id: str = ""
    model_a_version: str = ""
    model_b_id: str = ""
    model_b_version: str = ""
    traffic_split: float = 0.5  # Fraction for model A
    success_metric: str = "accuracy"
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    status: ExperimentStatus = ExperimentStatus.RUNNING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_hours: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """A/B testing results"""
    experiment_id: str
    model_a_metrics: PerformanceMetrics
    model_b_metrics: PerformanceMetrics
    statistical_significance: bool = False
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    winner: Optional[str] = None  # "A", "B", or "inconclusive"
    recommendation: str = ""
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


class ModelStorage(ABC):
    """Abstract base class for model storage backends"""
    
    @abstractmethod
    def save_model(self, model: Any, model_version: ModelVersion) -> str:
        """Save model to storage and return file path"""
        pass
    
    @abstractmethod
    def load_model(self, model_version: ModelVersion) -> Any:
        """Load model from storage"""
        pass
    
    @abstractmethod
    def delete_model(self, model_version: ModelVersion) -> bool:
        """Delete model from storage"""
        pass


class LocalFileStorage(ModelStorage):
    """Local filesystem storage backend"""
    
    def __init__(self, base_path: str = "./model_registry"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, model_version: ModelVersion) -> str:
        """Save model to local filesystem"""
        model_dir = self.base_path / model_version.model_id / model_version.version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension based on format
        extensions = {
            ModelFormat.PICKLE: ".pkl",
            ModelFormat.PYTORCH: ".pt",
            ModelFormat.ONNX: ".onnx",
            ModelFormat.TENSORFLOW: ".pb",
            ModelFormat.HUGGINGFACE: ".bin",
            ModelFormat.CUSTOM: ".model"
        }
        
        file_path = model_dir / f"model{extensions.get(model_version.format, '.pkl')}"
        
        # Save based on format
        if model_version.format == ModelFormat.PICKLE:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        elif model_version.format == ModelFormat.PYTORCH:
            import torch
            torch.save(model, file_path)
        else:
            # For other formats, assume model has a save method
            if hasattr(model, 'save'):
                model.save(str(file_path))
            else:
                # Fallback to pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
        
        # Calculate file size and checksum
        model_version.file_size = file_path.stat().st_size
        model_version.checksum = self._calculate_checksum(file_path)
        
        return str(file_path)
    
    def load_model(self, model_version: ModelVersion) -> Any:
        """Load model from local filesystem"""
        file_path = Path(model_version.file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Verify checksum if available
        if model_version.checksum:
            current_checksum = self._calculate_checksum(file_path)
            if current_checksum != model_version.checksum:
                raise ValueError(f"Model file corrupted: checksum mismatch")
        
        # Load based on format
        if model_version.format == ModelFormat.PICKLE:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif model_version.format == ModelFormat.PYTORCH:
            import torch
            return torch.load(file_path)
        else:
            # Try pickle fallback
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def delete_model(self, model_version: ModelVersion) -> bool:
        """Delete model from local filesystem"""
        if model_version.file_path:
            file_path = Path(model_version.file_path)
            if file_path.exists():
                file_path.unlink()
                return True
        return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


class ModelRegistry:
    """MLflow-style model registry with versioning and deployment management"""
    
    def __init__(self, storage: Optional[ModelStorage] = None, registry_path: str = "./model_registry"):
        """Initialize model registry.
        
        Args:
            storage: Model storage backend
            registry_path: Path for registry metadata
        """
        self.storage = storage or LocalFileStorage(registry_path)
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry metadata
        self.models: Dict[str, Dict[str, ModelVersion]] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.ab_experiments: Dict[str, ABTestConfig] = {}
        self.ab_results: Dict[str, ABTestResult] = {}
        
        # Load existing registry
        self._load_registry()
        
        # Background tasks
        self._running = True
        self._monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"Model registry initialized at {registry_path}")
    
    def register_model(self, 
                      model: Any, 
                      model_id: str,
                      version: str,
                      name: str,
                      description: str = "",
                      stage: ModelStage = ModelStage.DEVELOPMENT,
                      format: ModelFormat = ModelFormat.PICKLE,
                      tags: Optional[Dict[str, str]] = None,
                      metrics: Optional[Dict[str, float]] = None,
                      parameters: Optional[Dict[str, Any]] = None) -> ModelVersion:
        """Register a new model version.
        
        Args:
            model: Model object to register
            model_id: Unique model identifier
            version: Version string (semantic versioning recommended)
            name: Human-readable model name
            description: Model description
            stage: Deployment stage
            format: Model format
            tags: Metadata tags
            metrics: Initial performance metrics
            parameters: Model parameters/hyperparameters
            
        Returns:
            Registered model version
        """
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            stage=stage,
            format=format,
            tags=tags or {},
            metrics=metrics or {},
            parameters=parameters or {}
        )
        
        # Check if version already exists
        if model_id in self.models and version in self.models[model_id]:
            raise ValueError(f"Model {model_id} version {version} already exists")
        
        # Save model
        file_path = self.storage.save_model(model, model_version)
        model_version.file_path = file_path
        
        # Store in registry
        if model_id not in self.models:
            self.models[model_id] = {}
        self.models[model_id][version] = model_version
        
        # Initialize performance history
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model {model_id} version {version}")
        return model_version
    
    def get_model(self, model_id: str, version: Optional[str] = None, stage: Optional[ModelStage] = None) -> Any:
        """Load model from registry.
        
        Args:
            model_id: Model identifier
            version: Specific version (optional)
            stage: Get latest model in stage (optional)
            
        Returns:
            Loaded model object
        """
        model_version = self.get_model_version(model_id, version, stage)
        return self.storage.load_model(model_version)
    
    def get_model_version(self, model_id: str, version: Optional[str] = None, stage: Optional[ModelStage] = None) -> ModelVersion:
        """Get model version metadata.
        
        Args:
            model_id: Model identifier
            version: Specific version (optional)
            stage: Get latest model in stage (optional)
            
        Returns:
            Model version metadata
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_versions = self.models[model_id]
        
        if version:
            if version not in model_versions:
                raise ValueError(f"Model {model_id} version {version} not found")
            return model_versions[version]
        
        if stage:
            # Find latest version in stage
            stage_versions = [v for v in model_versions.values() if v.stage == stage]
            if not stage_versions:
                raise ValueError(f"No models found in stage {stage.value}")
            return max(stage_versions, key=lambda v: v.created_at)
        
        # Return latest version
        return max(model_versions.values(), key=lambda v: v.created_at)
    
    def list_models(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List all models in registry.
        
        Args:
            stage: Filter by stage (optional)
            
        Returns:
            List of model versions
        """
        all_versions = []
        for model_versions in self.models.values():
            all_versions.extend(model_versions.values())
        
        if stage:
            all_versions = [v for v in all_versions if v.stage == stage]
        
        return sorted(all_versions, key=lambda v: v.created_at, reverse=True)
    
    def update_model_stage(self, model_id: str, version: str, stage: ModelStage) -> ModelVersion:
        """Update model deployment stage.
        
        Args:
            model_id: Model identifier
            version: Model version
            stage: New stage
            
        Returns:
            Updated model version
        """
        model_version = self.get_model_version(model_id, version)
        old_stage = model_version.stage
        model_version.stage = stage
        
        self._save_registry()
        
        logger.info(f"Updated model {model_id} v{version} stage: {old_stage.value} -> {stage.value}")
        return model_version
    
    def delete_model_version(self, model_id: str, version: str) -> bool:
        """Delete a model version.
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            True if deleted successfully
        """
        if model_id not in self.models or version not in self.models[model_id]:
            return False
        
        model_version = self.models[model_id][version]
        
        # Delete from storage
        self.storage.delete_model(model_version)
        
        # Remove from registry
        del self.models[model_id][version]
        
        # Remove model if no versions left
        if not self.models[model_id]:
            del self.models[model_id]
            if model_id in self.performance_history:
                del self.performance_history[model_id]
        
        self._save_registry()
        
        logger.info(f"Deleted model {model_id} version {version}")
        return True
    
    def record_performance(self, performance: PerformanceMetrics):
        """Record model performance metrics.
        
        Args:
            performance: Performance metrics
        """
        model_id = performance.model_id
        
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        self.performance_history[model_id].append(performance)
        
        # Keep only recent metrics (last 1000 entries per model)
        if len(self.performance_history[model_id]) > 1000:
            self.performance_history[model_id] = self.performance_history[model_id][-1000:]
        
        self._save_registry()
        
        logger.debug(f"Recorded performance for model {model_id} v{performance.version}")
    
    def get_performance_history(self, model_id: str, 
                               version: Optional[str] = None,
                               environment: Optional[str] = None,
                               limit: int = 100) -> List[PerformanceMetrics]:
        """Get performance history for a model.
        
        Args:
            model_id: Model identifier
            version: Filter by version (optional)
            environment: Filter by environment (optional)
            limit: Maximum number of records
            
        Returns:
            List of performance metrics
        """
        if model_id not in self.performance_history:
            return []
        
        metrics = self.performance_history[model_id]
        
        # Apply filters
        if version:
            metrics = [m for m in metrics if m.version == version]
        
        if environment:
            metrics = [m for m in metrics if m.environment == environment]
        
        # Sort by timestamp (newest first) and limit
        metrics = sorted(metrics, key=lambda m: m.timestamp, reverse=True)
        return metrics[:limit]
    
    def create_ab_test(self, config: ABTestConfig) -> ABTestConfig:
        """Create A/B testing experiment.
        
        Args:
            config: A/B test configuration
            
        Returns:
            Created experiment configuration
        """
        if config.experiment_id in self.ab_experiments:
            raise ValueError(f"Experiment {config.experiment_id} already exists")
        
        # Validate models exist
        self.get_model_version(config.model_a_id, config.model_a_version)
        self.get_model_version(config.model_b_id, config.model_b_version)
        
        # Set end time if duration specified
        if config.duration_hours:
            config.end_time = config.start_time + timedelta(hours=config.duration_hours)
        
        self.ab_experiments[config.experiment_id] = config
        self._save_registry()
        
        logger.info(f"Created A/B test experiment: {config.experiment_id}")
        return config
    
    def get_ab_test(self, experiment_id: str) -> ABTestConfig:
        """Get A/B test configuration.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment configuration
        """
        if experiment_id not in self.ab_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return self.ab_experiments[experiment_id]
    
    def update_ab_test_status(self, experiment_id: str, status: ExperimentStatus):
        """Update A/B test status.
        
        Args:
            experiment_id: Experiment identifier
            status: New status
        """
        if experiment_id not in self.ab_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        self.ab_experiments[experiment_id].status = status
        
        if status == ExperimentStatus.COMPLETED and not self.ab_experiments[experiment_id].end_time:
            self.ab_experiments[experiment_id].end_time = datetime.now()
        
        self._save_registry()
        
        logger.info(f"Updated experiment {experiment_id} status to {status.value}")
    
    def analyze_ab_test(self, experiment_id: str) -> ABTestResult:
        """Analyze A/B test results.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            A/B test analysis results
        """
        experiment = self.get_ab_test(experiment_id)
        
        # Get performance metrics for both models
        model_a_metrics = self.get_performance_history(
            experiment.model_a_id, 
            experiment.model_a_version,
            limit=1000
        )
        model_b_metrics = self.get_performance_history(
            experiment.model_b_id, 
            experiment.model_b_version,
            limit=1000
        )
        
        # Filter metrics within experiment timeframe
        start_time = experiment.start_time
        end_time = experiment.end_time or datetime.now()
        
        model_a_metrics = [m for m in model_a_metrics 
                          if start_time <= m.timestamp <= end_time]
        model_b_metrics = [m for m in model_b_metrics 
                          if start_time <= m.timestamp <= end_time]
        
        if not model_a_metrics or not model_b_metrics:
            logger.warning(f"Insufficient data for experiment {experiment_id}")
            return ABTestResult(
                experiment_id=experiment_id,
                model_a_metrics=PerformanceMetrics(
                    model_id=experiment.model_a_id,
                    version=experiment.model_a_version
                ),
                model_b_metrics=PerformanceMetrics(
                    model_id=experiment.model_b_id,
                    version=experiment.model_b_version
                ),
                recommendation="Insufficient data for analysis"
            )
        
        # Calculate aggregate metrics
        model_a_agg = self._aggregate_metrics(model_a_metrics, experiment.success_metric)
        model_b_agg = self._aggregate_metrics(model_b_metrics, experiment.success_metric)
        
        # Perform statistical analysis
        statistical_significance, p_value, confidence_interval = self._statistical_test(
            model_a_metrics, model_b_metrics, experiment.success_metric, experiment.confidence_level
        )
        
        # Determine winner
        winner = None
        recommendation = "No significant difference found"
        
        if statistical_significance:
            metric_a = getattr(model_a_agg, experiment.success_metric, 0) or 0
            metric_b = getattr(model_b_agg, experiment.success_metric, 0) or 0
            
            if metric_a > metric_b:
                winner = "A"
                recommendation = f"Model A significantly outperforms Model B (p={p_value:.4f})"
            else:
                winner = "B"
                recommendation = f"Model B significantly outperforms Model A (p={p_value:.4f})"
        
        result = ABTestResult(
            experiment_id=experiment_id,
            model_a_metrics=model_a_agg,
            model_b_metrics=model_b_agg,
            statistical_significance=statistical_significance,
            p_value=p_value,
            confidence_interval=confidence_interval,
            winner=winner,
            recommendation=recommendation,
            sample_sizes={
                "model_a": len(model_a_metrics),
                "model_b": len(model_b_metrics)
            }
        )
        
        # Store results
        self.ab_results[experiment_id] = result
        self._save_registry()
        
        logger.info(f"Analyzed A/B test {experiment_id}: {recommendation}")
        return result
    
    def _aggregate_metrics(self, metrics_list: List[PerformanceMetrics], success_metric: str) -> PerformanceMetrics:
        """Aggregate performance metrics"""
        if not metrics_list:
            return PerformanceMetrics(model_id="", version="")
        
        # Calculate averages
        total_samples = sum(m.sample_count for m in metrics_list)
        
        agg_metrics = PerformanceMetrics(
            model_id=metrics_list[0].model_id,
            version=metrics_list[0].version,
            sample_count=total_samples
        )
        
        # Weighted averages based on sample count
        def weighted_avg(attr_name):
            values = [(getattr(m, attr_name) or 0, m.sample_count) for m in metrics_list]
            total_weighted = sum(v * w for v, w in values if v is not None)
            total_weights = sum(w for v, w in values if v is not None)
            return total_weighted / total_weights if total_weights > 0 else None
        
        agg_metrics.accuracy = weighted_avg('accuracy')
        agg_metrics.precision = weighted_avg('precision')
        agg_metrics.recall = weighted_avg('recall')
        agg_metrics.f1_score = weighted_avg('f1_score')
        agg_metrics.inference_time_ms = weighted_avg('inference_time_ms')
        agg_metrics.memory_usage_mb = weighted_avg('memory_usage_mb')
        agg_metrics.throughput_requests_per_sec = weighted_avg('throughput_requests_per_sec')
        
        return agg_metrics
    
    def _statistical_test(self, metrics_a: List[PerformanceMetrics], 
                         metrics_b: List[PerformanceMetrics],
                         success_metric: str, 
                         confidence_level: float) -> Tuple[bool, Optional[float], Optional[Tuple[float, float]]]:
        """Perform statistical significance test"""
        try:
            # Extract values for the success metric
            values_a = [getattr(m, success_metric) or 0 for m in metrics_a if getattr(m, success_metric) is not None]
            values_b = [getattr(m, success_metric) or 0 for m in metrics_b if getattr(m, success_metric) is not None]
            
            if len(values_a) < 10 or len(values_b) < 10:
                return False, None, None
            
            # Simple t-test approximation
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            std_a = np.std(values_a, ddof=1) if len(values_a) > 1 else 0
            std_b = np.std(values_b, ddof=1) if len(values_b) > 1 else 0
            
            # Standard error of difference
            se_diff = np.sqrt(std_a**2/len(values_a) + std_b**2/len(values_b))
            
            if se_diff == 0:
                return False, None, None
            
            # T-statistic
            t_stat = abs(mean_a - mean_b) / se_diff
            
            # Degrees of freedom (Welch's approximation)
            if std_a == 0 and std_b == 0:
                return False, None, None
            
            df = ((std_a**2/len(values_a) + std_b**2/len(values_b))**2) / \
                 ((std_a**2/len(values_a))**2/(len(values_a)-1) + (std_b**2/len(values_b))**2/(len(values_b)-1))
            
            # Critical value approximation for two-tailed test
            alpha = 1 - confidence_level
            critical_value = 2.0  # Rough approximation for common cases
            
            p_value = 2 * (1 - 0.5 * (1 + t_stat / np.sqrt(df + t_stat**2)))  # Rough p-value approximation
            
            is_significant = t_stat > critical_value
            
            # Confidence interval for difference
            margin_error = critical_value * se_diff
            diff = mean_a - mean_b
            confidence_interval = (diff - margin_error, diff + margin_error)
            
            return is_significant, p_value, confidence_interval
            
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return False, None, None
    
    def _save_registry(self):
        """Save registry metadata to file"""
        def serialize_enums(obj):
            """Custom serializer for enum values"""
            if hasattr(obj, 'value'):  # Handle enums
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return str(obj)
        
        registry_data = {
            'models': {
                model_id: {
                    version: asdict(model_version) 
                    for version, model_version in versions.items()
                }
                for model_id, versions in self.models.items()
            },
            'performance_history': {
                model_id: [asdict(metrics) for metrics in metrics_list]
                for model_id, metrics_list in self.performance_history.items()
            },
            'ab_experiments': {
                exp_id: asdict(config) 
                for exp_id, config in self.ab_experiments.items()
            },
            'ab_results': {
                exp_id: asdict(result) 
                for exp_id, result in self.ab_results.items()
            }
        }
        
        # Convert datetime objects and enums to strings
        registry_json = json.dumps(registry_data, default=serialize_enums, indent=2)
        
        registry_file = self.registry_path / "registry.json"
        with open(registry_file, 'w') as f:
            f.write(registry_json)
    
    def _load_registry(self):
        """Load registry metadata from file"""
        registry_file = self.registry_path / "registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Load models
            for model_id, versions in registry_data.get('models', {}).items():
                self.models[model_id] = {}
                for version, version_data in versions.items():
                    # Convert datetime strings back to datetime objects
                    if 'created_at' in version_data:
                        version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                    
                    # Handle enum values properly
                    if 'stage' in version_data:
                        if isinstance(version_data['stage'], str):
                            version_data['stage'] = ModelStage(version_data['stage'])
                        elif isinstance(version_data['stage'], dict) and 'value' in version_data['stage']:
                            version_data['stage'] = ModelStage(version_data['stage']['value'])
                    
                    if 'format' in version_data:
                        if isinstance(version_data['format'], str):
                            version_data['format'] = ModelFormat(version_data['format'])
                        elif isinstance(version_data['format'], dict) and 'value' in version_data['format']:
                            version_data['format'] = ModelFormat(version_data['format']['value'])
                    
                    self.models[model_id][version] = ModelVersion(**version_data)
            
            # Load performance history
            for model_id, metrics_list in registry_data.get('performance_history', {}).items():
                self.performance_history[model_id] = []
                for metrics_data in metrics_list:
                    if 'timestamp' in metrics_data:
                        metrics_data['timestamp'] = datetime.fromisoformat(metrics_data['timestamp'])
                    self.performance_history[model_id].append(PerformanceMetrics(**metrics_data))
            
            # Load A/B experiments
            for exp_id, exp_data in registry_data.get('ab_experiments', {}).items():
                if 'start_time' in exp_data:
                    exp_data['start_time'] = datetime.fromisoformat(exp_data['start_time'])
                if 'end_time' in exp_data and exp_data['end_time']:
                    exp_data['end_time'] = datetime.fromisoformat(exp_data['end_time'])
                exp_data['status'] = ExperimentStatus(exp_data['status'])
                
                self.ab_experiments[exp_id] = ABTestConfig(**exp_data)
            
            # Load A/B results
            for exp_id, result_data in registry_data.get('ab_results', {}).items():
                if 'generated_at' in result_data:
                    result_data['generated_at'] = datetime.fromisoformat(result_data['generated_at'])
                
                # Convert nested PerformanceMetrics
                for metrics_key in ['model_a_metrics', 'model_b_metrics']:
                    if metrics_key in result_data:
                        metrics_data = result_data[metrics_key]
                        if 'timestamp' in metrics_data:
                            metrics_data['timestamp'] = datetime.fromisoformat(metrics_data['timestamp'])
                        result_data[metrics_key] = PerformanceMetrics(**metrics_data)
                
                self.ab_results[exp_id] = ABTestResult(**result_data)
            
            logger.info(f"Loaded registry with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def _background_monitor(self):
        """Background thread for monitoring experiments"""
        while self._running:
            try:
                # Check for completed experiments
                for exp_id, experiment in self.ab_experiments.items():
                    if (experiment.status == ExperimentStatus.RUNNING and 
                        experiment.end_time and 
                        datetime.now() >= experiment.end_time):
                        
                        logger.info(f"Experiment {exp_id} time limit reached, completing...")
                        self.update_ab_test_status(exp_id, ExperimentStatus.COMPLETED)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Background monitor error: {e}")
                time.sleep(60)
    
    def shutdown(self):
        """Shutdown registry and cleanup resources"""
        self._running = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        self._save_registry()
        logger.info("Model registry shutdown complete")


# Example usage functions for testing
class DummyModel:
    """Dummy model class for testing"""
    def __init__(self, accuracy=0.85):
        self.accuracy = accuracy
        self.parameters = {"learning_rate": 0.001, "epochs": 100}
    
    def predict(self, x):
        return "prediction"


def example_dummy_model():
    """Create a dummy model for testing"""
    return DummyModel()


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # Register a model
    dummy_model = example_dummy_model()
    model_version = registry.register_model(
        model=dummy_model,
        model_id="audio_classifier",
        version="1.0.0",
        name="Audio Classification Model",
        description="Initial audio classification model",
        stage=ModelStage.STAGING,
        metrics={"accuracy": 0.85, "f1_score": 0.82}
    )
    
    print(f"Registered model: {model_version.model_id} v{model_version.version}")
    
    # Record performance
    performance = PerformanceMetrics(
        model_id="audio_classifier",
        version="1.0.0",
        accuracy=0.87,
        precision=0.85,
        recall=0.89,
        f1_score=0.87,
        inference_time_ms=45.2,
        sample_count=1000
    )
    registry.record_performance(performance)
    
    # List models
    models = registry.list_models()
    print(f"Total models: {len(models)}")
    
    # Create A/B test
    dummy_model_b = example_dummy_model()
    model_version_b = registry.register_model(
        model=dummy_model_b,
        model_id="audio_classifier",
        version="1.1.0",
        name="Audio Classification Model v1.1",
        description="Improved audio classification model",
        stage=ModelStage.TESTING,
        metrics={"accuracy": 0.88, "f1_score": 0.85}
    )
    
    ab_config = ABTestConfig(
        experiment_id="audio_classifier_v1_vs_v1.1",
        name="Audio Classifier A/B Test",
        description="Compare v1.0.0 vs v1.1.0",
        model_a_id="audio_classifier",
        model_a_version="1.0.0",
        model_b_id="audio_classifier",
        model_b_version="1.1.0",
        duration_hours=24
    )
    
    registry.create_ab_test(ab_config)
    print(f"Created A/B test: {ab_config.experiment_id}")
    
    registry.shutdown()