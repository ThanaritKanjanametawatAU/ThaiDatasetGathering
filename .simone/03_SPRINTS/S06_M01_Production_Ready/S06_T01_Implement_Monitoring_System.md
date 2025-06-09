# Task S06_T01: Implement Monitoring System

## Task Overview
Implement a comprehensive monitoring system for production audio enhancement services, tracking performance, quality metrics, system health, and user experience.

## Technical Requirements

### Core Implementation
- **Monitoring System** (`monitoring/production_monitor.py`)
  - Real-time metrics collection
  - Alert management
  - Dashboard visualization
  - Historical analysis

### Key Features
1. **Metrics Collection**
   - Processing latency
   - Quality scores
   - System resources
   - Error rates
   - User satisfaction

2. **Alert System**
   - Threshold-based alerts
   - Anomaly detection
   - Escalation policies
   - Alert routing

3. **Visualization**
   - Real-time dashboards
   - Historical trends
   - Drill-down capabilities
   - Custom views

## TDD Requirements

### Test Structure
```
tests/test_monitoring_system.py
- test_metric_collection()
- test_alert_generation()
- test_dashboard_updates()
- test_data_aggregation()
- test_alert_escalation()
- test_system_integration()
```

### Test Data Requirements
- Simulated metrics
- Alert scenarios
- Historical data
- System states

## Implementation Approach

### Phase 1: Core Monitoring
```python
class ProductionMonitor:
    def __init__(self, config):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.dashboard = DashboardService()
        
    def collect_metrics(self, service_name, metrics):
        # Collect and store metrics
        pass
    
    def check_health(self):
        # System health check
        pass
    
    def trigger_alert(self, condition, severity):
        # Alert management
        pass
```

### Phase 2: Advanced Monitoring
- Distributed tracing
- Log aggregation
- Predictive alerts
- Capacity planning

#### Distributed Tracing Implementation
```python
class DistributedTracingSystem:
    """Advanced distributed tracing for audio enhancement pipeline"""
    def __init__(self, service_name):
        self.service_name = service_name
        self.tracer = self._init_tracer()
        self.span_processors = []
        self.baggage_items = {}
        
    def _init_tracer(self):
        """Initialize OpenTelemetry tracer with custom configuration"""
        from opentelemetry import trace
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        # Configure tracer provider
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(self.service_name, "1.0.0")
        
        # Add Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        return tracer
    
    def create_distributed_trace(self, operation_name, attributes=None):
        """Create a distributed trace with correlation"""
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add custom attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            # Add audio-specific attributes
            span.set_attribute("audio.duration", attributes.get("duration", 0))
            span.set_attribute("audio.sample_rate", attributes.get("sample_rate", 0))
            span.set_attribute("audio.format", attributes.get("format", "unknown"))
            
            # Propagate context
            ctx = span.get_span_context()
            
            return {
                'trace_id': format(ctx.trace_id, '032x'),
                'span_id': format(ctx.span_id, '016x'),
                'trace_flags': ctx.trace_flags
            }
    
    def trace_audio_pipeline(self, audio_id):
        """Comprehensive tracing for audio processing pipeline"""
        with self.tracer.start_as_current_span("audio_enhancement_pipeline") as pipeline_span:
            pipeline_span.set_attribute("audio.id", audio_id)
            
            # Trace each stage
            stages = [
                ("load_audio", self._trace_load_audio),
                ("quality_analysis", self._trace_quality_analysis),
                ("enhancement", self._trace_enhancement),
                ("validation", self._trace_validation),
                ("output", self._trace_output)
            ]
            
            results = {}
            for stage_name, trace_func in stages:
                with self.tracer.start_as_current_span(stage_name) as stage_span:
                    try:
                        result = trace_func(audio_id, stage_span)
                        results[stage_name] = result
                        stage_span.set_status(trace.Status(trace.StatusCode.OK))
                    except Exception as e:
                        stage_span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        stage_span.record_exception(e)
                        raise
            
            return results
    
    def implement_trace_sampling(self):
        """Intelligent trace sampling to reduce overhead"""
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
        
        class AdaptiveSampler:
            """Adaptive sampling based on system load and error rates"""
            def __init__(self, base_rate=0.1):
                self.base_rate = base_rate
                self.error_sampler = TraceIdRatioBased(1.0)  # Always sample errors
                self.high_latency_sampler = TraceIdRatioBased(0.5)
                
            def should_sample(self, context, trace_id, name, kind, attributes, links):
                # Always sample errors
                if attributes and attributes.get("error", False):
                    return self.error_sampler.should_sample(
                        context, trace_id, name, kind, attributes, links
                    )
                
                # Sample high latency operations
                if attributes and attributes.get("latency_ms", 0) > 1000:
                    return self.high_latency_sampler.should_sample(
                        context, trace_id, name, kind, attributes, links
                    )
                
                # Adaptive sampling based on load
                current_load = self._get_system_load()
                if current_load > 0.8:
                    # Reduce sampling under high load
                    adjusted_rate = self.base_rate * 0.5
                else:
                    adjusted_rate = self.base_rate
                
                return TraceIdRatioBased(adjusted_rate).should_sample(
                    context, trace_id, name, kind, attributes, links
                )
        
        return AdaptiveSampler()
```

#### Metric Aggregation Strategies
```python
class MetricAggregationEngine:
    """Advanced metric aggregation with multiple strategies"""
    def __init__(self):
        self.aggregators = {
            'time_series': TimeSeriesAggregator(),
            'histogram': HistogramAggregator(),
            'percentile': PercentileAggregator(),
            'rate': RateAggregator()
        }
        self.metric_store = MetricStore()
        
    def aggregate_metrics(self, raw_metrics, aggregation_config):
        """Apply multiple aggregation strategies"""
        aggregated = {}
        
        for metric_name, metric_data in raw_metrics.items():
            config = aggregation_config.get(metric_name, {})
            
            # Determine aggregation strategy
            strategy = config.get('strategy', 'time_series')
            window = config.get('window', 60)  # seconds
            
            # Apply aggregation
            aggregator = self.aggregators[strategy]
            result = aggregator.aggregate(
                metric_data,
                window=window,
                **config.get('params', {})
            )
            
            aggregated[metric_name] = result
        
        return aggregated
    
    def compute_derived_metrics(self, base_metrics):
        """Compute complex derived metrics"""
        derived = {}
        
        # Audio quality index (composite metric)
        if all(m in base_metrics for m in ['pesq', 'stoi', 'snr']):
            derived['quality_index'] = (
                0.4 * base_metrics['pesq'] / 4.5 +  # Normalize PESQ
                0.3 * base_metrics['stoi'] +         # STOI already 0-1
                0.3 * np.clip(base_metrics['snr'] / 30, 0, 1)  # Normalize SNR
            )
        
        # Processing efficiency
        if 'processing_time' in base_metrics and 'audio_duration' in base_metrics:
            derived['real_time_factor'] = (
                base_metrics['audio_duration'] / base_metrics['processing_time']
            )
        
        # Error rate trends
        if 'errors' in base_metrics:
            error_series = base_metrics['errors']
            derived['error_trend'] = self._compute_trend(error_series)
            derived['error_anomalies'] = self._detect_anomalies(error_series)
        
        return derived
    
    def implement_streaming_aggregation(self):
        """Real-time streaming aggregation using Apache Flink concepts"""
        class StreamingAggregator:
            def __init__(self, window_size=60):
                self.window_size = window_size
                self.windows = defaultdict(deque)
                self.watermark = time.time()
                
            def process_event(self, event):
                """Process streaming metric event"""
                metric_name = event['metric']
                timestamp = event['timestamp']
                value = event['value']
                
                # Add to appropriate window
                window_key = int(timestamp // self.window_size)
                self.windows[metric_name].append({
                    'timestamp': timestamp,
                    'value': value,
                    'window': window_key
                })
                
                # Trigger window computation if needed
                if timestamp > self.watermark + self.window_size:
                    return self._compute_windows()
                
                return None
            
            def _compute_windows(self):
                """Compute aggregations for completed windows"""
                results = {}
                current_time = time.time()
                
                for metric_name, events in self.windows.items():
                    # Group by window
                    windowed_data = defaultdict(list)
                    
                    for event in events:
                        if event['timestamp'] < current_time - self.window_size:
                            windowed_data[event['window']].append(event['value'])
                    
                    # Compute aggregations
                    for window_key, values in windowed_data.items():
                        results[f"{metric_name}_window_{window_key}"] = {
                            'count': len(values),
                            'sum': sum(values),
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': min(values),
                            'max': max(values),
                            'p50': np.percentile(values, 50),
                            'p95': np.percentile(values, 95),
                            'p99': np.percentile(values, 99)
                        }
                    
                    # Clean old events
                    self.windows[metric_name] = deque(
                        e for e in events 
                        if e['timestamp'] >= current_time - 2 * self.window_size
                    )
                
                self.watermark = current_time
                return results
        
        return StreamingAggregator()
```

#### Alert Rule Generation
```python
class IntelligentAlertGenerator:
    """Generate and optimize alert rules using ML"""
    def __init__(self):
        self.alert_history = AlertHistory()
        self.ml_model = self._train_alert_model()
        
    def generate_alert_rules(self, metric_patterns):
        """Generate optimal alert rules from historical patterns"""
        rules = []
        
        for metric_name, pattern_data in metric_patterns.items():
            # Analyze metric behavior
            analysis = self._analyze_metric_behavior(pattern_data)
            
            # Generate rule candidates
            candidates = self._generate_rule_candidates(metric_name, analysis)
            
            # Evaluate and select best rules
            best_rules = self._select_optimal_rules(candidates, pattern_data)
            
            rules.extend(best_rules)
        
        return rules
    
    def _generate_rule_candidates(self, metric_name, analysis):
        """Generate candidate alert rules"""
        candidates = []
        
        # Static threshold rules
        if analysis['distribution'] == 'normal':
            mean, std = analysis['mean'], analysis['std']
            candidates.extend([
                {
                    'type': 'threshold',
                    'metric': metric_name,
                    'condition': f'value > {mean + 3*std}',
                    'severity': 'warning',
                    'description': '3-sigma deviation'
                },
                {
                    'type': 'threshold',
                    'metric': metric_name,
                    'condition': f'value > {mean + 4*std}',
                    'severity': 'critical',
                    'description': '4-sigma deviation'
                }
            ])
        
        # Anomaly detection rules
        candidates.append({
            'type': 'anomaly',
            'metric': metric_name,
            'algorithm': 'isolation_forest',
            'sensitivity': 0.05,
            'severity': 'warning'
        })
        
        # Trend-based rules
        if analysis['has_trend']:
            candidates.append({
                'type': 'trend',
                'metric': metric_name,
                'condition': 'rate_of_change > 0.1',
                'window': '5m',
                'severity': 'warning'
            })
        
        # Composite rules
        if 'correlated_metrics' in analysis:
            for correlated in analysis['correlated_metrics']:
                candidates.append({
                    'type': 'composite',
                    'metrics': [metric_name, correlated],
                    'condition': f'{metric_name} > threshold AND {correlated} < threshold',
                    'severity': 'critical'
                })
        
        return candidates
    
    def optimize_alert_thresholds(self, rule, historical_data):
        """Optimize thresholds to minimize false positives"""
        from scipy.optimize import minimize_scalar
        
        def objective(threshold):
            # Simulate alerts with this threshold
            alerts = self._simulate_alerts(rule, historical_data, threshold)
            
            # Calculate false positive rate
            fp_rate = self._calculate_false_positive_rate(alerts)
            
            # Calculate false negative rate
            fn_rate = self._calculate_false_negative_rate(alerts)
            
            # Combined objective (minimize both)
            return 0.7 * fp_rate + 0.3 * fn_rate
        
        # Find optimal threshold
        result = minimize_scalar(
            objective,
            bounds=(historical_data.min(), historical_data.max()),
            method='bounded'
        )
        
        return result.x
```

#### SLO/SLI Definitions
```python
class SLOManager:
    """Service Level Objective management with SLI tracking"""
    def __init__(self):
        self.slos = {}
        self.sli_calculators = {}
        self.error_budget_tracker = ErrorBudgetTracker()
        
    def define_audio_processing_slos(self):
        """Define SLOs for audio enhancement service"""
        slos = {
            'availability': {
                'target': 99.9,  # Three nines
                'window': '30d',
                'sli': {
                    'name': 'successful_requests_ratio',
                    'formula': 'sum(successful_requests) / sum(total_requests)',
                    'good_threshold': 'status_code < 500'
                }
            },
            'latency': {
                'target': 95,  # 95% of requests
                'threshold': 100,  # under 100ms
                'window': '7d',
                'sli': {
                    'name': 'request_latency_p95',
                    'formula': 'histogram_quantile(0.95, request_duration_seconds)',
                    'good_threshold': 'latency < 0.1'
                }
            },
            'quality': {
                'target': 99,  # 99% meet quality threshold
                'window': '7d',
                'sli': {
                    'name': 'audio_quality_score',
                    'formula': 'sum(quality_score > 3.5) / sum(processed_audio)',
                    'good_threshold': 'pesq > 3.5 AND stoi > 0.8'
                }
            },
            'throughput': {
                'target': 99.5,
                'threshold': 100,  # files per minute
                'window': '1d',
                'sli': {
                    'name': 'processing_throughput',
                    'formula': 'rate(processed_files[5m])',
                    'good_threshold': 'rate > 100'
                }
            }
        }
        
        return slos
    
    def calculate_error_budget(self, slo_name, time_window):
        """Calculate remaining error budget"""
        slo = self.slos[slo_name]
        
        # Get SLI measurements
        measurements = self.get_sli_measurements(slo_name, time_window)
        
        # Calculate achieved performance
        achieved = (sum(m['good'] for m in measurements) / 
                   len(measurements) * 100)
        
        # Calculate error budget
        total_budget = 100 - slo['target']
        consumed_budget = max(0, slo['target'] - achieved)
        remaining_budget = total_budget - consumed_budget
        
        # Time-based analysis
        burn_rate = consumed_budget / (time_window.total_seconds() / 3600)  # per hour
        
        return {
            'slo_target': slo['target'],
            'achieved_performance': achieved,
            'total_error_budget': total_budget,
            'consumed_budget': consumed_budget,
            'remaining_budget': remaining_budget,
            'burn_rate_per_hour': burn_rate,
            'time_until_exhausted': remaining_budget / burn_rate if burn_rate > 0 else float('inf')
        }
```

### Phase 3: Integration
- Cloud monitoring services
- Incident management
- SLA tracking
- Cost monitoring

## Acceptance Criteria
1. ✅ < 1s metric collection latency
2. ✅ 99.9% monitoring uptime
3. ✅ Alert response time < 30s
4. ✅ Support for 100+ metrics
5. ✅ Historical data retention 90 days

## Example Usage
```python
from monitoring import ProductionMonitor

# Initialize monitor
monitor = ProductionMonitor(config={
    'collection_interval': 10,
    'retention_days': 90,
    'alert_channels': ['email', 'slack', 'pagerduty']
})

# Collect metrics
monitor.collect_metrics('audio_enhancement', {
    'processing_time': 245,  # ms
    'quality_score': 3.8,
    'cpu_usage': 45.2,
    'memory_usage': 1024,  # MB
    'queue_size': 15
})

# Set up alerts
monitor.add_alert_rule(
    name='high_latency',
    condition='processing_time > 500',
    severity='warning',
    channels=['slack']
)

monitor.add_alert_rule(
    name='low_quality',
    condition='quality_score < 3.0',
    severity='critical',
    channels=['email', 'pagerduty']
)

# Check system health
health = monitor.check_health()
print(f"System status: {health.status}")
print(f"Services healthy: {health.healthy_services}/{health.total_services}")

# Get dashboard data
dashboard_data = monitor.get_dashboard_data(
    timeframe='last_hour',
    metrics=['latency', 'quality', 'throughput']
)

# Generate SLA report
sla_report = monitor.generate_sla_report(month='2024-01')
print(f"SLA compliance: {sla_report.compliance:.2f}%")
```

## Dependencies
- Prometheus for metrics
- Grafana for dashboards
- AlertManager for alerts
- ElasticSearch for logs
- Redis for caching

## Performance Targets
- Metric ingestion: > 10k/second
- Dashboard refresh: < 5 seconds
- Alert latency: < 30 seconds
- Storage efficiency: < 1GB/day

## Notes
- Implement data sampling for scale
- Consider metric cardinality
- Support for custom metrics
- Enable self-monitoring

## Advanced Monitoring Theory and Implementation

### Time Series Database Optimization
```python
class TimeSeriesOptimizer:
    """Optimize time series data storage and retrieval"""
    def __init__(self):
        self.compression_engine = CompressionEngine()
        self.retention_manager = RetentionManager()
        
    def implement_downsampling(self, metrics, retention_policy):
        """Implement intelligent downsampling for long-term storage"""
        downsampling_rules = {
            '1h': {'method': 'average', 'keep': ['min', 'max', 'avg', 'count']},
            '1d': {'method': 'average', 'keep': ['p50', 'p95', 'p99']},
            '1w': {'method': 'max', 'keep': ['max', 'p99']},
            '1M': {'method': 'max', 'keep': ['max']}
        }
        
        # Apply gorilla compression for recent data
        compressed_recent = self.compression_engine.gorilla_compress(
            metrics['recent'],
            timestamp_compression='delta-of-delta',
            value_compression='xor'
        )
        
        # Downsample older data
        downsampled = {}
        for age, rule in downsampling_rules.items():
            if age in metrics:
                downsampled[age] = self._apply_downsampling(
                    metrics[age],
                    rule['method'],
                    rule['keep']
                )
        
        return {
            'recent': compressed_recent,
            'historical': downsampled,
            'compression_ratio': self._calculate_compression_ratio(
                metrics, compressed_recent, downsampled
            )
        }
```

### Predictive Monitoring
```python
class PredictiveMonitor:
    """Predict issues before they occur"""
    def __init__(self):
        self.prediction_models = {
            'capacity': self._build_capacity_model(),
            'failure': self._build_failure_model(),
            'performance': self._build_performance_model()
        }
        
    def predict_capacity_exhaustion(self, resource_metrics, horizon='24h'):
        """Predict when resources will be exhausted"""
        # Extract trends using STL decomposition
        from statsmodels.tsa.seasonal import STL
        
        stl = STL(resource_metrics, seasonal=13)  # Daily seasonality
        result = stl.fit()
        
        # Project trend forward
        trend = result.trend
        seasonal = result.seasonal
        
        # ARIMA for trend prediction
        from statsmodels.tsa.arima.model import ARIMA
        
        model = ARIMA(trend, order=(2, 1, 2))
        model_fit = model.fit()
        
        # Forecast
        forecast_steps = self._horizon_to_steps(horizon)
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Add seasonal component
        seasonal_forecast = np.tile(seasonal[-24:], forecast_steps // 24 + 1)[:forecast_steps]
        
        total_forecast = forecast + seasonal_forecast
        
        # Find exhaustion point
        capacity_limit = resource_metrics.max() * 0.9  # 90% threshold
        exhaustion_point = np.where(total_forecast > capacity_limit)[0]
        
        if len(exhaustion_point) > 0:
            hours_until_exhaustion = exhaustion_point[0]
            return {
                'will_exhaust': True,
                'hours_until_exhaustion': hours_until_exhaustion,
                'predicted_peak': total_forecast.max(),
                'confidence_interval': model_fit.forecast(
                    steps=forecast_steps, 
                    alpha=0.05
                )[1]
            }
        else:
            return {
                'will_exhaust': False,
                'predicted_peak': total_forecast.max(),
                'capacity_headroom': capacity_limit - total_forecast.max()
            }
```