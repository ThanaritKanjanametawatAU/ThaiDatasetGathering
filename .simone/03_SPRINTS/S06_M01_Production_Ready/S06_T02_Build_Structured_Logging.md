# Task S06_T02: Build Structured Logging

## Task Overview
Build a comprehensive structured logging system that provides detailed, searchable, and analyzable logs for the audio enhancement pipeline in production environments.

## Technical Requirements

### Core Implementation
- **Structured Logging System** (`utils/structured_logging.py`)
  - JSON-formatted logs
  - Contextual information
  - Log aggregation
  - Search capabilities

### Key Features
1. **Log Structure**
   - Standardized format
   - Request correlation
   - Error tracking
   - Performance metrics

2. **Log Levels**
   - DEBUG: Detailed debugging
   - INFO: General information
   - WARNING: Warning conditions
   - ERROR: Error conditions
   - CRITICAL: Critical issues

3. **Context Management**
   - Request tracking
   - User identification
   - Session correlation
   - Distributed tracing

## TDD Requirements

### Test Structure
```
tests/test_structured_logging.py
- test_log_formatting()
- test_context_propagation()
- test_error_logging()
- test_performance_logging()
- test_log_aggregation()
- test_search_functionality()
```

### Test Data Requirements
- Log scenarios
- Error conditions
- Performance data
- Context information

## Implementation Approach

### Phase 1: Core Logging
```python
class StructuredLogger:
    def __init__(self, service_name):
        self.service_name = service_name
        self.context = LogContext()
        self.handlers = self._init_handlers()
        
    def log(self, level, message, **kwargs):
        # Structured log entry
        pass
    
    def with_context(self, **context):
        # Add context to logs
        pass
    
    def log_performance(self, operation, duration, **metrics):
        # Performance logging
        pass
```

### Phase 2: Advanced Features
- Log sampling
- Sensitive data masking
- Log enrichment
- Async logging

#### Advanced Log Processing Pipeline
```python
class LogProcessingPipeline:
    """Advanced log processing with multiple stages"""
    def __init__(self):
        self.processors = [
            SensitiveDataMasker(),
            LogEnricher(),
            LogCompressor(),
            LogRouter()
        ]
        self.async_queue = AsyncLogQueue(max_size=10000)
        
    def process_log_entry(self, log_entry):
        """Process log through pipeline stages"""
        processed = log_entry
        
        for processor in self.processors:
            processed = processor.process(processed)
            
        return processed
    
    async def async_log_processing(self):
        """Asynchronous log processing for high throughput"""
        batch_size = 100
        batch = []
        
        while True:
            try:
                # Get log entry with timeout
                log_entry = await asyncio.wait_for(
                    self.async_queue.get(),
                    timeout=1.0
                )
                
                batch.append(log_entry)
                
                # Process batch when full or on timeout
                if len(batch) >= batch_size:
                    await self._process_batch(batch)
                    batch = []
                    
            except asyncio.TimeoutError:
                # Process partial batch on timeout
                if batch:
                    await self._process_batch(batch)
                    batch = []


class SensitiveDataMasker:
    """Intelligent sensitive data detection and masking"""
    def __init__(self):
        self.patterns = self._compile_patterns()
        self.ml_detector = self._load_ml_detector()
        
    def _compile_patterns(self):
        """Compile regex patterns for sensitive data"""
        return {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            'ssn': re.compile(r'\b(?!000|666)[0-9]{3}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'),
            'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),
            'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b')
        }
    
    def process(self, log_entry):
        """Mask sensitive data in log entry"""
        if isinstance(log_entry, dict):
            return self._mask_dict(log_entry)
        elif isinstance(log_entry, str):
            return self._mask_string(log_entry)
        else:
            return log_entry
    
    def _mask_dict(self, data, depth=0, max_depth=10):
        """Recursively mask sensitive data in dictionary"""
        if depth > max_depth:
            return "[DEPTH_LIMIT_EXCEEDED]"
            
        masked = {}
        
        for key, value in data.items():
            # Check if key indicates sensitive data
            if self._is_sensitive_key(key):
                masked[key] = "[REDACTED]"
            elif isinstance(value, dict):
                masked[key] = self._mask_dict(value, depth + 1)
            elif isinstance(value, list):
                masked[key] = [self._mask_dict(item, depth + 1) if isinstance(item, dict) 
                              else self._mask_string(str(item)) 
                              for item in value]
            elif isinstance(value, str):
                masked[key] = self._mask_string(value)
            else:
                masked[key] = value
                
        return masked
    
    def _mask_string(self, text):
        """Mask sensitive patterns in string"""
        masked_text = text
        
        # Apply regex patterns
        for pattern_name, pattern in self.patterns.items():
            masked_text = pattern.sub(f'[{pattern_name.upper()}_REDACTED]', masked_text)
        
        # Apply ML-based detection for context-aware masking
        if self.ml_detector:
            entities = self.ml_detector.detect_entities(masked_text)
            for entity in entities:
                if entity['type'] in ['PERSON', 'LOCATION', 'ORGANIZATION']:
                    masked_text = masked_text.replace(
                        entity['text'], 
                        f'[{entity["type"]}_REDACTED]'
                    )
        
        return masked_text
    
    def _is_sensitive_key(self, key):
        """Check if dictionary key indicates sensitive data"""
        sensitive_keywords = [
            'password', 'pwd', 'secret', 'token', 'api_key', 'apikey',
            'private_key', 'privatekey', 'ssn', 'social_security',
            'credit_card', 'creditcard', 'bank_account', 'bankaccount'
        ]
        
        key_lower = key.lower()
        return any(keyword in key_lower for keyword in sensitive_keywords)
```

#### Log Enrichment and Correlation
```python
class LogEnricher:
    """Enrich logs with contextual information"""
    def __init__(self):
        self.enrichment_sources = {
            'geo': GeoIPEnricher(),
            'user': UserContextEnricher(),
            'system': SystemMetricsEnricher(),
            'trace': TraceContextEnricher()
        }
        self.cache = TTLCache(maxsize=10000, ttl=300)
        
    def process(self, log_entry):
        """Enrich log entry with additional context"""
        enriched = log_entry.copy()
        
        # Add standard enrichments
        enriched['@timestamp'] = datetime.utcnow().isoformat()
        enriched['@version'] = '1.0'
        
        # Apply specific enrichments based on log type
        log_type = log_entry.get('type', 'generic')
        
        if log_type == 'http_request':
            enriched = self._enrich_http_request(enriched)
        elif log_type == 'audio_processing':
            enriched = self._enrich_audio_processing(enriched)
        elif log_type == 'error':
            enriched = self._enrich_error(enriched)
        
        # Add correlation IDs
        enriched = self._add_correlation_ids(enriched)
        
        return enriched
    
    def _enrich_audio_processing(self, log_entry):
        """Enrich audio processing logs"""
        if 'audio_id' in log_entry:
            # Add audio metadata
            audio_meta = self._get_audio_metadata(log_entry['audio_id'])
            log_entry['audio_metadata'] = audio_meta
            
        if 'processing_stage' in log_entry:
            # Add stage-specific metrics
            stage_metrics = self._get_stage_metrics(log_entry['processing_stage'])
            log_entry['stage_metrics'] = stage_metrics
            
        # Add quality metrics if available
        if 'output_path' in log_entry:
            quality_metrics = self._compute_quality_metrics(log_entry['output_path'])
            log_entry['quality_metrics'] = quality_metrics
            
        return log_entry
    
    def _add_correlation_ids(self, log_entry):
        """Add various correlation IDs for tracking"""
        # Request correlation
        if 'request_id' not in log_entry:
            log_entry['request_id'] = self._get_current_request_id()
            
        # Session correlation
        if 'session_id' not in log_entry:
            log_entry['session_id'] = self._get_current_session_id()
            
        # Trace correlation
        trace_context = self.enrichment_sources['trace'].get_current_trace()
        if trace_context:
            log_entry['trace_id'] = trace_context['trace_id']
            log_entry['span_id'] = trace_context['span_id']
            
        # Business transaction ID
        if 'transaction_id' not in log_entry:
            log_entry['transaction_id'] = self._infer_transaction_id(log_entry)
            
        return log_entry
```

#### High-Performance Async Logging
```python
class AsyncStructuredLogger:
    """High-performance asynchronous structured logger"""
    def __init__(self, service_name, buffer_size=10000):
        self.service_name = service_name
        self.buffer = RingBuffer(buffer_size)
        self.writer_pool = ThreadPoolExecutor(max_workers=4)
        self.running = True
        self._start_background_tasks()
        
    def log(self, level, message, **kwargs):
        """Non-blocking log method"""
        log_entry = {
            'timestamp': time.time_ns(),
            'level': level,
            'message': message,
            'service': self.service_name,
            'thread_id': threading.get_ident(),
            'process_id': os.getpid(),
            **kwargs
        }
        
        # Add to ring buffer (lock-free)
        if not self.buffer.try_put(log_entry):
            # Buffer full, increment dropped counter
            self._increment_dropped_logs()
            
    def _start_background_tasks(self):
        """Start background log processing tasks"""
        # Start multiple writer threads
        for i in range(4):
            self.writer_pool.submit(self._writer_task, i)
            
        # Start batch processor
        self.writer_pool.submit(self._batch_processor)
        
    def _writer_task(self, worker_id):
        """Background task for writing logs"""
        local_buffer = []
        
        while self.running:
            try:
                # Get items from ring buffer
                item = self.buffer.get(timeout=0.1)
                if item:
                    local_buffer.append(item)
                    
                # Write when buffer reaches threshold
                if len(local_buffer) >= 100:
                    self._write_batch(local_buffer)
                    local_buffer = []
                    
            except Empty:
                # Write any remaining items
                if local_buffer:
                    self._write_batch(local_buffer)
                    local_buffer = []
    
    def _batch_processor(self):
        """Process log batches for optimization"""
        while self.running:
            try:
                # Collect logs for batch processing
                batch = self.buffer.get_batch(max_size=1000, timeout=1.0)
                
                if batch:
                    # Compress similar logs
                    compressed = self._compress_similar_logs(batch)
                    
                    # Apply sampling if needed
                    sampled = self._apply_sampling(compressed)
                    
                    # Send to log aggregator
                    self._send_to_aggregator(sampled)
                    
            except Exception as e:
                # Self-log errors
                self._self_log_error(e)
    
    def _compress_similar_logs(self, batch):
        """Compress similar log entries"""
        compressed = []
        groups = defaultdict(list)
        
        # Group by message template
        for log in batch:
            template = self._extract_template(log['message'])
            groups[template].append(log)
            
        # Compress each group
        for template, logs in groups.items():
            if len(logs) > 5:
                # Compress into single entry with count
                compressed.append({
                    'message': template,
                    'count': len(logs),
                    'first_occurrence': logs[0]['timestamp'],
                    'last_occurrence': logs[-1]['timestamp'],
                    'sample_details': logs[0]
                })
            else:
                compressed.extend(logs)
                
        return compressed
```

#### Log Search and Analysis Infrastructure
```python
class LogSearchEngine:
    """High-performance log search engine"""
    def __init__(self):
        self.index = InvertedIndex()
        self.query_parser = LogQueryParser()
        self.aggregator = LogAggregator()
        
    def search(self, query, time_range=None, limit=1000):
        """Search logs with complex queries"""
        # Parse query
        parsed_query = self.query_parser.parse(query)
        
        # Execute search
        results = self._execute_search(parsed_query, time_range)
        
        # Apply post-processing
        processed = self._post_process_results(results, limit)
        
        return processed
    
    def _execute_search(self, parsed_query, time_range):
        """Execute parsed search query"""
        if parsed_query['type'] == 'simple':
            return self.index.search(
                parsed_query['terms'],
                time_range=time_range
            )
        elif parsed_query['type'] == 'complex':
            # Build query execution plan
            plan = self._build_query_plan(parsed_query)
            
            # Execute plan with optimizations
            return self._execute_query_plan(plan)
    
    def aggregate(self, query, aggregations, time_range=None):
        """Perform aggregations on log data"""
        # Get matching logs
        logs = self.search(query, time_range, limit=None)
        
        # Apply aggregations
        results = {}
        for agg_name, agg_config in aggregations.items():
            results[agg_name] = self.aggregator.aggregate(
                logs,
                agg_config['type'],
                agg_config['field'],
                **agg_config.get('params', {})
            )
            
        return results
```

### Phase 3: Integration
- Central log aggregation
- Search infrastructure
- Analytics integration
- Compliance features

## Acceptance Criteria
1. ✅ Structured JSON format
2. ✅ < 1ms logging overhead
3. ✅ 100% request correlation
4. ✅ Searchable within 5 seconds
5. ✅ PII data protection

## Example Usage
```python
from utils import StructuredLogger

# Initialize logger
logger = StructuredLogger(service_name='audio_enhancement')

# Basic logging
logger.info("Processing started", 
    audio_id="audio_123",
    duration=30.5,
    format="wav"
)

# Context management
with logger.with_context(
    request_id="req_456",
    user_id="user_789",
    session_id="sess_abc"
):
    # All logs within this block include context
    logger.debug("Loading audio file", file_path="/data/audio.wav")
    
    try:
        result = enhance_audio(audio_data)
        logger.info("Enhancement completed",
            quality_score=result.quality,
            processing_time=result.duration
        )
    except Exception as e:
        logger.error("Enhancement failed",
            error_type=type(e).__name__,
            error_message=str(e),
            stack_trace=traceback.format_exc()
        )

# Performance logging
with logger.measure_performance("audio_enhancement"):
    enhanced = process_audio(audio)
    
# Structured error logging
logger.log_error(
    error=exception,
    context={
        'operation': 'speaker_separation',
        'audio_length': 45.2,
        'retry_count': 2
    }
)

# Audit logging
logger.audit("Configuration changed",
    user="admin@example.com",
    action="update_threshold",
    old_value=0.7,
    new_value=0.8
)
```

## Dependencies
- structlog for structured logging
- python-json-logger
- Elasticsearch for storage
- Kibana for visualization
- Logstash for processing

## Performance Targets
- Logging overhead: < 1ms
- Log ingestion: > 10k logs/second
- Search latency: < 100ms
- Storage: < 100GB/month

## Notes
- Implement log rotation
- Consider log sampling for high volume
- Mask sensitive information
- Enable log correlation across services

## Advanced Logging Theory and Best Practices

### Log Sampling Strategies
```python
class AdaptiveLogSampler:
    """Adaptive sampling based on system load and log importance"""
    def __init__(self):
        self.sampling_rates = {
            'debug': 0.01,
            'info': 0.1,
            'warning': 0.5,
            'error': 1.0,
            'critical': 1.0
        }
        self.load_adjuster = LoadAdjuster()
        
    def should_sample(self, log_entry):
        """Determine if log should be sampled"""
        # Always log errors and above
        if log_entry['level'] in ['error', 'critical']:
            return True
            
        # Adjust sampling based on system load
        load_factor = self.load_adjuster.get_load_factor()
        adjusted_rate = self.sampling_rates[log_entry['level']] * (1 - load_factor)
        
        # Importance-based sampling
        importance = self._calculate_importance(log_entry)
        final_rate = min(1.0, adjusted_rate * importance)
        
        return random.random() < final_rate
    
    def _calculate_importance(self, log_entry):
        """Calculate log importance score"""
        importance = 1.0
        
        # User-facing operations are more important
        if log_entry.get('user_facing', False):
            importance *= 2.0
            
        # Critical path operations
        if log_entry.get('critical_path', False):
            importance *= 1.5
            
        # Recent errors increase importance
        if self._recent_errors_detected():
            importance *= 1.3
            
        return min(importance, 3.0)  # Cap at 3x
```

### Log Compression and Storage Optimization
```python
class LogCompressionEngine:
    """Advanced log compression techniques"""
    def __init__(self):
        self.template_cache = LRUCache(maxsize=1000)
        self.dictionary_compressor = DictionaryCompressor()
        
    def compress_logs(self, logs):
        """Apply multiple compression techniques"""
        # Template extraction
        templated_logs = self._extract_templates(logs)
        
        # Dictionary compression
        dict_compressed = self.dictionary_compressor.compress(templated_logs)
        
        # Time series compression for timestamps
        time_compressed = self._compress_timestamps(dict_compressed)
        
        # Field-specific compression
        field_compressed = self._compress_fields(time_compressed)
        
        return {
            'compressed_data': field_compressed,
            'compression_ratio': len(logs) / len(field_compressed),
            'templates': self.template_cache.get_all()
        }
    
    def _extract_templates(self, logs):
        """Extract and replace common patterns with templates"""
        compressed = []
        
        for log in logs:
            message = log['message']
            
            # Extract variables from message
            template, variables = self._extract_variables(message)
            
            # Cache template
            template_id = self.template_cache.get_or_set(template)
            
            compressed.append({
                **log,
                'template_id': template_id,
                'variables': variables
            })
            
        return compressed
```