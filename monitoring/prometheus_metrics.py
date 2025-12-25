"""
Prometheus Metrics Module for Digital Twin Robotics Lab.

Provides comprehensive metrics instrumentation for all system components
including voice processing, robot control, simulation, and AI inference.

Usage:
    from metrics import MetricsRegistry, metrics_middleware
    
    # Initialize metrics
    registry = MetricsRegistry()
    
    # Use decorators to instrument functions
    @registry.track_latency("asr_processing")
    async def process_speech(audio):
        ...
    
    # Or track manually
    with registry.timer("llm_inference"):
        response = await nim_client.generate(prompt)
"""

import time
import asyncio
import functools
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import threading

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
    start_http_server,
)


class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition for a Prometheus metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

VOICE_METRICS = [
    # ASR Metrics
    MetricDefinition(
        name="asr_requests_total",
        description="Total number of ASR requests processed",
        metric_type=MetricType.COUNTER,
        labels=["language", "status"]
    ),
    MetricDefinition(
        name="asr_processing_seconds",
        description="ASR processing latency in seconds",
        metric_type=MetricType.HISTOGRAM,
        labels=["language"],
        buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
    ),
    MetricDefinition(
        name="asr_audio_duration_seconds",
        description="Duration of audio being processed",
        metric_type=MetricType.HISTOGRAM,
        labels=["language"],
        buckets=[1, 2, 5, 10, 30, 60, 120]
    ),
    MetricDefinition(
        name="asr_word_error_rate",
        description="Estimated word error rate",
        metric_type=MetricType.GAUGE,
        labels=["language"]
    ),
    MetricDefinition(
        name="asr_confidence_score",
        description="ASR confidence scores distribution",
        metric_type=MetricType.HISTOGRAM,
        labels=["language"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ),
    
    # Wake Word Metrics
    MetricDefinition(
        name="wake_word_detections_total",
        description="Total wake word detections",
        metric_type=MetricType.COUNTER,
        labels=["keyword", "confidence_level"]
    ),
    MetricDefinition(
        name="wake_word_false_positives_total",
        description="Estimated false positive wake word detections",
        metric_type=MetricType.COUNTER,
        labels=["keyword"]
    ),
    MetricDefinition(
        name="wake_word_detection_latency_seconds",
        description="Wake word detection latency",
        metric_type=MetricType.HISTOGRAM,
        labels=["keyword"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    ),
    
    # TTS Metrics
    MetricDefinition(
        name="tts_requests_total",
        description="Total TTS synthesis requests",
        metric_type=MetricType.COUNTER,
        labels=["voice", "language", "status"]
    ),
    MetricDefinition(
        name="tts_synthesis_seconds",
        description="TTS synthesis latency",
        metric_type=MetricType.HISTOGRAM,
        labels=["voice"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    ),
    MetricDefinition(
        name="tts_audio_duration_seconds",
        description="Generated audio duration",
        metric_type=MetricType.HISTOGRAM,
        labels=["voice"],
        buckets=[1, 2, 5, 10, 30, 60]
    ),
    
    # Noise Filtering Metrics
    MetricDefinition(
        name="noise_filter_snr_improvement_db",
        description="Signal-to-noise ratio improvement in dB",
        metric_type=MetricType.GAUGE,
        labels=["filter_type"]
    ),
    MetricDefinition(
        name="noise_filter_processing_seconds",
        description="Noise filtering processing time",
        metric_type=MetricType.HISTOGRAM,
        labels=["filter_type"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    ),
]

LLM_METRICS = [
    MetricDefinition(
        name="llm_requests_total",
        description="Total LLM inference requests",
        metric_type=MetricType.COUNTER,
        labels=["model", "intent_type", "status"]
    ),
    MetricDefinition(
        name="llm_inference_seconds",
        description="LLM inference latency",
        metric_type=MetricType.HISTOGRAM,
        labels=["model"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    ),
    MetricDefinition(
        name="llm_tokens_input_total",
        description="Total input tokens processed",
        metric_type=MetricType.COUNTER,
        labels=["model"]
    ),
    MetricDefinition(
        name="llm_tokens_output_total",
        description="Total output tokens generated",
        metric_type=MetricType.COUNTER,
        labels=["model"]
    ),
    MetricDefinition(
        name="llm_tokens_per_second",
        description="Token generation throughput",
        metric_type=MetricType.GAUGE,
        labels=["model"]
    ),
    MetricDefinition(
        name="llm_context_length",
        description="Context length distribution",
        metric_type=MetricType.HISTOGRAM,
        labels=["model"],
        buckets=[128, 256, 512, 1024, 2048, 4096, 8192]
    ),
    MetricDefinition(
        name="llm_intent_confidence",
        description="Intent extraction confidence scores",
        metric_type=MetricType.HISTOGRAM,
        labels=["intent_type"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ),
]

ROBOT_METRICS = [
    MetricDefinition(
        name="robot_commands_total",
        description="Total robot commands executed",
        metric_type=MetricType.COUNTER,
        labels=["robot_id", "command_type", "status"]
    ),
    MetricDefinition(
        name="robot_command_latency_seconds",
        description="Command execution latency",
        metric_type=MetricType.HISTOGRAM,
        labels=["robot_id", "command_type"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    ),
    MetricDefinition(
        name="robot_position_x",
        description="Robot X position in meters",
        metric_type=MetricType.GAUGE,
        labels=["robot_id"]
    ),
    MetricDefinition(
        name="robot_position_y",
        description="Robot Y position in meters",
        metric_type=MetricType.GAUGE,
        labels=["robot_id"]
    ),
    MetricDefinition(
        name="robot_orientation_yaw",
        description="Robot yaw orientation in radians",
        metric_type=MetricType.GAUGE,
        labels=["robot_id"]
    ),
    MetricDefinition(
        name="robot_velocity_linear",
        description="Robot linear velocity in m/s",
        metric_type=MetricType.GAUGE,
        labels=["robot_id"]
    ),
    MetricDefinition(
        name="robot_velocity_angular",
        description="Robot angular velocity in rad/s",
        metric_type=MetricType.GAUGE,
        labels=["robot_id"]
    ),
    MetricDefinition(
        name="robot_battery_percent",
        description="Robot battery percentage",
        metric_type=MetricType.GAUGE,
        labels=["robot_id"]
    ),
    MetricDefinition(
        name="robot_heartbeat_failures_total",
        description="Robot heartbeat failures",
        metric_type=MetricType.COUNTER,
        labels=["robot_id"]
    ),
    MetricDefinition(
        name="robot_emergency_stops_total",
        description="Emergency stop events",
        metric_type=MetricType.COUNTER,
        labels=["robot_id", "reason"]
    ),
]

FLEET_METRICS = [
    MetricDefinition(
        name="fleet_active_robots",
        description="Number of active robots in fleet",
        metric_type=MetricType.GAUGE,
        labels=["fleet_id"]
    ),
    MetricDefinition(
        name="fleet_tasks_queued",
        description="Number of tasks in queue",
        metric_type=MetricType.GAUGE,
        labels=["fleet_id", "priority"]
    ),
    MetricDefinition(
        name="fleet_tasks_completed_total",
        description="Total tasks completed",
        metric_type=MetricType.COUNTER,
        labels=["fleet_id", "task_type", "status"]
    ),
    MetricDefinition(
        name="fleet_task_duration_seconds",
        description="Task completion duration",
        metric_type=MetricType.HISTOGRAM,
        labels=["fleet_id", "task_type"],
        buckets=[10, 30, 60, 120, 300, 600, 1800, 3600]
    ),
    MetricDefinition(
        name="fleet_coordination_latency_seconds",
        description="Fleet coordination message latency",
        metric_type=MetricType.HISTOGRAM,
        labels=["fleet_id"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    ),
    MetricDefinition(
        name="fleet_collision_avoidance_events_total",
        description="Collision avoidance events triggered",
        metric_type=MetricType.COUNTER,
        labels=["fleet_id", "robot_id"]
    ),
]

SIMULATION_METRICS = [
    MetricDefinition(
        name="simulation_fps",
        description="Simulation frames per second",
        metric_type=MetricType.GAUGE,
        labels=["scene"]
    ),
    MetricDefinition(
        name="simulation_physics_step_seconds",
        description="Physics step duration",
        metric_type=MetricType.HISTOGRAM,
        labels=["scene"],
        buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    ),
    MetricDefinition(
        name="twin_sync_latency_ms",
        description="Digital twin synchronization latency in milliseconds",
        metric_type=MetricType.GAUGE,
        labels=["robot_id", "sync_mode"]
    ),
    MetricDefinition(
        name="twin_sync_messages_total",
        description="Total twin sync messages",
        metric_type=MetricType.COUNTER,
        labels=["robot_id", "direction"]
    ),
    MetricDefinition(
        name="twin_sync_errors_total",
        description="Twin sync errors",
        metric_type=MetricType.COUNTER,
        labels=["robot_id", "error_type"]
    ),
]

MAINTENANCE_METRICS = [
    MetricDefinition(
        name="component_wear_percent",
        description="Component wear percentage",
        metric_type=MetricType.GAUGE,
        labels=["robot_id", "component"]
    ),
    MetricDefinition(
        name="component_remaining_useful_life_hours",
        description="Predicted remaining useful life in hours",
        metric_type=MetricType.GAUGE,
        labels=["robot_id", "component"]
    ),
    MetricDefinition(
        name="maintenance_alerts_total",
        description="Maintenance alerts generated",
        metric_type=MetricType.COUNTER,
        labels=["robot_id", "component", "severity"]
    ),
    MetricDefinition(
        name="maintenance_events_total",
        description="Maintenance events completed",
        metric_type=MetricType.COUNTER,
        labels=["robot_id", "component", "event_type"]
    ),
]

INFERENCE_METRICS = [
    MetricDefinition(
        name="triton_inference_requests_total",
        description="Total Triton inference requests",
        metric_type=MetricType.COUNTER,
        labels=["model", "version", "status"]
    ),
    MetricDefinition(
        name="triton_inference_latency_seconds",
        description="Triton inference latency",
        metric_type=MetricType.HISTOGRAM,
        labels=["model"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ),
    MetricDefinition(
        name="triton_batch_size",
        description="Inference batch size distribution",
        metric_type=MetricType.HISTOGRAM,
        labels=["model"],
        buckets=[1, 2, 4, 8, 16, 32, 64]
    ),
    MetricDefinition(
        name="triton_queue_time_seconds",
        description="Time spent in inference queue",
        metric_type=MetricType.HISTOGRAM,
        labels=["model"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5]
    ),
    MetricDefinition(
        name="object_detection_objects_total",
        description="Total objects detected",
        metric_type=MetricType.COUNTER,
        labels=["class_name", "robot_id"]
    ),
    MetricDefinition(
        name="object_detection_confidence",
        description="Detection confidence distribution",
        metric_type=MetricType.HISTOGRAM,
        labels=["class_name"],
        buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ),
]

SYSTEM_METRICS = [
    MetricDefinition(
        name="gpu_utilization_percent",
        description="GPU utilization percentage",
        metric_type=MetricType.GAUGE,
        labels=["gpu_id", "gpu_name"]
    ),
    MetricDefinition(
        name="gpu_memory_used_bytes",
        description="GPU memory used in bytes",
        metric_type=MetricType.GAUGE,
        labels=["gpu_id"]
    ),
    MetricDefinition(
        name="gpu_memory_total_bytes",
        description="GPU total memory in bytes",
        metric_type=MetricType.GAUGE,
        labels=["gpu_id"]
    ),
    MetricDefinition(
        name="gpu_temperature_celsius",
        description="GPU temperature in Celsius",
        metric_type=MetricType.GAUGE,
        labels=["gpu_id"]
    ),
    MetricDefinition(
        name="redis_connections_active",
        description="Active Redis connections",
        metric_type=MetricType.GAUGE,
        labels=[]
    ),
    MetricDefinition(
        name="redis_memory_used_bytes",
        description="Redis memory usage",
        metric_type=MetricType.GAUGE,
        labels=[]
    ),
]


# =============================================================================
# METRICS REGISTRY
# =============================================================================

class MetricsRegistry:
    """
    Central registry for all Prometheus metrics.
    
    Provides a unified interface for creating, accessing, and managing
    metrics across all system components.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for metrics registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize the metrics registry."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._initialized = True
        
        # Register all metrics
        self._register_all_metrics()
    
    def _register_all_metrics(self):
        """Register all defined metrics."""
        all_metrics = (
            VOICE_METRICS +
            LLM_METRICS +
            ROBOT_METRICS +
            FLEET_METRICS +
            SIMULATION_METRICS +
            MAINTENANCE_METRICS +
            INFERENCE_METRICS +
            SYSTEM_METRICS
        )
        
        for metric_def in all_metrics:
            self._create_metric(metric_def)
    
    def _create_metric(self, definition: MetricDefinition):
        """Create a Prometheus metric from definition."""
        metric_class = {
            MetricType.COUNTER: Counter,
            MetricType.HISTOGRAM: Histogram,
            MetricType.GAUGE: Gauge,
            MetricType.SUMMARY: Summary,
            MetricType.INFO: Info,
        }[definition.metric_type]
        
        kwargs = {
            'name': definition.name,
            'documentation': definition.description,
            'labelnames': definition.labels,
            'registry': self._registry,
        }
        
        if definition.buckets and definition.metric_type == MetricType.HISTOGRAM:
            kwargs['buckets'] = definition.buckets
        
        self._metrics[definition.name] = metric_class(**kwargs)
    
    def get(self, name: str) -> Any:
        """Get a metric by name."""
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found")
        return self._metrics[name]
    
    def counter(self, name: str) -> Counter:
        """Get a counter metric."""
        return self.get(name)
    
    def histogram(self, name: str) -> Histogram:
        """Get a histogram metric."""
        return self.get(name)
    
    def gauge(self, name: str) -> Gauge:
        """Get a gauge metric."""
        return self.get(name)
    
    @contextmanager
    def timer(self, histogram_name: str, **labels):
        """
        Context manager for timing operations.
        
        Usage:
            with registry.timer("asr_processing_seconds", language="en-US"):
                result = process_audio(audio)
        """
        histogram = self.histogram(histogram_name)
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            histogram.labels(**labels).observe(duration)
    
    def track_latency(self, histogram_name: str, label_func: Optional[Callable] = None):
        """
        Decorator for tracking function latency.
        
        Usage:
            @registry.track_latency("llm_inference_seconds", lambda args: {"model": args[0].model})
            async def generate(self, prompt):
                ...
        """
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                labels = label_func(args) if label_func else {}
                with self.timer(histogram_name, **labels):
                    return await func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                labels = label_func(args) if label_func else {}
                with self.timer(histogram_name, **labels):
                    return func(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator
    
    def count(self, counter_name: str, value: int = 1, **labels):
        """Increment a counter."""
        self.counter(counter_name).labels(**labels).inc(value)
    
    def set_gauge(self, gauge_name: str, value: float, **labels):
        """Set a gauge value."""
        self.gauge(gauge_name).labels(**labels).set(value)
    
    def observe(self, histogram_name: str, value: float, **labels):
        """Observe a histogram value."""
        self.histogram(histogram_name).labels(**labels).observe(value)
    
    def generate_metrics(self) -> bytes:
        """Generate metrics output for Prometheus scraping."""
        return generate_latest(self._registry)
    
    def get_content_type(self) -> str:
        """Get the content type for metrics response."""
        return CONTENT_TYPE_LATEST


# =============================================================================
# METRICS COLLECTORS
# =============================================================================

class GPUMetricsCollector:
    """Collector for NVIDIA GPU metrics using pynvml."""
    
    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
        self._nvml_initialized = False
        self._init_nvml()
    
    def _init_nvml(self):
        """Initialize NVIDIA Management Library."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._pynvml = pynvml
        except Exception:
            self._nvml_initialized = False
    
    def collect(self):
        """Collect GPU metrics."""
        if not self._nvml_initialized:
            return
        
        try:
            device_count = self._pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                name = self._pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Utilization
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.registry.set_gauge(
                    "gpu_utilization_percent",
                    utilization.gpu,
                    gpu_id=str(i),
                    gpu_name=name
                )
                
                # Memory
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.registry.set_gauge(
                    "gpu_memory_used_bytes",
                    memory.used,
                    gpu_id=str(i)
                )
                self.registry.set_gauge(
                    "gpu_memory_total_bytes",
                    memory.total,
                    gpu_id=str(i)
                )
                
                # Temperature
                temp = self._pynvml.nvmlDeviceGetTemperature(
                    handle,
                    self._pynvml.NVML_TEMPERATURE_GPU
                )
                self.registry.set_gauge(
                    "gpu_temperature_celsius",
                    temp,
                    gpu_id=str(i)
                )
                
        except Exception:
            pass


class RobotMetricsCollector:
    """Collector for robot state metrics."""
    
    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
    
    def update_robot_state(
        self,
        robot_id: str,
        position: tuple,
        orientation: float,
        velocity: tuple,
        battery: float
    ):
        """Update robot state metrics."""
        x, y = position
        linear_vel, angular_vel = velocity
        
        self.registry.set_gauge("robot_position_x", x, robot_id=robot_id)
        self.registry.set_gauge("robot_position_y", y, robot_id=robot_id)
        self.registry.set_gauge("robot_orientation_yaw", orientation, robot_id=robot_id)
        self.registry.set_gauge("robot_velocity_linear", linear_vel, robot_id=robot_id)
        self.registry.set_gauge("robot_velocity_angular", angular_vel, robot_id=robot_id)
        self.registry.set_gauge("robot_battery_percent", battery, robot_id=robot_id)
    
    def record_command(
        self,
        robot_id: str,
        command_type: str,
        status: str,
        latency: float
    ):
        """Record a robot command execution."""
        self.registry.count(
            "robot_commands_total",
            robot_id=robot_id,
            command_type=command_type,
            status=status
        )
        self.registry.observe(
            "robot_command_latency_seconds",
            latency,
            robot_id=robot_id,
            command_type=command_type
        )


# =============================================================================
# HTTP SERVER
# =============================================================================

def start_metrics_server(port: int = 9090, registry: Optional[MetricsRegistry] = None):
    """
    Start a standalone HTTP server for Prometheus metrics.
    
    Args:
        port: Port to listen on (default: 9090)
        registry: MetricsRegistry instance (creates default if None)
    """
    if registry is None:
        registry = MetricsRegistry()
    
    start_http_server(port, registry=registry._registry)
    print(f"Metrics server started on port {port}")


# =============================================================================
# FASTAPI INTEGRATION
# =============================================================================

def create_metrics_endpoint(registry: MetricsRegistry):
    """
    Create a FastAPI endpoint for Prometheus metrics.
    
    Usage:
        from fastapi import FastAPI
        from fastapi.responses import Response
        
        app = FastAPI()
        registry = MetricsRegistry()
        
        @app.get("/metrics")
        async def metrics():
            return create_metrics_endpoint(registry)()
    """
    from fastapi.responses import Response
    
    def metrics_endpoint():
        return Response(
            content=registry.generate_metrics(),
            media_type=registry.get_content_type()
        )
    
    return metrics_endpoint


def metrics_middleware(registry: MetricsRegistry):
    """
    FastAPI middleware for automatic request metrics.
    
    Tracks:
    - Request count by method, path, status
    - Request latency distribution
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    import time
    
    # Create request metrics
    request_counter = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'path', 'status'],
        registry=registry._registry
    )
    
    request_latency = Histogram(
        'http_request_duration_seconds',
        'HTTP request latency',
        ['method', 'path'],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        registry=registry._registry
    )
    
    class MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.perf_counter()
            
            response = await call_next(request)
            
            duration = time.perf_counter() - start
            path = request.url.path
            
            request_counter.labels(
                method=request.method,
                path=path,
                status=response.status_code
            ).inc()
            
            request_latency.labels(
                method=request.method,
                path=path
            ).observe(duration)
            
            return response
    
    return MetricsMiddleware


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import random
    
    # Initialize registry
    registry = MetricsRegistry()
    
    # Example: Track ASR request
    registry.count(
        "asr_requests_total",
        language="en-US",
        status="success"
    )
    
    # Example: Track ASR latency
    with registry.timer("asr_processing_seconds", language="en-US"):
        time.sleep(random.uniform(0.1, 0.5))  # Simulate processing
    
    # Example: Track robot state
    robot_collector = RobotMetricsCollector(registry)
    robot_collector.update_robot_state(
        robot_id="robot_001",
        position=(5.2, 3.1),
        orientation=1.57,
        velocity=(0.5, 0.1),
        battery=87.5
    )
    
    # Example: Track LLM inference
    registry.count(
        "llm_requests_total",
        model="llama-3.1-8b",
        intent_type="navigate",
        status="success"
    )
    registry.observe(
        "llm_inference_seconds",
        0.75,
        model="llama-3.1-8b"
    )
    
    # Example: Track fleet metrics
    registry.set_gauge("fleet_active_robots", 5, fleet_id="main")
    registry.set_gauge("fleet_tasks_queued", 12, fleet_id="main", priority="high")
    
    # Example: Track twin sync
    registry.set_gauge(
        "twin_sync_latency_ms",
        15.3,
        robot_id="robot_001",
        sync_mode="MIRROR"
    )
    
    # Start metrics server
    print("Starting metrics server on port 9090...")
    start_metrics_server(9090, registry)
    
    # Keep running
    print("Metrics available at http://localhost:9090/metrics")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
