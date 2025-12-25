"""
OpenTelemetry Distributed Tracing for Digital Twin Robotics Lab.

This module provides comprehensive distributed tracing across all services using
OpenTelemetry, enabling end-to-end request visibility, latency analysis, and
debugging capabilities.
"""

import os
import time
import functools
from typing import Dict, Any, Optional, Callable, TypeVar
from contextlib import contextmanager
import asyncio

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.propagate import set_global_textmap, inject, extract
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
from opentelemetry.baggage import set_baggage, get_baggage


# Type variable for generic function decoration
F = TypeVar('F', bound=Callable[..., Any])


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        otlp_endpoint: Optional[str] = None,
        environment: str = "development",
        sample_rate: float = 1.0,
        enable_console_export: bool = False
    ):
        """
        Initialize tracing configuration.
        
        Args:
            service_name: Name of the service for trace identification
            service_version: Version of the service
            otlp_endpoint: OTLP collector endpoint (e.g., "http://jaeger:4317")
            environment: Deployment environment (development, staging, production)
            sample_rate: Fraction of traces to sample (0.0 to 1.0)
            enable_console_export: Whether to also export to console
        """
        self.service_name = service_name
        self.service_version = service_version
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"
        )
        self.environment = environment
        self.sample_rate = sample_rate
        self.enable_console_export = enable_console_export


class DistributedTracer:
    """
    Distributed tracing manager using OpenTelemetry.
    
    Provides:
    - Automatic span creation and propagation
    - Cross-service context propagation
    - Custom span attributes and events
    - Error tracking and status reporting
    - Integration with gRPC, HTTP, Redis, and async operations
    
    Example:
        >>> tracer = DistributedTracer(TracingConfig("cognitive-layer"))
        >>> tracer.initialize()
        >>> 
        >>> with tracer.span("process_voice_command") as span:
        ...     span.set_attribute("language", "en-US")
        ...     result = process_command(audio_data)
        ...     span.add_event("command_processed", {"intent": result.intent})
    """
    
    def __init__(self, config: TracingConfig):
        """Initialize the distributed tracer with configuration."""
        self.config = config
        self._tracer: Optional[trace.Tracer] = None
        self._provider: Optional[TracerProvider] = None
        self._propagator = TraceContextTextMapPropagator()
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize the tracing system.
        
        Sets up the TracerProvider with appropriate exporters and
        instruments common libraries (requests, aiohttp, redis, grpc).
        """
        if self._initialized:
            return
        
        # Create resource with service information
        resource = Resource.create({
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "deployment.environment": self.config.environment,
            "telemetry.sdk.language": "python",
        })
        
        # Create and configure TracerProvider
        self._provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter for production tracing
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True  # Use TLS in production
        )
        self._provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        # Optionally add console exporter for debugging
        if self.config.enable_console_export:
            console_exporter = ConsoleSpanExporter()
            self._provider.add_span_processor(BatchSpanProcessor(console_exporter))
        
        # Set as global provider
        trace.set_tracer_provider(self._provider)
        
        # Set global propagator for context propagation
        set_global_textmap(self._propagator)
        
        # Get tracer instance
        self._tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
        
        # Instrument common libraries
        self._instrument_libraries()
        
        self._initialized = True
    
    def _instrument_libraries(self) -> None:
        """Instrument common libraries for automatic tracing."""
        # HTTP requests
        RequestsInstrumentor().instrument()
        
        # Async HTTP
        AioHttpClientInstrumentor().instrument()
        
        # Redis
        RedisInstrumentor().instrument()
        
        # gRPC client and server
        GrpcInstrumentorClient().instrument()
        GrpcInstrumentorServer().instrument()
    
    @property
    def tracer(self) -> trace.Tracer:
        """Get the tracer instance, initializing if necessary."""
        if not self._initialized:
            self.initialize()
        return self._tracer
    
    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[Any] = None
    ):
        """
        Create a new span as a context manager.
        
        Args:
            name: Name of the span (e.g., "asr_transcribe")
            kind: Type of span (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
            attributes: Initial attributes to set on the span
            parent_context: Optional parent context for distributed tracing
            
        Yields:
            The created span object
            
        Example:
            >>> with tracer.span("process_audio", attributes={"format": "wav"}) as span:
            ...     result = transcribe(audio)
            ...     span.set_attribute("transcript_length", len(result))
        """
        ctx = parent_context or trace.get_current_span().get_span_context()
        
        with self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes or {}
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def trace(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Callable[[F], F]:
        """
        Decorator for tracing functions.
        
        Args:
            name: Span name (defaults to function name)
            kind: Type of span
            attributes: Static attributes to add to span
            
        Returns:
            Decorated function with automatic tracing
            
        Example:
            >>> @tracer.trace(attributes={"component": "asr"})
            ... def transcribe_audio(audio_data: bytes) -> str:
            ...     return riva_client.transcribe(audio_data)
        """
        def decorator(func: F) -> F:
            span_name = name or func.__name__
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.span(span_name, kind=kind, attributes=attributes) as span:
                    # Add function arguments as attributes (careful with sensitive data)
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    finally:
                        duration = time.perf_counter() - start_time
                        span.set_attribute("function.duration_ms", duration * 1000)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.span(span_name, kind=kind, attributes=attributes) as span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    start_time = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    finally:
                        duration = time.perf_counter() - start_time
                        span.set_attribute("function.duration_ms", duration * 1000)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def inject_context(self, carrier: Dict[str, str]) -> None:
        """
        Inject trace context into a carrier for propagation.
        
        Args:
            carrier: Dictionary to inject context headers into
            
        Example:
            >>> headers = {}
            >>> tracer.inject_context(headers)
            >>> requests.get("http://service/api", headers=headers)
        """
        inject(carrier)
    
    def extract_context(self, carrier: Dict[str, str]) -> Any:
        """
        Extract trace context from a carrier.
        
        Args:
            carrier: Dictionary containing context headers
            
        Returns:
            Extracted context for creating child spans
            
        Example:
            >>> context = tracer.extract_context(request.headers)
            >>> with tracer.span("handle_request", parent_context=context):
            ...     process_request()
        """
        return extract(carrier)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an event to the current span.
        
        Args:
            name: Event name
            attributes: Event attributes
            
        Example:
            >>> tracer.add_event("cache_hit", {"key": "user_session"})
        """
        span = trace.get_current_span()
        span.add_event(name, attributes or {})
    
    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set an attribute on the current span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        span = trace.get_current_span()
        span.set_attribute(key, value)
    
    def set_baggage(self, key: str, value: str) -> None:
        """
        Set baggage for propagation across services.
        
        Baggage is propagated to all downstream services.
        
        Args:
            key: Baggage key
            value: Baggage value
        """
        set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """
        Get baggage value.
        
        Args:
            key: Baggage key
            
        Returns:
            Baggage value or None if not set
        """
        return get_baggage(key)
    
    def get_trace_id(self) -> Optional[str]:
        """
        Get the current trace ID.
        
        Returns:
            Trace ID as hex string, or None if no active span
        """
        span = trace.get_current_span()
        if span:
            return format(span.get_span_context().trace_id, '032x')
        return None
    
    def get_span_id(self) -> Optional[str]:
        """
        Get the current span ID.
        
        Returns:
            Span ID as hex string, or None if no active span
        """
        span = trace.get_current_span()
        if span:
            return format(span.get_span_context().span_id, '016x')
        return None
    
    def shutdown(self) -> None:
        """Shutdown the tracer and flush pending spans."""
        if self._provider:
            self._provider.shutdown()


# Service-specific tracers
class VoiceProcessingTracer(DistributedTracer):
    """Tracer specialized for voice processing operations."""
    
    @contextmanager
    def asr_span(
        self,
        language: str,
        audio_duration_ms: float,
        sample_rate: int = 16000
    ):
        """
        Create a span for ASR transcription.
        
        Args:
            language: Language code (e.g., "en-US")
            audio_duration_ms: Duration of audio in milliseconds
            sample_rate: Audio sample rate in Hz
        """
        attributes = {
            "asr.language": language,
            "asr.audio_duration_ms": audio_duration_ms,
            "asr.sample_rate": sample_rate,
            "component": "riva_asr"
        }
        with self.span("asr_transcribe", kind=SpanKind.CLIENT, attributes=attributes) as span:
            yield span
    
    @contextmanager
    def tts_span(
        self,
        language: str,
        voice_name: str,
        text_length: int
    ):
        """
        Create a span for TTS synthesis.
        
        Args:
            language: Language code
            voice_name: Voice identifier
            text_length: Length of text to synthesize
        """
        attributes = {
            "tts.language": language,
            "tts.voice_name": voice_name,
            "tts.text_length": text_length,
            "component": "riva_tts"
        }
        with self.span("tts_synthesize", kind=SpanKind.CLIENT, attributes=attributes) as span:
            yield span
    
    @contextmanager
    def llm_span(
        self,
        model: str,
        prompt_tokens: int,
        intent: Optional[str] = None
    ):
        """
        Create a span for LLM inference.
        
        Args:
            model: Model name (e.g., "llama-3.1-8b")
            prompt_tokens: Number of tokens in prompt
            intent: Extracted intent (set after completion)
        """
        attributes = {
            "llm.model": model,
            "llm.prompt_tokens": prompt_tokens,
            "component": "nvidia_nim"
        }
        with self.span("llm_inference", kind=SpanKind.CLIENT, attributes=attributes) as span:
            if intent:
                span.set_attribute("llm.intent", intent)
            yield span


class RobotControlTracer(DistributedTracer):
    """Tracer specialized for robot control operations."""
    
    @contextmanager
    def navigation_span(
        self,
        robot_id: str,
        target_x: float,
        target_y: float,
        planning_algorithm: str = "nav2"
    ):
        """
        Create a span for navigation planning and execution.
        
        Args:
            robot_id: Unique robot identifier
            target_x: Target X coordinate
            target_y: Target Y coordinate
            planning_algorithm: Path planning algorithm used
        """
        attributes = {
            "robot.id": robot_id,
            "navigation.target_x": target_x,
            "navigation.target_y": target_y,
            "navigation.algorithm": planning_algorithm,
            "component": "nav2"
        }
        with self.span("navigate_to_pose", kind=SpanKind.INTERNAL, attributes=attributes) as span:
            yield span
    
    @contextmanager
    def command_span(
        self,
        robot_id: str,
        command_type: str,
        priority: int = 5
    ):
        """
        Create a span for robot command execution.
        
        Args:
            robot_id: Unique robot identifier
            command_type: Type of command (move, pick, place, etc.)
            priority: Command priority level
        """
        attributes = {
            "robot.id": robot_id,
            "command.type": command_type,
            "command.priority": priority,
            "component": "robot_control"
        }
        with self.span("execute_command", kind=SpanKind.INTERNAL, attributes=attributes) as span:
            yield span


class SimulationTracer(DistributedTracer):
    """Tracer specialized for simulation operations."""
    
    @contextmanager
    def sync_span(
        self,
        robot_id: str,
        data_type: str,
        direction: str
    ):
        """
        Create a span for digital twin synchronization.
        
        Args:
            robot_id: Robot being synchronized
            data_type: Type of data (pose, joint_states, sensor_data)
            direction: Sync direction (physical_to_sim, sim_to_physical)
        """
        attributes = {
            "robot.id": robot_id,
            "sync.data_type": data_type,
            "sync.direction": direction,
            "component": "isaac_sim"
        }
        with self.span("twin_sync", kind=SpanKind.INTERNAL, attributes=attributes) as span:
            yield span
    
    @contextmanager
    def physics_step_span(
        self,
        step_count: int,
        active_bodies: int
    ):
        """
        Create a span for physics simulation step.
        
        Args:
            step_count: Current simulation step number
            active_bodies: Number of active physics bodies
        """
        attributes = {
            "physics.step_count": step_count,
            "physics.active_bodies": active_bodies,
            "component": "isaac_sim_physics"
        }
        with self.span("physics_step", kind=SpanKind.INTERNAL, attributes=attributes) as span:
            yield span


# Global tracer instances for each service
_tracers: Dict[str, DistributedTracer] = {}


def get_tracer(service_name: str) -> DistributedTracer:
    """
    Get or create a tracer for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        DistributedTracer instance for the service
    """
    if service_name not in _tracers:
        config = TracingConfig(service_name)
        
        # Use specialized tracer based on service
        if "voice" in service_name or "cognitive" in service_name:
            _tracers[service_name] = VoiceProcessingTracer(config)
        elif "robot" in service_name or "control" in service_name:
            _tracers[service_name] = RobotControlTracer(config)
        elif "sim" in service_name or "twin" in service_name:
            _tracers[service_name] = SimulationTracer(config)
        else:
            _tracers[service_name] = DistributedTracer(config)
        
        _tracers[service_name].initialize()
    
    return _tracers[service_name]


# Convenience exports
__all__ = [
    'TracingConfig',
    'DistributedTracer',
    'VoiceProcessingTracer',
    'RobotControlTracer',
    'SimulationTracer',
    'get_tracer',
]
