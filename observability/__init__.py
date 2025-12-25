"""
Observability module for Digital Twin Robotics Lab.

Provides comprehensive distributed tracing using OpenTelemetry with
specialized instrumentation for:
- ROS 2 nodes and communications
- NVIDIA AI services (Riva, NIM, Triton)
- HTTP/gRPC services
- Redis operations
"""

from .tracing import (
    TracingConfig,
    DistributedTracer,
    VoiceProcessingTracer,
    RobotControlTracer,
    SimulationTracer,
    get_tracer,
)

from .ros2_instrumentation import (
    TracedNode,
    TracedPublisher,
    TracedServiceClient,
    TracedActionClient,
    trace_ros2_callback,
)

from .nvidia_instrumentation import (
    RivaTracer,
    NIMTracer,
    TritonTracer,
    NVIDIATracingMiddleware,
    trace_nvidia_call,
)


__all__ = [
    # Core tracing
    'TracingConfig',
    'DistributedTracer',
    'VoiceProcessingTracer',
    'RobotControlTracer',
    'SimulationTracer',
    'get_tracer',
    # ROS 2 instrumentation
    'TracedNode',
    'TracedPublisher',
    'TracedServiceClient',
    'TracedActionClient',
    'trace_ros2_callback',
    # NVIDIA instrumentation
    'RivaTracer',
    'NIMTracer',
    'TritonTracer',
    'NVIDIATracingMiddleware',
    'trace_nvidia_call',
]

__version__ = '1.0.0'
