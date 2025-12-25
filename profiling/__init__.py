"""
Performance Profiling Framework for Digital Twin Robotics Lab.

Provides comprehensive profiling capabilities for CPU, GPU, memory,
and latency analysis.
"""

from .profiler import (
    # Enums
    ProfilerType,
    GPUMetricType,
    
    # Data classes
    ProfileResult,
    GPUProfile,
    MemorySnapshot,
    LatencyProfile,
    
    # Profilers
    BaseProfiler,
    CPUProfiler,
    GPUProfiler,
    MemoryProfiler,
    LatencyProfiler,
    CombinedProfiler,
    
    # Decorators
    profile_function,
    profile_async_function
)

from .nvidia_profiler import (
    # CUDA timing
    CUDAEventTimer,
    
    # Nsight Systems
    NsightSystemsConfig,
    NsightSystemsProfiler,
    
    # PyTorch profiling
    PyTorchProfiler,
    
    # TensorRT profiling
    TensorRTProfiler,
    
    # Memory analysis
    GPUMemoryAnalyzer,
    
    # NVTX markers
    NVTXMarker,
    
    # Convenience functions
    profile_inference
)

from .middleware import (
    # Middleware
    PerformanceMiddleware,
    RequestContext,
    SLAConfig,
    
    # Data classes
    RequestMetrics,
    EndpointStats,
    
    # Router
    create_performance_router
)

__all__ = [
    # Core Profiling
    "ProfilerType",
    "GPUMetricType",
    "ProfileResult",
    "GPUProfile",
    "MemorySnapshot",
    "LatencyProfile",
    "BaseProfiler",
    "CPUProfiler",
    "GPUProfiler",
    "MemoryProfiler",
    "LatencyProfiler",
    "CombinedProfiler",
    "profile_function",
    "profile_async_function",
    
    # NVIDIA Profiling
    "CUDAEventTimer",
    "NsightSystemsConfig",
    "NsightSystemsProfiler",
    "PyTorchProfiler",
    "TensorRTProfiler",
    "GPUMemoryAnalyzer",
    "NVTXMarker",
    "profile_inference",
    
    # Middleware
    "PerformanceMiddleware",
    "RequestContext",
    "SLAConfig",
    "RequestMetrics",
    "EndpointStats",
    "create_performance_router"
]

__version__ = "1.0.0"
