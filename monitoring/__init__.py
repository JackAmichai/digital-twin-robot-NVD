"""
Monitoring Module for Digital Twin Robotics Lab.

This module provides comprehensive Prometheus metrics instrumentation
for all system components including voice processing, robot control,
simulation, and AI inference.

Components:
- MetricsRegistry: Central registry for all Prometheus metrics
- CognitiveMetrics: Voice processing metrics (ASR, TTS, Wake Word, LLM)
- RobotMetricsCollector: Robot state and command metrics
- GPUMetricsCollector: NVIDIA GPU metrics via pynvml

Quick Start:
    from monitoring import MetricsRegistry, CognitiveMetrics
    
    # Initialize
    registry = MetricsRegistry()
    cognitive_metrics = CognitiveMetrics(registry)
    
    # Track ASR request
    async with cognitive_metrics.track_asr_request(language="en-US") as tracker:
        result = await asr_client.transcribe(audio)
        tracker.set_confidence(result.confidence)
    
    # Track LLM inference
    async with cognitive_metrics.track_llm_request(model="llama-3.1-8b") as tracker:
        response = await nim_client.generate(prompt)
        tracker.set_intent(response.intent, response.confidence)

FastAPI Integration:
    from fastapi import FastAPI
    from monitoring import MetricsRegistry, metrics_middleware, create_metrics_endpoint
    
    app = FastAPI()
    registry = MetricsRegistry()
    
    # Add middleware for automatic request metrics
    app.add_middleware(metrics_middleware(registry))
    
    # Add /metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return create_metrics_endpoint(registry)()
"""

from monitoring.prometheus_metrics import (
    MetricsRegistry,
    MetricType,
    MetricDefinition,
    GPUMetricsCollector,
    RobotMetricsCollector,
    start_metrics_server,
    create_metrics_endpoint,
    metrics_middleware,
)

from monitoring.cognitive_metrics import (
    CognitiveMetrics,
    ASRTracker,
    TTSTracker,
    LLMTracker,
    track_asr,
    track_llm,
)

__all__ = [
    # Core Registry
    "MetricsRegistry",
    "MetricType",
    "MetricDefinition",
    
    # Collectors
    "GPUMetricsCollector",
    "RobotMetricsCollector",
    "CognitiveMetrics",
    
    # Trackers
    "ASRTracker",
    "TTSTracker",
    "LLMTracker",
    
    # Decorators
    "track_asr",
    "track_llm",
    
    # Server/Integration
    "start_metrics_server",
    "create_metrics_endpoint",
    "metrics_middleware",
]

__version__ = "1.0.0"
