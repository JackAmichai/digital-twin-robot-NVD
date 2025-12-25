"""
OpenTelemetry instrumentation for NVIDIA AI services.

Provides automatic tracing for:
- NVIDIA Riva ASR/TTS
- NVIDIA NIM (LLM inference)
- Triton Inference Server
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar
from contextlib import contextmanager
import asyncio

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .tracing import DistributedTracer, TracingConfig


F = TypeVar('F', bound=Callable[..., Any])


class RivaTracer:
    """
    Tracer for NVIDIA Riva ASR and TTS services.
    
    Provides detailed tracing for voice processing operations including:
    - Speech-to-text transcription
    - Text-to-speech synthesis
    - Streaming audio processing
    - Language detection
    
    Example:
        >>> riva_tracer = RivaTracer(config)
        >>> with riva_tracer.transcribe_span("en-US", 3000) as span:
        ...     result = riva_asr.transcribe(audio)
        ...     span.set_attribute("transcript", result.text)
    """
    
    def __init__(self, tracer: DistributedTracer):
        """Initialize with parent tracer."""
        self._tracer = tracer
    
    @contextmanager
    def transcribe_span(
        self,
        language: str,
        audio_duration_ms: float,
        sample_rate: int = 16000,
        streaming: bool = False
    ):
        """
        Create span for ASR transcription.
        
        Args:
            language: Language code (e.g., "en-US")
            audio_duration_ms: Audio duration in milliseconds
            sample_rate: Audio sample rate in Hz
            streaming: Whether using streaming API
        """
        attributes = {
            "riva.service": "asr",
            "riva.language": language,
            "riva.audio_duration_ms": audio_duration_ms,
            "riva.sample_rate": sample_rate,
            "riva.streaming": streaming,
            "ai.vendor": "nvidia",
            "ai.model_type": "speech_recognition"
        }
        
        with self._tracer.span(
            "riva_asr_transcribe",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            start_time = time.perf_counter()
            try:
                yield span
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("riva.processing_time_ms", duration_ms)
                span.set_attribute(
                    "riva.realtime_factor",
                    duration_ms / audio_duration_ms if audio_duration_ms > 0 else 0
                )
    
    @contextmanager
    def synthesize_span(
        self,
        language: str,
        voice_name: str,
        text_length: int,
        sample_rate: int = 22050
    ):
        """
        Create span for TTS synthesis.
        
        Args:
            language: Language code
            voice_name: Voice identifier
            text_length: Number of characters to synthesize
            sample_rate: Output audio sample rate
        """
        attributes = {
            "riva.service": "tts",
            "riva.language": language,
            "riva.voice_name": voice_name,
            "riva.text_length": text_length,
            "riva.sample_rate": sample_rate,
            "ai.vendor": "nvidia",
            "ai.model_type": "text_to_speech"
        }
        
        with self._tracer.span(
            "riva_tts_synthesize",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            start_time = time.perf_counter()
            try:
                yield span
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("riva.processing_time_ms", duration_ms)
                span.set_attribute(
                    "riva.chars_per_second",
                    text_length / (duration_ms / 1000) if duration_ms > 0 else 0
                )
    
    @contextmanager
    def streaming_session_span(self, language: str, session_type: str = "asr"):
        """
        Create span for streaming session lifecycle.
        
        Args:
            language: Language code
            session_type: Type of session ("asr" or "tts")
        """
        attributes = {
            "riva.service": f"{session_type}_streaming",
            "riva.language": language,
            "riva.session_type": "streaming",
            "ai.vendor": "nvidia"
        }
        
        with self._tracer.span(
            f"riva_{session_type}_streaming_session",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            yield span


class NIMTracer:
    """
    Tracer for NVIDIA NIM (LLM inference).
    
    Provides tracing for:
    - Text generation
    - Intent extraction
    - Embeddings generation
    - Batch inference
    
    Example:
        >>> nim_tracer = NIMTracer(tracer)
        >>> with nim_tracer.inference_span("llama-3.1-8b", 150) as span:
        ...     result = nim_client.generate(prompt)
        ...     span.set_attribute("nim.output_tokens", result.token_count)
    """
    
    def __init__(self, tracer: DistributedTracer):
        """Initialize with parent tracer."""
        self._tracer = tracer
    
    @contextmanager
    def inference_span(
        self,
        model: str,
        prompt_tokens: int,
        max_tokens: int = 256,
        temperature: float = 0.7
    ):
        """
        Create span for LLM inference.
        
        Args:
            model: Model identifier (e.g., "llama-3.1-8b")
            prompt_tokens: Number of tokens in prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        attributes = {
            "nim.model": model,
            "nim.prompt_tokens": prompt_tokens,
            "nim.max_tokens": max_tokens,
            "nim.temperature": temperature,
            "ai.vendor": "nvidia",
            "ai.model_type": "large_language_model"
        }
        
        with self._tracer.span(
            "nim_llm_inference",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            start_time = time.perf_counter()
            try:
                yield span
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("nim.latency_ms", duration_ms)
    
    @contextmanager
    def intent_extraction_span(
        self,
        model: str,
        text_length: int
    ):
        """
        Create span for intent extraction.
        
        Args:
            model: Model identifier
            text_length: Length of input text
        """
        attributes = {
            "nim.model": model,
            "nim.task": "intent_extraction",
            "nim.input_length": text_length,
            "ai.vendor": "nvidia"
        }
        
        with self._tracer.span(
            "nim_intent_extraction",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    @contextmanager
    def embeddings_span(
        self,
        model: str,
        text_count: int,
        embedding_dim: int = 4096
    ):
        """
        Create span for embeddings generation.
        
        Args:
            model: Embedding model identifier
            text_count: Number of texts to embed
            embedding_dim: Embedding dimension
        """
        attributes = {
            "nim.model": model,
            "nim.task": "embeddings",
            "nim.text_count": text_count,
            "nim.embedding_dim": embedding_dim,
            "ai.vendor": "nvidia"
        }
        
        with self._tracer.span(
            "nim_embeddings",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            yield span


class TritonTracer:
    """
    Tracer for Triton Inference Server.
    
    Provides tracing for:
    - Model inference requests
    - Batch processing
    - Model loading/unloading
    - Ensemble models
    
    Example:
        >>> triton_tracer = TritonTracer(tracer)
        >>> with triton_tracer.inference_span("yolov8", 8, "FP16") as span:
        ...     results = triton_client.infer("yolov8", inputs)
        ...     span.set_attribute("triton.detections", len(results))
    """
    
    def __init__(self, tracer: DistributedTracer):
        """Initialize with parent tracer."""
        self._tracer = tracer
    
    @contextmanager
    def inference_span(
        self,
        model_name: str,
        batch_size: int = 1,
        model_version: str = "1",
        precision: str = "FP32"
    ):
        """
        Create span for Triton inference.
        
        Args:
            model_name: Name of the model
            batch_size: Batch size for inference
            model_version: Model version string
            precision: Model precision (FP32, FP16, INT8)
        """
        attributes = {
            "triton.model_name": model_name,
            "triton.model_version": model_version,
            "triton.batch_size": batch_size,
            "triton.precision": precision,
            "ai.vendor": "nvidia",
            "ai.inference_server": "triton"
        }
        
        with self._tracer.span(
            f"triton_infer/{model_name}",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            start_time = time.perf_counter()
            try:
                yield span
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("triton.latency_ms", duration_ms)
                span.set_attribute(
                    "triton.throughput",
                    batch_size / (duration_ms / 1000) if duration_ms > 0 else 0
                )
    
    @contextmanager
    def async_inference_span(
        self,
        model_name: str,
        batch_size: int = 1,
        request_id: Optional[str] = None
    ):
        """
        Create span for async Triton inference.
        
        Args:
            model_name: Name of the model
            batch_size: Batch size
            request_id: Optional request identifier
        """
        attributes = {
            "triton.model_name": model_name,
            "triton.batch_size": batch_size,
            "triton.async": True,
            "ai.inference_server": "triton"
        }
        
        if request_id:
            attributes["triton.request_id"] = request_id
        
        with self._tracer.span(
            f"triton_async_infer/{model_name}",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            yield span
    
    @contextmanager
    def streaming_inference_span(
        self,
        model_name: str,
        stream_timeout: float = 30.0
    ):
        """
        Create span for streaming inference.
        
        Args:
            model_name: Name of the model
            stream_timeout: Stream timeout in seconds
        """
        attributes = {
            "triton.model_name": model_name,
            "triton.streaming": True,
            "triton.stream_timeout": stream_timeout,
            "ai.inference_server": "triton"
        }
        
        with self._tracer.span(
            f"triton_stream_infer/{model_name}",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            yield span
    
    @contextmanager
    def ensemble_inference_span(
        self,
        ensemble_name: str,
        component_models: list
    ):
        """
        Create span for ensemble model inference.
        
        Args:
            ensemble_name: Name of the ensemble
            component_models: List of component model names
        """
        attributes = {
            "triton.ensemble_name": ensemble_name,
            "triton.component_models": ",".join(component_models),
            "triton.component_count": len(component_models),
            "ai.inference_server": "triton"
        }
        
        with self._tracer.span(
            f"triton_ensemble/{ensemble_name}",
            kind=SpanKind.CLIENT,
            attributes=attributes
        ) as span:
            yield span


class NVIDIATracingMiddleware:
    """
    Unified tracing middleware for all NVIDIA AI services.
    
    Provides a single interface for tracing across Riva, NIM, and Triton.
    
    Example:
        >>> middleware = NVIDIATracingMiddleware(TracingConfig("cognitive-layer"))
        >>> 
        >>> # ASR tracing
        >>> with middleware.riva.transcribe_span("en-US", 3000):
        ...     transcript = asr_client.transcribe(audio)
        >>> 
        >>> # LLM tracing
        >>> with middleware.nim.inference_span("llama-3.1-8b", 100):
        ...     response = llm_client.generate(prompt)
        >>> 
        >>> # Object detection tracing
        >>> with middleware.triton.inference_span("yolov8", 16):
        ...     detections = triton_client.infer(images)
    """
    
    def __init__(self, config: TracingConfig):
        """
        Initialize the middleware.
        
        Args:
            config: Tracing configuration
        """
        self._tracer = DistributedTracer(config)
        self._tracer.initialize()
        
        self.riva = RivaTracer(self._tracer)
        self.nim = NIMTracer(self._tracer)
        self.triton = TritonTracer(self._tracer)
    
    @property
    def tracer(self) -> DistributedTracer:
        """Get the underlying tracer for custom spans."""
        return self._tracer
    
    @contextmanager
    def voice_command_pipeline(
        self,
        command_id: str,
        language: str = "en-US"
    ):
        """
        Create parent span for entire voice command pipeline.
        
        This creates a parent span that encompasses:
        1. ASR transcription
        2. NIM intent extraction
        3. Command execution
        4. TTS response
        
        Args:
            command_id: Unique command identifier
            language: Language code
        """
        attributes = {
            "pipeline": "voice_command",
            "command_id": command_id,
            "language": language,
            "services": "riva_asr,nim,riva_tts"
        }
        
        with self._tracer.span(
            "voice_command_pipeline",
            kind=SpanKind.INTERNAL,
            attributes=attributes
        ) as span:
            span.set_attribute("pipeline.start_time", time.time())
            try:
                yield span
            finally:
                span.set_attribute("pipeline.end_time", time.time())
    
    @contextmanager
    def vision_pipeline(
        self,
        frame_id: str,
        camera_id: str = "default"
    ):
        """
        Create parent span for computer vision pipeline.
        
        Args:
            frame_id: Unique frame identifier
            camera_id: Camera identifier
        """
        attributes = {
            "pipeline": "vision",
            "frame_id": frame_id,
            "camera_id": camera_id,
            "services": "triton"
        }
        
        with self._tracer.span(
            "vision_pipeline",
            kind=SpanKind.INTERNAL,
            attributes=attributes
        ) as span:
            yield span
    
    def shutdown(self) -> None:
        """Shutdown the middleware and flush pending spans."""
        self._tracer.shutdown()


def trace_nvidia_call(
    tracer: DistributedTracer,
    service: str,
    operation: str
) -> Callable[[F], F]:
    """
    Decorator for tracing NVIDIA API calls.
    
    Args:
        tracer: DistributedTracer instance
        service: Service name (riva, nim, triton)
        operation: Operation name
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.span(
                f"{service}_{operation}",
                kind=SpanKind.CLIENT,
                attributes={
                    "ai.vendor": "nvidia",
                    "ai.service": service,
                    "ai.operation": operation
                }
            ) as span:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    span.set_attribute(
                        "ai.latency_ms",
                        (time.perf_counter() - start_time) * 1000
                    )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.span(
                f"{service}_{operation}",
                kind=SpanKind.CLIENT,
                attributes={
                    "ai.vendor": "nvidia",
                    "ai.service": service,
                    "ai.operation": operation
                }
            ) as span:
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    span.set_attribute(
                        "ai.latency_ms",
                        (time.perf_counter() - start_time) * 1000
                    )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


__all__ = [
    'RivaTracer',
    'NIMTracer',
    'TritonTracer',
    'NVIDIATracingMiddleware',
    'trace_nvidia_call',
]
