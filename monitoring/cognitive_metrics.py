"""
Metrics Integration for Cognitive Service.

This module integrates Prometheus metrics into the cognitive service,
providing instrumentation for all voice processing operations.
"""

import time
import functools
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from monitoring.prometheus_metrics import MetricsRegistry


class CognitiveMetrics:
    """
    Metrics wrapper for the Cognitive Service.
    
    Provides convenient methods for tracking ASR, TTS, wake word,
    and LLM inference metrics.
    """
    
    def __init__(self, registry: Optional[MetricsRegistry] = None):
        """Initialize with metrics registry."""
        self.registry = registry or MetricsRegistry()
    
    # =========================================================================
    # ASR METRICS
    # =========================================================================
    
    @asynccontextmanager
    async def track_asr_request(self, language: str = "en-US"):
        """
        Context manager for tracking ASR request metrics.
        
        Usage:
            async with metrics.track_asr_request(language="en-US") as tracker:
                result = await asr_client.transcribe(audio)
                tracker.set_confidence(result.confidence)
                tracker.set_audio_duration(audio.duration)
        """
        start_time = time.perf_counter()
        tracker = ASRTracker(self.registry, language)
        
        try:
            yield tracker
            # Success
            self.registry.count(
                "asr_requests_total",
                language=language,
                status="success"
            )
        except Exception as e:
            # Error
            self.registry.count(
                "asr_requests_total",
                language=language,
                status="error"
            )
            raise
        finally:
            # Record latency
            duration = time.perf_counter() - start_time
            self.registry.observe(
                "asr_processing_seconds",
                duration,
                language=language
            )
    
    def record_asr_result(
        self,
        language: str,
        latency: float,
        confidence: float,
        audio_duration: float,
        success: bool = True
    ):
        """Record ASR result metrics directly."""
        status = "success" if success else "error"
        
        self.registry.count(
            "asr_requests_total",
            language=language,
            status=status
        )
        self.registry.observe(
            "asr_processing_seconds",
            latency,
            language=language
        )
        self.registry.observe(
            "asr_confidence_score",
            confidence,
            language=language
        )
        self.registry.observe(
            "asr_audio_duration_seconds",
            audio_duration,
            language=language
        )
    
    # =========================================================================
    # WAKE WORD METRICS
    # =========================================================================
    
    def record_wake_word_detection(
        self,
        keyword: str,
        confidence: float,
        latency: float,
        is_false_positive: bool = False
    ):
        """Record wake word detection event."""
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        
        self.registry.count(
            "wake_word_detections_total",
            keyword=keyword,
            confidence_level=confidence_level
        )
        self.registry.observe(
            "wake_word_detection_latency_seconds",
            latency,
            keyword=keyword
        )
        
        if is_false_positive:
            self.registry.count(
                "wake_word_false_positives_total",
                keyword=keyword
            )
    
    # =========================================================================
    # TTS METRICS
    # =========================================================================
    
    @asynccontextmanager
    async def track_tts_request(self, voice: str, language: str = "en-US"):
        """
        Context manager for tracking TTS synthesis metrics.
        
        Usage:
            async with metrics.track_tts_request(voice="Female-1", language="en-US") as tracker:
                audio = await tts_client.synthesize(text)
                tracker.set_audio_duration(audio.duration)
        """
        start_time = time.perf_counter()
        tracker = TTSTracker(self.registry, voice, language)
        
        try:
            yield tracker
            self.registry.count(
                "tts_requests_total",
                voice=voice,
                language=language,
                status="success"
            )
        except Exception:
            self.registry.count(
                "tts_requests_total",
                voice=voice,
                language=language,
                status="error"
            )
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.registry.observe(
                "tts_synthesis_seconds",
                duration,
                voice=voice
            )
    
    def record_tts_result(
        self,
        voice: str,
        language: str,
        latency: float,
        audio_duration: float,
        success: bool = True
    ):
        """Record TTS result metrics directly."""
        status = "success" if success else "error"
        
        self.registry.count(
            "tts_requests_total",
            voice=voice,
            language=language,
            status=status
        )
        self.registry.observe(
            "tts_synthesis_seconds",
            latency,
            voice=voice
        )
        self.registry.observe(
            "tts_audio_duration_seconds",
            audio_duration,
            voice=voice
        )
    
    # =========================================================================
    # LLM METRICS
    # =========================================================================
    
    @asynccontextmanager
    async def track_llm_request(self, model: str = "llama-3.1-8b"):
        """
        Context manager for tracking LLM inference metrics.
        
        Usage:
            async with metrics.track_llm_request(model="llama-3.1-8b") as tracker:
                result = await nim_client.generate(prompt)
                tracker.set_intent(result.intent)
                tracker.set_tokens(input_tokens=50, output_tokens=100)
        """
        start_time = time.perf_counter()
        tracker = LLMTracker(self.registry, model)
        
        try:
            yield tracker
            self.registry.count(
                "llm_requests_total",
                model=model,
                intent_type=tracker.intent_type or "unknown",
                status="success"
            )
        except Exception:
            self.registry.count(
                "llm_requests_total",
                model=model,
                intent_type=tracker.intent_type or "unknown",
                status="error"
            )
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.registry.observe(
                "llm_inference_seconds",
                duration,
                model=model
            )
            
            # Calculate tokens per second
            if tracker.output_tokens and duration > 0:
                tps = tracker.output_tokens / duration
                self.registry.set_gauge(
                    "llm_tokens_per_second",
                    tps,
                    model=model
                )
    
    def record_llm_result(
        self,
        model: str,
        intent_type: str,
        latency: float,
        input_tokens: int,
        output_tokens: int,
        confidence: float,
        success: bool = True
    ):
        """Record LLM inference result metrics directly."""
        status = "success" if success else "error"
        
        self.registry.count(
            "llm_requests_total",
            model=model,
            intent_type=intent_type,
            status=status
        )
        self.registry.observe(
            "llm_inference_seconds",
            latency,
            model=model
        )
        self.registry.count(
            "llm_tokens_input_total",
            value=input_tokens,
            model=model
        )
        self.registry.count(
            "llm_tokens_output_total",
            value=output_tokens,
            model=model
        )
        self.registry.observe(
            "llm_intent_confidence",
            confidence,
            intent_type=intent_type
        )
        
        if latency > 0:
            tps = output_tokens / latency
            self.registry.set_gauge(
                "llm_tokens_per_second",
                tps,
                model=model
            )
    
    # =========================================================================
    # NOISE FILTERING METRICS
    # =========================================================================
    
    def record_noise_filter_result(
        self,
        filter_type: str,
        processing_time: float,
        snr_improvement: float
    ):
        """Record noise filtering metrics."""
        self.registry.observe(
            "noise_filter_processing_seconds",
            processing_time,
            filter_type=filter_type
        )
        self.registry.set_gauge(
            "noise_filter_snr_improvement_db",
            snr_improvement,
            filter_type=filter_type
        )


class ASRTracker:
    """Helper class for tracking ASR metrics within context."""
    
    def __init__(self, registry: MetricsRegistry, language: str):
        self.registry = registry
        self.language = language
        self.confidence: Optional[float] = None
        self.audio_duration: Optional[float] = None
    
    def set_confidence(self, confidence: float):
        """Set the ASR confidence score."""
        self.confidence = confidence
        self.registry.observe(
            "asr_confidence_score",
            confidence,
            language=self.language
        )
    
    def set_audio_duration(self, duration: float):
        """Set the audio duration in seconds."""
        self.audio_duration = duration
        self.registry.observe(
            "asr_audio_duration_seconds",
            duration,
            language=self.language
        )


class TTSTracker:
    """Helper class for tracking TTS metrics within context."""
    
    def __init__(self, registry: MetricsRegistry, voice: str, language: str):
        self.registry = registry
        self.voice = voice
        self.language = language
        self.audio_duration: Optional[float] = None
    
    def set_audio_duration(self, duration: float):
        """Set the generated audio duration in seconds."""
        self.audio_duration = duration
        self.registry.observe(
            "tts_audio_duration_seconds",
            duration,
            voice=self.voice
        )


class LLMTracker:
    """Helper class for tracking LLM metrics within context."""
    
    def __init__(self, registry: MetricsRegistry, model: str):
        self.registry = registry
        self.model = model
        self.intent_type: Optional[str] = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.confidence: Optional[float] = None
    
    def set_intent(self, intent_type: str, confidence: Optional[float] = None):
        """Set the extracted intent type and confidence."""
        self.intent_type = intent_type
        self.confidence = confidence
        
        if confidence is not None:
            self.registry.observe(
                "llm_intent_confidence",
                confidence,
                intent_type=intent_type
            )
    
    def set_tokens(self, input_tokens: int, output_tokens: int):
        """Set token counts."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        
        self.registry.count(
            "llm_tokens_input_total",
            value=input_tokens,
            model=self.model
        )
        self.registry.count(
            "llm_tokens_output_total",
            value=output_tokens,
            model=self.model
        )
    
    def set_context_length(self, length: int):
        """Set the context length for this request."""
        self.registry.observe(
            "llm_context_length",
            length,
            model=self.model
        )


# =============================================================================
# DECORATORS
# =============================================================================

def track_asr(language_arg: str = "language"):
    """
    Decorator for tracking ASR function metrics.
    
    Usage:
        @track_asr(language_arg="lang")
        async def transcribe(audio, lang="en-US"):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            language = kwargs.get(language_arg, "en-US")
            registry = MetricsRegistry()
            metrics = CognitiveMetrics(registry)
            
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                metrics.registry.count(
                    "asr_requests_total",
                    language=language,
                    status="success"
                )
                return result
            except Exception:
                metrics.registry.count(
                    "asr_requests_total",
                    language=language,
                    status="error"
                )
                raise
            finally:
                duration = time.perf_counter() - start
                metrics.registry.observe(
                    "asr_processing_seconds",
                    duration,
                    language=language
                )
        return wrapper
    return decorator


def track_llm(model: str = "llama-3.1-8b"):
    """
    Decorator for tracking LLM inference metrics.
    
    Usage:
        @track_llm(model="llama-3.1-8b")
        async def generate_response(prompt):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            registry = MetricsRegistry()
            
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                registry.count(
                    "llm_requests_total",
                    model=model,
                    intent_type="unknown",
                    status="success"
                )
                return result
            except Exception:
                registry.count(
                    "llm_requests_total",
                    model=model,
                    intent_type="unknown",
                    status="error"
                )
                raise
            finally:
                duration = time.perf_counter() - start
                registry.observe(
                    "llm_inference_seconds",
                    duration,
                    model=model
                )
        return wrapper
    return decorator
