"""
FastAPI middleware for automatic performance profiling.

Integrates profiling capabilities directly into the API layer for:
- Request latency tracking
- Memory monitoring per request
- Slow query detection
- Performance anomaly alerting
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from collections import deque
import statistics

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    method: str
    path: str
    status_code: int
    start_time: datetime
    end_time: datetime
    duration_ms: float
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    db_queries: int = 0
    db_time_ms: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    gpu_time_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class EndpointStats:
    """Aggregated statistics for an endpoint."""
    path: str
    method: str
    request_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.request_count if self.request_count > 0 else 0
    
    @property
    def p50_ms(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        return sorted_latencies[len(sorted_latencies) // 2]
    
    @property
    def p95_ms(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        return sorted_latencies[int(len(sorted_latencies) * 0.95)]
    
    @property
    def p99_ms(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        return sorted_latencies[int(len(sorted_latencies) * 0.99)]
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count > 0 else 0


@dataclass
class SLAConfig:
    """SLA configuration for performance monitoring."""
    latency_threshold_ms: float = 500  # P95 latency threshold
    error_rate_threshold: float = 0.01  # 1% error rate
    throughput_threshold_rps: float = 100  # Minimum requests per second
    
    # Critical paths with stricter SLAs
    critical_paths: Dict[str, float] = field(default_factory=lambda: {
        "/api/v1/robots/{robot_id}/command": 100,  # 100ms for robot commands
        "/api/v1/voice/transcribe": 200,  # 200ms for voice processing
    })


# =============================================================================
# Performance Middleware
# =============================================================================

class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking request performance metrics.
    
    Features:
    - Request latency tracking with percentiles
    - Memory usage monitoring
    - Slow request detection and logging
    - SLA violation alerting
    - Metrics export for Prometheus
    
    Example:
        app = FastAPI()
        app.add_middleware(
            PerformanceMiddleware,
            slow_request_threshold_ms=500,
            track_memory=True
        )
    """
    
    def __init__(
        self,
        app: ASGIApp,
        slow_request_threshold_ms: float = 500,
        track_memory: bool = True,
        sla_config: Optional[SLAConfig] = None,
        on_slow_request: Optional[Callable[[RequestMetrics], None]] = None,
        on_sla_violation: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        super().__init__(app)
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self.track_memory = track_memory
        self.sla_config = sla_config or SLAConfig()
        self.on_slow_request = on_slow_request
        self.on_sla_violation = on_sla_violation
        
        # Metrics storage
        self._endpoint_stats: Dict[str, EndpointStats] = {}
        self._recent_requests: deque = deque(maxlen=10000)
        self._request_counter = 0
        
        # Memory tracking
        self._psutil_available = False
        try:
            import psutil
            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            logger.warning("psutil not available, memory tracking disabled")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with performance tracking."""
        request_id = f"req_{self._request_counter}"
        self._request_counter += 1
        
        # Start timing
        start_time = datetime.utcnow()
        start_perf = time.perf_counter()
        
        # Memory before (if enabled)
        memory_before = None
        if self.track_memory and self._psutil_available:
            process = self._psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)
        
        # Store request_id in state for downstream use
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Process request
        error = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error = str(e)
            raise
        finally:
            # Calculate duration
            end_time = datetime.utcnow()
            duration_ms = (time.perf_counter() - start_perf) * 1000
            
            # Memory after
            memory_after = None
            memory_delta = None
            if self.track_memory and self._psutil_available:
                process = self._psutil.Process()
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_delta = memory_after - memory_before if memory_before else None
            
            # Create metrics object
            metrics = RequestMetrics(
                request_id=request_id,
                method=request.method,
                path=self._normalize_path(request.url.path),
                status_code=status_code,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_delta,
                error=error
            )
            
            # Update stats
            self._update_stats(metrics)
            
            # Check for slow request
            if duration_ms > self.slow_request_threshold_ms:
                self._handle_slow_request(metrics)
            
            # Check SLA violations
            self._check_sla_violations(metrics)
            
            # Add response headers with timing info
            if hasattr(response, 'headers'):
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time-MS"] = str(round(duration_ms, 2))
        
        return response
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path by replacing IDs with placeholders."""
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}',
            path
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace robot-xxx, task-xxx patterns
        path = re.sub(r'/(robot|task|scenario|conv)-[a-zA-Z0-9]+', r'/{\1_id}', path)
        
        return path
    
    def _update_stats(self, metrics: RequestMetrics) -> None:
        """Update endpoint statistics."""
        key = f"{metrics.method}:{metrics.path}"
        
        if key not in self._endpoint_stats:
            self._endpoint_stats[key] = EndpointStats(
                path=metrics.path,
                method=metrics.method
            )
        
        stats = self._endpoint_stats[key]
        stats.request_count += 1
        stats.total_duration_ms += metrics.duration_ms
        stats.min_duration_ms = min(stats.min_duration_ms, metrics.duration_ms)
        stats.max_duration_ms = max(stats.max_duration_ms, metrics.duration_ms)
        stats.latencies.append(metrics.duration_ms)
        
        if metrics.status_code >= 400:
            stats.error_count += 1
        
        # Store recent request
        self._recent_requests.append(metrics)
    
    def _handle_slow_request(self, metrics: RequestMetrics) -> None:
        """Handle slow request detection."""
        logger.warning(
            f"Slow request detected: {metrics.method} {metrics.path} "
            f"took {metrics.duration_ms:.2f}ms (threshold: {self.slow_request_threshold_ms}ms)"
        )
        
        if self.on_slow_request:
            try:
                self.on_slow_request(metrics)
            except Exception as e:
                logger.error(f"Error in slow request callback: {e}")
    
    def _check_sla_violations(self, metrics: RequestMetrics) -> None:
        """Check for SLA violations."""
        # Check critical path SLA
        for path_pattern, threshold_ms in self.sla_config.critical_paths.items():
            if self._path_matches(metrics.path, path_pattern):
                if metrics.duration_ms > threshold_ms:
                    self._report_sla_violation(
                        "latency",
                        {
                            "path": metrics.path,
                            "duration_ms": metrics.duration_ms,
                            "threshold_ms": threshold_ms
                        }
                    )
                break
    
    def _path_matches(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern with wildcards."""
        import re
        regex = pattern.replace("{", "(?P<").replace("}", ">[^/]+)")
        return bool(re.match(f"^{regex}$", path))
    
    def _report_sla_violation(self, violation_type: str, details: Dict[str, Any]) -> None:
        """Report SLA violation."""
        logger.error(f"SLA violation ({violation_type}): {details}")
        
        if self.on_sla_violation:
            try:
                self.on_sla_violation(violation_type, details)
            except Exception as e:
                logger.error(f"Error in SLA violation callback: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "endpoints": {
                key: {
                    "path": stats.path,
                    "method": stats.method,
                    "request_count": stats.request_count,
                    "error_count": stats.error_count,
                    "error_rate": stats.error_rate,
                    "avg_duration_ms": stats.avg_duration_ms,
                    "min_duration_ms": stats.min_duration_ms if stats.min_duration_ms != float('inf') else 0,
                    "max_duration_ms": stats.max_duration_ms,
                    "p50_ms": stats.p50_ms,
                    "p95_ms": stats.p95_ms,
                    "p99_ms": stats.p99_ms
                }
                for key, stats in self._endpoint_stats.items()
            },
            "total_requests": sum(s.request_count for s in self._endpoint_stats.values()),
            "total_errors": sum(s.error_count for s in self._endpoint_stats.values())
        }
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Request count
        lines.append("# HELP http_requests_total Total HTTP requests")
        lines.append("# TYPE http_requests_total counter")
        for key, stats in self._endpoint_stats.items():
            lines.append(
                f'http_requests_total{{method="{stats.method}",path="{stats.path}"}} {stats.request_count}'
            )
        
        # Request duration
        lines.append("# HELP http_request_duration_ms HTTP request duration in milliseconds")
        lines.append("# TYPE http_request_duration_ms summary")
        for key, stats in self._endpoint_stats.items():
            lines.append(
                f'http_request_duration_ms{{method="{stats.method}",path="{stats.path}",quantile="0.5"}} {stats.p50_ms}'
            )
            lines.append(
                f'http_request_duration_ms{{method="{stats.method}",path="{stats.path}",quantile="0.95"}} {stats.p95_ms}'
            )
            lines.append(
                f'http_request_duration_ms{{method="{stats.method}",path="{stats.path}",quantile="0.99"}} {stats.p99_ms}'
            )
        
        return "\n".join(lines)


# =============================================================================
# Request Context for Distributed Tracing
# =============================================================================

class RequestContext:
    """
    Context manager for tracking request-scoped metrics.
    
    Use this to track custom metrics within a request:
    
        async def handler(request: Request):
            ctx = RequestContext.get(request)
            
            with ctx.time("database_query"):
                result = await db.query()
            
            ctx.increment("cache_hits")
    """
    
    _contexts: Dict[str, "RequestContext"] = {}
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.timings: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.custom_metrics: Dict[str, Any] = {}
    
    @classmethod
    def get(cls, request: Request) -> "RequestContext":
        """Get or create context for request."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        if request_id not in cls._contexts:
            cls._contexts[request_id] = cls(request_id)
        
        return cls._contexts[request_id]
    
    @classmethod
    def cleanup(cls, request_id: str) -> None:
        """Clean up context after request completes."""
        cls._contexts.pop(request_id, None)
    
    @asynccontextmanager
    async def time(self, operation: str):
        """Time an async operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(duration_ms)
    
    def time_sync(self, operation: str):
        """Context manager for timing synchronous operations."""
        class SyncTimer:
            def __init__(timer_self, ctx, op):
                timer_self.ctx = ctx
                timer_self.op = op
                timer_self.start = None
            
            def __enter__(timer_self):
                timer_self.start = time.perf_counter()
                return timer_self
            
            def __exit__(timer_self, *args):
                duration_ms = (time.perf_counter() - timer_self.start) * 1000
                if timer_self.op not in timer_self.ctx.timings:
                    timer_self.ctx.timings[timer_self.op] = []
                timer_self.ctx.timings[timer_self.op].append(duration_ms)
        
        return SyncTimer(self, operation)
    
    def increment(self, counter: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[counter] = self.counters.get(counter, 0) + value
    
    def set_metric(self, name: str, value: Any) -> None:
        """Set a custom metric."""
        self.custom_metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "request_id": self.request_id,
            "timings": {
                op: {
                    "count": len(times),
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times) if times else 0,
                    "max_ms": max(times) if times else 0
                }
                for op, times in self.timings.items()
            },
            "counters": self.counters,
            "custom_metrics": self.custom_metrics
        }


# =============================================================================
# Performance Dashboard Endpoint
# =============================================================================

def create_performance_router(middleware: PerformanceMiddleware):
    """Create FastAPI router for performance metrics endpoints."""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/performance", tags=["Performance"])
    
    @router.get("/metrics")
    async def get_metrics():
        """Get current performance metrics."""
        return middleware.get_metrics()
    
    @router.get("/metrics/prometheus")
    async def get_prometheus_metrics():
        """Get metrics in Prometheus format."""
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            middleware.get_prometheus_metrics(),
            media_type="text/plain"
        )
    
    @router.get("/endpoints")
    async def get_endpoint_stats():
        """Get per-endpoint statistics."""
        return {
            key: {
                "request_count": stats.request_count,
                "error_rate": f"{stats.error_rate * 100:.2f}%",
                "avg_ms": round(stats.avg_duration_ms, 2),
                "p50_ms": round(stats.p50_ms, 2),
                "p95_ms": round(stats.p95_ms, 2),
                "p99_ms": round(stats.p99_ms, 2)
            }
            for key, stats in middleware._endpoint_stats.items()
        }
    
    @router.get("/slow-requests")
    async def get_slow_requests(threshold_ms: float = 500, limit: int = 100):
        """Get recent slow requests."""
        slow_requests = [
            {
                "request_id": r.request_id,
                "method": r.method,
                "path": r.path,
                "duration_ms": round(r.duration_ms, 2),
                "status_code": r.status_code,
                "timestamp": r.start_time.isoformat()
            }
            for r in middleware._recent_requests
            if r.duration_ms > threshold_ms
        ][-limit:]
        
        return {"slow_requests": slow_requests}
    
    return router
