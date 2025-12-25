"""
Circuit Breaker - Resilience pattern implementation.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional, Any
from enum import Enum
from functools import wraps


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    
    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls * 100


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._stats = CircuitStats()
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try recovery."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        state = self.state
        self._stats.total_calls += 1
        
        if state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                self._stats.rejected_calls += 1
                raise CircuitOpenError(f"Circuit {self.name} is HALF_OPEN, max calls reached")
            self._half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self._stats.successful_calls += 1
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._reset()
        else:
            self._failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self._stats.failed_calls += 1
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._stats.last_failure_time = self._last_failure_time
        
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        elif self._failure_count >= self.failure_threshold:
            self._trip()
    
    def _trip(self) -> None:
        """Trip the circuit to OPEN state."""
        self._state = CircuitState.OPEN
        self._success_count = 0
    
    def _reset(self) -> None:
        """Reset circuit to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    def get_stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        return self._stats
    
    def force_open(self) -> None:
        """Manually open the circuit."""
        self._trip()
    
    def force_close(self) -> None:
        """Manually close the circuit."""
        self._reset()


class CircuitOpenError(Exception):
    """Raised when circuit is open."""
    pass


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0
) -> Callable:
    """
    Decorator to wrap function with circuit breaker.
    """
    cb = CircuitBreaker(name, failure_threshold, recovery_timeout)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = cb  # Expose for testing
        return wrapper
    
    return decorator
