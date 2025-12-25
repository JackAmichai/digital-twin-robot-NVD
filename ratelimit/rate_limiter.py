"""
Rate Limiter - Token bucket and sliding window algorithms.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional
from functools import wraps


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    per_user: bool = True


class RateLimiter:
    """
    Token bucket rate limiter.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, Dict] = {}
    
    def _get_bucket(self, key: str) -> Dict:
        """Get or create token bucket for key."""
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": self.config.burst_size,
                "last_update": time.time(),
            }
        return self._buckets[key]
    
    def _refill(self, bucket: Dict) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - bucket["last_update"]
        new_tokens = elapsed * self.config.requests_per_second
        bucket["tokens"] = min(self.config.burst_size, bucket["tokens"] + new_tokens)
        bucket["last_update"] = now
    
    def allow(self, key: str = "default") -> bool:
        """Check if request is allowed."""
        bucket = self._get_bucket(key)
        self._refill(bucket)
        
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False
    
    def get_wait_time(self, key: str = "default") -> float:
        """Get seconds to wait before next allowed request."""
        bucket = self._get_bucket(key)
        self._refill(bucket)
        
        if bucket["tokens"] >= 1:
            return 0.0
        
        tokens_needed = 1 - bucket["tokens"]
        return tokens_needed / self.config.requests_per_second
    
    def reset(self, key: str = "default") -> None:
        """Reset rate limit for key."""
        if key in self._buckets:
            del self._buckets[key]


class SlidingWindowLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = {}
    
    def allow(self, key: str = "default") -> bool:
        """Check if request is allowed."""
        now = time.time()
        cutoff = now - self.window_seconds
        
        if key not in self._requests:
            self._requests[key] = []
        
        # Remove old requests
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        
        if len(self._requests[key]) < self.max_requests:
            self._requests[key].append(now)
            return True
        return False


def rate_limit(rps: float = 10.0, burst: int = 20):
    """Decorator for rate limiting functions."""
    limiter = RateLimiter(RateLimitConfig(requests_per_second=rps, burst_size=burst))
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = kwargs.get("user_id", "default")
            if not limiter.allow(key):
                raise RateLimitExceeded(f"Rate limit exceeded, wait {limiter.get_wait_time(key):.2f}s")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass
