# Rate Limiting Module

## Overview
API throttling and quotas using token bucket algorithm.

## Files

### `rate_limiter.py`
Token bucket and sliding window rate limiters.

```python
class RateLimiter:
    def allow(key) -> bool
    def get_wait_time(key) -> float
    def reset(key)

class SlidingWindowLimiter:
    def allow(key) -> bool
```

## Usage

```python
from ratelimit import RateLimiter, RateLimitConfig, rate_limit

# Token bucket limiter
limiter = RateLimiter(RateLimitConfig(
    requests_per_second=10.0,
    burst_size=20,
))

if limiter.allow(user_id):
    process_request()
else:
    wait_time = limiter.get_wait_time(user_id)
    return {"error": f"Rate limited, retry in {wait_time}s"}

# Decorator
@rate_limit(rps=5.0, burst=10)
def api_endpoint(user_id: str):
    return {"status": "ok"}
```

## Algorithms
- **Token Bucket**: Allows bursts up to bucket size
- **Sliding Window**: Fixed requests per time window
