# Circuit Breakers Module

## Overview
Resilience patterns for fault-tolerant microservices.

## Files

### `circuit_breaker.py`
Circuit breaker pattern implementation.

```python
class CircuitBreaker:
    def __init__(name, failure_threshold=5, recovery_timeout=30)
    def call(func, *args, **kwargs) -> Any
    def get_stats() -> CircuitStats
    def force_open()
    def force_close()
```

**States:**
- `CLOSED`: Normal, requests pass through
- `OPEN`: Failing, requests rejected immediately
- `HALF_OPEN`: Testing recovery with limited calls

**Decorator:**
```python
@circuit_breaker(name="api", failure_threshold=5)
def call_external_api():
    ...
```

## Usage

```python
from resilience import CircuitBreaker, circuit_breaker

# Manual usage
cb = CircuitBreaker(
    name="triton-inference",
    failure_threshold=5,
    recovery_timeout=30.0,
)

try:
    result = cb.call(triton_client.infer, model, inputs)
except CircuitOpenError:
    # Use fallback or cached result
    result = get_cached_result()

# Decorator usage
@circuit_breaker(name="external-api", failure_threshold=3)
def call_external_service(data):
    return requests.post("https://api.example.com", json=data)

# Check stats
stats = cb.get_stats()
print(f"Failure rate: {stats.failure_rate}%")
```

## Flow
1. Calls pass through when CLOSED
2. After N failures, trips to OPEN
3. After timeout, transitions to HALF_OPEN
4. If test calls succeed, resets to CLOSED
5. If test calls fail, trips back to OPEN
