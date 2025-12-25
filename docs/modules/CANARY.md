# Canary Deployments Module

## Overview
Progressive rollout strategies for safe deployments.

## Files

### `canary_manager.py`
Manages canary deployments with auto promotion/rollback.

```python
class CanaryManager:
    def create_canary(config: CanaryConfig) -> CanaryDeployment
    def start_rollout(name) -> CanaryDeployment
    def advance_rollout(name) -> CanaryDeployment
    def complete_rollout(name) -> CanaryDeployment
    def rollback(name, reason) -> CanaryDeployment
    def pause(name) -> CanaryDeployment
```

**Rollout Statuses:**
- PENDING, IN_PROGRESS, PAUSED
- COMPLETED, ROLLED_BACK, FAILED

**Health Checks:**
- Success rate threshold (default 99%)
- Latency threshold (default 500ms)

### `traffic_router.py`
Routes traffic between versions.

```python
class TrafficRouter:
    def set_weights(service, canary_weight, canary_ver, stable_ver)
    def route(service, request_id) -> RouteDecision
    def get_endpoint(service, decision) -> str
```

## Usage

```python
from canary import CanaryManager, CanaryConfig, TrafficRouter

# Configure canary
config = CanaryConfig(
    name="fleet-api-v2",
    target_service="fleet-api",
    canary_version="v2.0.0",
    stable_version="v1.9.0",
    initial_weight=5,
    increment=10,
)

# Start rollout
manager = CanaryManager()
manager.create_canary(config)
manager.start_rollout("fleet-api-v2")

# Route traffic
router = TrafficRouter()
router.set_weights("fleet-api", 5, "v2.0.0", "v1.9.0")
decision = router.route("fleet-api")
endpoint = router.get_endpoint("fleet-api", decision)
```

## Rollout Flow
1. Start at 5% canary traffic
2. Monitor metrics for 5 minutes
3. Advance by 10% if healthy
4. Repeat until 100% or rollback
