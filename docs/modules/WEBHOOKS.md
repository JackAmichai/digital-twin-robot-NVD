# Webhook System Module

Comprehensive event-driven webhook system for real-time notifications to external services and integrations.

## Overview

The webhook system enables real-time event-driven communication between the Digital Twin Robotics Lab platform and external services. When significant events occur (robot connections, task completions, alerts), the system automatically notifies all subscribed endpoints with secure, signed payloads.

## Features

- **Event-Driven Architecture**: Subscribe to specific event types for targeted notifications
- **Secure Delivery**: HMAC-SHA256 payload signatures for authentication and integrity verification
- **Automatic Retries**: Exponential backoff retry mechanism (3 attempts by default)
- **Delivery Tracking**: Complete audit trail of all webhook deliveries with status codes
- **Concurrent Dispatch**: Parallel delivery to multiple subscribers for low latency
- **Configurable Timeouts**: 30-second default timeout with customizable settings

## Events

| Event | Description | Typical Use Case |
|-------|-------------|------------------|
| `robot.connected` | Robot came online | Inventory updates, status dashboards |
| `robot.disconnected` | Robot went offline | Incident response, alerting |
| `task.created` | New task assigned | Workflow orchestration, logging |
| `task.completed` | Task finished successfully | Reporting, chained workflows |
| `task.failed` | Task failed | Error handling, alerting |
| `alert.triggered` | System alert raised | Monitoring integration (PagerDuty, OpsGenie) |
| `maintenance.due` | Maintenance required | Scheduling systems, technician dispatch |

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Event Source   │────▶│  WebhookManager  │────▶│ External System  │
│  (Robot, Task)   │     │   (Dispatcher)   │     │  (Slack, PD)     │
└──────────────────┘     └────────┬─────────┘     └──────────────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Delivery Store  │
                         │ (Audit History)  │
                         └──────────────────┘
```

## Usage

### Register Webhook
```python
from webhooks import WebhookManager, WebhookEvent

manager = WebhookManager()

webhook = manager.register(
    url="https://api.example.com/webhook",
    events=[
        WebhookEvent.ROBOT_CONNECTED,
        WebhookEvent.TASK_COMPLETED,
    ],
)

print(f"Secret: {webhook.secret}")
```

### Dispatch Event
```python
deliveries = await manager.dispatch(
    WebhookEvent.TASK_COMPLETED,
    payload={
        "task_id": "T123",
        "robot_id": "R001",
        "duration_seconds": 45.2,
    },
)
```

### Verify Signature
```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature, expected)
```

## Payload Format
```json
{
    "event": "task.completed",
    "timestamp": "2024-01-15T10:30:00Z",
    "data": {
        "task_id": "T123",
        "robot_id": "R001"
    }
}
```

## Headers

| Header | Description |
|--------|-------------|
| `X-Webhook-Signature` | HMAC-SHA256 signature |
| `X-Webhook-Event` | Event type |
| `X-Webhook-ID` | Webhook subscription ID |

## Retry Policy

- Max retries: 3
- Backoff: 5s, 10s, 15s
- Success: 2xx status codes
