# Webhook System Module

Event-driven webhook notifications for external integrations.

## Features

- **Event Subscriptions**: Subscribe to specific events
- **HMAC Signatures**: Secure payload verification
- **Automatic Retries**: Exponential backoff on failures
- **Delivery Tracking**: Complete audit trail

## Events

| Event | Description |
|-------|-------------|
| `robot.connected` | Robot came online |
| `robot.disconnected` | Robot went offline |
| `task.created` | New task assigned |
| `task.completed` | Task finished successfully |
| `task.failed` | Task failed |
| `alert.triggered` | System alert |
| `maintenance.due` | Maintenance required |

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
