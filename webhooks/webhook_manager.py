"""Webhook management and delivery system."""

import asyncio
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4
import aiohttp


class WebhookEvent(Enum):
    """Supported webhook event types."""
    ROBOT_CONNECTED = "robot.connected"
    ROBOT_DISCONNECTED = "robot.disconnected"
    TASK_CREATED = "task.created"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    ALERT_TRIGGERED = "alert.triggered"
    MAINTENANCE_DUE = "maintenance.due"


@dataclass
class Webhook:
    """Webhook subscription."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    url: str = ""
    events: list[WebhookEvent] = field(default_factory=list)
    secret: str = field(default_factory=lambda: str(uuid4()))
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WebhookDelivery:
    """Webhook delivery record."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    webhook_id: str = ""
    event: WebhookEvent = WebhookEvent.ROBOT_CONNECTED
    payload: dict[str, Any] = field(default_factory=dict)
    status_code: int = 0
    response_body: str = ""
    delivered_at: datetime = field(default_factory=datetime.utcnow)
    success: bool = False
    attempts: int = 0


class WebhookManager:
    """Manage webhook subscriptions and deliveries."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 5.0):
        self._webhooks: dict[str, Webhook] = {}
        self._deliveries: list[WebhookDelivery] = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def register(
        self,
        url: str,
        events: list[WebhookEvent],
        secret: str | None = None,
    ) -> Webhook:
        """Register new webhook subscription."""
        webhook = Webhook(
            url=url,
            events=events,
            secret=secret or str(uuid4()),
        )
        self._webhooks[webhook.id] = webhook
        return webhook
    
    def unregister(self, webhook_id: str) -> bool:
        """Unregister webhook."""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            return True
        return False
    
    def get_webhook(self, webhook_id: str) -> Webhook | None:
        """Get webhook by ID."""
        return self._webhooks.get(webhook_id)
    
    def list_webhooks(self, event: WebhookEvent | None = None) -> list[Webhook]:
        """List webhooks, optionally filtered by event."""
        webhooks = [w for w in self._webhooks.values() if w.active]
        if event:
            webhooks = [w for w in webhooks if event in w.events]
        return webhooks
    
    async def dispatch(
        self,
        event: WebhookEvent,
        payload: dict[str, Any],
    ) -> list[WebhookDelivery]:
        """Dispatch event to all subscribed webhooks."""
        webhooks = self.list_webhooks(event)
        
        tasks = [
            self._deliver(webhook, event, payload)
            for webhook in webhooks
        ]
        
        return await asyncio.gather(*tasks)
    
    async def _deliver(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        payload: dict[str, Any],
    ) -> WebhookDelivery:
        """Deliver webhook with retries."""
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event=event,
            payload=payload,
        )
        
        full_payload = {
            "event": event.value,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": payload,
        }
        
        body = json.dumps(full_payload)
        signature = self._sign_payload(body, webhook.secret)
        
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Event": event.value,
            "X-Webhook-ID": webhook.id,
        }
        
        for attempt in range(self.max_retries):
            delivery.attempts = attempt + 1
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook.url,
                        data=body,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        delivery.status_code = resp.status
                        delivery.response_body = await resp.text()
                        delivery.success = 200 <= resp.status < 300
                        
                        if delivery.success:
                            break
            except Exception as e:
                delivery.response_body = str(e)
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        delivery.delivered_at = datetime.utcnow()
        self._deliveries.append(delivery)
        return delivery
    
    def _sign_payload(self, payload: str, secret: str) -> str:
        """Sign payload with HMAC-SHA256."""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
    
    def get_deliveries(
        self,
        webhook_id: str | None = None,
        limit: int = 100,
    ) -> list[WebhookDelivery]:
        """Get recent deliveries."""
        deliveries = self._deliveries
        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]
        return sorted(deliveries, key=lambda d: d.delivered_at, reverse=True)[:limit]
