#!/usr/bin/env python3
"""
System Orchestrator
Main entry point that coordinates all layers of the Digital Twin Robotics Lab
"""

import asyncio
import json
import logging
import signal
import sys
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('orchestrator')


class ServiceStatus(Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class ServiceInfo:
    name: str
    status: ServiceStatus
    last_heartbeat: float = 0.0
    error_count: int = 0


class SystemOrchestrator:
    """
    Coordinates the three-layer architecture:
    - Cognitive Layer (ASR + Intent)
    - Control Layer (ROS 2 + Nav2)
    - Simulation Layer (Isaac Sim)
    """
    
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis: Optional[redis.Redis] = None
        self.running = False
        
        # Service tracking
        self.services = {
            'cognitive': ServiceInfo('cognitive', ServiceStatus.STARTING),
            'ros2': ServiceInfo('ros2', ServiceStatus.STARTING),
            'isaac_sim': ServiceInfo('isaac_sim', ServiceStatus.STARTING),
        }
        
        # Callbacks
        self.on_status_change: Optional[Callable] = None
        
        # Channels
        self.channels = {
            'commands': 'robot_commands',
            'status': 'system_status',
            'heartbeat': 'service_heartbeat',
            'errors': 'system_errors',
        }
    
    async def connect(self):
        """Connect to Redis."""
        logger.info(f"Connecting to Redis: {self.redis_url}")
        self.redis = await redis.from_url(self.redis_url)
        await self.redis.ping()
        logger.info("Redis connected successfully")
    
    async def start(self):
        """Start the orchestrator."""
        self.running = True
        logger.info("=" * 60)
        logger.info("  Digital Twin Robotics Lab - System Orchestrator")
        logger.info("=" * 60)
        
        await self.connect()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._command_logger()),
            asyncio.create_task(self._status_publisher()),
        ]
        
        logger.info("Orchestrator started. Monitoring services...")
        
        # Wait for shutdown
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator shutting down...")
    
    async def stop(self):
        """Stop the orchestrator."""
        self.running = False
        if self.redis:
            await self.redis.close()
        logger.info("Orchestrator stopped")
    
    async def _heartbeat_monitor(self):
        """Monitor service heartbeats."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.channels['heartbeat'])
        
        while self.running:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0
                )
                if message and message['type'] == 'message':
                    data = json.loads(message['data'])
                    service_name = data.get('service')
                    if service_name in self.services:
                        self.services[service_name].status = ServiceStatus.HEALTHY
                        self.services[service_name].last_heartbeat = asyncio.get_event_loop().time()
                        self.services[service_name].error_count = 0
            except asyncio.TimeoutError:
                # Check for stale heartbeats
                current_time = asyncio.get_event_loop().time()
                for service in self.services.values():
                    if service.status == ServiceStatus.HEALTHY:
                        if current_time - service.last_heartbeat > 10.0:
                            service.status = ServiceStatus.DEGRADED
                            logger.warning(f"Service {service.name} heartbeat stale")
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
    
    async def _command_logger(self):
        """Log all robot commands for debugging."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.channels['commands'])
        
        while self.running:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0
                )
                if message and message['type'] == 'message':
                    data = json.loads(message['data'])
                    logger.info(f"COMMAND: {data.get('action', 'unknown')} → {data.get('target', 'N/A')}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"Command logger error: {e}")
    
    async def _status_publisher(self):
        """Publish system status periodically."""
        while self.running:
            status = {
                'services': {
                    name: {
                        'status': svc.status.value,
                        'error_count': svc.error_count,
                    }
                    for name, svc in self.services.items()
                },
                'healthy': all(
                    s.status in [ServiceStatus.HEALTHY, ServiceStatus.STARTING]
                    for s in self.services.values()
                ),
            }
            
            await self.redis.publish(
                self.channels['status'],
                json.dumps(status)
            )
            
            # Log status summary
            statuses = [f"{n}:{s.status.value[:3]}" for n, s in self.services.items()]
            logger.debug(f"Status: {' | '.join(statuses)}")
            
            await asyncio.sleep(5.0)
    
    async def send_command(self, action: str, target: str = None, coordinates: list = None):
        """Send a command to the robot."""
        command = {
            'action': action,
            'target': target,
            'coordinates': coordinates,
            'source': 'orchestrator',
        }
        await self.redis.publish(self.channels['commands'], json.dumps(command))
        logger.info(f"Sent command: {action} → {target}")
    
    def get_status(self) -> dict:
        """Get current system status."""
        return {
            name: svc.status.value
            for name, svc in self.services.items()
        }


async def main():
    orchestrator = SystemOrchestrator()
    
    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(orchestrator.stop()))
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        pass
    finally:
        await orchestrator.stop()


if __name__ == '__main__':
    asyncio.run(main())
