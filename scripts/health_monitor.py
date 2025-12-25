#!/usr/bin/env python3
"""
Health Monitor - Service health checking for Digital Twin Robotics Lab.

Monitors:
- Redis connectivity
- Cognitive service status
- ROS 2 node availability
- Isaac Sim bridge status

Provides health dashboard and alerting.
"""

import asyncio
import json
import time
import socket
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceHealth:
    """Health status for a single service."""
    name: str
    status: str  # 'healthy', 'unhealthy', 'unknown'
    latency_ms: float
    last_check: str
    details: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class HealthMonitor:
    """Monitor health of all system services."""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        riva_host: str = "localhost",
        riva_port: int = 50051,
        nim_host: str = "localhost",
        nim_port: int = 8000
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.riva_host = riva_host
        self.riva_port = riva_port
        self.nim_host = nim_host
        self.nim_port = nim_port
        
        self.health_status: Dict[str, ServiceHealth] = {}
        self.redis_client = None
    
    async def initialize(self):
        """Initialize connections."""
        try:
            import redis.asyncio as redis
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            logger.info("Health monitor initialized")
        except ImportError:
            logger.warning("redis-py not installed, using mock")
            self.redis_client = None
    
    def _check_port(self, host: str, port: int, timeout: float = 2.0) -> tuple[bool, float]:
        """Check if a port is open and return latency."""
        start = time.perf_counter()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            latency = (time.perf_counter() - start) * 1000
            return result == 0, latency
        except Exception as e:
            logger.error(f"Port check failed: {e}")
            return False, 0.0
    
    async def check_redis(self) -> ServiceHealth:
        """Check Redis health."""
        start = time.perf_counter()
        try:
            if self.redis_client:
                await self.redis_client.ping()
                latency = (time.perf_counter() - start) * 1000
                return ServiceHealth(
                    name="Redis",
                    status="healthy",
                    latency_ms=round(latency, 2),
                    last_check=datetime.now().isoformat(),
                    details="PONG received"
                )
            else:
                # Fallback to port check
                is_open, latency = self._check_port(self.redis_host, self.redis_port)
                return ServiceHealth(
                    name="Redis",
                    status="healthy" if is_open else "unhealthy",
                    latency_ms=round(latency, 2),
                    last_check=datetime.now().isoformat(),
                    details="Port open" if is_open else "Port closed"
                )
        except Exception as e:
            return ServiceHealth(
                name="Redis",
                status="unhealthy",
                latency_ms=0,
                last_check=datetime.now().isoformat(),
                details=str(e)
            )
    
    async def check_riva(self) -> ServiceHealth:
        """Check NVIDIA Riva ASR health."""
        is_open, latency = self._check_port(self.riva_host, self.riva_port)
        return ServiceHealth(
            name="Riva ASR",
            status="healthy" if is_open else "unhealthy",
            latency_ms=round(latency, 2),
            last_check=datetime.now().isoformat(),
            details="gRPC port accessible" if is_open else "gRPC port not accessible"
        )
    
    async def check_nim(self) -> ServiceHealth:
        """Check NVIDIA NIM LLM health."""
        is_open, latency = self._check_port(self.nim_host, self.nim_port)
        
        # Also try HTTP health check
        if is_open:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    start = time.perf_counter()
                    async with session.get(
                        f"http://{self.nim_host}:{self.nim_port}/v1/health/ready",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        latency = (time.perf_counter() - start) * 1000
                        if resp.status == 200:
                            return ServiceHealth(
                                name="NVIDIA NIM",
                                status="healthy",
                                latency_ms=round(latency, 2),
                                last_check=datetime.now().isoformat(),
                                details="LLM ready"
                            )
            except ImportError:
                pass
            except Exception:
                pass
        
        return ServiceHealth(
            name="NVIDIA NIM",
            status="healthy" if is_open else "unhealthy",
            latency_ms=round(latency, 2),
            last_check=datetime.now().isoformat(),
            details="HTTP port accessible" if is_open else "Service not accessible"
        )
    
    async def check_ros2_nodes(self) -> ServiceHealth:
        """Check ROS 2 nodes via ros2 CLI."""
        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["ros2", "node", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            latency = (time.perf_counter() - start) * 1000
            
            if result.returncode == 0:
                nodes = [n for n in result.stdout.strip().split('\n') if n]
                expected = ['cognitive_bridge', 'robot_state_publisher']
                found = sum(1 for n in expected if any(n in node for node in nodes))
                
                return ServiceHealth(
                    name="ROS 2 Nodes",
                    status="healthy" if found >= 1 else "unhealthy",
                    latency_ms=round(latency, 2),
                    last_check=datetime.now().isoformat(),
                    details=f"Found {len(nodes)} nodes: {', '.join(nodes[:5])}"
                )
            else:
                return ServiceHealth(
                    name="ROS 2 Nodes",
                    status="unhealthy",
                    latency_ms=round(latency, 2),
                    last_check=datetime.now().isoformat(),
                    details=f"ros2 error: {result.stderr[:100]}"
                )
        except FileNotFoundError:
            return ServiceHealth(
                name="ROS 2 Nodes",
                status="unknown",
                latency_ms=0,
                last_check=datetime.now().isoformat(),
                details="ros2 CLI not available"
            )
        except subprocess.TimeoutExpired:
            return ServiceHealth(
                name="ROS 2 Nodes",
                status="unhealthy",
                latency_ms=5000,
                last_check=datetime.now().isoformat(),
                details="ros2 command timed out"
            )
        except Exception as e:
            return ServiceHealth(
                name="ROS 2 Nodes",
                status="unhealthy",
                latency_ms=0,
                last_check=datetime.now().isoformat(),
                details=str(e)
            )
    
    async def check_isaac_sim(self) -> ServiceHealth:
        """Check Isaac Sim ROS 2 bridge via topic check."""
        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            latency = (time.perf_counter() - start) * 1000
            
            if result.returncode == 0:
                topics = result.stdout.strip().split('\n')
                isaac_topics = [t for t in topics if 'clock' in t or 'scan' in t or 'camera' in t]
                
                return ServiceHealth(
                    name="Isaac Sim Bridge",
                    status="healthy" if isaac_topics else "unknown",
                    latency_ms=round(latency, 2),
                    last_check=datetime.now().isoformat(),
                    details=f"Found {len(isaac_topics)} sim topics" if isaac_topics else "No sim topics yet"
                )
            else:
                return ServiceHealth(
                    name="Isaac Sim Bridge",
                    status="unknown",
                    latency_ms=round(latency, 2),
                    last_check=datetime.now().isoformat(),
                    details="Could not list topics"
                )
        except Exception as e:
            return ServiceHealth(
                name="Isaac Sim Bridge",
                status="unknown",
                latency_ms=0,
                last_check=datetime.now().isoformat(),
                details=str(e)
            )
    
    async def run_all_checks(self) -> Dict[str, ServiceHealth]:
        """Run all health checks concurrently."""
        checks = await asyncio.gather(
            self.check_redis(),
            self.check_riva(),
            self.check_nim(),
            self.check_ros2_nodes(),
            self.check_isaac_sim(),
            return_exceptions=True
        )
        
        services = ["redis", "riva", "nim", "ros2", "isaac"]
        for service, result in zip(services, checks):
            if isinstance(result, Exception):
                self.health_status[service] = ServiceHealth(
                    name=service,
                    status="error",
                    latency_ms=0,
                    last_check=datetime.now().isoformat(),
                    details=str(result)
                )
            else:
                self.health_status[service] = result
        
        return self.health_status
    
    def get_overall_status(self) -> str:
        """Get overall system health status."""
        if not self.health_status:
            return "unknown"
        
        statuses = [h.status for h in self.health_status.values()]
        if all(s == "healthy" for s in statuses):
            return "healthy"
        elif any(s == "unhealthy" for s in statuses):
            return "degraded"
        else:
            return "partial"
    
    def print_dashboard(self):
        """Print health dashboard to console."""
        print("\n" + "=" * 60)
        print("         DIGITAL TWIN ROBOTICS LAB - HEALTH DASHBOARD")
        print("=" * 60)
        print(f" Overall Status: {self.get_overall_status().upper()}")
        print(f" Timestamp: {datetime.now().isoformat()}")
        print("-" * 60)
        
        for service, health in self.health_status.items():
            status_icon = {
                "healthy": "✅",
                "unhealthy": "❌",
                "unknown": "❓",
                "error": "⚠️"
            }.get(health.status, "❓")
            
            print(f" {status_icon} {health.name:<20} | {health.latency_ms:>6.1f}ms | {health.details}")
        
        print("=" * 60 + "\n")
    
    async def publish_health(self):
        """Publish health status to Redis."""
        if self.redis_client:
            try:
                health_data = {
                    "overall": self.get_overall_status(),
                    "timestamp": datetime.now().isoformat(),
                    "services": {k: v.to_dict() for k, v in self.health_status.items()}
                }
                await self.redis_client.publish(
                    "system:health",
                    json.dumps(health_data)
                )
                await self.redis_client.set(
                    "system:health:latest",
                    json.dumps(health_data)
                )
            except Exception as e:
                logger.error(f"Failed to publish health: {e}")


async def main():
    """Main monitoring loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Monitor")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()
    
    monitor = HealthMonitor()
    await monitor.initialize()
    
    logger.info("Starting health monitoring...")
    
    try:
        while True:
            await monitor.run_all_checks()
            monitor.print_dashboard()
            await monitor.publish_health()
            
            if args.once:
                break
            
            await asyncio.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Health monitor stopped")


if __name__ == "__main__":
    asyncio.run(main())
