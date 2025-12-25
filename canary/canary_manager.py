"""
Canary Manager - Progressive deployment rollouts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class RolloutStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CanaryConfig:
    """Canary deployment configuration."""
    name: str
    target_service: str
    canary_version: str
    stable_version: str
    initial_weight: int = 5  # Start with 5% traffic
    increment: int = 10
    max_weight: int = 100
    interval_seconds: int = 300  # 5 min between increments
    success_threshold: float = 99.0  # Required success rate %
    latency_threshold_ms: float = 500


@dataclass
class CanaryMetrics:
    """Metrics for canary analysis."""
    success_rate: float
    avg_latency_ms: float
    error_count: int
    request_count: int
    p99_latency_ms: float = 0.0


@dataclass
class CanaryDeployment:
    """Active canary deployment state."""
    config: CanaryConfig
    status: RolloutStatus = RolloutStatus.PENDING
    current_weight: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics_history: List[CanaryMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "status": self.status.value,
            "current_weight": self.current_weight,
            "canary_version": self.config.canary_version,
            "stable_version": self.config.stable_version,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }


class CanaryManager:
    """
    Manages canary deployments with automatic promotion/rollback.
    """
    
    def __init__(self, metrics_provider=None):
        self.deployments: Dict[str, CanaryDeployment] = {}
        self.metrics_provider = metrics_provider
    
    def create_canary(self, config: CanaryConfig) -> CanaryDeployment:
        """Create a new canary deployment."""
        deployment = CanaryDeployment(
            config=config,
            status=RolloutStatus.PENDING,
            current_weight=0,
        )
        self.deployments[config.name] = deployment
        return deployment
    
    def start_rollout(self, name: str) -> CanaryDeployment:
        """Start canary rollout."""
        if name not in self.deployments:
            raise ValueError(f"Canary not found: {name}")
        
        deployment = self.deployments[name]
        deployment.status = RolloutStatus.IN_PROGRESS
        deployment.current_weight = deployment.config.initial_weight
        deployment.started_at = datetime.now()
        
        return deployment
    
    def advance_rollout(self, name: str) -> CanaryDeployment:
        """Advance canary to next weight increment."""
        deployment = self.deployments[name]
        
        if deployment.status != RolloutStatus.IN_PROGRESS:
            raise ValueError(f"Canary not in progress: {name}")
        
        # Check metrics before advancing
        metrics = self._get_current_metrics(deployment)
        deployment.metrics_history.append(metrics)
        
        if not self._check_health(deployment, metrics):
            return self.rollback(name, reason="Health check failed")
        
        # Advance weight
        new_weight = deployment.current_weight + deployment.config.increment
        if new_weight >= deployment.config.max_weight:
            return self.complete_rollout(name)
        
        deployment.current_weight = new_weight
        return deployment
    
    def complete_rollout(self, name: str) -> CanaryDeployment:
        """Complete canary and promote to 100%."""
        deployment = self.deployments[name]
        deployment.status = RolloutStatus.COMPLETED
        deployment.current_weight = 100
        deployment.completed_at = datetime.now()
        return deployment
    
    def rollback(self, name: str, reason: str = "") -> CanaryDeployment:
        """Rollback canary to stable version."""
        deployment = self.deployments[name]
        deployment.status = RolloutStatus.ROLLED_BACK
        deployment.current_weight = 0
        deployment.completed_at = datetime.now()
        return deployment
    
    def pause(self, name: str) -> CanaryDeployment:
        """Pause canary rollout."""
        deployment = self.deployments[name]
        deployment.status = RolloutStatus.PAUSED
        return deployment
    
    def _get_current_metrics(self, deployment: CanaryDeployment) -> CanaryMetrics:
        """Get current canary metrics."""
        if self.metrics_provider:
            return self.metrics_provider.get_metrics(deployment.config.target_service)
        
        # Default mock metrics
        return CanaryMetrics(
            success_rate=99.5,
            avg_latency_ms=150,
            error_count=5,
            request_count=1000,
        )
    
    def _check_health(self, deployment: CanaryDeployment, metrics: CanaryMetrics) -> bool:
        """Check if canary is healthy."""
        config = deployment.config
        
        if metrics.success_rate < config.success_threshold:
            return False
        if metrics.avg_latency_ms > config.latency_threshold_ms:
            return False
        
        return True
    
    def get_status(self, name: str) -> Dict[str, Any]:
        """Get canary deployment status."""
        if name not in self.deployments:
            return {"error": "Not found"}
        return self.deployments[name].to_dict()
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all canary deployments."""
        return [d.to_dict() for d in self.deployments.values()]
