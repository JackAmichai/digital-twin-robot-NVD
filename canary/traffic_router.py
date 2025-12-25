"""
Traffic Router - Route traffic between canary and stable.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import random


@dataclass
class RouteDecision:
    """Traffic routing decision."""
    target: str  # "canary" or "stable"
    version: str
    weight: int


class TrafficRouter:
    """
    Routes traffic between canary and stable versions.
    """
    
    def __init__(self):
        self.routes: Dict[str, Dict[str, Any]] = {}
    
    def set_weights(
        self,
        service: str,
        canary_weight: int,
        canary_version: str,
        stable_version: str
    ) -> None:
        """Set traffic weights for a service."""
        self.routes[service] = {
            "canary_weight": canary_weight,
            "stable_weight": 100 - canary_weight,
            "canary_version": canary_version,
            "stable_version": stable_version,
        }
    
    def route(self, service: str, request_id: str = "") -> RouteDecision:
        """Route a request to canary or stable."""
        if service not in self.routes:
            return RouteDecision(
                target="stable",
                version="default",
                weight=100,
            )
        
        route = self.routes[service]
        
        # Weighted random selection
        if random.randint(1, 100) <= route["canary_weight"]:
            return RouteDecision(
                target="canary",
                version=route["canary_version"],
                weight=route["canary_weight"],
            )
        
        return RouteDecision(
            target="stable",
            version=route["stable_version"],
            weight=route["stable_weight"],
        )
    
    def get_endpoint(self, service: str, decision: RouteDecision) -> str:
        """Get endpoint URL for routing decision."""
        # Example: service-canary.namespace.svc or service-stable.namespace.svc
        suffix = "canary" if decision.target == "canary" else "stable"
        return f"{service}-{suffix}.default.svc.cluster.local"
    
    def remove_route(self, service: str) -> None:
        """Remove routing rules for a service."""
        if service in self.routes:
            del self.routes[service]
    
    def get_weights(self, service: str) -> Optional[Dict[str, Any]]:
        """Get current weights for a service."""
        return self.routes.get(service)
