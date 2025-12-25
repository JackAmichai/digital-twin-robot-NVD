"""Service mesh configuration for Istio and Linkerd."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import yaml


class MeshProvider(Enum):
    """Supported service mesh providers."""
    ISTIO = "istio"
    LINKERD = "linkerd"


@dataclass
class TrafficPolicy:
    """Traffic management policy configuration."""
    
    retries: int = 3
    timeout_ms: int = 5000
    circuit_breaker_threshold: int = 5
    load_balancer: str = "round_robin"  # round_robin, least_conn, random


@dataclass
class MTLSConfig:
    """Mutual TLS configuration."""
    
    enabled: bool = True
    mode: str = "STRICT"  # STRICT, PERMISSIVE, DISABLE
    min_protocol_version: str = "TLSv1_3"


@dataclass
class ServiceMeshConfig:
    """Service mesh configuration."""
    
    provider: MeshProvider = MeshProvider.ISTIO
    namespace: str = "robotics-lab"
    mtls: MTLSConfig = field(default_factory=MTLSConfig)
    traffic_policy: TrafficPolicy = field(default_factory=TrafficPolicy)
    tracing_enabled: bool = True
    tracing_sample_rate: float = 0.1
    metrics_enabled: bool = True


def configure_mesh(config: ServiceMeshConfig) -> dict[str, Any]:
    """Generate mesh configuration based on provider."""
    if config.provider == MeshProvider.ISTIO:
        return _generate_istio_config(config)
    elif config.provider == MeshProvider.LINKERD:
        return _generate_linkerd_config(config)
    raise ValueError(f"Unsupported provider: {config.provider}")


def _generate_istio_config(config: ServiceMeshConfig) -> dict[str, Any]:
    """Generate Istio-specific configuration."""
    return {
        "apiVersion": "security.istio.io/v1beta1",
        "kind": "PeerAuthentication",
        "metadata": {
            "name": "default",
            "namespace": config.namespace,
        },
        "spec": {
            "mtls": {
                "mode": config.mtls.mode,
            },
        },
    }


def _generate_linkerd_config(config: ServiceMeshConfig) -> dict[str, Any]:
    """Generate Linkerd-specific configuration."""
    return {
        "apiVersion": "policy.linkerd.io/v1beta1",
        "kind": "Server",
        "metadata": {
            "name": "default",
            "namespace": config.namespace,
        },
        "spec": {
            "podSelector": {"matchLabels": {}},
            "port": 8080,
            "proxyProtocol": "HTTP/2",
        },
    }


def get_sidecar_config(
    config: ServiceMeshConfig,
    service_name: str,
    port: int = 8080,
) -> dict[str, Any]:
    """Get sidecar injection configuration."""
    base = {
        "service": service_name,
        "port": port,
        "namespace": config.namespace,
        "tracing": {
            "enabled": config.tracing_enabled,
            "sample_rate": config.tracing_sample_rate,
        },
        "metrics": {"enabled": config.metrics_enabled},
    }
    
    if config.provider == MeshProvider.ISTIO:
        base["annotations"] = {
            "sidecar.istio.io/inject": "true",
            "proxy.istio.io/config": yaml.dump({
                "tracing": {"sampling": config.tracing_sample_rate * 100}
            }),
        }
    elif config.provider == MeshProvider.LINKERD:
        base["annotations"] = {
            "linkerd.io/inject": "enabled",
            "config.linkerd.io/trace-collector": "collector.linkerd-jaeger:55678",
        }
    
    return base


def generate_virtual_service(
    config: ServiceMeshConfig,
    service_name: str,
    hosts: list[str],
    routes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate Istio VirtualService manifest."""
    if config.provider != MeshProvider.ISTIO:
        raise ValueError("VirtualService is Istio-specific")
    
    default_routes = routes or [{
        "destination": {"host": service_name, "port": {"number": 8080}},
        "weight": 100,
    }]
    
    return {
        "apiVersion": "networking.istio.io/v1beta1",
        "kind": "VirtualService",
        "metadata": {
            "name": service_name,
            "namespace": config.namespace,
        },
        "spec": {
            "hosts": hosts,
            "http": [{
                "route": default_routes,
                "timeout": f"{config.traffic_policy.timeout_ms}ms",
                "retries": {
                    "attempts": config.traffic_policy.retries,
                    "retryOn": "5xx,reset,connect-failure",
                },
            }],
        },
    }
