# Service Mesh Module

Service mesh integration supporting Istio and Linkerd for microservices communication.

## Features

- **Multi-Provider Support**: Istio and Linkerd
- **Mutual TLS**: Automatic encryption between services
- **Traffic Management**: Retries, timeouts, circuit breakers
- **Observability**: Distributed tracing, metrics collection

## Components

### MeshProvider
Supported service mesh providers:
- `ISTIO`: Full-featured service mesh
- `LINKERD`: Lightweight, Kubernetes-native

### ServiceMeshConfig
```python
from servicemesh import ServiceMeshConfig, MeshProvider

config = ServiceMeshConfig(
    provider=MeshProvider.ISTIO,
    namespace="robotics-lab",
    tracing_enabled=True,
    tracing_sample_rate=0.1,
)
```

### Traffic Policy
```python
from servicemesh.mesh_config import TrafficPolicy

policy = TrafficPolicy(
    retries=3,
    timeout_ms=5000,
    circuit_breaker_threshold=5,
    load_balancer="round_robin",
)
```

## Usage

### Configure Mesh
```python
from servicemesh import configure_mesh, ServiceMeshConfig

config = ServiceMeshConfig()
mesh_manifest = configure_mesh(config)
# Apply to Kubernetes
```

### Sidecar Injection
```python
from servicemesh import get_sidecar_config

sidecar = get_sidecar_config(
    config,
    service_name="fleet-controller",
    port=8080,
)
```

### Virtual Service (Istio)
```python
from servicemesh.mesh_config import generate_virtual_service

vs = generate_virtual_service(
    config,
    service_name="voice-service",
    hosts=["voice.robotics.local"],
)
```

## Security

### mTLS Configuration
- STRICT: Require mTLS for all connections
- PERMISSIVE: Allow both plaintext and mTLS
- DISABLE: No mTLS enforcement

## Integration

Works with:
- Kubernetes deployments
- Helm charts in `deploy/helm/`
- OpenTelemetry tracing
- Prometheus metrics
