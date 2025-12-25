"""Service mesh integration with Istio/Linkerd."""

from servicemesh.mesh_config import (
    ServiceMeshConfig,
    MeshProvider,
    configure_mesh,
    get_sidecar_config,
)

__all__ = [
    "ServiceMeshConfig",
    "MeshProvider",
    "configure_mesh",
    "get_sidecar_config",
]
