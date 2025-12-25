"""Edge deployment management for distributed robotics."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4
import asyncio


class NodeStatus(Enum):
    """Edge node status."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class HardwareType(Enum):
    """Edge hardware platforms."""
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    JETSON_ORIN = "jetson_orin"
    RASPBERRY_PI = "raspberry_pi"
    INDUSTRIAL_PC = "industrial_pc"
    CUSTOM = "custom"


@dataclass
class NodeResources:
    """Node resource specifications."""
    
    cpu_cores: int = 4
    memory_gb: float = 8.0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 64.0
    has_gpu: bool = False
    gpu_model: str = ""


@dataclass
class EdgeNode:
    """Edge computing node."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    hardware_type: HardwareType = HardwareType.JETSON_ORIN
    status: NodeStatus = NodeStatus.OFFLINE
    resources: NodeResources = field(default_factory=NodeResources)
    location: str = ""
    ip_address: str = ""
    last_heartbeat: datetime | None = None
    labels: dict[str, str] = field(default_factory=dict)
    
    @property
    def is_available(self) -> bool:
        """Check if node can accept workloads."""
        return self.status == NodeStatus.ONLINE


@dataclass
class WorkloadSpec:
    """Edge workload specification."""
    
    name: str
    image: str
    replicas: int = 1
    cpu_request: float = 0.5
    memory_request_mb: int = 512
    gpu_required: bool = False
    env_vars: dict[str, str] = field(default_factory=dict)
    ports: list[int] = field(default_factory=list)
    node_selector: dict[str, str] = field(default_factory=dict)
    priority: int = 0


@dataclass
class DeployedWorkload:
    """Deployed workload instance."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    spec: WorkloadSpec = field(default_factory=WorkloadSpec)
    node_id: str = ""
    status: str = "pending"
    deployed_at: datetime = field(default_factory=datetime.utcnow)


class EdgeCluster:
    """Manage cluster of edge nodes."""
    
    def __init__(self) -> None:
        self._nodes: dict[str, EdgeNode] = {}
        self._workloads: dict[str, DeployedWorkload] = {}
    
    def register_node(self, node: EdgeNode) -> None:
        """Register edge node."""
        self._nodes[node.id] = node
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister edge node."""
        self._nodes.pop(node_id, None)
    
    def get_node(self, node_id: str) -> EdgeNode | None:
        """Get node by ID."""
        return self._nodes.get(node_id)
    
    def list_nodes(
        self,
        status: NodeStatus | None = None,
        hardware: HardwareType | None = None,
    ) -> list[EdgeNode]:
        """List nodes with optional filters."""
        nodes = list(self._nodes.values())
        if status:
            nodes = [n for n in nodes if n.status == status]
        if hardware:
            nodes = [n for n in nodes if n.hardware_type == hardware]
        return nodes
    
    def update_heartbeat(self, node_id: str) -> None:
        """Update node heartbeat timestamp."""
        node = self._nodes.get(node_id)
        if node:
            node.last_heartbeat = datetime.utcnow()
            node.status = NodeStatus.ONLINE
    
    def find_suitable_node(self, spec: WorkloadSpec) -> EdgeNode | None:
        """Find node matching workload requirements."""
        available = [n for n in self._nodes.values() if n.is_available]
        
        for node in available:
            # Check GPU requirement
            if spec.gpu_required and not node.resources.has_gpu:
                continue
            
            # Check node selector
            if spec.node_selector:
                match = all(
                    node.labels.get(k) == v
                    for k, v in spec.node_selector.items()
                )
                if not match:
                    continue
            
            # Check resources
            if spec.memory_request_mb > node.resources.memory_gb * 1024:
                continue
            
            return node
        
        return None


class EdgeDeployer:
    """Deploy workloads to edge nodes."""
    
    def __init__(self, cluster: EdgeCluster):
        self.cluster = cluster
    
    async def deploy(self, spec: WorkloadSpec) -> DeployedWorkload | None:
        """Deploy workload to suitable node."""
        node = self.cluster.find_suitable_node(spec)
        if not node:
            return None
        
        workload = DeployedWorkload(
            spec=spec,
            node_id=node.id,
            status="deploying",
        )
        
        # Simulate deployment
        await asyncio.sleep(0.1)
        workload.status = "running"
        
        self.cluster._workloads[workload.id] = workload
        return workload
    
    async def undeploy(self, workload_id: str) -> bool:
        """Remove deployed workload."""
        workload = self.cluster._workloads.get(workload_id)
        if workload:
            workload.status = "terminated"
            del self.cluster._workloads[workload_id]
            return True
        return False
    
    def get_workload(self, workload_id: str) -> DeployedWorkload | None:
        """Get workload by ID."""
        return self.cluster._workloads.get(workload_id)
    
    def list_workloads(self, node_id: str | None = None) -> list[DeployedWorkload]:
        """List deployed workloads."""
        workloads = list(self.cluster._workloads.values())
        if node_id:
            workloads = [w for w in workloads if w.node_id == node_id]
        return workloads
    
    async def scale(self, workload_id: str, replicas: int) -> bool:
        """Scale workload replicas."""
        workload = self.cluster._workloads.get(workload_id)
        if workload:
            workload.spec.replicas = replicas
            return True
        return False
