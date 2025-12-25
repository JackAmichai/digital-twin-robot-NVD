# Edge Deployment Module

Distributed computing on NVIDIA Jetson and edge devices.

## Features

- **Multi-Platform**: Jetson Nano/Xavier/Orin, RPi, industrial PCs
- **Cluster Management**: Node registration, heartbeats
- **Workload Scheduling**: Resource-aware placement
- **GPU Support**: NVIDIA GPU workload scheduling

## Hardware Platforms

| Platform | GPU | Use Case |
|----------|-----|----------|
| Jetson Nano | Maxwell | Light inference |
| Jetson Xavier | Volta | Medium workloads |
| Jetson Orin | Ampere | Heavy AI/robotics |
| Raspberry Pi | None | Sensors, control |
| Industrial PC | Optional | Factory floor |

## Usage

### Register Edge Nodes
```python
from edge import EdgeNode, EdgeCluster, HardwareType, NodeResources

cluster = EdgeCluster()

node = EdgeNode(
    name="floor-controller-01",
    hardware_type=HardwareType.JETSON_ORIN,
    ip_address="192.168.1.101",
    location="warehouse-a",
    resources=NodeResources(
        cpu_cores=12,
        memory_gb=32,
        gpu_memory_gb=32,
        has_gpu=True,
        gpu_model="Orin",
    ),
    labels={"zone": "warehouse", "role": "controller"},
)

cluster.register_node(node)
cluster.update_heartbeat(node.id)
```

### Deploy Workloads
```python
from edge import EdgeDeployer, WorkloadSpec

deployer = EdgeDeployer(cluster)

spec = WorkloadSpec(
    name="object-detector",
    image="nvcr.io/nvidia/isaac/perception:latest",
    cpu_request=2.0,
    memory_request_mb=4096,
    gpu_required=True,
    env_vars={"MODEL": "yolov8"},
    ports=[8080],
    node_selector={"zone": "warehouse"},
)

workload = await deployer.deploy(spec)
print(f"Deployed to: {workload.node_id}")
```

### Cluster Operations
```python
# List available nodes
online = cluster.list_nodes(status=NodeStatus.ONLINE)
jetson_nodes = cluster.list_nodes(hardware=HardwareType.JETSON_ORIN)

# List workloads
workloads = deployer.list_workloads(node_id=node.id)

# Scale workload
await deployer.scale(workload.id, replicas=3)
```

## Integration

- Kubernetes edge (K3s/MicroK8s)
- NVIDIA Fleet Command
- Isaac ROS deployment
- Container runtime (Docker/Podman)
