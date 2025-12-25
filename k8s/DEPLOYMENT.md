# Kubernetes Deployment Guide

## 🚀 Digital Twin Robotics Lab - Production Kubernetes Deployment

This guide covers deploying the Digital Twin Robotics Lab to a production Kubernetes cluster using Helm.

## Prerequisites

### Cluster Requirements
- Kubernetes 1.25+ cluster
- NVIDIA GPU Operator installed
- At least 2 GPU nodes (NVIDIA A10G, T4, or better)
- Ingress controller (NGINX recommended)
- cert-manager (for TLS certificates)
- Prometheus Operator (optional, for monitoring)

### Local Tools
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

## Quick Start

### 1. Add Helm Repositories
```bash
# Add required chart repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

### 2. Create Namespace
```bash
kubectl create namespace digital-twin
```

### 3. Configure NGC Credentials
```bash
# Create secret for NVIDIA NGC registry access
kubectl create secret docker-registry ngc-registry \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=YOUR_NGC_API_KEY \
  -n digital-twin

# Create secret for NGC API access
kubectl create secret generic ngc-credentials \
  --from-literal=apikey=YOUR_NGC_API_KEY \
  -n digital-twin
```

### 4. Install GPU Operator (if not installed)
```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true
```

### 5. Deploy the Helm Chart
```bash
cd k8s/helm/digital-twin-robotics

# Install with default values
helm install digital-twin . -n digital-twin

# Or install with custom values
helm install digital-twin . -n digital-twin -f values-production.yaml
```

## Configuration

### Essential Values

Create a `values-production.yaml` file:

```yaml
global:
  environment: production
  imageRegistry: nvcr.io
  storageClass: gp3  # AWS EBS or your storage class

# NGC API Key (or use external secrets)
secrets:
  create: true
  ngcApiKey: "YOUR_NGC_API_KEY"

# Cognitive Service
cognitive:
  replicaCount: 3
  resources:
    limits:
      cpu: "4"
      memory: 8Gi
    requests:
      cpu: "2"
      memory: 4Gi

# NVIDIA Riva
riva:
  replicaCount: 2
  resources:
    limits:
      nvidia.com/gpu: "1"
      memory: 16Gi
    requests:
      nvidia.com/gpu: "1"
      memory: 12Gi

# NVIDIA NIM
nim:
  replicaCount: 1
  resources:
    limits:
      nvidia.com/gpu: "1"
      memory: 32Gi

# Triton Inference Server
triton:
  replicaCount: 2
  resources:
    limits:
      nvidia.com/gpu: "1"
      memory: 16Gi

# Ingress
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: robotics.yourdomain.com
  tls:
    - hosts:
        - robotics.yourdomain.com
      secretName: robotics-tls

# Monitoring
prometheus:
  enabled: true
grafana:
  enabled: true
  adminPassword: "secure-password"

# Autoscaling
autoscaling:
  cognitive:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
```

### GPU Node Selection

Label your GPU nodes:
```bash
kubectl label nodes <gpu-node-name> nvidia.com/gpu.present=true
kubectl label nodes <gpu-node-name> node-type=gpu-inference
```

The Helm chart automatically schedules GPU workloads on labeled nodes.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Kubernetes Cluster                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Ingress   │  │  Cognitive  │  │    Riva     │  │     NIM     │        │
│  │  (NGINX)    │──│   Service   │──│  ASR/TTS    │  │    LLM      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │                │
│         │                │                │                │                │
│  ┌──────┴────────────────┴────────────────┴────────────────┴─────┐         │
│  │                       Service Mesh                             │         │
│  └────────────────────────────────────────────────────────────────┘         │
│         │                │                │                │                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Triton    │  │   ROS 2     │  │  Isaac Sim  │  │   Redis     │        │
│  │  Inference  │  │   Nav2      │  │ Simulation  │  │   Cache     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │                │
│  ┌──────┴────────────────┴────────────────┴────────────────┴─────┐         │
│  │                    Persistent Volumes                          │         │
│  │  (Models, Simulation Data, Logs, ROS Workspace)               │         │
│  └────────────────────────────────────────────────────────────────┘         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ Prometheus  │  │   Grafana   │  │  AlertMgr   │    Monitoring Stack     │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Horizontal Pod Autoscaling

The chart includes HPA configurations for all services:

```yaml
# Cognitive Service HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: digital-twin-cognitive
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: digital-twin-cognitive
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

View HPA status:
```bash
kubectl get hpa -n digital-twin
kubectl describe hpa digital-twin-cognitive -n digital-twin
```

## Network Policies

Zero-trust networking is enabled by default. Services can only communicate with explicitly allowed endpoints:

- Cognitive Service → Riva, NIM, Triton, Redis
- Riva → DNS only (isolated inference)
- NIM → DNS only (isolated inference)
- Triton → DNS only (isolated inference)
- Redis → Cognitive Service only

## Monitoring & Observability

### Prometheus Metrics

Access Prometheus:
```bash
kubectl port-forward svc/digital-twin-prometheus 9090:9090 -n digital-twin
```

Key metrics:
- `asr_processing_seconds` - ASR latency histogram
- `llm_inference_seconds` - LLM inference time
- `robot_command_total` - Robot commands processed
- `twin_sync_latency_ms` - Digital twin sync lag
- `component_remaining_useful_life_hours` - Predictive maintenance

### Grafana Dashboards

Access Grafana:
```bash
kubectl port-forward svc/digital-twin-grafana 3000:3000 -n digital-twin
```

Pre-built dashboards:
- Digital Twin Overview
- Voice Processing Metrics
- Robot Fleet Status
- GPU Utilization
- Predictive Maintenance

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -n digital-twin
kubectl describe pod <pod-name> -n digital-twin

# Check events
kubectl get events -n digital-twin --sort-by='.lastTimestamp'
```

### GPU Not Detected

```bash
# Check GPU operator
kubectl get pods -n gpu-operator

# Check node GPU status
kubectl describe node <node-name> | grep -A 10 nvidia

# Check GPU resource availability
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
```

### Service Connectivity

```bash
# Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup digital-twin-riva

# Test service connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- curl http://digital-twin-cognitive:8080/health
```

### View Logs

```bash
# All pods
kubectl logs -f -l app.kubernetes.io/instance=digital-twin -n digital-twin

# Specific component
kubectl logs -f -l app.kubernetes.io/component=cognitive -n digital-twin
kubectl logs -f -l app.kubernetes.io/component=riva -n digital-twin
```

## Upgrades

### Upgrade Chart
```bash
helm upgrade digital-twin . -n digital-twin -f values-production.yaml
```

### Rolling Restart
```bash
kubectl rollout restart deployment -n digital-twin
```

### Rollback
```bash
helm rollback digital-twin <revision> -n digital-twin
helm history digital-twin -n digital-twin
```

## Uninstallation

```bash
# Uninstall the release
helm uninstall digital-twin -n digital-twin

# Delete PVCs (if needed)
kubectl delete pvc -l app.kubernetes.io/instance=digital-twin -n digital-twin

# Delete namespace
kubectl delete namespace digital-twin
```

## Security Considerations

1. **Secrets Management**: Use External Secrets Operator for production
2. **Network Policies**: Enabled by default, customize as needed
3. **Pod Security**: Runs as non-root where possible
4. **RBAC**: Minimal permissions configured
5. **TLS**: Always enable in production

## Support

For issues and feature requests, visit:
https://github.com/JackAmichai/digital-twin-robot-NVD/issues
