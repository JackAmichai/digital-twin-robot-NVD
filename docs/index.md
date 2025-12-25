# Digital Twin Robotics Lab Documentation

Welcome to the Digital Twin Robotics Lab documentation.

## Quick Start

```bash
# Clone repository
git clone https://github.com/JackAmichai/digital-twin-robot-NVD.git
cd digital-twin-robot-NVD

# Install dependencies
pip install -r requirements.txt

# Run the platform
python -m uvicorn main:app --reload
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Digital Twin Platform                     │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Voice Layer    │  Perception     │  Cognitive Layer        │
│  - Riva ASR/TTS │  - Object Det   │  - Intent Extraction    │
│  - Wake Word    │  - Triton Inf   │  - Scene Understanding  │
│  - Noise Filter │  - CUDA Accel   │  - Context Management   │
├─────────────────┴─────────────────┴─────────────────────────┤
│                    Fleet Management                          │
│  - Robot Registry  - Task Allocation  - State Sync          │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure                            │
│  - Kubernetes  - Monitoring  - CI/CD  - Security            │
└─────────────────────────────────────────────────────────────┘
```

## Modules

### Core Robotics
- [Fleet Management](modules/FLEET.md) - Robot registration and task allocation
- [Voice Processing](modules/VOICE.md) - NVIDIA Riva ASR/TTS integration
- [Perception](modules/PERCEPTION.md) - Object detection with Triton
- [Digital Twin Templates](modules/TEMPLATES.md) - Robot configuration templates
- [Simulation Scenarios](modules/SCENARIOS.md) - Behavior testing

### AI & ML
- [Cognitive](cognitive/) - Intent extraction and scene understanding
- [Model Registry](modules/REGISTRY.md) - MLflow-style model versioning
- [Maintenance](maintenance/) - Predictive maintenance ML

### Infrastructure
- [Infrastructure Guide](modules/INFRASTRUCTURE.md) - Kubernetes, Helm, monitoring
- [Edge Deployment](modules/EDGE.md) - Jetson and IoT deployment
- [Service Mesh](modules/SERVICEMESH.md) - Istio/Linkerd integration
- [Logging](modules/LOGGING.md) - ELK/Loki log aggregation

### DevOps
- [CI/CD Pipelines](.github/workflows/) - GitHub Actions workflows
- [Canary Deployments](modules/CANARY.md) - Progressive rollouts
- [Chaos Engineering](chaos/) - Resilience testing
- [Backup & Recovery](modules/BACKUP.md) - Data protection

### Security & Compliance
- [Security Scanning](security/) - SAST/DAST analysis
- [Secrets Management](modules/SECRETS.md) - HashiCorp Vault
- [Compliance](modules/COMPLIANCE.md) - Audit and reporting

### Platform Features
- [Feature Flags](modules/FEATURES.md) - Runtime feature toggling
- [A/B Testing](modules/EXPERIMENTS.md) - Experiment framework
- [Webhooks](modules/WEBHOOKS.md) - Event notifications
- [Plugins](modules/PLUGINS.md) - Extensibility system
- [Multi-tenancy](modules/TENANCY.md) - Tenant isolation

### Resilience
- [Circuit Breakers](modules/RESILIENCE.md) - Fault tolerance
- [Rate Limiting](modules/RATELIMIT.md) - Request throttling

### Extended Reality
- [AR/VR Integration](modules/XR.md) - Immersive robot control

## API Reference

See [OpenAPI Documentation](api/openapi/robotics-api.yaml) for full API specs.

## Configuration

Environment variables:
```bash
NVIDIA_RIVA_URL=localhost:50051
TRITON_URL=localhost:8001
REDIS_URL=redis://localhost:6379
VAULT_ADDR=http://localhost:8200
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Run tests: `pytest tests/`
4. Submit pull request

## License

MIT License - See LICENSE file
