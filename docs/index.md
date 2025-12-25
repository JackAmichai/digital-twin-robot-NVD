# Digital Twin Robotics Lab Documentation

Welcome to the Digital Twin Robotics Lab - a comprehensive NVIDIA-powered platform for robotics simulation, AI inference, and fleet management.

## Overview

The Digital Twin Robotics Lab provides enterprise-grade infrastructure for:

- **Robot Fleet Management**: Coordinate multiple robots with intelligent task allocation
- **Voice Interaction**: Natural language control using NVIDIA Riva ASR/TTS
- **AI Perception**: Real-time object detection with Triton Inference Server
- **Digital Twin Simulation**: Bi-directional sync with NVIDIA Isaac Sim
- **Predictive Maintenance**: ML-based failure prediction and scheduling

## Quick Start

```bash
# Clone repository
git clone https://github.com/JackAmichai/digital-twin-robot-NVD.git
cd digital-twin-robot-NVD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run the platform
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Digital Twin Robotics Platform                      │
├───────────────────┬───────────────────┬─────────────────────────────────┤
│   Voice Layer     │   Perception      │   Cognitive Layer               │
│   ─────────────   │   ──────────      │   ───────────────               │
│   • Riva ASR/TTS  │   • Object Det.   │   • Intent Extraction           │
│   • Wake Word     │   • Triton Inf.   │   • Scene Understanding         │
│   • Noise Filter  │   • CUDA Accel.   │   • Context Management          │
│   • Multi-lang    │   • TensorRT      │   • Spatial Reasoning           │
├───────────────────┴───────────────────┴─────────────────────────────────┤
│                        Fleet Management Layer                            │
│   • Robot Registry    • Task Allocation    • State Synchronization      │
│   • Health Monitoring • Predictive Maint.  • Digital Twin Sync          │
├─────────────────────────────────────────────────────────────────────────┤
│                        Infrastructure Layer                              │
│   • Kubernetes        • Service Mesh      • Secrets Management          │
│   • Monitoring        • Log Aggregation   • CI/CD Pipelines             │
│   • Edge Deployment   • Multi-tenancy     • Compliance Reporting        │
└─────────────────────────────────────────────────────────────────────────┘
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

Full OpenAPI documentation available at [api/openapi/robotics-api.yaml](api/openapi/robotics-api.yaml).

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/robots` | GET | List all registered robots |
| `/robots/{id}` | GET | Get robot details |
| `/robots/{id}/command` | POST | Send command to robot |
| `/tasks` | POST | Create new task |
| `/tasks/{id}/assign` | PUT | Assign task to robot |
| `/voice/recognize` | POST | Speech-to-text recognition |
| `/voice/synthesize` | POST | Text-to-speech synthesis |
| `/twin/sync` | POST | Synchronize digital twin state |
| `/health` | GET | Platform health check |

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# NVIDIA Services
NVIDIA_RIVA_URL=localhost:50051          # Riva ASR/TTS server
TRITON_URL=localhost:8001                # Triton Inference Server
CUDA_VISIBLE_DEVICES=0                   # GPU device selection

# Data Storage
REDIS_URL=redis://localhost:6379         # Redis for caching/state
DATABASE_URL=postgresql://localhost/dtlab # PostgreSQL database

# Security
VAULT_ADDR=http://localhost:8200         # HashiCorp Vault address
VAULT_TOKEN=hvs.xxxxx                    # Vault authentication token
JWT_SECRET=your-secret-key               # JWT signing secret

# Monitoring
PROMETHEUS_GATEWAY=localhost:9091        # Prometheus pushgateway
JAEGER_ENDPOINT=http://localhost:14268   # Distributed tracing

# Feature Flags
FEATURE_FLAGS_ENABLED=true               # Enable feature flag system
EXPERIMENT_TRAFFIC_PERCENT=10            # A/B testing traffic allocation

# Multi-tenancy
DEFAULT_TENANT=default                   # Default tenant identifier
TENANT_ISOLATION=strict                  # Isolation level (strict/soft)
```

### Kubernetes Configuration

For production deployments, use Helm:

```bash
# Add required repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Deploy the platform
helm install dtlab ./helm/charts/robotics-platform \
  --namespace digital-twin \
  --create-namespace \
  --values ./helm/values-production.yaml
```

## Development

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (for inference)
- Docker & Docker Compose (for services)
- kubectl & Helm (for Kubernetes deployment)

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v --cov=.

# Integration tests (requires Docker services)
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/ -v

# Performance tests
locust -f tests/load/locustfile.py --headless -u 100 -r 10

# Security scanning
bandit -r . -ll
safety check
```

### Code Quality

```bash
# Formatting
black . --line-length 100
isort . --profile black

# Linting
flake8 . --max-line-length 100
mypy . --strict

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Deployment Options

| Environment | Description | Documentation |
|-------------|-------------|---------------|
| **Local Development** | Docker Compose setup | [DEVELOPMENT.md](DEVELOPMENT.md) |
| **Kubernetes** | Production Helm charts | [modules/INFRASTRUCTURE.md](modules/INFRASTRUCTURE.md) |
| **Edge/Jetson** | Embedded deployment | [modules/EDGE.md](modules/EDGE.md) |
| **Cloud** | Multi-cloud support | [modules/CLOUD.md](modules/CLOUD.md) |

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Write tests** for your changes
4. **Ensure** all tests pass: `pytest tests/`
5. **Run** code quality checks: `black . && isort . && flake8 .`
6. **Commit** with conventional commits: `git commit -m "feat: add amazing feature"`
7. **Push** to your fork: `git push origin feature/amazing-feature`
8. **Open** a Pull Request

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## Support

- **Documentation**: [Full Docs](https://jackamichai.github.io/digital-twin-robot-NVD)
- **Issues**: [GitHub Issues](https://github.com/JackAmichai/digital-twin-robot-NVD/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JackAmichai/digital-twin-robot-NVD/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ using NVIDIA Technologies*
