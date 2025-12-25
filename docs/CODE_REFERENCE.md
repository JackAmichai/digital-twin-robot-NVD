# Digital Twin Robotics Lab - Code Reference Guide

## Overview

This document provides a comprehensive reference to all modules in the Digital Twin Robotics Lab platform. The platform integrates NVIDIA technologies for robotics simulation, AI inference, and voice processing.

## Quick Navigation

| Module | Purpose | Key Files | Status |
|--------|---------|-----------|--------|
| `fleet/` | Robot coordination | `fleet_manager.py` | ✅ Production |
| `voice/` | Voice processing | `asr.py`, `tts.py`, `wake_word.py` | ✅ Production |
| `simulation/` | Digital twin sync | `digital_twin.py` | ✅ Production |
| `perception/` | Object detection | `object_detector.py` | ✅ Production |
| `cognitive/` | NLU & reasoning | `intent_extractor.py`, `scene.py` | ✅ Production |
| `maintenance/` | Predictive ML | `predictor.py` | ✅ Production |
| `monitoring/` | Observability | Prometheus, Grafana, OpenTelemetry | ✅ Production |
| `profiling/` | Performance tools | `profiler.py` | ✅ Production |
| `security/` | Security scanning | `sast_scanner.py`, `dast_scanner.py` | ✅ Production |
| `backup/` | Backup & recovery | `backup_manager.py` | ✅ Production |
| `cost/` | Cost optimization | `cost_analyzer.py` | ✅ Production |
| `canary/` | Canary deployments | `canary_manager.py` | ✅ Production |
| `features/` | Feature flags | `feature_manager.py` | ✅ Production |
| `experiments/` | A/B testing | `experiment.py` | ✅ Production |
| `core/` | Dependency injection | `container.py` | ✅ Production |
| `resilience/` | Circuit breakers | `circuit_breaker.py` | ✅ Production |
| `ratelimit/` | Rate limiting | `rate_limiter.py` | ✅ Production |
| `servicemesh/` | Service mesh | `mesh_config.py` | ✅ Production |
| `logging_/` | Log aggregation | `aggregator.py` | ✅ Production |
| `secrets_/` | Secrets management | `vault_client.py` | ✅ Production |
| `tenancy/` | Multi-tenancy | `tenant_manager.py` | ✅ Production |
| `webhooks/` | Event webhooks | `webhook_manager.py` | ✅ Production |
| `plugins/` | Plugin system | `plugin_manager.py` | ✅ Production |
| `pipeline/` | Data pipelines | `data_pipeline.py` | ✅ Production |
| `registry/` | Model registry | `model_registry.py` | ✅ Production |
| `scenarios/` | Simulation scenarios | `scenario_runner.py` | ✅ Production |
| `templates/` | Robot templates | `twin_templates.py` | ✅ Production |
| `xr/` | AR/VR integration | `xr_integration.py` | ✅ Production |
| `edge/` | Edge deployment | `edge_deployment.py` | ✅ Production |
| `compliance/` | Compliance reporting | `compliance_reporter.py` | ✅ Production |
| `sdk/` | SDK generation | `sdk_generator.py` | ✅ Production |

## Module Details

### Core Robotics
- [Fleet Management](modules/FLEET.md) - Multi-robot coordination and task allocation
- [Voice Processing](modules/VOICE.md) - NVIDIA Riva ASR/TTS with wake word detection
- [Perception](modules/PERCEPTION.md) - Object detection with Triton Inference Server
- [Digital Twin Templates](modules/TEMPLATES.md) - Reusable robot configuration templates
- [Simulation Scenarios](modules/SCENARIOS.md) - Automated behavior testing framework

### AI & Machine Learning
- [Model Registry](modules/REGISTRY.md) - MLflow-style model versioning and staging
- [Data Pipeline](modules/PIPELINE.md) - ETL and streaming data processing

### Infrastructure & DevOps
- [Infrastructure Guide](modules/INFRASTRUCTURE.md) - Kubernetes, Helm, monitoring
- [Edge Deployment](modules/EDGE.md) - NVIDIA Jetson and IoT edge computing
- [Service Mesh](modules/SERVICEMESH.md) - Istio/Linkerd service mesh integration
- [Logging](modules/LOGGING.md) - ELK/Loki centralized log aggregation
- [Backup & Recovery](modules/BACKUP.md) - Automated backup and disaster recovery
- [Cost Optimization](modules/COST.md) - Resource usage analysis and recommendations
- [Canary Deployments](modules/CANARY.md) - Progressive rollout strategies

### Security & Compliance
- [Secrets Management](modules/SECRETS.md) - HashiCorp Vault integration
- [Compliance Reporting](modules/COMPLIANCE.md) - Audit trails and regulatory compliance

### Platform Features
- [Feature Flags](modules/FEATURES.md) - Runtime feature toggle system
- [A/B Testing](modules/EXPERIMENTS.md) - Experiment framework for features
- [Webhooks](modules/WEBHOOKS.md) - Event notifications to external systems
- [Plugins](modules/PLUGINS.md) - Extensible plugin architecture
- [Multi-tenancy](modules/TENANCY.md) - Tenant isolation and resource quotas
- [SDK Generation](modules/SDK.md) - Client SDK generation for Python/TypeScript

### Resilience Patterns
- [Circuit Breakers](modules/RESILIENCE.md) - Fault tolerance patterns
- [Rate Limiting](modules/RATELIMIT.md) - API throttling and request quotas

### Extended Reality
- [AR/VR Integration](modules/XR.md) - Immersive robot control and visualization
