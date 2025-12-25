# Infrastructure Module

## Kubernetes (`deploy/helm/`)
Helm charts with GPU support.

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

## Monitoring (`monitoring/`)

### Prometheus
- Robot fleet metrics
- Inference latency
- Alert rules

### Grafana
- Fleet dashboard
- Robot telemetry
- Voice pipeline stats

### OpenTelemetry
- Distributed tracing
- Span correlation

## Profiling (`profiling/`)

### CPU Profiling
```python
from profiling import CPUProfiler
profiler = CPUProfiler()
profiler.start()
# ... code ...
result = profiler.stop()
```

### GPU Profiling
```python
from profiling import GPUProfiler
profiler = GPUProfiler()
# ... code with CUDA ...
report = profiler.get_memory_usage()
```

## Security (`security/`)

### SAST
- Bandit (Python)
- Semgrep (multi-language)

### DAST
- OWASP ZAP

### Dependencies
- pip-audit
- Safety
- Trivy (containers)
