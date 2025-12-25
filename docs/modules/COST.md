# Cost Optimization Module

## Overview
Resource usage and cloud cost monitoring.

## Files

### `resource_monitor.py`
Track CPU, GPU, memory usage.

```python
class ResourceMonitor:
    def get_current_usage() -> ResourceUsage
    def get_k8s_resource_usage(namespace) -> Dict
    def get_average_usage(last_n) -> Dict
```

**Metrics Tracked:**
- CPU percent
- Memory percent/GB
- GPU utilization
- GPU memory
- Disk usage

### `cost_analyzer.py`
Calculate and optimize cloud costs.

```python
class CostAnalyzer:
    def calculate_costs(resources) -> List[ResourceCost]
    def get_optimization_recommendations(costs) -> List[Dict]
    def estimate_gpu_costs(gpu_hours_per_day, days) -> Dict
    def generate_report(costs) -> Dict
```

**Supported Providers:**
- AWS (p3.2xlarge GPU pricing)
- GCP (n1-standard + T4)
- Azure (NC6s_v3)

## Usage

```python
from cost import ResourceMonitor, CostAnalyzer

# Monitor resources
monitor = ResourceMonitor()
usage = monitor.get_current_usage()
print(f"CPU: {usage.cpu_percent}%, GPU: {usage.gpu_percent}%")

# Analyze costs
analyzer = CostAnalyzer(provider=CloudProvider.AWS)
costs = analyzer.calculate_costs(resources)
report = analyzer.generate_report(costs)
print(f"Monthly cost: ${report['total_monthly_cost']:.2f}")
```

## Recommendations
The analyzer provides:
- Underutilized resource detection (<30% usage)
- Reserved instance suggestions
- Spot instance savings estimates
