# Log Aggregation Module

Centralized logging with ELK Stack and Grafana Loki support.

## Features

- **Multiple Backends**: Elasticsearch, Loki, Console
- **Structured Logging**: JSON-formatted logs
- **Async Batching**: Efficient log shipping
- **Automatic Flushing**: Time and size-based

## Components

### LogBackend
```python
from logging_ import LogBackend

# Supported backends
LogBackend.ELASTICSEARCH  # ELK Stack
LogBackend.LOKI           # Grafana Loki
LogBackend.CONSOLE        # Local output
```

### StructuredLogger
```python
from logging_ import StructuredLogger, LogConfig

config = LogConfig(
    backend=LogBackend.ELASTICSEARCH,
    endpoint="http://elasticsearch:9200",
    index_prefix="robotics-lab",
)

logger = StructuredLogger("fleet-controller", config)
logger.info("Robot assigned", robot_id="R001", task_id="T123")
logger.error("Connection failed", error="timeout", retries=3)
```

### LogAggregator
```python
from logging_ import LogAggregator, LogConfig

config = LogConfig(
    backend=LogBackend.LOKI,
    endpoint="http://loki:3100",
    batch_size=100,
    flush_interval_seconds=5.0,
)

aggregator = LogAggregator(config)
await aggregator.start()

await aggregator.log({
    "level": "INFO",
    "message": "Task completed",
    "robot_id": "R001",
})

await aggregator.stop()
```

## Configuration

### Elasticsearch
```python
config = LogConfig(
    backend=LogBackend.ELASTICSEARCH,
    endpoint="http://elasticsearch:9200",
    index_prefix="robotics",  # Creates: robotics-YYYY.MM.DD
)
```

### Loki
```python
config = LogConfig(
    backend=LogBackend.LOKI,
    endpoint="http://loki:3100",
    extra_labels={"env": "prod", "service": "fleet"},
)
```

## Integration

- Works with Grafana dashboards
- OpenTelemetry trace correlation
- Kubernetes pod logging
- Prometheus metrics correlation
