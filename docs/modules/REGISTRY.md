# Model Registry Module

ML model versioning, staging, and deployment management.

## Features

- **Version Control**: Track model versions
- **Stage Management**: Dev → Staging → Production
- **Artifact Storage**: Persistent model storage
- **Metrics Tracking**: Performance metrics per version

## Model Stages

| Stage | Description |
|-------|-------------|
| `development` | Active development |
| `staging` | Testing/validation |
| `production` | Live deployment |
| `archived` | Deprecated versions |

## Usage

### Register Model
```python
from registry import register_model, ModelMetrics

model = register_model(
    model_name="object-detector",
    artifact_path="./trained_models/yolo_v8.pt",
    metrics=ModelMetrics(
        accuracy=0.95,
        precision=0.93,
        latency_ms=12.5,
    ),
    parameters={"epochs": 100, "batch_size": 32},
    tags={"framework": "pytorch", "dataset": "coco"},
)

print(f"Registered: {model.full_name}")  # object-detector:v1
```

### Load Model
```python
from registry import load_model, ModelStage

# Latest version
model = load_model("object-detector")

# Specific version
model = load_model("object-detector", version="v2")

# Production version
model = load_model("object-detector", stage=ModelStage.PRODUCTION)
```

### Model Registry Class
```python
from registry import ModelRegistry, ModelStage

registry = ModelRegistry("./models")

# Register
model = registry.register(
    "maintenance-predictor",
    "./models/predictor.onnx",
)

# List models
models = registry.list_models()
versions = registry.list_versions("maintenance-predictor")

# Promote to production
registry.transition_stage(
    "maintenance-predictor",
    "v1",
    ModelStage.PRODUCTION,
)
```

## Model Metrics
```python
from registry import ModelMetrics

metrics = ModelMetrics(
    accuracy=0.95,
    precision=0.93,
    recall=0.91,
    f1_score=0.92,
    latency_ms=15.0,
    custom={"mAP": 0.87, "FPS": 30},
)
```

## Integration

- Works with Triton Inference Server
- NVIDIA TensorRT optimization
- ONNX model support
- PyTorch/TensorFlow serialization
