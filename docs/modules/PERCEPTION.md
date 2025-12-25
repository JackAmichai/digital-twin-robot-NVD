# Perception Module

## Overview
Real-time object detection with NVIDIA Triton.

## Files

### `object_detector.py`
YOLO-based detection.

```python
class ObjectDetector:
    def detect(image) -> List[Detection]
    def detect_batch(images) -> List[List[Detection]]
```

### `triton_client.py`
Triton Inference Server client.

```python
class TritonInferenceClient:
    def infer(model_name, inputs) -> Dict
    def get_model_metadata(model_name) -> ModelMetadata
```

## Supported Models
- YOLOv8 (object detection)
- DETR (transformer detection)
- SAM (segmentation)

## Usage

```python
from perception import ObjectDetector

detector = ObjectDetector(
    model_name="yolov8",
    triton_url="localhost:8001"
)
detections = detector.detect(camera_frame)
for det in detections:
    print(f"{det.label}: {det.confidence:.2f}")
```
