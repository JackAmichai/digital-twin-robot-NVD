#!/usr/bin/env python3
"""
Object Detection Pipeline with NVIDIA TAO and Triton

Production-grade object detection system for warehouse robotics:
- NVIDIA TAO trained models (YOLO, SSD, FasterRCNN, DINO)
- Triton Inference Server deployment
- Real-time camera stream processing
- Multi-camera synchronization
- 3D object localization with depth
- ROS 2 integration for navigation

Supported Object Classes:
- Warehouse items: pallets, boxes, shelves, racks
- Safety: people, forklifts, warning signs, spills
- Navigation: doors, lanes, zones, obstacles
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable, Union
from enum import Enum
import threading
import queue
import time
import logging
import json
from collections import deque
from pathlib import Path

# TensorRT / Triton (conditional imports)
try:
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient
    from tritonclient.utils import InferenceServerException
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# OpenCV for image processing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2
    from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
    from geometry_msgs.msg import Pose, Point, Quaternion
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class ModelArchitecture(Enum):
    """Supported model architectures."""
    YOLOV4 = "yolov4"
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"
    SSD = "ssd"
    FASTERRCNN = "fasterrcnn"
    RETINANET = "retinanet"
    DINO = "dino"           # NVIDIA DINO transformer
    DETR = "detr"           # DEtection TRansformer


class ObjectClass(Enum):
    """Object classes for warehouse detection."""
    # Warehouse items
    PALLET = "pallet"
    BOX = "box"
    SHELF = "shelf"
    RACK = "rack"
    CONTAINER = "container"
    CART = "cart"
    
    # Safety
    PERSON = "person"
    FORKLIFT = "forklift"
    WARNING_SIGN = "warning_sign"
    SPILL = "spill"
    OBSTACLE = "obstacle"
    
    # Navigation
    DOOR = "door"
    LANE = "lane"
    ZONE_MARKER = "zone_marker"
    QR_CODE = "qr_code"
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """2D bounding box."""
    x_min: float      # Top-left x (pixels or normalized 0-1)
    y_min: float      # Top-left y
    x_max: float      # Bottom-right x
    y_max: float      # Bottom-right y
    normalized: bool = False  # True if coordinates are 0-1 normalized
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_pixels(self, img_width: int, img_height: int) -> 'BoundingBox':
        """Convert normalized coords to pixel coords."""
        if not self.normalized:
            return self
        return BoundingBox(
            x_min=self.x_min * img_width,
            y_min=self.y_min * img_height,
            x_max=self.x_max * img_width,
            y_max=self.y_max * img_height,
            normalized=False
        )
    
    def to_normalized(self, img_width: int, img_height: int) -> 'BoundingBox':
        """Convert pixel coords to normalized 0-1."""
        if self.normalized:
            return self
        return BoundingBox(
            x_min=self.x_min / img_width,
            y_min=self.y_min / img_height,
            x_max=self.x_max / img_width,
            y_max=self.y_max / img_height,
            normalized=True
        )
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box."""
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
            
        intersection = (x_max - x_min) * (y_max - y_min)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            'x_min': self.x_min,
            'y_min': self.y_min,
            'x_max': self.x_max,
            'y_max': self.y_max,
            'normalized': self.normalized
        }


@dataclass
class BoundingBox3D:
    """3D bounding box in world coordinates."""
    center_x: float
    center_y: float
    center_z: float
    width: float      # X dimension
    depth: float      # Y dimension
    height: float     # Z dimension
    yaw: float = 0.0  # Rotation around Z axis (radians)
    
    @property
    def corners(self) -> List[Tuple[float, float, float]]:
        """Get 8 corner points of the 3D box."""
        hw, hd, hh = self.width/2, self.depth/2, self.height/2
        cos_yaw, sin_yaw = np.cos(self.yaw), np.sin(self.yaw)
        
        corners_local = [
            (-hw, -hd, -hh), (hw, -hd, -hh), (hw, hd, -hh), (-hw, hd, -hh),
            (-hw, -hd, hh), (hw, -hd, hh), (hw, hd, hh), (-hw, hd, hh)
        ]
        
        corners_world = []
        for x, y, z in corners_local:
            x_rot = x * cos_yaw - y * sin_yaw + self.center_x
            y_rot = x * sin_yaw + y * cos_yaw + self.center_y
            z_world = z + self.center_z
            corners_world.append((x_rot, y_rot, z_world))
            
        return corners_world
    
    def to_dict(self) -> dict:
        return {
            'center': [self.center_x, self.center_y, self.center_z],
            'dimensions': [self.width, self.depth, self.height],
            'yaw': self.yaw
        }


@dataclass
class Detection:
    """Single object detection result."""
    object_class: ObjectClass
    confidence: float
    bbox_2d: BoundingBox
    bbox_3d: Optional[BoundingBox3D] = None
    track_id: Optional[int] = None          # For tracking
    velocity: Optional[Tuple[float, float, float]] = None  # m/s
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    camera_id: str = ""
    
    def to_dict(self) -> dict:
        return {
            'class': self.object_class.value,
            'confidence': self.confidence,
            'bbox_2d': self.bbox_2d.to_dict(),
            'bbox_3d': self.bbox_3d.to_dict() if self.bbox_3d else None,
            'track_id': self.track_id,
            'velocity': self.velocity,
            'attributes': self.attributes,
            'timestamp': self.timestamp,
            'camera_id': self.camera_id
        }


@dataclass
class DetectionResult:
    """Complete detection result for an image frame."""
    detections: List[Detection]
    image_width: int
    image_height: int
    timestamp: float
    camera_id: str
    inference_time_ms: float
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    
    @property
    def total_time_ms(self) -> float:
        return self.preprocessing_time_ms + self.inference_time_ms + self.postprocessing_time_ms
    
    def filter_by_confidence(self, min_confidence: float) -> 'DetectionResult':
        """Return result with only high-confidence detections."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(
            detections=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            timestamp=self.timestamp,
            camera_id=self.camera_id,
            inference_time_ms=self.inference_time_ms,
            preprocessing_time_ms=self.preprocessing_time_ms,
            postprocessing_time_ms=self.postprocessing_time_ms
        )
    
    def filter_by_class(self, classes: List[ObjectClass]) -> 'DetectionResult':
        """Return result with only specified classes."""
        filtered = [d for d in self.detections if d.object_class in classes]
        return DetectionResult(
            detections=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            timestamp=self.timestamp,
            camera_id=self.camera_id,
            inference_time_ms=self.inference_time_ms,
            preprocessing_time_ms=self.preprocessing_time_ms,
            postprocessing_time_ms=self.postprocessing_time_ms
        )
    
    def to_dict(self) -> dict:
        return {
            'detections': [d.to_dict() for d in self.detections],
            'image_size': [self.image_width, self.image_height],
            'timestamp': self.timestamp,
            'camera_id': self.camera_id,
            'timing': {
                'preprocessing_ms': self.preprocessing_time_ms,
                'inference_ms': self.inference_time_ms,
                'postprocessing_ms': self.postprocessing_time_ms,
                'total_ms': self.total_time_ms
            }
        }


@dataclass
class ModelConfig:
    """Configuration for a detection model."""
    name: str
    architecture: ModelArchitecture
    input_width: int = 640
    input_height: int = 640
    input_channels: int = 3
    
    # Classes
    class_names: List[str] = field(default_factory=list)
    class_mapping: Dict[int, ObjectClass] = field(default_factory=dict)
    
    # Preprocessing
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    input_format: str = "NCHW"  # NCHW or NHWC
    
    # Postprocessing
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 100
    
    # Triton settings
    triton_model_name: str = ""
    triton_model_version: str = "1"


# =============================================================================
# Image Preprocessing
# =============================================================================

class ImagePreprocessor:
    """
    Preprocesses images for neural network inference.
    Handles resizing, normalization, and format conversion.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: BGR image (H, W, 3) uint8
            
        Returns:
            Preprocessed tensor ready for inference
        """
        start = time.time()
        
        # Resize
        resized = cv2.resize(
            image, 
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply mean/std normalization
        mean = np.array(self.config.mean, dtype=np.float32)
        std = np.array(self.config.std, dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Change format if needed
        if self.config.input_format == "NCHW":
            # HWC -> CHW
            normalized = normalized.transpose(2, 0, 1)
            
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched.astype(np.float32)
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocess a batch of images."""
        processed = [self.preprocess(img)[0] for img in images]
        return np.stack(processed, axis=0)


# =============================================================================
# Postprocessing / NMS
# =============================================================================

class DetectionPostprocessor:
    """
    Postprocesses model outputs into Detection objects.
    Handles NMS, coordinate conversion, and class mapping.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def process_yolo_output(
        self,
        output: np.ndarray,
        img_width: int,
        img_height: int
    ) -> List[Detection]:
        """
        Process YOLO-style output.
        
        Args:
            output: Model output tensor
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # YOLO output format varies by version
        # Typical: [batch, num_boxes, 5 + num_classes]
        # 5 = [x_center, y_center, width, height, confidence]
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dim
            
        boxes = []
        scores = []
        class_ids = []
        
        for detection in output:
            if len(detection) < 5:
                continue
                
            confidence = detection[4]
            if confidence < self.config.confidence_threshold:
                continue
                
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            final_score = confidence * class_score
            if final_score < self.config.confidence_threshold:
                continue
                
            x_center, y_center, width, height = detection[:4]
            
            # Convert to corner format
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(float(final_score))
            class_ids.append(int(class_id))
            
        if not boxes:
            return []
            
        # Apply NMS
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        keep_indices = self._nms(boxes, scores, self.config.nms_threshold)
        
        # Create Detection objects
        for idx in keep_indices[:self.config.max_detections]:
            box = boxes[idx]
            
            # Map class ID to ObjectClass
            object_class = self.config.class_mapping.get(
                class_ids[idx], ObjectClass.UNKNOWN
            )
            
            bbox = BoundingBox(
                x_min=box[0] * img_width / self.config.input_width,
                y_min=box[1] * img_height / self.config.input_height,
                x_max=box[2] * img_width / self.config.input_width,
                y_max=box[3] * img_height / self.config.input_height,
                normalized=False
            )
            
            detections.append(Detection(
                object_class=object_class,
                confidence=scores[idx],
                bbox_2d=bbox,
                timestamp=time.time()
            ))
            
        return detections
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> List[int]:
        """Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []
            
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
                
            # Calculate IoU with remaining boxes
            remaining = order[1:]
            
            xx1 = np.maximum(boxes[i, 0], boxes[remaining, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[remaining, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[remaining, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[remaining, 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_remaining = (boxes[remaining, 2] - boxes[remaining, 0]) * \
                           (boxes[remaining, 3] - boxes[remaining, 1])
            
            iou = intersection / (area_i + area_remaining - intersection)
            
            # Keep boxes with IoU below threshold
            mask = iou <= iou_threshold
            order = remaining[mask]
            
        return keep


# =============================================================================
# Triton Inference Client
# =============================================================================

class TritonDetector:
    """
    Object detection using NVIDIA Triton Inference Server.
    
    Supports multiple model architectures with automatic preprocessing
    and postprocessing.
    
    Example Usage:
        >>> config = ModelConfig(
        ...     name="warehouse_detector",
        ...     architecture=ModelArchitecture.YOLOV8,
        ...     triton_model_name="yolov8_warehouse"
        ... )
        >>> detector = TritonDetector(config, triton_url="localhost:8001")
        >>> 
        >>> result = detector.detect(image)
        >>> for det in result.detections:
        ...     print(f"{det.object_class.value}: {det.confidence:.2f}")
    """
    
    def __init__(
        self,
        config: ModelConfig,
        triton_url: str = "localhost:8001",
        use_grpc: bool = True
    ):
        self.config = config
        self.triton_url = triton_url
        self.use_grpc = use_grpc
        
        self.preprocessor = ImagePreprocessor(config)
        self.postprocessor = DetectionPostprocessor(config)
        
        # Initialize Triton client
        self._client = None
        self._model_ready = False
        
        if HAS_TRITON:
            self._init_client()
        else:
            logger.warning("Triton client not available, using mock inference")
            
    def _init_client(self):
        """Initialize Triton client."""
        try:
            if self.use_grpc:
                self._client = grpcclient.InferenceServerClient(
                    url=self.triton_url,
                    verbose=False
                )
            else:
                self._client = httpclient.InferenceServerClient(
                    url=self.triton_url,
                    verbose=False
                )
                
            # Check model status
            model_name = self.config.triton_model_name or self.config.name
            if self._client.is_model_ready(model_name):
                self._model_ready = True
                logger.info(f"Model {model_name} is ready on Triton")
            else:
                logger.warning(f"Model {model_name} not ready on Triton")
                
        except Exception as e:
            logger.error(f"Failed to connect to Triton: {e}")
            
    def detect(self, image: np.ndarray, camera_id: str = "") -> DetectionResult:
        """
        Run object detection on an image.
        
        Args:
            image: BGR image (H, W, 3) uint8
            camera_id: Optional camera identifier
            
        Returns:
            DetectionResult with all detections
        """
        img_height, img_width = image.shape[:2]
        timestamp = time.time()
        
        # Preprocess
        preprocess_start = time.time()
        input_tensor = self.preprocessor.preprocess(image)
        preprocess_time = (time.time() - preprocess_start) * 1000
        
        # Inference
        inference_start = time.time()
        
        if self._model_ready and self._client:
            output = self._triton_inference(input_tensor)
        else:
            output = self._mock_inference(input_tensor)
            
        inference_time = (time.time() - inference_start) * 1000
        
        # Postprocess
        postprocess_start = time.time()
        detections = self.postprocessor.process_yolo_output(
            output, img_width, img_height
        )
        
        # Add camera_id to detections
        for det in detections:
            det.camera_id = camera_id
            det.timestamp = timestamp
            
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        return DetectionResult(
            detections=detections,
            image_width=img_width,
            image_height=img_height,
            timestamp=timestamp,
            camera_id=camera_id,
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocess_time,
            postprocessing_time_ms=postprocess_time
        )
        
    def _triton_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference on Triton server."""
        model_name = self.config.triton_model_name or self.config.name
        
        if self.use_grpc:
            inputs = [
                grpcclient.InferInput("images", input_tensor.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(input_tensor)
            
            outputs = [grpcclient.InferRequestedOutput("output0")]
            
            result = self._client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            return result.as_numpy("output0")
        else:
            inputs = [
                httpclient.InferInput("images", input_tensor.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(input_tensor)
            
            outputs = [httpclient.InferRequestedOutput("output0")]
            
            result = self._client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            return result.as_numpy("output0")
            
    def _mock_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Mock inference for testing without Triton."""
        # Generate some random detections
        num_boxes = np.random.randint(1, 5)
        num_classes = len(self.config.class_names) or 10
        
        output = []
        for _ in range(num_boxes):
            x = np.random.uniform(0.1, 0.9) * self.config.input_width
            y = np.random.uniform(0.1, 0.9) * self.config.input_height
            w = np.random.uniform(0.05, 0.3) * self.config.input_width
            h = np.random.uniform(0.05, 0.3) * self.config.input_height
            conf = np.random.uniform(0.5, 0.95)
            
            class_scores = np.random.uniform(0, 1, num_classes)
            class_scores = class_scores / class_scores.sum()  # Softmax-like
            
            box = [x, y, w, h, conf] + list(class_scores)
            output.append(box)
            
        return np.array([output])
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        camera_ids: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        """Run detection on a batch of images."""
        if camera_ids is None:
            camera_ids = [f"cam_{i}" for i in range(len(images))]
            
        results = []
        for image, cam_id in zip(images, camera_ids):
            result = self.detect(image, cam_id)
            results.append(result)
            
        return results


# =============================================================================
# 3D Localization
# =============================================================================

class DepthLocalizer:
    """
    Localizes 2D detections in 3D space using depth information.
    
    Uses camera intrinsics and depth map to project detections
    into world coordinates.
    """
    
    def __init__(
        self,
        fx: float = 554.25,  # Focal length x
        fy: float = 554.25,  # Focal length y
        cx: float = 320.0,   # Principal point x
        cy: float = 240.0,   # Principal point y
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # Camera intrinsic matrix
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
    def localize_detection(
        self,
        detection: Detection,
        depth_image: np.ndarray,
        transform: Optional[np.ndarray] = None
    ) -> Detection:
        """
        Add 3D location to a 2D detection using depth.
        
        Args:
            detection: 2D detection
            depth_image: Depth image (H, W) in meters
            transform: 4x4 camera-to-world transform
            
        Returns:
            Detection with bbox_3d populated
        """
        bbox = detection.bbox_2d
        if bbox.normalized:
            bbox = bbox.to_pixels(depth_image.shape[1], depth_image.shape[0])
            
        # Get depth at detection center
        cx, cy = int(bbox.center[0]), int(bbox.center[1])
        cx = np.clip(cx, 0, depth_image.shape[1] - 1)
        cy = np.clip(cy, 0, depth_image.shape[0] - 1)
        
        # Use median depth in ROI for robustness
        x1, y1 = int(bbox.x_min), int(bbox.y_min)
        x2, y2 = int(bbox.x_max), int(bbox.y_max)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_image.shape[1], x2)
        y2 = min(depth_image.shape[0], y2)
        
        roi_depth = depth_image[y1:y2, x1:x2]
        valid_depth = roi_depth[roi_depth > 0]
        
        if len(valid_depth) == 0:
            return detection
            
        depth = np.median(valid_depth)
        
        # Back-project to 3D
        x_3d = (cx - self.cx) * depth / self.fx
        y_3d = (cy - self.cy) * depth / self.fy
        z_3d = depth
        
        point_camera = np.array([x_3d, y_3d, z_3d, 1.0])
        
        # Transform to world coordinates if provided
        if transform is not None:
            point_world = transform @ point_camera
            x_3d, y_3d, z_3d = point_world[:3]
            
        # Estimate 3D box dimensions from 2D and depth
        width_3d = bbox.width * depth / self.fx
        height_3d = bbox.height * depth / self.fy
        depth_3d = width_3d  # Assume roughly cubic for unknown objects
        
        # Adjust based on object class
        class_depth_ratios = {
            ObjectClass.PALLET: 1.2,
            ObjectClass.BOX: 1.0,
            ObjectClass.PERSON: 0.5,
            ObjectClass.FORKLIFT: 2.0,
        }
        depth_3d *= class_depth_ratios.get(detection.object_class, 1.0)
        
        detection.bbox_3d = BoundingBox3D(
            center_x=float(x_3d),
            center_y=float(y_3d),
            center_z=float(z_3d),
            width=float(width_3d),
            depth=float(depth_3d),
            height=float(height_3d)
        )
        
        return detection
    
    def localize_all(
        self,
        result: DetectionResult,
        depth_image: np.ndarray,
        transform: Optional[np.ndarray] = None
    ) -> DetectionResult:
        """Add 3D locations to all detections in a result."""
        for detection in result.detections:
            self.localize_detection(detection, depth_image, transform)
        return result


# =============================================================================
# Object Tracker
# =============================================================================

class SimpleTracker:
    """
    Simple multi-object tracker using IoU matching.
    
    For production, consider using DeepSORT or ByteTrack.
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1
        
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update tracker with new detections.
        
        Returns detections with track_id assigned.
        """
        if not detections:
            # Age out old tracks
            self._age_tracks()
            return []
            
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match(detections)
        
        # Update matched tracks
        for det_idx, track_id in matched:
            detection = detections[det_idx]
            detection.track_id = track_id
            self.tracks[track_id]['bbox'] = detection.bbox_2d
            self.tracks[track_id]['class'] = detection.object_class
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['age'] = 0
            
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'bbox': detection.bbox_2d,
                'class': detection.object_class,
                'hits': 1,
                'age': 0
            }
            detection.track_id = track_id
            
        # Age unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]['age'] += 1
            
        # Remove old tracks
        self._age_tracks()
        
        return detections
    
    def _match(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using IoU."""
        if not self.tracks:
            return [], list(range(len(detections))), []
            
        # Build IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track_bbox = self.tracks[track_id]['bbox']
                iou_matrix[i, j] = det.bbox_2d.iou(track_bbox)
                
        # Greedy matching
        matched = []
        matched_dets = set()
        matched_tracks = set()
        
        while True:
            if len(matched_dets) >= len(detections) or len(matched_tracks) >= len(track_ids):
                break
                
            # Find best match
            max_iou = 0
            best_i, best_j = -1, -1
            
            for i in range(len(detections)):
                if i in matched_dets:
                    continue
                for j in range(len(track_ids)):
                    if j in matched_tracks:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_i, best_j = i, j
                        
            if max_iou < self.iou_threshold:
                break
                
            matched.append((best_i, track_ids[best_j]))
            matched_dets.add(best_i)
            matched_tracks.add(best_j)
            
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [track_ids[j] for j in range(len(track_ids)) if j not in matched_tracks]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _age_tracks(self):
        """Remove tracks that haven't been matched recently."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['age'] > self.max_age:
                to_remove.append(track_id)
                
        for track_id in to_remove:
            del self.tracks[track_id]


# =============================================================================
# ROS 2 Integration
# =============================================================================

if HAS_ROS:
    class DetectionNode(Node):
        """ROS 2 node for object detection."""
        
        def __init__(
            self,
            detector: TritonDetector,
            camera_topics: List[str],
            depth_topics: Optional[List[str]] = None
        ):
            super().__init__('object_detection')
            
            self.detector = detector
            self.localizer = DepthLocalizer()
            self.tracker = SimpleTracker()
            self.bridge = CvBridge()
            
            # Subscribers
            self.image_subs = []
            self.depth_subs = []
            self.latest_depth: Dict[str, np.ndarray] = {}
            
            for i, topic in enumerate(camera_topics):
                sub = self.create_subscription(
                    Image,
                    topic,
                    lambda msg, idx=i: self._image_callback(msg, idx),
                    10
                )
                self.image_subs.append(sub)
                
            if depth_topics:
                for i, topic in enumerate(depth_topics):
                    sub = self.create_subscription(
                        Image,
                        topic,
                        lambda msg, idx=i: self._depth_callback(msg, idx),
                        10
                    )
                    self.depth_subs.append(sub)
                    
            # Publishers
            self.detection_pub = self.create_publisher(
                Detection2DArray,
                '/detections',
                10
            )
            
            self.viz_pub = self.create_publisher(
                Image,
                '/detection_visualization',
                10
            )
            
            self.get_logger().info("Detection node initialized")
            
        def _image_callback(self, msg: Image, camera_idx: int):
            """Process incoming image."""
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
                
            camera_id = f"camera_{camera_idx}"
            
            # Run detection
            result = self.detector.detect(cv_image, camera_id)
            
            # Add 3D localization if depth available
            if camera_id in self.latest_depth:
                result = self.localizer.localize_all(result, self.latest_depth[camera_id])
                
            # Track objects
            result.detections = self.tracker.update(result.detections)
            
            # Publish results
            self._publish_detections(result, msg.header)
            
            # Publish visualization
            viz_image = self._draw_detections(cv_image, result)
            viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
            viz_msg.header = msg.header
            self.viz_pub.publish(viz_msg)
            
        def _depth_callback(self, msg: Image, camera_idx: int):
            """Store latest depth image."""
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.latest_depth[f"camera_{camera_idx}"] = depth
            except Exception as e:
                self.get_logger().error(f"Failed to convert depth: {e}")
                
        def _publish_detections(self, result: DetectionResult, header: Header):
            """Publish Detection2DArray message."""
            msg = Detection2DArray()
            msg.header = header
            
            for det in result.detections:
                det_msg = Detection2D()
                det_msg.header = header
                
                # Bounding box
                det_msg.bbox.center.position.x = det.bbox_2d.center[0]
                det_msg.bbox.center.position.y = det.bbox_2d.center[1]
                det_msg.bbox.size_x = det.bbox_2d.width
                det_msg.bbox.size_y = det.bbox_2d.height
                
                # Hypothesis
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = det.object_class.value
                hyp.hypothesis.score = det.confidence
                
                if det.bbox_3d:
                    hyp.pose.pose.position.x = det.bbox_3d.center_x
                    hyp.pose.pose.position.y = det.bbox_3d.center_y
                    hyp.pose.pose.position.z = det.bbox_3d.center_z
                    
                det_msg.results.append(hyp)
                msg.detections.append(det_msg)
                
            self.detection_pub.publish(msg)
            
        def _draw_detections(
            self,
            image: np.ndarray,
            result: DetectionResult
        ) -> np.ndarray:
            """Draw bounding boxes on image."""
            viz = image.copy()
            
            colors = {
                ObjectClass.PERSON: (0, 255, 0),      # Green
                ObjectClass.FORKLIFT: (0, 0, 255),   # Red
                ObjectClass.PALLET: (255, 128, 0),   # Orange
                ObjectClass.BOX: (255, 255, 0),      # Yellow
            }
            default_color = (128, 128, 128)
            
            for det in result.detections:
                bbox = det.bbox_2d
                color = colors.get(det.object_class, default_color)
                
                # Draw box
                pt1 = (int(bbox.x_min), int(bbox.y_min))
                pt2 = (int(bbox.x_max), int(bbox.y_max))
                cv2.rectangle(viz, pt1, pt2, color, 2)
                
                # Label
                label = f"{det.object_class.value}"
                if det.track_id:
                    label += f" #{det.track_id}"
                label += f" {det.confidence:.2f}"
                
                cv2.putText(viz, label, (pt1[0], pt1[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                           
            return viz


# =============================================================================
# Detection Pipeline Manager
# =============================================================================

class DetectionPipeline:
    """
    High-level detection pipeline manager.
    
    Coordinates multiple cameras, models, and post-processing steps.
    
    Example:
        >>> pipeline = DetectionPipeline()
        >>> pipeline.add_model(ModelConfig(
        ...     name="warehouse",
        ...     architecture=ModelArchitecture.YOLOV8,
        ...     triton_model_name="yolov8_warehouse"
        ... ))
        >>> pipeline.start()
        >>> 
        >>> result = pipeline.detect(image)
    """
    
    def __init__(
        self,
        triton_url: str = "localhost:8001",
        enable_tracking: bool = True,
        enable_3d: bool = True
    ):
        self.triton_url = triton_url
        self.enable_tracking = enable_tracking
        self.enable_3d = enable_3d
        
        self.detectors: Dict[str, TritonDetector] = {}
        self.tracker = SimpleTracker() if enable_tracking else None
        self.localizer = DepthLocalizer() if enable_3d else None
        
        self._callbacks: List[Callable[[DetectionResult], None]] = []
        
    def add_model(self, config: ModelConfig):
        """Add a detection model to the pipeline."""
        detector = TritonDetector(config, self.triton_url)
        self.detectors[config.name] = detector
        logger.info(f"Added model: {config.name}")
        
    def detect(
        self,
        image: np.ndarray,
        model_name: Optional[str] = None,
        camera_id: str = "",
        depth_image: Optional[np.ndarray] = None
    ) -> DetectionResult:
        """
        Run detection pipeline.
        
        Args:
            image: Input BGR image
            model_name: Specific model to use (or first available)
            camera_id: Camera identifier
            depth_image: Optional depth for 3D localization
            
        Returns:
            DetectionResult with all post-processing applied
        """
        # Select detector
        if model_name and model_name in self.detectors:
            detector = self.detectors[model_name]
        elif self.detectors:
            detector = list(self.detectors.values())[0]
        else:
            raise ValueError("No detection models configured")
            
        # Run detection
        result = detector.detect(image, camera_id)
        
        # Add 3D localization
        if self.enable_3d and self.localizer and depth_image is not None:
            result = self.localizer.localize_all(result, depth_image)
            
        # Track objects
        if self.enable_tracking and self.tracker:
            result.detections = self.tracker.update(result.detections)
            
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                
        return result
    
    def on_detection(self, callback: Callable[[DetectionResult], None]):
        """Register callback for detection results."""
        self._callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'models': list(self.detectors.keys()),
            'tracking_enabled': self.enable_tracking,
            '3d_enabled': self.enable_3d,
            'active_tracks': len(self.tracker.tracks) if self.tracker else 0
        }


# =============================================================================
# Pre-configured Models
# =============================================================================

def create_warehouse_config() -> ModelConfig:
    """Create config for warehouse object detection."""
    return ModelConfig(
        name="warehouse_detector",
        architecture=ModelArchitecture.YOLOV8,
        input_width=640,
        input_height=640,
        triton_model_name="yolov8_warehouse",
        class_names=[
            "pallet", "box", "shelf", "rack", "person",
            "forklift", "cart", "door", "obstacle"
        ],
        class_mapping={
            0: ObjectClass.PALLET,
            1: ObjectClass.BOX,
            2: ObjectClass.SHELF,
            3: ObjectClass.RACK,
            4: ObjectClass.PERSON,
            5: ObjectClass.FORKLIFT,
            6: ObjectClass.CART,
            7: ObjectClass.DOOR,
            8: ObjectClass.OBSTACLE
        },
        confidence_threshold=0.5,
        nms_threshold=0.45
    )


def create_safety_config() -> ModelConfig:
    """Create config for safety detection (people, forklifts)."""
    return ModelConfig(
        name="safety_detector",
        architecture=ModelArchitecture.YOLOV8,
        input_width=640,
        input_height=640,
        triton_model_name="yolov8_safety",
        class_names=["person", "forklift", "warning_sign", "spill"],
        class_mapping={
            0: ObjectClass.PERSON,
            1: ObjectClass.FORKLIFT,
            2: ObjectClass.WARNING_SIGN,
            3: ObjectClass.SPILL
        },
        confidence_threshold=0.6,  # Higher threshold for safety
        nms_threshold=0.4
    )


# =============================================================================
# Demo
# =============================================================================

def demo_detection():
    """Demonstrate object detection pipeline."""
    print("=== Object Detection Pipeline Demo ===\n")
    
    # Create pipeline
    pipeline = DetectionPipeline(
        triton_url="localhost:8001",
        enable_tracking=True,
        enable_3d=True
    )
    
    # Add warehouse model
    config = create_warehouse_config()
    pipeline.add_model(config)
    
    print(f"Pipeline Configuration:")
    print(f"  Models: {list(pipeline.detectors.keys())}")
    print(f"  Tracking: {pipeline.enable_tracking}")
    print(f"  3D Localization: {pipeline.enable_3d}")
    
    # Create dummy image
    if HAS_CV2:
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.uniform(1.0, 10.0, (480, 640)).astype(np.float32)
        
        print("\n--- Running Detection ---")
        
        # Simulate multiple frames
        for frame in range(5):
            result = pipeline.detect(
                image,
                camera_id="front_camera",
                depth_image=depth
            )
            
            print(f"\nFrame {frame + 1}:")
            print(f"  Detections: {len(result.detections)}")
            print(f"  Inference time: {result.inference_time_ms:.1f}ms")
            print(f"  Total time: {result.total_time_ms:.1f}ms")
            
            for det in result.detections:
                print(f"    - {det.object_class.value}: {det.confidence:.2f} "
                      f"(track #{det.track_id})")
                if det.bbox_3d:
                    pos = det.bbox_3d
                    print(f"      3D: ({pos.center_x:.2f}, {pos.center_y:.2f}, "
                          f"{pos.center_z:.2f})m")
    else:
        print("\nOpenCV not available, skipping image processing demo")
        
    print("\n--- Supported Object Classes ---")
    for obj_class in ObjectClass:
        print(f"  - {obj_class.value}")
        
    print("\n--- Supported Architectures ---")
    for arch in ModelArchitecture:
        print(f"  - {arch.value}")


if __name__ == "__main__":
    demo_detection()
