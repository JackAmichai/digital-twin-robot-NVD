"""XR (AR/VR) integration for immersive robotics control."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from uuid import uuid4
import asyncio


class XRDeviceType(Enum):
    """XR device types."""
    VR_HEADSET = "vr_headset"
    AR_GLASSES = "ar_glasses"
    CONTROLLER = "controller"
    HAND_TRACKER = "hand_tracker"
    SPATIAL_TRACKER = "spatial_tracker"


class TrackingState(Enum):
    """Device tracking state."""
    NOT_TRACKING = "not_tracking"
    LIMITED = "limited"
    TRACKING = "tracking"


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    """Rotation quaternion."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    """6DOF pose."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)


@dataclass
class XRDevice:
    """XR device representation."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    device_type: XRDeviceType = XRDeviceType.VR_HEADSET
    name: str = ""
    tracking_state: TrackingState = TrackingState.NOT_TRACKING
    pose: Pose = field(default_factory=Pose)
    battery_level: float = 1.0


@dataclass
class ControllerButton:
    """Controller button state."""
    pressed: bool = False
    touched: bool = False
    value: float = 0.0


@dataclass
class XRController(XRDevice):
    """XR controller with input state."""
    
    trigger: ControllerButton = field(default_factory=ControllerButton)
    grip: ControllerButton = field(default_factory=ControllerButton)
    thumbstick: Vector3 = field(default_factory=Vector3)
    buttons: dict[str, ControllerButton] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        self.device_type = XRDeviceType.CONTROLLER


@dataclass
class HandJoint:
    """Hand tracking joint."""
    pose: Pose = field(default_factory=Pose)
    radius: float = 0.01


@dataclass
class HandTracking:
    """Hand tracking data."""
    
    is_left: bool = True
    joints: dict[str, HandJoint] = field(default_factory=dict)
    tracking_confidence: float = 0.0
    
    JOINT_NAMES = [
        "wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
        "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
        "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
        "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
        "pinky_metacarpal", "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
    ]
    
    def get_pinch_strength(self) -> float:
        """Calculate pinch gesture strength."""
        thumb = self.joints.get("thumb_tip")
        index = self.joints.get("index_tip")
        if not thumb or not index:
            return 0.0
        
        dx = thumb.pose.position.x - index.pose.position.x
        dy = thumb.pose.position.y - index.pose.position.y
        dz = thumb.pose.position.z - index.pose.position.z
        distance = (dx*dx + dy*dy + dz*dz) ** 0.5
        
        return max(0, 1 - distance / 0.05)


@dataclass
class SpatialAnchor:
    """Spatial anchor for persistent positioning."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    pose: Pose = field(default_factory=Pose)
    persistent: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class XRSession:
    """Manage XR session and devices."""
    
    def __init__(self) -> None:
        self._devices: dict[str, XRDevice] = {}
        self._anchors: dict[str, SpatialAnchor] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._active = False
    
    async def start(self) -> None:
        """Start XR session."""
        self._active = True
    
    async def stop(self) -> None:
        """Stop XR session."""
        self._active = False
    
    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._active
    
    def register_device(self, device: XRDevice) -> None:
        """Register XR device."""
        self._devices[device.id] = device
        self._emit("device_connected", device)
    
    def unregister_device(self, device_id: str) -> None:
        """Unregister XR device."""
        device = self._devices.pop(device_id, None)
        if device:
            self._emit("device_disconnected", device)
    
    def get_device(self, device_id: str) -> XRDevice | None:
        """Get device by ID."""
        return self._devices.get(device_id)
    
    def get_headset(self) -> XRDevice | None:
        """Get primary headset."""
        for device in self._devices.values():
            if device.device_type in (XRDeviceType.VR_HEADSET, XRDeviceType.AR_GLASSES):
                return device
        return None
    
    def get_controllers(self) -> list[XRController]:
        """Get all controllers."""
        return [
            d for d in self._devices.values()
            if isinstance(d, XRController)
        ]
    
    def create_anchor(self, name: str, pose: Pose) -> SpatialAnchor:
        """Create spatial anchor."""
        anchor = SpatialAnchor(name=name, pose=pose)
        self._anchors[anchor.id] = anchor
        return anchor
    
    def get_anchor(self, anchor_id: str) -> SpatialAnchor | None:
        """Get anchor by ID."""
        return self._anchors.get(anchor_id)
    
    def list_anchors(self) -> list[SpatialAnchor]:
        """List all anchors."""
        return list(self._anchors.values())
    
    def on(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _emit(self, event: str, data: Any) -> None:
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            callback(data)
    
    async def update(self) -> None:
        """Update session state (call in render loop)."""
        for device in self._devices.values():
            self._emit("device_updated", device)
