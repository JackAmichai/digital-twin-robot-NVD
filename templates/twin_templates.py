"""Digital twin templates for robot instantiation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4
import json
from pathlib import Path


class RobotType(Enum):
    """Standard robot types."""
    MOBILE = "mobile"
    ARM = "arm"
    HUMANOID = "humanoid"
    DRONE = "drone"
    AMR = "amr"  # Autonomous Mobile Robot
    COBOT = "cobot"  # Collaborative Robot


class SensorType(Enum):
    """Sensor types."""
    LIDAR = "lidar"
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    IMU = "imu"
    ENCODER = "encoder"
    FORCE_TORQUE = "force_torque"
    PROXIMITY = "proximity"
    GPS = "gps"


@dataclass
class SensorConfig:
    """Sensor configuration."""
    
    sensor_type: SensorType
    name: str
    mount_point: str = ""
    update_rate_hz: float = 30.0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class JointConfig:
    """Robot joint configuration."""
    
    name: str
    joint_type: str = "revolute"  # revolute, prismatic, fixed
    limits: tuple[float, float] = (-3.14, 3.14)
    max_velocity: float = 1.0
    max_torque: float = 100.0


@dataclass
class TwinTemplate:
    """Digital twin template definition."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    robot_type: RobotType = RobotType.MOBILE
    description: str = ""
    version: str = "1.0.0"
    
    # Physical properties
    mass_kg: float = 10.0
    dimensions: dict[str, float] = field(default_factory=lambda: {
        "length": 0.5, "width": 0.5, "height": 0.3
    })
    
    # Components
    sensors: list[SensorConfig] = field(default_factory=list)
    joints: list[JointConfig] = field(default_factory=list)
    
    # Capabilities
    max_speed_mps: float = 1.0
    max_payload_kg: float = 5.0
    battery_capacity_wh: float = 100.0
    
    # Simulation
    urdf_path: str = ""
    usd_path: str = ""
    physics_material: str = "default"
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    custom_properties: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "robot_type": self.robot_type.value,
            "description": self.description,
            "version": self.version,
            "mass_kg": self.mass_kg,
            "dimensions": self.dimensions,
            "sensors": [
                {"type": s.sensor_type.value, "name": s.name, "rate": s.update_rate_hz}
                for s in self.sensors
            ],
            "joints": [
                {"name": j.name, "type": j.joint_type, "limits": j.limits}
                for j in self.joints
            ],
            "max_speed_mps": self.max_speed_mps,
            "max_payload_kg": self.max_payload_kg,
            "battery_capacity_wh": self.battery_capacity_wh,
            "urdf_path": self.urdf_path,
            "usd_path": self.usd_path,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TwinTemplate":
        """Create from dictionary."""
        sensors = [
            SensorConfig(
                sensor_type=SensorType(s["type"]),
                name=s["name"],
                update_rate_hz=s.get("rate", 30.0),
            )
            for s in data.get("sensors", [])
        ]
        joints = [
            JointConfig(
                name=j["name"],
                joint_type=j.get("type", "revolute"),
                limits=tuple(j.get("limits", [-3.14, 3.14])),
            )
            for j in data.get("joints", [])
        ]
        return cls(
            id=data.get("id", str(uuid4())),
            name=data["name"],
            robot_type=RobotType(data.get("robot_type", "mobile")),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            mass_kg=data.get("mass_kg", 10.0),
            dimensions=data.get("dimensions", {}),
            sensors=sensors,
            joints=joints,
            max_speed_mps=data.get("max_speed_mps", 1.0),
            max_payload_kg=data.get("max_payload_kg", 5.0),
            battery_capacity_wh=data.get("battery_capacity_wh", 100.0),
            urdf_path=data.get("urdf_path", ""),
            usd_path=data.get("usd_path", ""),
            tags=data.get("tags", []),
        )


class TemplateManager:
    """Manage digital twin templates."""
    
    def __init__(self, storage_path: Path | str = "./templates"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._templates: dict[str, TwinTemplate] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load templates from storage."""
        for path in self.storage_path.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
                template = TwinTemplate.from_dict(data)
                self._templates[template.id] = template
    
    def save(self, template: TwinTemplate) -> None:
        """Save template to storage."""
        self._templates[template.id] = template
        path = self.storage_path / f"{template.id}.json"
        with open(path, "w") as f:
            json.dump(template.to_dict(), f, indent=2)
    
    def get(self, template_id: str) -> TwinTemplate | None:
        """Get template by ID."""
        return self._templates.get(template_id)
    
    def find_by_name(self, name: str) -> TwinTemplate | None:
        """Find template by name."""
        for t in self._templates.values():
            if t.name == name:
                return t
        return None
    
    def list(
        self,
        robot_type: RobotType | None = None,
        tag: str | None = None,
    ) -> list[TwinTemplate]:
        """List templates with optional filters."""
        templates = list(self._templates.values())
        if robot_type:
            templates = [t for t in templates if t.robot_type == robot_type]
        if tag:
            templates = [t for t in templates if tag in t.tags]
        return templates
    
    def delete(self, template_id: str) -> bool:
        """Delete template."""
        if template_id in self._templates:
            del self._templates[template_id]
            path = self.storage_path / f"{template_id}.json"
            if path.exists():
                path.unlink()
            return True
        return False
    
    def instantiate(self, template_id: str, name: str) -> TwinTemplate:
        """Create instance from template."""
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        instance = TwinTemplate(
            name=name,
            robot_type=template.robot_type,
            description=f"Instance of {template.name}",
            mass_kg=template.mass_kg,
            dimensions=template.dimensions.copy(),
            sensors=template.sensors.copy(),
            joints=template.joints.copy(),
            max_speed_mps=template.max_speed_mps,
            max_payload_kg=template.max_payload_kg,
            battery_capacity_wh=template.battery_capacity_wh,
            urdf_path=template.urdf_path,
            usd_path=template.usd_path,
        )
        return instance
