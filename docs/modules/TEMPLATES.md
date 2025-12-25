# Digital Twin Templates Module

Reusable robot configurations for digital twin instantiation.

## Features

- **Robot Types**: Mobile, Arm, Humanoid, Drone, AMR, Cobot
- **Sensor Configs**: LiDAR, cameras, IMU, encoders
- **Joint Definitions**: Revolute, prismatic, fixed
- **Persistence**: JSON-based template storage

## Robot Types

| Type | Description |
|------|-------------|
| `mobile` | Wheeled mobile robots |
| `arm` | Robotic manipulators |
| `humanoid` | Bipedal robots |
| `drone` | Aerial vehicles |
| `amr` | Autonomous Mobile Robots |
| `cobot` | Collaborative robots |

## Usage

### Create Template
```python
from templates import TwinTemplate, RobotType, SensorConfig, SensorType

template = TwinTemplate(
    name="warehouse-amr",
    robot_type=RobotType.AMR,
    description="Warehouse logistics robot",
    mass_kg=50.0,
    dimensions={"length": 0.8, "width": 0.6, "height": 0.4},
    max_speed_mps=2.0,
    max_payload_kg=100.0,
    battery_capacity_wh=500.0,
    sensors=[
        SensorConfig(SensorType.LIDAR, "front_lidar", update_rate_hz=10),
        SensorConfig(SensorType.DEPTH_CAMERA, "nav_camera", update_rate_hz=30),
        SensorConfig(SensorType.IMU, "imu", update_rate_hz=100),
    ],
    tags=["warehouse", "logistics", "amr"],
)
```

### Template Manager
```python
from templates import TemplateManager

manager = TemplateManager("./robot_templates")

# Save template
manager.save(template)

# Find templates
amr_templates = manager.list(robot_type=RobotType.AMR)
warehouse_templates = manager.list(tag="warehouse")

# Instantiate
robot = manager.instantiate(template.id, "AMR-001")
```

### Sensor Configuration
```python
from templates import SensorConfig, SensorType

lidar = SensorConfig(
    sensor_type=SensorType.LIDAR,
    name="velodyne_vlp16",
    mount_point="top",
    update_rate_hz=10.0,
    parameters={"range_m": 100, "fov_deg": 360},
)
```

### Joint Configuration
```python
from templates.twin_templates import JointConfig

joints = [
    JointConfig(name="shoulder", limits=(-1.57, 1.57), max_torque=150),
    JointConfig(name="elbow", limits=(0, 2.5), max_torque=100),
    JointConfig(name="wrist", limits=(-3.14, 3.14), max_velocity=2.0),
]
```

## Integration

- URDF robot descriptions
- USD/Omniverse assets
- Isaac Sim environments
- ROS 2 robot models
