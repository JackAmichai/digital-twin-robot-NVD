# AR/VR Integration Module

Extended Reality (XR) integration for immersive robot control.

## Features

- **Device Management**: Headsets, controllers, trackers
- **Hand Tracking**: Full hand skeleton with gestures
- **Spatial Anchors**: Persistent world positioning
- **Event System**: Device connection and tracking events

## Device Types

| Type | Description |
|------|-------------|
| `vr_headset` | Virtual reality headset |
| `ar_glasses` | Augmented reality glasses |
| `controller` | Hand controller |
| `hand_tracker` | Hand tracking sensor |
| `spatial_tracker` | External tracker |

## Usage

### XR Session
```python
from xr import XRSession

session = XRSession()
await session.start()

# Event callbacks
session.on("device_connected", lambda d: print(f"Connected: {d.name}"))
session.on("device_updated", handle_tracking)

# Get devices
headset = session.get_headset()
controllers = session.get_controllers()
```

### Controller Input
```python
from xr import XRController

controller = session.get_controllers()[0]

# Check inputs
if controller.trigger.pressed:
    robot.grip_object()

if controller.grip.value > 0.5:
    robot.close_gripper(controller.grip.value)

# Thumbstick navigation
robot.move(
    controller.thumbstick.x * speed,
    controller.thumbstick.y * speed,
)
```

### Hand Tracking
```python
from xr import HandTracking

hand = HandTracking(is_left=True)

# Pinch gesture detection
pinch = hand.get_pinch_strength()
if pinch > 0.8:
    robot.select_object()

# Access joints
wrist = hand.joints["wrist"]
index_tip = hand.joints["index_tip"]
```

### Spatial Anchors
```python
from xr import Pose, Vector3

# Create anchor at robot location
anchor = session.create_anchor(
    name="robot_home",
    pose=Pose(position=Vector3(1.0, 0.0, 2.0)),
)

# Retrieve later
home = session.get_anchor(anchor.id)
robot.navigate_to(home.pose.position)
```

## Integration

- NVIDIA Omniverse XR
- OpenXR runtime
- Meta Quest support
- HoloLens 2 compatibility
- WebXR for browser access
