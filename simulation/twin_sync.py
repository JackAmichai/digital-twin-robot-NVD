#!/usr/bin/env python3
"""
Real-Time Digital Twin Synchronization

Provides bidirectional synchronization between physical robots and their
digital twins in NVIDIA Isaac Sim. Enables:
- Real robot → Simulation: Physical state drives simulation visualization
- Simulation → Real robot: Test movements in sim before physical execution
- Predictive simulation: Run scenarios ahead of real-time

Architecture:
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Physical Robot │────▶│   Twin Sync Hub  │────▶│   Isaac Sim     │
│   (ROS 2)       │◀────│   (This Module)  │◀────│   Simulation    │
└─────────────────┘     └──────────────────┘     └─────────────────┘

Sync Modes:
- MIRROR: Sim follows physical robot exactly
- SHADOW: Sim predicts physical robot path
- COMMAND: Physical robot follows sim commands
- HYBRID: Bidirectional with conflict resolution
"""

import asyncio
import threading
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Callable, Any, Tuple
from enum import Enum
from collections import deque
import numpy as np

# ROS 2 imports (conditional for non-ROS environments)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from geometry_msgs.msg import (
        PoseStamped, Pose, Point, Quaternion,
        Twist, TwistStamped, TransformStamped
    )
    from nav_msgs.msg import Odometry, Path
    from sensor_msgs.msg import JointState, LaserScan, Imu
    from std_msgs.msg import Header, String, Float32MultiArray, Bool
    from tf2_ros import TransformBroadcaster, Buffer, TransformListener
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    logging.warning("ROS 2 not available, running in simulation-only mode")

# Redis for state caching (optional)
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class SyncMode(Enum):
    """Synchronization modes for digital twin."""
    MIRROR = "mirror"       # Sim mirrors physical robot exactly
    SHADOW = "shadow"       # Sim predicts physical robot's future
    COMMAND = "command"     # Physical robot follows sim commands
    HYBRID = "hybrid"       # Bidirectional with arbitration
    PAUSED = "paused"       # Sync disabled


class ConflictResolution(Enum):
    """How to resolve conflicts in HYBRID mode."""
    PHYSICAL_PRIORITY = "physical_priority"  # Physical robot wins
    SIM_PRIORITY = "sim_priority"           # Simulation wins
    NEWEST_WINS = "newest_wins"             # Most recent update wins
    AVERAGE = "average"                     # Blend both states


@dataclass
class RobotPose:
    """6-DOF robot pose."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0
    frame_id: str = "world"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'RobotPose':
        return cls(**d)
    
    def distance_to(self, other: 'RobotPose') -> float:
        """Euclidean distance to another pose."""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def blend(self, other: 'RobotPose', alpha: float = 0.5) -> 'RobotPose':
        """Blend this pose with another (linear interpolation)."""
        return RobotPose(
            x=self.x * (1 - alpha) + other.x * alpha,
            y=self.y * (1 - alpha) + other.y * alpha,
            z=self.z * (1 - alpha) + other.z * alpha,
            roll=self.roll * (1 - alpha) + other.roll * alpha,
            pitch=self.pitch * (1 - alpha) + other.pitch * alpha,
            yaw=self._blend_angle(self.yaw, other.yaw, alpha),
            timestamp=max(self.timestamp, other.timestamp),
            frame_id=self.frame_id
        )
    
    @staticmethod
    def _blend_angle(a: float, b: float, alpha: float) -> float:
        """Blend two angles accounting for wraparound."""
        diff = b - a
        # Normalize to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return a + alpha * diff


@dataclass
class RobotVelocity:
    """Robot velocity (linear and angular)."""
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JointStates:
    """Robot joint states for articulated robots."""
    names: List[str] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    efforts: List[float] = field(default_factory=list)
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'names': self.names,
            'positions': self.positions,
            'velocities': self.velocities,
            'efforts': self.efforts,
            'timestamp': self.timestamp
        }


@dataclass
class TwinState:
    """Complete state of a robot twin."""
    robot_id: str
    pose: RobotPose = field(default_factory=RobotPose)
    velocity: RobotVelocity = field(default_factory=RobotVelocity)
    joints: Optional[JointStates] = None
    sync_mode: SyncMode = SyncMode.MIRROR
    is_physical: bool = True  # True for physical robot, False for sim
    last_sync: float = 0.0
    sync_latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'robot_id': self.robot_id,
            'pose': self.pose.to_dict(),
            'velocity': self.velocity.to_dict(),
            'joints': self.joints.to_dict() if self.joints else None,
            'sync_mode': self.sync_mode.value,
            'is_physical': self.is_physical,
            'last_sync': self.last_sync,
            'sync_latency_ms': self.sync_latency_ms
        }


@dataclass 
class SyncConfig:
    """Configuration for twin synchronization."""
    # Timing
    sync_rate_hz: float = 30.0          # Target sync frequency
    max_latency_ms: float = 100.0       # Maximum acceptable latency
    prediction_horizon_s: float = 0.5   # How far ahead to predict (SHADOW mode)
    
    # Thresholds
    position_threshold_m: float = 0.01  # Position change threshold
    rotation_threshold_rad: float = 0.02  # Rotation change threshold
    
    # Conflict resolution
    conflict_resolution: ConflictResolution = ConflictResolution.PHYSICAL_PRIORITY
    blend_factor: float = 0.5           # For AVERAGE resolution
    
    # History
    history_size: int = 100             # Frames to keep for prediction
    
    # Isaac Sim connection
    isaac_host: str = "localhost"
    isaac_port: int = 8211              # Omniverse streaming port
    
    # Redis caching
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_enabled: bool = True


# =============================================================================
# State Predictor
# =============================================================================

class StatePredictor:
    """
    Predicts future robot states based on historical data.
    Uses velocity-based extrapolation with smoothing.
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.pose_history: deque = deque(maxlen=history_size)
        self.velocity_history: deque = deque(maxlen=history_size)
        
    def add_state(self, pose: RobotPose, velocity: RobotVelocity):
        """Add a state observation."""
        self.pose_history.append(pose)
        self.velocity_history.append(velocity)
        
    def predict(self, time_ahead: float) -> Optional[RobotPose]:
        """
        Predict pose at time_ahead seconds in the future.
        
        Uses current velocity and historical acceleration for prediction.
        """
        if len(self.pose_history) < 2:
            return None
            
        current_pose = self.pose_history[-1]
        current_vel = self.velocity_history[-1] if self.velocity_history else None
        
        if current_vel is None:
            # Estimate velocity from pose history
            prev_pose = self.pose_history[-2]
            dt = current_pose.timestamp - prev_pose.timestamp
            if dt > 0:
                current_vel = RobotVelocity(
                    linear_x=(current_pose.x - prev_pose.x) / dt,
                    linear_y=(current_pose.y - prev_pose.y) / dt,
                    linear_z=(current_pose.z - prev_pose.z) / dt,
                    angular_z=(current_pose.yaw - prev_pose.yaw) / dt
                )
            else:
                return current_pose
                
        # Simple linear prediction
        predicted = RobotPose(
            x=current_pose.x + current_vel.linear_x * time_ahead,
            y=current_pose.y + current_vel.linear_y * time_ahead,
            z=current_pose.z + current_vel.linear_z * time_ahead,
            roll=current_pose.roll + current_vel.angular_x * time_ahead,
            pitch=current_pose.pitch + current_vel.angular_y * time_ahead,
            yaw=current_pose.yaw + current_vel.angular_z * time_ahead,
            timestamp=current_pose.timestamp + time_ahead,
            frame_id=current_pose.frame_id
        )
        
        return predicted
    
    def get_velocity_trend(self) -> Optional[Tuple[float, float]]:
        """Get velocity trend (acceleration) from history."""
        if len(self.velocity_history) < 5:
            return None
            
        velocities = list(self.velocity_history)[-5:]
        
        # Linear regression on velocity magnitudes
        v_mags = [np.sqrt(v.linear_x**2 + v.linear_y**2) for v in velocities]
        times = list(range(len(v_mags)))
        
        # Simple slope calculation
        mean_t = np.mean(times)
        mean_v = np.mean(v_mags)
        
        numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, v_mags))
        denominator = sum((t - mean_t) ** 2 for t in times)
        
        if denominator == 0:
            return (mean_v, 0.0)
            
        slope = numerator / denominator
        return (mean_v, slope)


# =============================================================================
# Isaac Sim Interface
# =============================================================================

class IsaacSimInterface:
    """
    Interface to NVIDIA Isaac Sim for digital twin visualization.
    
    Communicates with Isaac Sim via:
    - Omniverse Kit scripting API
    - ROS 2 bridge (when enabled)
    - Direct USD stage manipulation
    """
    
    def __init__(self, host: str = "localhost", port: int = 8211):
        self.host = host
        self.port = port
        self.connected = False
        self._robot_prims: Dict[str, str] = {}  # robot_id -> USD prim path
        
        # Try to import Isaac Sim modules
        try:
            from omni.isaac.core import World
            from omni.isaac.core.robots import Robot
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from pxr import Gf, UsdGeom
            self._has_isaac = True
        except ImportError:
            self._has_isaac = False
            logger.warning("Isaac Sim not available, using mock interface")
            
    async def connect(self) -> bool:
        """Establish connection to Isaac Sim."""
        if self._has_isaac:
            try:
                # In actual Isaac Sim environment, this would connect to the stage
                logger.info(f"Connecting to Isaac Sim at {self.host}:{self.port}")
                self.connected = True
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Isaac Sim: {e}")
                return False
        else:
            # Mock connection for testing
            logger.info("Using mock Isaac Sim interface")
            self.connected = True
            return True
            
    def register_robot(self, robot_id: str, usd_path: str, prim_path: str):
        """
        Register a robot model in the simulation.
        
        Args:
            robot_id: Unique identifier for the robot
            usd_path: Path to the robot USD file
            prim_path: USD prim path in the stage
        """
        self._robot_prims[robot_id] = prim_path
        
        if self._has_isaac:
            try:
                from omni.isaac.core.utils.stage import add_reference_to_stage
                add_reference_to_stage(usd_path, prim_path)
                logger.info(f"Registered robot {robot_id} at {prim_path}")
            except Exception as e:
                logger.error(f"Failed to register robot: {e}")
        else:
            logger.info(f"Mock: Registered robot {robot_id}")
            
    def update_robot_pose(self, robot_id: str, pose: RobotPose):
        """Update robot pose in simulation."""
        if robot_id not in self._robot_prims:
            logger.warning(f"Robot {robot_id} not registered")
            return
            
        prim_path = self._robot_prims[robot_id]
        
        if self._has_isaac:
            try:
                from pxr import Gf, UsdGeom
                from omni.isaac.core.utils.rotations import euler_angles_to_quat
                
                # Convert pose to USD transform
                position = Gf.Vec3d(pose.x, pose.y, pose.z)
                quat = euler_angles_to_quat(
                    np.array([pose.roll, pose.pitch, pose.yaw])
                )
                
                # Update prim transform
                # This would be done through the World/Robot API
                pass
            except Exception as e:
                logger.error(f"Failed to update pose: {e}")
        else:
            # Mock update - just log
            logger.debug(f"Mock: Updated {robot_id} pose to ({pose.x:.2f}, {pose.y:.2f}, {pose.yaw:.2f})")
            
    def update_robot_joints(self, robot_id: str, joints: JointStates):
        """Update robot joint positions in simulation."""
        if robot_id not in self._robot_prims:
            return
            
        if self._has_isaac:
            try:
                # Update joint drives
                pass
            except Exception as e:
                logger.error(f"Failed to update joints: {e}")
        else:
            logger.debug(f"Mock: Updated {robot_id} joints")
            
    def get_sim_pose(self, robot_id: str) -> Optional[RobotPose]:
        """Get current pose from simulation."""
        if robot_id not in self._robot_prims:
            return None
            
        if self._has_isaac:
            try:
                # Read from USD stage
                pass
            except Exception as e:
                logger.error(f"Failed to get sim pose: {e}")
                return None
        else:
            # Mock: return a default pose
            return RobotPose(timestamp=time.time())


# =============================================================================
# ROS 2 Twin Sync Node
# =============================================================================

if HAS_ROS:
    class TwinSyncNode(Node):
        """
        ROS 2 node for bidirectional twin synchronization.
        
        Subscribes to physical robot state and publishes to simulation,
        or vice versa depending on sync mode.
        """
        
        def __init__(
            self,
            robot_id: str,
            config: SyncConfig,
            isaac_interface: IsaacSimInterface
        ):
            super().__init__(f'twin_sync_{robot_id}')
            
            self.robot_id = robot_id
            self.config = config
            self.isaac = isaac_interface
            
            # State
            self.physical_state = TwinState(robot_id=robot_id, is_physical=True)
            self.sim_state = TwinState(robot_id=robot_id, is_physical=False)
            self.sync_mode = SyncMode.MIRROR
            
            # Predictor for SHADOW mode
            self.predictor = StatePredictor(config.history_size)
            
            # QoS profiles
            self.sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            
            self.control_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            
            # Subscribers - Physical robot state
            self.odom_sub = self.create_subscription(
                Odometry,
                f'/{robot_id}/odom',
                self._odom_callback,
                self.sensor_qos
            )
            
            self.joint_sub = self.create_subscription(
                JointState,
                f'/{robot_id}/joint_states',
                self._joint_callback,
                self.sensor_qos
            )
            
            # Publishers - Sim commands (COMMAND mode)
            self.sim_cmd_pub = self.create_publisher(
                Twist,
                f'/{robot_id}/sim_cmd_vel',
                self.control_qos
            )
            
            # Publishers - Twin state
            self.twin_state_pub = self.create_publisher(
                String,
                f'/{robot_id}/twin_state',
                self.control_qos
            )
            
            # Subscriber - Mode changes
            self.mode_sub = self.create_subscription(
                String,
                f'/{robot_id}/twin_mode',
                self._mode_callback,
                self.control_qos
            )
            
            # Sync timer
            period = 1.0 / config.sync_rate_hz
            self.sync_timer = self.create_timer(period, self._sync_loop)
            
            # TF broadcaster
            self.tf_broadcaster = TransformBroadcaster(self)
            
            # Statistics
            self.sync_count = 0
            self.total_latency = 0.0
            
            self.get_logger().info(f"TwinSync node initialized for {robot_id}")
            
        def _odom_callback(self, msg: Odometry):
            """Handle odometry from physical robot."""
            pose = msg.pose.pose
            twist = msg.twist.twist
            
            # Extract Euler angles from quaternion
            quat = pose.orientation
            roll, pitch, yaw = self._quat_to_euler(quat.x, quat.y, quat.z, quat.w)
            
            self.physical_state.pose = RobotPose(
                x=pose.position.x,
                y=pose.position.y,
                z=pose.position.z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                timestamp=time.time(),
                frame_id=msg.header.frame_id
            )
            
            self.physical_state.velocity = RobotVelocity(
                linear_x=twist.linear.x,
                linear_y=twist.linear.y,
                linear_z=twist.linear.z,
                angular_x=twist.angular.x,
                angular_y=twist.angular.y,
                angular_z=twist.angular.z,
                timestamp=time.time()
            )
            
            # Add to predictor
            self.predictor.add_state(
                self.physical_state.pose,
                self.physical_state.velocity
            )
            
        def _joint_callback(self, msg: JointState):
            """Handle joint states from physical robot."""
            self.physical_state.joints = JointStates(
                names=list(msg.name),
                positions=list(msg.position),
                velocities=list(msg.velocity) if msg.velocity else [],
                efforts=list(msg.effort) if msg.effort else [],
                timestamp=time.time()
            )
            
        def _mode_callback(self, msg: String):
            """Handle sync mode changes."""
            try:
                new_mode = SyncMode(msg.data)
                self.sync_mode = new_mode
                self.get_logger().info(f"Sync mode changed to: {new_mode.value}")
            except ValueError:
                self.get_logger().error(f"Invalid sync mode: {msg.data}")
                
        def _sync_loop(self):
            """Main synchronization loop."""
            sync_start = time.time()
            
            if self.sync_mode == SyncMode.PAUSED:
                return
                
            if self.sync_mode == SyncMode.MIRROR:
                self._do_mirror_sync()
            elif self.sync_mode == SyncMode.SHADOW:
                self._do_shadow_sync()
            elif self.sync_mode == SyncMode.COMMAND:
                self._do_command_sync()
            elif self.sync_mode == SyncMode.HYBRID:
                self._do_hybrid_sync()
                
            # Update statistics
            latency = (time.time() - sync_start) * 1000
            self.sync_count += 1
            self.total_latency += latency
            self.physical_state.sync_latency_ms = latency
            
            # Publish twin state
            self._publish_twin_state()
            
            # Broadcast TF
            self._broadcast_tf()
            
        def _do_mirror_sync(self):
            """Mirror mode: Sim follows physical exactly."""
            self.isaac.update_robot_pose(self.robot_id, self.physical_state.pose)
            
            if self.physical_state.joints:
                self.isaac.update_robot_joints(self.robot_id, self.physical_state.joints)
                
        def _do_shadow_sync(self):
            """Shadow mode: Sim predicts physical robot's future state."""
            predicted = self.predictor.predict(self.config.prediction_horizon_s)
            
            if predicted:
                self.isaac.update_robot_pose(self.robot_id, predicted)
            else:
                # Fall back to mirror
                self._do_mirror_sync()
                
        def _do_command_sync(self):
            """Command mode: Physical robot follows sim commands."""
            sim_pose = self.isaac.get_sim_pose(self.robot_id)
            
            if sim_pose:
                # Calculate velocity command to reach sim pose
                cmd = self._calculate_cmd_vel(
                    self.physical_state.pose,
                    sim_pose
                )
                self.sim_cmd_pub.publish(cmd)
                
        def _do_hybrid_sync(self):
            """Hybrid mode: Bidirectional with conflict resolution."""
            sim_pose = self.isaac.get_sim_pose(self.robot_id)
            physical_pose = self.physical_state.pose
            
            if sim_pose is None:
                # No sim data, fall back to mirror
                self._do_mirror_sync()
                return
                
            # Check for conflict (significant difference)
            distance = physical_pose.distance_to(sim_pose)
            
            if distance < self.config.position_threshold_m:
                # No conflict, mirror physical to sim
                self._do_mirror_sync()
            else:
                # Resolve conflict
                resolved_pose = self._resolve_conflict(physical_pose, sim_pose)
                
                # Update both
                self.isaac.update_robot_pose(self.robot_id, resolved_pose)
                
        def _resolve_conflict(
            self,
            physical: RobotPose,
            sim: RobotPose
        ) -> RobotPose:
            """Resolve conflict between physical and sim poses."""
            resolution = self.config.conflict_resolution
            
            if resolution == ConflictResolution.PHYSICAL_PRIORITY:
                return physical
            elif resolution == ConflictResolution.SIM_PRIORITY:
                return sim
            elif resolution == ConflictResolution.NEWEST_WINS:
                return physical if physical.timestamp > sim.timestamp else sim
            elif resolution == ConflictResolution.AVERAGE:
                return physical.blend(sim, self.config.blend_factor)
            else:
                return physical
                
        def _calculate_cmd_vel(
            self,
            current: RobotPose,
            target: RobotPose
        ) -> Twist:
            """Calculate velocity command to move from current to target."""
            cmd = Twist()
            
            # Position error
            dx = target.x - current.x
            dy = target.y - current.y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Angle to target
            angle_to_target = np.arctan2(dy, dx)
            angle_error = angle_to_target - current.yaw
            
            # Normalize angle
            while angle_error > np.pi:
                angle_error -= 2 * np.pi
            while angle_error < -np.pi:
                angle_error += 2 * np.pi
                
            # Simple proportional control
            if abs(angle_error) > 0.1:
                # Turn towards target
                cmd.angular.z = 0.5 * angle_error
            else:
                # Move forward
                cmd.linear.x = min(0.5, distance)
                cmd.angular.z = 0.2 * angle_error
                
            return cmd
            
        def _publish_twin_state(self):
            """Publish current twin state as JSON."""
            state_dict = {
                'physical': self.physical_state.to_dict(),
                'sim': self.sim_state.to_dict(),
                'sync_mode': self.sync_mode.value,
                'sync_count': self.sync_count,
                'avg_latency_ms': self.total_latency / max(1, self.sync_count)
            }
            
            msg = String()
            msg.data = json.dumps(state_dict)
            self.twin_state_pub.publish(msg)
            
        def _broadcast_tf(self):
            """Broadcast transform for the digital twin."""
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = f'{self.robot_id}_twin'
            
            pose = self.physical_state.pose
            t.transform.translation.x = pose.x
            t.transform.translation.y = pose.y
            t.transform.translation.z = pose.z
            
            qx, qy, qz, qw = self._euler_to_quat(pose.roll, pose.pitch, pose.yaw)
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            
            self.tf_broadcaster.sendTransform(t)
            
        @staticmethod
        def _quat_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
            """Convert quaternion to Euler angles (roll, pitch, yaw)."""
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp)
            else:
                pitch = np.arcsin(sinp)
                
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return roll, pitch, yaw
            
        @staticmethod
        def _euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
            """Convert Euler angles to quaternion."""
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            
            return x, y, z, w


# =============================================================================
# Twin Sync Manager
# =============================================================================

class TwinSyncManager:
    """
    High-level manager for digital twin synchronization.
    
    Coordinates multiple robot twins and provides:
    - Fleet-wide sync management
    - Metrics collection
    - Mode switching
    - Error recovery
    
    Example Usage:
        >>> manager = TwinSyncManager()
        >>> 
        >>> # Register robots
        >>> manager.register_robot(
        ...     robot_id="robot_1",
        ...     usd_path="/models/carter_v2.usd",
        ...     initial_mode=SyncMode.MIRROR
        ... )
        >>> 
        >>> # Start synchronization
        >>> await manager.start()
        >>> 
        >>> # Change mode
        >>> manager.set_mode("robot_1", SyncMode.SHADOW)
        >>> 
        >>> # Get status
        >>> status = manager.get_fleet_status()
    """
    
    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig()
        self.isaac = IsaacSimInterface(
            host=self.config.isaac_host,
            port=self.config.isaac_port
        )
        
        self._robots: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._redis_client = None
        
        # Initialize Redis if enabled
        if self.config.redis_enabled and HAS_REDIS:
            try:
                self._redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    decode_responses=True
                )
                self._redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self._redis_client = None
                
    def register_robot(
        self,
        robot_id: str,
        usd_path: str,
        prim_path: Optional[str] = None,
        initial_mode: SyncMode = SyncMode.MIRROR
    ):
        """Register a robot for twin synchronization."""
        prim_path = prim_path or f"/World/Robots/{robot_id}"
        
        self._robots[robot_id] = {
            'usd_path': usd_path,
            'prim_path': prim_path,
            'mode': initial_mode,
            'state': TwinState(robot_id=robot_id),
            'node': None
        }
        
        # Register with Isaac Sim
        self.isaac.register_robot(robot_id, usd_path, prim_path)
        
        logger.info(f"Registered robot {robot_id} for twin sync")
        
    async def start(self):
        """Start twin synchronization for all registered robots."""
        if self._running:
            return
            
        # Connect to Isaac Sim
        if not await self.isaac.connect():
            raise RuntimeError("Failed to connect to Isaac Sim")
            
        self._running = True
        
        # Start ROS nodes if available
        if HAS_ROS:
            rclpy.init()
            
            for robot_id, robot_data in self._robots.items():
                node = TwinSyncNode(
                    robot_id=robot_id,
                    config=self.config,
                    isaac_interface=self.isaac
                )
                robot_data['node'] = node
                
            # Spin in background
            self._spin_thread = threading.Thread(
                target=self._ros_spin_loop,
                daemon=True
            )
            self._spin_thread.start()
            
        logger.info("Twin synchronization started")
        
    def _ros_spin_loop(self):
        """Background ROS spinning."""
        executor = rclpy.executors.MultiThreadedExecutor()
        
        for robot_data in self._robots.values():
            if robot_data['node']:
                executor.add_node(robot_data['node'])
                
        while self._running and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            
    async def stop(self):
        """Stop twin synchronization."""
        self._running = False
        
        if HAS_ROS:
            for robot_data in self._robots.values():
                if robot_data['node']:
                    robot_data['node'].destroy_node()
            rclpy.shutdown()
            
        logger.info("Twin synchronization stopped")
        
    def set_mode(self, robot_id: str, mode: SyncMode):
        """Set sync mode for a specific robot."""
        if robot_id not in self._robots:
            raise ValueError(f"Unknown robot: {robot_id}")
            
        self._robots[robot_id]['mode'] = mode
        
        if HAS_ROS and self._robots[robot_id]['node']:
            self._robots[robot_id]['node'].sync_mode = mode
            
        # Cache in Redis
        if self._redis_client:
            self._redis_client.hset(
                f"twin:{robot_id}",
                "mode",
                mode.value
            )
            
        logger.info(f"Set {robot_id} sync mode to {mode.value}")
        
    def set_fleet_mode(self, mode: SyncMode):
        """Set sync mode for all robots."""
        for robot_id in self._robots:
            self.set_mode(robot_id, mode)
            
    def get_robot_state(self, robot_id: str) -> Optional[TwinState]:
        """Get current state of a robot twin."""
        if robot_id not in self._robots:
            return None
            
        if HAS_ROS and self._robots[robot_id]['node']:
            return self._robots[robot_id]['node'].physical_state
            
        return self._robots[robot_id]['state']
        
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get status of all robot twins."""
        status = {
            'total_robots': len(self._robots),
            'running': self._running,
            'robots': {}
        }
        
        for robot_id, robot_data in self._robots.items():
            state = self.get_robot_state(robot_id)
            
            status['robots'][robot_id] = {
                'mode': robot_data['mode'].value,
                'pose': state.pose.to_dict() if state else None,
                'last_sync': state.last_sync if state else 0,
                'latency_ms': state.sync_latency_ms if state else 0
            }
            
        return status
        
    def get_sync_metrics(self) -> Dict[str, Any]:
        """Get synchronization performance metrics."""
        metrics = {
            'total_syncs': 0,
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
            'robots': {}
        }
        
        total_latency = 0
        total_syncs = 0
        
        for robot_id, robot_data in self._robots.items():
            if HAS_ROS and robot_data['node']:
                node = robot_data['node']
                sync_count = node.sync_count
                avg_lat = node.total_latency / max(1, sync_count)
                
                metrics['robots'][robot_id] = {
                    'sync_count': sync_count,
                    'avg_latency_ms': avg_lat
                }
                
                total_syncs += sync_count
                total_latency += node.total_latency
                
        if total_syncs > 0:
            metrics['total_syncs'] = total_syncs
            metrics['avg_latency_ms'] = total_latency / total_syncs
            
        return metrics


# =============================================================================
# Standalone Demo
# =============================================================================

async def demo_twin_sync():
    """Demonstrate twin synchronization (standalone mode)."""
    print("=== Digital Twin Synchronization Demo ===\n")
    
    # Create manager
    config = SyncConfig(
        sync_rate_hz=30.0,
        prediction_horizon_s=0.5
    )
    manager = TwinSyncManager(config)
    
    # Register a robot
    manager.register_robot(
        robot_id="carter_1",
        usd_path="/models/carter_v2.usd",
        initial_mode=SyncMode.MIRROR
    )
    
    print("Registered robot: carter_1")
    print(f"Sync config:")
    print(f"  - Rate: {config.sync_rate_hz} Hz")
    print(f"  - Max latency: {config.max_latency_ms} ms")
    print(f"  - Prediction horizon: {config.prediction_horizon_s} s")
    
    # Simulate state updates
    predictor = StatePredictor()
    
    print("\n--- Simulating robot movement ---")
    
    for i in range(10):
        # Simulate moving robot
        pose = RobotPose(
            x=i * 0.1,
            y=np.sin(i * 0.5) * 0.5,
            yaw=i * 0.1,
            timestamp=time.time()
        )
        
        velocity = RobotVelocity(
            linear_x=0.1,
            linear_y=np.cos(i * 0.5) * 0.25,
            angular_z=0.1
        )
        
        predictor.add_state(pose, velocity)
        
        # Predict future
        predicted = predictor.predict(0.5)
        
        if predicted:
            print(f"Step {i}: Pose ({pose.x:.2f}, {pose.y:.2f}) -> "
                  f"Predicted ({predicted.x:.2f}, {predicted.y:.2f})")
                  
        await asyncio.sleep(0.1)
        
    print("\n--- Available Sync Modes ---")
    for mode in SyncMode:
        print(f"  - {mode.value}: ", end="")
        descriptions = {
            SyncMode.MIRROR: "Simulation mirrors physical robot exactly",
            SyncMode.SHADOW: "Simulation predicts physical robot's future",
            SyncMode.COMMAND: "Physical robot follows simulation commands",
            SyncMode.HYBRID: "Bidirectional with conflict resolution",
            SyncMode.PAUSED: "Synchronization disabled"
        }
        print(descriptions.get(mode, ""))
        
    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(demo_twin_sync())
