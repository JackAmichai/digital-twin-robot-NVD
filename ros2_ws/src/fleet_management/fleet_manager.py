#!/usr/bin/env python3
"""
Fleet Manager - Multi-Robot Coordination System

This module manages multiple AMRs (Autonomous Mobile Robots) with:
- Robot registration and tracking
- Traffic coordination at intersections
- Collision avoidance between robots
- Task allocation and load balancing
- Robot-to-robot communication via ROS 2 namespacing

Example Usage:
    # Each robot runs with its own namespace
    ros2 run fleet_management fleet_manager --ros-args -r __ns:=/robot_1
    ros2 run fleet_management fleet_manager --ros-args -r __ns:=/robot_2
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Bool
from visualization_msgs.msg import Marker, MarkerArray
import json
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class RobotState(Enum):
    """Robot operational states"""
    IDLE = "idle"
    NAVIGATING = "navigating"
    WAITING = "waiting"          # Waiting for traffic clearance
    CHARGING = "charging"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RobotInfo:
    """Information about a robot in the fleet"""
    robot_id: str
    namespace: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, theta
    velocity: Tuple[float, float] = (0.0, 0.0)  # linear, angular
    state: RobotState = RobotState.IDLE
    current_goal: Optional[Tuple[float, float]] = None
    battery_level: float = 100.0
    last_heartbeat: float = 0.0
    priority: int = 1  # Higher = more priority
    planned_path: List[Tuple[float, float]] = field(default_factory=list)


@dataclass 
class TrafficZone:
    """
    Traffic zone for intersection management
    
    Example zones in a warehouse:
    - Intersections where paths cross
    - Narrow aisles (single robot width)
    - Loading/unloading areas
    """
    zone_id: str
    center: Tuple[float, float]
    radius: float
    max_robots: int = 1  # Usually 1 for intersections
    current_robots: List[str] = field(default_factory=list)
    queue: List[str] = field(default_factory=list)


class FleetManager(Node):
    """
    Central fleet management node for multi-robot coordination
    
    Key Features:
    1. Robot Registration: Robots announce themselves on startup
    2. Position Tracking: Subscribes to all robot odometry
    3. Traffic Control: Manages access to shared zones
    4. Collision Avoidance: Monitors inter-robot distances
    5. Task Distribution: Assigns goals to available robots
    """
    
    # Safety parameters
    COLLISION_DISTANCE = 1.5      # Minimum distance between robots (meters)
    WARNING_DISTANCE = 3.0        # Distance to start slowing down
    HEARTBEAT_TIMEOUT = 5.0       # Seconds before robot considered offline
    
    def __init__(self):
        super().__init__('fleet_manager')
        
        # Fleet state
        self.robots: Dict[str, RobotInfo] = {}
        self.traffic_zones: Dict[str, TrafficZone] = {}
        self.lock = threading.Lock()
        
        # QoS for reliable communication
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # =====================================================
        # Publishers - Fleet-wide communication
        # =====================================================
        
        # Broadcast fleet status to all robots
        self.fleet_status_pub = self.create_publisher(
            String, '/fleet/status', reliable_qos
        )
        
        # Emergency stop for all robots
        self.emergency_stop_pub = self.create_publisher(
            Bool, '/fleet/emergency_stop', reliable_qos
        )
        
        # Visualization for Foxglove/RViz
        self.marker_pub = self.create_publisher(
            MarkerArray, '/fleet/visualization', 10
        )
        
        # =====================================================
        # Subscribers - Listen to all robots
        # =====================================================
        
        # Robot registration (robots announce themselves)
        self.create_subscription(
            String, '/fleet/register', self.handle_registration, reliable_qos
        )
        
        # Robot heartbeats
        self.create_subscription(
            String, '/fleet/heartbeat', self.handle_heartbeat, 10
        )
        
        # Traffic zone requests
        self.create_subscription(
            String, '/fleet/zone_request', self.handle_zone_request, reliable_qos
        )
        
        # =====================================================
        # Timers - Periodic tasks
        # =====================================================
        
        # Check for collisions every 100ms
        self.create_timer(0.1, self.check_collisions)
        
        # Publish fleet status every second
        self.create_timer(1.0, self.publish_fleet_status)
        
        # Check robot heartbeats every 2 seconds
        self.create_timer(2.0, self.check_heartbeats)
        
        # Update visualization every 500ms
        self.create_timer(0.5, self.publish_visualization)
        
        # Initialize default traffic zones
        self._init_traffic_zones()
        
        self.get_logger().info('Fleet Manager initialized')
    
    def _init_traffic_zones(self):
        """
        Initialize traffic zones for the warehouse
        
        Example warehouse layout:
        
            Zone A          Zone B          Zone C
              │               │               │
        ──────┼───────────────┼───────────────┼──────
              │               │               │
            Zone D          Zone E          Zone F
        
        Each intersection is a traffic zone where only
        one robot can pass at a time.
        """
        # Define intersection zones (customize for your warehouse)
        zones = [
            TrafficZone("intersection_A", (5.0, 5.0), 1.5, max_robots=1),
            TrafficZone("intersection_B", (10.0, 5.0), 1.5, max_robots=1),
            TrafficZone("intersection_C", (15.0, 5.0), 1.5, max_robots=1),
            TrafficZone("intersection_D", (5.0, 10.0), 1.5, max_robots=1),
            TrafficZone("intersection_E", (10.0, 10.0), 1.5, max_robots=1),
            TrafficZone("intersection_F", (15.0, 10.0), 1.5, max_robots=1),
            # Narrow aisle - only one robot at a time
            TrafficZone("narrow_aisle_1", (7.5, 7.5), 3.0, max_robots=1),
            # Loading dock - can have 2 robots
            TrafficZone("loading_dock", (0.0, 5.0), 2.0, max_robots=2),
        ]
        
        for zone in zones:
            self.traffic_zones[zone.zone_id] = zone
            self.get_logger().info(f'Traffic zone registered: {zone.zone_id}')
    
    def handle_registration(self, msg: String):
        """
        Handle robot registration
        
        Message format (JSON):
        {
            "robot_id": "amr_001",
            "namespace": "/robot_1",
            "priority": 1,
            "capabilities": ["navigation", "manipulation"]
        }
        """
        try:
            data = json.loads(msg.data)
            robot_id = data['robot_id']
            namespace = data['namespace']
            
            with self.lock:
                if robot_id not in self.robots:
                    # New robot - create subscriber for its odometry
                    self.robots[robot_id] = RobotInfo(
                        robot_id=robot_id,
                        namespace=namespace,
                        priority=data.get('priority', 1),
                        last_heartbeat=time.time()
                    )
                    
                    # Subscribe to this robot's odometry
                    # Example: /robot_1/odom
                    self.create_subscription(
                        Odometry,
                        f'{namespace}/odom',
                        lambda msg, rid=robot_id: self.handle_robot_odom(msg, rid),
                        10
                    )
                    
                    # Subscribe to this robot's planned path
                    # Example: /robot_1/plan
                    self.create_subscription(
                        Path,
                        f'{namespace}/plan',
                        lambda msg, rid=robot_id: self.handle_robot_path(msg, rid),
                        10
                    )
                    
                    # Create velocity command publisher for this robot
                    # Used for emergency stops and speed adjustments
                    setattr(self, f'cmd_vel_pub_{robot_id}',
                        self.create_publisher(Twist, f'{namespace}/cmd_vel_fleet', 10)
                    )
                    
                    self.get_logger().info(
                        f'Robot registered: {robot_id} (namespace: {namespace})'
                    )
                else:
                    # Robot reconnecting - update heartbeat
                    self.robots[robot_id].last_heartbeat = time.time()
                    self.get_logger().info(f'Robot reconnected: {robot_id}')
                    
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid registration message: {e}')
    
    def handle_robot_odom(self, msg: Odometry, robot_id: str):
        """
        Update robot position from odometry
        
        This is called for each robot's odometry topic.
        Example: /robot_1/odom, /robot_2/odom, etc.
        """
        with self.lock:
            if robot_id in self.robots:
                robot = self.robots[robot_id]
                
                # Extract position
                pos = msg.pose.pose.position
                orient = msg.pose.pose.orientation
                
                # Calculate yaw from quaternion
                # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
                cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                robot.position = (pos.x, pos.y, yaw)
                
                # Extract velocity
                robot.velocity = (
                    msg.twist.twist.linear.x,
                    msg.twist.twist.angular.z
                )
    
    def handle_robot_path(self, msg: Path, robot_id: str):
        """Store robot's planned path for trajectory prediction"""
        with self.lock:
            if robot_id in self.robots:
                self.robots[robot_id].planned_path = [
                    (pose.pose.position.x, pose.pose.position.y)
                    for pose in msg.poses
                ]
    
    def handle_heartbeat(self, msg: String):
        """
        Handle robot heartbeat messages
        
        Message format: {"robot_id": "amr_001", "state": "navigating", "battery": 85.5}
        """
        try:
            data = json.loads(msg.data)
            robot_id = data['robot_id']
            
            with self.lock:
                if robot_id in self.robots:
                    robot = self.robots[robot_id]
                    robot.last_heartbeat = time.time()
                    robot.state = RobotState(data.get('state', 'idle'))
                    robot.battery_level = data.get('battery', 100.0)
                    
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f'Invalid heartbeat: {e}')
    
    def handle_zone_request(self, msg: String):
        """
        Handle traffic zone access requests
        
        Message format:
        {
            "robot_id": "amr_001",
            "zone_id": "intersection_A",
            "action": "enter" | "exit"
        }
        
        Response published to /fleet/zone_response:
        {
            "robot_id": "amr_001",
            "zone_id": "intersection_A",
            "granted": true,
            "wait_time": 0
        }
        """
        try:
            data = json.loads(msg.data)
            robot_id = data['robot_id']
            zone_id = data['zone_id']
            action = data['action']
            
            response = {
                'robot_id': robot_id,
                'zone_id': zone_id,
                'granted': False,
                'wait_time': 0
            }
            
            with self.lock:
                if zone_id not in self.traffic_zones:
                    self.get_logger().warn(f'Unknown zone: {zone_id}')
                    return
                
                zone = self.traffic_zones[zone_id]
                
                if action == 'enter':
                    # Check if robot can enter zone
                    if len(zone.current_robots) < zone.max_robots:
                        # Grant access
                        zone.current_robots.append(robot_id)
                        response['granted'] = True
                        self.get_logger().info(
                            f'Robot {robot_id} granted access to {zone_id}'
                        )
                    else:
                        # Add to queue
                        if robot_id not in zone.queue:
                            zone.queue.append(robot_id)
                        response['wait_time'] = zone.queue.index(robot_id) + 1
                        self.get_logger().info(
                            f'Robot {robot_id} queued for {zone_id} '
                            f'(position {response["wait_time"]})'
                        )
                
                elif action == 'exit':
                    # Robot leaving zone
                    if robot_id in zone.current_robots:
                        zone.current_robots.remove(robot_id)
                        self.get_logger().info(
                            f'Robot {robot_id} exited {zone_id}'
                        )
                        
                        # Grant access to next in queue
                        if zone.queue and len(zone.current_robots) < zone.max_robots:
                            next_robot = zone.queue.pop(0)
                            zone.current_robots.append(next_robot)
                            # Notify the waiting robot
                            self._notify_zone_access(next_robot, zone_id)
            
            # Publish response
            response_pub = self.create_publisher(
                String, '/fleet/zone_response', 10
            )
            response_msg = String()
            response_msg.data = json.dumps(response)
            response_pub.publish(response_msg)
            
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f'Invalid zone request: {e}')
    
    def _notify_zone_access(self, robot_id: str, zone_id: str):
        """Notify a robot that it has been granted zone access"""
        response = {
            'robot_id': robot_id,
            'zone_id': zone_id,
            'granted': True,
            'wait_time': 0
        }
        response_pub = self.create_publisher(String, '/fleet/zone_response', 10)
        msg = String()
        msg.data = json.dumps(response)
        response_pub.publish(msg)
        self.get_logger().info(f'Notified {robot_id}: access granted to {zone_id}')
    
    def check_collisions(self):
        """
        Check for potential collisions between robots
        
        Algorithm:
        1. For each pair of robots, calculate distance
        2. If distance < COLLISION_DISTANCE: Emergency stop both
        3. If distance < WARNING_DISTANCE: Slow down lower priority robot
        
        This runs every 100ms for fast response.
        """
        with self.lock:
            robot_list = list(self.robots.values())
        
        for i, robot1 in enumerate(robot_list):
            for robot2 in robot_list[i+1:]:
                distance = self._calculate_distance(
                    robot1.position[:2], 
                    robot2.position[:2]
                )
                
                if distance < self.COLLISION_DISTANCE:
                    # EMERGENCY: Robots too close!
                    self._emergency_stop_robot(robot1.robot_id)
                    self._emergency_stop_robot(robot2.robot_id)
                    self.get_logger().error(
                        f'COLLISION WARNING: {robot1.robot_id} and {robot2.robot_id} '
                        f'are {distance:.2f}m apart! Emergency stop triggered.'
                    )
                
                elif distance < self.WARNING_DISTANCE:
                    # Slow down the lower priority robot
                    if robot1.priority < robot2.priority:
                        self._slow_down_robot(robot1.robot_id, distance)
                    else:
                        self._slow_down_robot(robot2.robot_id, distance)
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                           pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _emergency_stop_robot(self, robot_id: str):
        """Send emergency stop command to a robot"""
        with self.lock:
            if robot_id in self.robots:
                self.robots[robot_id].state = RobotState.EMERGENCY_STOP
        
        # Send zero velocity command
        cmd_pub = getattr(self, f'cmd_vel_pub_{robot_id}', None)
        if cmd_pub:
            stop_cmd = Twist()  # All zeros = stop
            cmd_pub.publish(stop_cmd)
    
    def _slow_down_robot(self, robot_id: str, distance: float):
        """Reduce robot speed based on proximity"""
        # Calculate speed factor: 0 at COLLISION_DISTANCE, 1 at WARNING_DISTANCE
        speed_factor = (distance - self.COLLISION_DISTANCE) / \
                       (self.WARNING_DISTANCE - self.COLLISION_DISTANCE)
        speed_factor = max(0.1, min(1.0, speed_factor))  # Clamp to [0.1, 1.0]
        
        with self.lock:
            if robot_id in self.robots:
                robot = self.robots[robot_id]
                if robot.state == RobotState.NAVIGATING:
                    robot.state = RobotState.WAITING
        
        self.get_logger().debug(
            f'Slowing {robot_id} to {speed_factor*100:.0f}% speed'
        )
    
    def check_heartbeats(self):
        """Check for offline robots (missed heartbeats)"""
        current_time = time.time()
        
        with self.lock:
            for robot_id, robot in list(self.robots.items()):
                if current_time - robot.last_heartbeat > self.HEARTBEAT_TIMEOUT:
                    self.get_logger().warn(
                        f'Robot {robot_id} heartbeat timeout - marking as ERROR'
                    )
                    robot.state = RobotState.ERROR
    
    def publish_fleet_status(self):
        """
        Publish fleet status for monitoring
        
        Example output:
        {
            "timestamp": 1703500000.0,
            "total_robots": 3,
            "active_robots": 2,
            "robots": {
                "amr_001": {"state": "navigating", "position": [5.2, 3.1], "battery": 85},
                "amr_002": {"state": "idle", "position": [0.0, 0.0], "battery": 100}
            },
            "traffic_zones": {
                "intersection_A": {"occupied_by": ["amr_001"], "queue": []}
            }
        }
        """
        with self.lock:
            status = {
                'timestamp': time.time(),
                'total_robots': len(self.robots),
                'active_robots': sum(
                    1 for r in self.robots.values() 
                    if r.state not in [RobotState.ERROR, RobotState.EMERGENCY_STOP]
                ),
                'robots': {
                    rid: {
                        'state': r.state.value,
                        'position': list(r.position[:2]),
                        'battery': r.battery_level,
                        'current_goal': r.current_goal
                    }
                    for rid, r in self.robots.items()
                },
                'traffic_zones': {
                    zid: {
                        'occupied_by': z.current_robots,
                        'queue': z.queue
                    }
                    for zid, z in self.traffic_zones.items()
                }
            }
        
        msg = String()
        msg.data = json.dumps(status)
        self.fleet_status_pub.publish(msg)
    
    def publish_visualization(self):
        """
        Publish visualization markers for Foxglove/RViz
        
        Displays:
        - Robot positions as colored spheres
        - Traffic zones as cylinders
        - Connections between nearby robots
        """
        marker_array = MarkerArray()
        
        with self.lock:
            # Robot markers
            for i, (robot_id, robot) in enumerate(self.robots.items()):
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = 'robots'
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = robot.position[0]
                marker.pose.position.y = robot.position[1]
                marker.pose.position.z = 0.5
                
                marker.scale.x = 0.8
                marker.scale.y = 0.8
                marker.scale.z = 0.8
                
                # Color based on state
                if robot.state == RobotState.NAVIGATING:
                    marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
                elif robot.state == RobotState.WAITING:
                    marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
                elif robot.state == RobotState.ERROR:
                    marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
                else:
                    marker.color.r, marker.color.g, marker.color.b = 0.5, 0.5, 0.5
                marker.color.a = 0.8
                
                marker_array.markers.append(marker)
            
            # Traffic zone markers
            for i, (zone_id, zone) in enumerate(self.traffic_zones.items()):
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = 'traffic_zones'
                marker.id = i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                marker.pose.position.x = zone.center[0]
                marker.pose.position.y = zone.center[1]
                marker.pose.position.z = 0.1
                
                marker.scale.x = zone.radius * 2
                marker.scale.y = zone.radius * 2
                marker.scale.z = 0.1
                
                # Red if occupied, green if free
                if zone.current_robots:
                    marker.color.r, marker.color.g, marker.color.b = 1.0, 0.3, 0.3
                else:
                    marker.color.r, marker.color.g, marker.color.b = 0.3, 1.0, 0.3
                marker.color.a = 0.3
                
                marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = FleetManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
