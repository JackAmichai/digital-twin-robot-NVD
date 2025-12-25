#!/usr/bin/env python3
"""
Robot Agent - Individual Robot Controller for Fleet

This module runs on each robot and handles:
- Registration with the fleet manager
- Heartbeat sending
- Traffic zone access requests
- Collision avoidance response
- Namespaced communication

Example Launch:
    # Robot 1
    ros2 run fleet_management robot_agent --ros-args \
        -p robot_id:=amr_001 \
        -p priority:=1 \
        -r __ns:=/robot_1
    
    # Robot 2 (higher priority)
    ros2 run fleet_management robot_agent --ros-args \
        -p robot_id:=amr_002 \
        -p priority:=2 \
        -r __ns:=/robot_2
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String, Bool
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import json
import time
import threading
from enum import Enum
from typing import Optional, Tuple, List
import math


class NavigationState(Enum):
    """Robot navigation states"""
    IDLE = "idle"
    NAVIGATING = "navigating"
    WAITING_FOR_ZONE = "waiting"
    GOAL_REACHED = "goal_reached"
    FAILED = "failed"


class RobotAgent(Node):
    """
    Individual robot agent that communicates with the fleet manager
    
    Key Responsibilities:
    1. Register with fleet manager on startup
    2. Send periodic heartbeats
    3. Request zone access before entering traffic zones
    4. Respond to fleet commands (emergency stop, slow down)
    5. Report position and status
    
    Topics (in robot namespace, e.g., /robot_1/...):
    - Subscribes: /odom, /battery_state, /plan
    - Publishes: /cmd_vel
    
    Fleet Topics (global):
    - Publishes: /fleet/register, /fleet/heartbeat, /fleet/zone_request
    - Subscribes: /fleet/status, /fleet/zone_response, /fleet/emergency_stop
    """
    
    HEARTBEAT_INTERVAL = 1.0  # seconds
    ZONE_CHECK_DISTANCE = 3.0  # meters - check zones when this close
    
    def __init__(self):
        super().__init__('robot_agent')
        
        # Declare parameters
        self.declare_parameter('robot_id', 'amr_001')
        self.declare_parameter('priority', 1)
        self.declare_parameter('capabilities', ['navigation'])
        
        # Get parameters
        self.robot_id = self.get_parameter('robot_id').value
        self.priority = self.get_parameter('priority').value
        self.capabilities = self.get_parameter('capabilities').value
        self.namespace = self.get_namespace()
        
        # State
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.battery_level: float = 100.0
        self.nav_state = NavigationState.IDLE
        self.current_goal: Optional[Tuple[float, float]] = None
        self.planned_path: List[Tuple[float, float]] = []
        self.emergency_stop_active = False
        self.waiting_for_zone: Optional[str] = None
        self.granted_zones: List[str] = []
        
        # Traffic zones we know about (received from fleet)
        self.traffic_zones: dict = {}
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # =====================================================
        # Fleet Communication (Global Topics)
        # =====================================================
        
        # Registration publisher
        self.register_pub = self.create_publisher(
            String, '/fleet/register', reliable_qos
        )
        
        # Heartbeat publisher
        self.heartbeat_pub = self.create_publisher(
            String, '/fleet/heartbeat', 10
        )
        
        # Zone request publisher
        self.zone_request_pub = self.create_publisher(
            String, '/fleet/zone_request', reliable_qos
        )
        
        # Fleet status subscriber
        self.create_subscription(
            String, '/fleet/status', self.handle_fleet_status, 10
        )
        
        # Zone response subscriber
        self.create_subscription(
            String, '/fleet/zone_response', self.handle_zone_response, reliable_qos
        )
        
        # Emergency stop subscriber
        self.create_subscription(
            Bool, '/fleet/emergency_stop', self.handle_emergency_stop, reliable_qos
        )
        
        # Fleet velocity override (from fleet manager)
        self.create_subscription(
            Twist, 'cmd_vel_fleet', self.handle_fleet_cmd_vel, 10
        )
        
        # =====================================================
        # Robot-Specific Topics (Namespaced)
        # =====================================================
        
        # Odometry subscriber
        self.create_subscription(
            Odometry, 'odom', self.handle_odom, 10
        )
        
        # Battery state subscriber
        self.create_subscription(
            BatteryState, 'battery_state', self.handle_battery, 10
        )
        
        # Path from Nav2 planner
        self.create_subscription(
            Path, 'plan', self.handle_path, 10
        )
        
        # Velocity command publisher (for emergency stops)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # =====================================================
        # Timers
        # =====================================================
        
        # Heartbeat timer
        self.create_timer(self.HEARTBEAT_INTERVAL, self.send_heartbeat)
        
        # Zone proximity checker
        self.create_timer(0.5, self.check_zone_proximity)
        
        # Register with fleet manager
        self.create_timer(0.1, self._register_once, callback_group=None)
        self._registered = False
        
        self.get_logger().info(
            f'Robot Agent initialized: {self.robot_id} '
            f'(namespace: {self.namespace}, priority: {self.priority})'
        )
    
    def _register_once(self):
        """Register with fleet manager (runs once)"""
        if self._registered:
            return
        
        registration = {
            'robot_id': self.robot_id,
            'namespace': self.namespace,
            'priority': self.priority,
            'capabilities': self.capabilities
        }
        
        msg = String()
        msg.data = json.dumps(registration)
        self.register_pub.publish(msg)
        
        self._registered = True
        self.get_logger().info(f'Registered with fleet manager')
    
    def handle_odom(self, msg: Odometry):
        """Update position from odometry"""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        
        # Calculate yaw from quaternion
        siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.position = (pos.x, pos.y, yaw)
        self.velocity = (
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        )
    
    def handle_battery(self, msg: BatteryState):
        """Update battery level"""
        self.battery_level = msg.percentage * 100
    
    def handle_path(self, msg: Path):
        """Store planned path from Nav2"""
        self.planned_path = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]
    
    def handle_fleet_status(self, msg: String):
        """
        Process fleet status updates
        
        Used to learn about traffic zones and other robots
        """
        try:
            status = json.loads(msg.data)
            
            # Update traffic zone information
            if 'traffic_zones' in status:
                self.traffic_zones = status['traffic_zones']
                
        except json.JSONDecodeError:
            pass
    
    def handle_zone_response(self, msg: String):
        """
        Handle traffic zone access responses
        
        Example response:
        {
            "robot_id": "amr_001",
            "zone_id": "intersection_A",
            "granted": true,
            "wait_time": 0
        }
        """
        try:
            response = json.loads(msg.data)
            
            # Only process messages for this robot
            if response['robot_id'] != self.robot_id:
                return
            
            zone_id = response['zone_id']
            granted = response['granted']
            
            if granted:
                self.get_logger().info(f'Zone access granted: {zone_id}')
                self.granted_zones.append(zone_id)
                
                # Resume navigation if we were waiting
                if self.waiting_for_zone == zone_id:
                    self.waiting_for_zone = None
                    self.nav_state = NavigationState.NAVIGATING
                    self._resume_navigation()
            else:
                wait_time = response.get('wait_time', 0)
                self.get_logger().info(
                    f'Zone access denied: {zone_id}, queue position: {wait_time}'
                )
                self.waiting_for_zone = zone_id
                self.nav_state = NavigationState.WAITING_FOR_ZONE
                self._pause_navigation()
                
        except (json.JSONDecodeError, KeyError):
            pass
    
    def handle_emergency_stop(self, msg: Bool):
        """Handle fleet-wide emergency stop"""
        self.emergency_stop_active = msg.data
        
        if self.emergency_stop_active:
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            self._stop_robot()
    
    def handle_fleet_cmd_vel(self, msg: Twist):
        """
        Handle velocity commands from fleet manager
        
        Fleet manager can override velocity for:
        - Emergency stops (zero velocity)
        - Speed reduction (collision avoidance)
        """
        # If it's a stop command, enforce it
        if msg.linear.x == 0.0 and msg.angular.z == 0.0:
            self.cmd_vel_pub.publish(msg)
    
    def send_heartbeat(self):
        """Send periodic heartbeat to fleet manager"""
        heartbeat = {
            'robot_id': self.robot_id,
            'state': self.nav_state.value,
            'battery': self.battery_level,
            'position': list(self.position[:2]),
            'velocity': list(self.velocity),
            'current_goal': self.current_goal,
            'waiting_for_zone': self.waiting_for_zone
        }
        
        msg = String()
        msg.data = json.dumps(heartbeat)
        self.heartbeat_pub.publish(msg)
    
    def check_zone_proximity(self):
        """
        Check if robot is approaching any traffic zones
        
        When close to a zone, request access before entering.
        """
        if self.nav_state != NavigationState.NAVIGATING:
            return
        
        for zone_id, zone_info in self.traffic_zones.items():
            # Skip if already granted access
            if zone_id in self.granted_zones:
                continue
            
            # Calculate distance to zone (assuming center is first 2 elements)
            # Zone format from fleet_status: {"occupied_by": [...], "queue": [...]}
            # We need zone positions - this would come from config
            zone_pos = self._get_zone_position(zone_id)
            if zone_pos is None:
                continue
            
            distance = math.sqrt(
                (self.position[0] - zone_pos[0])**2 +
                (self.position[1] - zone_pos[1])**2
            )
            
            if distance < self.ZONE_CHECK_DISTANCE:
                self._request_zone_access(zone_id)
    
    def _get_zone_position(self, zone_id: str) -> Optional[Tuple[float, float]]:
        """Get zone position (would typically come from config/map)"""
        # Hardcoded zone positions - should match fleet_manager
        zone_positions = {
            "intersection_A": (5.0, 5.0),
            "intersection_B": (10.0, 5.0),
            "intersection_C": (15.0, 5.0),
            "intersection_D": (5.0, 10.0),
            "intersection_E": (10.0, 10.0),
            "intersection_F": (15.0, 10.0),
            "narrow_aisle_1": (7.5, 7.5),
            "loading_dock": (0.0, 5.0),
        }
        return zone_positions.get(zone_id)
    
    def _request_zone_access(self, zone_id: str):
        """Request access to a traffic zone"""
        if self.waiting_for_zone == zone_id:
            return  # Already waiting
        
        request = {
            'robot_id': self.robot_id,
            'zone_id': zone_id,
            'action': 'enter'
        }
        
        msg = String()
        msg.data = json.dumps(request)
        self.zone_request_pub.publish(msg)
        
        self.get_logger().info(f'Requesting access to zone: {zone_id}')
    
    def exit_zone(self, zone_id: str):
        """Notify fleet manager that we've exited a zone"""
        if zone_id not in self.granted_zones:
            return
        
        request = {
            'robot_id': self.robot_id,
            'zone_id': zone_id,
            'action': 'exit'
        }
        
        msg = String()
        msg.data = json.dumps(request)
        self.zone_request_pub.publish(msg)
        
        self.granted_zones.remove(zone_id)
        self.get_logger().info(f'Exited zone: {zone_id}')
    
    def _pause_navigation(self):
        """Pause navigation (waiting for zone access)"""
        self._stop_robot()
        self.get_logger().info('Navigation paused - waiting for zone access')
    
    def _resume_navigation(self):
        """Resume navigation after zone access granted"""
        self.get_logger().info('Navigation resumed')
        # Nav2 will automatically resume when we stop sending stop commands
    
    def _stop_robot(self):
        """Send stop command to robot"""
        stop_cmd = Twist()  # All zeros
        self.cmd_vel_pub.publish(stop_cmd)
    
    def navigate_to(self, x: float, y: float, theta: float = 0.0):
        """
        Send navigation goal to Nav2
        
        Example:
            robot.navigate_to(5.0, 10.0, 1.57)  # Go to (5, 10), facing up
        """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available')
            return False
        
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        
        # Convert theta to quaternion
        goal.pose.pose.orientation.z = math.sin(theta / 2)
        goal.pose.pose.orientation.w = math.cos(theta / 2)
        
        self.current_goal = (x, y)
        self.nav_state = NavigationState.NAVIGATING
        
        self.get_logger().info(f'Navigating to ({x:.2f}, {y:.2f})')
        
        future = self.nav_client.send_goal_async(
            goal, 
            feedback_callback=self._nav_feedback_callback
        )
        future.add_done_callback(self._nav_goal_response_callback)
        
        return True
    
    def _nav_feedback_callback(self, feedback_msg):
        """Process navigation feedback"""
        feedback = feedback_msg.feedback
        # Could update progress here
    
    def _nav_goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().warn('Navigation goal rejected')
            self.nav_state = NavigationState.FAILED
            return
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_callback)
    
    def _nav_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        
        # Clear current goal
        self.current_goal = None
        self.nav_state = NavigationState.GOAL_REACHED
        
        self.get_logger().info('Navigation goal reached')
        
        # Exit any zones we're in
        for zone_id in list(self.granted_zones):
            self.exit_zone(zone_id)


def main(args=None):
    rclpy.init(args=args)
    node = RobotAgent()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
