#!/usr/bin/env python3
"""
Cognitive Bridge Node
Subscribes to Redis commands and publishes ROS 2 navigation goals
"""

import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import redis
import threading


class CognitiveBridge(Node):
    """Bridge between cognitive service and ROS 2 Nav2."""
    
    def __init__(self):
        super().__init__('cognitive_bridge')
        
        # Parameters
        self.declare_parameter('redis_url', 'redis://redis:6379')
        self.declare_parameter('redis_channel', 'robot_commands')
        
        redis_url = self.get_parameter('redis_url').value
        self.channel = self.get_parameter('redis_channel').value
        
        # Publishers
        self.goal_pub = self.create_publisher(
            PoseStamped, 
            '/goal_pose', 
            10
        )
        self.status_pub = self.create_publisher(
            String,
            '/cognitive/status',
            10
        )
        
        # Redis connection
        self.redis = redis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
        
        # Start Redis listener thread
        self.running = True
        self.listener_thread = threading.Thread(target=self._listen_redis)
        self.listener_thread.start()
        
        self.get_logger().info('Cognitive Bridge initialized')
        self.get_logger().info(f'Listening on Redis channel: {self.channel}')
    
    def _listen_redis(self):
        """Listen for commands on Redis channel."""
        self.pubsub.subscribe(self.channel)
        
        for message in self.pubsub.listen():
            if not self.running:
                break
                
            if message['type'] != 'message':
                continue
            
            try:
                data = json.loads(message['data'])
                self._process_command(data)
            except json.JSONDecodeError as e:
                self.get_logger().error(f'Invalid JSON: {e}')
    
    def _process_command(self, cmd: dict):
        """Process incoming command and publish appropriate ROS messages."""
        action = cmd.get('action', 'unknown')
        
        self.get_logger().info(f'Received command: {action}')
        
        if action in ['navigate', 'inspect']:
            self._send_nav_goal(cmd)
        elif action == 'stop':
            self._send_stop()
        elif action == 'status':
            self._publish_status()
        else:
            self.get_logger().warn(f'Unknown action: {action}')
    
    def _send_nav_goal(self, cmd: dict):
        """Send navigation goal to Nav2."""
        coords = cmd.get('coordinates')
        
        if not coords or len(coords) < 2:
            self.get_logger().error('No coordinates in command')
            return
        
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        
        goal.pose.position.x = float(coords[0])
        goal.pose.position.y = float(coords[1])
        goal.pose.position.z = 0.0
        
        # Default orientation (facing +X)
        goal.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal)
        
        target = cmd.get('target', 'unknown')
        self.get_logger().info(
            f'Published goal: {target} at ({coords[0]}, {coords[1]})'
        )
        
        # Publish status
        status = String()
        status.data = f'Navigating to {target}'
        self.status_pub.publish(status)
    
    def _send_stop(self):
        """Emergency stop - cancel navigation."""
        # Publish empty goal to cancel
        self.get_logger().info('STOP command received')
        
        status = String()
        status.data = 'Stopping'
        self.status_pub.publish(status)
    
    def _publish_status(self):
        """Publish current status."""
        status = String()
        status.data = 'Ready'
        self.status_pub.publish(status)
    
    def destroy_node(self):
        """Clean shutdown."""
        self.running = False
        self.pubsub.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CognitiveBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
