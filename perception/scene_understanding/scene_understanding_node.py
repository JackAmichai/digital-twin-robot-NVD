"""
ROS 2 Scene Understanding Node.

Publishes scene graphs from perception data with:
- Object detection integration
- Real-time scene graph updates
- Spatial query service
- Visualization markers
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Pose
from std_srvs.srv import Trigger

from .scene_graph import (
    SceneGraph,
    SceneObject,
    BoundingBox3D,
    SpatialRelation
)
from .spatial_reasoning import SpatialReasoner, QueryResult


class SceneUnderstandingNode(Node):
    """
    ROS 2 node for scene understanding.
    
    Subscribes to:
        - /detected_objects: Object detection results
        - /depth_camera/points: Point cloud for spatial analysis
        
    Publishes:
        - /scene_graph: JSON scene graph
        - /scene_markers: Visualization markers
        - /scene_description: Natural language description
        
    Services:
        - /query_scene: Spatial query service
        - /update_scene: Force scene update
    
    Example:
        >>> # Launch the node
        >>> ros2 run perception scene_understanding_node
        >>> 
        >>> # Query the scene
        >>> ros2 service call /query_scene perception_interfaces/srv/SceneQuery \
        ...     "query: 'find all boxes near the robot'"
    """
    
    def __init__(self):
        super().__init__('scene_understanding')
        
        # Parameters
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('near_threshold', 1.0)
        self.declare_parameter('far_threshold', 3.0)
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('scene_frame', 'world')
        
        self.update_rate = self.get_parameter('update_rate').value
        self.near_threshold = self.get_parameter('near_threshold').value
        self.far_threshold = self.get_parameter('far_threshold').value
        self.publish_markers = self.get_parameter('publish_markers').value
        self.scene_frame = self.get_parameter('scene_frame').value
        
        # Scene graph
        self.scene_graph = SceneGraph(
            near_threshold=self.near_threshold,
            far_threshold=self.far_threshold
        )
        self.reasoner = SpatialReasoner(self.scene_graph)
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.objects_sub = self.create_subscription(
            String,
            '/detected_objects',
            self.detected_objects_callback,
            qos_profile
        )
        
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/depth_camera/points',
            self.pointcloud_callback,
            qos_profile
        )
        
        # Publishers
        self.scene_graph_pub = self.create_publisher(
            String,
            '/scene_graph',
            qos_profile
        )
        
        self.markers_pub = self.create_publisher(
            MarkerArray,
            '/scene_markers',
            qos_profile
        )
        
        self.description_pub = self.create_publisher(
            String,
            '/scene_description',
            qos_profile
        )
        
        # Services
        self.query_srv = self.create_service(
            Trigger,  # Replace with custom service type
            '/query_scene',
            self.handle_query_service
        )
        
        self.update_srv = self.create_service(
            Trigger,
            '/update_scene',
            self.handle_update_service
        )
        
        # Timer for publishing
        self.create_timer(1.0 / self.update_rate, self.publish_scene)
        
        self.get_logger().info('Scene Understanding node initialized')
    
    def detected_objects_callback(self, msg: String):
        """
        Process detected objects from perception pipeline.
        
        Expected JSON format:
        {
            "objects": [
                {
                    "id": "box_1",
                    "class": "box",
                    "confidence": 0.95,
                    "bbox_3d": {
                        "min": [0.5, 0.0, 0.0],
                        "max": [1.0, 0.5, 0.5]
                    },
                    "attributes": {
                        "color": "red",
                        "material": "cardboard"
                    }
                }
            ]
        }
        """
        try:
            data = json.loads(msg.data)
            objects = data.get('objects', [])
            
            # Track which objects we've seen
            seen_ids = set()
            
            for obj_data in objects:
                obj_id = obj_data['id']
                seen_ids.add(obj_id)
                
                # Create bounding box
                bbox_data = obj_data.get('bbox_3d', {})
                bbox = BoundingBox3D(
                    min_point=np.array(bbox_data.get('min', [0, 0, 0])),
                    max_point=np.array(bbox_data.get('max', [1, 1, 1]))
                )
                
                # Create or update object
                scene_obj = SceneObject(
                    object_id=obj_id,
                    class_name=obj_data.get('class', 'unknown'),
                    position=bbox.center,
                    bbox_3d=bbox,
                    confidence=obj_data.get('confidence', 1.0),
                    attributes=obj_data.get('attributes', {})
                )
                
                # Add or update in scene graph
                if obj_id in self.scene_graph.objects:
                    self.scene_graph.update_object(obj_id, scene_obj)
                else:
                    self.scene_graph.add_object(scene_obj)
            
            # Remove objects that weren't seen (optional - persistence)
            # This can be configured based on tracking requirements
            
            self.get_logger().debug(f'Updated scene with {len(objects)} objects')
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse detected objects: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing objects: {e}')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Process point cloud for additional spatial analysis."""
        # Could be used for:
        # - Surface detection (finding tables, floors)
        # - Occlusion analysis
        # - Free space computation
        pass
    
    def publish_scene(self):
        """Publish scene graph and related data."""
        if not self.scene_graph.objects:
            return
        
        # Publish JSON scene graph
        scene_json = self.scene_graph.to_json()
        scene_msg = String()
        scene_msg.data = scene_json
        self.scene_graph_pub.publish(scene_msg)
        
        # Publish natural language description
        description = self.reasoner.get_scene_description()
        desc_msg = String()
        desc_msg.data = description
        self.description_pub.publish(desc_msg)
        
        # Publish visualization markers
        if self.publish_markers:
            markers = self.create_visualization_markers()
            self.markers_pub.publish(markers)
    
    def create_visualization_markers(self) -> MarkerArray:
        """
        Create RViz markers for scene visualization.
        
        Returns:
            MarkerArray with object bboxes and relation arrows
        """
        marker_array = MarkerArray()
        marker_id = 0
        
        # Object bounding boxes
        for obj in self.scene_graph.objects.values():
            # Bounding box marker
            bbox_marker = Marker()
            bbox_marker.header.frame_id = self.scene_frame
            bbox_marker.header.stamp = self.get_clock().now().to_msg()
            bbox_marker.id = marker_id
            bbox_marker.ns = 'scene_objects'
            bbox_marker.type = Marker.CUBE
            bbox_marker.action = Marker.ADD
            
            # Set pose
            bbox_marker.pose.position.x = float(obj.position[0])
            bbox_marker.pose.position.y = float(obj.position[1])
            bbox_marker.pose.position.z = float(obj.position[2])
            bbox_marker.pose.orientation.w = 1.0
            
            # Set scale from bounding box
            dimensions = obj.bbox_3d.max_point - obj.bbox_3d.min_point
            bbox_marker.scale.x = float(dimensions[0])
            bbox_marker.scale.y = float(dimensions[1])
            bbox_marker.scale.z = float(dimensions[2])
            
            # Color based on class
            bbox_marker.color = self._get_class_color(obj.class_name)
            bbox_marker.color.a = 0.5
            
            marker_array.markers.append(bbox_marker)
            marker_id += 1
            
            # Label marker
            label_marker = Marker()
            label_marker.header.frame_id = self.scene_frame
            label_marker.header.stamp = self.get_clock().now().to_msg()
            label_marker.id = marker_id
            label_marker.ns = 'scene_labels'
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            
            label_marker.pose.position.x = float(obj.position[0])
            label_marker.pose.position.y = float(obj.position[1])
            label_marker.pose.position.z = float(obj.position[2] + 0.3)
            
            label_marker.text = f"{obj.class_name}\n{obj.object_id}"
            label_marker.scale.z = 0.1
            label_marker.color.r = 1.0
            label_marker.color.g = 1.0
            label_marker.color.b = 1.0
            label_marker.color.a = 1.0
            
            marker_array.markers.append(label_marker)
            marker_id += 1
        
        # Relation arrows
        for edge in self.scene_graph.edges:
            if edge.relation in [SpatialRelation.ON_TOP_OF, SpatialRelation.INSIDE]:
                source_obj = self.scene_graph.get_object(edge.source)
                target_obj = self.scene_graph.get_object(edge.target)
                
                if source_obj and target_obj:
                    arrow_marker = Marker()
                    arrow_marker.header.frame_id = self.scene_frame
                    arrow_marker.header.stamp = self.get_clock().now().to_msg()
                    arrow_marker.id = marker_id
                    arrow_marker.ns = 'scene_relations'
                    arrow_marker.type = Marker.ARROW
                    arrow_marker.action = Marker.ADD
                    
                    # Arrow from source to target
                    start = Point()
                    start.x = float(source_obj.position[0])
                    start.y = float(source_obj.position[1])
                    start.z = float(source_obj.position[2])
                    
                    end = Point()
                    end.x = float(target_obj.position[0])
                    end.y = float(target_obj.position[1])
                    end.z = float(target_obj.position[2])
                    
                    arrow_marker.points = [start, end]
                    arrow_marker.scale.x = 0.02  # shaft diameter
                    arrow_marker.scale.y = 0.04  # head diameter
                    arrow_marker.scale.z = 0.05  # head length
                    
                    arrow_marker.color.r = 0.0
                    arrow_marker.color.g = 1.0
                    arrow_marker.color.b = 0.5
                    arrow_marker.color.a = 0.8
                    
                    marker_array.markers.append(arrow_marker)
                    marker_id += 1
        
        return marker_array
    
    def _get_class_color(self, class_name: str):
        """Get color for object class."""
        from std_msgs.msg import ColorRGBA
        
        color_map = {
            'box': ColorRGBA(r=0.8, g=0.4, b=0.2, a=1.0),
            'table': ColorRGBA(r=0.6, g=0.4, b=0.2, a=1.0),
            'robot': ColorRGBA(r=0.2, g=0.6, b=0.8, a=1.0),
            'shelf': ColorRGBA(r=0.4, g=0.4, b=0.4, a=1.0),
            'bin': ColorRGBA(r=0.2, g=0.8, b=0.2, a=1.0),
            'conveyor': ColorRGBA(r=0.6, g=0.6, b=0.6, a=1.0),
        }
        
        return color_map.get(class_name.lower(), ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0))
    
    def handle_query_service(self, request, response):
        """
        Handle spatial query service requests.
        
        This is a placeholder using Trigger service.
        In production, use a custom service with query string.
        """
        try:
            # For demonstration, describe the scene
            description = self.reasoner.get_scene_description()
            response.success = True
            response.message = description
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def handle_update_service(self, request, response):
        """Handle force update service requests."""
        try:
            self.scene_graph.build_graph()
            response.success = True
            response.message = f"Scene updated with {len(self.scene_graph.objects)} objects"
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def query_scene(self, query: str) -> QueryResult:
        """
        Query the scene using natural language.
        
        Args:
            query: Natural language query
            
        Returns:
            QueryResult with matching objects
            
        Example:
            >>> result = node.query_scene("find all boxes on the table")
            >>> for obj in result.objects:
            ...     print(f"Found: {obj.object_id}")
        """
        return self.reasoner.query(query)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = SceneUnderstandingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
