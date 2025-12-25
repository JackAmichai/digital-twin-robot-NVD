"""
Scene Understanding Module for Digital Twin Robotics Lab.

This module provides semantic scene understanding capabilities:
- Scene graph generation from perception data
- Spatial relationship extraction
- Natural language queries
- ROS 2 integration for real-time scene analysis

Components:
    - SceneGraph: Graph structure representing objects and spatial relations
    - SpatialReasoner: Query engine for spatial reasoning
    - SceneUnderstandingNode: ROS 2 node for publishing scene graphs

Example:
    >>> from perception.scene_understanding import SceneGraph, SpatialReasoner
    >>> 
    >>> # Create scene graph
    >>> scene = SceneGraph()
    >>> scene.add_object(SceneObject(
    ...     object_id="box_1",
    ...     class_name="box",
    ...     position=np.array([1.0, 0.5, 0.3]),
    ...     bbox_3d=BoundingBox3D(
    ...         min_point=np.array([0.8, 0.3, 0.0]),
    ...         max_point=np.array([1.2, 0.7, 0.6])
    ...     )
    ... ))
    >>> 
    >>> # Build relationships
    >>> scene.build_graph()
    >>> 
    >>> # Query the scene
    >>> reasoner = SpatialReasoner(scene)
    >>> result = reasoner.query("find all boxes near the robot")
    >>> print(result.explanation)
"""

from .scene_graph import (
    SceneGraph,
    SceneObject,
    SpatialEdge,
    BoundingBox3D,
    SpatialRelation,
)

from .spatial_reasoning import (
    SpatialReasoner,
    QueryResult,
    QueryOperator,
)

# ROS 2 node (import separately to avoid rclpy dependency issues)
# from .scene_understanding_node import SceneUnderstandingNode


__all__ = [
    # Scene Graph
    'SceneGraph',
    'SceneObject',
    'SpatialEdge',
    'BoundingBox3D',
    'SpatialRelation',
    
    # Spatial Reasoning
    'SpatialReasoner',
    'QueryResult',
    'QueryOperator',
]


__version__ = '1.0.0'
