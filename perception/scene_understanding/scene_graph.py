"""
Scene Graph Generation for Semantic Scene Understanding.

Builds structured scene graphs from object detections and depth information,
enabling spatial reasoning, natural language queries, and robot task planning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import time


class SpatialRelation(str, Enum):
    """Spatial relationships between objects."""
    # Horizontal relations
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    NEAR = "near"
    FAR_FROM = "far_from"
    
    # Vertical relations
    ABOVE = "above"
    BELOW = "below"
    ON_TOP_OF = "on_top_of"
    UNDER = "under"
    
    # Containment relations
    INSIDE = "inside"
    CONTAINS = "contains"
    NEXT_TO = "next_to"
    
    # Group relations
    BETWEEN = "between"
    SURROUNDED_BY = "surrounded_by"
    ALIGNED_WITH = "aligned_with"


class ObjectAttribute(str, Enum):
    """Attributes that can be associated with objects."""
    COLOR = "color"
    SIZE = "size"
    SHAPE = "shape"
    MATERIAL = "material"
    STATE = "state"
    ORIENTATION = "orientation"
    MOVABLE = "movable"
    GRASPABLE = "graspable"


@dataclass
class BoundingBox3D:
    """3D bounding box for an object."""
    center: np.ndarray  # [x, y, z]
    dimensions: np.ndarray  # [width, height, depth]
    rotation: Optional[np.ndarray] = None  # Quaternion [w, x, y, z]
    
    @property
    def min_point(self) -> np.ndarray:
        """Get minimum corner point."""
        return self.center - self.dimensions / 2
    
    @property
    def max_point(self) -> np.ndarray:
        """Get maximum corner point."""
        return self.center + self.dimensions / 2
    
    @property
    def volume(self) -> float:
        """Calculate volume."""
        return float(np.prod(self.dimensions))
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the bounding box."""
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
    
    def intersects(self, other: "BoundingBox3D") -> bool:
        """Check if two bounding boxes intersect."""
        return np.all(self.min_point <= other.max_point) and \
               np.all(self.max_point >= other.min_point)
    
    def distance_to(self, other: "BoundingBox3D") -> float:
        """Calculate distance between bounding box centers."""
        return float(np.linalg.norm(self.center - other.center))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "center": self.center.tolist(),
            "dimensions": self.dimensions.tolist(),
            "rotation": self.rotation.tolist() if self.rotation is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox3D":
        """Create from dictionary."""
        return cls(
            center=np.array(data["center"]),
            dimensions=np.array(data["dimensions"]),
            rotation=np.array(data["rotation"]) if data.get("rotation") else None
        )


@dataclass
class SceneObject:
    """
    An object in the scene graph.
    
    Attributes:
        object_id: Unique identifier
        class_name: Object class (e.g., "box", "table", "robot")
        confidence: Detection confidence score
        bbox_3d: 3D bounding box
        attributes: Object attributes (color, size, etc.)
        instance_id: Instance ID for tracking
        timestamp: Detection timestamp
    """
    object_id: str
    class_name: str
    confidence: float
    bbox_3d: BoundingBox3D
    attributes: Dict[str, Any] = field(default_factory=dict)
    instance_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    
    @property
    def position(self) -> np.ndarray:
        """Get object position (center of bounding box)."""
        return self.bbox_3d.center
    
    @property
    def size_category(self) -> str:
        """Categorize object size."""
        volume = self.bbox_3d.volume
        if volume < 0.001:  # < 1 liter
            return "tiny"
        elif volume < 0.01:  # < 10 liters
            return "small"
        elif volume < 0.1:  # < 100 liters
            return "medium"
        elif volume < 1.0:  # < 1000 liters
            return "large"
        else:
            return "very_large"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_3d": self.bbox_3d.to_dict(),
            "attributes": self.attributes,
            "instance_id": self.instance_id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneObject":
        """Create from dictionary."""
        return cls(
            object_id=data["object_id"],
            class_name=data["class_name"],
            confidence=data["confidence"],
            bbox_3d=BoundingBox3D.from_dict(data["bbox_3d"]),
            attributes=data.get("attributes", {}),
            instance_id=data.get("instance_id"),
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class SpatialEdge:
    """
    An edge in the scene graph representing a spatial relationship.
    
    Attributes:
        source_id: Source object ID
        target_id: Target object ID
        relation: Type of spatial relationship
        confidence: Confidence in the relationship
        distance: Distance between objects (meters)
    """
    source_id: str
    target_id: str
    relation: SpatialRelation
    confidence: float = 1.0
    distance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "distance": self.distance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialEdge":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=SpatialRelation(data["relation"]),
            confidence=data.get("confidence", 1.0),
            distance=data.get("distance")
        )


class SceneGraph:
    """
    Scene graph representation for spatial reasoning.
    
    The scene graph consists of:
    - Nodes: Objects in the scene with attributes
    - Edges: Spatial relationships between objects
    
    Example:
        >>> graph = SceneGraph()
        >>> 
        >>> # Add objects
        >>> box = SceneObject("box_1", "box", 0.95, bbox_3d)
        >>> table = SceneObject("table_1", "table", 0.98, table_bbox)
        >>> graph.add_object(box)
        >>> graph.add_object(table)
        >>> 
        >>> # Query relationships
        >>> relations = graph.get_relations("box_1")
        >>> # Returns: [("on_top_of", "table_1"), ("near", "robot_1")]
        >>> 
        >>> # Natural language query
        >>> objects = graph.query("objects on the table")
        >>> # Returns: [box_1, cup_1, ...]
    """
    
    def __init__(
        self,
        distance_threshold_near: float = 1.0,  # meters
        distance_threshold_far: float = 3.0,
        vertical_threshold: float = 0.1,
        horizontal_threshold: float = 0.3
    ):
        """
        Initialize scene graph.
        
        Args:
            distance_threshold_near: Max distance for "near" relation
            distance_threshold_far: Min distance for "far" relation
            vertical_threshold: Threshold for vertical relationships
            horizontal_threshold: Threshold for horizontal relationships
        """
        self._objects: Dict[str, SceneObject] = {}
        self._edges: List[SpatialEdge] = []
        self._adjacency: Dict[str, List[SpatialEdge]] = defaultdict(list)
        
        self.distance_threshold_near = distance_threshold_near
        self.distance_threshold_far = distance_threshold_far
        self.vertical_threshold = vertical_threshold
        self.horizontal_threshold = horizontal_threshold
        
        self._timestamp = time.time()
    
    @property
    def objects(self) -> Dict[str, SceneObject]:
        """Get all objects in the scene."""
        return self._objects.copy()
    
    @property
    def edges(self) -> List[SpatialEdge]:
        """Get all edges in the scene graph."""
        return self._edges.copy()
    
    @property
    def object_count(self) -> int:
        """Get number of objects."""
        return len(self._objects)
    
    @property
    def edge_count(self) -> int:
        """Get number of edges."""
        return len(self._edges)
    
    def add_object(self, obj: SceneObject) -> None:
        """Add an object to the scene graph."""
        self._objects[obj.object_id] = obj
        self._update_relations_for_object(obj)
    
    def remove_object(self, object_id: str) -> Optional[SceneObject]:
        """Remove an object from the scene graph."""
        if object_id not in self._objects:
            return None
        
        obj = self._objects.pop(object_id)
        
        # Remove related edges
        self._edges = [e for e in self._edges 
                      if e.source_id != object_id and e.target_id != object_id]
        
        # Update adjacency
        del self._adjacency[object_id]
        for adj_list in self._adjacency.values():
            adj_list[:] = [e for e in adj_list 
                         if e.source_id != object_id and e.target_id != object_id]
        
        return obj
    
    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get an object by ID."""
        return self._objects.get(object_id)
    
    def get_objects_by_class(self, class_name: str) -> List[SceneObject]:
        """Get all objects of a specific class."""
        return [obj for obj in self._objects.values() 
                if obj.class_name.lower() == class_name.lower()]
    
    def get_objects_by_attribute(
        self,
        attribute: str,
        value: Any
    ) -> List[SceneObject]:
        """Get objects with a specific attribute value."""
        return [obj for obj in self._objects.values()
                if obj.attributes.get(attribute) == value]
    
    def get_relations(
        self,
        object_id: str,
        relation_type: Optional[SpatialRelation] = None
    ) -> List[Tuple[SpatialRelation, str]]:
        """
        Get all relations for an object.
        
        Args:
            object_id: Object to query
            relation_type: Filter by relation type (optional)
            
        Returns:
            List of (relation, target_object_id) tuples
        """
        relations = []
        
        for edge in self._adjacency.get(object_id, []):
            if relation_type is None or edge.relation == relation_type:
                target = edge.target_id if edge.source_id == object_id else edge.source_id
                relations.append((edge.relation, target))
        
        return relations
    
    def get_objects_with_relation(
        self,
        object_id: str,
        relation: SpatialRelation
    ) -> List[SceneObject]:
        """Get objects that have a specific relation to the given object."""
        related_ids = [target for rel, target in self.get_relations(object_id)
                      if rel == relation]
        return [self._objects[oid] for oid in related_ids if oid in self._objects]
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[Tuple[str, SpatialRelation]]]:
        """
        Find a path of relations between two objects.
        
        Uses BFS to find the shortest path.
        
        Args:
            source_id: Starting object
            target_id: Target object
            max_depth: Maximum path length
            
        Returns:
            List of (object_id, relation) tuples forming the path, or None
        """
        if source_id not in self._objects or target_id not in self._objects:
            return None
        
        if source_id == target_id:
            return []
        
        visited = {source_id}
        queue = [(source_id, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            for relation, neighbor in self.get_relations(current):
                if neighbor == target_id:
                    return path + [(neighbor, relation)]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(neighbor, relation)]))
        
        return None
    
    def _update_relations_for_object(self, obj: SceneObject) -> None:
        """Update spatial relations for a newly added object."""
        for other_id, other_obj in self._objects.items():
            if other_id == obj.object_id:
                continue
            
            # Compute relations between obj and other_obj
            relations = self._compute_relations(obj, other_obj)
            
            for relation, confidence in relations:
                edge = SpatialEdge(
                    source_id=obj.object_id,
                    target_id=other_id,
                    relation=relation,
                    confidence=confidence,
                    distance=obj.bbox_3d.distance_to(other_obj.bbox_3d)
                )
                self._edges.append(edge)
                self._adjacency[obj.object_id].append(edge)
                self._adjacency[other_id].append(edge)
    
    def _compute_relations(
        self,
        obj1: SceneObject,
        obj2: SceneObject
    ) -> List[Tuple[SpatialRelation, float]]:
        """
        Compute spatial relations between two objects.
        
        Returns list of (relation, confidence) tuples.
        """
        relations = []
        
        pos1 = obj1.position
        pos2 = obj2.position
        
        diff = pos2 - pos1
        distance = np.linalg.norm(diff)
        
        # Distance-based relations
        if distance < self.distance_threshold_near:
            relations.append((SpatialRelation.NEAR, 1.0))
        elif distance > self.distance_threshold_far:
            relations.append((SpatialRelation.FAR_FROM, 1.0))
        
        # Horizontal relations (X-Y plane)
        horizontal_dist = np.linalg.norm(diff[:2])
        
        if horizontal_dist > self.horizontal_threshold:
            # Left/Right (X axis)
            if abs(diff[0]) > abs(diff[1]):
                if diff[0] > 0:
                    relations.append((SpatialRelation.LEFT_OF, 
                                     min(1.0, abs(diff[0]) / horizontal_dist)))
                else:
                    relations.append((SpatialRelation.RIGHT_OF,
                                     min(1.0, abs(diff[0]) / horizontal_dist)))
            # Front/Behind (Y axis)
            else:
                if diff[1] > 0:
                    relations.append((SpatialRelation.IN_FRONT_OF,
                                     min(1.0, abs(diff[1]) / horizontal_dist)))
                else:
                    relations.append((SpatialRelation.BEHIND,
                                     min(1.0, abs(diff[1]) / horizontal_dist)))
        
        # Vertical relations (Z axis)
        if abs(diff[2]) > self.vertical_threshold:
            if diff[2] > 0:
                relations.append((SpatialRelation.BELOW, 1.0))
                
                # Check if on top of
                if self._is_supported_by(obj1, obj2):
                    relations.append((SpatialRelation.ON_TOP_OF, 0.9))
            else:
                relations.append((SpatialRelation.ABOVE, 1.0))
                
                # Check if under
                if self._is_supported_by(obj2, obj1):
                    relations.append((SpatialRelation.UNDER, 0.9))
        
        # Containment relations
        if self._is_inside(obj1, obj2):
            relations.append((SpatialRelation.INSIDE, 0.95))
        elif self._is_inside(obj2, obj1):
            relations.append((SpatialRelation.CONTAINS, 0.95))
        
        # Next to (close and roughly same height)
        if distance < self.distance_threshold_near and abs(diff[2]) < self.vertical_threshold:
            relations.append((SpatialRelation.NEXT_TO, 0.8))
        
        return relations
    
    def _is_supported_by(self, obj1: SceneObject, obj2: SceneObject) -> bool:
        """Check if obj1 is physically supported by obj2."""
        # obj1 should be above obj2
        if obj1.position[2] <= obj2.position[2]:
            return False
        
        # obj1's bottom should be close to obj2's top
        obj1_bottom = obj1.bbox_3d.min_point[2]
        obj2_top = obj2.bbox_3d.max_point[2]
        
        if abs(obj1_bottom - obj2_top) > 0.05:  # 5cm tolerance
            return False
        
        # Check horizontal overlap
        obj1_min = obj1.bbox_3d.min_point[:2]
        obj1_max = obj1.bbox_3d.max_point[:2]
        obj2_min = obj2.bbox_3d.min_point[:2]
        obj2_max = obj2.bbox_3d.max_point[:2]
        
        overlap = np.all(obj1_min < obj2_max) and np.all(obj1_max > obj2_min)
        return overlap
    
    def _is_inside(self, obj1: SceneObject, obj2: SceneObject) -> bool:
        """Check if obj1 is inside obj2."""
        return (np.all(obj1.bbox_3d.min_point >= obj2.bbox_3d.min_point) and
                np.all(obj1.bbox_3d.max_point <= obj2.bbox_3d.max_point))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene graph to dictionary."""
        return {
            "objects": {oid: obj.to_dict() for oid, obj in self._objects.items()},
            "edges": [e.to_dict() for e in self._edges],
            "timestamp": self._timestamp,
            "config": {
                "distance_threshold_near": self.distance_threshold_near,
                "distance_threshold_far": self.distance_threshold_far,
                "vertical_threshold": self.vertical_threshold,
                "horizontal_threshold": self.horizontal_threshold
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneGraph":
        """Create scene graph from dictionary."""
        config = data.get("config", {})
        graph = cls(
            distance_threshold_near=config.get("distance_threshold_near", 1.0),
            distance_threshold_far=config.get("distance_threshold_far", 3.0),
            vertical_threshold=config.get("vertical_threshold", 0.1),
            horizontal_threshold=config.get("horizontal_threshold", 0.3)
        )
        
        # Add objects (this will also compute relations)
        for obj_data in data.get("objects", {}).values():
            obj = SceneObject.from_dict(obj_data)
            graph._objects[obj.object_id] = obj
        
        # Override with stored edges if available
        if "edges" in data:
            graph._edges = [SpatialEdge.from_dict(e) for e in data["edges"]]
            graph._adjacency.clear()
            for edge in graph._edges:
                graph._adjacency[edge.source_id].append(edge)
                graph._adjacency[edge.target_id].append(edge)
        
        graph._timestamp = data.get("timestamp", time.time())
        return graph
    
    def get_description(self, object_id: str) -> str:
        """
        Generate a natural language description of an object and its relations.
        
        Args:
            object_id: Object to describe
            
        Returns:
            Natural language description
        """
        obj = self.get_object(object_id)
        if not obj:
            return f"Unknown object: {object_id}"
        
        parts = [f"The {obj.class_name}"]
        
        # Add attributes
        if obj.attributes:
            attrs = []
            for key, value in obj.attributes.items():
                attrs.append(f"{value} {key}" if key != "color" else value)
            if attrs:
                parts[0] = f"The {' '.join(attrs)} {obj.class_name}"
        
        # Add relations
        relations = self.get_relations(object_id)
        if relations:
            relation_strs = []
            for rel, target_id in relations[:3]:  # Limit to 3 relations
                target = self.get_object(target_id)
                if target:
                    rel_str = rel.value.replace("_", " ")
                    relation_strs.append(f"{rel_str} the {target.class_name}")
            
            if relation_strs:
                parts.append("is " + ", ".join(relation_strs))
        
        return " ".join(parts) + "."


__all__ = [
    'SpatialRelation',
    'ObjectAttribute',
    'BoundingBox3D',
    'SceneObject',
    'SpatialEdge',
    'SceneGraph',
]
