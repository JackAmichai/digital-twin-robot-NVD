"""
Spatial Reasoning Engine for Scene Understanding.

Provides advanced spatial reasoning capabilities including:
- Natural language queries on scene graphs
- Spatial constraint satisfaction
- Reference resolution
- Path and reachability analysis
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .scene_graph import (
    SceneGraph,
    SceneObject,
    SpatialRelation,
    BoundingBox3D
)


class QueryOperator(str, Enum):
    """Query operators for spatial reasoning."""
    FIND = "find"
    COUNT = "count"
    EXISTS = "exists"
    NEAREST = "nearest"
    FARTHEST = "farthest"
    BETWEEN = "between"
    PATH = "path"
    REACHABLE = "reachable"


@dataclass
class QueryResult:
    """
    Result of a spatial query.
    
    Attributes:
        success: Whether the query succeeded
        objects: Matching objects (if applicable)
        count: Count result (if count query)
        exists: Boolean result (if exists query)
        path: Path result (if path query)
        explanation: Natural language explanation
    """
    success: bool
    objects: List[SceneObject] = None
    count: Optional[int] = None
    exists: Optional[bool] = None
    path: Optional[List[Tuple[str, SpatialRelation]]] = None
    explanation: str = ""
    
    def __post_init__(self):
        if self.objects is None:
            self.objects = []


class SpatialReasoner:
    """
    Spatial reasoning engine for scene graphs.
    
    Supports:
    - Natural language queries ("find the box on the table")
    - Programmatic spatial queries
    - Constraint satisfaction
    - Reference resolution
    
    Example:
        >>> reasoner = SpatialReasoner(scene_graph)
        >>> 
        >>> # Natural language query
        >>> result = reasoner.query("find all boxes near the robot")
        >>> print(result.objects)
        >>> 
        >>> # Programmatic query
        >>> boxes = reasoner.find_objects_with_relation(
        ...     "robot_1", SpatialRelation.NEAR, class_filter="box"
        ... )
        >>> 
        >>> # Reference resolution
        >>> obj = reasoner.resolve_reference("the red box on the left")
    """
    
    # Patterns for parsing natural language queries
    QUERY_PATTERNS = [
        # Find patterns
        (r"find (?:all |the )?(\w+)s? (?:that are |which are )?(\w+) (?:the |a )?(\w+)",
         QueryOperator.FIND, ["class", "relation", "reference"]),
        (r"find (?:all |the )?(\w+)s? (\w+) (?:the |a )?(\w+)",
         QueryOperator.FIND, ["class", "relation", "reference"]),
        (r"find (?:all |the )?(\w+)s?",
         QueryOperator.FIND, ["class"]),
        (r"what is (\w+) (?:the |a )?(\w+)",
         QueryOperator.FIND, ["relation", "reference"]),
        (r"where is (?:the |a )?(\w+)",
         QueryOperator.FIND, ["reference"]),
        
        # Count patterns
        (r"how many (\w+)s?",
         QueryOperator.COUNT, ["class"]),
        (r"count (?:all |the )?(\w+)s?",
         QueryOperator.COUNT, ["class"]),
        
        # Exists patterns
        (r"is there (?:a |an )?(\w+)",
         QueryOperator.EXISTS, ["class"]),
        (r"are there (?:any )?(\w+)s?",
         QueryOperator.EXISTS, ["class"]),
        
        # Nearest/Farthest patterns
        (r"(?:find |what is )?(?:the )?nearest (\w+) to (?:the |a )?(\w+)",
         QueryOperator.NEAREST, ["class", "reference"]),
        (r"(?:find |what is )?(?:the )?closest (\w+) to (?:the |a )?(\w+)",
         QueryOperator.NEAREST, ["class", "reference"]),
        (r"(?:find |what is )?(?:the )?farthest (\w+) from (?:the |a )?(\w+)",
         QueryOperator.FARTHEST, ["class", "reference"]),
    ]
    
    # Relation keywords mapping
    RELATION_KEYWORDS = {
        "on": SpatialRelation.ON_TOP_OF,
        "on_top_of": SpatialRelation.ON_TOP_OF,
        "above": SpatialRelation.ABOVE,
        "over": SpatialRelation.ABOVE,
        "below": SpatialRelation.BELOW,
        "under": SpatialRelation.UNDER,
        "beneath": SpatialRelation.UNDER,
        "left": SpatialRelation.LEFT_OF,
        "left_of": SpatialRelation.LEFT_OF,
        "right": SpatialRelation.RIGHT_OF,
        "right_of": SpatialRelation.RIGHT_OF,
        "near": SpatialRelation.NEAR,
        "close": SpatialRelation.NEAR,
        "close_to": SpatialRelation.NEAR,
        "next_to": SpatialRelation.NEXT_TO,
        "beside": SpatialRelation.NEXT_TO,
        "in_front_of": SpatialRelation.IN_FRONT_OF,
        "front": SpatialRelation.IN_FRONT_OF,
        "behind": SpatialRelation.BEHIND,
        "inside": SpatialRelation.INSIDE,
        "in": SpatialRelation.INSIDE,
        "contains": SpatialRelation.CONTAINS,
        "far": SpatialRelation.FAR_FROM,
        "far_from": SpatialRelation.FAR_FROM,
    }
    
    def __init__(self, scene_graph: SceneGraph):
        """
        Initialize spatial reasoner.
        
        Args:
            scene_graph: Scene graph to reason over
        """
        self.scene_graph = scene_graph
    
    def query(self, natural_language_query: str) -> QueryResult:
        """
        Process a natural language spatial query.
        
        Args:
            natural_language_query: Query in natural language
            
        Returns:
            QueryResult with matching objects or information
        """
        query_lower = natural_language_query.lower().strip()
        
        # Try each pattern
        for pattern, operator, param_names in self.QUERY_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                params = dict(zip(param_names, match.groups()))
                return self._execute_query(operator, params)
        
        # Default: try to find objects mentioned
        return self._fuzzy_query(query_lower)
    
    def _execute_query(
        self,
        operator: QueryOperator,
        params: Dict[str, str]
    ) -> QueryResult:
        """Execute a parsed query."""
        
        if operator == QueryOperator.FIND:
            return self._query_find(params)
        elif operator == QueryOperator.COUNT:
            return self._query_count(params)
        elif operator == QueryOperator.EXISTS:
            return self._query_exists(params)
        elif operator == QueryOperator.NEAREST:
            return self._query_nearest(params)
        elif operator == QueryOperator.FARTHEST:
            return self._query_farthest(params)
        else:
            return QueryResult(
                success=False,
                explanation=f"Unsupported query operator: {operator}"
            )
    
    def _query_find(self, params: Dict[str, str]) -> QueryResult:
        """Execute a FIND query."""
        class_name = params.get("class")
        relation_str = params.get("relation")
        reference = params.get("reference")
        
        # If only class specified, return all objects of that class
        if class_name and not relation_str and not reference:
            objects = self.scene_graph.get_objects_by_class(class_name)
            return QueryResult(
                success=True,
                objects=objects,
                explanation=f"Found {len(objects)} {class_name}(s)"
            )
        
        # If reference is specified, resolve it
        if reference:
            ref_objects = self.resolve_reference(reference)
            if not ref_objects:
                return QueryResult(
                    success=False,
                    explanation=f"Could not find reference: {reference}"
                )
            ref_obj = ref_objects[0]
            
            # Get relation
            relation = self.RELATION_KEYWORDS.get(relation_str)
            
            if relation:
                # Find objects with the relation
                if class_name:
                    objects = [
                        obj for obj in self.find_objects_with_relation(
                            ref_obj.object_id, relation
                        )
                        if obj.class_name.lower() == class_name.lower()
                    ]
                else:
                    objects = self.find_objects_with_relation(
                        ref_obj.object_id, relation
                    )
                
                return QueryResult(
                    success=True,
                    objects=objects,
                    explanation=f"Found {len(objects)} object(s) {relation_str} {reference}"
                )
            
            # If no relation, find what's near the reference
            objects = self.find_objects_with_relation(
                ref_obj.object_id, SpatialRelation.NEAR
            )
            if class_name:
                objects = [o for o in objects if o.class_name.lower() == class_name.lower()]
            
            return QueryResult(
                success=True,
                objects=objects,
                explanation=f"Found {len(objects)} object(s) near {reference}"
            )
        
        return QueryResult(success=False, explanation="Could not parse query")
    
    def _query_count(self, params: Dict[str, str]) -> QueryResult:
        """Execute a COUNT query."""
        class_name = params.get("class", "")
        objects = self.scene_graph.get_objects_by_class(class_name)
        
        return QueryResult(
            success=True,
            count=len(objects),
            objects=objects,
            explanation=f"There are {len(objects)} {class_name}(s)"
        )
    
    def _query_exists(self, params: Dict[str, str]) -> QueryResult:
        """Execute an EXISTS query."""
        class_name = params.get("class", "")
        objects = self.scene_graph.get_objects_by_class(class_name)
        exists = len(objects) > 0
        
        return QueryResult(
            success=True,
            exists=exists,
            objects=objects if exists else [],
            explanation=f"{'Yes' if exists else 'No'}, there {'are' if exists else 'are no'} {class_name}(s)"
        )
    
    def _query_nearest(self, params: Dict[str, str]) -> QueryResult:
        """Execute a NEAREST query."""
        class_name = params.get("class")
        reference = params.get("reference")
        
        # Resolve reference
        ref_objects = self.resolve_reference(reference)
        if not ref_objects:
            return QueryResult(
                success=False,
                explanation=f"Could not find reference: {reference}"
            )
        ref_obj = ref_objects[0]
        
        # Find nearest object of class
        nearest = self.find_nearest(
            ref_obj.object_id,
            class_filter=class_name
        )
        
        if nearest:
            return QueryResult(
                success=True,
                objects=[nearest],
                explanation=f"The nearest {class_name} to {reference} is {nearest.object_id}"
            )
        
        return QueryResult(
            success=False,
            explanation=f"No {class_name} found"
        )
    
    def _query_farthest(self, params: Dict[str, str]) -> QueryResult:
        """Execute a FARTHEST query."""
        class_name = params.get("class")
        reference = params.get("reference")
        
        ref_objects = self.resolve_reference(reference)
        if not ref_objects:
            return QueryResult(
                success=False,
                explanation=f"Could not find reference: {reference}"
            )
        ref_obj = ref_objects[0]
        
        farthest = self.find_farthest(
            ref_obj.object_id,
            class_filter=class_name
        )
        
        if farthest:
            return QueryResult(
                success=True,
                objects=[farthest],
                explanation=f"The farthest {class_name} from {reference} is {farthest.object_id}"
            )
        
        return QueryResult(
            success=False,
            explanation=f"No {class_name} found"
        )
    
    def _fuzzy_query(self, query: str) -> QueryResult:
        """Attempt fuzzy matching for unrecognized queries."""
        # Look for object class names in the query
        found_objects = []
        
        for obj in self.scene_graph.objects.values():
            if obj.class_name.lower() in query:
                found_objects.append(obj)
        
        if found_objects:
            return QueryResult(
                success=True,
                objects=found_objects,
                explanation=f"Found {len(found_objects)} matching object(s)"
            )
        
        return QueryResult(
            success=False,
            explanation="Could not understand the query"
        )
    
    def resolve_reference(
        self,
        reference: str,
        context: Optional[List[SceneObject]] = None
    ) -> List[SceneObject]:
        """
        Resolve a natural language reference to objects.
        
        Handles:
        - Class names ("the box", "a table")
        - Attributes ("the red box", "the large table")
        - Spatial descriptions ("the box on the left")
        - Ordinals ("the first box", "the second table")
        
        Args:
            reference: Natural language reference
            context: Optional context to resolve within
            
        Returns:
            List of matching objects
        """
        ref_lower = reference.lower().strip()
        
        # Remove articles
        ref_lower = re.sub(r'^(the|a|an)\s+', '', ref_lower)
        
        # Check for ordinals
        ordinal_match = re.match(r'(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+(\w+)', ref_lower)
        if ordinal_match:
            ordinal_str, class_name = ordinal_match.groups()
            ordinal = self._parse_ordinal(ordinal_str)
            objects = self.scene_graph.get_objects_by_class(class_name)
            if ordinal <= len(objects):
                return [objects[ordinal - 1]]
            return []
        
        # Check for color + class
        color_match = re.match(r'(\w+)\s+(\w+)', ref_lower)
        if color_match:
            potential_color, class_name = color_match.groups()
            objects = self.scene_graph.get_objects_by_class(class_name)
            colored = [o for o in objects if o.attributes.get("color", "").lower() == potential_color]
            if colored:
                return colored
            # If no match, fall through to class-only search
        
        # Check for spatial reference ("box on the left")
        spatial_match = re.match(r'(\w+)\s+on\s+the\s+(left|right)', ref_lower)
        if spatial_match:
            class_name, direction = spatial_match.groups()
            objects = self.scene_graph.get_objects_by_class(class_name)
            if direction == "left":
                objects.sort(key=lambda o: o.position[0])
                return objects[:1] if objects else []
            else:
                objects.sort(key=lambda o: -o.position[0])
                return objects[:1] if objects else []
        
        # Simple class name lookup
        objects = self.scene_graph.get_objects_by_class(ref_lower)
        if objects:
            return objects
        
        # Try to find partial matches
        for obj in self.scene_graph.objects.values():
            if ref_lower in obj.class_name.lower() or obj.class_name.lower() in ref_lower:
                objects.append(obj)
        
        return objects
    
    def _parse_ordinal(self, ordinal_str: str) -> int:
        """Parse ordinal string to integer."""
        ordinal_map = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5
        }
        
        if ordinal_str in ordinal_map:
            return ordinal_map[ordinal_str]
        
        # Try to parse numeric ordinal
        match = re.match(r'(\d+)', ordinal_str)
        if match:
            return int(match.group(1))
        
        return 1
    
    def find_objects_with_relation(
        self,
        reference_id: str,
        relation: SpatialRelation,
        class_filter: Optional[str] = None
    ) -> List[SceneObject]:
        """
        Find objects with a specific relation to a reference object.
        
        Args:
            reference_id: Reference object ID
            relation: Spatial relation to find
            class_filter: Optional class name filter
            
        Returns:
            List of matching objects
        """
        objects = self.scene_graph.get_objects_with_relation(reference_id, relation)
        
        if class_filter:
            objects = [o for o in objects if o.class_name.lower() == class_filter.lower()]
        
        return objects
    
    def find_nearest(
        self,
        reference_id: str,
        class_filter: Optional[str] = None,
        exclude_ids: Optional[Set[str]] = None
    ) -> Optional[SceneObject]:
        """
        Find the nearest object to a reference.
        
        Args:
            reference_id: Reference object ID
            class_filter: Optional class name filter
            exclude_ids: Object IDs to exclude
            
        Returns:
            Nearest object or None
        """
        ref_obj = self.scene_graph.get_object(reference_id)
        if not ref_obj:
            return None
        
        exclude = exclude_ids or set()
        exclude.add(reference_id)
        
        nearest = None
        min_distance = float('inf')
        
        for obj_id, obj in self.scene_graph.objects.items():
            if obj_id in exclude:
                continue
            if class_filter and obj.class_name.lower() != class_filter.lower():
                continue
            
            distance = ref_obj.bbox_3d.distance_to(obj.bbox_3d)
            if distance < min_distance:
                min_distance = distance
                nearest = obj
        
        return nearest
    
    def find_farthest(
        self,
        reference_id: str,
        class_filter: Optional[str] = None,
        exclude_ids: Optional[Set[str]] = None
    ) -> Optional[SceneObject]:
        """Find the farthest object from a reference."""
        ref_obj = self.scene_graph.get_object(reference_id)
        if not ref_obj:
            return None
        
        exclude = exclude_ids or set()
        exclude.add(reference_id)
        
        farthest = None
        max_distance = -1
        
        for obj_id, obj in self.scene_graph.objects.items():
            if obj_id in exclude:
                continue
            if class_filter and obj.class_name.lower() != class_filter.lower():
                continue
            
            distance = ref_obj.bbox_3d.distance_to(obj.bbox_3d)
            if distance > max_distance:
                max_distance = distance
                farthest = obj
        
        return farthest
    
    def find_objects_in_region(
        self,
        min_point: np.ndarray,
        max_point: np.ndarray,
        class_filter: Optional[str] = None
    ) -> List[SceneObject]:
        """
        Find objects within a 3D region.
        
        Args:
            min_point: Minimum corner of region [x, y, z]
            max_point: Maximum corner of region [x, y, z]
            class_filter: Optional class name filter
            
        Returns:
            List of objects in the region
        """
        results = []
        
        for obj in self.scene_graph.objects.values():
            if class_filter and obj.class_name.lower() != class_filter.lower():
                continue
            
            # Check if object center is within region
            pos = obj.position
            if np.all(pos >= min_point) and np.all(pos <= max_point):
                results.append(obj)
        
        return results
    
    def check_constraint(
        self,
        object_id: str,
        constraint: Dict[str, Any]
    ) -> bool:
        """
        Check if an object satisfies a spatial constraint.
        
        Constraint format:
        {
            "relation": "on_top_of",
            "reference": "table_1",
            "distance_max": 1.0,  # optional
            "distance_min": 0.5,  # optional
        }
        
        Args:
            object_id: Object to check
            constraint: Constraint specification
            
        Returns:
            True if constraint is satisfied
        """
        obj = self.scene_graph.get_object(object_id)
        if not obj:
            return False
        
        # Check relation constraint
        if "relation" in constraint and "reference" in constraint:
            relation = SpatialRelation(constraint["relation"])
            reference_id = constraint["reference"]
            
            relations = self.scene_graph.get_relations(object_id)
            if not any(r == relation and t == reference_id for r, t in relations):
                return False
        
        # Check distance constraints
        if "reference" in constraint:
            ref_obj = self.scene_graph.get_object(constraint["reference"])
            if ref_obj:
                distance = obj.bbox_3d.distance_to(ref_obj.bbox_3d)
                
                if "distance_max" in constraint and distance > constraint["distance_max"]:
                    return False
                if "distance_min" in constraint and distance < constraint["distance_min"]:
                    return False
        
        return True
    
    def get_scene_description(self, max_objects: int = 10) -> str:
        """
        Generate a natural language description of the scene.
        
        Args:
            max_objects: Maximum objects to describe
            
        Returns:
            Natural language scene description
        """
        objects = list(self.scene_graph.objects.values())[:max_objects]
        
        if not objects:
            return "The scene is empty."
        
        descriptions = []
        for obj in objects:
            desc = self.scene_graph.get_description(obj.object_id)
            descriptions.append(desc)
        
        # Summarize
        class_counts = {}
        for obj in self.scene_graph.objects.values():
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
        
        summary = "Scene contains: " + ", ".join(
            f"{count} {name}{'s' if count > 1 else ''}"
            for name, count in class_counts.items()
        ) + ".\n\n"
        
        return summary + "\n".join(descriptions)


__all__ = [
    'QueryOperator',
    'QueryResult',
    'SpatialReasoner',
]
