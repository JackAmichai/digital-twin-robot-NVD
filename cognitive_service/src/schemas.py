"""
Pydantic schemas for Digital Twin Robotics Lab.

These schemas define the data contracts between services:
- Cognitive Layer → Control Layer (RobotIntent)
- Control Layer → Simulation Layer (NavigationGoal)
- System-wide status messages
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    """Supported robot actions."""

    NAVIGATE = "navigate"
    MOVE_TO_ZONE = "move_to_zone"
    STOP = "stop"
    STATUS = "status"
    INSPECT = "inspect"
    RETURN_HOME = "return_home"
    UNKNOWN = "unknown"


class ZoneName(str, Enum):
    """Pre-defined warehouse zones."""

    LOADING_DOCK = "loading_dock"
    STORAGE = "storage"
    ASSEMBLY = "assembly"
    CHARGING = "charging"
    INSPECTION = "inspection"
    HOME = "home"


class Coordinates(BaseModel):
    """3D coordinates with orientation."""

    x: float = Field(..., description="X position in meters")
    y: float = Field(..., description="Y position in meters")
    z: float = Field(default=0.0, description="Z position in meters")
    theta: float = Field(default=0.0, description="Orientation in radians")

    @field_validator("theta")
    @classmethod
    def validate_theta(cls, v: float) -> float:
        """Ensure theta is within valid range."""
        import math

        while v > math.pi:
            v -= 2 * math.pi
        while v < -math.pi:
            v += 2 * math.pi
        return v


class RobotIntent(BaseModel):
    """
    Intent message from Cognitive Layer to Control Layer.

    This is the primary data contract for voice commands.
    """

    action: ActionType = Field(..., description="The intended action")
    target: Optional[str] = Field(None, description="Target zone or object name")
    coordinates: Optional[Coordinates] = Field(None, description="Target coordinates if known")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence score")
    raw_transcript: str = Field(..., description="Original speech transcript")
    timestamp: datetime = Field(default_factory=datetime.now, description="When intent was parsed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Clamp confidence to valid range."""
        return max(0.0, min(1.0, v))

    def is_navigation_action(self) -> bool:
        """Check if this intent requires navigation."""
        return self.action in {ActionType.NAVIGATE, ActionType.MOVE_TO_ZONE, ActionType.INSPECT}

    def requires_confirmation(self) -> bool:
        """Check if this intent needs user confirmation."""
        return self.confidence < 0.7 or self.action == ActionType.UNKNOWN


class NavigationGoal(BaseModel):
    """Navigation goal sent to Nav2."""

    goal_id: str = Field(..., description="Unique goal identifier")
    target_pose: Coordinates = Field(..., description="Target pose")
    frame_id: str = Field(default="map", description="Reference frame")
    timestamp: datetime = Field(default_factory=datetime.now)
    timeout_sec: float = Field(default=60.0, ge=0.0, description="Navigation timeout")


class NavigationStatus(str, Enum):
    """Navigation action status."""

    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


class NavigationResult(BaseModel):
    """Result of a navigation action."""

    goal_id: str
    status: NavigationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    distance_traveled: Optional[float] = None
    final_pose: Optional[Coordinates] = None
    error_message: Optional[str] = None


class ServiceHealth(BaseModel):
    """Health status for a service."""

    name: str = Field(..., description="Service name")
    status: str = Field(..., description="healthy, unhealthy, or unknown")
    latency_ms: float = Field(..., ge=0.0, description="Response latency")
    last_check: datetime = Field(default_factory=datetime.now)
    details: str = Field(default="", description="Additional status info")


class SystemStatus(BaseModel):
    """Overall system status."""

    overall_status: str = Field(..., description="healthy, degraded, or down")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: dict[str, ServiceHealth] = Field(default_factory=dict)
    active_goals: int = Field(default=0, ge=0)
    commands_processed: int = Field(default=0, ge=0)


class ASRResult(BaseModel):
    """Speech recognition result from Riva."""

    transcript: str = Field(..., description="Recognized text")
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_final: bool = Field(default=True)
    latency_ms: float = Field(..., ge=0.0)
    language: str = Field(default="en-US")
    alternatives: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Response from LLM intent extraction."""

    raw_response: str = Field(..., description="Raw LLM output")
    parsed_intent: Optional[RobotIntent] = Field(None)
    model: str = Field(default="llama-3.1-8b-instruct")
    tokens_used: int = Field(default=0, ge=0)
    latency_ms: float = Field(..., ge=0.0)


# Zone coordinate mappings (should match intent_parser.py)
ZONE_COORDINATES: dict[str, Coordinates] = {
    ZoneName.LOADING_DOCK.value: Coordinates(x=5.0, y=2.0, theta=0.0),
    ZoneName.STORAGE.value: Coordinates(x=-5.0, y=2.0, theta=3.14),
    ZoneName.ASSEMBLY.value: Coordinates(x=0.0, y=5.0, theta=1.57),
    ZoneName.CHARGING.value: Coordinates(x=0.0, y=-5.0, theta=-1.57),
    ZoneName.INSPECTION.value: Coordinates(x=3.0, y=0.0, theta=0.0),
    ZoneName.HOME.value: Coordinates(x=0.0, y=0.0, theta=0.0),
}


def get_zone_coordinates(zone: str) -> Optional[Coordinates]:
    """Get coordinates for a zone name."""
    return ZONE_COORDINATES.get(zone.lower().replace(" ", "_"))


# Example usage and validation
if __name__ == "__main__":
    # Test schema validation
    intent = RobotIntent(
        action=ActionType.MOVE_TO_ZONE,
        target="loading_dock",
        coordinates=Coordinates(x=5.0, y=2.0, theta=0.0),
        confidence=0.95,
        raw_transcript="go to the loading dock",
    )
    print(f"Valid intent: {intent.model_dump_json(indent=2)}")

    # Test zone lookup
    coords = get_zone_coordinates("storage")
    print(f"Storage coordinates: {coords}")

    # Test invalid confidence (should clamp)
    intent2 = RobotIntent(
        action=ActionType.STOP,
        confidence=1.5,  # Will be clamped to 1.0
        raw_transcript="stop",
    )
    print(f"Clamped confidence: {intent2.confidence}")
