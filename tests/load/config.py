"""
Load Testing Configuration.

Contains test scenarios, user profiles, and load patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class LoadProfile(str, Enum):
    """Load profile types."""
    SMOKE = "smoke"
    LOAD = "load"
    STRESS = "stress"
    SPIKE = "spike"
    ENDURANCE = "endurance"
    BREAKPOINT = "breakpoint"


@dataclass
class LoadScenario:
    """Load test scenario configuration."""
    name: str
    profile: LoadProfile
    users: int
    spawn_rate: int
    duration: str
    thresholds: Dict[str, float]
    tags: List[str] = field(default_factory=list)


# Load Scenarios
LOAD_SCENARIOS: Dict[str, LoadScenario] = {
    "smoke": LoadScenario(
        name="Smoke Test",
        profile=LoadProfile.SMOKE,
        users=5,
        spawn_rate=1,
        duration="1m",
        thresholds={"response_time_p95": 2000, "error_rate": 0.01},
        tags=["smoke", "quick"]
    ),
    
    "load_normal": LoadScenario(
        name="Normal Load",
        profile=LoadProfile.LOAD,
        users=50,
        spawn_rate=5,
        duration="10m",
        thresholds={"response_time_p95": 1000, "error_rate": 0.01},
        tags=["load", "standard"]
    ),
    
    "stress": LoadScenario(
        name="Stress Test",
        profile=LoadProfile.STRESS,
        users=200,
        spawn_rate=20,
        duration="20m",
        thresholds={"response_time_p95": 3000, "error_rate": 0.05},
        tags=["stress", "high-load"]
    ),
    
    "spike": LoadScenario(
        name="Spike Test",
        profile=LoadProfile.SPIKE,
        users=300,
        spawn_rate=100,
        duration="5m",
        thresholds={"response_time_p95": 5000, "error_rate": 0.10},
        tags=["spike", "burst"]
    ),
    
    "endurance": LoadScenario(
        name="Endurance Test",
        profile=LoadProfile.ENDURANCE,
        users=75,
        spawn_rate=5,
        duration="2h",
        thresholds={"response_time_p95": 1500, "error_rate": 0.01},
        tags=["endurance", "stability"]
    )
}


@dataclass
class SLADefinition:
    """Service Level Agreement definition."""
    name: str
    endpoint_pattern: str
    response_time_p95: int
    error_rate_max: float


SLA_DEFINITIONS: List[SLADefinition] = [
    SLADefinition("Voice Transcription", "/api/v1/voice/transcribe", 500, 0.1),
    SLADefinition("Robot Status", "/api/v1/robots/*/status", 50, 0.01),
    SLADefinition("Fleet Status", "/api/v1/fleet/status", 300, 0.1),
    SLADefinition("Simulation Sync", "/api/v1/simulation/sync", 30, 0.01),
    SLADefinition("Cognitive Message", "/api/v1/cognitive/message", 800, 0.1)
]


def get_scenario(name: str) -> Optional[LoadScenario]:
    """Get a load scenario by name."""
    return LOAD_SCENARIOS.get(name)
