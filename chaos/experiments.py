"""
Pre-defined Chaos Experiments for Digital Twin Robotics Lab.

Contains ready-to-run experiments for testing system resilience.
"""

from typing import List
from .engine import (
    ChaosExperiment,
    ChaosAction,
    SteadyStateHypothesis,
    FaultType
)


# =============================================================================
# Voice Processing Experiments
# =============================================================================

VOICE_LATENCY_EXPERIMENT = ChaosExperiment(
    name="voice-processing-latency",
    description="Test voice processing service under network latency to verify ASR/TTS degradation handling",
    steady_state=SteadyStateHypothesis(
        name="Voice processing healthy",
        probes=[
            {"type": "http", "url": "http://voice-processing:8080/health", "expected_status": 200},
            {"type": "http", "url": "http://voice-processing:8080/ready", "expected_status": 200}
        ],
        tolerance=0.95
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.NETWORK_LATENCY,
            target="voice-processing",
            parameters={"latency_ms": 300, "jitter_ms": 50},
            duration=120
        )
    ],
    tags=["voice", "network", "latency"]
)

VOICE_POD_FAILURE_EXPERIMENT = ChaosExperiment(
    name="voice-processing-pod-failure",
    description="Test voice service recovery when pods are killed",
    steady_state=SteadyStateHypothesis(
        name="Voice processing pods running",
        probes=[
            {"type": "http", "url": "http://voice-processing:8080/health"}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.POD_KILL,
            target="voice-processing",
            parameters={"selector": "app=voice-processing", "count": 1},
            duration=180
        )
    ],
    tags=["voice", "kubernetes", "pod-failure"]
)


# =============================================================================
# Robot Control Experiments
# =============================================================================

ROBOT_CONTROL_LATENCY_EXPERIMENT = ChaosExperiment(
    name="robot-control-network-latency",
    description="Test robot control under network latency - critical for safety",
    steady_state=SteadyStateHypothesis(
        name="Robot control responsive",
        probes=[
            {"type": "http", "url": "http://robot-control:8080/health", "timeout": 2}
        ],
        tolerance=0.99  # Higher tolerance for safety-critical service
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.NETWORK_LATENCY,
            target="robot-control",
            parameters={"latency_ms": 100, "jitter_ms": 20},  # Lower latency - safety critical
            duration=60
        )
    ],
    tags=["robot", "network", "latency", "safety-critical"]
)

ROBOT_CONTROL_PACKET_LOSS_EXPERIMENT = ChaosExperiment(
    name="robot-control-packet-loss",
    description="Test robot control under packet loss conditions",
    steady_state=SteadyStateHypothesis(
        name="Robot control healthy",
        probes=[
            {"type": "http", "url": "http://robot-control:8080/health"}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.NETWORK_PACKET_LOSS,
            target="robot-control",
            parameters={"loss_percent": 5},
            duration=60
        )
    ],
    tags=["robot", "network", "packet-loss"]
)


# =============================================================================
# Fleet Management Experiments
# =============================================================================

FLEET_MANAGER_FAILURE_EXPERIMENT = ChaosExperiment(
    name="fleet-manager-failure",
    description="Test fleet operations when fleet manager pod fails",
    steady_state=SteadyStateHypothesis(
        name="Fleet manager operational",
        probes=[
            {"type": "http", "url": "http://fleet-manager:8080/health"},
            {"type": "http", "url": "http://fleet-manager:8080/api/v1/fleet/status"}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.POD_KILL,
            target="fleet-manager",
            parameters={"selector": "app=fleet-manager"},
            duration=120
        )
    ],
    tags=["fleet", "kubernetes", "pod-failure"]
)

FLEET_DATABASE_PARTITION_EXPERIMENT = ChaosExperiment(
    name="fleet-database-partition",
    description="Test fleet manager behavior during database network partition",
    steady_state=SteadyStateHypothesis(
        name="Fleet manager can reach database",
        probes=[
            {"type": "http", "url": "http://fleet-manager:8080/health"}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.NETWORK_PARTITION,
            target="fleet-manager",
            parameters={"target_ip": "10.0.0.50"},  # Database IP
            duration=30
        )
    ],
    tags=["fleet", "network", "partition", "database"]
)


# =============================================================================
# Simulation Sync Experiments
# =============================================================================

SIMULATION_SYNC_LATENCY_EXPERIMENT = ChaosExperiment(
    name="simulation-sync-latency",
    description="Test digital twin sync under high latency - affects real-time fidelity",
    steady_state=SteadyStateHypothesis(
        name="Simulation sync operational",
        probes=[
            {"type": "http", "url": "http://simulation:8080/health"},
            {"type": "http", "url": "http://simulation:8080/api/v1/sync/status"}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.NETWORK_LATENCY,
            target="simulation",
            parameters={"latency_ms": 50, "jitter_ms": 10},  # Even 50ms affects real-time sync
            duration=60
        )
    ],
    tags=["simulation", "network", "latency", "real-time"]
)


# =============================================================================
# Cognitive Layer Experiments
# =============================================================================

COGNITIVE_LLM_TIMEOUT_EXPERIMENT = ChaosExperiment(
    name="cognitive-llm-timeout",
    description="Test cognitive layer when LLM inference times out",
    steady_state=SteadyStateHypothesis(
        name="Cognitive layer responsive",
        probes=[
            {"type": "http", "url": "http://cognitive:8080/health"}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.RESPONSE_DELAY,
            target="cognitive",
            parameters={"delay_ms": 10000, "endpoints": ["/api/v1/cognitive/intent"]},
            duration=60
        )
    ],
    tags=["cognitive", "timeout", "llm"]
)

COGNITIVE_REDIS_FAILURE_EXPERIMENT = ChaosExperiment(
    name="cognitive-redis-failure",
    description="Test cognitive layer when Redis (conversation context) fails",
    steady_state=SteadyStateHypothesis(
        name="Cognitive with Redis",
        probes=[
            {"type": "http", "url": "http://cognitive:8080/health"}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.POD_KILL,
            target="redis",
            parameters={"selector": "app=redis"},
            duration=60
        )
    ],
    tags=["cognitive", "redis", "dependency-failure"]
)


# =============================================================================
# Resource Exhaustion Experiments
# =============================================================================

GPU_MEMORY_EXHAUSTION_EXPERIMENT = ChaosExperiment(
    name="gpu-memory-exhaustion",
    description="Test inference services under GPU memory pressure",
    steady_state=SteadyStateHypothesis(
        name="Inference services healthy",
        probes=[
            {"type": "http", "url": "http://triton:8000/v2/health/ready"}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.MEMORY_STRESS,
            target="triton-inference",
            parameters={"memory_percent": 85, "gpu": True},
            duration=120
        )
    ],
    tags=["gpu", "memory", "inference", "resource"]
)

CPU_STRESS_EXPERIMENT = ChaosExperiment(
    name="control-plane-cpu-stress",
    description="Test control plane under high CPU load",
    steady_state=SteadyStateHypothesis(
        name="Control plane responsive",
        probes=[
            {"type": "http", "url": "http://robot-control:8080/health", "timeout": 5}
        ]
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.CPU_STRESS,
            target="control-plane",
            parameters={"cpu_percent": 90, "workers": 8},
            duration=120
        )
    ],
    tags=["cpu", "stress", "control-plane"]
)


# =============================================================================
# Multi-Service Experiments
# =============================================================================

CASCADING_FAILURE_EXPERIMENT = ChaosExperiment(
    name="cascading-failure",
    description="Test system behavior during cascading failures across services",
    steady_state=SteadyStateHypothesis(
        name="All services healthy",
        probes=[
            {"type": "http", "url": "http://voice-processing:8080/health"},
            {"type": "http", "url": "http://robot-control:8080/health"},
            {"type": "http", "url": "http://fleet-manager:8080/health"},
            {"type": "http", "url": "http://cognitive:8080/health"}
        ],
        tolerance=0.90
    ),
    actions=[
        ChaosAction(
            fault_type=FaultType.POD_KILL,
            target="redis",
            parameters={"selector": "app=redis"},
            duration=30
        ),
        ChaosAction(
            fault_type=FaultType.NETWORK_LATENCY,
            target="voice-processing",
            parameters={"latency_ms": 500},
            duration=60
        ),
        ChaosAction(
            fault_type=FaultType.CPU_STRESS,
            target="cognitive",
            parameters={"cpu_percent": 80},
            duration=60
        )
    ],
    tags=["cascading", "multi-service", "resilience"]
)


# =============================================================================
# Experiment Registry
# =============================================================================

ALL_EXPERIMENTS: List[ChaosExperiment] = [
    # Voice
    VOICE_LATENCY_EXPERIMENT,
    VOICE_POD_FAILURE_EXPERIMENT,
    # Robot Control
    ROBOT_CONTROL_LATENCY_EXPERIMENT,
    ROBOT_CONTROL_PACKET_LOSS_EXPERIMENT,
    # Fleet
    FLEET_MANAGER_FAILURE_EXPERIMENT,
    FLEET_DATABASE_PARTITION_EXPERIMENT,
    # Simulation
    SIMULATION_SYNC_LATENCY_EXPERIMENT,
    # Cognitive
    COGNITIVE_LLM_TIMEOUT_EXPERIMENT,
    COGNITIVE_REDIS_FAILURE_EXPERIMENT,
    # Resources
    GPU_MEMORY_EXHAUSTION_EXPERIMENT,
    CPU_STRESS_EXPERIMENT,
    # Multi-service
    CASCADING_FAILURE_EXPERIMENT
]


def get_experiment_by_name(name: str) -> ChaosExperiment:
    """Get experiment by name."""
    for exp in ALL_EXPERIMENTS:
        if exp.name == name:
            return exp
    raise ValueError(f"Unknown experiment: {name}")


def get_experiments_by_tag(tag: str) -> List[ChaosExperiment]:
    """Get all experiments with a specific tag."""
    return [exp for exp in ALL_EXPERIMENTS if tag in exp.tags]


def list_experiments() -> List[str]:
    """List all available experiment names."""
    return [exp.name for exp in ALL_EXPERIMENTS]
