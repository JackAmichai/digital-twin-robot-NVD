"""
Chaos Engineering Framework for Digital Twin Robotics Lab.

Provides failure injection testing to verify system resilience
against various fault conditions.
"""

from .engine import (
    ChaosEngine,
    ChaosExperiment,
    ChaosAction,
    SteadyStateHypothesis,
    ExperimentResult,
    FaultType,
    ExperimentState,
    FaultInjector,
    NetworkFaultInjector,
    ResourceFaultInjector,
    KubernetesFaultInjector,
    ApplicationFaultInjector,
    create_network_latency_experiment,
    create_pod_failure_experiment,
    create_resource_exhaustion_experiment
)

from .experiments import (
    ALL_EXPERIMENTS,
    get_experiment_by_name,
    get_experiments_by_tag,
    list_experiments,
    # Individual experiments
    VOICE_LATENCY_EXPERIMENT,
    VOICE_POD_FAILURE_EXPERIMENT,
    ROBOT_CONTROL_LATENCY_EXPERIMENT,
    ROBOT_CONTROL_PACKET_LOSS_EXPERIMENT,
    FLEET_MANAGER_FAILURE_EXPERIMENT,
    FLEET_DATABASE_PARTITION_EXPERIMENT,
    SIMULATION_SYNC_LATENCY_EXPERIMENT,
    COGNITIVE_LLM_TIMEOUT_EXPERIMENT,
    COGNITIVE_REDIS_FAILURE_EXPERIMENT,
    GPU_MEMORY_EXHAUSTION_EXPERIMENT,
    CPU_STRESS_EXPERIMENT,
    CASCADING_FAILURE_EXPERIMENT
)

__all__ = [
    # Core Engine
    "ChaosEngine",
    "ChaosExperiment",
    "ChaosAction",
    "SteadyStateHypothesis",
    "ExperimentResult",
    "FaultType",
    "ExperimentState",
    # Fault Injectors
    "FaultInjector",
    "NetworkFaultInjector",
    "ResourceFaultInjector",
    "KubernetesFaultInjector",
    "ApplicationFaultInjector",
    # Factory Functions
    "create_network_latency_experiment",
    "create_pod_failure_experiment",
    "create_resource_exhaustion_experiment",
    # Experiment Registry
    "ALL_EXPERIMENTS",
    "get_experiment_by_name",
    "get_experiments_by_tag",
    "list_experiments",
    # Pre-defined Experiments
    "VOICE_LATENCY_EXPERIMENT",
    "VOICE_POD_FAILURE_EXPERIMENT",
    "ROBOT_CONTROL_LATENCY_EXPERIMENT",
    "ROBOT_CONTROL_PACKET_LOSS_EXPERIMENT",
    "FLEET_MANAGER_FAILURE_EXPERIMENT",
    "FLEET_DATABASE_PARTITION_EXPERIMENT",
    "SIMULATION_SYNC_LATENCY_EXPERIMENT",
    "COGNITIVE_LLM_TIMEOUT_EXPERIMENT",
    "COGNITIVE_REDIS_FAILURE_EXPERIMENT",
    "GPU_MEMORY_EXHAUSTION_EXPERIMENT",
    "CPU_STRESS_EXPERIMENT",
    "CASCADING_FAILURE_EXPERIMENT"
]

__version__ = "1.0.0"
