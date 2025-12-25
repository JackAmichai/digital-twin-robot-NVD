"""
Chaos Engineering Framework for Digital Twin Robotics Lab.

Provides failure injection capabilities to test system resilience:
- Network failures (latency, packet loss, partition)
- Service failures (crash, resource exhaustion)
- Infrastructure failures (node, pod, container)
- Application failures (exceptions, timeouts)

Based on principles from Chaos Monkey, Litmus, and Chaos Toolkit.

Example:
    >>> from chaos.engine import ChaosEngine
    >>> engine = ChaosEngine()
    >>> 
    >>> # Run a chaos experiment
    >>> result = engine.run_experiment("network_latency", {
    ...     "target": "voice-processing",
    ...     "latency_ms": 500,
    ...     "duration_seconds": 60
    ... })
"""

import asyncio
import random
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)


class FaultType(str, Enum):
    """Types of faults that can be injected."""
    # Network faults
    NETWORK_LATENCY = "network_latency"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    NETWORK_PARTITION = "network_partition"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DNS_FAILURE = "dns_failure"
    
    # Service faults
    SERVICE_CRASH = "service_crash"
    SERVICE_RESTART = "service_restart"
    SERVICE_UNAVAILABLE = "service_unavailable"
    
    # Resource faults
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    IO_STRESS = "io_stress"
    
    # Application faults
    EXCEPTION_INJECTION = "exception_injection"
    RESPONSE_DELAY = "response_delay"
    ERROR_RESPONSE = "error_response"
    TIMEOUT = "timeout"
    
    # Infrastructure faults
    POD_KILL = "pod_kill"
    NODE_DRAIN = "node_drain"
    CONTAINER_KILL = "container_kill"


class ExperimentState(str, Enum):
    """State of a chaos experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ABORTED = "aborted"


@dataclass
class SteadyStateHypothesis:
    """
    Defines the expected steady state of the system.
    
    Used to verify system health before and after chaos injection.
    """
    name: str
    probes: List[Dict[str, Any]]
    tolerance: float = 0.95  # 95% success rate required
    
    async def verify(self) -> bool:
        """Verify all probes meet tolerance."""
        results = []
        for probe in self.probes:
            result = await self._run_probe(probe)
            results.append(result)
        
        success_rate = sum(results) / len(results) if results else 0
        return success_rate >= self.tolerance
    
    async def _run_probe(self, probe: Dict[str, Any]) -> bool:
        """Run a single probe."""
        probe_type = probe.get("type")
        
        if probe_type == "http":
            return await self._http_probe(probe)
        elif probe_type == "process":
            return await self._process_probe(probe)
        elif probe_type == "metric":
            return await self._metric_probe(probe)
        
        return False
    
    async def _http_probe(self, probe: Dict[str, Any]) -> bool:
        """HTTP health check probe."""
        import aiohttp
        
        url = probe.get("url")
        expected_status = probe.get("expected_status", 200)
        timeout = probe.get("timeout", 5)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    return response.status == expected_status
        except Exception:
            return False
    
    async def _process_probe(self, probe: Dict[str, Any]) -> bool:
        """Process existence probe."""
        import subprocess
        
        process_name = probe.get("process_name")
        try:
            result = subprocess.run(
                ["pgrep", "-f", process_name],
                capture_output=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def _metric_probe(self, probe: Dict[str, Any]) -> bool:
        """Prometheus metric probe."""
        # Simplified - would query Prometheus in production
        return True


@dataclass
class ChaosAction:
    """
    Defines a chaos action to be executed.
    """
    fault_type: FaultType
    target: str  # Service, pod, or resource name
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: int = 60  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type.value,
            "target": self.target,
            "parameters": self.parameters,
            "duration": self.duration
        }


@dataclass
class ExperimentResult:
    """
    Result of a chaos experiment.
    """
    experiment_id: str
    name: str
    state: ExperimentState
    start_time: datetime
    end_time: Optional[datetime] = None
    steady_state_before: bool = False
    steady_state_after: bool = False
    actions_executed: List[Dict[str, Any]] = field(default_factory=list)
    rollback_executed: bool = False
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Experiment is successful if system recovered."""
        return (
            self.state == ExperimentState.COMPLETED and
            self.steady_state_before and
            self.steady_state_after
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "state": self.state.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "steady_state_before": self.steady_state_before,
            "steady_state_after": self.steady_state_after,
            "actions_executed": self.actions_executed,
            "rollback_executed": self.rollback_executed,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics
        }


@dataclass
class ChaosExperiment:
    """
    Defines a complete chaos experiment.
    
    Structure:
    1. Verify steady state hypothesis (system is healthy)
    2. Inject chaos (execute actions)
    3. Wait for duration
    4. Rollback chaos
    5. Verify steady state again (system recovered)
    """
    name: str
    description: str
    steady_state: SteadyStateHypothesis
    actions: List[ChaosAction]
    rollback_actions: List[ChaosAction] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steady_state": {
                "name": self.steady_state.name,
                "probes": self.steady_state.probes
            },
            "actions": [a.to_dict() for a in self.actions],
            "rollback_actions": [a.to_dict() for a in self.rollback_actions],
            "tags": self.tags
        }


class FaultInjector(ABC):
    """Base class for fault injectors."""
    
    @abstractmethod
    async def inject(self, action: ChaosAction) -> bool:
        """Inject the fault."""
        pass
    
    @abstractmethod
    async def rollback(self, action: ChaosAction) -> bool:
        """Rollback the fault."""
        pass


class NetworkFaultInjector(FaultInjector):
    """
    Injects network-related faults using tc (traffic control).
    
    Supports:
    - Latency injection
    - Packet loss
    - Network partition
    - Bandwidth limiting
    """
    
    async def inject(self, action: ChaosAction) -> bool:
        """Inject network fault."""
        if action.fault_type == FaultType.NETWORK_LATENCY:
            return await self._inject_latency(action)
        elif action.fault_type == FaultType.NETWORK_PACKET_LOSS:
            return await self._inject_packet_loss(action)
        elif action.fault_type == FaultType.NETWORK_PARTITION:
            return await self._inject_partition(action)
        elif action.fault_type == FaultType.NETWORK_BANDWIDTH:
            return await self._inject_bandwidth_limit(action)
        
        return False
    
    async def rollback(self, action: ChaosAction) -> bool:
        """Remove network fault."""
        # Remove tc rules
        cmd = f"tc qdisc del dev eth0 root 2>/dev/null || true"
        return await self._execute_command(cmd)
    
    async def _inject_latency(self, action: ChaosAction) -> bool:
        """Add network latency."""
        latency_ms = action.parameters.get("latency_ms", 100)
        jitter_ms = action.parameters.get("jitter_ms", 10)
        interface = action.parameters.get("interface", "eth0")
        
        cmd = (
            f"tc qdisc add dev {interface} root netem "
            f"delay {latency_ms}ms {jitter_ms}ms distribution normal"
        )
        
        logger.info(f"Injecting latency: {latency_ms}ms ±{jitter_ms}ms on {interface}")
        return await self._execute_command(cmd)
    
    async def _inject_packet_loss(self, action: ChaosAction) -> bool:
        """Add packet loss."""
        loss_percent = action.parameters.get("loss_percent", 5)
        interface = action.parameters.get("interface", "eth0")
        
        cmd = f"tc qdisc add dev {interface} root netem loss {loss_percent}%"
        
        logger.info(f"Injecting packet loss: {loss_percent}% on {interface}")
        return await self._execute_command(cmd)
    
    async def _inject_partition(self, action: ChaosAction) -> bool:
        """Create network partition."""
        target_ip = action.parameters.get("target_ip")
        
        if not target_ip:
            return False
        
        cmd = f"iptables -A OUTPUT -d {target_ip} -j DROP"
        
        logger.info(f"Creating network partition to {target_ip}")
        return await self._execute_command(cmd)
    
    async def _inject_bandwidth_limit(self, action: ChaosAction) -> bool:
        """Limit network bandwidth."""
        rate = action.parameters.get("rate", "1mbit")
        interface = action.parameters.get("interface", "eth0")
        
        cmd = f"tc qdisc add dev {interface} root tbf rate {rate} burst 32kbit latency 400ms"
        
        logger.info(f"Limiting bandwidth to {rate} on {interface}")
        return await self._execute_command(cmd)
    
    async def _execute_command(self, cmd: str) -> bool:
        """Execute shell command."""
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False


class ResourceFaultInjector(FaultInjector):
    """
    Injects resource-related faults.
    
    Supports:
    - CPU stress
    - Memory stress
    - Disk I/O stress
    """
    
    def __init__(self):
        self._stress_processes: Dict[str, asyncio.subprocess.Process] = {}
    
    async def inject(self, action: ChaosAction) -> bool:
        """Inject resource fault."""
        if action.fault_type == FaultType.CPU_STRESS:
            return await self._inject_cpu_stress(action)
        elif action.fault_type == FaultType.MEMORY_STRESS:
            return await self._inject_memory_stress(action)
        elif action.fault_type == FaultType.IO_STRESS:
            return await self._inject_io_stress(action)
        
        return False
    
    async def rollback(self, action: ChaosAction) -> bool:
        """Stop resource stress."""
        key = f"{action.fault_type.value}_{action.target}"
        
        if key in self._stress_processes:
            process = self._stress_processes[key]
            process.terminate()
            await process.wait()
            del self._stress_processes[key]
            logger.info(f"Stopped stress process: {key}")
        
        return True
    
    async def _inject_cpu_stress(self, action: ChaosAction) -> bool:
        """Stress CPU."""
        cpu_percent = action.parameters.get("cpu_percent", 80)
        workers = action.parameters.get("workers", 4)
        
        # Using stress-ng
        cmd = f"stress-ng --cpu {workers} --cpu-load {cpu_percent} --timeout {action.duration}s"
        
        logger.info(f"Starting CPU stress: {cpu_percent}% with {workers} workers")
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self._stress_processes[f"{action.fault_type.value}_{action.target}"] = process
        return True
    
    async def _inject_memory_stress(self, action: ChaosAction) -> bool:
        """Stress memory."""
        memory_percent = action.parameters.get("memory_percent", 80)
        
        cmd = f"stress-ng --vm 1 --vm-bytes {memory_percent}% --timeout {action.duration}s"
        
        logger.info(f"Starting memory stress: {memory_percent}%")
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self._stress_processes[f"{action.fault_type.value}_{action.target}"] = process
        return True
    
    async def _inject_io_stress(self, action: ChaosAction) -> bool:
        """Stress disk I/O."""
        io_workers = action.parameters.get("io_workers", 4)
        
        cmd = f"stress-ng --io {io_workers} --timeout {action.duration}s"
        
        logger.info(f"Starting I/O stress with {io_workers} workers")
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self._stress_processes[f"{action.fault_type.value}_{action.target}"] = process
        return True


class KubernetesFaultInjector(FaultInjector):
    """
    Injects Kubernetes-related faults.
    
    Supports:
    - Pod kill
    - Node drain
    - Container kill
    """
    
    def __init__(self, namespace: str = "digital-twin"):
        self.namespace = namespace
    
    async def inject(self, action: ChaosAction) -> bool:
        """Inject Kubernetes fault."""
        if action.fault_type == FaultType.POD_KILL:
            return await self._kill_pod(action)
        elif action.fault_type == FaultType.NODE_DRAIN:
            return await self._drain_node(action)
        elif action.fault_type == FaultType.CONTAINER_KILL:
            return await self._kill_container(action)
        
        return False
    
    async def rollback(self, action: ChaosAction) -> bool:
        """Rollback is typically automatic in K8s (pod restart)."""
        if action.fault_type == FaultType.NODE_DRAIN:
            return await self._uncordon_node(action)
        return True
    
    async def _kill_pod(self, action: ChaosAction) -> bool:
        """Kill a pod."""
        pod_selector = action.parameters.get("selector", action.target)
        grace_period = action.parameters.get("grace_period", 0)
        
        cmd = (
            f"kubectl delete pod -l app={pod_selector} "
            f"-n {self.namespace} --grace-period={grace_period}"
        )
        
        logger.info(f"Killing pod: {pod_selector}")
        return await self._execute_kubectl(cmd)
    
    async def _drain_node(self, action: ChaosAction) -> bool:
        """Drain a node."""
        node_name = action.target
        
        cmd = f"kubectl drain {node_name} --ignore-daemonsets --delete-emptydir-data"
        
        logger.info(f"Draining node: {node_name}")
        return await self._execute_kubectl(cmd)
    
    async def _uncordon_node(self, action: ChaosAction) -> bool:
        """Uncordon a node."""
        node_name = action.target
        
        cmd = f"kubectl uncordon {node_name}"
        
        logger.info(f"Uncordoning node: {node_name}")
        return await self._execute_kubectl(cmd)
    
    async def _kill_container(self, action: ChaosAction) -> bool:
        """Kill a specific container in a pod."""
        pod_name = action.target
        container_name = action.parameters.get("container")
        
        cmd = (
            f"kubectl exec {pod_name} -n {self.namespace} -c {container_name} "
            f"-- kill 1"
        )
        
        logger.info(f"Killing container {container_name} in pod {pod_name}")
        return await self._execute_kubectl(cmd)
    
    async def _execute_kubectl(self, cmd: str) -> bool:
        """Execute kubectl command."""
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"kubectl failed: {stderr.decode()}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"kubectl execution failed: {e}")
            return False


class ApplicationFaultInjector(FaultInjector):
    """
    Injects application-level faults via middleware/hooks.
    
    Supports:
    - Exception injection
    - Response delays
    - Error responses
    """
    
    # Global registry of fault configurations
    _active_faults: Dict[str, Dict[str, Any]] = {}
    
    async def inject(self, action: ChaosAction) -> bool:
        """Register application fault."""
        fault_id = f"{action.target}_{action.fault_type.value}"
        
        self._active_faults[fault_id] = {
            "type": action.fault_type,
            "target": action.target,
            "parameters": action.parameters,
            "start_time": time.time(),
            "duration": action.duration
        }
        
        logger.info(f"Registered application fault: {fault_id}")
        return True
    
    async def rollback(self, action: ChaosAction) -> bool:
        """Remove application fault."""
        fault_id = f"{action.target}_{action.fault_type.value}"
        
        if fault_id in self._active_faults:
            del self._active_faults[fault_id]
            logger.info(f"Removed application fault: {fault_id}")
        
        return True
    
    @classmethod
    def should_inject_fault(cls, endpoint: str) -> Optional[Dict[str, Any]]:
        """Check if fault should be injected for endpoint."""
        for fault_id, fault in cls._active_faults.items():
            if fault["target"] in endpoint:
                # Check if still within duration
                elapsed = time.time() - fault["start_time"]
                if elapsed < fault["duration"]:
                    return fault
        return None


class ChaosEngine:
    """
    Main chaos engineering engine.
    
    Coordinates experiment execution, fault injection, and rollback.
    
    Example:
        >>> engine = ChaosEngine()
        >>> 
        >>> # Define experiment
        >>> experiment = ChaosExperiment(
        ...     name="voice-service-latency",
        ...     description="Test voice service under network latency",
        ...     steady_state=SteadyStateHypothesis(
        ...         name="Voice service healthy",
        ...         probes=[{"type": "http", "url": "http://voice:8080/health"}]
        ...     ),
        ...     actions=[
        ...         ChaosAction(
        ...             fault_type=FaultType.NETWORK_LATENCY,
        ...             target="voice-processing",
        ...             parameters={"latency_ms": 500},
        ...             duration=60
        ...         )
        ...     ]
        ... )
        >>> 
        >>> # Run experiment
        >>> result = await engine.run_experiment(experiment)
        >>> print(f"Success: {result.success}")
    """
    
    def __init__(self, namespace: str = "digital-twin"):
        self.namespace = namespace
        self.injectors: Dict[str, FaultInjector] = {
            "network": NetworkFaultInjector(),
            "resource": ResourceFaultInjector(),
            "kubernetes": KubernetesFaultInjector(namespace),
            "application": ApplicationFaultInjector()
        }
        self._running_experiments: Dict[str, ExperimentResult] = {}
        self._abort_signals: Set[str] = set()
    
    def get_injector(self, fault_type: FaultType) -> FaultInjector:
        """Get appropriate injector for fault type."""
        if fault_type in [FaultType.NETWORK_LATENCY, FaultType.NETWORK_PACKET_LOSS,
                          FaultType.NETWORK_PARTITION, FaultType.NETWORK_BANDWIDTH,
                          FaultType.DNS_FAILURE]:
            return self.injectors["network"]
        elif fault_type in [FaultType.CPU_STRESS, FaultType.MEMORY_STRESS,
                            FaultType.DISK_STRESS, FaultType.IO_STRESS]:
            return self.injectors["resource"]
        elif fault_type in [FaultType.POD_KILL, FaultType.NODE_DRAIN,
                            FaultType.CONTAINER_KILL]:
            return self.injectors["kubernetes"]
        else:
            return self.injectors["application"]
    
    async def run_experiment(
        self,
        experiment: ChaosExperiment,
        dry_run: bool = False
    ) -> ExperimentResult:
        """
        Run a chaos experiment.
        
        Args:
            experiment: Experiment to run
            dry_run: If True, don't actually inject faults
            
        Returns:
            ExperimentResult with outcome
        """
        experiment_id = f"exp_{int(time.time())}_{random.randint(1000, 9999)}"
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            name=experiment.name,
            state=ExperimentState.PENDING,
            start_time=datetime.now()
        )
        
        self._running_experiments[experiment_id] = result
        
        try:
            # Phase 1: Verify steady state before
            logger.info(f"[{experiment_id}] Verifying steady state before chaos...")
            result.steady_state_before = await experiment.steady_state.verify()
            
            if not result.steady_state_before:
                logger.error(f"[{experiment_id}] System not in steady state, aborting")
                result.state = ExperimentState.FAILED
                result.error = "System not in steady state before experiment"
                return result
            
            # Phase 2: Inject chaos
            result.state = ExperimentState.RUNNING
            logger.info(f"[{experiment_id}] Injecting chaos...")
            
            for action in experiment.actions:
                if experiment_id in self._abort_signals:
                    raise Exception("Experiment aborted")
                
                if not dry_run:
                    injector = self.get_injector(action.fault_type)
                    success = await injector.inject(action)
                    
                    result.actions_executed.append({
                        "action": action.to_dict(),
                        "success": success,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    if not success:
                        logger.warning(f"[{experiment_id}] Action failed: {action.fault_type}")
            
            # Phase 3: Wait for chaos duration
            max_duration = max(a.duration for a in experiment.actions) if experiment.actions else 60
            logger.info(f"[{experiment_id}] Chaos active for {max_duration}s...")
            
            await self._wait_with_abort_check(experiment_id, max_duration)
            
            # Phase 4: Rollback
            logger.info(f"[{experiment_id}] Rolling back chaos...")
            
            if not dry_run:
                for action in experiment.actions:
                    injector = self.get_injector(action.fault_type)
                    await injector.rollback(action)
                
                for action in experiment.rollback_actions:
                    injector = self.get_injector(action.fault_type)
                    await injector.inject(action)
            
            result.rollback_executed = True
            
            # Phase 5: Wait for recovery
            logger.info(f"[{experiment_id}] Waiting for system recovery...")
            await asyncio.sleep(10)  # Recovery grace period
            
            # Phase 6: Verify steady state after
            logger.info(f"[{experiment_id}] Verifying steady state after chaos...")
            result.steady_state_after = await experiment.steady_state.verify()
            
            result.state = ExperimentState.COMPLETED
            result.end_time = datetime.now()
            
            if result.success:
                logger.info(f"[{experiment_id}] ✅ Experiment PASSED - System recovered")
            else:
                logger.warning(f"[{experiment_id}] ❌ Experiment FAILED - System did not recover")
            
        except Exception as e:
            logger.error(f"[{experiment_id}] Experiment error: {e}")
            result.state = ExperimentState.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            
            # Emergency rollback
            for action in experiment.actions:
                try:
                    injector = self.get_injector(action.fault_type)
                    await injector.rollback(action)
                except Exception:
                    pass
        
        finally:
            del self._running_experiments[experiment_id]
            if experiment_id in self._abort_signals:
                self._abort_signals.remove(experiment_id)
        
        return result
    
    async def _wait_with_abort_check(self, experiment_id: str, duration: int):
        """Wait with periodic abort check."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            if experiment_id in self._abort_signals:
                raise Exception("Experiment aborted")
            await asyncio.sleep(1)
    
    def abort_experiment(self, experiment_id: str):
        """Signal experiment to abort."""
        self._abort_signals.add(experiment_id)
        logger.info(f"Abort signal sent for experiment: {experiment_id}")
    
    def list_running_experiments(self) -> List[Dict[str, Any]]:
        """List all running experiments."""
        return [r.to_dict() for r in self._running_experiments.values()]


# =============================================================================
# Pre-defined Experiments
# =============================================================================

def create_network_latency_experiment(
    target: str,
    latency_ms: int = 500,
    duration: int = 60
) -> ChaosExperiment:
    """Create a network latency experiment."""
    return ChaosExperiment(
        name=f"{target}-network-latency",
        description=f"Test {target} service under {latency_ms}ms network latency",
        steady_state=SteadyStateHypothesis(
            name=f"{target} healthy",
            probes=[
                {"type": "http", "url": f"http://{target}:8080/health", "expected_status": 200}
            ]
        ),
        actions=[
            ChaosAction(
                fault_type=FaultType.NETWORK_LATENCY,
                target=target,
                parameters={"latency_ms": latency_ms, "jitter_ms": latency_ms // 10},
                duration=duration
            )
        ],
        tags=["network", "latency"]
    )


def create_pod_failure_experiment(
    target: str,
    count: int = 1
) -> ChaosExperiment:
    """Create a pod failure experiment."""
    return ChaosExperiment(
        name=f"{target}-pod-failure",
        description=f"Test system resilience when {count} {target} pod(s) fail",
        steady_state=SteadyStateHypothesis(
            name=f"{target} pods running",
            probes=[
                {"type": "http", "url": f"http://{target}:8080/health"}
            ]
        ),
        actions=[
            ChaosAction(
                fault_type=FaultType.POD_KILL,
                target=target,
                parameters={"selector": target, "count": count},
                duration=120
            )
        ],
        tags=["kubernetes", "pod", "failure"]
    )


def create_resource_exhaustion_experiment(
    target: str,
    cpu_percent: int = 90,
    memory_percent: int = 80
) -> ChaosExperiment:
    """Create a resource exhaustion experiment."""
    return ChaosExperiment(
        name=f"{target}-resource-exhaustion",
        description=f"Test {target} under resource pressure (CPU: {cpu_percent}%, Memory: {memory_percent}%)",
        steady_state=SteadyStateHypothesis(
            name=f"{target} responsive",
            probes=[
                {"type": "http", "url": f"http://{target}:8080/health", "timeout": 5}
            ]
        ),
        actions=[
            ChaosAction(
                fault_type=FaultType.CPU_STRESS,
                target=target,
                parameters={"cpu_percent": cpu_percent, "workers": 4},
                duration=60
            ),
            ChaosAction(
                fault_type=FaultType.MEMORY_STRESS,
                target=target,
                parameters={"memory_percent": memory_percent},
                duration=60
            )
        ],
        tags=["resource", "stress", "cpu", "memory"]
    )


__all__ = [
    'FaultType',
    'ExperimentState',
    'SteadyStateHypothesis',
    'ChaosAction',
    'ChaosExperiment',
    'ExperimentResult',
    'ChaosEngine',
    'FaultInjector',
    'NetworkFaultInjector',
    'ResourceFaultInjector',
    'KubernetesFaultInjector',
    'ApplicationFaultInjector',
    'create_network_latency_experiment',
    'create_pod_failure_experiment',
    'create_resource_exhaustion_experiment',
]
