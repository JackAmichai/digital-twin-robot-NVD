"""Simulation scenario definition and execution."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4


class StepType(Enum):
    """Types of scenario steps."""
    COMMAND = "command"
    WAIT = "wait"
    ASSERT = "assert"
    PARALLEL = "parallel"
    CONDITION = "condition"


class StepStatus(Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScenarioStep:
    """Individual scenario step."""
    
    name: str
    step_type: StepType
    action: Callable[..., Awaitable[Any]] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    status: StepStatus = StepStatus.PENDING
    error: str | None = None
    result: Any = None
    duration_ms: float = 0.0


@dataclass
class ScenarioResult:
    """Result of scenario execution."""
    
    scenario_name: str
    passed: bool = False
    steps_total: int = 0
    steps_passed: int = 0
    steps_failed: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    step_results: list[ScenarioStep] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return self.steps_passed / self.steps_total


@dataclass
class Scenario:
    """Simulation scenario definition."""
    
    name: str
    description: str = ""
    steps: list[ScenarioStep] = field(default_factory=list)
    setup: Callable[[], Awaitable[None]] | None = None
    teardown: Callable[[], Awaitable[None]] | None = None
    tags: list[str] = field(default_factory=list)
    timeout_seconds: float = 300.0
    
    def add_command(
        self,
        name: str,
        action: Callable[..., Awaitable[Any]],
        **params: Any,
    ) -> "Scenario":
        """Add command step."""
        self.steps.append(ScenarioStep(
            name=name,
            step_type=StepType.COMMAND,
            action=action,
            params=params,
        ))
        return self
    
    def add_wait(self, name: str, seconds: float) -> "Scenario":
        """Add wait step."""
        self.steps.append(ScenarioStep(
            name=name,
            step_type=StepType.WAIT,
            params={"seconds": seconds},
        ))
        return self
    
    def add_assert(
        self,
        name: str,
        condition: Callable[[], Awaitable[bool]],
    ) -> "Scenario":
        """Add assertion step."""
        self.steps.append(ScenarioStep(
            name=name,
            step_type=StepType.ASSERT,
            action=condition,
        ))
        return self


class ScenarioRunner:
    """Execute simulation scenarios."""
    
    def __init__(self) -> None:
        self._scenarios: dict[str, Scenario] = {}
        self._results: list[ScenarioResult] = []
    
    def register(self, scenario: Scenario) -> None:
        """Register scenario for execution."""
        self._scenarios[scenario.name] = scenario
    
    def get_scenario(self, name: str) -> Scenario | None:
        """Get scenario by name."""
        return self._scenarios.get(name)
    
    def list_scenarios(self, tag: str | None = None) -> list[Scenario]:
        """List scenarios, optionally filtered by tag."""
        scenarios = list(self._scenarios.values())
        if tag:
            scenarios = [s for s in scenarios if tag in s.tags]
        return scenarios
    
    async def run(self, scenario_name: str) -> ScenarioResult:
        """Run a single scenario."""
        scenario = self._scenarios.get(scenario_name)
        if not scenario:
            return ScenarioResult(
                scenario_name=scenario_name,
                error=f"Scenario not found: {scenario_name}",
            )
        
        result = ScenarioResult(
            scenario_name=scenario_name,
            steps_total=len(scenario.steps),
        )
        
        start_time = datetime.utcnow()
        
        try:
            # Setup
            if scenario.setup:
                await scenario.setup()
            
            # Execute steps
            for step in scenario.steps:
                step_result = await self._execute_step(step)
                result.step_results.append(step_result)
                
                if step_result.status == StepStatus.PASSED:
                    result.steps_passed += 1
                elif step_result.status == StepStatus.FAILED:
                    result.steps_failed += 1
                    break  # Stop on first failure
            
            result.passed = result.steps_failed == 0
            
        except Exception as e:
            result.error = str(e)
            result.passed = False
        finally:
            # Teardown
            if scenario.teardown:
                try:
                    await scenario.teardown()
                except Exception:
                    pass
            
            result.duration_seconds = (
                datetime.utcnow() - start_time
            ).total_seconds()
        
        self._results.append(result)
        return result
    
    async def _execute_step(self, step: ScenarioStep) -> ScenarioStep:
        """Execute a single step."""
        step.status = StepStatus.RUNNING
        start = datetime.utcnow()
        
        try:
            if step.step_type == StepType.WAIT:
                await asyncio.sleep(step.params.get("seconds", 1))
                step.status = StepStatus.PASSED
            
            elif step.step_type == StepType.COMMAND:
                if step.action:
                    step.result = await asyncio.wait_for(
                        step.action(**step.params),
                        timeout=step.timeout_seconds,
                    )
                step.status = StepStatus.PASSED
            
            elif step.step_type == StepType.ASSERT:
                if step.action:
                    passed = await step.action()
                    step.status = StepStatus.PASSED if passed else StepStatus.FAILED
                    if not passed:
                        step.error = "Assertion failed"
        
        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error = f"Timeout after {step.timeout_seconds}s"
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
        
        step.duration_ms = (datetime.utcnow() - start).total_seconds() * 1000
        return step
    
    async def run_all(self, tag: str | None = None) -> list[ScenarioResult]:
        """Run all matching scenarios."""
        scenarios = self.list_scenarios(tag)
        results = []
        for scenario in scenarios:
            result = await self.run(scenario.name)
            results.append(result)
        return results
    
    def get_results(self) -> list[ScenarioResult]:
        """Get all execution results."""
        return self._results
