# Simulation Scenarios Module

Scenario-based testing for robot simulation and behavior validation.

## Features

- **Scenario Definition**: Declarative test scenarios
- **Step Types**: Commands, waits, assertions
- **Tag Filtering**: Run scenarios by category
- **Results Tracking**: Detailed execution reports

## Step Types

| Type | Description |
|------|-------------|
| `command` | Execute an action |
| `wait` | Pause for duration |
| `assert` | Verify condition |
| `parallel` | Run steps concurrently |
| `condition` | Conditional branching |

## Usage

### Define Scenario
```python
from scenarios import Scenario

async def move_robot(x, y):
    await robot.move_to(x, y)

async def check_position():
    pos = await robot.get_position()
    return pos.x == 10 and pos.y == 20

scenario = Scenario(
    name="navigation-test",
    description="Test robot navigation",
    tags=["navigation", "smoke"],
)

scenario.add_command("Move to start", move_robot, x=0, y=0)
scenario.add_wait("Stabilize", seconds=2.0)
scenario.add_command("Navigate", move_robot, x=10, y=20)
scenario.add_assert("Verify position", check_position)
```

### Run Scenario
```python
from scenarios import ScenarioRunner

runner = ScenarioRunner()
runner.register(scenario)

result = await runner.run("navigation-test")

print(f"Passed: {result.passed}")
print(f"Steps: {result.steps_passed}/{result.steps_total}")
print(f"Duration: {result.duration_seconds:.2f}s")
```

### Run by Tag
```python
# Run all navigation scenarios
results = await runner.run_all(tag="navigation")

for r in results:
    status = "✓" if r.passed else "✗"
    print(f"{status} {r.scenario_name}")
```

### Setup/Teardown
```python
async def setup():
    await simulation.reset()
    await robot.initialize()

async def teardown():
    await robot.stop()

scenario = Scenario(
    name="full-test",
    setup=setup,
    teardown=teardown,
)
```

## Result Analysis
```python
result = await runner.run("test")

for step in result.step_results:
    print(f"{step.name}: {step.status.value}")
    if step.error:
        print(f"  Error: {step.error}")
    print(f"  Duration: {step.duration_ms:.1f}ms")
```

## Integration

- Digital twin synchronization
- Isaac Sim environments
- Omniverse physics
- Hardware-in-the-loop testing
