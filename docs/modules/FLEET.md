# Fleet Management Module

## Overview
Multi-robot coordination with intelligent task allocation.

## Files

### `fleet_manager.py`
Central coordinator for robot fleet.

```python
class FleetManager:
    def register_robot(robot_id, capabilities) -> Robot
    def allocate_task(task) -> Optional[Robot]
    def get_fleet_status() -> FleetStatus
```

### `task_allocator.py`
Task assignment logic using weighted scoring.

**Algorithm:**
1. Filter robots by capabilities
2. Score by: proximity, battery, load
3. Select optimal robot
4. Assign and update state

### `robot.py`
Robot state machine.

**States:** IDLE, BUSY, CHARGING, MAINTENANCE, OFFLINE

## Usage

```python
from fleet import FleetManager

manager = FleetManager()
manager.register_robot("robot-001", ["pick", "place", "navigate"])
robot = manager.allocate_task(Task(type="pick", target="box-1"))
```
