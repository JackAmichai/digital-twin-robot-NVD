# Quick Start Guide

Get up and running with Digital Twin Robotics Lab in minutes.

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.0+ (optional for GPU features)
- Kubernetes cluster (optional for deployment)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/JackAmichai/digital-twin-robot-NVD.git
cd digital-twin-robot-NVD
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

## Running Locally

### Start Core Services

```bash
# Start with Docker Compose
docker-compose up -d

# Or run directly
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Services

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

## Your First Robot

### 1. Register a Robot

```python
from fleet import FleetManager, Robot, RobotCapability

manager = FleetManager()

robot = Robot(
    name="MyRobot-001",
    capabilities=[RobotCapability.NAVIGATION, RobotCapability.MANIPULATION],
)

manager.register_robot(robot)
```

### 2. Create a Task

```python
from fleet import Task, TaskPriority

task = Task(
    name="pickup-item",
    priority=TaskPriority.HIGH,
    required_capabilities=[RobotCapability.MANIPULATION],
)

assignment = await manager.allocate_task(task)
print(f"Assigned to: {assignment.robot_id}")
```

### 3. Monitor Status

```python
status = manager.get_robot_status(robot.id)
print(f"State: {status.state}")
print(f"Battery: {status.battery_level}%")
```

## Next Steps

- [Configure Voice Processing](modules/VOICE.md)
- [Set Up Object Detection](modules/PERCEPTION.md)
- [Deploy to Kubernetes](modules/INFRASTRUCTURE.md)
- [Configure Monitoring](modules/INFRASTRUCTURE.md#monitoring)

## Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Connection Refused**
```bash
# Check services are running
docker-compose ps

# View logs
docker-compose logs -f
```

## Getting Help

- GitHub Issues: [Report Bug](https://github.com/JackAmichai/digital-twin-robot-NVD/issues)
- Documentation: [Full Docs](index.md)
