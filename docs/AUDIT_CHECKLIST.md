# 🧪 Audit Commands Checklist

> Commands Copilot should propose/run and artifacts it should produce.

---

## 🖥️ System Prerequisites

```bash
# Verify GPU & Docker
nvidia-smi
docker --version
docker compose version

# Expected outputs:
# nvidia-smi: GPU name, driver version, CUDA version, memory usage
# docker: Docker version 24.x+
# compose: Docker Compose version v2.x+
```

---

## 🐳 Docker & Compose Validation

```bash
# Validate compose configuration
docker compose config

# Build all images
docker compose build --pull --no-cache

# Start services
docker compose up -d

# Check service health
docker compose ps
docker compose logs --tail=50

# Stop and cleanup
docker compose down -v

# Lint Dockerfiles
hadolint docker/Dockerfile.cognitive
hadolint docker/Dockerfile.ros2
hadolint docker/Dockerfile.isaac

# Expected artifacts:
# - docker-compose.yml validated output
# - Build logs for each service
# - Container health status
# - hadolint report (0 warnings)
```

---

## 🔒 Security Scanning

```bash
# Filesystem vulnerability scan
trivy fs . --severity HIGH,CRITICAL --format table
trivy fs . --severity HIGH,CRITICAL --format sarif -o trivy-fs.sarif

# Image vulnerability scan (after build)
trivy image dt_cognitive:latest --severity HIGH,CRITICAL
trivy image dt_ros2:latest --severity HIGH,CRITICAL
trivy image dt_isaac_sim:latest --severity HIGH,CRITICAL

# Python security scan
bandit -r cognitive_service/ -f json -o bandit-report.json
bandit -r scripts/ -f json -o bandit-scripts.json

# Expected artifacts:
# - trivy-fs.sarif (SARIF report for GitHub Security)
# - trivy-images.sarif (combined image scan)
# - bandit-report.json (Python security issues)
# - No HIGH or CRITICAL vulnerabilities
```

---

## 🐍 Python Linting & Type Checking

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Code formatting check
black --check --diff .

# Linting with ruff
ruff check . --output-format=github

# Type checking with mypy
mypy cognitive_service/ scripts/ tests/ --strict

# Run pre-commit on all files
pre-commit run --all-files

# Expected artifacts:
# - All checks pass with exit code 0
# - No formatting differences from black
# - No ruff violations
# - No mypy type errors
```

---

## 🧪 Python Testing

```bash
# Run unit tests with coverage
pytest tests/ -v --cov=. --cov-report=html --cov-report=xml

# Run integration tests
pytest tests/test_integration.py -v --tb=short

# Generate JUnit report for CI
pytest tests/ --junitxml=test-results.xml

# Expected artifacts:
# - test-results.xml (JUnit format)
# - htmlcov/ directory (HTML coverage report)
# - coverage.xml (Cobertura format)
# - All tests pass
```

---

## 🤖 ROS 2 Workspace Validation

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Update rosdep
rosdep update
rosdep install --from-paths ros2_ws/src -y --ignore-src

# Build workspace
cd ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Run colcon tests
colcon test
colcon test-result --verbose

# Expected artifacts:
# - Build logs (log/latest_build/)
# - Test results (log/latest_test/)
# - All packages build without warnings
# - All tests pass
```

---

## 🔍 ROS 2 Runtime Diagnostics

```bash
# Source workspace
source ros2_ws/install/setup.bash

# Run ROS 2 doctor
ros2 doctor --report

# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Check specific topics exist
ros2 topic info /cmd_vel
ros2 topic info /scan
ros2 topic info /odom
ros2 topic info /goal_pose
ros2 topic info /robot_intent

# Echo topics (once)
ros2 topic echo /cmd_vel --once
ros2 topic echo /scan --once

# Generate TF tree visualization
ros2 run tf2_tools view_frames
# Produces: frames.pdf

# Record a short bag for smoke test
ros2 bag record -a -o smoke_bag --max-duration 10

# Expected artifacts:
# - ros2_doctor_report.txt
# - node_list.txt
# - topic_list.txt
# - frames.pdf (TF tree)
# - smoke_bag/ (ROS bag)
```

---

## 🧭 Nav2 Validation

```bash
# Launch Nav2 stack
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=true \
    params_file:=config/nav2_params.yaml

# Check Nav2 lifecycle states
ros2 lifecycle list /controller_server
ros2 lifecycle list /planner_server
ros2 lifecycle list /recoveries_server
ros2 lifecycle list /bt_navigator

# Verify all nodes are active
ros2 lifecycle get /controller_server
# Expected: "active [3]"

# Send a test goal
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
    "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 0.0}}}}"

# Expected artifacts:
# - Nav2 launch logs
# - Lifecycle states for all Nav2 nodes
# - Goal acceptance confirmation
```

---

## 🎮 Isaac Sim Validation

```bash
# Run headless smoke test
python simulation/scripts/sim_smoke_test.py --headless --timeout 60

# Check ROS bridge topics
ros2 topic list | grep -E "(clock|scan|camera|odom)"

# Verify simulation clock
ros2 topic echo /clock --once

# Expected artifacts:
# - sim_smoke_test.log
# - simulation_frames/ (PNG sequence or MP4)
# - Test pass/fail status
```

---

## 🎤 Cognitive Service Validation

```bash
# Test ASR client with mock
python -m pytest cognitive_service/tests/ -v

# Test intent parser
python cognitive_service/src/intent_parser.py --test

# Validate Pydantic schemas
python -c "from cognitive_service.src.schemas import RobotIntent; print('Schema OK')"

# Expected artifacts:
# - ASR client test results
# - Intent parser test results
# - Schema validation confirmation
```

---

## 🔄 End-to-End Pipeline Test

```bash
# Run full pipeline test
python scripts/e2e_test.py --scenario navigate_to_dock

# Test scenarios:
# 1. Navigate to loading dock
python scripts/e2e_test.py --scenario navigate --target loading_dock

# 2. Emergency stop
python scripts/e2e_test.py --scenario stop

# 3. Ambiguous command
python scripts/e2e_test.py --scenario ambiguous

# Expected artifacts:
# - e2e_test_results.json
# - Per-scenario logs
# - Pass/fail summary
```

---

## 📊 CI Artifact Summary

| Artifact | Format | Purpose |
|----------|--------|---------|
| `test-results.xml` | JUnit XML | Test results for CI |
| `coverage.xml` | Cobertura | Code coverage |
| `htmlcov/` | HTML | Coverage visualization |
| `trivy-fs.sarif` | SARIF | Security vulnerabilities |
| `bandit-report.json` | JSON | Python security |
| `frames.pdf` | PDF | TF tree visualization |
| `smoke_bag/` | ROS bag | Runtime recording |
| `simulation_frames/` | PNG/MP4 | Sim visualization |
| `ros2_doctor_report.txt` | Text | ROS 2 health |
| `docker-compose-config.yml` | YAML | Validated compose |
| `hadolint-report.txt` | Text | Dockerfile lint |

---

## ✅ Pass Criteria

| Check | Requirement |
|-------|-------------|
| Docker build | All images build successfully |
| Compose config | Valid YAML, no errors |
| hadolint | 0 warnings (or documented exceptions) |
| trivy | No HIGH/CRITICAL vulnerabilities |
| bandit | No HIGH severity issues |
| black | No formatting differences |
| ruff | 0 violations |
| mypy | 0 type errors (strict mode) |
| pytest | 100% tests pass |
| colcon build | 0 warnings (-Werror) |
| colcon test | All ROS 2 tests pass |
| ros2 doctor | No errors |
| Nav2 lifecycle | All nodes active |
| Isaac Sim smoke | Robot reaches target |
| E2E scenarios | 3/3 pass |

---

## 🚀 Quick Start Commands

```bash
# One-liner: Full lint check
pre-commit run --all-files && pytest tests/ -v

# One-liner: Build and test Docker
docker compose build && docker compose up -d && docker compose ps && docker compose down

# One-liner: ROS 2 build and test
cd ros2_ws && colcon build && colcon test && colcon test-result --verbose

# One-liner: Security scan
trivy fs . --severity HIGH,CRITICAL && bandit -r . -ll
```

---

## 🔧 Makefile / Taskfile Targets

These commands should be available via `make` or `task`:

```bash
make setup          # Install dependencies
make lint           # Run all linters
make test           # Run all tests
make build          # Build Docker images
make up             # Start services
make down           # Stop services
make clean          # Remove build artifacts
make security       # Run security scans
make ros2-build     # Build ROS 2 workspace
make ros2-test      # Test ROS 2 packages
make ros2-diag      # Run ROS 2 diagnostics
make sim-smoke      # Run Isaac Sim smoke test
make e2e            # Run end-to-end tests
make ci             # Run full CI pipeline locally
make audit          # Generate audit report
```
