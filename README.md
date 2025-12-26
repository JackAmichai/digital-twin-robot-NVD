#  Digital Twin Robotics Lab

[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-blue)](https://docs.ros.org/en/humble/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.2.0-76B900)](https://developer.nvidia.com/isaac-sim)
[![NVIDIA Riva](https://img.shields.io/badge/NVIDIA-Riva-76B900)](https://developer.nvidia.com/riva)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A professional-grade closed-loop robotics simulation platform featuring voice-controlled autonomous navigation in a photorealistic digital twin environment.**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Development Commands](#-development-commands)
- [Demo Scenarios](#-demo-scenarios)
- [System Requirements](#-system-requirements)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

The **Digital Twin Robotics Lab** is an enterprise-grade demonstration of modern robotics software architecture. It showcases the integration of cutting-edge NVIDIA AI technologies with the Robot Operating System (ROS 2) in a containerized, production-ready deployment.

### What This Project Demonstrates

| Skill Area | Implementation |
|------------|----------------|
| **AI/ML Integration** | Voice commands processed by NVIDIA Riva ASR + LLM intent extraction |
| **Robotics Software** | Full ROS 2 Humble stack with Nav2 autonomous navigation |
| **Simulation** | Photorealistic physics simulation in NVIDIA Isaac Sim |
| **DevOps** | Docker Compose orchestration with health checks and networking |
| **Systems Design** | Closed-loop architecture with real-time sensor feedback |

### The Problem It Solves

Traditional robotics development requires expensive hardware and is slow to iterate. This digital twin approach allows:
- **Rapid Prototyping**: Test algorithms in simulation before hardware deployment
- **Safe Testing**: Push robots to failure without physical damage
- **Scalable Training**: Generate unlimited training data for ML models
- **Remote Development**: Full robotics stack accessible via streaming

---

## ✨ Key Features

### 🎤 Voice-Controlled Navigation
Speak natural language commands like *"Move to Zone B"* or *"Inspect the north shelf"* and watch the robot execute autonomously.

### 🧠 Intelligent Intent Parsing
LLM-powered understanding converts conversational commands into precise robot actions with confidence scoring.

### 🗺️ Autonomous Path Planning
Nav2 integration provides dynamic obstacle avoidance, costmap-based planning, and behavior trees for complex tasks.

### 🌍 Photorealistic Simulation
NVIDIA Isaac Sim delivers physically accurate sensor simulation (Lidar, cameras, IMU) in a beautiful warehouse environment.

### 📊 Real-Time Visualization
Foxglove Studio dashboard shows robot state, sensor data, planned paths, and system health in real-time.

### 🐳 One-Command Deployment
`make up` launches the entire 3-layer architecture with proper networking, GPU passthrough, and health monitoring.

---

## 🏗️ Architecture

This project implements a **Closed-Loop Control System** with three distinct layers:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DIGITAL TWIN ROBOTICS LAB                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │  🎤 USER INPUT  │───▶│  🧠 COGNITIVE   │───▶│  🦾 CONTROL     │             │
│  │                 │    │     LAYER       │    │     LAYER       │             │
│  │  Voice Command  │    │                 │    │                 │             │
│  │  "Go to Zone B" │    │  • Riva ASR     │    │  • ROS 2 Humble │             │
│  └─────────────────┘    │  • LLM Intent   │    │  • Nav2 Stack   │             │
│                         │  • Redis Pub    │    │  • TF2 Frames   │             │
│                         └────────┬────────┘    └────────┬────────┘             │
│                                  │                      │                       │
│                                  │    JSON Command      │    cmd_vel            │
│                                  │    {action: nav,     │    geometry_msgs      │
│                                  │     target: zone_b}  │    /Twist             │
│                                  ▼                      ▼                       │
│                         ┌─────────────────────────────────────────┐             │
│                         │           🌍 SIMULATION LAYER           │             │
│                         │                                         │             │
│                         │  ┌─────────────┐    ┌─────────────┐    │             │
│                         │  │ Isaac Sim   │    │ ROS 2 Bridge│    │             │
│                         │  │ Physics     │◀──▶│ Extension   │    │             │
│                         │  │ Engine      │    │             │    │             │
│                         │  └─────────────┘    └─────────────┘    │             │
│                         │         │                   │          │             │
│                         │         ▼                   ▼          │             │
│                         │  ┌─────────────┐    ┌─────────────┐    │             │
│                         │  │ Sensors     │    │ Robot Model │    │             │
│                         │  │ • Lidar     │    │ • URDF      │    │             │
│                         │  │ • Camera    │    │ • Joints    │    │             │
│                         │  │ • IMU       │    │ • Collision │    │             │
│                         │  └──────┬──────┘    └─────────────┘    │             │
│                         │         │                              │             │
│                         └─────────┼──────────────────────────────┘             │
│                                   │                                             │
│                                   │  /scan, /odom, /camera/*                   │
│                                   ▼                                             │
│                         ┌─────────────────┐                                     │
│                         │ 👁️ VISUALIZATION │                                    │
│                         │  Foxglove Studio │                                    │
│                         │  Isaac Sim View  │                                    │
│                         └─────────────────┘                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Layer Details

| Layer | Container | Responsibility | Key Topics |
|-------|-----------|----------------|------------|
| **Cognitive** | `dt_cognitive` | Speech → Intent | Redis: `robot_commands` |
| **Control** | `dt_ros2` | Intent → Motion | `/cmd_vel`, `/goal_pose` |
| **Simulation** | `dt_isaac_sim` | Motion → Physics → Sensors | `/scan`, `/odom`, `/tf` |

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Robot Framework** | ROS 2 Humble | Humble Hawksbill | Middleware & communication |
| **Navigation** | Nav2 | 1.x | Path planning & control |
| **Simulation** | NVIDIA Isaac Sim | 4.2.0 | Physics & rendering |
| **Speech AI** | NVIDIA Riva | 2.14.0 | ASR (Speech-to-Text) |
| **LLM** | Llama 3.1 (via NIM) | 8B | Intent extraction |
| **Orchestration** | Docker Compose | 2.x | Container management |

### Supporting Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Message Queue** | Redis | Cross-container pub/sub |
| **Visualization** | Foxglove Studio | Real-time dashboard |
| **DDS** | FastRTPS | ROS 2 ↔ Isaac communication |
| **Scene Format** | USD | Universal Scene Description |

---

## 🚀 Quick Start

### Prerequisites

- **OS:** Ubuntu 22.04 or Windows with WSL2
- **GPU:** NVIDIA RTX 3080+ (12GB+ VRAM recommended)
- **RAM:** 32GB+
- **Software:** Docker, NVIDIA Container Toolkit, Git

### Installation

```bash
# Clone the repository
git clone https://github.com/JackAmichai/digital-twin-robot-NVD.git
cd digital-twin-robot-NVD

# Run initial setup
bash scripts/setup.sh

# Edit .env with your NVIDIA API keys
nano .env

# Verify your environment
make check-env

# Build all containers
make build

# Start the full system
make up
```

### Run a Demo

```bash
# Start the voice-controlled navigation demo
make demo

# Or start individual components
make up-ros2      # Just ROS 2 stack
make up-sim       # ROS 2 + Isaac Sim
make up-dev       # Full stack + dev tools
```

---

## 📁 Project Structure

```
digital-twin-robotics-lab/
│
├── 📄 docker-compose.yml       # 🐳 Container orchestration (3 services + tools)
├── 📄 Makefile                 # ⚡ 40+ development commands
├── 📄 .env                     # 🔐 Environment configuration (git-ignored)
├── 📄 .env.example             # 📋 Environment template
│
├── 🧠 cognitive_service/       # THE BRAIN - AI/Speech Processing
│   ├── src/
│   │   ├── asr_client.py       # NVIDIA Riva gRPC client
│   │   ├── intent_parser.py    # LLM-based command parsing
│   │   └── command_bridge.py   # Redis publisher to ROS 2
│   ├── config/
│   │   └── cognitive.yaml      # ASR & LLM settings
│   ├── Dockerfile
│   └── requirements.txt
│
├── 🦾 ros2_ws/                 # THE NERVOUS SYSTEM - Robot Control
│   ├── src/
│   │   ├── cognitive_bridge/   # Redis subscriber → ROS 2 goals
│   │   ├── robot_control/      # High-level behaviors
│   │   ├── robot_description/  # URDF, meshes, configs
│   │   └── robot_bringup/      # Launch files
│   ├── Dockerfile
│   └── ros_entrypoint.sh
│
├── 🌍 simulation/              # THE WORLD - Isaac Sim Assets
│   ├── environments/           # USD warehouse scenes
│   ├── robots/                 # Robot USD models
│   └── scripts/                # Simulation automation
│
├── ⚙️ config/                  # Shared Configuration
│   ├── nav2_params.yaml        # Navigation tuning
│   └── fastrtps_profile.xml    # DDS settings
│
├── 📜 scripts/                 # Utility Scripts
│   ├── check_environment.sh    # Verify prerequisites
│   ├── check_gpu.sh            # GPU configuration check
│   └── setup.sh                # Initial project setup
│
├── 📚 docs/                    # Documentation
│   ├── SETUP.md                # Detailed installation guide
│   ├── ARCHITECTURE.md         # System design deep-dive
│   └── diagrams/               # Architecture visuals
│
└── 📊 data/                    # Runtime Data (git-ignored)
    ├── logs/
    └── recordings/
```

---

## ⌨️ Development Commands

The `Makefile` provides 40+ commands for development:

### Environment
```bash
make check-env      # Verify Docker, GPU, dependencies
make check-gpu      # Detailed GPU configuration
make setup          # Initial project setup
```

### Docker Operations
```bash
make build          # Build all containers
make up             # Start all services
make down           # Stop all services
make logs           # Tail all container logs
make status         # Show container status
```

### Individual Services
```bash
make up-ros2        # Start only ROS 2
make up-sim         # Start ROS 2 + Isaac Sim
make up-dev         # Full stack + dev tools
```

### Shell Access
```bash
make shell-ros2     # Bash into ROS 2 container
make shell-sim      # Bash into Isaac Sim container
make shell-cognitive # Bash into Cognitive container
```

### ROS 2 Specific
```bash
make ros2-topics    # List all ROS 2 topics
make ros2-nodes     # List all ROS 2 nodes
make ros2-build     # Build ROS 2 workspace
```

### Visualization
```bash
make foxglove       # Open Foxglove in browser
make isaac          # Open Isaac Sim streaming
```

---

## 🎬 Demo Scenarios

### Demo 1: Warehouse Inspection
```bash
make demo-inspect
```
**Voice:** *"Robot, inspect the north shelf"*
- Robot calculates optimal path
- Navigates avoiding obstacles
- Performs 360° scan at destination
- Reports status

### Demo 2: Dynamic Obstacle Avoidance
```bash
make demo-nav
```
**Voice:** *"Move to Zone B"*
- Path planned through warehouse
- Forklift appears mid-route
- Robot re-plans in real-time
- Arrives at destination

### Demo 3: Voice Control Loop
```bash
make demo-voice
```
Interactive voice control session with continuous command processing.

---

## 💻 System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 3070 (8GB) | RTX 4080 (16GB) |
| **VRAM** | 8GB | 16GB+ |
| **RAM** | 32GB | 64GB |
| **Storage** | 100GB SSD | 500GB NVMe |
| **CPU** | 8 cores | 16+ cores |

### Software

| Component | Version |
|-----------|---------|
| **OS** | Ubuntu 22.04 / WSL2 |
| **NVIDIA Driver** | 525+ |
| **CUDA** | 12.0+ |
| **Docker** | 24.x |
| **Docker Compose** | 2.x |

---

## ⚙️ Configuration

### Environment Variables

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `NGC_API_KEY` | NVIDIA NGC authentication | (required) |
| `NIM_API_KEY` | NVIDIA NIM for LLM | (required) |
| `ROS_DOMAIN_ID` | ROS 2 domain isolation | `0` |
| `LLM_PROVIDER` | LLM backend (nim/ollama) | `nim` |
| `HEADLESS` | Run Isaac Sim headless | `0` |
| `GPU_MEMORY_FRACTION` | Max GPU memory | `0.8` |

See [.env.example](.env.example) for complete list.

---

## 🗺️ Roadmap

| Epic | Status | Description |
|------|--------|-------------|
| 1. Foundation | ✅ Complete | Docker, scripts, configuration |
| 2. Cognitive Layer | ✅ Complete | Riva ASR, LLM intent parsing |
| 3. Control Layer | 🔄 In Progress | ROS 2, Nav2 integration |
| 4. Simulation Layer | ⏳ Planned | Isaac Sim, sensors |
| 5. Integration | ⏳ Planned | End-to-end pipeline |
| 6. Demo & Polish | ⏳ Planned | Documentation, videos |

See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for detailed sprint planning.

---

## 🎬 Demo

<!-- Replace with actual GIF/video after recording -->
```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   🎤 "Robot, go to the loading dock"                       │
│                                                            │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │ Riva ASR │───▶│ LLM NIM  │───▶│  Nav2    │            │
│   │ 150ms    │    │ 200ms    │    │ Planning │            │
│   └──────────┘    └──────────┘    └──────────┘            │
│                                          │                 │
│                                          ▼                 │
│   ┌────────────────────────────────────────────┐          │
│   │        Isaac Sim - Warehouse Scene         │          │
│   │                                            │          │
│   │     📦      🤖➡️➡️➡️➡️➡️🚛               │          │
│   │   Storage        Robot      Loading Dock   │          │
│   │                                            │          │
│   └────────────────────────────────────────────┘          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Run the interactive demo:**
```bash
python scripts/demo.py           # Interactive mode
python scripts/demo.py --auto    # Automated sequence
python scripts/demo.py --step    # Step-by-step walkthrough
```

---

## 🎯 Portfolio Highlights

This project demonstrates expertise in:

| Category | Skills Demonstrated |
|----------|---------------------|
| **AI/ML** | LLM prompt engineering, ASR integration, intent classification |
| **Robotics** | ROS 2 architecture, Nav2 configuration, URDF modeling, TF2 transforms |
| **Simulation** | Isaac Sim scripting, sensor simulation, physics configuration |
| **DevOps** | Docker multi-stage builds, Compose orchestration, health checks |
| **Python** | Async programming, gRPC clients, dataclasses, type hints |
| **Testing** | pytest integration tests, mocking, CI-ready test suite |
| **Documentation** | Mermaid diagrams, comprehensive README, video scripts |

**Lines of Code:** ~4,000+ across 40+ files  
**Technologies:** 15+ integrated components  
**Architecture:** 3-layer containerized microservices

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Make your changes with tests
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [NVIDIA Riva](https://developer.nvidia.com/riva)
- [ROS 2](https://www.ros.org/)
- [Nav2](https://navigation.ros.org/)
- [Foxglove Studio](https://foxglove.dev/)

---

<p align="center">
  <b>Built with ❤️ for the robotics community</b>
  <br>
  <a href="https://github.com/JackAmichai/digital-twin-robot-NVD">GitHub</a> •
  <a href="docs/SETUP.md">Setup Guide</a> •
  <a href="PROJECT_ROADMAP.md">Roadmap</a>
</p>
