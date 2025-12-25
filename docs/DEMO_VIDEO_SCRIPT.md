# Demo Video Recording Guide

## 🎬 Video Script: Digital Twin Robotics Lab

**Target Duration:** 3-5 minutes  
**Format:** Screen recording with voiceover  
**Resolution:** 1920x1080 (1080p)

---

## Pre-Recording Setup

### 1. Terminal Setup
```bash
# Clean terminal with dark theme
# Font: JetBrains Mono or similar, 14pt
# Set terminal to 120x40 characters

# Start services
docker compose up -d

# Wait for all services to be healthy
python scripts/health_monitor.py --once
```

### 2. Foxglove Setup
- Open Foxglove Studio
- Load `config/foxglove_layout.json`
- Connect to `ws://localhost:8765`
- Arrange windows: Terminal (left 40%), Foxglove (right 60%)

### 3. Isaac Sim Setup
- Launch Isaac Sim with warehouse scene
- Position camera for good overview
- Enable ROS 2 bridge

---

## Video Sections

### Section 1: Introduction (30 seconds)

**Visual:** Project title card, then terminal with banner

**Script:**
> "Hi, I'm [Name], and this is the Digital Twin Robotics Lab - 
> a professional-grade voice-controlled robotics simulation platform.
> 
> This project combines NVIDIA's AI stack with ROS 2 and Isaac Sim
> to create a complete closed-loop system where you can control
> a simulated robot using natural voice commands."

**Actions:**
```bash
python scripts/demo.py --overview
```

---

### Section 2: Architecture Overview (45 seconds)

**Visual:** Architecture diagram in terminal or slide

**Script:**
> "Let me walk you through the architecture.
> 
> Voice commands flow through five stages:
> 1. NVIDIA Riva captures and transcribes speech in real-time
> 2. The LLM - Llama 3.1 running on NVIDIA NIM - extracts the intent
> 3. Redis bridges the cognitive layer to the control layer
> 4. ROS 2 with Nav2 plans and executes the navigation
> 5. Isaac Sim provides physics-accurate simulation with sensor feedback
> 
> Everything runs in Docker containers, orchestrated with Docker Compose."

**Actions:**
- Show docker-compose.yml briefly
- Show architecture in demo.py output

---

### Section 3: Live Demo - Voice Commands (90 seconds)

**Visual:** Split screen - Terminal + Foxglove 3D view

**Script:**
> "Now let's see it in action. I'll send some voice commands
> and watch the robot respond in real-time."

**Actions:**
```bash
python scripts/demo.py
```

**Commands to demonstrate:**
1. `Robot, go to the loading dock`
   > "Watch as the command flows through each stage...
   > The robot plans a path and begins navigation."

2. `Navigate to storage area`
   > "The LLM correctly interprets 'storage area' as the storage zone."

3. `Stop`
   > "Emergency stop - the robot immediately halts."

4. `What's your status?`
   > "Status queries return the current position and state."

5. `Go to charging station`
   > "And finally, returning to the charging station."

---

### Section 4: Code Highlights (60 seconds)

**Visual:** VS Code with code files

**Script:**
> "Let me highlight some key implementation details."

**Files to show:**
1. `cognitive_service/src/intent_parser.py` (lines 20-50)
   > "The intent parser uses a carefully crafted prompt to teach
   > the LLM our warehouse zone mappings."

2. `ros2_ws/src/cognitive_bridge/bridge_node.py` (lines 40-70)
   > "The ROS 2 bridge subscribes to Redis and converts intents
   > to navigation goals that Nav2 can execute."

3. `simulation/scripts/launch_sim.py` (lines 60-90)
   > "Isaac Sim integration enables the ROS 2 bridge for
   > bidirectional communication between simulation and ROS."

---

### Section 5: Testing & Health Monitoring (30 seconds)

**Visual:** Terminal with test output

**Script:**
> "The project includes comprehensive testing and monitoring."

**Actions:**
```bash
# Run tests
pytest tests/test_integration.py -v --tb=short

# Show health dashboard
python scripts/health_monitor.py --once
```

---

### Section 6: Conclusion (30 seconds)

**Visual:** GitHub repo page or terminal

**Script:**
> "This project demonstrates:
> - Real-time speech processing with NVIDIA Riva
> - LLM-powered natural language understanding
> - Production-ready ROS 2 navigation
> - GPU-accelerated physics simulation
> - Clean, modular, containerized architecture
> 
> Check out the GitHub repo for full documentation and setup instructions.
> Thanks for watching!"

**Actions:**
- Show README.md
- Show GitHub stars/forks (if any)

---

## Post-Production Checklist

- [ ] Add intro/outro music (royalty-free)
- [ ] Add lower-third text overlays for key points
- [ ] Add zoom effects on code sections
- [ ] Sync audio with terminal typing
- [ ] Add chapter markers for YouTube
- [ ] Create thumbnail with robot + voice waves
- [ ] Write video description with links

---

## Thumbnail Ideas

1. Robot silhouette with sound waves and code
2. Split image: microphone → robot
3. NVIDIA + ROS 2 + Isaac Sim logos with "Voice Control"

---

## YouTube Description Template

```
🤖 Digital Twin Robotics Lab - Voice-Controlled Robot Simulation

Control a simulated robot using natural voice commands! This project 
integrates NVIDIA Riva (ASR), Llama 3.1 (NIM), ROS 2 Nav2, and 
NVIDIA Isaac Sim into a complete closed-loop system.

🔗 GitHub: https://github.com/JackAmichai/digital-twin-robot-NVD

⏱️ Timestamps:
0:00 - Introduction
0:30 - Architecture Overview
1:15 - Live Demo
2:45 - Code Walkthrough
3:45 - Testing & Monitoring
4:15 - Conclusion

🛠️ Technologies:
• NVIDIA Riva - Speech Recognition
• NVIDIA NIM - Llama 3.1 8B Instruct
• ROS 2 Humble - Robot Middleware
• Nav2 - Autonomous Navigation
• NVIDIA Isaac Sim 4.2 - Physics Simulation
• Docker Compose - Container Orchestration
• Redis - Message Broker
• Foxglove Studio - Visualization

#robotics #nvidia #ros2 #simulation #voicecontrol #ai #isaacsim
```
