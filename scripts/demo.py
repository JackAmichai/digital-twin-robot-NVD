#!/usr/bin/env python3
"""
Interactive Demo - Digital Twin Robotics Lab

This script provides an interactive demonstration of the complete
voice-controlled robotics simulation platform.

Features:
- Simulated voice commands (no microphone needed)
- Real-time visualization of the pipeline
- Step-by-step walkthrough mode
- Automated demo sequence

Usage:
    python scripts/demo.py              # Interactive mode
    python scripts/demo.py --auto       # Automated demo
    python scripts/demo.py --step       # Step-by-step walkthrough
"""

import asyncio
import json
import time
import sys
from datetime import datetime
from typing import Optional
import argparse

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_banner():
    """Print demo banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     🤖  DIGITAL TWIN ROBOTICS LAB  🤖                                ║
║                                                                      ║
║     Voice-Controlled Autonomous Navigation Demo                      ║
║                                                                      ║
║     NVIDIA Isaac Sim + ROS 2 + Riva ASR + NIM LLM                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.YELLOW}{'═' * 60}{Colors.ENDC}")
    print(f"{Colors.YELLOW}{Colors.BOLD}  {title}{Colors.ENDC}")
    print(f"{Colors.YELLOW}{'═' * 60}{Colors.ENDC}\n")


def print_step(step_num: int, description: str):
    """Print step indicator."""
    print(f"{Colors.GREEN}[Step {step_num}]{Colors.ENDC} {description}")


def print_data(label: str, data: str, indent: int = 2):
    """Print data with label."""
    spaces = " " * indent
    print(f"{spaces}{Colors.CYAN}{label}:{Colors.ENDC} {data}")


def print_json(data: dict, indent: int = 4):
    """Pretty print JSON data."""
    formatted = json.dumps(data, indent=2)
    for line in formatted.split('\n'):
        print(f"{' ' * indent}{Colors.DIM}{line}{Colors.ENDC}")


def simulate_typing(text: str, delay: float = 0.03):
    """Simulate typing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


class DemoSimulator:
    """Simulates the complete pipeline for demonstration."""
    
    # Zone coordinates (matching intent_parser.py)
    ZONES = {
        "loading_dock": {"x": 5.0, "y": 2.0, "theta": 0.0},
        "storage": {"x": -5.0, "y": 2.0, "theta": 3.14},
        "assembly": {"x": 0.0, "y": 5.0, "theta": 1.57},
        "charging": {"x": 0.0, "y": -5.0, "theta": -1.57},
        "inspection": {"x": 3.0, "y": 0.0, "theta": 0.0},
    }
    
    # Demo voice commands
    DEMO_COMMANDS = [
        "Robot, go to the loading dock",
        "Navigate to storage area",
        "Move to assembly station",
        "Stop",
        "What's your status?",
        "Go to charging station",
    ]
    
    def __init__(self):
        self.current_position = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.command_history = []
        self.is_moving = False
    
    def simulate_asr(self, command: str) -> dict:
        """Simulate ASR processing."""
        return {
            "transcript": command,
            "confidence": 0.95,
            "is_final": True,
            "latency_ms": 150
        }
    
    def simulate_intent_parsing(self, transcript: str) -> dict:
        """Simulate LLM intent parsing."""
        transcript_lower = transcript.lower()
        
        # Parse action
        if "stop" in transcript_lower:
            return {
                "action": "stop",
                "target": None,
                "confidence": 0.98,
                "reasoning": "Detected stop keyword"
            }
        elif "status" in transcript_lower or "where" in transcript_lower:
            return {
                "action": "status",
                "target": None,
                "confidence": 0.92,
                "reasoning": "Detected status query"
            }
        
        # Parse zone
        for zone in self.ZONES.keys():
            zone_readable = zone.replace("_", " ")
            if zone_readable in transcript_lower or zone in transcript_lower:
                return {
                    "action": "move_to_zone",
                    "target": zone,
                    "confidence": 0.94,
                    "reasoning": f"Detected navigation to {zone_readable}"
                }
        
        return {
            "action": "unknown",
            "target": None,
            "confidence": 0.3,
            "reasoning": "Could not parse intent"
        }
    
    def simulate_navigation(self, zone: str) -> dict:
        """Simulate navigation goal."""
        if zone not in self.ZONES:
            return {"success": False, "error": "Unknown zone"}
        
        target = self.ZONES[zone]
        
        # Calculate "distance"
        dx = target["x"] - self.current_position["x"]
        dy = target["y"] - self.current_position["y"]
        distance = (dx**2 + dy**2) ** 0.5
        
        # Simulate navigation time
        nav_time = distance / 0.5  # 0.5 m/s speed
        
        # Update position
        self.current_position = target.copy()
        
        return {
            "success": True,
            "target_zone": zone,
            "target_pose": target,
            "distance": round(distance, 2),
            "estimated_time": round(nav_time, 1),
            "path_planned": True
        }
    
    def get_status(self) -> dict:
        """Get robot status."""
        return {
            "position": self.current_position,
            "battery": 85,
            "is_moving": self.is_moving,
            "last_command": self.command_history[-1] if self.command_history else None
        }


async def run_single_command(sim: DemoSimulator, command: str, step_mode: bool = False):
    """Run a single command through the pipeline."""
    print(f"\n{Colors.BOLD}🎤 Voice Command:{Colors.ENDC}")
    simulate_typing(f'   "{command}"', delay=0.02)
    
    if step_mode:
        input(f"{Colors.DIM}   Press Enter to continue...{Colors.ENDC}")
    else:
        await asyncio.sleep(0.5)
    
    # Step 1: ASR
    print_step(1, "Speech Recognition (NVIDIA Riva)")
    asr_result = sim.simulate_asr(command)
    print_data("Transcript", f'"{asr_result["transcript"]}"')
    print_data("Confidence", f'{asr_result["confidence"]:.0%}')
    print_data("Latency", f'{asr_result["latency_ms"]}ms')
    
    if step_mode:
        input(f"{Colors.DIM}   Press Enter to continue...{Colors.ENDC}")
    else:
        await asyncio.sleep(0.3)
    
    # Step 2: Intent Parsing
    print_step(2, "Intent Parsing (NVIDIA NIM - Llama 3.1)")
    intent = sim.simulate_intent_parsing(asr_result["transcript"])
    print_data("Action", intent["action"])
    print_data("Target", str(intent["target"]))
    print_data("Confidence", f'{intent["confidence"]:.0%}')
    print_data("Reasoning", intent["reasoning"])
    
    if step_mode:
        input(f"{Colors.DIM}   Press Enter to continue...{Colors.ENDC}")
    else:
        await asyncio.sleep(0.3)
    
    # Step 3: Command Bridge
    print_step(3, "Command Bridge (Redis Pub/Sub)")
    robot_intent = {
        "action": intent["action"],
        "target": intent["target"],
        "parameters": sim.ZONES.get(intent["target"], {}),
        "confidence": intent["confidence"],
        "timestamp": datetime.now().isoformat()
    }
    print(f"   {Colors.DIM}Publishing to 'robot:commands':{Colors.ENDC}")
    print_json(robot_intent, indent=4)
    
    sim.command_history.append(robot_intent)
    
    if step_mode:
        input(f"{Colors.DIM}   Press Enter to continue...{Colors.ENDC}")
    else:
        await asyncio.sleep(0.3)
    
    # Step 4: ROS 2 Processing
    print_step(4, "ROS 2 Navigation (Nav2 Stack)")
    
    if intent["action"] == "move_to_zone" and intent["target"]:
        nav_result = sim.simulate_navigation(intent["target"])
        if nav_result["success"]:
            print_data("Target Zone", nav_result["target_zone"])
            print_data("Target Pose", f'x={nav_result["target_pose"]["x"]}, y={nav_result["target_pose"]["y"]}')
            print_data("Distance", f'{nav_result["distance"]}m')
            print_data("Est. Time", f'{nav_result["estimated_time"]}s')
            print(f"   {Colors.GREEN}✓ Navigation goal sent to Nav2{Colors.ENDC}")
        else:
            print(f"   {Colors.RED}✗ Navigation failed: {nav_result['error']}{Colors.ENDC}")
    
    elif intent["action"] == "stop":
        print(f"   {Colors.YELLOW}⚠ STOP command issued{Colors.ENDC}")
        print_data("Action", "Canceling current navigation goal")
        sim.is_moving = False
    
    elif intent["action"] == "status":
        status = sim.get_status()
        print_data("Position", f'x={status["position"]["x"]}, y={status["position"]["y"]}')
        print_data("Battery", f'{status["battery"]}%')
        print_data("Moving", str(status["is_moving"]))
    
    else:
        print(f"   {Colors.RED}✗ Unknown command - no action taken{Colors.ENDC}")
    
    if step_mode:
        input(f"{Colors.DIM}   Press Enter to continue...{Colors.ENDC}")
    else:
        await asyncio.sleep(0.5)
    
    # Step 5: Isaac Sim (visualization note)
    print_step(5, "Isaac Sim Visualization")
    print(f"   {Colors.DIM}[In full demo: Robot animates to target position]{Colors.ENDC}")
    print(f"   {Colors.DIM}[Lidar, camera, and odometry published to ROS 2]{Colors.ENDC}")


async def interactive_mode(sim: DemoSimulator):
    """Run interactive demo mode."""
    print_section("Interactive Mode")
    print("Enter voice commands to control the robot.")
    print("Type 'quit' to exit, 'help' for available commands.\n")
    
    available_zones = ", ".join(sim.ZONES.keys())
    
    while True:
        try:
            command = input(f"{Colors.CYAN}🎤 Enter command: {Colors.ENDC}")
            
            if command.lower() == 'quit':
                print("\nExiting demo...")
                break
            elif command.lower() == 'help':
                print(f"\n{Colors.YELLOW}Available commands:{Colors.ENDC}")
                print("  • 'go to [zone]' - Navigate to a zone")
                print("  • 'stop' - Stop the robot")
                print("  • 'status' - Get robot status")
                print(f"\n{Colors.YELLOW}Available zones:{Colors.ENDC}")
                print(f"  {available_zones}")
                print()
                continue
            elif not command.strip():
                continue
            
            await run_single_command(sim, command, step_mode=False)
            
        except KeyboardInterrupt:
            print("\n\nExiting demo...")
            break


async def automated_demo(sim: DemoSimulator):
    """Run automated demo sequence."""
    print_section("Automated Demo Sequence")
    print("Running through predefined commands...\n")
    
    for i, command in enumerate(sim.DEMO_COMMANDS, 1):
        print(f"\n{Colors.BOLD}{'─' * 60}{Colors.ENDC}")
        print(f"{Colors.BOLD}Demo Command {i}/{len(sim.DEMO_COMMANDS)}{Colors.ENDC}")
        await run_single_command(sim, command, step_mode=False)
        await asyncio.sleep(1)
    
    print_section("Demo Complete!")
    print("The robot successfully processed all voice commands.")
    print("\nFinal robot position:")
    status = sim.get_status()
    print_data("X", f'{status["position"]["x"]}m')
    print_data("Y", f'{status["position"]["y"]}m')
    print_data("Commands processed", str(len(sim.command_history)))


async def step_by_step_demo(sim: DemoSimulator):
    """Run step-by-step walkthrough."""
    print_section("Step-by-Step Walkthrough")
    print("This mode pauses after each step for explanation.")
    print("Press Enter to proceed through each stage.\n")
    
    input(f"{Colors.DIM}Press Enter to start...{Colors.ENDC}")
    
    # Just run first 3 commands in step mode
    for command in sim.DEMO_COMMANDS[:3]:
        print(f"\n{Colors.BOLD}{'─' * 60}{Colors.ENDC}")
        await run_single_command(sim, command, step_mode=True)
    
    print_section("Walkthrough Complete!")


def print_system_overview():
    """Print system architecture overview."""
    print_section("System Architecture")
    
    overview = f"""
{Colors.BOLD}Pipeline Flow:{Colors.ENDC}

  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  {Colors.CYAN}Microphone{Colors.ENDC}  │───▶│  {Colors.CYAN}Riva ASR{Colors.ENDC}   │───▶│  {Colors.CYAN}NIM LLM{Colors.ENDC}    │
  │  (Audio)    │    │  (Speech→   │    │  (Intent    │
  │             │    │   Text)     │    │   Parser)   │
  └─────────────┘    └─────────────┘    └──────┬──────┘
                                               │
                                               ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  {Colors.GREEN}Isaac Sim{Colors.ENDC}   │◀───│  {Colors.GREEN}Nav2 Stack{Colors.ENDC}  │◀───│  {Colors.GREEN}Redis{Colors.ENDC}       │
  │  (3D Sim)   │    │  (Path      │    │  (Message   │
  │             │    │   Planning) │    │   Bridge)   │
  └─────────────┘    └─────────────┘    └─────────────┘

{Colors.BOLD}Components:{Colors.ENDC}
  • {Colors.CYAN}NVIDIA Riva{Colors.ENDC}     - Real-time speech-to-text
  • {Colors.CYAN}NVIDIA NIM{Colors.ENDC}      - Llama 3.1 8B for intent extraction
  • {Colors.GREEN}Redis{Colors.ENDC}           - Cross-container pub/sub messaging
  • {Colors.GREEN}ROS 2 Humble{Colors.ENDC}    - Robot middleware & Nav2 navigation
  • {Colors.GREEN}Isaac Sim 4.2{Colors.ENDC}   - GPU-accelerated physics simulation
  • {Colors.YELLOW}Foxglove{Colors.ENDC}        - Real-time visualization dashboard
"""
    print(overview)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Digital Twin Robotics Lab Demo")
    parser.add_argument("--auto", action="store_true", help="Run automated demo")
    parser.add_argument("--step", action="store_true", help="Step-by-step walkthrough")
    parser.add_argument("--overview", action="store_true", help="Show system overview")
    args = parser.parse_args()
    
    print_banner()
    
    if args.overview:
        print_system_overview()
        return
    
    sim = DemoSimulator()
    
    if args.auto:
        await automated_demo(sim)
    elif args.step:
        await step_by_step_demo(sim)
    else:
        print_system_overview()
        await interactive_mode(sim)


if __name__ == "__main__":
    asyncio.run(main())
