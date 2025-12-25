"""
Load Testing Suite for Digital Twin Robotics Lab.

Comprehensive load testing with Locust for all platform components:
- Voice Processing API
- Robot Control Services
- Cognitive Layer
- Fleet Management
- Simulation Interface

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000
    
Or headless:
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
        --headless -u 100 -r 10 --run-time 5m
"""

import json
import random
import time
from typing import Dict, List, Any

from locust import HttpUser, task, between, events, tag
from locust.runners import MasterRunner


# =============================================================================
# Voice Processing Load Tests
# =============================================================================

class VoiceProcessingUser(HttpUser):
    """
    Simulates voice processing API clients.
    
    Tests:
    - Audio transcription (ASR)
    - Text-to-speech synthesis
    - Wake word detection
    - Intent extraction
    """
    
    weight = 3  # Higher weight = more users
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session."""
        self.session_id = f"session_{random.randint(10000, 99999)}"
        self.languages = ["en-US", "es-ES", "de-DE", "ja-JP"]
        
    @task(10)
    @tag('asr', 'voice')
    def transcribe_audio(self):
        """Test speech-to-text transcription."""
        audio_data = {
            "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
            "format": "wav",
            "sample_rate": 16000,
            "language": random.choice(self.languages),
            "session_id": self.session_id
        }
        
        with self.client.post(
            "/api/v1/voice/transcribe",
            json=audio_data,
            name="/api/v1/voice/transcribe",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "transcript" in data:
                    response.success()
                else:
                    response.failure("Missing transcript in response")
            elif response.status_code == 503:
                response.failure("Service unavailable")
    
    @task(5)
    @tag('tts', 'voice')
    def synthesize_speech(self):
        """Test text-to-speech synthesis."""
        texts = [
            "Robot arm moving to position A",
            "Task completed successfully",
            "Warning: Obstacle detected ahead",
            "System initialized and ready",
            "Processing your request now"
        ]
        
        tts_request = {
            "text": random.choice(texts),
            "voice": "en-US-Neural2-F",
            "language": "en-US",
            "speaking_rate": 1.0,
            "pitch": 0.0
        }
        
        self.client.post(
            "/api/v1/voice/synthesize",
            json=tts_request,
            name="/api/v1/voice/synthesize"
        )
    
    @task(3)
    @tag('intent', 'voice')
    def extract_intent(self):
        """Test intent extraction from text."""
        commands = [
            "Pick up the red box",
            "Move to station B",
            "Stop all robots",
            "What is the current status?",
            "Navigate to the charging station"
        ]
        
        intent_request = {
            "text": random.choice(commands),
            "context": {
                "robot_id": f"robot_{random.randint(1, 5)}",
                "location": "warehouse_floor"
            }
        }
        
        self.client.post(
            "/api/v1/cognitive/intent",
            json=intent_request,
            name="/api/v1/cognitive/intent"
        )


# =============================================================================
# Robot Control Load Tests
# =============================================================================

class RobotControlUser(HttpUser):
    """
    Simulates robot control clients.
    
    Tests:
    - Motion commands
    - Status polling
    - Navigation goals
    - Emergency stops
    """
    
    weight = 4
    wait_time = between(0.5, 2)
    
    def on_start(self):
        """Initialize robot session."""
        self.robot_id = f"robot_{random.randint(1, 10)}"
        
    @task(15)
    @tag('status', 'robot')
    def get_robot_status(self):
        """Poll robot status (high frequency)."""
        self.client.get(
            f"/api/v1/robots/{self.robot_id}/status",
            name="/api/v1/robots/[id]/status"
        )
    
    @task(8)
    @tag('telemetry', 'robot')
    def get_robot_telemetry(self):
        """Get detailed robot telemetry."""
        self.client.get(
            f"/api/v1/robots/{self.robot_id}/telemetry",
            name="/api/v1/robots/[id]/telemetry"
        )
    
    @task(5)
    @tag('motion', 'robot')
    def send_motion_command(self):
        """Send motion command to robot."""
        motion_command = {
            "type": random.choice(["move", "rotate", "gripper"]),
            "parameters": {
                "x": random.uniform(-5.0, 5.0),
                "y": random.uniform(-5.0, 5.0),
                "z": random.uniform(0.0, 2.0),
                "velocity": random.uniform(0.1, 1.0)
            }
        }
        
        self.client.post(
            f"/api/v1/robots/{self.robot_id}/motion",
            json=motion_command,
            name="/api/v1/robots/[id]/motion"
        )
    
    @task(3)
    @tag('navigation', 'robot')
    def send_navigation_goal(self):
        """Send navigation goal."""
        nav_goal = {
            "goal": {
                "x": random.uniform(-10.0, 10.0),
                "y": random.uniform(-10.0, 10.0),
                "theta": random.uniform(-3.14, 3.14)
            },
            "frame_id": "map",
            "planner": "NavfnROS"
        }
        
        self.client.post(
            f"/api/v1/robots/{self.robot_id}/navigate",
            json=nav_goal,
            name="/api/v1/robots/[id]/navigate"
        )
    
    @task(1)
    @tag('emergency', 'robot')
    def emergency_stop(self):
        """Test emergency stop (low frequency)."""
        self.client.post(
            f"/api/v1/robots/{self.robot_id}/emergency-stop",
            name="/api/v1/robots/[id]/emergency-stop"
        )


# =============================================================================
# Fleet Management Load Tests
# =============================================================================

class FleetManagementUser(HttpUser):
    """
    Simulates fleet management operations.
    """
    
    weight = 2
    wait_time = between(2, 5)
    
    @task(10)
    @tag('fleet', 'status')
    def get_fleet_status(self):
        """Get overall fleet status."""
        self.client.get(
            "/api/v1/fleet/status",
            name="/api/v1/fleet/status"
        )
    
    @task(8)
    @tag('fleet', 'robots')
    def list_robots(self):
        """List all robots in fleet."""
        params = {
            "status": random.choice(["all", "active", "idle", "charging"]),
            "limit": 50
        }
        
        self.client.get(
            "/api/v1/fleet/robots",
            params=params,
            name="/api/v1/fleet/robots"
        )
    
    @task(5)
    @tag('fleet', 'tasks')
    def create_task(self):
        """Create a new fleet task."""
        task_types = ["pick", "place", "transport", "inspect", "charge"]
        
        task_data = {
            "type": random.choice(task_types),
            "priority": random.randint(1, 10),
            "source": {
                "station_id": f"station_{random.randint(1, 20)}",
                "position": {"x": random.uniform(0, 50), "y": random.uniform(0, 50)}
            },
            "destination": {
                "station_id": f"station_{random.randint(1, 20)}",
                "position": {"x": random.uniform(0, 50), "y": random.uniform(0, 50)}
            }
        }
        
        self.client.post(
            "/api/v1/fleet/tasks",
            json=task_data,
            name="/api/v1/fleet/tasks"
        )
    
    @task(6)
    @tag('fleet', 'tasks')
    def get_task_status(self):
        """Query task status."""
        task_id = f"task_{random.randint(1, 1000)}"
        
        self.client.get(
            f"/api/v1/fleet/tasks/{task_id}",
            name="/api/v1/fleet/tasks/[id]"
        )


# =============================================================================
# Simulation Interface Load Tests
# =============================================================================

class SimulationUser(HttpUser):
    """
    Simulates digital twin synchronization clients.
    """
    
    weight = 2
    wait_time = between(0.1, 0.5)  # High frequency for real-time sync
    
    def on_start(self):
        """Initialize simulation session."""
        self.sim_session_id = f"sim_{random.randint(1000, 9999)}"
    
    @task(20)
    @tag('simulation', 'sync')
    def sync_robot_state(self):
        """Synchronize robot state with digital twin."""
        robot_id = f"robot_{random.randint(1, 10)}"
        
        state_data = {
            "robot_id": robot_id,
            "timestamp": time.time(),
            "pose": {
                "position": {"x": random.uniform(-10, 10), "y": random.uniform(-10, 10), "z": 0},
                "orientation": {"x": 0, "y": 0, "z": random.uniform(-1, 1), "w": random.uniform(-1, 1)}
            },
            "joint_states": {
                f"joint_{i}": random.uniform(-3.14, 3.14) for i in range(6)
            }
        }
        
        self.client.post(
            "/api/v1/simulation/sync",
            json=state_data,
            name="/api/v1/simulation/sync"
        )
    
    @task(10)
    @tag('simulation', 'scene')
    def get_scene_state(self):
        """Get current scene state."""
        self.client.get(
            "/api/v1/simulation/scene",
            name="/api/v1/simulation/scene"
        )


# =============================================================================
# Cognitive Layer Load Tests
# =============================================================================

class CognitiveUser(HttpUser):
    """
    Simulates cognitive processing clients.
    """
    
    weight = 2
    wait_time = between(1, 4)
    
    def on_start(self):
        """Initialize conversation session."""
        self.conversation_id = f"conv_{random.randint(10000, 99999)}"
        self.turn_count = 0
    
    @task(8)
    @tag('cognitive', 'conversation')
    def send_message(self):
        """Send a message in conversation."""
        messages = [
            "Show me the status of all robots",
            "Move robot 1 to the charging station",
            "What tasks are pending?",
            "Stop the robot near the conveyor",
            "Schedule maintenance for robot 3"
        ]
        
        self.turn_count += 1
        
        message_data = {
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_count,
            "message": random.choice(messages)
        }
        
        self.client.post(
            "/api/v1/cognitive/message",
            json=message_data,
            name="/api/v1/cognitive/message"
        )
    
    @task(4)
    @tag('cognitive', 'scene')
    def query_scene(self):
        """Query scene with natural language."""
        queries = [
            "find all boxes near the robot",
            "where is the red container?",
            "how many items are on table 1?"
        ]
        
        query_data = {
            "query": random.choice(queries),
            "scene_id": "main_scene"
        }
        
        self.client.post(
            "/api/v1/cognitive/scene-query",
            json=query_data,
            name="/api/v1/cognitive/scene-query"
        )


# =============================================================================
# Health Check User
# =============================================================================

class HealthCheckUser(HttpUser):
    """Lightweight health check user for monitoring."""
    
    weight = 1
    wait_time = between(5, 10)
    
    @task
    @tag('health')
    def health_check(self):
        """Check system health endpoints."""
        endpoints = [
            "/health",
            "/api/v1/health/voice",
            "/api/v1/health/robot-control",
            "/api/v1/health/fleet",
            "/api/v1/health/simulation"
        ]
        
        for endpoint in endpoints:
            self.client.get(endpoint, name=endpoint)


# =============================================================================
# Event Hooks
# =============================================================================

@events.test_start.add_listener  
def on_test_start(environment, **kwargs):
    """Log test start."""
    print("=" * 60)
    print("Digital Twin Robotics Lab - Load Test Started")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion."""
    print("=" * 60)
    print("Digital Twin Robotics Lab - Load Test Completed")
    print("=" * 60)
