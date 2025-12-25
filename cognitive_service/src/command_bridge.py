"""
Command Bridge
Sends parsed intents to ROS 2 via Redis pub/sub
"""

import json
import os
import redis
from typing import Optional
from intent_parser import RobotIntent, IntentParser
from asr_client import get_asr_client


class CommandBridge:
    """Bridge between cognitive service and ROS 2."""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.from_url(self.redis_url)
        self.channel = "robot_commands"
        self.parser = IntentParser(
            provider=os.getenv("LLM_PROVIDER", "nim")
        )
    
    def publish_intent(self, intent: RobotIntent):
        """Publish intent to Redis for ROS 2 bridge."""
        message = intent.to_json()
        self.redis.publish(self.channel, message)
        print(f"Published: {intent.action.value} → {intent.target}")
    
    def process_transcript(self, text: str, is_final: bool):
        """Process ASR transcript and publish if final."""
        if not is_final:
            return
        
        print(f"Processing: {text}")
        intent = self.parser.parse(text)
        
        if intent.confidence > 0.5:
            self.publish_intent(intent)
        else:
            print(f"Low confidence ({intent.confidence}), ignoring")
    
    def run_voice_control(self):
        """Run voice control loop."""
        asr = get_asr_client(
            server=os.getenv("RIVA_SERVER", "localhost:50051")
        )
        
        print("🎤 Voice control active. Speak commands...")
        try:
            asr.transcribe_stream(self.process_transcript)
        except KeyboardInterrupt:
            asr.stop()
            print("Voice control stopped")
    
    def send_command(self, text: str):
        """Send a text command directly."""
        intent = self.parser.parse(text)
        self.publish_intent(intent)


if __name__ == "__main__":
    import sys
    
    bridge = CommandBridge()
    
    if len(sys.argv) > 1:
        # Direct command mode
        command = " ".join(sys.argv[1:])
        bridge.send_command(command)
    else:
        # Voice control mode
        bridge.run_voice_control()
