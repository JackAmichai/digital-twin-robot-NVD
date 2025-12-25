#!/usr/bin/env python3
"""
Wake Word Detection with NVIDIA Riva

This module implements always-on wake word detection using NVIDIA Riva's
keyword spotting capability. The system listens continuously for trigger
phrases like "Hey Robot" before activating full speech recognition.

Architecture:
    ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
    │  Microphone  │────▶│  Wake Word      │────▶│  Full ASR        │
    │  (always on) │     │  Detection      │     │  (on trigger)    │
    └──────────────┘     └─────────────────┘     └──────────────────┘
                              │                         │
                              │ "Hey Robot" detected    │ Command text
                              ▼                         ▼
                         ┌──────────────────────────────────┐
                         │      Intent Parser / LLM        │
                         └──────────────────────────────────┘

Features:
- Low-power continuous listening for wake words
- Multiple wake phrase support ("Hey Robot", "OK Robot", "Robot")
- Configurable detection sensitivity
- Automatic timeout after activation
- Audio feedback on wake word detection

Example Usage:
    detector = WakeWordDetector()
    detector.start_listening()  # Blocks until wake word detected
    
    # Or with callback:
    detector.on_wake_word(lambda: print("Wake word detected!"))
    detector.start_async()
"""

import grpc
import numpy as np
import pyaudio
import threading
import time
import wave
import os
from typing import Callable, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

# Riva client imports
try:
    import riva.client
    from riva.client import ASRService
    import riva.client.audio_io
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False
    print("WARNING: Riva client not installed. Install with: pip install nvidia-riva-client")


class DetectorState(Enum):
    """Wake word detector states"""
    IDLE = "idle"
    LISTENING = "listening"          # Listening for wake word
    ACTIVATED = "activated"          # Wake word detected, ASR active
    PROCESSING = "processing"        # Processing command
    COOLDOWN = "cooldown"            # Brief pause after command


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection"""
    
    # Wake phrases to listen for
    wake_phrases: List[str] = field(default_factory=lambda: [
        "hey robot",
        "ok robot", 
        "hello robot",
        "robot"
    ])
    
    # Riva server configuration
    riva_server: str = "localhost:50051"
    
    # Audio configuration
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1600  # 100ms at 16kHz
    
    # Detection sensitivity (0.0 - 1.0)
    # Lower = more sensitive but more false positives
    sensitivity: float = 0.5
    
    # Timeout after wake word (seconds)
    # How long to listen for a command after wake word
    activation_timeout: float = 10.0
    
    # Cooldown between detections (seconds)
    cooldown_period: float = 1.0
    
    # Play audio feedback on wake word detection
    audio_feedback: bool = True
    feedback_sound: str = "assets/wake_sound.wav"
    
    # Language code for ASR
    language_code: str = "en-US"


class WakeWordDetector:
    """
    Continuous wake word detection using NVIDIA Riva
    
    This class provides always-on listening for wake words with low
    CPU/GPU usage, then activates full ASR when triggered.
    
    Example:
        config = WakeWordConfig(
            wake_phrases=["hey robot", "ok robot"],
            sensitivity=0.6
        )
        
        detector = WakeWordDetector(config)
        
        # Register callback
        def on_command(text: str):
            print(f"Command received: {text}")
        
        detector.on_command_received = on_command
        detector.start()
    """
    
    def __init__(self, config: Optional[WakeWordConfig] = None):
        self.config = config or WakeWordConfig()
        self.state = DetectorState.IDLE
        self.logger = logging.getLogger("WakeWordDetector")
        
        # Callbacks
        self.on_wake_word_detected: Optional[Callable[[], None]] = None
        self.on_command_received: Optional[Callable[[str], None]] = None
        self.on_state_change: Optional[Callable[[DetectorState], None]] = None
        
        # Threading
        self._stop_event = threading.Event()
        self._listen_thread: Optional[threading.Thread] = None
        
        # Audio
        self._audio = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        
        # Riva client
        self._riva_auth = None
        self._asr_service = None
        
        if RIVA_AVAILABLE:
            self._init_riva()
    
    def _init_riva(self):
        """Initialize Riva ASR client"""
        try:
            self._riva_auth = riva.client.Auth(
                uri=self.config.riva_server,
                use_ssl=False
            )
            self._asr_service = ASRService(self._riva_auth)
            self.logger.info(f"Connected to Riva server: {self.config.riva_server}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Riva: {e}")
            self._asr_service = None
    
    def _set_state(self, new_state: DetectorState):
        """Update state and notify callback"""
        old_state = self.state
        self.state = new_state
        
        if old_state != new_state:
            self.logger.info(f"State: {old_state.value} → {new_state.value}")
            if self.on_state_change:
                self.on_state_change(new_state)
    
    def start(self, blocking: bool = True):
        """
        Start wake word detection
        
        Args:
            blocking: If True, blocks until stop() is called
                     If False, runs in background thread
        """
        if self.state != DetectorState.IDLE:
            self.logger.warning("Detector already running")
            return
        
        self._stop_event.clear()
        
        if blocking:
            self._listen_loop()
        else:
            self._listen_thread = threading.Thread(
                target=self._listen_loop,
                daemon=True
            )
            self._listen_thread.start()
    
    def stop(self):
        """Stop wake word detection"""
        self._stop_event.set()
        
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
            self._listen_thread = None
        
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        self._set_state(DetectorState.IDLE)
        self.logger.info("Wake word detection stopped")
    
    def _listen_loop(self):
        """Main listening loop"""
        self.logger.info("Starting wake word detection...")
        self._set_state(DetectorState.LISTENING)
        
        # Open audio stream
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size
        )
        
        audio_buffer = []
        buffer_duration = 2.0  # seconds of audio to analyze
        buffer_frames = int(buffer_duration * self.config.sample_rate / self.config.chunk_size)
        
        while not self._stop_event.is_set():
            try:
                # Read audio chunk
                audio_data = self._stream.read(
                    self.config.chunk_size,
                    exception_on_overflow=False
                )
                
                if self.state == DetectorState.LISTENING:
                    # Add to buffer
                    audio_buffer.append(audio_data)
                    if len(audio_buffer) > buffer_frames:
                        audio_buffer.pop(0)
                    
                    # Check for wake word
                    if self._check_wake_word(audio_buffer):
                        self._handle_wake_word()
                
                elif self.state == DetectorState.ACTIVATED:
                    # Full ASR is active, handled elsewhere
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error in listen loop: {e}")
                time.sleep(0.1)
        
        self._stream.stop_stream()
        self._stream.close()
        self._stream = None
    
    def _check_wake_word(self, audio_buffer: List[bytes]) -> bool:
        """
        Check if wake word is present in audio buffer
        
        Uses Riva's streaming ASR with keyword boosting to detect
        wake phrases with high accuracy.
        """
        if not self._asr_service or not audio_buffer:
            return False
        
        try:
            # Combine audio chunks
            audio_data = b''.join(audio_buffer[-10:])  # Last ~1 second
            
            # Configure recognition with keyword boosting
            config = riva.client.RecognitionConfig(
                language_code=self.config.language_code,
                max_alternatives=1,
                enable_automatic_punctuation=False,
                audio_channel_count=self.config.channels,
                sample_rate_hertz=self.config.sample_rate,
                # Boost wake phrases
                speech_contexts=[
                    riva.client.SpeechContext(
                        phrases=self.config.wake_phrases,
                        boost=20.0  # Strong boost for wake phrases
                    )
                ]
            )
            
            # Run recognition
            response = self._asr_service.offline_recognize(
                audio_data,
                config
            )
            
            # Check for wake phrases in result
            for result in response.results:
                transcript = result.alternatives[0].transcript.lower().strip()
                confidence = result.alternatives[0].confidence
                
                # Check against wake phrases
                for phrase in self.config.wake_phrases:
                    if phrase in transcript and confidence > self.config.sensitivity:
                        self.logger.info(
                            f"Wake word detected: '{phrase}' "
                            f"(confidence: {confidence:.2f})"
                        )
                        return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Wake word check error: {e}")
            return False
    
    def _handle_wake_word(self):
        """Handle wake word detection"""
        self._set_state(DetectorState.ACTIVATED)
        
        # Play audio feedback
        if self.config.audio_feedback:
            self._play_feedback_sound()
        
        # Notify callback
        if self.on_wake_word_detected:
            self.on_wake_word_detected()
        
        # Start full ASR for command
        self._listen_for_command()
    
    def _play_feedback_sound(self):
        """Play audio feedback when wake word detected"""
        try:
            sound_path = self.config.feedback_sound
            if os.path.exists(sound_path):
                # Play sound file
                wf = wave.open(sound_path, 'rb')
                stream = self._audio.open(
                    format=self._audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
                
                stream.close()
                wf.close()
            else:
                # Generate simple beep
                self._play_beep()
                
        except Exception as e:
            self.logger.debug(f"Could not play feedback: {e}")
    
    def _play_beep(self, frequency: int = 800, duration: float = 0.2):
        """Generate and play a simple beep sound"""
        try:
            samples = int(self.config.sample_rate * duration)
            t = np.linspace(0, duration, samples, False)
            
            # Generate sine wave
            tone = np.sin(2 * np.pi * frequency * t)
            
            # Apply envelope to avoid clicks
            envelope = np.ones(samples)
            fade_samples = int(0.01 * self.config.sample_rate)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
            audio = (tone * envelope * 0.3 * 32767).astype(np.int16)
            
            # Play
            stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                output=True
            )
            stream.write(audio.tobytes())
            stream.close()
            
        except Exception as e:
            self.logger.debug(f"Beep error: {e}")
    
    def _listen_for_command(self):
        """
        Listen for a voice command after wake word detection
        
        This uses streaming ASR for real-time transcription with
        automatic endpoint detection.
        """
        self.logger.info("Listening for command...")
        
        if not self._asr_service:
            self.logger.error("Riva ASR not available")
            self._set_state(DetectorState.LISTENING)
            return
        
        try:
            # Configure streaming ASR
            config = riva.client.StreamingRecognitionConfig(
                config=riva.client.RecognitionConfig(
                    language_code=self.config.language_code,
                    max_alternatives=1,
                    enable_automatic_punctuation=True,
                    audio_channel_count=self.config.channels,
                    sample_rate_hertz=self.config.sample_rate,
                ),
                interim_results=True
            )
            
            # Start streaming recognition
            start_time = time.time()
            final_transcript = ""
            
            def audio_generator():
                """Generate audio chunks for streaming"""
                while time.time() - start_time < self.config.activation_timeout:
                    if self._stop_event.is_set():
                        break
                    if self._stream:
                        yield self._stream.read(
                            self.config.chunk_size,
                            exception_on_overflow=False
                        )
            
            # Stream audio and get responses
            responses = self._asr_service.streaming_response_generator(
                audio_chunks=audio_generator(),
                streaming_config=config
            )
            
            for response in responses:
                if self._stop_event.is_set():
                    break
                    
                for result in response.results:
                    if result.is_final:
                        final_transcript = result.alternatives[0].transcript
                        self.logger.info(f"Command: {final_transcript}")
                        break
                
                if final_transcript:
                    break
            
            # Process command
            if final_transcript:
                self._set_state(DetectorState.PROCESSING)
                
                if self.on_command_received:
                    self.on_command_received(final_transcript)
                
                # Cooldown before listening again
                self._set_state(DetectorState.COOLDOWN)
                time.sleep(self.config.cooldown_period)
            
            # Return to listening
            self._set_state(DetectorState.LISTENING)
            
        except Exception as e:
            self.logger.error(f"Error in command recognition: {e}")
            self._set_state(DetectorState.LISTENING)
    
    def __del__(self):
        """Cleanup"""
        self.stop()
        if self._audio:
            self._audio.terminate()


# ============================================
# ROS 2 Integration
# ============================================

class WakeWordNode:
    """
    ROS 2 node wrapper for wake word detection
    
    Topics Published:
    - /wake_word/detected (std_msgs/Bool): True when wake word detected
    - /wake_word/command (std_msgs/String): Recognized voice command
    - /wake_word/state (std_msgs/String): Current detector state
    
    Topics Subscribed:
    - /wake_word/enable (std_msgs/Bool): Enable/disable detection
    
    Example:
        ros2 run cognitive_service wake_word_node
    """
    
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Bool, String
        
        rclpy.init()
        self.node = rclpy.create_node('wake_word_detector')
        
        # Publishers
        self.detected_pub = self.node.create_publisher(Bool, '/wake_word/detected', 10)
        self.command_pub = self.node.create_publisher(String, '/wake_word/command', 10)
        self.state_pub = self.node.create_publisher(String, '/wake_word/state', 10)
        
        # Subscribers
        self.node.create_subscription(
            Bool, '/wake_word/enable', self._handle_enable, 10
        )
        
        # Wake word detector
        config = WakeWordConfig(
            riva_server=os.getenv('RIVA_SERVER', 'localhost:50051'),
            wake_phrases=['hey robot', 'ok robot', 'hello robot']
        )
        
        self.detector = WakeWordDetector(config)
        self.detector.on_wake_word_detected = self._on_wake_word
        self.detector.on_command_received = self._on_command
        self.detector.on_state_change = self._on_state_change
        
        self.enabled = False
    
    def _handle_enable(self, msg):
        """Handle enable/disable messages"""
        if msg.data and not self.enabled:
            self.detector.start(blocking=False)
            self.enabled = True
        elif not msg.data and self.enabled:
            self.detector.stop()
            self.enabled = False
    
    def _on_wake_word(self):
        """Called when wake word detected"""
        from std_msgs.msg import Bool
        msg = Bool()
        msg.data = True
        self.detected_pub.publish(msg)
    
    def _on_command(self, text: str):
        """Called when command recognized"""
        from std_msgs.msg import String
        msg = String()
        msg.data = text
        self.command_pub.publish(msg)
    
    def _on_state_change(self, state: DetectorState):
        """Called when state changes"""
        from std_msgs.msg import String
        msg = String()
        msg.data = state.value
        self.state_pub.publish(msg)
    
    def run(self):
        """Run the node"""
        import rclpy
        try:
            self.detector.start(blocking=False)
            self.enabled = True
            rclpy.spin(self.node)
        finally:
            self.detector.stop()
            self.node.destroy_node()
            rclpy.shutdown()


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Simple demo without ROS
    print("Wake Word Detector Demo")
    print("=" * 40)
    print("Say 'Hey Robot' to activate...")
    print("Press Ctrl+C to exit")
    print()
    
    config = WakeWordConfig(
        wake_phrases=["hey robot", "ok robot"],
        sensitivity=0.5,
        activation_timeout=10.0
    )
    
    detector = WakeWordDetector(config)
    
    def on_wake():
        print("\n🎤 WAKE WORD DETECTED! Listening for command...")
    
    def on_command(text):
        print(f"\n📝 Command: {text}\n")
    
    detector.on_wake_word_detected = on_wake
    detector.on_command_received = on_command
    
    try:
        detector.start(blocking=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()


if __name__ == '__main__':
    main()
