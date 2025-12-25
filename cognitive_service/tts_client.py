#!/usr/bin/env python3
"""
Text-to-Speech (TTS) Client with NVIDIA Riva

This module provides voice feedback to the user using NVIDIA Riva TTS.
It converts text responses to natural speech for robot status updates,
confirmations, and error messages.

Features:
- Multiple voice options (male, female, different styles)
- Multi-language support matching ASR languages
- SSML support for pronunciation control
- Audio caching for common phrases
- Async playback with queue management
- Volume and speed control

Example Usage:
    tts = TTSClient()
    
    # Simple speech
    tts.speak("Navigating to zone A")
    
    # With options
    tts.speak("Task complete", voice="female", speed=1.2)
    
    # Async (non-blocking)
    tts.speak_async("Processing your request")
    
    # With SSML for better pronunciation
    tts.speak_ssml('<speak>Moving <say-as interpret-as="cardinal">2</say-as> meters forward</speak>')

Voice Feedback Events:
- Wake word detected: "I'm listening"
- Command acknowledged: "Navigating to [location]"
- Task complete: "I've arrived at [location]"
- Error: "Sorry, I couldn't understand that"
- Low battery: "Battery low, returning to charging station"
"""

import grpc
import numpy as np
import pyaudio
import threading
import queue
import time
import hashlib
import os
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import wave
import io

# Riva TTS imports
try:
    import riva.client
    from riva.client import SpeechSynthesisService
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False
    print("WARNING: Riva client not installed. Install with: pip install nvidia-riva-client")


class VoiceStyle(Enum):
    """Available voice styles"""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    URGENT = "urgent"


class Voice(Enum):
    """Available TTS voices"""
    # English voices
    EN_US_FEMALE = "English-US.Female-1"
    EN_US_MALE = "English-US.Male-1"
    EN_GB_FEMALE = "English-GB.Female-1"
    
    # Spanish voices
    ES_ES_FEMALE = "Spanish-ES.Female-1"
    ES_ES_MALE = "Spanish-ES.Male-1"
    
    # German voices
    DE_DE_FEMALE = "German-DE.Female-1"
    DE_DE_MALE = "German-DE.Male-1"
    
    # French voices
    FR_FR_FEMALE = "French-FR.Female-1"
    
    # Chinese voices
    ZH_CN_FEMALE = "Mandarin-CN.Female-1"
    
    # Default
    DEFAULT = "English-US.Female-1"


@dataclass
class TTSConfig:
    """TTS Configuration"""
    
    # Riva server
    riva_server: str = "localhost:50051"
    
    # Audio settings
    sample_rate: int = 22050
    
    # Voice settings
    default_voice: str = "English-US.Female-1"
    default_language: str = "en-US"
    
    # Speech parameters
    speaking_rate: float = 1.0    # 0.5 to 2.0
    pitch: float = 0.0            # -20 to 20 semitones
    volume_gain_db: float = 0.0   # -96 to 16 dB
    
    # Caching
    enable_cache: bool = True
    cache_dir: str = "/tmp/tts_cache"
    max_cache_size_mb: int = 100
    
    # Playback
    async_playback: bool = True
    interrupt_on_new: bool = True  # Stop current speech for new


@dataclass
class SpeechResponse:
    """Response from TTS synthesis"""
    audio_data: bytes
    sample_rate: int
    duration_seconds: float
    text: str
    voice: str
    cached: bool = False


class TTSClient:
    """
    Text-to-Speech client using NVIDIA Riva
    
    Provides voice feedback for robot status updates and confirmations.
    
    Example:
        tts = TTSClient()
        
        # Blocking speech
        tts.speak("Hello, I am your robot assistant")
        
        # Non-blocking speech
        tts.speak_async("Starting navigation")
        
        # With language
        tts.speak("Navegando a la zona A", language="es-ES")
        
        # Stop current speech
        tts.stop()
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.logger = logging.getLogger("TTSClient")
        
        # Audio playback
        self._audio = pyaudio.PyAudio()
        self._playback_stream: Optional[pyaudio.Stream] = None
        
        # Async playback queue
        self._speech_queue: queue.Queue = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_speaking = False
        
        # Cache
        self._cache: Dict[str, bytes] = {}
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Riva client
        self._riva_auth = None
        self._tts_service = None
        
        if RIVA_AVAILABLE:
            self._init_riva()
        
        # Start playback thread
        if self.config.async_playback:
            self._start_playback_thread()
        
        # Pre-cache common phrases
        self._precache_common_phrases()
    
    def _init_riva(self):
        """Initialize Riva TTS client"""
        try:
            self._riva_auth = riva.client.Auth(
                uri=self.config.riva_server,
                use_ssl=False
            )
            self._tts_service = SpeechSynthesisService(self._riva_auth)
            self.logger.info(f"TTS connected to {self.config.riva_server}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Riva TTS: {e}")
    
    def _start_playback_thread(self):
        """Start background playback thread"""
        self._stop_event.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True
        )
        self._playback_thread.start()
    
    def _playback_loop(self):
        """Background thread for async audio playback"""
        while not self._stop_event.is_set():
            try:
                # Get next speech item (with timeout for clean shutdown)
                try:
                    audio_data, sample_rate = self._speech_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Play audio
                self._is_speaking = True
                self._play_audio(audio_data, sample_rate)
                self._is_speaking = False
                
                self._speech_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Playback error: {e}")
                self._is_speaking = False
    
    def _play_audio(self, audio_data: bytes, sample_rate: int):
        """Play audio through speakers"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Open stream
            stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            # Play in chunks for interruptibility
            chunk_size = 1024
            for i in range(0, len(audio_array), chunk_size):
                if self._stop_event.is_set():
                    break
                chunk = audio_array[i:i + chunk_size]
                stream.write(chunk.tobytes())
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
    
    def synthesize(self, text: str, 
                   voice: str = None,
                   language: str = None,
                   speaking_rate: float = None,
                   pitch: float = None) -> SpeechResponse:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            voice: Voice name (see Voice enum)
            language: Language code
            speaking_rate: Speed multiplier (0.5 - 2.0)
            pitch: Pitch adjustment (-20 to 20)
            
        Returns:
            SpeechResponse with audio data
        """
        voice = voice or self.config.default_voice
        language = language or self.config.default_language
        speaking_rate = speaking_rate or self.config.speaking_rate
        pitch = pitch or self.config.pitch
        
        # Check cache
        cache_key = self._get_cache_key(text, voice, speaking_rate, pitch)
        if self.config.enable_cache and cache_key in self._cache:
            self.logger.debug(f"Cache hit: {text[:30]}...")
            return SpeechResponse(
                audio_data=self._cache[cache_key],
                sample_rate=self.config.sample_rate,
                duration_seconds=len(self._cache[cache_key]) / (2 * self.config.sample_rate),
                text=text,
                voice=voice,
                cached=True
            )
        
        if not self._tts_service:
            self.logger.warning("TTS service not available")
            return SpeechResponse(
                audio_data=b"",
                sample_rate=self.config.sample_rate,
                duration_seconds=0,
                text=text,
                voice=voice
            )
        
        try:
            # Synthesize with Riva
            response = self._tts_service.synthesize(
                text,
                voice_name=voice,
                language_code=language,
                sample_rate_hz=self.config.sample_rate,
                audio_prompt_file=None
            )
            
            audio_data = response.audio
            
            # Cache if enabled
            if self.config.enable_cache:
                self._cache[cache_key] = audio_data
            
            duration = len(audio_data) / (2 * self.config.sample_rate)
            
            return SpeechResponse(
                audio_data=audio_data,
                sample_rate=self.config.sample_rate,
                duration_seconds=duration,
                text=text,
                voice=voice,
                cached=False
            )
            
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            return SpeechResponse(
                audio_data=b"",
                sample_rate=self.config.sample_rate,
                duration_seconds=0,
                text=text,
                voice=voice
            )
    
    def synthesize_ssml(self, ssml: str, voice: str = None) -> SpeechResponse:
        """
        Synthesize SSML to speech
        
        SSML allows fine-grained control over pronunciation:
        
        Example:
            <speak>
                Moving <say-as interpret-as="cardinal">2</say-as> meters
                <break time="500ms"/>
                forward.
            </speak>
        """
        voice = voice or self.config.default_voice
        
        if not self._tts_service:
            return SpeechResponse(b"", self.config.sample_rate, 0, ssml, voice)
        
        try:
            response = self._tts_service.synthesize(
                ssml,
                voice_name=voice,
                sample_rate_hz=self.config.sample_rate
            )
            
            return SpeechResponse(
                audio_data=response.audio,
                sample_rate=self.config.sample_rate,
                duration_seconds=len(response.audio) / (2 * self.config.sample_rate),
                text=ssml,
                voice=voice
            )
            
        except Exception as e:
            self.logger.error(f"SSML synthesis error: {e}")
            return SpeechResponse(b"", self.config.sample_rate, 0, ssml, voice)
    
    def speak(self, text: str, **kwargs) -> float:
        """
        Speak text (blocking)
        
        Args:
            text: Text to speak
            **kwargs: Options passed to synthesize()
            
        Returns:
            Duration in seconds
        """
        response = self.synthesize(text, **kwargs)
        
        if response.audio_data:
            self._play_audio(response.audio_data, response.sample_rate)
        
        return response.duration_seconds
    
    def speak_async(self, text: str, **kwargs):
        """
        Speak text (non-blocking)
        
        Adds to queue for background playback.
        """
        if self.config.interrupt_on_new and self._is_speaking:
            self.stop()
        
        response = self.synthesize(text, **kwargs)
        
        if response.audio_data:
            self._speech_queue.put((response.audio_data, response.sample_rate))
    
    def stop(self):
        """Stop current speech"""
        self._stop_event.set()
        
        # Clear queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        
        # Restart for future speech
        time.sleep(0.1)
        self._stop_event.clear()
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self._is_speaking or not self._speech_queue.empty()
    
    def wait_until_done(self, timeout: float = None):
        """Wait until all queued speech is complete"""
        self._speech_queue.join()
    
    def _get_cache_key(self, text: str, voice: str, 
                       rate: float, pitch: float) -> str:
        """Generate cache key for text"""
        content = f"{text}|{voice}|{rate}|{pitch}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _precache_common_phrases(self):
        """Pre-cache commonly used phrases for instant playback"""
        common_phrases = [
            "I'm listening",
            "I didn't catch that, please try again",
            "Navigating to the destination",
            "I've arrived",
            "Task complete",
            "Starting navigation",
            "Stopping",
            "Turning left",
            "Turning right",
            "Moving forward",
            "Battery low, returning to charging station",
            "Command received",
            "Processing",
            "Please wait",
            "Error encountered",
        ]
        
        if not self._tts_service:
            return
        
        self.logger.info("Pre-caching common phrases...")
        for phrase in common_phrases:
            try:
                self.synthesize(phrase)
            except Exception as e:
                self.logger.debug(f"Could not cache '{phrase}': {e}")
    
    def __del__(self):
        """Cleanup"""
        self._stop_event.set()
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
        if self._audio:
            self._audio.terminate()


# ============================================
# Robot Voice Feedback Manager
# ============================================

class RobotVoiceFeedback:
    """
    High-level voice feedback for robot events
    
    Provides contextual voice responses for common robot situations.
    
    Example:
        feedback = RobotVoiceFeedback()
        
        # When wake word detected
        feedback.on_wake_word()
        
        # When navigation starts
        feedback.on_navigation_start("warehouse zone A")
        
        # When navigation completes
        feedback.on_navigation_complete("warehouse zone A")
        
        # On error
        feedback.on_error("Path blocked by obstacle")
    """
    
    def __init__(self, tts_client: TTSClient = None, language: str = "en-US"):
        self.tts = tts_client or TTSClient()
        self.language = language
        self.enabled = True
        
        # Response templates by language
        self._templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load response templates for different languages"""
        return {
            "en-US": {
                "wake": "I'm listening",
                "wake_timeout": "I didn't hear a command",
                "command_ack": "Got it",
                "nav_start": "Navigating to {location}",
                "nav_complete": "I've arrived at {location}",
                "nav_failed": "I couldn't reach {location}. {reason}",
                "stop": "Stopping",
                "turning_left": "Turning left",
                "turning_right": "Turning right",
                "moving_forward": "Moving forward {distance}",
                "moving_backward": "Moving backward {distance}",
                "battery_low": "Battery at {percent} percent. Heading to charging station",
                "battery_critical": "Battery critical. Emergency return to charger",
                "obstacle": "Obstacle detected. Replanning route",
                "error": "Error: {message}",
                "unknown_command": "I didn't understand that. Please try again",
                "zone_wait": "Waiting for clearance at {zone}",
                "zone_clear": "Zone clear, proceeding",
                "task_queue": "Added to task queue. {count} tasks ahead",
            },
            "es-ES": {
                "wake": "Te escucho",
                "wake_timeout": "No escuché ningún comando",
                "command_ack": "Entendido",
                "nav_start": "Navegando hacia {location}",
                "nav_complete": "He llegado a {location}",
                "nav_failed": "No pude llegar a {location}. {reason}",
                "stop": "Parando",
                "turning_left": "Girando a la izquierda",
                "turning_right": "Girando a la derecha",
                "moving_forward": "Avanzando {distance}",
                "battery_low": "Batería al {percent} por ciento. Volviendo a cargar",
                "error": "Error: {message}",
                "unknown_command": "No entendí. Por favor, inténtalo de nuevo",
            },
            "de-DE": {
                "wake": "Ich höre zu",
                "command_ack": "Verstanden",
                "nav_start": "Navigiere zu {location}",
                "nav_complete": "Angekommen bei {location}",
                "stop": "Anhalten",
                "error": "Fehler: {message}",
            },
            "fr-FR": {
                "wake": "Je vous écoute",
                "command_ack": "Compris",
                "nav_start": "Navigation vers {location}",
                "nav_complete": "Arrivé à {location}",
                "stop": "Arrêt",
                "error": "Erreur: {message}",
            },
        }
    
    def _get_template(self, key: str) -> str:
        """Get template for current language, fallback to English"""
        templates = self._templates.get(self.language, self._templates["en-US"])
        return templates.get(key, self._templates["en-US"].get(key, ""))
    
    def _speak(self, text: str, **kwargs):
        """Speak if enabled"""
        if self.enabled and text:
            self.tts.speak_async(text, language=self.language, **kwargs)
    
    # ================== Event Handlers ==================
    
    def on_wake_word(self):
        """Called when wake word detected"""
        self._speak(self._get_template("wake"))
    
    def on_wake_timeout(self):
        """Called when no command after wake word"""
        self._speak(self._get_template("wake_timeout"))
    
    def on_command_acknowledged(self):
        """Called when command is understood"""
        self._speak(self._get_template("command_ack"))
    
    def on_navigation_start(self, location: str):
        """Called when navigation begins"""
        text = self._get_template("nav_start").format(location=location)
        self._speak(text)
    
    def on_navigation_complete(self, location: str):
        """Called when robot arrives at destination"""
        text = self._get_template("nav_complete").format(location=location)
        self._speak(text)
    
    def on_navigation_failed(self, location: str, reason: str = ""):
        """Called when navigation fails"""
        text = self._get_template("nav_failed").format(
            location=location, 
            reason=reason
        )
        self._speak(text)
    
    def on_stop(self):
        """Called when robot stops"""
        self._speak(self._get_template("stop"))
    
    def on_turn(self, direction: str):
        """Called when robot turns"""
        if direction.lower() == "left":
            self._speak(self._get_template("turning_left"))
        else:
            self._speak(self._get_template("turning_right"))
    
    def on_move(self, direction: str, distance: str = ""):
        """Called when robot moves"""
        if direction.lower() == "forward":
            text = self._get_template("moving_forward").format(distance=distance)
        else:
            text = self._get_template("moving_backward").format(distance=distance)
        self._speak(text)
    
    def on_battery_low(self, percent: int):
        """Called when battery is low"""
        text = self._get_template("battery_low").format(percent=percent)
        self._speak(text)
    
    def on_battery_critical(self):
        """Called when battery is critical"""
        self._speak(self._get_template("battery_critical"))
    
    def on_obstacle_detected(self):
        """Called when obstacle blocks path"""
        self._speak(self._get_template("obstacle"))
    
    def on_error(self, message: str):
        """Called on error"""
        text = self._get_template("error").format(message=message)
        self._speak(text)
    
    def on_unknown_command(self):
        """Called when command not understood"""
        self._speak(self._get_template("unknown_command"))
    
    def on_zone_wait(self, zone: str):
        """Called when waiting for traffic zone"""
        text = self._get_template("zone_wait").format(zone=zone)
        self._speak(text)
    
    def on_zone_clear(self):
        """Called when zone access granted"""
        self._speak(self._get_template("zone_clear"))
    
    def set_language(self, language: str):
        """Change feedback language"""
        self.language = language
        self.tts.config.default_language = language
    
    def set_enabled(self, enabled: bool):
        """Enable/disable voice feedback"""
        self.enabled = enabled


def main():
    """Demo TTS functionality"""
    logging.basicConfig(level=logging.INFO)
    
    print("Text-to-Speech Demo")
    print("=" * 50)
    
    tts = TTSClient()
    feedback = RobotVoiceFeedback(tts)
    
    # Demo events
    print("\nSimulating robot events...")
    
    print("1. Wake word detected")
    feedback.on_wake_word()
    time.sleep(2)
    
    print("2. Navigation starting")
    feedback.on_navigation_start("warehouse zone A")
    time.sleep(3)
    
    print("3. Navigation complete")
    feedback.on_navigation_complete("warehouse zone A")
    time.sleep(3)
    
    print("4. Error example")
    feedback.on_error("Path blocked")
    time.sleep(3)
    
    # Spanish example
    print("\n5. Spanish feedback")
    feedback.set_language("es-ES")
    feedback.on_wake_word()
    time.sleep(2)
    feedback.on_navigation_start("zona de carga")
    time.sleep(3)
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
