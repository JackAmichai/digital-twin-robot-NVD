"""
NVIDIA Riva ASR Client
Speech-to-Text for Digital Twin Robotics Lab
"""

import grpc
import pyaudio
import queue
import threading
from typing import Generator, Optional, Callable

# Riva imports (available when running in Riva container)
try:
    import riva.client
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False


class ASRClient:
    """Real-time speech recognition using NVIDIA Riva."""
    
    def __init__(
        self,
        server: str = "localhost:50051",
        language: str = "en-US",
        sample_rate: int = 16000,
    ):
        self.server = server
        self.language = language
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        if RIVA_AVAILABLE:
            self.auth = riva.client.Auth(uri=server)
            self.asr_service = riva.client.ASRService(self.auth)
        
    def _get_config(self) -> "riva.client.StreamingRecognitionConfig":
        """Create Riva streaming config."""
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.language,
                sample_rate_hertz=self.sample_rate,
                max_alternatives=1,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )
        return config
    
    def _audio_generator(self) -> Generator[bytes, None, None]:
        """Generate audio chunks from queue."""
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                yield chunk
            except queue.Empty:
                continue
    
    def _capture_audio(self):
        """Capture audio from microphone."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
        )
        
        while self.is_running:
            data = stream.read(1024, exception_on_overflow=False)
            self.audio_queue.put(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def transcribe_stream(
        self, 
        on_transcript: Callable[[str, bool], None]
    ):
        """
        Stream audio and get transcriptions.
        
        Args:
            on_transcript: Callback(text, is_final)
        """
        if not RIVA_AVAILABLE:
            raise RuntimeError("Riva not available")
        
        self.is_running = True
        
        # Start audio capture thread
        capture_thread = threading.Thread(target=self._capture_audio)
        capture_thread.start()
        
        try:
            responses = self.asr_service.streaming_response_generator(
                audio_chunks=self._audio_generator(),
                streaming_config=self._get_config(),
            )
            
            for response in responses:
                if not response.results:
                    continue
                    
                result = response.results[0]
                transcript = result.alternatives[0].transcript
                is_final = result.is_final
                
                on_transcript(transcript, is_final)
                
        finally:
            self.is_running = False
            capture_thread.join()
    
    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file."""
        if not RIVA_AVAILABLE:
            raise RuntimeError("Riva not available")
        
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        response = self.asr_service.offline_recognize(
            audio_data,
            riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.language,
                sample_rate_hertz=self.sample_rate,
            ),
        )
        
        return response.results[0].alternatives[0].transcript
    
    def stop(self):
        """Stop streaming."""
        self.is_running = False


# Mock client for development without Riva
class MockASRClient:
    """Mock ASR for development/testing."""
    
    def transcribe_stream(self, on_transcript):
        """Simulate transcription."""
        import time
        test_commands = [
            "move to zone b",
            "inspect the north shelf",
            "stop",
        ]
        for cmd in test_commands:
            time.sleep(2)
            on_transcript(cmd, True)
    
    def transcribe_file(self, audio_path: str) -> str:
        return "mock transcription"
    
    def stop(self):
        pass


def get_asr_client(server: str = "localhost:50051") -> ASRClient:
    """Get appropriate ASR client."""
    if RIVA_AVAILABLE:
        return ASRClient(server=server)
    return MockASRClient()


if __name__ == "__main__":
    def print_transcript(text: str, is_final: bool):
        prefix = "FINAL" if is_final else "INTERIM"
        print(f"[{prefix}] {text}")
    
    client = get_asr_client()
    print("Listening... (Ctrl+C to stop)")
    try:
        client.transcribe_stream(print_transcript)
    except KeyboardInterrupt:
        client.stop()
